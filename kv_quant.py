"""KV-cache fake quantization, optionally wrapped in the transform of a KVBundle.

Entry points:

* ``quantize_past_kv_inplace(cache, bits, block_size, bundle=None, ...)`` —
  post-RoPE: call once after prefill to replace every layer's cached K/V with
  its fake-quantized version.
* ``install_pre_rope_kv_quant(model, bits, block_size, bundle=None, ...)`` —
  pre-RoPE: patches ``apply_rotary_pos_emb`` so K is fake-quantized before
  RoPE, and hooks every ``self_attn.v_proj`` to fake-quantize V.

Quantization is asymmetric uniform RTN with one scale per group:
  - ``per_channel`` — groups of ``block_size`` tokens per channel (K default);
  - ``per_token``   — groups of ``block_size`` channels per token (always V);
  - ``per_tensor``  — ``block_size × block_size`` (token × channel) tiles.

Transforms (per site and layer, from the bundle):
  - none: plain RTN;
  - dense orthogonal R: ``y = x @ R``, inverse ``y @ R.t()``;
  - Hadamard: ``y = P_o · D_o · H · (x[..., perm] * signs)``. Shared and
    per-head orients act on each head; per-layer orients act on the flattened
    ``num_kv_heads * head_dim`` axis, with one full-width Hadamard or, with
    ``hadamard_block=head_dim``, one Hadamard per head-sized block.

Decode steps (seq == 1) are not quantized.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Callable, List, Optional

import torch

from gha import _build_sylvester_hadamard
from kv_bundle import (
    LAYOUT_PER_LAYER, SITE_K_POST_ROPE, SITE_K_PRE_ROPE, SITE_V, KVBundle,
)

V_MODE = "per_token"


# ---------------------------------------------------------------------------
# Fake quantization
# ---------------------------------------------------------------------------

def _quant_dequant_asym(x: torch.Tensor, dim: int, bits: int) -> torch.Tensor:
    """Asymmetric uniform RTN along ``dim``; returns the dequantized tensor."""
    qmax = (1 << bits) - 1
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    scale = ((xmax - xmin) / qmax).clamp(min=1e-12)
    q = torch.round((x - xmin) / scale).clamp_(0, qmax)
    return q * scale + xmin


def _fake_quant_per_channel(x: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    """Groups of ``block_size`` tokens (axis -2) per channel."""
    seq = x.shape[-2]
    if seq <= 1:
        return x
    n_full = (seq // block_size) * block_size
    parts = []
    if n_full > 0:
        lead = x.shape[:-2]
        d = x.shape[-1]
        x_full = x[..., :n_full, :].reshape(*lead, n_full // block_size, block_size, d)
        x_full = _quant_dequant_asym(x_full, dim=-2, bits=bits)
        parts.append(x_full.reshape(*lead, n_full, d))
    if n_full < seq:
        parts.append(_quant_dequant_asym(x[..., n_full:, :], dim=-2, bits=bits))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)


def _fake_quant_per_token(x: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    """Groups of ``block_size`` channels (axis -1) per token."""
    d = x.shape[-1]
    n_full = (d // block_size) * block_size
    parts = []
    if n_full > 0:
        lead = x.shape[:-1]
        x_full = x[..., :n_full].reshape(*lead, n_full // block_size, block_size)
        x_full = _quant_dequant_asym(x_full, dim=-1, bits=bits)
        parts.append(x_full.reshape(*lead, n_full))
    if n_full < d:
        parts.append(_quant_dequant_asym(x[..., n_full:], dim=-1, bits=bits))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _fake_quant_per_tensor(x: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    """``(block_size, block_size)`` tiles over (token, channel); token / channel
    tails form their own strips, and the leftover corner gets one scale."""
    seq, d = x.shape[-2], x.shape[-1]
    if seq <= 1:
        return x
    nT = (seq // block_size) * block_size
    nD = (d // block_size) * block_size
    lead = x.shape[:-2]

    def qd(t, dims):
        qmax = (1 << bits) - 1
        xmin = t.amin(dim=dims, keepdim=True)
        xmax = t.amax(dim=dims, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp(min=1e-12)
        return torch.round((t - xmin) / scale).clamp_(0, qmax) * scale + xmin

    out = x.clone()
    if nT > 0 and nD > 0:
        tiles = x[..., :nT, :nD].reshape(
            *lead, nT // block_size, block_size, nD // block_size, block_size)
        out[..., :nT, :nD] = qd(tiles, (-3, -1)).reshape(*lead, nT, nD)
    if nT > 0 and nD < d:
        strip = x[..., :nT, nD:].reshape(*lead, nT // block_size, block_size, d - nD)
        out[..., :nT, nD:] = qd(strip, (-2, -1)).reshape(*lead, nT, d - nD)
    if nT < seq and nD > 0:
        strip = x[..., nT:, :nD].reshape(*lead, seq - nT, nD // block_size, block_size)
        out[..., nT:, :nD] = qd(strip, (-3, -1)).reshape(*lead, seq - nT, nD)
    if nT < seq and nD < d:
        out[..., nT:, nD:] = qd(x[..., nT:, nD:], (-2, -1))
    return out


_FAKE_QUANT = {
    "per_channel": _fake_quant_per_channel,
    "per_token": _fake_quant_per_token,
    "per_tensor": _fake_quant_per_tensor,
}


def fake_quant(x: torch.Tensor, bits: int, block_size: int, mode: str) -> torch.Tensor:
    """Fake-quantize ``x`` (computed in float32, returned in ``x.dtype``)."""
    if mode not in _FAKE_QUANT:
        raise ValueError(f"unknown scaling mode {mode!r}; expected one of {tuple(_FAKE_QUANT)}")
    return _FAKE_QUANT[mode](x.float(), bits, block_size).to(x.dtype)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _hadamard_matrix(n: int, device: str, dtype_str: str) -> torch.Tensor:
    """Normalized n x n Sylvester-Hadamard matrix (symmetric, orthogonal)."""
    assert n > 0 and (n & (n - 1)) == 0, f"Hadamard size must be a power of 2, got {n}"
    return _build_sylvester_hadamard(n, "cpu", torch.float64).to(
        dtype=getattr(torch, dtype_str), device=device)


def _blocked_hadamard(x: torch.Tensor, block: Optional[int]) -> torch.Tensor:
    """Hadamard along the last dim: one full-width matrix (``block=None``) or
    one ``block × block`` matrix per contiguous block. Self-inverse."""
    F = x.shape[-1]
    dt = str(x.dtype).split(".")[-1]
    if block is None or block == F:
        return torch.matmul(x, _hadamard_matrix(F, str(x.device), dt).t())
    if F % block != 0:
        raise ValueError(f"blocked Hadamard: last dim {F} not divisible by block {block}")
    lead = x.shape[:-1]
    yb = torch.matmul(x.reshape(*lead, F // block, block),
                      _hadamard_matrix(block, str(x.device), dt).t())
    return yb.reshape(*lead, F)


def _bview(t: torch.Tensor, x: torch.Tensor, head_axis: int) -> torch.Tensor:
    """View a ``(D,)`` or ``(H, D)`` tensor so it broadcasts against ``x``
    (heads at ``head_axis``, channels last)."""
    t = t.to(x.device)
    if t.ndim == 1:
        return t
    view = [1] * x.ndim
    view[head_axis], view[-1] = t.shape
    return t.view(view)


def _gather(x: torch.Tensor, perm: torch.Tensor, head_axis: int) -> torch.Tensor:
    return torch.gather(x, -1, _bview(perm, x, head_axis).expand(x.shape))


def _scale(x: torch.Tensor, signs: torch.Tensor, head_axis: int) -> torch.Tensor:
    return x * _bview(signs, x, head_axis).to(x.dtype)


def _hadamard_roundtrip(x, quant, signs, perm, signs_out, perm_out, head_axis, block):
    """``y = P_o·D_o·H·(x[perm]·signs)`` → ``quant(y)`` → inverse transform.
    ``perm_out=None`` with ``signs_out`` set means D_o only."""
    y = _blocked_hadamard(_scale(_gather(x, perm, head_axis), signs, head_axis), block)
    if signs_out is not None:
        y = _scale(y, signs_out, head_axis)
        if perm_out is not None:
            y = _gather(y, perm_out, head_axis)
    y = quant(y)
    if signs_out is not None:
        if perm_out is not None:
            y = _gather(y, torch.argsort(perm_out, dim=-1), head_axis)
        y = _scale(y, signs_out, head_axis)
    x = _scale(_blocked_hadamard(y, block), signs, head_axis)
    return _gather(x, torch.argsort(perm, dim=-1), head_axis)


def _dense(x: torch.Tensor, R: torch.Tensor, block: Optional[int],
           inverse: bool = False) -> torch.Tensor:
    """``x @ R`` (``inverse``: ``x @ R.t()``) along the last dim. Per-layer R is
    ``(F, F)`` (``block=None``); per-head R is ``(H, D, D)`` (``block=D``)."""
    Rr = R.to(device=x.device, dtype=x.dtype)
    if block is None:
        return torch.matmul(x, Rr.transpose(-1, -2) if inverse else Rr)
    F = x.shape[-1]
    lead = x.shape[:-1]
    xb = x.reshape(*lead, F // block, block)
    eq = "...he,hde->...hd" if inverse else "...hd,hde->...he"
    return torch.einsum(eq, xb, Rr).reshape(*lead, F)


def _flatten_heads(x: torch.Tensor, head_axis: int):
    """``(..., H, ..., D)`` -> ``(..., H*D)`` with the head axis moved next to
    the channels; returns the flat tensor and an undo function."""
    ndim = x.ndim
    h = head_axis % ndim
    axes = [i for i in range(ndim) if i not in (h, ndim - 1)] + [h, ndim - 1]
    xp = x.permute(*axes).contiguous()
    inv = [axes.index(i) for i in range(ndim)]

    def undo(y_flat: torch.Tensor) -> torch.Tensor:
        return y_flat.reshape(xp.shape).permute(*inv).contiguous()

    return xp.reshape(*xp.shape[:-2], -1), undo


def _make_site_quant(bits: int, block_size: int, bundle: Optional[KVBundle],
                     out_mode: str, k_mode: str, hadamard_block: Optional[int]):
    """Return ``quant(x, site, li, head_axis)``: fake-quantize ``x`` (heads at
    ``head_axis``, channels last) with the bundle's transform for (site, li)."""
    if out_mode not in ("none", "d_out", "dp_out"):
        raise ValueError(f"unknown out_mode {out_mode!r}")

    def quant(x: torch.Tensor, site: str, li: int, head_axis: int) -> torch.Tensor:
        mode = V_MODE if site == SITE_V else k_mode
        q = lambda y: fake_quant(y, bits, block_size, mode)
        if bundle is None:
            return q(x)
        R = bundle.get_layer_matrix(site, li)
        if R is not None:
            flat, undo = _flatten_heads(x, head_axis)
            block = bundle.head_dim if R.ndim == 3 else None
            return undo(_dense(q(_dense(flat, R, block)), R, block, inverse=True))
        orient = bundle.get_layer(site, li)
        if orient is None:
            return q(x)
        signs, perm = orient
        out = bundle.get_layer_out(site, li) if out_mode != "none" else None
        so, po = out if out is not None else (None, None)
        if out_mode == "d_out":
            po = None
        if bundle.layout(site) != LAYOUT_PER_LAYER:
            return _hadamard_roundtrip(x, q, signs, perm, so, po, head_axis, None)
        flat, undo = _flatten_heads(x, head_axis)
        if flat.shape[-1] != signs.numel():
            raise ValueError(f"per-layer {site}: flat dim {flat.shape[-1]} != "
                             f"bundle dim {signs.numel()}")
        return undo(_hadamard_roundtrip(flat, q, signs, perm, so, po, -1, hadamard_block))

    return quant


# ---------------------------------------------------------------------------
# Post-RoPE: quantize the cache once after prefill
# ---------------------------------------------------------------------------

def _iter_cache_layers(cache):
    """Yield ``(idx, k, v, setter)`` for every layer of a DynamicCache."""
    for i, layer in enumerate(cache.layers):
        def setter(k_new, v_new, lyr=layer):
            lyr.keys = k_new
            lyr.values = v_new

        yield i, layer.keys, layer.values, setter


def quantize_past_kv_inplace(
    cache, bits: int, block_size: int,
    bundle: Optional[KVBundle] = None,
    out_mode: str = "none",
    k_mode: str = "per_channel",
    hadamard_block: Optional[int] = None,
) -> None:
    """Replace each layer's cached K/V (``(B, H, S, D)``) in place with its
    fake-quantized version."""
    quant = _make_site_quant(bits, block_size, bundle, out_mode, k_mode, hadamard_block)
    for li, k, v, setter in _iter_cache_layers(cache):
        setter(quant(k, SITE_K_POST_ROPE, li, -3), quant(v, SITE_V, li, -3))


# ---------------------------------------------------------------------------
# Pre-RoPE: apply_rotary_pos_emb wrapper + v_proj hooks
# ---------------------------------------------------------------------------

class _PreRopeKVQuantHandle:
    def __init__(self, defer: bool = False):
        self._restores: List[Callable[[], None]] = []
        # Independent prefill: the hooks leave the prefill forward unquantized
        # and stash the quantized cache entries here, per layer, in cache
        # layout (B, H_kv, S, D); K is stored after RoPE.
        self.defer = defer
        self._pending_k: dict = {}
        self._pending_v: dict = {}

    def apply_deferred(self, cache) -> None:
        """Independent prefill: overwrite every prefill layer of ``cache``
        with the stashed quantized K/V. Call once, right after the prefill
        forward."""
        if not self.defer:
            raise RuntimeError("apply_deferred() requires defer=True")
        n = 0
        for li, k, v, setter in _iter_cache_layers(cache):
            if li not in self._pending_k or li not in self._pending_v:
                raise RuntimeError(
                    f"deferred pre-RoPE quant: no stashed K/V for layer {li} "
                    f"(have K={sorted(self._pending_k)}, V={sorted(self._pending_v)})")
            k_new, v_new = self._pending_k[li], self._pending_v[li]
            if k_new.shape != k.shape or v_new.shape != v.shape:
                raise RuntimeError(
                    f"deferred pre-RoPE quant: layer {li} shape mismatch "
                    f"K {tuple(k_new.shape)} vs cache {tuple(k.shape)}, "
                    f"V {tuple(v_new.shape)} vs cache {tuple(v.shape)}")
            setter(k_new.to(k.dtype), v_new.to(v.dtype))
            n += 1
        if n != len(self._pending_k):
            raise RuntimeError(
                f"deferred pre-RoPE quant: cache has {n} layers but "
                f"{len(self._pending_k)} were stashed")
        self._pending_k.clear()
        self._pending_v.clear()

    def remove(self) -> None:
        for fn in reversed(self._restores):
            fn()
        self._restores.clear()


def install_pre_rope_kv_quant(
    model, bits: int, block_size: int,
    bundle: Optional[KVBundle] = None,
    out_mode: str = "none",
    k_mode: str = "per_channel",
    hadamard_block: Optional[int] = None,
    defer: bool = False,
) -> _PreRopeKVQuantHandle:
    """Install the pre-RoPE K wrapper and the v_proj V hooks.

    ``defer=False`` (sequential prefill): K (before RoPE) and V are
    fake-quantized inside the prefill forward, so later layers, prefill
    attention and the first generated token see the quantized KV.
    ``defer=True`` (independent prefill): the forward stays unquantized; the
    hooks stash the quantized cache entries, and the caller writes them with
    ``handle.apply_deferred(cache)`` after the prefill.
    """
    handle = _PreRopeKVQuantHandle(defer=defer)
    quant = _make_site_quant(bits, block_size, bundle, out_mode, k_mode, hadamard_block)
    cfg = model.config
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    cur_layer = [0]

    # K: wrap apply_rotary_pos_emb once per attention module class.
    for mod_name in dict.fromkeys(type(l.self_attn).__module__ for l in model.model.layers):
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "apply_rotary_pos_emb"):
            print(f"[kv_quant] warning: {mod_name} has no apply_rotary_pos_emb; "
                  "pre-RoPE K quant not applied for this module's attentions.")
            continue
        orig = mod.apply_rotary_pos_emb

        def wrapped(q, k, *args, _orig=orig, **kwargs):
            if k.shape[-2] > 1:
                li = cur_layer[0]
                k_q = quant(k, SITE_K_PRE_ROPE, li, -3)
                if handle.defer:
                    handle._pending_k[li] = _orig(q, k_q, *args, **kwargs)[1]
                else:
                    k = k_q
            return _orig(q, k, *args, **kwargs)

        mod.apply_rotary_pos_emb = wrapped
        handle._restores.append(lambda m=mod, o=orig: setattr(m, "apply_rotary_pos_emb", o))

    for li, layer in enumerate(model.model.layers):
        def set_layer(_module, _args, _li=li):
            cur_layer[0] = _li

        def v_hook(_module, _inputs, output, _li=li):
            if output.shape[-2] <= 1:
                return output
            B, S, F = output.shape
            if bundle is None or (bundle.get_layer_matrix(SITE_V, _li) is None
                                  and bundle.get_layer(SITE_V, _li) is None):
                v_q = fake_quant(output, bits, block_size, V_MODE)
            else:
                v_q = quant(output.reshape(B, S, F // head_dim, head_dim),
                            SITE_V, _li, -2).reshape(B, S, F)
            if handle.defer:
                handle._pending_v[_li] = (v_q.view(B, S, F // head_dim, head_dim)
                                          .transpose(1, 2).contiguous())
                return output
            return v_q

        handle._restores.append(layer.self_attn.register_forward_pre_hook(set_layer).remove)
        handle._restores.append(layer.self_attn.v_proj.register_forward_hook(v_hook).remove)

    return handle
