"""KV-cache transform bundles: schema, loader and builders.

A bundle specifies the transform applied to K/V before quantization (and
inverted after dequantization), per site:
    k_pre_rope   — K quantized before RoPE
    k_post_rope  — K quantized after RoPE
    v            — V

Hadamard layouts (``y = H(x[..., perm] * signs)``, optional output side
``signs_out`` / ``perm_out``):
* ``shared``    — ``(head_dim,)``: one orient for all layers and heads (RHT-PH).
* ``per_head``  — ``(num_layers, num_kv_heads, head_dim)``: one per (layer, head).
* ``per_layer`` — ``(num_layers, num_kv_heads * head_dim)``: one per layer over
  the flattened channel axis.

Dense layouts (``y = x @ R``, R orthogonal; DFRot https://arxiv.org/abs/2412.00648,
ButterflyQuant https://arxiv.org/abs/2509.09679, GSR https://arxiv.org/abs/2505.03810):
* ``dense_per_head``  — ``(num_layers, num_kv_heads, head_dim, head_dim)``.
* ``dense_per_layer`` — ``(num_layers, F, F)``, ``F = num_kv_heads * head_dim``.

A missing site means plain RTN at that site.
"""
from __future__ import annotations

from typing import Optional

import torch

SITE_K_PRE_ROPE = "k_pre_rope"
SITE_K_POST_ROPE = "k_post_rope"
SITE_V = "v"
ALL_SITES = (SITE_K_PRE_ROPE, SITE_K_POST_ROPE, SITE_V)

LAYOUT_SHARED = "shared"
LAYOUT_PER_HEAD = "per_head"
LAYOUT_PER_LAYER = "per_layer"

LAYOUT_DENSE_PER_HEAD = "dense_per_head"
LAYOUT_DENSE_PER_LAYER = "dense_per_layer"
DENSE_LAYOUTS = (LAYOUT_DENSE_PER_HEAD, LAYOUT_DENSE_PER_LAYER)

CONVENTION = "y = fwht(x[..., perm] * signs)"
CONVENTION_DENSE = "y = x @ R  (R orthogonal; inverse y @ R.t())"


class KVBundle:
    def __init__(self, payload: dict):
        meta = payload.get("meta", {})
        if meta.get("convention") not in (None, CONVENTION, CONVENTION_DENSE):
            raise ValueError(f"unsupported convention {meta.get('convention')!r}")
        self.head_dim = int(meta["head_dim"])
        self.meta = meta
        self._matrix: dict = {}      # dense sites: R
        self._signs: dict = {}       # Hadamard sites: signs / perm
        self._perm: dict = {}
        self._layout: dict = {}
        self._signs_out: dict = {}   # optional output side D_o / P_o
        self._perm_out: dict = {}

        # Layout is declared in meta["layouts"]; older bundles without it are
        # shared (1-D) or per_head.
        declared_layouts = meta.get("layouts", {})

        for site in ALL_SITES:
            raw = payload.get(site)
            if raw is None:
                continue
            if "matrix" in raw:
                self._parse_dense_site(site, raw, declared_layouts)
                continue
            signs = raw["signs"].to(torch.float32)
            perm = raw["perm"].to(torch.long)
            layout = declared_layouts.get(site)
            if layout is None:
                layout = LAYOUT_SHARED if signs.ndim == 1 else LAYOUT_PER_HEAD
            if perm.shape != signs.shape:
                raise ValueError(f"site {site}: perm shape {tuple(perm.shape)} "
                                 f"!= signs shape {tuple(signs.shape)}")
            if layout == LAYOUT_SHARED:
                ok = signs.shape == (self.head_dim,)
            elif layout == LAYOUT_PER_HEAD:
                ok = signs.ndim == 3 and signs.shape[-1] == self.head_dim
            elif layout == LAYOUT_PER_LAYER:
                ok = signs.ndim == 2 and signs.shape[-1] % self.head_dim == 0
            else:
                raise ValueError(f"unknown layout {layout!r} for site {site}")
            if not ok:
                raise ValueError(f"site {site} ({layout}): bad signs shape "
                                 f"{tuple(signs.shape)} for head_dim={self.head_dim}")
            self._signs[site] = signs
            self._perm[site] = perm
            self._layout[site] = layout

            signs_out_raw = raw.get("signs_out")
            perm_out_raw = raw.get("perm_out")
            if (signs_out_raw is None) != (perm_out_raw is None):
                raise ValueError(
                    f"site {site}: signs_out and perm_out must both be present "
                    f"or both absent")
            if signs_out_raw is None:
                continue
            signs_out = signs_out_raw.to(torch.float32)
            perm_out = perm_out_raw.to(torch.long)
            if signs_out.shape != signs.shape or perm_out.shape != perm.shape:
                raise ValueError(
                    f"site {site}: signs_out{tuple(signs_out.shape)}/"
                    f"perm_out{tuple(perm_out.shape)} must match "
                    f"signs{tuple(signs.shape)}/perm{tuple(perm.shape)}")
            self._signs_out[site] = signs_out
            self._perm_out[site] = perm_out

    def _parse_dense_site(self, site: str, raw: dict, declared_layouts: dict) -> None:
        """Parse and validate a dense site entry (``{"matrix": R}``)."""
        R = raw["matrix"].to(torch.float32)
        layout = declared_layouts.get(site)
        if layout is None:
            layout = LAYOUT_DENSE_PER_HEAD if R.ndim == 4 else LAYOUT_DENSE_PER_LAYER
        if layout == LAYOUT_DENSE_PER_HEAD:
            if R.ndim != 4 or R.shape[-1] != self.head_dim or R.shape[-2] != self.head_dim:
                raise ValueError(
                    f"site {site} (dense_per_head): expected "
                    f"(num_layers, num_kv_heads, {self.head_dim}, {self.head_dim}), "
                    f"got matrix{tuple(R.shape)}")
        elif layout == LAYOUT_DENSE_PER_LAYER:
            if R.ndim != 3 or R.shape[-1] != R.shape[-2]:
                raise ValueError(
                    f"site {site} (dense_per_layer): expected square "
                    f"(num_layers, F, F), got matrix{tuple(R.shape)}")
            if R.shape[-1] % self.head_dim != 0:
                raise ValueError(
                    f"site {site} (dense_per_layer): F={R.shape[-1]} is not a "
                    f"multiple of head_dim={self.head_dim}")
        else:
            raise ValueError(f"unknown dense layout {layout!r} for site {site}")
        self._matrix[site] = R.contiguous()
        self._layout[site] = layout

    def _slice(self, store: dict, site: str, layer_idx: int) -> Optional[tuple]:
        if site not in store[0]:
            return None
        if self._layout[site] == LAYOUT_SHARED:
            return store[0][site], store[1][site]
        return store[0][site][layer_idx], store[1][site][layer_idx]

    def get_layer(self, site: str, layer_idx: int) -> Optional[tuple]:
        """``(signs, perm)`` of one layer: ``(head_dim,)`` shared,
        ``(num_kv_heads, head_dim)`` per_head, ``(num_kv_heads*head_dim,)``
        per_layer; ``None`` for a missing (or dense) site."""
        return self._slice((self._signs, self._perm), site, layer_idx)

    def get_layer_matrix(self, site: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Dense R of one layer: ``(num_kv_heads, head_dim, head_dim)`` or
        ``(F, F)``; ``None`` for a non-dense site."""
        if site not in self._matrix:
            return None
        return self._matrix[site][layer_idx]

    def layout(self, site: str) -> Optional[str]:
        return self._layout.get(site)

    def has_output_side(self, site: str) -> bool:
        return site in self._signs_out

    def get_layer_out(self, site: str, layer_idx: int) -> Optional[tuple]:
        """``(signs_out, perm_out)`` of one layer, shaped like ``get_layer``;
        ``None`` if the site has no output side."""
        return self._slice((self._signs_out, self._perm_out), site, layer_idx)

    def check_head_dim(self, model_head_dim: int) -> None:
        if self.head_dim != model_head_dim:
            raise ValueError(
                f"bundle head_dim={self.head_dim} but model head_dim={model_head_dim}")


def load_bundle(path: str) -> KVBundle:
    return KVBundle(torch.load(path, map_location="cpu", weights_only=False))


def save_bundle(path: str, payload: dict) -> None:
    torch.save(payload, path)


def make_random_bundle(head_dim: int, seed: int, num_layers: Optional[int] = None,
                       num_kv_heads: Optional[int] = None) -> dict:
    """RHT (randomized Hadamard transform, as in QuaRot https://arxiv.org/abs/2404.00456)
    bundle: Rademacher signs, identity perm, the same for all three
    sites. Shared (``num_layers=None``): one ``(head_dim,)`` vector.
    Per-layer: one ``(num_kv_heads*head_dim,)`` vector copied to every layer."""
    shared = num_layers is None
    dim = head_dim if shared else num_kv_heads * head_dim
    g = torch.Generator()
    g.manual_seed(seed)
    signs = (torch.randint(0, 2, (dim,), generator=g, dtype=torch.int64) * 2 - 1).to(torch.float32)
    perm = torch.arange(dim, dtype=torch.long)
    if shared:
        meta = {"head_dim": head_dim, "kind": "random", "seed": seed,
                "convention": CONVENTION, "layout_kind": "shared",
                "layouts": {s: LAYOUT_SHARED for s in ALL_SITES},
                "note": "Rademacher D shared across K (pre/post-RoPE) and V; P=I."}
    else:
        signs = signs.unsqueeze(0).expand(num_layers, dim).contiguous()
        perm = perm.unsqueeze(0).expand(num_layers, dim).contiguous()
        meta = {"head_dim": head_dim, "num_layers": num_layers,
                "num_kv_heads": num_kv_heads, "full_dim": dim, "kind": "random",
                "seed": seed, "convention": CONVENTION, "layout_kind": "pl",
                "layouts": {s: LAYOUT_PER_LAYER for s in ALL_SITES},
                "note": ("Rademacher D over the flat (num_kv_heads*head_dim) "
                         "channel axis, shared across all layers and across "
                         "K (pre/post-RoPE) and V; P=I. Models with the same "
                         "full_dim and seed produce identical D.")}
    entry = {"signs": signs, "perm": perm}
    return {"meta": meta, **{s: {k: v.clone() for k, v in entry.items()} for s in ALL_SITES}}


def make_perm_only_pl_bundle(
    head_dim: int, num_layers: int, num_kv_heads: int, *,
    kind: str,
    per_site_perms: dict,
    per_site_perm_out: Optional[dict] = None,
    note: str = "",
    extra_meta: Optional[dict] = None,
) -> dict:
    """Permutation-only per-layer bundle (MassDiff from PeRQ
    https://arxiv.org/abs/2601.22347, RotateKV https://arxiv.org/abs/2501.16383): signs = +1 and
    one global cross-head permutation per (site, layer), shape
    ``(num_layers, num_kv_heads*head_dim)``, gather convention
    ``out[..., p] = x[..., perm[p]]``.

    ``per_site_perms`` are applied before the Hadamard (missing site =
    identity). ``per_site_perm_out`` are applied after it, with
    ``signs_out = +1``; evaluate those with ``--gha-out-mode dp_out``. The
    Hadamard granularity is chosen at eval time (``--hadamard-granularity``).
    """
    flat_dim = num_kv_heads * head_dim
    ident = torch.arange(flat_dim, dtype=torch.long).unsqueeze(0).expand(
        num_layers, flat_dim).contiguous()
    ones = torch.ones(num_layers, flat_dim, dtype=torch.float32)
    per_site_perm_out = per_site_perm_out or {}

    def _site_entry(site: str) -> dict:
        perm = per_site_perms.get(site)
        perm = ident.clone() if perm is None else perm.to(torch.long).contiguous()
        if perm.shape != (num_layers, flat_dim):
            raise ValueError(
                f"site {site}: perm shape {tuple(perm.shape)} != "
                f"({num_layers}, {flat_dim})")
        entry = {"signs": ones.clone(), "perm": perm}
        perm_out = per_site_perm_out.get(site)
        if perm_out is not None:
            perm_out = perm_out.to(torch.long).contiguous()
            if perm_out.shape != (num_layers, flat_dim):
                raise ValueError(
                    f"site {site}: perm_out shape {tuple(perm_out.shape)} != "
                    f"({num_layers}, {flat_dim})")
            entry["perm_out"] = perm_out
            entry["signs_out"] = ones.clone()   # +1: no sign flip on output side
        return entry

    meta = {
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "full_dim": flat_dim,
        "kind": kind,
        "convention": CONVENTION,
        "layout_kind": "pl",
        "layouts": {s: LAYOUT_PER_LAYER for s in ALL_SITES},
        "note": note,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "meta": meta,
        SITE_K_PRE_ROPE: _site_entry(SITE_K_PRE_ROPE),
        SITE_K_POST_ROPE: _site_entry(SITE_K_POST_ROPE),
        SITE_V: _site_entry(SITE_V),
    }


def make_dense_r_bundle(
    head_dim: int, num_layers: int, num_kv_heads: int, *,
    layout: str,
    per_site_matrix: dict,
    kind: str,
    note: str = "",
    extra_meta: Optional[dict] = None,
) -> dict:
    """Dense orthogonal-R bundle (DFRot, ButterflyQuant, GSR).
    ``per_site_matrix[site]`` is ``(num_layers, num_kv_heads, head_dim,
    head_dim)`` for ``dense_per_head`` or ``(num_layers, F, F)`` for
    ``dense_per_layer``; sites not given are left out (plain RTN)."""
    if layout not in DENSE_LAYOUTS:
        raise ValueError(f"make_dense_r_bundle: layout {layout!r} not a dense layout")
    flat_dim = num_kv_heads * head_dim
    if layout == LAYOUT_DENSE_PER_HEAD:
        want = (num_layers, num_kv_heads, head_dim, head_dim)
    else:
        want = (num_layers, flat_dim, flat_dim)

    payload_sites = {}
    for site in ALL_SITES:
        R = per_site_matrix.get(site)
        if R is None:
            continue
        R = R.to(torch.float32).contiguous()
        if tuple(R.shape) != want:
            raise ValueError(
                f"site {site}: matrix shape {tuple(R.shape)} != expected {want}")
        payload_sites[site] = {"matrix": R}

    meta = {
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "full_dim": flat_dim,
        "kind": kind,
        "convention": CONVENTION_DENSE,
        "layout_kind": "dense_ph" if layout == LAYOUT_DENSE_PER_HEAD else "dense_pl",
        "layouts": {s: layout for s in payload_sites},
        "note": note,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"meta": meta, **payload_sites}
