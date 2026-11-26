"""Generate GHA bundles (per-head and per-layer) for K (pre/post-RoPE) and V.

1. Sample 128 random ``seqlen``-token windows of WikiText-2 train.
2. Run the model and capture K before / after RoPE and V at every layer.
3. Reduce the captured tokens of each (site, layer) to one per-channel
   representative vector per ``--reps`` entry:
     med  — median over all tokens;
     mean — mean over all tokens;
     medm — per-batch median, then median over batches;
     ext  — per-batch signed value of maximum magnitude, then median over batches.
4. Fit GHA on each vector (per head, or on the flattened layer vector), giving
   ``(perm, signs)`` and the output side ``signs_out = sign(y)``,
   ``perm_out = argsort(|y|, descending)`` for ``y = H(rep[perm] * signs)``.

Output: bundles/gha_{ph,pl}_<model>__wikitext2__<rep>__seed<N>.pt

This module also provides the capture / naming helpers shared by the other
generate_*_bundle.py scripts.
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import time
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import datasets  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

import gha as gha_mod  # noqa: E402
import kv_bundle  # noqa: E402

SEED = 0
NSAMPLES = 128
ANCHOR = 1024                     # GHA max_search_range anchor
CALIB = "wikitext2"
REPS = ("med", "mean", "medm", "ext")
SITES = ("k_pre", "k_post", "v")                   # capture sites
SITE_OF = dict(zip(SITES, kv_bundle.ALL_SITES))    # capture site -> bundle site


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqlen", type=int, default=4096,
                    help="calibration window length (default 4096)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="calibration window-sampling seed (paper: 0-9)")
    ap.add_argument("--out-dir", default="bundles")


def model_short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def bundle_path(out_dir: str, stem: str, model_id: str, tag: str, seed: int) -> str:
    return os.path.join(out_dir, f"{stem}_{model_short(model_id)}__{CALIB}__{tag}__seed{seed}.pt")


def model_dims(cfg):
    """``(num_layers, num_kv_heads, head_dim)`` of a model config."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return int(cfg.num_hidden_layers), int(num_kv_heads), int(head_dim)


def flatten_heads(t: torch.Tensor) -> torch.Tensor:
    """``(L, H, D)`` -> ``(L, H*D)``, head-major (channel ``h*D + d``)."""
    L, H, D = t.shape
    return t.reshape(L, H * D).contiguous()


def flatten_perm(perm: torch.Tensor) -> torch.Tensor:
    """Per-head perm ``(L, H, D)`` -> block-diagonal flat perm ``(L, H*D)``:
    ``flat[h*D + d] = h*D + perm[h, d]``."""
    L, H, D = perm.shape
    offset = torch.arange(H, dtype=torch.long).view(1, H, 1) * D
    return (perm.to(torch.long) + offset).reshape(L, H * D).contiguous()


def check_perm(name: str, perm: torch.Tensor) -> None:
    """Raise unless every row of ``perm`` is a permutation of ``0..n-1``."""
    ref = torch.arange(perm.shape[-1], dtype=perm.dtype).expand_as(perm)
    if not torch.equal(perm.sort(dim=-1).values, ref):
        raise ValueError(f"{name}: not a permutation along the last dim")


def _load_calib(nsamples: int, seqlen: int, seed: int, tokenizer) -> List[torch.Tensor]:
    """``nsamples`` random ``(1, seqlen)`` windows of tokenized WikiText-2 train."""
    traindata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(seed)
    out = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        out.append(enc.input_ids[:, i:i + seqlen])
    return out


def _install_capture(model, num_layers: int, num_kv_heads: int, head_dim: int):
    """Hook K (around ``apply_rotary_pos_emb``) and V (``v_proj`` output) of every
    layer. Returns ``(sinks, restore)``; ``sinks[site][layer]`` collects one
    ``(S, H, D)`` float32 CPU tensor per forward."""
    sinks = {s: {li: [] for li in range(num_layers)} for s in SITES}
    cur_layer = [0]
    restores = []

    def accum(site: str, li: int, x: torch.Tensor):
        x = x.detach()
        if x.dim() == 4:                      # K: (1, H, S, D) -> (S, H, D)
            x = x.squeeze(0).permute(1, 0, 2).contiguous()
        else:                                 # V: (1, S, H*D) -> (S, H, D)
            x = x.squeeze(0).reshape(-1, num_kv_heads, head_dim)
        sinks[site][li].append(x.to("cpu", torch.float32))

    for mod_name in dict.fromkeys(type(l.self_attn).__module__ for l in model.model.layers):
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "apply_rotary_pos_emb"):
            continue
        orig = mod.apply_rotary_pos_emb

        def wrapped(q, k, *args, _orig=orig, **kwargs):
            if k.shape[-2] > 1:
                accum("k_pre", cur_layer[0], k)
            q_out, k_out = _orig(q, k, *args, **kwargs)
            if k_out.shape[-2] > 1:
                accum("k_post", cur_layer[0], k_out)
            return q_out, k_out

        mod.apply_rotary_pos_emb = wrapped
        restores.append(lambda m=mod, o=orig: setattr(m, "apply_rotary_pos_emb", o))

    for li, layer in enumerate(model.model.layers):
        def set_layer(_m, _args, _li=li):
            cur_layer[0] = _li

        def v_hook(_m, _inp, output, _li=li):
            if output.shape[-2] > 1:
                accum("v", _li, output)

        restores.append(layer.self_attn.register_forward_pre_hook(set_layer).remove)
        restores.append(layer.self_attn.v_proj.register_forward_hook(v_hook).remove)

    def restore():
        for fn in restores:
            fn()

    return sinks, restore


def capture_kv(model_id: str, seqlen: int, seed: int, nsamples: int = None):
    """Load the model, run ``nsamples`` calibration windows (default
    ``NSAMPLES``) and return ``(sinks, (num_layers, num_kv_heads, head_dim))``."""
    nsamples = NSAMPLES if nsamples is None else nsamples
    print(f"loading model {model_id}")
    path = model_id if os.path.isdir(model_id) else snapshot_download(model_id, local_files_only=True)
    tok = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype="auto", device_map="auto", attn_implementation="sdpa").eval()
    dims = model_dims(model.config)
    print(f"  L, H_kv, D = {dims}; calib {CALIB}: {nsamples} x {seqlen} tokens, seed {seed}")
    batches = _load_calib(nsamples, seqlen, seed, tok)
    sinks, restore = _install_capture(model, *dims)
    try:
        for bi, ids in enumerate(batches):
            t = time.perf_counter()
            with torch.inference_mode():
                model(input_ids=ids.to(model.device), use_cache=False)
            print(f"  batch {bi + 1}/{len(batches)} in {time.perf_counter() - t:.1f}s", flush=True)
    finally:
        restore()
    del model
    torch.cuda.empty_cache()
    return sinks, dims


# ---------------------------------------------------------------------------
# Reduction and fit
# ---------------------------------------------------------------------------

def _reduce_all(sinks, reps: List[str], clear: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
    """``{rep: {site: (L, H, D)}}`` per-channel representatives."""
    out: Dict[str, Dict[str, torch.Tensor]] = {r: {} for r in reps}
    for site in SITES:
        per_layer = {r: [] for r in reps}
        for batches in sinks[site].values():           # list of (S, H, D)
            cat = None
            for r in reps:
                if r == "medm":
                    v = torch.stack([b.median(dim=0).values for b in batches]).median(dim=0).values
                elif r == "ext":
                    v = torch.stack([b.gather(0, b.abs().argmax(dim=0, keepdim=True)).squeeze(0)
                                     for b in batches]).median(dim=0).values
                else:
                    cat = torch.cat(batches, dim=0) if cat is None else cat
                    v = cat.median(dim=0).values if r == "med" else cat.mean(dim=0)
                per_layer[r].append(v)
            if clear:
                batches.clear()
        for r in reps:
            out[r][site] = torch.stack(per_layer[r])
    return out


def _fit_site(rep: torch.Tensor, device, tag: str = ""):
    """Fit GHA on every vector along the last dim of ``rep`` (``(L, H, D)`` per
    head or ``(L, F)`` per layer). Returns ``(perm, signs, signs_out,
    perm_out)``, each shaped like ``rep``."""
    d = rep.shape[-1]
    rows = rep.reshape(-1, d)
    H_mat = gha_mod._build_sylvester_hadamard(d, device=device, dtype=torch.float32)
    msr = gha_mod.pick_max_search_range(d, anchor=ANCHOR)
    perm = torch.empty(rows.shape, dtype=torch.int64)
    signs = torch.empty(rows.shape, dtype=torch.float32)
    signs_out = torch.empty(rows.shape, dtype=torch.float32)
    perm_out = torch.empty(rows.shape, dtype=torch.int64)
    t0 = time.perf_counter()
    for i in range(rows.shape[0]):
        x = rows[i].to(device=device, dtype=torch.float32)
        p, s = gha_mod.gha(x, h=H_mat, max_search_range=msr)
        y = H_mat @ (x.index_select(0, p) * s)
        d_o = torch.sign(y)
        perm[i] = p.cpu()
        signs[i] = s.cpu()
        signs_out[i] = torch.where(d_o == 0, torch.ones_like(d_o), d_o).cpu()
        perm_out[i] = y.abs().argsort(descending=True).cpu()
    print(f"  fit {tag}: {tuple(rep.shape)} msr={msr} in {time.perf_counter() - t0:.1f}s")
    return tuple(t.reshape(rep.shape) for t in (perm, signs, signs_out, perm_out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--reps", default=",".join(REPS),
                    help=f"comma-separated subset of {REPS} (default: all)")
    args = ap.parse_args()
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    if not reps or any(r not in REPS for r in reps):
        ap.error(f"--reps must be a non-empty subset of {REPS}")

    outputs = {(layout, rep): bundle_path(args.out_dir, f"gha_{layout}", args.model, rep, args.seed)
               for layout in ("ph", "pl") for rep in reps}
    todo = {k: p for k, p in outputs.items() if not os.path.exists(p)}
    if not todo:
        print("all requested bundles already exist")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    sinks, (L, H, D) = capture_kv(args.model, args.seqlen, args.seed)
    rep_pool = _reduce_all(sinks, sorted({rep for _, rep in todo}, key=reps.index))
    del sinks
    fit_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    for (layout, rep), out_path in todo.items():
        layout_tag = kv_bundle.LAYOUT_PER_HEAD if layout == "ph" else kv_bundle.LAYOUT_PER_LAYER
        payload = {"meta": {
            "head_dim": D, "num_layers": L, "num_kv_heads": H, "model": args.model,
            "kind": "gha", "calib": CALIB, "rep": rep, "nsamples": NSAMPLES,
            "seqlen": args.seqlen, "seed": args.seed, "anchor": ANCHOR,
            "has_output_side": True, "convention": kv_bundle.CONVENTION,
            "layout_kind": layout, "layouts": {s: layout_tag for s in kv_bundle.ALL_SITES},
        }}
        for cap, site in SITE_OF.items():
            r = rep_pool[rep][cap]
            p, s, so, po = _fit_site(r if layout == "ph" else r.reshape(L, H * D),
                                     fit_dev, tag=f"{site}/{layout}/{rep}")
            payload[site] = {"signs": s, "perm": p, "signs_out": so, "perm_out": po}
        kv_bundle.save_bundle(out_path, payload)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
