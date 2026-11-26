"""Generate MassDiff+GHA bundles: the MassDiff permutation P_m (PeRQ,
https://arxiv.org/abs/2601.22347) decides which
channels form each head-sized block, then GHA is refit inside every new block.

Per site and layer, with the per-channel mean |x| representative x:
  1. P_m = MassDiff(x)                       (global, across heads)
  2. per-head GHA on the reblocked u = x[P_m] -> (P_g, signs, signs_out, perm_out)
  3. input perm = P_m[flatten(P_g)], so x[perm] = u[flatten(P_g)].

Every site is a per-layer bundle; with ``--hadamard-granularity per_head`` the
runtime applies the composed perm and signs, then one Hadamard per head-sized
block. Eval with ``--gha-out-mode none``.

Output: bundles/massdiff_gha_pl_<model>__wikitext2__meanabs__seed<N>.pt
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from generate_gha_bundle import (
    ANCHOR, CALIB, NSAMPLES, SITE_OF, _fit_site, add_common_args, bundle_path,
    capture_kv, check_perm, flatten_heads, flatten_perm,
)
from generate_perm_bundle import _reduce_metric, massdiff_perm
import kv_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()

    out_path = bundle_path(args.out_dir, "massdiff_gha_pl", args.model, "meanabs", args.seed)
    if os.path.exists(out_path):
        print(f"bundle already exists; skipping: {out_path}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    sinks, (L, H, D) = capture_kv(args.model, args.seqlen, args.seed)
    reps = _reduce_metric(sinks, "mean_abs")          # {cap_site: (L, H, D)} float64
    del sinks
    fit_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    payload = {"meta": {
        "head_dim": D, "num_layers": L, "num_kv_heads": H, "full_dim": H * D,
        "model": args.model, "kind": "massdiff_gha", "calib": CALIB, "rep": "meanabs",
        "metric": "mean_abs", "block_size": D, "nsamples": NSAMPLES,
        "seqlen": args.seqlen, "seed": args.seed, "anchor": ANCHOR,
        "has_output_side": True, "convention": kv_bundle.CONVENTION, "layout_kind": "pl",
        "layouts": {s: kv_bundle.LAYOUT_PER_LAYER for s in kv_bundle.ALL_SITES},
        "note": ("MassDiff reblock P_m composed with a per-block GHA refit: perm = "
                 "P_m[P_g], signs = GHA signs. Eval with --gha-out-mode none "
                 "--hadamard-granularity per_head."),
    }}
    t0 = time.perf_counter()
    for cap, site in SITE_OF.items():
        p_m = massdiff_perm(reps[cap])                                       # (L, F)
        u = torch.gather(reps[cap].reshape(L, H * D), -1, p_m).reshape(L, H, D)
        perm_g, signs_g, signs_out_g, perm_out_g = _fit_site(
            u.to(torch.float32), fit_dev, tag=f"{site}/massdiff_gha")
        payload[site] = {
            "signs": flatten_heads(signs_g),
            "perm": torch.gather(p_m, -1, flatten_perm(perm_g)).contiguous(),
            "signs_out": flatten_heads(signs_out_g),
            "perm_out": flatten_perm(perm_out_g),
        }
        check_perm(f"{site}/perm", payload[site]["perm"])
    print(f"  all sites done in {time.perf_counter() - t0:.1f}s")
    kv_bundle.save_bundle(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
