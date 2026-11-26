"""Generate permutation-only per-layer bundles (signs = +1) for the MassDiff and
RotateKV baselines. Both act on the flattened ``num_kv_heads * head_dim`` axis
and are evaluated with ``--hadamard-granularity per_head``.

  massdiff — PeRQ's MassDiff permutation (https://arxiv.org/abs/2601.22347, Algorithm 1),
    applied here to K and V with block size = head_dim (the paper uses it at
    the FFN down-projection input). Channels are taken in descending order of
    mean |x| over the calibration tokens and each goes to the non-full block
    with the smallest current l1 mass. Input-side ``perm``; eval with
    ``--gha-out-mode none``. Tag ``meanabs``.
  rotatekv — RotateKV's key reordering (https://arxiv.org/abs/2501.16383):
    channels sorted ascending by the signed
    channel sum after the per-head Hadamard (the sum is reduced first and then
    rotated, which is equivalent since the Hadamard is linear). K only, stored
    as output-side ``perm_out``; eval with ``--gha-out-mode dp_out``. Tag
    ``sumsigned``.

Output: bundles/{massdiff,rotatekv}_pl_<model>__wikitext2__<tag>__seed<N>.pt
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import torch

from generate_gha_bundle import (
    CALIB, NSAMPLES, SITES, add_common_args, bundle_path, capture_kv, check_perm,
)
import kv_bundle  # noqa: E402
import kv_quant  # noqa: E402


def _reduce_metric(sinks, metric: str) -> Dict[str, torch.Tensor]:
    """``{site: (L, H, D)}`` float64 per-channel signed sum (``sum_signed``) or
    mean |x| (``mean_abs``) over all captured tokens; clears ``sinks``."""
    out = {}
    for site in SITES:
        per_layer = []
        for batches in sinks[site].values():
            cat = torch.cat(batches, dim=0).to(torch.float64)       # (T, H, D)
            if metric == "sum_signed":
                per_layer.append(cat.sum(dim=0))
            elif metric == "mean_abs":
                per_layer.append(cat.abs().mean(dim=0))
            else:
                raise ValueError(f"unknown metric {metric!r}")
            batches.clear()
        out[site] = torch.stack(per_layer)
    return out


def rotatekv_perm(rep_LHD: torch.Tensor) -> torch.Tensor:
    """``(L, H*D)`` ascending argsort of the per-head-Hadamard-rotated signed sum."""
    L, H, D = rep_LHD.shape
    Hd = kv_quant._hadamard_matrix(D, "cpu", "float64")
    return torch.matmul(rep_LHD.to(torch.float64), Hd.t()).reshape(L, H * D).argsort(dim=-1)


def massdiff_perm(rep_LHD: torch.Tensor) -> torch.Tensor:
    """MassDiff with ``num_kv_heads`` blocks of ``head_dim`` channels.
    ``rep_LHD`` is the per-channel mean |x|; returns an ``(L, H*D)`` perm.

    The paper scores a block by the sample mean of its l1 norm; by linearity
    that equals the sum of the block's per-channel mean |x|, used here. The
    first ``H`` channels seed one block each (the paper's greedy rule on empty
    blocks, ties to the lowest index)."""
    L, H, D = rep_LHD.shape
    F = H * D
    out = torch.empty(L, F, dtype=torch.long)
    rep_LHD = rep_LHD.to(torch.float64)
    for li in range(L):
        mass = rep_LHD[li].reshape(F)
        order = mass.argsort(descending=True).tolist()
        blocks = [[c] for c in order[:H]]
        block_sum = [float(mass[c]) for c in order[:H]]
        for c in order[H:]:
            mc = float(mass[c])
            b = min((i for i in range(H) if len(blocks[i]) < D), key=lambda i: block_sum[i] + mc)
            blocks[b].append(c)
            block_sum[b] += mc
        out[li] = torch.tensor([c for bl in blocks for c in bl], dtype=torch.long)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--method", required=True, choices=["massdiff", "rotatekv"])
    args = ap.parse_args()

    metric, tag = (("mean_abs", "meanabs") if args.method == "massdiff"
                   else ("sum_signed", "sumsigned"))
    out_path = bundle_path(args.out_dir, f"{args.method}_pl", args.model, tag, args.seed)
    if os.path.exists(out_path):
        print(f"bundle already exists; skipping: {out_path}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    sinks, (L, H, D) = capture_kv(args.model, args.seqlen, args.seed)
    reps = _reduce_metric(sinks, metric)
    del sinks

    t0 = time.perf_counter()
    meta = {"metric": metric, "calib": CALIB, "nsamples": NSAMPLES, "seqlen": args.seqlen,
            "seed": args.seed, "model": args.model}
    if args.method == "massdiff":
        payload = kv_bundle.make_perm_only_pl_bundle(
            D, L, H, kind="massdiff",
            per_site_perms={kv_bundle.SITE_K_PRE_ROPE: massdiff_perm(reps["k_pre"]),
                            kv_bundle.SITE_K_POST_ROPE: massdiff_perm(reps["k_post"]),
                            kv_bundle.SITE_V: massdiff_perm(reps["v"])},
            note=("PeRQ MassDiff input-side permutation (block = head_dim), K and V. "
                  "Eval with --gha-out-mode none --hadamard-granularity per_head."),
            extra_meta={**meta, "block_size": D})
    else:
        payload = kv_bundle.make_perm_only_pl_bundle(
            D, L, H, kind="rotatekv", per_site_perms={},
            per_site_perm_out={kv_bundle.SITE_K_PRE_ROPE: rotatekv_perm(reps["k_pre"]),
                               kv_bundle.SITE_K_POST_ROPE: rotatekv_perm(reps["k_post"])},
            note=("RotateKV key reordering (signed channel sum after the per-head "
                  "Hadamard, ascending), output-side, K only. Eval with "
                  "--gha-out-mode dp_out --hadamard-granularity per_head."),
            extra_meta=meta)
    print(f"  perm computed in {time.perf_counter() - t0:.1f}s")
    for site in kv_bundle.ALL_SITES:
        for key in ("perm", "perm_out"):
            if key in payload[site]:
                check_perm(f"{site}/{key}", payload[site][key])
    kv_bundle.save_bundle(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
