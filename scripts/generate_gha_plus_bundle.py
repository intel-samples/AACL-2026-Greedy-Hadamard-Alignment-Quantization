"""Generate the output-side variants of a per-head GHA bundle (paper Table 2).

With ``y = H(rep[perm] * signs)`` per head (GHA's input side):
  gha_plus_dout_pl   — output D_o = sign(y), P_o = I.
  gha_plus_dpout_pl  — output D_o = sign(y), P_o = argsort(|y|, descending)
                       over all heads (K and V).
  gha_reorder_pl     — output D_o = I; K gets RotateKV's reordering
                       (https://arxiv.org/abs/2501.16383) in GHA's
                       space (signed channel sum of y, ascending, over all
                       heads); V keeps GHA's per-head P_o.
A permutation within one head cannot change a per-token, head-sized quant
group, so the reorders act across heads. The bundles are per-layer: per-head
tensors are flattened head-major (block-diagonal, exact with
``--hadamard-granularity per_head``). Eval with ``--gha-out-mode dp_out
--hadamard-granularity per_head --k-scaling per_token``.

The model, calibration windows and rep are those recorded in the source
bundle's meta.

Output: bundles/{gha_plus_dout_pl,gha_plus_dpout_pl}_<model>__wikitext2__<rep>__seed<N>.pt
        bundles/gha_reorder_pl_<model>__wikitext2__sumsigned__seed<N>.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from generate_gha_bundle import (
    CALIB, SITE_OF, _reduce_all, bundle_path, capture_kv, check_perm,
    flatten_heads, flatten_perm,
)
from generate_perm_bundle import _reduce_metric
import kv_bundle  # noqa: E402
import kv_quant  # noqa: E402


def _gha_y(rep_HD: torch.Tensor, signs: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """``(H, D)`` float64 ``y = H(rep[perm] * signs)`` per head for one layer."""
    x = torch.gather(rep_HD.to(torch.float64), -1, perm.to(torch.long)) * signs.to(torch.float64)
    return torch.matmul(x, kv_quant._hadamard_matrix(rep_HD.shape[-1], "cpu", "float64").t())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gha-bundle", required=True,
                    help="per-head GHA bundle, e.g. bundles/gha_ph_<model>__wikitext2__med__seed0.pt")
    ap.add_argument("--out-dir", default="bundles")
    args = ap.parse_args()

    gha = torch.load(args.gha_bundle, map_location="cpu", weights_only=False)
    gm = gha["meta"]
    if gm.get("layout_kind") != "ph":
        raise ValueError(f"--gha-bundle must be per-head; got layout_kind={gm.get('layout_kind')!r}")
    model, rep, seed = gm["model"], gm["rep"], gm["seed"]
    out = {
        "gha_plus_dout": bundle_path(args.out_dir, "gha_plus_dout_pl", model, rep, seed),
        "gha_plus_dpout": bundle_path(args.out_dir, "gha_plus_dpout_pl", model, rep, seed),
        "gha_reorder": bundle_path(args.out_dir, "gha_reorder_pl", model,
                                   ("" if rep == "med" else f"{rep}_") + "sumsigned", seed),
    }
    if all(os.path.exists(p) for p in out.values()):
        print("all bundles already exist")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    sinks, dims = capture_kv(model, gm["seqlen"], seed, nsamples=gm["nsamples"])
    L, H, D = dims
    if dims != (gm["num_layers"], gm["num_kv_heads"], gm["head_dim"]):
        raise ValueError(f"model dims {dims} do not match the GHA bundle")
    F = H * D
    reps = _reduce_all(sinks, [rep], clear=False)[rep]
    sums = _reduce_metric(sinks, "sum_signed")
    del sinks

    abs_desc, sum_asc = {}, {}
    for cap, site in SITE_OF.items():
        g = gha[site]
        abs_desc[site] = torch.stack([_gha_y(reps[cap][li], g["signs"][li], g["perm"][li])
                                      .abs().reshape(F).argsort(descending=True)
                                      for li in range(L)])
        if site != kv_bundle.SITE_V:
            sum_asc[site] = torch.stack([_gha_y(sums[cap][li], g["signs"][li], g["perm"][li])
                                         .reshape(F).argsort() for li in range(L)])

    identity = torch.arange(F, dtype=torch.long).expand(L, F).contiguous()
    perm_out = {
        "gha_plus_dout": {s: identity.clone() for s in kv_bundle.ALL_SITES},
        "gha_plus_dpout": abs_desc,
        "gha_reorder": {**sum_asc, kv_bundle.SITE_V: flatten_perm(gha[kv_bundle.SITE_V]["perm_out"])},
    }
    notes = {
        "gha_plus_dout": "GHA input D·P, output D_o = sign(y), P_o = I.",
        "gha_plus_dpout": "GHA input D·P, output D_o = sign(y), P_o = cross-head |y| descending.",
        "gha_reorder": ("GHA input D·P, output D_o = I, K: cross-head signed-sum ascending "
                        "reorder of y; V: GHA's per-head P_o."),
    }
    for kind, path in out.items():
        if os.path.exists(path):
            print(f"exists, skipped: {path}")
            continue
        meta = {
            "head_dim": D, "num_layers": L, "num_kv_heads": H, "full_dim": F, "model": model,
            "kind": kind, "calib": CALIB, "rep": "sumsigned" if kind == "gha_reorder" else rep,
            **({"source_rep": rep} if kind == "gha_reorder" else {}),
            "nsamples": gm["nsamples"], "seqlen": gm["seqlen"], "seed": seed,
            "has_output_side": True, "convention": kv_bundle.CONVENTION, "layout_kind": "pl",
            "layouts": {s: kv_bundle.LAYOUT_PER_LAYER for s in kv_bundle.ALL_SITES},
            "note": notes[kind] + " Eval with --gha-out-mode dp_out --hadamard-granularity per_head.",
            "source_gha_bundle": os.path.basename(args.gha_bundle),
        }
        payload = {"meta": meta}
        for site in kv_bundle.ALL_SITES:
            g = gha[site]
            payload[site] = {
                "signs": flatten_heads(g["signs"]),
                "perm": flatten_perm(g["perm"]),
                "signs_out": (torch.ones(L, F, dtype=torch.float32) if kind == "gha_reorder"
                              else flatten_heads(g["signs_out"])),
                "perm_out": perm_out[kind][site],
            }
            check_perm(f"{kind}/{site}/perm_out", payload[site]["perm_out"])
        kv_bundle.save_bundle(path, payload)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
