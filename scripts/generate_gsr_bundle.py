"""Generate GSR (grouped sequency-arranged rotation, Choi et al., ACL SRW 2025,
https://arxiv.org/abs/2505.03810) bundles.

The rotation is block-diagonal with 128-wide Walsh blocks; a Walsh matrix is
the normalized Sylvester-Hadamard matrix with rows sorted by sequency (number
of sign changes along the row). Stored as dense ``R = Wᵀ`` for ``y = x @ R``.
Deterministic: only the model config is read.

Output: bundles/gsr_{ph,pl}_<model>__fixed__g128.pt
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from generate_gha_bundle import model_dims, model_short  # noqa: E402
import kv_bundle  # noqa: E402
import kv_quant  # noqa: E402

GROUP_SIZE = 128


def walsh_matrix(n: int) -> torch.Tensor:
    """Normalized Walsh matrix: Hadamard rows in ascending-sequency order."""
    H = kv_quant._hadamard_matrix(n, "cpu", "float32")
    sequency = (H[:, 1:] * H[:, :-1] < 0).sum(dim=1)
    return H[sequency.argsort(stable=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", required=True, choices=["per_head", "per_layer"])
    ap.add_argument("--out-dir", default="bundles")
    args = ap.parse_args()

    lay = "ph" if args.layout == "per_head" else "pl"
    out_path = os.path.join(args.out_dir, f"gsr_{lay}_{model_short(args.model)}__fixed__g{GROUP_SIZE}.pt")
    if os.path.exists(out_path):
        print(f"bundle already exists; skipping: {out_path}")
        return

    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model, local_files_only=True)
    L, H, D = model_dims(AutoConfig.from_pretrained(path))
    n = D if lay == "ph" else H * D
    R = torch.block_diag(*[walsh_matrix(GROUP_SIZE).t()] * (n // GROUP_SIZE)).contiguous()
    if lay == "ph":
        shape, layout = (L, H, D, D), kv_bundle.LAYOUT_DENSE_PER_HEAD
    else:
        shape, layout = (L, n, n), kv_bundle.LAYOUT_DENSE_PER_LAYER
    payload = kv_bundle.make_dense_r_bundle(
        D, L, H, layout=layout, kind="gsr",
        per_site_matrix={s: R.expand(shape).clone() for s in kv_bundle.ALL_SITES},
        note=("Fixed grouped sequency-arranged rotation: block-diagonal Walsh "
              "matrices, stored as dense R = W.T. No calibration."),
        extra_meta={"model": args.model, "transform": "walsh_sequency",
                    "group_size": GROUP_SIZE, "calibration": "none", "matrix_shape": shape})
    os.makedirs(args.out_dir, exist_ok=True)
    kv_bundle.save_bundle(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
