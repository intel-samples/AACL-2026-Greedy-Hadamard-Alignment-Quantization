"""Generate an RHT bundle (randomized Hadamard transform, as in QuaRot
https://arxiv.org/abs/2404.00456): Rademacher signs, identity perm.

* ``shared`` (RHT-PH): one ``(head_dim,)`` vector for every layer, head and
  site; model-agnostic.
* ``pl`` (RHT-PL): one ``(num_kv_heads*head_dim,)`` vector, copied to every
  layer; dimensions are read from the model config.

Usage:
    python scripts/generate_random_bundle.py --head-dim 128 --seed 0 \\
        --out bundles/rh_sh_d128__seed0.pt
    python scripts/generate_random_bundle.py --layout pl --model Qwen/Qwen3-4B \\
        --seed 0 --out bundles/rh_pl_Qwen3-4B__seed0.pt
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from kv_bundle import make_random_bundle, save_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="shared", choices=["shared", "pl"])
    ap.add_argument("--head-dim", type=int, help="required for --layout shared")
    ap.add_argument("--model", help="HF id; required for --layout pl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.layout == "shared":
        if args.head_dim is None:
            ap.error("--head-dim required for --layout shared")
        payload = make_random_bundle(args.head_dim, args.seed)
    else:
        if args.model is None:
            ap.error("--model required for --layout pl")
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
        path = (args.model if os.path.isdir(args.model)
                else snapshot_download(args.model, local_files_only=True))
        cfg = AutoConfig.from_pretrained(path)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        payload = make_random_bundle(int(head_dim), args.seed,
                                     int(cfg.num_hidden_layers), int(num_kv_heads))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_bundle(args.out, payload)
    print(f"wrote {args.out}  (layout={args.layout}, seed={args.seed}, "
          f"signs {tuple(payload['k_pre_rope']['signs'].shape)})")


if __name__ == "__main__":
    main()
