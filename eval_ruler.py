"""Minimal RULER-4096 evaluator (RULER: https://arxiv.org/abs/2404.06654), aligned
with the no_press baseline of NVIDIA kvpress (https://github.com/NVIDIA/kvpress).

Prompt and scoring follow kvpress so numbers are directly comparable.

Defaults reproduce the paper's Table 1 setting (2-bit asymmetric RTN, K per_channel,
post-RoPE; pre-RoPE uses sequential prefill). Fixed for every run: V per_token,
block 128, SDPA attention, all 13 RULER tasks.

Usage:
    python eval_ruler.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --kv-bundle bundles/<bundle>.pt --out results/<name>.json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import kv_bundle
import kv_quant

# RULER-4096 test split from the HF hub (resolved through the default HF cache;
# revision pinned so every run reads the same snapshot).
DATA_REPO = "simonjegou/ruler"
DATA_FILE = "4096/test-00000-of-00001.parquet"
DATA_REVISION = "24adceac8a0e6532936e8d721cd9e9084d2e4686"

# Private-use character marking the end of the context inside the rendered
# chat template; it never occurs in RULER text.
CONTEXT_END = "\ue000"

# Fixed evaluation settings (paper Tables 1-2).
KV_BLOCK_SIZE = 128
ATTN_IMPL = "sdpa"


def load_rows(limit=None):
    path = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset",
                           revision=DATA_REVISION)
    df = pq.read_table(path).to_pandas()
    if limit:
        df = df.groupby("task", group_keys=False).head(limit)
    return df.reset_index(drop=True)


def build_prompt(tokenizer, context, question, answer_prefix):
    """Render the context as the user turn of the chat template, then append
    the question, the template's closing / assistant header and answer_prefix,
    so that the context tokens form a prefix shared by every question."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + CONTEXT_END}],
        add_generation_prompt=True, tokenize=False)
    head, _, tail = rendered.partition(CONTEXT_END)

    def ids(text):
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    return torch.cat([ids(head), ids(question + tail + answer_prefix)], dim=1)


def score(task, pred, refs):
    """RULER string match on the cleaned prediction (control characters
    removed): QA tasks score 1 if any reference occurs in it, the other tasks
    score the fraction of references that occur (case-insensitive)."""
    pred = "".join(ch for ch in pred.strip() if ord(ch) >= 32).strip()
    text = pred.lower()
    hits = [str(r).lower() in text for r in refs]
    if not hits:
        return 0.0, pred
    if task.partition("_")[0] == "qa":
        return float(any(hits)), pred
    return sum(hits) / len(hits), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="per-task row cap (quick checks)")
    ap.add_argument("--kv-quant", default="post_rope",
                    choices=["none", "post_rope", "pre_rope"],
                    help="KV cache fake-quant mode. post_rope (default) "
                         "quantizes the prefill past_kv once; pre_rope hooks "
                         "K before RoPE; none disables KV quant entirely.")
    ap.add_argument("--pre-rope-prefill", default="sequential",
                    choices=["sequential", "independent"],
                    help="Only for --kv-quant pre_rope. sequential (default, "
                         "the paper's pre-RoPE setting): K (before RoPE) and V "
                         "are fake-quantized inside the prefill forward, layer "
                         "by layer, so each layer sees inputs perturbed by the "
                         "quantized caches of earlier layers, and prefill "
                         "attention and the first generated token observe the "
                         "quantized KV. independent: run an unquantized "
                         "prefill, then quantize each layer's pre-RoPE K "
                         "(RoPE re-applied) and V once and write them into the "
                         "cache (same flow as post_rope).")
    ap.add_argument("--bits", type=int, default=2,
                    help="KV fake-quant bit width, used for both K and V.")
    ap.add_argument("--k-scaling", default="per_channel",
                    choices=["per_channel", "per_token"],
                    help="K fake-quant grouping. per_channel (default, Table 1) "
                         "groups along the token dim in blocks of 128; "
                         "per_token (Table 2) groups along the channel dim. "
                         "V is always per_token.")
    ap.add_argument("--kv-bundle", default=None,
                    help="Bundle holding the K (k_pre_rope / k_post_rope, "
                         "picked by --kv-quant) and V transforms. Unset = no "
                         "transform (direct quantization).")
    ap.add_argument("--gha-out-mode", default="none",
                    choices=["none", "d_out", "dp_out"],
                    help="Output-side transform. none: T = H·D·P. d_out: "
                         "T = D_o·H·D·P. dp_out: T = P_o·D_o·H·D·P. d_out / "
                         "dp_out need a bundle with signs_out/perm_out.")
    ap.add_argument("--hadamard-granularity", default="full",
                    choices=["full", "per_head"],
                    help="Hadamard block granularity on the per-layer (PL) "
                         "bundle path. full (default): one Hadamard over the "
                         "whole num_kv_heads*head_dim axis. per_head: "
                         "per-head-block (block=head_dim) Hadamard while the "
                         "permutation still acts globally across heads (used "
                         "by the massdiff_pl / rotatekv_pl permutation "
                         "baselines). Ignored on shared/per-head bundle "
                         "layouts (already head-local).")
    args = ap.parse_args()
    if args.pre_rope_prefill != "sequential" and args.kv_quant != "pre_rope":
        ap.error("--pre-rope-prefill independent requires --kv-quant pre_rope")

    rows = load_rows(args.limit)
    print(f"loaded {len(rows)} rows across {rows['task'].nunique()} tasks")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        attn_implementation=ATTN_IMPL,
    )
    model.eval()
    model.generation_config.max_length = None

    print(f"kv_quant={args.kv_quant} pre_rope_prefill={args.pre_rope_prefill} "
          f"bits={args.bits} "
          f"k_scaling={args.k_scaling} "
          f"kv_bundle={args.kv_bundle}")

    head_dim = getattr(model.config, "head_dim", None) or \
        (model.config.hidden_size // model.config.num_attention_heads)
    hadamard_block = head_dim if args.hadamard_granularity == "per_head" else None
    print(f"hadamard_granularity={args.hadamard_granularity} "
          f"(block={hadamard_block})")
    bundle = None
    if args.kv_bundle:
        bundle = kv_bundle.load_bundle(args.kv_bundle)
        bundle.check_head_dim(head_dim)
        print(f"  bundle meta: {bundle.meta}")

    if args.gha_out_mode != "none" and (
            bundle is None or not any(bundle.has_output_side(s) for s in kv_bundle.ALL_SITES)):
        ap.error(f"--gha-out-mode {args.gha_out_mode} needs a bundle with signs_out/perm_out")

    quant_kw = dict(bits=args.bits, block_size=KV_BLOCK_SIZE, bundle=bundle,
                    out_mode=args.gha_out_mode, k_mode=args.k_scaling,
                    hadamard_block=hadamard_block)
    prerope_handle = None
    if args.kv_quant == "pre_rope":
        prerope_handle = kv_quant.install_pre_rope_kv_quant(
            model, **quant_kw, defer=(args.pre_rope_prefill == "independent"))

    # Prefill-then-quantize flow: post_rope, and pre_rope with independent prefill.
    prefill_then_quant = (args.kv_quant == "post_rope" or
                          (args.kv_quant == "pre_rope"
                           and args.pre_rope_prefill == "independent"))

    def generate_with_kv_quant(input_ids, attn_mask, max_new):
        """Run generation with the requested KV-quant mode."""
        if prefill_then_quant:
            past = DynamicCache()
            with torch.inference_mode():
                pref = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            past = pref.past_key_values
            if args.kv_quant == "post_rope":
                kv_quant.quantize_past_kv_inplace(past, **quant_kw)
            else:
                # pre_rope + independent prefill: write the stashed quantized K/V.
                prerope_handle.apply_deferred(past)
            next_tok = pref.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del pref
            with torch.inference_mode():
                out = model.generate(
                    input_ids=next_tok,
                    past_key_values=past,
                    attention_mask=torch.cat(
                        [attn_mask, torch.ones_like(next_tok)], dim=1),
                    max_new_tokens=max_new - 1,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
            return torch.cat([input_ids, next_tok, out[:, 1:]], dim=1)
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

    records = []
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        max_new = int(row["max_new_tokens"])
        input_ids = build_prompt(
            tok, row["context"], row["question"], row["answer_prefix"]).to(model.device)
        attn = torch.ones_like(input_ids)
        out = generate_with_kv_quant(input_ids, attn, max_new)
        raw_pred = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        golds = list(row["answer"])
        s, pred_clean = score(row["task"], raw_pred, golds)
        records.append({
            "task": row["task"],
            "question": row["question"],
            "answer_prefix": row["answer_prefix"],
            "golds": [str(g) for g in golds],
            "pred": pred_clean,
            "score": s,
            "input_tokens": int(input_ids.shape[1]),
        })

    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r["score"])
    summary = {t: sum(v) / len(v) for t, v in by_task.items()}
    overall = sum(summary.values()) / len(summary)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model,
                   "kv_quant": args.kv_quant,
                   "pre_rope_prefill": (args.pre_rope_prefill
                                        if args.kv_quant == "pre_rope" else None),
                   "summary": summary,
                   "overall": overall, "records": records}, f, indent=2)

    print(f"\n=== {args.model} ===")
    for t in sorted(summary):
        print(f"  {t:<18s} {summary[t]:.4f}  (n={len(by_task[t])})")
    print(f"  {'OVERALL':<18s} {overall:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
