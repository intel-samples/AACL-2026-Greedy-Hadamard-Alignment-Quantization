"""GSM8K (https://arxiv.org/abs/2110.14168) evaluator sharing eval_ruler.py's KV-cache fake-quant machinery
(same bundles, same quantizer, same pre_rope hook / post_rope prefill-quant
generation path), so results are directly comparable to the RULER runs.

Task: 5-shot chain-of-thought generation (exemplars = first 5 train rows, at
most 256 new tokens, greedy); the prediction is the last number in the output,
compared to the gold after '####'.

KV-quant semantics (important):
  * pre_rope  — the hook fake-quantizes K (before RoPE) and V of the prompt
                prefill. Decode steps (seq == 1) are passed through, so tokens
                generated during a CoT decode keep unquantized KV.
                --pre-rope-prefill sequential (default): quantization happens
                inside the prefill forward. independent: unquantized prefill,
                then the quantized K/V are written into the cache once before
                decoding (same flow as post_rope).
  * post_rope — (default) quantizes the prefill past_kv once, then decode is
                unquantized.

Fixed settings (the paper's Table 3): asymmetric RTN, K per_channel /
V per_token, block 128, full-width Hadamard on per-layer bundles, no
output-side transform, SDPA attention. Defaults: 2-bit, post-RoPE.

The dataset (openai/gsm8k, main) is read from the local HF cache; this script
keeps HF_HUB_OFFLINE=1 like the rest of the repo.

Usage:
  python eval_gsm8k.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --kv-quant pre_rope \
      --kv-bundle bundles/gha_pl_Meta-Llama-3.1-8B-Instruct__wikitext2__medm__seed0.pt \
      --out results-reasoning/gsm8k/.../model.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import kv_bundle
import kv_quant

# Fixed evaluation settings (paper Table 3).
N_SHOT = 5
MAX_NEW_TOKENS = 256
KV_BLOCK_SIZE = 128
K_SCALING = "per_channel"
ATTN_IMPL = "sdpa"
INSTRUCTION = ("\n\nSolve this step by step, then give the final answer on a "
               "new line as: #### <number>.")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_task(limit=None):
    """Return the GSM8K test split as a list of dict rows (first `limit` rows
    if `limit` is set)."""
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    return rows[:limit] if limit else rows


def gsm8k_fewshot():
    """The first ``N_SHOT`` train rows as (question, CoT answer ending in
    '#### <number>') exemplars."""
    train = load_dataset("openai/gsm8k", "main", split="train")[:N_SHOT]
    return list(zip(train["question"], train["answer"]))


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_gold(answer: str) -> str:
    """Gold = the number after '####', normalized (strip commas/trailing .0)."""
    tail = answer.split("####")[-1].strip()
    return _norm_num(tail)


def _norm_num(s: str) -> str:
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return s.strip()
    v = m.group(0)
    # normalize 18.0 -> 18, keep genuine decimals
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def gsm8k_extract_pred(text: str) -> str:
    """Prediction = the LAST number in the generated text (post-CoT answer)."""
    nums = _NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return ""
    return _norm_num(nums[-1])


def gsm8k_prompt(tok, question: str, shots) -> torch.Tensor:
    """Chat-templated prompt: each exemplar as a user question + assistant
    solution turn, then the test question. Returns ``(1, L)`` input ids."""
    msgs = []
    for q, a in shots:
        msgs += [{"role": "user", "content": q + INSTRUCTION},
                 {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question + INSTRUCTION})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True, enable_thinking=False)["input_ids"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test rows (quick checks)")
    ap.add_argument("--kv-quant", default="post_rope",
                    choices=["none", "post_rope", "pre_rope"],
                    help="KV fake-quant mode. post_rope (default) quantizes the "
                         "prefill cache once. pre_rope hooks K (before RoPE) "
                         "and V of the prefill forward; decode steps stay "
                         "unquantized. none disables it.")
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
    ap.add_argument("--kv-bundle", default=None,
                    help="Bundle holding the K and V transforms. Unset = no "
                         "transform (direct quantization).")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.pre_rope_prefill != "sequential" and args.kv_quant != "pre_rope":
        raise SystemExit("--pre-rope-prefill independent requires --kv-quant pre_rope")

    rows = load_task(args.limit)
    print(f"[gsm8k] loaded {len(rows)} rows")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto",
        attn_implementation=ATTN_IMPL)
    model.eval()
    model.generation_config.max_length = None

    head_dim = getattr(model.config, "head_dim", None) or \
        (model.config.hidden_size // model.config.num_attention_heads)
    print(f"kv_quant={args.kv_quant} bits={args.bits} bundle={args.kv_bundle}")

    bundle = None
    if args.kv_bundle:
        bundle = kv_bundle.load_bundle(args.kv_bundle)
        bundle.check_head_dim(head_dim)
        print(f"  bundle meta: {bundle.meta}")

    quant_kw = dict(bits=args.bits, block_size=KV_BLOCK_SIZE, bundle=bundle,
                    out_mode="none", k_mode=K_SCALING, hadamard_block=None)

    # pre_rope: install the hook once; it fires on every multi-token forward
    # (decode steps with seq == 1 are passed through).
    prerope_handle = None
    if args.kv_quant == "pre_rope":
        prerope_handle = kv_quant.install_pre_rope_kv_quant(
            model, **quant_kw, defer=(args.pre_rope_prefill == "independent"))

    def _quantize_prefill(past):
        """Prefill-then-quantize flow (one shot, in place): post_rope
        quantizes the cache; pre_rope + independent prefill writes the stashed
        pre-RoPE-quantized K/V."""
        if args.kv_quant == "pre_rope":
            prerope_handle.apply_deferred(past)
            return
        kv_quant.quantize_past_kv_inplace(past, **quant_kw)

    try:
        records = run_gsm8k(args, model, tok, rows, _quantize_prefill)
    finally:
        if prerope_handle is not None:
            prerope_handle.remove()

    n = len(records)
    acc = sum(r["score"] for r in records) / max(n, 1)
    summary = {
        "task": "gsm8k",
        "model": args.model,
        "n": n,
        "accuracy": acc,
        "kv_quant": args.kv_quant,
        "pre_rope_prefill": (args.pre_rope_prefill
                             if args.kv_quant == "pre_rope" else None),
        "bits": args.bits,
        "kv_bundle": args.kv_bundle,
        "limit": args.limit,
    }
    out = {"summary": summary, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[gsm8k] accuracy = {acc:.4f} over {n} examples")
    print(f"wrote {args.out}")


# ---------------------------------------------------------------------------
# GSM8K runner (generation)
# ---------------------------------------------------------------------------

def run_gsm8k(args, model, tok, rows, quantize_prefill):
    shots = gsm8k_fewshot()
    print(f"[gsm8k] {len(shots)}-shot CoT (exemplars from train split)")
    records = []
    for row in tqdm(rows, total=len(rows)):
        gold = gsm8k_gold(row["answer"])
        ids = gsm8k_prompt(tok, row["question"], shots=shots).to(model.device)
        attn = torch.ones_like(ids)
        out = _generate(args, model, tok, ids, attn, MAX_NEW_TOKENS, quantize_prefill)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred = gsm8k_extract_pred(text)
        score = 1.0 if (pred != "" and pred == gold) else 0.0
        records.append({
            "question": row["question"],
            "gold": gold, "pred": pred, "score": score,
            "gen": text[:2000],
            "input_tokens": int(ids.shape[1]),
        })
    return records


def _generate(args, model, tok, input_ids, attn_mask, max_new, quantize_prefill):
    """Same two paths as eval_ruler.generate_with_kv_quant."""
    if args.kv_quant == "post_rope" or (args.kv_quant == "pre_rope"
                                        and args.pre_rope_prefill == "independent"):
        past = DynamicCache()
        with torch.inference_mode():
            pref = model(input_ids=input_ids, attention_mask=attn_mask,
                         past_key_values=past, use_cache=True)
        past = pref.past_key_values
        quantize_prefill(past)
        next_tok = pref.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del pref
        with torch.inference_mode():
            out = model.generate(
                input_ids=next_tok, past_key_values=past,
                attention_mask=torch.cat([attn_mask, torch.ones_like(next_tok)], dim=1),
                max_new_tokens=max_new - 1, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True)
        return torch.cat([input_ids, next_tok, out[:, 1:]], dim=1)
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id)


if __name__ == "__main__":
    main()
