# CLAUDE.md

Official KV-cache quantization code for the AACL-IJCNLP 2026 paper "Less Random,
More Incoherent: Greedy Hadamard Alignment for LLM Quantization" (Tables 1-3;
the hidden-state experiments are not in this repo). See @README.md.

## Layout

- `gha.py` — GHA itself: Algorithm 1 `_psa`, Algorithm 2 `_greedy_z`, `gha()`; also the only Sylvester-Hadamard builder.
- `kv_quant.py` — RTN fake quantization, pre-/post-RoPE hooks, Hadamard and dense-R transforms. V is always per_token.
- `kv_bundle.py` — bundle schema (sites `k_pre_rope`, `k_post_rope`, `v`; layouts shared / per_head / per_layer / dense_*).
- `eval_ruler.py`, `eval_gsm8k.py` — evaluators (Hugging Face, offline by default).
- `scripts/generate_*_bundle.py` — calibration; `generate_gha_bundle.py` also holds the shared capture/naming helpers.
- `scripts/table*.sh` — one-command reproduction; `scripts/summarize_*.py` — paper tables from result JSONs.

## Commands

- Setup: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (versions are pinned; keep them).
- Dry run of a table script (no GPU): `DRY_RUN=1 bash scripts/table1_gha.sh`
- Quick eval check: `python eval_ruler.py ... --limit 2` (first N rows per task).
- Summaries: `python scripts/summarize_ruler_main.py [--post-root DIR --pre-root DIR]`, `python scripts/summarize_gsm8k_main.py [--root DIR --pre-root DIR]`.
- There is no test suite; verify changes as described below.

## Rules

- Paper configurations must stay bit-identical. After touching `kv_quant.py`, `kv_bundle.py`, `gha.py` or a generator, compare against the previous code: hash the quantized cache and logits on GPU for every bundle type x {post, pre sequential, pre independent}, regenerate bundles and compare tensors with `torch.equal`, and diff eval records.
- `--pre-rope-prefill sequential` (default) is the paper's setting; `independent` (unquantized prefill, cache quantized once) matches deployment. Do not change the default.
- Paper statistics: Table 1 GHA (median) = rep `medm`; Tables 2-3 "median" = `med`; Table 3 GHA (m) Post-PH matches `medm`; MassDiff(+GHA) = `meanabs` with `--hadamard-granularity per_head`; std uses ddof=0 over seeds 0-9; bold = mean above RHT in the same column.
- Bundle names always end in `__seed<N>`; calibration is WikiText-2 train, 128 x 4096 tokens.
- Table scripts never set GPUs: the user prefixes `CUDA_VISIBLE_DEVICES=...`. New ones follow the `scripts/table<N>_<name>.sh` naming and the existing runner.
- Comments and docs: concise, factual, English, no development-stage wording ("legacy", "today's", internal experiment numbers). Do not mention numeric precision (bf16/fp32) in code comments or docs, except the intended "bf16 (baseline)" row label in `scripts/summarize_*.py`; say "unquantized" for runs without KV quantization.
- Do not copy third-party code; reimplement and keep results identical. Cite papers by arXiv link.
- README follows the minimal QuaRot style; keep it short.
