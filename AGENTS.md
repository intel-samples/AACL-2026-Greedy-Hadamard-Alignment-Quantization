# AGENTS.md

This repository reproduces the paper's KV-cache quantization experiments (Tables 1-3).
It does not contain the hidden-state experiments or the deployment codec.
See [README.md](README.md) for Python 3.11 setup and reproduction commands. Keep dependency versions pinned.

## Where to make changes

- [gha.py](gha.py) implements the alignment algorithms and the Sylvester-Hadamard builder. Reuse that builder instead of adding another.
- [kv_quant.py](kv_quant.py) applies transforms and quantization at the pre- and post-RoPE hooks. [kv_bundle.py](kv_bundle.py) defines the bundle sites and layouts.
- [scripts/generate_gha_bundle.py](scripts/generate_gha_bundle.py) contains the shared calibration capture and naming helpers. Calibrate on WikiText-2 train with 128 sequences of 4096 tokens. Seeded bundle names end in `__seed<N>.pt`.
- [eval_ruler.py](eval_ruler.py) and [eval_gsm8k.py](eval_gsm8k.py) run offline by default. For new table scripts, follow the runner in [scripts/table1_gha.sh](scripts/table1_gha.sh) and name the script `scripts/table<N>_<name>.sh`. The caller sets `CUDA_VISIBLE_DEVICES`, not the script.

## Verification

There is no automated test suite. Choose the check that exercises the change:

- For table script changes, inspect the generated commands without using a GPU. For example: `DRY_RUN=1 MODELS=qwen SEEDS=0 bash scripts/table1_gha.sh`.
- For an evaluator smoke check, add `--limit 2` to the evaluation command (first two rows per task). Check `nvidia-smi` first and use a free GPU. Ask before starting full calibration or evaluation runs.
- After changing `gha.py`, `kv_quant.py`, `kv_bundle.py`, or a bundle generator, compare against the previous code on the same inputs. On GPU, compare hashes of the quantized cache and logits for every bundle type in all three modes: post-RoPE, pre-RoPE sequential, and pre-RoPE independent. Regenerate bundles and compare tensors with `torch.equal`; diff the evaluation records. Paper configurations must remain bit-identical. A dry run or a smoke check alone does not establish this.
- For reporting changes, run [scripts/summarize_ruler_main.py](scripts/summarize_ruler_main.py) or [scripts/summarize_gsm8k_main.py](scripts/summarize_gsm8k_main.py) on existing result JSONs; no evaluation run is needed.

## Code Review Rules

- Keep `--pre-rope-prefill sequential` as the paper default: it quantizes K/V during prefill. The `independent` mode quantizes the cache once after unquantized prefill. V is always `per_token`.
- MassDiff and MassDiff+GHA use `meanabs` with `--hadamard-granularity per_head`. Compute population standard deviation (`ddof=0`) over seeds 0-9. Bold a mean only when it exceeds RHT in the same column.

## Writing conventions

Keep comments and documentation concise, factual, and in English. Describe runs without KV quantization as "unquantized"; preserve the existing baseline row label in the summary scripts. Keep the README short and avoid development-stage labels. Cite papers with arXiv links.