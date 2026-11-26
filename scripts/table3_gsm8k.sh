#!/usr/bin/env bash
# Reproduce paper Table 3 (GSM8K https://arxiv.org/abs/2110.14168, 5-shot CoT, 2-bit, K per_channel / V per_token,
# LLaMA-3.1 8B Instruct, reusing the WikiText-2 RULER bundles): RHT (rh_pl / rh_sh),
# GHA (m) = rep med, GHA (e) = rep ext, PL + PH, Post + Pre, seeds 0-9, plus the
# unquantized row. The printed GHA (m) Post-PH cell is the medm bundle, so gha_ph/medm
# is also run for Post. -> 130 evals + 1 unquantized eval + 10 GHA calibrations.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/table3_gsm8k.sh
#   CUDA_VISIBLE_DEVICES=0 SEEDS="0 1" MODELS=qwen bash scripts/table3_gsm8k.sh
#   DRY_RUN=1 bash scripts/table3_gsm8k.sh            # list bundles to generate + eval commands
# Several copies may run at once on different GPUs (one CUDA_VISIBLE_DEVICES each):
# finished results are skipped, a job running elsewhere is skipped (flock), and
# bundle calibration is serialized (one capture at a time; needs ~200+ GB host RAM).
#
# Env:
#   PY             python (default: py311 env)
#   SEEDS          calibration seeds (default "0 1 2 3 4 5 6 7 8 9")
#   MODELS         subset of "llama" (default "llama")
#   REUSE_BUNDLES  read-only dir with existing bundles (e.g. ../kv-cache-quant-ruler/bundles);
#                  bundles missing there are generated into bundles/
#   LIMIT=N        quick check: first N rows per task, results go to results-limN-*
#   DRY_RUN=1      print only
#
# Outputs: results-reasoning/gsm8k/kpc-vpt/{rh_pl,rh_sh}/{post,pre}/seed<S>/2bit/<model>.json,
#          results-reasoning/gsm8k/kpc-vpt/gha_{pl,ph}/{med,ext}/{post,pre}/seed<S>/2bit/<model>.json,
#          results-reasoning/gsm8k/kpc-vpt/gha_ph/medm/post/seed<S>/2bit/<model>.json,
#          results-reasoning/gsm8k/bf16/<model>.json
# Missing bundles are generated into bundles/ first. Logs: logs/<script name>/;
# failed jobs are listed at the end (re-run to retry).
# Summary: python3 scripts/summarize_gsm8k_main.py --bits 2
# ---- runner (shared by scripts/table*.sh) ----------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # repo root

PY="${PY:-/home/yey/miniforge3/envs/py311/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"
REUSE_BUNDLES="${REUSE_BUNDLES:-}"
LIM="${LIMIT:+-lim$LIMIT}"                      # results-limN-* for quick checks
LIMARG="${LIMIT:+--limit $LIMIT}"
BUNDLES="bundles"
LOCKS="logs/.locks"
LOG_DIR="logs/$(basename "$0" .sh)${LIM}"
mkdir -p "$BUNDLES/.locks" "$LOCKS" "$LOG_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# key | HF id | result short name | bundle short name
MODEL_TABLE=(
  "llama|meta-llama/Meta-Llama-3.1-8B-Instruct|llama3.1-8b-instruct|Meta-Llama-3.1-8B-Instruct"
  "qwen|Qwen/Qwen3-4B|qwen3-4b|Qwen3-4B"
)
model_row () {  # $1=key -> sets HF SHORT BSHORT
  local row k
  for row in "${MODEL_TABLE[@]}"; do
    IFS='|' read -r k HF SHORT BSHORT <<< "$row"
    [[ "$k" == "$1" ]] && return 0
  done
  echo "ERROR: unknown model key '$1' (use: llama qwen)" >&2; exit 1
}
hf_for_bshort () {
  local row k hf short bshort
  for row in "${MODEL_TABLE[@]}"; do
    IFS='|' read -r k hf short bshort <<< "$row"
    [[ "$bshort" == "$1" ]] && { echo "$hf"; return 0; }
  done
  return 1
}

# Job line: seed|label|output|bundle|copy-to|args for $PY ("@BUNDLE@" = bundle path)
JOBS=()
add_job () { local s="$1" l="$2" o="$3" b="$4" c="$5"; shift 5; JOBS+=("$s|$l|$o|$b|$c|$*"); }

bundle_path () {  # $1=name -> $REUSE_BUNDLES/<name> if present, else bundles/<name>
  if [[ -n "$REUSE_BUNDLES" && -f "$REUSE_BUNDLES/$1" ]]; then echo "$REUSE_BUNDLES/$1"
  else echo "$BUNDLES/$1"; fi
}

gen_cmd () {  # $1=bundle name -> generator args for $PY (paper calibration settings)
  local n="$1" m
  if [[ "$n" =~ ^gha_(ph|pl)_(.+)__wikitext2__(med|mean|medm|ext)__seed([0-9]+)\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[2]}")" || return 1
    # one capture per (model, seed) writes all 8 GHA bundles (PH/PL x med/mean/medm/ext)
    echo "scripts/generate_gha_bundle.py --model $m --seed ${BASH_REMATCH[4]} --out-dir $BUNDLES"
  elif [[ "$n" =~ ^rh_pl_(.+)__seed([0-9]+)\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[1]}")" || return 1
    echo "scripts/generate_random_bundle.py --layout pl --model $m --seed ${BASH_REMATCH[2]} --out $BUNDLES/$n"
  elif [[ "$n" =~ ^rh_sh_d128__seed([0-9]+)\.pt$ ]]; then
    echo "scripts/generate_random_bundle.py --layout shared --head-dim 128 --seed ${BASH_REMATCH[1]} --out $BUNDLES/$n"
  elif [[ "$n" =~ ^gsr_ph_(.+)__fixed__g128\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[1]}")" || return 1
    echo "scripts/generate_gsr_bundle.py --model $m --layout per_head --out-dir $BUNDLES"
  elif [[ "$n" =~ ^(dfrot|butterfly)_(ph|pl)_(.+)__wikitext2__2bit_(g100|s600)__seed([0-9]+)\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[3]}")" || return 1
    local lay=per_head; [[ "${BASH_REMATCH[2]}" == pl ]] && lay=per_layer
    echo "scripts/generate_dense_r_bundle.py --model $m --method ${BASH_REMATCH[1]} --layout $lay --seed ${BASH_REMATCH[5]} --out-dir $BUNDLES"
  elif [[ "$n" =~ ^massdiff_pl_(.+)__wikitext2__meanabs__seed([0-9]+)\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[1]}")" || return 1
    echo "scripts/generate_perm_bundle.py --model $m --method massdiff --seed ${BASH_REMATCH[2]} --out-dir $BUNDLES"
  elif [[ "$n" =~ ^massdiff_gha_pl_(.+)__wikitext2__meanabs__seed([0-9]+)\.pt$ ]]; then
    m="$(hf_for_bshort "${BASH_REMATCH[1]}")" || return 1
    echo "scripts/generate_massdiff_gha_bundle.py --model $m --seed ${BASH_REMATCH[2]} --out-dir $BUNDLES"
  else
    echo "ERROR: no generator rule for bundle $n" >&2; return 1
  fi
}

ensure_bundle () {  # $1=name; generate into bundles/ if missing
  local n="$1" cmd log group
  [[ -f "$(bundle_path "$n")" ]] && return 0
  cmd="$(gen_cmd "$n")" || return 1
  log="$LOG_DIR/bundle__${n%.pt}.log"
  group="$n"   # all GHA files of one (model, seed) come from one capture
  [[ "$n" =~ ^gha_(ph|pl)_(.+)__wikitext2__.+__(seed[0-9]+)\.pt$ ]] && \
    group="gha_${BASH_REMATCH[2]}__${BASH_REMATCH[3]}"
  (
    flock 8                         # another copy generating the same bundle: wait for it
    [[ -f "$BUNDLES/$n" ]] && exit 0
    flock 7                         # one calibration capture at a time (~200+ GB host RAM)
    echo "[bundle $(date '+%m-%d %H:%M')] $n"
    # shellcheck disable=SC2086
    "$PY" $cmd > "$log" 2>&1 && [[ -f "$BUNDLES/$n" ]] || { echo "[bundle FAIL] $n log=$log"; exit 1; }
  ) 8> "$BUNDLES/.locks/$group.lock" 7> "$BUNDLES/.locks/capture.lock"
}

want_seed () { [[ "$1" == "-" || " $SEEDS " == *" $1 "* ]]; }

run_job () {
  local seed label out bundle copy args log lock bpath="" t0 rc
  IFS='|' read -r seed label out bundle copy args <<< "$1"
  want_seed "$seed" || return 0
  if [[ -f "$out" ]]; then
    if [[ -n "$copy" && ! -f "$copy" ]]; then mkdir -p "$(dirname "$copy")"; cp "$out" "$copy"; fi
    echo "[skip] $label (exists)"; return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -n "$bundle" ]]; then
      bpath="$(bundle_path "$bundle")"
      [[ -f "$bpath" ]] || echo "[dry] bundle: $PY $(gen_cmd "$bundle")"
    fi
    echo "[dry] $label: $PY ${args//@BUNDLE@/$bpath}"
    return 0
  fi
  log="$LOG_DIR/$(echo "$out" | sed 's|^results[^/]*/||; s|/|__|g; s|\.json$||').log"
  lock="$LOCKS/$(echo "$out" | md5sum | cut -c1-16)"
  (
    flock -n 9 || { echo "[skip] $label (running elsewhere)"; exit 0; }
    [[ -f "$out" ]] && exit 0
    if [[ -n "$bundle" ]]; then
      ensure_bundle "$bundle" || { echo "$label (bundle $bundle)" >> "$LOG_DIR/failed.txt"; exit 0; }
      bpath="$(bundle_path "$bundle")"
    fi
    mkdir -p "$(dirname "$out")"
    t0=$(date +%s)
    echo "[start $(date '+%m-%d %H:%M')] $label"
    # shellcheck disable=SC2086
    "$PY" ${args//@BUNDLE@/$bpath} > "$log" 2>&1
    rc=$?
    if (( rc == 0 )) && [[ -f "$out" ]]; then
      if [[ -n "$copy" ]]; then mkdir -p "$(dirname "$copy")"; cp "$out" "$copy"; fi
      echo "[done  $(date '+%m-%d %H:%M')] $label ($(( ($(date +%s) - t0) / 60 )) min)"
    else
      echo "[FAIL  $(date '+%m-%d %H:%M')] $label rc=$rc log=$log"
      echo "$label log=$log" >> "$LOG_DIR/failed.txt"
    fi
  ) 9> "$lock"
}

run_jobs () {  # $1 = summarizer command to print at the end
  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "ERROR: set CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=0 bash $0" >&2; exit 1
    fi
    "$PY" -c "import pandas, pyarrow, datasets, transformers, torch" \
      || { echo "ERROR: python env not usable; pass PY=/path/to/python" >&2; exit 1; }
  fi
  echo "${#JOBS[@]} jobs (SEEDS=$SEEDS${LIMIT:+, LIMIT=$LIMIT}), logs in $LOG_DIR"
  rm -f "$LOG_DIR/failed.txt"
  local j
  for j in "${JOBS[@]}"; do run_job "$j"; done
  if [[ -s "$LOG_DIR/failed.txt" ]]; then
    echo "== FAILED (re-run this script to retry):"; cat "$LOG_DIR/failed.txt"; exit 1
  fi
  echo "== finished. Summary: $1"
}

# ---- jobs: unquantized first, then seed-major --------------------------------
GROOT="results${LIM}-reasoning/gsm8k"
for mk in ${MODELS:-llama}; do
  model_row "$mk"
  out="$GROOT/bf16/$SHORT.json"
  add_job - "unquantized $mk" "$out" "" "" \
    eval_gsm8k.py --model "$HF" --kv-quant none --out "$out" $LIMARG
done
for S in $SEEDS; do
  for mk in ${MODELS:-llama}; do
    model_row "$mk"
    for T in post pre; do
      # cell | result series | bundle name
      CELLS=(
        "RHT $T-PL|rh_pl/$T|rh_pl_${BSHORT}__seed$S.pt"
        "RHT $T-PH|rh_sh/$T|rh_sh_d128__seed$S.pt"
        "GHA(m) $T-PL|gha_pl/med/$T|gha_pl_${BSHORT}__wikitext2__med__seed$S.pt"
        "GHA(m) $T-PH|gha_ph/med/$T|gha_ph_${BSHORT}__wikitext2__med__seed$S.pt"
        "GHA(e) $T-PL|gha_pl/ext/$T|gha_pl_${BSHORT}__wikitext2__ext__seed$S.pt"
        "GHA(e) $T-PH|gha_ph/ext/$T|gha_ph_${BSHORT}__wikitext2__ext__seed$S.pt"
      )
      [[ "$T" == post ]] && CELLS+=("GHA medm post-PH|gha_ph/medm/post|gha_ph_${BSHORT}__wikitext2__medm__seed$S.pt")
      for c in "${CELLS[@]}"; do
        IFS='|' read -r name ser b <<< "$c"
        out="$GROOT/kpc-vpt/$ser/seed$S/2bit/$SHORT.json"
        add_job "$S" "$name $mk seed$S" "$out" "$b" "" \
          eval_gsm8k.py --model "$HF" --kv-quant "${T}_rope" --bits 2 \
          --kv-bundle @BUNDLE@ --out "$out" $LIMARG
      done
    done
  done
done

run_jobs "python3 scripts/summarize_gsm8k_main.py --bits 2"
