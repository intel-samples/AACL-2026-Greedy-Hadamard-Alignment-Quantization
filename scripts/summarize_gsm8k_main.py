"""Paper Table 3: GSM8K accuracy (×100) of LLaMA-3.1 8B Instruct.

    | Model | Method | Post-PL | Post-PH | Pre-PL | Pre-PH |

Reads the eval_gsm8k.py result tree::

    <root>/gsm8k/bf16/<model>.json
    <root>/gsm8k/kpc-vpt/gha_<pl|ph>/<rep>/<pre|post>/seed<N>/<bits>bit/<model>.json
    <root>/gsm8k/kpc-vpt/rh_<pl|sh>/<pre|post>/seed<N>/<bits>bit/<model>.json

Rows: GHA (med) — the paper's GHA (m), except that its printed Post-PH entry
matches GHA (medm); GHA (ext) — the paper's GHA (e); RH. Cells are
``mean ± std`` (ddof=0) over all seeds, ``(n=N)``; ``-`` = no run.
**Bold** = mean above RH in the same column (bf16 and RH rows never bold).

Output: analysis/table_gsm8k_main.md (one section per --bits value).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from summarize_ruler_main import (
    BUCKET, COLUMNS, MODEL_DISPLAY, REPO, RH_VARIANT, _path_key, bold_above,
    fmt_cell, read_summary,
)

MODEL = "llama3.1-8b-instruct"

# (display, variant stem, rep)
METHODS = [
    ("GHA (med)",  "gha", "med"),
    ("GHA (medm)", "gha", "medm"),
    ("GHA (ext)",  "gha", "ext"),
    ("RH",         "rh",  None),
]


def collect(root: Path):
    """``(bf16_accuracy or None, {(variant, rep, timing, bits): [accuracy, ...]})``
    for ``MODEL``, one accuracy per seed."""
    task_root = root / "gsm8k"
    bf16_path = task_root / "bf16" / f"{MODEL}.json"
    bf16 = read_summary(bf16_path)["accuracy"] if bf16_path.is_file() else None
    runs: dict = defaultdict(list)
    bucket = task_root / BUCKET
    for path in sorted(bucket.rglob(f"{MODEL}.json"), key=_path_key) if bucket.is_dir() else []:
        parts = path.relative_to(bucket).parts
        if parts[0] in ("gha_pl", "gha_ph") and len(parts) == 6:
            variant, rep, timing, seed, bits, _ = parts
        elif parts[0] in ("rh_pl", "rh_sh") and len(parts) == 5:
            (variant, timing, seed, bits, _), rep = parts, None
        else:
            continue
        if seed.startswith("seed"):
            runs[(variant, rep, timing, bits)].append(read_summary(path)["accuracy"])
    return bf16, runs


def build_table(bf16, runs, bits: int) -> list[str]:
    btag = f"{bits}bit"
    disp = MODEL_DISPLAY[MODEL]

    def cell(stem, rep, timing, family):
        variant = RH_VARIANT[family] if stem == "rh" else f"{stem}_{family}"
        return fmt_cell(runs.get((variant, rep, timing, btag), []))

    rh = [cell("rh", None, tm, fam)[1] for _, tm, fam in COLUMNS]
    lines = [
        f"## {btag}",
        "",
        "| Model | Method | " + " | ".join(c[0] for c in COLUMNS) + " |",
        "|---|---|" + "|".join(["---:"] * len(COLUMNS)) + "|",
        f"| {disp} | *bf16 (baseline)* | "
        + " | ".join([f"{bf16 * 100:.2f}" if bf16 is not None else "-"] * len(COLUMNS)) + " |",
    ]
    for name, stem, rep in METHODS:
        cells = []
        for ci, (_, tm, fam) in enumerate(COLUMNS):
            txt, mean = cell(stem, rep, tm, fam)
            cells.append(txt if stem == "rh" else bold_above(txt, mean, rh[ci]))
        lines.append(f"| {disp} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "results-reasoning"),
                    help="result tree root")
    ap.add_argument("--pre-root", default=None,
                    help="read the Pre columns from this root instead (e.g. "
                         "results-reasoning-independent)")
    ap.add_argument("--out", default=str(REPO / "analysis" / "table_gsm8k_main.md"))
    ap.add_argument("--bits", type=int, nargs="+", default=[2])
    args = ap.parse_args()

    bf16, runs = collect(Path(args.root))
    if args.pre_root:
        _, pre = collect(Path(args.pre_root))
        runs = {k: v for k, v in runs.items() if k[2] != "pre"}
        runs.update({k: v for k, v in pre.items() if k[2] == "pre"})
    lines = ["# GSM8K — main comparison (GHA vs RH)", "",
             "GSM8K accuracy (×100), **mean ± std** over calibration seeds with the "
             "seed count as `(n=N)`. `-` = run not available. **Bold** = mean above "
             "`RH` in the same column.", ""]
    for bits in args.bits:
        lines.extend(build_table(bf16, runs, bits))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
