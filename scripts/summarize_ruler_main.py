"""Paper Table 1: RULER-4096 overall accuracy (×100) at a given bit-width.

    | Model | Method | Post-PL | Post-PH | Pre-PL | Pre-PH |

Reads ``<root>/kpc-vpt/<variant>/[<rep>/][seed<N>/]<bits>bit/<model>.json`` under
the post- and pre-RoPE result roots, and ``<root>/baseline/<model>.json`` for
the unquantized row. Cells are ``mean ± std`` (ddof=0) over all calibration
seeds, with the seed count as ``(n=N)``; ``-`` = no run. **Bold** = mean above
RH in the same column of the same model (the bf16 and RH rows are never bold).

Output: analysis/table_ruler_main_<B>bit.md for each --bits value.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
BUCKET = "kpc-vpt"                                # K per_channel, V per_token

MODEL_DISPLAY = {
    "llama3.1-8b-instruct": "LLaMA-3.1 8B",
    "qwen3-4b": "Qwen3 4B",
}

# (display, variant stem, rep): the result dir is <stem>_<pl|ph>/<rep>; RH is
# rh_pl (PL) / rh_sh (PH) without a rep level. Sources: DFRot arXiv 2412.00648,
# ButterflyQuant arXiv 2509.09679, GSR arXiv 2505.03810, MassDiff (PeRQ) arXiv
# 2601.22347, RH = randomized Hadamard transform as in QuaRot arXiv 2404.00456.
METHODS = [
    ("GHA (mean)",     "gha",          "mean"),
    ("GHA (medm)",     "gha",          "medm"),
    ("GHA (ext)",      "gha",          "ext"),
    ("DFRot",          "dfrot",        "2bit_g100"),
    ("ButterflyQuant", "butterfly",    "2bit_s600"),
    ("GSR",            "gsr",          "fixed_g128"),
    ("MassDiff",       "massdiff",     "meanabs"),
    ("MassDiff+GHA",   "massdiff_gha", "meanabs"),
    ("RH",             "rh",           None),
]
RH_VARIANT = {"pl": "rh_pl", "ph": "rh_sh"}

# (display, timing, family)
COLUMNS = [
    ("Post-PL", "post", "pl"),
    ("Post-PH", "post", "ph"),
    ("Pre-PL",  "pre",  "pl"),
    ("Pre-PH",  "pre",  "ph"),
]


def read_summary(path: Path) -> dict:
    """The ``summary`` dict of a result JSON, plus ``overall`` if present."""
    d = json.loads(path.read_text())
    s = dict(d.get("summary") or {})
    if "overall" in d:
        s["overall"] = d["overall"]
    return s


def _path_key(path: Path):
    """Sort key ordering seed<N> dirs numerically."""
    return [(0, int(p[4:])) if p.startswith("seed") and p[4:].isdigit() else (1, p)
            for p in path.parts]


def collect(root: Path) -> dict:
    """``{(variant, rep, bits): {model: [summary, ...]}}`` with one summary
    per seed (``rep`` is None for plain / RH)."""
    runs: dict = defaultdict(lambda: defaultdict(list))
    bucket = root / BUCKET
    if not bucket.is_dir():
        return runs
    for path in sorted(bucket.rglob("*.json"), key=_path_key):
        parts = path.relative_to(bucket).parts
        variant, mid, bits = parts[0], list(parts[1:-2]), parts[-2]
        if not bits.endswith("bit") or path.stem not in MODEL_DISPLAY:
            continue
        if mid and mid[-1].startswith("seed"):
            mid = mid[:-1]
        runs[(variant, "/".join(mid) or None, int(bits[:-3]))][path.stem].append(
            read_summary(path))
    return runs


def fmt_cell(vals, with_n: bool = True) -> tuple[str, float | None]:
    """(display text, mean) of fractional scores, shown ×100."""
    if not vals:
        return "-", None
    a = np.asarray(vals, dtype=float) * 100.0
    mean = float(a.mean())
    if not with_n:
        return f"{mean:.2f}", mean
    if a.size == 1:
        return f"{mean:.2f} (n=1)", mean
    return f"{mean:.2f} ± {float(a.std(ddof=0)):.2f} (n={a.size})", mean


def bold_above(txt: str, mean: float | None, ref: float | None) -> str:
    return f"**{txt}**" if mean is not None and ref is not None and mean > ref else txt


def build_table(runs_by_timing: dict, base: dict, bits: int) -> list[str]:
    def vals(variant, rep, timing, model):
        runs = runs_by_timing[timing].get((variant, rep, bits), {}).get(model, [])
        return [s["overall"] for s in runs if "overall" in s]

    def variant(stem, family):
        return RH_VARIANT[family] if stem == "rh" else f"{stem}_{family}"

    bf16_txt = ", ".join(f"{MODEL_DISPLAY[m]} {base[m] * 100:.2f}"
                         for m in MODEL_DISPLAY if m in base) or "n/a"
    lines = [
        f"# RULER-4096 — main comparison, {bits}-bit",
        "",
        f"Overall accuracy (×100) at {bits}-bit, **mean ± std** over calibration "
        "seeds with the seed count as `(n=N)`. `-` = run not available. "
        "**Bold** = mean above `RH` in the same column of the same model. "
        "Columns are {Post,Pre}-RoPE × {PL (per-layer), PH (per-head)}.",
        "",
        f"**bf16 baseline (unquantized, overall ×100):** {bf16_txt}.",
        "",
        f"`plain` is RTN without a transform ({BUCKET}); MassDiff and MassDiff+GHA "
        "are per-layer only.",
        "",
        "| Model | Method | " + " | ".join(c[0] for c in COLUMNS) + " |",
        "|---|---|" + "|".join(["---:"] * len(COLUMNS)) + "|",
    ]
    for model, disp in MODEL_DISPLAY.items():
        rh = [fmt_cell(vals(RH_VARIANT[fam], None, tm, model))[1] for _, tm, fam in COLUMNS]
        b = base.get(model)
        lines.append(f"| {disp} | *bf16 (baseline)* | "
                     + " | ".join([f"{b * 100:.2f}" if b is not None else "-"] * len(COLUMNS))
                     + " |")
        plain = [bold_above(*fmt_cell(vals("plain", None, tm, model), with_n=False), rh[ci])
                 for ci, (_, tm, _) in enumerate(COLUMNS)]
        lines.append(f"| {disp} | *plain (baseline)* | " + " | ".join(plain) + " |")
        for name, stem, rep in METHODS:
            cells = []
            for ci, (_, tm, fam) in enumerate(COLUMNS):
                txt, mean = fmt_cell(vals(variant(stem, fam), rep, tm, model))
                cells.append(txt if stem == "rh" else bold_above(txt, mean, rh[ci]))
            lines.append(f"| {disp} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post-root", default=str(REPO / "results-post-rope"))
    ap.add_argument("--pre-root", default=str(REPO / "results-pre-rope"))
    ap.add_argument("--out-dir", default=str(REPO / "analysis"))
    ap.add_argument("--bits", type=int, nargs="+", default=[2],
                    help="bit-width(s) to tabulate, one file each. Default: 2.")
    args = ap.parse_args()

    runs_by_timing, base = {}, {}
    for timing, root in (("post", Path(args.post_root)), ("pre", Path(args.pre_root))):
        runs_by_timing[timing] = collect(root)
        for m in MODEL_DISPLAY:
            p = root / "baseline" / f"{m}.json"
            if m not in base and p.is_file():
                base[m] = read_summary(p)["overall"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bits in args.bits:
        out = out_dir / f"table_ruler_main_{bits}bit.md"
        out.write_text("\n".join(build_table(runs_by_timing, base, bits)) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
