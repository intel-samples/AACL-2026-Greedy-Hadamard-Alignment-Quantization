#!/usr/bin/env python3
"""Convert GHA ``.pt`` bundles to ``.safetensors`` files.

Standalone (does not import kv_bundle.py). Dependencies: torch (only to read
the ``.pt``) and numpy. The safetensors container is written by hand; if the
``safetensors`` package is installed, it is additionally used to cross-check
every written file.

Scope: GHA preconditioning only, i.e. bundles with ``meta.kind == "gha"``
(``gha_ph_*`` / ``gha_pl_*``). Their output side (``signs_out`` / ``perm_out``,
used only by the GHA+ ``--gha-out-mode d_out/dp_out`` runs) is dropped. Other
kinds found in an input directory are skipped.

What the converter writes:
  * tensors ``<site>.signs`` (I8, ±1) and ``<site>.perm`` (I16, or I32 if the
    last dim exceeds 32768), for every exported site;
  * ``__metadata__`` = {"format": FORMAT, "format_version": FORMAT_VERSION,
    META_KEY: <JSON>} (constants below); the meta holds head_dim, num_layers,
    num_kv_heads, full_dim, convention, ``sites`` = {site: {layout,
    hadamard_block}} and the provenance fields of the source bundle
    (null-valued ones dropped), plus model_revision and source_file.

Hadamard block: ``head_dim`` for per_head sites; for per_layer sites ``F``
(one full-width Hadamard, the paper's GHA setting) unless ``--pl-block head``.

Every output is written to a temp file, re-parsed with an independent reader,
validated (shapes, ±1 signs, valid permutations, Hadamard block sizes,
cross-site L/H consistency), compared value-by-value with the source, and only
then moved into place.

Usage:
  python3 scripts/convert_gha_bundles.py ../kv-cache-quant-ruler/bundles --out-dir release
  python3 scripts/convert_gha_bundles.py ../kv-cache-quant-ruler/bundles/gha_pl_Qwen3-4B__wikitext2__medm__seed0.pt \\
      --out-dir release --k-site pre_rope
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Format constants (format tag / metadata key names are set only here)
# --------------------------------------------------------------------------
FORMAT = "kvbundle"
FORMAT_VERSION = 1
META_KEY = "kvbundle_meta"
CONVENTION = "y = fwht(x[..., perm] * signs)"

SITES = ("k_pre_rope", "k_post_rope", "v")
FIELDS = ("signs", "perm")
LAYOUT_NDIM = {"shared": 1, "per_head": 3, "per_layer": 2}

ACCEPTED_KINDS = {"gha"}
# Source meta keys that are replaced (layouts -> sites) or that only describe
# the dropped output side.
DROPPED_META_KEYS = {"layouts", "layout_kind", "has_output_side", "eval_config",
                     "sites", "convention"}
# Kept even when null (model_revision may be unknown).
KEEP_IF_NULL = {"model", "model_revision"}

K_SITE_CHOICES = {"both": ("k_pre_rope", "k_post_rope"),
                  "pre_rope": ("k_pre_rope",),
                  "post_rope": ("k_post_rope",)}
K_SITE_SUFFIX = {"both": "", "pre_rope": "__kpre", "post_rope": "__kpost"}

_NP_TO_ST = {np.dtype("int8"): "I8", np.dtype("int16"): "I16",
             np.dtype("int32"): "I32", np.dtype("int64"): "I64"}
_ST_TO_NP = {v: k.newbyteorder("<") for k, v in _NP_TO_ST.items()}


def _pow2(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n > 0 and n & (n - 1) == 0


# --------------------------------------------------------------------------
# safetensors container, dependency-free
# --------------------------------------------------------------------------
def write_safetensors(path: str, tensors: dict, metadata: dict) -> None:
    """tensors: name -> C-contiguous little-endian integer ndarray."""
    # Larger itemsize first so every tensor starts at a multiple of its itemsize.
    names = sorted(tensors, key=lambda k: (-tensors[k].dtype.itemsize, k))
    header, blobs, off = {}, [], 0
    for k in names:
        a = np.ascontiguousarray(tensors[k], dtype=tensors[k].dtype.newbyteorder("<"))
        b = a.tobytes()
        header[k] = {"dtype": _NP_TO_ST[np.dtype(a.dtype.str.lstrip("<>|="))],
                     "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)                  # pad so the data section is 8-aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def read_safetensors(path: str):
    """Return (header_metadata, {name: ndarray}). Uses data_offsets."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    hm = header.pop("__metadata__", {})
    out = {}
    for k, info in header.items():
        s, e = info["data_offsets"]
        if info["dtype"] not in _ST_TO_NP:
            raise ValueError(f"{k}: unsupported dtype {info['dtype']}")
        out[k] = np.frombuffer(data[s:e], dtype=_ST_TO_NP[info["dtype"]]).reshape(info["shape"])
    return hm, out


# --------------------------------------------------------------------------
# Validation (structure, values, cross-site / meta shape consistency)
# --------------------------------------------------------------------------
def validate(hm: dict, tensors: dict) -> tuple[dict, dict]:
    """Validate a parsed file; return (meta, {site: {signs, perm}})."""
    # format tag / version
    if hm.get("format") != FORMAT:
        raise ValueError(f"format={hm.get('format')!r}, expected {FORMAT!r}")
    if int(hm["format_version"]) > FORMAT_VERSION:
        raise ValueError(f"format_version {hm['format_version']} > {FORMAT_VERSION}")
    meta = json.loads(hm[META_KEY])
    # tensor names are <site>.<field>
    sites: dict = {}
    for k, a in tensors.items():
        site, _, field = k.partition(".")
        if site not in SITES or field not in FIELDS:
            raise ValueError(f"unexpected tensor {k!r}")
        sites.setdefault(site, {})[field] = a
    if not sites:
        raise ValueError("no sites in file")
    # meta.sites matches the tensor sites
    specs = meta.get("sites")
    if not isinstance(specs, dict) or set(specs) != set(sites):
        raise ValueError(f"meta.sites {sorted(specs or {})} != tensor sites {sorted(sites)}")
    D = meta.get("head_dim")
    # head_dim is a power of two
    if not _pow2(D):
        raise ValueError(f"head_dim={D!r} is not a power-of-two int")
    LH = set()
    for site, t in sites.items():
        spec = specs[site]
        if not isinstance(spec, dict) or spec.get("layout") not in LAYOUT_NDIM:
            raise ValueError(f"{site}: bad sites entry {spec!r}")
        layout, blk = spec["layout"], spec.get("hadamard_block")
        if set(t) != set(FIELDS):
            raise ValueError(f"{site}: fields {sorted(t)}, expected {FIELDS}")
        s, p = t["signs"], t["perm"]
        # shapes match the layout
        if s.shape != p.shape or s.ndim != LAYOUT_NDIM[layout]:
            raise ValueError(f"{site} ({layout}): signs{s.shape} perm{p.shape}")
        n = s.shape[-1]
        if (layout == "per_layer" and n % D) or (layout != "per_layer" and n != D):
            raise ValueError(f"{site} ({layout}): last dim {n} vs head_dim {D}")
        # signs are ±1, perm is a permutation
        if not np.all(np.abs(s.astype(np.int64)) == 1):
            raise ValueError(f"{site}: signs not all ±1")
        if not np.array_equal(np.sort(p, axis=-1), np.broadcast_to(np.arange(n), p.shape)):
            raise ValueError(f"{site}: perm is not a permutation along the last axis")
        # Hadamard block is a valid power of two
        if not _pow2(blk) or n % blk or (layout != "per_layer" and blk != D):
            raise ValueError(f"{site} ({layout}): invalid hadamard_block {blk!r}")
        if layout == "per_head":
            LH.add((s.shape[0], s.shape[1]))
        elif layout == "per_layer":
            LH.add((s.shape[0], n // D))
    # All non-shared sites must agree on (L, H), and with the meta fields.
    if len(LH) > 1:
        raise ValueError(f"sites disagree on (num_layers, num_kv_heads): {sorted(LH)}")
    if LH:
        (L, H), = LH
        want = {"num_layers": L, "num_kv_heads": H, "full_dim": H * D}
        got = {k: meta.get(k) for k in want}
        if got != want:
            raise ValueError(f"meta {got} != tensor-derived {want}")
    return meta, sites


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def hf_cache_revision(model_id: str | None) -> str | None:
    """Commit sha the local HF cache resolves 'main' to (best effort)."""
    if not model_id:
        return None
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    ref = Path(hub) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def _source_layout(site_raw: dict, declared: str | None) -> str:
    if declared is not None:
        return declared
    return {1: "shared", 3: "per_head", 2: "per_layer"}[site_raw["signs"].ndim]


def build(src: dict, src_name: str, pl_block: str, k_site: str,
          revision: str | None) -> tuple[dict, dict, dict]:
    """Return (tensors, header_metadata, expected) for one ``.pt`` bundle.
    ``expected`` maps site -> (signs int64 ndarray, perm int64 ndarray) taken
    straight from the source, for post-write comparison."""
    meta_in = src["meta"]
    D = int(meta_in["head_dim"])
    declared = meta_in.get("layouts") or {}
    keep_sites = [s for s in SITES
                  if s in src and (s == "v" or s in K_SITE_CHOICES[k_site])]
    missing = [s for s in ("v", *K_SITE_CHOICES[k_site]) if s not in src]
    if missing:
        raise ValueError(f"source has no {missing} site(s)")

    tensors, sites_meta, expected, LH = {}, {}, {}, set()
    for site in keep_sites:
        raw = src[site]
        layout = _source_layout(raw, declared.get(site))
        if layout not in LAYOUT_NDIM:
            raise ValueError(f"{site}: unsupported layout {layout!r}")
        signs = raw["signs"].detach().cpu().to(torch.float64).numpy()
        perm = raw["perm"].detach().cpu().to(torch.int64).numpy()
        if not np.all(np.abs(signs) == 1):
            raise ValueError(f"{site}: source signs not all ±1")
        n = perm.shape[-1]
        pdt = np.int16 if n <= 32768 else np.int32
        tensors[f"{site}.signs"] = signs.astype(np.int8)
        tensors[f"{site}.perm"] = perm.astype(pdt)
        expected[site] = (signs.astype(np.int64), perm)

        if layout == "per_layer":
            blk = n if pl_block == "full" else D
            LH.add((perm.shape[0], n // D))
        else:
            blk = D
            if layout == "per_head":
                LH.add(perm.shape[:2])
        sites_meta[site] = {"layout": layout, "hadamard_block": int(blk)}

    if len(LH) > 1:
        raise ValueError(f"sites disagree on (num_layers, num_kv_heads): {sorted(LH)}")

    # Meta: required fields first, then provenance (null-valued dropped).
    meta = {"head_dim": D}
    if LH:
        (L, H), = LH
        for k, v in (("num_layers", L), ("num_kv_heads", H)):
            if meta_in.get(k) is not None and int(meta_in[k]) != v:
                raise ValueError(f"source meta {k}={meta_in[k]} but tensors say {v}")
        meta.update(num_layers=int(L), num_kv_heads=int(H), full_dim=int(H * D))
    meta["model"] = meta_in.get("model")
    meta["model_revision"] = (revision or meta_in.get("model_revision")
                              or hf_cache_revision(meta_in.get("model")))
    for k, v in meta_in.items():
        if k in meta or k in DROPPED_META_KEYS:
            continue
        if v is None and k not in KEEP_IF_NULL:
            continue
        meta[k] = v
    meta["convention"] = CONVENTION
    meta["sites"] = sites_meta
    meta["source_file"] = src_name

    hm = {"format": FORMAT, "format_version": str(FORMAT_VERSION),
          META_KEY: json.dumps(meta)}
    return tensors, hm, expected


def verify(path: str, expected: dict, hm_written: dict) -> dict:
    hm, tensors = read_safetensors(path)
    meta, sites = validate(hm, tensors)
    if json.loads(hm[META_KEY]) != json.loads(hm_written[META_KEY]):
        raise AssertionError("meta changed on round trip")
    if set(sites) != set(expected):
        raise AssertionError(f"sites {sorted(sites)} != {sorted(expected)}")
    for site, (s_ref, p_ref) in expected.items():
        if not np.array_equal(sites[site]["signs"].astype(np.int64), s_ref):
            raise AssertionError(f"{site}.signs differs from source")
        if not np.array_equal(sites[site]["perm"].astype(np.int64), p_ref):
            raise AssertionError(f"{site}.perm differs from source")
    try:  # optional cross-check with the reference safetensors library
        from safetensors.numpy import load_file
    except ImportError:
        return meta
    lib = load_file(path)
    if set(lib) != set(tensors) or any(not np.array_equal(lib[k], tensors[k]) for k in lib):
        raise AssertionError("safetensors library reads different tensors")
    return meta


def collect_inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*.pt")) if p.is_dir() else [p])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help=".pt bundle files and/or directories (non-recursive)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pl-block", choices=["full", "head"], default="full",
                    help="hadamard_block for per_layer sites: full = F (paper's "
                         "GHA setting, default) or head = head_dim")
    ap.add_argument("--k-site", choices=list(K_SITE_CHOICES), default="both",
                    help="K site(s) to export (v is always exported); a single "
                         "K site adds __kpre / __kpost to the file name")
    ap.add_argument("--model-revision",
                    help="model commit sha to record (default: source meta, then "
                         "the local HF cache's refs/main)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing output files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_skip = 0
    failed = []
    for p in collect_inputs(args.inputs):
        name = p.name
        out = Path(args.out_dir) / (p.stem + K_SITE_SUFFIX[args.k_site] + ".safetensors")
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            src = torch.load(p, map_location="cpu", weights_only=True)
            kind = src.get("meta", {}).get("kind")
            if kind not in ACCEPTED_KINDS:
                n_skip += 1
                continue
            if out.exists() and not args.overwrite:
                print(f"[exists] {out} (use --overwrite)")
                n_skip += 1
                continue
            tensors, hm, expected = build(src, name, args.pl_block, args.k_site,
                                          args.model_revision)
            write_safetensors(str(tmp), tensors, hm)
            meta = verify(str(tmp), expected, hm)
            os.replace(tmp, out)
        except Exception as e:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            failed.append(name)
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            continue
        n_ok += 1
        blocks = sorted({v["hadamard_block"] for v in meta["sites"].values()})
        layouts = sorted({v["layout"] for v in meta["sites"].values()})
        print(f"[ok]   {name} -> {out.name}  ({p.stat().st_size / 1e6:.1f} MB -> "
              f"{out.stat().st_size / 1e6:.2f} MB, {'/'.join(layouts)}, "
              f"block={blocks}, sites={list(meta['sites'])}, "
              f"rev={(meta['model_revision'] or '-')[:10]})")
    print(f"\n{n_ok} converted, {n_skip} skipped (other kinds / existing), "
          f"{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
