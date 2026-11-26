"""Generate dense orthogonal-R bundles for the DFRot and ButterflyQuant baselines.

Both fit an orthogonal R per (site, layer[, head]) that lowers the quantization
error of ``x @ R`` under 2-bit per-token RTN in groups of 128 channels; the
runtime applies ``y = x @ R`` and inverts with ``y @ R.t()``.

  dfrot     — DFRot (https://arxiv.org/abs/2412.00648, §3.3-3.4): alternating orthogonal
    Procrustes, ``R = U Vᵀ`` from ``SVD(Xᵀ diag(w) Q(XR))``, with tokens of
    massive activation up-weighted by gamma = 100; 100 iterations. Tag 2bit_g100.
  butterfly — ButterflyQuant (https://arxiv.org/abs/2509.09679, §3.2-3.4): R is a product of
    log2(d) layers of d/2 Givens rotations, trained from identity with Adam on
    a reconstruction + uniformity loss through a straight-through quantizer,
    600 steps, then expanded to a dense matrix. Tag 2bit_s600.

``--layout per_head``: one (D, D) R per head; ``per_layer``: one (F, F) R per
layer, F = num_kv_heads * head_dim. Each fit uses up to 8192 calibration tokens.

Output: bundles/{dfrot,butterfly}_{ph,pl}_<model>__wikitext2__<tag>__seed<N>.pt
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from generate_gha_bundle import CALIB, NSAMPLES, SITE_OF, add_common_args, bundle_path, capture_kv
import kv_bundle  # noqa: E402
import kv_quant  # noqa: E402

BITS = 2
BLOCK_SIZE = 128
GAMMA = 100.0          # DFRot massive-activation weight
DFROT_ITERS = 100
BF_STEPS = 600
BF_LAMBDA = 0.1        # ButterflyQuant uniformity-loss weight
BF_LR = 1e-2
MAX_TOKENS = 8192


def _quant(x2d: torch.Tensor) -> torch.Tensor:
    """The eval fake-quant, per-token groups along the channel axis."""
    return kv_quant.fake_quant(x2d, BITS, BLOCK_SIZE, "per_token")


# ---------------------------------------------------------------------------
# DFRot
# ---------------------------------------------------------------------------

def _massive_weight(X: torch.Tensor) -> torch.Tensor:
    """Per-token weight: GAMMA for tokens whose max |x| exceeds mean + 3 std
    over the calibration tokens, 1 otherwise."""
    linf = X.abs().amax(dim=-1)
    w = torch.ones_like(linf)
    w[linf > linf.mean() + 3.0 * linf.std()] = GAMMA
    return w


def fit_dfrot(X: torch.Tensor) -> torch.Tensor:
    """X: (T, d). Returns the (d, d) orthogonal R minimizing the weighted
    ‖XR − Q(XR)‖² by alternating Q and the Procrustes solution for R."""
    Xd = X.to(torch.float64)
    Xw = Xd * _massive_weight(Xd).unsqueeze(-1)
    R = torch.eye(Xd.shape[1], dtype=torch.float64)
    for _ in range(DFROT_ITERS):
        Yq = _quant((Xd @ R).to(torch.float32)).to(torch.float64)
        U, _, Vh = torch.linalg.svd(Xw.transpose(-1, -2) @ Yq)
        R = U @ Vh
    return R.to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# ButterflyQuant
# ---------------------------------------------------------------------------

def _butterfly_pairs(d: int):
    """Per butterfly layer s, the index pairs (i, i + 2^s) for every i whose
    bit s is 0."""
    n_layers = int(round(math.log2(d)))
    assert (1 << n_layers) == d, f"d={d} must be a power of 2"
    pairs = []
    for s in range(n_layers):
        i = torch.tensor([i for i in range(d) if not i & (1 << s)], dtype=torch.long)
        pairs.append((i, i + (1 << s)))
    return pairs


def butterfly_to_dense(angles: torch.Tensor, d: int, pairs) -> torch.Tensor:
    """Dense (d, d) R with ``x @ R`` = the butterfly applied to x
    (differentiable in ``angles``, shape (n_layers, d/2))."""
    R = torch.eye(d, dtype=angles.dtype, device=angles.device)
    for s, (ii, jj) in enumerate(pairs):
        c, sn = torch.cos(angles[s]), torch.sin(angles[s])
        ri, rj = R[:, ii], R[:, jj]
        R = R.clone()
        R[:, ii] = ri * c - rj * sn
        R[:, jj] = ri * sn + rj * c
    return R


def _ste_quant(x: torch.Tensor) -> torch.Tensor:
    """``_quant`` with a straight-through round, so gradients reach R."""
    d = x.shape[-1]
    n_full = (d // BLOCK_SIZE) * BLOCK_SIZE
    qmax = (1 << BITS) - 1

    def asym(xg):
        xmin = xg.amin(dim=-1, keepdim=True)
        xmax = xg.amax(dim=-1, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp(min=1e-12)
        u = (xg - xmin) / scale
        return (u + (torch.round(u) - u).detach()).clamp(0, qmax) * scale + xmin

    lead = x.shape[:-1]
    parts = []
    if n_full > 0:
        parts.append(asym(x[..., :n_full].reshape(*lead, n_full // BLOCK_SIZE, BLOCK_SIZE))
                     .reshape(*lead, n_full))
    if n_full < d:
        parts.append(asym(x[..., n_full:]))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _uniformity_kl(Yq: torch.Tensor) -> torch.Tensor:
    """KL(soft histogram of the codes ‖ uniform) on a global min-max grid."""
    n_bins = 1 << BITS
    lo, hi = Yq.detach().min(), Yq.detach().max()
    if (hi - lo) < 1e-9:
        return Yq.new_zeros(())
    u = ((Yq - lo) / (hi - lo)).clamp(0, 1) * (n_bins - 1)
    centers = torch.arange(n_bins, device=Yq.device, dtype=Yq.dtype)
    w = (1.0 - (u.reshape(-1, 1) - centers.reshape(1, -1)).abs()).clamp(min=0.0)
    hist = w.sum(dim=0)
    hist = hist / hist.sum().clamp(min=1e-12)
    unif = torch.full_like(hist, 1.0 / n_bins)
    return (hist * (hist.clamp(min=1e-12).log() - unif.log())).sum()


def fit_butterfly(X: torch.Tensor, device: str) -> torch.Tensor:
    """X: (T, d). Returns the dense (d, d) R of the trained butterfly."""
    Xd = X.to(device=device, dtype=torch.float32)
    d = Xd.shape[1]
    pairs = [(ii.to(device), jj.to(device)) for ii, jj in _butterfly_pairs(d)]
    angles = torch.zeros(len(pairs), d // 2, device=device, requires_grad=True)
    opt = torch.optim.Adam([angles], lr=BF_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BF_STEPS)
    for _ in range(BF_STEPS):
        opt.zero_grad()
        Y = Xd @ butterfly_to_dense(angles, d, pairs)
        Yq = _ste_quant(Y)
        loss = ((Y - Yq) ** 2).mean() + BF_LAMBDA * _uniformity_kl(Yq)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        R = butterfly_to_dense(angles.detach(), d, pairs)
    return R.to("cpu", torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--method", required=True, choices=["dfrot", "butterfly"])
    ap.add_argument("--layout", required=True, choices=["per_head", "per_layer"])
    args = ap.parse_args()

    tag = f"{BITS}bit_g{int(GAMMA)}" if args.method == "dfrot" else f"{BITS}bit_s{BF_STEPS}"
    lay = "ph" if args.layout == "per_head" else "pl"
    out_path = bundle_path(args.out_dir, f"{args.method}_{lay}", args.model, tag, args.seed)
    if os.path.exists(out_path):
        print(f"bundle already exists; skipping: {out_path}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    sinks, (L, H, D) = capture_kv(args.model, args.seqlen, args.seed)
    device = "cuda" if args.method == "butterfly" and torch.cuda.is_available() else "cpu"

    def fit(X):
        return fit_dfrot(X) if args.method == "dfrot" else fit_butterfly(X, device)

    print(f"fitting {args.method} R ({args.layout}) on {device}")
    per_site_matrix = {}
    t0 = time.perf_counter()
    for cap, site in SITE_OF.items():
        Rs = []
        for li in range(L):
            X = torch.cat(sinks[cap][li], dim=0)                     # (T, H, D)
            if X.shape[0] > MAX_TOKENS:
                X = X[torch.linspace(0, X.shape[0] - 1, MAX_TOKENS).long()]
            Rs.append(torch.stack([fit(X[:, h]) for h in range(H)]) if args.layout == "per_head"
                      else fit(X.reshape(X.shape[0], H * D)))
            sinks[cap][li].clear()
            if (li + 1) % 8 == 0 or li == L - 1:
                print(f"  {cap}: layer {li + 1}/{L} [{time.perf_counter() - t0:.1f}s]", flush=True)
        per_site_matrix[site] = torch.stack(Rs)

    meta = {"method": args.method, "calib": CALIB, "nsamples": NSAMPLES, "seqlen": args.seqlen,
            "seed": args.seed, "model": args.model, "bits": BITS, "block_size": BLOCK_SIZE,
            "max_tokens": MAX_TOKENS}
    if args.method == "dfrot":
        meta.update({"gamma": GAMMA, "iters": DFROT_ITERS})
    else:
        meta.update({"steps": BF_STEPS, "lambda": BF_LAMBDA, "lr": BF_LR})
    payload = kv_bundle.make_dense_r_bundle(
        D, L, H, per_site_matrix=per_site_matrix, kind=args.method,
        layout=kv_bundle.LAYOUT_DENSE_PER_HEAD if lay == "ph" else kv_bundle.LAYOUT_DENSE_PER_LAYER,
        note=f"{args.method} dense orthogonal R ({tag}). Eval with --gha-out-mode none.",
        extra_meta=meta)
    worst = max(((R := m.reshape(-1, m.shape[-1], m.shape[-1])).transpose(-1, -2) @ R
                 - torch.eye(R.shape[-1])).abs().amax().item() for m in per_site_matrix.values())
    print(f"  max |RᵀR − I| = {worst:.2e}")
    if worst >= 1e-3:
        raise ValueError(f"non-orthogonal R (max deviation {worst:.2e})")
    kv_bundle.save_bundle(out_path, payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
