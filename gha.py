"""Greedy Hadamard Alignment (GHA)"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _build_sylvester_hadamard(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    hadamard = torch.tensor([[1.0]], device=device, dtype=dtype)
    while hadamard.shape[0] < d:
        hadamard = torch.cat([
            torch.cat([hadamard, hadamard], dim=1),
            torch.cat([hadamard, -hadamard], dim=1),
        ], dim=0)
    return hadamard / math.sqrt(float(d))


def _resolve_hadamard(
    d: int,
    h: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if h is None:
        if not _is_power_of_two(d):
            raise ValueError(
                f"input dimension d={d} is not a power of two; pass an explicit Hadamard matrix via h"
            )
        return _build_sylvester_hadamard(d=d, device=device, dtype=dtype)

    if h.ndim != 2 or h.shape[0] != d or h.shape[1] != d:
        raise ValueError(f"h must have shape ({d}, {d}), got {tuple(h.shape)}")
    return h.to(device=device, dtype=dtype)


def _psa(
    x: torch.Tensor,
    z: torch.Tensor,
    hadamard: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 1: signed permutation aligning ``x`` with ``u = Hᵀz``."""
    d = x.shape[0]
    u = hadamard.T @ z

    idx_x = x.abs().argsort(descending=True)
    idx_u = u.abs().argsort(descending=True)

    permutation = torch.empty(d, dtype=torch.long, device=x.device)
    permutation[idx_u] = idx_x

    px = x[permutation]
    signs = torch.where(u * px < 0.0, -torch.ones_like(px), torch.ones_like(px)).to(torch.float32)
    return permutation, signs


def _surrogate_objective_batch(sorted_abs_x: torch.Tensor, u_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized surrogate objective for a batch of candidate u vectors."""
    sorted_abs_u_batch = u_batch.abs().sort(descending=True, dim=1).values
    diffs = sorted_abs_u_batch - sorted_abs_x.unsqueeze(0)
    return diffs.pow(2).sum(dim=1)


def _greedy_z(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    max_search_range: int | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Algorithm 2: greedy flat-target search, fixing one signed coordinate of
    ``z`` per step to minimize the surrogate objective."""
    d = x.shape[0]
    hadamard_t = hadamard.T.contiguous()
    rho = torch.linalg.norm(x)
    flat_level = float((rho / math.sqrt(d)).item())

    sorted_abs_x = x.abs().sort(descending=True).values
    all_coords = torch.arange(d, device=x.device)

    z = torch.zeros(d, device=x.device, dtype=x.dtype)
    current_u = torch.zeros(d, device=x.device, dtype=x.dtype)
    available_mask = torch.ones(d, device=x.device, dtype=torch.bool)

    if max_search_range is None or max_search_range < 0:
        window = d
    else:
        window = int(max_search_range)

    iterator = range(d)
    if show_progress:
        iterator = tqdm(iterator)
    for _ in iterator:
        available_coords = all_coords[available_mask]
        candidate_coords = available_coords[:window]
        candidate_rows = hadamard_t[candidate_coords]

        candidate_u_pos = current_u.unsqueeze(0) + flat_level * candidate_rows
        candidate_u_neg = current_u.unsqueeze(0) - flat_level * candidate_rows
        candidate_u_batch = torch.stack((candidate_u_pos, candidate_u_neg), dim=1).reshape(-1, d)
        candidate_objectives = _surrogate_objective_batch(sorted_abs_x=sorted_abs_x, u_batch=candidate_u_batch)

        best_index = torch.argmin(candidate_objectives).item()
        best_coord = int(candidate_coords[best_index // 2].item())
        z_level = flat_level if best_index % 2 == 0 else -flat_level

        z[best_coord] = z_level
        available_mask[best_coord] = False
        current_u = current_u + z_level * hadamard_t[best_coord]

    return z


def gha(
    x: torch.Tensor,
    h: torch.Tensor | None = None,
    *,
    max_search_range: int | None = None,
    show_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy Hadamard Alignment: returns ``(perm, signs)`` for ``x``.

    Each greedy step scores only the first ``max_search_range`` still-available
    coordinates (``None`` or negative: all of them), so a step costs
    ``O(max_search_range * d log d)`` instead of ``O(d^2 log d)``.
    ``h`` defaults to the Sylvester Hadamard matrix (``d`` a power of two).
    """
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional vector")
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)

    d = x.shape[0]
    hadamard = _resolve_hadamard(d=d, h=h, device=x.device, dtype=x.dtype)
    z = _greedy_z(x, hadamard, max_search_range=max_search_range,
                  show_progress=show_progress)
    return _psa(x=x, z=z, hadamard=hadamard)


def pick_max_search_range(d: int, anchor: int = 1024) -> int:
    if d <= 0:
        raise ValueError("d must be a positive integer")
    if anchor <= 0:
        raise ValueError("anchor must be a positive integer")
    if not _is_power_of_two(anchor):
        raise ValueError("anchor must be a power of two")
    if d <= anchor:
        return anchor

    interval_upper_bound = 1 << math.ceil(math.log2(d))
    return max(1, anchor * anchor // interval_upper_bound)


if __name__ == "__main__":
    ## Example usage and comparison to random Hadamard projections ##
    import matplotlib.pyplot as plt
    from time import perf_counter

    torch.manual_seed(42)
    d = 8192
    num_rand_samples = 1000
    search_range_anchor = 1024
    emsr = pick_max_search_range(d, anchor=search_range_anchor)
    max_search_ranges = list(set([emsr, 32]))
    bins = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"d={d}, anchor={search_range_anchor}, empirical_search_range={emsr}, "
        f"random-Hadamard samples={num_rand_samples}"
    )

    x = torch.randn(d, device=device)
    outlier_indices = torch.randperm(d, device=device)[:3]
    x[outlier_indices] *= 24

    hadamard = _build_sylvester_hadamard(d=d, device=x.device, dtype=x.dtype)

    gha_results = {}
    for mr in max_search_ranges:
        label = f"gha(max_search_range={mr})" + (" [full]" if mr >= d or mr < 0 else " [windowed]")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        P, D = gha(x, h=hadamard, max_search_range=mr, show_progress=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - t0

        y = hadamard @ (D.to(x.dtype) * x[P])
        absmax = y.abs().max().item()
        rng = (y.max() - y.min()).item()
        gha_results[mr] = {"label": label, "absmax": absmax, "range": rng, "elapsed": elapsed}
        print(f"{label:<36s}  elapsed={elapsed:7.2f}s  absmax={absmax:.4f}  range={rng:.4f}")

    absmax_random = []
    range_random = []
    for _ in range(num_rand_samples):
        D_rand = (torch.randint(0, 2, (d,), dtype=x.dtype, device=x.device) * 2 - 1)
        y_rand = hadamard @ (D_rand * x)
        absmax_random.append(y_rand.abs().max().item())
        range_random.append((y_rand.max() - y_rand.min()).item())

    absmax_random_t = torch.tensor(absmax_random)
    range_random_t = torch.tensor(range_random)
    print(
        f"Random: absmax mean={absmax_random_t.mean():.4f} min={absmax_random_t.min():.4f} max={absmax_random_t.max():.4f}, "
        f"range  mean={range_random_t.mean():.4f} min={range_random_t.min():.4f} max={range_random_t.max():.4f}"
    )

    # Dynamic color / linestyle assignment so max_search_ranges may have any length.
    color_cycle = ["red", "green", "orange", "purple", "brown", "magenta", "teal", "olive"]
    linestyle_cycle = ["--", "-.", ":", "--", "-.", ":", "--", "-."]
    colors = {mr: color_cycle[i % len(color_cycle)] for i, mr in enumerate(max_search_ranges)}
    linestyles = {mr: linestyle_cycle[i % len(linestyle_cycle)] for i, mr in enumerate(max_search_ranges)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(absmax_random, bins=bins, alpha=0.7, color="steelblue",
                 label=f"Random Hadamard (n={num_rand_samples})")
    for mr in max_search_ranges:
        r = gha_results[mr]
        axes[0].axvline(r["absmax"], color=colors[mr], linestyle=linestyles[mr], linewidth=2,
                        label=f"{r['label']} = {r['absmax']:.4f}")
    axes[0].set_xlabel("absmax(y)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"absmax of y = H D P x  (d={d})")
    axes[0].legend()

    axes[1].hist(range_random, bins=bins, alpha=0.7, color="steelblue",
                 label=f"Random Hadamard (n={num_rand_samples})")
    for mr in max_search_ranges:
        r = gha_results[mr]
        axes[1].axvline(r["range"], color=colors[mr], linestyle=linestyles[mr], linewidth=2,
                        label=f"{r['label']} = {r['range']:.4f}")
    axes[1].set_xlabel("range(y) = max - min")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"range of y = H D P x  (d={d})")
    axes[1].legend()

    plt.tight_layout()
    out_path = "gha_vs_random.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")
