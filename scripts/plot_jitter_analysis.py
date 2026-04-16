"""Quantify how much the quantile-grid token jitters with n_cells vs. signal.

For each (gene, pert) with >= n_max cells:
  - full_token = tokenize(all cells, K)
  - for n in [10, 20, 50, 100, 200, 500]:
      - repeat R times: sub-sample n cells, tokenize, compute W1 vs full_token

Compares sampling jitter to "signal" — the W1 between distinct (gene, pert)
tokens at full n. Outputs a publication-style panel.

Run:
    python scripts/plot_jitter_analysis.py \
        --adata data/mini_perturb_seq.h5ad \
        --output-dir eval_results/quantile_tokenization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from dist_vae.data import samples_to_quantile_grid


JITTER = "#d35400"
SIGNAL = "#1f4e79"
ACCENT = "#2c3e50"


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def _load_matrix(adata: anndata.AnnData) -> np.ndarray:
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X)


def _tokenize(samples: np.ndarray, K: int) -> np.ndarray:
    return (
        samples_to_quantile_grid(
            torch.tensor(samples, dtype=torch.float32), K
        )
        .numpy()
        .astype(np.float64)
    )


def _w1_between_tokens(a: np.ndarray, b: np.ndarray) -> float:
    """W1 (MAE) between two tokens of the same length."""
    return float(np.mean(np.abs(a - b)))


def _collect_pairs(
    adata: anndata.AnnData,
    perturbation_key: str,
    min_cells: int,
    max_pairs: int,
) -> list[tuple[str, str, np.ndarray]]:
    """Pairs (gene, pert, samples) with at least min_cells cells."""
    X = _load_matrix(adata)
    perts_col = adata.obs[perturbation_key].values
    counts = adata.obs[perturbation_key].value_counts()
    valid = counts[counts >= min_cells].index.tolist()
    out: list[tuple[str, str, np.ndarray]] = []
    for p in valid:
        mask = perts_col == p
        X_p = X[mask]
        if X_p.shape[0] < min_cells:
            continue
        for g in range(X_p.shape[1]):
            vals = X_p[:, g]
            if vals.std() < 1e-6:
                continue
            out.append((adata.var_names[g], p, vals.copy()))
            if len(out) >= max_pairs:
                return out
    return out


def compute_jitter(
    pairs: list[tuple[str, str, np.ndarray]],
    K: int,
    n_grid: list[int],
    R: int,
    seed: int = 0,
) -> np.ndarray:
    """Return W1 jitter matrix of shape (n_pairs, n_ns, R)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((len(pairs), len(n_grid), R))
    for i, (_, _, samples) in enumerate(pairs):
        full_token = _tokenize(samples, K)
        for j, n in enumerate(n_grid):
            if n > len(samples):
                W[i, j, :] = np.nan
                continue
            for r in range(R):
                idx = rng.choice(len(samples), size=n, replace=False)
                sub_token = _tokenize(samples[idx], K)
                W[i, j, r] = _w1_between_tokens(sub_token, full_token)
    return W


def compute_signal(
    pairs: list[tuple[str, str, np.ndarray]], K: int
) -> np.ndarray:
    """Return W1 between all distinct pairs of full-n tokens (flat array)."""
    tokens = np.stack([_tokenize(s, K) for _, _, s in pairs])
    diffs = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(_w1_between_tokens(tokens[i], tokens[j]))
    return np.asarray(diffs)


def build_panel(
    adata: anndata.AnnData,
    perturbation_key: str,
    K: int,
    n_grid: list[int],
    R: int,
    min_cells: int,
    max_pairs: int,
    out_png: Path,
    out_pdf: Path,
    seed: int = 0,
) -> dict:
    _set_style()
    print("  Collecting pairs ...")
    pairs = _collect_pairs(
        adata, perturbation_key, min_cells=min_cells, max_pairs=max_pairs
    )
    print(f"  {len(pairs)} (gene, pert) pairs with >= {min_cells} cells")

    print("  Computing jitter ...")
    W = compute_jitter(pairs, K, n_grid, R, seed=seed)  # (P, N, R)

    # Flatten across (pairs, R) for each n to get distribution over all trials.
    per_n_flat = [W[:, j, :].ravel() for j in range(len(n_grid))]
    per_n_flat = [x[np.isfinite(x)] for x in per_n_flat]

    j_mean = np.array([x.mean() for x in per_n_flat])
    j_q1 = np.array([np.quantile(x, 0.25) for x in per_n_flat])
    j_q3 = np.array([np.quantile(x, 0.75) for x in per_n_flat])
    j_med = np.array([np.median(x) for x in per_n_flat])

    print("  Computing signal W1 (all distinct full-n pairs) ...")
    signal = compute_signal(pairs, K)
    sig_med = float(np.median(signal))

    # SNR at each n: median_signal / median_jitter
    snr = sig_med / j_med

    # --- Figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    # (a) Jitter vs n_cells, with signal median as reference
    ax = axes[0]
    ax.plot(n_grid, j_mean, color=JITTER, lw=2.0, marker="o", ms=6, label="jitter mean")
    ax.fill_between(n_grid, j_q1, j_q3, color=JITTER, alpha=0.2, label="jitter IQR")
    ax.axhline(
        sig_med,
        color=SIGNAL,
        ls="--",
        lw=1.4,
        label=f"signal median = {sig_med:.3g}  (W1 between distinct (gene, pert) tokens)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.set_xlabel("n_cells used to tokenize")
    ax.set_ylabel("W1 (sub-sample token  vs  full-n token)")
    ax.set_title(
        f"(a)  Sampling jitter of the K={K} token vs. n_cells  ({len(pairs)} pairs, R={R})",
        loc="left",
    )
    ax.legend(loc="upper right")

    # (b) Signal vs jitter distribution at a single representative n
    ax = axes[1]
    lo = min(min(per_n_flat[0]), signal.min())
    hi = max(max(per_n_flat[-1]), signal.max())
    bins = np.logspace(np.log10(max(lo, 1e-4)), np.log10(hi * 1.05), 60)
    ax.hist(
        signal,
        bins=bins,
        color=SIGNAL,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
        label=f"signal: W1 between pairs (median {sig_med:.3g})",
    )
    # Overlay jitter at n=20 and n=200
    n_marks = [n_grid[0], n_grid[len(n_grid) // 2], n_grid[-1]]
    colors = ["#b03a2e", "#d35400", "#f1c40f"]
    for n, col in zip(n_marks, colors):
        j = n_grid.index(n)
        ax.hist(
            per_n_flat[j],
            bins=bins,
            color=col,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            label=f"jitter at n_cells={n}  (median {np.median(per_n_flat[j]):.3g})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("W1")
    ax.set_ylabel("count")
    ax.set_title("(b)  Signal vs jitter distributions  (log-x)", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    # (c) Signal-to-jitter ratio
    ax = axes[2]
    ax.plot(n_grid, snr, color=ACCENT, lw=2.0, marker="o", ms=6)
    for thr, col, style in [(1.0, "#e74c3c", ":"), (5.0, "#f39c12", "--"), (10.0, "#27ae60", "-.")]:
        ax.axhline(thr, color=col, ls=style, lw=1.0, alpha=0.85)
        ax.text(
            n_grid[0],
            thr * 1.08,
            f"SNR={thr:.0f}",
            color=col,
            fontsize=9,
            va="bottom",
            ha="left",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid])
    ax.set_xlabel("n_cells")
    ax.set_ylabel("SNR = median(signal W1) / median(jitter W1)")
    ax.set_title("(c)  Signal-to-jitter ratio vs. n_cells", loc="left")

    fig.suptitle(
        f"Quantile-grid token (K={K}) — sampling jitter vs. signal",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "K": K,
        "n_grid": list(n_grid),
        "jitter_median": j_med.tolist(),
        "jitter_mean": j_mean.tolist(),
        "signal_median": sig_med,
        "snr": snr.tolist(),
        "n_pairs": len(pairs),
        "R": R,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, default="data/mini_perturb_seq.h5ad")
    parser.add_argument("--perturbation-key", type=str, default="perturbation")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=700)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--R", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=str, default="eval_results/quantile_tokenization"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.adata} ...")
    adata = anndata.read_h5ad(args.adata)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    n_grid = [10, 20, 50, 100, 200, 500]
    stats = build_panel(
        adata,
        args.perturbation_key,
        K=args.K,
        n_grid=n_grid,
        R=args.R,
        min_cells=args.min_cells,
        max_pairs=args.max_pairs,
        out_png=out_dir / f"jitter_K{args.K}.png",
        out_pdf=out_dir / f"jitter_K{args.K}.pdf",
        seed=args.seed,
    )
    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
