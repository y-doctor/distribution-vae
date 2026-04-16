"""Visualize what quantile-grid 'tokenization' of distributions looks like.

Generates a set of plots illustrating how variable-length 1D distributions are
converted into fixed-size quantile-grid vectors (tokens), with no learned
parameters. Saves PNGs under eval_results/quantile_tokenization/.

Run:
    python scripts/plot_quantile_tokenization.py \
        --adata data/mini_perturb_seq.h5ad \
        --output-dir eval_results/quantile_tokenization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from dist_vae.data import samples_to_quantile_grid


def sample_from_quantile_grid(
    grid: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n_samples from the distribution implied by a quantile grid.

    Uses inverse-CDF sampling: draw uniform u_j in [0, 1], then linear
    interpolation of the grid at those u_j values.
    """
    K = len(grid)
    q_grid = np.linspace(0.0, 1.0, K)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    return np.interp(u, q_grid, grid)


def pick_example_distributions(
    adata: anndata.AnnData,
    perturbation_key: str,
    n_examples: int = 6,
    min_cells: int = 50,
    seed: int = 0,
) -> list[tuple[str, str, np.ndarray]]:
    """Pick a diverse set of (gene, pert, samples) triples.

    Selection mixes high-variance, sparse (high zero-fraction), bimodal, and
    low-variance distributions so the plots show a range of shapes.
    """
    rng = np.random.default_rng(seed)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    perts = adata.obs[perturbation_key].values
    pert_counts = adata.obs[perturbation_key].value_counts()
    valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()

    # For each (gene, pert) compute summary stats, then pick diverse shapes.
    candidates = []
    for pert in valid_perts[:20]:
        cell_mask = perts == pert
        X_p = X[cell_mask]
        for gene_idx in range(X_p.shape[1]):
            vals = X_p[:, gene_idx]
            zero_frac = float((vals == 0).mean())
            std = float(vals.std())
            if std < 1e-6:
                continue
            candidates.append(
                {
                    "gene": adata.var_names[gene_idx],
                    "pert": pert,
                    "values": vals,
                    "zero_frac": zero_frac,
                    "std": std,
                    "mean": float(vals.mean()),
                    "n": len(vals),
                }
            )

    # Pick a diverse subset by sampling from different std/zero-frac buckets.
    candidates.sort(key=lambda c: c["std"])
    buckets = np.array_split(candidates, n_examples)
    picks = []
    for b in buckets:
        if len(b) == 0:
            continue
        chosen = b[rng.integers(0, len(b))]
        picks.append((chosen["gene"], chosen["pert"], chosen["values"]))
    return picks


def plot_tokenization_walkthrough(
    samples: np.ndarray, gene: str, pert: str, out_path: Path
) -> None:
    """4-panel walkthrough: samples -> sort -> quantile grids at K=32/64/256."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].hist(samples, bins=40, color="#4a90e2", edgecolor="white")
    axes[0].set_title(f"Raw samples (n={len(samples)})\n{gene} | {pert}")
    axes[0].set_xlabel("expression")
    axes[0].set_ylabel("count")

    sorted_s = np.sort(samples)
    quantiles = np.linspace(0, 1, len(sorted_s))
    axes[1].plot(quantiles, sorted_s, color="#4a90e2", lw=1.2)
    axes[1].set_title("Sorted samples = empirical quantile fn")
    axes[1].set_xlabel("quantile q")
    axes[1].set_ylabel("expression")

    samples_t = torch.tensor(samples, dtype=torch.float32)
    for ax, K, color in zip(axes[2:], [32, 256], ["#e67e22", "#27ae60"]):
        grid = samples_to_quantile_grid(samples_t, K).numpy()
        q_grid = np.linspace(0, 1, K)
        ax.plot(
            quantiles, sorted_s, color="#bbbbbb", lw=1.0, label="empirical"
        )
        ax.plot(q_grid, grid, color=color, lw=1.4, marker="o", ms=3, label=f"token (K={K})")
        ax.set_title(f"Tokenized: {K}-dim vector")
        ax.set_xlabel("quantile q")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Quantile-grid tokenization = sort samples and interpolate onto a uniform grid",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_gallery(
    picks: list[tuple[str, str, np.ndarray]], grid_size: int, out_path: Path
) -> None:
    """Gallery: for several distributions, show histogram + quantile-grid token."""
    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (gene, pert, samples) in enumerate(picks):
        samples_t = torch.tensor(samples, dtype=torch.float32)
        token = samples_to_quantile_grid(samples_t, grid_size).numpy()
        q = np.linspace(0, 1, grid_size)

        axes[i, 0].hist(samples, bins=40, color="#4a90e2", edgecolor="white")
        axes[i, 0].set_ylabel(f"{gene}\n{pert}\n(n={len(samples)})", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("Raw samples")

        axes[i, 1].plot(q, token, color="#27ae60", lw=1.2)
        axes[i, 1].fill_between(q, token.min(), token, color="#27ae60", alpha=0.15)
        if i == 0:
            axes[i, 1].set_title(f"Quantile-grid token (K={grid_size})")
        axes[i, 1].set_xlabel("quantile q" if i == n - 1 else "")

    fig.suptitle(
        "Each distribution becomes a fixed-size vector — the quantile-grid 'token'",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_tradeoff(
    samples: np.ndarray, gene: str, pert: str, out_path: Path
) -> None:
    """Show how different grid sizes K trade resolution vs dimensionality."""
    samples_t = torch.tensor(samples, dtype=torch.float32)
    sorted_s = np.sort(samples)
    emp_q = np.linspace(0, 1, len(sorted_s))

    Ks = [8, 16, 32, 64, 128, 256]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, K in zip(axes.flat, Ks):
        grid = samples_to_quantile_grid(samples_t, K).numpy()
        q = np.linspace(0, 1, K)
        ax.plot(emp_q, sorted_s, color="#bbbbbb", lw=1.0, label="empirical")
        ax.plot(q, grid, color="#e67e22", lw=1.2, marker="o", ms=3, label=f"K={K}")
        ax.set_title(f"K = {K}  ({K}-dim vector)")
        ax.legend(fontsize=8, loc="upper left")

    for ax in axes[-1]:
        ax.set_xlabel("quantile q")
    for ax in axes[:, 0]:
        ax.set_ylabel("expression")

    fig.suptitle(
        f"Resolution vs. token size — {gene} | {pert} (n={len(samples)})", y=1.0
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_token_matrix(
    adata: anndata.AnnData,
    perturbation_key: str,
    grid_size: int,
    out_path: Path,
    n_max: int = 500,
    min_cells: int = 50,
) -> None:
    """Heatmap of the full tokenized dataset (distributions x grid points)."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    perts = adata.obs[perturbation_key].values
    pert_counts = adata.obs[perturbation_key].value_counts()
    valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()

    tokens = []
    labels = []
    for pert in valid_perts:
        cell_mask = perts == pert
        X_p = X[cell_mask]
        for gene_idx in range(X_p.shape[1]):
            vals = X_p[:, gene_idx]
            if vals.std() < 1e-6:
                continue
            token = samples_to_quantile_grid(
                torch.tensor(vals, dtype=torch.float32), grid_size
            ).numpy()
            tokens.append(token)
            labels.append((adata.var_names[gene_idx], pert))
            if len(tokens) >= n_max:
                break
        if len(tokens) >= n_max:
            break

    tokens = np.stack(tokens)  # (N, K)

    # Sort rows by their median for a readable heatmap.
    order = np.argsort(np.median(tokens, axis=1))
    tokens_sorted = tokens[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})
    im = axes[0].imshow(
        tokens_sorted, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    axes[0].set_title(f"Token matrix: {tokens.shape[0]} distributions x {grid_size} grid points")
    axes[0].set_xlabel("grid index (quantile q)")
    axes[0].set_ylabel("distribution (sorted by median expression)")
    fig.colorbar(im, ax=axes[0], label="expression")

    # Right panel: overlay ~50 tokens to show the diversity of shapes.
    q = np.linspace(0, 1, grid_size)
    sample_idx = np.linspace(0, tokens.shape[0] - 1, 50).astype(int)
    for i in sample_idx:
        axes[1].plot(q, tokens[i], color="#333333", alpha=0.15, lw=0.8)
    axes[1].set_title("50 overlaid tokens (shape diversity)")
    axes[1].set_xlabel("quantile q")
    axes[1].set_ylabel("expression")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sample_size_effect(
    adata: anndata.AnnData,
    perturbation_key: str,
    grid_size: int,
    out_path: Path,
    seed: int = 0,
) -> None:
    """Show how token jitter scales with n_cells — why smoothing (VAE) helps."""
    rng = np.random.default_rng(seed)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Pick a gene-pert with lots of cells so we can subsample.
    perts = adata.obs[perturbation_key].values
    pert_counts = adata.obs[perturbation_key].value_counts()
    pert = pert_counts.index[0]
    cell_mask = perts == pert
    X_p = X[cell_mask]
    # Find a gene with reasonable variance.
    gene_idx = int(np.argmax(X_p.std(axis=0)))
    vals = X_p[:, gene_idx]
    gene = adata.var_names[gene_idx]

    sizes = [20, 50, 200, len(vals)]
    fig, axes = plt.subplots(1, len(sizes), figsize=(16, 4), sharey=True)
    q = np.linspace(0, 1, grid_size)
    for ax, n in zip(axes, sizes):
        # Plot 5 random subsamples to show jitter.
        for _ in range(5):
            idx = rng.choice(len(vals), size=min(n, len(vals)), replace=False)
            sub = vals[idx]
            token = samples_to_quantile_grid(
                torch.tensor(sub, dtype=torch.float32), grid_size
            ).numpy()
            ax.plot(q, token, lw=1.0, alpha=0.7)
        ax.set_title(f"n_cells = {n}")
        ax.set_xlabel("quantile q")
    axes[0].set_ylabel("expression")
    fig.suptitle(
        f"Sampling noise in the token vs. n_cells — {gene} | {pert} (5 resamples each)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_gallery(
    picks: list[tuple[str, str, np.ndarray]],
    grid_size: int,
    out_path: Path,
    seed: int = 0,
) -> None:
    """Side-by-side original vs reconstructed histograms for several distributions.

    Reconstruction: tokenize with K=grid_size, then inverse-CDF sample the same
    number of points from the token. The token is lossy, so the reconstructed
    histogram shows how much of the shape the token preserves.
    """
    rng = np.random.default_rng(seed)
    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n), sharex="row")
    if n == 1:
        axes = axes[None, :]

    for i, (gene, pert, samples) in enumerate(picks):
        token = samples_to_quantile_grid(
            torch.tensor(samples, dtype=torch.float32), grid_size
        ).numpy()
        recon = sample_from_quantile_grid(token, len(samples), rng)

        lo = float(min(samples.min(), recon.min()))
        hi = float(max(samples.max(), recon.max()))
        bins = np.linspace(lo, hi, 41)

        axes[i, 0].hist(samples, bins=bins, color="#4a90e2", edgecolor="white")
        axes[i, 0].set_ylabel(f"{gene}\n{pert}\n(n={len(samples)})", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("Original histogram")

        axes[i, 1].hist(recon, bins=bins, color="#27ae60", edgecolor="white")
        if i == 0:
            axes[i, 1].set_title(f"Reconstructed from K={grid_size} token")
        if i == n - 1:
            axes[i, 0].set_xlabel("expression")
            axes[i, 1].set_xlabel("expression")

    fig.suptitle(
        "Tokenize -> resample: how much of the histogram survives the quantile grid?",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_by_K(
    samples: np.ndarray,
    gene: str,
    pert: str,
    out_path: Path,
    seed: int = 0,
) -> None:
    """For one distribution, show original hist + reconstructions at K in {8..256}."""
    rng = np.random.default_rng(seed)
    Ks = [8, 16, 32, 64, 128, 256]

    lo = float(samples.min())
    hi = float(samples.max())
    bins = np.linspace(lo, hi, 41)

    fig, axes = plt.subplots(1, len(Ks) + 1, figsize=(20, 3.5), sharey=True)
    axes[0].hist(samples, bins=bins, color="#4a90e2", edgecolor="white")
    axes[0].set_title(f"Original\n{gene} | {pert} (n={len(samples)})")
    axes[0].set_xlabel("expression")
    axes[0].set_ylabel("count")

    for ax, K in zip(axes[1:], Ks):
        token = samples_to_quantile_grid(
            torch.tensor(samples, dtype=torch.float32), K
        ).numpy()
        recon = sample_from_quantile_grid(token, len(samples), rng)
        ax.hist(recon, bins=bins, color="#27ae60", edgecolor="white")
        ax.set_title(f"Reconstructed (K={K})")
        ax.set_xlabel("expression")

    fig.suptitle(
        "Original histogram vs reconstruction sampled from the quantile-grid token",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _reconstruction_losses(
    samples: np.ndarray, grid_size: int
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (W1, Cramer-RMSE, reference_quantile_fn, token_on_ref) for one dist.

    Evaluates both the empirical quantile function and the K-point token on a
    common high-resolution reference grid, then computes L1 (W1) and
    sqrt(L2-mean-squared) = Cramer-RMSE between them.
    """
    M = 4096
    sorted_s = np.sort(samples).astype(np.float64)
    ref_q = np.linspace(0.0, 1.0, M)
    emp_q = np.linspace(0.0, 1.0, len(sorted_s))
    emp_on_ref = np.interp(ref_q, emp_q, sorted_s)

    token = samples_to_quantile_grid(
        torch.tensor(samples, dtype=torch.float32), grid_size
    ).numpy().astype(np.float64)
    tok_q = np.linspace(0.0, 1.0, grid_size)
    tok_on_ref = np.interp(ref_q, tok_q, token)

    w1 = float(np.mean(np.abs(emp_on_ref - tok_on_ref)))
    cramer_rmse = float(np.sqrt(np.mean((emp_on_ref - tok_on_ref) ** 2)))
    return w1, cramer_rmse, emp_on_ref, tok_on_ref


def _pick_genes_and_perts(
    adata: anndata.AnnData,
    perturbation_key: str,
    n_genes: int = 3,
    n_perts: int = 3,
    min_cells: int = 100,
) -> tuple[list[str], list[str]]:
    """Pick n_perts with most cells, and n_genes with diverse variance."""
    pert_counts = adata.obs[perturbation_key].value_counts()
    perts = pert_counts[pert_counts >= min_cells].index.tolist()[:n_perts]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Rank genes by variance averaged over the chosen perts, then pick 3 spread out.
    stds = []
    for g in range(adata.n_vars):
        s = 0.0
        for p in perts:
            mask = (adata.obs[perturbation_key] == p).values
            s += float(X[mask, g].std())
        stds.append(s / len(perts))
    order = np.argsort(stds)
    # Pick low, middle, high variance.
    pick_idx = np.linspace(len(order) * 0.25, len(order) * 0.95, n_genes).astype(int)
    genes = [adata.var_names[order[i]] for i in pick_idx]
    return genes, perts


def plot_grid_3x3_histograms(
    adata: anndata.AnnData,
    perturbation_key: str,
    genes: list[str],
    perts: list[str],
    grid_size: int,
    out_path: Path,
    seed: int = 0,
) -> None:
    """3x3 (genes x perts) grid of overlaid original vs reconstructed histograms."""
    rng = np.random.default_rng(seed)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    n_rows, n_cols = len(genes), len(perts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for i, g in enumerate(genes):
        for j, p in enumerate(perts):
            mask = (adata.obs[perturbation_key] == p).values
            samples = X[mask, gene_to_idx[g]]
            if samples.std() < 1e-6:
                axes[i, j].set_title(f"{g} | {p} (constant)")
                continue

            w1, cramer_rmse, _, _ = _reconstruction_losses(samples, grid_size)
            token = samples_to_quantile_grid(
                torch.tensor(samples, dtype=torch.float32), grid_size
            ).numpy()
            recon = sample_from_quantile_grid(token, len(samples), rng)

            lo = float(min(samples.min(), recon.min()))
            hi = float(max(samples.max(), recon.max()))
            bins = np.linspace(lo, hi, 41)
            ax = axes[i, j]
            ax.hist(samples, bins=bins, color="#4a90e2", alpha=0.55, label="original")
            ax.hist(recon, bins=bins, color="#27ae60", alpha=0.55, label="reconstructed")
            ax.set_title(f"{g} | {p} (n={len(samples)})", fontsize=10)
            ax.text(
                0.98,
                0.95,
                f"W1={w1:.3g}\nCramer-RMSE={cramer_rmse:.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85),
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="center right")
            if i == n_rows - 1:
                ax.set_xlabel("expression")
            if j == 0:
                ax.set_ylabel("count")

    fig.suptitle(
        f"Original vs reconstructed histograms at K={grid_size} (3 genes x 3 perts)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_grid_3x3_quantiles(
    adata: anndata.AnnData,
    perturbation_key: str,
    genes: list[str],
    perts: list[str],
    grid_size: int,
    out_path: Path,
) -> None:
    """3x3 grid of overlaid empirical vs token quantile functions."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    n_rows, n_cols = len(genes), len(perts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    ref_q = np.linspace(0.0, 1.0, 4096)
    for i, g in enumerate(genes):
        for j, p in enumerate(perts):
            mask = (adata.obs[perturbation_key] == p).values
            samples = X[mask, gene_to_idx[g]]
            if samples.std() < 1e-6:
                axes[i, j].set_title(f"{g} | {p} (constant)")
                continue

            w1, cramer_rmse, emp_on_ref, tok_on_ref = _reconstruction_losses(
                samples, grid_size
            )
            ax = axes[i, j]
            ax.plot(ref_q, emp_on_ref, color="#4a90e2", lw=1.5, label="empirical")
            ax.plot(
                ref_q, tok_on_ref, color="#27ae60", lw=1.2, ls="--", label=f"token (K={grid_size})"
            )
            ax.set_title(f"{g} | {p} (n={len(samples)})", fontsize=10)
            ax.text(
                0.02,
                0.95,
                f"W1={w1:.3g}\nCramer-RMSE={cramer_rmse:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85),
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="lower right")
            if i == n_rows - 1:
                ax.set_xlabel("quantile q")
            if j == 0:
                ax.set_ylabel("expression")

    fig.suptitle(
        f"Empirical vs tokenized quantile functions at K={grid_size} (3 genes x 3 perts)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_loss_vs_K(
    adata: anndata.AnnData,
    perturbation_key: str,
    Ks: list[int],
    out_path: Path,
    min_cells: int = 50,
    max_dists: int = 300,
) -> dict:
    """Diminishing-returns: reconstruction loss vs K averaged across distributions.

    For each (gene, pert) with >= min_cells cells, evaluate the empirical
    quantile function and the K-point token on a common reference grid and
    compute W1 and Cramer-RMSE. Plot mean + IQR across distributions.

    Returns a dict with the numerical results.
    """
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    perts_col = adata.obs[perturbation_key].values
    pert_counts = adata.obs[perturbation_key].value_counts()
    valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()

    all_samples: list[np.ndarray] = []
    for p in valid_perts:
        mask = perts_col == p
        X_p = X[mask]
        for g in range(X_p.shape[1]):
            vals = X_p[:, g]
            if vals.std() < 1e-6:
                continue
            all_samples.append(vals.copy())
            if len(all_samples) >= max_dists:
                break
        if len(all_samples) >= max_dists:
            break

    w1_mat = np.zeros((len(all_samples), len(Ks)))
    cramer_mat = np.zeros_like(w1_mat)
    for i, s in enumerate(all_samples):
        for j, K in enumerate(Ks):
            w1, cramer_rmse, _, _ = _reconstruction_losses(s, K)
            w1_mat[i, j] = w1
            cramer_mat[i, j] = cramer_rmse

    def stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.mean(mat, axis=0),
            np.quantile(mat, 0.25, axis=0),
            np.quantile(mat, 0.75, axis=0),
        )

    w1_mean, w1_q1, w1_q3 = stats(w1_mat)
    cr_mean, cr_q1, cr_q3 = stats(cramer_mat)

    # Diminishing returns: normalize loss-reduction curve to [0, 1] and pick the
    # smallest K that captures 90 / 95 / 99 % of the loss reduction attainable
    # at Ks[-1]. "Knee" picked as the 95% point.
    def achieved(mean: np.ndarray) -> np.ndarray:
        baseline = mean[0]
        best = mean[-1]
        denom = max(baseline - best, 1e-12)
        return (baseline - mean) / denom

    w1_frac = achieved(w1_mean)
    cr_frac = achieved(cr_mean)

    def first_at(frac: np.ndarray, thr: float) -> int:
        for idx, f in enumerate(frac):
            if f >= thr:
                return Ks[idx]
        return Ks[-1]

    knees = {
        "W1_90": first_at(w1_frac, 0.90),
        "W1_95": first_at(w1_frac, 0.95),
        "W1_99": first_at(w1_frac, 0.99),
        "cramer_90": first_at(cr_frac, 0.90),
        "cramer_95": first_at(cr_frac, 0.95),
        "cramer_99": first_at(cr_frac, 0.99),
    }
    knee_K = knees["W1_95"]  # recommended default

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    for ax, mean, q1, q3, label, color in zip(
        axes[:2],
        [w1_mean, cr_mean],
        [w1_q1, cr_q1],
        [w1_q3, cr_q3],
        ["W1 (MAE between quantile fns)", "Cramer-RMSE (RMS between quantile fns)"],
        ["#e67e22", "#8e44ad"],
    ):
        ax.plot(Ks, mean, color=color, lw=2.0, marker="o", label="mean")
        ax.fill_between(Ks, q1, q3, color=color, alpha=0.2, label="IQR (25-75%)")
        ax.axvline(
            knee_K,
            color="#333333",
            ls="--",
            lw=1.2,
            label=f"95% knee @ K={knee_K}",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("token size K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    # Third panel: fraction of K=max loss reduction captured at each K.
    ax = axes[2]
    ax.plot(Ks, w1_frac, color="#e67e22", marker="o", lw=2.0, label="W1")
    ax.plot(Ks, cr_frac, color="#8e44ad", marker="s", lw=2.0, label="Cramer-RMSE")
    for thr, col, style in [(0.90, "#888888", ":"), (0.95, "#555555", "--"), (0.99, "#222222", "-.")]:
        ax.axhline(thr, color=col, ls=style, lw=1.0, alpha=0.7)
        ax.text(Ks[0], thr, f"  {int(thr*100)}%", color=col, fontsize=8, va="center")
    ax.axvline(knee_K, color="#333333", ls="--", lw=1.2, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("token size K")
    ax.set_ylabel("fraction of K=max loss-reduction captured")
    ax.set_title("Diminishing returns (normalized)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.0, 1.02)

    knees_str = (
        f"W1: 90% @ K={knees['W1_90']}  |  95% @ K={knees['W1_95']}"
        f"  |  99% @ K={knees['W1_99']}\n"
        f"Cramer: 90% @ K={knees['cramer_90']}  |  95% @ K={knees['cramer_95']}"
        f"  |  99% @ K={knees['cramer_99']}"
    )
    fig.suptitle(
        f"Reconstruction loss vs. K over {len(all_samples)} (gene, pert) distributions "
        f"(min {min_cells} cells)\n{knees_str}",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "Ks": list(Ks),
        "w1_mean": w1_mean.tolist(),
        "cramer_rmse_mean": cr_mean.tolist(),
        "knees": knees,
        "recommended_K": knee_K,
        "n_distributions": len(all_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, default="data/mini_perturb_seq.h5ad")
    parser.add_argument("--perturbation-key", type=str, default="perturbation")
    parser.add_argument("--grid-size", type=int, default=256)
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

    print("Picking example distributions ...")
    picks = pick_example_distributions(
        adata, args.perturbation_key, n_examples=6, seed=args.seed
    )

    print("Plot 1/5: tokenization walkthrough ...")
    gene, pert, samples = picks[len(picks) // 2]
    plot_tokenization_walkthrough(
        samples, gene, pert, out_dir / "01_walkthrough.png"
    )

    print("Plot 2/5: distribution gallery ...")
    plot_gallery(picks, args.grid_size, out_dir / "02_gallery.png")

    print("Plot 3/5: resolution tradeoff ...")
    plot_resolution_tradeoff(
        samples, gene, pert, out_dir / "03_resolution_tradeoff.png"
    )

    print("Plot 4/5: token matrix heatmap ...")
    plot_token_matrix(
        adata,
        args.perturbation_key,
        args.grid_size,
        out_dir / "04_token_matrix.png",
    )

    print("Plot 5/7: sampling-noise effect ...")
    plot_sample_size_effect(
        adata,
        args.perturbation_key,
        args.grid_size,
        out_dir / "05_sampling_noise.png",
        seed=args.seed,
    )

    print("Plot 6/7: reconstruction gallery (original vs reconstructed hist) ...")
    plot_reconstruction_gallery(
        picks,
        args.grid_size,
        out_dir / "06_reconstruction_gallery.png",
        seed=args.seed,
    )

    print("Plot 7/10: reconstruction by K ...")
    plot_reconstruction_by_K(
        samples, gene, pert, out_dir / "07_reconstruction_by_K.png", seed=args.seed
    )

    print("Plot 8/10: 3x3 histogram grid ...")
    genes_3, perts_3 = _pick_genes_and_perts(
        adata, args.perturbation_key, n_genes=3, n_perts=3
    )
    print(f"  genes: {genes_3}")
    print(f"  perts: {perts_3}")
    plot_grid_3x3_histograms(
        adata,
        args.perturbation_key,
        genes_3,
        perts_3,
        args.grid_size,
        out_dir / "08_3x3_histograms.png",
        seed=args.seed,
    )

    print("Plot 9/10: 3x3 quantile grid ...")
    plot_grid_3x3_quantiles(
        adata,
        args.perturbation_key,
        genes_3,
        perts_3,
        args.grid_size,
        out_dir / "09_3x3_quantiles.png",
    )

    print("Plot 10/10: loss vs K (diminishing returns) ...")
    Ks = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    stats = plot_loss_vs_K(
        adata,
        args.perturbation_key,
        Ks,
        out_dir / "10_loss_vs_K.png",
        min_cells=50,
        max_dists=300,
    )
    print(
        f"  knees = {stats['knees']}  "
        f"(recommended K = {stats['recommended_K']}, "
        f"over {stats['n_distributions']} distributions)"
    )

    print(f"Saved plots under {out_dir}")


if __name__ == "__main__":
    main()
