"""Evaluation metrics and plotting for the Distribution VAE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dist_vae.losses import cramer_distance, ks_distance_smooth, wasserstein1_distance
from dist_vae.model import DistributionVAE


@torch.no_grad()
def evaluate_reconstruction(
    model: DistributionVAE,
    dataset: Dataset,
    n_samples: int = 100,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate reconstruction quality on a dataset.

    Computes Cramer, KS, and Wasserstein-1 distances on held-out data.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to evaluate on.
        n_samples: Number of samples to evaluate.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with mean and std of each metric.
    """
    model.eval()
    device = next(model.parameters()).device

    # Limit dataset size
    n_samples = min(n_samples, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(n_samples))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_cramer, all_ks, all_w1 = [], [], []

    for batch in loader:
        grids = batch[0].to(device)
        recon, _, _, _ = model(grids)

        all_cramer.append(cramer_distance(grids, recon).cpu())
        all_ks.append(ks_distance_smooth(grids, recon).cpu())
        all_w1.append(wasserstein1_distance(grids, recon).cpu())

    cramer_vals = torch.cat(all_cramer)
    ks_vals = torch.cat(all_ks)
    w1_vals = torch.cat(all_w1)

    return {
        "cramer_mean": cramer_vals.mean().item(),
        "cramer_std": cramer_vals.std().item(),
        "ks_mean": ks_vals.mean().item(),
        "ks_std": ks_vals.std().item(),
        "wasserstein1_mean": w1_vals.mean().item(),
        "wasserstein1_std": w1_vals.std().item(),
        "n_samples": n_samples,
    }


def plot_reconstructions(
    model: DistributionVAE,
    dataset: Dataset,
    n_examples: int = 9,
    save_path: str | Path | None = None,
) -> None:
    """Plot input vs reconstructed quantile functions in a grid."""
    model.eval()
    device = next(model.parameters()).device

    n_cols = int(np.ceil(np.sqrt(n_examples)))
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    with torch.no_grad():
        for i in range(n_examples):
            grid, _, _ = dataset[i]
            grid_in = grid.unsqueeze(0).to(device)
            recon, _, _, _ = model(grid_in)

            ax = axes[i]
            x = np.linspace(0, 1, grid.shape[-1])
            ax.plot(x, grid.cpu().numpy(), label="Input", alpha=0.8)
            ax.plot(x, recon[0].cpu().numpy(), label="Recon", alpha=0.8, linestyle="--")
            ax.set_title(f"Sample {i}")
            if i == 0:
                ax.legend(fontsize=8)

    for i in range(n_examples, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latent_space(
    model: DistributionVAE,
    dataset: Dataset,
    color_by: str = "perturbation",
    method: str = "umap",
    save_path: str | Path | None = None,
) -> None:
    """Plot 2D embedding of the latent space."""
    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_mu, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            grids = batch[0].to(device)
            label_idx = batch[2] if color_by == "perturbation" else batch[1]
            mu, _ = model.encoder(grids)
            all_mu.append(mu.cpu())
            all_labels.append(label_idx)

    latents = torch.cat(all_mu).numpy()
    labels = torch.cat(all_labels).numpy()

    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            method = "pca"
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)

    embedding = reducer.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap="tab20", alpha=0.6, s=10,
    )
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"Latent space ({method.upper()}, colored by {color_by})")
    plt.colorbar(scatter, ax=ax, label=color_by)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_interpolations(
    model: DistributionVAE,
    dataset: Dataset,
    idx_pairs: list[tuple[int, int]],
    n_steps: int = 8,
    save_path: str | Path | None = None,
) -> None:
    """Plot latent space interpolations between distribution pairs."""
    model.eval()
    device = next(model.parameters()).device

    n_pairs = len(idx_pairs)
    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(2 * n_steps, 2 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, (idx_a, idx_b) in enumerate(idx_pairs):
            grid_a, _, _ = dataset[idx_a]
            grid_b, _, _ = dataset[idx_b]
            grid_a = grid_a.unsqueeze(0).to(device)
            grid_b = grid_b.unsqueeze(0).to(device)

            mu_a, _ = model.encoder(grid_a)
            mu_b, _ = model.encoder(grid_b)

            x = np.linspace(0, 1, model.grid_size)
            for col, alpha in enumerate(np.linspace(0, 1, n_steps)):
                z = (1 - alpha) * mu_a + alpha * mu_b
                recon = model.decoder(z)
                axes[row, col].plot(x, recon[0].cpu().numpy(), color="steelblue")
                axes[row, col].set_title(f"α={alpha:.2f}", fontsize=8)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latent_statistics(
    model: DistributionVAE,
    dataset: Dataset,
    save_path: str | Path | None = None,
) -> None:
    """Plot latent space statistics: dimension histograms and correlation matrix."""
    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_mu = []

    with torch.no_grad():
        for batch in loader:
            grids = batch[0].to(device)
            mu, _ = model.encoder(grids)
            all_mu.append(mu.cpu())

    latents = torch.cat(all_mu).numpy()
    n_dims = latents.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograms of first 8 dimensions
    n_show = min(8, n_dims)
    for i in range(n_show):
        axes[0].hist(latents[:, i], bins=30, alpha=0.5, label=f"z_{i}")
    axes[0].set_title("Latent dimension histograms")
    axes[0].legend(fontsize=7)

    # Correlation matrix
    corr = np.corrcoef(latents.T)
    im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Latent correlation matrix")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_eval_report(
    model: DistributionVAE,
    dataset: Dataset,
    output_dir: str | Path,
) -> dict:
    """Generate a complete evaluation report with all metrics and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics = evaluate_reconstruction(model, dataset, n_samples=min(500, len(dataset)))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_reconstructions(model, dataset, n_examples=9, save_path=output_dir / "reconstructions.png")

    n_data = len(dataset)
    if n_data >= 2:
        pairs = [(0, min(1, n_data - 1))]
        if n_data > 3:
            pairs.append((2, 3))
        plot_interpolations(model, dataset, idx_pairs=pairs, save_path=output_dir / "interpolations.png")

    if n_data >= 10:
        plot_latent_space(model, dataset, method="pca", save_path=output_dir / "latent_pca.png")

    plot_latent_statistics(model, dataset, save_path=output_dir / "latent_statistics.png")

    print(f"Evaluation report saved to {output_dir}")
    return metrics
