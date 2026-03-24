"""Fixed infrastructure for autoresearch experiments.

DO NOT MODIFY THIS FILE. The AI agent modifies train.py only.

This file provides:
- Data loading and train/val splitting (mini Perturb-seq real data only)
- The evaluation metric (val_kl_divergence) used to judge experiments
- A deterministic data pipeline so all experiments are comparable
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# ---- Constants (fixed across all experiments) ----

GRID_SIZE = 256
VAL_FRACTION = 0.1
SEED = 42
BATCH_SIZE = 256
TIME_BUDGET = 300  # 5 minutes wall-clock for training
DATA_PATH = Path(__file__).parent.parent / "data" / "mini_perturb_seq.h5ad"


def load_dataset() -> Dataset:
    """Load the mini Perturb-seq real dataset. Raises if not found."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data not found at {DATA_PATH}. "
            "Run: python scripts/make_mini_dataset.py"
        )

    import anndata as ad

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dist_vae.data import PerturbationDistributionDataset

    adata = ad.read_h5ad(DATA_PATH)
    dataset = PerturbationDistributionDataset(
        adata,
        perturbation_key="perturbation",
        grid_size=GRID_SIZE,
        min_cells=20,
    )
    print(f"Loaded real data: {len(dataset)} distributions from {DATA_PATH.name}")
    return dataset


def get_splits(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Deterministic train/val split."""
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    return train_ds, val_ds


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders with fixed seed."""
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
    )
    return train_loader, val_loader


def _kl_divergence_quantile(
    sorted_x: torch.Tensor, sorted_y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """KL divergence between two distributions estimated from their quantile grids.

    For quantile functions Q evaluated at uniform probability points,
    the density is f(Q(p)) ≈ dp / (Q(p_{i+1}) - Q(p_i)).
    KL(P || Q_recon) = E_P[log(f_P / f_recon)] = mean(log(delta_recon / delta_input)).

    Args:
        sorted_x: Input quantile grid (P), shape (batch, grid_size).
        sorted_y: Reconstructed quantile grid (Q), shape (batch, grid_size).
        eps: Small constant for numerical stability.

    Returns:
        Per-sample KL divergence, shape (batch,).
    """
    dx = torch.diff(sorted_x, dim=-1).clamp(min=eps)
    dy = torch.diff(sorted_y, dim=-1).clamp(min=eps)
    log_ratio = torch.log(dy / dx)
    return torch.mean(log_ratio, dim=-1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute validation metrics. This is the FIXED evaluation function.

    The primary metric is val_kl_divergence (lower absolute value is better):
    KL divergence between the original and reconstructed distributions,
    estimated from their quantile grids.

    Also reports val_cramer, val_w1, latent_kl, and active_dims.
    """
    model.eval()

    total_kl_divergence = 0.0
    total_cramer = 0.0
    total_w1 = 0.0
    total_latent_kl = 0.0
    n_batches = 0
    all_mu = []

    for batch in val_loader:
        grids = batch[0].to(device)
        recon, mu, logvar, z = model(grids)

        # KL divergence between input and reconstructed distributions (PRIMARY METRIC)
        kl_div = _kl_divergence_quantile(grids, recon).mean()
        # Cramer distance (MSE on quantile grids)
        cramer = torch.mean((grids - recon) ** 2, dim=-1).mean()
        # Wasserstein-1 (MAE on quantile grids)
        w1 = torch.mean(torch.abs(grids - recon), dim=-1).mean()
        # Latent KL (posterior vs prior)
        latent_kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=-1).mean()

        total_kl_divergence += kl_div.item()
        total_cramer += cramer.item()
        total_w1 += w1.item()
        total_latent_kl += latent_kl.item()
        n_batches += 1
        all_mu.append(mu.cpu())

    n = max(n_batches, 1)
    val_kl_divergence = total_kl_divergence / n
    val_cramer = total_cramer / n
    val_w1 = total_w1 / n
    val_latent_kl = total_latent_kl / n

    # Count active latent dimensions (std > 0.1 across validation set)
    latents = torch.cat(all_mu)
    dim_stds = latents.std(dim=0)
    active_dims = int((dim_stds > 0.1).sum().item())
    total_dims = latents.shape[1]

    return {
        "val_kl_divergence": val_kl_divergence,
        "val_cramer": val_cramer,
        "val_w1": val_w1,
        "val_latent_kl": val_latent_kl,
        "active_dims": active_dims,
        "total_dims": total_dims,
        "latent_mean_std": dim_stds.mean().item(),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Print metrics in a parseable format."""
    print(f"val_kl_divergence={metrics['val_kl_divergence']:.6f}")
    print(f"val_cramer={metrics['val_cramer']:.6f}")
    print(f"val_w1={metrics['val_w1']:.6f}")
    print(f"val_latent_kl={metrics['val_latent_kl']:.4f}")
    print(f"active_dims={metrics['active_dims']}/{metrics['total_dims']}")
    print(f"latent_mean_std={metrics['latent_mean_std']:.4f}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
