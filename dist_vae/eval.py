"""Evaluation metrics and plotting for the Distribution VAE."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

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
    raise NotImplementedError


def plot_reconstructions(
    model: DistributionVAE,
    dataset: Dataset,
    n_examples: int = 9,
    save_path: str | Path | None = None,
) -> None:
    """Plot input vs reconstructed quantile functions in a grid.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to sample from.
        n_examples: Number of examples to plot (arranged in a square grid).
        save_path: Path to save the figure. If None, displays interactively.
    """
    raise NotImplementedError


def plot_latent_space(
    model: DistributionVAE,
    dataset: Dataset,
    color_by: str = "perturbation",
    method: str = "umap",
    save_path: str | Path | None = None,
) -> None:
    """Plot 2D embedding of the latent space.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to encode.
        color_by: What to color points by ('perturbation' or 'gene').
        method: Dimensionality reduction method ('umap' or 'pca').
        save_path: Path to save the figure.
    """
    raise NotImplementedError


def plot_interpolations(
    model: DistributionVAE,
    dataset: Dataset,
    idx_pairs: list[tuple[int, int]],
    n_steps: int = 8,
    save_path: str | Path | None = None,
) -> None:
    """Plot latent space interpolations between distribution pairs.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to sample from.
        idx_pairs: List of (idx_a, idx_b) pairs to interpolate between.
        n_steps: Number of interpolation steps.
        save_path: Path to save the figure.
    """
    raise NotImplementedError


def plot_latent_statistics(
    model: DistributionVAE,
    dataset: Dataset,
    save_path: str | Path | None = None,
) -> None:
    """Plot latent space statistics: dimension histograms and correlation matrix.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to encode.
        save_path: Path to save the figure.
    """
    raise NotImplementedError


def generate_eval_report(
    model: DistributionVAE,
    dataset: Dataset,
    output_dir: str | Path,
) -> dict:
    """Generate a complete evaluation report with all metrics and plots.

    Args:
        model: Trained DistributionVAE.
        dataset: Dataset to evaluate on.
        output_dir: Directory to save all outputs.

    Returns:
        Dictionary of all computed metrics.
    """
    raise NotImplementedError
