"""Distributional loss functions operating on sorted quantile grids.

All functions expect inputs of shape (batch, grid_size) where values are
sorted (non-decreasing) along the grid dimension.
"""

import torch
import torch.nn as nn


def cramer_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Cramer distance: L2 between quantile functions (MSE over grid points).

    Args:
        sorted_x: Sorted quantile grid, shape (batch, grid_size).
        sorted_y: Sorted quantile grid, shape (batch, grid_size).

    Returns:
        Per-sample Cramer distance, shape (batch,).
    """
    raise NotImplementedError


def wasserstein1_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Wasserstein-1 distance: L1 between quantile functions (MAE over grid points).

    Args:
        sorted_x: Sorted quantile grid, shape (batch, grid_size).
        sorted_y: Sorted quantile grid, shape (batch, grid_size).

    Returns:
        Per-sample W1 distance, shape (batch,).
    """
    raise NotImplementedError


def ks_distance_smooth(
    sorted_x: torch.Tensor,
    sorted_y: torch.Tensor,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Smooth Kolmogorov-Smirnov distance via logsumexp approximation of max.

    Computes a differentiable approximation of sup|F(t) - G(t)| using
    logsumexp as a smooth max over pointwise absolute differences.

    Args:
        sorted_x: Sorted quantile grid, shape (batch, grid_size).
        sorted_y: Sorted quantile grid, shape (batch, grid_size).
        temperature: Smoothing temperature for logsumexp. Lower = closer to true max.

    Returns:
        Per-sample smooth KS distance, shape (batch,).
    """
    raise NotImplementedError


class CombinedDistributionLoss(nn.Module):
    """Weighted combination of distributional losses.

    Args:
        weights: Dictionary mapping loss names to weights.
            Supported keys: 'cramer', 'wasserstein1', 'ks_smooth'.
        ks_temperature: Temperature for smooth KS distance.
    """

    def __init__(self, weights: dict[str, float], ks_temperature: float = 0.01) -> None:
        raise NotImplementedError

    def forward(
        self, sorted_x: torch.Tensor, sorted_y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted combination of distributional losses.

        Args:
            sorted_x: Sorted quantile grid, shape (batch, grid_size).
            sorted_y: Sorted quantile grid, shape (batch, grid_size).

        Returns:
            Tuple of (total_loss, component_dict) where total_loss is a scalar
            and component_dict maps loss names to per-sample values (batch,).
        """
        raise NotImplementedError
