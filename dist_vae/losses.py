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
    return torch.mean((sorted_x - sorted_y) ** 2, dim=-1)


def wasserstein1_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Wasserstein-1 distance: L1 between quantile functions (MAE over grid points).

    Args:
        sorted_x: Sorted quantile grid, shape (batch, grid_size).
        sorted_y: Sorted quantile grid, shape (batch, grid_size).

    Returns:
        Per-sample W1 distance, shape (batch,).
    """
    return torch.mean(torch.abs(sorted_x - sorted_y), dim=-1)


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
    abs_diff = torch.abs(sorted_x - sorted_y)
    return temperature * torch.logsumexp(abs_diff / temperature, dim=-1)


class CombinedDistributionLoss(nn.Module):
    """Weighted combination of distributional losses.

    Args:
        weights: Dictionary mapping loss names to weights.
            Supported keys: 'cramer', 'wasserstein1', 'ks_smooth'.
        ks_temperature: Temperature for smooth KS distance.
    """

    def __init__(self, weights: dict[str, float], ks_temperature: float = 0.01) -> None:
        super().__init__()
        self.weights = weights
        self.ks_temperature = ks_temperature

        self._loss_fns: dict[str, callable] = {
            "cramer": cramer_distance,
            "wasserstein1": wasserstein1_distance,
            "ks_smooth": lambda x, y: ks_distance_smooth(x, y, temperature=self.ks_temperature),
        }

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
        components: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=sorted_x.device, dtype=sorted_x.dtype)

        for name, weight in self.weights.items():
            if weight > 0 and name in self._loss_fns:
                loss_val = self._loss_fns[name](sorted_x, sorted_y)
                components[name] = loss_val
                total = total + weight * loss_val.mean()

        return total, components
