"""Distributional loss functions operating on sorted quantile grids.

All functions expect inputs of shape (batch, grid_size) where values are
sorted (non-decreasing) along the grid dimension.
"""

import torch
import torch.nn as nn


def cramer_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Cramer distance: L2 between quantile functions (MSE over grid points).

    Equivalent to the squared Wasserstein-2 distance when computed on quantile
    functions evaluated at uniform probability points.

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


def kl_divergence_quantile(
    sorted_x: torch.Tensor,
    sorted_y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence estimated from quantile grids via density ratios.

    For a quantile function Q evaluated at uniform points p_i = i/(N-1),
    the density at Q(p_i) is f(Q(p_i)) = 1 / Q'(p_i), approximated by
    finite differences: f_i ≈ dp / (Q(p_{i+1}) - Q(p_i)).

    KL(P || Q_recon) = E_P[log(f_P / f_recon)]
                     = mean(log(delta_recon / delta_input))

    where delta = Q(p_{i+1}) - Q(p_i) are the quantile spacings.

    Args:
        sorted_x: Input quantile grid (P), shape (batch, grid_size).
        sorted_y: Reconstructed quantile grid (Q), shape (batch, grid_size).
        eps: Small constant for numerical stability.

    Returns:
        Per-sample KL divergence, shape (batch,).
    """
    # Quantile spacings (proportional to 1/density)
    dx = torch.diff(sorted_x, dim=-1).clamp(min=eps)
    dy = torch.diff(sorted_y, dim=-1).clamp(min=eps)

    # KL = E_P[log(f_P/f_Q)] = E_P[log(dy/dx)] since f ∝ 1/delta
    log_ratio = torch.log(dy / dx)

    return torch.mean(log_ratio, dim=-1)


class CombinedDistributionLoss(nn.Module):
    """Weighted combination of distributional losses.

    Args:
        weights: Dictionary mapping loss names to weights.
            Supported keys: 'cramer', 'wasserstein1', 'kl_divergence'.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

        self._loss_fns: dict[str, callable] = {
            "cramer": cramer_distance,
            "wasserstein1": wasserstein1_distance,
            "kl_divergence": kl_divergence_quantile,
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
