"""Distribution VAE model: encoder, decoder, and full VAE.

The encoder maps fixed-size quantile grids to a latent space.
The decoder maps latent vectors back to quantile grids, enforcing monotonicity.
"""

import torch
import torch.nn as nn


class DistributionEncoder(nn.Module):
    """1D CNN encoder for quantile grids.

    Architecture: Conv1d stack with GELU activations and stride-2 downsampling,
    followed by adaptive average pooling and linear projection to mu and logvar.

    Args:
        grid_size: Size of the input quantile grid.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Base hidden dimension for conv layers.
    """

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int = 128) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode quantile grid to latent parameters.

        Args:
            x: Quantile grid, shape (batch, grid_size).

        Returns:
            Tuple of (mu, logvar), each shape (batch, latent_dim).
        """
        raise NotImplementedError


class DistributionDecoder(nn.Module):
    """1D CNN decoder with monotonicity enforcement.

    Architecture: Linear → reshape → ConvTranspose1d stack → interpolate.
    Monotonicity is enforced via start_value + cumsum(softplus(deltas)).

    Args:
        grid_size: Size of the output quantile grid.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Base hidden dimension for conv layers.
    """

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int = 128) -> None:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to monotonic quantile grid.

        Args:
            z: Latent vector, shape (batch, latent_dim).

        Returns:
            Monotonically non-decreasing quantile grid, shape (batch, grid_size).
        """
        raise NotImplementedError


class DistributionVAE(nn.Module):
    """Variational Autoencoder for 1D empirical distributions.

    Encodes quantile grids into a latent space and decodes them back,
    with monotonicity-enforced outputs and distributional losses.

    Args:
        grid_size: Size of the quantile grid.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Base hidden dimension for encoder/decoder.
        beta: KL divergence weight (can be ramped via warmup).
        loss_config: Dictionary of loss weights for CombinedDistributionLoss.
    """

    def __init__(
        self,
        grid_size: int = 256,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        beta: float = 0.01,
        loss_config: dict[str, float] | None = None,
    ) -> None:
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z = mu + std * epsilon.

        Args:
            mu: Mean of the latent distribution, shape (batch, latent_dim).
            logvar: Log variance, shape (batch, latent_dim).

        Returns:
            Sampled latent vector, shape (batch, latent_dim).
        """
        raise NotImplementedError

    def forward(
        self, quantile_grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            quantile_grid: Input quantile grid, shape (batch, grid_size).

        Returns:
            Tuple of (reconstruction, mu, logvar, z).
        """
        raise NotImplementedError

    def compute_loss(
        self,
        input_grid: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss with distributional reconstruction loss and KL divergence.

        Args:
            input_grid: Original quantile grid, shape (batch, grid_size).
            recon: Reconstructed quantile grid, shape (batch, grid_size).
            mu: Latent mean, shape (batch, latent_dim).
            logvar: Latent log variance, shape (batch, latent_dim).

        Returns:
            Dictionary with keys 'total', 'recon', 'kl', plus per-component
            reconstruction losses.
        """
        raise NotImplementedError

    def encode_distribution(
        self, raw_samples: torch.Tensor, n_valid: int | None = None
    ) -> torch.Tensor:
        """Convenience: encode raw samples to latent mean.

        Sorts samples, interpolates to quantile grid, and returns mu.

        Args:
            raw_samples: Raw 1D samples, shape (batch, n_samples).
            n_valid: Number of valid samples per batch element (if padded).

        Returns:
            Latent mean mu, shape (batch, latent_dim).
        """
        raise NotImplementedError

    def decode_to_samples(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Convenience: decode latent vector to sorted samples.

        Args:
            z: Latent vector, shape (batch, latent_dim).
            n_samples: Number of output samples.

        Returns:
            Sorted samples, shape (batch, n_samples).
        """
        raise NotImplementedError

    @staticmethod
    def samples_to_grid(
        samples: torch.Tensor, grid_size: int = 256, n_valid: int | None = None
    ) -> torch.Tensor:
        """Convert raw samples to a quantile grid.

        Args:
            samples: Raw 1D samples, shape (batch, n_samples).
            grid_size: Target grid size.
            n_valid: Number of valid samples per batch element.

        Returns:
            Quantile grid, shape (batch, grid_size).
        """
        raise NotImplementedError
