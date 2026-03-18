"""Distribution VAE model: encoder, decoder, and full VAE.

The encoder maps fixed-size quantile grids to a latent space.
The decoder maps latent vectors back to quantile grids, enforcing monotonicity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_vae.data import quantile_grid_to_samples, samples_to_quantile_grid
from dist_vae.losses import CombinedDistributionLoss


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
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode quantile grid to latent parameters.

        Args:
            x: Quantile grid, shape (batch, grid_size).

        Returns:
            Tuple of (mu, logvar), each shape (batch, latent_dim).
        """
        h = x.unsqueeze(1)  # (batch, 1, grid_size)
        h = self.convs(h)   # (batch, hidden_dim, 1)
        h = h.squeeze(-1)   # (batch, hidden_dim)
        return self.fc_mu(h), self.fc_logvar(h)


class DistributionDecoder(nn.Module):
    """1D CNN decoder with monotonicity enforcement.

    Architecture: Linear -> reshape -> ConvTranspose1d stack -> interpolate.
    Monotonicity is enforced via start_value + cumsum(softplus(deltas)).

    Args:
        grid_size: Size of the output quantile grid.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Base hidden dimension for conv layers.
    """

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.initial_length = 8

        self.fc = nn.Linear(latent_dim, hidden_dim * self.initial_length)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 4, 1, kernel_size=4, stride=2, padding=1),
        )

        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to monotonic quantile grid.

        Args:
            z: Latent vector, shape (batch, latent_dim).

        Returns:
            Monotonically non-decreasing quantile grid, shape (batch, grid_size).
        """
        h = self.fc(z)  # (batch, hidden_dim * initial_length)
        h = h.view(-1, self.hidden_dim, self.initial_length)  # (batch, hidden_dim, 8)
        h = self.deconvs(h)  # (batch, 1, ~64)
        h = h.squeeze(1)     # (batch, ~64)

        # Interpolate to target grid size
        if h.shape[-1] != self.grid_size:
            h = F.interpolate(
                h.unsqueeze(1), size=self.grid_size, mode="linear", align_corners=True
            ).squeeze(1)

        # Enforce monotonicity: start + cumsum(softplus(deltas))
        start = h[:, :1]
        deltas = F.softplus(h[:, 1:])
        return torch.cat([start, start + torch.cumsum(deltas, dim=-1)], dim=-1)


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
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.current_beta = beta
        self.free_bits = free_bits

        self.encoder = DistributionEncoder(grid_size, latent_dim, hidden_dim)
        self.decoder = DistributionDecoder(grid_size, latent_dim, hidden_dim)

        if loss_config is None:
            loss_config = {"cramer": 1.0}
        self.loss_fn = CombinedDistributionLoss(loss_config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z = mu + std * epsilon."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self, quantile_grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            quantile_grid: Input quantile grid, shape (batch, grid_size).

        Returns:
            Tuple of (reconstruction, mu, logvar, z).
        """
        mu, logvar = self.encoder(quantile_grid)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def compute_loss(
        self,
        input_grid: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss with distributional reconstruction loss and KL divergence.

        When free_bits > 0, applies per-dimension KL floor: each latent dimension
        must contribute at least `free_bits` nats before the KL penalty kicks in.
        This prevents posterior collapse by ensuring the model uses each dimension.
        """
        # Reconstruction loss
        recon_total, recon_components = self.loss_fn(input_grid, recon)

        # Per-dimension KL: 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (batch, latent_dim)

        if self.free_bits > 0:
            # Free-bits: clamp per-dimension KL to at least free_bits nats
            # Average over batch first, then clamp, then sum over dims
            kl_dim_mean = kl_per_dim.mean(dim=0)  # (latent_dim,)
            kl = torch.sum(torch.clamp(kl_dim_mean, min=self.free_bits))
        else:
            kl = kl_per_dim.sum(dim=-1).mean()

        total = recon_total + self.current_beta * kl

        result = {"total": total, "recon": recon_total, "kl": kl}
        for name, val in recon_components.items():
            result[f"recon_{name}"] = val.mean()
        return result

    def encode_distribution(
        self, raw_samples: torch.Tensor, n_valid: int | None = None
    ) -> torch.Tensor:
        """Convenience: encode raw samples to latent mean."""
        grid = samples_to_quantile_grid(raw_samples, self.grid_size, n_valid=n_valid)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        mu, _ = self.encoder(grid)
        return mu

    def decode_to_samples(self, z: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Convenience: decode latent vector to sorted samples."""
        grid = self.decoder(z)
        return quantile_grid_to_samples(grid, n_samples)

    @staticmethod
    def samples_to_grid(
        samples: torch.Tensor, grid_size: int = 256, n_valid: int | None = None
    ) -> torch.Tensor:
        """Convert raw samples to a quantile grid."""
        return samples_to_quantile_grid(samples, grid_size, n_valid=n_valid)
