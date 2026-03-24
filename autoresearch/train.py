"""Distribution VAE training — THE FILE THE AI AGENT MODIFIES.

This is a self-contained training script for the Distribution VAE.
The AI agent experiments by modifying the model architecture, loss functions,
hyperparameters, optimizer, scheduler, and training loop in this file.

The evaluation metric is val_kl_divergence (lower absolute value is better) —
the KL divergence between original and reconstructed distributions, computed by
the fixed evaluate() function in prepare.py. The agent's goal is to minimize
|val_kl_divergence| while keeping all latent dimensions active (no posterior collapse).

Usage:
    python autoresearch/train.py > run.log 2>&1

Output (parsed by the agent):
    val_kl_divergence=X.XXXXXX    (primary metric, lower absolute value is better)
    active_dims=N/M               (latent utilization, higher is better)
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed infrastructure — do not modify prepare.py
sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    GRID_SIZE,
    TIME_BUDGET,
    load_dataset,
    get_splits,
    get_dataloaders,
    evaluate,
    print_metrics,
    get_device,
)

# ============================================================================
# HYPERPARAMETERS — feel free to modify
# ============================================================================

LATENT_DIM = 16
HIDDEN_DIM = 128
BETA = 0.0001            # KL weight
BETA_WARMUP_EPOCHS = 20  # Linear warmup from 0 to BETA
FREE_BITS = 0.0          # Per-dim KL floor (0 = disabled)
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 256
SEED = 42

# ============================================================================
# LOSS FUNCTIONS — feel free to modify or add new ones
# ============================================================================


def cramer_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Cramer distance: MSE between quantile grids. Shape: (batch,)."""
    return torch.mean((sorted_x - sorted_y) ** 2, dim=-1)


def wasserstein1_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Wasserstein-1: MAE between quantile grids. Shape: (batch,)."""
    return torch.mean(torch.abs(sorted_x - sorted_y), dim=-1)


def ks_distance_smooth(
    sorted_x: torch.Tensor, sorted_y: torch.Tensor, temperature: float = 100.0
) -> torch.Tensor:
    """Smooth KS distance. Shape: (batch,)."""
    abs_diff = torch.abs(sorted_x - sorted_y)
    return torch.logsumexp(temperature * abs_diff, dim=-1) / temperature


def density_matching_loss(
    sorted_x: torch.Tensor, sorted_y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Loss on quantile spacing ratios — directly targets the KL divergence eval metric.

    KL(P||Q) ≈ mean(log(delta_recon / delta_input)), so we minimize
    the squared log-ratio of quantile spacings.
    """
    dx = torch.diff(sorted_x, dim=-1).clamp(min=eps)
    dy = torch.diff(sorted_y, dim=-1).clamp(min=eps)
    log_ratio = torch.log(dy / dx)
    return torch.mean(log_ratio ** 2, dim=-1)


# ============================================================================
# MODEL ARCHITECTURE — feel free to modify
# ============================================================================


class DistributionEncoder(nn.Module):
    """1D CNN encoder: quantile grid -> (mu, logvar)."""

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
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
        h = self.convs(x.unsqueeze(1)).squeeze(-1)
        return self.fc_mu(h), self.fc_logvar(h)


class DistributionDecoder(nn.Module):
    """1D CNN decoder with monotonicity enforcement: z -> quantile grid."""

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.initial_length = 8

        self.fc = nn.Linear(latent_dim, hidden_dim * self.initial_length)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 4, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.hidden_dim, self.initial_length)
        h = self.deconvs(h).squeeze(1)
        if h.shape[-1] != self.grid_size:
            h = F.interpolate(
                h.unsqueeze(1), size=self.grid_size, mode="linear", align_corners=True
            ).squeeze(1)
        # Enforce monotonicity: start + cumsum(softplus(deltas))
        start = h[:, :1]
        deltas = F.softplus(h[:, 1:])
        return torch.cat([start, start + torch.cumsum(deltas, dim=-1)], dim=-1)


class DistributionVAE(nn.Module):
    """Full VAE: encoder -> reparameterize -> decoder."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        beta: float = BETA,
        free_bits: float = FREE_BITS,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.current_beta = beta
        self.free_bits = free_bits

        self.encoder = DistributionEncoder(grid_size, latent_dim, hidden_dim)
        self.decoder = DistributionDecoder(grid_size, latent_dim, hidden_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
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
        # Reconstruction loss: KL divergence on quantile spacings
        # Matches the val_kl_divergence eval metric in prepare.py
        eps = 1e-8
        dx = torch.diff(input_grid, dim=-1).clamp(min=eps)
        dy = torch.diff(recon, dim=-1).clamp(min=eps)
        log_ratio = torch.log(dy / dx)
        recon_loss = torch.mean(log_ratio, dim=-1).abs().mean()

        # KL divergence
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if self.free_bits > 0:
            kl_dim_mean = kl_per_dim.mean(dim=0)
            kl = torch.sum(torch.clamp(kl_dim_mean, min=self.free_bits))
        else:
            kl = kl_per_dim.sum(dim=-1).mean()

        total = recon_loss + self.current_beta * kl
        return {"total": total, "recon": recon_loss, "kl": kl}


# ============================================================================
# TRAINING LOOP — feel free to modify
# ============================================================================


def train() -> None:
    """Main training function. Runs within TIME_BUDGET seconds."""
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load data (fixed)
    dataset = load_dataset()
    train_ds, val_ds = get_splits(dataset)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = DistributionVAE(
        grid_size=GRID_SIZE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        beta=BETA,
        free_bits=FREE_BITS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- Training loop (time-budgeted) ----
    import copy
    start_time = time.time()
    epoch = 0
    best_val_metric = float("inf")
    best_state = None

    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            break

        # KL warmup
        if BETA_WARMUP_EPOCHS > 0 and epoch < BETA_WARMUP_EPOCHS:
            model.current_beta = BETA * (epoch / BETA_WARMUP_EPOCHS)
        else:
            model.current_beta = BETA

        # Train epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            grids = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(grids)
            losses = model.compute_loss(grids, recon, mu, logvar)
            losses["total"].backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += losses["total"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Eval every epoch for optimal checkpointing
        if True:
            metrics = evaluate(model, val_loader, device)
            vkl = metrics["val_kl_divergence"]
            ad = metrics["active_dims"]
            td = metrics["total_dims"]
            print(
                f"Epoch {epoch:4d} | train_loss={avg_loss:.6f} | "
                f"val_kl_divergence={vkl:.6f} | active_dims={ad}/{td} | "
                f"elapsed={elapsed:.0f}s"
            )
            if abs(vkl) < best_val_metric:
                best_val_metric = abs(vkl)
                best_state = copy.deepcopy(model.state_dict())

        epoch += 1

    # ---- Final evaluation using best checkpoint ----
    total_time = time.time() - start_time
    print(f"\nTraining complete: {epoch} epochs in {total_time:.1f}s")
    print(f"Best |val_kl| during training: {best_val_metric:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, device)
    print("\n--- RESULTS ---")
    print_metrics(metrics)
    print(f"epochs={epoch}")
    print(f"n_params={n_params}")


if __name__ == "__main__":
    train()
