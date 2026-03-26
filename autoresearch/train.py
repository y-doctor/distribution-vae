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

LATENT_DIM = 24
HIDDEN_DIM = 256
BETA = 0.00005           # KL weight (lower to let recon dominate)
BETA_WARMUP_EPOCHS = 30  # Linear warmup from 0 to BETA
FREE_BITS = 0.5          # Per-dim KL floor — prevents dimension collapse
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 256
SEED = 42
DENSITY_WEIGHT = 0.15    # Weight for density matching loss (targets KL metric)
TAIL_WEIGHT = 2.0        # Extra weight on tail quantiles (top 20%)
ZERO_EPS = 0.01          # Threshold for counting "zero" values in quantile grid
MASK_STEEPNESS = 100.0   # Steepness of the sigmoid mask at the zero-nonzero transition
MASK_MARGIN = 0.02       # Margin so mask is ~1 at position 0 when zero_frac=0

# ============================================================================
# LOSS FUNCTIONS — feel free to modify or add new ones
# ============================================================================


def cramer_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Cramer distance: MSE between quantile grids. Shape: (batch,)."""
    return torch.mean((sorted_x - sorted_y) ** 2, dim=-1)


def tail_weighted_cramer(
    sorted_x: torch.Tensor, sorted_y: torch.Tensor, tail_weight: float = 2.0
) -> torch.Tensor:
    """Cramer distance with extra weight on tail quantiles (top 20%)."""
    n = sorted_x.shape[-1]
    weights = torch.ones(n, device=sorted_x.device)
    tail_start = int(0.8 * n)
    weights[tail_start:] = tail_weight
    weights = weights / weights.mean()
    sq_diff = (sorted_x - sorted_y) ** 2
    return torch.mean(sq_diff * weights.unsqueeze(0), dim=-1)


def wasserstein1_distance(sorted_x: torch.Tensor, sorted_y: torch.Tensor) -> torch.Tensor:
    """Wasserstein-1: MAE between quantile grids. Shape: (batch,)."""
    return torch.mean(torch.abs(sorted_x - sorted_y), dim=-1)


def density_matching_loss(
    sorted_x: torch.Tensor, sorted_y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Loss on quantile spacing ratios — directly targets the KL divergence eval metric."""
    dx = torch.diff(sorted_x, dim=-1).clamp(min=eps)
    dy = torch.diff(sorted_y, dim=-1).clamp(min=eps)
    log_ratio = torch.log(dy / dx)
    return torch.mean(log_ratio ** 2, dim=-1)


def extract_zero_frac(x: torch.Tensor, eps: float = ZERO_EPS) -> torch.Tensor:
    """Extract zero-fraction from quantile grid. Shape: (batch,) -> (batch, 1)."""
    return (x < eps).float().mean(dim=-1, keepdim=True)


# ============================================================================
# MODEL ARCHITECTURE — zero-fraction conditioned VAE with scale normalization
# ============================================================================


class DistributionEncoder(nn.Module):
    """1D CNN encoder with scale normalization + zero_frac conditioning.

    Normalizes the input grid by its range before encoding, so the latent
    space only needs to capture the shape, not the scale. zero_frac and
    scale info are passed as additional channels.
    """

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        c1, c2, c3 = hidden_dim // 4, hidden_dim // 2, hidden_dim

        # 3 input channels: (normalized_grid, zero_frac_broadcast, scale_broadcast)
        self.conv1 = nn.Conv1d(3, c1, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(c3)

        # Residual block
        self.res_conv1 = nn.Conv1d(c3, c3, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm1d(c3)
        self.res_conv2 = nn.Conv1d(c3, c3, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm1d(c3)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, zero_frac: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, G = x.shape
        # Normalize grid by scale (range), so model sees shape only
        x_norm = x / (scale + 1e-6)
        # 3-channel input: normalized grid, zero_frac, log(scale)
        zf_ch = zero_frac.expand(-1, G).unsqueeze(1)
        sc_ch = torch.log1p(scale).expand(-1, G).unsqueeze(1)
        h = torch.cat([x_norm.unsqueeze(1), zf_ch, sc_ch], dim=1)  # (B, 3, G)
        h = F.gelu(self.bn1(self.conv1(h)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        r = F.gelu(self.res_bn1(self.res_conv1(h)))
        h = h + self.res_bn2(self.res_conv2(r))
        h = F.gelu(h)
        h = self.pool(h).squeeze(-1)
        return self.fc_mu(h), self.fc_logvar(h)


class DistributionDecoder(nn.Module):
    """MLP decoder with mask margin fix + scale denormalization.

    Predicts shape in normalized [0,1] space, applies zero-frac mask
    with margin (so zf=0 distributions aren't suppressed at position 0),
    then scales back to original range.
    """

    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.register_buffer("positions", torch.linspace(0, 1, grid_size))

        # Input: z (latent_dim) + zero_frac (1) + log_scale (1)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, grid_size),
        )

    def forward(
        self, z: torch.Tensor, zero_frac: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        log_scale = torch.log1p(scale)
        h = self.mlp(torch.cat([z, zero_frac, log_scale], dim=-1))

        # Build the normalized shape via cumsum(sharp_softplus)
        start = h[:, :1]
        deltas = F.softplus(h[:, 1:], beta=15)
        shape = torch.cat([start, start + torch.cumsum(deltas, dim=-1)], dim=-1)

        # Mask with margin: when zero_frac=0, mask at position 0 = sigmoid(100*0.02) ≈ 0.88
        # This prevents suppression of non-zero-inflated distributions
        mask = torch.sigmoid(
            MASK_STEEPNESS * (self.positions.unsqueeze(0) - zero_frac + MASK_MARGIN)
        )

        # Apply mask in normalized space, then scale back
        return shape * mask * scale


class DistributionVAE(nn.Module):
    """Zero-fraction + scale-normalized VAE with KL balancing.

    Embedding = [zero_frac, log_scale, mu_1, ..., mu_N]
    """

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
        zero_frac = extract_zero_frac(x)  # (B, 1)
        scale = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values)  # (B, 1)
        mu, logvar = self.encoder(x, zero_frac, scale)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, zero_frac, scale)
        return recon, mu, logvar, z

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the full embedding: [zero_frac, log_scale, mu_1, ..., mu_N]."""
        zero_frac = extract_zero_frac(x)
        scale = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values)
        mu, _ = self.encoder(x, zero_frac, scale)
        return torch.cat([zero_frac, torch.log1p(scale), mu], dim=-1)

    def compute_loss(
        self,
        input_grid: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Reconstruction loss: tail-weighted Cramer + density matching
        cramer = tail_weighted_cramer(input_grid, recon, tail_weight=TAIL_WEIGHT).mean()
        density = density_matching_loss(input_grid, recon).mean()
        recon_loss = cramer + DENSITY_WEIGHT * density

        # KL divergence with free_bits for balanced latent usage
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

    # Optimizer + cosine annealing scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-5)

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

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Eval every 5 epochs to save time for more training
        if epoch % 5 == 0 or elapsed > TIME_BUDGET - 15:
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
