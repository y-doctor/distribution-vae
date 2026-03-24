"""Plot input vs reconstructed distributions from the best VAE checkpoint."""

from __future__ import annotations

import sys
import copy
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    GRID_SIZE,
    TIME_BUDGET,
    load_dataset,
    get_splits,
    get_dataloaders,
    evaluate,
    get_device,
)
from train import (
    DistributionVAE,
    LATENT_DIM,
    HIDDEN_DIM,
    BETA,
    FREE_BITS,
    BATCH_SIZE,
    SEED,
    LR,
    WEIGHT_DECAY,
    GRAD_CLIP,
    BETA_WARMUP_EPOCHS,
)


def train_and_get_best_model():
    """Train the model and return the best checkpoint + data."""
    torch.manual_seed(SEED)
    device = get_device()

    dataset = load_dataset()
    train_ds, val_ds = get_splits(dataset)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)

    model = DistributionVAE(
        grid_size=GRID_SIZE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        beta=BETA,
        free_bits=FREE_BITS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_time = time.time()
    epoch = 0
    best_val_metric = float("inf")
    best_state = None

    print("Training model...")
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            break

        if BETA_WARMUP_EPOCHS > 0 and epoch < BETA_WARMUP_EPOCHS:
            model.current_beta = BETA * (epoch / BETA_WARMUP_EPOCHS)
        else:
            model.current_beta = BETA

        model.train()
        for batch in train_loader:
            grids = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(grids)
            losses = model.compute_loss(grids, recon, mu, logvar)
            losses["total"].backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Eval every epoch for best checkpoint
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, val_loader, device)
            vkl = metrics["val_kl_divergence"]
            if abs(vkl) < best_val_metric:
                best_val_metric = abs(vkl)
                best_state = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | val_kl={vkl:.6f} | best={best_val_metric:.6f} | elapsed={elapsed:.0f}s")
        epoch += 1

    print(f"Training complete: {epoch} epochs, best |val_kl|={best_val_metric:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, val_loader, device


def plot_reconstructions(model, val_loader, device, n_examples=8):
    """Create a figure showing input vs reconstructed distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    # Get a batch of validation data
    batch = next(iter(val_loader))
    grids = batch[0].to(device)

    with torch.no_grad():
        recon, mu, logvar, z = model(grids)

    grids_np = grids.cpu().numpy()
    recon_np = recon.cpu().numpy()

    # Pick n_examples evenly spaced indices
    n = min(n_examples, len(grids_np))
    indices = np.linspace(0, len(grids_np) - 1, n, dtype=int)

    # Quantile positions (uniform grid from 0 to 1)
    quantiles = np.linspace(0, 1, GRID_SIZE)

    fig, axes = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(quantiles, grids_np[idx], "b-", linewidth=2, label="Input", alpha=0.8)
        ax.plot(quantiles, recon_np[idx], "r--", linewidth=2, label="Reconstructed", alpha=0.8)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Value")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribution VAE: Input vs Reconstructed Quantile Functions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(__file__).parent / "reconstruction_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)

    # Also make a density comparison plot
    fig2, axes2 = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 8))
    axes2 = axes2.flatten()

    for i, idx in enumerate(indices):
        ax = axes2[i]
        # Approximate density as 1/spacing between quantiles
        eps = 1e-8
        input_spacing = np.diff(grids_np[idx])
        recon_spacing = np.diff(recon_np[idx])
        input_density = 1.0 / np.maximum(input_spacing, eps)
        recon_density = 1.0 / np.maximum(recon_spacing, eps)

        # Normalize to make them comparable
        input_density = input_density / input_density.sum()
        recon_density = recon_density / recon_density.sum()

        midpoints = 0.5 * (grids_np[idx][:-1] + grids_np[idx][1:])

        ax.plot(midpoints, input_density, "b-", linewidth=1.5, label="Input density", alpha=0.8)
        ax.plot(midpoints, recon_density, "r--", linewidth=1.5, label="Recon density", alpha=0.8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density (approx)")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Distribution VAE: Input vs Reconstructed Density (approx)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path2 = Path(__file__).parent / "density_plot.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Density plot saved to: {out_path2}")
    plt.close(fig2)

    return out_path, out_path2


if __name__ == "__main__":
    model, val_loader, device = train_and_get_best_model()
    plot_reconstructions(model, val_loader, device)
