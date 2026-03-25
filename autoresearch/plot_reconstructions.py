"""Plot input vs reconstructed distributions from the best VAE checkpoint.

Uses dist_vae.eval plotting functions to show histogram overlays with
perturbation/gene labels.
"""

from __future__ import annotations

import sys
import copy
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from dist_vae.eval import (
    plot_reconstructions,
    plot_reconstructions_hist,
    plot_latent_space,
)


def train_and_get_best_model():
    """Train the model and return the best checkpoint + raw dataset."""
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-5)

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

        scheduler.step()

        # Eval every 5 epochs for best checkpoint
        if epoch % 5 == 0:
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

    return model, dataset, val_ds


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    model, dataset, val_ds = train_and_get_best_model()

    out_dir = Path(__file__).parent

    # Histogram plots with perturbation/gene labels (the good ones)
    print("\nGenerating histogram plots...")
    plot_reconstructions_hist(
        model, dataset, n_examples=9, n_bins=40,
        save_path=out_dir / "reconstruction_hist.png",
    )
    print(f"Histogram plot saved to: {out_dir / 'reconstruction_hist.png'}")

    # Quantile function plots with labels
    print("Generating quantile plots...")
    plot_reconstructions(
        model, dataset, n_examples=9,
        save_path=out_dir / "reconstruction_plot.png",
    )
    print(f"Quantile plot saved to: {out_dir / 'reconstruction_plot.png'}")

    # Latent space colored by perturbation
    print("Generating latent space plot...")
    plot_latent_space(
        model, dataset, color_by="perturbation", method="pca",
        save_path=out_dir / "latent_space.png",
    )
    print(f"Latent space plot saved to: {out_dir / 'latent_space.png'}")
