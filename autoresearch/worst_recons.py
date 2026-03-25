"""Train the model and plot the 9 worst-reconstructed samples by MSE."""
import sys, copy, time, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from prepare import GRID_SIZE, TIME_BUDGET, load_dataset, get_splits, get_dataloaders, evaluate, get_device
from train import (DistributionVAE, LATENT_DIM, HIDDEN_DIM, BETA, FREE_BITS, BATCH_SIZE,
                   SEED, LR, WEIGHT_DECAY, GRAD_CLIP, BETA_WARMUP_EPOCHS, extract_zero_frac)

torch.manual_seed(SEED)
device = get_device()
print(f"Device: {device}")

dataset = load_dataset()
train_ds, val_ds = get_splits(dataset)
train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Total dataset: {len(dataset)}")

model = DistributionVAE(grid_size=GRID_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                        beta=BETA, free_bits=FREE_BITS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-5)

start_time = time.time()
epoch = 0
best_val_metric = float('inf')
best_state = None

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
        losses['total'].backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    scheduler.step()
    epoch += 1
    if (epoch - 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, val_loader, device)
            vkl = metrics['val_kl_divergence']
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | val_kl={vkl:.6f} | elapsed={elapsed:.0f}s")
            if abs(vkl) < best_val_metric:
                best_val_metric = abs(vkl)
                best_state = copy.deepcopy(model.state_dict())

if best_state:
    model.load_state_dict(best_state)
model.eval()
total_time = time.time() - start_time
print(f"\nTraining complete: {epoch} epochs in {total_time:.1f}s")

# ---- Find 9 worst-reconstructed samples by MSE across the FULL dataset ----
print("Computing MSE for all samples in the full dataset...")

all_mse = []
all_grids = []
all_recons = []

# Use batch processing for efficiency
from torch.utils.data import DataLoader as _DL
full_loader = _DL(dataset, batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    for batch in full_loader:
        grids = batch[0].to(device)
        recon, _, _, _ = model(grids)
        mse = torch.mean((grids - recon) ** 2, dim=-1)  # (batch,)
        all_mse.append(mse.cpu())
        all_grids.append(grids.cpu())
        all_recons.append(recon.cpu())

all_mse = torch.cat(all_mse)       # (N,)
all_grids = torch.cat(all_grids)   # (N, grid_size)
all_recons = torch.cat(all_recons) # (N, grid_size)

# Sort by MSE descending, take top 9
worst_indices = torch.argsort(all_mse, descending=True)[:9]
print(f"Worst 9 MSE values: {all_mse[worst_indices].tolist()}")

# ---- Plot quantile functions for the 9 worst samples ----
positions = np.linspace(0, 255, GRID_SIZE)  # x-axis: position 0-255

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()

for plot_i, dataset_idx in enumerate(worst_indices.tolist()):
    grid = all_grids[dataset_idx].numpy()    # (grid_size,)
    recon = all_recons[dataset_idx].numpy()  # (grid_size,)
    mse_val = all_mse[dataset_idx].item()

    # zero_frac: fraction of quantile grid entries < 0.01
    zero_frac = float((grid < 0.01).mean())

    # Get metadata
    meta = dataset.get_metadata(dataset_idx)
    pert_name = meta.get('perturbation_name', meta.get('perturbation', 'unknown'))
    gene_name = meta.get('gene_name', meta.get('gene', 'unknown'))

    ax = axes[plot_i]
    ax.plot(positions, grid, color='steelblue', linewidth=1.2, label='Input', alpha=0.85)
    ax.plot(positions, recon, color='coral', linewidth=1.2, label='Recon', alpha=0.85, linestyle='--')

    title = f"{pert_name}/{gene_name}\nzf={zero_frac:.0%}, MSE={mse_val:.4f}"
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel('Quantile position', fontsize=7)
    ax.set_ylabel('Value', fontsize=7)
    ax.tick_params(labelsize=7)
    if plot_i == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.suptitle('9 Worst Reconstructions by MSE (Input vs Reconstruction — Quantile Functions)',
             fontsize=11, y=1.01)
plt.tight_layout()

out_path = Path(__file__).parent / 'worst_recons.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
plt.close()
