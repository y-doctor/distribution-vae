"""Train the model and generate a diagnostic plot showing the 9 worst reconstructions by MSE."""
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
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

model = DistributionVAE(
    grid_size=GRID_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
    beta=BETA, free_bits=FREE_BITS
).to(device)
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
            if abs(vkl) < best_val_metric:
                best_val_metric = abs(vkl)
                best_state = copy.deepcopy(model.state_dict())

print(f"Training complete: {epoch} epochs in {time.time() - start_time:.1f}s")

if best_state:
    model.load_state_dict(best_state)
model.eval()

# ---- Find 9 worst-reconstructed samples by MSE from the full dataset ----
print(f"Scanning full dataset ({len(dataset)} samples) for worst reconstructions...")

all_mse = []
with torch.no_grad():
    for idx in range(len(dataset)):
        grid, _, _ = dataset[idx]
        recon, _, _, _ = model(grid.unsqueeze(0).to(device))
        mse = torch.mean((grid.to(device) - recon[0]) ** 2).item()
        all_mse.append((mse, idx))

# Sort descending by MSE, take top 9
all_mse.sort(key=lambda x: x[0], reverse=True)
worst_9 = all_mse[:9]
print("Worst 9 MSE values:", [f"{m:.6f}" for m, _ in worst_9])

# ---- Plot quantile functions as lines (x=position 0-255, y=value) ----
positions = np.arange(GRID_SIZE)  # 0..255

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

with torch.no_grad():
    for i, (mse, idx) in enumerate(worst_9):
        grid, _, _ = dataset[idx]
        recon, _, _, _ = model(grid.unsqueeze(0).to(device))

        input_vals = grid.cpu().numpy()          # shape (256,)
        recon_vals = recon[0].cpu().numpy()      # shape (256,)

        zero_frac = (grid < 0.01).float().mean().item()
        meta = dataset.get_metadata(idx)
        pert_name = meta.get('perturbation_name', f'idx{idx}')
        gene_name = meta.get('gene_name', '')

        ax = axes[i]
        ax.plot(positions, input_vals, color='steelblue', linewidth=1.0, label='Input', alpha=0.85)
        ax.plot(positions, recon_vals, color='coral', linewidth=1.0, label='Recon', alpha=0.85, linestyle='--')

        title = f"{pert_name}/{gene_name}\nzero_frac={zero_frac:.0%}, MSE={mse:.5f}"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Quantile position (0–255)', fontsize=7)
        ax.set_ylabel('Value', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7)

plt.suptitle('9 Worst Reconstructions by MSE (full dataset)', fontsize=11, y=1.01)
plt.tight_layout()

out_path = Path(__file__).parent / 'worst_recons.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {out_path}")
