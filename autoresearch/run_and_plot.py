"""Train the model and generate histogram plots on validation data."""
import sys, copy, time, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from prepare import GRID_SIZE, TIME_BUDGET, load_dataset, get_splits, get_dataloaders, evaluate, print_metrics, get_device
from train import (DistributionVAE, LATENT_DIM, HIDDEN_DIM, BETA, FREE_BITS, BATCH_SIZE,
                    SEED, LR, WEIGHT_DECAY, GRAD_CLIP, BETA_WARMUP_EPOCHS)

torch.manual_seed(SEED); device = get_device()
dataset = load_dataset()
train_ds, val_ds = get_splits(dataset)
train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
model = DistributionVAE(grid_size=GRID_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                         beta=BETA, free_bits=FREE_BITS).to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-5)

start_time = time.time(); epoch = 0; best_val_metric = float('inf'); best_state = None
while True:
    elapsed = time.time() - start_time
    if elapsed > TIME_BUDGET: break
    if BETA_WARMUP_EPOCHS > 0 and epoch < BETA_WARMUP_EPOCHS:
        model.current_beta = BETA * (epoch / BETA_WARMUP_EPOCHS)
    else: model.current_beta = BETA
    model.train()
    for batch in train_loader:
        grids = batch[0].to(device); optimizer.zero_grad()
        recon, mu, logvar, z = model(grids)
        losses = model.compute_loss(grids, recon, mu, logvar)
        losses['total'].backward()
        if GRAD_CLIP > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    scheduler.step(); epoch += 1
    if (epoch - 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, val_loader, device)
            vkl = metrics['val_kl_divergence']
            if abs(vkl) < best_val_metric:
                best_val_metric = abs(vkl); best_state = copy.deepcopy(model.state_dict())

if best_state: model.load_state_dict(best_state)
metrics = evaluate(model, val_loader, device)
print_metrics(metrics)
print(f"epochs={epoch}"); print(f"n_params={n_params}")

# Plot histograms on a mix of zero-inflated and high-variance val samples
indices = [0, 5, 99, 187, 199, 498, 596, 898, 997]
model.eval(); fig, axes = plt.subplots(3, 3, figsize=(12, 9)); axes = axes.flatten()
with torch.no_grad():
    for i, idx in enumerate(indices):
        grid, _, _ = dataset[idx]
        recon, _, _, _ = model(grid.unsqueeze(0).to(device))
        ax = axes[i]
        ax.hist(grid.cpu().numpy(), bins=40, density=True, alpha=0.5, color='steelblue', label='Input', edgecolor='none')
        ax.hist(recon[0].cpu().numpy(), bins=40, density=True, alpha=0.5, color='coral', label='Recon', edgecolor='none')
        meta = dataset.get_metadata(idx)
        zf = (grid < 0.01).float().mean().item()
        ax.set_title(f"{meta['perturbation_name']}/{meta['gene_name']} (zf={zf:.0%})", fontsize=9)
        ax.set_xlabel('value', fontsize=8); ax.set_yticks([])
        if i == 0: ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig('autoresearch/reconstruction_hist.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to autoresearch/reconstruction_hist.png"); plt.close()
