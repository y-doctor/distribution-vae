# Project Status

**Last updated**: 2026-03-18 00:15 UTC
**Updated by**: First synthetic training run session

## What works
- `dist_vae/losses.py` — All 3 loss functions + ks_distance_smooth + CombinedDistributionLoss (47 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing, training curve plots
- `dist_vae/eval.py` — All evaluation functions and plotting (reconstructions, latent PCA, interpolations, latent statistics)
- `scripts/` — All CLI scripts + new generate_synthetic_dataset.py
- Full synthetic training verified: 100 epochs, loss converges, reconstructions reasonable
- Evaluation pipeline generates all plots successfully
- Package installable via `pip install -e ".[dev]"`
- All 47 tests pass on CPU
- Saved synthetic dataset at `data/synthetic_2k.h5ad` (2.1 MB)

## What's broken / blocked
- Nothing currently broken
- Partial latent collapse (z_1 near-zero variance) — needs hyperparameter tuning
- High latent correlations — poor disentanglement with current beta=0.01
- Sharp distribution features (step functions) poorly reconstructed

## What's in progress
- (none)

## Next priorities
1. Hyperparameter tuning: increase beta, reduce latent_dim, try hidden_dim=256
2. Download and test with real Perturb-seq data
3. Add integration tests for training loop
4. Profile memory usage on large datasets

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Last tested on: CPU, Linux
