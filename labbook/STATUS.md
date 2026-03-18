# Project Status

**Last updated**: 2026-03-18 00:30 UTC
**Updated by**: Synthetic training + real data session

## What works
- `dist_vae/losses.py` — All 3 loss functions + ks_distance_smooth + CombinedDistributionLoss (47 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing, training curve plots
- `dist_vae/eval.py` — All evaluation functions and plotting (reconstructions, latent PCA, interpolations, latent statistics)
- `scripts/` — All CLI scripts + generate_synthetic_dataset.py + make_mini_dataset.py
- Full synthetic training verified: 100 epochs, loss converges, reconstructions reasonable
- Evaluation pipeline generates all plots successfully
- Package installable via `pip install -e ".[dev]"`
- All 47 tests pass on CPU
- Saved synthetic dataset at `data/synthetic_2k.h5ad` (2.1 MB)
- Saved mini Norman et al. dataset at `data/mini_perturb_seq.h5ad` (4.6 MB) — 9452 cells x 100 genes x 10 perturbations

## What's broken / blocked
- Nothing currently broken
- Partial latent collapse (z_1 near-zero variance) — needs hyperparameter tuning
- High latent correlations — poor disentanglement with current beta=0.01
- Sharp distribution features (step functions) poorly reconstructed

## What's in progress
- (none)

## Key finding: posterior collapse on real data
- 1000-epoch run on mini Norman data achieves Cramer=0.020 but KL=2.0, latent range [-0.1, 0.04]
- Model learns a single zero-inflated template, latent space is nearly unused
- beta=0.01 is too aggressive — need to reduce to 0.001 or use free-bits

## What's in the repo (data files)
- `data/synthetic_2k.h5ad` — 2000 synthetic distributions (2.1 MB, committed)
- `data/mini_perturb_seq.h5ad` — Mini Norman 2019 Perturb-seq: 9452 cells x 100 genes x 10 perts (4.6 MB, committed)
- `data/sample_perturb_seq.h5ad` — Full preprocessed Norman 2019: 111k cells x 500 HVGs (gitignored, regenerate via download script)

## Next priorities
1. Train on mini real data: `python scripts/train.py --config configs/default.yaml --adata data/mini_perturb_seq.h5ad`
2. Hyperparameter tuning: increase beta, reduce latent_dim, try hidden_dim=256
3. Add integration tests for training loop
4. Profile memory usage on large datasets

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Last tested on: CPU, Linux
