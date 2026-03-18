# Project Status

**Last updated**: 2026-03-18 06:30 UTC
**Updated by**: Session — 100-trial hyperopt on mini Norman

## What works
- `dist_vae/losses.py` — All loss functions + CombinedDistributionLoss (47 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE with free-bits support
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing, perturbation/gene labels in snapshots
- `dist_vae/eval.py` — All evaluation functions with perturbation/gene labels in all plots
- `scripts/` — All CLI scripts (train, evaluate, encode, download, hyperopt)
- **Posterior collapse fixed** — beta=0.0001 + latent_dim=16 gives Cramer=0.0092, all 16 dims active
- **Hyperopt complete** — 100 trials on mini Norman: best is latent_dim=16, hidden_dim=256, beta=0.0017, lr=0.00125, batch_size=64, Cramer-only loss
- Package installable via `pip install -e ".[dev]"`
- All 47 tests pass on CPU

## What's broken / blocked
- Latent correlations still high — disentanglement could be improved
- Tail reconstruction still imperfect for zero-inflated distributions
- Only tested on mini Norman (100 genes, 10 perturbations) — needs full-scale validation

## What's in progress
- Nothing — session ending

## Next priorities
1. Train with best hyperopt config (configs/best_hyperopt.yaml) for full 500 epochs and evaluate
2. Test on full 500-gene Norman dataset
3. Investigate total correlation penalty for disentanglement
4. Add integration tests for training loop

## What's in the repo (data files)
- `data/synthetic_2k.h5ad` — 2000 synthetic distributions (2.1 MB, committed)
- `data/mini_perturb_seq.h5ad` — Mini Norman 2019: 9452 cells x 100 genes x 10 perts (4.6 MB, committed)

## Eval results
- `eval_results/real_1k_epochs/` — Baseline: beta=0.01, d=32, 1000 epochs (COLLAPSED)
- `eval_results/beta_0.001/` — beta=0.001, d=32, 500 epochs
- `eval_results/beta_0.0001/` — beta=0.0001, d=32, 500 epochs
- `eval_results/beta_0.0001_dim16/` — **BEST**: beta=0.0001, d=16, 500 epochs
- `eval_results/beta_0.001_freebits/` — beta=0.001, d=32, free_bits=0.5, 500 epochs

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Last tested on: CPU, Linux
