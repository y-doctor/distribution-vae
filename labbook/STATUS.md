# Project Status

**Last updated**: 2026-03-18 00:30 UTC
**Updated by**: Session close-out — hyperopt module + report

## What works
- `dist_vae/losses.py` — All 3 loss functions + CombinedDistributionLoss (17 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities (14 tests pass)
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE (14 tests pass, 1 CUDA skipped)
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing, epoch_callback
- `dist_vae/eval.py` — All evaluation functions and plotting
- `dist_vae/hyperopt.py` — Optuna-based hyperparameter optimization with pruning (14 tests pass)
- `scripts/` — All 5 CLI scripts implemented (train, evaluate, encode, download, hyperopt)
- End-to-end synthetic training verified (loss decreases correctly)
- Package installable via `pip install -e ".[dev,hyperopt]"`
- All 60 tests pass on CPU

## What's broken / blocked
- Nothing currently broken
- CUDA test skipped (no GPU in test environment)
- Real data (Norman et al.) not yet tested (requires download)

## What's in progress
- Nothing — session ending

## Next priorities
1. Run hyperopt on synthetic data end-to-end
2. Download and test with real Perturb-seq data
3. Tune hyperparameters on real data using hyperopt module
4. Add integration tests for training loop
5. Profile and optimize for large datasets

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Optuna version: 4.8.0
- Last tested on: CPU, Linux
