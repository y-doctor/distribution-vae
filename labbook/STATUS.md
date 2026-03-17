# Project Status

**Last updated**: 2026-03-17 16:00 UTC
**Updated by**: Full implementation session

## What works
- `dist_vae/losses.py` — All 3 loss functions + CombinedDistributionLoss (17 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities (14 tests pass)
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE (14 tests pass, 1 CUDA skipped)
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing
- `dist_vae/eval.py` — All evaluation functions and plotting
- `scripts/` — All 4 CLI scripts implemented
- End-to-end synthetic training verified (loss decreases correctly)
- Package installable via `pip install -e ".[dev]"`
- All 45 tests pass on CPU

## What's broken / blocked
- Nothing currently broken
- CUDA test skipped (no GPU in test environment)
- Real data (Norman et al.) not yet tested (requires download)

## What's in progress
- Nothing

## Next priorities
1. Download and test with real Perturb-seq data
2. Tune hyperparameters on real data
3. Add more evaluation tests
4. Profile and optimize for large datasets

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Last tested on: CPU, Linux
