# Pulse: per-cell set-transformer classifier build
**Date**: 2026-04-17 15:15 UTC
**Task**: Prototype the per-cell set-transformer classifier (labbook/DECISIONS.md 2026-04-17)
**Status**: on track

## Progress this segment
- Built `dist_vae/rl_cell_model.py` in 4 components: `GeneModuleAttention`, `CellSetTransformer`, `PertNTCCrossAttention`, `PerturbationCellClassifier`. 19 unit tests, all pass.
- Extended `PerturbationClassificationDataset` with `return_cells=True` (raw cells in place of quantile tokens). 3 new tests, all pass.
- Added `scripts/train_rl_cell.py` CLI that reuses the existing `GRPOTrainer` unchanged (same `(ntc, pert, gene_ids) → logits` signature).
- Phase-3 10-pert smoke run: 59,594 params, 10 epochs on `mini_perturb_seq_with_ntc.h5ad`. Reward climbed 0.046 → 0.29, top-1 0.08 → 0.20 (random 0.10). Model is learning.
- Phase-4 50-pert run started (30 epochs, 500g/50p, d=48, n_modules=16, 2+2 attention layers). Running in background.

## Current blockers
- None.

## Next 30 min plan
- Monitor the 50p run; compare final reward to the MLP baseline (50 epochs: reward 0.57, top-1 0.14).
- Write a final labbook entry with results and architecture diagram.
- Commit the whole thing: model, dataset edit, configs, script, tests. Update TASKS.md.
