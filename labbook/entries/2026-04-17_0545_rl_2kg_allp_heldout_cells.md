# Full-scale run: 2000 HVGs x 236 perturbations, held-out cell split, test-time ensembling
**Date**: 2026-04-17 05:45 UTC
**Duration**: ~80 minutes (60 min training + 20 min eval)
**Goal**: Push the GRPO classifier to the full Norman dataset scale (2k HVGs, 236 perts) and implement two levers the previous evals were missing: (1) held-out **cell** split, (2) **test-time ensembling** over multiple subsamples.

## What I did

1. Regenerated `data/mini_perturb_seq_2kg_allp_ntc.h5ad` via `make_mini_dataset.py --n-genes 2000 --n-perts 300 --keep-controls`. Result: 111,445 cells x 2000 genes x 237 perts (236 non-control + control).

2. Added **held-out cell split** to `dist_vae/rl_data.py`:
   - `val_fraction`, `split_seed`, `mode` params on `PerturbationClassificationDataset`.
   - Splits each pert's cells (and NTC cells) into train/val pools; `__getitem__` samples from the chosen pool.
   - The oracle `delta_mean_profiles` still uses all cells (the target profile is pert-property, not data-dependent).
   - Back-compatible: `val_fraction=0.0` preserves prior behavior.

3. Added **test-time ensembling** to `scripts/eval_rl_perturbation.py`:
   - New `--n-ensemble N` flag. Each "trial" averages model logits over N fresh subsamples before argmax.
   - New `--val-fraction` / `--mode` flags so eval can run on the held-out cell split.

4. Wrote `configs/rl_perturbation_2kg_allp.yaml` (MLP, 20% holdout, 150 epochs, samples_per_epoch=1200, group_size=64).

5. Smoke-tested runtime (2 epochs at 160 samples = 14s → ~26s/epoch at 1200 samples), then ran full 150 epochs (~65 min).

6. Ran held-out eval three ways: train-cell eval (ens=1), val-cell eval (ens=1, 5, 10).

## Results

**Training** (150 epochs, MLP, 272K params):
- Final train-time mean reward: **0.852** (random baseline 0.23)
- Final train-time top-1: 0.422 (random 0.004)
- Entropy: 5.45 → 0.08 (smooth decay from log(236))

**Held-out evaluation** (R=20 trials per pert):

| eval                  | top-1 | top-10 | MRR  | mean reward | P(reward ≥ 0.9) |
|-----------------------|------:|-------:|-----:|------------:|----------------:|
| train cells, ens=1    | 0.369 | 0.453  | 0.408 | **0.812** | **0.531** |
| val cells, ens=1      | 0.198 | 0.379  | 0.271 | 0.718 | 0.329 |
| val cells, ens=5      | 0.208 | 0.383  | 0.279 | 0.727 | 0.344 |
| val cells, ens=10     | 0.213 | 0.385  | 0.283 | 0.729 | 0.351 |

Random baselines (for 236 classes): top-1 = 0.004, mean cos-sim = 0.228.

## Interpretation

### Generalization gap
The train-cell / val-cell gap on **mean reward** is 0.094 (0.812 → 0.718). This is a meaningful but not huge gap — the model does generalize across held-out cells, just less confidently. Absolute bio-equivalence (P ≥ 0.9) drops from 53% to 33%.

### Test-time ensembling helps modestly, as expected
Going from N=1 to N=10 subsamples per trial:
- Mean reward: 0.718 → 0.729 (+0.011)
- Top-1: 0.198 → 0.213 (+0.015)
- P(reward ≥ 0.9): 0.329 → 0.351 (+0.022)

Small but real. Diminishing returns after ~5 samples. This suggests sampling noise explains **only a fraction** of the train/val gap — most of the gap is genuine generalization.

### Scale sanity check
| scale | random rwd | train-time rwd | held-out rwd (ens=1) | lift over random |
|-------|-----------:|---------------:|---------------------:|-----------------:|
| 50 perts, 500 HVGs | 0.170 | 0.725 | 0.716 (subsample) | +0.55 |
| 236 perts, 2k HVGs | 0.228 | 0.852 | 0.718 (val cell) | +0.49 |

The model is learning comparable amounts of signal at both scales. At 236 perts on held-out cells it's still holding a +0.49 lift over random.

### What the gap says about the reward
The ~0.10 train/val gap in mean reward roughly corresponds to the n^-0.5 sampling noise at n=100 cells (jitter W1 ≈ 0.036, signal W1 ≈ 0.23 → SNR ≈ 6). So the cell-split methodology is revealing real robustness-to-resampling limits that the old same-cell eval was hiding.

## Files

- `data/mini_perturb_seq_2kg_allp_ntc.h5ad` — full 111K-cell dataset, 2k HVGs, 236 perts + control (gitignored due to size).
- `dist_vae/rl_data.py` — added val_fraction / split_seed / mode.
- `scripts/eval_rl_perturbation.py` — added --n-ensemble / --val-fraction / --mode / --split-seed CLI flags; small UMAP title fix.
- `scripts/train_rl.py` — reads val_fraction and split_seed from config.data.
- `configs/rl_perturbation_2kg_allp.yaml` — new config.
- `eval_results/rl_perturbation_2kg_allp/{train_ens1,val_ens1,val_ens5,val_ens10}/` — one eval per run.
- `eval_results/rl_perturbation_2kg_allp/ensemble_comparison.png` — summary figure.

## Next steps

- Try **train-time multi-subsample input** (lever #2 from earlier plan). The test-time ensemble only buys ~0.01 reward; the bigger potential win is making the model inherently noise-robust.
- Increase `n_cells_per_pert` from 100 to 300 or 500 → the jitter analysis says this should close most of the generalization gap.
- Add regularization (weight_decay, dropout) and re-run the attention variant on this scale to see if it pays off here.

## Branch

`claude/rl-perturbation-classifier-BNzAX`.
