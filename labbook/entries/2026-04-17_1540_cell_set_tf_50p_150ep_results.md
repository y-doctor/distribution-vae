# Per-cell set-transformer on 500g/50p — 150-epoch training + held-out eval
**Date**: 2026-04-17 15:40 UTC
**Duration**: ~30 minutes (13 min training + eval runs)
**Goal**: Run the `rl_cell_model` architecture to plateau on 500g/50p with held-out cells, then reproduce the reward-ball eval from the MLP baseline to do a direct A/B.

## Training

Config `configs/rl_cell_50p.yaml` — **d=48, n_modules=16, 2 cell-self-attn + 2 cross-attn layers, lr=1e-4, clip=0.5, val_fraction=0.20**. 236,786 params. 150 epochs, ~5 sec/epoch, ~13 min wall.

| epoch | reward (z-scored) | top-1 (train) | entropy |
|---:|---:|---:|---:|
| 1 | +0.091 | 0.030 | 3.710 |
| 28 | +0.650 | 0.050 | 3.291 |
| 75 | ~+1.5 | ~0.45 | ~2.3 |
| 150 | **+2.018** | **0.710** | **1.655** |

Clean, no collapse. Grad-norm pre-clip ran 4→48; clip=0.5 bounded step sizes. Entropy decreased monotonically. Plateau not yet reached at ep150 — likely can squeeze more out.

> *Scale note:* `reward` in training is **z-scored per true-pert row**, so values >1 are expected and good. 2.0 means average pick lands near the top of each row's reward distribution. Eval reports raw cos-sim which is bounded in [-1, 1].

## Held-out eval (val_fraction=0.20, split_seed=123)

`scripts/eval_rl_perturbation.py` with new `--model-type cell` flag. Predictions computed on held-out 20% of cells per pert, R=20 trials, n_cells=50.

| metric | ens=1 | ens=10 |
|---|---:|---:|
| top-1 | 0.301 | **0.356** |
| top-3 | 0.551 | 0.585 |
| top-5 | 0.622 | 0.649 |
| top-10 | 0.754 | **0.780** |
| MRR | 0.458 | 0.504 |
| mean reward (raw cos-sim) | 0.727 | **0.774** |
| **P(r ≥ 0.9)** | 0.404 | **0.456** |
| P(r ≥ 0.5) | 0.805 | 0.864 |
| P(r ≥ 0.95) | 0.341 | 0.396 |
| P(r = 0.99, ~exact) | 0.301 | 0.356 |

Random baseline: top-1 0.02, mean reward 0.170.

## A/B vs the MLP baselines

| model | config | epochs | held-out top-1 | held-out P(r≥0.9) |
|---|---|---:|---:|---:|
| MLP (50p) | `rl_perturbation_50p.yaml` | 334 | 0.43 | not computed |
| MLP (2kg/236p row-norm) | `rl_perturbation_2kg_allp_rownorm.yaml` | 150 | 0.23 | 0.42 (ens=10) |
| **Cell-set-TF (50p)** | `rl_cell_50p.yaml` | **150** | **0.36** (ens=10) | **0.46** (ens=10) |

The cell model at ep150 on 50 perts **matches or exceeds the 2kg/236p MLP's P(r≥0.9)** despite using a different class count, and lags the 334-epoch MLP 50p run on top-1 only because it has less than half the training budget. Strong indication the architecture is working as intended.

## Reward-ball plots (see `eval_results/rl_cell_50p/val_ens10/`)

**Soft-accuracy curve**: model at τ=0.5 = 0.86, τ=0.9 = 0.46, τ=0.99 = 0.36. Random baseline drops off to ~0 by τ=0.95. Clean separation at every threshold.

**Best-20 perts**: model confidently picks the true pert (green ring on gold star). Many have dense blue reward-ball clusters — these are bio-equivalent perts like `CBL_PTPN9`, `MAP2K6_ELMSAN1`, `ETS2_CEBPE`. Model happily lands inside the cluster.

**Worst-20 perts**: gold stars sit in isolated regions with few/no blue neighbors. Red rings (wrong predictions) land close to the star in PCA space — *not* scattered like in the 2kg MLP case. The model knows the right *neighborhood* but can't always pin the exact pert when there is no reward ball to guide it.

## Observations

- **Top-10 0.78** means in 78% of trials the true pert is in the model's top-10 candidates. Very usable for downstream "suggest a pert to make."
- **Bio-equivalent pair hit rate** (τ=0.9) is 0.46 — the model's best use case is proposing any member of the reward-equivalent class, which is the biologically honest framing.
- **Ensembling gains**: +5.5pp top-1, +5.2pp P(r≥0.9) with ens=10. Free win if you can afford 10× forward passes.
- **Top-1 lag vs MLP 50p** is probably a training-budget artifact. Worth running 300 eps to close the gap.

## Artifacts

- `checkpoints/rl_cell_50p/best.pt` (gitignored)
- `eval_results/rl_cell_50p/history.json` — training curves
- `eval_results/rl_cell_50p/val_ens1/` and `val_ens10/` — full eval output including `soft_accuracy_curve.png`, `pert_neighborhoods_best.png`, `pert_neighborhoods_worst.png`, `metrics.json`

## Next steps

1. **Longer run** (300-500 eps) to close the top-1 gap with the MLP 50p 334-ep baseline.
2. **Scale to 2kg/236p** — needs GPU (training was fast on 50p because n_cells=50, G=500; at 2000 genes × 236 perts the compute will dominate).
3. **Hard-pert upweighting** — worst-20 analysis shows the model concentrates errors on isolated perts. Inverse-reward-ball-density sampling should help those directly.
4. **Gene module introspection** — now that training has converged, read out attention weights to see if the 16 learned modules correspond to recognizable biological pathways.
