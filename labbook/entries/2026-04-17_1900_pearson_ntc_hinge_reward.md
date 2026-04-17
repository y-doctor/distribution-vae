# Reward-function upgrade: Pearson + NTC-noise-baseline hinge
**Date**: 2026-04-17 19:00 UTC
**Duration**: ~90 minutes (code + two 150-ep runs + eval)
**Goal**: Fix the generalization gap on 2kg/105-singles — the cos-sim + row-norm MLP saw train top-1 0.21 → held-out 0.04 — by replacing the reward function with something more principled.

## Motivation

Critical review of the cos-sim reward revealed several issues (see today's earlier chat log):
1. Cos-sim is inflated by the "baseline response direction" many perts share (generic stress / cell-cycle genes moving together).
2. The 0.9 / 0.5 thresholds we were using to define a "reward ball" were arbitrary — not tied to a noise floor.
3. With 2000 HVGs, most of the cos-sim signal comes from low-variance noise genes.
4. Row-normalization after the fact flattened out any notion of "in vs out of the ball."

The singles-only dataset also removed the paired-pert degeneracy we had built the ball metric around, so cos-sim was scoring mostly-unrelated perts as moderately similar.

## Changes

1. **Reward metric: Pearson correlation** (`reward.metric: "pearson"`).
   Mean-centers each profile before computing cosine. Removes the shared up-regulation direction between any two perts. New helper `dist_vae.losses.pearson_correlation`.

2. **NTC-noise-baseline hinge** (`reward.hinge: "ntc_baseline"`).
   Per true pert i:
     - Sample K=200 independent subsamples of n_cells NTC cells.
     - Compute each as a pseudo-profile: `mean(ntc_subsample_k) - mean(all_ntc)`.
     - Correlate each with `profile[i]` → K-sized null distribution.
     - Take the 95th percentile → `baseline[i]`.
   Any reward ≤ `baseline[i]` gets hinged to 0.
   The baseline is per-pert because different profiles have different susceptibilities to noise correlation. Implemented as `PerturbationClassificationDataset.compute_ntc_noise_baseline`.

3. **row_normalize disabled** when using the NTC-baseline hinge — the hinge already provides cross-pert balance by zeroing out anything below each row's noise floor.

4. **n_cells bumped 100 → 200** per stream, to reduce trial-time noise that was previously flooring generalization.

## Files

- `dist_vae/losses.py` — `pearson_correlation(x, y, dim, eps)`.
- `dist_vae/rl_data.py` — `compute_ntc_noise_baseline(profiles, n_cells, metric, K, quantile, seed)`.
- `dist_vae/rl_train.py` — `GRPOTrainer.__init__` now reads `reward.metric`, `reward.hinge`, `reward.hinge_*` and applies them before optional row-normalization.
- `scripts/eval_rl_perturbation.py` — reads metric from checkpoint config, scores per-trial rewards / soft-accuracy / ball plots / metrics.json with the matching metric. `plot_confusion_vs_reward` now takes optional `hinge_baseline` to display the effective (post-hinge) reward-similarity matrix; the x-axis label reflects the actual metric.
- `configs/rl_2kg_singles_mlp_pearson.yaml` — new config using Pearson + NTC hinge, no row-normalize, n_cells=200.
- Tests: 4 new for Pearson, 3 new for NTC-baseline. Full suite 95 passed / 2 skipped.

## Training

Ran 150 epochs on `data/mini_perturb_seq_2kg_allp_ntc.h5ad` filtered to 105 single-gene perts (no `_` in name), val_fraction=0.20. Full run log saved as `eval_results/rl_2kg_singles_mlp_pearson/full_run.log`.

Process exited after epoch 142 (not gracefully — either external kill or OOM), but `best.pt` was already saved at the high-water reward 0.6526 on epoch 142, so nothing was lost. `history.json` reconstructed from the log post-hoc.

Trajectory:
| epoch | reward (raw Pearson-hinge) | train top-1 | entropy | grad-norm |
|---:|---:|---:|---:|---:|
| 1 | 0.231 | 0.005 | 4.63 | 0.5 |
| 50 | 0.55 | 0.14 | 3.38 | 3.6 |
| 100 | 0.62 | 0.24 | 2.25 | 4.5 |
| 142 | **0.653** | **0.30** | 1.68 | 4.8 |

Clean convergence — reward monotonic, entropy dropped 4.63 → 1.68 (log(105) = 4.65 uniform baseline), grad-norm small and stable throughout. **Train top-1 0.30** vs the cos-sim + row-norm baseline's 0.21 — a real improvement at training time.

## A/B held-out eval

Same val split (val_fraction=0.20, split_seed=123), same R=20 trials, n_cells=200.

| metric | cos-sim + row-norm (baseline) | **pearson + NTC-hinge** | Δ |
|---|---:|---:|---:|
| top-1 (ens=1) | 0.043 | **0.075** | +3.2pp, +74% rel |
| top-1 (ens=10) | 0.049 | 0.072 | +2.3pp |
| top-10 (ens=1) | 0.252 | **0.332** | +8.0pp |
| top-10 (ens=10) | 0.270 | 0.337 | +6.7pp |
| MRR (ens=1) | 0.119 | 0.161 | +4.2pp |
| mean reward | 0.474 (cos) | 0.519 (pearson) | different scales |
| P(r ≥ 0.5, ens=1) | 0.498 | **0.589** | +9.1pp |
| P(r ≥ 0.9, ens=1) | 0.063 | 0.083 | +2.0pp |

Genuine improvement everywhere. Biggest relative gains are at the looser thresholds (top-10, P(r ≥ 0.5)): the model's ranking of candidate perts is noticeably better, even when it doesn't land exactly.

**Ensembling barely helps** (+0 to +0.5pp typically), same pattern as the cos-sim run. Model is already stable across subsamples — it's not sampling-noise-limited.

## Clustermap interpretation

`confusion_vs_reward_sim.png` now shows Pearson on the left (previously mislabeled as cos-sim) with the NTC-baseline hinge applied. The hinge-baseline distribution across 105 perts is:
- range: 0.088 – 0.287
- mean: 0.213

Most off-diagonal Pearson values in the 2k-singles matrix are in the 0.2–0.5 range, so the hinge is **milder than expected on this data**. It zeros out anti-correlated and weakly-correlated pairs but leaves the broadly-related structure intact. The hinge would bite harder on a dataset with more isolated perts.

## Best-20 / worst-20 ball analysis (Pearson space)

- **Best-20**: top 5 (CEBPB, IKZF3, ATL1, PRTG, CEBPE) near-perfect — μr = 1.00, P(r≥0.9) = 1.00. Rows 3–4 have μr 0.82–0.94 but P(r≥0.9) = 0 — model gets *close* on average but doesn't land inside the tight 0.9 ball. Consistent with "ranks are good, exact identification is hard."
- **Worst-20**: 20 perts with μr in [-0.35, 0.29]. ZC3HAV1 has strongly anti-correlated predictions (μr = -0.35). Red rings scatter — these perts produce transcriptomic responses too weak or too similar to others to disambiguate given n_cells=200.

## Observations / open questions

1. **Train–val gap is still large**: 0.30 → 0.075. Fewer absolute errors at every threshold, but same structural issue — model memorizes training cells somewhat.
2. **Ensembling isn't helping**. Worth investigating why — possibly because the per-gene MLP is deterministic-enough that 10 fresh subsamples land at the same argmax.
3. **Row_normalize with hinge together?** Didn't test. Might compress the useful dynamic range above the hinge. Left as a future ablation.
4. **Noise floor is pert-specific**: `baseline[i]` varies from 0.088 to 0.287. A pert with baseline 0.09 is very easy to clear; one with baseline 0.29 is hard. Might be worth ranking perts by baseline and looking for patterns — maybe perts with large profiles are "easier" in this sense.

## Next steps

1. **Extend training** — reward was still climbing at ep142. 250-300 ep might push top-1 past 0.10 on held-out.
2. **Try per-trial (trial-local) reward** — was option A in the critical review. Compute the "true profile" per-trial from the 200 cells actually sampled, not from the oracle.
3. **Cell-set-transformer on the same data** — direct A/B vs the MLP using the same Pearson+hinge reward.
4. **Compare to singles on the larger 236-pert matrix with combos re-enabled** — the hinge should be much stricter there (more degenerate pairs), and we'd recover the "reward ball" semantics.

## Artifacts

- `checkpoints/rl_2kg_singles_mlp_pearson/best.pt` (gitignored)
- `eval_results/rl_2kg_singles_mlp_pearson/history.json`
- `eval_results/rl_2kg_singles_mlp_pearson/full_run.log`
- `eval_results/rl_2kg_singles_mlp_pearson/val_ens1/` + `val_ens10/`
- `eval_results/rl_2kg_singles_mlp/` (cos-sim + row-norm A/B partner, 150 ep)
