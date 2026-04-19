# Long-training plateau run on the rescale-hinge config
**Date**: 2026-04-19 21:42 UTC
**Duration**: ~45 minutes (405-ep run + eval + analysis)
**Goal**: Answer "how much does longer training help?" for the Pearson +
NTC-hinge + linear-rescale configuration. Prior run stopped at a fixed 150
epochs with the reward still clearly climbing.

## Setup

Same config as `rl_2kg_singles_mlp_pearson_rescale.yaml` but with 600-epoch
budget and plateau early-stopping (window=30, patience=50, min_delta=0.002).
See `configs/rl_2kg_singles_mlp_pearson_rescale_long.yaml`.

## Training

Plateau-stopped at **ep 405/600**. Trajectory of train reward (rescaled
scale):

| epoch | reward | train top-1 | entropy |
|---:|---:|---:|---:|
| 89 | 0.449 | 0.105 | 2.80 |
| 150 | 0.593 | 0.32 | 1.69 |
| 214 | 0.726 | 0.51 | 0.61 |
| 395 | 0.791 | 0.555 | 0.03 |
| 405 (stop) | **0.7919** | **0.555** | 0.03 |

Policy entropy dropped to ~0.03 nats (log(105) = 4.65 = uniform baseline),
so the policy is effectively deterministic at the plateau. `pg_loss`
shrank toward zero, grad-norm stable around 2–3. Plateau detector worked
as intended: smoothed reward hadn't improved by 0.002 in 50 epochs.

## Three-way held-out A/B

Same val split (val_fraction=0.20, split_seed=123), R=20 trials, n_cells=200.

| metric | binary-hinge (142 ep) | rescale-short (150 ep) | **rescale-long (405 ep)** |
|---|---:|---:|---:|
| train top-1 | 0.30 | 0.32 | **0.555** |
| held-out top-1 (ens=1) | 0.075 | 0.091 | **0.105** |
| held-out top-1 (ens=10) | 0.072 | 0.101 | **0.111** |
| held-out top-3 (ens=10) | — | 0.193 | 0.238 |
| held-out top-5 (ens=10) | — | 0.272 | 0.313 |
| held-out top-10 (ens=1) | 0.332 | 0.345 | **0.397** |
| held-out top-10 (ens=10) | 0.337 | 0.343 | **0.424** |
| MRR (ens=1) | 0.161 | 0.180 | **0.204** |
| MRR (ens=10) | — | 0.190 | **0.216** |
| mean reward (ens=1) | 0.519 | 0.513 | 0.470 ↓ |
| P(r ≥ 0.5, ens=1) | 0.589 | 0.582 | 0.516 ↓ |
| P(r ≥ 0.8, ens=1) | — | 0.214 | 0.188 |
| P(r ≥ 0.9, ens=1) | 0.083 | 0.112 | 0.112 |
| P(r ≥ 0.9, ens=10) | — | 0.122 | 0.119 |

## Observations

1. **Training longer helps ranking, not exact ID.** top-10 ens=10 went
   +8pp (0.343 → 0.424, +24% rel); MRR +0.026 (+14% rel); but top-1 ens=10
   gained only +1pp (0.101 → 0.111). The model got much better at
   "right answer is somewhere in my top 10" without getting much better at
   "right answer is my first pick."

2. **Classic overfit pattern on train.** Train top-1 went from 0.32 → 0.555
   (+73%), held-out only +15%. Entropy collapse from 1.69 → 0.03 means the
   policy is fully deterministic — zero hedging. The reward table rewards
   that near the diagonal but not enough in the "mostly right" band to make
   up for the dropped entropy bonus.

3. **P(r ≥ 0.9) plateaued at 0.112.** Tight-reward-ball accuracy did
   **not** improve with 3× the training. This is the biology ceiling: 0.112
   is the model's asymptotic rate of landing on a pert that is truly
   bio-equivalent (r ≥ 0.9) to the ground truth, given the data.

4. **Mean reward and P(r ≥ 0.5) DROPPED.** Going deterministic cost us
   ~5pp on "broadly-right" metrics (0.513 → 0.470 mean, 0.582 → 0.516
   P(r ≥ 0.5)) while the ranking-metric gains were ~3–8pp. The 0-entropy
   policy commits to one pick per input; when that pick is wrong, there's
   no fallback probability mass on the nearby bio-equivalents that would
   have given ~0.7 reward. Effectively the policy traded "usually close"
   for "occasionally exact."

5. **Plateau-stop worked as intended** — detector window=30, patience=50,
   min_delta=0.002 triggered at ep 405 after the smoothed reward flat-lined
   around 0.79 for ~35 epochs.

## Implications

"Just train longer" is a modest lever on this problem; the remaining gap
is structural, not budget-driven:

- The train/held-out ratio (0.555 / 0.105 ≈ 5.3×) is huge — regularization
  is probably higher-ROI than more training.
- The entropy collapse suggests `entropy_coef` should stay higher for
  longer, or we should add a floor on entropy. Currently the
  entropy_coef=0.3 coefficient becomes irrelevant once entropy is near 0
  because the entropy bonus scales with entropy.
- P(r ≥ 0.9) = 0.112 is a biology ceiling. The only ways past it are:
  (a) a stronger architecture that resolves reward-degenerate pairs using
  information the MLP discards (e.g. cell-level heterogeneity via the
  set-transformer), or (b) more training data.

## Files / artifacts

- `configs/rl_2kg_singles_mlp_pearson_rescale_long.yaml`
- `dist_vae/rl_train.py` — new `PlateauDetector` class, three `training.plateau_*` config keys (default-off)
- `tests/test_rl_train.py` — +5 tests for PlateauDetector (14 total passing)
- `checkpoints/rl_2kg_singles_mlp_pearson_rescale_long/best.pt` (gitignored)
- `eval_results/rl_2kg_singles_mlp_pearson_rescale_long/` — history.json,
  full_run.log, val_ens{1,10}/ with full plot set + metrics.json

## Next steps

1. **Raise entropy floor.** Either clip log-probs, add a KL-to-uniform
   floor, or use a scheduled entropy coefficient that doesn't let the
   policy collapse below ~0.5 nats. Primary remedy for the overfit.
2. **`hinge_multiplier=2.0 + rescale`** — the stricter-threshold A/B. Could
   help by pruning the "close but not quite" predictions that are hurting
   mean reward at entropy=0.
3. **Port rescale hinge to the `rl_cell` set-transformer path** on 2kg
   singles. Given the biology ceiling on P(r ≥ 0.9), architecture changes
   are the highest-leverage move.
4. Re-run the binary-hinge baseline with the same plateau-stop to verify
   the reward-shape win persists at training convergence (currently the
   150-vs-405-ep comparison is the only one at "actual plateau").
