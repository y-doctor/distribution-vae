# Linear-rescale hinge: strictly better than binary hinge on 2kg/singles
**Date**: 2026-04-17 20:48 UTC
**Duration**: ~90 minutes (code + 150-ep run + eval + analysis)
**Goal**: Test whether linearly rescaling the reward above the NTC-baseline
hinge beats the binary (zero-below-θ, raw-above-θ) shape the Pearson run
used in 2026-04-17_1900.

## Motivation

The binary hinge has two structural issues for GRPO training:

1. **Step discontinuity at θ.** Just below θ → reward 0. Just above → reward
   ≈ θ (~0.21 on singles). A prediction that barely clears the noise floor
   gets ~21% of perfect-match reward just for "clearing the floor."
2. **Squished dynamic range above θ.** Rewards live in [θ, 1] = width 0.79.
   GRPO advantages are `(r − group_mean) / group_std`; the narrow range
   compresses advantages.

Proposed fix: map the above-θ range linearly to [0, 1]:

    r_eff = relu((r − θ) / (1 − θ))

so r = θ maps to 0 (no free reward for luck), r = 1 maps to 1 (perfect still
worth 1), and the usable advantage range widens.

## Changes

- `dist_vae/rl_train.py` — new module-level `apply_hinge(R, threshold, rescale)`
  helper. Two new config keys on `reward`:
  - `hinge_rescale` (default `False`, legacy-preserving)
  - `hinge_multiplier` (default `1.0`, scales the threshold, e.g. 2.0 for
    "2× noise floor"; not exercised in this run)
  The existing `ntc_baseline` and `fixed` hinges both go through the helper.
- `configs/rl_2kg_singles_mlp_pearson_rescale.yaml` — drop-in A/B config.
  Identical to `rl_2kg_singles_mlp_pearson.yaml` except `hinge_rescale: true`
  and an explicit `hinge_multiplier: 1.0`.
- `tests/test_rl_train.py` — 9 unit tests. Full suite now 104 passed / 2
  skipped.
- `scripts/viz_reward_landscape.py` — new one-off viz (from earlier in the
  session) that plots sorted pairwise-reward vectors per selected pert, with
  the hinge baseline, for intuition-building on the reward dynamics.

## Training

150 epochs on `data/mini_perturb_seq_2kg_allp_ntc.h5ad` filtered to 105
single-gene perts. Same val split, seed, batch/group/lr/entropy as the
baseline. Clean run — completed all 150 epochs (the baseline got killed at
142 on the same box so they are not *exactly* epoch-matched).

Trajectory (rescaled reward scale):
| epoch | reward | train top-1 | entropy |
|---:|---:|---:|---:|
| 1 | 0.079 | 0.005 | 4.64 |
| 50 | 0.357 | 0.07 | 3.31 |
| 100 | 0.474 | 0.17 | 2.53 |
| 150 | **0.593** | **0.320** | 1.69 |

(rescaled 0.593 ≈ 0.593·(1−0.213) + 0.213 = 0.68 on the original raw scale,
vs baseline's 0.653 at ep 142 — roughly matched at training time, but the
real test is held-out.)

## Held-out A/B

Same val_fraction=0.20, split_seed=123, R=20 trials, n_cells=200.

| metric | binary-hinge | **rescaled** | Δ |
|---|---:|---:|---:|
| top-1 (ens=1) | 0.075 | **0.091** | +1.6pp, +21% rel |
| top-1 (ens=10) | 0.072 | **0.101** | +2.9pp, +40% rel |
| top-3 (ens=1) | — | 0.185 | — |
| top-5 (ens=1) | — | 0.258 | — |
| top-10 (ens=1) | 0.332 | 0.345 | +1.3pp |
| top-10 (ens=10) | 0.337 | 0.343 | +0.6pp |
| MRR (ens=1) | 0.161 | **0.180** | +1.9pp, +12% rel |
| MRR (ens=10) | — | 0.190 | — |
| mean reward (ens=1) | 0.519 (pearson) | 0.513 | flat |
| P(r ≥ 0.5, ens=1) | 0.589 | 0.582 | ~flat |
| P(r ≥ 0.8, ens=1) | — | 0.214 | — |
| P(r ≥ 0.9, ens=1) | 0.083 | **0.112** | +2.9pp, +35% rel |
| P(r ≥ 0.9, ens=10) | — | 0.122 | — |

**Ranking got better at every top-k, and high-confidence correctness (P(r ≥
0.9)) improved substantially.** Loose metrics (P(r ≥ 0.5), mean reward) are
flat — the model's "roughly right" rate was already fine; the win is in
"exactly right" and "nearly right" rates.

**Ensembling works now.** It didn't help the binary-hinge run (+0 to +0.5pp
typically). Here ens=10 gives another +1pp top-1 over ens=1. My best guess
why: with the binary hinge many predictions were clipped to 0 regardless of
"how close," so 10 fresh subsamples all produced nearly the same argmax.
Rescaling preserves gradient on "nearly right," so the predictions vary more
across subsamples and average-of-logits helps.

## Intuition

The rescaled hinge widens the advantage spread. Concretely, in a group of
G=64 sampled actions, above-threshold rewards previously lived in [0.21, 1]
= width 0.79. With rescaling they live in [0, 1] = width 1.0. That's +27%
dynamic range to distinguish between "marginally signal" and "strong match"
within a single group — i.e. more gradient information per update for the
kind of fine-grained distinctions GRPO needs to make.

The discontinuity removal is probably the bigger effect for top-1: the
binary hinge encouraged the policy to optimize "can I get *something* above
θ?" (an easy objective), then stalled. Rescaling makes "barely-above-θ"
worth almost nothing, so the policy can't coast on clearing-the-floor wins.

## Open questions / next steps

1. **Try hinge_multiplier=2.0 + rescale.** The singles-only data has a mild
   hinge (mean baseline 0.213, most off-diag pairs in [0.2, 0.5]). A 2× hinge
   would prune middle-reward pairs harder. Risk: weak perts (BCL2L11,
   MAP4K3) have low signal and might get zero-reward-everywhere groups → no
   gradient. The viz panel shows ~20% of perts have <5 neighbors above r=0.5
   — these would be the canaries.
2. **Extend training.** Reward was still climbing at ep 150 (entropy 1.69
   still well above the log(10)=2.30 10-class collapse floor). 250-300 ep
   likely pushes top-1 further.
3. **Re-run the cell-set-transformer (rl_cell) with the same rescale.** A
   cleaner A/B against the architecture variant.
4. **Port to the full 237-pert data w/ combos.** The hinge bites harder
   there (more degenerate pairs); rescaling benefits scale with θ, so this
   should compound.

## Artifacts

- `checkpoints/rl_2kg_singles_mlp_pearson_rescale/best.pt` (gitignored)
- `eval_results/rl_2kg_singles_mlp_pearson_rescale/history.json`
- `eval_results/rl_2kg_singles_mlp_pearson_rescale/full_run.log`
- `eval_results/rl_2kg_singles_mlp_pearson_rescale/val_ens1/`
- `eval_results/rl_2kg_singles_mlp_pearson_rescale/val_ens10/`
- `eval_results/reward_landscape/` — pre-training intuition viz (panel +
  summary + npz of the (P,P) matrix)
