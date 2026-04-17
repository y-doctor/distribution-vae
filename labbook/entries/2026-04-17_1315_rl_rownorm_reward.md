# Row-normalized reward: z-score per true pert before GRPO
**Date**: 2026-04-17 13:15 UTC
**Duration**: ~1.5h (code + 60 min training + eval)
**Goal**: Test whether pre-normalizing the (P, P) reward table per row (z-score each row across predictions) gives a more balanced gradient signal across perts, on top of GRPO's existing group normalization.

## Motivation (user's observation)

With `group_size=64 < P=236`, GRPO's group-normalized advantage is an approximation of the row-wide normalization. Perts with unique profiles (no near-duplicates, small row mean, small row std) produce outsized gradient magnitudes; perts with many near-duplicates give small advantages regardless. Pre-row-normalizing the reward table equalizes dynamic range across true perts.

## Implementation

- `dist_vae/rl_train.py`: precompute `reward_table = cos-sim(profiles, profiles)` once at trainer init. If `reward.row_normalize=True`, z-score each row: `R = (R - row_mean) / row_std`. `_compute_reward` now indexes into this table instead of computing cos-sim per batch.
- New config `configs/rl_perturbation_2kg_allp_rownorm.yaml` — identical to `rl_perturbation_2kg_allp.yaml` except `reward.row_normalize: true`.
- All 61 tests still pass.

## Results (same seed, same schedule, 150 epochs each)

### Training
| metric | raw reward | row-norm | Δ |
|---|---:|---:|---:|
| final train top-1 | 0.422 | **0.507** | **+8.5pp** |
| final entropy | 0.081 | 0.111 | — |

Training curves track each other early (ep 30-60), then row-norm pulls ahead from ~ep 80 onwards.

### Held-out cell eval (R=20 trials, 20% cell holdout)

Without test-time ensembling (ens=1):
| metric | raw | row-norm | Δ |
|---|---:|---:|---:|
| top-1 | 0.198 | 0.205 | +0.7pp |
| top-3 | 0.291 | **0.335** | **+4.4pp** |
| top-10 | 0.379 | **0.450** | **+7.1pp** |
| MRR | 0.271 | **0.297** | +2.6pp |
| mean reward (raw cos-sim) | 0.718 | 0.720 | ~tie |
| P(reward ≥ 0.9) | 0.329 | **0.383** | **+5.4pp** |

With 10x test-time ensembling:
| metric | raw | row-norm | Δ |
|---|---:|---:|---:|
| top-1 | 0.213 | **0.231** | +1.8pp |
| top-3 | 0.303 | **0.351** | **+4.8pp** |
| top-10 | 0.385 | **0.456** | **+7.1pp** |
| MRR | 0.283 | **0.316** | +3.3pp |
| mean reward | 0.729 | **0.748** | **+1.9pp** |
| P(reward ≥ 0.9) | 0.351 | **0.416** | **+6.5pp** |

## Interpretation

**The model ranks the right pert higher across the board.** Top-10 jumps from 0.38 to 0.45 (ens=1) and to 0.46 (ens=10). P(reward ≥ 0.9) — the bio-equivalent hit rate — goes from 0.33 to 0.38 (ens=1) and to 0.42 (ens=10).

Mean reward on the RAW cos-sim scale is roughly tied, which is consistent with the theoretical argument: row-normalizing doesn't change where the model lands *on the reward manifold*, it changes how evenly the model learns across the 236 true-pert targets. The model is now investing more learning capacity into distinguishing perts that were previously deprioritized (perts with many near-duplicates had small gradients and were underserved).

**The top-10 gain is the cleanest signal** — ranks improve even when the exact top-1 is still hard.

## Does this break the "bio-equivalence is fine" principle?

No. Two perts with cos-sim > 0.9 still give each other high raw cos-sim rewards (row-norm preserves row ORDER). We only rescaled the dynamic range. The model picking a near-duplicate still gets a positive normalized reward; it's just that normally-orthogonal perts now give sharper negative signal.

## Files

- `dist_vae/rl_train.py` — modified.
- `configs/rl_perturbation_2kg_allp_rownorm.yaml` — new.
- `eval_results/rl_perturbation_2kg_allp_rownorm/` — training log, history, checkpoint, val_ens1/ and val_ens10/ evals, plots.

## Recommended default

Flip `reward.row_normalize` ON for all future runs. It's a free win — no compute overhead, no architectural change, gets uniformly better held-out ranking.

## Branch

`claude/rl-perturbation-classifier-BNzAX`
