# Cross-gene attention + richer held-out evaluation
**Date**: 2026-04-16 23:30 UTC
**Duration**: ~45 minutes
**Goal**: Add a transformer encoder stack on top of the per-gene features in the RL classifier and compare to the MLP baseline at matched training budget. Also enrich the held-out eval with UMAP, per-pert reward distributions, and reward-threshold metrics.

## What I did

1. Extended `dist_vae/rl_model.py`: added `n_attn_layers` / `n_heads` config knobs. When `n_attn_layers > 0`, a `nn.TransformerEncoder` is inserted after the per-gene MLP and before the mean+max pool; otherwise it's `nn.Identity()` (back-compat).
2. New config `configs/rl_perturbation_50p_attn.yaml`: same as `rl_perturbation_50p.yaml` plus `n_attn_layers=2, n_heads=4`. Resulting model is 250K params vs 150K baseline.
3. Ran 50 epochs of GRPO with the new architecture on `data/mini_perturb_seq_500g_50p_ntc.h5ad`.
4. Extended `scripts/eval_rl_perturbation.py` to produce:
   - **UMAP of the model's prediction-probability vectors** across R=30 trials per pert, colored two ways: by true pert, and by "bio-equivalence cluster" (single-linkage on reward cos-sim ≥ 0.9). Shows whether trials from the same pert cluster in prediction space.
   - **Per-pert reward boxplot**: distribution of per-trial `cos-sim(profile[pred], profile[true])` for each pert, sorted by mean.
   - **Summary JSON** with top-k (1, 3, 5, 10), MRR, mean reward, and reward-threshold hit rates (≥ 0.5 / 0.8 / 0.9 / 0.95 / 0.99).
5. Re-ran the new eval on both checkpoints for apples-to-apples.

## Results

**Training-curve comparison at matched epochs** (same config otherwise):

| epoch | MLP reward | Attn reward | MLP top-1 | Attn top-1 |
|---|---|---|---|---|
| 5  | 0.311 | 0.311 | 0.020 | 0.030 |
| 10 | 0.340 | 0.337 | 0.024 | 0.041 |
| 20 | 0.485 | 0.480 | 0.044 | 0.051 |
| 30 | 0.519 | **0.567** | 0.054 | **0.115** |
| 40 | 0.535 | **0.615** | 0.084 | **0.155** |
| 50 | 0.574 | **0.637** | 0.142 | **0.186** |

Attention pulls ahead from epoch 30 onward. At ep 50, attention has +11% relative reward and +31% relative top-1. Entropy at ep 50 is 1.44 — still far from saturation; the attention model is still climbing.

**Held-out evaluation (R=30 fresh subsamples per pert)**:

|  | MLP (334 ep) | Attn (50 ep) |
|---|---|---|
| top-1 | 0.427 | 0.154 |
| top-3 | 0.547 | 0.275 |
| top-5 | 0.574 | 0.363 |
| top-10 | 0.637 | 0.582 |
| MRR | 0.513 | 0.277 |
| mean reward | **0.716** | 0.616 |
| P(reward ≥ 0.9) | **0.474** | 0.231 |
| P(reward ≥ 0.5) | **0.758** | 0.669 |

The 334-epoch MLP still wins on absolute numbers — 7× more training, of course. The key comparison is **at matched 50 epochs** (above), where attention is decisively ahead. A 300+ ep attention run would likely surpass the baseline.

**Metric interpretation**:
- For the MLP at 334 epochs: **47% of trials land in a bio-equivalent pert** (cos-sim ≥ 0.9). That's the fairest "did the model get it right?" metric given the reward-degeneracies in the delta-mean profile table (11 pert pairs have cos-sim > 0.9).
- **76% within-class hit rate** (cos-sim ≥ 0.5). The remaining errors are genuinely off-target predictions.
- The mean held-out reward of 0.72 vs the random baseline of 0.17 says the model is firmly in the signal regime.

## Key plots

- `eval_results/rl_perturbation_50p/{umap_predictions,per_pert_rewards,confusion_*,training_curves}.png`
- `eval_results/rl_perturbation_50p_attn/{umap_predictions,per_pert_rewards,confusion_*,training_curves}.png`
- `eval_results/rl_perturbation_50p{,_attn}/metrics.json`

The UMAP of the MLP's output-probability vectors is particularly satisfying: each pert's 30 trials form a tight cluster, and bio-equivalent perts (ETS2/MAPK1/ETS2_MAPK1, CEBPE clusters) sit in adjacent regions of the space. This is the confusion-matrix story as a continuous geometry.

## Problems encountered

1. Initial eval-script load failed because the checkpoint had `gene_attn` keys that the default model constructor didn't instantiate. Fixed by threading `n_attn_layers` / `n_heads` from the saved config into the eval-time model constructor.
2. `enable_nested_tensor` warning from `TransformerEncoder` — harmless performance hint, ignored.

## Next steps

- Run attention at 300+ epochs to get a fair head-to-head with the 334-epoch MLP baseline.
- Consider a CLS-token readout instead of mean+max pool for the attention model.
- Replace the delta-mean reward with something less degenerate (delta quantile, VAE-latent profile).

## Branch

`claude/rl-perturbation-classifier-BNzAX`
