# Long attention run (300 ep) vs 334-ep MLP baseline — attention overfits
**Date**: 2026-04-17 00:10 UTC
**Duration**: ~2 hours (training) + eval
**Goal**: Train the transformer-attention classifier for 300 epochs to do a fair head-to-head against the 334-epoch MLP baseline.

## What I did

- Ran `scripts/train_rl.py --config configs/rl_perturbation_50p_attn.yaml` for 300 epochs (same dataset, same entropy_coef=0.3, lr=1e-4, samples_per_epoch=300, group_size=50 as baseline).
- Checkpoint to `checkpoints/rl_perturbation_50p_attn_long/best.pt` (best mean reward epoch).
- Evaluated with `scripts/eval_rl_perturbation.py` (R=30 fresh subsamples per pert).

## Training dynamics

| epoch | reward | top-1 | entropy |
|---:|---:|---:|---:|
| 50  | 0.64 | 0.19 | 1.44 |
| 100 | 0.77 | 0.40 | 0.30 |
| 150 | 0.80 | 0.50 | 0.08 |
| 200 | 0.81 | 0.50 | 0.06 |
| 250 | 0.81 | 0.52 | 0.05 |
| 300 | 0.82 | 0.53 | 0.03 |

Plateaued cleanly around ep 180. Training reward peak 0.826 (vs 0.725 for MLP baseline at 334 ep).

## Head-to-head — **held-out evaluation** (R=30 fresh 100-cell subsamples per pert)

| metric | MLP (334 ep) | Attn (300 ep) | Δ |
|---|---:|---:|---:|
| **training mean reward** | 0.725 | **0.826** | +0.10 |
| **held-out mean reward** | 0.716 | 0.718 | ~tie |
| held-out top-1 | **0.427** | 0.363 | -0.06 |
| held-out top-3 | **0.547** | 0.491 | -0.06 |
| held-out top-5 | 0.574 | 0.558 | -0.02 |
| held-out top-10 | 0.637 | **0.645** | +0.01 |
| held-out MRR | **0.513** | 0.466 | -0.05 |
| P(reward ≥ 0.5) | 0.758 | **0.770** | +0.01 |
| P(reward ≥ 0.8) | **0.541** | 0.495 | -0.05 |
| P(reward ≥ 0.9) | **0.474** | 0.408 | -0.07 |

## Interpretation

**Attention overfits.** The 50-epoch comparison was misleading because both models were still far from their respective plateaus — it captured "attention learns faster", but not "attention generalizes better". At convergence the picture flips:

- **Training reward** moved up 0.10 absolute for attention (the model memorizes training subsamples better).
- **Held-out reward** stayed flat (0.716 → 0.718). The extra capacity isn't buying biological generalization.
- **Strict top-k and reward-threshold metrics regress slightly** for attention. This is consistent with overfitting: the model is more confident on perts it has seen but less reliable on fresh subsamples.

The MLP's **training-to-held-out gap is ~0.009**; attention's is **~0.108**. That 12× generalization gap is the story.

Why? Attention has 1.7× the parameters (250K vs 150K) without any added regularization. With only ~30 distinct cell-subsets per pert per epoch at samples_per_epoch=300, the larger model can memorize fine-grained sampling noise that doesn't transfer.

## What would help attention

- **Dropout** in the attention layers (currently 0.0).
- **Weight decay** > 0.
- **Data augmentation**: more diverse cell subsamples per pert per epoch (increase samples_per_epoch).
- **More NTC subsampling diversity** (e.g. resample NTC per-call instead of sharing one NTC sample across genes).
- **Early stopping at ep 50-100** where attention was still improving held-out — suggests sweet spot before memorization kicks in.

## Conclusion for this session

**The 334-ep MLP baseline is the best result so far.** Cross-gene attention does learn faster and hits a higher training ceiling, but overfits relative to the MLP under the current training regime. Attention may become useful with regularization or more data per epoch — both parked as followups.

## Files

- `eval_results/rl_perturbation_50p_attn_long/` — full artifacts for the 300-ep attention run.
- Training curves, confusion matrix, confusion-vs-reward-sim, UMAP, per-pert rewards, metrics.json all present.
- Saved `checkpoints/rl_perturbation_50p_attn_long/best.pt`.

## Branch

`claude/rl-perturbation-classifier-BNzAX`.
