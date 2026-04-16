# RL perturbation-classifier (GRPO) kickoff
**Date**: 2026-04-16 21:15 UTC
**Duration**: ~2 hours
**Goal**: Train a model to predict "which perturbation was this?" from a (NTC phenotype, perturbed phenotype) pair using GRPO (Group Relative Policy Optimization) with a cosine-similarity-based reward.

## Setup

- **Input**: two K=64 quantile-grid tokens per gene (one for 100 random NTC cells, one for 100 random perturbed cells), for each of 100 expression genes.
- **Output**: softmax policy over 10 perturbations.
- **Reward**: cosine similarity between `delta_mean_expression[predicted_pert]` and `delta_mean_expression[true_pert]` where `delta_mean[p] = mean(X_pert) - mean(X_NTC)` per gene.
- **Dataset**: `data/mini_perturb_seq_with_ntc.h5ad` — regenerated from the full Norman source via the new `--keep-controls` flag. 21307 cells × 100 genes × 11 perts (10 knockdowns + 11855 NTC cells).
- **GRPO**: batch 8, group G=10 (full enumeration over 10 classes so advantages have no sampling noise), entropy bonus 0.05, lr 3e-4, 50 epochs.

## Key changes

- `scripts/make_mini_dataset.py`: added `--keep-controls` flag. Produces `data/mini_perturb_seq_with_ntc.h5ad`.
- `dist_vae/losses.py`: added `cosine_similarity(x, y, dim, eps)`.
- `dist_vae/rl_data.py`: new `PerturbationClassificationDataset` with per-idx resampling, gene-vocabulary builder (union of expression genes and pert-target genes, compound perts like `UBASH3B_OSR2` split on `_`), and `compute_delta_mean_profiles()`.
- `dist_vae/rl_model.py`: new `PerturbationClassifier`. Final architecture:
  - gene embedding `nn.Embedding(n_all_genes=110, 32)`
  - per-gene MLP: `LayerNorm → Linear(3K+d, 128) → GELU → Linear(128, 64)` where the input concat is `(ntc_token, pert_token, pert-ntc delta_token, gene_emb)`
  - mean+max pool across genes → `LayerNorm → Linear(128, 128) → GELU → Linear(128, 10)`
  - 59k params total.
- `dist_vae/rl_train.py`: new `GRPOTrainer` — full-enumeration group advantages (`(r - mean) / std` within each input), entropy bonus, optional KL-to-ref-policy (default off).
- `scripts/train_rl.py`, `configs/rl_perturbation.yaml`: CLI + config.
- Tests: `tests/test_cosine_similarity.py`, `tests/test_rl_data.py`, `tests/test_rl_model.py` — 14 tests pass. Full suite: 61 passed, 2 skipped.

## Results

50-epoch GRPO run on `data/mini_perturb_seq_with_ntc.h5ad`:

| metric               | start | best    | final |
|----------------------|-------|---------|-------|
| mean reward (cos-sim)| 0.34  | **0.785** | 0.78  |
| top-1 accuracy       | 0.17  | **0.50**  | 0.50  |
| policy entropy       | 1.52  | —       | 0.006 |

Random baselines: mean off-diagonal cos-sim = 0.047, top-1 acc = 0.10.

Training curves saved to `eval_results/rl_perturbation/training_curves.png`. Best checkpoint at `checkpoints/rl_perturbation/best.pt`.

## Why top-1 plateaus at ~50%

The delta-mean profile matrix is not injective across perts:
- ETS2 vs MAPK1: cos-sim 0.99 — effectively indistinguishable by this reward
- ZBTB1 vs ZBTB25: cos-sim 0.75 (same TF family)
- CEBPE/ETS2/MAPK1/RUNX1T1: pairwise cos-sim ~0.55–0.65

When top-1 is measured against true-pert (not reward-equivalent class), the accuracy ceiling is ~50% even under an optimal policy. The high mean reward (0.78 vs 1.0 max) confirms the model is picking the true pert OR a reward-equivalent alternative most of the time.

## Problems encountered and fixes

1. **Initial architecture collapsed to "always predict CEBPE"** — mean-pool across 100 genes + tiny MLP (hidden 64) discarded too much per-gene signal. Supervised CE also failed to escape random baseline. **Fixed** by adding explicit `delta_token = pert - ntc` as an input channel, widening hidden → 128, adding mean+max pooling, and using a multilayer head with LayerNorm. Supervised sanity check then reached 100% train acc in 15 epochs.
2. **Policy entropy collapsing to zero instantly** with narrow d_feat. **Fixed** by widening d_feat=64 and using `entropy_coef=0.05`.
3. **Logits too peaky at init** — left mitigated by the `1/sqrt(d_feat)` scaling in an earlier draft; dropped once the new head (Linear(128, n_perts)) was introduced since it behaves sensibly without scaling.

## Open questions / next steps

- Replace the Linear classification head with the originally-planned `dot(global_feat, pert_target_embedding)` head once the simpler baseline is established. The `pert_embeddings()` method is kept on the model for this.
- Train a larger G for the gene vocabulary (include many non-selected Norman perts as vocabulary tokens) — might help generalization.
- Try KL-to-ref-policy with `kl_coef > 0` once baseline is solid.
- Held-out validation split: currently we train on the whole 1000-samples/epoch stream; add an eval-only split for faithful accuracy.
- Zero-inflation-aware tokens (still queued in `labbook/TASKS.md`) may improve signal.

## Branch

`claude/rl-perturbation-classifier-BNzAX`
