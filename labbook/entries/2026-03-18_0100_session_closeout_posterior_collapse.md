# Session Close-out: Real Data Training & Posterior Collapse Diagnosis
**Date**: 2026-03-18 01:00 UTC
**Duration**: ~60 minutes
**Goal**: Train on real Perturb-seq data, evaluate quality, diagnose issues

## What I did
1. Ran 1000-epoch training on mini Norman dataset (1000 distributions, 10 perturbations x 100 genes)
2. Ran full evaluation pipeline — reconstructions, latent PCA, interpolations, latent statistics
3. Diagnosed severe posterior collapse: latent range [-0.1, 0.04], KL=2.0, most dims unused
4. Compared real vs synthetic results to understand the failure mode
5. Documented findings and proposed fixes

## Key changes
- `eval_results/real_1k_epochs/`: All evaluation outputs (training curves, reconstructions, latent PCA, interpolations, metrics)
- `labbook/entries/2026-03-18_0045_real_data_1k_epochs.md`: Detailed analysis of training results

## Results
- Cramer distance 0.020 (looks good numerically but misleading)
- Model learned a single zero-inflated template shape — all reconstructions look the same
- Posterior collapse: beta=0.01 too aggressive for 1000 distributions with 32 latent dims
- Latent space barely used — interpolations produce no meaningful variation

## Problems encountered
- Posterior collapse is the main blocker for real-data use
- Need hyperparameter tuning before the model is useful on Perturb-seq data

## Next steps
1. **Fix posterior collapse** — try beta=0.001, reduce latent_dim to 8-16, or add free-bits
2. **Hyperparameter sweep** — systematic search over beta, latent_dim, hidden_dim
3. **Consider data augmentation** — more genes or perturbations to increase diversity
4. **Integration tests** — add training loop tests to prevent regressions

## Open questions
- Is zero-inflated structure causing the model to shortcut by learning one template?
- Would W1 loss help differentiate distributions in the tail?
- Should we log-transform quantile grids before feeding to the model?
