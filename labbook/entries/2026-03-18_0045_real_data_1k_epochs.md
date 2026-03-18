# 1000-Epoch Training on Mini Norman Dataset
**Date**: 2026-03-18 00:45 UTC
**Duration**: ~20 minutes
**Goal**: Full training run on real Perturb-seq data, evaluate reconstruction and latent quality

## What I did
1. Ran 1000-epoch training on `data/mini_perturb_seq.h5ad` (1000 distributions from 10 perturbations x 100 genes)
2. Ran full evaluation pipeline
3. Saved all outputs to `eval_results/real_1k_epochs/`

## Key changes
- `eval_results/real_1k_epochs/`: Training curves, reconstruction snapshots, latent PCA, interpolations, latent statistics, metrics

## Results

### Training metrics (1000 epochs, CPU)
- Train loss: 5436 → 0.042 (recon: 5436 → 0.021)
- Val loss: 5262 → 0.044 (recon: 5262 → 0.023)
- Best val recon loss: **0.0207**
- KL divergence stabilized at ~2.0 (very low)

### Evaluation metrics (500 distributions)
| Metric | Mean | Std |
|--------|------|-----|
| Cramer | 0.020 | 0.034 |
| KS | 0.597 | 0.420 |
| W1 | 0.078 | 0.062 |

### Key observations
1. **Reconstruction is excellent numerically** — Cramer 0.020 vs 0.573 on synthetic (28x better)
2. **But posterior collapse is severe**:
   - Latent dims crammed into [-0.1, 0.04] range; z_2 is a delta spike
   - KL = 2.0 means posterior ≈ prior; latent space is barely used
   - PCA shows 1D manifold — bulk of points collapsed into tight cluster
3. **Interpolations are nearly constant** — moving through latent space produces minimal change
4. **All reconstructions look similar** — model learned the average zero-inflated gene expression shape but doesn't differentiate between distributions
5. **Root cause**: beta=0.01 is too aggressive for 1000 distributions with 32 latent dims. KL penalty crushes the posterior before the encoder learns meaningful structure.

### Comparison: Synthetic vs Real
| | Synthetic (100 ep) | Real (1000 ep) |
|---|---|---|
| Cramer | 0.573 | 0.020 |
| KL | 22.8 | 2.0 |
| Latent range | [-6, 6] | [-0.1, 0.04] |
| Collapse? | Partial (1 dim) | Severe (most dims) |

## Problems encountered
- Posterior collapse on real data due to low KL / aggressive beta

## Next steps
- **Immediate**: Try beta=0.001 or beta=0.0001 to allow latent space to learn
- **Or**: Reduce latent_dim to 8 or 16 to force efficient use
- **Or**: Use free-bits / KL thresholding to prevent collapse
- Consider whether the data is too homogeneous (zero-inflated distributions are similar) and needs more genes or perturbations

## Open questions
- Is the zero-inflated structure causing the model to "cheat" by just learning a single template?
- Would adding the Wasserstein-1 loss help differentiate distributions in the upper tail?
