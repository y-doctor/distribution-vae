# Fix posterior collapse on real Perturb-seq data
**Date**: 2026-03-18 02:00 UTC
**Duration**: ~90 minutes
**Goal**: Fix posterior collapse where model collapses to single template on mini Norman data

## What I did
1. Added perturbation/gene labels to all evaluation and training plots
2. Ran systematic beta sweep: beta=0.001 and beta=0.0001, both 500 epochs on mini Norman data
3. Implemented free-bits (per-dimension KL floor) in DistributionVAE
4. Tested beta=0.001 + free_bits=0.5 and beta=0.0001 + latent_dim=16
5. Generated full evaluation reports for all 4 configurations
6. Compared against baseline (beta=0.01, latent_dim=32, 1000 epochs)

## Key changes
- `dist_vae/eval.py`: Added `_get_label()` helper; `plot_reconstructions()` shows perturbation/gene titles; `plot_latent_space()` uses actual names in legend; `plot_interpolations()` labels endpoints
- `dist_vae/train.py`: Added `_get_snapshot_labels()` to show perturbation/gene names in training reconstruction snapshots
- `dist_vae/model.py`: Added `free_bits` parameter to DistributionVAE; `compute_loss()` now supports per-dimension KL floor
- `scripts/train.py`, `scripts/evaluate.py`: Pass `free_bits` from config
- `configs/default.yaml`: Added `free_bits: 0.0`
- `configs/example_perturb_seq.yaml`: Updated to best settings (beta=0.0001, latent_dim=16, 500 epochs)

## Results

### Full comparison table (500 epochs unless noted, mini Norman data)

| Setting | Cramer | W1 | KL | Latent range | Active dims | Mean dim std |
|---------|--------|-----|-----|-------------|-------------|-------------|
| baseline (beta=0.01, d=32, 1000ep) | 0.0200 | 0.078 | 2.0 | [-0.1, 0.04] | ~0/32 | ~0.01 |
| beta=0.001, d=32 | 0.0162 | 0.068 | 8.0 | [-1.6, 8.5] | 27/32 | 0.25 |
| beta=0.001+freebits=0.5, d=32 | 0.0132 | 0.060 | 19.1 | [-2.8, 7.0] | 32/32 | 0.51 |
| beta=0.0001, d=32 | 0.0101 | 0.048 | 40.9 | [-5.8, 9.4] | 29/32 | 0.63 |
| **beta=0.0001, d=16** | **0.0092** | **0.051** | **34.5** | **[-11.9, 4.7]** | **16/16** | **0.72** |

### Key findings
1. **beta=0.01 was way too high** for 1000 real distributions — the KL penalty dominated, forcing posterior to prior
2. **beta=0.0001 resolves collapse completely** — all latent dims active, wide range, 2x better Cramer
3. **Smaller latent_dim=16 helps** — forces each dim to carry more info, best overall reconstruction
4. **Free-bits works** — doubles KL usage at beta=0.001 (8.0 -> 19.1), but not as effective as just lowering beta
5. Reconstruction quality: all models capture zero-inflated structure; tail accuracy improves with lower beta

## Problems encountered
- Training runs take ~30 min each on CPU for 500 epochs (1000 distributions). Ran 2 pairs in parallel.

## Next steps
- Try even longer training (1000 epochs) with best settings
- Consider adding W1 loss for better tail reconstruction
- Investigate latent correlations — disentanglement is still imperfect
- Test on full 500-gene Norman dataset

## Open questions
- Is the parabolic structure in the latent PCA (visible in all models) intrinsic to the data or a model artifact?
- Should we use beta=0.0001 as default for all real data, or make it data-size-dependent?
