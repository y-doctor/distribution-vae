# Quantile-grid tokenization visualizations
**Date**: 2026-04-16 17:10 UTC
**Duration**: ~20 minutes
**Goal**: Visualize what "tokenizing" empirical distributions via the quantile grid looks like, so we can reason about whether the VAE is needed at all or whether the grid alone is a sufficient token.

## What I did
- Wrote `scripts/plot_quantile_tokenization.py` — standalone plotting script that takes an AnnData file and produces 5 figures illustrating the tokenization.
- Ran it on `data/mini_perturb_seq.h5ad` (9452 cells x 100 genes x 10 perts).
- Installed missing deps in the environment: matplotlib, anndata, torch (CPU), and `pip install -e .`.

## Key changes
- `scripts/plot_quantile_tokenization.py`: new script. Produces:
  1. `01_walkthrough.png` — one distribution shown as (histogram → sorted samples → K=32 token → K=256 token).
  2. `02_gallery.png` — 6 diverse (gene, pert) distributions; side-by-side raw histogram and 256-dim token.
  3. `03_resolution_tradeoff.png` — same distribution tokenized at K ∈ {8, 16, 32, 64, 128, 256} overlaid on the empirical quantile function.
  4. `04_token_matrix.png` — heatmap of 500 tokens x 256 grid points (sorted by median), plus 50 overlaid tokens to show shape diversity.
  5. `05_sampling_noise.png` — the same (gene, pert) resampled 5x at n_cells ∈ {20, 50, 200, full}, showing how token jitter shrinks as n_cells grows.

## Results
- Tokenization is visually intuitive: zero-inflated distributions show up as long flat plateaus at 0 followed by a steep ramp in the higher quantiles.
- K=32 already captures the bulk shape on zero-inflated real Perturb-seq genes; K=256 preserves fine tail structure.
- The sampling-noise plot is the key one for the VAE motivation: at n_cells=20 the tokens jitter wildly across resamples of the same (gene, pert); at n_cells≈1500 they are nearly identical. A smoothing model (VAE) helps where n_cells is small.

## Problems encountered
- Env missing matplotlib/anndata/torch — installed on the fly.
- `packaging` RECORD file conflict when pip-installing anndata; bypassed with `--ignore-installed packaging`.

## Next steps
- If we want a pure "grid-as-token" baseline for downstream ML, add a short script that just encodes the full dataset as an (N, K) matrix and saves it.
- Compare grid-only vs VAE-latent on a downstream task to quantify the information loss of compression.

## Open questions
- Is there a better non-uniform quantile grid (e.g. denser near tails) that captures zero-inflated distributions with fewer points?
