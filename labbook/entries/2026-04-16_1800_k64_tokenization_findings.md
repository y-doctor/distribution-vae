# K=64 quantile-grid tokenization — findings and decision
**Date**: 2026-04-16 18:00 UTC
**Duration**: ~3 hours
**Goal**: Answer "can we just tokenize distributions using the simple quantile
grid and skip the VAE?" — quantify fidelity, pick a default K, and measure
sampling noise.

## Answer

**Yes, for most downstream uses, ship the raw K=64 quantile grid as the
distribution embedding.** The VAE remains valuable for (a) denoising tokens
from perturbations with < 50 cells and (b) ~4x further compactness
(d=16 latent vs. 64-dim grid), but it is not required for a faithful,
downstream-usable embedding.

## What I did

Built three analyses (see `eval_results/quantile_tokenization/`):

1. **Fidelity sweep** (`scripts/plot_k64_panel.py`) — for each of 300 (gene,
   pert) distributions from the mini Norman dataset (min 50 cells), evaluated
   empirical quantile function vs K-point token interpolant on an M=4096
   reference grid. Computed W1 (MAE) and Cramer-RMSE (RMS) for
   K in {4, 8, 16, 32, 64, 128, 256, 512, 1024}.

2. **Diminishing-returns analysis** — for each loss, normalized the
   reduction curve so 0 corresponds to K=4 and 1 corresponds to K=1024,
   then identified the smallest K that crossed 90 / 95 / 99 %.

3. **Jitter vs signal analysis** (`scripts/plot_jitter_analysis.py`) — for
   80 (gene, pert) pairs with >= 700 cells, bootstrapped 30 sub-samples at
   n_cells in {10, 20, 50, 100, 200, 500}, tokenized each, computed W1 vs
   the full-n token (jitter). Compared to the W1 between distinct full-n
   tokens (signal), and reported SNR = median(signal) / median(jitter).

## Results

### Fidelity (Perturb-seq log-normalized counts)

Knees (fraction of K=1024 loss-reduction captured):

| Metric       | 90% @ K = | 95% @ K = | 99% @ K = |
|--------------|-----------|-----------|-----------|
| W1           | 32        | 64        | 256       |
| Cramer-RMSE  | 64        | 128       | 512       |

At K=64:
- 97% of W1 reduction and 91% of Cramer reduction captured.
- Per-dist W1 over 300 dists: median = 0.006, p90 = 0.011, p99 = 0.019
  (expression units on log-normalized counts).

### Jitter vs signal at K=64

Signal: median W1 between full-n tokens of distinct (gene, pert) = **0.202**.

| n_cells | jitter median W1 | SNR = signal/jitter |
|--------:|-----------------:|--------------------:|
| 10      | 0.094            | 2.2                 |
| 20      | 0.072            | 2.8                 |
| 50      | 0.050            | 4.0                 |
| 100     | 0.036            | 5.5                 |
| 200     | 0.026            | 7.8                 |
| 500     | 0.013            | 15.0                |

Jitter scales ~n_cells^-0.5 as expected from empirical-quantile theory.
SNR crosses 5 around n=80 and 10 around n=300.

## Why K=64 (not K=256 or K=16)

- K=256 (current library default for `grid_size`): essentially lossless but
  wastes ~4x the dimensionality relative to the real information content.
- K=16: captures only ~80% of W1 reduction; loses tail fidelity on
  zero-inflated distributions (the dominant shape in Perturb-seq).
- K=64: sits at the knee. Captures 97% of W1 reduction, 4x smaller than
  the default, only 4x larger than the d=16 VAE latent.

## Why "just use the grid" works

- The quantile grid IS a complete sufficient statistic for a 1D distribution.
- At K=64, the per-distribution reconstruction W1 is ~0.006 expression units
  — effectively invisible in downstream use.
- Zero-inflation is captured exactly (grid points in the zero-plateau
  region are exactly 0).
- The embedding is permutation-invariant by construction, no learned
  parameters, CPU-fast.

## When to still use the VAE

- **n_cells < 50 per (gene, pert)**: token jitter is comparable to
  between-distribution signal. VAE prior acts as a smoother.
- **GPU memory / attention-style models over many distributions**: d=16
  latent is 4x lighter than a 64-dim grid.
- **Generative tasks** (sample new distributions, interpolate): needs
  a density over distributions, which the VAE provides.

## Key changes

- `scripts/plot_quantile_tokenization.py` — 10-plot exploratory set.
- `scripts/plot_k64_panel.py` — publication-ready 2x3 figure.
- `scripts/plot_jitter_analysis.py` — jitter vs signal vs SNR.
- `eval_results/quantile_tokenization/` — all figures, plus README.md
  indexing them.

## Open questions / next steps

- Change `dist_vae/data.py` `grid_size` default to 64? (not done here —
  separate PR to avoid mixing analysis and API changes).
- Zero-inflation-aware tokens: split into (zero_fraction, K-point grid over
  non-zero values). Likely ~2x further compactness at same fidelity.
- Train the VAE at K=64 tokens and re-run posterior-collapse diagnostics —
  smaller input may change the collapse dynamics.
- Add an `encode_as_grid.py` script as a parallel to `encode_dataset.py`
  for the VAE-free path.

## Read next

For anyone picking this up:
- Start with `eval_results/quantile_tokenization/README.md` + `panel_K64.png`.
- Then `jitter_K64.png` for the sampling-noise story.
- `labbook/DECISIONS.md` for the one-paragraph architectural-decision form.
