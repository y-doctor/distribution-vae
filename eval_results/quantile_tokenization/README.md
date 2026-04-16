# Quantile-grid tokenization — plot index

Findings and artifacts from the analysis that justified switching the default
distribution embedding to the **K=64 quantile grid** (no VAE required for
n_cells >= ~100).

## Headline

For per-(gene, pert) Perturb-seq distributions on log-normalized counts
(data/mini_perturb_seq.h5ad):

- **K=64** captures 97% of the attainable W1 reconstruction-loss reduction
  (vs. K=1024) and 91% of the Cramer-RMSE reduction.
- Per-distribution W1 at K=64 across 300 distributions:
  median = 0.006, p90 = 0.011, p99 = 0.019 (log-normalized count units).
- Sampling jitter scales as ~n_cells^-0.5. Signal-to-jitter ratio reaches
  5 at n_cells ~= 80 and 10 at n_cells ~= 300.
- The **VAE's strongest use case** is denoising tokens at n_cells < 50,
  where jitter is comparable to between-distribution signal.

## Files

- `panel_K64.{png,pdf}` — **main publication figure**: 2x3 panel.
  - (a) Tokenization concept.
  - (b) Three histogram reconstructions at K=64.
  - (c) Three quantile-function reconstructions at K=64.
  - (d) Loss-vs-K with mean + IQR.
  - (e) Normalized diminishing-returns with 90/95/99% knees.
  - (f) Per-distribution W1 at K=64 across 300 distributions.
- `jitter_K64.{png,pdf}` — **jitter analysis**: 3-panel.
  - (a) Jitter (W1 sub-sample vs full-n token) vs n_cells.
  - (b) Signal vs jitter histograms (log-x).
  - (c) Signal-to-jitter ratio vs n_cells.

## Supporting plots (exploratory)

- `01_walkthrough.png` — samples -> sort -> K=32/256 token for one distribution.
- `02_gallery.png` — 6 diverse (gene, pert) raw hists and their K=256 tokens.
- `03_resolution_tradeoff.png` — one distribution tokenized at K in {8..256}.
- `04_token_matrix.png` — 500-distribution x 256-grid heatmap + overlay.
- `05_sampling_noise.png` — 5 bootstrap resamples at n_cells in {20, 50, 200, full}.
- `06_reconstruction_gallery.png` — original vs reconstructed hist, K=256, 6 dists.
- `07_reconstruction_by_K.png` — one distribution's reconstruction at K in {8..256}.
- `08_3x3_histograms.png` — 3 genes x 3 perts hist comparison at K=256 with W1.
- `09_3x3_quantiles.png` — same 3x3 as quantile-function overlays.
- `10_loss_vs_K.png` — loss vs K with normalized diminishing returns.

## How to regenerate

```bash
# Main panel + K sweep
python scripts/plot_k64_panel.py --adata data/mini_perturb_seq.h5ad \
    --output-dir eval_results/quantile_tokenization

# Jitter analysis
python scripts/plot_jitter_analysis.py --adata data/mini_perturb_seq.h5ad \
    --output-dir eval_results/quantile_tokenization

# Exploratory plot set
python scripts/plot_quantile_tokenization.py --adata data/mini_perturb_seq.h5ad \
    --output-dir eval_results/quantile_tokenization
```

## See also

- `labbook/entries/2026-04-16_1800_k64_tokenization_findings.md` — full
  narrative and decision.
- `labbook/DECISIONS.md` — architectural decision entry.
- `dist_vae/data.py:samples_to_quantile_grid` — the tokenization function itself.
