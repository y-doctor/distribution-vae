# Publication-ready K=64 justification panel
**Date**: 2026-04-16 17:45 UTC
**Duration**: ~25 minutes
**Goal**: Produce a single polished 2x3 figure that justifies K=64 as the quantile-grid tokenization size.

## What I did
- Wrote `scripts/plot_k64_panel.py` — self-contained, clean-style (publication) figure with consistent palette (blue=original, orange=K=64 token).
- Ran it on `data/mini_perturb_seq.h5ad` across 300 (gene, pert) distributions (min 50 cells).
- Saves both `panel_K64.png` (220 DPI, 17.5x10.5 in) and `panel_K64.pdf`.

## Key changes
- `scripts/plot_k64_panel.py`: new 2x3 panel figure.

## Results
Knees (fraction of loss-reduction at K=1024 captured):
- W1:    90% @ K=32, 95% @ K=64, 99% @ K=256
- Cramer: 90% @ K=64, 95% @ K=128, 99% @ K=512

At K=64:
- Captures 97% of attainable W1 reduction, 91% of Cramer reduction.
- Per-distribution W1 over 300 dists: median = 0.006, p90 = 0.011, p99 = 0.019 (in log-normalized count units).

Recommendation: K=64 is the right default. It is 4x more compact than the current K=256 grid_size at essentially no loss of reconstruction quality, and only 4x larger than the VAE latent (d=16) — suggesting the VAE's compression advantage is modest.

## Problems encountered
- Initial panel (b) header used axis-relative coords that collided with the figure suptitle. Fixed by computing the bbox of the top sub-axis and placing the label in figure coords.

## Next steps
- Consider changing `dist_vae/data.py` default `grid_size` from 256 to 64 and retraining the VAE at the smaller token size to see if the latent compresses further (d=8 maybe).
- Use the loss-vs-K numbers to motivate a "grid-only baseline" section in the eventual writeup.

## Open questions
- Does K=64 change the VAE's posterior-collapse dynamics?
- Can a non-uniform quantile grid (denser near tails) get to 95% W1 reduction at K<64?
