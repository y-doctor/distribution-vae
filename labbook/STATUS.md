# Project Status

**Last updated**: 2026-04-17 20:48 UTC
**Updated by**: Session — reward-function intuition viz + linear-rescale hinge A/B

## What works
- `dist_vae/losses.py` — All loss functions + CombinedDistributionLoss (47 tests pass)
- `dist_vae/data.py` — SyntheticDistributionDataset, PerturbationDistributionDataset, quantile grid utilities
- `dist_vae/model.py` — DistributionEncoder, DistributionDecoder, DistributionVAE with free-bits support
- `dist_vae/train.py` — Trainer with KL warmup, gradient clipping, checkpointing, perturbation/gene labels in snapshots
- `dist_vae/eval.py` — All evaluation functions with perturbation/gene labels in all plots
- `scripts/` — All CLI scripts (train, evaluate, encode, download, hyperopt) + quantile-grid analysis plots
- **Posterior collapse fixed** — beta=0.0001 + latent_dim=16 gives Cramer=0.0092, all 16 dims active
- **K=64 quantile-grid tokenization validated as standalone embedding** (no VAE required for n_cells >= 100) — see eval_results/quantile_tokenization/
- **GRPO perturbation-classifier on K=64 tokens** — 50 epochs: mean reward 0.78, top-1 acc 50% (vs 10% random). `dist_vae/rl_{data,model,train}.py`, `scripts/train_rl.py`, `eval_results/rl_perturbation/`. Trained on new `data/mini_perturb_seq_with_ntc.h5ad` (includes 11855 NTC cells).
- **GRPO scaled to 500 HVGs x 50 perts** — 334 epochs: train mean reward 0.84, train top-1 0.60, **held-out top-1 0.43** (vs 0.02 random = 21x). Confusion matrix shows errors cluster exactly at reward-degenerate pairs. `configs/rl_perturbation_50p.yaml`, `scripts/eval_rl_perturbation.py`, `eval_results/rl_perturbation_50p/`. Trained on new `data/mini_perturb_seq_500g_50p_ntc.h5ad`.
- **Cross-gene transformer attention** — optional `n_attn_layers` on `PerturbationClassifier`. 50-epoch A/B: at matched budget attention gives reward 0.64 vs MLP 0.57, top-1 0.19 vs 0.14 — attention is still climbing at ep 50. Full eval includes UMAP of prediction vectors (per-pert clusters + bio-equiv groups), per-pert reward boxplot, and top-k / reward-threshold summary metrics. See `configs/rl_perturbation_50p_attn.yaml`, `eval_results/rl_perturbation_50p_attn/`. At 300 epochs attention overfits: train reward 0.83, held-out 0.72 (tied with MLP).
- **Full scale: 2k HVGs x 236 perts with held-out cell split + test-time ensembling** — 150-epoch MLP. Train-cell mean reward 0.812, held-out (val-cell) 0.718 with 10x ensemble 0.729. Ensembling contributes +0.011 reward / +0.015 top-1. See `data/mini_perturb_seq_2kg_allp_ntc.h5ad`, `configs/rl_perturbation_2kg_allp.yaml`, `eval_results/rl_perturbation_2kg_allp/`.
- **Row-normalized reward** — pre-z-score each row of the (P, P) reward table before GRPO. Train top-1 0.42 → 0.51 (+8.5pp); held-out top-10 0.38 → 0.46 (+7.1pp, ens=1); P(reward >= 0.9) 0.33 → 0.38 (ens=1). Recommended default. `configs/rl_perturbation_2kg_allp_rownorm.yaml`, `eval_results/rl_perturbation_2kg_allp_rownorm/`.
- **Per-cell set-transformer classifier (rl_cell)** — `dist_vae/rl_cell_model.py`. Raw-cells in, K=16 learned gene modules, 2-layer cell self-attn + 2-layer pert→NTC cross-attn + CLS pool. 150-ep 500g/50p held-out: top-1 0.36, P(r≥0.9) 0.46, top-10 0.78 (ens=10). Trained in ~13 min on CPU. See `eval_results/rl_cell_50p/val_ens10/`.
- **Reward-landscape intuition viz** — `scripts/viz_reward_landscape.py` plots the sorted Pearson-reward surface per pert with the NTC baseline, showing (a) crowding on strong perts (CEBPA has 10 bio-equiv neighbors above r=0.5), (b) low-support on weak perts (BCL2L11 has 0 above r=0.5), (c) off-diagonal distribution peaks near the mean baseline 0.213 — hinge is milder than expected on singles. `eval_results/reward_landscape/`.
- **Linear-rescale hinge** — `r_eff = relu((r - θ) / (1 - θ))` above the NTC baseline. New `apply_hinge` helper + `hinge_rescale`/`hinge_multiplier` config flags in `dist_vae/rl_train.py`. 150-ep 2kg/singles A/B vs the binary hinge: held-out top-1 0.075 → 0.091 (ens=1), 0.072 → 0.101 (ens=10), P(r≥0.9) 0.083 → 0.112, MRR 0.161 → 0.180. Ensembling also starts helping (+1pp top-1 from ens=10). `configs/rl_2kg_singles_mlp_pearson_rescale.yaml`, `eval_results/rl_2kg_singles_mlp_pearson_rescale/`.
- Package installable via `pip install -e ".[dev]"`
- All 61 tests pass on CPU

## Key findings (2026-04-16)
- K=64 quantile grid captures 97% of W1 loss-reduction attainable at K=1024; per-dist W1 median 0.006, p99 0.019
- Sampling jitter scales n_cells^-0.5; SNR (signal/jitter) reaches 5 at n~80, 10 at n~300
- VAE's main remaining value: denoising low-n_cells tokens and ~4x further compactness

## What's broken / blocked
- Latent correlations still high — disentanglement could be improved
- Tail reconstruction still imperfect for zero-inflated distributions (but grid reconstruction is near-lossless)
- Only tested on mini Norman (100 genes, 10 perturbations) — needs full-scale validation

## What's in progress
- Reward-function evaluation: linear rescaling above NTC hinge is a clear win (above). Next knob to test: `hinge_multiplier=2.0` + rescale (stricter threshold) — risk is zero-reward groups on weak perts.

## Next priorities
1. A/B: `hinge_multiplier=2.0` + rescale on 2kg/singles — does the stricter threshold help top-1 more, or does it kill weak perts?
2. Port rescale to the rl_cell (set-transformer) path and re-run 50p A/B.
3. Port rescale + combos to full 237-pert data — hinge bites harder there, more to gain.
4. Extend training past 150 ep — reward was still climbing, entropy still high.
5. Consider changing `grid_size` default 256 -> 64 in dist_vae/data.py
6. Add `scripts/encode_as_grid.py` as a VAE-free baseline encoder

## What's in the repo (data files)
- `data/synthetic_2k.h5ad` — 2000 synthetic distributions (2.1 MB, committed)
- `data/mini_perturb_seq.h5ad` — Mini Norman 2019: 9452 cells x 100 genes x 10 perts (4.6 MB, committed)
- `data/mini_perturb_seq_2kg_allp_ntc.h5ad` — 2000 HVGs × all 236 perts, with NTC (gitignored, ~100 MB). Rebuild with `python scripts/download_2kg_data.py`.

## Eval results
- `eval_results/real_1k_epochs/` — Baseline: beta=0.01, d=32, 1000 epochs (COLLAPSED)
- `eval_results/beta_0.001/` — beta=0.001, d=32, 500 epochs
- `eval_results/beta_0.0001/` — beta=0.0001, d=32, 500 epochs
- `eval_results/beta_0.0001_dim16/` — **BEST**: beta=0.0001, d=16, 500 epochs
- `eval_results/beta_0.001_freebits/` — beta=0.001, d=32, free_bits=0.5, 500 epochs

## Environment
- Python version: 3.11
- PyTorch version: 2.10.0
- Last tested on: CPU, Linux
