# Project Status

**Last updated**: 2026-04-19 22:26 UTC
**Updated by**: Session — hinge_multiplier=2.0 A/B broke the P(r≥0.9) ceiling

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
- **Plateau-stop long run on rescale hinge** — `PlateauDetector` early-stop class in `dist_vae/rl_train.py` + `configs/rl_2kg_singles_mlp_pearson_rescale_long.yaml`. Ran to ep 405/600 (plateau detected: window=30, patience=50, min_delta=0.002). Train top-1 0.32 → **0.555**, held-out top-1 ens=10 0.101 → **0.111**, top-10 ens=10 0.343 → **0.424**, MRR ens=10 0.190 → **0.216**. Longer training = real ranking gains but widening train/held-out gap (5.0× ratio). See entries/2026-04-19_2142_long_training_plateau_results.md.
- **hinge_multiplier=2.0 + rescale on 2kg/singles** — plateau-stopped at ep 387. Held-out ens=10: top-1 **0.174** (from 0.111, +57% rel), top-10 **0.531** (from 0.424, +25% rel), MRR **0.293** (from 0.216, +36% rel), **P(r≥0.9) 0.188** (from 0.119, **+58% rel**). Proves the earlier "biology ceiling" at P(r≥0.9)=0.12 was actually a plateau-trap artifact — doubling the hinge destroys the modest-reward local optimum and forces fine discrimination. P(r≥0.5) preserved at 0.526 (broad-class unchanged). No dead-group regression. `configs/rl_2kg_singles_mlp_pearson_rescale_hinge2x.yaml`, `eval_results/rl_2kg_singles_mlp_pearson_rescale_hinge2x/`. See entries/2026-04-19_2226_hinge2x_broke_the_ceiling.md.
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
- Reward-function tuning: `hinge_multiplier=2.0 + rescale` is now the leading config on 2kg/singles. Next knobs to explore: 3× hinge, entropy floor, combo-data sanity check, rl_cell port.

## Next priorities
1. **Try `hinge_multiplier=3.0`** — is there more above-plateau reward to extract? Higher risk of dead groups on weak perts but worth testing after the 2× surprise.
2. **Port 2× hinge + rescale to rl_cell (set-transformer) path.** Architecture A/B under the best reward shape found.
3. **Sanity check on 237-pert combo data.** Combos introduce genuine reward-degenerate pairs; does 2× hinge still help there, or does the biology ceiling become real?
4. **Entropy-floor regularization.** Entropy collapsed to 0.02 under 2× hinge too — prevent this with KL-to-uniform or scheduled entropy_coef to close the 4.2× train/val gap.
5. Re-run binary-hinge with plateau-stop for a true convergence A/B vs rescale.
6. Consider changing `grid_size` default 256 -> 64 in dist_vae/data.py
7. Add `scripts/encode_as_grid.py` as a VAE-free baseline encoder

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
