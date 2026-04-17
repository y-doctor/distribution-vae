# Task Board

## TODO
- [ ] Change `dist_vae/data.py` `grid_size` default 256 -> 64 (after K=64 analysis, see 2026-04-16 entry) — S
- [ ] Add `scripts/encode_as_grid.py` as VAE-free baseline encoder (produces (N, 64) matrix from AnnData) — S
- [ ] Prototype zero-inflation-aware tokens: (zero_fraction, K-point grid over non-zeros) — S
- [ ] Retrain VAE at K=64 input; re-run posterior-collapse diagnostics at smaller grid — M
- [ ] Run hyperopt on synthetic data end-to-end — S
- [ ] End-to-end test: Norman et al. full dataset → train → encode → eval — M
- [ ] Tune hyperparameters on real data using hyperopt module — M
- [ ] Hyperparameter sweep: beta, latent_dim, hidden_dim, free_bits — M
- [ ] Investigate log-transforming quantile grids before model input — S
- [ ] Try W1 loss for better tail differentiation — S
- [ ] Investigate total correlation penalty for better disentanglement — S
- [ ] Add integration tests for training loop — S
- [ ] Profile memory usage on large datasets — S
- [ ] Test best settings (beta=0.0001, d=16) on full 500-gene Norman dataset — M

## IN PROGRESS
- [ ] Run autoresearch agent overnight — M — started 2026-03-24
- [ ] Prototype per-cell set-transformer classifier (`dist_vae/rl_cell_model.py`) — L — started 2026-04-17 — 19 model-unit tests + 3 dataset tests pass; 10-pert smoke reward 0.29 at ep10 (random 0.10); 50-pert 20-epoch scale-up running

## RECENTLY ADDED
- [ ] Run autoresearch agent on GPU for extended experiment session — M
- [ ] Analyze autoresearch results and integrate best findings into main library — M

## DONE
- [x] **Cross-gene transformer attention + UMAP/per-pert-reward evaluation** — completed 2026-04-16 23:30
  - Added optional `n_attn_layers` to PerturbationClassifier (TransformerEncoder after per-gene MLP)
  - At matched 50-epoch budget, attention beats MLP: reward 0.637 vs 0.574, top-1 0.186 vs 0.142
  - Extended eval with UMAP of output-probability vectors, per-pert reward boxplot, top-k / reward-threshold metrics
  - See entries/2026-04-16_2330_rl_50p_attention_and_umap.md
- [x] **Scale RL classifier to 500 HVGs x 50 perts + held-out eval + confusion matrix** — completed 2026-04-16 23:00
  - 334-epoch run: train reward 0.84, train top-1 0.60, held-out top-1 0.43 (vs 0.02 random = 21x)
  - Found and fixed early entropy collapse (entropy_coef 0.1 -> 0.3, lr 3e-4 -> 1e-4)
  - New eval script scripts/eval_rl_perturbation.py produces confusion matrix + confusion-vs-reward-sim figure
  - Errors cluster exactly at reward-degenerate pairs — model is reward-bounded, not architecturally limited
  - See entries/2026-04-16_2300_rl_50pert_500hvg_scaleup.md
- [x] **RL perturbation-classifier on K=64 tokens (GRPO kickoff)** — completed 2026-04-16 21:15
  - 50-epoch training: mean reward 0.78, top-1 acc 50% (vs 10% random)
  - New: rl_data/rl_model/rl_train modules, train_rl CLI, config, 14 tests
  - Regenerated data/mini_perturb_seq_with_ntc.h5ad with --keep-controls
  - See entries/2026-04-16_2115_rl_perturbation_classifier_kickoff.md
- [x] **Quantile-grid tokenization fidelity + jitter analysis (K=64 justification)** — completed 2026-04-16 18:00
  - 300-dist fidelity sweep: K=64 captures 97% of W1 reduction, per-dist median W1 = 0.006
  - 80-pair jitter sweep: SNR reaches 5 at n~80, 10 at n~300; jitter ~ n_cells^-0.5
  - See entries/2026-04-16_1800_k64_tokenization_findings.md, panel_K64.png, jitter_K64.png
- [x] **Fix posterior collapse on real data** — completed 2026-03-18 02:15
  - beta=0.0001 + latent_dim=16 is best: Cramer=0.0092, all 16 dims active, mean std=0.72
  - Implemented free-bits as alternative approach (works but less effective than lowering beta)
  - Updated configs/example_perturb_seq.yaml with best settings
  - See entries/2026-03-18_0200_fix_posterior_collapse.md
- [x] Add perturbation/gene labels to all plots — completed 2026-03-18 01:30
- [x] Implement free-bits (per-dim KL floor) in DistributionVAE — completed 2026-03-18 01:30
- [x] Implement `dist_vae/hyperopt.py` — completed 2026-03-18 00:30 — 14 tests pass
- [x] Implement `scripts/hyperopt.py` CLI — completed 2026-03-18 00:30
- [x] Implement `tests/test_hyperopt.py` — completed 2026-03-18 00:30
- [x] Add epoch_callback to Trainer.train() — completed 2026-03-18 00:30
- [x] Create HTML project report (report.html) — completed 2026-03-18 00:15
- [x] Create project scaffold — completed 2026-03-17 15:00
- [x] Implement `dist_vae/losses.py` — completed 2026-03-17 15:15 — 17 tests pass
- [x] Implement `tests/test_losses.py` — completed 2026-03-17 15:15
- [x] Implement `dist_vae/data.py` — completed 2026-03-17 15:30 — 14 tests pass
- [x] Implement `tests/test_data.py` — completed 2026-03-17 15:30
- [x] Implement `dist_vae/model.py` — completed 2026-03-17 15:45 — 14 tests pass
- [x] Implement `tests/test_model.py` — completed 2026-03-17 15:45
- [x] Implement `dist_vae/train.py` — completed 2026-03-17 15:50
- [x] Implement `dist_vae/eval.py` — completed 2026-03-17 15:55
- [x] Implement all scripts — completed 2026-03-17 16:00
- [x] Create `notebooks/quickstart.ipynb` — completed 2026-03-17 16:00
- [x] End-to-end test: synthetic data → train → encode → eval — completed 2026-03-17 16:00
- [x] Write README.md content — completed 2026-03-17 15:00
- [x] First full synthetic training run (100 epochs) — completed 2026-03-18 00:15
- [x] Add ks_distance_smooth to losses.py — completed 2026-03-18 00:15
- [x] Generate and save synthetic dataset to data/ — completed 2026-03-18 00:15
- [x] Download Norman et al. and create mini dataset — completed 2026-03-18 00:30
- [x] Train 1000 epochs on mini Norman data — completed 2026-03-18 00:45 — posterior collapse diagnosed
