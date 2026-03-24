# Task Board

## TODO
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

## RECENTLY ADDED
- [ ] Run autoresearch agent on GPU for extended experiment session — M
- [ ] Analyze autoresearch results and integrate best findings into main library — M

## DONE
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
