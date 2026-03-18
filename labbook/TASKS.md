# Task Board

## TODO
- [ ] Run hyperopt on synthetic data end-to-end — S
- [ ] End-to-end test: Norman et al. → train → encode → eval — M
- [ ] Tune hyperparameters on real data using hyperopt module — M
- [ ] **Fix posterior collapse on real data** — M — highest priority
  - Try beta=0.001 or 0.0001
  - Reduce latent_dim to 8 or 16
  - Implement free-bits / KL thresholding
- [ ] Hyperparameter sweep: beta, latent_dim, hidden_dim — M
- [ ] Investigate log-transforming quantile grids before model input — S
- [ ] Try W1 loss for better tail differentiation — S
- [ ] Investigate total correlation penalty for better disentanglement — S
- [ ] Add integration tests for training loop — S
- [ ] Profile memory usage on large datasets — S

## IN PROGRESS
(none)

## DONE
- [x] Implement `dist_vae/hyperopt.py` — completed 2026-03-18 00:30 — 14 tests pass
- [x] Implement `scripts/hyperopt.py` CLI — completed 2026-03-18 00:30
- [x] Implement `tests/test_hyperopt.py` — completed 2026-03-18 00:30
- [x] Add epoch_callback to Trainer.train() — completed 2026-03-18 00:30
- [x] Create HTML project report (report.html) — completed 2026-03-18 00:15
- [x] Create project scaffold — completed 2026-03-17 15:00 — entries/2026-03-17_1500_initial_scaffold.md
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
