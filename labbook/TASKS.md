# Task Board

## TODO
- [ ] Hyperparameter tuning: try beta=0.02-0.05, latent_dim=16, hidden_dim=256 — M
- [ ] End-to-end test: Norman et al. → train → encode → eval — M
- [ ] Tune hyperparameters on real data — M
- [ ] Add integration tests for training loop — S
- [ ] Profile memory usage on large datasets — S
- [ ] Investigate total correlation penalty for better disentanglement — S

## IN PROGRESS
(none)

## DONE
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
- [x] First full synthetic training run (100 epochs) — completed 2026-03-18 00:15 — entries/2026-03-18_0015_first_synthetic_training_run.md
- [x] Add ks_distance_smooth to losses.py — completed 2026-03-18 00:15
- [x] Generate and save synthetic dataset to data/ — completed 2026-03-18 00:15
