# Architectural Decisions

## [2026-03-17] Quantile function representation
**Context**: Need a fixed-size representation of variable-length 1D distributions
**Decision**: Sort samples → interpolate to fixed grid. This IS the quantile function.
**Alternatives considered**: Deep Sets, set transformers, kernel mean embeddings, histogram binning
**Rationale**: Simplest possible approach. Sorting is O(N log N), interpolation is O(K). No learned parameters in the preprocessing. Permutation invariant by construction. The quantile function is a complete sufficient statistic for the distribution.

## [2026-03-17] Monotonic decoder via cumsum(softplus(deltas))
**Context**: Decoder output must be a valid quantile function (non-decreasing)
**Decision**: Predict start value + positive increments via softplus, then cumsum
**Alternatives considered**: Sorting the output (non-differentiable wrt ordering), penalty loss for violations, monotonic neural networks
**Rationale**: Hard constraint is better than soft penalty. Cumsum+softplus is simple, differentiable, and guarantees validity. No extra hyperparameters.

## [2026-03-17] Cramer distance as default loss
**Context**: Need a differentiable distributional distance for reconstruction loss
**Decision**: Cramer distance (L2 between quantile functions) as primary; W1 (L1) and smooth-KS as options
**Alternatives considered**: KL divergence (requires density estimation), MMD (kernel choice), sliced Wasserstein
**Rationale**: On quantile grids, Cramer = MSE and W1 = MAE — trivially simple, fast, fully differentiable. No kernel or bandwidth choices. Smooth KS gives a max-deviation signal complementary to the average-deviation losses.

## [2026-03-18] Optuna for hyperparameter optimization
**Context**: Need automated hyperparameter tuning for the VAE
**Decision**: Use Optuna with MedianPruner, optimizing val_recon loss. Add epoch_callback to Trainer rather than subclassing.
**Alternatives considered**: Ray Tune (heavier dependency), custom grid search (less flexible), subclassing Trainer (code duplication)
**Rationale**: Optuna is the standard for PyTorch HPO — lightweight, supports pruning, persistent storage, and distributed search. The epoch_callback approach is minimal and backwards-compatible (default None). val_recon is the right metric because total loss includes beta-scaled KL which varies across trials.

## [2026-03-17] Module boundaries for parallel development
**Context**: Multiple agents will develop this simultaneously
**Decision**: losses.py is pure functions (no state), model.py depends only on losses, data.py is independent, train/eval depend on everything
**Rationale**: Minimizes coupling. Two agents can work on losses and data simultaneously with zero conflict.

## [2026-03-18] Smooth KS distance implementation
**Context**: eval.py imported `ks_distance_smooth` from losses.py but it was never implemented
**Decision**: Implement as logsumexp(temperature * |x-y|) / temperature — a differentiable soft-max approximation of the KS statistic
**Alternatives considered**: True max (non-differentiable), p-norm approximation
**Rationale**: logsumexp is a standard smooth-max, temperature=100 gives a close approximation. Simple, differentiable, no extra dependencies.

## [2026-03-18] Save synthetic dataset as h5ad in data/
**Context**: Re-generating synthetic data every run wastes time and makes results harder to compare
**Decision**: Generate once, save to `data/synthetic_2k.h5ad` (2.1 MB), commit to repo
**Alternatives considered**: .pt file (less portable), .npz (no metadata), regenerate each time
**Rationale**: h5ad matches the real data format (AnnData), small enough to commit, enables consistent benchmarking

## [2026-03-18] Commit mini Norman dataset to repo
**Context**: Every new session needs real Perturb-seq data to test with, but downloading 700 MB from Zenodo is slow and flaky
**Decision**: Create `data/mini_perturb_seq.h5ad` (4.6 MB): 9452 cells x 100 top-variance genes x 10 perturbations from Norman 2019, preprocessed (normalize_total + log1p). Commit to repo.
**Alternatives considered**: Download each session (slow, unreliable), Git LFS (extra setup), full 500-gene dataset (too large to commit)
**Rationale**: 4.6 MB is small enough to commit directly. 10 perturbations and 100 genes give 1000 distributions — enough for meaningful training/eval. Any session can immediately use `--adata data/mini_perturb_seq.h5ad`.
