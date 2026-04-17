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

## [2026-03-18] Fix posterior collapse: beta=0.0001 + latent_dim=16
**Context**: Model collapsed to single template on real Perturb-seq data (KL=2.0, latent range [-0.1, 0.04]) with beta=0.01 and latent_dim=32.
**Decision**: Use beta=0.0001 and latent_dim=16 for real data. Also implemented free-bits as a configurable fallback.
**Alternatives considered**:
- beta=0.001 (KL=8.0, Cramer=0.016 — better but not enough latent usage)
- beta=0.001 + free_bits=0.5 (KL=19.1, Cramer=0.013 — good but not as clean as just lowering beta)
- beta=0.0001 + latent_dim=32 (KL=40.9, Cramer=0.010 — good but 3 dims still underused)
**Rationale**: beta=0.0001 with d=16 achieves best Cramer (0.0092), all 16 dims fully active (mean std=0.72), and the most compact representation. For 1000 distributions, 16 dims is sufficient — d=32 has more dims than needed and some go unused. The smaller latent forces each dimension to carry more information.

## [2026-03-18] Free-bits implementation
**Context**: Needed a mechanism to prevent posterior collapse without relying solely on beta tuning.
**Decision**: Implement per-dimension KL floor (free-bits): each latent dim must contribute >= `free_bits` nats before KL penalty applies. Default 0.0 preserves existing behavior.
**Alternatives considered**: Cyclical annealing, delta-VAE, skip connections, aggressive decoder
**Rationale**: Free-bits is simple, well-understood (Kingma et al. 2016), and compatible with any beta. Doubles KL usage at beta=0.001 (8.0 -> 19.1). Good fallback when beta can't be lowered further.

## [2026-04-16] GRPO perturbation classifier on K=64 tokens with delta-mean-cos-sim reward
**Context**: Want to validate that the K=64 quantile-grid tokens carry enough signal to support a downstream prediction task — given a (NTC, perturbed) phenotype pair, identify the perturbation. RL chosen over supervised CE because it opens the door to non-differentiable rewards, partial-credit signals, and larger action spaces later.
**Decision**: Use GRPO with full group enumeration (G=10 since we only have 10 classes), entropy bonus 0.05, Adam lr 3e-4. Reward = cosine_similarity(delta_mean_expression[pred], delta_mean_expression[true]) where delta_mean[p] = mean(X_pert) - mean(X_NTC) per gene. Model: per-gene MLP over (ntc_token, pert_token, pert-ntc, gene_embed), mean+max pool, 2-layer head to 10 classes.
**Alternatives considered**:
- Supervised cross-entropy: works (100% train acc) but doesn't generalize to the "partial credit" desideratum.
- Embedding-dot classification head (originally planned): underfit at 10 classes because of shared-gradient interference between the input path and the classifier path; dropped in favor of a simple 2-layer Linear head. The `pert_embeddings()` method is preserved on the model for later re-enablement.
- G=4 sampling: more noise for no benefit on a 10-class problem; G=10 is strictly superior here.
**Rationale**: Full-enumeration GRPO gives exact group advantages. Delta-mean cos-sim gives partial credit for reward-equivalent perts (ETS2/MAPK1 are cos-sim 0.99 — effectively the same pert by this reward). The architecture's per-gene `delta_token = pert - ntc` input channel is what unlocks learning: without it both supervised and RL training stall at random. Final 50-epoch results: mean reward 0.78, top-1 acc 50% (vs 10% random); top-1 ceiling is capped by reward degeneracies. See `labbook/entries/2026-04-16_2115_rl_perturbation_classifier_kickoff.md`, `eval_results/rl_perturbation/training_curves.png`, `dist_vae/rl_model.py`.

## [2026-04-16] Default quantile-grid size K=64 for direct-use tokenization
**Context**: Open question — can we just use the quantile grid directly as a per-(gene, pert) embedding, skipping the VAE? And what is the right K?
**Decision**: K=64 is the recommended default for downstream tokenization when n_cells per (gene, pert) >= ~100. The raw K=64 quantile-grid vector is a standalone embedding that does not require VAE encoding for most uses.
**Alternatives considered**:
- K=256 (current library default `grid_size`): near-lossless but 4x wider than needed.
- K=16: misses zero-inflated tail detail; captures only ~80% of attainable W1 reduction.
- Keeping the VAE as the only supported path: unjustified given the reconstruction fidelity of the raw grid.
**Rationale**: Over 300 (gene, pert) distributions from the mini Norman dataset, K=64 captures 97% of the attainable W1 loss reduction (vs. K=1024) and 91% of Cramer-RMSE reduction; median per-dist W1 = 0.006, p99 = 0.019. Separately, jitter analysis shows sampling noise scales n_cells^-0.5: at n=100 the SNR (signal/jitter) is ~5.5, at n=500 it is ~15. The VAE remains useful for (a) denoising tokens at n_cells < 50, (b) ~4x further compactness (d=16 latent vs 64-dim grid), and (c) generative sampling. But it is not required for a faithful embedding. See `eval_results/quantile_tokenization/panel_K64.png`, `jitter_K64.png`, and `labbook/entries/2026-04-16_1800_k64_tokenization_findings.md`.

## [2026-03-24] Autoresearch framework (autonomous AI experimentation)
**Context**: Manual hyperparameter/architecture tuning is slow. Want autonomous AI agent to experiment 24/7.
**Decision**: Create `autoresearch/` directory following Karpathy's autoresearch pattern: single mutable file (train.py), fixed evaluation infrastructure (prepare.py), agent instructions (program.md), git-based experiment tracking.
**Alternatives considered**: Extend existing hyperopt module (too constrained — only tunes hyperparameters, not architecture), custom experiment framework (over-engineering)
**Rationale**: The autoresearch pattern is simple and proven. Single-file modification keeps diffs reviewable. Fixed time budget (5 min) makes experiments comparable. Git commits as experiment tracker — keep on improvement, reset on failure. Agent can modify anything: architecture, losses, optimizer, training loop.

## [2026-04-17] Linear-rescale hinge: r_eff = relu((r - θ) / (1 - θ))
**Context**: The Pearson + NTC-baseline hinge used a binary shape: zero below θ, raw Pearson above. This has a step discontinuity at θ (barely-above-floor predictions get ~θ reward for "free") and squishes the usable reward range into [θ, 1], compressing GRPO advantages.
**Decision**: Add a `hinge_rescale` config flag. When true, map above-threshold rewards linearly to [0, 1]: `r_eff = relu((r − θ) / (1 − θ))`. Default remains False (legacy-preserving). Also add `hinge_multiplier` to scale θ (default 1.0).
**Alternatives considered**:
- **Keep binary hinge**: simplest but the step at θ is a known GRPO failure mode; observed empirical loss in training (ensembling didn't help, top-1 plateaued early).
- **Exponential (r^p, p > 1)** above θ: winner-take-all; would zero out ~20% of perts (BCL2L11, MAP4K3) whose ceilings sit around r=0.4 and harm their group gradients.
- **Concave (sqrt) above θ**: "credit for any progress"; potentially good for weak-signal perts but not the failure mode we see (mediocre-plateau).
- **Stricter threshold (2× NTC baseline) + binary**: risk of dead groups on weak perts without addressing the discontinuity issue. Deferred as a follow-up ablation with rescale enabled.
**Rationale**: Standard RL practice is linear rewards, whitened per batch (RLHF, PPO). Under GRPO the rescaling is nonlinear (not an affine transform) because the threshold moves θ→0, changing relative spacing within groups. On 2kg/singles, 150-ep A/B vs binary hinge: held-out top-1 0.075→0.091 (ens=1, +21%), 0.072→0.101 (ens=10, +40%), P(r ≥ 0.9) 0.083→0.112 (+35%), MRR 0.161→0.180. Ensembling now helps (was flat before). Loose metrics (mean reward, P(r ≥ 0.5)) are flat — the win is concentrated in high-confidence correctness. See `labbook/entries/2026-04-17_2048_linear_rescale_hinge_results.md`.

## [2026-04-17] Per-cell set-transformer classifier with learned gene modules
**Context**: The quantile-grid classifier (`rl_model.PerturbationClassifier`) collapses each pert's cell population into a single K=64 quantile summary per gene before doing any learning. This throws away (a) per-cell heterogeneity (bimodal / subpopulation responses) and (b) per-cell matching to specific NTC baselines. The ball-plot analysis shows the remaining headroom lives in "isolated perts" — perts with no bio-equivalent neighbors and noisy signal — where population-level summaries may hide the signal.
**Decision**: Build a parallel classifier that operates on raw cells:
1. Each cell's (G,) expression vector is mapped through cross-attention from K learned "module queries" into K gene tokens `(g_emb * expr)`. Output per cell: (K, d) module activations. Discovered modules are soft, learned, interpretable.
2. Pool modules to (d,) per cell and add a pert-type vs ntc-type embedding.
3. Self-attend cells within each stream (pert, ntc).
4. Cross-attend: pert-cell queries attend to NTC-cell key/values, preserving per-cell NTC-baseline matching (not a global summary subtraction).
5. Pool via a learned CLS query and classify to (P,) logits.
**Alternatives considered**:
- **Profile-space prediction head** (predict (G,) profile, score via cos-sim to pert-profile table): rejected by director — predicting a 2k expression vector is harder than needed, and the end use is "pick a pert to make," which wants a categorical head.
- **Gene-target attention bias** (use `pert_target_gene_ids` as attention prior toward the perturbed gene): rejected — the model would learn to cue off "is the target gene down?" instead of the downstream transcriptomic cascade we care about. Information-leak.
- **Strictly additive hybrid**: keep quantile-grid path, bolt on the cell path, concat before head. Safer but harder to attribute wins. Deferred.
- **Simple linear 2000→128 projection before set transformer**: the learned-modules variant replaces this with parameter-efficient, interpretable attention at ~similar cost.
**Rationale**: A set transformer preserves subpopulation structure that quantile grids smooth away. Cross-attention to NTC gives per-cell baseline matching that global NTC subtraction can't. Learned gene modules replace brittle linear dim-reduction with soft, interpretable clustering. Expected to help most on the "isolated perts" failure mode identified by the reward-ball analysis (`eval_results/rl_perturbation_2kg_allp_rownorm/val_ens10/pert_neighborhoods_worst.png`).
**Files**: `dist_vae/rl_cell_model.py`, `dist_vae/rl_cell_data.py`, `configs/rl_cell_*.yaml`, `scripts/train_rl_cell.py`, `tests/test_rl_cell_model.py`, `tests/test_rl_cell_data.py`. GRPOTrainer is reused as-is — the new (dataset, model) pair match its existing interface.
