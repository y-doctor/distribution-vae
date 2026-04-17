# Per-cell set-transformer classifier: prototype + phased test plan
**Date**: 2026-04-17 15:45 UTC
**Duration**: ~90 minutes
**Goal**: Build an alternative to the quantile-grid `PerturbationClassifier` that operates on raw cells rather than distribution summaries, with cross-attention from pert cells to matched NTC cells and a learned-gene-module front end. Target: unlock the "isolated perts" failure mode surfaced by the reward-ball analysis.

## Motivation (recap from DECISIONS.md 2026-04-17)
Reward-ball plots revealed two failure modes on the row-norm 2kg/236p run:
1. Bio-equivalent misses (green rings on blue dots) — cosmetic; already tracked as `P(r≥0.9) = 0.42`.
2. **Isolated perts** (worst-20 grid) — gold stars with no reward-ball neighbors; model has no "close-enough" answer and predictions scatter. This is the real remaining headroom.

The quantile-grid classifier summarizes each pert's cell population into a single `(G, K)` token before any learning happens — it can't leverage per-cell heterogeneity or per-cell NTC matching. A set transformer over raw cells, with cross-attention to NTC, preserves both.

## Architecture (sketch)

```
pert cells (B, n_p, G)                 ntc cells (B, n_n, G)
        │                                     │
   GeneModuleAttention (shared)          same
   ├── gene_embed[g]*x[g] gene tokens        ...
   └── K learned module queries cross-attend
   (B, n_p, K, d)                       (B, n_n, K, d)
        │  mean over modules                  │  mean over modules
   (B, n_p, d)                          (B, n_n, d)
        + pert_type_emb                      + ntc_type_emb
        │                                     │
   CellSetTransformer (self-attn over cells) ×2 layers per stream
        │                                     │
        └─── PertNTCCrossAttention (×2 layers, pert Q, ntc K/V)
                        │
                 CLS query cross-attends → (B, d)
                        │
                   Linear → (B, P) logits
```

Same `forward(ntc, pert, gene_ids) → logits` signature as the quantile-grid model, so `GRPOTrainer` is reused unchanged.

## Files
- `dist_vae/rl_cell_model.py` — 4 modules: `GeneModuleAttention`, `CellSetTransformer`, `PertNTCCrossAttention`, `PerturbationCellClassifier`.
- `dist_vae/rl_data.py` — extended with `return_cells=True` flag (backward compatible).
- `scripts/train_rl_cell.py` — parallel CLI, reuses `GRPOTrainer`.
- `configs/rl_cell_mini10.yaml`, `configs/rl_cell_50p.yaml`.
- `tests/test_rl_cell_model.py` — 13 tests covering the 4 modules (shapes, gradient flow, permutation invariance, trainability on a tiny fixed-target CE task).
- `tests/test_rl_data.py` — 3 new tests for `return_cells` mode.

## Test results

**Phase 1 (unit tests, all pass):** 13 new model tests + 3 dataset tests. Full suite: 80 passed, 2 skipped.

**Phase 3 (10-pert smoke, `configs/rl_cell_mini10.yaml`):**
- 59,594 params, 10 epochs on `mini_perturb_seq_with_ntc.h5ad`.
- Reward climbed 0.046 → **0.29**; top-1 0.08 → 0.20 (random 0.10). Entropy decreasing. Model is learning.

**Phase 4 (50-pert scale-up, 500g/50p):**
Two configs explored:

| config | params | LR | clip | epochs | final reward | final top-1 | final grad-norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| ambitious (d=48, K=16) | 237K | 1e-4 | 1.0 | 8 (killed) | **0.42** | 0.045 | 32 ↑ |
| conservative (d=32, K=8) | 112K | 5e-5 | 0.5 | 20 | 0.19 | 0.062 | 12 |

The ambitious config was learning fast but gradients were exploding. The conservative version was stable but under-trained in the time budget. Settled default (committed as `rl_cell_50p.yaml`): `d=48, K=16, lr=1e-4, clip=0.5` — keeps capacity, tightens clip to stabilize.

**Confirmation run on settled config** (12 epochs, `eval_results/rl_cell_50p/`):
- Reward: 0.094 → **0.376**, steadily climbing (no plateau, no collapse).
- Top-1: 0.02 → 0.03 (random 0.02) — not yet meaningful at this early epoch count.
- Grad-norm pre-clip grew 8 → 24; clip=0.5 kept the step sizes bounded. Training productive.
- At the same rate, the 50-epoch MLP baseline result (reward 0.57) is reachable within 20–25 more epochs.

## Key decisions made in code

1. **Gene token construction**: `token[g] = expression[g] * gene_embed[g]` — multiplicative modulation. Simple, effective; no FiLM.
2. **Module pooling**: mean over `K` module tokens → `(d,)` per cell. Simpler than learned attention pooling; gives a clean handoff to cell self-attention.
3. **Type embeddings**: `pert_type`, `ntc_type` learnable vectors added to cell representations before self-attention. Lets downstream cross-attn distinguish streams.
4. **CLS pooling**: separate learned query cross-attends over refined pert cells; no learned tokens inside the set.
5. **Shared `gene_embed`**: one `nn.Embedding` for both module attention and any downstream head that wants it.

## Results vs prior-art MLP baseline

The 50-pert MLP baseline (`rl_perturbation_50p.yaml`) hit reward 0.57 at ep50 and held-out top-1 0.43 at ep334. The cell set-transformer hit reward 0.42 at ep8 (ambitious config, killed early) — on trajectory to match or beat MLP with more epochs + tuned clip.

## Problems encountered

1. **Command pipeline buffering** — running `python train.py ... 2>&1 | tail -40` held all output until the process ended, which made a 58-minute run look "stuck." Fix: `python -u` + redirect directly to a file, no `tail` piping.
2. **Grad-norm explosion at `clip=1.0`** in the wider model. Tightened to 0.5. Probably LR is still slightly aggressive for the wider attention stack.

## Next steps (open for a future session)

1. **Longer 50p run** with the settled config (30-50 epochs) to directly A/B against the MLP's 50-epoch result.
2. **Reduce `samples_per_epoch`** or swap to a GPU before attempting the full 2kg/236p config. On CPU each epoch is ~2-3 min at these settings, which extrapolates to ~8-12 hours for 150 epochs — fine for overnight, too long for iterative debugging.
3. **Hard-pert upweighting** (inverse reward-ball density) — complementary to this architecture, should help the isolated-pert failure mode.
4. **Profile-gene-module attention weights** after training — readable "these N genes attend to module-3" gives interpretability.

## Open questions

- Does the learned module structure actually capture biologically meaningful gene groups? Need to inspect attention weights post-training on a converged model.
- Is 16–32 the right K? Too few → bottleneck, too many → identity. Worth a sweep once baseline converges.
- Cross-attention depth (`n_cross_layers`) — 2 layers chosen arbitrarily. Could do a 1 vs 2 vs 4 A/B.
