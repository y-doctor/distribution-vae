# Autoresearch: Distribution VAE

You are an autonomous AI research agent. Your job is to improve a Variational Autoencoder that encodes 1D empirical distributions (from Perturb-seq single-cell experiments) into fixed-dimensional latent representations. You do this by repeatedly modifying `train.py`, running it, checking if the result improved, and keeping or discarding the change.

**You run experiments forever. NEVER STOP. The human might be asleep.**

## Setup (do this once at the start)

1. Read this file (`program.md`) completely
2. Read `prepare.py` — this is the fixed evaluation infrastructure, do NOT modify it
3. Read `train.py` — this is the file you modify
4. Create a new git branch: `git checkout -b autoresearch/<your_tag>` where `<your_tag>` is a short identifier (e.g., `autoresearch/run-001`)
5. Initialize the results file:
   ```
   echo "commit\tval_kl_divergence\tval_w1\tactive_dims\tepochs\tn_params\tstatus\tdescription" > results.tsv
   ```
6. Run the baseline: `python autoresearch/train.py > run.log 2>&1`
7. Parse the output for the metrics (see "Parsing output" below)
8. Record the baseline in `results.tsv`

## The experiment loop (repeat forever)

```
while True:
    1. Formulate a hypothesis (what change, why it might help)
    2. Edit train.py
    3. Git commit the change: git add autoresearch/train.py && git commit -m "<brief description>"
    4. Run: python autoresearch/train.py > run.log 2>&1   (timeout: 10 minutes)
    5. Parse output for val_kl_divergence and active_dims
    6. Decision:
       - If |val_kl_divergence| IMPROVED (decreased) AND active_dims > 0: KEEP
         → Record in results.tsv with status=keep
       - If |val_kl_divergence| got WORSE or EQUAL: DISCARD
         → git reset --hard HEAD~1
         → Record in results.tsv with status=discard
       - If CRASHED: read the error, attempt a fix or discard
         → git reset --hard HEAD~1
         → Record in results.tsv with status=crash
    7. Go to step 1 immediately — NEVER STOP
```

## What you CAN modify

Only `autoresearch/train.py`. Specifically:

- **Hyperparameters**: LATENT_DIM, HIDDEN_DIM, BETA, LR, WEIGHT_DECAY, GRAD_CLIP, BATCH_SIZE, etc.
- **Model architecture**: Change the encoder (different layer types, depths, widths, activations, normalization). Change the decoder (different upsampling strategies, monotonicity enforcement). Add skip connections, attention, residual blocks, etc.
- **Loss functions**: Modify reconstruction loss (combine Cramer + W1, add KS, use different weights). Modify KL (free bits, cyclical annealing, etc.). Add regularization terms.
- **Optimizer**: Try different optimizers (Adam, SGD+momentum, AdamW with different configs). Try learning rate schedules (cosine, warmup+decay, cyclic).
- **Training loop**: Change the training procedure (gradient accumulation, EMA, etc.).

## What you CANNOT modify

- `prepare.py` — the evaluation function and data pipeline are fixed
- Do NOT add new files or dependencies beyond what's already imported
- Do NOT change the output format — the `--- RESULTS ---` block must be parseable

## Parsing output

After each run, grep the output for:
```
val_kl_divergence=X.XXXXXX     ← PRIMARY METRIC (lower absolute value is better)
val_w1=X.XXXXXX         ← secondary (lower is better)
active_dims=N/M          ← latent utilization (MUST stay > 0, ideally N == M)
epochs=N                 ← how many epochs fit in the time budget
n_params=N               ← model size
```

The **primary metric is val_kl_divergence**. A change is KEPT only if |val_kl_divergence| strictly decreases (closer to 0). KL divergence can be negative due to the quantile-based estimation, so always compare absolute values.

**IMPORTANT**: If active_dims drops to 0, that's posterior collapse — always discard, even if val_kl_divergence looks good.

## Recording results

Append to `results.tsv` (tab-separated):
```
<commit_hash>\t<val_kl_divergence>\t<val_w1>\t<active_dims>\t<epochs>\t<n_params>\t<status>\t<description>
```

Example:
```
a1b2c3d\t0.042000\t0.051000\t16/16\t487\t123456\tbaseline\tBaseline run with default settings
e4f5g6h\t0.031000\t0.048000\t16/16\t487\t123456\tkeep\tIncrease hidden_dim to 192
i7j8k9l\t0.058000\t0.053000\t12/16\t487\t123456\tdiscard\tTry W1 loss instead of Cramer
```

## The metric

**val_kl_divergence** = KL divergence between the original and reconstructed distributions, estimated from their quantile grids. It measures how faithfully the VAE preserves the distributional shape — including density ratios, not just location.

For quantile functions Q, the density is f(Q(p)) ≈ dp / dQ. The KL is computed as mean(log(delta_recon / delta_input)) over quantile spacings.

Lower |val_kl_divergence| = better reconstruction (0 = perfect). The value can be negative.

## Research ideas to try (starting suggestions)

These are starting points — you should generate your own ideas too:

1. **Architecture changes**:
   - Deeper encoder/decoder (more conv layers)
   - Wider encoder (hidden_dim=192 or 256)
   - Add batch normalization or layer normalization
   - Residual connections in encoder/decoder
   - Replace Conv1d with dilated convolutions for longer-range dependencies
   - Attention mechanism in the bottleneck
   - Different activation functions (SiLU/Swish, Mish)

2. **Loss function changes**:
   - Combine Cramer + W1 loss (weighted sum)
   - Add smooth KS loss as a regularizer
   - Spectral loss (FFT-based frequency matching)
   - Tail-weighted Cramer (upweight the edges of the quantile grid)

3. **Training improvements**:
   - Cosine annealing learning rate schedule
   - Learning rate warmup (first few epochs)
   - Gradient accumulation for effective larger batch size
   - Exponential moving average (EMA) of model weights
   - Cyclical KL annealing instead of linear warmup

4. **Regularization**:
   - Dropout in encoder
   - Spectral normalization
   - Total correlation penalty for better disentanglement
   - L2 regularization on latent means

5. **Decoder improvements**:
   - Different monotonicity enforcement (e.g., softmax + cumsum instead of softplus + cumsum)
   - Separate scale and shift parameters
   - Predict quantile spacings directly

## Rules

- **One change at a time**: Make one conceptual change per experiment so you can attribute improvements correctly
- **Simplicity**: All else equal, simpler is better. If removing code maintains performance, that's a win.
- **No posterior collapse**: If active_dims drops to 0, always discard. A good model uses all its latent dimensions.
- **NEVER STOP**: The human might be away. Keep running experiments continuously.
- **Time budget**: Training runs for 5 minutes (300 seconds). Your changes should work within this budget.
- **Parameter budget**: Keep model size reasonable. Massive models that barely train in 5 minutes are not useful.

## Crash recovery

If `train.py` crashes:
1. Read the traceback from `run.log`
2. If it's an obvious syntax/shape error from your edit, attempt to fix it
3. If you can't fix it quickly, discard: `git reset --hard HEAD~1`
4. Record in results.tsv with status=crash
5. Move on to a different idea

## When to STOP (never, but here are soft goals)

You can note milestones:
- |val_kl_divergence| < 0.05: Good reconstruction
- |val_kl_divergence| < 0.01: Excellent — near-perfect density matching
- |val_kl_divergence| < 0.005: Outstanding — diminishing returns territory
- active_dims = M/M consistently: Healthy latent space

Even after hitting these milestones, KEEP GOING. There's always more to find.
