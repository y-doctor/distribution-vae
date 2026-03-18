# distribution-vae

A PyTorch library for encoding arbitrary-sized 1D empirical distributions into fixed-dimensional latent representations via a Variational Autoencoder (VAE).

**Primary use case:** In Perturb-seq experiments, each (gene, perturbation) pair yields a variable-length distribution of expression values across cells. This library compresses each distribution into a compact latent vector suitable for downstream ML — perturbation classification, regression, clustering, and more.

## Core Insight

**Sorting is all you need.** A sorted empirical sample is the quantile function (inverse CDF) — a canonical, permutation-invariant, order-preserving representation of any 1D distribution. Interpolate it to a fixed grid, and you have a fixed-size input for a standard encoder-decoder.

```
Raw samples (variable size N)
    → sort
    → interpolate onto fixed quantile grid (size K, e.g. 256)
    → encoder → latent z
    → decoder → reconstructed quantile grid (size K)
    → loss: distributional distance between input and output grids
```

The decoder enforces **monotonicity** (quantile functions are non-decreasing) by predicting a start value + `softplus(deltas)`, then cumsumming.

## Architecture

```
Input: variable-length 1D samples
         │
         ▼
    ┌─────────┐
    │  Sort +  │
    │ Interp.  │  → fixed quantile grid (batch, grid_size)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Encoder  │  1D CNN: Conv1d → GELU → stride downsample
    │  (CNN)   │  → AdaptiveAvgPool → Linear
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  Latent  │  μ, log σ² → reparameterize → z
    │  Space   │  (batch, latent_dim)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Decoder  │  Linear → ConvTranspose1d → interpolate
    │  (CNN)   │  → monotonicity: start + cumsum(softplus(Δ))
    └────┬────┘
         │
         ▼
    Reconstructed quantile grid (batch, grid_size)
```

## Quick Start

```bash
# Install
git clone https://github.com/<user>/distribution-vae.git
cd distribution-vae
pip install -e ".[dev]"

# Train on synthetic data (no download needed)
python scripts/train.py --config configs/default.yaml --synthetic

# Download sample Perturb-seq data (Norman et al. 2019)
python scripts/download_sample_data.py

# Train on real data
python scripts/train.py --config configs/example_perturb_seq.yaml \
    --adata data/sample_perturb_seq.h5ad

# Encode dataset → latent matrix
python scripts/encode_dataset.py \
    --checkpoint checkpoints/best.pt \
    --adata data/sample_perturb_seq.h5ad \
    --output latents.h5ad

# Evaluate + generate plots
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --adata data/sample_perturb_seq.h5ad \
    --output-dir eval_results/

# Run tests
pytest tests/ -v
```

## Usage with Custom Data

Any AnnData object works. The key requirement is an `.obs` column identifying perturbations:

```python
import anndata as ad
from dist_vae.data import PerturbationDistributionDataset
from dist_vae.model import DistributionVAE

# Load your data
adata = ad.read_h5ad("your_data.h5ad")

# Create dataset
dataset = PerturbationDistributionDataset(
    adata=adata,
    perturbation_key="perturbation",  # column in .obs
    grid_size=256,
    min_cells=20,
)

# Create and train model
model = DistributionVAE(
    grid_size=256,
    latent_dim=32,
    hidden_dim=128,
    beta=0.01,
)
```

## Configuration

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `grid_size` | 256 | Quantile grid resolution |
| `model` | `latent_dim` | 32 | Latent space dimensionality |
| `model` | `hidden_dim` | 128 | Hidden layer width |
| `model` | `beta` | 0.01 | KL divergence weight |
| `model` | `beta_warmup_epochs` | 10 | Epochs to linearly ramp beta |
| `data` | `perturbation_key` | "perturbation" | AnnData .obs column |
| `data` | `min_cells` | 20 | Min cells per distribution |
| `training` | `epochs` | 100 | Training epochs |
| `training` | `batch_size` | 256 | Batch size |
| `training` | `lr` | 1e-3 | Learning rate |
| `training` | `weight_decay` | 1e-4 | AdamW weight decay |
| `training` | `grad_clip` | 1.0 | Gradient norm clipping |
| `loss` | `cramer` | 1.0 | Cramer distance weight |
| `loss` | `wasserstein1` | 0.0 | Wasserstein-1 weight |
| `loss` | `ks_smooth` | 0.0 | Smooth KS distance weight |

## Module Overview

| Module | Purpose | Dependencies |
|--------|---------|-------------|
| `dist_vae/losses.py` | Distributional loss functions | torch only |
| `dist_vae/model.py` | Encoder, decoder, VAE | losses.py |
| `dist_vae/data.py` | Dataset classes, quantile grid utilities | anndata, scipy |
| `dist_vae/train.py` | Training loop | model, data, losses |
| `dist_vae/eval.py` | Evaluation metrics and plotting | model, data, matplotlib |

## Citation

If you use this library in your research, please cite:

```bibtex
@software{distribution_vae,
  title={distribution-vae: VAE for 1D empirical distributions},
  url={https://github.com/<user>/distribution-vae},
  year={2026},
}
```

## License

MIT
