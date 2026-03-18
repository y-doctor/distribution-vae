"""CLI script for training the Distribution VAE."""

import argparse
from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides to config dict."""
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Try to parse as int, float, bool, or keep as string
        for parser in [int, float]:
            try:
                value = parser(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "null" or value.lower() == "none":
                value = None
        d[parts[-1]] = value
    return config


def get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Train a Distribution VAE model."""
    parser = argparse.ArgumentParser(description="Train a Distribution VAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--adata", type=str, default=None, help="Path to AnnData .h5ad file")
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Config overrides in dot notation (e.g., model.latent_dim=64)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    device = get_device()
    print(f"Using device: {device}")

    # Seed
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)

    # Dataset
    model_cfg = config.get("model", {})
    grid_size = model_cfg.get("grid_size", 256)

    if args.synthetic:
        from dist_vae.data import SyntheticDistributionDataset
        full_dataset = SyntheticDistributionDataset(
            n_distributions=2000, grid_size=grid_size, seed=seed,
        )
    elif args.adata:
        import anndata as ad
        from dist_vae.data import PerturbationDistributionDataset
        adata = ad.read_h5ad(args.adata)
        data_cfg = config.get("data", {})
        full_dataset = PerturbationDistributionDataset(
            adata=adata,
            perturbation_key=data_cfg.get("perturbation_key", "perturbation"),
            grid_size=grid_size,
            min_cells=data_cfg.get("min_cells", 20),
            gene_subset=data_cfg.get("gene_subset"),
        )
    else:
        raise ValueError("Must specify either --synthetic or --adata")

    # Train/val split
    val_frac = config.get("training", {}).get("val_fraction", 0.1)
    n_val = max(1, int(len(full_dataset) * val_frac))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Dataset: {len(full_dataset)} total, {n_train} train, {n_val} val")

    # Model
    from dist_vae.model import DistributionVAE
    loss_cfg = config.get("loss", {})
    loss_weights = {
        "cramer": loss_cfg.get("cramer", 1.0),
        "wasserstein1": loss_cfg.get("wasserstein1", 0.0),
        "kl_divergence": loss_cfg.get("kl_divergence", 0.0),
    }

    model = DistributionVAE(
        grid_size=grid_size,
        latent_dim=model_cfg.get("latent_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        beta=model_cfg.get("beta", 0.01),
        loss_config=loss_weights,
        free_bits=model_cfg.get("free_bits", 0.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Train
    from dist_vae.train import Trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    history = trainer.train()

    print(f"\nTraining complete. Best val recon loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
