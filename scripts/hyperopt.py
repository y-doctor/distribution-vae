"""CLI script for hyperparameter optimization of the Distribution VAE."""

import argparse
import sys
from pathlib import Path

import torch

# Reuse config utilities from the train script
sys.path.insert(0, str(Path(__file__).parent))
from train import apply_overrides, get_device, load_config


def main() -> None:
    """Run hyperparameter optimization for the Distribution VAE."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Distribution VAE"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to base config YAML file"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data"
    )
    parser.add_argument(
        "--adata", type=str, default=None, help="Path to AnnData .h5ad file"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=30,
        help="Training epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/best_hyperopt.yaml",
        help="Path to save best config (default: configs/best_hyperopt.yaml)",
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///study.db) for resumable search",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
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
            n_distributions=2000, grid_size=grid_size, seed=seed
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
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Dataset: {len(full_dataset)} total, {n_train} train, {n_val} val")
    print(f"Running {args.n_trials} trials, {args.n_epochs} epochs each\n")

    # Run hyperopt
    from dist_vae.hyperopt import best_config_to_yaml, run_hyperopt

    best_config, study = run_hyperopt(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        base_config=config,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        device=device,
        study_name=args.study_name,
        storage=args.storage,
    )

    # Report results
    print(f"\n{'=' * 60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_recon: {study.best_value:.6f}")
    print(f"Best hyperparameters:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    # Save best config
    best_config_to_yaml(best_config, args.output)
    print(f"\nBest config saved to: {args.output}")
    print(f"Train with: python scripts/train.py --config {args.output} --synthetic")


if __name__ == "__main__":
    main()
