"""CLI script for training the Distribution VAE."""

import argparse


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
    _args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
