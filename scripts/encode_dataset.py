"""CLI script for encoding all distributions in a dataset to latent vectors."""

import argparse


def main() -> None:
    """Encode all distributions in an AnnData file to latent representations."""
    parser = argparse.ArgumentParser(description="Encode distributions to latent vectors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output path for latent AnnData")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    _args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
