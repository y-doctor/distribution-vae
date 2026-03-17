"""CLI script for running evaluation and generating plots."""

import argparse


def main() -> None:
    """Run evaluation on a trained model and generate report."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Distribution VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for evaluation")
    _args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
