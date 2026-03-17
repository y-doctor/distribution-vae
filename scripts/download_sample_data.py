"""Download sample Perturb-seq data (Norman et al. 2019) for testing."""

import argparse


def main() -> None:
    """Download and preprocess Norman et al. 2019 Perturb-seq data."""
    parser = argparse.ArgumentParser(description="Download sample Perturb-seq data")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min-cells", type=int, default=30, help="Min cells per perturbation")
    parser.add_argument("--output", type=str, default="data/sample_perturb_seq.h5ad", help="Output path")
    _args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
