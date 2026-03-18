"""Generate and save a synthetic dataset for training/evaluation without re-generating each time."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from dist_vae.data import SyntheticDistributionDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic distribution dataset")
    parser.add_argument("--n-distributions", type=int, default=2000, help="Number of distributions")
    parser.add_argument("--grid-size", type=int, default=256, help="Quantile grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/synthetic_2k.h5ad", help="Output path")
    args = parser.parse_args()

    print(f"Generating {args.n_distributions} synthetic distributions (grid_size={args.grid_size}, seed={args.seed})...")
    dataset = SyntheticDistributionDataset(
        n_distributions=args.n_distributions,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    # Save as AnnData: rows = distributions, columns = quantile grid points
    grids = dataset.grids.numpy()
    gene_indices = np.array([dataset[i][1] for i in range(len(dataset))])
    pert_indices = np.array([dataset[i][2] for i in range(len(dataset))])

    adata = ad.AnnData(
        X=grids,
        obs={
            "gene_idx": gene_indices,
            "pert_idx": pert_indices,
        },
        var={
            "quantile": np.linspace(0, 1, args.grid_size),
        },
    )
    adata.var_names = [f"q_{i:03d}" for i in range(args.grid_size)]
    adata.obs_names = [f"dist_{i:04d}" for i in range(args.n_distributions)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(adata)} distributions to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
