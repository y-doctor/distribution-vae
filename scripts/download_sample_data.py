"""Download sample Perturb-seq data (Norman et al. 2019) for testing."""

import argparse
from pathlib import Path


def main() -> None:
    """Download and preprocess Norman et al. 2019 Perturb-seq data."""
    parser = argparse.ArgumentParser(description="Download sample Perturb-seq data")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min-cells", type=int, default=30, help="Min cells per perturbation")
    parser.add_argument("--output", type=str, default="data/sample_perturb_seq.h5ad", help="Output path")
    args = parser.parse_args()

    import anndata as ad
    import numpy as np
    import scanpy as sc

    print("Downloading Norman et al. 2019 dataset...")

    # Try loading from scanpy datasets
    try:
        adata = sc.datasets.norman_2019()
        print(f"Loaded via scanpy: {adata.shape[0]} cells x {adata.shape[1]} genes")
    except Exception as e:
        print(f"scanpy download failed: {e}")
        print("Trying alternative source...")
        # Fallback: try scperturb
        try:
            adata = sc.read(
                "https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad",
                backup_url="https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad",
            )
            print(f"Loaded from scperturb: {adata.shape[0]} cells x {adata.shape[1]} genes")
        except Exception as e2:
            raise RuntimeError(
                f"Could not download Norman et al. data from any source.\n"
                f"scanpy error: {e}\nscperturb error: {e2}"
            )

    # Identify perturbation key
    pert_key = None
    for candidate in ["gene", "perturbation", "guide_identity", "condition"]:
        if candidate in adata.obs.columns:
            pert_key = candidate
            break

    if pert_key is None:
        print(f"Available obs columns: {list(adata.obs.columns)}")
        raise ValueError("Could not identify perturbation column in adata.obs")

    print(f"Using perturbation key: '{pert_key}'")

    # Filter perturbations with enough cells
    pert_counts = adata.obs[pert_key].value_counts()
    valid_perts = pert_counts[pert_counts >= args.min_cells].index
    adata = adata[adata.obs[pert_key].isin(valid_perts)].copy()
    print(f"After filtering (>= {args.min_cells} cells): {adata.shape[0]} cells, {len(valid_perts)} perturbations")

    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select HVGs
    if adata.shape[1] > args.n_hvgs:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs)
        adata = adata[:, adata.var["highly_variable"]].copy()

    print(f"Final: {adata.shape[0]} cells x {adata.shape[1]} genes, {adata.obs[pert_key].nunique()} perturbations")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"Saved to {output_path}")

    # Summary
    print(f"\nDataset summary:")
    print(f"  Cells: {adata.shape[0]}")
    print(f"  Genes: {adata.shape[1]}")
    print(f"  Perturbations: {adata.obs[pert_key].nunique()}")
    print(f"  Perturbation key: '{pert_key}'")
    print(f"  Obs columns: {list(adata.obs.columns)}")


if __name__ == "__main__":
    main()
