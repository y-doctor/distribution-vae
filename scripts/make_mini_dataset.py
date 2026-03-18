"""Create a mini Perturb-seq dataset for fast iteration.

Subsets the full dataset to a small number of genes and perturbations,
keeping all cells for those perturbations so distribution shapes are preserved.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mini Perturb-seq dataset")
    parser.add_argument("--input", type=str, default="data/sample_perturb_seq.h5ad")
    parser.add_argument("--output", type=str, default="data/mini_perturb_seq.h5ad")
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-perts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import anndata as ad
    import numpy as np
    import scanpy as sc

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    print(f"Full dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

    rng = np.random.default_rng(args.seed)

    # Identify perturbation key (same logic as download script)
    pert_key = None
    for candidate in ["gene", "perturbation", "guide_identity", "condition"]:
        if candidate in adata.obs.columns:
            pert_key = candidate
            break
    assert pert_key is not None, f"No perturbation key found in {list(adata.obs.columns)}"

    # Pick perturbations with the most cells (more representative distributions)
    pert_counts = adata.obs[pert_key].value_counts()
    # Exclude control-like perturbations if present
    non_control = pert_counts[~pert_counts.index.str.lower().str.contains("control")]
    if len(non_control) >= args.n_perts:
        pert_counts = non_control
    top_perts = pert_counts.head(args.n_perts * 3).index.tolist()
    selected_perts = list(rng.choice(top_perts, size=min(args.n_perts, len(top_perts)), replace=False))

    # Subset cells to selected perturbations
    adata = adata[adata.obs[pert_key].isin(selected_perts)].copy()

    # Pick genes with highest variance (most interesting distributions)
    if adata.shape[1] > args.n_genes:
        gene_vars = np.array(adata.X.toarray().var(axis=0)).flatten() if hasattr(adata.X, 'toarray') else np.var(adata.X, axis=0)
        top_gene_idx = np.argsort(gene_vars)[-args.n_genes:]
        adata = adata[:, top_gene_idx].copy()

    print(f"\nMini dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Perturbations ({adata.obs[pert_key].nunique()}): {sorted(adata.obs[pert_key].unique())}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"Saved to {output_path}")

    # Summary stats
    print(f"\nPer-perturbation cell counts:")
    for p in sorted(selected_perts):
        n = int((adata.obs[pert_key] == p).sum())
        print(f"  {p}: {n} cells")


if __name__ == "__main__":
    main()
