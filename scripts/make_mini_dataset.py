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
    parser.add_argument(
        "--keep-controls",
        action="store_true",
        help="Keep non-targeting-control (NTC) cells. If set, the control "
        "perturbation is guaranteed to be included in the output file.",
    )
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

    if args.keep_controls:
        # Find the most-abundant control perturbation and guarantee its inclusion.
        control_mask = pert_counts.index.str.lower().str.contains("control")
        control_perts = pert_counts[control_mask]
        if len(control_perts) == 0:
            print("WARNING: --keep-controls set but no 'control' perturbation found.")
            non_control = pert_counts
        else:
            top_control = control_perts.index[0]
            print(
                f"Including control perturbation '{top_control}' "
                f"({int(control_perts.iloc[0])} cells)."
            )
            non_control = pert_counts[~control_mask]
    else:
        # Exclude control-like perturbations if present (back-compat).
        non_control = pert_counts[~pert_counts.index.str.lower().str.contains("control")]

    if len(non_control) >= args.n_perts:
        pert_counts_for_picking = non_control
    else:
        pert_counts_for_picking = pert_counts
    top_perts = pert_counts_for_picking.head(args.n_perts * 3).index.tolist()
    selected_perts = list(
        rng.choice(
            top_perts,
            size=min(args.n_perts, len(top_perts)),
            replace=False,
        )
    )

    if args.keep_controls and len(control_perts) > 0 and top_control not in selected_perts:
        selected_perts.append(top_control)
        print(f"Appended control '{top_control}' to selected perturbations.")

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
