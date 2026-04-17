"""Re-materialize the 2kg x 236-pert Perturb-seq dataset with NTC.

Produces ``data/mini_perturb_seq_2kg_allp_ntc.h5ad`` — the input used by
``configs/rl_perturbation_2kg_allp.yaml`` and
``configs/rl_perturbation_2kg_allp_rownorm.yaml``, and therefore required by
``scripts/eval_rl_perturbation.py`` on the 2kg run.

This file is gitignored (it's ~100 MB after preprocessing), so rerun this
script in any new session that needs to evaluate the 2kg/236p checkpoint.

Usage::

    python scripts/download_2kg_data.py
    # then, with a trained checkpoint:
    python scripts/eval_rl_perturbation.py \
        --adata data/mini_perturb_seq_2kg_allp_ntc.h5ad \
        --checkpoint eval_results/rl_perturbation_2kg_allp_rownorm/checkpoints/best.pt \
        --history eval_results/rl_perturbation_2kg_allp_rownorm/history.json \
        --output-dir eval_results/rl_perturbation_2kg_allp_rownorm/val_ens1/ \
        --val-fraction 0.20 --mode val --n-repeats 20

Matches what ``scripts/download_sample_data.py`` + ``scripts/make_mini_dataset.py``
would produce for (n_hvgs=2000, all perts, min_cells=30, keep controls).
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUTPUT_DEFAULT = "data/mini_perturb_seq_2kg_allp_ntc.h5ad"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DEFAULT,
        help=f"Destination h5ad path (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--n-hvgs", type=int, default=2000,
        help="Number of highly variable genes to keep (default: 2000)",
    )
    parser.add_argument(
        "--min-cells", type=int, default=30,
        help="Minimum cells per perturbation to include (default: 30)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the output already exists.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists. Use --force to rebuild.")
        return

    import anndata as ad
    import numpy as np
    import scanpy as sc

    print("Downloading Norman et al. 2019 via scanpy ...")
    try:
        adata = sc.datasets.norman_2019()
        print(f"  loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    except Exception as e:
        print(f"scanpy download failed: {e}")
        print("Falling back to scperturb Zenodo mirror ...")
        adata = sc.read(
            str(out_path.parent / "NormanWeissman2019_filtered.h5ad"),
            backup_url=(
                "https://zenodo.org/records/7041849/files/"
                "NormanWeissman2019_filtered.h5ad"
            ),
        )
        print(f"  loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    pert_key = None
    for candidate in ("gene", "perturbation", "guide_identity", "condition"):
        if candidate in adata.obs.columns:
            pert_key = candidate
            break
    assert pert_key is not None, (
        f"No perturbation key in {list(adata.obs.columns)}"
    )
    print(f"  perturbation key: '{pert_key}'")

    pert_counts = adata.obs[pert_key].value_counts()
    valid = pert_counts[pert_counts >= args.min_cells].index
    adata = adata[adata.obs[pert_key].isin(valid)].copy()
    n_perts = adata.obs[pert_key].nunique()
    n_ntc = int(
        adata.obs[pert_key].astype(str).str.lower().str.contains("control").sum()
    )
    print(
        f"  keeping {len(valid)} perts (>= {args.min_cells} cells) "
        f"→ {adata.shape[0]} cells, {n_ntc} of them NTC"
    )

    print("Normalize + log1p ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if adata.shape[1] > args.n_hvgs:
        print(f"Selecting top {args.n_hvgs} HVGs ...")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs)
        adata = adata[:, adata.var["highly_variable"]].copy()

    print(
        f"Final: {adata.shape[0]} cells x {adata.shape[1]} genes, "
        f"{adata.obs[pert_key].nunique()} perturbations"
    )

    # Rename the pert-key column to 'perturbation' so it matches the configs
    # (configs/rl_perturbation_2kg_allp*.yaml expects perturbation_key='perturbation').
    if pert_key != "perturbation":
        adata.obs["perturbation"] = adata.obs[pert_key].astype(str)
        print(f"  aliased obs column '{pert_key}' → 'perturbation'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    top = adata.obs["perturbation"].value_counts().head(5)
    print("\nTop-5 pert cell counts:")
    for p, n in top.items():
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main()
