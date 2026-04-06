"""Plot gene-gene covariance matrices for perturbations in mini Norman dataset.

Row 1: Raw covariance matrices for a control perturbation + 6 others.
Row 2: Difference (perturbation - control) covariance matrices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gene-gene covariance matrices")
    parser.add_argument("--input", type=str, default="data/mini_perturb_seq.h5ad")
    parser.add_argument("--control", type=str, default="BAK1", help="Perturbation to use as control")
    parser.add_argument("--perturbations", nargs="+", default=None,
                        help="Perturbations to show (default: 6 largest after control)")
    parser.add_argument("--output", type=str, default="gene_gene_covariance.png")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    pert_key = "perturbation"

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)

    # Select perturbations
    control = args.control
    if args.perturbations:
        perturbations = args.perturbations
    else:
        pert_counts = adata.obs[pert_key].value_counts()
        perturbations = [p for p in pert_counts.index if p != control][:6]

    all_perts = [control] + perturbations

    # Compute gene-gene covariance per perturbation
    cov_matrices = {}
    cell_counts = {}
    for p in all_perts:
        mask = (adata.obs[pert_key] == p).values
        cell_counts[p] = int(mask.sum())
        cov_matrices[p] = np.cov(X[mask], rowvar=False)

    # Difference matrices
    diff_matrices = {p: cov_matrices[p] - cov_matrices[control] for p in perturbations}

    # Color scales (95th percentile of absolute values)
    vmax_raw = np.percentile(np.abs(np.concatenate([cov_matrices[p].ravel() for p in all_perts])), 95)
    vmax_diff = np.percentile(np.abs(np.concatenate([diff_matrices[p].ravel() for p in perturbations])), 95)

    n_cols = len(all_perts)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Row 1: raw covariance
    for i, p in enumerate(all_perts):
        ax = axes[0, i]
        im1 = ax.imshow(cov_matrices[p], cmap="RdBu_r", vmin=-vmax_raw, vmax=vmax_raw, aspect="equal")
        ax.set_title(f"{p}\n({cell_counts[p]} cells)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Gene-gene covariance", fontsize=11)

    fig.colorbar(im1, ax=axes[0, :].tolist(), shrink=0.6, label="Covariance")

    # Row 2: control slot is blank, then differences
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, f"(control:\n{control})", ha="center", va="center",
                    fontsize=10, style="italic", transform=axes[1, 0].transAxes)

    for i, p in enumerate(perturbations):
        ax = axes[1, i + 1]
        im2 = ax.imshow(diff_matrices[p], cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff, aspect="equal")
        ax.set_title(f"{p} − {control}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f"Δ Covariance\n(pert − {control})", fontsize=11)

    fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.6, label="ΔCovariance")

    fig.suptitle(
        f"Gene-gene covariance matrices: Norman mini dataset ({adata.n_vars} genes)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
