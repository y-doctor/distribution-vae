"""Overlay model confusion onto the reward-similarity heatmap.

Single big figure where each (true_i, pred_j) cell shows:
    - Background color = reward the model would have earned for predicting j
      when truth was i (Pearson on delta-mean profiles, NTC-hinged)
    - Overlay circle = model actually picked j when truth was i (marker size
      proportional to P(pred=j | true=i), ring color = fell above/below hinge)

The question this visualizes: when the model is *wrong*, does it land on a
high-reward neighbor (good mistake) or a low-reward one (bad mistake)?

Usage:
    python scripts/plot_confusion_on_reward.py \
        --adata data/mini_perturb_seq_2kg_allp_ntc.h5ad \
        --eval-dir eval_results/rl_2kg_singles_mlp_pearson_rescale_hinge2x/val_ens10 \
        --singles-only \
        --hinge-multiplier 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch

from dist_vae.losses import pearson_correlation
from dist_vae.rl_data import PerturbationClassificationDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", default="data/mini_perturb_seq_2kg_allp_ntc.h5ad")
    p.add_argument("--eval-dir", required=True,
                   help="Directory with confusion.npy and true_p.npy (val_ensN/).")
    p.add_argument("--out-name", default="confusion_overlaid_on_reward.png")
    p.add_argument("--singles-only", action="store_true")
    p.add_argument("--n-cells", type=int, default=200,
                   help="n_cells for the NTC-noise-baseline subsamples.")
    p.add_argument("--hinge-multiplier", type=float, default=1.0)
    p.add_argument("--baseline-quantile", type=float, default=0.95)
    p.add_argument("--baseline-K", type=int, default=200)
    p.add_argument("--figsize", type=float, default=22.0)
    p.add_argument("--min-confusion", type=float, default=0.02,
                   help="Skip overlay markers below this P(pred|true).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_path = eval_dir / args.out_name

    confusion_path = eval_dir / "confusion.npy"
    if not confusion_path.exists():
        raise SystemExit(f"Missing {confusion_path}")

    print(f"Loading {args.adata} ...")
    adata = ad.read_h5ad(args.adata)
    if args.singles_only:
        keep = ~adata.obs["perturbation"].str.contains("_").fillna(False)
        adata = adata[keep].copy()

    ds = PerturbationClassificationDataset(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        n_cells_per_pert=args.n_cells,
        n_cells_ntc=args.n_cells,
        grid_size=64,
        min_cells=30,
        samples_per_epoch=10,
        seed=42,
        val_fraction=0.0,
    )
    names = list(ds.perturbation_names)
    P = len(names)
    print(f"  {P} perts")

    profiles = ds.compute_delta_mean_profiles()      # (P, G)
    baseline = ds.compute_ntc_noise_baseline(
        profiles,
        n_cells=args.n_cells,
        metric="pearson",
        K=args.baseline_K,
        quantile=args.baseline_quantile,
        seed=42,
    )   # (P,)
    effective_threshold = baseline * args.hinge_multiplier   # (P,)
    print(f"  hinge threshold (x{args.hinge_multiplier}): "
          f"mean {float(effective_threshold.mean()):.3f}, "
          f"range [{float(effective_threshold.min()):.3f}, "
          f"{float(effective_threshold.max()):.3f}]")

    R = pearson_correlation(
        profiles.unsqueeze(1),
        profiles.unsqueeze(0),
        dim=-1,
    ).numpy()                                        # (P, P)

    # Apply hinge (zero-out below threshold[i] for row i).
    thr = effective_threshold.numpy()[:, None]
    R_hinged = np.where(R > thr, R, 0.0)

    # Load confusion.
    C = np.load(confusion_path)
    if C.shape != (P, P):
        raise SystemExit(f"confusion shape {C.shape} != {(P, P)}")
    C_norm = C / C.sum(axis=1, keepdims=True).clip(min=1)    # P(pred | true)

    # Reorder by hierarchical clustering of the raw reward similarity so
    # that bio-equivalent clusters land near the diagonal.
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    D = 1 - R.copy()
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)
    R_sorted = R_hinged[order][:, order]
    C_sorted = C_norm[order][:, order]
    thr_sorted = thr.squeeze()[order]
    labels = [names[i] for i in order]

    # Plot.
    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))

    vmax = float(max(abs(R_sorted.min()), abs(R_sorted.max()), 1e-6))
    im = ax.imshow(R_sorted, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="Pearson r (reward), hinged",
                        fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(P))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted perturbation", fontsize=12)
    ax.set_ylabel("True perturbation", fontsize=12)

    # Overlay confusion markers.
    rows, cols = np.where(C_sorted >= args.min_confusion)
    freqs = C_sorted[rows, cols]
    reward_values = R_sorted[rows, cols]
    # Hinge threshold depends only on the TRUE row.
    thr_per_pair = thr_sorted[rows]

    # Size by frequency. Scale so typical diagonal (freq ~0.2-1.0) is visible.
    sizes = 60 + 360 * freqs ** 0.7

    # Marker color: green edge if prediction lands on a real reward
    # (raw Pearson > hinge threshold), red edge otherwise.
    above_hinge = R[order][:, order][rows, cols] > thr_per_pair
    is_diag = rows == cols
    edge_colors = np.where(
        is_diag, "#000000",
        np.where(above_hinge, "#1a7f3c", "#b00020"),
    )

    ax.scatter(
        cols, rows,
        s=sizes, facecolors="none",
        edgecolors=edge_colors,
        linewidths=1.4,
        zorder=5,
    )

    # Legend (manual, hand-built).
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="#000000",
               markersize=12, label="correct (diagonal)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="#1a7f3c",
               markersize=12, label="wrong pert, reward ABOVE hinge"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="#b00020",
               markersize=12, label="wrong pert, reward BELOW hinge (~zero reward)"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="gray",
               markersize=7, label="circle size ∝ P(pred | true)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9, title="Model prediction overlay",
              title_fontsize=10)

    hinge_note = (f"  [hinge x{args.hinge_multiplier:g}]"
                  if args.hinge_multiplier != 1.0 else "")
    top1 = float(np.diag(C_norm).mean())
    ax.set_title(
        f"Confusion overlaid on reward heatmap{hinge_note}\n"
        f"Pearson of delta-mean profiles as background; "
        f"model predictions as circles (size = frequency, color = reward above/below hinge).  "
        f"Top-1 = {top1:.3f}",
        fontsize=13, loc="left", pad=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
