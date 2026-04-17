"""Visualize the Pearson + NTC-hinge reward landscape for a few perturbations.

For each selected true perturbation i, plots the reward the model would get
if it predicted any other pert j in the vocabulary — i.e. the full reward
surface that GRPO is navigating. The raw Pearson curve, the hinged curve, and
the per-pert NTC noise-baseline are shown together so the dynamics are legible.

Perts are sampled along signal-strength quantiles so the panel spans
"easy / medium / hard" from a single run.

Usage:
    python scripts/viz_reward_landscape.py \
        --adata data/mini_perturb_seq_2kg_allp_ntc.h5ad \
        --singles-only \
        --output-dir eval_results/reward_landscape \
        --n-cells 200
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
    p.add_argument("--output-dir", default="eval_results/reward_landscape")
    p.add_argument("--singles-only", action="store_true")
    p.add_argument("--n-cells", type=int, default=200,
                   help="n_cells for the NTC-noise-baseline subsamples.")
    p.add_argument("--baseline-K", type=int, default=200)
    p.add_argument("--baseline-quantile", type=float, default=0.95)
    p.add_argument("--n-perts-to-plot", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_perts_by_signal(
    profiles: torch.Tensor, names: list[str], n: int
) -> list[int]:
    """Pick n perts spread across signal-norm quantiles: strongest, strong,
    median, weak-ish, weak, weakest. Falls back to quantile indices when n != 6.
    """
    signal = profiles.norm(dim=1).numpy()          # (P,)
    order = np.argsort(signal)[::-1]               # descending
    P = len(order)

    # Spread n picks across the ranked list: include rank 0 (strongest) and
    # rank P-1 (weakest) explicitly, plus n-2 quantile stripes in between.
    if n <= 1:
        return [int(order[0])]
    if n == 2:
        return [int(order[0]), int(order[-1])]

    picks = [0]
    interior_q = np.linspace(0.05, 0.95, n - 2)
    for q in interior_q:
        picks.append(int(round(q * (P - 1))))
    picks.append(P - 1)
    # Dedup while preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for rk in picks:
        idx = int(order[rk])
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def plot_landscape_panel(
    reward_matrix: torch.Tensor,       # (P, P) raw Pearson
    baseline: torch.Tensor,            # (P,)
    names: list[str],
    selected_idx: list[int],
    out_path: Path,
    signal_norm: np.ndarray,
    top_k_annotate: int = 5,
) -> None:
    """2×3 (or similar) grid; each subplot = reward curve for one true pert."""
    n = len(selected_idx)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False,
    )
    P = reward_matrix.shape[0]
    signal_q = np.argsort(np.argsort(-signal_norm))  # descending-rank per pert

    for ax_idx, pert_i in enumerate(selected_idx):
        ax = axes[ax_idx // ncols][ax_idx % ncols]

        rewards = reward_matrix[pert_i].numpy().copy()   # (P,)
        thr = float(baseline[pert_i])
        # Sort descending so the self-reward (1.0) sits at x=0.
        order = np.argsort(-rewards)
        sorted_rewards = rewards[order]
        sorted_names = [names[j] for j in order]
        hinged = np.where(sorted_rewards > thr, sorted_rewards, 0.0)

        xs = np.arange(P)
        # Area: hinged (effective) reward.
        ax.fill_between(xs, 0.0, hinged, color="tab:orange", alpha=0.55,
                        label="effective reward (hinged)")
        # Raw Pearson curve.
        ax.plot(xs, sorted_rewards, color="black", lw=1.0,
                label="raw Pearson")
        # Hinge baseline.
        ax.axhline(thr, color="tab:red", lw=1.0, ls="--",
                   label=f"baseline = {thr:.3f}")
        # Zero line.
        ax.axhline(0.0, color="gray", lw=0.6)

        # Annotate top-k neighbors (excluding self at rank 0).
        for rk in range(1, min(top_k_annotate + 1, P)):
            if sorted_rewards[rk] < 0.05:
                break
            ax.annotate(
                sorted_names[rk],
                xy=(rk, sorted_rewards[rk]),
                xytext=(rk + 3, sorted_rewards[rk] + 0.03),
                fontsize=7,
                color="tab:blue",
                arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.5),
            )
        # Mark self at rank 0.
        ax.scatter([0], [sorted_rewards[0]], color="tab:green",
                   s=30, zorder=5, label="true pert (self)")

        # How many perts clear the hinge?
        n_above = int((rewards > thr).sum()) - 1  # exclude self
        n_strong = int((rewards > 0.5).sum()) - 1  # heuristic "strong neighbor"
        sig = signal_norm[pert_i]
        rank = int(signal_q[pert_i]) + 1
        ax.set_title(
            f"{names[pert_i]}   "
            f"(signal ||Δ||={sig:.2f}, rank {rank}/{P})\n"
            f"{n_above} perts clear hinge, {n_strong} with r>0.5",
            fontsize=10,
        )
        ax.set_xlabel("prediction rank (0 = highest-reward pert)")
        ax.set_ylabel("reward")
        ax.set_ylim(-0.3, 1.05)
        ax.set_xlim(-2, P + 2)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # Hide unused cells.
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(
        "Reward landscape: what the model could earn for each possible prediction\n"
        "(Pearson of delta-mean profiles, NTC-noise hinge at 95th pctile)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    reward_matrix: torch.Tensor,
    baseline: torch.Tensor,
    signal_norm: np.ndarray,
    out_path: Path,
) -> None:
    """Overview figure: baseline distribution, signal-norm distribution,
    and a reward-threshold curve (how many pairs clear each τ)."""
    P = reward_matrix.shape[0]
    off_diag = reward_matrix[~torch.eye(P, dtype=torch.bool)].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(baseline.numpy(), bins=30, color="tab:red", alpha=0.75)
    ax.set_title(
        f"NTC-noise baseline per pert\n"
        f"(mean {float(baseline.mean()):.3f}, "
        f"range [{float(baseline.min()):.3f}, {float(baseline.max()):.3f}])"
    )
    ax.set_xlabel("baseline Pearson (95th pctile of NTC null)")
    ax.set_ylabel("perts")

    ax = axes[1]
    ax.hist(off_diag, bins=60, color="tab:blue", alpha=0.75)
    ax.axvline(float(baseline.mean()), color="tab:red", ls="--",
               label=f"mean baseline = {float(baseline.mean()):.3f}")
    ax.set_title(
        f"Off-diagonal pairwise Pearson ({len(off_diag):,} pairs)\n"
        f"mean {off_diag.mean():.3f}, p95 {np.quantile(off_diag, 0.95):.3f}, "
        f"p99 {np.quantile(off_diag, 0.99):.3f}"
    )
    ax.set_xlabel("Pearson between two perts' Δ profiles")
    ax.set_ylabel("pairs")
    ax.legend()

    ax = axes[2]
    taus = np.linspace(-0.1, 1.0, 100)
    frac_above = np.array([(off_diag > t).mean() for t in taus])
    ax.plot(taus, frac_above, color="black")
    ax.axvline(float(baseline.mean()), color="tab:red", ls="--",
               label=f"mean baseline")
    ax.axvline(0.5, color="gray", ls=":", label="τ = 0.5")
    ax.axvline(0.9, color="gray", ls=":", label="τ = 0.9")
    ax.set_title("Fraction of off-diag pairs above threshold τ")
    ax.set_xlabel("τ")
    ax.set_ylabel("P(r > τ)")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.adata} ...")
    adata = ad.read_h5ad(args.adata)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if args.singles_only:
        keep = ~adata.obs["perturbation"].str.contains("_").fillna(False)
        # keep also the control label (no underscore) — stays by default
        adata = adata[keep].copy()
        print(f"  singles-only filter: {adata.n_obs:,} cells remain")

    ds = PerturbationClassificationDataset(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        n_cells_per_pert=args.n_cells,
        n_cells_ntc=args.n_cells,
        grid_size=64,
        min_cells=30,
        samples_per_epoch=10,
        seed=args.seed,
        val_fraction=0.0,   # oracle uses the full cell pool
    )
    names = list(ds.perturbation_names)
    P = len(names)
    print(f"  {P} perturbations after min_cells filter")

    print("Computing delta-mean profiles ...")
    profiles = ds.compute_delta_mean_profiles()      # (P, G)
    signal = profiles.norm(dim=1).numpy()

    print(
        f"Computing NTC-noise baseline "
        f"(n_cells={args.n_cells}, K={args.baseline_K}, "
        f"q={args.baseline_quantile}) ..."
    )
    baseline = ds.compute_ntc_noise_baseline(
        profiles,
        n_cells=args.n_cells,
        metric="pearson",
        K=args.baseline_K,
        quantile=args.baseline_quantile,
        seed=args.seed,
    )
    print(f"  baseline: mean {float(baseline.mean()):.3f}, "
          f"range [{float(baseline.min()):.3f}, {float(baseline.max()):.3f}]")

    print("Computing (P, P) pairwise Pearson reward matrix ...")
    R = pearson_correlation(
        profiles.unsqueeze(1),
        profiles.unsqueeze(0),
        dim=-1,
    )
    # torch pearson between a vector and itself is 1.0 up to float error.
    # Don't zero the diagonal — we want to show self at reward 1.0.

    selected = select_perts_by_signal(profiles, names, args.n_perts_to_plot)
    print(f"Selected perts: {[names[i] for i in selected]}")

    panel_path = out_dir / "reward_landscape_panel.png"
    summary_path = out_dir / "reward_summary.png"

    plot_landscape_panel(
        reward_matrix=R,
        baseline=baseline,
        names=names,
        selected_idx=selected,
        out_path=panel_path,
        signal_norm=signal,
    )
    print(f"Saved {panel_path}")

    plot_summary(
        reward_matrix=R,
        baseline=baseline,
        signal_norm=signal,
        out_path=summary_path,
    )
    print(f"Saved {summary_path}")

    # Save the raw tables for later analysis too.
    np.savez(
        out_dir / "reward_landscape.npz",
        reward_matrix=R.numpy(),
        baseline=baseline.numpy(),
        signal_norm=signal,
        names=np.array(names),
        selected=np.array(selected),
    )
    print(f"Saved {out_dir / 'reward_landscape.npz'}")


if __name__ == "__main__":
    main()
