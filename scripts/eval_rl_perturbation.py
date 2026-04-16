"""Evaluate the GRPO-trained 50-pert classifier and build publication plots.

Produces:
- training_curves_300s_1k.png  : mean reward, top-1 acc, entropy, pg_loss over epochs
- confusion_matrix.png          : (true_pert, pred_pert) confusion heatmap at argmax
- confusion_vs_reward_sim.png   : two-panel — reward-similarity matrix vs confusion
                                  matrix, side-by-side, reordered so degenerate perts
                                  cluster together

The predictions are computed at argmax (greedy policy) on fresh random
100-cell subsamples, R=20 repeats per pert, so each row of the confusion
matrix integrates over sampling noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from dist_vae.losses import cosine_similarity
from dist_vae.rl_data import PerturbationClassificationDataset
from dist_vae.rl_model import PerturbationClassifier


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def plot_training_curves(history_path: Path, out_path: Path, n_perts: int) -> None:
    h = json.load(open(history_path))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    x = list(range(1, len(h["mean_reward"]) + 1))

    axes[0, 0].plot(x, h["mean_reward"], color="#d35400", lw=1.8)
    axes[0, 0].axhline(0.17, color="#888888", ls="--", lw=1,
                       label="random (off-diag mean cos-sim = 0.17)")
    axes[0, 0].set_title("Mean reward (cos-sim)")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(x, h["top1_acc"], color="#1f4e79", lw=1.8)
    axes[0, 1].axhline(1.0 / n_perts, color="#888888", ls="--", lw=1,
                       label=f"random ({1.0/n_perts:.2f})")
    axes[0, 1].set_title("Top-1 accuracy")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(x, h["entropy"], color="#27ae60", lw=1.8)
    axes[1, 0].axhline(np.log(n_perts), color="#888888", ls="--", lw=1,
                       label=f"log({n_perts}) = {np.log(n_perts):.2f}")
    axes[1, 0].set_title("Policy entropy")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("nats")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(x, h["pg_loss"], color="#8e44ad", lw=1.8)
    axes[1, 1].set_title("Policy-gradient loss")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("loss")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(
        f"GRPO perturbation classifier (500 HVGs x {n_perts} perts, 300 samples/ep)",
        fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_confusion(
    model: PerturbationClassifier,
    dataset: PerturbationClassificationDataset,
    R: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """Return (P, P) confusion matrix of greedy predictions."""
    model.eval()
    P = len(dataset.perturbation_names)
    gene_ids = dataset.vocab.expression_gene_ids.to(device)
    C = np.zeros((P, P), dtype=np.int64)
    rng = np.random.default_rng(0)

    # Re-implement a deterministic per-pert sampler so we don't mutate the
    # global dataset RNG.
    pert_cells = dataset._pert_cells
    ntc_cells = dataset._ntc_cells
    n_p = dataset.n_cells_per_pert
    n_n = dataset.n_cells_ntc
    from dist_vae.data import samples_to_quantile_grid

    with torch.no_grad():
        for true_p in range(P):
            pert_mat = pert_cells[true_p]
            for _ in range(R):
                pi = rng.choice(pert_mat.shape[0], size=min(n_p, pert_mat.shape[0]), replace=False)
                ni = rng.choice(ntc_cells.shape[0], size=min(n_n, ntc_cells.shape[0]), replace=False)
                p_sub = torch.from_numpy(pert_mat[pi]).float()
                n_sub = torch.from_numpy(ntc_cells[ni]).float()
                p_tok = samples_to_quantile_grid(p_sub.T, dataset.grid_size)
                n_tok = samples_to_quantile_grid(n_sub.T, dataset.grid_size)
                logits = model(
                    n_tok.unsqueeze(0).to(device),
                    p_tok.unsqueeze(0).to(device),
                    gene_ids,
                )
                pred = int(logits.argmax(dim=-1).item())
                C[true_p, pred] += 1

    return C


def compute_reward_matrix(profiles: torch.Tensor) -> np.ndarray:
    P = profiles.shape[0]
    M = torch.zeros(P, P)
    for i in range(P):
        for j in range(P):
            M[i, j] = cosine_similarity(profiles[i], profiles[j])
    return M.numpy()


def plot_confusion(
    C: np.ndarray,
    pert_names: list[str],
    out_path: Path,
) -> None:
    """Single-panel confusion heatmap (row-normalized probabilities)."""
    P = C.shape[0]
    C_norm = C / C.sum(axis=1, keepdims=True).clip(min=1)
    # Re-order so the diagonal is most prominent: sort by argmax if not already.
    order = np.arange(P)
    C_sorted = C_norm[order][:, order]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(C_sorted, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(P))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels([pert_names[i] for i in order], rotation=90, fontsize=7)
    ax.set_yticklabels([pert_names[i] for i in order], fontsize=7)
    ax.set_xlabel("Predicted perturbation")
    ax.set_ylabel("True perturbation")
    top1 = np.diag(C_norm).mean()
    ax.set_title(
        f"Confusion matrix (greedy argmax, R=20 subsamples/pert)\n"
        f"Diagonal = P(pred = true).  Mean diagonal = {top1:.3f}",
        loc="left",
    )
    fig.colorbar(im, ax=ax, label="P(predicted | true)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_vs_reward(
    C: np.ndarray,
    R_sim: np.ndarray,
    pert_names: list[str],
    out_path: Path,
) -> None:
    """Side-by-side reward-similarity matrix and confusion matrix.

    Perturbations are reordered by single-linkage hierarchical clustering of
    the reward-similarity matrix so that near-degenerate pairs cluster
    together and the two heatmaps can be compared directly.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    P = C.shape[0]
    # Convert cos-sim to distance, then cluster
    D = 1 - R_sim.copy()
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)

    R_sorted = R_sim[order][:, order]
    C_norm = C / C.sum(axis=1, keepdims=True).clip(min=1)
    C_sorted = C_norm[order][:, order]
    labels = [pert_names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))

    im0 = axes[0].imshow(R_sorted, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(P)); axes[0].set_yticks(np.arange(P))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[0].set_yticklabels(labels, fontsize=6)
    axes[0].set_title(
        "Reward similarity:  cos-sim of delta-mean profiles\n"
        "(structure the RL reward sees)", loc="left")
    fig.colorbar(im0, ax=axes[0], label="cos-sim", fraction=0.04)

    im1 = axes[1].imshow(C_sorted, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(np.arange(P)); axes[1].set_yticks(np.arange(P))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_yticklabels(labels, fontsize=6)
    top1 = np.diag(C_norm).mean()
    axes[1].set_title(
        f"Model confusion matrix  (top-1 acc = {top1:.3f})\n"
        "P(predicted | true) — same pert ordering as left", loc="left")
    fig.colorbar(im1, ax=axes[1], label="P(pred | true)", fraction=0.04)

    fig.suptitle(
        "Reward degeneracy (left) explains where the classifier mistakes cluster (right)",
        fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cells", type=int, default=100)
    args = parser.parse_args()

    _set_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.adata} ...")
    adata = anndata.read_h5ad(args.adata)
    dataset = PerturbationClassificationDataset(
        adata,
        n_cells_per_pert=args.n_cells,
        n_cells_ntc=args.n_cells,
        grid_size=args.grid_size,
        samples_per_epoch=32,
    )
    P = len(dataset.perturbation_names)
    print(f"  {P} perts, {len(dataset.vocab.names)} genes in vocab")

    print("Computing delta-mean reward-similarity matrix ...")
    profiles = dataset.compute_delta_mean_profiles()
    R_sim = compute_reward_matrix(profiles)

    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]["model"]
    model = PerturbationClassifier(
        n_all_genes=len(dataset.vocab.names),
        n_perts=P,
        pert_target_gene_ids=dataset.vocab.pert_target_gene_ids,
        grid_size=int(cfg.get("grid_size", 64)),
        d_embed=int(cfg.get("d_embed", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        d_feat=int(cfg.get("d_feat", 64)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint from epoch {ckpt['epoch']}, "
          f"train-time mean reward = {ckpt['metrics']['mean_reward']:.4f}")

    print(f"Computing confusion matrix (R={args.n_repeats} subsamples/pert) ...")
    C = compute_confusion(model, dataset, R=args.n_repeats)

    top1 = np.diag(C / C.sum(axis=1, keepdims=True).clip(min=1)).mean()
    print(f"  held-out greedy top-1 acc: {top1:.4f}")

    # Top-5 accuracy
    top5 = 0
    for true_p in range(P):
        row = C[true_p].copy()
        # Top-5 predictions: sum of 5 largest entries
        top5_preds = np.argsort(-row)[:5]
        top5 += (true_p in top5_preds) * row.sum()
        # Alternative: count fraction of R trials where true_p is in the top-5
        # predictions. We only have argmax predictions here, so we approximate.
    # Simpler: we have only argmax predictions, report only top-1.

    print("Plotting training curves ...")
    plot_training_curves(
        Path(args.history), out / "training_curves_300s_1k.png", n_perts=P
    )

    print("Plotting confusion matrix ...")
    plot_confusion(C, dataset.perturbation_names, out / "confusion_matrix.png")

    print("Plotting confusion vs reward similarity ...")
    plot_confusion_vs_reward(
        C, R_sim, dataset.perturbation_names, out / "confusion_vs_reward_sim.png"
    )

    # Save raw confusion matrix
    np.save(out / "confusion.npy", C)

    print(f"Saved plots under {out}")


if __name__ == "__main__":
    main()
