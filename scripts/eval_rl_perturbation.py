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


def compute_predictions(
    model: PerturbationClassifier,
    dataset: PerturbationClassificationDataset,
    R: int = 20,
    n_ensemble: int = 1,
    device: str = "cpu",
) -> dict:
    """For each (pert, repeat) sample the model and collect logits + predictions.

    Args:
        n_ensemble: if > 1, each "trial" averages the model's logits across
            n_ensemble independent 100-cell subsamples before argmax. This
            is the test-time ensembling lever.

    Returns a dict with keys:
      - confusion: (P, P) int64 confusion counts, argmax predictions.
      - logits:    (P*R, P) float32 ensembled logits per trial.
      - probs:     (P*R, P) softmax probabilities per trial.
      - true_p:    (P*R,) int64 true pert index per trial.
      - pred_p:    (P*R,) int64 argmax prediction per trial.
    """
    model.eval()
    P = len(dataset.perturbation_names)
    gene_ids = dataset.vocab.expression_gene_ids.to(device)
    C = np.zeros((P, P), dtype=np.int64)
    rng = np.random.default_rng(0)

    pert_cells = dataset._pert_cells
    ntc_cells = dataset._ntc_cells
    n_p = dataset.n_cells_per_pert
    n_n = dataset.n_cells_ntc
    from dist_vae.data import samples_to_quantile_grid

    all_logits = np.zeros((P * R, P), dtype=np.float32)
    all_true = np.zeros(P * R, dtype=np.int64)
    all_pred = np.zeros(P * R, dtype=np.int64)

    with torch.no_grad():
        row = 0
        for true_p in range(P):
            pert_mat = pert_cells[true_p]
            for _ in range(R):
                ens_logits = np.zeros(P, dtype=np.float32)
                for _e in range(n_ensemble):
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
                    ).squeeze(0)
                    ens_logits += logits.cpu().numpy()
                ens_logits /= max(n_ensemble, 1)
                pred = int(np.argmax(ens_logits))
                C[true_p, pred] += 1
                all_logits[row] = ens_logits
                all_true[row] = true_p
                all_pred[row] = pred
                row += 1

    probs = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    return {
        "confusion": C,
        "logits": all_logits,
        "probs": probs,
        "true_p": all_true,
        "pred_p": all_pred,
    }


def compute_confusion(
    model: PerturbationClassifier,
    dataset: PerturbationClassificationDataset,
    R: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """Backward-compatible: just the (P, P) confusion matrix."""
    return compute_predictions(model, dataset, R=R, device=device)["confusion"]


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


def _bio_equivalence_clusters(R_sim: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Group perts into clusters via single-linkage on cos-sim > threshold.

    Returns an (P,) array of cluster ids. Perts in the same cluster are
    reward-equivalent up to `threshold`.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    P = R_sim.shape[0]
    D = 1 - R_sim.copy()
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="single")
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return cluster_ids - 1  # 0-indexed


def plot_umap_predictions(
    pred_bundle: dict,
    pert_names: list[str],
    R_sim: np.ndarray,
    out_path: Path,
    seed: int = 0,
) -> None:
    """UMAP of the model's predicted-probability vectors, colored two ways.

    Each point is one trial (one 100-cell subsample of one pert). Its feature
    is the model's (P,)-dim probability distribution over perts. UMAP groups
    trials whose predictions agree — so trials from the same pert should
    cluster, and reward-equivalent perts should merge.
    """
    import umap

    probs = pred_bundle["probs"]
    true_p = pred_bundle["true_p"]
    P = probs.shape[1]

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.05, metric="cosine", random_state=seed
    )
    Z = reducer.fit_transform(probs)

    cluster_ids = _bio_equivalence_clusters(R_sim, threshold=0.9)
    n_clusters = int(cluster_ids.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: color by true pert (50 colors, cycling).
    cmap = mpl.cm.get_cmap("tab20", max(P, 20))
    for p in range(P):
        m = true_p == p
        axes[0].scatter(
            Z[m, 0], Z[m, 1],
            s=18, alpha=0.75, edgecolor="white", linewidth=0.2,
            color=cmap(p % 20),
        )
        # Write the pert label at the centroid.
        if m.sum() > 0:
            cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
            axes[0].text(
                cx, cy, pert_names[p],
                fontsize=6.5, ha="center", va="center", weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.5),
            )
    axes[0].set_title(
        f"UMAP of model-output probability vectors ({P} perts, R={probs.shape[0]//P} trials each)\n"
        "colored by TRUE perturbation — trials from the same pert should cluster",
        loc="left",
    )
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")

    # Panel 2: color by bio-equivalence cluster (reward-sim > 0.9).
    cmap2 = mpl.cm.get_cmap("tab20", max(n_clusters, 10))
    for c in range(n_clusters):
        members = np.where(cluster_ids == c)[0]
        m = np.isin(true_p, members)
        if m.sum() == 0:
            continue
        size = len(members)
        label = (
            ", ".join(pert_names[mi] for mi in members[:3])
            + (f" (+{size - 3})" if size > 3 else "")
        )
        axes[1].scatter(
            Z[m, 0], Z[m, 1],
            s=18, alpha=0.75, edgecolor="white", linewidth=0.2,
            color=cmap2(c % 20),
            label=label if size > 1 else None,
        )
    axes[1].set_title(
        "Same UMAP colored by BIO-EQUIVALENCE CLUSTER (reward cos-sim >= 0.9)\n"
        "within-cluster overlap is expected; between-cluster separation is what we want",
        loc="left",
    )
    axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
    # Only show legend for multi-member clusters.
    axes[1].legend(fontsize=7, loc="upper right", ncol=1, frameon=True)

    fig.suptitle(
        "Model output geometry: where does each perturbation land in prediction space?",
        fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_pert_rewards(
    pred_bundle: dict,
    profiles: torch.Tensor,
    pert_names: list[str],
    out_path: Path,
) -> None:
    """Distribution of per-trial reward for each perturbation, sorted by mean.

    Shows which perts the model is confident about (high mean, tight spread)
    vs. which perts are fuzzy (low mean, wide spread).
    """
    pred_p = pred_bundle["pred_p"]
    true_p = pred_bundle["true_p"]
    P = len(pert_names)
    profiles_np = profiles.numpy()

    # Per-trial reward = cos-sim(profile[pred], profile[true]).
    norms = np.linalg.norm(profiles_np, axis=1, keepdims=True).clip(min=1e-8)
    unit = profiles_np / norms
    rewards = (unit[pred_p] * unit[true_p]).sum(axis=1)  # (P*R,)

    # Group per true pert.
    per_pert_rewards = [rewards[true_p == p] for p in range(P)]
    means = np.array([r.mean() for r in per_pert_rewards])
    order = np.argsort(-means)

    fig, ax = plt.subplots(figsize=(14, 9))
    parts = ax.boxplot(
        [per_pert_rewards[i] for i in order],
        vert=False,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#1f4e79", lw=1.2),
    )
    for patch, idx in zip(parts["boxes"], order):
        # Color boxes by their mean reward (green good, red bad).
        v = means[idx]
        color = plt.cm.RdYlGn((v + 1) / 2)
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.85)

    ax.set_yticks(range(1, P + 1))
    ax.set_yticklabels([pert_names[i] for i in order], fontsize=7)
    ax.axvline(0.9, color="#27ae60", ls="--", lw=1, alpha=0.7, label="bio-equivalent (>=0.9)")
    ax.axvline(0.5, color="#f39c12", ls="--", lw=1, alpha=0.7, label="same broad class (>=0.5)")
    ax.axvline(0.0, color="#c0392b", ls="--", lw=1, alpha=0.7, label="zero (orthogonal)")
    ax.axvline(rewards.mean(), color="#1f4e79", ls="-", lw=1.5, alpha=0.9,
               label=f"overall mean = {rewards.mean():.3f}")
    ax.set_xlabel("per-trial reward  =  cos-sim(profile[pred], profile[true])")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        f"Per-pert reward distribution (R=30 subsamples each). "
        f"{int((rewards >= 0.9).mean() * 100)}% of trials land in a bio-equivalent pert; "
        f"{int((rewards >= 0.5).mean() * 100)}% within broad class.",
        loc="left", fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def report_summary_metrics(
    pred_bundle: dict,
    profiles: torch.Tensor,
    R_sim: np.ndarray,
    pert_names: list[str],
    out_path: Path,
) -> dict:
    """Compute top-k / reward-threshold metrics and write JSON summary."""
    logits = pred_bundle["logits"]
    true_p = pred_bundle["true_p"]
    P = logits.shape[1]

    # Top-k.
    topk = {}
    for k in (1, 3, 5, 10):
        topk_preds = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        hits = np.any(topk_preds == true_p[:, None], axis=1)
        topk[f"top_{k}"] = float(hits.mean())

    # MRR.
    ranks = (logits > logits[np.arange(len(true_p)), true_p][:, None]).sum(axis=1) + 1
    mrr = float(np.mean(1.0 / ranks))

    # Per-trial reward and threshold hits.
    profiles_np = profiles.numpy()
    norms = np.linalg.norm(profiles_np, axis=1, keepdims=True).clip(min=1e-8)
    unit = profiles_np / norms
    rewards = (unit[pred_bundle["pred_p"]] * unit[true_p]).sum(axis=1)
    thr = {}
    for t in (0.5, 0.8, 0.9, 0.95, 0.99):
        thr[f"reward_ge_{t}"] = float((rewards >= t).mean())

    off_diag = R_sim - np.eye(P)
    random_reward = float(off_diag[~np.eye(P, dtype=bool)].mean())

    summary = {
        "top_k": topk,
        "mrr": mrr,
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "reward_std": float(rewards.std()),
        "reward_threshold_rates": thr,
        "random_reward_baseline": random_reward,
        "random_top_1": 1.0 / P,
        "n_trials": len(true_p),
    }
    import json

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument(
        "--n-ensemble", type=int, default=1,
        help="Test-time ensembling: average logits over N fresh subsamples per trial.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.0,
        help="If > 0, eval on the held-out cells (val split) from the training-time config.",
    )
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument(
        "--mode", type=str, default="val", choices=["train", "val"],
        help="With --val-fraction > 0, which split to eval on.",
    )
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
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        mode=args.mode if args.val_fraction > 0 else "train",
    )
    if args.val_fraction > 0:
        print(f"  evaluating on {args.mode} split (val_fraction={args.val_fraction})")
        print(f"  pool sizes — pert0: {dataset._pert_cells[0].shape[0]} (full {dataset._pert_cells_full[0].shape[0]})")
        print(f"               NTC:   {dataset._ntc_cells.shape[0]} (full {dataset._ntc_cells_full.shape[0]})")
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
        n_attn_layers=int(cfg.get("n_attn_layers", 0)),
        n_heads=int(cfg.get("n_heads", 4)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint from epoch {ckpt['epoch']}, "
          f"train-time mean reward = {ckpt['metrics']['mean_reward']:.4f}")

    print(f"Computing predictions (R={args.n_repeats} trials/pert, n_ensemble={args.n_ensemble}) ...")
    pred_bundle = compute_predictions(
        model, dataset, R=args.n_repeats, n_ensemble=args.n_ensemble
    )
    C = pred_bundle["confusion"]

    top1 = np.diag(C / C.sum(axis=1, keepdims=True).clip(min=1)).mean()
    print(f"  held-out greedy top-1 acc: {top1:.4f}")

    print("Plotting training curves ...")
    plot_training_curves(
        Path(args.history), out / "training_curves.png", n_perts=P
    )

    print("Plotting confusion matrix ...")
    plot_confusion(C, dataset.perturbation_names, out / "confusion_matrix.png")

    print("Plotting confusion vs reward similarity ...")
    plot_confusion_vs_reward(
        C, R_sim, dataset.perturbation_names, out / "confusion_vs_reward_sim.png"
    )

    print("Plotting UMAP of prediction-probability vectors ...")
    plot_umap_predictions(
        pred_bundle, dataset.perturbation_names, R_sim, out / "umap_predictions.png"
    )

    print("Plotting per-pert reward distributions ...")
    plot_per_pert_rewards(
        pred_bundle, profiles, dataset.perturbation_names, out / "per_pert_rewards.png"
    )

    print("Computing summary metrics ...")
    summary = report_summary_metrics(
        pred_bundle, profiles, R_sim, dataset.perturbation_names, out / "metrics.json"
    )
    print("== held-out metrics ==")
    for k, v in summary["top_k"].items():
        print(f"  {k}: {v:.3f}")
    print(f"  MRR: {summary['mrr']:.3f}")
    print(f"  mean reward: {summary['mean_reward']:.3f} "
          f"(random baseline {summary['random_reward_baseline']:.3f})")
    for k, v in summary["reward_threshold_rates"].items():
        print(f"  {k}: {v:.3f}")

    np.save(out / "confusion.npy", C)
    np.save(out / "logits.npy", pred_bundle["logits"])
    np.save(out / "true_p.npy", pred_bundle["true_p"])

    print(f"Saved plots and metrics under {out}")


if __name__ == "__main__":
    main()
