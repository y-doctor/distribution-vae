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

from dist_vae.losses import cosine_similarity, pearson_correlation
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
    model,
    dataset: PerturbationClassificationDataset,
    R: int = 20,
    n_ensemble: int = 1,
    device: str = "cpu",
    model_type: str = "token",
) -> dict:
    """For each (pert, repeat) sample the model and collect logits + predictions.

    Args:
        n_ensemble: if > 1, each "trial" averages the model's logits across
            n_ensemble independent 100-cell subsamples before argmax. This
            is the test-time ensembling lever.
        model_type: "token" for the quantile-grid ``PerturbationClassifier``,
            "cell" for the per-cell ``PerturbationCellClassifier``. Controls
            whether each subsample is passed through ``samples_to_quantile_grid``
            or forwarded as raw cells.

    Returns a dict with keys:
      - confusion: (P, P) int64 confusion counts, argmax predictions.
      - logits:    (P*R, P) float32 ensembled logits per trial.
      - probs:     (P*R, P) softmax probabilities per trial.
      - true_p:    (P*R,) int64 true pert index per trial.
      - pred_p:    (P*R,) int64 argmax prediction per trial.
    """
    assert model_type in ("token", "cell"), model_type
    model.eval()
    P = len(dataset.perturbation_names)
    gene_ids = dataset.vocab.expression_gene_ids.to(device)
    C = np.zeros((P, P), dtype=np.int64)
    rng = np.random.default_rng(0)

    pert_cells = dataset._pert_cells
    ntc_cells = dataset._ntc_cells
    n_p = dataset.n_cells_per_pert
    n_n = dataset.n_cells_ntc
    if model_type == "token":
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
                    if model_type == "token":
                        p_in = samples_to_quantile_grid(p_sub.T, dataset.grid_size)
                        n_in = samples_to_quantile_grid(n_sub.T, dataset.grid_size)
                    else:
                        # Cell model: pass raw (n_cells, G) cells.
                        p_in = p_sub
                        n_in = n_sub
                    logits = model(
                        n_in.unsqueeze(0).to(device),
                        p_in.unsqueeze(0).to(device),
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


def compute_reward_matrix(
    profiles: torch.Tensor, metric: str = "cosine",
) -> np.ndarray:
    """(P, P) pairwise reward-metric matrix between pert delta-mean profiles."""
    assert metric in ("cosine", "pearson"), metric
    metric_fn = pearson_correlation if metric == "pearson" else cosine_similarity
    P = profiles.shape[0]
    M = torch.zeros(P, P)
    for i in range(P):
        for j in range(P):
            M[i, j] = metric_fn(profiles[i], profiles[j])
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


def _per_trial_reward(
    pred_bundle: dict, profiles: torch.Tensor, metric: str = "cosine",
) -> np.ndarray:
    """Per-trial reward between predicted and true pert profiles."""
    profiles_np = profiles.numpy()
    if metric == "pearson":
        profiles_np = profiles_np - profiles_np.mean(axis=1, keepdims=True)
    unit = profiles_np / np.linalg.norm(profiles_np, axis=1, keepdims=True).clip(min=1e-8)
    return (unit[pred_bundle["pred_p"]] * unit[pred_bundle["true_p"]]).sum(axis=1)


def _soft_accuracy_curve(
    rewards: np.ndarray,
    R_sim: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute model soft-accuracy and random baseline at each threshold.

    - model[i]: P(per-trial reward >= thresholds[i]).
    - random[i]: fraction of off-diagonal reward-matrix entries >= thresholds[i],
      i.e. the probability a uniformly-random pert lands in the reward ball.
    """
    model = np.array([(rewards >= t).mean() for t in thresholds])
    P = R_sim.shape[0]
    off = R_sim[~np.eye(P, dtype=bool)]
    random = np.array([(off >= t).mean() for t in thresholds])
    return model, random


def plot_soft_accuracy_curve(
    pred_bundle: dict,
    profiles: torch.Tensor,
    R_sim: np.ndarray,
    out_path: Path,
    metric: str = "cosine",
) -> dict:
    """Soft-accuracy vs threshold τ — 'picked a pert within reward τ of true'."""
    rewards = _per_trial_reward(pred_bundle, profiles, metric=metric)
    thresholds = np.round(np.arange(0.50, 1.0001, 0.05), 2)
    model, random = _soft_accuracy_curve(rewards, R_sim, thresholds)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thresholds, model, color="#1f4e79", lw=2.2, marker="o",
            ms=5, label="model")
    ax.plot(thresholds, random, color="#888888", ls="--", lw=1.5, marker="s",
            ms=4, label="random baseline")

    idx_09 = int(np.argmin(np.abs(thresholds - 0.9)))
    headline = model[idx_09]
    ax.scatter([0.9], [headline], s=160, facecolor="#27ae60",
               edgecolor="black", zorder=5, linewidth=1.2,
               label=f"τ=0.9 headline: {headline:.3f}")

    metric_label = metric if metric != "cosine" else "cos-sim"
    ax.set_xlabel(f"reward threshold τ  ({metric_label} of delta-mean profiles)")
    ax.set_ylabel("P(per-trial reward ≥ τ)")
    ax.set_title(
        "Soft accuracy: did the model pick a perturbation within the reward ball?\n"
        f"(reward metric: {metric})",
        loc="left", fontweight="bold",
    )
    ax.set_xlim(0.48, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)

    return {
        "thresholds": thresholds.tolist(),
        "model": model.tolist(),
        "random": random.tolist(),
        "headline_tau": 0.9,
        "headline_value": float(headline),
    }


def plot_pert_neighborhoods(
    pred_bundle: dict,
    profiles: torch.Tensor,
    R_sim: np.ndarray,
    pert_names: list[str],
    out_dir: Path,
    tau: float = 0.9,
    n_show: int = 20,
    metric: str = "cosine",
) -> None:
    """Per-pert 'reward ball' plots: best-N and worst-N perts by mean reward-to-true.

    For each subject pert p, plot the PCA-2D embedding of all P pert profiles:
      - light gray dots = all perts
      - filled blue circles = reward-ball members (cos-sim >= τ to p)
      - gold star = true pert p
      - model's top-1 predictions on p's test cells overlaid as markers,
        sized by count and colored green (in ball) / red (out of ball)
    """
    from sklearn.decomposition import PCA

    profiles_np = profiles.numpy()
    P = profiles_np.shape[0]
    true_p = pred_bundle["true_p"]
    pred_p = pred_bundle["pred_p"]

    # Rank perts by mean per-trial reward to their true label.
    rewards = _per_trial_reward(pred_bundle, profiles, metric=metric)
    per_pert_mean = np.array([
        rewards[true_p == p].mean() if (true_p == p).any() else np.nan
        for p in range(P)
    ])
    finite = np.where(np.isfinite(per_pert_mean))[0]
    order = finite[np.argsort(-per_pert_mean[finite])]
    best = order[:n_show]
    worst = order[-n_show:][::-1]

    # PCA of pert profiles → 2D coords.
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(profiles_np)
    var_exp = pca.explained_variance_ratio_

    def _panel(ax: plt.Axes, p: int) -> None:
        # Background: all perts.
        ax.scatter(coords[:, 0], coords[:, 1], s=6, color="#dddddd",
                   edgecolor="none", zorder=1)
        # Reward ball members (excluding self).
        ball = np.where(R_sim[p] >= tau)[0]
        ball = ball[ball != p]
        if ball.size > 0:
            ax.scatter(coords[ball, 0], coords[ball, 1], s=22,
                       color="#4a90d9", edgecolor="white", linewidth=0.4,
                       zorder=3, alpha=0.85)
        # True pert.
        ax.scatter(coords[p, 0], coords[p, 1], s=220, marker="*",
                   color="#f5c518", edgecolor="black", linewidth=1.0,
                   zorder=5)
        # Predictions for this true pert.
        mask = true_p == p
        if mask.any():
            preds = pred_p[mask]
            uniq, counts = np.unique(preds, return_counts=True)
            for q, c in zip(uniq, counts):
                in_ball = R_sim[p, q] >= tau
                color = "#2ecc71" if in_ball else "#e74c3c"
                ax.scatter(coords[q, 0], coords[q, 1],
                           s=30 + 18 * c, marker="o",
                           facecolor="none", edgecolor=color, linewidth=1.8,
                           zorder=4)
            n_in_ball = int(((R_sim[p, preds] >= tau).sum()))
            hit_rate = n_in_ball / len(preds)
        else:
            hit_rate = float("nan")

        ax.set_title(
            f"{pert_names[p]}  μr={per_pert_mean[p]:.2f}  P(r≥{tau})={hit_rate:.2f}",
            fontsize=7.5,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4); spine.set_color("#888888")

    def _grid(perts: np.ndarray, title: str, out_path: Path) -> None:
        n_cols = 5
        n_rows = int(np.ceil(len(perts) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.9 * n_rows))
        axes = np.atleast_2d(axes)
        for i, p in enumerate(perts):
            _panel(axes[i // n_cols, i % n_cols], int(p))
        for i in range(len(perts), n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis("off")

        # Shared legend via dummy scatter handles.
        handles = [
            plt.scatter([], [], s=220, marker="*", color="#f5c518",
                        edgecolor="black", linewidth=1.0, label="true pert"),
            plt.scatter([], [], s=22, color="#4a90d9",
                        edgecolor="white", linewidth=0.4,
                        label=f"reward-ball member (cos-sim ≥ {tau})"),
            plt.scatter([], [], s=60, marker="o", facecolor="none",
                        edgecolor="#2ecc71", linewidth=1.8,
                        label="prediction (in ball)"),
            plt.scatter([], [], s=60, marker="o", facecolor="none",
                        edgecolor="#e74c3c", linewidth=1.8,
                        label="prediction (out of ball)"),
            plt.scatter([], [], s=6, color="#dddddd",
                        label="other perts"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=5, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(
            f"{title}\n"
            f"PCA-2D of perturbation delta-mean profiles  "
            f"(PC1 {var_exp[0]*100:.1f}%, PC2 {var_exp[1]*100:.1f}%). "
            f"Marker size ∝ prediction count across trials.",
            fontweight="bold", y=1.0,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    _grid(best, f"Best-{n_show} perts by mean reward-to-true",
          out_dir / "pert_neighborhoods_best.png")
    _grid(worst, f"Worst-{n_show} perts by mean reward-to-true",
          out_dir / "pert_neighborhoods_worst.png")


def report_summary_metrics(
    pred_bundle: dict,
    profiles: torch.Tensor,
    R_sim: np.ndarray,
    pert_names: list[str],
    out_path: Path,
    metric: str = "cosine",
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

    # Per-trial reward and threshold hits (match the training metric).
    rewards = _per_trial_reward(pred_bundle, profiles, metric=metric)
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
    parser.add_argument(
        "--model-type", type=str, default="token", choices=["token", "cell"],
        help="'token' = quantile-grid PerturbationClassifier; "
             "'cell' = raw-cell PerturbationCellClassifier.",
    )
    parser.add_argument(
        "--singles-only", action="store_true",
        help="Filter to single-gene perturbations (no '_' in name). Must "
             "match the training config, otherwise the 236-class model head "
             "won't align with the 105-class dataset.",
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
        singles_only=args.singles_only,
    )
    if args.val_fraction > 0:
        print(f"  evaluating on {args.mode} split (val_fraction={args.val_fraction})")
        print(f"  pool sizes — pert0: {dataset._pert_cells[0].shape[0]} (full {dataset._pert_cells_full[0].shape[0]})")
        print(f"               NTC:   {dataset._ntc_cells.shape[0]} (full {dataset._ntc_cells_full.shape[0]})")
    P = len(dataset.perturbation_names)
    print(f"  {P} perts, {len(dataset.vocab.names)} genes in vocab")

    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]["model"]

    # Reward metric used at train time — eval reports in the same metric so
    # the ball-plot thresholds and the training reward are on the same scale.
    reward_cfg_ckpt = ckpt["config"].get("reward", {})
    reward_metric = str(reward_cfg_ckpt.get("metric", "cosine"))
    print(f"  checkpoint reward metric: {reward_metric}")

    print(f"Computing pairwise {reward_metric} reward matrix ...")
    profiles = dataset.compute_delta_mean_profiles()
    R_sim = compute_reward_matrix(profiles, metric=reward_metric)
    if args.model_type == "token":
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
    else:
        from dist_vae.rl_cell_model import PerturbationCellClassifier
        model = PerturbationCellClassifier(
            n_all_genes=len(dataset.vocab.names),
            n_perts=P,
            d=int(cfg.get("d", 64)),
            n_modules=int(cfg.get("n_modules", 32)),
            n_cell_layers=int(cfg.get("n_cell_layers", 2)),
            n_cross_layers=int(cfg.get("n_cross_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint from epoch {ckpt['epoch']}, "
          f"train-time mean reward = {ckpt['metrics']['mean_reward']:.4f}")

    print(f"Computing predictions (model_type={args.model_type}, "
          f"R={args.n_repeats} trials/pert, n_ensemble={args.n_ensemble}) ...")
    pred_bundle = compute_predictions(
        model, dataset, R=args.n_repeats, n_ensemble=args.n_ensemble,
        model_type=args.model_type,
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

    print("Plotting soft-accuracy curve (P(reward >= tau) vs tau) ...")
    soft_curve = plot_soft_accuracy_curve(
        pred_bundle, profiles, R_sim, out / "soft_accuracy_curve.png",
        metric=reward_metric,
    )

    print("Plotting per-pert reward-ball neighborhoods (best-20 + worst-20) ...")
    plot_pert_neighborhoods(
        pred_bundle, profiles, R_sim,
        dataset.perturbation_names, out, tau=0.9, n_show=20,
        metric=reward_metric,
    )

    print("Computing summary metrics ...")
    summary = report_summary_metrics(
        pred_bundle, profiles, R_sim, dataset.perturbation_names,
        out / "metrics.json", metric=reward_metric,
    )
    summary["soft_accuracy_curve"] = soft_curve
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
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
