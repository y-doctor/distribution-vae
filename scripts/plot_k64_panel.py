"""Publication-quality panel figure justifying K=64 as the quantile-grid token size.

Produces a single 2x3 figure (PNG + PDF) summarizing:
  (a) The tokenization: raw samples -> sorted -> K=64 grid (one distribution).
  (b) Original vs reconstructed histograms for three representative shapes.
  (c) Empirical vs K=64 quantile functions for the same three shapes.
  (d) Mean reconstruction loss (W1, Cramer-RMSE) vs K, log-log with IQR.
  (e) Normalized loss-reduction fraction vs K with 90/95/99% knees.
  (f) Per-distribution W1 at K=64 across the full dataset (quality dispersion).

Run:
    python scripts/plot_k64_panel.py \
        --adata data/mini_perturb_seq.h5ad \
        --output-dir eval_results/quantile_tokenization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import gridspec

from dist_vae.data import samples_to_quantile_grid


# --- Style -------------------------------------------------------------------

ORIG = "#1f4e79"  # deep blue
TOKEN = "#d35400"  # warm orange
ACCENT = "#2c3e50"  # slate for reference lines
W1_C = "#d35400"
CR_C = "#6c3483"


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
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


# --- Helpers -----------------------------------------------------------------


def _load_matrix(adata: anndata.AnnData) -> np.ndarray:
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X)


def _tokenize(samples: np.ndarray, K: int) -> np.ndarray:
    return (
        samples_to_quantile_grid(
            torch.tensor(samples, dtype=torch.float32), K
        )
        .numpy()
        .astype(np.float64)
    )


def _reconstruction_losses_on_ref(
    samples: np.ndarray, K: int, M: int = 4096
) -> tuple[float, float]:
    """Return (W1, Cramer-RMSE) between empirical and K-point token on M-point grid."""
    sorted_s = np.sort(samples).astype(np.float64)
    ref_q = np.linspace(0.0, 1.0, M)
    emp_on_ref = np.interp(ref_q, np.linspace(0.0, 1.0, len(sorted_s)), sorted_s)

    token = _tokenize(samples, K)
    tok_on_ref = np.interp(ref_q, np.linspace(0.0, 1.0, K), token)

    w1 = float(np.mean(np.abs(emp_on_ref - tok_on_ref)))
    cramer_rmse = float(np.sqrt(np.mean((emp_on_ref - tok_on_ref) ** 2)))
    return w1, cramer_rmse


def _sample_from_grid(grid: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    q = np.linspace(0.0, 1.0, len(grid))
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, q, grid)


def _collect_distributions(
    adata: anndata.AnnData,
    perturbation_key: str,
    min_cells: int,
    max_dists: int,
) -> list[tuple[str, str, np.ndarray]]:
    X = _load_matrix(adata)
    perts_col = adata.obs[perturbation_key].values
    counts = adata.obs[perturbation_key].value_counts()
    valid = counts[counts >= min_cells].index.tolist()
    out: list[tuple[str, str, np.ndarray]] = []
    for p in valid:
        mask = perts_col == p
        X_p = X[mask]
        for g in range(X_p.shape[1]):
            vals = X_p[:, g]
            if vals.std() < 1e-6:
                continue
            out.append((adata.var_names[g], p, vals.copy()))
            if len(out) >= max_dists:
                return out
    return out


def _pick_representative_three(
    dists: list[tuple[str, str, np.ndarray]], seed: int = 0
) -> list[tuple[str, str, np.ndarray]]:
    """Pick three distributions spanning a range of shapes: peaky, zero-inflated, heavy."""
    rng = np.random.default_rng(seed)
    # Score each by (zero_frac, std) and bucket.
    scored = []
    for g, p, s in dists:
        zf = float((s == 0).mean())
        sd = float(s.std())
        scored.append((g, p, s, zf, sd))
    # Bucket 1: high zero frac (> 0.7), mid variance
    b1 = [x for x in scored if x[3] > 0.7 and 0.3 < x[4] < 1.0]
    # Bucket 2: moderate zero frac (0.1 - 0.5), wider spread
    b2 = [x for x in scored if 0.05 < x[3] < 0.4 and x[4] > 0.6]
    # Bucket 3: near-continuous (low zero), tight-ish
    b3 = [x for x in scored if x[3] < 0.05 and x[4] > 0.3]
    picks = []
    for b in (b1, b2, b3):
        if not b:
            continue
        chosen = b[rng.integers(0, len(b))]
        picks.append((chosen[0], chosen[1], chosen[2]))
    while len(picks) < 3 and scored:
        chosen = scored[rng.integers(0, len(scored))]
        picks.append((chosen[0], chosen[1], chosen[2]))
    return picks[:3]


# --- Panels ------------------------------------------------------------------


def _panel_a_concept(ax, samples: np.ndarray, gene: str, pert: str, K: int) -> None:
    """Concept panel: histogram backdrop + empirical & K-token quantile fn."""
    sorted_s = np.sort(samples)
    emp_q = np.linspace(0.0, 1.0, len(sorted_s))
    token = _tokenize(samples, K)
    tok_q = np.linspace(0.0, 1.0, K)

    # Background histogram, normalized, rotated to share the same y-axis range
    # as the quantile function. We draw it as a vertical density on the right.
    ax.plot(emp_q, sorted_s, color=ORIG, lw=2.0, label=f"empirical ({len(sorted_s)} cells)")
    ax.plot(
        tok_q,
        token,
        color=TOKEN,
        lw=0.0,
        marker="o",
        ms=4.0,
        mfc=TOKEN,
        mec="white",
        mew=0.5,
        label=f"K={K} token",
    )
    ax.plot(tok_q, token, color=TOKEN, lw=1.2, alpha=0.8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("quantile q")
    ax.set_ylabel("expression")
    ax.set_title(f"(a)  Tokenize: n cells -> K={K} numbers", loc="left")
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.05,
        f"{gene} | {pert}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=9,
    )


def _panel_b_histograms(
    gs_cell, fig, picks, K: int, seed: int
) -> None:
    """3 stacked histogram comparisons (original vs K-token)."""
    rng = np.random.default_rng(seed)
    sub = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_cell, hspace=0.7
    )
    top_ax = None
    for i, (gene, pert, samples) in enumerate(picks):
        ax = fig.add_subplot(sub[i])
        if i == 0:
            top_ax = ax
        token = _tokenize(samples, K)
        recon = _sample_from_grid(token, len(samples), rng)
        lo = float(min(samples.min(), recon.min()))
        hi = float(max(samples.max(), recon.max()))
        bins = np.linspace(lo, hi, 36)
        ax.hist(samples, bins=bins, color=ORIG, alpha=0.55, label="original")
        ax.hist(recon, bins=bins, color=TOKEN, alpha=0.55, label=f"K={K} recon.")
        w1, _ = _reconstruction_losses_on_ref(samples, K)
        ax.set_title(
            f"{gene} | {pert}   (n={len(samples)},  W1={w1:.3g})",
            fontsize=9,
            loc="left",
            fontweight="normal",
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
        if i == 2:
            ax.set_xlabel("expression")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    if top_ax is not None:
        bbox = top_ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + 0.025,
            f"(b)  Histogram fidelity at K={K}",
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )


def _panel_c_quantiles(ax, picks, K: int) -> None:
    """Overlay empirical vs K-token quantile fns for three distributions."""
    colors = ["#1f4e79", "#6c3483", "#117a65"]
    M = 2048
    ref_q = np.linspace(0.0, 1.0, M)
    labels = []
    for (gene, pert, samples), c in zip(picks, colors):
        sorted_s = np.sort(samples)
        emp_on_ref = np.interp(
            ref_q, np.linspace(0.0, 1.0, len(sorted_s)), sorted_s
        )
        token = _tokenize(samples, K)
        tok_on_ref = np.interp(ref_q, np.linspace(0.0, 1.0, K), token)
        w1, _ = _reconstruction_losses_on_ref(samples, K)
        lbl = f"{gene} | {pert}  (W1={w1:.3g})"
        ax.plot(ref_q, emp_on_ref, color=c, lw=1.8, label=lbl)
        ax.plot(ref_q, tok_on_ref, color=c, lw=1.2, ls=(0, (2, 2)), alpha=0.9)
        labels.append(lbl)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("quantile q")
    ax.set_ylabel("expression")
    ax.set_title(
        f"(c)  Quantile-function fidelity at K={K}  (solid = empirical, dashed = token)",
        loc="left",
    )
    ax.legend(loc="upper left")


def _panel_d_loss_vs_K(
    ax, Ks, w1_mean, w1_q1, w1_q3, cr_mean, cr_q1, cr_q3, K_star: int
) -> None:
    """Dual-axis: W1 and Cramer-RMSE mean with IQR vs K, log-log."""
    ax.plot(Ks, w1_mean, color=W1_C, lw=2.0, marker="o", ms=5, label="W1")
    ax.fill_between(Ks, w1_q1, w1_q3, color=W1_C, alpha=0.15)

    ax.plot(
        Ks, cr_mean, color=CR_C, lw=2.0, marker="s", ms=5, label="Cramer-RMSE"
    )
    ax.fill_between(Ks, cr_q1, cr_q3, color=CR_C, alpha=0.15)

    ax.axvline(K_star, color=ACCENT, ls="--", lw=1.2, alpha=0.8)
    ax.text(
        K_star * 1.05,
        ax.get_ylim()[1] if False else w1_mean[0],
        f"K={K_star}",
        color=ACCENT,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], fontsize=8)
    ax.set_xlabel("token size K")
    ax.set_ylabel("reconstruction loss")
    ax.set_title("(d)  Loss vs. K  (mean and IQR over 300 distributions)", loc="left")
    ax.legend(loc="lower left")


def _panel_e_diminishing(
    ax, Ks, w1_frac, cr_frac, K_star: int, knees: dict
) -> None:
    """Normalized loss-reduction captured vs K with 90/95/99 threshold lines."""
    ax.plot(Ks, w1_frac, color=W1_C, lw=2.0, marker="o", ms=5, label="W1")
    ax.plot(Ks, cr_frac, color=CR_C, lw=2.0, marker="s", ms=5, label="Cramer-RMSE")

    for thr, style, alpha in [(0.90, ":", 0.7), (0.95, "--", 0.8), (0.99, "-.", 0.6)]:
        ax.axhline(thr, color="#555555", ls=style, lw=0.8, alpha=alpha)
        ax.text(
            Ks[0],
            thr + 0.005,
            f"{int(thr*100)}%",
            color="#555555",
            fontsize=8,
            va="bottom",
            ha="left",
        )

    ax.axvline(K_star, color=ACCENT, ls="--", lw=1.3, alpha=0.85)
    ax.text(
        K_star * 1.05,
        0.02,
        f"K={K_star}",
        color=ACCENT,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )

    # Mark intersection of K=64 with W1 curve
    j = Ks.index(K_star)
    ax.plot([K_star], [w1_frac[j]], marker="o", ms=9, mfc="none", mec=W1_C, mew=2.0)
    ax.plot([K_star], [cr_frac[j]], marker="s", ms=9, mfc="none", mec=CR_C, mew=2.0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], fontsize=8)
    ax.set_xlabel("token size K")
    ax.set_ylabel("fraction of loss reduction captured")
    ax.set_title(
        f"(e)  Diminishing returns: K={K_star} captures {w1_frac[j]*100:.0f}% (W1), "
        f"{cr_frac[j]*100:.0f}% (Cramer)",
        loc="left",
    )
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower right")


def _panel_f_per_dist_W1(ax, w1_values: np.ndarray, K_star: int) -> None:
    """Histogram of per-distribution W1 at K=K_star with summary stats."""
    med = float(np.median(w1_values))
    p90 = float(np.quantile(w1_values, 0.90))
    p99 = float(np.quantile(w1_values, 0.99))

    ax.hist(
        w1_values,
        bins=40,
        color=TOKEN,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(med, color=ACCENT, ls="-", lw=1.2, label=f"median = {med:.3g}")
    ax.axvline(p90, color=ACCENT, ls="--", lw=1.0, label=f"p90 = {p90:.3g}")
    ax.axvline(p99, color=ACCENT, ls=":", lw=1.0, label=f"p99 = {p99:.3g}")
    ax.set_xlabel(f"per-distribution W1 at K={K_star}")
    ax.set_ylabel("count")
    ax.set_title(
        f"(f)  Per-distribution W1 at K={K_star}  (n={len(w1_values)} distributions)",
        loc="left",
    )
    ax.legend(loc="upper right")


# --- Main --------------------------------------------------------------------


def build_panel(
    adata: anndata.AnnData,
    perturbation_key: str,
    K_star: int,
    Ks: list[int],
    out_png: Path,
    out_pdf: Path,
    min_cells: int = 50,
    max_dists: int = 300,
    seed: int = 0,
) -> dict:
    _set_style()

    print("  Collecting distributions ...")
    dists = _collect_distributions(
        adata, perturbation_key, min_cells=min_cells, max_dists=max_dists
    )
    print(f"  {len(dists)} distributions (min {min_cells} cells)")

    print("  Picking representative examples ...")
    picks = _pick_representative_three(dists, seed=seed)
    for g, p, s in picks:
        print(f"    {g} | {p}  (n={len(s)}, zero_frac={(s==0).mean():.2f})")

    print("  Computing loss sweep ...")
    w1_mat = np.zeros((len(dists), len(Ks)))
    cr_mat = np.zeros_like(w1_mat)
    for i, (_, _, s) in enumerate(dists):
        for j, K in enumerate(Ks):
            w1, cr = _reconstruction_losses_on_ref(s, K)
            w1_mat[i, j] = w1
            cr_mat[i, j] = cr

    w1_mean = w1_mat.mean(axis=0)
    w1_q1 = np.quantile(w1_mat, 0.25, axis=0)
    w1_q3 = np.quantile(w1_mat, 0.75, axis=0)
    cr_mean = cr_mat.mean(axis=0)
    cr_q1 = np.quantile(cr_mat, 0.25, axis=0)
    cr_q3 = np.quantile(cr_mat, 0.75, axis=0)

    def achieved(mean: np.ndarray) -> np.ndarray:
        return (mean[0] - mean) / max(mean[0] - mean[-1], 1e-12)

    w1_frac = achieved(w1_mean)
    cr_frac = achieved(cr_mean)

    def first_at(frac: np.ndarray, thr: float) -> int:
        for idx, f in enumerate(frac):
            if f >= thr:
                return Ks[idx]
        return Ks[-1]

    knees = {
        "W1_90": first_at(w1_frac, 0.90),
        "W1_95": first_at(w1_frac, 0.95),
        "W1_99": first_at(w1_frac, 0.99),
        "cramer_90": first_at(cr_frac, 0.90),
        "cramer_95": first_at(cr_frac, 0.95),
        "cramer_99": first_at(cr_frac, 0.99),
    }

    w1_at_star = w1_mat[:, Ks.index(K_star)]

    # --- Figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(17.5, 10.5))
    outer = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        wspace=0.32,
        hspace=0.45,
        left=0.055,
        right=0.985,
        top=0.92,
        bottom=0.075,
    )

    # Panel (a): concept — use the first representative distribution.
    ax_a = fig.add_subplot(outer[0, 0])
    _panel_a_concept(ax_a, picks[0][2], picks[0][0], picks[0][1], K_star)

    # Panel (b): three stacked histograms (sub-gridspec)
    _panel_b_histograms(outer[0, 1], fig, picks, K_star, seed=seed)

    # Panel (c): quantile-function overlays
    ax_c = fig.add_subplot(outer[0, 2])
    _panel_c_quantiles(ax_c, picks, K_star)

    # Panel (d): loss vs K
    ax_d = fig.add_subplot(outer[1, 0])
    _panel_d_loss_vs_K(
        ax_d, Ks, w1_mean, w1_q1, w1_q3, cr_mean, cr_q1, cr_q3, K_star
    )

    # Panel (e): diminishing returns
    ax_e = fig.add_subplot(outer[1, 1])
    _panel_e_diminishing(ax_e, Ks, w1_frac, cr_frac, K_star, knees)

    # Panel (f): per-distribution W1 at K_star
    ax_f = fig.add_subplot(outer[1, 2])
    _panel_f_per_dist_W1(ax_f, w1_at_star, K_star)

    fig.suptitle(
        f"Quantile-grid tokenization at K={K_star}: fidelity vs. compactness",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "K_star": K_star,
        "Ks": list(Ks),
        "w1_mean": w1_mean.tolist(),
        "cramer_mean": cr_mean.tolist(),
        "knees": knees,
        "per_dist_W1_at_Kstar": {
            "median": float(np.median(w1_at_star)),
            "p90": float(np.quantile(w1_at_star, 0.90)),
            "p99": float(np.quantile(w1_at_star, 0.99)),
        },
        "n_distributions": len(dists),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, default="data/mini_perturb_seq.h5ad")
    parser.add_argument("--perturbation-key", type=str, default="perturbation")
    parser.add_argument("--K-star", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=str, default="eval_results/quantile_tokenization"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.adata} ...")
    adata = anndata.read_h5ad(args.adata)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    Ks = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    stats = build_panel(
        adata,
        args.perturbation_key,
        K_star=args.K_star,
        Ks=Ks,
        out_png=out_dir / f"panel_K{args.K_star}.png",
        out_pdf=out_dir / f"panel_K{args.K_star}.pdf",
        seed=args.seed,
    )
    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
