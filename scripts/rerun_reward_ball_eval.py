"""Regenerate the reward-ball plots for a cached eval run.

Given an eval directory that already has ``logits.npy`` and ``true_p.npy``
(produced by a prior run of ``eval_rl_perturbation.py``), this script
reconstructs ``pred_bundle`` offline and runs only the new reward-ball
plotting functions — no model checkpoint required.

Profiles and ``R_sim`` are rebuilt from the AnnData, so the pert ordering
used at training time must match the one alphabetical-sort inside
``PerturbationClassificationDataset`` (it always does, given identical
``val_fraction`` and ``split_seed``).

Usage::

    python scripts/rerun_reward_ball_eval.py \\
        --adata data/mini_perturb_seq_2kg_allp_ntc.h5ad \\
        --eval-dir eval_results/rl_perturbation_2kg_allp_rownorm/val_ens1 \\
        --val-fraction 0.20 --split-seed 123 --mode val
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dist_vae.rl_data import PerturbationClassificationDataset

# Load eval_rl_perturbation module by path so we can call its helpers.
_spec = importlib.util.spec_from_file_location(
    "eval_rl_perturbation", ROOT / "scripts" / "eval_rl_perturbation.py"
)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--n-show", type=int, default=20)
    args = parser.parse_args()

    _eval_mod._set_style()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Load cached logits / true_p ----------------------------------------
    logits = np.load(eval_dir / "logits.npy")
    true_p = np.load(eval_dir / "true_p.npy")
    print(f"Loaded {logits.shape} logits from {eval_dir}")
    pred_p = np.argmax(logits, axis=1).astype(np.int64)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    P = int(logits.shape[1])
    C = np.zeros((P, P), dtype=np.int64)
    for t, q in zip(true_p, pred_p):
        C[int(t), int(q)] += 1

    pred_bundle = {
        "confusion": C,
        "logits": logits.astype(np.float32),
        "probs": probs.astype(np.float32),
        "true_p": true_p.astype(np.int64),
        "pred_p": pred_p,
    }
    top1 = float(np.diag(C / C.sum(axis=1, keepdims=True).clip(min=1)).mean())
    print(f"  derived top-1 from cached logits: {top1:.4f}")

    # --- Rebuild profiles + R_sim from AnnData ------------------------------
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
        mode=args.mode,
    )
    if len(dataset.perturbation_names) != P:
        raise SystemExit(
            f"Pert count mismatch: logits have {P} classes but dataset has "
            f"{len(dataset.perturbation_names)}. Did the AnnData change?"
        )
    profiles = dataset.compute_delta_mean_profiles()
    R_sim = _eval_mod.compute_reward_matrix(profiles)
    print(f"  profiles: {profiles.shape}, R_sim: {R_sim.shape}")

    # --- Run the new reward-ball plots --------------------------------------
    print("Plotting soft-accuracy curve ...")
    curve = _eval_mod.plot_soft_accuracy_curve(
        pred_bundle, profiles, R_sim, eval_dir / "soft_accuracy_curve.png"
    )
    print(f"  headline P(reward >= 0.9) = {curve['headline_value']:.3f}")

    print(f"Plotting per-pert reward-ball neighborhoods "
          f"(best-{args.n_show}, worst-{args.n_show}) ...")
    _eval_mod.plot_pert_neighborhoods(
        pred_bundle,
        profiles,
        R_sim,
        dataset.perturbation_names,
        eval_dir,
        tau=args.tau,
        n_show=args.n_show,
    )

    # --- Extend metrics.json with the soft-accuracy curve -------------------
    metrics_path = eval_dir / "metrics.json"
    existing: dict = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except Exception:
            existing = {}
    existing["soft_accuracy_curve"] = curve
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"Updated {metrics_path} with soft_accuracy_curve.")
    print(f"Done. Plots under {eval_dir}")


if __name__ == "__main__":
    main()
