"""Smoke tests for the reward-ball visualization helpers in eval_rl_perturbation.

These don't require a trained checkpoint or the full 2kg/236p AnnData. They
synthesize pert profiles + predictions and verify the new plotting functions
run end-to-end and produce well-formed output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "eval_rl_perturbation.py"


@pytest.fixture(scope="module")
def eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_rl_perturbation", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_rl_perturbation"] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_bundle(P: int = 20, R: int = 15, n_genes: int = 40, seed: int = 0):
    """Synthesize (profiles, R_sim, pred_bundle) with known degenerate pairs.

    Half the perts come in near-duplicate pairs so the reward matrix has
    off-diagonal high-sim entries — the model's "errors" should concentrate
    there to give a realistic test case for the soft-accuracy curve.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(P // 2, n_genes)).astype(np.float32)
    jitter = 0.05 * rng.normal(size=(P // 2, n_genes)).astype(np.float32)
    profiles_np = np.concatenate([base, base + jitter], axis=0)  # 2*(P/2)=P perts
    if profiles_np.shape[0] != P:
        # If P is odd, pad with one more random pert.
        extra = rng.normal(size=(P - profiles_np.shape[0], n_genes)).astype(np.float32)
        profiles_np = np.concatenate([profiles_np, extra], axis=0)
    profiles = torch.from_numpy(profiles_np)

    unit = profiles_np / np.linalg.norm(profiles_np, axis=1, keepdims=True).clip(min=1e-8)
    R_sim = unit @ unit.T

    true_p = np.repeat(np.arange(P), R).astype(np.int64)
    # Model predicts either the true pert, its degenerate partner, or a random one.
    pred_p = np.zeros_like(true_p)
    for i, t in enumerate(true_p):
        roll = rng.random()
        if roll < 0.5:
            pred_p[i] = t  # correct
        elif roll < 0.8:
            # degenerate partner (swap halves).
            partner = (t + P // 2) % P
            pred_p[i] = partner
        else:
            pred_p[i] = rng.integers(0, P)

    # Build logits consistent with pred_p at argmax.
    logits = rng.normal(size=(P * R, P)).astype(np.float32)
    for i, p in enumerate(pred_p):
        logits[i, p] = logits[i].max() + 1.0
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    C = np.zeros((P, P), dtype=np.int64)
    for t, q in zip(true_p, pred_p):
        C[t, q] += 1

    bundle = {
        "confusion": C,
        "logits": logits,
        "probs": probs,
        "true_p": true_p,
        "pred_p": pred_p,
    }
    return profiles, R_sim, bundle


def test_soft_accuracy_curve_runs_and_shape(tmp_path, eval_module):
    profiles, R_sim, bundle = _synthetic_bundle()
    out = tmp_path / "curve.png"
    result = eval_module.plot_soft_accuracy_curve(bundle, profiles, R_sim, out)

    assert out.exists() and out.stat().st_size > 0
    assert set(result) >= {"thresholds", "model", "random", "headline_tau", "headline_value"}
    assert len(result["thresholds"]) == len(result["model"]) == len(result["random"])
    # Curves monotonically non-increasing in tau.
    m = np.array(result["model"])
    r = np.array(result["random"])
    assert np.all(np.diff(m) <= 1e-9)
    assert np.all(np.diff(r) <= 1e-9)
    # Model should beat random at tau=0.9 in this synthetic setup (degenerate pairs).
    idx = int(np.argmin(np.abs(np.array(result["thresholds"]) - 0.9)))
    assert result["model"][idx] > result["random"][idx]
    assert 0.0 <= result["headline_value"] <= 1.0


def test_pert_neighborhoods_runs(tmp_path, eval_module):
    profiles, R_sim, bundle = _synthetic_bundle()
    pert_names = [f"P{i}" for i in range(profiles.shape[0])]
    eval_module.plot_pert_neighborhoods(
        bundle, profiles, R_sim, pert_names, tmp_path, tau=0.9, n_show=5
    )
    best = tmp_path / "pert_neighborhoods_best.png"
    worst = tmp_path / "pert_neighborhoods_worst.png"
    assert best.exists() and best.stat().st_size > 0
    assert worst.exists() and worst.stat().st_size > 0


def test_soft_accuracy_curve_writes_json_compatible(tmp_path, eval_module):
    """Soft-accuracy curve return value must be JSON-serializable."""
    profiles, R_sim, bundle = _synthetic_bundle()
    out = tmp_path / "curve.png"
    result = eval_module.plot_soft_accuracy_curve(bundle, profiles, R_sim, out)
    (tmp_path / "metrics.json").write_text(json.dumps({"soft_accuracy_curve": result}))
