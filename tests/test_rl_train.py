"""Tests for the hinge / rescale logic in dist_vae.rl_train."""

import math

import torch

from dist_vae.rl_train import PlateauDetector, apply_hinge


def test_hinge_binary_keeps_raw_above_threshold() -> None:
    """Default (rescale=False): keep raw reward above threshold, zero below."""
    R = torch.tensor([0.1, 0.2, 0.3, 0.7, 1.0])
    out = apply_hinge(R, threshold=0.25, rescale=False)
    expected = torch.tensor([0.0, 0.0, 0.3, 0.7, 1.0])
    assert torch.allclose(out, expected)


def test_hinge_binary_step_at_threshold() -> None:
    """At exactly threshold: zero. Just above: raw value (step discontinuity)."""
    R = torch.tensor([0.25, 0.2501])
    out = apply_hinge(R, threshold=0.25, rescale=False)
    assert float(out[0]) == 0.0
    assert abs(float(out[1]) - 0.2501) < 1e-6


def test_rescale_maps_threshold_to_zero_and_one_to_one() -> None:
    """Rescaled: r=theta -> 0; r=1 -> 1; linear in between."""
    theta = 0.3
    R = torch.tensor([theta, 1.0])
    out = apply_hinge(R, threshold=theta, rescale=True)
    assert abs(float(out[0]) - 0.0) < 1e-6
    assert abs(float(out[1]) - 1.0) < 1e-6


def test_rescale_below_threshold_is_zero() -> None:
    """Values at or below the threshold clamp to zero."""
    R = torch.tensor([-0.5, 0.0, 0.2, 0.3])
    out = apply_hinge(R, threshold=0.3, rescale=True)
    assert torch.allclose(out, torch.zeros_like(R), atol=1e-6)


def test_rescale_is_linear_above_threshold() -> None:
    """Halfway between theta and 1 should map to 0.5."""
    theta = 0.2
    mid = 0.5 * (theta + 1.0)        # 0.6
    R = torch.tensor([mid])
    out = apply_hinge(R, threshold=theta, rescale=True)
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_rescale_wider_dynamic_range_than_binary() -> None:
    """For Pearson-shaped rewards the rescaled range is strictly wider than
    the binary-hinge range, for any threshold > 0."""
    theta = 0.3
    R = torch.tensor([0.4, 0.6, 0.9])
    binary = apply_hinge(R, threshold=theta, rescale=False)
    rescaled = apply_hinge(R, threshold=theta, rescale=True)
    # Binary spans [0.4, 0.9] = 0.5; rescaled spans [(0.4-0.3)/0.7, (0.9-0.3)/0.7]
    # = [0.143, 0.857] = 0.714. The rescaled spread is strictly larger.
    assert (rescaled.max() - rescaled.min()) > (binary.max() - binary.min())


def test_hinge_broadcasts_per_row_threshold() -> None:
    """Per-row threshold (P, 1) broadcasts over the (P, P) reward matrix."""
    R = torch.tensor([
        [0.1, 0.5, 0.9],
        [0.2, 0.4, 0.8],
    ])
    theta = torch.tensor([[0.2], [0.5]])   # (2, 1)
    out = apply_hinge(R, threshold=theta, rescale=False)
    expected = torch.tensor([
        [0.0, 0.5, 0.9],
        [0.0, 0.0, 0.8],
    ])
    assert torch.allclose(out, expected)


def test_hinge_broadcasts_per_row_with_rescale() -> None:
    """Per-row threshold with rescale=True rescales each row by its own theta."""
    R = torch.tensor([
        [0.1, 0.5, 0.9],
        [0.2, 0.4, 0.8],
    ])
    theta = torch.tensor([[0.2], [0.5]])
    out = apply_hinge(R, threshold=theta, rescale=True)
    # Row 0: theta=0.2, denom=0.8 -> [0, 0.375, 0.875]
    # Row 1: theta=0.5, denom=0.5 -> [0, 0, 0.6]
    expected = torch.tensor([
        [0.0, 0.375, 0.875],
        [0.0, 0.0, 0.6],
    ])
    assert torch.allclose(out, expected, atol=1e-6)


def test_rescale_preserves_monotonicity() -> None:
    """A monotonic transform: rank order above the threshold is preserved."""
    theta = 0.1
    R = torch.tensor([0.2, 0.3, 0.5, 0.6, 0.9])
    out = apply_hinge(R, threshold=theta, rescale=True)
    diffs = out[1:] - out[:-1]
    assert (diffs >= 0).all()


# -------------------------- PlateauDetector ----------------------------------

def test_plateau_waits_for_full_window() -> None:
    """No stop signal (and NaN smoothed) before the first `window` epochs."""
    d = PlateauDetector(window=5, patience=3, min_delta=0.001)
    for i in range(4):
        should_stop, smoothed = d.update([0.1] * (i + 1))
        assert not should_stop
        assert math.isnan(smoothed)


def test_plateau_triggers_when_no_improvement() -> None:
    """A flat reward history triggers early stop after `patience` epochs."""
    d = PlateauDetector(window=3, patience=2, min_delta=0.001)
    # Fill the window — this sets the initial best_smoothed.
    for i in range(3):
        should_stop, _ = d.update([0.5] * (i + 1))
        assert not should_stop
    # One more flat update = 1 epoch without improvement, still not stopping.
    should_stop, _ = d.update([0.5] * 4)
    assert not should_stop
    # Second flat update = 2 epochs = patience reached, stop.
    should_stop, _ = d.update([0.5] * 5)
    assert should_stop


def test_plateau_resets_on_improvement() -> None:
    """A fresh high resets the counter. Stop signal comes `patience` epochs
    AFTER the last improvement, not before."""
    d = PlateauDetector(window=3, patience=2, min_delta=0.001)
    # Window fills: best_smoothed <- mean([0.10, 0.10, 0.10]) = 0.10.
    history = [0.10, 0.10, 0.10]
    should_stop, _ = d.update(history)
    assert not should_stop
    # Improvement. best_smoothed -> 0.20 (mean of [0.10, 0.10, 0.40]).
    history.append(0.40)
    should_stop, _ = d.update(history)
    assert not should_stop
    # Now two flat appends, counter -> 2 -> stop.
    history.append(0.10)
    should_stop, _ = d.update(history)
    assert not should_stop                    # counter = 1
    history.append(0.10)
    should_stop, _ = d.update(history)
    assert should_stop                        # counter = 2 = patience


def test_plateau_noisy_history_below_min_delta() -> None:
    """Small oscillations that don't exceed min_delta should count as flat."""
    d = PlateauDetector(window=5, patience=3, min_delta=0.01)
    history = [0.50, 0.51, 0.50, 0.49, 0.50]  # smoothed 0.500
    d.update(history)
    # Subsequent window-smoothed values oscillate within +/- min_delta of 0.500.
    for r in [0.49, 0.51, 0.50, 0.49]:        # window stays in [0.498, 0.504]
        history.append(r)
        d.update(history)
    # After `patience` epochs of flat-ish smoothed values -> stop.
    should_stop, _ = d.update(history)
    assert should_stop


def test_plateau_min_delta_is_respected() -> None:
    """Improvements smaller than min_delta should NOT reset the counter."""
    d = PlateauDetector(window=2, patience=2, min_delta=0.05)
    history = [0.50, 0.50]
    d.update(history)                 # best_smoothed = 0.50
    # +0.01 improvement < 0.05 min_delta: counter advances.
    history.extend([0.51, 0.51])
    should_stop, _ = d.update(history[:3])
    assert not should_stop
    should_stop, _ = d.update(history[:4])
    assert should_stop
