"""Tests for the hinge / rescale logic in dist_vae.rl_train."""

import torch

from dist_vae.rl_train import apply_hinge


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
