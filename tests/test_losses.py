"""Tests for distributional loss functions."""

import torch

from dist_vae.losses import (
    CombinedDistributionLoss,
    cramer_distance,
    kl_divergence_quantile,
    wasserstein1_distance,
)


class TestCramerDistance:
    def test_non_negative(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        y = sample_quantile_grids(8, 256)
        d = cramer_distance(x, y)
        assert (d >= 0).all()

    def test_zero_for_identical(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        d = cramer_distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_differentiable(self, sample_quantile_grids):
        x = sample_quantile_grids(4, 64)
        x.requires_grad_(True)
        y = sample_quantile_grids(4, 64)
        d = cramer_distance(x, y).sum()
        d.backward()
        assert x.grad is not None

    def test_batch_dimension(self, sample_quantile_grids):
        x = sample_quantile_grids(16, 128)
        y = sample_quantile_grids(16, 128)
        d = cramer_distance(x, y)
        assert d.shape == (16,)


class TestWasserstein1Distance:
    def test_non_negative(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        y = sample_quantile_grids(8, 256)
        d = wasserstein1_distance(x, y)
        assert (d >= 0).all()

    def test_zero_for_identical(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        d = wasserstein1_distance(x, x)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_differentiable(self, sample_quantile_grids):
        x = sample_quantile_grids(4, 64)
        x.requires_grad_(True)
        y = sample_quantile_grids(4, 64)
        d = wasserstein1_distance(x, y).sum()
        d.backward()
        assert x.grad is not None

    def test_batch_dimension(self, sample_quantile_grids):
        x = sample_quantile_grids(16, 128)
        y = sample_quantile_grids(16, 128)
        d = wasserstein1_distance(x, y)
        assert d.shape == (16,)


class TestKLDivergenceQuantile:
    def test_zero_for_identical(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        d = kl_divergence_quantile(x, x)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)

    def test_differentiable(self, sample_quantile_grids):
        x = sample_quantile_grids(4, 64)
        y = sample_quantile_grids(4, 64)
        y.requires_grad_(True)
        d = kl_divergence_quantile(x, y).sum()
        d.backward()
        assert y.grad is not None

    def test_batch_dimension(self, sample_quantile_grids):
        x = sample_quantile_grids(16, 128)
        y = sample_quantile_grids(16, 128)
        d = kl_divergence_quantile(x, y)
        assert d.shape == (16,)

    def test_positive_for_different_distributions(self):
        """KL should be positive when distributions differ significantly."""
        torch.manual_seed(42)
        # Narrow distribution vs wide distribution
        x = torch.sort(torch.randn(4, 128) * 0.1)[0]  # narrow
        y = torch.sort(torch.randn(4, 128) * 2.0)[0]   # wide
        d = kl_divergence_quantile(x, y)
        # KL(narrow || wide) should be meaningfully non-zero
        assert d.abs().mean() > 0.01

    def test_detects_location_shift(self):
        """KL should be ~0 for a pure location shift (same shape)."""
        torch.manual_seed(42)
        x = torch.sort(torch.randn(4, 256))[0]
        y = x + 5.0  # shift location, keep shape identical
        d = kl_divergence_quantile(x, y)
        # Same spacings → log(dy/dx) = log(1) = 0 (small eps noise from clamping)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-4)

    def test_detects_scale_change(self):
        """KL should be non-zero for a scale change (different shape)."""
        torch.manual_seed(42)
        x = torch.sort(torch.randn(4, 256))[0]
        y = x * 2.0  # scale up → spacings double → log(2) per interval
        d = kl_divergence_quantile(x, y)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(d, expected * torch.ones_like(d), atol=0.05)


class TestCombinedDistributionLoss:
    def test_respects_weights(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 64)
        y = sample_quantile_grids(8, 64)

        loss_cramer_only = CombinedDistributionLoss({"cramer": 1.0, "wasserstein1": 0.0})
        total_c, comp_c = loss_cramer_only(x, y)

        loss_w1_only = CombinedDistributionLoss({"cramer": 0.0, "wasserstein1": 1.0})
        total_w, comp_w = loss_w1_only(x, y)

        # Cramer-only should equal cramer distance mean
        expected_c = cramer_distance(x, y).mean()
        assert torch.allclose(total_c, expected_c)

        # W1-only should equal wasserstein1 distance mean
        expected_w = wasserstein1_distance(x, y).mean()
        assert torch.allclose(total_w, expected_w)

    def test_returns_components(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 64)
        y = sample_quantile_grids(8, 64)
        loss_fn = CombinedDistributionLoss({"cramer": 1.0, "wasserstein1": 0.5})
        total, components = loss_fn(x, y)
        assert "cramer" in components
        assert "wasserstein1" in components
        assert components["cramer"].shape == (8,)

    def test_zero_weight_disables_component(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 64)
        y = sample_quantile_grids(8, 64)
        loss_fn = CombinedDistributionLoss({"cramer": 1.0, "wasserstein1": 0.0})
        _, components = loss_fn(x, y)
        assert "wasserstein1" not in components

    def test_kl_divergence_component(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 64)
        y = sample_quantile_grids(8, 64)
        loss_fn = CombinedDistributionLoss({"cramer": 1.0, "kl_divergence": 0.5})
        total, components = loss_fn(x, y)
        assert "cramer" in components
        assert "kl_divergence" in components


class TestDistanceOrdering:
    def test_closer_distributions_have_smaller_distance(self):
        torch.manual_seed(42)
        # Create three sorted grids: a, b (close to a), c (far from a)
        a = torch.sort(torch.randn(1, 128))[0]
        b = a + 0.01 * torch.randn(1, 128)
        b = torch.sort(b)[0]
        c = a + 10.0 * torch.randn(1, 128)
        c = torch.sort(c)[0]

        for loss_fn in [cramer_distance, wasserstein1_distance]:
            d_ab = loss_fn(a, b)
            d_ac = loss_fn(a, c)
            assert d_ab < d_ac, f"{loss_fn.__name__}: d(a,b)={d_ab.item()} >= d(a,c)={d_ac.item()}"
