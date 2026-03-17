"""Tests for distributional loss functions."""

import torch

from dist_vae.losses import (
    CombinedDistributionLoss,
    cramer_distance,
    ks_distance_smooth,
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


class TestKSDistanceSmooth:
    def test_non_negative(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        y = sample_quantile_grids(8, 256)
        d = ks_distance_smooth(x, y)
        assert (d >= 0).all()

    def test_zero_for_identical(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 256)
        d = ks_distance_smooth(x, x)
        # logsumexp(0/temp, ...) = temp * log(grid_size) ≈ small but not exactly zero
        # With identical inputs abs_diff=0, so result = temp * log(grid_size)
        # This is a known property — it's the "bias" of the smooth max at zero
        assert (d < 0.1).all()  # should be small

    def test_differentiable(self, sample_quantile_grids):
        x = sample_quantile_grids(4, 64)
        x.requires_grad_(True)
        y = sample_quantile_grids(4, 64)
        d = ks_distance_smooth(x, y).sum()
        d.backward()
        assert x.grad is not None

    def test_batch_dimension(self, sample_quantile_grids):
        x = sample_quantile_grids(16, 128)
        y = sample_quantile_grids(16, 128)
        d = ks_distance_smooth(x, y)
        assert d.shape == (16,)

    def test_temperature_effect(self, sample_quantile_grids):
        x = sample_quantile_grids(8, 128)
        y = sample_quantile_grids(8, 128)
        d_low = ks_distance_smooth(x, y, temperature=0.001)
        d_high = ks_distance_smooth(x, y, temperature=1.0)
        true_max = torch.abs(x - y).max(dim=-1).values
        # Lower temperature should be closer to true max
        assert torch.mean(torch.abs(d_low - true_max)) < torch.mean(torch.abs(d_high - true_max))


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
