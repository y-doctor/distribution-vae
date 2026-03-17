"""Tests for distributional loss functions."""

import pytest
import torch

from dist_vae.losses import (
    CombinedDistributionLoss,
    cramer_distance,
    ks_distance_smooth,
    wasserstein1_distance,
)


class TestCramerDistance:
    def test_non_negative(self, sample_quantile_grids):
        """Cramer distance should be non-negative."""
        raise NotImplementedError

    def test_zero_for_identical(self, sample_quantile_grids):
        """Cramer distance should be zero when inputs are identical."""
        raise NotImplementedError

    def test_differentiable(self, sample_quantile_grids):
        """Cramer distance should support backpropagation."""
        raise NotImplementedError

    def test_batch_dimension(self, sample_quantile_grids):
        """Output shape should match batch dimension."""
        raise NotImplementedError


class TestWasserstein1Distance:
    def test_non_negative(self, sample_quantile_grids):
        """Wasserstein-1 distance should be non-negative."""
        raise NotImplementedError

    def test_zero_for_identical(self, sample_quantile_grids):
        """Wasserstein-1 distance should be zero when inputs are identical."""
        raise NotImplementedError

    def test_differentiable(self, sample_quantile_grids):
        """Wasserstein-1 distance should support backpropagation."""
        raise NotImplementedError

    def test_batch_dimension(self, sample_quantile_grids):
        """Output shape should match batch dimension."""
        raise NotImplementedError


class TestKSDistanceSmooth:
    def test_non_negative(self, sample_quantile_grids):
        """Smooth KS distance should be non-negative."""
        raise NotImplementedError

    def test_zero_for_identical(self, sample_quantile_grids):
        """Smooth KS distance should be approximately zero for identical inputs."""
        raise NotImplementedError

    def test_differentiable(self, sample_quantile_grids):
        """Smooth KS distance should support backpropagation."""
        raise NotImplementedError

    def test_batch_dimension(self, sample_quantile_grids):
        """Output shape should match batch dimension."""
        raise NotImplementedError

    def test_temperature_effect(self, sample_quantile_grids):
        """Lower temperature should give values closer to true max."""
        raise NotImplementedError


class TestCombinedDistributionLoss:
    def test_respects_weights(self, sample_quantile_grids):
        """Combined loss should weight components correctly."""
        raise NotImplementedError

    def test_returns_components(self, sample_quantile_grids):
        """Combined loss should return individual component losses."""
        raise NotImplementedError

    def test_zero_weight_disables_component(self, sample_quantile_grids):
        """Components with zero weight should not contribute to total."""
        raise NotImplementedError


class TestDistanceOrdering:
    def test_closer_distributions_have_smaller_distance(self):
        """Known distribution pairs: closer distributions should have smaller distances."""
        raise NotImplementedError
