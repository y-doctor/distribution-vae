"""Tests for dataset classes and quantile grid utilities."""

import pytest
import torch

from dist_vae.data import (
    SyntheticDistributionDataset,
    quantile_grid_to_samples,
    samples_to_quantile_grid,
)


class TestSamplesToQuantileGrid:
    def test_known_input(self):
        """Known sorted input should produce expected grid."""
        raise NotImplementedError

    def test_output_is_sorted(self):
        """Output quantile grid should be monotonically non-decreasing."""
        raise NotImplementedError

    def test_different_grid_sizes(self):
        """Should produce grids of the requested size."""
        raise NotImplementedError

    def test_variable_input_sizes(self):
        """Different input sizes should produce same-size grids."""
        raise NotImplementedError

    def test_batch_support(self):
        """Should handle batched inputs."""
        raise NotImplementedError


class TestQuantileGridToSamples:
    def test_roundtrip(self):
        """samples_to_quantile_grid → quantile_grid_to_samples should approximately invert."""
        raise NotImplementedError

    def test_output_size(self):
        """Should produce the requested number of samples."""
        raise NotImplementedError

    def test_output_is_sorted(self):
        """Output samples should be sorted."""
        raise NotImplementedError


class TestSyntheticDistributionDataset:
    def test_deterministic_with_seed(self):
        """Same seed should produce identical datasets."""
        raise NotImplementedError

    def test_correct_shapes(self):
        """Dataset items should have correct shapes."""
        raise NotImplementedError

    def test_grid_is_monotonic(self):
        """All quantile grids should be monotonically non-decreasing."""
        raise NotImplementedError

    def test_length(self):
        """Dataset length should match n_distributions."""
        raise NotImplementedError

    def test_different_seeds_differ(self):
        """Different seeds should produce different datasets."""
        raise NotImplementedError

    def test_returns_tuple(self):
        """__getitem__ should return (grid, gene_idx, pert_idx) tuple."""
        raise NotImplementedError
