"""Tests for dataset classes and quantile grid utilities."""

import torch

from dist_vae.data import (
    SyntheticDistributionDataset,
    quantile_grid_to_samples,
    samples_to_quantile_grid,
)


class TestSamplesToQuantileGrid:
    def test_known_input(self):
        # Already sorted input of size 5 -> grid of size 5 should be identity
        samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        grid = samples_to_quantile_grid(samples, grid_size=5)
        assert torch.allclose(grid, samples)

    def test_output_is_sorted(self):
        torch.manual_seed(0)
        samples = torch.randn(8, 200)
        grid = samples_to_quantile_grid(samples, grid_size=64)
        diffs = grid[:, 1:] - grid[:, :-1]
        assert (diffs >= -1e-6).all()

    def test_different_grid_sizes(self):
        samples = torch.randn(4, 100)
        for gs in [32, 64, 128, 256]:
            grid = samples_to_quantile_grid(samples, grid_size=gs)
            assert grid.shape == (4, gs)

    def test_variable_input_sizes(self):
        g1 = samples_to_quantile_grid(torch.randn(50), grid_size=64)
        g2 = samples_to_quantile_grid(torch.randn(500), grid_size=64)
        assert g1.shape == g2.shape == (64,)

    def test_batch_support(self):
        samples = torch.randn(16, 100)
        grid = samples_to_quantile_grid(samples, grid_size=64)
        assert grid.shape == (16, 64)


class TestQuantileGridToSamples:
    def test_roundtrip(self):
        torch.manual_seed(0)
        # Use same size for near-exact roundtrip
        samples = torch.sort(torch.randn(4, 128))[0]
        grid = samples_to_quantile_grid(samples, grid_size=128)
        recovered = quantile_grid_to_samples(grid, n_samples=128)
        assert torch.allclose(samples, recovered, atol=0.01)

    def test_output_size(self):
        grid = torch.sort(torch.randn(4, 64))[0]
        samples = quantile_grid_to_samples(grid, n_samples=200)
        assert samples.shape == (4, 200)

    def test_output_is_sorted(self):
        grid = torch.sort(torch.randn(4, 64))[0]
        samples = quantile_grid_to_samples(grid, n_samples=200)
        diffs = samples[:, 1:] - samples[:, :-1]
        assert (diffs >= -1e-6).all()


class TestSyntheticDistributionDataset:
    def test_deterministic_with_seed(self):
        ds1 = SyntheticDistributionDataset(n_distributions=20, grid_size=64, seed=42)
        ds2 = SyntheticDistributionDataset(n_distributions=20, grid_size=64, seed=42)
        for i in range(20):
            g1, _, _ = ds1[i]
            g2, _, _ = ds2[i]
            assert torch.allclose(g1, g2)

    def test_correct_shapes(self):
        ds = SyntheticDistributionDataset(n_distributions=10, grid_size=128, seed=0)
        grid, gene_idx, pert_idx = ds[0]
        assert grid.shape == (128,)
        assert isinstance(gene_idx, int)
        assert isinstance(pert_idx, int)

    def test_grid_is_monotonic(self):
        ds = SyntheticDistributionDataset(n_distributions=50, grid_size=64, seed=42)
        for i in range(len(ds)):
            grid, _, _ = ds[i]
            diffs = grid[1:] - grid[:-1]
            assert (diffs >= -1e-6).all(), f"Non-monotonic at index {i}"

    def test_length(self):
        ds = SyntheticDistributionDataset(n_distributions=37, grid_size=64, seed=0)
        assert len(ds) == 37

    def test_different_seeds_differ(self):
        ds1 = SyntheticDistributionDataset(n_distributions=5, grid_size=64, seed=1)
        ds2 = SyntheticDistributionDataset(n_distributions=5, grid_size=64, seed=2)
        g1, _, _ = ds1[0]
        g2, _, _ = ds2[0]
        assert not torch.allclose(g1, g2)

    def test_returns_tuple(self):
        ds = SyntheticDistributionDataset(n_distributions=5, grid_size=64, seed=0)
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 3
