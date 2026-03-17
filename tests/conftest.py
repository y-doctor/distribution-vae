"""Shared test fixtures for distribution-vae tests."""

import pytest
import torch


@pytest.fixture
def sample_quantile_grids():
    """Generate random sorted quantile grids for testing."""
    def _make(batch_size: int = 8, grid_size: int = 256) -> torch.Tensor:
        grids = torch.randn(batch_size, grid_size)
        grids, _ = torch.sort(grids, dim=-1)
        return grids
    return _make


@pytest.fixture
def synthetic_dataset():
    """Create a small SyntheticDistributionDataset for testing."""
    def _make(n: int = 50, grid_size: int = 64):
        from dist_vae.data import SyntheticDistributionDataset
        return SyntheticDistributionDataset(
            n_distributions=n, grid_size=grid_size, seed=42
        )
    return _make


@pytest.fixture
def small_model():
    """Create a small DistributionVAE with random weights for testing."""
    def _make(grid_size: int = 64, latent_dim: int = 8, hidden_dim: int = 32):
        from dist_vae.model import DistributionVAE
        return DistributionVAE(
            grid_size=grid_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            beta=0.01,
        )
    return _make
