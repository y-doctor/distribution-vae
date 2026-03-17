"""Dataset classes and quantile grid utilities.

Provides datasets for loading Perturb-seq data from AnnData objects
and generating synthetic distributions for testing.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def samples_to_quantile_grid(
    samples: torch.Tensor, grid_size: int, n_valid: int | None = None
) -> torch.Tensor:
    """Convert raw samples to a fixed-size quantile grid.

    Sorts the samples and interpolates onto a uniform grid of the specified size.

    Args:
        samples: Raw 1D samples, shape (batch, n_samples) or (n_samples,).
        grid_size: Number of points in the output quantile grid.
        n_valid: If provided, only use the first n_valid samples (rest are padding).

    Returns:
        Quantile grid, shape (batch, grid_size) or (grid_size,).
    """
    raise NotImplementedError


def quantile_grid_to_samples(grid: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Convert a quantile grid back to samples via interpolation.

    Args:
        grid: Quantile grid, shape (batch, grid_size) or (grid_size,).
        n_samples: Number of output samples.

    Returns:
        Sorted samples, shape (batch, n_samples) or (n_samples,).
    """
    raise NotImplementedError


class SyntheticDistributionDataset(Dataset):
    """Dataset of synthetic distributions from random Gaussian mixtures.

    Each distribution is a random mixture of 1-4 Gaussians, sampled and
    converted to a quantile grid. Fully deterministic given a seed.

    Args:
        n_distributions: Number of distributions to generate.
        grid_size: Size of the quantile grid.
        n_components_range: Range of number of Gaussian components (min, max).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_distributions: int = 1000,
        grid_size: int = 256,
        n_components_range: tuple[int, int] = (1, 4),
        seed: int = 42,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Get a quantile grid and metadata indices.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (quantile_grid, gene_index, perturbation_index).
            For synthetic data, gene_index and perturbation_index are
            deterministic mappings from idx.
        """
        raise NotImplementedError


class PerturbationDistributionDataset(Dataset):
    """Dataset of per-(gene, perturbation) expression distributions from AnnData.

    Extracts distributions by grouping cells by perturbation, then creating
    one quantile grid per (gene, perturbation) pair.

    Args:
        adata: AnnData object with cells x genes expression matrix.
        perturbation_key: Column in adata.obs identifying perturbations.
        grid_size: Size of the quantile grid.
        min_cells: Minimum number of cells for a perturbation to be included.
        gene_subset: Optional list of gene names to use (default: all genes).
    """

    def __init__(
        self,
        adata: "anndata.AnnData",
        perturbation_key: str = "perturbation",
        grid_size: int = 256,
        min_cells: int = 20,
        gene_subset: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_anndata(
        cls,
        adata: "anndata.AnnData",
        **kwargs,
    ) -> "PerturbationDistributionDataset":
        """Create dataset from an AnnData object.

        Args:
            adata: AnnData object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            PerturbationDistributionDataset instance.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Get a quantile grid and metadata indices.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (quantile_grid, gene_index, perturbation_index).
        """
        raise NotImplementedError

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a specific distribution.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary with 'gene_name', 'perturbation_name', 'n_cells'.
        """
        raise NotImplementedError
