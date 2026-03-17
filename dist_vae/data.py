"""Dataset classes and quantile grid utilities.

Provides datasets for loading Perturb-seq data from AnnData objects
and generating synthetic distributions for testing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    squeeze = False
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
        squeeze = True

    if n_valid is not None:
        samples = samples[:, :n_valid]

    sorted_samples, _ = torch.sort(samples, dim=-1)
    n = sorted_samples.shape[-1]

    if n == grid_size:
        result = sorted_samples
    else:
        # Interpolate using F.interpolate on 1D signal
        # Shape: (batch, 1, n) -> (batch, 1, grid_size) -> (batch, grid_size)
        result = F.interpolate(
            sorted_samples.unsqueeze(1).float(),
            size=grid_size,
            mode="linear",
            align_corners=True,
        ).squeeze(1)

    if squeeze:
        result = result.squeeze(0)
    return result


def quantile_grid_to_samples(grid: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Convert a quantile grid back to samples via interpolation.

    Args:
        grid: Quantile grid, shape (batch, grid_size) or (grid_size,).
        n_samples: Number of output samples.

    Returns:
        Sorted samples, shape (batch, n_samples) or (n_samples,).
    """
    squeeze = False
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
        squeeze = True

    grid_size = grid.shape[-1]
    if n_samples == grid_size:
        result = grid
    else:
        result = F.interpolate(
            grid.unsqueeze(1).float(),
            size=n_samples,
            mode="linear",
            align_corners=True,
        ).squeeze(1)

    if squeeze:
        result = result.squeeze(0)
    return result


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
        super().__init__()
        self.n_distributions = n_distributions
        self.grid_size = grid_size

        rng = torch.Generator()
        rng.manual_seed(seed)

        self.grids = []
        n_samples_per_dist = 500

        for _ in range(n_distributions):
            # Random number of components
            n_comp = torch.randint(
                n_components_range[0], n_components_range[1] + 1, (1,), generator=rng
            ).item()

            # Random mixture parameters
            means = torch.randn(n_comp, generator=rng) * 3.0
            stds = torch.abs(torch.randn(n_comp, generator=rng)) * 0.5 + 0.1
            weights = torch.rand(n_comp, generator=rng)
            weights = weights / weights.sum()

            # Sample from mixture
            samples = []
            for j in range(n_comp):
                n_j = max(1, int(weights[j].item() * n_samples_per_dist))
                comp_samples = means[j] + stds[j] * torch.randn(n_j, generator=rng)
                samples.append(comp_samples)
            all_samples = torch.cat(samples)

            # Convert to quantile grid
            grid = samples_to_quantile_grid(all_samples, grid_size)
            self.grids.append(grid)

        self.grids = torch.stack(self.grids)

    def __len__(self) -> int:
        return self.n_distributions

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Get a quantile grid and metadata indices.

        Returns:
            Tuple of (quantile_grid, gene_index, perturbation_index).
            For synthetic data, gene_index and perturbation_index are
            deterministic mappings from idx.
        """
        gene_idx = idx % 100
        pert_idx = idx // 100
        return self.grids[idx], gene_idx, pert_idx


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
        import numpy as np
        import scipy.sparse as sp

        super().__init__()
        self.grid_size = grid_size

        # Determine genes to use
        if gene_subset is not None:
            gene_mask = adata.var_names.isin(gene_subset)
            self.gene_names = list(adata.var_names[gene_mask])
        else:
            self.gene_names = list(adata.var_names)
            gene_mask = np.ones(adata.n_vars, dtype=bool)

        gene_indices = np.where(gene_mask)[0]

        # Group cells by perturbation
        perturbations = adata.obs[perturbation_key]
        pert_counts = perturbations.value_counts()
        valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()
        self.perturbation_names = sorted(valid_perts)

        # Build index: list of (gene_idx, pert_idx) and precompute grids
        self.grids = []
        self.index_map: list[tuple[int, int, int]] = []  # (gene_local_idx, pert_idx, n_cells)

        X = adata.X

        for pert_idx, pert_name in enumerate(self.perturbation_names):
            cell_mask = (perturbations == pert_name).values
            n_cells = int(cell_mask.sum())

            # Densify only this perturbation's cells
            X_pert = X[cell_mask][:, gene_indices]
            if sp.issparse(X_pert):
                X_pert = X_pert.toarray()
            X_pert = torch.tensor(X_pert, dtype=torch.float32)  # (n_cells, n_genes)

            for gene_local_idx in range(len(self.gene_names)):
                gene_values = X_pert[:, gene_local_idx]  # (n_cells,)
                grid = samples_to_quantile_grid(gene_values, grid_size)
                self.grids.append(grid)
                self.index_map.append((gene_local_idx, pert_idx, n_cells))

        self.grids = torch.stack(self.grids) if self.grids else torch.empty(0, grid_size)

    @classmethod
    def from_anndata(
        cls,
        adata: "anndata.AnnData",
        **kwargs,
    ) -> "PerturbationDistributionDataset":
        """Create dataset from an AnnData object."""
        return cls(adata, **kwargs)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Get a quantile grid and metadata indices."""
        gene_idx, pert_idx, _ = self.index_map[idx]
        return self.grids[idx], gene_idx, pert_idx

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a specific distribution."""
        gene_idx, pert_idx, n_cells = self.index_map[idx]
        return {
            "gene_name": self.gene_names[gene_idx],
            "perturbation_name": self.perturbation_names[pert_idx],
            "n_cells": n_cells,
        }
