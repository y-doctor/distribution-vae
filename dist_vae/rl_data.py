"""Dataset for RL perturbation-classification training.

Produces (ntc_tokens, pert_tokens, pert_idx) triples from a Perturb-seq
AnnData object. Each call to __getitem__ subsamples fresh cells, tokenizes
each gene's expression as a K-point quantile grid, and returns the resulting
(n_genes, K) tensors for both the non-targeting control (NTC) cells and the
perturbed cells, plus the integer perturbation id.

This module is intentionally independent from the VAE training pipeline — it
provides the supervised/RL training path on raw K=64 quantile tokens (see
labbook/DECISIONS.md 2026-04-16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from dist_vae.data import samples_to_quantile_grid


@dataclass
class GeneVocabulary:
    """Shared index for expression genes and perturbation-target genes.

    Attributes:
        names: List of all gene symbols. Length == n_all_genes.
        name_to_idx: Mapping from gene symbol to embedding index.
        expression_gene_ids: Tensor of shape (n_expression_genes,); embedding
            index for each column of the expression matrix, in adata-var order.
        pert_target_gene_ids: List of lists; for each perturbation, the
            embedding indices of its target genes (1 or more for compounds).
    """

    names: list[str]
    name_to_idx: dict[str, int]
    expression_gene_ids: torch.Tensor
    pert_target_gene_ids: list[list[int]]


def _build_gene_vocabulary(
    expression_gene_names: list[str],
    perturbation_names: list[str],
) -> GeneVocabulary:
    """Union the expression genes and pert-target genes into a single vocab."""
    all_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    for n in expression_gene_names:
        if n not in name_to_idx:
            name_to_idx[n] = len(all_names)
            all_names.append(n)

    pert_target_gene_ids: list[list[int]] = []
    for p in perturbation_names:
        targets = p.split("_") if "_" in p else [p]
        ids: list[int] = []
        for t in targets:
            if t not in name_to_idx:
                name_to_idx[t] = len(all_names)
                all_names.append(t)
            ids.append(name_to_idx[t])
        pert_target_gene_ids.append(ids)

    expression_gene_ids = torch.tensor(
        [name_to_idx[n] for n in expression_gene_names], dtype=torch.long
    )
    return GeneVocabulary(
        names=all_names,
        name_to_idx=name_to_idx,
        expression_gene_ids=expression_gene_ids,
        pert_target_gene_ids=pert_target_gene_ids,
    )


class PerturbationClassificationDataset(Dataset):
    """Dataset yielding (ntc_tokens, pert_tokens, pert_idx) triples.

    Resamples fresh cells on every __getitem__ call so that each training
    step sees a new estimate of the K-point quantile token per gene.

    Args:
        adata: AnnData with a perturbation label column.
        perturbation_key: Column in adata.obs identifying perturbation ids.
        control_label: Substring (case-insensitive) used to identify NTC rows.
        n_cells_per_pert: Number of cells to subsample per (gene, pert) token.
        n_cells_ntc: Number of NTC cells to subsample per sample.
        grid_size: K, the size of the quantile-grid token.
        min_cells: Minimum cells per perturbation to include that pert.
        samples_per_epoch: Length of the dataset (i.e., steps per epoch).
        seed: Seed for per-call sampling RNGs.
    """

    def __init__(
        self,
        adata,
        perturbation_key: str = "perturbation",
        control_label: str = "control",
        n_cells_per_pert: int = 100,
        n_cells_ntc: int = 100,
        grid_size: int = 64,
        min_cells: int = 30,
        samples_per_epoch: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_cells_per_pert = n_cells_per_pert
        self.n_cells_ntc = n_cells_ntc
        self.grid_size = grid_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

        pert_col = adata.obs[perturbation_key].astype(str)
        is_control = pert_col.str.lower().str.contains(control_label.lower())
        if int(is_control.sum()) < min_cells:
            raise ValueError(
                f"Not enough NTC cells matching '{control_label}': "
                f"found {int(is_control.sum())}, need >= {min_cells}"
            )

        pert_counts = pert_col[~is_control].value_counts()
        valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()
        self.perturbation_names = sorted(valid_perts)

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Pre-densify per pert and the NTC pool once.
        self._pert_cells: list[np.ndarray] = []
        for p in self.perturbation_names:
            mask = (pert_col == p).values
            self._pert_cells.append(X[mask])
        self._ntc_cells: np.ndarray = X[is_control.values]

        self.gene_names = list(adata.var_names)
        self.vocab = _build_gene_vocabulary(
            self.gene_names, self.perturbation_names
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pert_idx = idx % len(self.perturbation_names)
        rng = np.random.default_rng(self.seed + idx * 6151)

        pert_mat = self._pert_cells[pert_idx]
        ntc_mat = self._ntc_cells

        n_p = min(self.n_cells_per_pert, pert_mat.shape[0])
        n_n = min(self.n_cells_ntc, ntc_mat.shape[0])

        pert_idx_cells = rng.choice(pert_mat.shape[0], size=n_p, replace=False)
        ntc_idx_cells = rng.choice(ntc_mat.shape[0], size=n_n, replace=False)

        pert_sub = torch.from_numpy(pert_mat[pert_idx_cells]).float()  # (n_p, G)
        ntc_sub = torch.from_numpy(ntc_mat[ntc_idx_cells]).float()     # (n_n, G)

        # Per-gene tokens: transpose to (G, n_cells) then tokenize batched.
        pert_tokens = samples_to_quantile_grid(pert_sub.T, self.grid_size)
        ntc_tokens = samples_to_quantile_grid(ntc_sub.T, self.grid_size)

        return ntc_tokens, pert_tokens, int(pert_idx)

    def compute_delta_mean_profiles(self) -> torch.Tensor:
        """Return (P, n_expression_genes) delta-mean reward profile tensor."""
        ntc_mean = self._ntc_cells.mean(axis=0)  # (G,)
        profiles = np.zeros((len(self.perturbation_names), ntc_mean.shape[0]), dtype=np.float32)
        for i, cells in enumerate(self._pert_cells):
            profiles[i] = cells.mean(axis=0) - ntc_mean
        return torch.from_numpy(profiles)
