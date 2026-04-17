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
        val_fraction: float = 0.0,
        split_seed: int = 123,
        mode: str = "train",
        return_cells: bool = False,
        singles_only: bool = False,
    ) -> None:
        """See class docstring.

        Additional args (held-out cell split):
            val_fraction: If > 0, reserve this fraction of each pert's cells
                (and NTC cells) for a held-out evaluation pool. Deterministic
                given split_seed.
            split_seed: Seed for the train/val cell split (shared between
                train and val datasets so they partition the same cells).
            mode: "train" samples from the train pool; "val" samples from the
                val pool. Passes silently when val_fraction == 0 (always
                "train").
            return_cells: If True, ``__getitem__`` returns raw (n_cells, G)
                cell matrices instead of (G, K) quantile tokens. Used by the
                per-cell set-transformer classifier in ``rl_cell_model``.
            singles_only: If True, drop perts whose name contains ``_``
                (paired / combo perts in the Norman 2019 dual-CRISPRa
                convention). Leaves singles + the control NTC pool.
        """
        super().__init__()
        self.n_cells_per_pert = n_cells_per_pert
        self.n_cells_ntc = n_cells_ntc
        self.grid_size = grid_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        assert mode in ("train", "val"), f"mode must be train or val, got {mode}"
        self.mode = mode
        self.return_cells = return_cells

        pert_col = adata.obs[perturbation_key].astype(str)
        is_control = pert_col.str.lower().str.contains(control_label.lower())
        if int(is_control.sum()) < min_cells:
            raise ValueError(
                f"Not enough NTC cells matching '{control_label}': "
                f"found {int(is_control.sum())}, need >= {min_cells}"
            )

        pert_counts = pert_col[~is_control].value_counts()
        valid_perts = pert_counts[pert_counts >= min_cells].index.tolist()
        if singles_only:
            valid_perts = [p for p in valid_perts if "_" not in p]
        self.perturbation_names = sorted(valid_perts)
        self.singles_only = singles_only

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Pre-densify per pert and the NTC pool once. Always keep the FULL
        # (un-split) cell arrays for oracle profile computation; the
        # sampling pool (self._pert_cells / self._ntc_cells) is the split subset.
        split_rng = np.random.default_rng(split_seed)
        self._pert_cells_full: list[np.ndarray] = []
        self._pert_cells: list[np.ndarray] = []
        for p in self.perturbation_names:
            mask = (pert_col == p).values
            cells = X[mask]
            self._pert_cells_full.append(cells)
            if val_fraction > 0.0:
                idx = split_rng.permutation(cells.shape[0])
                n_val = max(1, int(round(val_fraction * cells.shape[0])))
                if mode == "train":
                    cells = cells[idx[n_val:]]
                else:
                    cells = cells[idx[:n_val]]
            self._pert_cells.append(cells)

        ntc_full = X[is_control.values]
        self._ntc_cells_full: np.ndarray = ntc_full
        if val_fraction > 0.0:
            idx = split_rng.permutation(ntc_full.shape[0])
            n_val = max(1, int(round(val_fraction * ntc_full.shape[0])))
            if mode == "train":
                ntc_split = ntc_full[idx[n_val:]]
            else:
                ntc_split = ntc_full[idx[:n_val]]
        else:
            ntc_split = ntc_full
        self._ntc_cells: np.ndarray = ntc_split

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

        if self.return_cells:
            return ntc_sub, pert_sub, int(pert_idx)

        # Per-gene tokens: transpose to (G, n_cells) then tokenize batched.
        pert_tokens = samples_to_quantile_grid(pert_sub.T, self.grid_size)
        ntc_tokens = samples_to_quantile_grid(ntc_sub.T, self.grid_size)

        return ntc_tokens, pert_tokens, int(pert_idx)

    def compute_delta_mean_profiles(self) -> torch.Tensor:
        """Return (P, n_expression_genes) delta-mean reward profile tensor.

        Uses the FULL un-split cell pools to give an oracle estimate that
        does not depend on the train/val split. The profile is a property of
        the perturbation's effect, not of the training data.
        """
        ntc_mean = self._ntc_cells_full.mean(axis=0)  # (G,)
        profiles = np.zeros((len(self.perturbation_names), ntc_mean.shape[0]), dtype=np.float32)
        for i, cells in enumerate(self._pert_cells_full):
            profiles[i] = cells.mean(axis=0) - ntc_mean
        return torch.from_numpy(profiles)

    def compute_ntc_noise_baseline(
        self,
        profiles: torch.Tensor,
        n_cells: int,
        metric: str = "pearson",
        K: int = 200,
        quantile: float = 0.95,
        seed: int = 42,
    ) -> torch.Tensor:
        """Per-pert correlation floor from NTC-only "noise" pseudo-profiles.

        For each true pert i with profile ``profiles[i]``:
            1. Draw K independent subsamples of ``n_cells`` NTC cells.
            2. Form pseudo-profiles ``pseudo_k = mean(ntc_subsample_k) -
               mean(all_ntc)`` — a noise-only "delta".
            3. Correlate each pseudo-profile with ``profiles[i]`` under the
               requested metric.
            4. Take the ``quantile`` of that K-sized null distribution.

        Returned shape: ``(P,)`` — per-pert floor such that any predicted
        pert j with ``metric(profiles[i], profiles[j]) <= baseline[i]`` is
        indistinguishable from "predicting NTC pretending to be i."

        Args:
            profiles: (P, G) delta-mean profile table.
            n_cells: Number of NTC cells per subsample (match training n_cells_*).
            metric: "pearson" or "cosine".
            K: Number of subsamples (null size).
            quantile: Quantile of the null (0.95 = one-tailed 5% threshold).
            seed: RNG seed for reproducibility.
        """
        from dist_vae.losses import cosine_similarity, pearson_correlation

        assert metric in ("pearson", "cosine"), metric
        metric_fn = pearson_correlation if metric == "pearson" else cosine_similarity

        ntc = self._ntc_cells_full                           # (N, G)
        N = ntc.shape[0]
        if N < n_cells:
            raise ValueError(
                f"Need {n_cells} NTC cells per subsample but pool has only {N}."
            )
        ntc_mean = ntc.mean(axis=0, keepdims=True)           # (1, G)
        rng = np.random.default_rng(seed)
        pseudo = np.zeros((K, ntc.shape[1]), dtype=np.float32)
        for k in range(K):
            idx = rng.choice(N, size=n_cells, replace=False)
            pseudo[k] = ntc[idx].mean(axis=0) - ntc_mean[0]   # (G,)
        pseudo_t = torch.from_numpy(pseudo)                   # (K, G)

        # (P, 1, G) vs (1, K, G) -> (P, K)
        corr = metric_fn(
            profiles.unsqueeze(1),
            pseudo_t.unsqueeze(0),
            dim=-1,
        )
        return torch.quantile(corr, q=float(quantile), dim=1)
