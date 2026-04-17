"""Tests for dist_vae.rl_data.PerturbationClassificationDataset."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from dist_vae.rl_data import PerturbationClassificationDataset


@pytest.fixture
def tiny_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_ctrl = 400
    n_per_pert = 150
    gene_names = [f"G{i:02d}" for i in range(20)]
    perts = ["control"] + ["A", "B", "C"]
    rows = []
    pert_labels = []
    for p in perts:
        n = n_ctrl if p == "control" else n_per_pert
        mean_shift = rng.uniform(-0.5, 0.5, size=20) if p != "control" else np.zeros(20)
        rows.append(rng.normal(loc=mean_shift, scale=1.0, size=(n, 20)))
        pert_labels.extend([p] * n)
    X = np.concatenate(rows, axis=0).astype(np.float32)
    obs = pd.DataFrame({"perturbation": pert_labels})
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X, obs=obs, var=var)


def test_dataset_lengths_and_shapes(tiny_adata):
    ds = PerturbationClassificationDataset(
        tiny_adata,
        n_cells_per_pert=50,
        n_cells_ntc=50,
        grid_size=32,
        samples_per_epoch=16,
        min_cells=10,
    )
    assert len(ds) == 16
    assert sorted(ds.perturbation_names) == ["A", "B", "C"]

    ntc, pert, p_idx = ds[0]
    assert ntc.shape == (20, 32)
    assert pert.shape == (20, 32)
    assert 0 <= p_idx < 3
    assert ntc.dtype == torch.float32


def test_resampling_gives_different_grids(tiny_adata):
    ds = PerturbationClassificationDataset(
        tiny_adata,
        n_cells_per_pert=40,
        n_cells_ntc=40,
        grid_size=32,
        samples_per_epoch=8,
        min_cells=10,
    )
    # Two different idx should give different samples due to the per-idx RNG.
    ntc_a, pert_a, p_a = ds[0]
    ntc_b, pert_b, p_b = ds[3]
    if p_a == p_b:
        assert not torch.allclose(pert_a, pert_b)
    assert not torch.allclose(ntc_a, ntc_b)


def test_delta_mean_profiles(tiny_adata):
    ds = PerturbationClassificationDataset(
        tiny_adata,
        n_cells_per_pert=50,
        n_cells_ntc=50,
        grid_size=32,
        samples_per_epoch=8,
        min_cells=10,
    )
    profiles = ds.compute_delta_mean_profiles()
    assert profiles.shape == (3, 20)
    # Different perts should give different profile vectors (with high prob).
    assert not torch.allclose(profiles[0], profiles[1])


def test_gene_vocab_shapes(tiny_adata):
    ds = PerturbationClassificationDataset(
        tiny_adata,
        n_cells_per_pert=50,
        n_cells_ntc=50,
        grid_size=32,
        samples_per_epoch=4,
        min_cells=10,
    )
    vocab = ds.vocab
    assert len(vocab.expression_gene_ids) == 20
    # Perturbation targets A/B/C each add a new gene.
    assert len(vocab.names) == 20 + 3
    assert len(vocab.pert_target_gene_ids) == 3
    for ids in vocab.pert_target_gene_ids:
        assert all(0 <= i < len(vocab.names) for i in ids)
