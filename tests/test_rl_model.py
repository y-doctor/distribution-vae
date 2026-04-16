"""Tests for dist_vae.rl_model.PerturbationClassifier."""

import torch

from dist_vae.rl_model import PerturbationClassifier


def _make_model(n_perts: int = 3, n_all_genes: int = 12, grid_size: int = 16):
    targets = [[0], [1, 2], [3]]
    model = PerturbationClassifier(
        n_all_genes=n_all_genes,
        n_perts=n_perts,
        pert_target_gene_ids=targets,
        grid_size=grid_size,
        d_embed=8,
        hidden_dim=16,
        d_feat=8,
    )
    return model


def test_forward_shapes() -> None:
    B, G, K = 2, 5, 16
    model = _make_model(grid_size=K)
    ntc = torch.randn(B, G, K)
    pert = torch.randn(B, G, K)
    gene_ids = torch.arange(G, dtype=torch.long)
    logits = model(ntc, pert, gene_ids)
    assert logits.shape == (B, 3)
    assert torch.isfinite(logits).all()


def test_compound_pert_averages_embeddings() -> None:
    # Pert 1 targets genes {1, 2}: its pert-embedding should equal mean.
    torch.manual_seed(0)
    model = _make_model()
    emb = model.pert_embeddings()
    expected_1 = 0.5 * (model.gene_embed.weight[1] + model.gene_embed.weight[2])
    assert torch.allclose(emb[1], expected_1, atol=1e-6)


def test_gradient_flows_to_embedding() -> None:
    B, G, K = 2, 5, 16
    model = _make_model(grid_size=K)
    ntc = torch.randn(B, G, K)
    pert = torch.randn(B, G, K)
    gene_ids = torch.arange(G, dtype=torch.long)
    logits = model(ntc, pert, gene_ids)
    loss = logits.sum()
    loss.backward()
    assert model.gene_embed.weight.grad is not None
    assert model.gene_embed.weight.grad.abs().sum() > 0.0


def test_determinism_for_fixed_seed() -> None:
    torch.manual_seed(42)
    model = _make_model()
    ntc = torch.randn(2, 5, 16)
    pert = torch.randn(2, 5, 16)
    gene_ids = torch.arange(5)
    out1 = model(ntc, pert, gene_ids)
    out2 = model(ntc, pert, gene_ids)
    assert torch.allclose(out1, out2)
