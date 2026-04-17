"""Unit tests for the per-cell set-transformer classifier components."""

from __future__ import annotations

import pytest
import torch

from dist_vae.rl_cell_model import (
    CellSetTransformer,
    GeneModuleAttention,
    PertNTCCrossAttention,
    PerturbationCellClassifier,
)


def _make_gene_ids(G: int, V: int) -> torch.Tensor:
    """Map each of G expression slots to an embedding id in [0, V)."""
    return torch.arange(G, dtype=torch.long) % V


# -------------------------- GeneModuleAttention ---------------------------


def test_gene_module_attention_shape_2d():
    mod = GeneModuleAttention(n_all_genes=200, d=16, n_modules=8, n_heads=4)
    x = torch.randn(4, 150)
    gene_ids = _make_gene_ids(150, 200)
    out = mod(x, gene_ids)
    assert out.shape == (4, 8, 16)
    assert torch.isfinite(out).all()


def test_gene_module_attention_shape_3d():
    """Module attention should work over (B, n_cells, G)."""
    mod = GeneModuleAttention(n_all_genes=200, d=16, n_modules=8, n_heads=4)
    x = torch.randn(2, 7, 150)
    gene_ids = _make_gene_ids(150, 200)
    out = mod(x, gene_ids)
    assert out.shape == (2, 7, 8, 16)


def test_gene_module_attention_gradients_flow():
    mod = GeneModuleAttention(n_all_genes=100, d=8, n_modules=4, n_heads=2)
    x = torch.randn(3, 50, requires_grad=True)
    gene_ids = _make_gene_ids(50, 100)
    out = mod(x, gene_ids)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"nan grad for {name}"


def test_gene_module_attention_shared_embedding():
    """Shared gene embedding can be injected and is preserved."""
    emb = torch.nn.Embedding(50, 8)
    mod = GeneModuleAttention(
        n_all_genes=50, d=8, n_modules=4, n_heads=2, gene_embed=emb
    )
    assert mod.gene_embed is emb


def test_gene_module_attention_rejects_bad_gene_ids():
    mod = GeneModuleAttention(n_all_genes=100, d=8, n_modules=4, n_heads=2)
    x = torch.randn(2, 30)
    with pytest.raises(AssertionError):
        mod(x, _make_gene_ids(25, 100))  # wrong G


def test_gene_module_attention_queries_are_learnable():
    mod = GeneModuleAttention(n_all_genes=100, d=8, n_modules=4, n_heads=2)
    # Queries must be nn.Parameter (appear in .parameters()).
    param_ids = {id(p) for p in mod.parameters()}
    assert id(mod.module_queries) in param_ids


# -------------------------- CellSetTransformer ----------------------------


def test_cell_set_transformer_shape():
    mod = CellSetTransformer(d=16, n_heads=4, n_layers=2)
    x = torch.randn(3, 10, 16)
    out = mod(x)
    assert out.shape == (3, 10, 16)
    assert torch.isfinite(out).all()


def test_cell_set_transformer_permutation_invariant_in_pooling():
    """Self-attn output is equivariant; check a pooled mean is invariant to a permutation of input cells."""
    mod = CellSetTransformer(d=16, n_heads=4, n_layers=2, dropout=0.0)
    mod.eval()  # disable dropout for the invariance check
    x = torch.randn(1, 6, 16)
    out_a = mod(x).mean(dim=1)
    perm = torch.randperm(6)
    out_b = mod(x[:, perm]).mean(dim=1)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_cell_set_transformer_gradients_flow():
    mod = CellSetTransformer(d=8, n_heads=2, n_layers=1, dropout=0.0)
    x = torch.randn(2, 5, 8, requires_grad=True)
    out = mod(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in mod.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ------------------------ PertNTCCrossAttention ---------------------------


def test_cross_attn_shape_preserves_query_cardinality():
    """Cross-attn output has n_p queries regardless of n_n."""
    mod = PertNTCCrossAttention(d=16, n_heads=4, n_layers=2, dropout=0.0)
    pert = torch.randn(2, 10, 16)
    ntc = torch.randn(2, 23, 16)
    out = mod(pert, ntc)
    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


def test_cross_attn_handles_mismatched_stream_sizes():
    """n_p != n_n must be supported (usually is the case in practice)."""
    mod = PertNTCCrossAttention(d=8, n_heads=2, n_layers=1, dropout=0.0)
    for n_p, n_n in [(4, 4), (4, 10), (20, 5)]:
        out = mod(torch.randn(1, n_p, 8), torch.randn(1, n_n, 8))
        assert out.shape == (1, n_p, 8)


def test_cross_attn_gradients_flow_both_streams():
    mod = PertNTCCrossAttention(d=8, n_heads=2, n_layers=2, dropout=0.0)
    pert = torch.randn(2, 5, 8, requires_grad=True)
    ntc = torch.randn(2, 7, 8, requires_grad=True)
    out = mod(pert, ntc)
    out.sum().backward()
    assert pert.grad is not None and torch.isfinite(pert.grad).all()
    assert ntc.grad is not None and torch.isfinite(ntc.grad).all()


def test_cross_attn_ntc_actually_contributes():
    """If we zero out NTC, the output should differ from nonzero NTC
    (sanity: cross-attn is not a no-op)."""
    mod = PertNTCCrossAttention(d=8, n_heads=2, n_layers=1, dropout=0.0)
    mod.eval()
    pert = torch.randn(1, 4, 8)
    ntc_a = torch.randn(1, 6, 8)
    ntc_b = torch.zeros(1, 6, 8)
    out_a = mod(pert, ntc_a)
    out_b = mod(pert, ntc_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


# --------------------- PerturbationCellClassifier (full) -------------------


def _mini_classifier(
    n_all_genes: int = 60,
    n_perts: int = 5,
    d: int = 16,
    n_modules: int = 4,
) -> PerturbationCellClassifier:
    return PerturbationCellClassifier(
        n_all_genes=n_all_genes,
        n_perts=n_perts,
        d=d,
        n_modules=n_modules,
        n_cell_layers=2,
        n_cross_layers=2,
        n_heads=4,
        dropout=0.0,
    )


def test_full_classifier_forward_shape():
    model = _mini_classifier()
    G, P = 50, 5
    ntc = torch.randn(3, 8, G)
    pert = torch.randn(3, 6, G)
    gene_ids = _make_gene_ids(G, 60)
    logits = model(ntc, pert, gene_ids)
    assert logits.shape == (3, P)
    assert torch.isfinite(logits).all()


def test_full_classifier_signature_matches_grpo_trainer():
    """GRPOTrainer calls ``model(ntc, pert, gene_ids)`` — must work."""
    model = _mini_classifier()
    ntc = torch.randn(2, 4, 40)
    pert = torch.randn(2, 4, 40)
    gene_ids = _make_gene_ids(40, 60)
    logits = model(ntc, pert, gene_ids)
    assert logits.shape == (2, 5)


def test_full_classifier_gradients_flow_to_every_parameter():
    model = _mini_classifier()
    ntc = torch.randn(2, 5, 40)
    pert = torch.randn(2, 5, 40)
    gene_ids = _make_gene_ids(40, 60)
    logits = model(ntc, pert, gene_ids)
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no grad for: {missing}"
    bad = [n for n, p in model.named_parameters()
           if not torch.isfinite(p.grad).all()]
    assert not bad, f"nan/inf grad in: {bad}"


def test_full_classifier_permutation_invariant_in_cells():
    """Predictions must NOT depend on cell ordering within pert/NTC sets."""
    model = _mini_classifier()
    model.eval()
    torch.manual_seed(0)
    G = 40
    gene_ids = _make_gene_ids(G, 60)
    ntc = torch.randn(1, 10, G)
    pert = torch.randn(1, 10, G)
    out_a = model(ntc, pert, gene_ids)
    perm_n = torch.randperm(10)
    perm_p = torch.randperm(10)
    out_b = model(ntc[:, perm_n], pert[:, perm_p], gene_ids)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "classifier should be permutation-invariant in both cell sets"
    )


def test_full_classifier_parameter_count_reasonable():
    """Sanity: K=32, d=64, 2000 genes → <2M params on this small config."""
    m = PerturbationCellClassifier(
        n_all_genes=2000, n_perts=236, d=64, n_modules=32,
        n_cell_layers=2, n_cross_layers=2, n_heads=4, dropout=0.0,
    )
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 2_000_000, (
        f"parameter count {n:,} outside expected range"
    )


def test_full_classifier_trainable_step():
    """One optimizer step should reduce a fixed-label CE loss on random input."""
    model = _mini_classifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    G = 40
    gene_ids = _make_gene_ids(G, 60)
    torch.manual_seed(0)
    ntc = torch.randn(2, 8, G)
    pert = torch.randn(2, 8, G)
    target = torch.tensor([2, 3])

    loss_before = torch.nn.functional.cross_entropy(
        model(ntc, pert, gene_ids), target
    ).item()
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(ntc, pert, gene_ids), target
        )
        loss.backward()
        opt.step()
    loss_after = loss.item()
    assert loss_after < loss_before, (
        f"model failed to fit a tiny sample: loss {loss_before:.3f} → {loss_after:.3f}"
    )
