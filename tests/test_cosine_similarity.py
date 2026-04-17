"""Tests for dist_vae.losses.cosine_similarity."""

import torch

from dist_vae.losses import cosine_similarity


def test_identity_equals_one() -> None:
    x = torch.randn(4, 16)
    assert torch.allclose(cosine_similarity(x, x), torch.ones(4), atol=1e-5)


def test_opposite_equals_minus_one() -> None:
    x = torch.randn(4, 16)
    assert torch.allclose(cosine_similarity(x, -x), -torch.ones(4), atol=1e-5)


def test_orthogonal_is_zero() -> None:
    x = torch.tensor([[1.0, 0.0, 0.0]])
    y = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(cosine_similarity(x, y), torch.zeros(1), atol=1e-6)


def test_broadcasting_3d() -> None:
    x = torch.randn(2, 5, 8)
    y = torch.randn(2, 1, 8)
    out = cosine_similarity(x, y, dim=-1)
    assert out.shape == (2, 5)
    for i in range(2):
        for j in range(5):
            ref = cosine_similarity(x[i, j], y[i, 0])
            assert torch.allclose(out[i, j], ref, atol=1e-5)


def test_bounded_in_unit_interval() -> None:
    x = torch.randn(100, 32)
    y = torch.randn(100, 32)
    out = cosine_similarity(x, y)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_zero_vector_does_not_nan() -> None:
    x = torch.zeros(3, 8)
    y = torch.randn(3, 8)
    out = cosine_similarity(x, y)
    assert torch.all(torch.isfinite(out))
