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


# -------------------------- pearson_correlation ---------------------------


def test_pearson_self_is_one():
    from dist_vae.losses import pearson_correlation
    x = torch.randn(4, 10)
    assert torch.allclose(pearson_correlation(x, x), torch.ones(4), atol=1e-5)


def test_pearson_opposite_is_minus_one():
    from dist_vae.losses import pearson_correlation
    x = torch.randn(4, 10)
    assert torch.allclose(pearson_correlation(x, -x), -torch.ones(4), atol=1e-5)


def test_pearson_zero_for_orthogonal_after_centering():
    """pearson(a, b) = 0 when a and b are orthogonal *after* mean-centering."""
    from dist_vae.losses import pearson_correlation
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    y = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    # Both already zero-mean; cos(x, y) = 0, so pearson should also be 0.
    assert torch.allclose(pearson_correlation(x, y), torch.zeros(1), atol=1e-6)


def test_pearson_invariant_to_constant_offset():
    """Pearson is mean-centered, so adding a constant shouldn't change it."""
    from dist_vae.losses import pearson_correlation
    x = torch.randn(3, 20)
    y = torch.randn(3, 20)
    a = pearson_correlation(x, y)
    b = pearson_correlation(x + 7.5, y - 3.2)
    assert torch.allclose(a, b, atol=1e-5)


def test_pearson_matches_numpy_corrcoef():
    """Sanity: pearson_correlation matches numpy.corrcoef on known inputs."""
    import numpy as np
    from dist_vae.losses import pearson_correlation
    rng = np.random.default_rng(0)
    a = rng.normal(size=20).astype("float32")
    b = rng.normal(size=20).astype("float32")
    expected = float(np.corrcoef(a, b)[0, 1])
    out = pearson_correlation(torch.from_numpy(a), torch.from_numpy(b))
    assert abs(float(out) - expected) < 1e-5
