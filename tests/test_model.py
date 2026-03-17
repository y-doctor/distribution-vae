"""Tests for the Distribution VAE model."""

import pytest
import torch

from dist_vae.model import DistributionDecoder, DistributionEncoder, DistributionVAE


class TestDistributionEncoder:
    def test_output_shapes(self):
        enc = DistributionEncoder(grid_size=64, latent_dim=16, hidden_dim=32)
        x = torch.randn(4, 64)
        mu, logvar = enc(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_different_grid_sizes(self):
        for gs in [32, 64, 128, 256]:
            enc = DistributionEncoder(grid_size=gs, latent_dim=8, hidden_dim=32)
            x = torch.randn(2, gs)
            mu, logvar = enc(x)
            assert mu.shape == (2, 8)


class TestDistributionDecoder:
    def test_output_shape(self):
        dec = DistributionDecoder(grid_size=64, latent_dim=8, hidden_dim=32)
        z = torch.randn(4, 8)
        out = dec(z)
        assert out.shape == (4, 64)

    def test_monotonicity(self):
        dec = DistributionDecoder(grid_size=64, latent_dim=8, hidden_dim=32)
        z = torch.randn(4, 8)
        out = dec(z)
        diffs = out[:, 1:] - out[:, :-1]
        assert (diffs >= 0).all(), "Decoder output is not monotonically non-decreasing"

    def test_monotonicity_many_samples(self):
        dec = DistributionDecoder(grid_size=128, latent_dim=16, hidden_dim=64)
        z = torch.randn(100, 16)
        out = dec(z)
        diffs = out[:, 1:] - out[:, :-1]
        assert (diffs >= 0).all(), "Monotonicity violated in batch of 100"


class TestDistributionVAE:
    def test_forward_shapes(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8)
        x = sample_quantile_grids(4, 64)
        recon, mu, logvar, z = model(x)
        assert recon.shape == (4, 64)
        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)
        assert z.shape == (4, 8)

    def test_reconstruction_is_monotonic(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8)
        x = sample_quantile_grids(8, 64)
        recon, _, _, _ = model(x)
        diffs = recon[:, 1:] - recon[:, :-1]
        assert (diffs >= 0).all()

    def test_kl_loss_non_negative(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8)
        x = sample_quantile_grids(4, 64)
        recon, mu, logvar, z = model(x)
        losses = model.compute_loss(x, recon, mu, logvar)
        assert losses["kl"] >= 0

    def test_compute_loss_keys(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8)
        x = sample_quantile_grids(4, 64)
        recon, mu, logvar, z = model(x)
        losses = model.compute_loss(x, recon, mu, logvar)
        assert "total" in losses
        assert "recon" in losses
        assert "kl" in losses

    def test_encode_distribution(self, small_model):
        model = small_model(grid_size=64, latent_dim=8)
        model.eval()
        samples = torch.randn(2, 200)
        mu = model.encode_distribution(samples)
        assert mu.shape == (2, 8)

    def test_decode_to_samples(self, small_model):
        model = small_model(grid_size=64, latent_dim=8)
        z = torch.randn(2, 8)
        samples = model.decode_to_samples(z, n_samples=100)
        assert samples.shape == (2, 100)

    def test_samples_to_grid(self):
        samples = torch.randn(4, 200)
        grid = DistributionVAE.samples_to_grid(samples, grid_size=64)
        assert grid.shape == (4, 64)
        # Should be sorted
        diffs = grid[:, 1:] - grid[:, :-1]
        assert (diffs >= -1e-6).all()

    def test_reparameterize(self, small_model):
        model = small_model(grid_size=64, latent_dim=8)
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (4, 8)

    def test_cpu_execution(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8)
        x = sample_quantile_grids(4, 64)
        recon, mu, logvar, z = model(x)
        losses = model.compute_loss(x, recon, mu, logvar)
        losses["total"].backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, small_model, sample_quantile_grids):
        model = small_model(grid_size=64, latent_dim=8).cuda()
        x = sample_quantile_grids(4, 64).cuda()
        recon, mu, logvar, z = model(x)
        losses = model.compute_loss(x, recon, mu, logvar)
        assert losses["total"].device.type == "cuda"
