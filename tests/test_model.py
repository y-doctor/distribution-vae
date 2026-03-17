"""Tests for the Distribution VAE model."""

import pytest
import torch

from dist_vae.model import DistributionDecoder, DistributionEncoder, DistributionVAE


class TestDistributionEncoder:
    def test_output_shapes(self):
        """Encoder should output mu and logvar with correct shapes."""
        raise NotImplementedError

    def test_different_grid_sizes(self):
        """Encoder should work with different grid sizes."""
        raise NotImplementedError


class TestDistributionDecoder:
    def test_output_shape(self):
        """Decoder should output grid of correct size."""
        raise NotImplementedError

    def test_monotonicity(self):
        """Decoder output must be monotonically non-decreasing."""
        raise NotImplementedError

    def test_monotonicity_many_samples(self):
        """Monotonicity should hold across many random latent vectors."""
        raise NotImplementedError


class TestDistributionVAE:
    def test_forward_shapes(self, small_model, sample_quantile_grids):
        """Forward pass should return correct output shapes."""
        raise NotImplementedError

    def test_reconstruction_is_monotonic(self, small_model, sample_quantile_grids):
        """Reconstructed quantile grids must be monotonically non-decreasing."""
        raise NotImplementedError

    def test_kl_loss_non_negative(self, small_model, sample_quantile_grids):
        """KL divergence should be non-negative."""
        raise NotImplementedError

    def test_compute_loss_keys(self, small_model, sample_quantile_grids):
        """compute_loss should return dict with expected keys."""
        raise NotImplementedError

    def test_encode_distribution(self, small_model):
        """encode_distribution convenience method should work."""
        raise NotImplementedError

    def test_decode_to_samples(self, small_model):
        """decode_to_samples convenience method should work."""
        raise NotImplementedError

    def test_samples_to_grid(self):
        """samples_to_grid static method should produce sorted grids."""
        raise NotImplementedError

    def test_reparameterize(self, small_model):
        """Reparameterization should produce same shape as input."""
        raise NotImplementedError

    def test_cpu_execution(self, small_model, sample_quantile_grids):
        """Model should work on CPU."""
        raise NotImplementedError

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, small_model, sample_quantile_grids):
        """Model should work on CUDA if available."""
        raise NotImplementedError
