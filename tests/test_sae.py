"""Tests for JumpReLU Sparse Autoencoder."""

import pytest
import torch
import numpy as np
from ace_code.sae_anomaly.sae import JumpReLUSAE, JumpReLU


class TestJumpReLU:
    """Tests for JumpReLU activation function."""

    def test_init(self):
        """Test JumpReLU initialization."""
        n_features = 100
        activation = JumpReLU(n_features)
        assert activation.threshold.shape == (n_features,)

    def test_forward_basic(self):
        """Test basic forward pass."""
        n_features = 100
        activation = JumpReLU(n_features, initial_threshold=0.5)

        # Create input above and below threshold
        z = torch.tensor([[0.3, 0.7, 0.5, 1.0, 0.1]])
        result = activation(z)

        # Values <= 0.5 should be 0, values > 0.5 should pass through
        assert result[0, 0].item() == 0.0  # 0.3 <= 0.5
        assert result[0, 1].item() == 0.7  # 0.7 > 0.5
        assert result[0, 2].item() == 0.0  # 0.5 <= 0.5 (not strictly greater)
        assert result[0, 3].item() == 1.0  # 1.0 > 0.5
        assert result[0, 4].item() == 0.0  # 0.1 <= 0.5

    def test_sparsity(self):
        """Test that JumpReLU produces sparse outputs."""
        n_features = 1000
        activation = JumpReLU(n_features, initial_threshold=0.5)

        # Random input centered around 0.5
        z = torch.randn(32, n_features) * 0.3 + 0.5
        result = activation(z)

        # Should have significant sparsity
        sparsity = (result == 0).float().mean()
        assert sparsity > 0.3  # At least 30% zeros


class TestJumpReLUSAE:
    """Tests for JumpReLU Sparse Autoencoder."""

    @pytest.fixture
    def sae(self):
        """Create SAE for testing."""
        return JumpReLUSAE(
            d_model=256,
            expansion_factor=8,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, sae):
        """Test SAE initialization."""
        assert sae.d_model == 256
        assert sae.n_latents == 256 * 8
        assert sae.W_enc.shape == (256, 2048)
        assert sae.W_dec.shape == (2048, 256)
        assert sae.b_enc.shape == (2048,)
        assert sae.b_dec.shape == (256,)

    def test_encode(self, sae):
        """Test encoding produces sparse features."""
        x = torch.randn(4, 256)
        features = sae.encode(x)

        assert features.shape == (4, 2048)
        # Features should be non-negative (due to JumpReLU)
        assert (features >= 0).all()

    def test_decode(self, sae):
        """Test decoding reconstructs dimensions."""
        features = torch.rand(4, 2048)
        x_hat = sae.decode(features)

        assert x_hat.shape == (4, 256)

    def test_forward(self, sae):
        """Test full forward pass."""
        x = torch.randn(4, 256)

        # Without features
        x_hat = sae(x)
        assert x_hat.shape == x.shape

        # With features
        x_hat, features = sae(x, return_features=True)
        assert x_hat.shape == x.shape
        assert features.shape == (4, 2048)

    def test_get_feature_activations(self, sae):
        """Test getting feature activations."""
        x = torch.randn(4, 256)
        features = sae.get_feature_activations(x)

        assert features.shape == (4, 2048)
        assert (features >= 0).all()

    def test_compute_loss(self, sae):
        """Test loss computation."""
        x = torch.randn(4, 256)

        # Total loss
        loss = sae.compute_loss(x, l1_coefficient=0.01)
        assert loss.item() > 0

        # Component losses
        losses = sae.compute_loss(x, l1_coefficient=0.01, return_components=True)
        assert "total" in losses
        assert "mse" in losses
        assert "l1" in losses
        assert "sparsity" in losses

    def test_sparsity_stats(self, sae):
        """Test sparsity statistics."""
        x = torch.randn(4, 256)
        stats = sae.get_sparsity_stats(x)

        assert "mean_active" in stats
        assert "max_active" in stats
        assert "sparsity_ratio" in stats
        assert 0 <= stats["sparsity_ratio"] <= 1

    def test_save_load(self, sae, tmp_path):
        """Test saving and loading SAE."""
        # Save
        save_path = tmp_path / "sae.pt"
        sae.save(save_path)
        assert save_path.exists()

        # Load
        loaded_sae = JumpReLUSAE.load(save_path, device="cpu")
        assert loaded_sae.d_model == sae.d_model
        assert loaded_sae.n_latents == sae.n_latents

        # Check weights match
        x = torch.randn(4, 256)
        original_features = sae.encode(x)
        loaded_features = loaded_sae.encode(x)
        assert torch.allclose(original_features, loaded_features)

    def test_batch_processing(self, sae):
        """Test batch processing."""
        batch_sizes = [1, 4, 16, 64]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 256)
            x_hat, features = sae(x, return_features=True)

            assert x_hat.shape == (batch_size, 256)
            assert features.shape == (batch_size, 2048)

    def test_gradient_flow(self, sae):
        """Test that gradients flow through SAE."""
        x = torch.randn(4, 256, requires_grad=True)
        loss = sae.compute_loss(x)
        loss.backward()

        assert x.grad is not None
        assert sae.W_enc.grad is not None
        assert sae.W_dec.grad is not None
