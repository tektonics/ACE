"""Tests for SAE Trainer."""

import pytest
import torch
from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.sae_anomaly.trainer import SAETrainer, SAETrainerConfig


class TestSAETrainer:
    """Tests for SAE training."""

    @pytest.fixture
    def sae(self):
        """Create SAE for testing."""
        return JumpReLUSAE(
            d_model=256,
            expansion_factor=4,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.fixture
    def trainer(self, sae):
        """Create trainer for testing."""
        config = SAETrainerConfig(
            learning_rate=1e-3,
            batch_size=32,
            l1_coefficient=1e-3,
        )
        return SAETrainer(sae, config)

    @pytest.fixture
    def sample_activations(self):
        """Create sample activation data."""
        # Simulate 1000 activation vectors
        return torch.randn(1000, 256)

    def test_init(self, trainer, sae):
        """Test trainer initialization."""
        assert trainer.sae is sae
        assert trainer.global_step == 0
        assert trainer.optimizer is not None

    def test_compute_loss(self, trainer, sample_activations):
        """Test loss computation."""
        batch = sample_activations[:32]
        loss, metrics = trainer._compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "loss/total" in metrics
        assert "loss/mse" in metrics
        assert "loss/sparsity" in metrics
        assert "features/n_active" in metrics

    def test_train_step(self, trainer, sample_activations):
        """Test single training step."""
        batch = sample_activations[:32]

        initial_params = {
            name: param.clone()
            for name, param in trainer.sae.named_parameters()
        }

        metrics = trainer.train_step(batch)

        # Check parameters changed
        for name, param in trainer.sae.named_parameters():
            assert not torch.equal(param, initial_params[name]), f"{name} didn't change"

        assert trainer.global_step == 1
        assert "loss/total" in metrics

    def test_train_reduces_loss(self, trainer, sample_activations):
        """Test that training reduces loss over time."""
        # Get initial loss
        batch = sample_activations[:32]
        _, initial_metrics = trainer._compute_loss(batch)
        initial_loss = initial_metrics["loss/mse"]

        # Train for a few steps
        for _ in range(50):
            trainer.train_step(batch)

        # Check loss reduced
        _, final_metrics = trainer._compute_loss(batch)
        final_loss = final_metrics["loss/mse"]

        # Should improve reconstruction
        assert final_loss < initial_loss, "Training should reduce MSE loss"

    def test_train_full(self, trainer, sample_activations):
        """Test full training loop."""
        history = trainer.train(sample_activations, n_epochs=1)

        assert "loss/total" in history
        assert len(history["loss/total"]) > 0
        assert trainer.global_step > 0

    def test_decoder_normalization(self, trainer, sample_activations):
        """Test that decoder columns are normalized."""
        batch = sample_activations[:32]
        trainer.train_step(batch)

        # Check decoder norms
        norms = trainer.sae.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sparsity_with_l1(self, sae, sample_activations):
        """Test that L1 coefficient affects sparsity."""
        # High L1 coefficient
        config_high = SAETrainerConfig(l1_coefficient=0.1, batch_size=32)
        trainer_high = SAETrainer(sae, config_high)

        # Train a bit
        for _ in range(20):
            trainer_high.train_step(sample_activations[:32])

        # Check sparsity
        features = sae.encode(sample_activations[:32])
        sparsity = (features == 0).float().mean()

        # With high L1, should be fairly sparse
        assert sparsity > 0.5, "High L1 should produce sparse features"


class TestSAETrainerConfig:
    """Tests for trainer configuration."""

    def test_default_values(self):
        """Test default configuration."""
        config = SAETrainerConfig()

        assert config.expansion_factor == 8
        assert config.learning_rate == 3e-4
        assert config.batch_size == 4096
        assert config.l1_coefficient == 5e-3

    def test_custom_values(self):
        """Test custom configuration."""
        config = SAETrainerConfig(
            learning_rate=1e-4,
            l1_coefficient=0.01,
        )

        assert config.learning_rate == 1e-4
        assert config.l1_coefficient == 0.01
