"""Tests for the steering intervention module."""

import pytest
import torch

from ace_code.intervention.steering import SteeringVector, SteeringConfig


class TestSteeringVector:
    """Tests for SteeringVector class."""

    def test_creation(self):
        vector = torch.randn(768)
        sv = SteeringVector(vector=vector, layer=10)

        assert sv.layer == 10
        assert sv.vector.shape == (768,)

    def test_normalize(self):
        vector = torch.tensor([3.0, 4.0])  # norm = 5
        sv = SteeringVector(vector=vector, layer=5)

        normalized = sv.normalize()

        assert torch.isclose(normalized.vector.norm(), torch.tensor(1.0), atol=1e-6)
        assert normalized._normalized

    def test_scale(self):
        vector = torch.ones(10)
        sv = SteeringVector(vector=vector, layer=5)

        scaled = sv.scale(2.5)

        assert torch.allclose(scaled.vector, torch.ones(10) * 2.5)

    def test_to_device(self):
        vector = torch.randn(100)
        sv = SteeringVector(vector=vector, layer=5)

        sv_cpu = sv.to("cpu")
        assert sv_cpu.device == torch.device("cpu")

    def test_norm_property(self):
        vector = torch.tensor([3.0, 4.0])
        sv = SteeringVector(vector=vector, layer=5)

        assert abs(sv.norm - 5.0) < 1e-6

    def test_get_hook_fn(self):
        vector = torch.ones(768)
        sv = SteeringVector(vector=vector, layer=5)

        hook_fn = sv.get_hook_fn(strength=2.0)

        # Test hook function
        activation = torch.zeros(1, 10, 768)
        result = hook_fn(activation, None)

        # Should add 2.0 * vector to all positions
        assert torch.allclose(result, torch.ones(1, 10, 768) * 2.0)

    def test_get_hook_fn_specific_positions(self):
        vector = torch.ones(768)
        sv = SteeringVector(vector=vector, layer=5)

        hook_fn = sv.get_hook_fn(strength=1.0, positions=[0, -1])

        activation = torch.zeros(1, 5, 768)
        result = hook_fn(activation, None)

        # Only positions 0 and -1 (4) should be modified
        assert result[0, 0, 0] == 1.0
        assert result[0, 4, 0] == 1.0
        assert result[0, 2, 0] == 0.0  # Middle position unchanged


class TestSteeringConfig:
    """Tests for SteeringConfig dataclass."""

    def test_default_config(self):
        config = SteeringConfig()

        assert config.strength == 1.0
        assert config.normalize is True
        assert config.component == "resid_post"

    def test_custom_config(self):
        config = SteeringConfig(
            layers=[10, 11, 12],
            strength=0.5,
            positions=[0, 5, -1],
        )

        assert config.layers == [10, 11, 12]
        assert config.strength == 0.5
        assert len(config.positions) == 3
