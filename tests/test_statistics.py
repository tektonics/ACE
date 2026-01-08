"""Tests for statistical utilities."""

import pytest
import torch

from ace_code.utils.statistics import (
    compute_t_statistic,
    normalize_vector,
    cosine_similarity,
    top_k_indices,
    compute_effect_size,
    find_significant_features,
)


class TestNormalizeVector:
    """Tests for normalize_vector."""

    def test_normalizes_to_unit_length(self):
        vec = torch.tensor([3.0, 4.0])
        normalized = normalize_vector(vec)

        assert torch.isclose(normalized.norm(), torch.tensor(1.0), atol=1e-6)

    def test_batch_normalization(self):
        batch = torch.randn(5, 10)
        normalized = normalize_vector(batch, dim=-1)

        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)

    def test_handles_zero_vector(self):
        vec = torch.zeros(5)
        normalized = normalize_vector(vec)

        # Should not produce NaN
        assert not torch.isnan(normalized).any()


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self):
        vec = torch.randn(10)
        sim = cosine_similarity(vec, vec)

        assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        sim = cosine_similarity(a, b)

        assert torch.isclose(sim, torch.tensor(0.0), atol=1e-6)

    def test_opposite_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        sim = cosine_similarity(a, b)

        assert torch.isclose(sim, torch.tensor(-1.0), atol=1e-6)


class TestTopKIndices:
    """Tests for top_k_indices."""

    def test_finds_top_k(self):
        values = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
        top_vals, top_idx = top_k_indices(values, k=3)

        assert len(top_idx) == 3
        assert 1 in top_idx  # 5.0 is largest
        assert top_vals[0] == 5.0

    def test_finds_smallest_k(self):
        values = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
        bot_vals, bot_idx = top_k_indices(values, k=2, largest=False)

        assert len(bot_idx) == 2
        assert 0 in bot_idx  # 1.0 is smallest


class TestComputeTStatistic:
    """Tests for compute_t_statistic."""

    def test_basic_t_statistic(self):
        # Group 1: mean ~= 0
        pos = torch.randn(100, 10)
        # Group 2: mean ~= 1 for first feature
        neg = torch.randn(100, 10)
        neg[:, 0] += 3.0  # Add clear difference to first feature

        t_stats = compute_t_statistic(pos, neg)

        assert len(t_stats) == 10
        # First feature should have large negative t (neg > pos)
        assert t_stats[0] < -2.0


class TestFindSignificantFeatures:
    """Tests for find_significant_features."""

    def test_finds_above_threshold(self):
        t_stats = torch.tensor([0.5, 3.0, -2.5, 1.0, -4.0])
        indices, values = find_significant_features(t_stats, threshold=2.0)

        assert 1 in indices.tolist()  # 3.0
        assert 2 in indices.tolist()  # -2.5
        assert 4 in indices.tolist()  # -4.0

    def test_returns_top_k(self):
        t_stats = torch.tensor([0.5, 3.0, -2.5, 1.0, -4.0])
        indices, values = find_significant_features(t_stats, top_k=2)

        assert len(indices) == 2
        assert 4 in indices.tolist()  # -4.0 has highest absolute value


class TestComputeEffectSize:
    """Tests for compute_effect_size (Cohen's d)."""

    def test_effect_size_same_distributions(self):
        pos = torch.randn(50, 5)
        neg = torch.randn(50, 5)

        d = compute_effect_size(pos, neg)

        # Effect size should be small for same distributions
        assert d.abs().mean() < 0.5

    def test_effect_size_different_distributions(self):
        pos = torch.zeros(50, 5)
        neg = torch.ones(50, 5) * 2

        d = compute_effect_size(pos, neg)

        # Effect size should be large
        assert d.abs().mean() > 1.0
