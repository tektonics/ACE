"""Tests for statistical utilities."""

import pytest
import numpy as np
import torch
from ace_code.utils.statistics import (
    compute_t_statistic,
    compute_feature_significance,
    compute_activation_frequency,
    filter_by_frequency,
    welch_t_test,
)


class TestTStatistic:
    """Tests for t-statistic computation."""

    def test_basic_difference(self):
        """Test t-statistic with clear difference."""
        # Group 1 has higher mean
        incorrect = np.array([5.0, 6.0, 7.0, 5.5, 6.5])
        correct = np.array([1.0, 2.0, 1.5, 2.5, 1.8])

        t_stat = compute_t_statistic(incorrect, correct)

        # Should be positive (incorrect > correct)
        assert t_stat > 0
        # Should be significant (large difference)
        assert t_stat > 5.0

    def test_no_difference(self):
        """Test t-statistic with no difference."""
        group1 = np.array([1.0, 2.0, 3.0, 2.0, 2.5])
        group2 = np.array([1.0, 2.0, 3.0, 2.0, 2.5])

        t_stat = compute_t_statistic(group1, group2)

        # Should be close to 0
        assert abs(t_stat) < 0.1

    def test_negative_difference(self):
        """Test t-statistic with negative difference."""
        incorrect = np.array([1.0, 2.0, 1.5])
        correct = np.array([5.0, 6.0, 5.5])

        t_stat = compute_t_statistic(incorrect, correct)

        # Should be negative (incorrect < correct)
        assert t_stat < 0

    def test_torch_input(self):
        """Test with torch tensors."""
        incorrect = torch.tensor([5.0, 6.0, 7.0])
        correct = torch.tensor([1.0, 2.0, 1.5])

        t_stat = compute_t_statistic(incorrect, correct)

        assert isinstance(t_stat, float)
        assert t_stat > 0

    def test_edge_case_small_sample(self):
        """Test with very small samples."""
        incorrect = np.array([5.0])
        correct = np.array([1.0])

        # Should handle gracefully
        t_stat = compute_t_statistic(incorrect, correct)
        assert t_stat == 0.0  # Not enough samples


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        group1 = np.array([10.0, 11.0, 12.0, 10.5, 11.5] * 10)
        group2 = np.array([1.0, 2.0, 1.5, 2.5, 1.8] * 10)

        t_stat, p_value = welch_t_test(group1, group2)

        assert t_stat > 0
        assert p_value < 0.01  # Highly significant

    def test_no_significant_difference(self):
        """Test detection of no significant difference."""
        np.random.seed(42)
        group1 = np.random.normal(5.0, 1.0, 50)
        group2 = np.random.normal(5.0, 1.0, 50)

        _, p_value = welch_t_test(group1, group2)

        # Should not be significant
        assert p_value > 0.05


class TestFeatureSignificance:
    """Tests for multi-feature significance computation."""

    def test_basic_significance(self):
        """Test computing significance for multiple features."""
        n_samples = 100
        n_features = 10

        # Create data where first 3 features are significant
        np.random.seed(42)
        correct = np.random.normal(0, 1, (n_samples, n_features))
        incorrect = np.random.normal(0, 1, (n_samples, n_features))

        # Make first 3 features significantly different
        incorrect[:, 0] += 2.0
        incorrect[:, 1] += 1.5
        incorrect[:, 2] += 1.8

        t_stats, p_values, sig_mask = compute_feature_significance(
            incorrect, correct, n_features,
            alpha=0.05,
            bonferroni_correction=False,
        )

        assert len(t_stats) == n_features
        assert len(p_values) == n_features
        assert len(sig_mask) == n_features

        # First 3 features should have positive t-stats
        assert t_stats[0] > 0
        assert t_stats[1] > 0
        assert t_stats[2] > 0

        # First 3 features should be significant
        assert sig_mask[0] == True
        assert sig_mask[1] == True
        assert sig_mask[2] == True

    def test_bonferroni_correction(self):
        """Test that Bonferroni correction is more conservative."""
        n_samples = 50
        n_features = 100

        np.random.seed(42)
        correct = np.random.normal(0, 1, (n_samples, n_features))
        incorrect = np.random.normal(0, 1, (n_samples, n_features))

        # Add mild effect to first feature
        incorrect[:, 0] += 0.8

        _, _, sig_without = compute_feature_significance(
            incorrect, correct, n_features,
            bonferroni_correction=False,
        )

        _, _, sig_with = compute_feature_significance(
            incorrect, correct, n_features,
            bonferroni_correction=True,
        )

        # Bonferroni should find fewer significant features
        assert sig_with.sum() <= sig_without.sum()


class TestActivationFrequency:
    """Tests for activation frequency computation."""

    def test_basic_frequency(self):
        """Test basic frequency computation."""
        activations = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.0, 0.8],
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        frequencies = compute_activation_frequency(activations, threshold=0.0)

        # Feature 0: 2/4 = 0.5
        assert frequencies[0] == 0.5
        # Feature 1: 0/4 = 0.0
        assert frequencies[1] == 0.0
        # Feature 2: 3/4 = 0.75
        assert frequencies[2] == 0.75

    def test_torch_input(self):
        """Test with torch tensor input."""
        activations = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        frequencies = compute_activation_frequency(activations)

        assert frequencies[0] == 0.5
        assert frequencies[1] == 0.5


class TestFrequencyFilter:
    """Tests for frequency-based filtering."""

    def test_basic_filter(self):
        """Test basic frequency filtering."""
        feature_indices = np.array([0, 1, 2, 3, 4])
        frequencies = np.array([0.01, 0.05, 0.02, 0.10, 0.015])

        # Filter to max 2%
        filtered = filter_by_frequency(feature_indices, frequencies, max_frequency=0.02)

        # Should keep indices 0, 2, 4 (frequencies <= 0.02)
        assert 0 in filtered
        assert 1 not in filtered  # 0.05 > 0.02
        assert 2 in filtered
        assert 3 not in filtered  # 0.10 > 0.02
        assert 4 in filtered

    def test_all_filtered(self):
        """Test when all features are filtered out."""
        feature_indices = np.array([0, 1, 2])
        frequencies = np.array([0.5, 0.6, 0.7])

        filtered = filter_by_frequency(feature_indices, frequencies, max_frequency=0.02)

        assert len(filtered) == 0

    def test_none_filtered(self):
        """Test when no features are filtered out."""
        feature_indices = np.array([0, 1, 2])
        frequencies = np.array([0.01, 0.01, 0.01])

        filtered = filter_by_frequency(feature_indices, frequencies, max_frequency=0.02)

        assert len(filtered) == 3
