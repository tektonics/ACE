"""Statistical utilities for SAE feature analysis."""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from scipy import stats


def welch_t_test(
    group1: Union[np.ndarray, torch.Tensor],
    group2: Union[np.ndarray, torch.Tensor],
) -> Tuple[float, float]:
    """
    Perform Welch's t-test for unequal variances.

    Args:
        group1: First group of samples
        group2: Second group of samples

    Returns:
        Tuple of (t-statistic, p-value)
    """
    if isinstance(group1, torch.Tensor):
        group1 = group1.cpu().numpy()
    if isinstance(group2, torch.Tensor):
        group2 = group2.cpu().numpy()

    # Handle edge cases
    if len(group1) < 2 or len(group2) < 2:
        return 0.0, 1.0

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return float(t_stat), float(p_value)


def compute_t_statistic(
    activations_incorrect: Union[np.ndarray, torch.Tensor],
    activations_correct: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute t-statistic for incorrectness feature identification.

    This implements the formula from Tahimic & Cheng (2025):
    t^{incorrect}_{l,j} = (μ(incorrect) - μ(correct)) /
                          sqrt(σ²(incorrect)/N_incorrect + σ²(correct)/N_correct)

    Args:
        activations_incorrect: Feature activations for incorrect/buggy code samples
        activations_correct: Feature activations for correct code samples

    Returns:
        t-statistic value (positive = feature activates more for incorrect code)
    """
    if isinstance(activations_incorrect, torch.Tensor):
        activations_incorrect = activations_incorrect.cpu().numpy()
    if isinstance(activations_correct, torch.Tensor):
        activations_correct = activations_correct.cpu().numpy()

    # Ensure 1D arrays
    activations_incorrect = np.asarray(activations_incorrect).flatten()
    activations_correct = np.asarray(activations_correct).flatten()

    n_incorrect = len(activations_incorrect)
    n_correct = len(activations_correct)

    # Handle edge cases
    if n_incorrect < 2 or n_correct < 2:
        return 0.0

    # Compute means
    mu_incorrect = np.mean(activations_incorrect)
    mu_correct = np.mean(activations_correct)

    # Compute variances
    var_incorrect = np.var(activations_incorrect, ddof=1)
    var_correct = np.var(activations_correct, ddof=1)

    # Compute pooled standard error (Welch's formula)
    se = np.sqrt(var_incorrect / n_incorrect + var_correct / n_correct)

    # Avoid division by zero
    if se < 1e-10:
        return 0.0

    # Compute t-statistic
    t_stat = (mu_incorrect - mu_correct) / se

    return float(t_stat)


def compute_feature_significance(
    activations_incorrect: Union[np.ndarray, torch.Tensor],
    activations_correct: Union[np.ndarray, torch.Tensor],
    n_features: int,
    alpha: float = 0.05,
    bonferroni_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute significance of all SAE features for incorrectness detection.

    Args:
        activations_incorrect: Shape (n_incorrect_samples, n_features)
        activations_correct: Shape (n_correct_samples, n_features)
        n_features: Number of SAE features
        alpha: Significance level
        bonferroni_correction: Whether to apply Bonferroni correction

    Returns:
        Tuple of:
            - t_statistics: Array of t-statistics for each feature
            - p_values: Array of p-values for each feature
            - significant_mask: Boolean mask of significant features
    """
    if isinstance(activations_incorrect, torch.Tensor):
        activations_incorrect = activations_incorrect.cpu().numpy()
    if isinstance(activations_correct, torch.Tensor):
        activations_correct = activations_correct.cpu().numpy()

    t_statistics = np.zeros(n_features)
    p_values = np.ones(n_features)

    for j in range(n_features):
        feat_incorrect = activations_incorrect[:, j]
        feat_correct = activations_correct[:, j]

        t_stat = compute_t_statistic(feat_incorrect, feat_correct)
        _, p_val = welch_t_test(feat_incorrect, feat_correct)

        t_statistics[j] = t_stat
        p_values[j] = p_val

    # Apply correction
    if bonferroni_correction:
        corrected_alpha = alpha / n_features
    else:
        corrected_alpha = alpha

    # Features significant for incorrectness (positive t-stat, low p-value)
    significant_mask = (t_statistics > 0) & (p_values < corrected_alpha)

    return t_statistics, p_values, significant_mask


def compute_activation_frequency(
    activations: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute activation frequency for each feature.

    Used to filter out features that activate too frequently on general text.

    Args:
        activations: Shape (n_samples, n_features)
        threshold: Minimum activation value to count as "active"

    Returns:
        Array of activation frequencies (0 to 1) for each feature
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()

    n_samples = activations.shape[0]
    active_counts = np.sum(activations > threshold, axis=0)

    return active_counts / n_samples


def filter_by_frequency(
    feature_indices: np.ndarray,
    activation_frequencies: np.ndarray,
    max_frequency: float = 0.02,
) -> np.ndarray:
    """
    Filter features by activation frequency.

    Discards features that activate on > max_frequency of general text data.

    Args:
        feature_indices: Indices of candidate features
        activation_frequencies: Activation frequency for each feature
        max_frequency: Maximum allowed activation frequency (default 2%)

    Returns:
        Filtered feature indices
    """
    mask = activation_frequencies[feature_indices] <= max_frequency
    return feature_indices[mask]
