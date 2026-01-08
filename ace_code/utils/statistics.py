"""
Statistical utilities for ACE-Code feature analysis.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy import stats


def compute_t_statistic(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute t-statistic for comparing two groups of activations.

    This is used to identify features that discriminate between
    "correct" and "buggy" code by measuring the statistical
    significance of activation differences.

    Args:
        positive_activations: Activations from positive (correct) examples
            Shape: (n_positive, d_features)
        negative_activations: Activations from negative (buggy) examples
            Shape: (n_negative, d_features)

    Returns:
        t-statistics for each feature, shape: (d_features,)
    """
    pos = positive_activations.detach().cpu().numpy()
    neg = negative_activations.detach().cpu().numpy()

    # Compute t-statistic for each feature
    t_stats, _ = stats.ttest_ind(pos, neg, axis=0, equal_var=False)

    return torch.from_numpy(t_stats).float()


def normalize_vector(
    vector: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L2 normalize a vector or batch of vectors.

    Args:
        vector: Input vector(s)
        dim: Dimension to normalize along
        eps: Small constant for numerical stability

    Returns:
        Normalized vector(s)
    """
    norm = torch.linalg.norm(vector, dim=dim, keepdim=True)
    return vector / (norm + eps)


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    Args:
        a: First vector(s)
        b: Second vector(s)
        dim: Dimension to compute similarity along

    Returns:
        Cosine similarity value(s)
    """
    a_norm = normalize_vector(a, dim=dim)
    b_norm = normalize_vector(b, dim=dim)
    return (a_norm * b_norm).sum(dim=dim)


def top_k_indices(
    values: torch.Tensor,
    k: int,
    largest: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k indices and values.

    Args:
        values: Input tensor
        k: Number of top values to return
        largest: If True, return largest values; if False, return smallest

    Returns:
        Tuple of (values, indices)
    """
    return torch.topk(values, k=k, largest=largest)


def compute_effect_size(
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Cohen's d effect size.

    Args:
        positive: Positive group activations
        negative: Negative group activations

    Returns:
        Effect size for each feature
    """
    pos_mean = positive.mean(dim=0)
    neg_mean = negative.mean(dim=0)

    pos_var = positive.var(dim=0)
    neg_var = negative.var(dim=0)

    n_pos = positive.shape[0]
    n_neg = negative.shape[0]

    # Pooled standard deviation
    pooled_std = torch.sqrt(
        ((n_pos - 1) * pos_var + (n_neg - 1) * neg_var) / (n_pos + n_neg - 2)
    )

    # Cohen's d
    d = (pos_mean - neg_mean) / (pooled_std + 1e-8)

    return d


def find_significant_features(
    t_stats: torch.Tensor,
    threshold: float = 2.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find features with significant t-statistics.

    Args:
        t_stats: T-statistics for each feature
        threshold: Minimum absolute t-statistic threshold
        top_k: If provided, return only top-k features

    Returns:
        Tuple of (feature_indices, t_stat_values)
    """
    # Handle NaN values
    t_stats = torch.nan_to_num(t_stats, nan=0.0)

    if top_k is not None:
        # Return top-k by absolute value
        abs_t = t_stats.abs()
        values, indices = torch.topk(abs_t, k=min(top_k, len(t_stats)))
        return indices, t_stats[indices]
    else:
        # Return all above threshold
        mask = t_stats.abs() > threshold
        indices = torch.where(mask)[0]
        return indices, t_stats[indices]
