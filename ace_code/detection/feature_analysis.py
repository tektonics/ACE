"""
Feature Analysis Utilities

This module provides tools for analyzing and interpreting SAE features,
including vocabulary projection, feature clustering, and visualization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger

try:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class VocabProjection:
    """
    Vocabulary projection results for a feature direction.

    Attributes:
        feature_idx: Index of the feature (if applicable)
        direction: The feature direction vector
        top_tokens: Top predicted tokens and scores
        bottom_tokens: Bottom (most negative) tokens and scores
    """

    feature_idx: Optional[int] = None
    direction: Optional[torch.Tensor] = None
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)
    bottom_tokens: List[Tuple[str, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        top_str = ", ".join(f"'{t}':{s:.2f}" for t, s in self.top_tokens[:5])
        return f"VocabProjection(top=[{top_str}...])"


class FeatureAnalyzer:
    """
    Analyzer for interpreting SAE features.

    This class provides tools for understanding what SAE features
    represent by projecting them to vocabulary space, clustering
    similar features, and comparing feature activations.

    Usage:
        >>> analyzer = FeatureAnalyzer(model, sae)
        >>> proj = analyzer.project_to_vocab(feature_idx=42)
        >>> print(f"Feature 42 predicts: {proj.top_tokens[:5]}")
    """

    def __init__(
        self,
        model: Any,  # ACEModel
        sae: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize feature analyzer.

        Args:
            model: ACEModel instance (for tokenizer and unembedding)
            sae: Sparse Autoencoder (optional, can be set later)
        """
        self.model = model
        self.sae = sae

        # Cache unembedding matrix
        self._W_U: Optional[torch.Tensor] = None

    @property
    def W_U(self) -> torch.Tensor:
        """Get (cached) unembedding matrix."""
        if self._W_U is None:
            self._W_U = self.model.get_unembedding_matrix()
        return self._W_U

    def set_sae(self, sae: torch.nn.Module) -> None:
        """Set the SAE for analysis."""
        self.sae = sae

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """
        Get the decoder direction for a specific feature.

        Args:
            feature_idx: Index of the feature

        Returns:
            Feature direction vector, shape (d_model,)
        """
        if self.sae is None:
            raise RuntimeError("SAE not set")

        if hasattr(self.sae, 'W_dec'):
            return self.sae.W_dec[feature_idx]
        elif hasattr(self.sae, 'decoder'):
            return self.sae.decoder.weight[:, feature_idx]
        else:
            raise RuntimeError("Cannot access SAE decoder")

    def project_to_vocab(
        self,
        feature_idx: Optional[int] = None,
        direction: Optional[torch.Tensor] = None,
        top_k: int = 20,
    ) -> VocabProjection:
        """
        Project a feature direction to vocabulary space.

        This shows which tokens a feature direction most strongly
        predicts when added to the residual stream.

        Args:
            feature_idx: Index of SAE feature (mutually exclusive with direction)
            direction: Custom direction vector (mutually exclusive with feature_idx)
            top_k: Number of top/bottom tokens to return

        Returns:
            VocabProjection with top and bottom tokens
        """
        if feature_idx is not None and direction is not None:
            raise ValueError("Specify either feature_idx or direction, not both")

        if feature_idx is not None:
            direction = self.get_feature_direction(feature_idx)
        elif direction is None:
            raise ValueError("Must specify either feature_idx or direction")

        # Project through unembedding
        logits = direction @ self.W_U  # (vocab_size,)

        # Get top tokens
        top_values, top_indices = torch.topk(logits, k=top_k)
        top_tokens = [
            (self.model.tokenizer.decode([idx]), val.item())
            for idx, val in zip(top_indices.tolist(), top_values.tolist())
        ]

        # Get bottom tokens
        bot_values, bot_indices = torch.topk(logits, k=top_k, largest=False)
        bottom_tokens = [
            (self.model.tokenizer.decode([idx]), val.item())
            for idx, val in zip(bot_indices.tolist(), bot_values.tolist())
        ]

        return VocabProjection(
            feature_idx=feature_idx,
            direction=direction,
            top_tokens=top_tokens,
            bottom_tokens=bottom_tokens,
        )

    def find_similar_features(
        self,
        feature_idx: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Find features with similar directions.

        Args:
            feature_idx: Target feature index
            top_k: Number of similar features to return

        Returns:
            List of (feature_idx, similarity) tuples
        """
        if self.sae is None:
            raise RuntimeError("SAE not set")

        target_dir = self.get_feature_direction(feature_idx)

        # Get all feature directions
        if hasattr(self.sae, 'W_dec'):
            all_dirs = self.sae.W_dec  # (n_features, d_model)
        else:
            all_dirs = self.sae.decoder.weight.T

        # Compute cosine similarities
        target_norm = target_dir / (target_dir.norm() + 1e-8)
        all_norms = all_dirs / (all_dirs.norm(dim=1, keepdim=True) + 1e-8)
        similarities = all_norms @ target_norm  # (n_features,)

        # Get top-k (excluding self)
        values, indices = torch.topk(similarities, k=top_k + 1)

        results = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            if idx != feature_idx:
                results.append((idx, sim))

        return results[:top_k]

    def cluster_features(
        self,
        n_clusters: int = 10,
        feature_subset: Optional[List[int]] = None,
    ) -> Dict[int, List[int]]:
        """
        Cluster SAE features by direction similarity.

        Args:
            n_clusters: Number of clusters
            feature_subset: Subset of features to cluster (None for all)

        Returns:
            Dictionary mapping cluster_id to list of feature indices
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for clustering: pip install scikit-learn")

        if self.sae is None:
            raise RuntimeError("SAE not set")

        # Get feature directions
        if hasattr(self.sae, 'W_dec'):
            all_dirs = self.sae.W_dec
        else:
            all_dirs = self.sae.decoder.weight.T

        # Subset if specified
        if feature_subset is not None:
            dirs = all_dirs[feature_subset].detach().cpu().numpy()
            indices = feature_subset
        else:
            dirs = all_dirs.detach().cpu().numpy()
            indices = list(range(len(dirs)))

        # Compute similarity matrix
        sim_matrix = sklearn_cosine(dirs)

        # Cluster
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
        )
        labels = clustering.fit_predict(sim_matrix)

        # Group by cluster
        clusters = {}
        for idx, label in zip(indices, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return clusters

    def analyze_feature_activation_pattern(
        self,
        feature_idx: int,
        input_ids: torch.Tensor,
        layer: int,
    ) -> Dict[str, Any]:
        """
        Analyze how a feature activates across token positions.

        Args:
            feature_idx: Feature to analyze
            input_ids: Input token IDs
            layer: Layer to extract activations from

        Returns:
            Dictionary with activation analysis
        """
        # Get residual stream
        resid = self.model.get_residual_stream(input_ids, layer=layer)

        # Encode through SAE
        if hasattr(self.sae, 'encode'):
            features = self.sae.encode(resid)
        else:
            # Manual encoding
            features = F.relu(resid @ self.sae.W_enc.T + self.sae.b_enc)

        # Get activations for target feature
        acts = features[0, :, feature_idx]  # (seq_len,)

        # Get token strings
        tokens = [
            self.model.tokenizer.decode([tid])
            for tid in input_ids[0].tolist()
        ]

        return {
            "feature_idx": feature_idx,
            "activations": acts.tolist(),
            "tokens": tokens,
            "max_activation": acts.max().item(),
            "mean_activation": acts.mean().item(),
            "sparsity": (acts > 0).float().mean().item(),
            "peak_position": acts.argmax().item(),
            "peak_token": tokens[acts.argmax().item()],
        }

    def compare_features(
        self,
        feature_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Compare multiple features.

        Args:
            feature_indices: List of feature indices to compare

        Returns:
            Dictionary with comparison results
        """
        directions = [self.get_feature_direction(idx) for idx in feature_indices]
        directions = torch.stack(directions)  # (n, d_model)

        # Compute pairwise similarities
        norms = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = norms @ norms.T

        # Get vocab projections
        projections = [
            self.project_to_vocab(feature_idx=idx, top_k=5)
            for idx in feature_indices
        ]

        return {
            "feature_indices": feature_indices,
            "similarity_matrix": sim_matrix.tolist(),
            "projections": projections,
            "mean_similarity": sim_matrix.mean().item(),
        }
