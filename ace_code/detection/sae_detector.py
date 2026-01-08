"""
SAE-Based Error Detection

This module implements the detection component of ACE-Code using
Sparse Autoencoders (SAEs). SAEs learn interpretable feature directions
in the activation space, allowing us to identify features that activate
when the model is confused or about to make an error.

Key Concepts:
- SAE features are sparse, interpretable directions in activation space
- "Incorrectness" features fire when the model is uncertain/confused
- We identify these by comparing activations on correct vs buggy code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ace_code.utils.statistics import compute_t_statistic, find_significant_features

# Optional: SAE Lens for loading pre-trained SAEs
try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except ImportError:
    SAE_LENS_AVAILABLE = False
    SAE = None


@dataclass
class ErrorFeature:
    """
    A feature that indicates potential errors or confusion.

    Attributes:
        feature_idx: Index in the SAE feature dictionary
        t_statistic: T-statistic comparing activation on buggy vs correct code
        mean_activation_buggy: Mean activation on buggy examples
        mean_activation_correct: Mean activation on correct examples
        vocab_projection: Top tokens associated with this feature
        interpretation: Human-readable interpretation
    """

    feature_idx: int
    t_statistic: float
    mean_activation_buggy: float = 0.0
    mean_activation_correct: float = 0.0
    vocab_projection: List[Tuple[str, float]] = field(default_factory=list)
    interpretation: str = ""

    @property
    def effect_size(self) -> float:
        """Compute effect size (difference in means)."""
        return self.mean_activation_buggy - self.mean_activation_correct

    def __repr__(self) -> str:
        return (
            f"ErrorFeature(idx={self.feature_idx}, "
            f"t={self.t_statistic:.2f}, "
            f"effect={self.effect_size:.3f})"
        )


class SAEDetector:
    """
    Sparse Autoencoder-based error/incorrectness detector.

    This class uses SAEs to detect when a model is likely to produce
    incorrect output. It identifies SAE features that discriminate
    between correct and buggy code, then monitors those features
    during inference.

    Architecture:
    1. Load pre-trained SAE for the target layer
    2. Collect activations on positive/negative examples
    3. Identify "incorrectness" features via t-statistics
    4. Monitor these features during inference

    Usage:
        >>> detector = SAEDetector(model, layer=12)
        >>> detector.load_sae("path/to/sae")
        >>> detector.identify_error_features(code_pairs)
        >>> score = detector.detect(prompt)
    """

    def __init__(
        self,
        model: Any,  # ACEModel
        layer: int,
        position: str = "resid_post",
    ):
        """
        Initialize SAE detector.

        Args:
            model: ACEModel instance
            layer: Layer to attach SAE to
            position: Where in the layer to extract activations
        """
        self.model = model
        self.layer = layer
        self.position = position
        self.hook_name = f"blocks.{layer}.hook_{position}"

        # SAE components
        self.sae: Optional[nn.Module] = None
        self.sae_loaded = False

        # Error features
        self.error_features: List[ErrorFeature] = []

        # Thresholds
        self.detection_threshold = 0.5

        logger.info(f"Initialized SAEDetector for layer {layer}")

    def load_sae(
        self,
        sae_path: Optional[str] = None,
        sae_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Load a pre-trained Sparse Autoencoder.

        Args:
            sae_path: Path to local SAE checkpoint
            sae_id: ID for loading from SAE Lens / Neuronpedia
            device: Device to load SAE on (default: model device)
        """
        if device is None:
            device = str(self.model.device)

        if sae_path is not None:
            self.sae = self._load_sae_from_path(sae_path, device)
        elif sae_id is not None:
            self.sae = self._load_sae_from_id(sae_id, device)
        else:
            # Create a simple SAE for demonstration
            logger.warning("No SAE path provided, creating simple SAE")
            self.sae = SimpleSAE(
                d_model=self.model.d_model,
                n_features=self.model.d_model * 4,  # 4x expansion
            ).to(device)

        self.sae_loaded = True
        logger.info(f"SAE loaded with {self.sae.n_features if hasattr(self.sae, 'n_features') else 'unknown'} features")

    def _load_sae_from_path(self, path: str, device: str) -> nn.Module:
        """Load SAE from a local checkpoint."""
        logger.info(f"Loading SAE from {path}")
        checkpoint = torch.load(path, map_location=device)

        # Try different checkpoint formats
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        elif "state_dict" in checkpoint:
            sae = SimpleSAE(
                d_model=checkpoint.get("d_model", self.model.d_model),
                n_features=checkpoint.get("n_features", self.model.d_model * 4),
            )
            sae.load_state_dict(checkpoint["state_dict"])
            return sae.to(device)
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    def _load_sae_from_id(self, sae_id: str, device: str) -> nn.Module:
        """Load SAE from SAE Lens / Neuronpedia."""
        if not SAE_LENS_AVAILABLE:
            raise ImportError("sae_lens required for loading by ID: pip install sae_lens")

        logger.info(f"Loading SAE: {sae_id}")
        sae = SAE.from_pretrained(sae_id, device=device)
        return sae

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations through the SAE to get feature activations.

        Args:
            activations: Residual stream activations, shape (batch, seq_len, d_model)

        Returns:
            SAE feature activations, shape (batch, seq_len, n_features)
        """
        if not self.sae_loaded:
            raise RuntimeError("SAE not loaded. Call load_sae() first.")

        return self.sae.encode(activations)

    def collect_feature_activations(
        self,
        input_ids: torch.Tensor,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Collect SAE feature activations for an input.

        Args:
            input_ids: Input token IDs
            position: Token position to analyze (-1 for last)

        Returns:
            Feature activation vector, shape (n_features,)
        """
        # Get residual stream activations
        resid = self.model.get_residual_stream(
            input_ids,
            layer=self.layer,
            position=self.position.replace("resid_", ""),
        )

        # Get activations at specified position
        if position == -1:
            act = resid[0, -1, :]  # Last token
        else:
            act = resid[0, position, :]

        # Encode through SAE
        features = self.encode(act.unsqueeze(0).unsqueeze(0))

        return features.squeeze()

    def identify_error_features(
        self,
        positive_inputs: List[torch.Tensor],
        negative_inputs: List[torch.Tensor],
        t_threshold: float = 2.0,
        top_k: int = 20,
    ) -> List[ErrorFeature]:
        """
        Identify SAE features that discriminate buggy from correct code.

        This is the core feature identification process:
        1. Collect SAE activations on positive (correct) examples
        2. Collect SAE activations on negative (buggy) examples
        3. Compute t-statistics comparing the two distributions
        4. Select features with high |t| that fire more on buggy code

        Args:
            positive_inputs: List of input_ids for correct code
            negative_inputs: List of input_ids for buggy code
            t_threshold: Minimum |t-statistic| for significance
            top_k: Maximum number of features to return

        Returns:
            List of ErrorFeature objects
        """
        logger.info(f"Identifying error features from {len(positive_inputs)} pos / {len(negative_inputs)} neg examples")

        if not self.sae_loaded:
            raise RuntimeError("SAE not loaded. Call load_sae() first.")

        # Collect feature activations
        pos_features = []
        neg_features = []

        with torch.no_grad():
            for inp in positive_inputs:
                feat = self.collect_feature_activations(inp)
                pos_features.append(feat)

            for inp in negative_inputs:
                feat = self.collect_feature_activations(inp)
                neg_features.append(feat)

        pos_features = torch.stack(pos_features)  # (n_pos, n_features)
        neg_features = torch.stack(neg_features)  # (n_neg, n_features)

        # Compute t-statistics
        # Positive t = buggy > correct (we want features that fire MORE on buggy)
        t_stats = compute_t_statistic(neg_features, pos_features)

        # Find significant features
        indices, t_values = find_significant_features(
            t_stats,
            threshold=t_threshold,
            top_k=top_k,
        )

        # Build ErrorFeature objects
        self.error_features = []
        for idx, t_val in zip(indices.tolist(), t_values.tolist()):
            feature = ErrorFeature(
                feature_idx=idx,
                t_statistic=t_val,
                mean_activation_buggy=neg_features[:, idx].mean().item(),
                mean_activation_correct=pos_features[:, idx].mean().item(),
            )

            # Get vocabulary projection for interpretation
            feature.vocab_projection = self._project_to_vocab(idx)

            self.error_features.append(feature)

        # Sort by absolute t-statistic
        self.error_features.sort(key=lambda f: abs(f.t_statistic), reverse=True)

        logger.info(f"Identified {len(self.error_features)} error features")
        return self.error_features

    def _project_to_vocab(self, feature_idx: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Project an SAE feature to vocabulary space.

        This helps interpret what a feature "means" by showing
        which tokens it most strongly predicts.

        Args:
            feature_idx: Index of the SAE feature
            top_k: Number of top tokens to return

        Returns:
            List of (token, score) tuples
        """
        # Get SAE decoder weight for this feature
        if hasattr(self.sae, 'W_dec'):
            feature_dir = self.sae.W_dec[feature_idx]  # (d_model,)
        elif hasattr(self.sae, 'decoder'):
            feature_dir = self.sae.decoder.weight[:, feature_idx]
        else:
            return []

        # Project through unembedding matrix
        W_U = self.model.get_unembedding_matrix()  # (d_model, vocab_size)
        logits = feature_dir @ W_U  # (vocab_size,)

        # Get top tokens
        values, indices = torch.topk(logits, k=top_k)

        results = []
        for v, idx in zip(values.tolist(), indices.tolist()):
            token = self.model.tokenizer.decode([idx])
            results.append((token, v))

        return results

    def detect(
        self,
        input_ids: torch.Tensor,
        position: int = -1,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[int, float]]:
        """
        Detect potential errors in the input.

        Args:
            input_ids: Input token IDs
            position: Token position to check (-1 for last)
            threshold: Detection threshold (default: self.detection_threshold)

        Returns:
            Tuple of (is_error, error_score, feature_activations)
        """
        if not self.error_features:
            logger.warning("No error features identified. Call identify_error_features() first.")
            return False, 0.0, {}

        if threshold is None:
            threshold = self.detection_threshold

        # Get feature activations
        features = self.collect_feature_activations(input_ids, position)

        # Compute error score based on error feature activations
        feature_acts = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for ef in self.error_features:
            act = features[ef.feature_idx].item()
            feature_acts[ef.feature_idx] = act

            # Weight by t-statistic
            weight = abs(ef.t_statistic)
            weighted_sum += act * weight
            weight_sum += weight

        error_score = weighted_sum / (weight_sum + 1e-8)

        # Normalize to [0, 1] range
        error_score = torch.sigmoid(torch.tensor(error_score)).item()

        is_error = error_score > threshold

        return is_error, error_score, feature_acts

    def get_error_direction(self) -> torch.Tensor:
        """
        Get the combined "error direction" in activation space.

        This averages the SAE feature directions weighted by their
        discriminative power (t-statistic), giving a single vector
        that points towards "incorrectness".

        Returns:
            Error direction vector, shape (d_model,)
        """
        if not self.error_features:
            raise RuntimeError("No error features identified")

        # Get decoder weights
        if hasattr(self.sae, 'W_dec'):
            W_dec = self.sae.W_dec
        elif hasattr(self.sae, 'decoder'):
            W_dec = self.sae.decoder.weight.T
        else:
            raise RuntimeError("Cannot access SAE decoder weights")

        # Weighted average of feature directions
        direction = torch.zeros(self.model.d_model, device=W_dec.device)
        weight_sum = 0.0

        for ef in self.error_features:
            weight = abs(ef.t_statistic)
            direction += W_dec[ef.feature_idx] * weight
            weight_sum += weight

        direction = direction / (weight_sum + 1e-8)

        # Normalize
        direction = direction / (direction.norm() + 1e-8)

        return direction


class SimpleSAE(nn.Module):
    """
    A simple Sparse Autoencoder implementation.

    This is used when no pre-trained SAE is available. It provides
    the basic SAE interface with learnable encoder/decoder weights.

    Architecture:
    - Encoder: x -> ReLU(W_enc @ x + b_enc)
    - Decoder: features -> W_dec @ features + b_dec
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features

        # Encoder
        self.W_enc = nn.Parameter(torch.randn(n_features, d_model, dtype=dtype) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features, dtype=dtype))

        # Decoder
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model, dtype=dtype) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        # x: (..., d_model)
        # output: (..., n_features)
        pre_act = x @ self.W_enc.T + self.b_enc
        return F.relu(pre_act)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstructed activations."""
        # features: (..., n_features)
        # output: (..., d_model)
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode."""
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features
