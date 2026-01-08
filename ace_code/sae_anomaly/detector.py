"""
Runtime Anomaly Detection using SAE Features

This module implements the runtime inference logic for anomaly detection.
Once "incorrectness" features have been identified through feature discovery,
this detector checks those specific features during inference to flag
potentially buggy code.

Algorithm:
1. Harvest: Get residual stream x at the final prompt token
2. Encode: Run a(x) = SAE(x)
3. Score: Look up activation values of pre-identified "Incorrectness Features"
4. Flag: If Σ(a_j × weight_j) > Threshold, FLAG as suspicious

Expected Performance:
- "Incorrect-Predicting" features: F1 ≈ 0.821
- "Correct-Predicting" features: F1 ≈ 0.50 (unreliable, ignored)
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import numpy as np
import torch
from ace_code.sae_anomaly.sidecar import SidecarModel
from ace_code.sae_anomaly.discovery import FeatureDiscoveryResult
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyLevel(Enum):
    """Anomaly severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    anomaly_level: AnomalyLevel
    threshold: float

    # Active features
    n_active_features: int
    active_feature_indices: List[int]
    active_feature_values: List[float]

    # Top contributing features
    top_features: List[Dict[str, Any]]

    # Metadata
    input_text: str
    model_name: str
    target_layer: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "anomaly_level": self.anomaly_level.value,
            "threshold": self.threshold,
            "n_active_features": self.n_active_features,
            "active_feature_indices": self.active_feature_indices,
            "active_feature_values": self.active_feature_values,
            "top_features": self.top_features,
            "input_text": self.input_text[:200] + "..." if len(self.input_text) > 200 else self.input_text,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
        }


@dataclass
class DetectorConfig:
    """Configuration for anomaly detector."""

    # Thresholds
    anomaly_threshold: float = 0.5
    low_threshold: float = 0.3
    medium_threshold: float = 0.5
    high_threshold: float = 0.7
    critical_threshold: float = 0.9

    # Scoring
    use_weighted_score: bool = True
    normalize_by_n_features: bool = False

    # Feature interpretation
    top_k_features_to_report: int = 10
    min_activation_to_report: float = 0.01

    # Routing
    route_to_verification: bool = True  # Route flagged inputs to Phase 4


class AnomalyDetector:
    """
    Runtime anomaly detector using SAE incorrectness features.

    This is the main interface for Phase 1 anomaly detection at inference time.
    It uses pre-computed incorrectness features to quickly flag potentially
    buggy code generation.

    Usage:
        sidecar = SidecarModel().load()
        discovery_result = FeatureDiscoveryResult.load("discovery.json")
        detector = AnomalyDetector(sidecar, discovery_result)

        result = detector.detect("def sort_list(lst):")
        if result.is_anomaly:
            # Route to Phase 4 formal verification
            pass
    """

    def __init__(
        self,
        sidecar: SidecarModel,
        discovery_result: FeatureDiscoveryResult,
        config: Optional[DetectorConfig] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            sidecar: Loaded SidecarModel instance
            discovery_result: Pre-computed feature discovery result
            config: Detector configuration
        """
        self.sidecar = sidecar
        self.discovery = discovery_result
        self.config = config or DetectorConfig()

        # Convert to tensors for efficient lookup
        self._feature_indices = torch.tensor(
            discovery_result.incorrectness_features,
            device=sidecar.model.device,
        )
        self._feature_weights = torch.tensor(
            discovery_result.feature_weights,
            dtype=torch.float32,
            device=sidecar.model.device,
        )

        logger.info(
            f"Initialized AnomalyDetector with "
            f"{len(self._feature_indices)} incorrectness features"
        )

    def _compute_anomaly_score(
        self,
        feature_activations: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute weighted anomaly score.

        Score = Σ(a_j × weight_j)

        Args:
            feature_activations: Activations for incorrectness features

        Returns:
            Tuple of (score, per_feature_contributions)
        """
        if self.config.use_weighted_score:
            contributions = feature_activations * self._feature_weights
        else:
            contributions = feature_activations

        score = contributions.sum().item()

        if self.config.normalize_by_n_features:
            score = score / len(self._feature_indices)

        return score, contributions

    def _determine_level(self, score: float) -> AnomalyLevel:
        """Determine anomaly severity level from score."""
        if score >= self.config.critical_threshold:
            return AnomalyLevel.CRITICAL
        elif score >= self.config.high_threshold:
            return AnomalyLevel.HIGH
        elif score >= self.config.medium_threshold:
            return AnomalyLevel.MEDIUM
        elif score >= self.config.low_threshold:
            return AnomalyLevel.LOW
        else:
            return AnomalyLevel.NONE

    def _get_top_features(
        self,
        feature_activations: torch.Tensor,
        contributions: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Get top contributing features with details."""
        # Sort by contribution
        sorted_indices = torch.argsort(contributions, descending=True)
        top_k = min(self.config.top_k_features_to_report, len(sorted_indices))

        top_features = []
        for i in range(top_k):
            idx = sorted_indices[i].item()
            activation = feature_activations[idx].item()

            if activation < self.config.min_activation_to_report:
                continue

            top_features.append({
                "feature_index": int(self._feature_indices[idx].item()),
                "local_index": idx,
                "activation": activation,
                "weight": self._feature_weights[idx].item(),
                "contribution": contributions[idx].item(),
                "t_statistic": float(self.discovery.feature_t_statistics[idx]),
            })

        return top_features

    @torch.no_grad()
    def detect(
        self,
        text: Union[str, List[str]],
        return_features: bool = False,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect anomalies in input text.

        Args:
            text: Input prompt or code
            return_features: Whether to include raw feature activations

        Returns:
            DetectionResult or list of results for batch input
        """
        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]

        results = []
        for input_text in text:
            result = self._detect_single(input_text)
            results.append(result)

        if is_batch:
            return results
        return results[0]

    def _detect_single(self, text: str) -> DetectionResult:
        """Detect anomaly for single input."""
        # Get SAE features
        all_features = self.sidecar.get_sae_features(text, position="final")

        # Extract incorrectness features
        feature_activations = all_features[0, self._feature_indices]

        # Compute score
        score, contributions = self._compute_anomaly_score(feature_activations)

        # Determine level and flag
        level = self._determine_level(score)
        is_anomaly = score >= self.config.anomaly_threshold

        # Get active features
        active_mask = feature_activations > 0
        active_indices = self._feature_indices[active_mask].cpu().tolist()
        active_values = feature_activations[active_mask].cpu().tolist()

        # Get top features
        top_features = self._get_top_features(feature_activations, contributions)

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_level=level,
            threshold=self.config.anomaly_threshold,
            n_active_features=len(active_indices),
            active_feature_indices=active_indices,
            active_feature_values=active_values,
            top_features=top_features,
            input_text=text,
            model_name=self.sidecar.config.model_name,
            target_layer=self.sidecar.config.target_layer,
        )

    def batch_detect(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[DetectionResult]:
        """
        Batch detection for multiple inputs.

        Args:
            texts: List of input texts
            batch_size: Processing batch size

        Returns:
            List of DetectionResults
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.detect(batch)
            results.extend(batch_results)
        return results

    def should_route_to_verification(self, result: DetectionResult) -> bool:
        """
        Determine if result should be routed to formal verification (Phase 4).

        Args:
            result: Detection result

        Returns:
            Whether to route to verification
        """
        if not self.config.route_to_verification:
            return False

        return result.is_anomaly

    def get_detection_summary(
        self,
        results: List[DetectionResult],
    ) -> Dict[str, Any]:
        """
        Get summary statistics for batch detection.

        Args:
            results: List of detection results

        Returns:
            Summary statistics dict
        """
        n_total = len(results)
        n_anomalies = sum(1 for r in results if r.is_anomaly)
        scores = [r.anomaly_score for r in results]

        level_counts = {}
        for level in AnomalyLevel:
            level_counts[level.value] = sum(
                1 for r in results if r.anomaly_level == level
            )

        return {
            "n_total": n_total,
            "n_anomalies": n_anomalies,
            "anomaly_rate": n_anomalies / n_total if n_total > 0 else 0,
            "mean_score": np.mean(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "level_distribution": level_counts,
        }

    def calibrate_threshold(
        self,
        validation_correct: List[str],
        validation_incorrect: List[str],
        target_recall: float = 0.8,
    ) -> float:
        """
        Calibrate detection threshold on validation data.

        Args:
            validation_correct: Prompts with correct code
            validation_incorrect: Prompts with incorrect code
            target_recall: Target recall for incorrect detection

        Returns:
            Calibrated threshold
        """
        logger.info("Calibrating detection threshold...")

        # Get scores for both classes
        correct_scores = []
        for text in validation_correct:
            result = self._detect_single(text)
            correct_scores.append(result.anomaly_score)

        incorrect_scores = []
        for text in validation_incorrect:
            result = self._detect_single(text)
            incorrect_scores.append(result.anomaly_score)

        # Find threshold that achieves target recall
        all_scores = sorted(incorrect_scores)
        threshold_idx = int((1 - target_recall) * len(all_scores))
        threshold = all_scores[threshold_idx] if threshold_idx < len(all_scores) else all_scores[-1]

        # Compute metrics at this threshold
        true_positives = sum(1 for s in incorrect_scores if s >= threshold)
        false_positives = sum(1 for s in correct_scores if s >= threshold)
        recall = true_positives / len(incorrect_scores) if incorrect_scores else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logger.info(
            f"Calibrated threshold: {threshold:.4f}, "
            f"Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}"
        )

        # Update config
        self.config.anomaly_threshold = threshold

        return threshold

    def save_config(self, path: Union[str, Path]) -> None:
        """Save detector configuration."""
        path = Path(path)
        config_dict = {
            "anomaly_threshold": self.config.anomaly_threshold,
            "low_threshold": self.config.low_threshold,
            "medium_threshold": self.config.medium_threshold,
            "high_threshold": self.config.high_threshold,
            "critical_threshold": self.config.critical_threshold,
            "use_weighted_score": self.config.use_weighted_score,
            "normalize_by_n_features": self.config.normalize_by_n_features,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, path: Union[str, Path]) -> DetectorConfig:
        """Load detector configuration."""
        path = Path(path)
        with open(path) as f:
            config_dict = json.load(f)
        return DetectorConfig(**config_dict)
