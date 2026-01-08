"""Detection module using Sparse Autoencoders (SAEs)."""

from ace_code.detection.sae_detector import SAEDetector, ErrorFeature
from ace_code.detection.feature_analysis import FeatureAnalyzer, VocabProjection

__all__ = [
    "SAEDetector",
    "ErrorFeature",
    "FeatureAnalyzer",
    "VocabProjection",
]
