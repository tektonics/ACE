"""
ACE-Code: Automated Circuit Extraction for Code Models

Phase 1: SAE-Based Anomaly Flagging
A negative restraint system that acts as a high-speed error alarm by detecting
features correlated with bugs rather than certifying correctness.
"""

__version__ = "0.1.0"

from ace_code.sae_anomaly import (
    JumpReLUSAE,
    SidecarModel,
    FeatureDiscovery,
    AnomalyDetector,
    SAEAnomalyPipeline,
)

__all__ = [
    "JumpReLUSAE",
    "SidecarModel",
    "FeatureDiscovery",
    "AnomalyDetector",
    "SAEAnomalyPipeline",
]
