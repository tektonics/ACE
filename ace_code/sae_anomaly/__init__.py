"""
SAE-Based Anomaly Flagging Module (Phase 1)

This module implements a "negative restraint system" that acts as a high-speed
error alarm by detecting features correlated with bugs rather than certifying
correctness. It exploits the finding that models possess robust "anomaly detectors"
(incorrectness features) but lack reliable "validity assessors" (correctness features).

Based on:
- Tahimic & Cheng (2025): Mechanistic Interpretability of Code Correctness via SAEs
- Nguyen et al. (2025): Deploying Interpretability to Production with Rakuten
- Lieberum et al. (2024): Gemma Scope - JumpReLU SAE Architecture
"""

from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.sae_anomaly.sidecar import SidecarModel
from ace_code.sae_anomaly.discovery import FeatureDiscovery
from ace_code.sae_anomaly.detector import AnomalyDetector
from ace_code.sae_anomaly.pipeline import SAEAnomalyPipeline
from ace_code.sae_anomaly.trainer import SAETrainer, SAETrainerConfig, train_sae_on_code_corpus

__all__ = [
    "JumpReLUSAE",
    "SidecarModel",
    "FeatureDiscovery",
    "AnomalyDetector",
    "SAEAnomalyPipeline",
    "SAETrainer",
    "SAETrainerConfig",
    "train_sae_on_code_corpus",
]
