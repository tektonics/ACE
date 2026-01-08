"""
ACE-Code: Automated Circuit Extraction for Code Models

A mechanistic interpretability toolkit for understanding and steering
code-capable language models through circuit discovery, sparse autoencoders,
and activation steering.
"""

__version__ = "0.1.0"
__author__ = "ACE-Code Team"

from ace_code.core.model import ACEModel, load_hooked_model
from ace_code.data.mutation import MutationGenerator, CodePair
from ace_code.discovery.cdt import ContextualDecomposition, CDTResult
from ace_code.detection.sae_detector import SAEDetector, ErrorFeature
from ace_code.intervention.steering import SteeringVector, PISCES
from ace_code.pipeline import ACEPipeline

__all__ = [
    # Core
    "ACEModel",
    "load_hooked_model",
    # Data
    "MutationGenerator",
    "CodePair",
    # Discovery
    "ContextualDecomposition",
    "CDTResult",
    # Detection
    "SAEDetector",
    "ErrorFeature",
    # Intervention
    "SteeringVector",
    "PISCES",
    # Pipeline
    "ACEPipeline",
]
