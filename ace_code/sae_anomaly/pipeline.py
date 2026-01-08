"""
SAE Anomaly Flagging Pipeline (Phase 1)

This module provides the main pipeline that orchestrates all components
of Phase 1: SAE-Based Anomaly Flagging. It serves as the primary interface
for both feature discovery and runtime anomaly detection.

The pipeline acts as a "negative restraint system" - a high-speed error alarm
that detects features correlated with bugs rather than certifying correctness.

Usage:
    # Feature Discovery (offline)
    pipeline = SAEAnomalyPipeline.from_config("config.yaml")
    pipeline.run_discovery()
    pipeline.save("artifacts/")

    # Runtime Detection (online)
    pipeline = SAEAnomalyPipeline.load("artifacts/")
    result = pipeline.detect("def sort_list(lst): return lst.sort()")
    if result.is_anomaly:
        # Route to Phase 4 Formal Verification
        pass
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import torch
from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.sae_anomaly.sidecar import SidecarModel, SidecarConfig
from ace_code.sae_anomaly.discovery import FeatureDiscovery, DiscoveryConfig, FeatureDiscoveryResult
from ace_code.sae_anomaly.detector import AnomalyDetector, DetectorConfig, DetectionResult
from ace_code.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    target_layer: int = 12
    device: str = "auto"
    dtype: str = "float16"
    quantization: Optional[str] = "8bit"

    # SAE configuration
    sae_expansion_factor: int = 8
    sae_path: Optional[str] = None

    # Discovery configuration
    dataset_name: str = "mbpp"
    n_samples: Optional[int] = None
    temperature: float = 0.0
    max_new_tokens: int = 256
    top_k_features: int = 100
    min_t_statistic: float = 2.0
    max_frequency: float = 0.02

    # Detection configuration
    anomaly_threshold: float = 0.5
    use_weighted_score: bool = True

    # Processing
    chunk_size: int = 128
    chunk_overlap: int = 32

    # Output
    output_dir: str = "artifacts"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_sidecar_config(self) -> SidecarConfig:
        """Convert to SidecarConfig."""
        return SidecarConfig(
            model_name=self.model_name,
            target_layer=self.target_layer,
            device=self.device,
            dtype=self.dtype,
            quantization=self.quantization,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            sae_expansion_factor=self.sae_expansion_factor,
            sae_path=self.sae_path,
        )

    def to_discovery_config(self) -> DiscoveryConfig:
        """Convert to DiscoveryConfig."""
        return DiscoveryConfig(
            dataset_name=self.dataset_name,
            n_samples=self.n_samples,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_k_features=self.top_k_features,
            min_t_statistic=self.min_t_statistic,
            max_frequency=self.max_frequency,
            output_dir=self.output_dir,
        )

    def to_detector_config(self) -> DetectorConfig:
        """Convert to DetectorConfig."""
        return DetectorConfig(
            anomaly_threshold=self.anomaly_threshold,
            use_weighted_score=self.use_weighted_score,
        )


class SAEAnomalyPipeline:
    """
    Main pipeline for SAE-Based Anomaly Flagging.

    This pipeline orchestrates:
    1. Model loading (sidecar architecture)
    2. Feature discovery (offline, on MBPP dataset)
    3. Runtime detection (online, for each input)

    The pipeline implements the findings from Tahimic & Cheng (2025):
    - Models have robust "incorrectness" features (F1 ≈ 0.821)
    - Models lack reliable "correctness" features (F1 ≈ 0.50)
    - This asymmetry is exploited for anomaly detection
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        setup_logging(level=self.config.log_level)

        self._sidecar: Optional[SidecarModel] = None
        self._discovery_result: Optional[FeatureDiscoveryResult] = None
        self._detector: Optional[AnomalyDetector] = None
        self._is_loaded = False

        logger.info(f"Initialized SAEAnomalyPipeline with model: {self.config.model_name}")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "SAEAnomalyPipeline":
        """Create pipeline from configuration file."""
        config = PipelineConfig.from_yaml(config_path)
        return cls(config)

    @classmethod
    def load(cls, artifacts_dir: Union[str, Path]) -> "SAEAnomalyPipeline":
        """
        Load a complete pipeline from saved artifacts.

        Args:
            artifacts_dir: Directory containing saved pipeline artifacts

        Returns:
            Loaded pipeline ready for detection
        """
        artifacts_dir = Path(artifacts_dir)

        # Load config
        config_path = artifacts_dir / "config.yaml"
        if config_path.exists():
            config = PipelineConfig.from_yaml(config_path)
        else:
            config = PipelineConfig()

        pipeline = cls(config)

        # Load sidecar model
        sidecar_config = config.to_sidecar_config()
        sidecar_config.sae_path = str(artifacts_dir / "sae.pt")
        pipeline._sidecar = SidecarModel(sidecar_config)
        pipeline._sidecar.load()

        # Load discovery result
        discovery_path = artifacts_dir / "discovery_result.json"
        pipeline._discovery_result = FeatureDiscoveryResult.load(discovery_path)

        # Initialize detector
        detector_config = config.to_detector_config()
        detector_config_path = artifacts_dir / "detector_config.json"
        if detector_config_path.exists():
            detector_config = AnomalyDetector.load_config(detector_config_path)

        pipeline._detector = AnomalyDetector(
            pipeline._sidecar,
            pipeline._discovery_result,
            detector_config,
        )

        pipeline._is_loaded = True
        logger.info(f"Loaded pipeline from {artifacts_dir}")

        return pipeline

    def load_model(self) -> "SAEAnomalyPipeline":
        """Load the sidecar model."""
        logger.info("Loading sidecar model...")

        sidecar_config = self.config.to_sidecar_config()
        self._sidecar = SidecarModel(sidecar_config)
        self._sidecar.load()

        return self

    def run_discovery(
        self,
        general_text_activations: Optional[torch.Tensor] = None,
    ) -> FeatureDiscoveryResult:
        """
        Run feature discovery to identify incorrectness features.

        This is an offline process that should be run once before deployment.

        Args:
            general_text_activations: Optional activations on general text
                for frequency filtering

        Returns:
            FeatureDiscoveryResult with selected features
        """
        if self._sidecar is None:
            self.load_model()

        logger.info("Running feature discovery...")

        discovery_config = self.config.to_discovery_config()
        discovery = FeatureDiscovery(self._sidecar, discovery_config)

        self._discovery_result = discovery.discover(general_text_activations)

        # Initialize detector with discovered features
        detector_config = self.config.to_detector_config()
        self._detector = AnomalyDetector(
            self._sidecar,
            self._discovery_result,
            detector_config,
        )

        self._is_loaded = True
        logger.info(
            f"Discovery complete. Found {self._discovery_result.n_selected_features} "
            "incorrectness features."
        )

        return self._discovery_result

    def detect(self, text: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect anomalies in input text.

        This is the main runtime interface for Phase 1.

        Args:
            text: Input prompt or code (single or batch)

        Returns:
            DetectionResult(s) with anomaly score and flag
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Pipeline not ready. Run run_discovery() first or load() from artifacts."
            )

        return self._detector.detect(text)

    def should_verify(self, result: DetectionResult) -> bool:
        """
        Check if result should be routed to Phase 4 formal verification.

        Args:
            result: Detection result

        Returns:
            Whether to route to verification
        """
        return self._detector.should_route_to_verification(result)

    def calibrate(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
        target_recall: float = 0.8,
    ) -> float:
        """
        Calibrate detection threshold on validation data.

        Args:
            correct_prompts: Prompts with correct code
            incorrect_prompts: Prompts with incorrect code
            target_recall: Target recall for incorrect detection

        Returns:
            Calibrated threshold
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not ready.")

        return self._detector.calibrate_threshold(
            correct_prompts,
            incorrect_prompts,
            target_recall,
        )

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save pipeline artifacts for later loading.

        Saves:
        - config.yaml: Pipeline configuration
        - sae.pt: SAE weights
        - discovery_result.json: Discovered features
        - detector_config.json: Detector settings

        Args:
            output_dir: Output directory (defaults to config.output_dir)
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving pipeline artifacts to {output_dir}")

        # Save config
        self.config.to_yaml(output_dir / "config.yaml")

        # Save SAE
        if self._sidecar and self._sidecar._sae:
            self._sidecar._sae.save(output_dir / "sae.pt")

        # Save discovery result
        if self._discovery_result:
            self._discovery_result.save(output_dir / "discovery_result.json")

        # Save detector config
        if self._detector:
            self._detector.save_config(output_dir / "detector_config.json")

        logger.info("Pipeline artifacts saved successfully")

    @property
    def sidecar(self) -> SidecarModel:
        """Get the sidecar model."""
        if self._sidecar is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._sidecar

    @property
    def discovery_result(self) -> FeatureDiscoveryResult:
        """Get the discovery result."""
        if self._discovery_result is None:
            raise RuntimeError("Discovery not run. Call run_discovery() first.")
        return self._discovery_result

    @property
    def detector(self) -> AnomalyDetector:
        """Get the detector."""
        if self._detector is None:
            raise RuntimeError("Detector not initialized.")
        return self._detector

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "model_name": self.config.model_name,
            "target_layer": self.config.target_layer,
            "is_loaded": self._is_loaded,
        }

        if self._discovery_result:
            stats["n_incorrectness_features"] = self._discovery_result.n_selected_features
            stats["n_correct_samples"] = self._discovery_result.n_correct_samples
            stats["n_incorrect_samples"] = self._discovery_result.n_incorrect_samples

        if self._detector:
            stats["anomaly_threshold"] = self._detector.config.anomaly_threshold

        return stats


def create_default_pipeline() -> SAEAnomalyPipeline:
    """
    Create a pipeline with recommended defaults.

    Uses:
    - Llama-3.1-8B-Instruct as sidecar model
    - Layer 12 for extraction
    - 8x SAE expansion factor
    - 100 top incorrectness features

    Returns:
        Configured SAEAnomalyPipeline
    """
    config = PipelineConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        target_layer=12,
        device="auto",
        dtype="float16",
        quantization="8bit",
        sae_expansion_factor=8,
        top_k_features=100,
        min_t_statistic=2.0,
        anomaly_threshold=0.5,
    )
    return SAEAnomalyPipeline(config)
