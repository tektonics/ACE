"""Tests for Anomaly Detector."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from ace_code.sae_anomaly.detector import (
    AnomalyDetector,
    DetectorConfig,
    DetectionResult,
    AnomalyLevel,
)
from ace_code.sae_anomaly.discovery import FeatureDiscoveryResult, DiscoveryConfig


class TestDetectorConfig:
    """Tests for detector configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DetectorConfig()

        assert config.anomaly_threshold == 0.5
        assert config.use_weighted_score == True
        assert config.top_k_features_to_report == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = DetectorConfig(
            anomaly_threshold=0.7,
            low_threshold=0.4,
            use_weighted_score=False,
        )

        assert config.anomaly_threshold == 0.7
        assert config.low_threshold == 0.4
        assert config.use_weighted_score == False


class TestAnomalyLevel:
    """Tests for anomaly level enum."""

    def test_level_values(self):
        """Test that all levels have correct values."""
        assert AnomalyLevel.NONE.value == "none"
        assert AnomalyLevel.LOW.value == "low"
        assert AnomalyLevel.MEDIUM.value == "medium"
        assert AnomalyLevel.HIGH.value == "high"
        assert AnomalyLevel.CRITICAL.value == "critical"


class TestDetectionResult:
    """Tests for detection result."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DetectionResult(
            is_anomaly=True,
            anomaly_score=0.75,
            anomaly_level=AnomalyLevel.HIGH,
            threshold=0.5,
            n_active_features=5,
            active_feature_indices=[1, 2, 3, 4, 5],
            active_feature_values=[0.1, 0.2, 0.3, 0.4, 0.5],
            top_features=[{"feature_index": 1, "activation": 0.5}],
            input_text="test code",
            model_name="test-model",
            target_layer=12,
        )

        d = result.to_dict()

        assert d["is_anomaly"] == True
        assert d["anomaly_score"] == 0.75
        assert d["anomaly_level"] == "high"
        assert d["threshold"] == 0.5
        assert d["n_active_features"] == 5

    def test_long_text_truncation(self):
        """Test that long text is truncated in dict."""
        long_text = "x" * 500

        result = DetectionResult(
            is_anomaly=False,
            anomaly_score=0.1,
            anomaly_level=AnomalyLevel.NONE,
            threshold=0.5,
            n_active_features=0,
            active_feature_indices=[],
            active_feature_values=[],
            top_features=[],
            input_text=long_text,
            model_name="test",
            target_layer=12,
        )

        d = result.to_dict()

        assert len(d["input_text"]) < len(long_text)
        assert "..." in d["input_text"]


class TestAnomalyDetector:
    """Tests for anomaly detector."""

    @pytest.fixture
    def mock_sidecar(self):
        """Create mock sidecar model."""
        sidecar = Mock()
        sidecar.model.device = "cpu"
        sidecar.config.model_name = "test-model"
        sidecar.config.target_layer = 12

        # Mock get_sae_features to return tensor
        def mock_features(text, position="final"):
            n_latents = 100
            # Return random features
            return torch.rand(1, n_latents)

        sidecar.get_sae_features = mock_features

        return sidecar

    @pytest.fixture
    def mock_discovery(self):
        """Create mock discovery result."""
        return FeatureDiscoveryResult(
            incorrectness_features=np.array([0, 1, 2, 3, 4]),
            feature_t_statistics=np.array([5.0, 4.0, 3.0, 2.5, 2.0]),
            feature_weights=np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
            n_correct_samples=50,
            n_incorrect_samples=50,
            n_total_features=100,
            n_selected_features=5,
            model_name="test-model",
            target_layer=12,
            config=DiscoveryConfig(),
        )

    @pytest.fixture
    def detector(self, mock_sidecar, mock_discovery):
        """Create detector with mocks."""
        config = DetectorConfig(anomaly_threshold=0.3)
        return AnomalyDetector(mock_sidecar, mock_discovery, config)

    def test_init(self, detector, mock_discovery):
        """Test detector initialization."""
        assert len(detector._feature_indices) == 5
        assert len(detector._feature_weights) == 5

    def test_determine_level(self, detector):
        """Test anomaly level determination."""
        assert detector._determine_level(0.1) == AnomalyLevel.NONE
        assert detector._determine_level(0.4) == AnomalyLevel.LOW
        assert detector._determine_level(0.6) == AnomalyLevel.MEDIUM
        assert detector._determine_level(0.8) == AnomalyLevel.HIGH
        assert detector._determine_level(0.95) == AnomalyLevel.CRITICAL

    def test_detect_single(self, detector):
        """Test single input detection."""
        result = detector.detect("def test(): pass")

        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_anomaly, bool)
        assert result.anomaly_score >= 0
        assert result.model_name == "test-model"
        assert result.target_layer == 12

    def test_detect_batch(self, detector):
        """Test batch detection."""
        texts = ["def a(): pass", "def b(): pass", "def c(): pass"]
        results = detector.detect(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DetectionResult)

    def test_should_route_to_verification(self, detector):
        """Test routing decision."""
        # Anomaly should be routed
        anomaly_result = Mock()
        anomaly_result.is_anomaly = True
        assert detector.should_route_to_verification(anomaly_result) == True

        # Non-anomaly should not be routed
        normal_result = Mock()
        normal_result.is_anomaly = False
        assert detector.should_route_to_verification(normal_result) == False

    def test_get_detection_summary(self, detector):
        """Test summary statistics."""
        results = [
            DetectionResult(
                is_anomaly=True, anomaly_score=0.7,
                anomaly_level=AnomalyLevel.HIGH, threshold=0.5,
                n_active_features=3, active_feature_indices=[],
                active_feature_values=[], top_features=[],
                input_text="", model_name="", target_layer=0,
            ),
            DetectionResult(
                is_anomaly=False, anomaly_score=0.2,
                anomaly_level=AnomalyLevel.NONE, threshold=0.5,
                n_active_features=1, active_feature_indices=[],
                active_feature_values=[], top_features=[],
                input_text="", model_name="", target_layer=0,
            ),
        ]

        summary = detector.get_detection_summary(results)

        assert summary["n_total"] == 2
        assert summary["n_anomalies"] == 1
        assert summary["anomaly_rate"] == 0.5
        assert summary["mean_score"] == 0.45
        assert summary["max_score"] == 0.7
        assert summary["min_score"] == 0.2
