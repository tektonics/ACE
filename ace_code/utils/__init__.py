"""Utility functions for ACE-Code."""

from ace_code.utils.device import get_device, DeviceManager
from ace_code.utils.logging import setup_logging, get_logger
from ace_code.utils.statistics import (
    compute_t_statistic,
    compute_feature_significance,
    welch_t_test,
)

__all__ = [
    "get_device",
    "DeviceManager",
    "setup_logging",
    "get_logger",
    "compute_t_statistic",
    "compute_feature_significance",
    "welch_t_test",
]
