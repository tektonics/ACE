"""
Device management utilities for ACE-Code.
"""

import gc
from typing import Optional, Union

import torch
from loguru import logger


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Get the best available device.

    Args:
        preferred: Preferred device ("cuda", "mps", "cpu")

    Returns:
        torch.device object
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif preferred == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device")

    return device


def clear_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Clear GPU memory and run garbage collection.

    Args:
        device: Device to clear memory for (None for all)
    """
    gc.collect()

    if torch.cuda.is_available():
        if device is None:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Log memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.debug(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "device_name": torch.cuda.get_device_name(),
        "device_count": torch.cuda.device_count(),
    }
