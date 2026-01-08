"""Device management utilities for ACE-Code."""

from typing import Optional, Union
import torch


class DeviceManager:
    """Manages device allocation for model and tensor operations."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize device manager.

        Args:
            device: Device specification ('cuda', 'cpu', 'auto', or specific like 'cuda:0')
        """
        self._device = self._resolve_device(device)

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        """Resolve device string to torch.device."""
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device."""
        return tensor.to(self._device)

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device.type == "cpu"

    def get_memory_info(self) -> dict:
        """Get memory information for CUDA devices."""
        if not self.is_cuda:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}

        return {
            "allocated": torch.cuda.memory_allocated(self._device),
            "reserved": torch.cuda.memory_reserved(self._device),
            "max_allocated": torch.cuda.max_memory_allocated(self._device),
        }


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device for computation.

    Args:
        device: Device specification ('cuda', 'cpu', 'auto', or None for auto)

    Returns:
        torch.device instance
    """
    manager = DeviceManager(device)
    return manager.device
