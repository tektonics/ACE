"""Activation caching and hook management for ACE-Code."""

from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ActivationRecord:
    """Record of a single activation capture."""

    layer_idx: int
    position: str  # 'all', 'final', or specific index
    activation: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActivationCache:
    """
    Cache for storing and managing model activations.

    Supports efficient storage and retrieval of activations
    for SAE analysis and feature extraction.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize activation cache.

        Args:
            max_size: Maximum number of records to store (None for unlimited)
        """
        self.max_size = max_size
        self._cache: Dict[str, ActivationRecord] = {}
        self._order: List[str] = []  # For LRU eviction

    def store(
        self,
        key: str,
        layer_idx: int,
        activation: torch.Tensor,
        position: str = "all",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an activation in the cache.

        Args:
            key: Unique identifier for this activation
            layer_idx: Layer index the activation came from
            activation: The activation tensor
            position: Position descriptor ('all', 'final', etc.)
            metadata: Additional metadata
        """
        if self.max_size is not None and len(self._cache) >= self.max_size:
            # Evict oldest entry
            if self._order:
                oldest_key = self._order.pop(0)
                del self._cache[oldest_key]

        record = ActivationRecord(
            layer_idx=layer_idx,
            position=position,
            activation=activation.detach().clone(),
            metadata=metadata or {},
        )

        self._cache[key] = record
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

        logger.debug(f"Stored activation: {key}, shape={activation.shape}")

    def get(self, key: str) -> Optional[ActivationRecord]:
        """Retrieve an activation record by key."""
        return self._cache.get(key)

    def get_activation(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve just the activation tensor by key."""
        record = self._cache.get(key)
        return record.activation if record else None

    def get_all_for_layer(self, layer_idx: int) -> List[ActivationRecord]:
        """Get all cached activations for a specific layer."""
        return [r for r in self._cache.values() if r.layer_idx == layer_idx]

    def clear(self) -> None:
        """Clear all cached activations."""
        self._cache.clear()
        self._order.clear()

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class HookManager:
    """
    Manager for registering and coordinating model hooks.

    Provides a clean interface for adding/removing hooks
    and collecting activations from specific layers.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize hook manager.

        Args:
            model: The model to manage hooks for
        """
        self.model = model
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._activation_store: Dict[str, torch.Tensor] = {}

    def _make_hook(
        self,
        name: str,
        extract_fn: Optional[Callable] = None,
    ) -> Callable:
        """
        Create a hook function that stores activations.

        Args:
            name: Name to store activation under
            extract_fn: Optional function to extract specific values from output

        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if extract_fn is not None:
                value = extract_fn(output)
            else:
                # Default: get first element if tuple
                if isinstance(output, tuple):
                    value = output[0]
                else:
                    value = output
            self._activation_store[name] = value.detach()

        return hook

    def register_hook(
        self,
        name: str,
        module: nn.Module,
        hook_type: str = "forward",
        extract_fn: Optional[Callable] = None,
    ) -> str:
        """
        Register a hook on a module.

        Args:
            name: Unique name for the hook
            module: Module to hook
            hook_type: Type of hook ('forward', 'backward', 'forward_pre')
            extract_fn: Optional extraction function

        Returns:
            Hook name
        """
        hook_fn = self._make_hook(name, extract_fn)

        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = module.register_full_backward_hook(hook_fn)
        elif hook_type == "forward_pre":
            handle = module.register_forward_pre_hook(
                lambda m, i: self._activation_store.update({f"{name}_input": i[0].detach()})
            )
            handle = module.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

        self._hooks[name] = handle
        logger.debug(f"Registered {hook_type} hook: {name}")
        return name

    def register_layer_hook(
        self,
        layer_idx: int,
        layer_getter: Callable[[nn.Module, int], nn.Module],
        name_prefix: str = "layer",
    ) -> str:
        """
        Register a hook on a specific layer.

        Args:
            layer_idx: Layer index
            layer_getter: Function to get layer module from model
            name_prefix: Prefix for hook name

        Returns:
            Hook name
        """
        name = f"{name_prefix}_{layer_idx}"
        layer = layer_getter(self.model, layer_idx)
        return self.register_hook(name, layer)

    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Get stored activation by name."""
        return self._activation_store.get(name)

    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        """Get all stored activations."""
        return dict(self._activation_store)

    def clear_activations(self) -> None:
        """Clear all stored activations."""
        self._activation_store.clear()

    def remove_hook(self, name: str) -> None:
        """Remove a specific hook."""
        if name in self._hooks:
            self._hooks[name].remove()
            del self._hooks[name]
            logger.debug(f"Removed hook: {name}")

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all hooks")

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_all_hooks()
        self.clear_activations()


def extract_final_token(
    activation: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract activation at the final token position.

    Args:
        activation: Shape (batch, seq_len, hidden_dim)
        attention_mask: Optional attention mask

    Returns:
        Shape (batch, hidden_dim)
    """
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(activation.size(0), device=activation.device)
        return activation[batch_indices, seq_lengths]
    else:
        return activation[:, -1, :]
