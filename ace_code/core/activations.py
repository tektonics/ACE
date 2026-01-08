"""
Activation Cache and Collection Utilities

This module provides utilities for collecting, storing, and manipulating
activation tensors from transformer models during forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class ActivationCache:
    """
    A structured cache for storing and accessing model activations.

    This class provides efficient storage and retrieval of activation
    tensors collected during model forward passes, with support for
    selective caching and memory management.
    """

    _cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    _metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize internal storage."""
        self._cache = {}
        self._metadata = {}

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get an activation tensor by key."""
        if key not in self._cache:
            raise KeyError(f"Activation '{key}' not found in cache. Available: {list(self._cache.keys())}")
        return self._cache[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Store an activation tensor."""
        self._cache[key] = value
        self._metadata[key] = {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    def __len__(self) -> int:
        """Get the number of cached activations."""
        return len(self._cache)

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())

    def values(self) -> List[torch.Tensor]:
        """Get all cached tensors."""
        return list(self._cache.values())

    def items(self) -> List[Tuple[str, torch.Tensor]]:
        """Get all (key, tensor) pairs."""
        return list(self._cache.items())

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Get an activation with optional default."""
        return self._cache.get(key, default)

    def get_residual_stream(
        self,
        layer: int,
        position: str = "post",
    ) -> torch.Tensor:
        """
        Get residual stream activations for a specific layer.

        Args:
            layer: Layer index
            position: "pre" or "post" the layer

        Returns:
            Residual stream tensor
        """
        key = f"blocks.{layer}.hook_resid_{position}"
        return self[key]

    def get_attention_output(self, layer: int) -> torch.Tensor:
        """Get attention output for a specific layer."""
        key = f"blocks.{layer}.hook_attn_out"
        return self[key]

    def get_mlp_output(self, layer: int) -> torch.Tensor:
        """Get MLP output for a specific layer."""
        key = f"blocks.{layer}.hook_mlp_out"
        return self[key]

    def get_attention_pattern(self, layer: int) -> torch.Tensor:
        """Get attention pattern for a specific layer."""
        key = f"blocks.{layer}.attn.hook_pattern"
        return self[key]

    def filter_by_layer(self, layer: int) -> "ActivationCache":
        """Get a new cache containing only activations from a specific layer."""
        filtered = ActivationCache()
        layer_prefix = f"blocks.{layer}."
        for key, value in self._cache.items():
            if key.startswith(layer_prefix):
                filtered[key] = value
        return filtered

    def filter_by_component(self, component: str) -> "ActivationCache":
        """Get a new cache containing only activations for a specific component."""
        filtered = ActivationCache()
        for key, value in self._cache.items():
            if component in key:
                filtered[key] = value
        return filtered

    def stack_layers(
        self,
        component: str = "resid_post",
        layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Stack activations from multiple layers into a single tensor.

        Args:
            component: Component name to stack (e.g., "resid_post")
            layers: List of layer indices to include (None for all)

        Returns:
            Stacked tensor of shape (n_layers, batch, seq_len, d_model)
        """
        tensors = []

        if layers is None:
            # Infer layers from cache keys
            layers = sorted(set(
                int(key.split(".")[1])
                for key in self._cache.keys()
                if key.startswith("blocks.")
            ))

        for layer in layers:
            key = f"blocks.{layer}.hook_{component}"
            if key in self._cache:
                tensors.append(self._cache[key])

        if not tensors:
            raise ValueError(f"No activations found for component '{component}'")

        return torch.stack(tensors, dim=0)

    def to(self, device: Union[str, torch.device]) -> "ActivationCache":
        """Move all cached tensors to a device."""
        for key in self._cache:
            self._cache[key] = self._cache[key].to(device)
        return self

    def detach(self) -> "ActivationCache":
        """Detach all cached tensors from computation graph."""
        for key in self._cache:
            self._cache[key] = self._cache[key].detach()
        return self

    def clone(self) -> "ActivationCache":
        """Create a deep copy of the cache."""
        new_cache = ActivationCache()
        for key, value in self._cache.items():
            new_cache[key] = value.clone()
        return new_cache

    def clear(self) -> None:
        """Clear all cached activations."""
        self._cache.clear()
        self._metadata.clear()

    def memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = 0
        for tensor in self._cache.values():
            total += tensor.numel() * tensor.element_size()
        return total

    def __repr__(self) -> str:
        """String representation of the cache."""
        mem_mb = self.memory_usage() / (1024 * 1024)
        return f"ActivationCache(n_activations={len(self)}, memory={mem_mb:.2f}MB)"


def collect_activations(
    model: Any,  # ACEModel
    input_ids: torch.Tensor,
    layers: Optional[List[int]] = None,
    components: Optional[List[str]] = None,
    detach: bool = True,
) -> ActivationCache:
    """
    Collect activations from a model for specified layers and components.

    This is a convenience function that wraps model.run_with_cache()
    and returns a structured ActivationCache object.

    Args:
        model: ACEModel instance
        input_ids: Input token IDs
        layers: List of layer indices to collect (None for all)
        components: List of components to collect (None for default set)
        detach: Whether to detach tensors from computation graph

    Returns:
        ActivationCache containing the collected activations

    Example:
        >>> cache = collect_activations(model, input_ids, layers=[10, 11, 12])
        >>> resid = cache.get_residual_stream(layer=11, position="post")
    """
    if components is None:
        components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]

    # Build names filter
    def names_filter(name: str) -> bool:
        # Check if any component matches
        for comp in components:
            if comp in name:
                # Check layer filter
                if layers is not None:
                    try:
                        layer_idx = int(name.split(".")[1])
                        if layer_idx in layers:
                            return True
                    except (IndexError, ValueError):
                        pass
                else:
                    return True
        return False

    # Run with cache
    _, raw_cache = model.run_with_cache(input_ids, names_filter=names_filter)

    # Build structured cache
    cache = ActivationCache()
    for key, value in raw_cache.items():
        if detach:
            cache[key] = value.detach()
        else:
            cache[key] = value

    logger.debug(f"Collected {len(cache)} activations, {cache.memory_usage() / 1e6:.2f}MB")

    return cache


def compute_activation_difference(
    cache_positive: ActivationCache,
    cache_negative: ActivationCache,
    layer: int,
    position: int = -1,
    component: str = "resid_post",
) -> torch.Tensor:
    """
    Compute the difference between activations from two caches.

    This is used for computing steering vectors by taking the
    difference between "positive" (correct) and "negative" (buggy)
    prompt activations.

    Args:
        cache_positive: Cache from positive (correct) prompt
        cache_negative: Cache from negative (buggy) prompt
        layer: Layer index to compute difference at
        position: Token position to compute difference at (-1 for last)
        component: Component to use (e.g., "resid_post")

    Returns:
        Difference vector of shape (d_model,)
    """
    key = f"blocks.{layer}.hook_{component}"

    act_pos = cache_positive[key]
    act_neg = cache_negative[key]

    # Get activations at specified position
    if position == -1:
        pos_vec = act_pos[0, -1, :]  # Last token
        neg_vec = act_neg[0, -1, :]
    else:
        pos_vec = act_pos[0, position, :]
        neg_vec = act_neg[0, position, :]

    return pos_vec - neg_vec


def compute_mean_activation(
    caches: List[ActivationCache],
    layer: int,
    position: int = -1,
    component: str = "resid_post",
) -> torch.Tensor:
    """
    Compute mean activation across multiple caches.

    Args:
        caches: List of activation caches
        layer: Layer index
        position: Token position (-1 for last)
        component: Component name

    Returns:
        Mean activation vector
    """
    key = f"blocks.{layer}.hook_{component}"

    vectors = []
    for cache in caches:
        act = cache[key]
        if position == -1:
            vectors.append(act[0, -1, :])
        else:
            vectors.append(act[0, position, :])

    return torch.stack(vectors).mean(dim=0)
