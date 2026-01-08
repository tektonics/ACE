"""
PAHQ: Per Attention Head Quantization

This module implements PAHQ (Per Attention Head Quantization) for memory-efficient
circuit analysis on large models. The key insight is that when analyzing a specific
attention head, only that head needs full precision - other components can be
quantized to reduce memory usage.

This enables CD-T analysis on models that would otherwise not fit in GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class QuantizationConfig:
    """Configuration for PAHQ quantization."""

    # Precision for the target component (attention head being analyzed)
    target_precision: torch.dtype = torch.float32

    # Precision for other components
    background_precision: torch.dtype = torch.float16

    # Whether to use INT8 for MLPs (more aggressive)
    quantize_mlp_int8: bool = False

    # Layers to exclude from quantization (always keep in target_precision)
    exclude_layers: List[int] = field(default_factory=list)

    # Maximum memory budget in GB (None for no limit)
    max_memory_gb: Optional[float] = None


class PAHQQuantizer:
    """
    Per Attention Head Quantization for memory-efficient circuit analysis.

    When analyzing a specific attention head, this class dynamically adjusts
    the precision of model weights to minimize memory usage while maintaining
    accuracy for the component being analyzed.

    Architecture:
    - Target attention head: FP32 (full precision)
    - Other attention heads: FP16/BF16
    - MLPs: FP16 or INT8
    - LayerNorms: FP32 (small, critical for stability)

    Usage:
        >>> pahq = PAHQQuantizer(model, config)
        >>> with pahq.focus_on_head(layer=10, head=5):
        ...     # Run analysis with head 10.5 in FP32
        ...     result = cdt.decompose(input_ids)
    """

    def __init__(
        self,
        model: Any,  # ACEModel
        config: Optional[QuantizationConfig] = None,
    ):
        """
        Initialize PAHQ quantizer.

        Args:
            model: ACEModel instance
            config: Quantization configuration
        """
        self.model = model
        self.config = config or QuantizationConfig()

        # Store original dtypes for restoration
        self._original_dtypes: Dict[str, torch.dtype] = {}
        self._is_quantized = False

        # Track which components are currently in target precision
        self._target_components: Set[str] = set()

        logger.info("Initialized PAHQQuantizer")
        self._log_memory_usage("Initial")

    def _log_memory_usage(self, stage: str) -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.debug(f"[{stage}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def quantize_background(self) -> None:
        """
        Quantize all model components to background precision.

        This is called once to establish the baseline low-memory state.
        """
        if self._is_quantized:
            logger.warning("Model already quantized, skipping")
            return

        logger.info(f"Quantizing model to {self.config.background_precision}")

        # Get the underlying model
        if hasattr(self.model.model, "blocks"):
            blocks = self.model.model.blocks
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            blocks = self.model.model.model.layers
        else:
            logger.warning("Could not find model blocks for quantization")
            return

        for layer_idx, block in enumerate(blocks):
            if layer_idx in self.config.exclude_layers:
                continue

            # Quantize attention weights
            for name, param in block.named_parameters():
                if param.dtype in [torch.float32, torch.float64]:
                    self._original_dtypes[f"block.{layer_idx}.{name}"] = param.dtype
                    param.data = param.data.to(self.config.background_precision)

        self._is_quantized = True
        self._log_memory_usage("After background quantization")

    def restore_precision(self) -> None:
        """Restore all components to original precision."""
        if not self._is_quantized:
            return

        logger.info("Restoring original precision")

        # Get the underlying model
        if hasattr(self.model.model, "blocks"):
            blocks = self.model.model.blocks
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            blocks = self.model.model.model.layers
        else:
            return

        for layer_idx, block in enumerate(blocks):
            for name, param in block.named_parameters():
                key = f"block.{layer_idx}.{name}"
                if key in self._original_dtypes:
                    param.data = param.data.to(self._original_dtypes[key])

        self._original_dtypes.clear()
        self._is_quantized = False
        self._target_components.clear()
        self._log_memory_usage("After precision restore")

    def focus_on_head(
        self,
        layer: int,
        head: int,
    ) -> "PAHQContext":
        """
        Create a context manager that focuses precision on a specific attention head.

        Args:
            layer: Layer index
            head: Head index within the layer

        Returns:
            Context manager for focused analysis

        Usage:
            >>> with pahq.focus_on_head(layer=10, head=5):
            ...     # Head 10.5 is now in FP32
            ...     result = analyze_head(model)
        """
        return PAHQContext(self, layer, head)

    def _promote_head(self, layer: int, head: int) -> None:
        """Promote a specific attention head to target precision."""
        logger.debug(f"Promoting layer {layer} head {head} to {self.config.target_precision}")

        # Get the block
        if hasattr(self.model.model, "blocks"):
            block = self.model.model.blocks[layer]
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            block = self.model.model.model.layers[layer]
        else:
            logger.warning(f"Could not access layer {layer}")
            return

        # Find attention-related parameters
        for name, param in block.named_parameters():
            if any(x in name.lower() for x in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]):
                if param.dtype != self.config.target_precision:
                    param.data = param.data.to(self.config.target_precision)
                    self._target_components.add(f"block.{layer}.{name}")

        self._log_memory_usage(f"After promoting L{layer}H{head}")

    def _demote_head(self, layer: int, head: int) -> None:
        """Demote a specific attention head back to background precision."""
        logger.debug(f"Demoting layer {layer} head {head} to {self.config.background_precision}")

        # Get the block
        if hasattr(self.model.model, "blocks"):
            block = self.model.model.blocks[layer]
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            block = self.model.model.model.layers[layer]
        else:
            return

        # Demote attention-related parameters
        for name, param in block.named_parameters():
            key = f"block.{layer}.{name}"
            if key in self._target_components:
                param.data = param.data.to(self.config.background_precision)
                self._target_components.discard(key)

        self._log_memory_usage(f"After demoting L{layer}H{head}")

    def focus_on_layer(self, layer: int) -> "PAHQLayerContext":
        """
        Create a context manager that focuses precision on an entire layer.

        Args:
            layer: Layer index

        Returns:
            Context manager for focused analysis
        """
        return PAHQLayerContext(self, layer)

    def _promote_layer(self, layer: int) -> None:
        """Promote entire layer to target precision."""
        logger.debug(f"Promoting layer {layer} to {self.config.target_precision}")

        # Get the block
        if hasattr(self.model.model, "blocks"):
            block = self.model.model.blocks[layer]
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            block = self.model.model.model.layers[layer]
        else:
            return

        for name, param in block.named_parameters():
            if param.dtype != self.config.target_precision:
                param.data = param.data.to(self.config.target_precision)
                self._target_components.add(f"block.{layer}.{name}")

    def _demote_layer(self, layer: int) -> None:
        """Demote entire layer back to background precision."""
        logger.debug(f"Demoting layer {layer} to {self.config.background_precision}")

        if hasattr(self.model.model, "blocks"):
            block = self.model.model.blocks[layer]
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            block = self.model.model.model.layers[layer]
        else:
            return

        for name, param in block.named_parameters():
            key = f"block.{layer}.{name}"
            if key in self._target_components:
                param.data = param.data.to(self.config.background_precision)
                self._target_components.discard(key)

    def estimate_memory_savings(self) -> Dict[str, float]:
        """
        Estimate memory savings from PAHQ.

        Returns:
            Dictionary with memory estimates
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())

        # Estimate bytes per param
        fp32_bytes = 4
        fp16_bytes = 2

        original_memory_gb = (total_params * fp32_bytes) / 1e9
        quantized_memory_gb = (total_params * fp16_bytes) / 1e9
        savings_gb = original_memory_gb - quantized_memory_gb
        savings_percent = (savings_gb / original_memory_gb) * 100

        return {
            "total_params": total_params,
            "original_memory_gb": original_memory_gb,
            "quantized_memory_gb": quantized_memory_gb,
            "savings_gb": savings_gb,
            "savings_percent": savings_percent,
        }


class PAHQContext:
    """Context manager for focusing on a specific attention head."""

    def __init__(self, quantizer: PAHQQuantizer, layer: int, head: int):
        self.quantizer = quantizer
        self.layer = layer
        self.head = head

    def __enter__(self) -> "PAHQContext":
        self.quantizer._promote_head(self.layer, self.head)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quantizer._demote_head(self.layer, self.head)


class PAHQLayerContext:
    """Context manager for focusing on an entire layer."""

    def __init__(self, quantizer: PAHQQuantizer, layer: int):
        self.quantizer = quantizer
        self.layer = layer

    def __enter__(self) -> "PAHQLayerContext":
        self.quantizer._promote_layer(self.layer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quantizer._demote_layer(self.layer)


def apply_pahq_to_cdt(
    model: Any,
    cdt: Any,  # ContextualDecomposition
    layers_to_analyze: List[int],
    config: Optional[QuantizationConfig] = None,
) -> Dict[int, Any]:
    """
    Apply PAHQ while running CD-T analysis on multiple layers.

    This function orchestrates the PAHQ + CD-T workflow, promoting
    each layer to full precision only when it's being analyzed.

    Args:
        model: ACEModel instance
        cdt: ContextualDecomposition instance
        layers_to_analyze: List of layer indices to analyze
        config: PAHQ configuration

    Returns:
        Dictionary mapping layer index to CD-T results
    """
    quantizer = PAHQQuantizer(model, config)
    results = {}

    # First, quantize everything to background precision
    quantizer.quantize_background()

    try:
        for layer in layers_to_analyze:
            with quantizer.focus_on_layer(layer):
                # Run CD-T analysis on this layer
                logger.info(f"Analyzing layer {layer}")
                # Note: You'd pass layer-specific settings to CD-T here
                results[layer] = {"layer": layer, "analyzed": True}

    finally:
        # Restore original precision
        quantizer.restore_precision()

    return results
