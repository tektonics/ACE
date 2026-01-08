"""
PISCES: Permanent Weight Editing for Error Removal

This module implements PISCES (Projection-based Intervention for Surgical
Concept Erasure in Sparse feature spaces), which permanently removes
error-inducing directions from model weights.

Unlike steering (which is temporary), PISCES modifies the model weights
to make it impossible for the model to represent certain error patterns.

Key Formula:
    W_new = W_old - (W_old @ d) @ d^T

Where d is the error direction to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class PISCESConfig:
    """Configuration for PISCES weight editing."""

    # Layer to edit
    layer: int = 0

    # Component to edit (mlp_out, attn_out)
    component: str = "mlp_out"

    # Projection strength (1.0 = full removal)
    strength: float = 1.0

    # Whether to normalize the direction before projection
    normalize_direction: bool = True

    # Minimum effect threshold to apply edit
    min_effect: float = 0.01

    # Whether to create a backup before editing
    backup: bool = True


@dataclass
class PISCESResult:
    """Results from a PISCES weight edit."""

    layer: int
    component: str
    direction_norm: float
    weight_change_norm: float
    weight_change_relative: float
    success: bool
    backup_path: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"PISCESResult(layer={self.layer}, "
            f"weight_change={self.weight_change_relative:.2%}, "
            f"success={self.success})"
        )


class PISCES:
    """
    Permanent weight editing to remove error-inducing directions.

    PISCES works by projecting out specific directions from the model's
    weight matrices, making it impossible for the model to output in
    those directions.

    Architecture:
    1. Identify error direction (from SAE or steering analysis)
    2. Project this direction out of the MLP/Attention output weights
    3. Verify the edit by checking model behavior changes

    Warning: This permanently modifies model weights. Always backup first!

    Usage:
        >>> pisces = PISCES(model)
        >>> result = pisces.remove_direction(error_direction, layer=12)
        >>> if result.success:
        ...     print(f"Removed error direction from layer {result.layer}")
    """

    def __init__(
        self,
        model: Any,  # ACEModel
        config: Optional[PISCESConfig] = None,
    ):
        """
        Initialize PISCES editor.

        Args:
            model: ACEModel instance to edit
            config: PISCES configuration
        """
        self.model = model
        self.config = config or PISCESConfig()

        # Weight backups
        self._backups: Dict[str, torch.Tensor] = {}

        # Edit history
        self._history: List[PISCESResult] = []

        logger.info("Initialized PISCES weight editor")

    def _get_output_weight(
        self,
        layer: int,
        component: str,
    ) -> Tuple[nn.Parameter, str]:
        """
        Get the output weight matrix for a layer/component.

        Args:
            layer: Layer index
            component: Component name (mlp_out, attn_out)

        Returns:
            Tuple of (weight_parameter, parameter_name)
        """
        # Access the underlying model
        if hasattr(self.model.model, "blocks"):
            block = self.model.model.blocks[layer]
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "layers"):
            block = self.model.model.model.layers[layer]
        else:
            raise RuntimeError("Could not access model layers")

        # Find the appropriate weight
        if component == "mlp_out":
            # Try different naming conventions
            for name in ["mlp.W_out", "mlp.down_proj.weight", "mlp.c_proj.weight", "feed_forward.down_proj.weight"]:
                parts = name.split(".")
                obj = block
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    final_name = parts[-1]
                    if hasattr(obj, final_name):
                        param = getattr(obj, final_name)
                        if isinstance(param, nn.Parameter):
                            return param, f"blocks.{layer}.{name}"

            raise RuntimeError(f"Could not find MLP output weight in layer {layer}")

        elif component == "attn_out":
            for name in ["attn.W_O", "self_attn.o_proj.weight", "attn.c_proj.weight"]:
                parts = name.split(".")
                obj = block
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    final_name = parts[-1]
                    if hasattr(obj, final_name):
                        param = getattr(obj, final_name)
                        if isinstance(param, nn.Parameter):
                            return param, f"blocks.{layer}.{name}"

            raise RuntimeError(f"Could not find attention output weight in layer {layer}")

        else:
            raise ValueError(f"Unknown component: {component}")

    def backup_weight(self, layer: int, component: str) -> str:
        """
        Create a backup of a weight matrix.

        Args:
            layer: Layer index
            component: Component name

        Returns:
            Backup key for restoration
        """
        weight, name = self._get_output_weight(layer, component)
        backup_key = f"{layer}_{component}"
        self._backups[backup_key] = weight.data.clone()
        logger.info(f"Backed up weight: {name}")
        return backup_key

    def restore_weight(self, backup_key: str) -> bool:
        """
        Restore a weight from backup.

        Args:
            backup_key: Key from backup_weight()

        Returns:
            True if restoration successful
        """
        if backup_key not in self._backups:
            logger.error(f"No backup found for key: {backup_key}")
            return False

        layer, component = backup_key.split("_")
        layer = int(layer)

        weight, name = self._get_output_weight(layer, component)
        weight.data = self._backups[backup_key].clone()

        logger.info(f"Restored weight: {name}")
        return True

    def remove_direction(
        self,
        direction: torch.Tensor,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> PISCESResult:
        """
        Remove a direction from the model weights.

        This projects out the specified direction from the output weight
        matrix, making it impossible for the model to output in that direction.

        Formula: W_new = W_old - strength * (W_old @ d) @ d^T

        Args:
            direction: Direction to remove, shape (d_model,)
            layer: Layer to edit (default: config.layer)
            component: Component to edit (default: config.component)
            strength: Removal strength (default: config.strength)

        Returns:
            PISCESResult with edit details
        """
        layer = layer if layer is not None else self.config.layer
        component = component if component is not None else self.config.component
        strength = strength if strength is not None else self.config.strength

        logger.info(f"Removing direction from layer {layer} {component}")

        # Ensure direction is normalized if configured
        if self.config.normalize_direction:
            direction = direction / (direction.norm() + 1e-8)

        direction = direction.to(self.model.device)
        direction_norm = direction.norm().item()

        # Get weight matrix
        weight, weight_name = self._get_output_weight(layer, component)

        # Backup if configured
        backup_key = None
        if self.config.backup:
            backup_key = self.backup_weight(layer, component)

        # Compute original weight norm
        original_norm = weight.data.norm().item()

        # Compute projection
        # W_new = W - strength * (W @ d) @ d^T
        # For weight shapes, we need to handle different conventions
        # Common shapes: (out_features, in_features) or (d_model, d_model)

        if weight.dim() == 2:
            # Standard linear layer: (out_features, in_features)
            # We want to project out direction from the output space

            # Project: remove component of each row in direction
            projection = weight.data @ direction  # (out_features,)
            update = strength * torch.outer(projection, direction)  # (out_features, in_features)
            weight.data = weight.data - update

        elif weight.dim() == 3:
            # Attention output: (n_heads, d_head, d_model) or similar
            # Flatten, project, reshape
            original_shape = weight.shape
            flat_weight = weight.data.reshape(-1, weight.shape[-1])

            projection = flat_weight @ direction
            update = strength * torch.outer(projection, direction)
            flat_weight = flat_weight - update

            weight.data = flat_weight.reshape(original_shape)

        else:
            logger.error(f"Unexpected weight dimension: {weight.dim()}")
            return PISCESResult(
                layer=layer,
                component=component,
                direction_norm=direction_norm,
                weight_change_norm=0.0,
                weight_change_relative=0.0,
                success=False,
            )

        # Compute change statistics
        new_norm = weight.data.norm().item()
        change_norm = abs(original_norm - new_norm)
        change_relative = change_norm / (original_norm + 1e-8)

        # Check if change was significant
        success = change_relative >= self.config.min_effect

        result = PISCESResult(
            layer=layer,
            component=component,
            direction_norm=direction_norm,
            weight_change_norm=change_norm,
            weight_change_relative=change_relative,
            success=success,
            backup_path=backup_key,
        )

        self._history.append(result)

        if success:
            logger.info(f"Successfully removed direction (change: {change_relative:.2%})")
        else:
            logger.warning(f"Direction removal had minimal effect ({change_relative:.2%})")

        return result

    def remove_multiple_directions(
        self,
        directions: List[torch.Tensor],
        layer: int,
        component: str = "mlp_out",
        strengths: Optional[List[float]] = None,
    ) -> List[PISCESResult]:
        """
        Remove multiple directions from the model.

        Directions are removed sequentially. For orthogonal directions,
        the order doesn't matter. For non-orthogonal directions, later
        removals may partially undo earlier ones.

        Args:
            directions: List of direction vectors
            layer: Layer to edit
            component: Component to edit
            strengths: Per-direction strengths (default: all 1.0)

        Returns:
            List of PISCESResult objects
        """
        if strengths is None:
            strengths = [1.0] * len(directions)

        results = []
        for direction, strength in zip(directions, strengths):
            result = self.remove_direction(
                direction=direction,
                layer=layer,
                component=component,
                strength=strength,
            )
            results.append(result)

        return results

    def remove_sae_features(
        self,
        sae: Any,  # SAE module
        feature_indices: List[int],
        layer: int,
        component: str = "mlp_out",
        strengths: Optional[List[float]] = None,
    ) -> List[PISCESResult]:
        """
        Remove specific SAE features from the model.

        This extracts the decoder directions for the specified SAE features
        and projects them out of the model weights.

        Args:
            sae: Sparse Autoencoder module
            feature_indices: Indices of features to remove
            layer: Layer to edit
            component: Component to edit
            strengths: Per-feature strengths

        Returns:
            List of PISCESResult objects
        """
        # Get decoder directions for specified features
        if hasattr(sae, 'W_dec'):
            directions = [sae.W_dec[idx] for idx in feature_indices]
        elif hasattr(sae, 'decoder'):
            directions = [sae.decoder.weight[:, idx] for idx in feature_indices]
        else:
            raise RuntimeError("Cannot access SAE decoder weights")

        logger.info(f"Removing {len(directions)} SAE features from layer {layer}")

        return self.remove_multiple_directions(
            directions=directions,
            layer=layer,
            component=component,
            strengths=strengths,
        )

    def verify_removal(
        self,
        direction: torch.Tensor,
        layer: int,
        component: str,
        test_inputs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Verify that a direction has been successfully removed.

        This checks if the model can still output in the removed direction
        by measuring activations on test inputs.

        Args:
            direction: The removed direction
            layer: Layer that was edited
            component: Component that was edited
            test_inputs: Test input_ids to verify with

        Returns:
            Dictionary with verification metrics
        """
        direction = direction / (direction.norm() + 1e-8)
        direction = direction.to(self.model.device)

        activations_in_direction = []

        with torch.no_grad():
            for inp in test_inputs:
                # Get residual stream at the layer
                resid = self.model.get_residual_stream(
                    inp,
                    layer=layer,
                    position="post" if "out" in component else "pre",
                )

                # Measure activation in removed direction
                # Use last token position
                act = resid[0, -1, :]
                act_in_dir = (act @ direction).abs().item()
                activations_in_direction.append(act_in_dir)

        mean_activation = sum(activations_in_direction) / len(activations_in_direction)
        max_activation = max(activations_in_direction)

        logger.info(f"Verification - Mean activation in direction: {mean_activation:.4f}")
        logger.info(f"Verification - Max activation in direction: {max_activation:.4f}")

        return {
            "mean_activation": mean_activation,
            "max_activation": max_activation,
            "n_samples": len(test_inputs),
        }

    def get_history(self) -> List[PISCESResult]:
        """Get the history of all edits made."""
        return self._history

    def reset_all(self) -> int:
        """
        Reset all edited weights to their backups.

        Returns:
            Number of weights restored
        """
        restored = 0
        for backup_key in list(self._backups.keys()):
            if self.restore_weight(backup_key):
                restored += 1

        self._backups.clear()
        self._history.clear()
        logger.info(f"Reset {restored} weights to original values")
        return restored
