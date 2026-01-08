"""
Activation Steering for Code Models

This module implements inference-time steering, which modifies model behavior
by adding steering vectors to the residual stream during forward passes.

Key Concepts:
- Steering vectors are computed as the difference between activations on
  positive (correct) and negative (buggy) examples
- Adding the steering vector nudges the model toward "correct" behavior
- This is a temporary fix that doesn't modify model weights

Formula:
    v_steer = Activation(x+) - Activation(x-)
    x_new = x_old + alpha * v_steer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from ace_code.core.activations import ActivationCache, compute_activation_difference


@dataclass
class SteeringConfig:
    """Configuration for steering vector application."""

    # Layer(s) to apply steering
    layers: List[int] = field(default_factory=list)

    # Steering strength (alpha)
    strength: float = 1.0

    # Token positions to apply steering (-1 for all)
    positions: List[int] = field(default_factory=lambda: [-1])

    # Component to steer (resid_pre, resid_post, attn_out, mlp_out)
    component: str = "resid_post"

    # Whether to normalize the steering vector
    normalize: bool = True

    # Maximum norm for the steering vector
    max_norm: Optional[float] = None


class SteeringVector:
    """
    A steering vector for modifying model activations.

    Steering vectors capture the "direction" of a concept in activation
    space. By adding this direction during inference, we can push the
    model toward or away from that concept.

    Usage:
        >>> sv = SteeringVector.from_prompts(model, positive_prompt, negative_prompt, layer=12)
        >>> output = sv.apply(model, input_ids)
    """

    def __init__(
        self,
        vector: torch.Tensor,
        layer: int,
        component: str = "resid_post",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize steering vector.

        Args:
            vector: The steering vector, shape (d_model,)
            layer: Layer to apply the vector
            component: Component name (resid_pre, resid_post, etc.)
            metadata: Optional metadata about how the vector was created
        """
        self.vector = vector
        self.layer = layer
        self.component = component
        self.metadata = metadata or {}

        # Normalize by default
        self._normalized = False

    @property
    def norm(self) -> float:
        """Get the L2 norm of the steering vector."""
        return self.vector.norm().item()

    @property
    def device(self) -> torch.device:
        """Get the device of the steering vector."""
        return self.vector.device

    def normalize(self) -> "SteeringVector":
        """Return a normalized copy of this steering vector."""
        norm = self.vector.norm()
        if norm > 0:
            new_vector = self.vector / norm
        else:
            new_vector = self.vector
        sv = SteeringVector(
            vector=new_vector,
            layer=self.layer,
            component=self.component,
            metadata={**self.metadata, "normalized": True},
        )
        sv._normalized = True
        return sv

    def scale(self, factor: float) -> "SteeringVector":
        """Return a scaled copy of this steering vector."""
        return SteeringVector(
            vector=self.vector * factor,
            layer=self.layer,
            component=self.component,
            metadata={**self.metadata, "scale_factor": factor},
        )

    def to(self, device: Union[str, torch.device]) -> "SteeringVector":
        """Move steering vector to a device."""
        return SteeringVector(
            vector=self.vector.to(device),
            layer=self.layer,
            component=self.component,
            metadata=self.metadata,
        )

    @classmethod
    def from_prompts(
        cls,
        model: Any,  # ACEModel
        positive_prompt: str,
        negative_prompt: str,
        layer: int,
        position: int = -1,
        component: str = "resid_post",
    ) -> "SteeringVector":
        """
        Create a steering vector from positive and negative prompts.

        Args:
            model: ACEModel instance
            positive_prompt: Prompt for desired behavior
            negative_prompt: Prompt for undesired behavior
            layer: Layer to extract activations from
            position: Token position (-1 for last)
            component: Component to use

        Returns:
            SteeringVector instance
        """
        logger.info(f"Computing steering vector from prompts at layer {layer}")

        # Tokenize
        pos_tokens = model.tokenize(positive_prompt)["input_ids"]
        neg_tokens = model.tokenize(negative_prompt)["input_ids"]

        # Get activations
        with torch.no_grad():
            pos_resid = model.get_residual_stream(
                pos_tokens,
                layer=layer,
                position=component.replace("resid_", ""),
            )
            neg_resid = model.get_residual_stream(
                neg_tokens,
                layer=layer,
                position=component.replace("resid_", ""),
            )

        # Compute difference at specified position
        if position == -1:
            pos_act = pos_resid[0, -1, :]
            neg_act = neg_resid[0, -1, :]
        else:
            pos_act = pos_resid[0, position, :]
            neg_act = neg_resid[0, position, :]

        vector = pos_act - neg_act

        return cls(
            vector=vector,
            layer=layer,
            component=component,
            metadata={
                "positive_prompt": positive_prompt[:100],
                "negative_prompt": negative_prompt[:100],
                "position": position,
            },
        )

    @classmethod
    def from_caches(
        cls,
        positive_caches: List[ActivationCache],
        negative_caches: List[ActivationCache],
        layer: int,
        position: int = -1,
        component: str = "resid_post",
    ) -> "SteeringVector":
        """
        Create a steering vector from multiple activation caches.

        This averages across multiple examples for a more robust vector.

        Args:
            positive_caches: List of caches from positive examples
            negative_caches: List of caches from negative examples
            layer: Layer to use
            position: Token position (-1 for last)
            component: Component name

        Returns:
            SteeringVector instance
        """
        hook_name = f"blocks.{layer}.hook_{component}"

        # Collect positive activations
        pos_vectors = []
        for cache in positive_caches:
            act = cache[hook_name]
            if position == -1:
                pos_vectors.append(act[0, -1, :])
            else:
                pos_vectors.append(act[0, position, :])

        # Collect negative activations
        neg_vectors = []
        for cache in negative_caches:
            act = cache[hook_name]
            if position == -1:
                neg_vectors.append(act[0, -1, :])
            else:
                neg_vectors.append(act[0, position, :])

        # Average
        pos_mean = torch.stack(pos_vectors).mean(dim=0)
        neg_mean = torch.stack(neg_vectors).mean(dim=0)

        vector = pos_mean - neg_mean

        return cls(
            vector=vector,
            layer=layer,
            component=component,
            metadata={
                "n_positive": len(positive_caches),
                "n_negative": len(negative_caches),
                "position": position,
            },
        )

    @classmethod
    def from_error_direction(
        cls,
        error_direction: torch.Tensor,
        layer: int,
        component: str = "resid_post",
        negate: bool = True,
    ) -> "SteeringVector":
        """
        Create a steering vector from an SAE error direction.

        The error direction points towards "incorrectness", so we
        typically negate it to steer away from errors.

        Args:
            error_direction: Direction vector from SAE analysis
            layer: Layer to apply
            component: Component name
            negate: Whether to negate (steer away from error)

        Returns:
            SteeringVector instance
        """
        vector = -error_direction if negate else error_direction

        return cls(
            vector=vector,
            layer=layer,
            component=component,
            metadata={
                "source": "sae_error_direction",
                "negated": negate,
            },
        )

    def get_hook_fn(
        self,
        strength: float = 1.0,
        positions: Optional[List[int]] = None,
    ) -> Callable:
        """
        Get a hook function that applies this steering vector.

        Args:
            strength: Scaling factor (alpha)
            positions: Token positions to apply steering (None for all)

        Returns:
            Hook function compatible with TransformerLens
        """
        vector = self.vector * strength

        def steering_hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            # activation shape: (batch, seq_len, d_model)
            if positions is None:
                # Apply to all positions
                return activation + vector.unsqueeze(0).unsqueeze(0)
            else:
                # Apply only to specified positions
                result = activation.clone()
                for pos in positions:
                    if pos == -1:
                        result[:, -1, :] += vector
                    else:
                        result[:, pos, :] += vector
                return result

        return steering_hook

    def apply(
        self,
        model: Any,  # ACEModel
        input_ids: torch.Tensor,
        strength: float = 1.0,
        positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Apply steering vector during model forward pass.

        Args:
            model: ACEModel instance
            input_ids: Input token IDs
            strength: Steering strength (alpha)
            positions: Token positions to steer (None for all)

        Returns:
            Model logits with steering applied
        """
        hook_name = f"blocks.{self.layer}.hook_{self.component}"
        hook_fn = self.get_hook_fn(strength=strength, positions=positions)

        return model.run_with_hooks(
            input_ids,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    def save(self, path: str) -> None:
        """Save steering vector to file."""
        torch.save({
            "vector": self.vector,
            "layer": self.layer,
            "component": self.component,
            "metadata": self.metadata,
        }, path)
        logger.info(f"Saved steering vector to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SteeringVector":
        """Load steering vector from file."""
        data = torch.load(path, map_location=device)
        return cls(
            vector=data["vector"],
            layer=data["layer"],
            component=data["component"],
            metadata=data["metadata"],
        )


class MultiLayerSteering:
    """
    Apply steering vectors across multiple layers.

    This allows for more nuanced control by steering at multiple
    points in the model's computation.
    """

    def __init__(self, steering_vectors: List[SteeringVector]):
        """
        Initialize multi-layer steering.

        Args:
            steering_vectors: List of SteeringVector objects for different layers
        """
        self.steering_vectors = steering_vectors

        # Verify no duplicate layers
        layers = [sv.layer for sv in steering_vectors]
        if len(layers) != len(set(layers)):
            logger.warning("Multiple steering vectors for same layer")

    def apply(
        self,
        model: Any,  # ACEModel
        input_ids: torch.Tensor,
        strengths: Optional[Dict[int, float]] = None,
        positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Apply all steering vectors during forward pass.

        Args:
            model: ACEModel instance
            input_ids: Input token IDs
            strengths: Per-layer strength overrides {layer: alpha}
            positions: Token positions to steer

        Returns:
            Model logits with all steering applied
        """
        if strengths is None:
            strengths = {}

        # Build hooks
        hooks = []
        for sv in self.steering_vectors:
            strength = strengths.get(sv.layer, 1.0)
            hook_name = f"blocks.{sv.layer}.hook_{sv.component}"
            hook_fn = sv.get_hook_fn(strength=strength, positions=positions)
            hooks.append((hook_name, hook_fn))

        return model.run_with_hooks(input_ids, fwd_hooks=hooks)


def find_optimal_strength(
    model: Any,
    steering_vector: SteeringVector,
    test_inputs: List[torch.Tensor],
    expected_outputs: List[torch.Tensor],
    strength_range: Tuple[float, float] = (0.1, 3.0),
    n_steps: int = 10,
) -> Tuple[float, float]:
    """
    Find optimal steering strength via grid search.

    Args:
        model: ACEModel instance
        steering_vector: SteeringVector to optimize
        test_inputs: List of test input_ids
        expected_outputs: List of expected output token ids
        strength_range: Range of strengths to try
        n_steps: Number of steps in grid search

    Returns:
        Tuple of (optimal_strength, accuracy)
    """
    strengths = torch.linspace(strength_range[0], strength_range[1], n_steps)
    best_strength = 1.0
    best_accuracy = 0.0

    for strength in strengths:
        correct = 0
        for inp, expected in zip(test_inputs, expected_outputs):
            logits = steering_vector.apply(model, inp, strength=strength.item())
            pred = logits[0, -1, :].argmax()
            if pred == expected[0, -1]:
                correct += 1

        accuracy = correct / len(test_inputs)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_strength = strength.item()

        logger.debug(f"Strength {strength.item():.2f}: accuracy {accuracy:.2%}")

    logger.info(f"Optimal strength: {best_strength:.2f} (accuracy: {best_accuracy:.2%})")
    return best_strength, best_accuracy
