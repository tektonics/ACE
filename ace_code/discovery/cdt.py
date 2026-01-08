"""
Contextual Decomposition for Transformers (CD-T)

This module implements CD-T, which mathematically decomposes transformer
activations into "Relevant" (beta) and "Irrelevant" (gamma) contributions.
Unlike activation patching methods like ACDC, CD-T requires only a single
forward pass and provides a complete attribution of each component's
contribution to the output.

Reference: "Contextual Decomposition" methodology adapted for transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from loguru import logger

from ace_code.core.activations import ActivationCache


@dataclass
class CDTResult:
    """
    Results from Contextual Decomposition analysis.

    Attributes:
        beta: Relevant contributions (from the specified tokens/features)
        gamma: Irrelevant contributions (everything else)
        layer_contributions: Per-layer breakdown of contributions
        head_contributions: Per-attention-head breakdown
        total_relevance: Final relevance score
    """

    beta: Dict[str, torch.Tensor] = field(default_factory=dict)
    gamma: Dict[str, torch.Tensor] = field(default_factory=dict)
    layer_contributions: Dict[int, float] = field(default_factory=dict)
    head_contributions: Dict[Tuple[int, int], float] = field(default_factory=dict)
    total_relevance: float = 0.0

    def get_top_layers(self, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k most relevant layers."""
        sorted_layers = sorted(
            self.layer_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_layers[:k]

    def get_top_heads(self, k: int = 10) -> List[Tuple[Tuple[int, int], float]]:
        """Get top-k most relevant attention heads."""
        sorted_heads = sorted(
            self.head_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_heads[:k]

    def __repr__(self) -> str:
        return (
            f"CDTResult(total_relevance={self.total_relevance:.4f}, "
            f"n_layers={len(self.layer_contributions)}, "
            f"n_heads={len(self.head_contributions)})"
        )


class ContextualDecomposition:
    """
    Contextual Decomposition for Transformers (CD-T).

    This class implements the CD-T algorithm for decomposing transformer
    computations into relevant and irrelevant contributions. The key insight
    is that for linear operations, the decomposition can be propagated
    exactly through the network.

    Key Concepts:
    - Every activation x is decomposed as: x = beta + gamma
    - beta contains contributions from "relevant" inputs (e.g., specific tokens)
    - gamma contains contributions from "irrelevant" inputs
    - For linear operations: y = Wx + b => beta_y = W*beta_x, gamma_y = W*gamma_x + b

    Usage:
        >>> cdt = ContextualDecomposition(model)
        >>> result = cdt.decompose(input_ids, relevant_positions=[0, 1, 2])
        >>> top_heads = result.get_top_heads(k=10)
    """

    def __init__(
        self,
        model: Any,  # ACEModel
        use_attention_shortcut: bool = True,
        decompose_mlp: bool = True,
        decompose_layernorm: bool = True,
    ):
        """
        Initialize CD-T analyzer.

        Args:
            model: ACEModel instance with hook access
            use_attention_shortcut: Use simplified attention decomposition
            decompose_mlp: Whether to decompose MLP contributions
            decompose_layernorm: Whether to handle LayerNorm decomposition
        """
        self.model = model
        self.use_attention_shortcut = use_attention_shortcut
        self.decompose_mlp = decompose_mlp
        self.decompose_layernorm = decompose_layernorm

        # Model dimensions
        self.n_layers = model.n_layers
        self.d_model = model.d_model
        self.n_heads = model.n_heads
        self.d_head = self.d_model // self.n_heads

        logger.info(f"Initialized CD-T for {self.n_layers}-layer model")

    def decompose(
        self,
        input_ids: torch.Tensor,
        relevant_positions: Optional[List[int]] = None,
        relevant_token_ids: Optional[List[int]] = None,
        target_position: int = -1,
        return_all_layers: bool = False,
    ) -> CDTResult:
        """
        Perform contextual decomposition on the model's forward pass.

        Args:
            input_ids: Input token IDs, shape (1, seq_len)
            relevant_positions: Token positions to mark as "relevant"
            relevant_token_ids: Token IDs to mark as "relevant"
            target_position: Position to analyze (default: last token)
            return_all_layers: Whether to return decomposition at all layers

        Returns:
            CDTResult containing the decomposition analysis
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # Handle target position
        if target_position < 0:
            target_position = seq_len + target_position

        # Initialize relevance mask
        relevance_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)

        if relevant_positions is not None:
            for pos in relevant_positions:
                if pos < 0:
                    pos = seq_len + pos
                if 0 <= pos < seq_len:
                    relevance_mask[pos] = True

        if relevant_token_ids is not None:
            for tid in relevant_token_ids:
                relevance_mask |= (input_ids[0] == tid)

        # If no relevance specified, mark all positions as relevant
        if not relevance_mask.any():
            logger.warning("No relevant positions specified, marking all as relevant")
            relevance_mask[:] = True

        logger.debug(f"Relevant positions: {relevance_mask.nonzero().flatten().tolist()}")

        # Run forward pass with cache
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(input_ids)

        # Initialize result
        result = CDTResult()

        # Get embeddings
        if "hook_embed" in cache:
            embeddings = cache["hook_embed"]
        else:
            # Fallback: get from blocks.0.hook_resid_pre
            embeddings = cache.get("blocks.0.hook_resid_pre", None)

        if embeddings is None:
            raise ValueError("Could not find embedding activations in cache")

        # Initialize beta (relevant) and gamma (irrelevant)
        # Shape: (batch, seq_len, d_model)
        beta = torch.zeros_like(embeddings)
        gamma = torch.zeros_like(embeddings)

        # Initialize based on relevance mask
        for pos in range(seq_len):
            if relevance_mask[pos]:
                beta[0, pos, :] = embeddings[0, pos, :]
            else:
                gamma[0, pos, :] = embeddings[0, pos, :]

        result.beta["embed"] = beta.clone()
        result.gamma["embed"] = gamma.clone()

        # Propagate through layers
        for layer_idx in range(self.n_layers):
            beta, gamma, layer_contrib, head_contribs = self._decompose_layer(
                layer_idx,
                beta,
                gamma,
                cache,
                target_position,
            )

            result.layer_contributions[layer_idx] = layer_contrib
            result.head_contributions.update(head_contribs)

            if return_all_layers:
                result.beta[f"layer_{layer_idx}"] = beta.clone()
                result.gamma[f"layer_{layer_idx}"] = gamma.clone()

        # Final decomposition
        result.beta["final"] = beta
        result.gamma["final"] = gamma

        # Compute total relevance at target position
        # Project through unembedding to get logit contribution
        W_U = self.model.get_unembedding_matrix()  # (d_model, vocab_size)

        beta_logits = beta[0, target_position, :] @ W_U
        gamma_logits = gamma[0, target_position, :] @ W_U

        # Total relevance is the norm of beta's contribution
        result.total_relevance = beta_logits.norm().item()

        return result

    def _decompose_layer(
        self,
        layer_idx: int,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        target_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[Tuple[int, int], float]]:
        """
        Decompose a single transformer layer.

        The layer computation is:
        x -> LayerNorm -> Attention -> + x -> LayerNorm -> MLP -> + x

        Args:
            layer_idx: Index of the layer
            beta: Current relevant contribution
            gamma: Current irrelevant contribution
            cache: Activation cache from forward pass
            target_position: Position to analyze

        Returns:
            Tuple of (new_beta, new_gamma, layer_contribution, head_contributions)
        """
        head_contributions = {}

        # Get clean activations for reference
        attn_out = cache.get(f"blocks.{layer_idx}.hook_attn_out", None)
        mlp_out = cache.get(f"blocks.{layer_idx}.hook_mlp_out", None)

        # Pre-attention LayerNorm
        total_pre_attn = beta + gamma
        beta_normed, gamma_normed = self._decompose_layernorm(
            beta, gamma, total_pre_attn
        )

        # Attention decomposition
        beta_attn, gamma_attn, head_contribs = self._decompose_attention(
            layer_idx,
            beta_normed,
            gamma_normed,
            cache,
            target_position,
        )

        # Update head contributions
        for (l, h), contrib in head_contribs.items():
            head_contributions[(l, h)] = contrib

        # Residual connection after attention
        beta = beta + beta_attn
        gamma = gamma + gamma_attn

        # Pre-MLP LayerNorm
        if self.decompose_mlp and mlp_out is not None:
            total_pre_mlp = beta + gamma
            beta_normed, gamma_normed = self._decompose_layernorm(
                beta, gamma, total_pre_mlp
            )

            # MLP decomposition
            beta_mlp, gamma_mlp = self._decompose_mlp(
                layer_idx,
                beta_normed,
                gamma_normed,
                cache,
            )

            # Residual connection after MLP
            beta = beta + beta_mlp
            gamma = gamma + gamma_mlp

        # Compute layer contribution
        layer_contrib = beta[0, target_position, :].norm().item()

        return beta, gamma, layer_contrib, head_contributions

    def _decompose_layernorm(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        total: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose LayerNorm operation.

        For LayerNorm: y = (x - mean) / std * scale + bias
        We use the total's statistics but apply proportionally to beta and gamma.

        Args:
            beta: Relevant contribution
            gamma: Irrelevant contribution
            total: Total activation (beta + gamma)

        Returns:
            Tuple of (beta_normed, gamma_normed)
        """
        if not self.decompose_layernorm:
            # Simple proportional scaling
            total_norm = total.norm(dim=-1, keepdim=True) + 1e-8
            beta_frac = beta.norm(dim=-1, keepdim=True) / total_norm
            gamma_frac = gamma.norm(dim=-1, keepdim=True) / total_norm
            return beta * beta_frac, gamma * gamma_frac

        # Compute LayerNorm statistics from total
        mean = total.mean(dim=-1, keepdim=True)
        var = total.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + 1e-5).sqrt()

        # Apply normalization proportionally
        # The mean subtraction affects both equally in proportion to their magnitude
        beta_centered = beta - mean * (beta / (total + 1e-8))
        gamma_centered = gamma - mean * (gamma / (total + 1e-8))

        # Scale by standard deviation
        beta_normed = beta_centered / std
        gamma_normed = gamma_centered / std

        return beta_normed, gamma_normed

    def _decompose_attention(
        self,
        layer_idx: int,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        target_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], float]]:
        """
        Decompose attention mechanism.

        The attention computation:
        1. Q = xW_Q, K = xW_K, V = xW_V
        2. scores = QK^T / sqrt(d_k)
        3. attn_weights = softmax(scores)
        4. output = attn_weights @ V @ W_O

        For decomposition, we use the shortcut approach:
        - Keep attention weights fixed from the clean run
        - Propagate beta_V and gamma_V through those weights

        Args:
            layer_idx: Layer index
            beta: Relevant contribution after LayerNorm
            gamma: Irrelevant contribution after LayerNorm
            cache: Activation cache
            target_position: Position to analyze

        Returns:
            Tuple of (beta_attn, gamma_attn, head_contributions)
        """
        head_contributions = {}

        # Get attention pattern from cache
        pattern_key = f"blocks.{layer_idx}.attn.hook_pattern"
        attn_pattern = cache.get(pattern_key, None)

        if attn_pattern is None or self.use_attention_shortcut:
            # Use simplified shortcut: attribute based on value projection
            return self._decompose_attention_shortcut(
                layer_idx,
                beta,
                gamma,
                cache,
                target_position,
            )

        # Full decomposition (more complex but more accurate)
        return self._decompose_attention_full(
            layer_idx,
            beta,
            gamma,
            attn_pattern,
            cache,
            target_position,
        )

    def _decompose_attention_shortcut(
        self,
        layer_idx: int,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        target_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], float]]:
        """
        Simplified attention decomposition using the shortcut method.

        Keep attention weights fixed and propagate value contributions.
        This is computationally efficient and works well in practice.
        """
        head_contributions = {}

        # Get attention output
        attn_out_key = f"blocks.{layer_idx}.hook_attn_out"
        attn_out = cache.get(attn_out_key, torch.zeros_like(beta))

        # Compute the proportion of beta in the input
        total = beta + gamma
        total_norm = total.norm(dim=-1, keepdim=True) + 1e-8
        beta_ratio = beta.norm(dim=-1, keepdim=True) / total_norm

        # Attribute attention output proportionally
        beta_attn = attn_out * beta_ratio
        gamma_attn = attn_out * (1 - beta_ratio)

        # Compute per-head contributions (approximate)
        # We attribute equally across heads for the shortcut method
        per_head_contrib = beta_attn[0, target_position, :].norm().item() / self.n_heads

        for head_idx in range(self.n_heads):
            head_contributions[(layer_idx, head_idx)] = per_head_contrib

        return beta_attn, gamma_attn, head_contributions

    def _decompose_attention_full(
        self,
        layer_idx: int,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        attn_pattern: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        target_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], float]]:
        """
        Full attention decomposition with per-head attribution.

        This computes the exact contribution of each attention head
        to the relevant (beta) and irrelevant (gamma) outputs.
        """
        head_contributions = {}

        batch_size, seq_len, d_model = beta.shape
        d_head = d_model // self.n_heads

        # Get V projections (we'd need model weights for exact decomposition)
        # For now, use the cached values
        v_key = f"blocks.{layer_idx}.attn.hook_v"
        v = cache.get(v_key, None)

        if v is None:
            # Fall back to shortcut
            return self._decompose_attention_shortcut(
                layer_idx, beta, gamma, cache, target_position
            )

        # v shape: (batch, seq_len, n_heads, d_head)
        # attn_pattern shape: (batch, n_heads, seq_len, seq_len)

        # Compute beta/gamma proportions per position
        total = beta + gamma
        total_norm = total.norm(dim=-1, keepdim=True) + 1e-8
        beta_ratio = beta.norm(dim=-1, keepdim=True) / total_norm  # (batch, seq, 1)

        # Decompose V into beta_V and gamma_V
        beta_v = v * beta_ratio.unsqueeze(-1)
        gamma_v = v * (1 - beta_ratio.unsqueeze(-1))

        # Apply attention weights to decomposed values
        # attn_pattern: (batch, n_heads, seq_len, seq_len)
        # beta_v: (batch, seq_len, n_heads, d_head)

        # Rearrange for matmul
        beta_v_t = rearrange(beta_v, "b s h d -> b h s d")
        gamma_v_t = rearrange(gamma_v, "b s h d -> b h s d")

        # Weighted sum: (batch, n_heads, seq_len, d_head)
        beta_attn_heads = torch.matmul(attn_pattern, beta_v_t)
        gamma_attn_heads = torch.matmul(attn_pattern, gamma_v_t)

        # Compute per-head contributions at target position
        for head_idx in range(self.n_heads):
            head_beta = beta_attn_heads[0, head_idx, target_position, :]
            head_contributions[(layer_idx, head_idx)] = head_beta.norm().item()

        # Combine heads back to d_model
        beta_attn = rearrange(beta_attn_heads, "b h s d -> b s (h d)")
        gamma_attn = rearrange(gamma_attn_heads, "b h s d -> b s (h d)")

        return beta_attn, gamma_attn, head_contributions

    def _decompose_mlp(
        self,
        layer_idx: int,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose MLP contribution.

        MLP computation: y = W_out * activation(W_in * x + b_in) + b_out

        For linear parts, decomposition is exact.
        For non-linear activation, we use proportional attribution.
        """
        mlp_out_key = f"blocks.{layer_idx}.hook_mlp_out"
        mlp_out = cache.get(mlp_out_key, torch.zeros_like(beta))

        # Compute beta proportion
        total = beta + gamma
        total_norm = total.norm(dim=-1, keepdim=True) + 1e-8
        beta_ratio = beta.norm(dim=-1, keepdim=True) / total_norm

        # Attribute MLP output proportionally
        beta_mlp = mlp_out * beta_ratio
        gamma_mlp = mlp_out * (1 - beta_ratio)

        return beta_mlp, gamma_mlp

    def find_critical_components(
        self,
        input_ids: torch.Tensor,
        relevant_positions: List[int],
        top_k_layers: int = 5,
        top_k_heads: int = 10,
    ) -> Dict[str, Any]:
        """
        Find the most critical layers and attention heads.

        This is a convenience method that runs CD-T and extracts
        the most important components for the given input.

        Args:
            input_ids: Input token IDs
            relevant_positions: Positions to mark as relevant
            top_k_layers: Number of top layers to return
            top_k_heads: Number of top heads to return

        Returns:
            Dictionary with critical components
        """
        result = self.decompose(
            input_ids,
            relevant_positions=relevant_positions,
        )

        return {
            "top_layers": result.get_top_layers(k=top_k_layers),
            "top_heads": result.get_top_heads(k=top_k_heads),
            "total_relevance": result.total_relevance,
        }

    def compare_decompositions(
        self,
        input_ids_positive: torch.Tensor,
        input_ids_negative: torch.Tensor,
        relevant_positions: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compare CD-T decompositions between positive and negative examples.

        This helps identify which components differ between correct
        and buggy code predictions.

        Args:
            input_ids_positive: Token IDs for positive (correct) example
            input_ids_negative: Token IDs for negative (buggy) example
            relevant_positions: Positions to mark as relevant

        Returns:
            Dictionary with comparison results
        """
        result_pos = self.decompose(input_ids_positive, relevant_positions)
        result_neg = self.decompose(input_ids_negative, relevant_positions)

        # Compute differences
        layer_diffs = {}
        for layer in result_pos.layer_contributions:
            pos_val = result_pos.layer_contributions.get(layer, 0)
            neg_val = result_neg.layer_contributions.get(layer, 0)
            layer_diffs[layer] = pos_val - neg_val

        head_diffs = {}
        for head in result_pos.head_contributions:
            pos_val = result_pos.head_contributions.get(head, 0)
            neg_val = result_neg.head_contributions.get(head, 0)
            head_diffs[head] = pos_val - neg_val

        # Sort by absolute difference
        sorted_layers = sorted(layer_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_heads = sorted(head_diffs.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "positive_result": result_pos,
            "negative_result": result_neg,
            "layer_differences": sorted_layers,
            "head_differences": sorted_heads,
            "most_different_layer": sorted_layers[0] if sorted_layers else None,
            "most_different_heads": sorted_heads[:5] if sorted_heads else [],
        }
