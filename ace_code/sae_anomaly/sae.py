"""
JumpReLU Sparse Autoencoder Implementation

Implements the JumpReLU SAE architecture from Gemma Scope (Lieberum et al., 2024).
The SAE decomposes dense residual stream activations into sparse, interpretable features.

Math:
    Given residual stream x ∈ R^d:
    - Encoder: a(x) = JumpReLU_θ(x @ W_enc + b_enc)
    - JumpReLU: JumpReLU_θ(z) = z ⊙ H(z - θ)
    - Decoder: x̂ = a(x) @ W_dec + b_dec

Where:
    - H is the Heaviside step function
    - θ is a learnable threshold vector
    - W_enc, b_enc are encoder weights and biases
    - W_dec, b_dec are decoder weights and biases
"""

from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from ace_code.utils.device import get_device
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


class JumpReLU(nn.Module):
    """
    JumpReLU activation function with learnable thresholds.

    JumpReLU_θ(z) = z ⊙ H(z - θ)

    Where H is the Heaviside step function:
    - H(x) = 1 if x > 0
    - H(x) = 0 otherwise

    This creates a "jump" at the threshold, promoting sparsity.
    """

    def __init__(self, n_features: int, initial_threshold: float = 0.0):
        """
        Initialize JumpReLU activation.

        Args:
            n_features: Number of features (latents)
            initial_threshold: Initial threshold value
        """
        super().__init__()
        self.threshold = nn.Parameter(
            torch.full((n_features,), initial_threshold)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply JumpReLU activation.

        Args:
            z: Pre-activation tensor of shape (..., n_features)

        Returns:
            Activated tensor with same shape
        """
        # Heaviside step function: 1 where z > threshold, 0 otherwise
        mask = (z > self.threshold).float()
        # Element-wise multiplication
        return z * mask

    def get_active_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Get binary mask of active features."""
        return (z > self.threshold).float()


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder for residual stream decomposition.

    Architecture:
        - Encoder: Projects d-dimensional input to n_latents sparse features
        - Decoder: Reconstructs input from sparse features

    The SAE is trained to minimize reconstruction error while maintaining
    sparsity through the JumpReLU activation and L0/L1 penalties.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 8,
        n_latents: Optional[int] = None,
        initial_threshold: float = 0.0,
        normalize_decoder: bool = True,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize JumpReLU SAE.

        Args:
            d_model: Model hidden dimension (e.g., 4096 for Llama-8B)
            expansion_factor: Ratio of n_latents to d_model (8x-16x typical)
            n_latents: Explicit number of latents (overrides expansion_factor)
            initial_threshold: Initial JumpReLU threshold
            normalize_decoder: Whether to normalize decoder columns
            device: Device to place model on
            dtype: Data type for weights
        """
        super().__init__()

        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.n_latents = n_latents or (d_model * expansion_factor)
        self.normalize_decoder = normalize_decoder
        self._device = get_device(device)
        self._dtype = dtype

        logger.info(
            f"Initializing JumpReLUSAE: d_model={d_model}, "
            f"n_latents={self.n_latents}, expansion={self.n_latents/d_model:.1f}x"
        )

        # Encoder: d_model -> n_latents
        self.W_enc = nn.Parameter(
            torch.randn(d_model, self.n_latents, dtype=dtype) * 0.02
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.n_latents, dtype=dtype)
        )

        # JumpReLU activation
        self.activation = JumpReLU(self.n_latents, initial_threshold)

        # Decoder: n_latents -> d_model
        self.W_dec = nn.Parameter(
            torch.randn(self.n_latents, d_model, dtype=dtype) * 0.02
        )
        self.b_dec = nn.Parameter(
            torch.zeros(d_model, dtype=dtype)
        )

        # Initialize decoder columns to unit norm
        if normalize_decoder:
            self._normalize_decoder()

        self.to(self._device)

    def _normalize_decoder(self) -> None:
        """Normalize decoder weight columns to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True)
            self.W_dec.div_(norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature activations.

        a(x) = JumpReLU_θ(x @ W_enc + b_enc)

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Sparse feature activations of shape (..., n_latents)
        """
        # Linear projection
        z = x @ self.W_enc + self.b_enc
        # Apply JumpReLU
        a = self.activation(z)
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstructed input.

        x̂ = a @ W_dec + b_dec

        Args:
            a: Sparse feature activations of shape (..., n_latents)

        Returns:
            Reconstructed input of shape (..., d_model)
        """
        return a @ self.W_dec + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through SAE.

        Args:
            x: Input tensor of shape (..., d_model)
            return_features: Whether to also return sparse features

        Returns:
            If return_features=False: Reconstructed x̂
            If return_features=True: (x̂, features)
        """
        features = self.encode(x)
        x_hat = self.decode(features)

        if return_features:
            return x_hat, features
        return x_hat

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sparse feature activations for input.

        This is the key method for anomaly detection - we extract feature
        activations and check specific "incorrectness" features.

        Args:
            x: Input tensor of shape (batch, d_model) or (batch, seq, d_model)

        Returns:
            Feature activations of shape (batch, n_latents) or (batch, seq, n_latents)
        """
        return self.encode(x)

    def get_active_features(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get indices and values of active (non-zero) features.

        Args:
            x: Input tensor of shape (batch, d_model)
            top_k: If specified, return only top-k features by activation

        Returns:
            Tuple of (feature_indices, feature_values)
        """
        features = self.encode(x)

        if top_k is not None:
            # Get top-k features per sample
            values, indices = torch.topk(features, min(top_k, features.size(-1)), dim=-1)
            return indices, values

        # Get all non-zero features
        # For batched input, return list of active features per sample
        active_mask = features > 0
        return active_mask.nonzero(), features[active_mask]

    def compute_loss(
        self,
        x: torch.Tensor,
        l1_coefficient: float = 0.0,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SAE training loss.

        Loss = MSE(x, x̂) + λ * L1(features)

        Args:
            x: Input tensor
            l1_coefficient: L1 sparsity penalty weight
            return_components: Whether to return loss components separately

        Returns:
            Total loss or dict of loss components
        """
        x_hat, features = self.forward(x, return_features=True)

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_hat, x)

        # Sparsity loss (L1 on features)
        l1_loss = features.abs().mean()

        # Total loss
        total_loss = mse_loss + l1_coefficient * l1_loss

        if return_components:
            return {
                "total": total_loss,
                "mse": mse_loss,
                "l1": l1_loss,
                "sparsity": (features > 0).float().mean(),
            }
        return total_loss

    def get_sparsity_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute sparsity statistics for input batch.

        Args:
            x: Input tensor

        Returns:
            Dict with sparsity metrics
        """
        features = self.encode(x)
        active = (features > 0).float()

        return {
            "mean_active": active.sum(dim=-1).mean().item(),
            "max_active": active.sum(dim=-1).max().item(),
            "sparsity_ratio": (1 - active.mean()).item(),
            "mean_activation": features[features > 0].mean().item() if (features > 0).any() else 0.0,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save SAE weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "expansion_factor": self.expansion_factor,
            "normalize_decoder": self.normalize_decoder,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Saved SAE to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "JumpReLUSAE":
        """Load SAE from file."""
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        sae = cls(
            d_model=state["d_model"],
            n_latents=state["n_latents"],
            normalize_decoder=state["normalize_decoder"],
            device=device,
        )
        sae.load_state_dict(state["state_dict"])
        logger.info(f"Loaded SAE from {path}")
        return sae

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        layer: int,
        device: Optional[str] = None,
        **kwargs,
    ) -> "JumpReLUSAE":
        """
        Load a pretrained SAE from SAE Lens or similar registry.

        Args:
            model_name: Model the SAE was trained on (e.g., "llama-3.1-8b")
            layer: Layer the SAE was trained for
            device: Device to load onto
            **kwargs: Additional arguments for SAE loading

        Returns:
            Loaded JumpReLUSAE instance
        """
        try:
            from sae_lens import SAE as SAELensSAE

            logger.info(f"Loading pretrained SAE for {model_name} layer {layer}")

            # Load from SAE Lens
            sae_lens_model = SAELensSAE.from_pretrained(
                release=f"{model_name}-res-jb",
                sae_id=f"blocks.{layer}.hook_resid_post",
                device=device or "cpu",
            )

            # Convert to our format
            d_model = sae_lens_model.cfg.d_in
            n_latents = sae_lens_model.cfg.d_sae

            sae = cls(
                d_model=d_model,
                n_latents=n_latents,
                device=device,
            )

            # Copy weights (may need adjustment based on SAE Lens format)
            with torch.no_grad():
                sae.W_enc.copy_(sae_lens_model.W_enc)
                sae.b_enc.copy_(sae_lens_model.b_enc)
                sae.W_dec.copy_(sae_lens_model.W_dec)
                sae.b_dec.copy_(sae_lens_model.b_dec)

            return sae

        except ImportError:
            logger.warning("SAE Lens not available, cannot load pretrained SAE")
            raise
        except Exception as e:
            logger.error(f"Failed to load pretrained SAE: {e}")
            raise
