"""
Sidecar Model Architecture for SAE-Based Anomaly Detection

Instead of probing the large main model (e.g., Llama-70B) directly for every token,
we deploy a smaller "Sidecar Model" to run in parallel for efficient SAE analysis.

Based on:
- Nguyen et al. (2025): Deploying Interpretability to Production with Rakuten

Recommended configurations:
- Llama-3.1-8B for production: Layer 12 extraction
- Gemma-2-2B for lightweight: Layer 19 extraction
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from ace_code.core.model import ACEModel
from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.utils.device import get_device
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SidecarConfig:
    """Configuration for the Sidecar model."""

    # Model selection
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Layer configuration (where "incorrectness" features peak)
    target_layer: int = 12  # Layer 12 for Llama-8B, Layer 19 for Gemma-2-2B

    # Device and precision
    device: str = "auto"
    dtype: str = "float16"
    quantization: Optional[str] = "8bit"  # 8bit recommended for efficiency

    # Input processing
    chunk_size: int = 128  # Process inputs in chunks
    overlap: int = 32  # Overlap between chunks for context

    # SAE configuration
    sae_expansion_factor: int = 8  # 8x to 16x expansion
    sae_path: Optional[str] = None  # Path to pretrained SAE

    def __post_init__(self):
        """Set target layer based on model if not explicitly set."""
        layer_defaults = {
            "meta-llama/Llama-3.1-8B": 12,
            "meta-llama/Llama-3.1-8B-Instruct": 12,
            "google/gemma-2-2b": 19,
            "google/gemma-2-2b-it": 19,
        }
        if self.model_name in layer_defaults:
            self.target_layer = layer_defaults[self.model_name]


class ChunkedProcessor:
    """
    Processes long inputs in overlapping chunks.

    This maintains context while allowing efficient processing of
    long sequences without memory issues.
    """

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 128,
        overlap: int = 32,
    ):
        """
        Initialize chunked processor.

        Args:
            tokenizer: Tokenizer instance
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between consecutive chunks
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Split input into overlapping chunks.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            List of chunk dictionaries with 'input_ids', 'attention_mask', 'start_idx'
        """
        batch_size, seq_len = input_ids.shape
        chunks = []

        start_idx = 0
        while start_idx < seq_len:
            end_idx = min(start_idx + self.chunk_size, seq_len)

            chunk_ids = input_ids[:, start_idx:end_idx]
            chunk_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None

            chunks.append({
                "input_ids": chunk_ids,
                "attention_mask": chunk_mask,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

            if end_idx >= seq_len:
                break
            start_idx += self.stride

        return chunks

    def merge_activations(
        self,
        chunk_activations: List[Tuple[torch.Tensor, int, int]],
        seq_len: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """
        Merge activations from overlapping chunks.

        Uses simple averaging in overlapping regions.

        Args:
            chunk_activations: List of (activation, start_idx, end_idx)
            seq_len: Original sequence length
            hidden_dim: Hidden dimension

        Returns:
            Merged activation tensor
        """
        # Initialize output and count tensors
        merged = torch.zeros(1, seq_len, hidden_dim, device=chunk_activations[0][0].device)
        counts = torch.zeros(1, seq_len, 1, device=chunk_activations[0][0].device)

        for activation, start_idx, end_idx in chunk_activations:
            merged[:, start_idx:end_idx, :] += activation
            counts[:, start_idx:end_idx, :] += 1

        # Average overlapping regions
        merged = merged / counts.clamp(min=1)
        return merged


class SidecarModel:
    """
    Sidecar model for efficient SAE-based anomaly detection.

    Runs a smaller model (e.g., Llama-3.1-8B) in parallel with the main
    production model to extract residual stream activations for SAE analysis.

    Key features:
    - Efficient 8-bit quantization
    - Chunked processing for long inputs
    - Targeted layer extraction
    - Integrated SAE encoding
    """

    def __init__(self, config: Optional[SidecarConfig] = None):
        """
        Initialize Sidecar model.

        Args:
            config: Sidecar configuration (defaults to Llama-3.1-8B)
        """
        self.config = config or SidecarConfig()
        self._model: Optional[ACEModel] = None
        self._sae: Optional[JumpReLUSAE] = None
        self._processor: Optional[ChunkedProcessor] = None
        self._is_loaded = False

        logger.info(
            f"Initializing SidecarModel with {self.config.model_name}, "
            f"target_layer={self.config.target_layer}"
        )

    def load(self) -> "SidecarModel":
        """Load the sidecar model and SAE."""
        logger.info("Loading sidecar model...")

        # Load the base model
        self._model = ACEModel(
            model_name=self.config.model_name,
            device=self.config.device,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
        )
        self._model.load()

        # Initialize chunked processor
        self._processor = ChunkedProcessor(
            tokenizer=self._model.tokenizer,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )

        # Load or initialize SAE
        if self.config.sae_path:
            self._sae = JumpReLUSAE.load(
                self.config.sae_path,
                device=self.config.device,
            )
        else:
            # Initialize SAE with model dimensions
            self._sae = JumpReLUSAE(
                d_model=self._model.model_dim,
                expansion_factor=self.config.sae_expansion_factor,
                device=self.config.device,
            )
            logger.warning(
                "No pretrained SAE path provided. "
                "SAE initialized with random weights. "
                "Feature discovery will not work correctly."
            )

        self._is_loaded = True
        logger.info("Sidecar model loaded successfully")
        return self

    @property
    def model(self) -> ACEModel:
        """Get the underlying ACE model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def sae(self) -> JumpReLUSAE:
        """Get the SAE encoder."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._sae

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self._model.tokenizer

    def set_sae(self, sae: JumpReLUSAE) -> None:
        """
        Set a custom SAE for the sidecar.

        Args:
            sae: JumpReLUSAE instance
        """
        self._sae = sae

    @torch.no_grad()
    def extract_activation(
        self,
        text: Union[str, List[str]],
        position: str = "final",
    ) -> torch.Tensor:
        """
        Extract residual stream activation from input text.

        Args:
            text: Input text or list of texts
            position: Which position to extract ('final', 'all', or int for specific)

        Returns:
            Activation tensor:
                - If position='final': shape (batch, hidden_dim)
                - If position='all': shape (batch, seq_len, hidden_dim)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize
        inputs = self._model.tokenize(text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get activation at target layer
        if position == "final":
            return self._model.get_final_token_activation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=self.config.target_layer,
            )
        elif position == "all":
            return self._model.get_residual_stream(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=self.config.target_layer,
            )
        elif isinstance(position, int):
            residual = self._model.get_residual_stream(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=self.config.target_layer,
            )
            return residual[:, position, :]
        else:
            raise ValueError(f"Unknown position: {position}")

    @torch.no_grad()
    def extract_activation_chunked(
        self,
        text: str,
        position: str = "final",
    ) -> torch.Tensor:
        """
        Extract activation from long text using chunked processing.

        Args:
            text: Input text
            position: Which position to extract

        Returns:
            Activation tensor
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize
        inputs = self._model.tokenize(text, max_length=8192)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        seq_len = input_ids.size(1)

        # If short enough, process directly
        if seq_len <= self.config.chunk_size:
            return self.extract_activation(text, position)

        # Process in chunks
        chunks = self._processor.chunk_input(input_ids, attention_mask)
        chunk_activations = []

        for chunk in chunks:
            activation = self._model.get_residual_stream(
                input_ids=chunk["input_ids"],
                attention_mask=chunk["attention_mask"],
                layer_idx=self.config.target_layer,
            )
            chunk_activations.append((
                activation,
                chunk["start_idx"],
                chunk["end_idx"],
            ))

        # Merge activations
        merged = self._processor.merge_activations(
            chunk_activations,
            seq_len,
            self._model.model_dim,
        )

        if position == "final":
            # Get last non-padding position
            seq_length = attention_mask.sum(dim=1) - 1
            return merged[0, seq_length[0].item(), :]
        elif position == "all":
            return merged
        else:
            return merged[:, position, :]

    @torch.no_grad()
    def get_sae_features(
        self,
        text: Union[str, List[str]],
        position: str = "final",
    ) -> torch.Tensor:
        """
        Get SAE feature activations for input text.

        This is the primary method for anomaly detection - it extracts
        the sparse feature representation that can be analyzed for
        "incorrectness" features.

        Args:
            text: Input text or list of texts
            position: Position to extract ('final' recommended for prompts)

        Returns:
            Sparse feature activations of shape (batch, n_latents)
        """
        # Get residual stream activation
        activation = self.extract_activation(text, position)

        # Encode through SAE
        features = self._sae.get_feature_activations(activation)

        return features

    @torch.no_grad()
    def get_specific_features(
        self,
        text: Union[str, List[str]],
        feature_indices: torch.Tensor,
        position: str = "final",
    ) -> torch.Tensor:
        """
        Get activations of specific SAE features.

        Used at runtime to check only the pre-identified "incorrectness" features.

        Args:
            text: Input text
            feature_indices: Indices of features to check
            position: Position to extract

        Returns:
            Feature activations for specified indices, shape (batch, len(feature_indices))
        """
        all_features = self.get_sae_features(text, position)
        return all_features[:, feature_indices]

    def probe_for_anomaly(
        self,
        text: str,
        incorrectness_features: torch.Tensor,
        feature_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Probe input for potential code anomalies.

        This is the main runtime interface for anomaly detection.

        Args:
            text: Input prompt/code
            incorrectness_features: Indices of incorrectness-correlated features
            feature_weights: Optional weights for each feature
            threshold: Anomaly score threshold for flagging

        Returns:
            Dict with 'is_anomaly', 'score', 'active_features', 'details'
        """
        # Get feature activations
        features = self.get_sae_features(text, position="final")

        # Extract incorrectness features
        target_features = features[0, incorrectness_features]

        if feature_weights is None:
            feature_weights = torch.ones_like(target_features)

        # Compute weighted anomaly score
        anomaly_score = (target_features * feature_weights).sum().item()

        # Get active features
        active_mask = target_features > 0
        active_indices = incorrectness_features[active_mask]
        active_values = target_features[active_mask]

        return {
            "is_anomaly": anomaly_score > threshold,
            "score": anomaly_score,
            "threshold": threshold,
            "n_active_features": active_mask.sum().item(),
            "active_features": {
                "indices": active_indices.tolist(),
                "values": active_values.tolist(),
            },
            "details": {
                "model": self.config.model_name,
                "layer": self.config.target_layer,
                "n_incorrectness_features": len(incorrectness_features),
            },
        }
