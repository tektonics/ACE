"""
ACE-Code Model Loading and Hooking Module

This module provides the core functionality for loading transformer models
with hooks that enable access to the residual stream for mechanistic
interpretability analysis.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

try:
    from transformer_lens import HookedTransformer
    from transformer_lens.hook_points import HookPoint
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None
    HookPoint = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name: str = "gemma-2-2b"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_transformer_lens: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None

    # Hook configuration
    hook_layers: Optional[List[int]] = None
    hook_components: List[str] = field(
        default_factory=lambda: ["resid_pre", "resid_post", "attn_out", "mlp_out"]
    )


class ACEModel:
    """
    Wrapper class for transformer models with hook capabilities.

    This class provides a unified interface for loading models with
    TransformerLens (preferred) or HuggingFace Transformers, enabling
    access to internal activations for circuit analysis.
    """

    def __init__(
        self,
        model: Union[HookedTransformer, nn.Module],
        tokenizer: Any,
        config: ModelConfig,
    ):
        """
        Initialize ACEModel wrapper.

        Args:
            model: The underlying transformer model
            tokenizer: The tokenizer for the model
            config: Model configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._is_hooked = isinstance(model, HookedTransformer) if HookedTransformer else False
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List[Tuple[str, Callable]] = []

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        if self._is_hooked:
            return self.model.cfg.device
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        if self._is_hooked:
            return self.model.cfg.dtype
        return next(self.model.parameters()).dtype

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model."""
        if self._is_hooked:
            return self.model.cfg.n_layers
        # Fallback for HuggingFace models
        return getattr(self.model.config, "num_hidden_layers", 12)

    @property
    def d_model(self) -> int:
        """Get the model dimension."""
        if self._is_hooked:
            return self.model.cfg.d_model
        return getattr(self.model.config, "hidden_size", 768)

    @property
    def n_heads(self) -> int:
        """Get the number of attention heads."""
        if self._is_hooked:
            return self.model.cfg.n_heads
        return getattr(self.model.config, "num_attention_heads", 12)

    def tokenize(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text.

        Args:
            text: Input text or list of texts
            return_tensors: Format for output tensors
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length

        Returns:
            Dictionary of tokenized inputs
        """
        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_type: str = "logits",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_type: What to return ("logits", "loss", "both")

        Returns:
            Model outputs based on return_type
        """
        if self._is_hooked:
            return self.model(input_ids, return_type=return_type)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[Union[str, List[str], Callable]] = None,
        return_type: str = "logits",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and cache specified activations.

        This is the primary method for collecting activations during
        circuit analysis. It hooks into the residual stream and caches
        the specified activation tensors.

        Args:
            input_ids: Input token IDs
            names_filter: Filter for which activations to cache
            return_type: What to return from the model

        Returns:
            Tuple of (model_output, activation_cache)
        """
        if self._is_hooked:
            logits, cache = self.model.run_with_cache(
                input_ids,
                names_filter=names_filter,
                return_type=return_type,
            )
            return logits, dict(cache)
        else:
            # Fallback: manual hooking for HuggingFace models
            return self._run_with_manual_cache(input_ids, names_filter)

    def _run_with_manual_cache(
        self,
        input_ids: torch.Tensor,
        names_filter: Optional[Union[str, List[str], Callable]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Manual activation caching for non-TransformerLens models."""
        cache = {}
        hooks = []

        def make_hook(name: str) -> Callable:
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                if isinstance(output, tuple):
                    cache[name] = output[0].detach().clone()
                else:
                    cache[name] = output.detach().clone()
            return hook_fn

        # Register hooks on transformer layers
        for i, layer in enumerate(self.model.model.layers):
            hook_name = f"blocks.{i}.hook_resid_post"
            hooks.append(layer.register_forward_hook(make_hook(hook_name)))

        try:
            outputs = self.model(input_ids)
            logits = outputs.logits
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return logits, cache

    def run_with_hooks(
        self,
        input_ids: torch.Tensor,
        fwd_hooks: List[Tuple[str, Callable]],
        return_type: str = "logits",
    ) -> torch.Tensor:
        """
        Run forward pass with custom hooks for activation patching.

        This method enables intervention experiments by allowing custom
        hook functions to modify activations during the forward pass.

        Args:
            input_ids: Input token IDs
            fwd_hooks: List of (hook_name, hook_fn) tuples
            return_type: What to return from the model

        Returns:
            Model output with hooks applied
        """
        if self._is_hooked:
            return self.model.run_with_hooks(
                input_ids,
                fwd_hooks=fwd_hooks,
                return_type=return_type,
            )
        else:
            raise NotImplementedError(
                "Custom hooks require TransformerLens. "
                "Please reload the model with use_transformer_lens=True"
            )

    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        layer: int,
        position: str = "post",
    ) -> torch.Tensor:
        """
        Get residual stream activations at a specific layer.

        Args:
            input_ids: Input token IDs
            layer: Layer index
            position: "pre" or "post" the layer

        Returns:
            Residual stream tensor of shape (batch, seq_len, d_model)
        """
        hook_name = f"blocks.{layer}.hook_resid_{position}"
        _, cache = self.run_with_cache(input_ids, names_filter=[hook_name])
        return cache[hook_name]

    def get_attention_patterns(
        self,
        input_ids: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """
        Get attention patterns for a specific layer.

        Args:
            input_ids: Input token IDs
            layer: Layer index

        Returns:
            Attention patterns of shape (batch, n_heads, seq_len, seq_len)
        """
        if self._is_hooked:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            _, cache = self.run_with_cache(input_ids, names_filter=[hook_name])
            return cache[hook_name]
        else:
            raise NotImplementedError(
                "Attention pattern extraction requires TransformerLens"
            )

    def get_unembedding_matrix(self) -> torch.Tensor:
        """
        Get the unembedding (output) matrix W_U.

        This matrix is used for vocabulary projection to interpret
        feature directions in terms of predicted tokens.

        Returns:
            Unembedding matrix of shape (d_model, vocab_size)
        """
        if self._is_hooked:
            return self.model.W_U
        else:
            # For HuggingFace models, this is typically the lm_head
            return self.model.lm_head.weight.T

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        tokens = self.tokenize(prompt, padding=False)
        input_ids = tokens["input_ids"]

        if self._is_hooked:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )
        else:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def to(self, device: Union[str, torch.device]) -> "ACEModel":
        """Move model to specified device."""
        if self._is_hooked:
            self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)
        return self

    def eval(self) -> "ACEModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self) -> "ACEModel":
        """Set model to training mode."""
        self.model.train()
        return self


def load_hooked_model(
    model_name: str = "gemma-2-2b",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_transformer_lens: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> ACEModel:
    """
    Load a transformer model with hooks for mechanistic interpretability.

    This is the primary entry point for loading models in ACE-Code.
    It preferentially uses TransformerLens for full hook access, but
    falls back to HuggingFace Transformers if needed.

    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on ("cuda", "cpu", "auto")
        dtype: Data type for model weights
        use_transformer_lens: Whether to use TransformerLens (recommended)
        load_in_8bit: Load with 8-bit quantization
        load_in_4bit: Load with 4-bit quantization
        cache_dir: Directory to cache downloaded models
        **kwargs: Additional arguments passed to the model loader

    Returns:
        ACEModel wrapper with hook capabilities

    Example:
        >>> model = load_hooked_model("gemma-2-2b", device="cuda")
        >>> logits, cache = model.run_with_cache(input_ids)
        >>> resid = cache["blocks.10.hook_resid_post"]
    """
    config = ModelConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        use_transformer_lens=use_transformer_lens,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
    )

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    if use_transformer_lens and TRANSFORMER_LENS_AVAILABLE:
        logger.info("Using TransformerLens for hook access")

        try:
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
                cache_dir=cache_dir,
                **kwargs,
            )
            tokenizer = model.tokenizer

        except Exception as e:
            logger.warning(f"TransformerLens loading failed: {e}")
            logger.info("Falling back to HuggingFace Transformers")
            use_transformer_lens = False

    if not use_transformer_lens or not TRANSFORMER_LENS_AVAILABLE:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Neither TransformerLens nor HuggingFace Transformers is available. "
                "Please install at least one: pip install transformer_lens transformers"
            )

        logger.info("Using HuggingFace Transformers (limited hook access)")

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "cache_dir": cache_dir,
        }

        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = device if device != "cuda" else "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ace_model = ACEModel(model, tokenizer, config)
    ace_model.eval()

    logger.info(f"Model loaded successfully: {config.model_name}")
    logger.info(f"  Layers: {ace_model.n_layers}")
    logger.info(f"  Model dim: {ace_model.d_model}")
    logger.info(f"  Attention heads: {ace_model.n_heads}")

    return ace_model
