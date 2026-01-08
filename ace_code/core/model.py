"""Core model wrapper for ACE-Code with activation access."""

from typing import Optional, Dict, Any, List, Union, Callable
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ace_code.utils.device import get_device
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


class ACEModel:
    """
    Wrapper for loading and accessing language models with activation hooks.

    Supports both HuggingFace Transformers and optional TransformerLens integration.
    Provides hook-based access to residual stream activations for SAE analysis.
    """

    # Model dimension mappings for common models
    MODEL_DIMENSIONS = {
        "meta-llama/Llama-3.1-8B": 4096,
        "meta-llama/Llama-3.1-8B-Instruct": 4096,
        "google/gemma-2-2b": 2304,
        "google/gemma-2-2b-it": 2304,
    }

    # Target layer mappings (middle-to-late layers for incorrectness detection)
    TARGET_LAYERS = {
        "meta-llama/Llama-3.1-8B": 12,
        "meta-llama/Llama-3.1-8B-Instruct": 12,
        "google/gemma-2-2b": 19,
        "google/gemma-2-2b-it": 19,
    }

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        use_transformer_lens: bool = False,
    ):
        """
        Initialize ACE model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('cuda', 'cpu', 'auto')
            dtype: Data type ('float16', 'bfloat16', 'float32')
            quantization: Quantization mode ('4bit', '8bit', None)
            use_transformer_lens: Whether to use TransformerLens for activation access
        """
        self.model_name = model_name
        self.device = get_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.quantization = quantization
        self.use_transformer_lens = use_transformer_lens

        # Model components
        self._model = None
        self._tokenizer = None
        self._hooks = {}
        self._activation_cache = {}

        # Model metadata
        self.model_dim = self.MODEL_DIMENSIONS.get(model_name, 4096)
        self.target_layer = self.TARGET_LAYERS.get(model_name, 12)

        logger.info(
            f"Initializing ACEModel: {model_name}, "
            f"device={self.device}, dtype={dtype}, quantization={quantization}"
        )

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Resolve dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float16)

    def load(self) -> "ACEModel":
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = str(self.device)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model.eval()

        logger.info(f"Model loaded successfully. Target layer: {self.target_layer}")
        return self

    @property
    def model(self) -> nn.Module:
        """Get the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the transformer layer module for hooking."""
        # Handle different model architectures
        if hasattr(self._model, "model"):
            # LLaMA-style architecture
            if hasattr(self._model.model, "layers"):
                return self._model.model.layers[layer_idx]
        if hasattr(self._model, "transformer"):
            # GPT-style architecture
            if hasattr(self._model.transformer, "h"):
                return self._model.transformer.h[layer_idx]
        raise ValueError(f"Unsupported model architecture: {type(self._model)}")

    def register_activation_hook(
        self,
        layer_idx: int,
        hook_fn: Optional[Callable] = None,
        hook_name: Optional[str] = None,
    ) -> str:
        """
        Register a forward hook to capture activations at a specific layer.

        Args:
            layer_idx: Layer index to hook
            hook_fn: Custom hook function. If None, uses default caching hook.
            hook_name: Name for the hook (for removal later)

        Returns:
            Hook name/identifier
        """
        if hook_name is None:
            hook_name = f"layer_{layer_idx}_hook"

        if hook_fn is None:
            # Default hook: cache the output activation
            def hook_fn(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                self._activation_cache[hook_name] = activation.detach()

        layer = self._get_layer_module(layer_idx)
        handle = layer.register_forward_hook(hook_fn)
        self._hooks[hook_name] = handle

        logger.debug(f"Registered activation hook: {hook_name} at layer {layer_idx}")
        return hook_name

    def remove_hook(self, hook_name: str) -> None:
        """Remove a registered hook."""
        if hook_name in self._hooks:
            self._hooks[hook_name].remove()
            del self._hooks[hook_name]
            logger.debug(f"Removed hook: {hook_name}")

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for name, handle in list(self._hooks.items()):
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all hooks")

    def get_cached_activation(self, hook_name: str) -> Optional[torch.Tensor]:
        """Get cached activation from a hook."""
        return self._activation_cache.get(hook_name)

    def clear_activation_cache(self) -> None:
        """Clear all cached activations."""
        self._activation_cache.clear()

    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch)

        Returns:
            Dictionary of tokenized inputs
        """
        return self._tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Model output logits
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.logits

    @torch.no_grad()
    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get residual stream activations at a specific layer.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_idx: Layer to extract from (defaults to target_layer)

        Returns:
            Residual stream tensor of shape (batch, seq_len, hidden_dim)
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states is tuple of (embedding, layer1, layer2, ...)
        # So layer_idx=12 means outputs.hidden_states[13]
        hidden_states = outputs.hidden_states[layer_idx + 1]
        return hidden_states

    @torch.no_grad()
    def get_final_token_activation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get activation at the final token position (for prompt-based analysis).

        This is the key extraction point for SAE-based anomaly detection,
        as described in Tahimic & Cheng (2025).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_idx: Layer to extract from

        Returns:
            Activation tensor of shape (batch, hidden_dim)
        """
        residual = self.get_residual_stream(input_ids, attention_mask, layer_idx)

        # Get the last non-padding token position for each sequence
        if attention_mask is not None:
            # Sum of attention mask gives sequence length
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(residual.size(0), device=residual.device)
            final_activations = residual[batch_indices, seq_lengths]
        else:
            # No padding, just take the last token
            final_activations = residual[:, -1, :]

        return final_activations

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        inputs = self.tokenize(prompt)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        gen_kwargs.update(kwargs)

        output_ids = self._model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        # Decode only the generated part
        generated = self._tokenizer.decode(
            output_ids[0, input_ids.size(1):],
            skip_special_tokens=True,
        )
        return generated
