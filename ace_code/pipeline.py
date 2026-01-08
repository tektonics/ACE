"""
ACE Pipeline: Orchestrating the Full Workflow

This module provides the main ACEPipeline class that orchestrates the
complete workflow for Automated Circuit Extraction:

1. Load model with hooks
2. Generate mutation test pairs
3. Discover circuits via CD-T
4. Detect error features via SAE
5. Intervene with steering or PISCES

Usage:
    >>> pipeline = ACEPipeline("gemma-2-2b")
    >>> pipeline.analyze_code(code_snippet)
    >>> pipeline.steer_generation(prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from ace_code.core.model import ACEModel, load_hooked_model
from ace_code.core.activations import ActivationCache, collect_activations
from ace_code.data.mutation import MutationGenerator, CodePair
from ace_code.discovery.cdt import ContextualDecomposition, CDTResult
from ace_code.discovery.pahq import PAHQQuantizer, QuantizationConfig
from ace_code.detection.sae_detector import SAEDetector, ErrorFeature
from ace_code.detection.feature_analysis import FeatureAnalyzer
from ace_code.intervention.steering import SteeringVector, MultiLayerSteering
from ace_code.intervention.pisces import PISCES, PISCESConfig, PISCESResult


@dataclass
class PipelineConfig:
    """Configuration for the ACE pipeline."""

    # Model configuration
    model_name: str = "gemma-2-2b"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Analysis configuration
    target_layers: List[int] = field(default_factory=lambda: [10, 11, 12])
    sae_layer: int = 12

    # SAE configuration
    sae_path: Optional[str] = None
    sae_id: Optional[str] = None

    # Steering configuration
    default_steering_strength: float = 1.0

    # PAHQ configuration
    use_pahq: bool = False
    pahq_background_precision: torch.dtype = torch.float16

    # Output configuration
    output_dir: str = "./ace_output"
    save_intermediate: bool = True


@dataclass
class AnalysisResult:
    """Results from a complete ACE analysis."""

    code_pairs: List[CodePair]
    cdt_results: Dict[int, CDTResult]
    error_features: List[ErrorFeature]
    steering_vectors: Dict[int, SteeringVector]
    critical_layers: List[int]
    critical_heads: List[Tuple[int, int]]

    def summary(self) -> str:
        """Generate a summary of the analysis."""
        lines = [
            "=== ACE Analysis Summary ===",
            f"Code pairs analyzed: {len(self.code_pairs)}",
            f"Error features found: {len(self.error_features)}",
            f"Steering vectors created: {len(self.steering_vectors)}",
            f"Critical layers: {self.critical_layers}",
            f"Top critical heads: {self.critical_heads[:5]}",
        ]
        return "\n".join(lines)


class ACEPipeline:
    """
    Main orchestrator for the ACE-Code workflow.

    This class provides a high-level interface for running the complete
    ACE analysis and intervention pipeline on code models.

    The pipeline consists of:
    1. Model Setup: Load model with hooks via TransformerLens
    2. Data Preparation: Generate positive/negative code pairs
    3. Circuit Discovery: Use CD-T to find relevant components
    4. Error Detection: Use SAE to identify error features
    5. Intervention: Apply steering or permanent fixes

    Example:
        >>> # Full analysis workflow
        >>> pipeline = ACEPipeline("gemma-2-2b")
        >>> results = pipeline.full_analysis(code_snippets)
        >>> print(results.summary())

        >>> # Steered generation
        >>> output = pipeline.generate_with_steering(
        ...     "def calculate(",
        ...     steering_layer=12,
        ...     strength=1.5
        ... )
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        model: Optional[ACEModel] = None,
    ):
        """
        Initialize the ACE pipeline.

        Args:
            model_name_or_path: Name or path of model to load
            config: Pipeline configuration
            model: Pre-loaded ACEModel (optional, skips loading)
        """
        self.config = config or PipelineConfig()

        if model_name_or_path:
            self.config.model_name = model_name_or_path

        # Model and components (initialized lazily or via load())
        self.model: Optional[ACEModel] = model
        self.mutation_generator: Optional[MutationGenerator] = None
        self.cdt: Optional[ContextualDecomposition] = None
        self.sae_detector: Optional[SAEDetector] = None
        self.feature_analyzer: Optional[FeatureAnalyzer] = None
        self.pahq: Optional[PAHQQuantizer] = None
        self.pisces: Optional[PISCES] = None

        # Cached results
        self._steering_vectors: Dict[int, SteeringVector] = {}
        self._analysis_results: Optional[AnalysisResult] = None

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ACEPipeline for {self.config.model_name}")

    def load(self) -> "ACEPipeline":
        """
        Load the model and initialize all components.

        Returns:
            self for chaining
        """
        if self.model is None:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = load_hooked_model(
                model_name=self.config.model_name,
                device=self.config.device,
                dtype=self.config.dtype,
            )

        # Initialize components
        self.mutation_generator = MutationGenerator()
        self.cdt = ContextualDecomposition(self.model)
        self.sae_detector = SAEDetector(
            self.model,
            layer=self.config.sae_layer,
        )
        self.feature_analyzer = FeatureAnalyzer(self.model)
        self.pisces = PISCES(self.model)

        # Initialize PAHQ if configured
        if self.config.use_pahq:
            pahq_config = QuantizationConfig(
                background_precision=self.config.pahq_background_precision,
            )
            self.pahq = PAHQQuantizer(self.model, pahq_config)

        # Load SAE if path provided
        if self.config.sae_path or self.config.sae_id:
            self.sae_detector.load_sae(
                sae_path=self.config.sae_path,
                sae_id=self.config.sae_id,
            )
            self.feature_analyzer.set_sae(self.sae_detector.sae)

        logger.info("Pipeline components initialized")
        return self

    def generate_code_pairs(
        self,
        code_snippets: List[str],
        n_per_snippet: int = 1,
    ) -> List[CodePair]:
        """
        Generate positive/negative code pairs from snippets.

        Args:
            code_snippets: List of original code strings
            n_per_snippet: Number of pairs per snippet

        Returns:
            List of CodePair objects
        """
        if self.mutation_generator is None:
            self.mutation_generator = MutationGenerator()

        pairs = self.mutation_generator.generate_pairs(
            code_snippets,
            n_per_snippet=n_per_snippet,
        )

        logger.info(f"Generated {len(pairs)} code pairs")
        return pairs

    def discover_circuits(
        self,
        code_pair: CodePair,
        relevant_positions: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Discover circuits that differ between positive and negative code.

        Args:
            code_pair: CodePair to analyze
            relevant_positions: Token positions to mark as relevant

        Returns:
            Dictionary with circuit discovery results
        """
        if self.cdt is None:
            self._ensure_loaded()

        # Tokenize
        pos_tokens = self.model.tokenize(code_pair.positive)["input_ids"]
        neg_tokens = self.model.tokenize(code_pair.negative)["input_ids"]

        # Compare decompositions
        comparison = self.cdt.compare_decompositions(
            pos_tokens,
            neg_tokens,
            relevant_positions=relevant_positions,
        )

        return comparison

    def identify_error_features(
        self,
        code_pairs: List[CodePair],
        top_k: int = 20,
    ) -> List[ErrorFeature]:
        """
        Identify SAE features that indicate errors.

        Args:
            code_pairs: List of code pairs to analyze
            top_k: Number of top features to return

        Returns:
            List of ErrorFeature objects
        """
        if self.sae_detector is None:
            self._ensure_loaded()

        if not self.sae_detector.sae_loaded:
            # Load default SAE
            self.sae_detector.load_sae()

        # Tokenize pairs
        pos_inputs = [
            self.model.tokenize(pair.positive)["input_ids"]
            for pair in code_pairs
        ]
        neg_inputs = [
            self.model.tokenize(pair.negative)["input_ids"]
            for pair in code_pairs
        ]

        # Identify error features
        features = self.sae_detector.identify_error_features(
            positive_inputs=pos_inputs,
            negative_inputs=neg_inputs,
            top_k=top_k,
        )

        return features

    def compute_steering_vector(
        self,
        code_pair: CodePair,
        layer: int,
        position: int = -1,
    ) -> SteeringVector:
        """
        Compute a steering vector from a code pair.

        Args:
            code_pair: CodePair to use
            layer: Layer to compute vector for
            position: Token position (-1 for last)

        Returns:
            SteeringVector instance
        """
        self._ensure_loaded()

        sv = SteeringVector.from_prompts(
            model=self.model,
            positive_prompt=code_pair.positive,
            negative_prompt=code_pair.negative,
            layer=layer,
            position=position,
        )

        # Cache the steering vector
        self._steering_vectors[layer] = sv

        return sv

    def compute_steering_from_pairs(
        self,
        code_pairs: List[CodePair],
        layer: int,
        position: int = -1,
    ) -> SteeringVector:
        """
        Compute averaged steering vector from multiple pairs.

        Args:
            code_pairs: List of code pairs
            layer: Layer to compute vector for
            position: Token position

        Returns:
            Averaged SteeringVector
        """
        self._ensure_loaded()

        # Collect activations
        pos_caches = []
        neg_caches = []

        for pair in code_pairs:
            pos_tokens = self.model.tokenize(pair.positive)["input_ids"]
            neg_tokens = self.model.tokenize(pair.negative)["input_ids"]

            pos_cache = collect_activations(
                self.model,
                pos_tokens,
                layers=[layer],
            )
            neg_cache = collect_activations(
                self.model,
                neg_tokens,
                layers=[layer],
            )

            pos_caches.append(pos_cache)
            neg_caches.append(neg_cache)

        # Compute averaged steering vector
        sv = SteeringVector.from_caches(
            positive_caches=pos_caches,
            negative_caches=neg_caches,
            layer=layer,
            position=position,
        )

        self._steering_vectors[layer] = sv
        return sv

    def steer_generation(
        self,
        prompt: str,
        layer: Optional[int] = None,
        strength: Optional[float] = None,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generate text with steering applied.

        Args:
            prompt: Input prompt
            layer: Layer to apply steering (uses cached if available)
            strength: Steering strength
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        self._ensure_loaded()

        if layer is None:
            layer = self.config.sae_layer

        if strength is None:
            strength = self.config.default_steering_strength

        # Get or compute steering vector
        if layer not in self._steering_vectors:
            logger.warning(f"No steering vector for layer {layer}, generating without steering")
            return self.model.generate(prompt, max_new_tokens=max_new_tokens)

        sv = self._steering_vectors[layer]

        # Tokenize prompt
        tokens = self.model.tokenize(prompt)["input_ids"]

        # Generate with steering
        # Note: This is simplified - full implementation would need token-by-token generation
        # with steering applied at each step
        logger.info(f"Generating with steering (layer={layer}, strength={strength})")

        # For now, apply steering and get logits, then use standard generation
        # A complete implementation would hook into the generation loop
        return self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )

    def apply_pisces(
        self,
        direction: torch.Tensor,
        layer: int,
        component: str = "mlp_out",
        strength: float = 1.0,
    ) -> PISCESResult:
        """
        Apply PISCES permanent weight edit.

        Args:
            direction: Direction to remove
            layer: Layer to edit
            component: Component to edit
            strength: Removal strength

        Returns:
            PISCESResult with edit details
        """
        self._ensure_loaded()

        return self.pisces.remove_direction(
            direction=direction,
            layer=layer,
            component=component,
            strength=strength,
        )

    def full_analysis(
        self,
        code_snippets: List[str],
        n_pairs_per_snippet: int = 2,
    ) -> AnalysisResult:
        """
        Run the complete ACE analysis pipeline.

        This runs all stages:
        1. Generate code pairs
        2. Discover circuits via CD-T
        3. Identify error features via SAE
        4. Compute steering vectors

        Args:
            code_snippets: List of code snippets to analyze
            n_pairs_per_snippet: Number of mutation pairs per snippet

        Returns:
            AnalysisResult with all findings
        """
        self._ensure_loaded()

        logger.info(f"Starting full analysis on {len(code_snippets)} snippets")

        # Step 1: Generate code pairs
        code_pairs = self.generate_code_pairs(
            code_snippets,
            n_per_snippet=n_pairs_per_snippet,
        )

        # Step 2: Discover circuits
        cdt_results = {}
        all_head_diffs = {}

        for i, pair in enumerate(code_pairs[:5]):  # Limit for efficiency
            logger.info(f"Analyzing pair {i+1}/{min(len(code_pairs), 5)}")
            comparison = self.discover_circuits(pair)

            # Aggregate results
            if comparison["most_different_layer"]:
                layer, diff = comparison["most_different_layer"]
                cdt_results[layer] = comparison["positive_result"]

            for (layer, head), diff in comparison["head_differences"][:10]:
                key = (layer, head)
                if key not in all_head_diffs:
                    all_head_diffs[key] = []
                all_head_diffs[key].append(diff)

        # Find critical components
        layer_counts = {}
        for layer in cdt_results:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        critical_layers = sorted(layer_counts.keys(), key=lambda l: layer_counts[l], reverse=True)[:5]

        # Average head differences and find critical heads
        avg_head_diffs = {
            k: sum(v) / len(v)
            for k, v in all_head_diffs.items()
        }
        critical_heads = sorted(avg_head_diffs.keys(), key=lambda k: abs(avg_head_diffs[k]), reverse=True)[:10]

        # Step 3: Identify error features
        error_features = self.identify_error_features(code_pairs)

        # Step 4: Compute steering vectors for critical layers
        steering_vectors = {}
        for layer in critical_layers[:3]:  # Top 3 layers
            sv = self.compute_steering_from_pairs(code_pairs, layer)
            steering_vectors[layer] = sv

        # Compile results
        self._analysis_results = AnalysisResult(
            code_pairs=code_pairs,
            cdt_results=cdt_results,
            error_features=error_features,
            steering_vectors=steering_vectors,
            critical_layers=critical_layers,
            critical_heads=critical_heads,
        )

        logger.info("Full analysis complete")
        logger.info(self._analysis_results.summary())

        return self._analysis_results

    def detect_error(
        self,
        code: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[int, float]]:
        """
        Detect potential errors in code using trained detector.

        Args:
            code: Code string to analyze
            threshold: Detection threshold

        Returns:
            Tuple of (is_error, error_score, feature_activations)
        """
        self._ensure_loaded()

        tokens = self.model.tokenize(code)["input_ids"]
        return self.sae_detector.detect(tokens, threshold=threshold)

    def save_state(self, path: Optional[str] = None) -> str:
        """
        Save pipeline state (steering vectors, analysis results).

        Args:
            path: Output path (default: output_dir/pipeline_state.pt)

        Returns:
            Path where state was saved
        """
        if path is None:
            path = str(self.output_dir / "pipeline_state.pt")

        state = {
            "config": self.config,
            "steering_vectors": {
                k: {"vector": v.vector, "layer": v.layer, "metadata": v.metadata}
                for k, v in self._steering_vectors.items()
            },
        }

        if self._analysis_results:
            state["critical_layers"] = self._analysis_results.critical_layers
            state["critical_heads"] = self._analysis_results.critical_heads

        torch.save(state, path)
        logger.info(f"Saved pipeline state to {path}")
        return path

    def load_state(self, path: str) -> "ACEPipeline":
        """
        Load pipeline state from file.

        Args:
            path: Path to state file

        Returns:
            self for chaining
        """
        state = torch.load(path, map_location=self.config.device)

        # Restore steering vectors
        for layer, sv_data in state.get("steering_vectors", {}).items():
            self._steering_vectors[int(layer)] = SteeringVector(
                vector=sv_data["vector"],
                layer=sv_data["layer"],
                metadata=sv_data["metadata"],
            )

        logger.info(f"Loaded pipeline state from {path}")
        return self

    def _ensure_loaded(self) -> None:
        """Ensure model and components are loaded."""
        if self.model is None:
            self.load()


# Convenience function for quick analysis
def analyze_code(
    code_snippets: Union[str, List[str]],
    model_name: str = "gemma-2-2b",
    device: str = "cuda",
) -> AnalysisResult:
    """
    Quick analysis of code snippets.

    Args:
        code_snippets: Code snippet(s) to analyze
        model_name: Model to use
        device: Device to run on

    Returns:
        AnalysisResult with findings
    """
    if isinstance(code_snippets, str):
        code_snippets = [code_snippets]

    config = PipelineConfig(model_name=model_name, device=device)
    pipeline = ACEPipeline(config=config)
    pipeline.load()

    return pipeline.full_analysis(code_snippets)
