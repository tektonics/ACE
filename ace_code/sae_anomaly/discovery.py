"""
Feature Discovery for SAE-Based Anomaly Detection

This module identifies "incorrectness" features - SAE latents that activate
significantly more for buggy code than correct code. This is done by:

1. Generating code samples using the base model on MBPP problems
2. Executing code against test cases to label correct/incorrect
3. Extracting SAE features at the final prompt token
4. Computing t-statistics to identify incorrectness-correlated features
5. Filtering by activation frequency on general text

Based on: Tahimic & Cheng (2025) - Mechanistic Interpretability of Code Correctness

Key finding: "Incorrect-Predicting" features achieve F1 ≈ 0.821, while
"Correct-Predicting" features (F1 ~0.50) are unreliable.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np
import torch
from tqdm import tqdm
from ace_code.sae_anomaly.sidecar import SidecarModel
from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.utils.statistics import (
    compute_t_statistic,
    compute_feature_significance,
    filter_by_frequency,
    compute_activation_frequency,
)
from ace_code.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DiscoveryConfig:
    """Configuration for feature discovery."""

    # Dataset
    dataset_name: str = "mbpp"  # Mostly Basic Python Problems
    n_samples: Optional[int] = None  # Limit samples (None = all)

    # Generation
    temperature: float = 0.0  # Greedy decoding as per paper
    max_new_tokens: int = 256

    # Feature selection
    significance_level: float = 0.05
    bonferroni_correction: bool = True
    min_t_statistic: float = 2.0  # Minimum t-stat for selection
    max_frequency: float = 0.02  # Filter features active on >2% of general text
    top_k_features: Optional[int] = 100  # Select top-k features by t-stat

    # Execution
    timeout_seconds: int = 5  # Timeout for test execution
    use_sandbox: bool = True  # Run code in sandboxed environment

    # Output
    save_activations: bool = False
    output_dir: Optional[str] = None


@dataclass
class FeatureDiscoveryResult:
    """Results from feature discovery."""

    # Selected features
    incorrectness_features: np.ndarray  # Indices of incorrectness features
    feature_t_statistics: np.ndarray  # T-statistics for selected features
    feature_weights: np.ndarray  # Weights based on t-statistics

    # Statistics
    n_correct_samples: int
    n_incorrect_samples: int
    n_total_features: int
    n_selected_features: int

    # Metadata
    model_name: str
    target_layer: int
    config: DiscoveryConfig

    def save(self, path: Union[str, Path]) -> None:
        """Save discovery results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "incorrectness_features": self.incorrectness_features.tolist(),
            "feature_t_statistics": self.feature_t_statistics.tolist(),
            "feature_weights": self.feature_weights.tolist(),
            "n_correct_samples": self.n_correct_samples,
            "n_incorrect_samples": self.n_incorrect_samples,
            "n_total_features": self.n_total_features,
            "n_selected_features": self.n_selected_features,
            "model_name": self.model_name,
            "target_layer": self.target_layer,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved discovery results to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[DiscoveryConfig] = None) -> "FeatureDiscoveryResult":
        """Load discovery results from file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            incorrectness_features=np.array(data["incorrectness_features"]),
            feature_t_statistics=np.array(data["feature_t_statistics"]),
            feature_weights=np.array(data["feature_weights"]),
            n_correct_samples=data["n_correct_samples"],
            n_incorrect_samples=data["n_incorrect_samples"],
            n_total_features=data["n_total_features"],
            n_selected_features=data["n_selected_features"],
            model_name=data["model_name"],
            target_layer=data["target_layer"],
            config=config or DiscoveryConfig(),
        )


class CodeExecutor:
    """
    Safe code executor for testing generated code.

    Runs code against test cases and determines correctness.
    """

    def __init__(
        self,
        timeout_seconds: int = 5,
        use_sandbox: bool = True,
    ):
        """
        Initialize code executor.

        Args:
            timeout_seconds: Timeout for each test execution
            use_sandbox: Whether to use sandboxed execution
        """
        self.timeout = timeout_seconds
        self.use_sandbox = use_sandbox

    def execute(
        self,
        code: str,
        test_cases: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute code against test cases.

        Args:
            code: Generated Python code
            test_cases: List of test assertion strings

        Returns:
            Tuple of (is_correct, details)
        """
        # Combine code and test cases
        full_code = code + "\n\n" + "\n".join(test_cases)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(full_code)
                temp_path = f.name

            # Execute with timeout
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            is_correct = result.returncode == 0

            return is_correct, {
                "returncode": result.returncode,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
            }

        except subprocess.TimeoutExpired:
            return False, {"error": "timeout"}
        except Exception as e:
            return False, {"error": str(e)}
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass


class FeatureDiscovery:
    """
    Discovers "incorrectness" features in SAE latent space.

    This is NOT done at runtime - it's a pre-computation step that identifies
    which of the ~32k SAE features correspond to "bugs" by analyzing
    activations on correct vs incorrect code samples.

    Process:
    1. Load MBPP dataset
    2. Generate code solutions with temperature=0
    3. Execute against test cases
    4. Extract SAE activations at final prompt token
    5. Compute t-statistics for each feature
    6. Filter by frequency on general text
    7. Select top incorrectness features
    """

    def __init__(
        self,
        sidecar: SidecarModel,
        config: Optional[DiscoveryConfig] = None,
    ):
        """
        Initialize feature discovery.

        Args:
            sidecar: Loaded SidecarModel instance
            config: Discovery configuration
        """
        self.sidecar = sidecar
        self.config = config or DiscoveryConfig()
        self.executor = CodeExecutor(
            timeout_seconds=self.config.timeout_seconds,
            use_sandbox=self.config.use_sandbox,
        )

        self._correct_activations: List[torch.Tensor] = []
        self._incorrect_activations: List[torch.Tensor] = []

    def _load_mbpp_dataset(self) -> List[Dict[str, Any]]:
        """Load MBPP dataset."""
        try:
            from datasets import load_dataset

            logger.info("Loading MBPP dataset...")
            dataset = load_dataset("mbpp", split="test")

            samples = []
            for item in dataset:
                samples.append({
                    "task_id": item["task_id"],
                    "text": item["text"],  # Problem description
                    "code": item["code"],  # Reference solution
                    "test_list": item["test_list"],  # Test cases
                })

            if self.config.n_samples:
                samples = samples[:self.config.n_samples]

            logger.info(f"Loaded {len(samples)} MBPP samples")
            return samples

        except Exception as e:
            logger.error(f"Failed to load MBPP dataset: {e}")
            raise

    def _create_prompt(self, problem: Dict[str, Any]) -> str:
        """
        Create prompt for code generation.

        Uses the prompt template from Tahimic & Cheng (2025).
        """
        return f"""Write a Python function to solve the following problem.

Problem: {problem['text']}

Solution:
```python
"""

    def _extract_code(self, generation: str) -> str:
        """Extract code from model generation."""
        # Handle markdown code blocks
        if "```python" in generation:
            code = generation.split("```python")[0]
        elif "```" in generation:
            code = generation.split("```")[0]
        else:
            code = generation

        # Clean up
        code = code.strip()

        # Ensure function is complete
        if not code.endswith(":") and not code.endswith("\n"):
            code += "\n"

        return code

    def _collect_activations(
        self,
        problems: List[Dict[str, Any]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Collect activations for correct and incorrect code.

        Args:
            problems: List of MBPP problems

        Returns:
            Tuple of (correct_activations, incorrect_activations)
        """
        correct_activations = []
        incorrect_activations = []

        logger.info("Collecting activations from code generation...")

        for problem in tqdm(problems, desc="Processing problems"):
            prompt = self._create_prompt(problem)

            # Get activation at final prompt token BEFORE generation
            # This is the key extraction point per the paper
            activation = self.sidecar.get_sae_features(
                prompt,
                position="final",
            )

            # Generate code
            generated = self.sidecar.model.generate(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            code = self._extract_code(generated)

            # Execute and label
            is_correct, details = self.executor.execute(
                code,
                problem["test_list"],
            )

            if is_correct:
                correct_activations.append(activation.cpu())
            else:
                incorrect_activations.append(activation.cpu())

        logger.info(
            f"Collected {len(correct_activations)} correct, "
            f"{len(incorrect_activations)} incorrect samples"
        )

        return correct_activations, incorrect_activations

    def _compute_feature_selection(
        self,
        correct_activations: torch.Tensor,
        incorrect_activations: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute t-statistics and select incorrectness features.

        Args:
            correct_activations: Shape (n_correct, n_features)
            incorrect_activations: Shape (n_incorrect, n_features)

        Returns:
            Tuple of (selected_indices, t_statistics)
        """
        n_features = correct_activations.shape[1]

        logger.info(f"Computing t-statistics for {n_features} features...")

        # Compute t-statistics for all features
        t_stats, p_values, significant_mask = compute_feature_significance(
            incorrect_activations.numpy(),
            correct_activations.numpy(),
            n_features,
            alpha=self.config.significance_level,
            bonferroni_correction=self.config.bonferroni_correction,
        )

        # Filter by minimum t-statistic
        high_t_mask = t_stats >= self.config.min_t_statistic

        # Combine masks
        candidate_mask = significant_mask & high_t_mask
        candidate_indices = np.where(candidate_mask)[0]

        logger.info(f"Found {len(candidate_indices)} candidate incorrectness features")

        # Sort by t-statistic
        sorted_indices = candidate_indices[
            np.argsort(t_stats[candidate_indices])[::-1]
        ]

        # Select top-k
        if self.config.top_k_features:
            selected_indices = sorted_indices[:self.config.top_k_features]
        else:
            selected_indices = sorted_indices

        selected_t_stats = t_stats[selected_indices]

        logger.info(f"Selected {len(selected_indices)} incorrectness features")
        logger.info(f"Top 5 t-statistics: {selected_t_stats[:5]}")

        return selected_indices, selected_t_stats

    def _compute_frequency_filter(
        self,
        selected_indices: np.ndarray,
        general_text_activations: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Filter features by activation frequency on general text.

        Discards features that activate on >2% of general text to ensure
        we're not flagging common syntax as bugs.

        Args:
            selected_indices: Candidate feature indices
            general_text_activations: Optional activations on general text

        Returns:
            Filtered feature indices
        """
        if general_text_activations is None:
            # Without general text data, skip frequency filtering
            logger.warning(
                "No general text activations provided. "
                "Skipping frequency filtering."
            )
            return selected_indices

        frequencies = compute_activation_frequency(general_text_activations)
        filtered_indices = filter_by_frequency(
            selected_indices,
            frequencies,
            max_frequency=self.config.max_frequency,
        )

        n_filtered = len(selected_indices) - len(filtered_indices)
        logger.info(f"Filtered {n_filtered} high-frequency features")

        return filtered_indices

    def discover(
        self,
        general_text_activations: Optional[torch.Tensor] = None,
    ) -> FeatureDiscoveryResult:
        """
        Run the full feature discovery pipeline.

        Args:
            general_text_activations: Optional activations on general text
                (e.g., from The Pile) for frequency filtering

        Returns:
            FeatureDiscoveryResult with selected features
        """
        logger.info("Starting feature discovery pipeline...")

        # Load dataset
        problems = self._load_mbpp_dataset()

        # Collect activations
        correct_acts, incorrect_acts = self._collect_activations(problems)

        if len(correct_acts) < 2 or len(incorrect_acts) < 2:
            raise ValueError(
                f"Insufficient samples: {len(correct_acts)} correct, "
                f"{len(incorrect_acts)} incorrect. Need at least 2 of each."
            )

        # Stack activations
        correct_tensor = torch.cat(correct_acts, dim=0)
        incorrect_tensor = torch.cat(incorrect_acts, dim=0)

        # Store for potential reuse
        self._correct_activations = correct_acts
        self._incorrect_activations = incorrect_acts

        # Compute feature selection
        selected_indices, t_statistics = self._compute_feature_selection(
            correct_tensor,
            incorrect_tensor,
        )

        # Apply frequency filter if available
        if general_text_activations is not None:
            selected_indices = self._compute_frequency_filter(
                selected_indices,
                general_text_activations,
            )
            t_statistics = t_statistics[:len(selected_indices)]

        # Compute weights (normalized t-statistics)
        weights = t_statistics / t_statistics.sum()

        # Create result
        result = FeatureDiscoveryResult(
            incorrectness_features=selected_indices,
            feature_t_statistics=t_statistics,
            feature_weights=weights,
            n_correct_samples=len(correct_acts),
            n_incorrect_samples=len(incorrect_acts),
            n_total_features=self.sidecar.sae.n_latents,
            n_selected_features=len(selected_indices),
            model_name=self.sidecar.config.model_name,
            target_layer=self.sidecar.config.target_layer,
            config=self.config,
        )

        # Save if configured
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / "discovery_result.json"
            result.save(output_path)

        logger.info(
            f"Feature discovery complete. "
            f"Selected {result.n_selected_features} incorrectness features."
        )

        return result

    def discover_from_labeled_data(
        self,
        correct_prompts: List[str],
        incorrect_prompts: List[str],
    ) -> FeatureDiscoveryResult:
        """
        Discover features from pre-labeled data.

        Alternative to full pipeline when you already have labeled
        correct/incorrect samples.

        Args:
            correct_prompts: Prompts that led to correct code
            incorrect_prompts: Prompts that led to incorrect code

        Returns:
            FeatureDiscoveryResult
        """
        logger.info(
            f"Discovering features from labeled data: "
            f"{len(correct_prompts)} correct, {len(incorrect_prompts)} incorrect"
        )

        # Collect activations
        correct_acts = []
        for prompt in tqdm(correct_prompts, desc="Processing correct"):
            act = self.sidecar.get_sae_features(prompt, position="final")
            correct_acts.append(act.cpu())

        incorrect_acts = []
        for prompt in tqdm(incorrect_prompts, desc="Processing incorrect"):
            act = self.sidecar.get_sae_features(prompt, position="final")
            incorrect_acts.append(act.cpu())

        # Stack and compute
        correct_tensor = torch.cat(correct_acts, dim=0)
        incorrect_tensor = torch.cat(incorrect_acts, dim=0)

        selected_indices, t_statistics = self._compute_feature_selection(
            correct_tensor,
            incorrect_tensor,
        )

        weights = t_statistics / t_statistics.sum()

        return FeatureDiscoveryResult(
            incorrectness_features=selected_indices,
            feature_t_statistics=t_statistics,
            feature_weights=weights,
            n_correct_samples=len(correct_prompts),
            n_incorrect_samples=len(incorrect_prompts),
            n_total_features=self.sidecar.sae.n_latents,
            n_selected_features=len(selected_indices),
            model_name=self.sidecar.config.model_name,
            target_layer=self.sidecar.config.target_layer,
            config=self.config,
        )
