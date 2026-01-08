"""
Mutation Testing for Code Pair Generation

This module implements semantics-preserving and semantics-breaking code
mutations to generate (Positive, Negative) code pairs for circuit analysis.
These pairs are essential for identifying the "Incorrectness" features.
"""

from __future__ import annotations

import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger

# Optional: tree-sitter for AST-based mutations
try:
    import tree_sitter
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not available. Using regex-based mutations only.")


class MutationType(Enum):
    """Types of code mutations for generating test pairs."""

    # Semantics-preserving mutations (should not affect correctness)
    VARIABLE_RENAME = "variable_rename"
    WHITESPACE_CHANGE = "whitespace_change"
    COMMENT_REMOVAL = "comment_removal"

    # Semantics-breaking mutations (induces errors)
    TYPE_HINT_REMOVAL = "type_hint_removal"
    RETURN_TYPE_REMOVAL = "return_type_removal"
    VARIABLE_MISSPELL = "variable_misspell"
    OPERATOR_SWAP = "operator_swap"
    OFF_BY_ONE = "off_by_one"
    NONE_RETURN = "none_return"
    ARGUMENT_SWAP = "argument_swap"


@dataclass
class CodePair:
    """
    A pair of code snippets for contrastive analysis.

    Attributes:
        positive: The correct (original) code
        negative: The mutated (buggy) code
        mutation_type: Type of mutation applied
        mutation_details: Additional information about the mutation
        expected_token: The token the model should predict (if applicable)
    """

    positive: str
    negative: str
    mutation_type: MutationType
    mutation_details: Dict[str, Any] = field(default_factory=dict)
    expected_token: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"CodePair(mutation={self.mutation_type.value}, "
            f"pos_len={len(self.positive)}, neg_len={len(self.negative)})"
        )


class BaseMutator(ABC):
    """Base class for code mutators."""

    @abstractmethod
    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply mutation to code.

        Args:
            code: Original code string

        Returns:
            Tuple of (mutated_code, mutation_details)
        """
        pass

    @abstractmethod
    def can_mutate(self, code: str) -> bool:
        """Check if this mutator can be applied to the code."""
        pass


class TypeHintRemovalMutator(BaseMutator):
    """
    Remove type hints from function signatures and variables.

    Example:
        def calculate(price: float) -> float:  =>  def calculate(price):
    """

    # Patterns for type hints
    PARAM_TYPE_PATTERN = re.compile(r":\s*[A-Za-z_][A-Za-z0-9_\[\], ]*(?=\s*[,)=])")
    RETURN_TYPE_PATTERN = re.compile(r"\)\s*->\s*[A-Za-z_][A-Za-z0-9_\[\], ]*\s*:")
    VAR_TYPE_PATTERN = re.compile(r":\s*[A-Za-z_][A-Za-z0-9_\[\], ]*\s*=")

    def can_mutate(self, code: str) -> bool:
        """Check if code contains type hints."""
        return bool(
            self.PARAM_TYPE_PATTERN.search(code)
            or self.RETURN_TYPE_PATTERN.search(code)
        )

    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Remove type hints from the code."""
        original = code
        removed_hints = []

        # Remove parameter type hints
        matches = list(self.PARAM_TYPE_PATTERN.finditer(code))
        for match in reversed(matches):
            removed_hints.append(match.group())
            code = code[: match.start()] + code[match.end() :]

        # Remove return type hints
        def remove_return_type(match):
            removed_hints.append(match.group())
            return "):"

        code = self.RETURN_TYPE_PATTERN.sub(remove_return_type, code)

        return code, {"removed_hints": removed_hints, "count": len(removed_hints)}


class VariableRenameMutator(BaseMutator):
    """
    Rename variables to semantically meaningless names.

    This creates "distractor" mutations that test whether the model
    relies on variable names or actual code structure.

    Example:
        def calculate_total(price, quantity):  =>  def calculate_total(x, y):
    """

    DISTRACTOR_NAMES = ["x", "y", "z", "a", "b", "c", "tmp", "val", "var"]

    # Pattern to find variable/parameter names
    VAR_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\b", re.IGNORECASE)

    # Keywords to skip
    PYTHON_KEYWORDS = {
        "def",
        "class",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "return",
        "yield",
        "import",
        "from",
        "as",
        "try",
        "except",
        "finally",
        "with",
        "lambda",
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "None",
        "pass",
        "break",
        "continue",
        "raise",
        "assert",
        "global",
        "nonlocal",
        "del",
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "self",
        "cls",
    }

    def can_mutate(self, code: str) -> bool:
        """Check if code has renameable variables."""
        # Find function definition with parameters
        return bool(re.search(r"def\s+\w+\s*\([^)]+\)", code))

    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Rename parameters to distractor names."""
        # Find function parameters
        func_match = re.search(r"def\s+\w+\s*\(([^)]+)\)", code)
        if not func_match:
            return code, {"renamed": []}

        params_str = func_match.group(1)

        # Extract parameter names (handling type hints)
        param_names = []
        for param in params_str.split(","):
            param = param.strip()
            if ":" in param:
                param_name = param.split(":")[0].strip()
            elif "=" in param:
                param_name = param.split("=")[0].strip()
            else:
                param_name = param.strip()

            if param_name and param_name not in self.PYTHON_KEYWORDS:
                param_names.append(param_name)

        # Create rename mapping
        renames = {}
        available_distractors = list(self.DISTRACTOR_NAMES)
        random.shuffle(available_distractors)

        for i, name in enumerate(param_names):
            if i < len(available_distractors):
                renames[name] = available_distractors[i]

        # Apply renames
        mutated = code
        for old_name, new_name in sorted(renames.items(), key=lambda x: -len(x[0])):
            # Use word boundaries to avoid partial matches
            pattern = rf"\b{re.escape(old_name)}\b"
            mutated = re.sub(pattern, new_name, mutated)

        return mutated, {"renames": renames}


class OperatorSwapMutator(BaseMutator):
    """
    Swap operators to introduce bugs.

    Example:
        x <= y  =>  x < y
        a + b   =>  a - b
    """

    OPERATOR_SWAPS = {
        "<=": "<",
        ">=": ">",
        "==": "=",
        "!=": "==",
        "+": "-",
        "-": "+",
        "*": "/",
        "and": "or",
        "or": "and",
    }

    def can_mutate(self, code: str) -> bool:
        """Check if code has swappable operators."""
        for op in self.OPERATOR_SWAPS:
            if op in code:
                return True
        return False

    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Swap one operator in the code."""
        for op, replacement in self.OPERATOR_SWAPS.items():
            if op in code:
                # Replace first occurrence only
                mutated = code.replace(op, replacement, 1)
                return mutated, {"original": op, "replacement": replacement}

        return code, {}


class OffByOneMutator(BaseMutator):
    """
    Introduce off-by-one errors in loop bounds or indices.

    Example:
        range(n)     =>  range(n-1)
        arr[i]       =>  arr[i+1]
        for i in range(len(arr)):  =>  for i in range(len(arr)-1):
    """

    RANGE_PATTERN = re.compile(r"range\(([^)]+)\)")
    INDEX_PATTERN = re.compile(r"\[(\w+)\]")

    def can_mutate(self, code: str) -> bool:
        """Check if code has range() or array indices."""
        return bool(self.RANGE_PATTERN.search(code) or self.INDEX_PATTERN.search(code))

    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Introduce off-by-one error."""
        # Try to modify range() first
        match = self.RANGE_PATTERN.search(code)
        if match:
            arg = match.group(1)
            if "-1" not in arg and "+1" not in arg:
                # Add -1 to range argument
                new_arg = f"{arg}-1"
                mutated = code[: match.start()] + f"range({new_arg})" + code[match.end() :]
                return mutated, {"type": "range", "original": arg, "mutated": new_arg}

        # Try to modify index
        match = self.INDEX_PATTERN.search(code)
        if match:
            idx = match.group(1)
            if idx.isidentifier():
                mutated = code[: match.start()] + f"[{idx}+1]" + code[match.end() :]
                return mutated, {"type": "index", "original": idx, "mutated": f"{idx}+1"}

        return code, {}


class ArgumentSwapMutator(BaseMutator):
    """
    Swap function arguments to introduce bugs.

    Example:
        subtract(a, b)  =>  subtract(b, a)
    """

    CALL_PATTERN = re.compile(r"(\w+)\(([^)]+,[^)]+)\)")

    def can_mutate(self, code: str) -> bool:
        """Check if code has function calls with multiple arguments."""
        return bool(self.CALL_PATTERN.search(code))

    def mutate(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Swap first two arguments of a function call."""
        match = self.CALL_PATTERN.search(code)
        if not match:
            return code, {}

        func_name = match.group(1)
        args_str = match.group(2)

        # Split arguments
        args = [a.strip() for a in args_str.split(",")]
        if len(args) >= 2:
            # Swap first two arguments
            args[0], args[1] = args[1], args[0]
            new_args = ", ".join(args)
            mutated = (
                code[: match.start()]
                + f"{func_name}({new_args})"
                + code[match.end() :]
            )
            return mutated, {
                "function": func_name,
                "original_order": args_str,
                "swapped_order": new_args,
            }

        return code, {}


class MutationGenerator:
    """
    Generator for creating (Positive, Negative) code pairs.

    This class orchestrates the mutation process, applying various
    mutators to generate training data for circuit analysis.

    Example:
        >>> generator = MutationGenerator()
        >>> pairs = generator.generate_pairs(code_snippets, n_per_snippet=3)
        >>> for pair in pairs:
        ...     print(f"Mutation: {pair.mutation_type}")
        ...     print(f"Original: {pair.positive[:50]}...")
        ...     print(f"Mutated: {pair.negative[:50]}...")
    """

    def __init__(
        self,
        mutation_types: Optional[List[MutationType]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the mutation generator.

        Args:
            mutation_types: List of mutation types to use (None for all)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Initialize mutators
        self._mutators: Dict[MutationType, BaseMutator] = {
            MutationType.TYPE_HINT_REMOVAL: TypeHintRemovalMutator(),
            MutationType.VARIABLE_RENAME: VariableRenameMutator(),
            MutationType.OPERATOR_SWAP: OperatorSwapMutator(),
            MutationType.OFF_BY_ONE: OffByOneMutator(),
            MutationType.ARGUMENT_SWAP: ArgumentSwapMutator(),
        }

        # Filter to requested types
        if mutation_types:
            self._active_types = [t for t in mutation_types if t in self._mutators]
        else:
            self._active_types = list(self._mutators.keys())

        logger.info(f"Initialized MutationGenerator with {len(self._active_types)} mutation types")

    def generate_pair(
        self,
        code: str,
        mutation_type: Optional[MutationType] = None,
    ) -> Optional[CodePair]:
        """
        Generate a single (Positive, Negative) code pair.

        Args:
            code: Original code snippet
            mutation_type: Specific mutation to apply (None for random)

        Returns:
            CodePair or None if no valid mutation could be applied
        """
        if mutation_type is None:
            # Find applicable mutation types
            applicable = [
                t
                for t in self._active_types
                if self._mutators[t].can_mutate(code)
            ]
            if not applicable:
                logger.debug("No applicable mutations for code snippet")
                return None
            mutation_type = random.choice(applicable)

        mutator = self._mutators.get(mutation_type)
        if mutator is None or not mutator.can_mutate(code):
            return None

        mutated_code, details = mutator.mutate(code)

        # Verify mutation actually changed the code
        if mutated_code == code:
            return None

        return CodePair(
            positive=code,
            negative=mutated_code,
            mutation_type=mutation_type,
            mutation_details=details,
        )

    def generate_pairs(
        self,
        code_snippets: List[str],
        n_per_snippet: int = 1,
        include_all_types: bool = False,
    ) -> List[CodePair]:
        """
        Generate multiple code pairs from a list of snippets.

        Args:
            code_snippets: List of original code snippets
            n_per_snippet: Number of pairs to generate per snippet
            include_all_types: If True, generate one pair per mutation type

        Returns:
            List of CodePair objects
        """
        pairs = []

        for code in code_snippets:
            if include_all_types:
                # Generate one pair per applicable mutation type
                for mutation_type in self._active_types:
                    pair = self.generate_pair(code, mutation_type)
                    if pair is not None:
                        pairs.append(pair)
            else:
                # Generate n_per_snippet random pairs
                for _ in range(n_per_snippet):
                    pair = self.generate_pair(code)
                    if pair is not None:
                        pairs.append(pair)

        logger.info(f"Generated {len(pairs)} code pairs from {len(code_snippets)} snippets")
        return pairs

    def generate_type_prediction_pairs(
        self,
        typed_code: str,
        target_parameter: Optional[str] = None,
    ) -> Optional[CodePair]:
        """
        Generate pairs specifically for type prediction analysis.

        This creates pairs where the positive has type hints and the
        negative has them removed, useful for analyzing type inference circuits.

        Args:
            typed_code: Code with type hints
            target_parameter: Specific parameter to focus on

        Returns:
            CodePair for type prediction analysis
        """
        mutator = self._mutators[MutationType.TYPE_HINT_REMOVAL]

        if not mutator.can_mutate(typed_code):
            return None

        untyped_code, details = mutator.mutate(typed_code)

        # Extract expected type token if possible
        expected_token = None
        if details.get("removed_hints"):
            # Get the first removed type
            first_hint = details["removed_hints"][0]
            # Extract type name
            type_match = re.search(r":\s*(\w+)", first_hint)
            if type_match:
                expected_token = type_match.group(1)

        return CodePair(
            positive=typed_code,
            negative=untyped_code,
            mutation_type=MutationType.TYPE_HINT_REMOVAL,
            mutation_details=details,
            expected_token=expected_token,
        )

    @staticmethod
    def create_completion_pair(
        prefix: str,
        correct_completion: str,
        incorrect_completion: str,
    ) -> CodePair:
        """
        Create a code pair from explicit completions.

        This is useful when you have specific correct and incorrect
        continuations for a code prefix.

        Args:
            prefix: The code prefix before the completion
            correct_completion: The correct code continuation
            incorrect_completion: The incorrect code continuation

        Returns:
            CodePair for completion analysis
        """
        return CodePair(
            positive=prefix + correct_completion,
            negative=prefix + incorrect_completion,
            mutation_type=MutationType.VARIABLE_MISSPELL,  # Generic
            mutation_details={
                "prefix": prefix,
                "correct": correct_completion,
                "incorrect": incorrect_completion,
            },
            expected_token=correct_completion.split()[0] if correct_completion else None,
        )


# Example code snippets for testing
EXAMPLE_CODE_SNIPPETS = [
    '''def calculate_total(price: float, quantity: int) -> float:
    """Calculate the total price."""
    return price * quantity
''',
    '''def find_max(numbers: list[int]) -> int:
    """Find maximum value in a list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
''',
    '''def is_palindrome(text: str) -> bool:
    """Check if text is a palindrome."""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
''',
    '''def binary_search(arr: list[int], target: int) -> int:
    """Perform binary search on sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
]


def generate_test_pairs(n_pairs: int = 10) -> List[CodePair]:
    """Generate test code pairs using example snippets."""
    generator = MutationGenerator(seed=42)
    return generator.generate_pairs(
        EXAMPLE_CODE_SNIPPETS,
        n_per_snippet=n_pairs // len(EXAMPLE_CODE_SNIPPETS) + 1,
    )[:n_pairs]
