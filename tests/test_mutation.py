"""Tests for the mutation testing module."""

import pytest

from ace_code.data.mutation import (
    MutationGenerator,
    MutationType,
    CodePair,
    TypeHintRemovalMutator,
    VariableRenameMutator,
    OperatorSwapMutator,
    EXAMPLE_CODE_SNIPPETS,
)


class TestTypeHintRemovalMutator:
    """Tests for TypeHintRemovalMutator."""

    def test_can_mutate_with_type_hints(self):
        mutator = TypeHintRemovalMutator()
        code = "def calculate(price: float) -> float:\n    return price * 1.1"
        assert mutator.can_mutate(code)

    def test_cannot_mutate_without_type_hints(self):
        mutator = TypeHintRemovalMutator()
        code = "def calculate(price):\n    return price * 1.1"
        assert not mutator.can_mutate(code)

    def test_removes_parameter_type_hints(self):
        mutator = TypeHintRemovalMutator()
        code = "def calculate(price: float, quantity: int) -> float:"
        mutated, details = mutator.mutate(code)

        assert ": float" not in mutated or mutated.count(": float") < code.count(": float")
        assert ": int" not in mutated
        assert "removed_hints" in details

    def test_removes_return_type(self):
        mutator = TypeHintRemovalMutator()
        code = "def foo(x: int) -> str:\n    return str(x)"
        mutated, details = mutator.mutate(code)

        assert "-> str" not in mutated


class TestVariableRenameMutator:
    """Tests for VariableRenameMutator."""

    def test_can_mutate_function_with_params(self):
        mutator = VariableRenameMutator()
        code = "def calculate(price, quantity):\n    return price * quantity"
        assert mutator.can_mutate(code)

    def test_renames_parameters(self):
        mutator = VariableRenameMutator()
        code = "def calculate(price, quantity):\n    return price * quantity"
        mutated, details = mutator.mutate(code)

        assert "price" not in mutated or mutated != code
        assert "renames" in details


class TestOperatorSwapMutator:
    """Tests for OperatorSwapMutator."""

    def test_can_mutate_with_operators(self):
        mutator = OperatorSwapMutator()
        code = "if x <= y:\n    return x + y"
        assert mutator.can_mutate(code)

    def test_swaps_operator(self):
        mutator = OperatorSwapMutator()
        code = "if x <= y:\n    return True"
        mutated, details = mutator.mutate(code)

        assert mutated != code
        assert "original" in details
        assert "replacement" in details


class TestMutationGenerator:
    """Tests for MutationGenerator."""

    def test_initialization(self):
        generator = MutationGenerator()
        assert len(generator._active_types) > 0

    def test_generate_pair(self):
        generator = MutationGenerator(seed=42)
        code = "def foo(x: int) -> int:\n    return x + 1"

        pair = generator.generate_pair(code)

        assert pair is not None
        assert isinstance(pair, CodePair)
        assert pair.positive == code
        assert pair.negative != code

    def test_generate_pairs_from_snippets(self):
        generator = MutationGenerator(seed=42)

        pairs = generator.generate_pairs(EXAMPLE_CODE_SNIPPETS, n_per_snippet=1)

        assert len(pairs) >= 1
        for pair in pairs:
            assert isinstance(pair, CodePair)
            assert pair.positive != pair.negative

    def test_generate_type_prediction_pairs(self):
        generator = MutationGenerator()
        typed_code = "def add(a: int, b: int) -> int:\n    return a + b"

        pair = generator.generate_type_prediction_pairs(typed_code)

        assert pair is not None
        assert pair.mutation_type == MutationType.TYPE_HINT_REMOVAL


class TestCodePair:
    """Tests for CodePair dataclass."""

    def test_code_pair_creation(self):
        pair = CodePair(
            positive="def foo(): pass",
            negative="def foo(): return None",
            mutation_type=MutationType.NONE_RETURN,
        )

        assert pair.positive != pair.negative
        assert pair.mutation_type == MutationType.NONE_RETURN

    def test_code_pair_repr(self):
        pair = CodePair(
            positive="abc",
            negative="xyz",
            mutation_type=MutationType.VARIABLE_RENAME,
        )

        repr_str = repr(pair)
        assert "variable_rename" in repr_str
