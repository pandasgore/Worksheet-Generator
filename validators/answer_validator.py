"""
answer_validator.py

SymPy-based scaffolding that checks whether a learner response matches one (or
several) acceptable answers. The validator keeps the logic lightweight so we
can plug it directly into generators, doc builders, or even live tutoring
agents without depending on any UI code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sympy import Expr, sympify
from sympy.core.sympify import SympifyError
from sympy import simplify


@dataclass
class ValidationResult:
    """
    Represents the outcome of a single validation attempt.
    """

    correct: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class AnswerValidator:
    """
    Validates symbolic or numeric answers using SymPy.

    Usage:
        validator = AnswerValidator()
        result = validator.validate("2/3", learner_response)
        if result.correct:
            ...
    """

    def __init__(self, *, tolerance: float = 1e-6):
        self.tolerance = tolerance

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate(
        self,
        expected: Any,
        actual: Any,
        *,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Compares `actual` against `expected`. Accepts primitives (str, int,
        float), SymPy expressions, sequences of acceptable answers, or nested
        mappings (for multi-part problems).
        """

        return self._validate_single(expected, actual, variables or {})

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _validate_single(
        self,
        expected: Any,
        actual: Any,
        variables: Dict[str, Any],
    ) -> ValidationResult:
        if isinstance(expected, Mapping):
            return self._validate_mapping(expected, actual, variables)

        if isinstance(expected, (Sequence, set)) and not isinstance(expected, (str, bytes)):
            return self._validate_sequence(expected, actual, variables)

        return self._validate_leaf(expected, actual, variables)

    def _validate_mapping(
        self,
        expected: Mapping[str, Any],
        actual: Any,
        variables: Dict[str, Any],
    ) -> ValidationResult:
        if not isinstance(actual, Mapping):
            return ValidationResult(
                correct=False,
                message="Expected a mapping/dict style response.",
                details={"received_type": type(actual).__name__},
            )

        component_results: Dict[str, bool] = {}
        messages: Dict[str, str] = {}
        all_correct = True

        for key, expected_value in expected.items():
            if key not in actual:
                component_results[key] = False
                messages[key] = "Missing component."
                all_correct = False
                continue

            result = self._validate_single(expected_value, actual[key], variables)
            component_results[key] = result.correct
            messages[key] = result.message
            if not result.correct:
                all_correct = False

        message = "All parts correct." if all_correct else "One or more parts incorrect."
        return ValidationResult(
            correct=all_correct,
            message=message,
            details={
                "component_results": component_results,
                "component_messages": messages,
            },
        )

    def _validate_sequence(
        self,
        expected_options: Iterable[Any],
        actual: Any,
        variables: Dict[str, Any],
    ) -> ValidationResult:
        errors: List[str] = []
        for option in expected_options:
            result = self._validate_single(option, actual, variables)
            if result.correct:
                result.message = "Matched one of the acceptable answers."
                return result
            errors.append(result.message)

        return ValidationResult(
            correct=False,
            message="Response did not match any acceptable answer.",
            details={"attempts": errors},
        )

    def _validate_leaf(
        self,
        expected: Any,
        actual: Any,
        variables: Dict[str, Any],
    ) -> ValidationResult:
        try:
            expected_expr = self._to_expr(expected, variables)
            actual_expr = self._to_expr(actual, variables)
        except ValueError as exc:
            return ValidationResult(
                correct=False,
                message=str(exc),
                details={"expected": expected, "actual": actual},
            )

        if self._expressions_match(expected_expr, actual_expr):
            return ValidationResult(
                correct=True,
                message="Answers are equivalent.",
                details={
                    "expected_expr": str(expected_expr),
                    "actual_expr": str(actual_expr),
                },
            )

        return ValidationResult(
            correct=False,
            message="Answers differ.",
            details={
                "expected_expr": str(expected_expr),
                "actual_expr": str(actual_expr),
            },
        )

    def _to_expr(self, value: Any, variables: Dict[str, Any]) -> Expr:
        if isinstance(value, Expr):
            return value
        if isinstance(value, (int, float, complex, Decimal)):
            return sympify(value)
        if isinstance(value, str):
            try:
                return sympify(value, locals=variables)
            except SympifyError as exc:
                raise ValueError(f"Unable to parse expression '{value}'.") from exc
        if isinstance(value, bool):
            return sympify(int(value))

        raise ValueError(f"Unsupported answer type: {type(value).__name__}")

    def _expressions_match(self, expected: Expr, actual: Expr) -> bool:
        if expected.free_symbols or actual.free_symbols:
            return simplify(expected - actual) == 0

        expected_val = expected.evalf()
        actual_val = actual.evalf()
        return abs(expected_val - actual_val) <= self.tolerance


def validate_answer(
    expected: Any,
    actual: Any,
    *,
    tolerance: float = 1e-6,
    variables: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Convenience function for one-off validations.
    """

    validator = AnswerValidator(tolerance=tolerance)
    return validator.validate(expected, actual, variables=variables)


if __name__ == "__main__":
    validator = AnswerValidator()
    demo = validator.validate("2/3", "0.6666667")
    print(demo)

