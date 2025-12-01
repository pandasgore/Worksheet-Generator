"""
difficulty_scaler.py

Lightweight scaffolding for calibrating worksheet difficulty. The real agent
will eventually read a richer learner model plus analytics, but this module
provides the plumbing so other components (problem planner, generators, etc.)
have a single place to ask, "How hard should this next problem be?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.grade_progression import get_grade_info

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


@dataclass
class LearnerProfile:
    """
    Minimal container for runtime personalization signals.

    Fields can grow over time—ideally we would persist this somewhere else and
    simply pass a snapshot into the scaler when planning a worksheet.
    """

    grade: int
    mastery: Dict[str, float] = field(default_factory=dict)  # 0.0–1.0 scale
    recent_results: List[bool] = field(default_factory=list)
    accommodations: Dict[str, Any] = field(default_factory=dict)


def default_profile(grade: int) -> LearnerProfile:
    """
    Helper so callers that do not yet track performance can still produce a
    valid profile object.
    """

    return LearnerProfile(grade=grade, mastery={}, recent_results=[], accommodations={})


class DifficultyScaler:
    """
    Computes a difficulty recommendation anchored to grade-level expectations.

    The idea is for callers to provide a domain ("fractions", "algebra", etc.)
    plus whatever student performance context they have. The scaler returns a
    `difficulty_level` along with quantitative targets (number ranges, multi-step
    expectations, etc.) so downstream generators know what to build.
    """

    def __init__(self, grade: int):
        self.grade = grade
        self.grade_info = get_grade_info(grade)
        if not self.grade_info:
            raise ValueError(f"Invalid grade: {grade}. Must be between 1 and 8.")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def suggest_level(
        self,
        domain: str,
        profile: Optional[LearnerProfile] = None,
        target: str = "medium",
    ) -> Dict[str, Any]:
        """
        Returns a recommendation with the shape:
        {
            "difficulty_level": "easy|medium|hard",
            "quantitative_targets": {
                "number_range": (lo, hi),
                "multi_step": bool,
                "include_word_problems": bool,
            },
            "rationale": "...human readable summary..."
        }
        """

        profile = profile or default_profile(self.grade)
        level = self._resolve_level(domain, profile, target)
        number_range = self._scaled_number_range(domain, level)
        rationale = self._build_rationale(domain, profile, target, level)

        return {
            "difficulty_level": level,
            "quantitative_targets": {
                "number_range": number_range,
                "multi_step": level == "hard",
                "include_word_problems": level != "easy",
            },
            "rationale": rationale,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _resolve_level(
        self,
        domain: str,
        profile: LearnerProfile,
        target: str,
    ) -> str:
        base_level = target if target in DIFFICULTY_LEVELS else "medium"
        mastery = profile.mastery.get(domain, 0.5)
        trend = self._compute_trend(profile.recent_results)

        # Nudges are intentionally conservative; tune once we collect data.
        if mastery > 0.75 and trend >= 0:
            return "hard"
        if mastery < 0.35 or trend < 0:
            return "easy"
        return base_level

    @staticmethod
    def _compute_trend(results: List[bool], window: int = 5) -> int:
        """
        Very rough measure: +1 if most recent answers are correct, -1 if most
        are incorrect, otherwise 0. Replace with EWMA or Bayesian model later.
        """

        if not results:
            return 0

        recent = results[-window:]
        score = sum(1 if r else -1 for r in recent)
        if score > 0:
            return 1
        if score < 0:
            return -1
        return 0

    def _scaled_number_range(
        self,
        domain: str,
        level: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Uses the grade's numeric expectations to compute a target range.
        Many domains ultimately draw on the same number set, so grabbing the
        base `numbers["range"]` keeps us anchored to the grade progression.
        """

        numbers_info = self.grade_info.get("numbers")
        if not numbers_info or "range" not in numbers_info:
            return None

        low, high = numbers_info["range"]
        span = max(high - low, 1)
        adjustment = max(span // 3, 5)

        if level == "easy":
            return (low, max(low + span // 2, low + adjustment))
        if level == "hard":
            return (low, high + adjustment)
        return (low, high)

    @staticmethod
    def _build_rationale(
        domain: str,
        profile: LearnerProfile,
        requested: str,
        final: str,
    ) -> str:
        """
        Produces a short natural-language blurb so teachers/debuggers know why
        a certain difficulty was chosen.
        """

        mastery = profile.mastery.get(domain, 0.5)
        trend = DifficultyScaler._compute_trend(profile.recent_results)
        trend_label = {1: "improving", 0: "steady", -1: "declining"}[trend]
        return (
            f"Requested {requested}, mastery={mastery:.2f}, "
            f"trend={trend_label}; returning {final} for domain '{domain}'."
        )


if __name__ == "__main__":
    # Quick smoke test
    scaler = DifficultyScaler(grade=5)
    sample_profile = LearnerProfile(
        grade=5,
        mastery={"fractions": 0.82},
        recent_results=[True, True, False, True, True],
    )
    print(scaler.suggest_level("fractions", sample_profile, target="medium"))

