"""
problem_planner.py

Translates interpreted teacher input + difficulty guidance into a concrete
problem blueprint. The idea is for generators (DOCX/PDF/etc.) to receive a
single, well-structured object that explains exactly what to build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.difficulty_scaler import DifficultyScaler, LearnerProfile, default_profile
from core.topic_interpreter import interpret_concept


@dataclass
class ProblemBlueprint:
    """
    Minimal schema describing one worksheet prompt.

    Fields can and should expand (distractors, representations, rubric, etc.).
    For now we keep it intentionally small so humans can still reason about it
    when debugging the flow end-to-end.
    """

    concept_text: str
    domain: str
    subskills: List[str]
    difficulty: str
    quantitative_targets: Dict[str, Any] = field(default_factory=dict)
    scaffolds: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)
    diagram_types: List[str] = field(default_factory=list)  # Diagrams to show with problems
    answer_diagram_types: List[str] = field(default_factory=list)  # Diagrams for answer key (student creates)


class ProblemPlanner:
    """
    High-level orchestrator. Typical usage:
        planner = ProblemPlanner(grade=4)
        plan = planner.plan("add fractions with like denominators")
    `plan` returns a dict containing the concept analysis, difficulty metadata,
    and the blueprint itself so downstream steps can decide how much detail
    they need.
    """

    def __init__(self, grade: int):
        self.grade = grade
        self.scaler = DifficultyScaler(grade)

    def plan(
        self,
        concept_text: str,
        profile: Optional[LearnerProfile] = None,
        target_difficulty: str = "medium",
    ) -> Dict[str, Any]:
        profile = profile or default_profile(self.grade)
        concept_analysis = interpret_concept(concept_text, self.grade)
        domain = self._select_domain(concept_analysis)
        difficulty_plan = self.scaler.suggest_level(domain, profile, target_difficulty)
        
        # If LLM analysis provided custom number ranges, use them
        llm_analysis = concept_analysis.get("llm_analysis")
        if llm_analysis and llm_analysis.get("number_range"):
            llm_range = llm_analysis["number_range"]
            if isinstance(llm_range, list) and len(llm_range) == 2:
                difficulty_plan["quantitative_targets"]["number_range"] = llm_range
        
        # Get diagram types from concept analysis
        diagram_types = concept_analysis.get("diagram_types", [])
        answer_diagram_types = concept_analysis.get("answer_diagram_types", [])
        
        blueprint = ProblemBlueprint(
            concept_text=concept_text,
            domain=domain,
            subskills=concept_analysis["subskills"] or [domain],
            difficulty=difficulty_plan["difficulty_level"],
            quantitative_targets=difficulty_plan["quantitative_targets"],
            scaffolds=self._suggest_scaffolds(domain, concept_analysis["subskills"]),
            validation_checks=self._default_validation_checks(domain),
            diagram_types=diagram_types,
            answer_diagram_types=answer_diagram_types,
        )

        return {
            "concept_analysis": concept_analysis,
            "difficulty_plan": difficulty_plan,
            "blueprint": blueprint,
        }

    # ------------------------------------------------------------------ #
    # Helper methods (initial heuristics / placeholders)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_domain(concept_analysis: Dict[str, Any]) -> str:
        alignment = concept_analysis.get("domain_alignment", {})
        matched = alignment.get("matched") or []
        if matched:
            return matched[0]
        domains = concept_analysis.get("domains") or []
        for domain in domains:
            if domain != "unknown":
                return domain
        # Default fallback keeps us anchored to numeric fluency.
        return "numbers"

    @staticmethod
    def _suggest_scaffolds(domain: str, subskills: List[str]) -> List[str]:
        """
        Quick heuristics so templates have something to work with. Replace with
        data-driven decisions later.
        """

        scaffolds: List[str] = []
        if domain in {"fractions", "decimals"}:
            scaffolds.append("visual_model_prompt")
        if "word_problems" in subskills:
            scaffolds.append("step_by_step_plan")
        if domain in {"algebra", "functions"}:
            scaffolds.append("table_of_values")
        return scaffolds

    @staticmethod
    def _default_validation_checks(domain: str) -> List[str]:
        base_checks = ["answer_exists", "answer_unique"]
        if domain in {"operations", "fractions"}:
            base_checks.append("simplify_final_answer")
        if domain in {"geometry"}:
            base_checks.append("units_present")
        return base_checks


if __name__ == "__main__":
    planner = ProblemPlanner(grade=6)
    plan = planner.plan("ratio word problems with unit rates")
    print(plan["blueprint"])

