"""
generate_problems_tool.py

Exposes a single entry point that the agent (or a CLI) can call to produce a
`WorksheetPlan` from raw teacher text.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from generators.universal_generator import UniversalGenerator
from core.difficulty_scaler import LearnerProfile, default_profile


def run_generate_problems(
    concept_text: str,
    *,
    grade: int,
    num_problems: int = 5,
    profile: Optional[LearnerProfile] = None,
    target_difficulty: str = "medium",
) -> Dict[str, Any]:
    generator = UniversalGenerator(grade)
    plan = generator.generate(
        concept_text,
        num_problems=num_problems,
        profile=profile or default_profile(grade),
        target_difficulty=target_difficulty,
    )
    serializable = {
        "concept_text": plan.concept_text,
        "grade": plan.grade,
        "difficulty_plan": plan.difficulty_plan,
        "concept_analysis": plan.concept_analysis,
        "teacher_notes": plan.teacher_notes,
        "problems": [
            {
                "identifier": spec.identifier,
                "prompt_stub": spec.prompt_stub,
                "answer_plan": spec.answer_plan,
                "metadata": spec.metadata,
                # Include diagram spec if present (for problem display)
                "diagram_spec": {
                    "diagram_type": spec.diagram_spec.diagram_type,
                    "params": spec.diagram_spec.params,
                    "width": spec.diagram_spec.width,
                    "height": spec.diagram_spec.height,
                    "show_labels": spec.diagram_spec.show_labels,
                    "show_measurements": spec.diagram_spec.show_measurements,
                    "title": spec.diagram_spec.title,
                } if spec.diagram_spec else None,
                # Include answer diagram spec if present (for answer key)
                "answer_diagram_spec": {
                    "diagram_type": spec.answer_diagram_spec.diagram_type,
                    "params": spec.answer_diagram_spec.params,
                    "width": spec.answer_diagram_spec.width,
                    "height": spec.answer_diagram_spec.height,
                    "show_labels": spec.answer_diagram_spec.show_labels,
                    "show_measurements": spec.answer_diagram_spec.show_measurements,
                    "title": spec.answer_diagram_spec.title,
                } if spec.answer_diagram_spec else None,
            }
            for spec in plan.problems
        ],
    }
    return serializable


if __name__ == "__main__":
    sample = run_generate_problems(
        "decimals word problems with money context",
        grade=5,
        num_problems=3,
    )
    print(sample["concept_text"])
    for problem in sample["problems"]:
        print(problem["identifier"], "->", problem["prompt_stub"])

