"""
universal_generator.py

Produces structured problem templates that the ADK agent can expand into full
worksheets. This module intentionally avoids rendering final problem text so
that the LLM-driven agent remains responsible for natural language generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.difficulty_scaler import LearnerProfile, default_profile
from core.problem_planner import ProblemBlueprint, ProblemPlanner


@dataclass
class DiagramSpec:
    """Specification for a diagram to accompany a problem."""
    diagram_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    width: float = 3.0
    height: float = 2.5
    show_labels: bool = True
    show_measurements: bool = True
    title: Optional[str] = None


@dataclass
class ProblemSpec:
    """Template describing the intent of a single problem."""

    identifier: str
    prompt_stub: str
    answer_plan: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagram_spec: Optional[DiagramSpec] = None  # Diagram to show with the problem
    answer_diagram_spec: Optional[DiagramSpec] = None  # Diagram to show in answer key


@dataclass
class WorksheetPlan:
    concept_text: str
    grade: int
    problems: List[ProblemSpec]
    difficulty_plan: Dict[str, Any]
    concept_analysis: Dict[str, Any]
    teacher_notes: List[str] = field(default_factory=list)


class UniversalGenerator:
    """
    Facade for topic interpretation, difficulty scaling, and blueprint planning.
    Returns templates that downstream LLM agents can turn into complete content.
    """

    def __init__(self, grade: int):
        self.grade = grade
        self.planner = ProblemPlanner(grade)

    def generate(
        self,
        concept_text: str,
        *,
        num_problems: int = 5,
        profile: Optional[LearnerProfile] = None,
        target_difficulty: str = "medium",
    ) -> WorksheetPlan:
        profile = profile or default_profile(self.grade)
        plan = self.planner.plan(concept_text, profile, target_difficulty)
        blueprint: ProblemBlueprint = plan["blueprint"]
        problems = self._build_problem_specs(blueprint, num_problems)
        teacher_notes = self._default_teacher_notes(plan)

        return WorksheetPlan(
            concept_text=concept_text,
            grade=self.grade,
            problems=problems,
            difficulty_plan=plan["difficulty_plan"],
            concept_analysis=plan["concept_analysis"],
            teacher_notes=teacher_notes,
        )

    def _build_problem_specs(
        self,
        blueprint: ProblemBlueprint,
        count: int,
    ) -> List[ProblemSpec]:
        specs: List[ProblemSpec] = []
        diagram_types = getattr(blueprint, 'diagram_types', []) or []
        answer_diagram_types = getattr(blueprint, 'answer_diagram_types', []) or []
        
        for idx in range(count):
            identifier = f"{blueprint.domain}_{idx + 1}"
            prompt_stub = self._craft_prompt_stub(blueprint, idx)
            answer_plan = self._craft_answer_plan(blueprint)
            metadata = {
                "difficulty": blueprint.difficulty,
                "quantitative_targets": blueprint.quantitative_targets,
                "scaffolds": blueprint.scaffolds,
                "validation_checks": blueprint.validation_checks,
                "sequence_index": idx,
            }
            
            # Create diagram spec if diagram types are available (for problem display)
            diagram_spec = None
            if diagram_types:
                # Cycle through available diagram types if we have more problems than types
                diagram_type = diagram_types[idx % len(diagram_types)]
                diagram_spec = self._create_diagram_spec(diagram_type, blueprint, idx)
            
            # Create answer diagram spec if answer diagram types are available (for answer key)
            answer_diagram_spec = None
            if answer_diagram_types:
                # Cycle through available answer diagram types
                answer_diagram_type = answer_diagram_types[idx % len(answer_diagram_types)]
                answer_diagram_spec = self._create_diagram_spec(answer_diagram_type, blueprint, idx)
            
            specs.append(
                ProblemSpec(
                    identifier=identifier,
                    prompt_stub=prompt_stub,
                    answer_plan=answer_plan,
                    metadata=metadata,
                    diagram_spec=diagram_spec,
                    answer_diagram_spec=answer_diagram_spec,
                )
            )
        return specs
    
    def _create_diagram_spec(
        self,
        diagram_type: str,
        blueprint: ProblemBlueprint,
        idx: int,
    ) -> DiagramSpec:
        """Create a diagram specification with appropriate parameters for the problem."""
        params = {}
        
        # Get number range for scaling diagram values
        number_range = blueprint.quantitative_targets.get("number_range", [1, 10])
        min_val, max_val = number_range[0], number_range[1]
        
        # Generate varied parameters based on diagram type and problem index
        import random
        random.seed(idx * 42)  # Deterministic but varied
        
        if diagram_type == "right_triangle":
            base = random.randint(max(3, min_val), min(12, max_val))
            height = random.randint(max(3, min_val), min(12, max_val))
            params = {
                "base": base,
                "height": height,
                "show_right_angle": True,
                "labels": {"base": f"{base}", "height": f"{height}"},
            }
        elif diagram_type == "triangle":
            params = {
                "vertices": [(0, 0), (random.randint(4, 8), 0), (random.randint(1, 3), random.randint(3, 6))],
                "vertex_labels": ["A", "B", "C"],
            }
        elif diagram_type == "rectangle":
            width = random.randint(max(4, min_val), min(10, max_val))
            height = random.randint(max(2, min_val), min(8, max_val))
            params = {"width": width, "height": height}
        elif diagram_type == "square":
            side = random.randint(max(3, min_val), min(8, max_val))
            params = {"side": side}
        elif diagram_type == "circle":
            radius = random.randint(max(2, min_val), min(6, max_val))
            params = {"radius": radius, "show_radius": True, "show_diameter": idx % 2 == 0}
        elif diagram_type == "rectangular_prism":
            params = {
                "length": random.randint(4, 8),
                "width": random.randint(2, 5),
                "height": random.randint(3, 6),
            }
        elif diagram_type == "cube":
            params = {"side": random.randint(3, 6)}
        elif diagram_type == "cylinder":
            params = {"radius": random.randint(2, 4), "height": random.randint(4, 8)}
        elif diagram_type == "cone":
            params = {"radius": random.randint(2, 4), "height": random.randint(4, 8)}
        elif diagram_type == "pyramid":
            params = {"base": random.randint(4, 8), "height": random.randint(4, 8)}
        elif diagram_type == "angle":
            params = {"degrees": random.choice([30, 45, 60, 90, 120, 135])}
        elif diagram_type == "angle_pair":
            angle1 = random.randint(20, 70)
            params = {"angle1": angle1, "angle2": 90 - angle1, "type": "complementary"}
        elif diagram_type == "transversal":
            params = {"highlight_angles": [random.randint(1, 4), random.randint(5, 8)]}
        elif diagram_type == "plotted_points":
            num_points = random.randint(2, 4)
            points = [(random.randint(-5, 5), random.randint(-5, 5)) for _ in range(num_points)]
            labels = [chr(65 + i) for i in range(num_points)]
            params = {"points": points, "labels": labels}
        elif diagram_type == "fraction_circle":
            denominator = random.choice([2, 3, 4, 5, 6, 8])
            numerator = random.randint(1, denominator - 1)
            params = {"numerator": numerator, "denominator": denominator}
        elif diagram_type == "fraction_bar":
            denominator = random.choice([2, 3, 4, 5, 6, 8, 10])
            numerator = random.randint(1, denominator - 1)
            params = {"numerator": numerator, "denominator": denominator}
        elif diagram_type == "bar_graph":
            categories = ["A", "B", "C", "D", "E"][:random.randint(3, 5)]
            data = {cat: random.randint(2, 12) for cat in categories}
            params = {"data": data, "xlabel": "Category", "ylabel": "Count"}
        elif diagram_type == "dot_plot":
            data = [random.randint(1, 6) for _ in range(random.randint(10, 20))]
            params = {"data": data}
        elif diagram_type == "box_plot":
            base = random.randint(10, 30)
            data = sorted([base + random.randint(0, 40) for _ in range(15)])
            params = {"data": data}
        elif diagram_type == "double_number_line":
            ratio = random.randint(2, 5)
            top_values = [i * 2 for i in range(5)]
            bottom_values = [i * ratio for i in range(5)]
            params = {
                "top_values": top_values,
                "bottom_values": bottom_values,
                "top_label": "Miles",
                "bottom_label": "Hours",
            }
        elif diagram_type == "tape_diagram":
            parts = [random.randint(2, 5), random.randint(2, 5)]
            params = {"parts": parts, "labels": ["Part A", "Part B"]}
        
        return DiagramSpec(
            diagram_type=diagram_type,
            params=params,
            show_labels=True,
            show_measurements=True,
        )

    @staticmethod
    def _craft_prompt_stub(blueprint: ProblemBlueprint, idx: int) -> str:
        number_range = blueprint.quantitative_targets.get("number_range")
        range_text = (
            f"using numbers between {number_range[0]} and {number_range[1]}"
            if number_range
            else "using grade-appropriate numbers"
        )
        subskill_text = ", ".join(blueprint.subskills)
        scaffold_text = (
            f"Include scaffolds: {', '.join(blueprint.scaffolds)}."
            if blueprint.scaffolds
            else ""
        )
        return (
            f"Problem #{idx + 1}: craft a {blueprint.difficulty} {blueprint.domain} "
            f"question that reinforces {subskill_text} {range_text}. "
            f"{scaffold_text}".strip()
        )

    @staticmethod
    def _craft_answer_plan(blueprint: ProblemBlueprint) -> str:
        validations = ", ".join(blueprint.validation_checks)
        return (
            f"Ensure the solution demonstrates {', '.join(blueprint.subskills)}. "
            f"Apply validation steps: {validations}."
        )

    def _default_teacher_notes(self, plan: Dict[str, Any]) -> List[str]:
        blueprint: ProblemBlueprint = plan["blueprint"]
        difficulty_level = plan["difficulty_plan"]["difficulty_level"]
        notes = [
            f"Focus domain: {blueprint.domain}",
            f"Difficulty level resolved to '{difficulty_level}' "
            f"based on current learner profile.",
        ]
        if blueprint.scaffolds:
            notes.append("Suggested scaffolds: " + ", ".join(blueprint.scaffolds))
        return notes


if __name__ == "__main__":
    generator = UniversalGenerator(grade=6)
    plan = generator.generate("ratio word problems", num_problems=3)
    for spec in plan.problems:
        print(spec.identifier, "->", spec.prompt_stub)

