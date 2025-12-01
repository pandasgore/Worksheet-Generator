from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


# =============================================================================
# DIAGRAM SCHEMAS
# =============================================================================

class DiagramSpecPayload(BaseModel):
    """Specification for a diagram to include with a problem."""
    diagram_type: str = Field(..., description="Type of diagram (e.g., 'right_triangle', 'circle', 'bar_graph')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the diagram")
    width: float = Field(3.0, description="Diagram width in inches")
    height: float = Field(2.5, description="Diagram height in inches")
    show_labels: bool = Field(True, description="Whether to show labels")
    show_measurements: bool = Field(True, description="Whether to show measurements")
    title: Optional[str] = Field(None, description="Optional title for the diagram")


# =============================================================================
# WORKSHEET SCHEMAS
# =============================================================================

class WorksheetRequest(BaseModel):
    """Defines the inputs the Worksheet toolchain expects from teachers."""

    concept_text: str = Field(..., description="Math concept or skill to cover.")
    grade: int = Field(..., ge=1, le=8, description="Target grade level (1-8).")
    num_problems: int = Field(
        5,
        ge=1,
        le=25,
        description="How many problems to include in the worksheet.",
    )
    target_difficulty: Literal["easy", "medium", "hard"] = Field(
        "medium", description="Overall rigor for the worksheet."
    )


class ProblemTemplatePayload(BaseModel):
    identifier: str
    prompt_stub: str
    answer_plan: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    diagram_spec: Optional[DiagramSpecPayload] = Field(None, description="Optional diagram for this problem")
    answer_diagram_spec: Optional[DiagramSpecPayload] = Field(None, description="Diagram to show in answer key")


class WorksheetPlanPayload(BaseModel):
    concept_text: str
    grade: int
    difficulty_plan: Dict[str, Any]
    concept_analysis: Dict[str, Any]
    teacher_notes: List[str] = Field(default_factory=list)
    problems: List[ProblemTemplatePayload]

    @property
    def num_problems(self) -> int:
        return len(self.problems)


class WorksheetDocumentProblem(BaseModel):
    prompt: str
    answer: str = ""
    solution_steps: List[str] = Field(default_factory=list)
    diagram: Optional[DiagramSpecPayload] = Field(None, description="Optional diagram shown with the problem")
    answer_diagram: Optional[DiagramSpecPayload] = Field(None, description="Diagram to show in answer key (for 'create a diagram' type problems)")


class WorksheetDocumentPayload(BaseModel):
    title: Optional[str] = None
    concept: str
    grade: int
    difficulty: Literal["easy", "medium", "hard", "mixed"]
    num_problems: int
    problems: List[WorksheetDocumentProblem]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocxBuildRequest(BaseModel):
    worksheet: WorksheetDocumentPayload
    output_path: str = "worksheet.docx"
    include_answer_key: bool = True


class PdfBuildRequest(BaseModel):
    worksheet: WorksheetDocumentPayload
    output_path: str = "worksheet.pdf"
    include_answer_key: bool = True


class BuildArtifactResponse(BaseModel):
    artifact_path: str
    artifact_type: Literal["docx", "pdf"]


