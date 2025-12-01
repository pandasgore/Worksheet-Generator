
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from schemas.worksheet import (
    WorksheetRequest,
    WorksheetPlanPayload,
    ProblemTemplatePayload,
    WorksheetDocumentPayload,
    WorksheetDocumentProblem,
    DocxBuildRequest,
    PdfBuildRequest,
    BuildArtifactResponse,
)
from tools.generate_problems_tool import run_generate_problems
from tools.build_docx_tool import build_docx_from_payload
from tools.build_pdf_tool import build_pdf_from_payload


def _to_plan_payload(raw: dict) -> WorksheetPlanPayload:
    problems = [
        ProblemTemplatePayload(**problem_dict) for problem_dict in raw["problems"]
    ]
    return WorksheetPlanPayload(
        concept_text=raw["concept_text"],
        grade=raw["grade"],
        difficulty_plan=raw["difficulty_plan"],
        concept_analysis=raw["concept_analysis"],
        teacher_notes=raw.get("teacher_notes", []),
        problems=problems,
    )


class WorksheetToolset(BaseToolset):
    """ADK toolset exposing the worksheet pipeline to the LLM agent."""

    def __init__(self, *, artifact_dir: Path):
        super().__init__()
        self._artifact_dir = artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._latest_artifacts: dict[str, dict] = {}
        self._latest_plan: Optional[WorksheetPlanPayload] = None
        WorksheetToolset._SELF_REF = self

    def clear_artifacts(self) -> None:
        """Resets the artifact tracking for a new run."""
        self._latest_artifacts.clear()
        self._latest_plan = None

    async def get_tools(self, readonly_context=None):
        # Explicitly return named tools without the 'name' argument in constructor
        return [
            FunctionTool(self.generate_plan),
            FunctionTool(self.build_docx),
            FunctionTool(self.build_pdf),
            FunctionTool(self.build_docx_problems),
            FunctionTool(self.build_pdf_problems),
            FunctionTool(self.BuildDocxProblems),
            FunctionTool(self.BuildPdfProblems),
            # Extra alias for occasional hallucinated name
            FunctionTool(self.BuildDocxProblemsProblems),
            FunctionTool(self.generate_and_build_docx),
            FunctionTool(self.generate_and_build_pdf),
        ]

    def generate_plan(
        self,
        concept_text: str,
        grade: int,
        num_problems: int = 5,
        target_difficulty: str = "medium",
    ) -> dict:
        """Create a structured worksheet plan for the requested concept."""
        print(f"[DEBUG] generate_plan called for concept: {concept_text}, grade: {grade}, difficulty: {target_difficulty}")

        try:
            # Coerce difficulty to allowed values to prevent Pydantic validation errors
            difficulty_map = {
                "simple": "easy",
                "basic": "easy",
                "beginner": "easy",
                "advanced": "hard",
                "complex": "hard",
            }
            cleaned_difficulty = target_difficulty.lower().strip()
            cleaned_difficulty = difficulty_map.get(cleaned_difficulty, cleaned_difficulty)
            
            if cleaned_difficulty not in {"easy", "medium", "hard"}:
                cleaned_difficulty = "medium"

            request = WorksheetRequest(
                concept_text=concept_text,
                grade=grade,
                num_problems=num_problems,
                target_difficulty=cleaned_difficulty,
            )
            plan_dict = run_generate_problems(
                concept_text=request.concept_text,
                grade=request.grade,
                num_problems=request.num_problems,
                target_difficulty=request.target_difficulty,
            )
            plan_payload = _to_plan_payload(plan_dict)
            # Store plan directly on self (no longer using fragile _SELF_REF)
            self._latest_plan = plan_payload
            print(f"[DEBUG] generate_plan complete: {len(plan_payload.problems)} problem templates created")
            return plan_payload.model_dump()
        except Exception as e:
            print(f"[ERROR] generate_plan failed: {type(e).__name__}: {e}")
            # Return a minimal error response so the agent knows something went wrong
            return {
                "error": str(e),
                "concept_text": concept_text,
                "grade": grade,
                "problems": [],
                "difficulty_plan": {"difficulty_level": target_difficulty},
                "concept_analysis": {"error": "Plan generation failed"},
                "teacher_notes": [f"Error during plan generation: {e}"],
            }

    def build_docx(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.docx",
        include_answer_key: bool = True,
    ) -> dict:
        """
        Render the provided worksheet data into DOCX artifacts (student and teacher versions).
        """
        print(f"[DEBUG] build_docx called with {len(problems) if problems else 0} problems")

        try:
            payload = self._build_payload_from_inputs(
                problems=problems,
                concept=concept,
                grade=grade,
                difficulty=difficulty,
                title=title,
                num_problems=num_problems,
            )
            request = DocxBuildRequest(
                worksheet=payload,
                output_path=output_path,
                include_answer_key=include_answer_key,
            )
            paths = self._render_docx(request)
            artifact = {
                "artifact_type": "docx",
                "artifact_path": paths["teacher"],  # Default path for backward compatibility
                "teacher_path": paths["teacher"],
                "student_path": paths["student"],
            }
            self._latest_artifacts["docx"] = artifact
            print(f"[DEBUG] DOCX built: teacher={paths['teacher']}, student={paths['student']}")
            return artifact
        except Exception as e:
            print(f"[ERROR] build_docx failed: {type(e).__name__}: {e}")
            return {"error": str(e), "artifact_type": "docx", "artifact_path": None}

    def build_pdf(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.pdf",
        include_answer_key: bool = True,
    ) -> dict:
        """Render the provided worksheet data into PDF artifacts (student and teacher versions)."""
        print(f"[DEBUG] build_pdf called with {len(problems) if problems else 0} problems")
        
        try:
            payload = self._build_payload_from_inputs(
                problems=problems,
                concept=concept,
                grade=grade,
                difficulty=difficulty,
                title=title,
                num_problems=num_problems,
            )
            request = PdfBuildRequest(
                worksheet=payload,
                output_path=output_path,
                include_answer_key=include_answer_key,
            )
            paths = self._render_pdf(request)
            artifact = {
                "artifact_type": "pdf",
                "artifact_path": paths["teacher"],  # Default path for backward compatibility
                "teacher_path": paths["teacher"],
                "student_path": paths["student"],
            }
            self._latest_artifacts["pdf"] = artifact
            print(f"[DEBUG] PDF built: teacher={paths['teacher']}, student={paths['student']}")
            return artifact
        except Exception as e:
            print(f"[ERROR] build_pdf failed: {type(e).__name__}: {e}")
            return {"error": str(e), "artifact_type": "pdf", "artifact_path": None}

    # Aliases for LLMs that hallucinate alternate tool names -----------------

    def build_docx_problems(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.docx",
        include_answer_key: bool = True,
    ) -> dict:
        return self.build_docx(
            problems=problems,
            concept=concept,
            grade=grade,
            difficulty=difficulty,
            title=title,
            num_problems=num_problems,
            output_path=output_path,
            include_answer_key=include_answer_key,
        )

    def build_pdf_problems(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.pdf",
        include_answer_key: bool = True,
    ) -> dict:
        return self.build_pdf(
            problems=problems,
            concept=concept,
            grade=grade,
            difficulty=difficulty,
            title=title,
            num_problems=num_problems,
            output_path=output_path,
            include_answer_key=include_answer_key,
        )

    # Exact CamelCase entry points for Gemini hallucinations ---------------

    def BuildDocxProblems(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.docx",
        include_answer_key: bool = True,
    ) -> dict:
        return self.build_docx_problems(
            problems=problems,
            concept=concept,
            grade=grade,
            difficulty=difficulty,
            title=title,
            num_problems=num_problems,
            output_path=output_path,
            include_answer_key=include_answer_key,
        )

    # Some model variants occasionally hallucinate an extra 'Problems' suffix
    # in the tool name, e.g. `BuildDocxProblemsProblems`. Provide a thin alias
    # so those calls still resolve correctly.
    def BuildDocxProblemsProblems(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.docx",
        include_answer_key: bool = True,
    ) -> dict:
        return self.build_docx_problems(
            problems=problems,
            concept=concept,
            grade=grade,
            difficulty=difficulty,
            title=title,
            num_problems=num_problems,
            output_path=output_path,
            include_answer_key=include_answer_key,
        )

    def BuildPdfProblems(
        self,
        problems: List[WorksheetDocumentProblem],
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        title: Optional[str] = None,
        num_problems: Optional[int] = None,
        output_path: str = "worksheet.pdf",
        include_answer_key: bool = True,
    ) -> dict:
        return self.build_pdf_problems(
            problems=problems,
            concept=concept,
            grade=grade,
            difficulty=difficulty,
            title=title,
            num_problems=num_problems,
            output_path=output_path,
            include_answer_key=include_answer_key,
        )

    # Legacy one-shot helpers -------------------------------------------------

    def generate_and_build_docx(
        self,
        concept_text: str,
        *,
        grade: int,
        num_problems: int = 5,
        target_difficulty: str = "medium",
        output_path: str = "worksheet.docx",
    ) -> dict:
        # Use instance method instead of static call
        plan_dict = self.generate_plan(
            concept_text=concept_text,
            grade=grade,
            num_problems=num_problems,
            target_difficulty=target_difficulty,
        )
        self._latest_plan = WorksheetPlanPayload(**plan_dict)
        placeholder_problems = [
            {
                "prompt": spec.prompt_stub,
                "answer": "",
                "solution_steps": [],
            }
            for spec in self._latest_plan.problems
        ]
        return self.build_docx(
            problems=placeholder_problems,
            concept=self._latest_plan.concept_text,
            grade=self._latest_plan.grade,
            difficulty=self._latest_plan.difficulty_plan.get(
                "difficulty_level", "mixed"
            ),
            title=f"Grade {self._latest_plan.grade} Worksheet (Auto Placeholder)",
            num_problems=len(placeholder_problems),
            output_path=output_path,
        )

    def generate_and_build_pdf(
        self,
        concept_text: str,
        *,
        grade: int,
        num_problems: int = 5,
        target_difficulty: str = "medium",
        output_path: str = "worksheet.pdf",
    ) -> dict:
        docx_info = self.generate_and_build_docx(
            concept_text=concept_text,
            grade=grade,
            num_problems=num_problems,
            target_difficulty=target_difficulty,
            output_path="__tmp_auto.docx",
        )
        placeholder_problems = [
            {
                "prompt": spec.prompt_stub,
                "answer": "",
                "solution_steps": [],
            }
            for spec in self._latest_plan.problems
        ]
        pdf = self.build_pdf(
            problems=placeholder_problems,
            concept=self._latest_plan.concept_text,
            grade=self._latest_plan.grade,
            difficulty=self._latest_plan.difficulty_plan.get(
                "difficulty_level", "mixed"
            ),
            title=f"Grade {self._latest_plan.grade} Worksheet (Auto Placeholder)",
            num_problems=len(placeholder_problems),
            output_path=output_path,
            
        )
        return {"docx": docx_info, "pdf": pdf}

    def _render_docx(self, request: DocxBuildRequest) -> dict:
        """Render both student and teacher DOCX versions."""
        # Teacher version (with answer key)
        teacher_path = self._next_artifact_path(request.output_path.replace('.docx', '_teacher.docx'))
        build_docx_from_payload(
            worksheet_data=request.worksheet.model_dump(),
            output_path=str(teacher_path),
            include_answer_key=True,
        )
        
        # Student version (without answer key)
        student_path = self._next_artifact_path(request.output_path.replace('.docx', '_student.docx'))
        build_docx_from_payload(
            worksheet_data=request.worksheet.model_dump(),
            output_path=str(student_path),
            include_answer_key=False,
        )
        
        return {"teacher": str(teacher_path), "student": str(student_path)}

    def _render_pdf(self, request: PdfBuildRequest) -> dict:
        """Render both student and teacher PDF versions."""
        # Teacher version (with answer key)
        teacher_path = self._next_artifact_path(request.output_path.replace('.pdf', '_teacher.pdf'))
        build_pdf_from_payload(
            worksheet_data=request.worksheet.model_dump(),
            output_path=str(teacher_path),
            include_answer_key=True,
        )
        
        # Student version (without answer key)
        student_path = self._next_artifact_path(request.output_path.replace('.pdf', '_student.pdf'))
        build_pdf_from_payload(
            worksheet_data=request.worksheet.model_dump(),
            output_path=str(student_path),
            include_answer_key=False,
        )
        
        return {"teacher": str(teacher_path), "student": str(student_path)}

    def _next_artifact_path(self, filename: str) -> Path:
        base = Path(filename)
        suffix = base.suffix or ".docx"
        stem = base.stem or "worksheet"

        if base.is_absolute():
            target_dir = base.parent
        else:
            parent = base.parent
            if parent and str(parent) not in ("", "."):
                target_dir = (self._artifact_dir / parent).resolve()
            else:
                target_dir = self._artifact_dir

        target_dir.mkdir(parents=True, exist_ok=True)

        candidate = target_dir / (stem + suffix)
        counter = 1
        while candidate.exists():
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate

    def latest_artifacts(self) -> dict[str, dict]:
        return {key: value.copy() for key, value in self._latest_artifacts.items()}

    def latest_plan(self) -> Optional[WorksheetPlanPayload]:
        """Expose the most recent plan so callers (e.g. runtime/UI) can summarize."""
        return self._latest_plan

    def auto_build_from_response_text(self, text: str) -> bool:
        """
        Attempts to parse a natural language agent response for structured problems
        and, if successful, builds any missing artifacts automatically.
        Falls back to generating placeholder artifacts from the plan if parsing fails
        but a plan exists.
        """
        print(f"[DEBUG] auto_build_from_response_text called, text length: {len(text) if text else 0}")

        if not self._latest_plan:
            print("[DEBUG] No plan available for auto-build. This usually means generate_plan was never called or failed.")
            return False

        print(f"[DEBUG] Plan exists with {len(self._latest_plan.problems)} problem templates")
        
        problems = self._parse_problems_from_text(text) if text else []
        print(f"[DEBUG] Parsed {len(problems)} problems from response text")
        
        # If no problems parsed from text, check if we can fallback to plan stubs
        if not problems:
            from schemas.worksheet import DiagramSpecPayload
            
            print("[DEBUG] No problems parsed from text. Using plan stubs as fallback.")
            fallback_problems = []
            for spec in self._latest_plan.problems:
                # Convert diagram spec if present (for problem display)
                # Handle both DiagramSpecPayload objects and dicts
                diagram = None
                if spec.diagram_spec:
                    try:
                        if isinstance(spec.diagram_spec, DiagramSpecPayload):
                            diagram = spec.diagram_spec
                        elif isinstance(spec.diagram_spec, dict):
                            diagram = DiagramSpecPayload(**spec.diagram_spec)
                        else:
                            # Object with model_dump method (Pydantic)
                            diagram = DiagramSpecPayload(**spec.diagram_spec.model_dump())
                    except Exception as e:
                        print(f"[DEBUG] Could not convert diagram_spec: {e}")
                
                # Convert answer diagram spec if present (for answer key)
                answer_diagram = None
                answer_diagram_spec = getattr(spec, 'answer_diagram_spec', None)
                if answer_diagram_spec:
                    try:
                        if isinstance(answer_diagram_spec, DiagramSpecPayload):
                            answer_diagram = answer_diagram_spec
                        elif isinstance(answer_diagram_spec, dict):
                            answer_diagram = DiagramSpecPayload(**answer_diagram_spec)
                        else:
                            # Object with model_dump method (Pydantic)
                            answer_diagram = DiagramSpecPayload(**answer_diagram_spec.model_dump())
                    except Exception as e:
                        print(f"[DEBUG] Could not convert answer_diagram_spec: {e}")
                
                fallback_problems.append(
                    WorksheetDocumentProblem(
                        prompt=spec.prompt_stub,
                        answer="(Generated by AI Plan - Answer Key Pending)",
                        solution_steps=[],
                        diagram=diagram,
                        answer_diagram=answer_diagram,
                    )
                )
            problems = fallback_problems
            print(f"[DEBUG] Created {len(problems)} fallback problems from plan stubs")

        if not problems:
            print("[DEBUG] No problems available (parsed or plan). Auto-build aborted.")
            return False

        built = False
        metadata = {
            "concept": self._latest_plan.concept_text,
            "grade": self._latest_plan.grade,
            "difficulty": self._latest_plan.difficulty_plan.get(
                "difficulty_level", "mixed"
            ),
            "title": f"Grade {self._latest_plan.grade} Worksheet",
            "num_problems": len(problems),
        }
        print(f"[DEBUG] Auto-build metadata: {metadata}")

        try:
            if "docx" not in self._latest_artifacts:
                print("[DEBUG] Auto-building DOCX...")
                self.build_docx(problems=problems, **metadata)
                built = True
            if "pdf" not in self._latest_artifacts:
                print("[DEBUG] Auto-building PDF...")
                self.build_pdf(problems=problems, **metadata)
                built = True
        except Exception as e:
            print(f"[ERROR] Auto-build failed with exception: {type(e).__name__}: {e}")
            return False
            
        print(f"[DEBUG] Auto-build completed, built={built}")
        return built

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    _SELF_REF: Optional["WorksheetToolset"] = None

    @classmethod
    def _get_self(cls) -> Optional["WorksheetToolset"]:
        return cls._SELF_REF

    _PROBLEM_REGEX = re.compile(
        r"(Problem\s+\d+\s*:.*?)(?=\n\s*Problem\s+\d+\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _build_payload_from_inputs(
        self,
        *,
        problems: List[Any],
        concept: Optional[str],
        grade: Optional[int],
        difficulty: Optional[str],
        title: Optional[str],
        num_problems: Optional[int],
    ) -> WorksheetDocumentPayload:
        from schemas.worksheet import DiagramSpecPayload
        
        if not problems:
            raise ValueError("Provide at least one problem when building artifacts.")

        plan = self._latest_plan

        resolved_grade = grade or (plan.grade if plan else None)
        resolved_concept = concept or (plan.concept_text if plan else None)
        resolved_difficulty = (difficulty or (plan.difficulty_plan.get("difficulty_level") if plan else None) or "mixed").lower()
        if resolved_difficulty not in {"easy", "medium", "hard", "mixed"}:
            resolved_difficulty = "mixed"

        if resolved_grade is None or resolved_concept is None:
            raise ValueError(
                "grade and concept are required (either via arguments or the last plan)."
            )

        coerced_problems = [
            self._coerce_problem_input(raw, idx)
            for idx, raw in enumerate(problems, start=1)
        ]
        
        # Merge answer_diagram_spec from the plan into coerced problems
        # This ensures answer diagrams are included even when LLM doesn't specify them
        if plan and plan.problems:
            for idx, problem in enumerate(coerced_problems):
                if idx < len(plan.problems):
                    plan_problem = plan.problems[idx]
                    # Add answer_diagram from plan if problem doesn't have one
                    if not problem.answer_diagram:
                        answer_diagram_spec = getattr(plan_problem, 'answer_diagram_spec', None)
                        if answer_diagram_spec:
                            try:
                                if isinstance(answer_diagram_spec, DiagramSpecPayload):
                                    problem.answer_diagram = answer_diagram_spec
                                elif isinstance(answer_diagram_spec, dict):
                                    problem.answer_diagram = DiagramSpecPayload(**answer_diagram_spec)
                                else:
                                    problem.answer_diagram = DiagramSpecPayload(**answer_diagram_spec.model_dump())
                            except Exception:
                                pass
                    # Also add regular diagram from plan if problem doesn't have one
                    if not problem.diagram:
                        diagram_spec = getattr(plan_problem, 'diagram_spec', None)
                        if diagram_spec:
                            try:
                                if isinstance(diagram_spec, DiagramSpecPayload):
                                    problem.diagram = diagram_spec
                                elif isinstance(diagram_spec, dict):
                                    problem.diagram = DiagramSpecPayload(**diagram_spec)
                                else:
                                    problem.diagram = DiagramSpecPayload(**diagram_spec.model_dump())
                            except Exception:
                                pass
        resolved_num = num_problems or len(coerced_problems)
        resolved_title = title or f"Grade {resolved_grade} Worksheet"

        return WorksheetDocumentPayload(
            title=resolved_title,
            concept=resolved_concept,
            grade=resolved_grade,
            difficulty=resolved_difficulty,
            num_problems=resolved_num,
            problems=coerced_problems,
            metadata={"source": "agent_generated"},
        )

    @staticmethod
    def _coerce_problem_input(raw: Any, idx: int):
        from schemas.worksheet import DiagramSpecPayload
        
        if isinstance(raw, WorksheetDocumentProblem):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(
                f"Problem #{idx} must be an object with prompt/answer fields."
            )
        prompt = raw.get("prompt") or raw.get("question")
        if not prompt:
            raise ValueError(f"Problem #{idx} is missing a 'prompt' string.")
        answer = raw.get("answer", "")
        steps = raw.get("solution_steps") or raw.get("steps") or []
        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(steps, list):
            raise ValueError(f"Problem #{idx} solution_steps must be a list.")
        
        # Handle diagram spec if present (for problem display)
        diagram = None
        diagram_spec = raw.get("diagram") or raw.get("diagram_spec")
        diagram_data = raw.get("diagram_data")  # LLM-provided data
        
        if diagram_spec and isinstance(diagram_spec, dict):
            try:
                diagram = DiagramSpecPayload(**diagram_spec)
            except Exception:
                pass
        elif diagram_data and isinstance(diagram_data, dict):
            # Convert LLM diagram_data to DiagramSpecPayload
            # Detect diagram type from the data keys
            if "base" in diagram_data and "height" in diagram_data:
                diagram_type = "triangle"
            elif "width" in diagram_data or "length" in diagram_data:
                diagram_type = "rectangle"
            elif "radius" in diagram_data:
                diagram_type = "circle"
            elif "numerator" in diagram_data and "denominator" in diagram_data:
                diagram_type = "fraction_circle"
                diagram_data = diagram_data  # Already in correct format
            else:
                # Assume bar graph for named keys with numeric values
                diagram_type = "bar_graph"
                diagram_data = {"data": diagram_data}  # Wrap for bar graph format
            
            try:
                diagram = DiagramSpecPayload(
                    diagram_type=diagram_type,
                    params=diagram_data
                )
            except Exception:
                pass
        
        # Handle answer diagram spec if present (for answer key)
        answer_diagram = None
        answer_diagram_data = raw.get("answer_diagram") or raw.get("answer_diagram_spec")
        if answer_diagram_data and isinstance(answer_diagram_data, dict):
            try:
                answer_diagram = DiagramSpecPayload(**answer_diagram_data)
            except Exception:
                pass
        
        return WorksheetDocumentProblem(
            prompt=prompt,
            answer=str(answer) if answer is not None else "",
            solution_steps=[str(step) for step in steps],
            diagram=diagram,
            answer_diagram=answer_diagram,
        )

    def _parse_problems_from_text(self, text: str) -> List[WorksheetDocumentProblem]:
        problems: List[WorksheetDocumentProblem] = []
        if not text:
            return problems

        for match in self._PROBLEM_REGEX.finditer(text):
            block = match.group(0).strip()
            prompt_body = re.sub(
                r"^Problem\s+\d+\s*:\s*",
                "",
                block,
                flags=re.IGNORECASE,
            ).strip()
            if not prompt_body:
                continue

            answer_match = re.search(
                r"Answer\s*:?(.*)",
                prompt_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if answer_match:
                prompt_text = prompt_body[: answer_match.start()].strip()
                answer_section = answer_match.group(1).strip()
            else:
                prompt_text = prompt_body.strip()
                answer_section = ""

            steps_match = re.search(
                r"(Solution Steps?|Steps?)\s*:?(.*)",
                answer_section,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if steps_match:
                answer_text = answer_section[: steps_match.start()].strip()
                steps_text = steps_match.group(2).strip()
            else:
                answer_text = answer_section.strip()
                steps_text = ""

            steps: List[str] = []
            if steps_text:
                for line in re.split(r"[\n\r]+", steps_text):
                    clean = line.strip(" -*\t")
                    if clean:
                        steps.append(clean)

            prompt_text = prompt_text or block
            problems.append(
                WorksheetDocumentProblem(
                    prompt=prompt_text,
                    answer=answer_text,
                    solution_steps=steps,
                )
            )

        return problems
