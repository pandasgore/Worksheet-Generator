from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from pydantic import BaseModel, Field


class TeacherProfile(BaseModel):
    """Structured representation of long-term teacher preferences."""

    name: str = "Unknown Teacher"
    preferred_grade: int = Field(ge=1, le=8, default=5)
    default_problem_count: int = Field(ge=1, le=25, default=5)
    include_teacher_notes: bool = False
    include_solution_guidance: bool = False


class WorksheetMemoryService(InMemoryMemoryService):
    """Bridges ADK memory with our persisted teacher profile file."""

    def __init__(self, memory_file: Path):
        super().__init__()
        self._memory_file = memory_file
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, Any] = self._read_file()
        self.teacher_profile = TeacherProfile(
            **self._store.get("teacher_profile", {})
        )

    def _read_file(self) -> Dict[str, Any]:
        if not self._memory_file.exists():
            return {"teacher_profile": TeacherProfile().model_dump(), "recent_requests": []}
        try:
            return json.loads(self._memory_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"teacher_profile": TeacherProfile().model_dump(), "recent_requests": []}

    def persist_teacher_profile(self) -> None:
        self._store["teacher_profile"] = self.teacher_profile.model_dump()
        self._memory_file.write_text(
            json.dumps(self._store, indent=2), encoding="utf-8"
        )

    def append_request(self, request: Dict[str, Any]) -> None:
        history = self._store.setdefault("recent_requests", [])
        history.append(request)
        self._store["recent_requests"] = history[-10:]
        self.persist_teacher_profile()


