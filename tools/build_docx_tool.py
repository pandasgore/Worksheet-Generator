"""
build_docx_tool.py

Wraps the DOCX builder so the ADK agent can provide fully formed worksheet data
and receive an artifact path in return.
"""

from __future__ import annotations

from pathlib import Path

from formatting.docx_builder import build_docx
from schemas.worksheet import WorksheetDocumentPayload


def build_docx_from_payload(
    worksheet_data: dict,
    output_path: str | Path,
    *,
    include_answer_key: bool = True,
) -> Path:
    payload = WorksheetDocumentPayload(**worksheet_data)
    return build_docx(payload, output_path, include_answer_key=include_answer_key)


if __name__ == "__main__":
    sample = {
        "concept": "Ratios",
        "grade": 6,
        "difficulty": "medium",
        "num_problems": 2,
        "problems": [
            {"prompt": "Write the ratio of 8 apples to 12 oranges in simplest form.", "answer": "2:3"},
            {"prompt": "Simplify the ratio 9:6.", "answer": "3:2"},
        ],
    }
    output = build_docx_from_payload(sample, "sample_output.docx")
    print(f"Wrote {output}")
