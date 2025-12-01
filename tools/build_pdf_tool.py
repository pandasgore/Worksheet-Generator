"""
build_pdf_tool.py

Utility wrapper around `PdfBuilder`. Accepts fully populated worksheet data and
returns the generated artifact path.
"""

from __future__ import annotations

from pathlib import Path

from formatting.pdf_builder import PdfBuilder
from schemas.worksheet import WorksheetDocumentPayload


def build_pdf_from_payload(
    worksheet_data: dict,
    output_path: str | Path,
    *,
    include_answer_key: bool = True,
) -> Path:
    payload = WorksheetDocumentPayload(**worksheet_data)
    builder = PdfBuilder()
    return builder.build(payload, output_path, include_answer_key=include_answer_key)


if __name__ == "__main__":
    sample = {
        "concept": "Fractions",
        "grade": 4,
        "difficulty": "mixed",
        "num_problems": 2,
        "problems": [
            {"prompt": "Shade 3/4 of a rectangle divided into 4 equal parts.", "answer": "3 parts shaded"},
            {"prompt": "What is 2/3 of 12?", "answer": "8"},
        ],
    }
    output = build_pdf_from_payload(sample, "sample_output.pdf")
    print(f"Wrote {output}")
