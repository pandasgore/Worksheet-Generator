
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

from flask import abort, Flask, render_template, request, send_from_directory, url_for

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.agent import WorksheetAgentRuntime
from core.grade_progression import GRADE_PROGRESSION

ARTIFACT_DIR = (Path(__file__).resolve().parent.parent / "artifacts").resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

runtime = WorksheetAgentRuntime(artifacts_dir=ARTIFACT_DIR)
app = Flask(__name__)


def get_topics_map():
    """
    Flattens the nested GRADE_PROGRESSION into a simple dictionary:
    { grade (int): [topic1, topic2, ...] }
    """
    topics_map = {}
    for grade, info in GRADE_PROGRESSION.items():
        grade_topics = []
        # Iterate over all keys that aren't metadata
        for key, value in info.items():
            if key in ["description", "reference"]:
                continue
            if isinstance(value, dict):
                # e.g. "numbers": {"range": ...} - maybe skip or extract keys?
                # For now, we primarily want the lists of skills
                pass
            elif isinstance(value, list):
                # Flatten the list of skill strings
                # Replace underscores with spaces for display
                formatted_skills = [skill.replace("_", " ") for skill in value]
                grade_topics.extend(formatted_skills)
        
        # Sort and deduplicate
        topics_map[grade] = sorted(list(set(grade_topics)))
    return topics_map


@app.route("/", methods=["GET"])
def index():
    topics_map = get_topics_map()
    return render_template("index.html", result=None, topics_map=topics_map)


@app.route("/generate", methods=["POST"])
def generate():
    topic = request.form.get("topic", "").strip()
    grade = int(request.form.get("grade") or 1)
    difficulty = request.form.get("difficulty") or "medium"
    num_problems = int(request.form.get("num_problems") or 5)

    topics_map = get_topics_map()

    if not topic:
        return render_template(
            "index.html",
            result={
                "prompt": "",
                "responses": ["Please enter or select a topic before generating a worksheet."],
                "docx_url": None,
                "pdf_url": None,
                "docx_info": None,
                "pdf_info": None,
            },
            topics_map=topics_map
        )

    prompt = build_prompt(topic, grade, difficulty, num_problems)
    responses = run_agent_workflow(prompt)
    artifacts = runtime.latest_artifacts()
    docx_info = artifacts.get("docx")
    pdf_info = artifacts.get("pdf")

    # DOCX URLs (student and teacher versions)
    docx_student_url = None
    docx_teacher_url = None
    if docx_info:
        if docx_info.get("student_path"):
            student_path = Path(docx_info["student_path"])
            if student_path.exists():
                docx_student_url = url_for("download_artifact", filename=student_path.name)
        if docx_info.get("teacher_path"):
            teacher_path = Path(docx_info["teacher_path"])
            if teacher_path.exists():
                docx_teacher_url = url_for("download_artifact", filename=teacher_path.name)

    # PDF URLs (student and teacher versions)
    pdf_student_url = None
    pdf_teacher_url = None
    if pdf_info:
        if pdf_info.get("student_path"):
            student_path = Path(pdf_info["student_path"])
            if student_path.exists():
                pdf_student_url = url_for("download_artifact", filename=student_path.name)
        if pdf_info.get("teacher_path"):
            teacher_path = Path(pdf_info["teacher_path"])
            if teacher_path.exists():
                pdf_teacher_url = url_for("download_artifact", filename=teacher_path.name)

    result = {
        "prompt": prompt,
        "responses": responses,
        "docx_student_url": docx_student_url,
        "docx_teacher_url": docx_teacher_url,
        "pdf_student_url": pdf_student_url,
        "pdf_teacher_url": pdf_teacher_url,
        "docx_info": docx_info,
        "pdf_info": pdf_info,
    }
    return render_template("index.html", result=result, topics_map=topics_map)


@app.route("/artifacts/<path:filename>")
def download_artifact(filename: str):
    file_path = ARTIFACT_DIR / filename
    if not file_path.exists():
        abort(404)
    return send_from_directory(str(ARTIFACT_DIR), filename, as_attachment=True)


def build_prompt(topic: str, grade: int, difficulty: str, num_problems: int) -> str:
    return (
        f"Create a {difficulty} worksheet for grade {grade} students on the topic "
        f"{topic}. Include approximately {num_problems} problems."
    )


def run_agent_workflow(prompt: str):
    try:
        responses = runtime.run(prompt, user_id="web_teacher", session_id="web_session")
    except Exception as exc:
        responses = [f"Agent workflow failed: {exc}"]
    return responses


if __name__ == "__main__":
    app.run(debug=True, port=5000)
