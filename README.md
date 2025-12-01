# 📝 AI Math Worksheet Generator

An AI-powered tool that generates complete math worksheets for grades 1-8. Teachers enter a topic, and the system creates problems, diagrams, and answer keys automatically.

## ✨ What It Does

| Feature | Description |
|---------|-------------|
| **Any Math Topic** | Fractions, geometry, algebra, statistics, and more |
| **Grades 1-8** | Age-appropriate difficulty scaling |
| **Auto Diagrams** | Triangles, graphs, charts generated automatically |
| **PDF & DOCX** | Download worksheets in either format |
| **Answer Keys** | Separate student and teacher versions |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
# Windows (Command Prompt):
set GOOGLE_API_KEY=your-gemini-api-key

# Windows (PowerShell):
$env:GOOGLE_API_KEY="your-gemini-api-key"

# Mac/Linux:
export GOOGLE_API_KEY="your-gemini-api-key"

# 3. Run the web app
python web/app.py

# 4. Open http://localhost:5000
```

---

## 🤖 ADK Features Used

This project demonstrates **Google's Agent Development Kit (ADK)**:

| Feature | Implementation |
|---------|----------------|
| **LlmAgent** | Gemini-powered worksheet orchestrator |
| **LoopAgent** | Retry logic with artifact verification |
| **BaseAgent** | Custom ArtifactCheckAgent for validation |
| **BaseToolset** | Custom tools: `generate_plan`, `build_pdf`, `build_docx` |
| **Sessions** | InMemorySessionService for state management |
| **Runner** | Async event-driven execution pipeline |

---

## 📁 Project Structure

```
worksheet-generator/
├── web/                    # Flask web interface
│   ├── app.py              # Server & API
│   └── templates/          # HTML frontend
│
├── agent/                  # ADK agent layer
│   ├── agent.py            # Runtime & event loop
│   ├── planner.py          # Agent configuration
│   └── tools.py            # WorksheetToolset
│
├── core/                   # Intelligence layer
│   ├── topic_interpreter.py
│   ├── problem_planner.py
│   └── grade_progression.py
│
├── formatting/             # Output generation
│   ├── pdf_builder.py      # PDF worksheets
│   ├── docx_builder.py     # Word documents
│   └── diagram_generator.py # Math diagrams
│
└── artifacts/              # Generated files
```

---

## 🔄 How It Works

```
Teacher Input → AI Agent → Generated Worksheet
     │              │              │
     ▼              ▼              ▼
  Grade: 6      Analyzes       Problems
  Topic         Generates      Diagrams
  Difficulty    Validates      Answer Key
```

**Step by step:**
1. Teacher enters: grade, topic, difficulty, number of problems
2. ADK agent calls `generate_plan` to analyze the topic
3. Agent creates problems with `build_problems`
4. Agent generates PDF/DOCX with `build_pdf` / `build_docx`
5. Teacher downloads the worksheet

---

## 📊 Supported Diagrams

| Category | Types |
|----------|-------|
| **Geometry** | Triangles, rectangles, circles, 3D shapes |
| **Graphs** | Bar graphs, line graphs, pie charts |
| **Fractions** | Fraction circles, fraction bars |
| **Statistics** | Dot plots, box plots, histograms |

---

## 🔧 Tech Stack

- **AI Agent**: Google ADK + Gemini 
- **Web**: Flask
- **PDF**: ReportLab
- **DOCX**: python-docx
- **Diagrams**: Matplotlib
- **Validation**: Pydantic

---

## 📓 Kaggle Demo

See `kaggle_demo.py` for a single-file demonstration of the entire system that runs in a Kaggle notebook.

---

## 📜 License

MIT License
