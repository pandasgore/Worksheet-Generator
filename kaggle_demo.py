"""
================================================================================
AI MATH WORKSHEET GENERATOR - ADK CAPSTONE PROJECT
================================================================================

AI-powered math worksheet generator using Google's Agent Development Kit (ADK).

================================================================================
ADK FEATURES DEMONSTRATED (6 Key Concepts)
================================================================================

1. MULTI-AGENT SYSTEM
   ├── LlmAgent: "worksheet_orchestrator" - LLM-powered problem generation
   ├── LoopAgent: "worksheet_loop" - Iterative retry with max_iterations
   ├── BaseAgent: "ArtifactCheckAgent" - Custom non-LLM verification logic
   └── Sequential Agents: LlmAgent → ArtifactCheckAgent in pipeline

2. CUSTOM TOOLS (BaseToolset + FunctionTool)
   ├── generate_plan() - Topic analysis and problem planning
   ├── build_problems() - Convert JSON to Problem objects with diagrams
   └── build_pdf() - Generate PDF worksheet artifact

3. SESSIONS & STATE MANAGEMENT
   ├── InMemorySessionService - Session lifecycle management
   ├── create_session() - Initialize conversation state
   └── Session ID tracking across agent runs

4. RUNNER PIPELINE
   ├── Runner - Async execution of agent hierarchy
   ├── run_async() - Event stream processing
   └── Event handling for function calls/responses

5. AGENT ORCHESTRATION PATTERNS
   ├── Tool-calling workflow (generate → build → output)
   ├── Artifact verification before completion
   └── Escalation to stop LoopAgent when done

6. CUSTOM AGENT LOGIC (BaseAgent)
   ├── _run_async_impl() - Custom async generator
   ├── EventActions(escalate=True/False) - Control loop flow
   └── State inspection without LLM calls

================================================================================

Requirements: pip install google-adk reportlab matplotlib numpy pydantic
Kaggle Setup: Add GOOGLE_API_KEY to secrets, run cells in order
ADK Docs: https://google.github.io/adk-docs/
"""

# =============================================================================
# CELL 1: IMPORTS AND SETUP
# =============================================================================

# Install dependencies (for Kaggle notebook)
# !pip install google-adk reportlab matplotlib numpy pydantic

import os
import io
import re
import json
import random
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field



# For diagram generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge
import numpy as np

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


from google.adk.agents import LlmAgent, LoopAgent, BaseAgent
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.events import Event
from google.genai import types as genai_types

# For data validation
from pydantic import BaseModel, Field

print("✅ All imports successful (including Google ADK)!")

# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================

# Set your API key - works in Kaggle or local environment
# In Kaggle: Add to Secrets as GOOGLE_API_KEY
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("✅ API key loaded from Kaggle Secrets!")
except:
    API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    if API_KEY:
        print("✅ API key loaded from environment!")
    else:
        print("⚠️ No API key found - agent execution will fail")

# Set API key in environment for ADK
# ADK reads from GOOGLE_API_KEY environment variable
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

# Output directory for artifacts
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# CELL 3: DATA MODELS
# =============================================================================

@dataclass
class DiagramSpec:
    """Specification for a diagram."""
    diagram_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    width: float = 3.0
    height: float = 2.5

@dataclass
class Problem:
    """A single worksheet problem."""
    number: int
    question: str
    answer: str
    solution_steps: List[str] = field(default_factory=list)
    diagram: Optional[DiagramSpec] = None
    answer_diagram: Optional[DiagramSpec] = None

@dataclass
class Worksheet:
    """Complete worksheet data."""
    title: str
    grade: int
    difficulty: str
    concept: str
    problems: List[Problem]

print("✅ Data models defined!")

# =============================================================================
# CELL 4: GRADE PROGRESSION DATA
# =============================================================================

GRADE_INFO = {
    1: {"number_range": [0, 20], "operations": ["addition", "subtraction"]},
    2: {"number_range": [0, 100], "operations": ["addition", "subtraction", "skip counting"]},
    3: {"number_range": [0, 1000], "operations": ["multiplication", "division"]},
    4: {"number_range": [0, 10000], "operations": ["multi-digit operations", "fractions"]},
    5: {"number_range": [0, 100000], "operations": ["decimals", "fractions", "volume"]},
    6: {"number_range": [-100, 100], "operations": ["ratios", "integers", "expressions"]},
    7: {"number_range": [-1000, 1000], "operations": ["proportions", "equations", "geometry"]},
    8: {"number_range": [-10000, 10000], "operations": ["linear equations", "functions", "Pythagorean"]},
}

# Domain detection keywords
DOMAIN_KEYWORDS = {
    "geometry": ["triangle", "rectangle", "circle", "area", "perimeter", "volume", "angle"],
    "fractions": ["fraction", "numerator", "denominator", "mixed number"],
    "algebra": ["equation", "variable", "expression", "solve", "linear"],
    "statistics": ["mean", "median", "mode", "graph", "plot", "data", "probability"],
    "ratios": ["ratio", "proportion", "rate", "percent", "discount", "tax"],
    "measurement": ["measure", "length", "time", "convert", "elapsed"],
    "operations": ["add", "subtract", "multiply", "divide", "integer"],
}

# Diagram type mapping - Only for problems where diagrams help understanding
# NOT for computation problems where students solve it themselves
DIAGRAM_KEYWORDS = {
    "triangle": ["triangle", "area of triangle"],
    "right_triangle": ["right triangle", "pythagorean"],
    "rectangle": ["rectangle", "area of rectangle", "perimeter"],
    "circle": ["circle", "radius", "diameter", "circumference"],
    "bar_graph": ["bar graph", "bar chart", "interpreting bar"],
    "pie_chart": ["pie chart", "circle graph"],
    "box_plot": ["box plot", "box and whisker"],
    "dot_plot": ["dot plot"],
    "double_number_line": ["double number line", "number line"],
    # fraction_circle only for identification, NOT computation
    "fraction_circle": ["identify fraction", "what fraction", "shade fraction"],
}

# Topics where students compute the answer themselves - NO diagrams
NO_DIAGRAM_TOPICS = [
    "adding fraction", "add fraction", 
    "subtracting fraction", "subtract fraction",
    "multiplying fraction", "multiply fraction",
    "dividing fraction", "divide fraction",
    "comparing fraction", "compare fraction",
    "equivalent fraction",
    "simplify fraction", "simplifying fraction",
]

print("✅ Grade progression data loaded!")

# =============================================================================
# CELL 5: TOPIC INTERPRETER
# =============================================================================

def interpret_topic(concept: str, grade: int) -> Dict[str, Any]:
    """
    Analyze a math topic and determine appropriate parameters.
    
    Args:
        concept: The math concept/topic
        grade: Target grade level (1-8)
        
    Returns:
        Dictionary with domain, diagram types, number ranges, etc.
    """
    text = concept.lower()
    
    # Detect domain
    domain = "general"
    for dom, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            domain = dom
            break
    
    # Detect diagram type
    diagram_type = None
    answer_diagram_type = None
    
    # Skip diagrams for computation topics - students solve these themselves
    is_computation = any(topic in text for topic in NO_DIAGRAM_TOPICS)
    
    if not is_computation:
        # Check if this is a "create diagram" topic
        create_keywords = ["create", "make", "draw", "plot", "construct"]
        is_create_diagram = any(kw in text for kw in create_keywords)
        
        for diag_type, keywords in DIAGRAM_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                if is_create_diagram:
                    answer_diagram_type = diag_type  # Show in answer key
                else:
                    diagram_type = diag_type  # Show with problem
                break
    
    # Get grade-appropriate number range
    grade_data = GRADE_INFO.get(grade, GRADE_INFO[6])
    
    return {
        "concept": concept,
        "grade": grade,
        "domain": domain,
        "diagram_type": diagram_type,
        "answer_diagram_type": answer_diagram_type,
        "number_range": grade_data["number_range"],
        "operations": grade_data["operations"],
    }

# Test the interpreter
print("\n📚 Topic Interpretation Examples:")
for topic in ["area of triangles", "bar graphs", "solving equations"]:
    result = interpret_topic(topic, 6)
    print(f"  '{topic}' → domain={result['domain']}, diagram={result['diagram_type']}")

# =============================================================================
# CELL 6: DIAGRAM GENERATOR
# =============================================================================

class DiagramGenerator:
    """Generates mathematical diagrams as PNG images."""
    
    COLORS = {
        "primary": "#2563eb",
        "accent": "#f59e0b",
        "fill": "#dbeafe",
        "stroke": "#1e40af",
    }
    
    def generate(self, spec: DiagramSpec) -> bytes:
        """Generate a diagram and return PNG bytes."""
        fig, ax = plt.subplots(figsize=(spec.width, spec.height))
        
        # Route to appropriate drawing method
        method_name = f"_draw_{spec.diagram_type}"
        if hasattr(self, method_name):
            getattr(self, method_name)(ax, spec.params)
        else:
            self._draw_placeholder(ax, spec.diagram_type)
        
        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def _draw_triangle(self, ax, params):
        """Draw a triangle with dimensions - clean worksheet style."""
        base = params.get("base", 6)
        height = params.get("height", 4)
        
        # Draw isoceles triangle
        vertices = [(0, 0), (base, 0), (base/2, height)]
        triangle = Polygon(vertices, fill=True,
                          facecolor=self.COLORS["fill"],
                          edgecolor=self.COLORS["stroke"], linewidth=2.5)
        ax.add_patch(triangle)
        
        # Draw dashed height line from apex to base
        ax.plot([base/2, base/2], [0, height], '--', 
                color=self.COLORS["stroke"], linewidth=1.5, alpha=0.7)
        
        # Base label (below)
        ax.annotate(f"{base}", (base/2, -0.8), ha='center', fontsize=12,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=self.COLORS["accent"], linewidth=1.5))
        
        # Height label (to the side of dashed line)
        ax.annotate(f"{height}", (base/2 + 0.8, height/2), ha='left', fontsize=12,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=self.COLORS["accent"], linewidth=1.5))
        
        ax.set_xlim(-1.5, base + 2)
        ax.set_ylim(-1.5, height + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_right_triangle(self, ax, params):
        """Draw a right triangle with right angle marker."""
        base = params.get("base", 6)
        height = params.get("height", 4)
        
        vertices = [(0, 0), (base, 0), (0, height)]
        triangle = Polygon(vertices, fill=True,
                          facecolor=self.COLORS["fill"],
                          edgecolor=self.COLORS["stroke"], linewidth=2.5)
        ax.add_patch(triangle)
        
        # Right angle marker
        size = min(base, height) * 0.15
        ax.add_patch(Rectangle((0, 0), size, size, fill=False,
                               edgecolor=self.COLORS["stroke"], linewidth=1.5))
        
        # Labels with styled boxes
        ax.annotate(f"{base}", (base/2, -0.8), ha='center', fontsize=12,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=self.COLORS["accent"], linewidth=1.5))
        ax.annotate(f"{height}", (-1, height/2), ha='right', fontsize=12,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=self.COLORS["accent"], linewidth=1.5))
        
        ax.set_xlim(-2, base + 1.5)
        ax.set_ylim(-1.5, height + 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_rectangle(self, ax, params):
        """Draw a rectangle with dimensions."""
        width = params.get("width", 6)
        height = params.get("height", 4)
        
        rect = Rectangle((0, 0), width, height, fill=True,
                         facecolor=self.COLORS["fill"],
                         edgecolor=self.COLORS["stroke"], linewidth=2)
        ax.add_patch(rect)
        
        # Labels
        ax.annotate(f"{width}", (width/2, -0.5), ha='center', fontsize=11,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax.annotate(f"{height}", (-0.5, height/2), ha='right', fontsize=11,
                   fontweight='bold', color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        ax.set_xlim(-1.5, width + 1)
        ax.set_ylim(-1, height + 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_circle(self, ax, params):
        """Draw a circle with radius."""
        radius = params.get("radius", 3)
        center = (radius + 0.5, radius + 0.5)
        
        circle = Circle(center, radius, fill=True,
                       facecolor=self.COLORS["fill"],
                       edgecolor=self.COLORS["stroke"], linewidth=2)
        ax.add_patch(circle)
        
        # Radius line
        ax.plot([center[0], center[0] + radius], [center[1], center[1]],
               color=self.COLORS["accent"], linewidth=2)
        ax.plot(*center, 'o', color=self.COLORS["stroke"], markersize=6)
        
        # Label
        ax.annotate(f"r = {radius}", (center[0] + radius/2, center[1] + 0.4),
                   ha='center', fontsize=11, fontweight='bold',
                   color=self.COLORS["accent"],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        ax.set_xlim(-0.5, 2*radius + 1.5)
        ax.set_ylim(-0.5, 2*radius + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_bar_graph(self, ax, params):
        """Draw a bar graph."""
        data = params.get("data", {"A": 5, "B": 8, "C": 3, "D": 6})
        title = params.get("title", "")
        
        categories = list(data.keys())
        values = list(data.values())
        colors = [self.COLORS["primary"], self.COLORS["accent"], "#10b981", "#7c3aed"]
        bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
        
        bars = ax.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5)
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.annotate(str(val), (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                       ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    def _draw_fraction_circle(self, ax, params):
        """Draw a fraction circle with clear shading."""
        numerator = params.get("numerator", 3)
        denominator = params.get("denominator", 4)
        radius = 2.2
        center = (radius + 0.5, radius + 0.5)
        
        angle_per_part = 360 / denominator
        for i in range(denominator):
            start_angle = 90 - i * angle_per_part
            # Filled parts are accent color, empty parts are light fill
            color = self.COLORS["accent"] if i < numerator else "#e5e7eb"
            wedge = Wedge(center, radius, start_angle - angle_per_part, start_angle,
                         facecolor=color, edgecolor=self.COLORS["stroke"], linewidth=2)
            ax.add_patch(wedge)
        
        # Label
        ax.annotate(f"{numerator}/{denominator}", (center[0], -0.3),
                   ha='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=self.COLORS["accent"], linewidth=2))
        
        ax.set_xlim(-0.5, 2*radius + 1.5)
        ax.set_ylim(-1, 2*radius + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_double_number_line(self, ax, params):
        """Draw a double number line."""
        top_values = params.get("top_values", [0, 2, 4, 6, 8])
        bottom_values = params.get("bottom_values", [0, 5, 10, 15, 20])
        top_label = params.get("top_label", "Miles")
        bottom_label = params.get("bottom_label", "Hours")
        
        line_len = 6
        
        # Top line
        ax.plot([0, line_len], [2, 2], color=self.COLORS["primary"], linewidth=2)
        ax.annotate(top_label, (-0.3, 2), ha='right', va='center', fontsize=10,
                   fontweight='bold', color=self.COLORS["primary"])
        
        # Bottom line
        ax.plot([0, line_len], [0, 0], color=self.COLORS["accent"], linewidth=2)
        ax.annotate(bottom_label, (-0.3, 0), ha='right', va='center', fontsize=10,
                   fontweight='bold', color=self.COLORS["accent"])
        
        # Tick marks and values
        for i, (top_val, bot_val) in enumerate(zip(top_values, bottom_values)):
            x = i * line_len / (len(top_values) - 1)
            ax.plot([x, x], [1.8, 2.2], color=self.COLORS["primary"], linewidth=1.5)
            ax.plot([x, x], [-0.2, 0.2], color=self.COLORS["accent"], linewidth=1.5)
            ax.plot([x, x], [0.2, 1.8], '--', color='#ccc', linewidth=1)
            ax.annotate(str(top_val), (x, 2.4), ha='center', fontsize=9,
                       color=self.COLORS["primary"])
            ax.annotate(str(bot_val), (x, -0.4), ha='center', fontsize=9,
                       color=self.COLORS["accent"])
        
        ax.set_xlim(-1.5, line_len + 0.5)
        ax.set_ylim(-1, 3)
        ax.axis('off')
    
    def _draw_placeholder(self, ax, diagram_type):
        """Draw a placeholder for unsupported diagram types."""
        ax.text(0.5, 0.5, f"[{diagram_type}]", ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
        ax.axis('off')

# Test diagram generation
print("\n🎨 Testing Diagram Generator...")
diagram_gen = DiagramGenerator()
test_spec = DiagramSpec("triangle", {"base": 5, "height": 4})
img_bytes = diagram_gen.generate(test_spec)
print(f"  Generated triangle diagram: {len(img_bytes)} bytes")

# =============================================================================
# CELL 7: ADK TOOLSET AND AGENTS
# =============================================================================


class WorksheetToolset(BaseToolset):
    """
    ADK BaseToolset exposing worksheet generation capabilities to agents.
    
    This demonstrates how to create custom tools that LlmAgents can invoke
    to perform specific tasks. The toolset pattern is central to ADK:
    
    1. Inherit from BaseToolset
    2. Define tool methods with clear docstrings (used by LLM)
    3. Implement async get_tools() returning FunctionTool objects
    4. Track state across tool invocations
    """
    
    def __init__(self):
        super().__init__()
        self._current_plan = None
        self._topic_analysis = None
        self._generated_problems = []
        self._diagram_generator = DiagramGenerator()
        self._latest_artifacts: Dict[str, Any] = {}
    
    # =====================
    # ADK get_tools() method - Required by BaseToolset
    # =====================
    async def get_tools(self, readonly_context=None):
        """
        Return FunctionTool objects for each tool method.
        This is called by ADK when setting up the agent's available tools.
        """
        return [
            FunctionTool(self.generate_plan),
            FunctionTool(self.build_problems),
            FunctionTool(self.build_pdf),
        ]
    
    def latest_artifacts(self) -> Dict[str, Any]:
        """Return the latest built artifacts (for LoopAgent checking)."""
        return self._latest_artifacts
    
    
    def generate_plan(
        self,
        concept_text: str,
        grade: int,
        num_problems: int = 5,
        target_difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """
        Create a structured worksheet plan for the requested concept.
        
        This tool analyzes the topic and creates a blueprint for problem generation.
        Call this FIRST before build_problems or build_pdf.
        
        Args:
            concept_text: The math topic (e.g., "area of triangles", "adding fractions")
            grade: Target grade level (1-8)
            num_problems: Number of problems to generate (default: 5)
            target_difficulty: "easy", "medium", or "hard" (default: "medium")
            
        Returns:
            A plan dictionary with domain analysis, problem templates, and number ranges
        """
        print(f"  [Tool] generate_plan called: {concept_text}, grade {grade}, {num_problems} problems")
        
        # Analyze the topic
        self._topic_analysis = interpret_topic(concept_text, grade)
        
        # Create problem templates
        self._current_plan = {
            "concept": concept_text,
            "grade": grade,
            "difficulty": target_difficulty,
            "num_problems": num_problems,
            "domain": self._topic_analysis["domain"],
            "diagram_type": self._topic_analysis["diagram_type"],
            "answer_diagram_type": self._topic_analysis["answer_diagram_type"],
            "number_range": self._topic_analysis["number_range"],
            "problem_templates": []
        }
        
        for i in range(num_problems):
            template = {
                "index": i + 1,
                "skill": f"{concept_text} problem {i+1}",
                "difficulty": target_difficulty,
                "requires_diagram": self._topic_analysis["diagram_type"] is not None
            }
            self._current_plan["problem_templates"].append(template)
        
        return {
            "status": "success",
            "concept": concept_text,
            "domain": self._topic_analysis["domain"],
            "num_templates": num_problems,
            "diagram_type": self._topic_analysis["diagram_type"],
            "number_range": self._topic_analysis["number_range"],
            "message": f"Created plan with {num_problems} problem templates for {concept_text}"
        }
    
    
    def build_problems(self, problems_json: str) -> Dict[str, Any]:
        """
        Convert problem specifications into Problem objects with diagrams.
        
        Call this AFTER generate_plan to create the actual problem objects.
        The problems should be provided as a JSON string.
        
        Args:
            problems_json: JSON string with format:
                {"problems": [{"question": "...", "answer": "...", "steps": [...]}]}
            
        Returns:
            Status and count of problems built
        """
        print(f"  [Tool] build_problems called")
        
        try:
            problems_data = json.loads(problems_json) if isinstance(problems_json, str) else problems_json
            if isinstance(problems_data, dict):
                problems_data = problems_data.get("problems", [])
        except Exception as e:
            print(f"  [Tool] JSON parse error: {e}")
            problems_data = []
        
        self._generated_problems = []
        for i, raw in enumerate(problems_data, 1):
            diagram = None
            answer_diagram = None
            question_text = raw.get("question", f"Problem {i}")
            
            # Use diagram_data from LLM if provided (preferred)
            llm_diagram_data = raw.get("diagram_data", {})
            
            # Create diagrams based on topic analysis
            if self._topic_analysis and self._topic_analysis.get("diagram_type"):
                diagram_type = self._topic_analysis["diagram_type"]
                
                # Use LLM-provided data if available, otherwise try extraction
                if llm_diagram_data:
                    # For bar graphs, wrap in "data" key
                    if diagram_type == "bar_graph" and "data" not in llm_diagram_data:
                        params = {"data": llm_diagram_data}
                    else:
                        params = llm_diagram_data
                else:
                    # Fallback to extraction from text
                    params = self._generate_diagram_params(
                        diagram_type,
                        self._topic_analysis["number_range"],
                        question_text
                    )
                diagram = DiagramSpec(diagram_type, params)
            
            if self._topic_analysis and self._topic_analysis.get("answer_diagram_type"):
                answer_diagram = DiagramSpec(
                    self._topic_analysis["answer_diagram_type"],
                    llm_diagram_data or {}
                )
            
            problem = Problem(
                number=i,
                question=question_text,
                answer=raw.get("answer", "N/A"),
                solution_steps=raw.get("steps", []),
                diagram=diagram,
                answer_diagram=answer_diagram
            )
            self._generated_problems.append(problem)
        
        return {
            "status": "success",
            "problems_built": len(self._generated_problems),
            "message": f"Built {len(self._generated_problems)} problem objects with diagrams"
        }
    
    
    def build_pdf(
        self,
        title: Optional[str] = None,
        concept: Optional[str] = None,
        grade: Optional[int] = None,
        difficulty: Optional[str] = None,
        problems: Optional[str] = None  # JSON string fallback
    ) -> Dict[str, Any]:
        """
        Build the final PDF worksheet with problems and answer key.
        
        Call this AFTER build_problems to generate the PDF artifact.
        
        Args:
            title: Worksheet title (auto-generated if not provided)
            concept: The math concept (uses plan if not provided)
            grade: Grade level (uses plan if not provided)
            difficulty: Difficulty level (uses plan if not provided)
            problems: Optional JSON string of problems if build_problems wasn't called
            
        Returns:
            Status and path to the generated PDF
        """
        print(f"  [Tool] build_pdf called")
        
        # Use plan values as defaults
        if self._current_plan:
            concept = concept or self._current_plan.get("concept", "Math")
            grade = grade or self._current_plan.get("grade", 6)
            difficulty = difficulty or self._current_plan.get("difficulty", "medium")
        else:
            concept = concept or "Math"
            grade = grade or 6
            difficulty = difficulty or "medium"
        
        if title is None:
            title = f"Grade {grade} {difficulty.capitalize()} Worksheet: {concept.title()}"
        
        # Build problems from JSON if not already built
        if not self._generated_problems and problems:
            self.build_problems(problems)
        
        if not self._generated_problems:
            return {"status": "error", "message": "No problems to build. Call build_problems first."}
        
        worksheet = Worksheet(
            title=title,
            grade=grade,
            difficulty=difficulty,
            concept=concept,
            problems=self._generated_problems
        )
        
        safe_concept = re.sub(r'[^\w\s-]', '', concept).replace(' ', '_')
        output_filename = f"worksheet_{safe_concept}_grade{grade}.pdf"
        output_path = OUTPUT_DIR / output_filename
        
        builder = PDFBuilder()
        builder.build(worksheet, output_path)
        
        # Track artifact for LoopAgent verification
        self._latest_artifacts["pdf"] = {"path": str(output_path)}
        
        return {
            "status": "success",
            "path": str(output_path),
            "num_problems": len(self._generated_problems),
            "message": f"PDF created at {output_path}"
        }
    
    def _extract_diagram_data_from_text(self, text: str, diagram_type: str) -> Optional[Dict]:
        """Extract diagram parameters from problem text if present."""
        if not text:
            return None
        
        # Debug: show extraction attempt
        if diagram_type == "bar_graph":
            data = {}
            
            # Pattern 1: "Name: 12" (colon format)
            p1 = re.findall(r'([A-Z][a-z]+):\s*(\d+)', text)
            
            # Pattern 2: "Name read/own/has/scored/sold 12" (verb format)
            p2 = re.findall(r'([A-Z][a-z]+)\s+(?:read|reads|own|owns|has|have|had|got|scored|sold|ate|collected|earned)\s+(\d+)', text, re.IGNORECASE)
            
            # Pattern 3: "• Name 12 items" (bullet format)
            p3 = re.findall(r'[•\-\*]\s*([A-Z][a-z]+)\s+(\d+)', text)
            
            # Pattern 4: "X families own Y pets" (quantity format)
            p4 = re.findall(r'(\d+)\s+(?:families|people|students|kids|children)\s+(?:own|have|read)\s+(\d+)', text, re.IGNORECASE)
            if p4:
                # Convert to label format like "1 pet: 3 families"
                data = {f"{v} pets": int(k) for k, v in p4[:6]}
                if len(data) >= 2:
                    return {"data": data}
            
            all_matches = p1 + p2 + p3
            
            if len(all_matches) >= 2:
                seen = set()
                for name, val in all_matches:
                    if name not in seen:
                        data[name] = int(val)
                        seen.add(name)
                if len(data) >= 2:
                    return {"data": data}
        
        elif diagram_type == "fraction_circle":
            # Look for fractions like "2/8" or "3/4"
            match = re.search(r'(\d+)/(\d+)', text)
            if match:
                return {"numerator": int(match.group(1)), "denominator": int(match.group(2))}
        
        elif diagram_type in ["triangle", "right_triangle"]:
            # Flexible patterns for base and height
            base_match = re.search(r'base[^.]*?(\d+)', text, re.IGNORECASE)
            height_match = re.search(r'height[^.]*?(\d+)', text, re.IGNORECASE)
            if base_match and height_match:
                return {"base": int(base_match.group(1)), "height": int(height_match.group(1))}
        
        elif diagram_type == "rectangle":
            # Look for dimensions
            width_match = re.search(r'width\s*(?:of|is|=)?\s*(\d+)', text, re.IGNORECASE)
            height_match = re.search(r'(?:height|length)\s*(?:of|is|=)?\s*(\d+)', text, re.IGNORECASE)
            if width_match and height_match:
                return {"width": int(width_match.group(1)), "height": int(height_match.group(1))}
            # Also try "X by Y" pattern
            dim_match = re.search(r'(\d+)\s*(?:by|x|×)\s*(\d+)', text, re.IGNORECASE)
            if dim_match:
                return {"width": int(dim_match.group(1)), "height": int(dim_match.group(2))}
        
        return None
    
    def _generate_diagram_params(self, diagram_type: str, number_range: List[int], problem_text: str = "") -> Dict:
        """Generate parameters for diagram based on type, extracting from text if possible."""
        
        # First try to extract from problem text
        extracted = self._extract_diagram_data_from_text(problem_text, diagram_type)
        if extracted:
            return extracted
        
        # Fall back to random generation
        min_val = max(2, abs(number_range[0]) // 10)
        max_val = min(12, abs(number_range[1]) // 5)
        
        if diagram_type in ["triangle", "right_triangle"]:
            return {"base": random.randint(min_val, max_val), 
                    "height": random.randint(min_val, max_val)}
        elif diagram_type == "rectangle":
            return {"width": random.randint(min_val, max_val),
                    "height": random.randint(min_val, max_val)}
        elif diagram_type == "circle":
            return {"radius": random.randint(min_val, max_val)}
        elif diagram_type == "bar_graph":
            categories = ["A", "B", "C", "D"]
            return {"data": {c: random.randint(2, 10) for c in categories}}
        elif diagram_type == "fraction_circle":
            den = random.choice([2, 3, 4, 5, 6, 8])
            return {"numerator": random.randint(1, den-1), "denominator": den}
        elif diagram_type == "double_number_line":
            ratio = random.randint(2, 5)
            return {
                "top_values": [i*2 for i in range(5)],
                "bottom_values": [i*ratio for i in range(5)],
                "top_label": "Miles",
                "bottom_label": "Hours"
            }
        return {}


# ==========================
# ADK AGENTS AND LOOP
# ==========================

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import EventActions
from typing import AsyncGenerator

class ArtifactCheckAgent(BaseAgent):
    """
    ADK BaseAgent for verifying artifact creation.
    
    This demonstrates the BaseAgent pattern for custom logic:
    - Doesn't call LLM, just inspects state
    - Yields Event objects with EventActions
    - Used inside LoopAgent to control iteration
    """
    
    def __init__(self, name: str, toolset: 'WorksheetToolset'):
        super().__init__(name=name)
        self._toolset = toolset
        self._iteration_count = 0
    
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Check if PDF artifact has been created."""
        self._iteration_count += 1
        artifacts = self._toolset.latest_artifacts()
        have_pdf = "pdf" in artifacts
        
        print(f"  [ArtifactCheck] Iteration {self._iteration_count}: pdf={have_pdf}")
        
        if have_pdf:
            # Escalate to stop the loop
            print(f"  [ArtifactCheck] PDF found, stopping loop")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
            )
        else:
            # Continue the loop
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False),
            )


def build_worksheet_agent(toolset: WorksheetToolset) -> LoopAgent:
    """
    Build the complete ADK agent hierarchy.
    
    Architecture:
    ┌─────────────────────────────────────────────┐
    │            LoopAgent (max_iterations=2)     │
    │  ┌───────────────────────────────────────┐  │
    │  │      LlmAgent (worksheet_orchestrator)│  │
    │  │      - Uses WorksheetToolset          │  │
    │  │      - Calls generate_plan            │  │
    │  │      - Generates problems             │  │
    │  │      - Calls build_pdf                │  │
    │  └───────────────────────────────────────┘  │
    │  ┌───────────────────────────────────────┐  │
    │  │      ArtifactCheckAgent               │  │
    │  │      - Verifies PDF was created       │  │
    │  │      - Escalates to stop loop         │  │
    │  └───────────────────────────────────────┘  │
    └─────────────────────────────────────────────┘
    """
    
    instruction = """You are a math worksheet generator. Create educational worksheets.

WORKFLOW:
1. Call `generate_plan` with concept, grade, num_problems, difficulty.

2. Create problems as JSON with this EXACT format:
   {
     "problems": [
       {
         "question": "The problem text...",
         "answer": "The answer",
         "steps": ["Step 1", "Step 2"],
         "diagram_data": {"label1": value1, "label2": value2}  // For bar graphs, geometry, etc.
       }
     ]
   }

   DIAGRAM DATA EXAMPLES:
   - Bar graphs: {"diagram_data": {"Amy": 12, "Ben": 8, "Cathy": 15}}
   - Triangles: {"diagram_data": {"base": 8, "height": 6}}
   - Fractions: {"diagram_data": {"numerator": 2, "denominator": 8}}
   
   The diagram_data MUST match the values in your question!

3. Call `build_problems` with your JSON.
4. Call `build_pdf` to create the worksheet.

RULES:
- No LaTeX - write fractions as "3/4"
- Use Unicode: π, ≈, ², ³
- Make problems age-appropriate and engaging

Complete all 4 steps to finish the worksheet."""

    # Create the main LLM agent
    worksheet_agent = LlmAgent(
        name="worksheet_orchestrator",
        description="Plans and builds math worksheets using specialized tools.",
        instruction=instruction,
        model="gemini-2.0-flash",
        tools=[toolset],
    )
    
    # Wrap in LoopAgent with artifact checker
    loop_agent = LoopAgent(
        name="worksheet_loop",
        sub_agents=[
            worksheet_agent,
            ArtifactCheckAgent(name="artifact_checker", toolset=toolset)
        ],
        max_iterations=2,
    )
    
    return loop_agent


# ==========================
# ADK RUNNER
# ==========================

async def run_worksheet_agent(
    concept: str, 
    grade: int, 
    difficulty: str, 
    num_problems: int
) -> tuple[Dict[str, Any], WorksheetToolset]:
    """
    Execute the ADK worksheet agent pipeline.
    
    This demonstrates the complete ADK execution flow:
    1. Create session service for state management
    2. Create toolset with custom tools
    3. Build agent hierarchy (LoopAgent -> LlmAgent + CheckAgent)
    4. Create Runner with agent and services
    5. Execute via run_async() and process events
    
    Args:
        concept: Math topic
        grade: Target grade level
        difficulty: easy/medium/hard
        num_problems: Number of problems
        
    Returns:
        Tuple of (result dict, toolset with state)
    """
    print(f"\n  [ADK] Initializing agent pipeline...")
    
    # Create ADK services
    session_service = InMemorySessionService()
    
    # Create toolset with custom tools
    toolset = WorksheetToolset()
    
    # Build agent hierarchy
    agent = build_worksheet_agent(toolset)
    
    # Create runner
    runner = Runner(
        agent=agent,
        app_name="worksheet_generator",
        session_service=session_service,
    )
    
    # Create session
    session = await session_service.create_session(
        app_name="worksheet_generator",
        user_id="demo_user"
    )
    
    # Build the prompt
    prompt_text = f"""Create a {difficulty} worksheet for grade {grade} students on: {concept}
Generate exactly {num_problems} problems.

Remember to:
1. Call generate_plan(concept_text="{concept}", grade={grade}, num_problems={num_problems}, target_difficulty="{difficulty}")
2. Create {num_problems} creative word problems as JSON
3. Call build_problems with your JSON
4. Call build_pdf to create the worksheet"""

    # Wrap in Content object for ADK (matching main implementation)
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt_text)]
    )

    print(f"  [ADK] Sending prompt to agent...")
    print(f"  [ADK] Agent: {agent.name}")
    
    # Track results with safety limits (matching main implementation)
    result = {"status": "pending", "events": 0, "tool_calls": []}
    MAX_EVENTS = 50
    
    try:
        async for event in runner.run_async(
            session_id=session.id,
            user_id="demo_user",
            new_message=user_content
        ):
            result["events"] += 1
            
            # Safety limit
            if result["events"] > MAX_EVENTS:
                print(f"  [ADK] Max events reached ({MAX_EVENTS})")
                break
            
            # Early exit if PDF already created
            if toolset._latest_artifacts.get("pdf"):
                print(f"  [ADK] PDF created, exiting early")
                break
            
            # Log function calls safely (check for None)
            if hasattr(event, 'content') and event.content:
                parts = getattr(event.content, 'parts', None)
                if parts:
                    for part in parts:
                        func_call = getattr(part, 'function_call', None)
                        if func_call is not None:
                            func_name = getattr(func_call, 'name', None)
                            if func_name:
                                print(f"  [ADK] Tool call: {func_name}")
                                result["tool_calls"].append(func_name)
        
        # Check if PDF was created
        if toolset._latest_artifacts.get("pdf"):
            result["status"] = "success"
            result["path"] = toolset._latest_artifacts["pdf"]["path"]
        else:
            result["status"] = "incomplete"
    
    except Exception as e:
        print(f"  [ADK] Error: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    
    return result, toolset


print("\n✅ ADK Components Defined!")
print("  ┌─────────────────────────────────────────────────┐")
print("  │ WorksheetToolset (BaseToolset)                  │")
print("  │   ├── generate_plan   - Topic analysis          │")
print("  │   ├── build_problems  - Create Problem objects  │")
print("  │   └── build_pdf       - Generate PDF artifact   │")
print("  ├─────────────────────────────────────────────────┤")
print("  │ LlmAgent (worksheet_orchestrator)               │")
print("  │   └── Autonomous multi-step reasoning           │")
print("  ├─────────────────────────────────────────────────┤")
print("  │ ArtifactCheckAgent (BaseAgent)                  │")
print("  │   └── Verifies PDF creation                     │")
print("  ├─────────────────────────────────────────────────┤")
print("  │ LoopAgent                                       │")
print("  │   └── Orchestrates retry logic                  │")
print("  ├─────────────────────────────────────────────────┤")
print("  │ Runner + InMemorySessionService                 │")
print("  │   └── Async execution with state management     │")
print("  └─────────────────────────────────────────────────┘")

# =============================================================================
# CELL 8: TEXT FORMATTER & PDF BUILDER
# =============================================================================

def format_math_text(text: str) -> str:
    """
    Format math text for middle school readability:
    - Replace * with visible bullet (•) for multiplication (avoids confusion with variable x)
    - Add proper spacing for clarity
    - Keep fractions as-is (will be rendered specially in PDF)
    """
    # Replace asterisk with bullet point (• is more visible than · for multiplication)
    # Add extra spacing around operators for clarity
    text = text.replace(' * ', '  •  ')
    text = text.replace('*', '  •  ')
    
    # Also convert any × or · to bullet for consistency and visibility
    text = text.replace('×', '  •  ')
    text = text.replace('·', '  •  ')
    
    # Replace common division notation with extra spacing
    text = text.replace(' / ', '  ÷  ')
    
    # Add space after colons for better readability
    text = text.replace(': ', ':  ')
    
    # Clean up any triple+ spaces
    while '   ' in text:
        text = text.replace('   ', '  ')
    
    return text


class PDFBuilder:
    """Builds PDF worksheets with problems, diagrams, and answer keys."""
    
    def __init__(self):
        self.pagesize = letter
        self.margin = 72
        self.line_height = 16
        self.diagram_gen = DiagramGenerator()
    
    def _draw_stacked_fraction(self, c, numerator: str, denominator: str, 
                                x: float, y: float, font_size: int = 12,
                                with_parens: bool = False) -> float:
        """
        Draw a stacked fraction (numerator over denominator with line).
        If with_parens=True, draws tall parentheses around the fraction.
        
        Returns the width used by the fraction (including parentheses if any).
        """
        c.setFont("Helvetica", font_size)
        
        # Calculate widths with generous padding
        num_width = c.stringWidth(str(numerator), "Helvetica", font_size)
        den_width = c.stringWidth(str(denominator), "Helvetica", font_size)
        frac_width = max(num_width, den_width) + 12  # more padding for clarity
        
        # Calculate fraction vertical bounds
        top_y = y + font_size * 0.5
        bottom_y = y - font_size * 1.1
        frac_height = top_y - bottom_y
        
        current_x = x
        paren_width = font_size * 0.4  # slightly wider parens
        
        # Draw opening parenthesis if needed
        if with_parens:
            self._draw_tall_paren(c, "(", current_x, y, frac_height, font_size)
            current_x += paren_width + 3  # gap after open paren
        
        # Draw numerator (centered above line)
        num_x = current_x + (frac_width - num_width) / 2
        c.drawString(num_x, y + font_size * 0.35, str(numerator))
        
        # Draw fraction line (slightly thicker for visibility)
        line_y = y
        c.setLineWidth(1.2)
        c.line(current_x, line_y, current_x + frac_width, line_y)
        c.setLineWidth(1)
        
        # Draw denominator (centered below line)
        den_x = current_x + (frac_width - den_width) / 2
        c.drawString(den_x, y - font_size * 0.85, str(denominator))
        
        current_x += frac_width
        
        # Draw closing parenthesis if needed
        if with_parens:
            current_x += 3  # gap before close paren
            self._draw_tall_paren(c, ")", current_x, y, frac_height, font_size)
            current_x += paren_width
        
        total_width = current_x - x
        return total_width

    def _draw_tall_paren(self, c, paren: str, x: float, y: float, 
                          height: float, font_size: int = 12) -> float:
        """
        Draw a tall parenthesis that spans the given height.
        Uses a larger font size to create properly scaled parentheses that
        extend both ABOVE and BELOW the fraction.
        Returns the width used.
        """
        # Use 1.8x font for parenthesis - tall enough to span fraction
        paren_font_size = font_size * 1.8
        
        # y is the fraction LINE position (middle of fraction)
        # Parenthesis should be centered on this line
        # A parenthesis character's visual center is about 40% up from baseline
        # So we need to position the baseline lower to center the paren
        paren_y = y - paren_font_size * 0.35
        
        c.setFont("Helvetica", paren_font_size)
        c.drawString(x, paren_y, paren)
        
        # Reset font
        c.setFont("Helvetica", font_size)
        
        # Return width of the parenthesis character
        paren_width = c.stringWidth(paren, "Helvetica", paren_font_size)
        return paren_width
    
    def _parse_and_draw_text_with_fractions(self, c, text: str, x: float, y: float, 
                                             font_size: int = 12) -> float:
        """
        Parse text and render fractions as stacked fractions.
        Returns the new y position.
        """
        # Pattern to find fractions like "3/4" or "-3/4"
        fraction_pattern = r'(-?\d+)/(\d+)'
        
        c.setFont("Helvetica", font_size)
        current_x = x
        last_end = 0
        
        # Check if this text contains fractions
        import re
        matches = list(re.finditer(fraction_pattern, text))
        
        if not matches:
            # No fractions, just draw the text normally
            c.drawString(x, y, format_math_text(text))
            return y - self.line_height
        
        # Has fractions - need to draw piece by piece
        for match in matches:
            # Draw text before this fraction
            before_text = format_math_text(text[last_end:match.start()])
            if before_text:
                c.drawString(current_x, y, before_text)
                current_x += c.stringWidth(before_text, "Helvetica", font_size)
            
            # Draw the stacked fraction
            numerator = match.group(1)
            denominator = match.group(2)
            frac_width = self._draw_stacked_fraction(c, numerator, denominator, 
                                                      current_x, y, font_size)
            current_x += frac_width + 4  # small gap after fraction
            
            last_end = match.end()
        
        # Draw any remaining text after the last fraction
        remaining = format_math_text(text[last_end:])
        if remaining:
            c.drawString(current_x, y, remaining)
        
        return y - self.line_height * 1.8  # Extra space for stacked fractions
    
    def build(self, worksheet: Worksheet, output_path: str) -> Path:
        """Build a complete PDF worksheet."""
        output_path = Path(output_path)
        c = pdf_canvas.Canvas(str(output_path), pagesize=self.pagesize)
        width, height = self.pagesize
        y = height - self.margin
        
        # Title
        y = self._draw_header(c, worksheet, y)
        
        # Problems
        y = self._draw_problems(c, worksheet.problems, y)
        
        # Answer Key (new page)
        c.showPage()
        y = height - self.margin
        y = self._draw_answer_key(c, worksheet.problems, y)
        
        c.save()
        return output_path
    
    def _draw_header(self, c, worksheet: Worksheet, y: float) -> float:
        """Draw worksheet header."""
        c.setFont("Helvetica-Bold", 18)
        c.drawString(self.margin, y, worksheet.title)
        y -= 25
        
        c.setFont("Helvetica", 12)
        c.drawString(self.margin, y, f"Grade: {worksheet.grade} • Difficulty: {worksheet.difficulty.capitalize()}")
        y -= 15
        c.drawString(self.margin, y, f"Concept: {worksheet.concept}")
        y -= 10
        
        # Divider
        c.line(self.margin, y, self.pagesize[0] - self.margin, y)
        y -= 30
        
        # Name/Date
        c.drawString(self.margin, y, "Name: _______________________")
        c.drawString(300, y, "Date: _______________")
        y -= 30
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(self.margin, y, "Problems")
        y -= 25
        
        return y
    
    def _draw_problems(self, c, problems: List[Problem], y: float) -> float:
        """Draw all problems with proper math formatting."""
        c.setFont("Helvetica", 12)
        
        for problem in problems:
            # Check for page break
            if y < 200:
                c.showPage()
                y = self.pagesize[1] - self.margin
                c.setFont("Helvetica", 12)
            
            # Problem text with stacked fractions
            text = f"{problem.number}. {problem.question}"
            y = self._draw_math_text(c, text, y)
            
            # Draw diagram if present
            if problem.diagram:
                y -= 10
                y = self._draw_diagram(c, problem.diagram, y)
            
            # Work space
            y -= 100
        
        return y
    
    def _draw_answer_key(self, c, problems: List[Problem], y: float) -> float:
        """Draw answer key section with proper math formatting."""
        c.setFont("Helvetica-Bold", 16)
        c.drawString(self.margin, y, "Answer Key")
        y -= 25
        
        c.setFont("Helvetica", 11)
        
        for problem in problems:
            if y < 150:
                c.showPage()
                y = self.pagesize[1] - self.margin
                c.setFont("Helvetica", 11)
            
            # Question with stacked fractions
            text = f"{problem.number}. {problem.question}"
            y = self._draw_math_text(c, text, y, font_size=11)
            
            # Answer with stacked fractions
            c.setFont("Helvetica-Bold", 11)
            y = self._draw_math_text(c, f"Answer: {problem.answer}", y, font_size=11)
            c.setFont("Helvetica", 11)
            
            # Solution steps with stacked fractions
            if problem.solution_steps:
                y -= 5
                c.drawString(self.margin, y, "Steps:")
                y -= self.line_height
                for step in problem.solution_steps:
                    y = self._draw_math_text(c, f"  • {step}", y, font_size=10)
            
            # Answer diagram if present
            if problem.answer_diagram:
                y -= 10
                y = self._draw_diagram(c, problem.answer_diagram, y)
            
            y -= 15
        
        return y
    
    def _draw_math_text(self, c, text: str, y: float, font_size: int = 12) -> float:
        """
        Draw text with proper math formatting and wrapping:
        - Fractions rendered as stacked (numerator over denominator)
        - Parenthesized fractions get tall parentheses
        - Multiplication shown as · (dot)
        - Long lines wrap properly
        """
        # Format multiplication symbols
        text = format_math_text(text)
        
        # Check for fractions (optionally with parentheses)
        fraction_pattern = r'(\()?(-?\d+)/(\d+)(\))?'
        matches = list(re.finditer(fraction_pattern, text))
        
        if not matches:
            # No fractions, use regular wrapped text
            return self._draw_wrapped_text(c, text, y, font_size)
        
        # Has fractions - need smart wrapping
        c.setFont("Helvetica", font_size)
        max_width = self.pagesize[0] - 2 * self.margin
        
        # Build segments: each segment is either text or a fraction
        segments = []
        last_end = 0
        for match in matches:
            # Text before fraction
            if match.start() > last_end:
                segments.append(('text', text[last_end:match.start()]))
            # The fraction itself
            open_paren = match.group(1)
            close_paren = match.group(4)
            has_parens = (open_paren == '(' and close_paren == ')')
            segments.append(('fraction', match.group(2), match.group(3), has_parens))
            last_end = match.end()
        # Text after last fraction
        if last_end < len(text):
            segments.append(('text', text[last_end:]))
        
        # Calculate width of each segment
        # Use larger gaps for better readability
        FRACTION_GAP = 10  # pixels gap after each fraction
        
        def get_segment_width(seg):
            if seg[0] == 'text':
                return c.stringWidth(seg[1], "Helvetica", font_size)
            else:  # fraction
                num, den, has_parens = seg[1], seg[2], seg[3]
                num_w = c.stringWidth(str(num), "Helvetica", font_size)
                den_w = c.stringWidth(str(den), "Helvetica", font_size)
                frac_w = max(num_w, den_w) + 12  # padding inside fraction
                if has_parens:
                    frac_w += font_size * 0.6 + 10  # parentheses width
                return frac_w + FRACTION_GAP  # gap after
        
        # Render segments with wrapping
        current_x = self.margin
        has_fraction_on_line = False
        
        for seg in segments:
            seg_width = get_segment_width(seg)
            
            # Check if we need to wrap
            if current_x + seg_width > self.margin + max_width and current_x > self.margin:
                # Wrap to new line
                y -= self.line_height * (1.8 if has_fraction_on_line else 1.0)
                current_x = self.margin
                has_fraction_on_line = False
                
                # Check for page break
                if y <= self.margin:
                    c.showPage()
                    y = self.pagesize[1] - self.margin
                    c.setFont("Helvetica", font_size)
            
            if seg[0] == 'text':
                # For text segments, we may need to wrap within the segment
                words = seg[1].split()
                for i, word in enumerate(words):
                    word_with_space = word + (' ' if i < len(words) - 1 else '')
                    word_width = c.stringWidth(word_with_space, "Helvetica", font_size)
                    
                    if current_x + word_width > self.margin + max_width and current_x > self.margin:
                        y -= self.line_height * (1.8 if has_fraction_on_line else 1.0)
                        current_x = self.margin
                        has_fraction_on_line = False
                        
                        if y <= self.margin:
                            c.showPage()
                            y = self.pagesize[1] - self.margin
                            c.setFont("Helvetica", font_size)
                    
                    # Check if this is an operator that should be vertically centered
                    # (when appearing between fractions)
                    stripped_word = word_with_space.strip()
                    is_operator = stripped_word in ['•', '÷', '+', '-', '=']
                    if is_operator and has_fraction_on_line:
                        # The fraction line is at y - that's our vertical center
                        # To center a character, we need baseline at: center - (char_height * 0.4)
                        
                        if stripped_word == '÷':
                            # Division symbol - BIGGER and centered
                            big_font = font_size * 1.6
                            c.setFont("Helvetica", big_font)
                            # Center of ÷ is about 45% up from baseline
                            operator_y = y - big_font * 0.45
                            c.drawString(current_x, operator_y, stripped_word)
                            current_x += c.stringWidth(stripped_word, "Helvetica", big_font)
                            c.setFont("Helvetica", font_size)
                        else:
                            # Bullet (•) and other operators - centered
                            # Center of • is about 50% up from baseline
                            operator_y = y - font_size * 0.5
                            c.drawString(current_x, operator_y, stripped_word)
                            current_x += c.stringWidth(stripped_word, "Helvetica", font_size)
                        # Add spacing after operator
                        current_x += font_size * 0.3
                    else:
                        c.drawString(current_x, y, word_with_space)
                        current_x += word_width
            else:
                # Draw fraction with generous spacing
                num, den, has_parens = seg[1], seg[2], seg[3]
                frac_width = self._draw_stacked_fraction(
                    c, num, den, current_x, y, font_size, with_parens=has_parens
                )
                current_x += frac_width + FRACTION_GAP
                has_fraction_on_line = True
        
        return y - self.line_height * (1.8 if has_fraction_on_line else 1.0)
    
    def _draw_diagram(self, c, diagram: DiagramSpec, y: float) -> float:
        """Draw a diagram on the canvas."""
        try:
            img_bytes = self.diagram_gen.generate(diagram)
            img = ImageReader(io.BytesIO(img_bytes))
            
            img_width = diagram.width * 72
            img_height = diagram.height * 72
            
            x = (self.pagesize[0] - img_width) / 2
            c.drawImage(img, x, y - img_height, width=img_width, height=img_height)
            
            return y - img_height - 10
        except Exception as e:
            print(f"  ⚠️ Diagram error: {e}")
            return y
    
    def _draw_wrapped_text(self, c, text: str, y: float, font_size: int = 12) -> float:
        """Draw text with word wrapping."""
        c.setFont("Helvetica", font_size)
        max_width = self.pagesize[0] - 2 * self.margin
        
        words = text.split()
        line = ""
        
        for word in words:
            test_line = f"{line} {word}".strip()
            if c.stringWidth(test_line, "Helvetica", font_size) < max_width:
                line = test_line
            else:
                c.drawString(self.margin, y, line)
                y -= self.line_height
                line = word
        
        if line:
            c.drawString(self.margin, y, line)
            y -= self.line_height
        
        return y

print("✅ PDF Builder ready!")

# =============================================================================
# CELL 9: MAIN WORKFLOW (ADK-POWERED)
# =============================================================================

async def generate_worksheet_adk(
    concept: str,
    grade: int = 6,
    difficulty: str = "medium",
    num_problems: int = 5,
    output_filename: str = None
) -> Path:
    """
    Main entry point using ADK agents.
    
    This showcases the full ADK pipeline:
    1. Creates an agent with custom toolset
    2. Runs the agent via ADK Runner
    3. Agent autonomously calls tools to generate worksheet
    
    Args:
        concept: The math topic (e.g., "area of triangles")
        grade: Target grade level (1-8)
        difficulty: "easy", "medium", or "hard"
        num_problems: Number of problems to generate
        output_filename: Output PDF filename (auto-generated if None)
        
    Returns:
        Path to the generated PDF file
    """
    print(f"\n{'='*60}")
    print(f"🤖 ADK WORKSHEET GENERATOR")
    print(f"{'='*60}")
    print(f"  Topic: {concept}")
    print(f"  Grade: {grade}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Problems: {num_problems}")
    
    if output_filename is None:
        safe_concept = re.sub(r'[^\w\s-]', '', concept).replace(' ', '_')
        output_filename = f"worksheet_{safe_concept}_grade{grade}.pdf"
    
    # Run ADK agent pipeline
    print(f"\n🚀 Starting ADK Agent Pipeline...")
    result, toolset = await run_worksheet_agent(concept, grade, difficulty, num_problems)
    
    output_path = OUTPUT_DIR / output_filename
    
    if result["status"] == "success":
        print(f"\n✅ WORKSHEET COMPLETE!")
        print(f"  Output: {output_path}")
        print(f"  Agent Events: {result['events']}")
        print(f"  Tools Called: {', '.join(result.get('tool_calls', []))}")
    else:
        print(f"\n❌ ADK Pipeline Error: {result.get('error', 'Unknown error')}")
        print(f"  Events processed: {result['events']}")
    
    print(f"{'='*60}\n")
    return output_path


def generate_worksheet(concept: str, grade: int = 6, difficulty: str = "medium",
                       num_problems: int = 5, output_filename: str = None) -> Path:
    """
    Synchronous wrapper for ADK worksheet generation.
    Use this in non-async contexts.
    """
    return asyncio.run(generate_worksheet_adk(concept, grade, difficulty, num_problems, output_filename))

# =============================================================================
# CELL 10: GENERATE WORKSHEETS
# =============================================================================

async def generate_worksheets():
    """Generate sample worksheets using the ADK pipeline."""
    
    # Clear old worksheets first
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("🗑️ Cleared old worksheets\n")
    
    # Worksheet 1: Geometry
    print("📐 Generating: Area of Triangles (Grade 6)")
    pdf1 = await generate_worksheet_adk(
        concept="area of triangles",
        grade=6,
        difficulty="medium",
        num_problems=3
    )
    
    # Worksheet 2: Fractions
    print("\n🍕 Generating: Adding Fractions (Grade 5)")
    pdf2 = await generate_worksheet_adk(
        concept="adding fractions",
        grade=5,
        difficulty="easy",
        num_problems=3
    )
    
    # Worksheet 3: Statistics
    print("\n📊 Generating: Bar Graphs (Grade 4)")
    pdf3 = await generate_worksheet_adk(
        concept="interpreting bar graphs",
        grade=4,
        difficulty="medium",
        num_problems=3
    )
    
    print(f"\n✅ Complete! Files in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"  📄 {f.name}")
    
    return [pdf1, pdf2, pdf3]


await generate_worksheets()

# =============================================================================
# CELL 11: VIEW/DOWNLOAD FILES IN KAGGLE
# =============================================================================
# Run this cell to get download links for generated PDFs

from IPython.display import FileLink, display, HTML

print("📥 View your worksheets:\n")
for pdf_file in sorted(OUTPUT_DIR.glob("*.pdf")):
    display(FileLink(str(pdf_file), result_html_prefix=f"📄 "))



