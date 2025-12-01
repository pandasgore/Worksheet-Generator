"""
llm_topic_analyzer.py

Uses an LLM to analyze custom/unknown topics and provide intelligent 
suggestions for diagram types, difficulty parameters, and problem structure.
This provides a fallback when keyword-based detection fails.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# Grade-level characteristics for reference
GRADE_CHARACTERISTICS = {
    1: {
        "number_range": [0, 20],
        "operations": ["counting", "addition", "subtraction"],
        "concepts": ["single-digit numbers", "basic shapes", "comparison"],
    },
    2: {
        "number_range": [0, 100],
        "operations": ["addition", "subtraction", "skip counting"],
        "concepts": ["place value", "basic fractions", "time to hour/half-hour"],
    },
    3: {
        "number_range": [0, 1000],
        "operations": ["multiplication", "division", "fractions"],
        "concepts": ["area", "perimeter", "fractions on number line"],
    },
    4: {
        "number_range": [0, 10000],
        "operations": ["multi-digit multiplication", "division with remainders"],
        "concepts": ["equivalent fractions", "decimals", "angles"],
    },
    5: {
        "number_range": [0, 1000000],
        "operations": ["fraction operations", "decimal operations"],
        "concepts": ["volume", "coordinate plane", "order of operations"],
    },
    6: {
        "number_range": [-100, 100],
        "operations": ["ratios", "rates", "integers"],
        "concepts": ["expressions", "equations", "statistical measures"],
    },
    7: {
        "number_range": [-1000, 1000],
        "operations": ["proportions", "percent calculations"],
        "concepts": ["probability", "geometry transformations", "linear relationships"],
    },
    8: {
        "number_range": [-10000, 10000],
        "operations": ["exponents", "scientific notation", "linear equations"],
        "concepts": ["functions", "Pythagorean theorem", "systems of equations"],
    },
}

# Available diagram types for suggestion
AVAILABLE_DIAGRAMS = [
    "right_triangle", "triangle", "rectangle", "square", "circle",
    "rectangular_prism", "cylinder", "cone", "coordinate_grid",
    "plotted_points", "line_graph", "bar_graph", "dot_plot",
    "box_plot", "histogram", "pie_chart", "fraction_circle",
    "fraction_bar", "double_number_line", "tape_diagram",
    "number_line", "angle", "none"
]


class LLMTopicAnalyzer:
    """
    Analyzes custom topics using an LLM to determine appropriate
    problem parameters and diagram types.
    """
    
    def __init__(self):
        self._model = None
        self._initialized = False
    
    def _init_model(self):
        """Lazily initialize the Gemini model."""
        if self._initialized:
            return self._model is not None
            
        self._initialized = True
        
        if not GENAI_AVAILABLE:
            print("[LLMTopicAnalyzer] google-generativeai not installed")
            return False
            
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[LLMTopicAnalyzer] GOOGLE_API_KEY not set")
            return False
        
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-2.0-flash")
            return True
        except Exception as e:
            print(f"[LLMTopicAnalyzer] Failed to initialize: {e}")
            return False
    
    def analyze_topic(self, topic: str, grade: int) -> Dict[str, Any]:
        """
        Analyze a custom topic and return structured parameters.
        
        Returns:
            Dictionary with:
            - domain: The math domain (geometry, algebra, etc.)
            - subskills: List of specific skills covered
            - diagram_type: Suggested diagram type or "none"
            - answer_diagram_type: Diagram type for answer key (if students create)
            - number_range: Appropriate number range for this grade/topic
            - problem_style: "word_problem", "computation", "visual", etc.
            - scaffolds: Suggested scaffolding approaches
        """
        # Try LLM analysis first
        if self._init_model():
            result = self._analyze_with_llm(topic, grade)
            if result:
                return result
        
        # Fall back to heuristic analysis
        return self._analyze_with_heuristics(topic, grade)
    
    def _analyze_with_llm(self, topic: str, grade: int) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze the topic."""
        grade_info = GRADE_CHARACTERISTICS.get(grade, GRADE_CHARACTERISTICS[6])
        
        prompt = f"""Analyze this math topic for grade {grade} students and respond with ONLY valid JSON.

Topic: "{topic}"

Grade {grade} typically covers:
- Number range: {grade_info['number_range']}
- Operations: {', '.join(grade_info['operations'])}
- Concepts: {', '.join(grade_info['concepts'])}

Available diagram types: {', '.join(AVAILABLE_DIAGRAMS)}

Respond with this exact JSON structure (no markdown, no explanation):
{{
    "domain": "one of: geometry, fractions, algebra, statistics, ratios, measurement, number_operations, or the most appropriate domain",
    "subskills": ["list", "of", "specific", "skills"],
    "diagram_type": "best diagram type from the list, or 'none' if no diagram helps the PROBLEM",
    "answer_diagram_type": "diagram type for ANSWER KEY if students CREATE diagrams, otherwise 'none'",
    "number_range": [min, max],
    "problem_style": "word_problem, computation, visual, or mixed",
    "scaffolds": ["helpful", "scaffolding", "suggestions"],
    "is_diagram_creation": true or false (whether students CREATE a diagram vs analyze one)
}}

Important:
- If this topic involves students CREATING diagrams (like "make a bar graph" or "plot points"), set diagram_type to "none" and put the diagram in answer_diagram_type
- If this topic involves students ANALYZING given diagrams (like "find the area of this triangle"), put the diagram in diagram_type
- Use grade-appropriate number ranges
"""
        
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up response - remove markdown code blocks if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            
            result = json.loads(text)
            
            # Validate and clean the result
            return self._validate_analysis(result, grade)
            
        except Exception as e:
            print(f"[LLMTopicAnalyzer] LLM analysis failed: {e}")
            return None
    
    def _validate_analysis(self, result: Dict[str, Any], grade: int) -> Dict[str, Any]:
        """Validate and clean up LLM analysis result."""
        grade_info = GRADE_CHARACTERISTICS.get(grade, GRADE_CHARACTERISTICS[6])
        
        # Ensure required fields exist with defaults
        validated = {
            "domain": result.get("domain", "unknown"),
            "subskills": result.get("subskills", []),
            "diagram_type": result.get("diagram_type", "none"),
            "answer_diagram_type": result.get("answer_diagram_type", "none"),
            "number_range": result.get("number_range", grade_info["number_range"]),
            "problem_style": result.get("problem_style", "word_problem"),
            "scaffolds": result.get("scaffolds", []),
            "is_diagram_creation": result.get("is_diagram_creation", False),
            "llm_analyzed": True,  # Flag that LLM was used
        }
        
        # Validate diagram types
        if validated["diagram_type"] not in AVAILABLE_DIAGRAMS:
            validated["diagram_type"] = "none"
        if validated["answer_diagram_type"] not in AVAILABLE_DIAGRAMS:
            validated["answer_diagram_type"] = "none"
        
        # Ensure number_range is valid
        if not isinstance(validated["number_range"], list) or len(validated["number_range"]) != 2:
            validated["number_range"] = grade_info["number_range"]
        
        return validated
    
    def _analyze_with_heuristics(self, topic: str, grade: int) -> Dict[str, Any]:
        """
        Fallback heuristic analysis when LLM is unavailable.
        Uses keyword matching and grade-level defaults.
        """
        text = topic.lower()
        grade_info = GRADE_CHARACTERISTICS.get(grade, GRADE_CHARACTERISTICS[6])
        
        # Detect domain from keywords
        domain = "unknown"
        domain_keywords = {
            "geometry": ["triangle", "rectangle", "circle", "area", "perimeter", "volume", "angle", "shape", 
                        "polygon", "quadrilateral", "parallelogram", "trapezoid", "prism", "cylinder", "cone", 
                        "sphere", "pythagorean", "congruent", "similar", "transformation", "reflection", "rotation"],
            "fractions": ["fraction", "numerator", "denominator", "mixed number", "improper fraction", 
                         "equivalent", "simplify", "reduce", "common denominator"],
            "algebra": ["equation", "variable", "expression", "solve", "linear", "quadratic", "inequality",
                       "slope", "intercept", "function", "system", "polynomial", "factor", "exponent"],
            "statistics": ["mean", "median", "mode", "graph", "plot", "data", "probability", "range",
                          "distribution", "sample", "survey", "outcome", "experiment", "event", "random"],
            "ratios": ["ratio", "proportion", "rate", "percent", "percentage", "scale", "unit rate",
                      "discount", "tax", "tip", "interest", "markup", "markdown", "commission"],
            "measurement": ["measure", "length", "weight", "capacity", "time", "convert", "elapsed",
                           "metric", "customary", "unit", "ruler", "protractor", "scale"],
            "number_operations": ["add", "subtract", "multiply", "divide", "operation", "integer", 
                                 "negative", "absolute value", "order of operations", "place value",
                                 "rounding", "estimation", "decimal", "whole number"],
        }
        
        for dom, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domain = dom
                break
        
        # Detect if this is a diagram creation topic
        creation_keywords = ["create", "make", "draw", "plot", "construct", "graph"]
        is_creation = any(kw in text for kw in creation_keywords)
        
        # Suggest diagram type based on keywords
        diagram_type = "none"
        answer_diagram_type = "none"
        
        diagram_keywords = {
            "triangle": ["triangle"],
            "rectangle": ["rectangle", "rectangular"],
            "circle": ["circle", "radius", "diameter"],
            "bar_graph": ["bar graph", "bar chart"],
            "line_graph": ["line graph", "line plot"],
            "pie_chart": ["pie chart", "circle graph"],
            "coordinate_grid": ["coordinate", "ordered pair", "quadrant"],
            "dot_plot": ["dot plot"],
            "box_plot": ["box plot", "box and whisker"],
            "histogram": ["histogram"],
            "fraction_circle": ["fraction"],
            "double_number_line": ["number line", "double number"],
        }
        
        for diag, keywords in diagram_keywords.items():
            if any(kw in text for kw in keywords):
                if is_creation:
                    answer_diagram_type = diag
                else:
                    diagram_type = diag
                break
        
        # Extract subskills
        subskills = []
        skill_keywords = ["word problem", "solving", "calculating", "finding", "comparing", "converting"]
        for skill in skill_keywords:
            if skill in text:
                subskills.append(skill.replace(" ", "_"))
        
        return {
            "domain": domain,
            "subskills": subskills or [topic.replace(" ", "_")],
            "diagram_type": diagram_type,
            "answer_diagram_type": answer_diagram_type,
            "number_range": grade_info["number_range"],
            "problem_style": "word_problem" if "word problem" in text else "mixed",
            "scaffolds": [],
            "is_diagram_creation": is_creation,
            "llm_analyzed": False,  # Flag that heuristics were used
        }


# Singleton instance for reuse
_analyzer_instance = None

def get_topic_analyzer() -> LLMTopicAnalyzer:
    """Get the singleton topic analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = LLMTopicAnalyzer()
    return _analyzer_instance


def analyze_custom_topic(topic: str, grade: int) -> Dict[str, Any]:
    """
    Convenience function to analyze a custom topic.
    
    Args:
        topic: The custom topic text
        grade: Target grade level (1-8)
        
    Returns:
        Analysis dictionary with domain, diagram suggestions, etc.
    """
    analyzer = get_topic_analyzer()
    return analyzer.analyze_topic(topic, grade)


if __name__ == "__main__":
    # Test the analyzer
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    test_cases = [
        ("compound interest", 8),
        ("elapsed time word problems", 3),
        ("create a bar graph from data", 5),
        ("area of triangles", 6),
        ("two-step equations", 7),
        ("probability experiments", 7),
    ]
    
    print("LLM Topic Analyzer Tests")
    print("=" * 60)
    
    for topic, grade in test_cases:
        print(f"\nTopic: '{topic}' (Grade {grade})")
        result = analyze_custom_topic(topic, grade)
        print(f"  Domain: {result['domain']}")
        print(f"  Diagram: {result['diagram_type']}")
        print(f"  Answer Diagram: {result['answer_diagram_type']}")
        print(f"  Number Range: {result['number_range']}")
        print(f"  LLM Analyzed: {result['llm_analyzed']}")

