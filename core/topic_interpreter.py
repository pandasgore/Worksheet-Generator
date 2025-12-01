
"""
topic_interpreter.py

This module interprets ANY teacher-provided text input describing a math concept.
Its job is to:
- Understand what mathematical domain the concept belongs to (fractions, algebra, geometry, etc.)
- Identify the underlying subskills (e.g., "adding fractions", "solving equations")
- Validate whether the concept fits the student’s grade level
- Produce a structured interpretation for the problem_planner to use

"""

import re
from core.grade_progression import GRADE_PROGRESSION, get_grade_info


META_GRADE_KEYS = {"description", "reference"}
KNOWN_DOMAINS = sorted(
    {
        key
        for grade in GRADE_PROGRESSION.values()
        for key in grade.keys()
        if key not in META_GRADE_KEYS
    }
)

# Build a reverse map from skill strings to domains based on GRADE_PROGRESSION
SKILL_TO_DOMAIN = {}
for grade_data in GRADE_PROGRESSION.values():
    for domain, content in grade_data.items():
        if domain in META_GRADE_KEYS:
            continue
        
        # Helper to register skills
        def register_skills(data, dom):
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        # Normalize: "statistical_questions_variability" -> "statistical questions variability"
                        clean_skill = item.replace("_", " ").lower()
                        SKILL_TO_DOMAIN[clean_skill] = dom
            elif isinstance(data, dict):
                for k, v in data.items():
                    if k not in ["range", "description", "reference"]:
                        # Register the key itself as a skill (e.g. "rounding", "place_value")
                        clean_key = k.replace("_", " ").lower()
                        SKILL_TO_DOMAIN[clean_key] = dom
                        # Recurse
                        register_skills(v, dom)

        register_skills(content, domain)


DOMAIN_KEYWORDS = {
    "numbers": [
        "count", "counting", "count to", "number line", "whole number",
        "place value", "round", "integer", "absolute value",
        "scientific notation"
    ],
    "operations": [
        "add", "addition", "sum", "subtract", "subtraction", "difference",
        "plus", "minus", "multiply", "multiplication", "product",
        "division", "divide", "dividing", "quotient", "mixed operation"
    ],
    "fractions": [
        "fraction", "fractions", "numerator", "denominator",
        "mixed number", "improper", "equivalent", "simplify",
        "halves", "thirds", "fourths", "unit fraction"
    ],
    "decimals": [
        "decimal", "decimals", "decimal point", "tenths",
        "hundredths", "thousandths", "place to the right"
    ],
    "geometry": [
        "area", "perimeter", "volume", "surface area", "shapes",
        "triangle", "rectangle", "quadrilateral", "polygon",
        "line", "segment", "ray", "angle", "coordinate plane",
        "transform", "rotation", "reflection", "dilation"
    ],
    "measurement": [
        "measure", "measurement", "length", "width", "height",
        "elapsed time", "clock", "weight", "mass", "capacity",
        "temperature", "convert units", "unit conversion"
    ],
    "data": [
        "data", "table", "chart", "graph", "bar graph",
        "picture graph", "line plot", "frequency table"
    ],
    "algebra": [
        "equation", "expression", "variable", "solve",
        "evaluate", "two-step", "multi-step", "substitute",
        "linear", "quadratic", "inequality"
    ],
    "algebraic_thinking": [
        "pattern", "pattern rule", "unknown addend",
        "true or false equation", "input output table",
        "number pattern", "function machine"
    ],
    "ratios": [
        "ratio", "ratios", "rate", "rates", "unit rate", "percent",
        "proportion", "proportions", "proportional",
        "constant of proportionality", "scale drawing",
        "scale drawings", "double number line"
    ],
    "statistics": [
        "mean", "median", "mode", "dot plot", "histogram",
        "box plot", "scatter plot", "two-way table",
        "sample", "population", "probability", "chance",
        "random", "distribution", "trend line", "variability", "statistical question"
    ],
    "functions": [
        "function", "function notation", "f(x)", "input-output",
        "rate of change", "linear function", "nonlinear function",
        "bivariate", "dependent variable", "independent variable"
    ]
}

for domain in KNOWN_DOMAINS:
    DOMAIN_KEYWORDS.setdefault(domain, [])


def get_grade_domains(grade_info: dict):
    """
    Returns the set of domains explicitly supported in the grade specification.
    """
    return {
        key for key in grade_info.keys()
        if key not in META_GRADE_KEYS
    }


def summarize_domain_alignment(requested_domains, grade_domains):
    """
    Builds a diagnostic summary so downstream components can understand
    which requested domains are recognized and supported.
    """
    recognized = [dom for dom in requested_domains if dom in KNOWN_DOMAINS]
    matched = sorted(set(recognized) & grade_domains)
    missing = sorted(set(recognized) - grade_domains)

    return {
        "requested": requested_domains,
        "recognized": recognized,
        "available": sorted(grade_domains),
        "matched": matched,
        "missing": missing,
        "unrecognized": [
            dom for dom in requested_domains
            if dom not in KNOWN_DOMAINS and dom != "unknown"
        ]
    }


def determine_domain(concept_text: str):
    """
    Determines the mathematical domain(s) the concept belongs to.
    Returns a list of matched domain names.
    """

    text = concept_text.lower().strip()
    matched_domains = []

    # 1. Exact match against known skills from GRADE_PROGRESSION
    #    This handles cases like "statistical questions variability" perfectly.
    if text in SKILL_TO_DOMAIN:
        matched_domains.append(SKILL_TO_DOMAIN[text])

    # 2. Fuzzy/Keyword matching
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain in matched_domains:
            continue # Already matched this domain
            
        for kw in keywords:
            escaped_kw = re.escape(kw.strip())
            # Use boundary check, but handle non-word chars
            pattern = rf"(?<!\w){escaped_kw}(?!\w)"
            if re.search(pattern, text):
                matched_domains.append(domain)
                break  # move to next domain after first hit

    # If nothing matches, fallback to "unknown"
    return matched_domains if matched_domains else ["unknown"]


def extract_subskills(concept_text: str):
    """
    Pulls keywords that describe specific subskills.
    This will help the problem planner design accurate templates.
    """

    text = concept_text.lower()

    subskills = []

    # If the text itself is a known skill, add it as a subskill
    if text in SKILL_TO_DOMAIN:
        subskills.append(text)

    # Example pattern extraction (expand later)
    if "adding" in text or "add" in text:
        subskills.append("addition")
    if "subtract" in text:
        subskills.append("subtraction")
    if "multiply" in text:
        subskills.append("multiplication")
    if "divide" in text or "division" in text:
        subskills.append("division")
    if "word problem" in text:
        subskills.append("word_problems")

    # Fractions
    if "fraction" in text:
        if "add" in text:
            subskills.append("fraction_addition")
        if "multiply" in text:
            subskills.append("fraction_multiplication")
        if "divide" in text:
            subskills.append("fraction_division")

    # Algebra
    if "equation" in text:
        subskills.append("equations")
    if "expression" in text:
        subskills.append("expressions")

    return list(set(subskills))  # remove duplicates


def detect_diagram_types(concept_text: str, domains: list) -> list:
    """
    Detect what diagram types would be helpful for this concept.
    Returns a list of diagram type strings.
    """
    text = concept_text.lower()
    diagram_types = []
    
    # Geometry - 2D Shapes
    if any(d in domains for d in ["geometry"]) or any(word in text for word in 
           ["area", "perimeter", "triangle", "rectangle", "circle", "polygon", "shape"]):
        
        if "right triangle" in text or "right-triangle" in text:
            diagram_types.append("right_triangle")
        elif "triangle" in text:
            diagram_types.append("triangle")
        
        if "rectangle" in text and "prism" not in text:
            diagram_types.append("rectangle")
        elif "square" in text:
            diagram_types.append("square")
        
        if "parallelogram" in text:
            diagram_types.append("parallelogram")
        
        if "trapezoid" in text:
            diagram_types.append("trapezoid")
        
        if "rhombus" in text:
            diagram_types.append("rhombus")
        
        if "circle" in text or "circumference" in text or "radius" in text or "diameter" in text:
            diagram_types.append("circle")
        
        if "polygon" in text:
            if "regular" in text:
                diagram_types.append("regular_polygon")
            else:
                diagram_types.append("triangle")  # Default to simple polygon
    
    # Geometry - 3D Shapes (check each independently, not elif)
    if any(word in text for word in ["volume", "surface area", "prism", "pyramid", 
                                      "cylinder", "cone", "sphere", "cube", "3d", "three-dimensional"]):
        if "rectangular prism" in text or "box" in text:
            diagram_types.append("rectangular_prism")
        if "triangular prism" in text:
            diagram_types.append("triangular_prism")
        if "cube" in text:
            diagram_types.append("cube")
        if "cylinder" in text or "cylinders" in text:
            diagram_types.append("cylinder")
        if "cone" in text or "cones" in text:
            diagram_types.append("cone")
        if "sphere" in text or "spheres" in text:
            diagram_types.append("sphere")
        if "pyramid" in text or "pyramids" in text:
            diagram_types.append("pyramid")
        # Default to rectangular prism if just "prism" mentioned
        if "prism" in text and "rectangular_prism" not in diagram_types and "triangular_prism" not in diagram_types:
            diagram_types.append("rectangular_prism")
    
    # Angles - use word boundaries to avoid matching "rectangle"
    import re
    if re.search(r'\bangle\b', text) or re.search(r'\bangles\b', text):
        if "transversal" in text or "parallel" in text:
            diagram_types.append("transversal")
        elif "complementary" in text or "supplementary" in text:
            diagram_types.append("angle_pair")
        else:
            diagram_types.append("angle")
    
    # Coordinate Plane - be specific to avoid matching "dot plot", "box plot", etc.
    # Only match actual coordinate plane topics
    coord_keywords = ["coordinate", "coordinate plane", "coordinate grid", "ordered pair", "quadrant", "x-axis", "y-axis"]
    # Exclude statistical plots which use "plot" but aren't coordinate plane diagrams
    is_statistical_plot = any(kw in text for kw in ["dot plot", "box plot", "box and whisker", "scatter plot"])
    
    if any(word in text for word in coord_keywords) and not is_statistical_plot:
        if "point" in text or "ordered pair" in text:
            diagram_types.append("plotted_points")
        elif "line" in text or "linear" in text:
            diagram_types.append("line_graph")
        elif "shape" in text or "polygon" in text:
            diagram_types.append("shape_on_grid")
        else:
            diagram_types.append("coordinate_grid")
    
    # Fractions
    if any(d in domains for d in ["fractions"]) or "fraction" in text:
        if "number line" in text:
            diagram_types.append("fraction_number_line")
        elif "bar" in text or "model" in text or "rectangle" in text:
            diagram_types.append("fraction_bar")
        else:
            diagram_types.append("fraction_circle")
    
    # Statistics & Data - these are typically "student creates" diagrams
    # Only include as reference diagrams if the topic is about READING/INTERPRETING existing diagrams
    # NOT when students need to CREATE them (those go in answer_diagram_types only)
    if any(d in domains for d in ["statistics", "data"]) or any(word in text for word in 
           ["mean", "median", "mode", "graph", "plot", "chart", "data"]):
        # Check if this is about reading/interpreting existing diagrams
        reading_keywords = ["read", "interpret", "analyze", "given", "shown", "from the", "use the", "look at"]
        is_reading_diagram = any(kw in text for kw in reading_keywords)
        
        if is_reading_diagram:
            # These are reference diagrams - show with the problem
            if "box" in text or "whisker" in text:
                diagram_types.append("box_plot")
            elif "dot plot" in text:
                diagram_types.append("dot_plot")
            elif "histogram" in text:
                diagram_types.append("histogram")
            elif "pie" in text or "circle graph" in text:
                diagram_types.append("pie_chart")
            elif "bar" in text:
                diagram_types.append("bar_graph")
        # If not reading, these will be handled as answer_diagram_types only
    
    # Ratios & Proportions - double number lines and tape diagrams are typically "student creates"
    # Don't add to diagram_types (shown with problem) - only to answer_diagram_types
    # Exception: if topic explicitly says to USE an existing diagram
    if any(d in domains for d in ["ratios"]) or any(word in text for word in 
           ["ratio", "proportion", "rate", "percent"]):
        reading_keywords = ["read", "interpret", "given", "shown", "from the", "use the", "look at"]
        is_reading_diagram = any(kw in text for kw in reading_keywords)
        
        if is_reading_diagram:
            if "double number line" in text or "number line" in text:
                diagram_types.append("double_number_line")
            elif "tape" in text or "bar model" in text:
                diagram_types.append("tape_diagram")
    
    # Don't add tape diagram / bar model as a reference diagram by default
    # Students typically create these themselves
    
    # Measurement
    if "ruler" in text or "measure" in text and "length" in text:
        diagram_types.append("ruler")
    if "protractor" in text or ("measure" in text and "angle" in text):
        diagram_types.append("protractor")
    if "scale drawing" in text or "scale model" in text:
        diagram_types.append("scale_drawing")
    
    return list(set(diagram_types))  # Remove duplicates


def detect_answer_diagram_types(concept_text: str) -> list:
    """
    Detect when a topic requires students to CREATE diagrams.
    These diagrams should appear in the answer key as the solution.
    
    Returns a list of diagram type strings for the answer key.
    """
    text = concept_text.lower()
    answer_diagram_types = []
    
    # Keywords that indicate students need to CREATE/DRAW diagrams
    create_keywords = ["create", "construct", "draw", "make", "plot", "graph", "sketch"]
    
    # Check for dot plots
    if "dot plot" in text or "dotplot" in text:
        if any(kw in text for kw in create_keywords) or "plot" in text:
            answer_diagram_types.append("dot_plot")
    
    # Check for box plots / box and whisker
    if "box plot" in text or "box and whisker" in text or "boxplot" in text:
        if any(kw in text for kw in create_keywords) or "plot" in text:
            answer_diagram_types.append("box_plot")
    
    # Check for histograms - include if topic is about histograms
    if "histogram" in text:
        answer_diagram_types.append("histogram")
    
    # Check for bar graphs - include if topic is about bar graphs
    if "bar graph" in text or "bar chart" in text:
        answer_diagram_types.append("bar_graph")
    
    # Check for pie charts - include if topic is about pie charts
    if "pie chart" in text or "circle graph" in text:
        answer_diagram_types.append("pie_chart")
    
    # Check for line graphs/plots - include if topic mentions them
    if "line graph" in text or "line plot" in text:
        answer_diagram_types.append("line_graph")
    
    # Check for coordinate plane plotting
    if any(word in text for word in ["coordinate", "ordered pair", "quadrant"]):
        if any(kw in text for kw in create_keywords + ["plot"]):
            answer_diagram_types.append("plotted_points")
    
    # Check for number lines (including double number lines)
    if "number line" in text:
        if "double" in text:
            answer_diagram_types.append("double_number_line")
        elif any(kw in text for kw in create_keywords):
            answer_diagram_types.append("fraction_number_line")
    
    # Check for tape diagrams - include if topic is about tape diagrams/bar models
    if "tape diagram" in text or "bar model" in text:
        answer_diagram_types.append("tape_diagram")
    
    # Check for fraction models
    if "fraction" in text and any(word in text for word in ["model", "diagram", "represent", "shade"]):
        answer_diagram_types.append("fraction_circle")
    
    return list(set(answer_diagram_types))


def interpret_concept(concept_text: str, grade: int):
    """
    Main entry point.
    - Identifies domain
    - Extracts subskills
    - Checks grade-level compatibility
    - Detects appropriate diagram types
    - Returns structured meaning for the planner
    """

    grade_info = get_grade_info(grade)
    if not grade_info:
        raise ValueError(f"Invalid grade: {grade}. Must be between 1 and 8.")

    domains = determine_domain(concept_text)
    subskills = extract_subskills(concept_text)
    grade_domains = get_grade_domains(grade_info)
    domain_alignment = summarize_domain_alignment(domains, grade_domains)
    grade_supported = bool(domain_alignment["matched"])
    
    # Detect diagram types for this concept (diagrams to show with problems)
    diagram_types = detect_diagram_types(concept_text, domains)
    
    # Detect answer diagram types (diagrams students need to CREATE - show in answer key)
    answer_diagram_types = detect_answer_diagram_types(concept_text)
    
    # If domain is unknown or we have no useful information, try LLM/heuristic analysis
    llm_analysis = None
    if domains == ["unknown"] or (not diagram_types and not answer_diagram_types and not subskills):
        try:
            from core.llm_topic_analyzer import analyze_custom_topic
            llm_analysis = analyze_custom_topic(concept_text, grade)
            
            # Merge insights into our results (works for both LLM and heuristic analysis)
            if llm_analysis:
                # Update domains if analysis found something
                if llm_analysis.get("domain") and llm_analysis["domain"] != "unknown":
                    domains = [llm_analysis["domain"]]
                    domain_alignment = summarize_domain_alignment(domains, grade_domains)
                    grade_supported = bool(domain_alignment["matched"])
                
                # Update diagram types from analysis
                if llm_analysis.get("diagram_type") and llm_analysis["diagram_type"] != "none":
                    diagram_types = [llm_analysis["diagram_type"]]
                if llm_analysis.get("answer_diagram_type") and llm_analysis["answer_diagram_type"] != "none":
                    answer_diagram_types = [llm_analysis["answer_diagram_type"]]
                
                # Update subskills
                if llm_analysis.get("subskills"):
                    subskills = list(set(subskills + llm_analysis["subskills"]))
        except Exception as e:
            print(f"[interpret_concept] Topic analysis failed (will use defaults): {e}")
            llm_analysis = None

    return {
        "raw_text": concept_text,
        "domains": domains,
        "subskills": subskills,
        "grade_supported": grade_supported,
        "grade_details": grade_info,
        "domain_alignment": domain_alignment,
        "diagram_types": diagram_types,
        "answer_diagram_types": answer_diagram_types,
        "llm_analysis": llm_analysis,  # Include LLM analysis for debugging/advanced use
    }


if __name__ == "__main__":
    # Test known skill
    print("Test 1:", interpret_concept("statistical questions variability", grade=6))
    # Test generic
    print("Test 2:", interpret_concept("adding big numbers", grade=4))
