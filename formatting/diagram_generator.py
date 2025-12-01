"""
diagram_generator.py

Comprehensive diagram generation system for math worksheets.
Supports geometry shapes, coordinate planes, fraction models, data visualizations,
and more. All diagrams are generated as high-quality images that can be embedded
inline with problems in both PDF and DOCX formats.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import (
    Circle, Rectangle, Polygon, FancyArrowPatch, Arc, Wedge,
    RegularPolygon, FancyBboxPatch
)
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import numpy as np


class DiagramType(Enum):
    """Supported diagram types."""
    # Geometry - 2D Shapes
    TRIANGLE = "triangle"
    RIGHT_TRIANGLE = "right_triangle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    PARALLELOGRAM = "parallelogram"
    TRAPEZOID = "trapezoid"
    RHOMBUS = "rhombus"
    CIRCLE = "circle"
    SEMICIRCLE = "semicircle"
    REGULAR_POLYGON = "regular_polygon"
    
    # Geometry - 3D Shapes
    RECTANGULAR_PRISM = "rectangular_prism"
    CUBE = "cube"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    PYRAMID = "pyramid"
    TRIANGULAR_PRISM = "triangular_prism"
    
    # Geometry - Angles
    ANGLE = "angle"
    ANGLE_PAIR = "angle_pair"
    TRANSVERSAL = "transversal"
    
    # Coordinate Plane
    COORDINATE_GRID = "coordinate_grid"
    PLOTTED_POINTS = "plotted_points"
    LINE_GRAPH = "line_graph"
    SHAPE_ON_GRID = "shape_on_grid"
    
    # Fractions
    FRACTION_CIRCLE = "fraction_circle"
    FRACTION_BAR = "fraction_bar"
    FRACTION_NUMBER_LINE = "fraction_number_line"
    
    # Data & Statistics
    BAR_GRAPH = "bar_graph"
    DOT_PLOT = "dot_plot"
    BOX_PLOT = "box_plot"
    HISTOGRAM = "histogram"
    PIE_CHART = "pie_chart"
    
    # Ratios & Proportions
    DOUBLE_NUMBER_LINE = "double_number_line"
    TAPE_DIAGRAM = "tape_diagram"
    
    # Measurement
    RULER = "ruler"
    PROTRACTOR = "protractor"
    SCALE_DRAWING = "scale_drawing"


@dataclass
class DiagramSpec:
    """Specification for generating a diagram."""
    diagram_type: DiagramType
    params: Dict[str, Any] = field(default_factory=dict)
    width: float = 3.0  # inches
    height: float = 2.5  # inches
    show_labels: bool = True
    show_measurements: bool = True
    title: Optional[str] = None
    
    # Minimum and maximum sizes for readability
    MIN_WIDTH = 2.5
    MAX_WIDTH = 6.0
    MIN_HEIGHT = 2.0
    MAX_HEIGHT = 5.0
    
    def __post_init__(self):
        """Validate and normalize the diagram specification."""
        # Ensure minimum size for readability
        self.width = max(self.MIN_WIDTH, min(self.MAX_WIDTH, self.width))
        self.height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, self.height))
        
        # Normalize parameters based on diagram type
        self._normalize_params()
    
    def _normalize_params(self):
        """Normalize parameters to ensure readable diagram output."""
        # Prevent extremely large or small values that cause label issues
        numeric_params = ['base', 'height', 'width', 'radius', 'side', 'length']
        
        for key in numeric_params:
            if key in self.params:
                val = self.params[key]
                if isinstance(val, (int, float)):
                    # Clamp values to reasonable range
                    self.params[key] = max(1, min(20, val))
        
        # Ensure data lists aren't too long for visualizations
        if 'data' in self.params:
            data = self.params['data']
            if isinstance(data, list) and len(data) > 30:
                # Sample the data to prevent overcrowding
                step = len(data) // 30
                self.params['data'] = data[::step][:30]
            elif isinstance(data, dict) and len(data) > 10:
                # Limit categories for bar graphs/pie charts
                items = list(data.items())[:10]
                self.params['data'] = dict(items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_type": self.diagram_type.value,
            "params": self.params,
            "width": self.width,
            "height": self.height,
            "show_labels": self.show_labels,
            "show_measurements": self.show_measurements,
            "title": self.title,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagramSpec":
        return cls(
            diagram_type=DiagramType(data["diagram_type"]),
            params=data.get("params", {}),
            width=data.get("width", 3.0),
            height=data.get("height", 2.5),
            show_labels=data.get("show_labels", True),
            show_measurements=data.get("show_measurements", True),
            title=data.get("title"),
        )


class DiagramGenerator:
    """
    Generates mathematical diagrams for worksheets.
    
    Usage:
        generator = DiagramGenerator()
        spec = DiagramSpec(
            diagram_type=DiagramType.RIGHT_TRIANGLE,
            params={"base": 6, "height": 8, "show_right_angle": True}
        )
        image_bytes = generator.generate(spec)
    """
    
    # Color scheme for consistent, professional appearance
    COLORS = {
        "primary": "#2563eb",      # Blue
        "secondary": "#7c3aed",    # Purple
        "accent": "#f59e0b",       # Amber
        "success": "#10b981",      # Green
        "shape_fill": "#dbeafe",   # Light blue
        "shape_stroke": "#1e40af", # Dark blue
        "grid": "#e5e7eb",         # Light gray
        "axis": "#374151",         # Dark gray
        "text": "#1f2937",         # Near black
        "highlight": "#fef3c7",    # Light yellow
        "label_bg": "#ffffff",     # White background for labels
    }
    
    # Standard label configuration for readability
    LABEL_CONFIG = {
        "fontsize": 10,
        "fontweight": "bold",
        "ha": "center",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        }
    }
    
    # Smaller label for tight spaces
    SMALL_LABEL_CONFIG = {
        "fontsize": 9,
        "fontweight": "normal",
        "ha": "center",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        }
    }
    
    def __init__(self):
        # Configure matplotlib for high-quality output
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'font.size': 10,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2,
        })
        # Track label positions for collision avoidance
        self._label_positions = []
    
    def _reset_labels(self):
        """Reset label tracking for new diagram."""
        self._label_positions = []
    
    def _add_label(self, ax, text: str, x: float, y: float, color: str = None, 
                   config: dict = None, offset_direction: str = "auto"):
        """
        Add a label with automatic positioning to avoid overlaps.
        
        Args:
            ax: Matplotlib axes
            text: Label text
            x, y: Target position
            color: Text color (defaults to COLORS["text"])
            config: Label configuration dict (defaults to LABEL_CONFIG)
            offset_direction: "auto", "up", "down", "left", "right", or None for no offset
        """
        if config is None:
            config = self.LABEL_CONFIG.copy()
        else:
            config = {**self.LABEL_CONFIG, **config}
        
        if color:
            config["color"] = color
        else:
            config["color"] = self.COLORS["text"]
        
        # Calculate offset to avoid collisions
        final_x, final_y = x, y
        if offset_direction != "none":
            final_x, final_y = self._find_clear_position(x, y, text, offset_direction)
        
        # Add the annotation
        ax.annotate(text, (final_x, final_y), **config)
        
        # Track this label's position
        self._label_positions.append((final_x, final_y, len(text)))
    
    def _find_clear_position(self, x: float, y: float, text: str, 
                              direction: str = "auto") -> Tuple[float, float]:
        """Find a position that doesn't overlap with existing labels."""
        # Estimate label size (rough approximation)
        label_width = len(text) * 0.08
        label_height = 0.15
        
        # Check for collisions and adjust
        for lx, ly, llen in self._label_positions:
            other_width = llen * 0.08
            # Check if positions overlap
            if (abs(x - lx) < (label_width + other_width) / 2 and 
                abs(y - ly) < label_height * 2):
                # Offset based on direction
                if direction == "auto":
                    # Try to move away from the other label
                    if y >= ly:
                        y += label_height * 1.5
                    else:
                        y -= label_height * 1.5
                elif direction == "up":
                    y += label_height * 1.5
                elif direction == "down":
                    y -= label_height * 1.5
                elif direction == "left":
                    x -= label_width
                elif direction == "right":
                    x += label_width
        
        return x, y
    
    def _add_measurement_label(self, ax, text: str, x: float, y: float, 
                                rotation: float = 0, color: str = None):
        """Add a measurement label with white background for readability."""
        config = {
            "fontsize": 11,
            "fontweight": "bold",
            "ha": "center",
            "va": "center",
            "rotation": rotation,
            "color": color or self.COLORS["accent"],
            "bbox": {
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": self.COLORS["accent"],
                "alpha": 0.95,
                "linewidth": 1,
            }
        }
        ax.annotate(text, (x, y), **config)
    
    def generate(self, spec: DiagramSpec, format: str = "png") -> bytes:
        """
        Generate a diagram based on the specification.
        
        Args:
            spec: DiagramSpec describing what to draw
            format: Output format ('png', 'pdf', 'svg')
            
        Returns:
            Image data as bytes
        """
        # Reset label tracking for new diagram
        self._reset_labels()
        
        # Auto-adjust size based on diagram complexity
        adjusted_width, adjusted_height = self._calculate_optimal_size(spec)
        
        # Create figure with extra padding for labels
        fig, ax = plt.subplots(figsize=(adjusted_width, adjusted_height))
        
        # Route to appropriate drawing method
        draw_method = self._get_draw_method(spec.diagram_type)
        draw_method(ax, spec)
        
        # Add title if specified
        if spec.title:
            ax.set_title(spec.title, fontsize=11, fontweight='bold', 
                        color=self.COLORS["text"], pad=15)
        
        # Ensure tight layout with padding for labels
        fig.tight_layout(pad=0.5)
        
        # Save to bytes with higher DPI for clarity
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def _calculate_optimal_size(self, spec: DiagramSpec) -> Tuple[float, float]:
        """Calculate optimal diagram size based on content complexity."""
        base_width = spec.width
        base_height = spec.height
        
        # Adjust for data-heavy diagrams
        if spec.diagram_type in [DiagramType.BAR_GRAPH, DiagramType.DOT_PLOT, 
                                  DiagramType.HISTOGRAM, DiagramType.PIE_CHART]:
            data = spec.params.get("data", {})
            if isinstance(data, dict):
                num_items = len(data)
            elif isinstance(data, list):
                num_items = len(set(data))
            else:
                num_items = 5
            
            # Expand width for many categories
            if num_items > 5:
                base_width = min(6.0, base_width * (1 + (num_items - 5) * 0.1))
        
        # Adjust for complex geometry
        if spec.diagram_type in [DiagramType.RECTANGULAR_PRISM, DiagramType.TRIANGULAR_PRISM,
                                  DiagramType.PYRAMID, DiagramType.TRANSVERSAL]:
            base_width = max(base_width, 3.5)
            base_height = max(base_height, 3.0)
        
        # Adjust for coordinate grids
        if spec.diagram_type in [DiagramType.COORDINATE_GRID, DiagramType.PLOTTED_POINTS,
                                  DiagramType.LINE_GRAPH, DiagramType.SHAPE_ON_GRID]:
            x_range = spec.params.get("x_range", (-5, 5))
            y_range = spec.params.get("y_range", (-5, 5))
            x_span = abs(x_range[1] - x_range[0])
            y_span = abs(y_range[1] - y_range[0])
            
            # Maintain aspect ratio for grids
            if x_span > y_span:
                base_width = max(base_width, 3.5)
            else:
                base_height = max(base_height, 3.0)
        
        return base_width, base_height
    
    def generate_to_file(self, spec: DiagramSpec, path: Union[str, Path], 
                         format: str = "png") -> Path:
        """Generate diagram and save to file."""
        path = Path(path)
        data = self.generate(spec, format)
        path.write_bytes(data)
        return path
    
    def _get_draw_method(self, diagram_type: DiagramType):
        """Get the appropriate drawing method for a diagram type."""
        methods = {
            # 2D Geometry
            DiagramType.TRIANGLE: self._draw_triangle,
            DiagramType.RIGHT_TRIANGLE: self._draw_right_triangle,
            DiagramType.RECTANGLE: self._draw_rectangle,
            DiagramType.SQUARE: self._draw_square,
            DiagramType.PARALLELOGRAM: self._draw_parallelogram,
            DiagramType.TRAPEZOID: self._draw_trapezoid,
            DiagramType.RHOMBUS: self._draw_rhombus,
            DiagramType.CIRCLE: self._draw_circle,
            DiagramType.SEMICIRCLE: self._draw_semicircle,
            DiagramType.REGULAR_POLYGON: self._draw_regular_polygon,
            
            # 3D Geometry
            DiagramType.RECTANGULAR_PRISM: self._draw_rectangular_prism,
            DiagramType.CUBE: self._draw_cube,
            DiagramType.CYLINDER: self._draw_cylinder,
            DiagramType.CONE: self._draw_cone,
            DiagramType.SPHERE: self._draw_sphere,
            DiagramType.PYRAMID: self._draw_pyramid,
            DiagramType.TRIANGULAR_PRISM: self._draw_triangular_prism,
            
            # Angles
            DiagramType.ANGLE: self._draw_angle,
            DiagramType.ANGLE_PAIR: self._draw_angle_pair,
            DiagramType.TRANSVERSAL: self._draw_transversal,
            
            # Coordinate Plane
            DiagramType.COORDINATE_GRID: self._draw_coordinate_grid,
            DiagramType.PLOTTED_POINTS: self._draw_plotted_points,
            DiagramType.LINE_GRAPH: self._draw_line_graph,
            DiagramType.SHAPE_ON_GRID: self._draw_shape_on_grid,
            
            # Fractions
            DiagramType.FRACTION_CIRCLE: self._draw_fraction_circle,
            DiagramType.FRACTION_BAR: self._draw_fraction_bar,
            DiagramType.FRACTION_NUMBER_LINE: self._draw_fraction_number_line,
            
            # Data & Statistics
            DiagramType.BAR_GRAPH: self._draw_bar_graph,
            DiagramType.DOT_PLOT: self._draw_dot_plot,
            DiagramType.BOX_PLOT: self._draw_box_plot,
            DiagramType.HISTOGRAM: self._draw_histogram,
            DiagramType.PIE_CHART: self._draw_pie_chart,
            
            # Ratios
            DiagramType.DOUBLE_NUMBER_LINE: self._draw_double_number_line,
            DiagramType.TAPE_DIAGRAM: self._draw_tape_diagram,
            
            # Measurement
            DiagramType.RULER: self._draw_ruler,
            DiagramType.PROTRACTOR: self._draw_protractor,
            DiagramType.SCALE_DRAWING: self._draw_scale_drawing,
        }
        return methods.get(diagram_type, self._draw_placeholder)
    
    # =========================================================================
    # 2D GEOMETRY SHAPES
    # =========================================================================
    
    def _draw_triangle(self, ax, spec: DiagramSpec):
        """Draw a general triangle with optional measurements."""
        params = spec.params
        # Default to a scalene triangle
        vertices = params.get("vertices", [(0, 0), (4, 0), (1.5, 3)])
        labels = params.get("vertex_labels", ["A", "B", "C"])
        side_lengths = params.get("side_lengths", None)
        
        # Create triangle patch
        triangle = Polygon(vertices, fill=True,
                          facecolor=self.COLORS["shape_fill"],
                          edgecolor=self.COLORS["shape_stroke"],
                          linewidth=2)
        ax.add_patch(triangle)
        
        # Add vertex labels
        if spec.show_labels:
            offsets = [(-0.3, -0.3), (0.3, -0.3), (0, 0.3)]
            for i, (vertex, label) in enumerate(zip(vertices, labels)):
                ax.annotate(label, vertex, 
                           xytext=(vertex[0] + offsets[i][0], vertex[1] + offsets[i][1]),
                           fontsize=12, fontweight='bold', color=self.COLORS["text"])
        
        # Add side length labels
        if spec.show_measurements and side_lengths:
            self._add_side_labels(ax, vertices, side_lengths)
        
        self._setup_shape_axes(ax, vertices)
    
    def _draw_right_triangle(self, ax, spec: DiagramSpec):
        """Draw a right triangle with right angle marker and clear labels."""
        params = spec.params
        base = params.get("base", 6)
        height = params.get("height", 4)
        show_right_angle = params.get("show_right_angle", True)
        labels = params.get("labels", {"base": str(base), "height": str(height)})
        
        # Scale factor to normalize diagram size
        scale = max(base, height)
        padding = scale * 0.2
        
        vertices = [(0, 0), (base, 0), (0, height)]
        
        # Create triangle
        triangle = Polygon(vertices, fill=True,
                          facecolor=self.COLORS["shape_fill"],
                          edgecolor=self.COLORS["shape_stroke"],
                          linewidth=2.5)
        ax.add_patch(triangle)
        
        # Add right angle marker
        if show_right_angle:
            marker_size = min(base, height) * 0.1
            right_angle = Rectangle((0, 0), marker_size, marker_size,
                                    fill=False, edgecolor=self.COLORS["shape_stroke"],
                                    linewidth=1.5)
            ax.add_patch(right_angle)
        
        # Add measurement labels with white backgrounds
        if spec.show_measurements:
            # Base label - below and centered
            base_label = labels.get("base", str(base))
            ax.annotate(base_label, (base/2, -padding * 0.7),
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
            
            # Height label - left of the shape
            height_label = labels.get("height", str(height))
            ax.annotate(height_label, (-padding * 0.7, height/2),
                       ha='right', va='center', fontsize=12, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
            
            # Hypotenuse label if provided - along the hypotenuse
            if "hypotenuse" in labels:
                hyp_label = labels["hypotenuse"]
                # Position along hypotenuse with offset
                mid_x = base/2 + padding * 0.4
                mid_y = height/2 + padding * 0.4
                ax.annotate(hyp_label, (mid_x, mid_y),
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           color=self.COLORS["secondary"],
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor=self.COLORS["secondary"], alpha=0.95, linewidth=1.5))
        
        # Vertex labels with clear positioning
        if spec.show_labels:
            vertex_labels = params.get("vertex_labels", ["A", "B", "C"])
            # Position labels outside the triangle
            positions = [
                (vertices[0][0] - padding * 0.5, vertices[0][1] - padding * 0.5),  # Bottom left
                (vertices[1][0] + padding * 0.5, vertices[1][1] - padding * 0.5),  # Bottom right
                (vertices[2][0] - padding * 0.5, vertices[2][1] + padding * 0.5),  # Top
            ]
            for pos, label in zip(positions, vertex_labels):
                ax.annotate(label, pos, fontsize=13, fontweight='bold',
                           color=self.COLORS["text"],
                           bbox=dict(boxstyle='circle,pad=0.2', facecolor='white',
                                    edgecolor=self.COLORS["shape_stroke"], alpha=0.95))
        
        self._setup_shape_axes(ax, vertices, padding=padding * 1.5)
    
    def _draw_rectangle(self, ax, spec: DiagramSpec):
        """Draw a rectangle with dimensions and clear labels."""
        params = spec.params
        width = params.get("width", 6)
        height = params.get("height", 4)
        
        # Scale factor for padding
        scale = max(width, height)
        padding = scale * 0.15
        
        rect = Rectangle((0, 0), width, height, fill=True,
                         facecolor=self.COLORS["shape_fill"],
                         edgecolor=self.COLORS["shape_stroke"],
                         linewidth=2.5)
        ax.add_patch(rect)
        
        # Add right angle markers at corners (only bottom-left visible)
        marker_size = min(width, height) * 0.1
        right_angle = Rectangle((0, 0), marker_size, marker_size,
                               fill=False, edgecolor=self.COLORS["shape_stroke"],
                               linewidth=1.5)
        ax.add_patch(right_angle)
        
        if spec.show_measurements:
            # Width label - below the rectangle
            ax.annotate(f"{width}", (width/2, -padding),
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
            
            # Height label - left of the rectangle
            ax.annotate(f"{height}", (-padding, height/2),
                       ha='right', va='center', fontsize=12, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
        
        vertices = [(0, 0), (width, 0), (width, height), (0, height)]
        self._setup_shape_axes(ax, vertices, padding=padding * 2)
    
    def _draw_square(self, ax, spec: DiagramSpec):
        """Draw a square with side length."""
        params = spec.params
        side = params.get("side", 4)
        spec.params["width"] = side
        spec.params["height"] = side
        self._draw_rectangle(ax, spec)
    
    def _draw_parallelogram(self, ax, spec: DiagramSpec):
        """Draw a parallelogram with base, height, and slant."""
        params = spec.params
        base = params.get("base", 6)
        height = params.get("height", 4)
        slant = params.get("slant", 2)
        
        vertices = [(0, 0), (base, 0), (base + slant, height), (slant, height)]
        
        para = Polygon(vertices, fill=True,
                      facecolor=self.COLORS["shape_fill"],
                      edgecolor=self.COLORS["shape_stroke"],
                      linewidth=2)
        ax.add_patch(para)
        
        # Add height line (dashed)
        ax.plot([slant, slant], [0, height], '--', 
               color=self.COLORS["accent"], linewidth=1.5)
        
        if spec.show_measurements:
            ax.annotate(f"{base}", (base/2, -0.5), ha='center', fontsize=11,
                       color=self.COLORS["text"])
            ax.annotate(f"h = {height}", (slant - 0.8, height/2), ha='center',
                       va='center', fontsize=10, color=self.COLORS["accent"], rotation=90)
        
        self._setup_shape_axes(ax, vertices, padding=1.0)
    
    def _draw_trapezoid(self, ax, spec: DiagramSpec):
        """Draw a trapezoid with parallel sides and height."""
        params = spec.params
        base1 = params.get("base1", 8)  # Bottom base
        base2 = params.get("base2", 4)  # Top base
        height = params.get("height", 4)
        
        # Center the top base
        offset = (base1 - base2) / 2
        vertices = [(0, 0), (base1, 0), (base1 - offset, height), (offset, height)]
        
        trap = Polygon(vertices, fill=True,
                      facecolor=self.COLORS["shape_fill"],
                      edgecolor=self.COLORS["shape_stroke"],
                      linewidth=2)
        ax.add_patch(trap)
        
        # Height line
        mid_x = base1 / 2
        ax.plot([mid_x, mid_x], [0, height], '--',
               color=self.COLORS["accent"], linewidth=1.5)
        
        if spec.show_measurements:
            ax.annotate(f"{base1}", (base1/2, -0.5), ha='center', fontsize=11,
                       color=self.COLORS["text"])
            ax.annotate(f"{base2}", (base1/2, height + 0.4), ha='center', fontsize=11,
                       color=self.COLORS["text"])
            ax.annotate(f"h = {height}", (mid_x + 0.5, height/2), ha='left',
                       va='center', fontsize=10, color=self.COLORS["accent"])
        
        self._setup_shape_axes(ax, vertices, padding=1.0)
    
    def _draw_rhombus(self, ax, spec: DiagramSpec):
        """Draw a rhombus with diagonals."""
        params = spec.params
        d1 = params.get("diagonal1", 6)  # Horizontal diagonal
        d2 = params.get("diagonal2", 4)  # Vertical diagonal
        
        vertices = [(0, d2/2), (d1/2, 0), (d1, d2/2), (d1/2, d2)]
        
        rhombus = Polygon(vertices, fill=True,
                         facecolor=self.COLORS["shape_fill"],
                         edgecolor=self.COLORS["shape_stroke"],
                         linewidth=2)
        ax.add_patch(rhombus)
        
        # Draw diagonals
        ax.plot([0, d1], [d2/2, d2/2], '--', color=self.COLORS["accent"], linewidth=1.5)
        ax.plot([d1/2, d1/2], [0, d2], '--', color=self.COLORS["secondary"], linewidth=1.5)
        
        if spec.show_measurements:
            ax.annotate(f"d₁ = {d1}", (d1/2, d2/2 - 0.4), ha='center', fontsize=10,
                       color=self.COLORS["accent"])
            ax.annotate(f"d₂ = {d2}", (d1/2 + 0.5, d2/2), ha='left', fontsize=10,
                       color=self.COLORS["secondary"])
        
        self._setup_shape_axes(ax, vertices, padding=1.0)
    
    def _draw_circle(self, ax, spec: DiagramSpec):
        """Draw a circle with radius/diameter and clear labels."""
        params = spec.params
        radius = params.get("radius", 3)
        show_radius = params.get("show_radius", True)
        show_diameter = params.get("show_diameter", False)
        show_center = params.get("show_center", True)
        
        padding = radius * 0.3
        center = (radius + padding, radius + padding)
        
        circle = Circle(center, radius, fill=True,
                       facecolor=self.COLORS["shape_fill"],
                       edgecolor=self.COLORS["shape_stroke"],
                       linewidth=2.5)
        ax.add_patch(circle)
        
        # Center point
        if show_center:
            ax.plot(*center, 'o', color=self.COLORS["shape_stroke"], markersize=8)
            ax.annotate("O", (center[0] - padding * 0.5, center[1] + padding * 0.5),
                       fontsize=12, fontweight='bold', color=self.COLORS["text"],
                       bbox=dict(boxstyle='circle,pad=0.15', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95))
        
        # Radius line and label
        if show_radius:
            # Draw radius line at an angle to avoid overlap with diameter
            angle = 45 if show_diameter else 0
            rad = math.radians(angle)
            end_x = center[0] + radius * math.cos(rad)
            end_y = center[1] + radius * math.sin(rad)
            
            ax.plot([center[0], end_x], [center[1], end_y],
                   color=self.COLORS["accent"], linewidth=2.5)
            ax.plot(end_x, end_y, 'o', color=self.COLORS["accent"], markersize=6)
            
            if spec.show_measurements:
                # Position label along the radius, outside the circle
                label_x = center[0] + (radius * 0.5 + padding * 0.3) * math.cos(rad)
                label_y = center[1] + (radius * 0.5 + padding * 0.3) * math.sin(rad)
                ax.annotate(f"r = {radius}", (label_x, label_y),
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           color=self.COLORS["accent"],
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
        
        # Diameter line and label
        if show_diameter:
            ax.plot([center[0] - radius, center[0] + radius], [center[1], center[1]],
                   color=self.COLORS["secondary"], linewidth=2.5)
            # End points
            ax.plot([center[0] - radius, center[0] + radius], [center[1], center[1]],
                   'o', color=self.COLORS["secondary"], markersize=6)
            
            if spec.show_measurements:
                # Position label below the diameter line
                ax.annotate(f"d = {2*radius}", (center[0], center[1] - padding),
                           ha='center', va='top', fontsize=12, fontweight='bold',
                           color=self.COLORS["secondary"],
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor=self.COLORS["secondary"], alpha=0.95, linewidth=1.5))
        
        ax.set_xlim(-padding * 0.5, 2*radius + padding * 2.5)
        ax.set_ylim(-padding * 0.5, 2*radius + padding * 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_semicircle(self, ax, spec: DiagramSpec):
        """Draw a semicircle."""
        params = spec.params
        radius = params.get("radius", 4)
        
        # Draw semicircle using arc
        theta = np.linspace(0, np.pi, 100)
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)
        
        ax.fill_between(x, 0, y, color=self.COLORS["shape_fill"])
        ax.plot(x, y, color=self.COLORS["shape_stroke"], linewidth=2)
        ax.plot([0, 2*radius], [0, 0], color=self.COLORS["shape_stroke"], linewidth=2)
        
        if spec.show_measurements:
            ax.annotate(f"r = {radius}", (radius, radius/2), ha='center', fontsize=11,
                       color=self.COLORS["text"])
        
        ax.set_xlim(-0.5, 2*radius + 0.5)
        ax.set_ylim(-0.5, radius + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_regular_polygon(self, ax, spec: DiagramSpec):
        """Draw a regular polygon (pentagon, hexagon, etc.)."""
        params = spec.params
        sides = params.get("sides", 6)
        radius = params.get("radius", 3)
        
        center = (radius + 0.5, radius + 0.5)
        
        polygon = RegularPolygon(center, sides, radius=radius,
                                facecolor=self.COLORS["shape_fill"],
                                edgecolor=self.COLORS["shape_stroke"],
                                linewidth=2)
        ax.add_patch(polygon)
        
        # Calculate side length for label
        if spec.show_measurements:
            side_length = 2 * radius * math.sin(math.pi / sides)
            ax.annotate(f"s ≈ {side_length:.1f}", (center[0], -0.3),
                       ha='center', fontsize=10, color=self.COLORS["text"])
        
        ax.set_xlim(-0.5, 2*radius + 1.5)
        ax.set_ylim(-0.5, 2*radius + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # 3D GEOMETRY SHAPES
    # =========================================================================
    
    def _draw_rectangular_prism(self, ax, spec: DiagramSpec):
        """Draw a 3D rectangular prism (box) in isometric view."""
        params = spec.params
        length = params.get("length", 5)
        width = params.get("width", 3)
        height = params.get("height", 4)
        
        # Isometric projection factors
        iso_x = 0.866  # cos(30°)
        iso_y = 0.5    # sin(30°)
        
        # Front face vertices
        front = [(0, 0), (length, 0), (length, height), (0, height)]
        
        # Back face offset
        dx = width * iso_x
        dy = width * iso_y
        back = [(x + dx, y + dy) for x, y in front]
        
        # Draw back edges (dashed for hidden)
        ax.plot([back[0][0], back[1][0]], [back[0][1], back[1][1]], '--',
               color=self.COLORS["shape_stroke"], linewidth=1, alpha=0.5)
        ax.plot([back[0][0], back[3][0]], [back[0][1], back[3][1]], '--',
               color=self.COLORS["shape_stroke"], linewidth=1, alpha=0.5)
        
        # Draw front face
        front_poly = Polygon(front, fill=True,
                            facecolor=self.COLORS["shape_fill"],
                            edgecolor=self.COLORS["shape_stroke"],
                            linewidth=2)
        ax.add_patch(front_poly)
        
        # Draw top face
        top = [front[3], front[2], back[2], back[3]]
        top_poly = Polygon(top, fill=True,
                          facecolor=self.COLORS["highlight"],
                          edgecolor=self.COLORS["shape_stroke"],
                          linewidth=2)
        ax.add_patch(top_poly)
        
        # Draw right face
        right = [front[1], front[2], back[2], back[1]]
        right_poly = Polygon(right, fill=True,
                            facecolor=self.COLORS["shape_fill"],
                            edgecolor=self.COLORS["shape_stroke"],
                            linewidth=2, alpha=0.8)
        ax.add_patch(right_poly)
        
        padding = max(length, height, width) * 0.15
        
        if spec.show_measurements:
            # Length label - below the front face
            ax.annotate(f"l = {length}", (length/2, -padding * 1.2),
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95, linewidth=1.5))
            # Height label - left of the front face
            ax.annotate(f"h = {height}", (-padding * 1.2, height/2),
                       ha='right', va='center', fontsize=11, fontweight='bold',
                       color=self.COLORS["text"], rotation=90,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95, linewidth=1.5))
            # Width label - along the top edge going back
            ax.annotate(f"w = {width}", (length + dx/2 + padding * 0.5, dy/2 + height + padding * 0.3),
                       ha='left', va='bottom', fontsize=11, fontweight='bold',
                       color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95, linewidth=1.5))
        
        ax.set_xlim(-padding * 2.5, length + dx + padding * 2)
        ax.set_ylim(-padding * 2.5, height + dy + padding * 2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_cube(self, ax, spec: DiagramSpec):
        """Draw a cube."""
        side = spec.params.get("side", 4)
        spec.params["length"] = side
        spec.params["width"] = side
        spec.params["height"] = side
        self._draw_rectangular_prism(ax, spec)
    
    def _draw_cylinder(self, ax, spec: DiagramSpec):
        """Draw a cylinder with clear measurement labels."""
        params = spec.params
        radius = params.get("radius", 2)
        height = params.get("height", 5)
        
        padding = max(radius, height) * 0.15
        
        # Draw bottom ellipse
        ellipse_height = radius * 0.3
        bottom = patches.Ellipse((radius, 0), 2*radius, ellipse_height,
                                fill=True, facecolor=self.COLORS["shape_fill"],
                                edgecolor=self.COLORS["shape_stroke"], linewidth=2.5)
        ax.add_patch(bottom)
        
        # Draw sides
        ax.plot([0, 0], [0, height], color=self.COLORS["shape_stroke"], linewidth=2.5)
        ax.plot([2*radius, 2*radius], [0, height], color=self.COLORS["shape_stroke"], linewidth=2.5)
        
        # Draw top ellipse
        top = patches.Ellipse((radius, height), 2*radius, ellipse_height,
                             fill=True, facecolor=self.COLORS["highlight"],
                             edgecolor=self.COLORS["shape_stroke"], linewidth=2.5)
        ax.add_patch(top)
        
        # Draw radius line on top
        ax.plot([radius, 2*radius], [height, height], '--',
               color=self.COLORS["accent"], linewidth=2)
        ax.plot(2*radius, height, 'o', color=self.COLORS["accent"], markersize=5)
        
        if spec.show_measurements:
            # Radius label - inside the top, centered
            ax.annotate(f"r = {radius}", (radius + radius/2, height + padding * 0.8),
                       ha='center', va='bottom', fontsize=11, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
            # Height label - left side
            ax.annotate(f"h = {height}", (-padding * 1.5, height/2),
                       ha='right', va='center', fontsize=11, fontweight='bold',
                       color=self.COLORS["text"], rotation=90,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95, linewidth=1.5))
        
        ax.set_xlim(-padding * 3, 2*radius + padding * 2)
        ax.set_ylim(-padding * 2, height + padding * 3)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_cone(self, ax, spec: DiagramSpec):
        """Draw a cone with clear measurement labels."""
        params = spec.params
        radius = params.get("radius", 2)
        height = params.get("height", 5)
        
        padding = max(radius, height) * 0.15
        
        # Draw base ellipse
        ellipse_height = radius * 0.3
        base = patches.Ellipse((radius, 0), 2*radius, ellipse_height,
                              fill=True, facecolor=self.COLORS["shape_fill"],
                              edgecolor=self.COLORS["shape_stroke"], linewidth=2.5)
        ax.add_patch(base)
        
        # Draw cone sides
        apex = (radius, height)
        ax.fill([0, 2*radius, apex[0]], [0, 0, apex[1]],
               color=self.COLORS["shape_fill"], edgecolor=self.COLORS["shape_stroke"],
               linewidth=2.5)
        
        # Height line (dashed)
        ax.plot([radius, radius], [0, height], '--',
               color=self.COLORS["accent"], linewidth=2)
        ax.plot(radius, height, 'o', color=self.COLORS["accent"], markersize=6)
        
        if spec.show_measurements:
            # Radius label - below the base
            ax.annotate(f"r = {radius}", (radius, -padding * 1.5),
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["shape_stroke"], alpha=0.95, linewidth=1.5))
            # Height label - right of the height line
            ax.annotate(f"h = {height}", (radius + padding, height/2),
                       ha='left', va='center', fontsize=11, fontweight='bold',
                       color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=1.5))
        
        ax.set_xlim(-padding * 2, 2*radius + padding * 2)
        ax.set_ylim(-padding * 3, height + padding * 2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_sphere(self, ax, spec: DiagramSpec):
        """Draw a sphere with cross-section."""
        params = spec.params
        radius = params.get("radius", 3)
        
        center = (radius + 0.5, radius + 0.5)
        
        # Main circle
        circle = Circle(center, radius, fill=True,
                       facecolor=self.COLORS["shape_fill"],
                       edgecolor=self.COLORS["shape_stroke"],
                       linewidth=2)
        ax.add_patch(circle)
        
        # Equator ellipse to show 3D effect
        ellipse = patches.Ellipse(center, 2*radius, radius*0.4,
                                 fill=False, edgecolor=self.COLORS["shape_stroke"],
                                 linewidth=1, linestyle='--')
        ax.add_patch(ellipse)
        
        # Radius line
        ax.plot([center[0], center[0] + radius], [center[1], center[1]],
               color=self.COLORS["accent"], linewidth=2)
        ax.plot(*center, 'o', color=self.COLORS["shape_stroke"], markersize=5)
        
        if spec.show_measurements:
            ax.annotate(f"r = {radius}", (center[0] + radius/2, center[1] + 0.3),
                       ha='center', fontsize=11, color=self.COLORS["accent"])
        
        ax.set_xlim(-0.5, 2*radius + 1.5)
        ax.set_ylim(-0.5, 2*radius + 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_pyramid(self, ax, spec: DiagramSpec):
        """Draw a square pyramid."""
        params = spec.params
        base = params.get("base", 4)
        height = params.get("height", 5)
        
        # Base as a parallelogram (isometric view)
        iso_factor = 0.5
        base_vertices = [(1, 0), (1 + base, 0), 
                        (1 + base + base*iso_factor, base*iso_factor),
                        (1 + base*iso_factor, base*iso_factor)]
        
        # Apex
        apex = (1 + base/2 + base*iso_factor/2, base*iso_factor/2 + height)
        
        # Draw base (back edges dashed)
        ax.plot([base_vertices[2][0], base_vertices[3][0]], 
               [base_vertices[2][1], base_vertices[3][1]], '--',
               color=self.COLORS["shape_stroke"], linewidth=1, alpha=0.5)
        ax.plot([base_vertices[3][0], base_vertices[0][0]], 
               [base_vertices[3][1], base_vertices[0][1]], '--',
               color=self.COLORS["shape_stroke"], linewidth=1, alpha=0.5)
        
        # Front face
        front = Polygon([base_vertices[0], base_vertices[1], apex],
                       fill=True, facecolor=self.COLORS["shape_fill"],
                       edgecolor=self.COLORS["shape_stroke"], linewidth=2)
        ax.add_patch(front)
        
        # Right face
        right = Polygon([base_vertices[1], base_vertices[2], apex],
                       fill=True, facecolor=self.COLORS["highlight"],
                       edgecolor=self.COLORS["shape_stroke"], linewidth=2, alpha=0.9)
        ax.add_patch(right)
        
        # Height line (dashed)
        base_center = (1 + base/2 + base*iso_factor/2, base*iso_factor/2)
        ax.plot([base_center[0], apex[0]], [base_center[1], apex[1]], '--',
               color=self.COLORS["accent"], linewidth=1.5)
        
        if spec.show_measurements:
            ax.annotate(f"base = {base}", (1 + base/2, -0.4), ha='center', fontsize=10,
                       color=self.COLORS["text"])
            ax.annotate(f"h = {height}", (apex[0] + 0.4, (base_center[1] + apex[1])/2),
                       ha='left', fontsize=10, color=self.COLORS["accent"])
        
        ax.set_xlim(0, 2 + base + base*iso_factor)
        ax.set_ylim(-1, height + base*iso_factor + 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_triangular_prism(self, ax, spec: DiagramSpec):
        """Draw a triangular prism."""
        params = spec.params
        base = params.get("base", 4)
        tri_height = params.get("triangle_height", 3)
        length = params.get("length", 5)
        
        # Isometric offset
        dx = length * 0.7
        dy = length * 0.3
        
        # Front triangle
        front = [(0, 0), (base, 0), (base/2, tri_height)]
        back = [(x + dx, y + dy) for x, y in front]
        
        # Draw back triangle (dashed hidden edges)
        ax.plot([back[0][0], back[1][0]], [back[0][1], back[1][1]], '--',
               color=self.COLORS["shape_stroke"], linewidth=1, alpha=0.5)
        
        # Front face
        front_poly = Polygon(front, fill=True,
                            facecolor=self.COLORS["shape_fill"],
                            edgecolor=self.COLORS["shape_stroke"], linewidth=2)
        ax.add_patch(front_poly)
        
        # Top face
        top = [front[2], front[1], back[1], back[2]]
        top_poly = Polygon(top, fill=True,
                          facecolor=self.COLORS["highlight"],
                          edgecolor=self.COLORS["shape_stroke"], linewidth=2)
        ax.add_patch(top_poly)
        
        # Connect edges
        ax.plot([front[0][0], back[0][0]], [front[0][1], back[0][1]],
               color=self.COLORS["shape_stroke"], linewidth=2)
        
        if spec.show_measurements:
            ax.annotate(f"b = {base}", (base/2, -0.4), ha='center', fontsize=10,
                       color=self.COLORS["text"])
            ax.annotate(f"l = {length}", (base + dx/2 + 0.2, dy/2), ha='left',
                       fontsize=10, color=self.COLORS["text"])
        
        ax.set_xlim(-1, base + dx + 1)
        ax.set_ylim(-1, tri_height + dy + 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # ANGLES
    # =========================================================================
    
    def _draw_angle(self, ax, spec: DiagramSpec):
        """Draw an angle with arc and measurement."""
        params = spec.params
        degrees = params.get("degrees", 45)
        radius = params.get("radius", 3)
        
        # Draw rays
        ax.plot([0, radius], [0, 0], color=self.COLORS["shape_stroke"], linewidth=2)
        end_x = radius * math.cos(math.radians(degrees))
        end_y = radius * math.sin(math.radians(degrees))
        ax.plot([0, end_x], [0, end_y], color=self.COLORS["shape_stroke"], linewidth=2)
        
        # Draw arc
        arc_radius = radius * 0.3
        arc = Arc((0, 0), 2*arc_radius, 2*arc_radius, angle=0,
                 theta1=0, theta2=degrees, color=self.COLORS["accent"], linewidth=2)
        ax.add_patch(arc)
        
        # Angle label
        if spec.show_measurements:
            label_angle = math.radians(degrees / 2)
            label_x = arc_radius * 1.5 * math.cos(label_angle)
            label_y = arc_radius * 1.5 * math.sin(label_angle)
            ax.annotate(f"{degrees}°", (label_x, label_y), ha='center', va='center',
                       fontsize=12, color=self.COLORS["accent"], fontweight='bold')
        
        # Vertex point
        ax.plot(0, 0, 'o', color=self.COLORS["shape_stroke"], markersize=6)
        
        ax.set_xlim(-0.5, radius + 0.5)
        ax.set_ylim(-0.5, max(end_y, 0.5) + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_angle_pair(self, ax, spec: DiagramSpec):
        """Draw complementary or supplementary angles."""
        params = spec.params
        angle1 = params.get("angle1", 60)
        angle2 = params.get("angle2", 30)
        pair_type = params.get("type", "complementary")  # or "supplementary"
        radius = params.get("radius", 3)
        
        # Draw rays
        ax.plot([-radius, radius], [0, 0], color=self.COLORS["shape_stroke"], linewidth=2)
        
        # First angle ray
        end1_x = radius * math.cos(math.radians(angle1))
        end1_y = radius * math.sin(math.radians(angle1))
        ax.plot([0, end1_x], [0, end1_y], color=self.COLORS["shape_stroke"], linewidth=2)
        
        if pair_type == "supplementary":
            # Second angle on opposite side
            end2_x = -radius * math.cos(math.radians(180 - angle1 - angle2))
            end2_y = radius * math.sin(math.radians(180 - angle1 - angle2))
            ax.plot([0, end2_x], [0, end2_y], color=self.COLORS["shape_stroke"], linewidth=2)
        
        # Draw arcs
        arc_radius = radius * 0.25
        arc1 = Arc((0, 0), 2*arc_radius, 2*arc_radius, angle=0,
                  theta1=0, theta2=angle1, color=self.COLORS["accent"], linewidth=2)
        ax.add_patch(arc1)
        
        arc2 = Arc((0, 0), 2*arc_radius*1.3, 2*arc_radius*1.3, angle=0,
                  theta1=angle1, theta2=angle1+angle2, color=self.COLORS["secondary"], linewidth=2)
        ax.add_patch(arc2)
        
        if spec.show_measurements:
            # Angle 1 label
            label1_angle = math.radians(angle1 / 2)
            ax.annotate(f"{angle1}°", (arc_radius*1.8*math.cos(label1_angle), 
                                       arc_radius*1.8*math.sin(label1_angle)),
                       ha='center', fontsize=10, color=self.COLORS["accent"])
            
            # Angle 2 label
            label2_angle = math.radians(angle1 + angle2/2)
            ax.annotate(f"{angle2}°", (arc_radius*2.2*math.cos(label2_angle),
                                       arc_radius*2.2*math.sin(label2_angle)),
                       ha='center', fontsize=10, color=self.COLORS["secondary"])
        
        ax.set_xlim(-radius - 0.5, radius + 0.5)
        ax.set_ylim(-0.5, radius + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_transversal(self, ax, spec: DiagramSpec):
        """Draw parallel lines with a transversal."""
        params = spec.params
        highlight_angles = params.get("highlight_angles", [])
        
        # Draw two parallel lines
        ax.plot([0, 6], [1, 1], color=self.COLORS["shape_stroke"], linewidth=2)
        ax.plot([0, 6], [3, 3], color=self.COLORS["shape_stroke"], linewidth=2)
        
        # Parallel line markers
        ax.annotate("›", (2.8, 1), fontsize=14, color=self.COLORS["shape_stroke"])
        ax.annotate("›", (3.2, 1), fontsize=14, color=self.COLORS["shape_stroke"])
        ax.annotate("›", (2.8, 3), fontsize=14, color=self.COLORS["shape_stroke"])
        ax.annotate("›", (3.2, 3), fontsize=14, color=self.COLORS["shape_stroke"])
        
        # Draw transversal
        ax.plot([1, 5], [0, 4], color=self.COLORS["accent"], linewidth=2)
        
        # Label angles (1-8)
        angle_positions = [
            (2.6, 1.3), (3.4, 1.3), (2.6, 0.7), (3.4, 0.7),  # Lower intersection
            (3.6, 3.3), (4.4, 3.3), (3.6, 2.7), (4.4, 2.7),  # Upper intersection
        ]
        
        if spec.show_labels:
            for i, pos in enumerate(angle_positions, 1):
                color = self.COLORS["secondary"] if i in highlight_angles else self.COLORS["text"]
                fontweight = 'bold' if i in highlight_angles else 'normal'
                ax.annotate(str(i), pos, fontsize=9, color=color, fontweight=fontweight)
        
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # COORDINATE PLANE
    # =========================================================================
    
    def _draw_coordinate_grid(self, ax, spec: DiagramSpec):
        """Draw a coordinate grid."""
        params = spec.params
        x_range = params.get("x_range", (-5, 5))
        y_range = params.get("y_range", (-5, 5))
        
        self._setup_coordinate_axes(ax, x_range, y_range)
    
    def _draw_plotted_points(self, ax, spec: DiagramSpec):
        """Draw points on a coordinate grid."""
        params = spec.params
        points = params.get("points", [(2, 3), (-1, 4), (3, -2)])
        labels = params.get("labels", [])
        x_range = params.get("x_range", (-6, 6))
        y_range = params.get("y_range", (-6, 6))
        
        self._setup_coordinate_axes(ax, x_range, y_range)
        
        colors = [self.COLORS["accent"], self.COLORS["secondary"], 
                 self.COLORS["success"], self.COLORS["primary"]]
        
        for i, point in enumerate(points):
            color = colors[i % len(colors)]
            ax.plot(*point, 'o', color=color, markersize=10, zorder=5)
            
            # Label
            label = labels[i] if i < len(labels) else f"({point[0]}, {point[1]})"
            ax.annotate(label, (point[0] + 0.3, point[1] + 0.3),
                       fontsize=9, color=color, fontweight='bold')
    
    def _draw_line_graph(self, ax, spec: DiagramSpec):
        """Draw a line on coordinate grid."""
        params = spec.params
        points = params.get("points", [(-3, -1), (3, 5)])
        x_range = params.get("x_range", (-6, 6))
        y_range = params.get("y_range", (-6, 6))
        show_equation = params.get("show_equation", True)
        
        self._setup_coordinate_axes(ax, x_range, y_range)
        
        # Plot line
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        ax.plot(x_vals, y_vals, color=self.COLORS["accent"], linewidth=2, zorder=4)
        
        # Plot points
        for point in points:
            ax.plot(*point, 'o', color=self.COLORS["secondary"], markersize=8, zorder=5)
        
        # Show equation if requested
        if show_equation and len(points) >= 2:
            slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]) if points[1][0] != points[0][0] else float('inf')
            intercept = points[0][1] - slope * points[0][0]
            if slope != float('inf'):
                eq = f"y = {slope:.1f}x + {intercept:.1f}" if intercept >= 0 else f"y = {slope:.1f}x - {abs(intercept):.1f}"
                ax.annotate(eq, (x_range[1] - 1, y_range[1] - 0.5),
                           fontsize=10, color=self.COLORS["accent"])
    
    def _draw_shape_on_grid(self, ax, spec: DiagramSpec):
        """Draw a polygon on coordinate grid."""
        params = spec.params
        vertices = params.get("vertices", [(1, 1), (4, 1), (4, 3), (1, 3)])
        x_range = params.get("x_range", (-1, 6))
        y_range = params.get("y_range", (-1, 5))
        
        self._setup_coordinate_axes(ax, x_range, y_range)
        
        # Draw shape
        shape = Polygon(vertices, fill=True,
                       facecolor=self.COLORS["shape_fill"],
                       edgecolor=self.COLORS["accent"],
                       linewidth=2, alpha=0.7, zorder=3)
        ax.add_patch(shape)
        
        # Label vertices
        if spec.show_labels:
            labels = params.get("labels", [chr(65+i) for i in range(len(vertices))])
            for i, (vertex, label) in enumerate(zip(vertices, labels)):
                ax.plot(*vertex, 'o', color=self.COLORS["secondary"], markersize=8, zorder=5)
                ax.annotate(f"{label}({vertex[0]},{vertex[1]})", 
                           (vertex[0] + 0.2, vertex[1] + 0.3),
                           fontsize=8, color=self.COLORS["text"])
    
    def _setup_coordinate_axes(self, ax, x_range, y_range):
        """Set up coordinate grid with axes."""
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # Grid
        ax.grid(True, color=self.COLORS["grid"], linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        
        # Axes through origin
        ax.axhline(y=0, color=self.COLORS["axis"], linewidth=1.5, zorder=2)
        ax.axvline(x=0, color=self.COLORS["axis"], linewidth=1.5, zorder=2)
        
        # Axis labels
        ax.set_xlabel("x", fontsize=11, color=self.COLORS["text"])
        ax.set_ylabel("y", fontsize=11, color=self.COLORS["text"])
        
        # Tick marks
        ax.set_xticks(range(int(x_range[0]), int(x_range[1]) + 1))
        ax.set_yticks(range(int(y_range[0]), int(y_range[1]) + 1))
        ax.tick_params(colors=self.COLORS["text"], labelsize=8)
        
        ax.set_aspect('equal')
    
    # =========================================================================
    # FRACTIONS
    # =========================================================================
    
    def _draw_fraction_circle(self, ax, spec: DiagramSpec):
        """Draw a circle divided into fraction parts with clear labeling."""
        params = spec.params
        numerator = params.get("numerator", 3)
        denominator = params.get("denominator", 4)
        radius = params.get("radius", 2)
        
        padding = radius * 0.3
        center = (radius + padding, radius + padding)
        
        # Draw wedges first (filled), then outline
        angle_per_part = 360 / denominator
        for i in range(denominator):
            start_angle = 90 - i * angle_per_part
            color = self.COLORS["accent"] if i < numerator else self.COLORS["shape_fill"]
            wedge = Wedge(center, radius, start_angle - angle_per_part, start_angle,
                         facecolor=color, edgecolor=self.COLORS["shape_stroke"],
                         linewidth=1.5)
            ax.add_patch(wedge)
        
        # Draw circle outline on top for clean edges
        circle = Circle(center, radius, fill=False,
                       edgecolor=self.COLORS["shape_stroke"], linewidth=2.5)
        ax.add_patch(circle)
        
        # Fraction label with clear background
        if spec.show_labels:
            ax.annotate(f"{numerator}/{denominator}", (center[0], -padding * 0.5),
                       ha='center', va='top', fontsize=16, fontweight='bold',
                       color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=2))
        
        ax.set_xlim(-padding, 2*radius + padding * 2)
        ax.set_ylim(-padding * 2, 2*radius + padding * 2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_fraction_bar(self, ax, spec: DiagramSpec):
        """Draw a bar model divided into fraction parts with clear labeling."""
        params = spec.params
        numerator = params.get("numerator", 3)
        denominator = params.get("denominator", 5)
        width = params.get("width", 6)
        height = params.get("height", 1.2)
        
        padding = 0.6
        part_width = width / denominator
        
        for i in range(denominator):
            color = self.COLORS["accent"] if i < numerator else self.COLORS["shape_fill"]
            rect = Rectangle((i * part_width, 0), part_width, height,
                            facecolor=color, edgecolor=self.COLORS["shape_stroke"],
                            linewidth=2)
            ax.add_patch(rect)
        
        # Fraction label with clear background
        if spec.show_labels:
            ax.annotate(f"{numerator}/{denominator}", (width/2, -padding * 0.8),
                       ha='center', va='top', fontsize=16, fontweight='bold',
                       color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=self.COLORS["accent"], alpha=0.95, linewidth=2))
        
        ax.set_xlim(-padding, width + padding)
        ax.set_ylim(-padding * 2, height + padding)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_fraction_number_line(self, ax, spec: DiagramSpec):
        """Draw a number line with fractions marked."""
        params = spec.params
        denominator = params.get("denominator", 4)
        highlight = params.get("highlight", [])  # Fractions to highlight
        start = params.get("start", 0)
        end = params.get("end", 1)
        
        # Main line
        ax.plot([0, 6], [0, 0], color=self.COLORS["axis"], linewidth=2)
        
        # End markers
        ax.plot([0, 0], [-0.2, 0.2], color=self.COLORS["axis"], linewidth=2)
        ax.plot([6, 6], [-0.2, 0.2], color=self.COLORS["axis"], linewidth=2)
        
        # Division marks
        for i in range(denominator + 1):
            x = i * 6 / denominator
            ax.plot([x, x], [-0.1, 0.1], color=self.COLORS["axis"], linewidth=1.5)
            
            # Label
            if i == 0:
                label = str(start)
            elif i == denominator:
                label = str(end)
            else:
                label = f"{i}/{denominator}"
            
            color = self.COLORS["accent"] if i in highlight else self.COLORS["text"]
            fontweight = 'bold' if i in highlight else 'normal'
            ax.annotate(label, (x, -0.4), ha='center', fontsize=10,
                       color=color, fontweight=fontweight)
            
            # Highlight point
            if i in highlight:
                ax.plot(x, 0, 'o', color=self.COLORS["accent"], markersize=10)
        
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-1, 0.5)
        ax.axis('off')
    
    # =========================================================================
    # DATA & STATISTICS
    # =========================================================================
    
    def _draw_bar_graph(self, ax, spec: DiagramSpec):
        """Draw a bar graph with clear, readable labels."""
        params = spec.params
        data = params.get("data", {"A": 5, "B": 8, "C": 3, "D": 6})
        title = params.get("title", "")
        xlabel = params.get("xlabel", "Category")
        ylabel = params.get("ylabel", "Value")
        
        categories = list(data.keys())
        values = list(data.values())
        max_val = max(values) if values else 1
        
        colors = [self.COLORS["primary"], self.COLORS["accent"], 
                 self.COLORS["secondary"], self.COLORS["success"]]
        bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
        
        # Create bars with proper spacing
        bar_width = 0.7
        bars = ax.bar(categories, values, width=bar_width, color=bar_colors, 
                     edgecolor='white', linewidth=1.5)
        
        # Value labels ABOVE bars (not inside) with white background
        for bar, val in zip(bars, values):
            label_y = bar.get_height() + max_val * 0.05  # 5% above bar
            ax.annotate(
                str(int(val) if val == int(val) else f"{val:.1f}"),
                (bar.get_x() + bar.get_width()/2, label_y),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', 
                color=self.COLORS["text"],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.9)
            )
        
        # Set y-axis to give space for labels
        ax.set_ylim(0, max_val * 1.2)
        
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', color=self.COLORS["text"], pad=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=self.COLORS["text"], labelsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    def _draw_dot_plot(self, ax, spec: DiagramSpec):
        """Draw a dot plot with clear spacing and readable values."""
        params = spec.params
        data = params.get("data", [1, 2, 2, 3, 3, 3, 4, 4, 5])
        title = params.get("title", "")
        xlabel = params.get("xlabel", "Value")
        
        # Count occurrences
        from collections import Counter
        counts = Counter(data)
        max_count = max(counts.values()) if counts else 1
        
        # Calculate appropriate dot size based on data range
        min_val, max_val = min(data), max(data)
        data_range = max_val - min_val + 1
        
        # Adjust dot size to prevent overlap
        dot_size = min(12, max(6, 60 / data_range))
        
        # Plot dots with proper vertical spacing
        for value, count in counts.items():
            for i in range(count):
                ax.plot(value, i + 0.5, 'o', color=self.COLORS["primary"], 
                       markersize=dot_size, markeredgecolor='white', markeredgewidth=1)
        
        # Set axes with padding
        ax.set_xlim(min_val - 0.8, max_val + 0.8)
        ax.set_ylim(-0.5, max_count + 1)
        ax.set_xticks(range(min_val, max_val + 1))
        
        # Draw baseline
        ax.axhline(y=0, color=self.COLORS["axis"], linewidth=2, zorder=1)
        
        # Style x-axis labels
        ax.tick_params(axis='x', colors=self.COLORS["text"], labelsize=10)
        ax.set_yticks([])
        
        # Add frequency labels on the right side if many dots
        if max_count > 4:
            for value, count in counts.items():
                ax.annotate(f"({count})", (value, count + 0.2), ha='center', va='bottom',
                           fontsize=8, color=self.COLORS["secondary"])
        
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.COLORS["axis"])
        
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', color=self.COLORS["text"], pad=10)
    
    def _draw_box_plot(self, ax, spec: DiagramSpec):
        """Draw a box and whisker plot with clear, non-overlapping labels."""
        params = spec.params
        data = params.get("data", [12, 15, 18, 22, 25, 28, 30, 35, 40])
        title = params.get("title", "")
        show_values = params.get("show_values", True)
        xlabel = params.get("xlabel", "")
        
        # Calculate statistics
        data_sorted = sorted(data)
        min_val = min(data)
        max_val = max(data)
        q1 = np.percentile(data, 25)
        median = np.median(data)
        q3 = np.percentile(data, 75)
        data_range = max_val - min_val
        
        # Create box plot
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                       boxprops=dict(facecolor=self.COLORS["shape_fill"], 
                                    edgecolor=self.COLORS["primary"], linewidth=2),
                       medianprops=dict(color=self.COLORS["accent"], linewidth=3),
                       whiskerprops=dict(color=self.COLORS["primary"], linewidth=2),
                       capprops=dict(color=self.COLORS["primary"], linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor=self.COLORS["secondary"],
                                      markersize=8, markeredgecolor='white'))
        
        # Add value labels BELOW the plot to avoid overlap
        if show_values:
            label_y = 0.4  # Below the box
            
            # Min value
            ax.annotate(f"Min={min_val:.0f}", (min_val, label_y), ha='center', va='top',
                       fontsize=9, color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
            # Q1 - offset if too close to min
            q1_label_x = q1
            if (q1 - min_val) / data_range < 0.15:
                q1_label_x = min_val + data_range * 0.15
            ax.annotate(f"Q1={q1:.0f}", (q1_label_x, label_y - 0.25), ha='center', va='top',
                       fontsize=9, color=self.COLORS["primary"],
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
            # Median - always centered, shown above
            ax.annotate(f"Median={median:.0f}", (median, 1.6), ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color=self.COLORS["accent"],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=self.COLORS["accent"], alpha=0.95))
            
            # Q3 - offset if too close to max
            q3_label_x = q3
            if (max_val - q3) / data_range < 0.15:
                q3_label_x = max_val - data_range * 0.15
            ax.annotate(f"Q3={q3:.0f}", (q3_label_x, label_y - 0.25), ha='center', va='top',
                       fontsize=9, color=self.COLORS["primary"],
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
            # Max value
            ax.annotate(f"Max={max_val:.0f}", (max_val, label_y), ha='center', va='top',
                       fontsize=9, color=self.COLORS["text"],
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        
        # Set axis limits with padding
        padding = data_range * 0.1
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(-0.3 if show_values else 0.3, 2.0)
        
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', colors=self.COLORS["text"], labelsize=9)
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', color=self.COLORS["text"], pad=10)
    
    def _draw_histogram(self, ax, spec: DiagramSpec):
        """Draw a histogram with clear bars and frequency labels."""
        params = spec.params
        data = params.get("data", [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6])
        bins = params.get("bins", 6)
        title = params.get("title", "")
        xlabel = params.get("xlabel", "Value")
        ylabel = params.get("ylabel", "Frequency")
        show_counts = params.get("show_counts", True)
        
        # Create histogram and get bin info
        counts, bin_edges, patches = ax.hist(data, bins=bins, color=self.COLORS["primary"], 
                                              edgecolor='white', linewidth=2, alpha=0.9)
        
        max_count = max(counts) if len(counts) > 0 else 1
        
        # Add count labels above each bar
        if show_counts:
            for i, (count, patch) in enumerate(zip(counts, patches)):
                if count > 0:  # Only label non-empty bins
                    x = patch.get_x() + patch.get_width() / 2
                    y = count + max_count * 0.05
                    ax.annotate(f"{int(count)}", (x, y), ha='center', va='bottom',
                               fontsize=10, fontweight='bold', color=self.COLORS["text"],
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                        edgecolor='none', alpha=0.85))
        
        # Set y-axis to accommodate labels
        ax.set_ylim(0, max_count * 1.25)
        
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color=self.COLORS["text"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=self.COLORS["text"], labelsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', color=self.COLORS["text"], pad=10)
    
    def _draw_pie_chart(self, ax, spec: DiagramSpec):
        """Draw a pie chart."""
        params = spec.params
        data = params.get("data", {"A": 30, "B": 25, "C": 20, "D": 25})
        title = params.get("title", "")
        
        labels = list(data.keys())
        sizes = list(data.values())
        
        colors = [self.COLORS["primary"], self.COLORS["accent"],
                 self.COLORS["secondary"], self.COLORS["success"],
                 "#f472b6", "#a78bfa"]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.0f%%',
                                          colors=colors[:len(sizes)],
                                          startangle=90,
                                          textprops={'fontsize': 10, 'color': self.COLORS["text"]})
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
        
        if title:
            ax.set_title(title, fontsize=11, fontweight='bold', color=self.COLORS["text"])
    
    # =========================================================================
    # RATIOS & PROPORTIONS
    # =========================================================================
    
    def _draw_double_number_line(self, ax, spec: DiagramSpec):
        """Draw a double number line for ratios."""
        params = spec.params
        top_values = params.get("top_values", [0, 2, 4, 6, 8])
        bottom_values = params.get("bottom_values", [0, 3, 6, 9, 12])
        top_label = params.get("top_label", "Miles")
        bottom_label = params.get("bottom_label", "Hours")
        
        line_length = 6
        
        # Top line
        ax.plot([0, line_length], [1, 1], color=self.COLORS["primary"], linewidth=2)
        ax.annotate(top_label, (-0.8, 1), ha='right', va='center', fontsize=10,
                   color=self.COLORS["primary"], fontweight='bold')
        
        # Bottom line
        ax.plot([0, line_length], [0, 0], color=self.COLORS["accent"], linewidth=2)
        ax.annotate(bottom_label, (-0.8, 0), ha='right', va='center', fontsize=10,
                   color=self.COLORS["accent"], fontweight='bold')
        
        # Marks and labels
        num_marks = len(top_values)
        for i in range(num_marks):
            x = i * line_length / (num_marks - 1)
            
            # Top marks
            ax.plot([x, x], [0.9, 1.1], color=self.COLORS["primary"], linewidth=1.5)
            ax.annotate(str(top_values[i]), (x, 1.3), ha='center', fontsize=10,
                       color=self.COLORS["primary"])
            
            # Bottom marks
            ax.plot([x, x], [-0.1, 0.1], color=self.COLORS["accent"], linewidth=1.5)
            ax.annotate(str(bottom_values[i]), (x, -0.3), ha='center', fontsize=10,
                       color=self.COLORS["accent"])
            
            # Connecting line
            ax.plot([x, x], [0, 1], '--', color=self.COLORS["grid"], linewidth=1, alpha=0.5)
        
        ax.set_xlim(-1.5, line_length + 0.5)
        ax.set_ylim(-0.8, 1.8)
        ax.axis('off')
    
    def _draw_tape_diagram(self, ax, spec: DiagramSpec):
        """Draw a tape diagram for ratios."""
        params = spec.params
        parts = params.get("parts", [3, 2])  # Ratio parts
        labels = params.get("labels", ["Part A", "Part B"])
        total = params.get("total", None)
        
        colors = [self.COLORS["primary"], self.COLORS["accent"], 
                 self.COLORS["secondary"], self.COLORS["success"]]
        
        total_parts = sum(parts)
        unit_width = 5 / total_parts
        
        x = 0
        for i, (num_parts, label) in enumerate(zip(parts, labels)):
            width = num_parts * unit_width
            rect = Rectangle((x, 0), width, 1,
                            facecolor=colors[i % len(colors)],
                            edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Label inside
            ax.annotate(label, (x + width/2, 0.5), ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold')
            
            # Parts count
            ax.annotate(f"{num_parts}", (x + width/2, -0.3), ha='center',
                       fontsize=10, color=colors[i % len(colors)], fontweight='bold')
            
            x += width
        
        # Total label
        if total:
            ax.annotate(f"Total = {total}", (2.5, 1.3), ha='center',
                       fontsize=11, color=self.COLORS["text"], fontweight='bold')
        
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.8, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # MEASUREMENT
    # =========================================================================
    
    def _draw_ruler(self, ax, spec: DiagramSpec):
        """Draw a ruler with measurements."""
        params = spec.params
        length = params.get("length", 6)  # inches
        highlight_range = params.get("highlight", None)  # (start, end) to highlight
        
        # Ruler body
        rect = Rectangle((0, 0), length, 0.8,
                        facecolor=self.COLORS["highlight"],
                        edgecolor=self.COLORS["shape_stroke"],
                        linewidth=2)
        ax.add_patch(rect)
        
        # Inch marks
        for i in range(length + 1):
            ax.plot([i, i], [0, 0.5], color=self.COLORS["shape_stroke"], linewidth=2)
            ax.annotate(str(i), (i, -0.2), ha='center', fontsize=10,
                       color=self.COLORS["text"])
            
            # Half-inch marks
            if i < length:
                ax.plot([i + 0.5, i + 0.5], [0, 0.35], color=self.COLORS["shape_stroke"], linewidth=1.5)
                
                # Quarter-inch marks
                ax.plot([i + 0.25, i + 0.25], [0, 0.25], color=self.COLORS["shape_stroke"], linewidth=1)
                ax.plot([i + 0.75, i + 0.75], [0, 0.25], color=self.COLORS["shape_stroke"], linewidth=1)
        
        # Highlight range
        if highlight_range:
            start, end = highlight_range
            highlight = Rectangle((start, 0), end - start, 0.8,
                                 facecolor=self.COLORS["accent"],
                                 alpha=0.3)
            ax.add_patch(highlight)
        
        ax.set_xlim(-0.5, length + 0.5)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
    
    def _draw_protractor(self, ax, spec: DiagramSpec):
        """Draw a protractor with angle marked."""
        params = spec.params
        highlight_angle = params.get("highlight_angle", None)
        
        # Semicircle
        theta = np.linspace(0, np.pi, 100)
        radius = 3
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)
        
        ax.fill_between(x, 0, y, color=self.COLORS["shape_fill"], alpha=0.5)
        ax.plot(x, y, color=self.COLORS["shape_stroke"], linewidth=2)
        ax.plot([0, 2*radius], [0, 0], color=self.COLORS["shape_stroke"], linewidth=2)
        
        # Degree marks
        for deg in range(0, 181, 10):
            angle_rad = math.radians(deg)
            inner_r = radius * 0.85
            outer_r = radius * 0.95 if deg % 30 == 0 else radius * 0.9
            
            x1, y1 = radius + inner_r * math.cos(angle_rad), inner_r * math.sin(angle_rad)
            x2, y2 = radius + outer_r * math.cos(angle_rad), outer_r * math.sin(angle_rad)
            
            ax.plot([x1, x2], [y1, y2], color=self.COLORS["shape_stroke"], linewidth=1)
            
            # Labels for major angles
            if deg % 30 == 0:
                label_r = radius * 0.75
                lx, ly = radius + label_r * math.cos(angle_rad), label_r * math.sin(angle_rad)
                ax.annotate(str(deg), (lx, ly), ha='center', va='center', fontsize=8,
                           color=self.COLORS["text"])
        
        # Center point
        ax.plot(radius, 0, 'o', color=self.COLORS["shape_stroke"], markersize=5)
        
        # Highlight angle
        if highlight_angle:
            angle_rad = math.radians(highlight_angle)
            ax.plot([radius, radius + radius*0.9*math.cos(angle_rad)],
                   [0, radius*0.9*math.sin(angle_rad)],
                   color=self.COLORS["accent"], linewidth=2)
            
            # Arc
            arc = Arc((radius, 0), radius*0.4, radius*0.4, angle=0,
                     theta1=0, theta2=highlight_angle, color=self.COLORS["accent"], linewidth=2)
            ax.add_patch(arc)
            
            ax.annotate(f"{highlight_angle}°", 
                       (radius + 0.5*math.cos(math.radians(highlight_angle/2)),
                        0.5*math.sin(math.radians(highlight_angle/2))),
                       fontsize=11, color=self.COLORS["accent"], fontweight='bold')
        
        ax.set_xlim(-0.5, 2*radius + 0.5)
        ax.set_ylim(-0.5, radius + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_scale_drawing(self, ax, spec: DiagramSpec):
        """Draw a scale drawing comparison."""
        params = spec.params
        actual_dims = params.get("actual", (20, 15))  # feet
        scale = params.get("scale", "1 in = 5 ft")
        
        # Calculate drawing dimensions
        scale_factor = 5  # from scale string
        draw_width = actual_dims[0] / scale_factor
        draw_height = actual_dims[1] / scale_factor
        
        # Draw scaled rectangle
        rect = Rectangle((0.5, 0.5), draw_width, draw_height,
                        facecolor=self.COLORS["shape_fill"],
                        edgecolor=self.COLORS["shape_stroke"],
                        linewidth=2)
        ax.add_patch(rect)
        
        # Dimension labels
        ax.annotate(f'{draw_width}" ({actual_dims[0]} ft)', (0.5 + draw_width/2, 0.2),
                   ha='center', fontsize=10, color=self.COLORS["text"])
        ax.annotate(f'{draw_height}" ({actual_dims[1]} ft)', (-0.1, 0.5 + draw_height/2),
                   ha='center', va='center', fontsize=10, color=self.COLORS["text"], rotation=90)
        
        # Scale label
        ax.annotate(f"Scale: {scale}", (0.5 + draw_width/2, 0.5 + draw_height + 0.4),
                   ha='center', fontsize=11, fontweight='bold', color=self.COLORS["accent"])
        
        ax.set_xlim(-0.5, draw_width + 1.5)
        ax.set_ylim(-0.3, draw_height + 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _setup_shape_axes(self, ax, vertices, padding=0.5):
        """Configure axes for shape display."""
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        ax.set_xlim(min(xs) - padding, max(xs) + padding)
        ax.set_ylim(min(ys) - padding, max(ys) + padding)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _add_side_labels(self, ax, vertices, lengths):
        """Add side length labels to a polygon."""
        n = len(vertices)
        for i, length in enumerate(lengths):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            
            # Offset label slightly from midpoint
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Perpendicular offset
            perp_x = -dy / (math.sqrt(dx*dx + dy*dy) + 0.001) * 0.3
            perp_y = dx / (math.sqrt(dx*dx + dy*dy) + 0.001) * 0.3
            
            ax.annotate(str(length), (mid_x + perp_x, mid_y + perp_y),
                       ha='center', va='center', fontsize=10, color=self.COLORS["text"])
    
    def _draw_placeholder(self, ax, spec: DiagramSpec):
        """Draw a placeholder for unimplemented diagram types."""
        ax.text(0.5, 0.5, f"Diagram: {spec.diagram_type.value}",
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes, color=self.COLORS["text"])
        ax.text(0.5, 0.3, "(Not yet implemented)",
               ha='center', va='center', fontsize=10,
               transform=ax.transAxes, color=self.COLORS["text"], alpha=0.5)
        ax.axis('off')


# =============================================================================
# DIAGRAM DETECTION UTILITIES
# =============================================================================

def detect_diagram_needs(concept_text: str, domain: str) -> List[DiagramSpec]:
    """
    Detect what diagrams would be helpful for a given math concept.
    
    Args:
        concept_text: The math concept description
        domain: The mathematical domain (geometry, fractions, etc.)
        
    Returns:
        List of DiagramSpec suggestions
    """
    specs = []
    text_lower = concept_text.lower()
    
    # Geometry detection
    if domain == "geometry" or any(word in text_lower for word in 
        ["area", "perimeter", "volume", "surface", "angle", "triangle", 
         "rectangle", "circle", "polygon", "shape"]):
        
        if "right triangle" in text_lower or "right-triangle" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.RIGHT_TRIANGLE,
                params={"base": 6, "height": 8, "show_right_angle": True}
            ))
        elif "triangle" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.TRIANGLE,
                params={}
            ))
        
        if "rectangle" in text_lower and "prism" not in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.RECTANGLE,
                params={"width": 6, "height": 4}
            ))
        
        if "circle" in text_lower or "circumference" in text_lower or "radius" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.CIRCLE,
                params={"radius": 3, "show_radius": True}
            ))
        
        if "prism" in text_lower:
            if "triangular" in text_lower:
                specs.append(DiagramSpec(
                    diagram_type=DiagramType.TRIANGULAR_PRISM,
                    params={}
                ))
            else:
                specs.append(DiagramSpec(
                    diagram_type=DiagramType.RECTANGULAR_PRISM,
                    params={}
                ))
        
        if "cylinder" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.CYLINDER,
                params={"radius": 2, "height": 5}
            ))
        
        if "cone" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.CONE,
                params={"radius": 2, "height": 5}
            ))
        
        if "pyramid" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.PYRAMID,
                params={}
            ))
        
        if "angle" in text_lower:
            if "transversal" in text_lower or "parallel" in text_lower:
                specs.append(DiagramSpec(
                    diagram_type=DiagramType.TRANSVERSAL,
                    params={}
                ))
            else:
                specs.append(DiagramSpec(
                    diagram_type=DiagramType.ANGLE,
                    params={"degrees": 45}
                ))
    
    # Fractions detection
    if domain == "fractions" or "fraction" in text_lower:
        specs.append(DiagramSpec(
            diagram_type=DiagramType.FRACTION_CIRCLE,
            params={"numerator": 3, "denominator": 4}
        ))
    
    # Coordinate plane detection
    if "coordinate" in text_lower or "graph" in text_lower or "plot" in text_lower:
        if "point" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.PLOTTED_POINTS,
                params={"points": [(2, 3), (-1, 4)]}
            ))
        elif "line" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.LINE_GRAPH,
                params={}
            ))
        else:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.COORDINATE_GRID,
                params={}
            ))
    
    # Statistics detection
    if domain == "statistics" or domain == "data":
        if "box" in text_lower or "whisker" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.BOX_PLOT,
                params={}
            ))
        elif "dot plot" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.DOT_PLOT,
                params={}
            ))
        elif "histogram" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.HISTOGRAM,
                params={}
            ))
        elif "bar" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.BAR_GRAPH,
                params={}
            ))
    
    # Ratios detection
    if domain == "ratios" or "ratio" in text_lower or "proportion" in text_lower:
        if "number line" in text_lower or "double" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.DOUBLE_NUMBER_LINE,
                params={}
            ))
        elif "tape" in text_lower or "diagram" in text_lower:
            specs.append(DiagramSpec(
                diagram_type=DiagramType.TAPE_DIAGRAM,
                params={}
            ))
    
    return specs


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Create test output directory
    test_dir = Path(__file__).parent.parent / "artifacts" / "diagram_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    generator = DiagramGenerator()
    
    # Test various diagram types
    test_specs = [
        ("right_triangle", DiagramSpec(
            diagram_type=DiagramType.RIGHT_TRIANGLE,
            params={"base": 6, "height": 8, "labels": {"base": "6 cm", "height": "8 cm", "hypotenuse": "10 cm"}}
        )),
        ("circle", DiagramSpec(
            diagram_type=DiagramType.CIRCLE,
            params={"radius": 4, "show_radius": True, "show_diameter": True}
        )),
        ("rectangular_prism", DiagramSpec(
            diagram_type=DiagramType.RECTANGULAR_PRISM,
            params={"length": 5, "width": 3, "height": 4}
        )),
        ("coordinate_points", DiagramSpec(
            diagram_type=DiagramType.PLOTTED_POINTS,
            params={"points": [(2, 3), (-3, 1), (4, -2)], "labels": ["A", "B", "C"]}
        )),
        ("fraction_circle", DiagramSpec(
            diagram_type=DiagramType.FRACTION_CIRCLE,
            params={"numerator": 3, "denominator": 8}
        )),
        ("bar_graph", DiagramSpec(
            diagram_type=DiagramType.BAR_GRAPH,
            params={"data": {"Mon": 5, "Tue": 8, "Wed": 3, "Thu": 6, "Fri": 9}, "title": "Daily Sales"}
        )),
        ("angle", DiagramSpec(
            diagram_type=DiagramType.ANGLE,
            params={"degrees": 60}
        )),
        ("transversal", DiagramSpec(
            diagram_type=DiagramType.TRANSVERSAL,
            params={"highlight_angles": [1, 5]}
        )),
    ]
    
    print("Generating test diagrams...")
    for name, spec in test_specs:
        path = test_dir / f"{name}.png"
        generator.generate_to_file(spec, path)
        print(f"  Created: {path}")
    
    print(f"\nAll test diagrams saved to: {test_dir}")

