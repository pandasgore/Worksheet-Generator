from __future__ import annotations

import io
import tempfile
from pathlib import Path
import sys
from typing import Iterable, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from schemas.worksheet import WorksheetDocumentPayload, WorksheetDocumentProblem, DiagramSpecPayload
from formatting.text_cleaner import clean_math_text, format_for_middle_school, extract_fractions, extract_markdown_table
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# Import diagram generator (lazy to avoid import errors if matplotlib not installed)
_diagram_generator = None

def get_diagram_generator():
    """Lazy load the diagram generator."""
    global _diagram_generator
    if _diagram_generator is None:
        try:
            from formatting.diagram_generator import DiagramGenerator
            _diagram_generator = DiagramGenerator()
        except ImportError:
            print("Warning: Diagram generation not available (matplotlib not installed)")
            return None
    return _diagram_generator


def _friendly_difficulty_label(level: str) -> str:
    level = (level or "").lower()
    if level in {"easy", "medium", "hard"}:
        return level.capitalize()
    return "Mixed"


class PdfBuilder:
    def __init__(
        self,
        *,
        pagesize=letter,
        left_margin: int = 72,
        right_margin: int = 72,
        top_margin: int = 72,
        line_height: int = 16,
    ):
        self.pagesize = pagesize
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.top_margin = top_margin
        self.line_height = line_height

    def build(
        self,
        worksheet: WorksheetDocumentPayload,
        output_path: str | Path,
        *,
        include_answer_key: bool = True,
    ) -> Path:
        output_path = Path(output_path)
        canvas = pdf_canvas.Canvas(str(output_path), pagesize=self.pagesize)
        width, height = self.pagesize
        cursor_y = height - self.top_margin

        cursor_y = self._draw_header(canvas, worksheet, cursor_y)
        cursor_y = self._draw_instructions(canvas, cursor_y)
        cursor_y = self._draw_problems(canvas, worksheet.problems, cursor_y)

        if include_answer_key and worksheet.problems:
            canvas.showPage()
            cursor_y = height - self.top_margin
            cursor_y = self._draw_answer_key(canvas, worksheet.problems, cursor_y)

        canvas.save()
        return output_path

    def _draw_header(
        self,
        canvas: pdf_canvas.Canvas,
        worksheet: WorksheetDocumentPayload,
        cursor_y: float,
    ) -> float:
        canvas.setFont("Helvetica-Bold", 18)
        title = worksheet.title or f"Grade {worksheet.grade} Worksheet"
        canvas.drawString(self.left_margin, cursor_y, title)
        cursor_y -= self.line_height * 1.5

        canvas.setFont("Helvetica", 12)
        canvas.drawString(self.left_margin, cursor_y, f"Name: ______________________    Date: ______________________")
        cursor_y -= self.line_height

        meta = " • ".join(
            [
                f"Grade: {worksheet.grade}",
                f"Concept: {worksheet.concept}",
                f"Difficulty: {_friendly_difficulty_label(worksheet.difficulty)}",
            ]
        )
        canvas.drawString(self.left_margin, cursor_y, meta)
        cursor_y -= self.line_height
        return cursor_y

    def _draw_instructions(
        self,
        canvas: pdf_canvas.Canvas,
        cursor_y: float,
    ) -> float:
        canvas.setFont("Helvetica", 11)
        canvas.drawString(
            self.left_margin,
            cursor_y,
            "Show your work clearly. You can write in the spaces under each problem.",
        )
        return cursor_y - self.line_height * 1.5

    def _draw_diagram(
        self,
        canvas: pdf_canvas.Canvas,
        diagram_spec: DiagramSpecPayload,
        cursor_y: float,
    ) -> float:
        """Draw a diagram inline with the problem."""
        generator = get_diagram_generator()
        if generator is None:
            return cursor_y
        
        try:
            from formatting.diagram_generator import DiagramSpec, DiagramType
            
            # Convert payload to DiagramSpec
            spec = DiagramSpec(
                diagram_type=DiagramType(diagram_spec.diagram_type),
                params=diagram_spec.params,
                width=diagram_spec.width,
                height=diagram_spec.height,
                show_labels=diagram_spec.show_labels,
                show_measurements=diagram_spec.show_measurements,
                title=diagram_spec.title,
            )
            
            # Generate the diagram as PNG bytes
            image_bytes = generator.generate(spec, format="png")
            
            # Calculate image dimensions in points (72 points per inch)
            img_width = spec.width * 72
            img_height = spec.height * 72
            
            # Check if we need a new page
            if cursor_y - img_height < self.top_margin:
                canvas.showPage()
                cursor_y = self.pagesize[1] - self.top_margin
            
            # Draw the image
            img = ImageReader(io.BytesIO(image_bytes))
            # Center the image horizontally
            page_width = self.pagesize[0]
            x_pos = (page_width - img_width) / 2
            
            canvas.drawImage(img, x_pos, cursor_y - img_height, 
                           width=img_width, height=img_height)
            
            cursor_y -= img_height + self.line_height
            
        except Exception as e:
            print(f"Warning: Could not generate diagram: {e}")
        
        return cursor_y

    def _draw_problems(
        self,
        canvas: pdf_canvas.Canvas,
        problems: Iterable[WorksheetDocumentProblem],
        cursor_y: float,
    ) -> float:
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(self.left_margin, cursor_y, "Problems")
        cursor_y -= self.line_height * 1.5

        canvas.setFont("Helvetica", 12)
        # Give about 2 inches of space per problem for work
        workspace_height = 144  

        for idx, problem in enumerate(problems, start=1):
            # Calculate needed height including potential diagram
            diagram_height = 0
            if problem.diagram:
                diagram_height = (problem.diagram.height * 72) + self.line_height
            
            # Check if we have enough space for the problem text + diagram + workspace
            # approximate text height as 3 lines to be safe for check
            needed_height = (self.line_height * 3) + diagram_height + workspace_height
            if cursor_y - needed_height < self.top_margin:
                 canvas.showPage()
                 cursor_y = self.pagesize[1] - self.top_margin
                 canvas.setFont("Helvetica", 12)

            # Clean LaTeX notation and format for middle school
            cleaned_prompt = clean_math_text(problem.prompt.strip())
            cursor_y = self._draw_math_text(
                canvas,
                f"{idx}. {cleaned_prompt}",
                cursor_y,
            )
            
            # Draw diagram if present (inline with the problem)
            if problem.diagram:
                cursor_y -= self.line_height * 0.5  # Small gap before diagram
                cursor_y = self._draw_diagram(canvas, problem.diagram, cursor_y)
            
            # Add workspace
            cursor_y -= workspace_height
            
            # Optional: Draw a light line at the bottom of the workspace
            # canvas.setLineWidth(0.5)
            # canvas.setStrokeColorRGB(0.8, 0.8, 0.8)
            # canvas.line(self.left_margin, cursor_y + 10, self.pagesize[0] - self.right_margin, cursor_y + 10)
            # canvas.setStrokeColorRGB(0, 0, 0)

            cursor_y -= self.line_height

        return cursor_y

    def _draw_answer_key(
        self,
        canvas: pdf_canvas.Canvas,
        problems: Iterable[WorksheetDocumentProblem],
        cursor_y: float,
    ) -> float:
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(self.left_margin, cursor_y, "Answer Key")
        cursor_y -= self.line_height * 1.5

        canvas.setFont("Helvetica", 12)
        for idx, problem in enumerate(problems, start=1):
            # Calculate space needed for this answer including potential diagram
            answer_diagram_height = 0
            if problem.answer_diagram:
                answer_diagram_height = (problem.answer_diagram.height * 72) + self.line_height
            
            # Check if we need a new page before starting this answer
            needed_height = (self.line_height * 5) + answer_diagram_height
            if cursor_y - needed_height < self.top_margin:
                canvas.showPage()
                cursor_y = self.pagesize[1] - self.top_margin
                canvas.setFont("Helvetica", 12)
            
            # Clean LaTeX notation and format for middle school
            cleaned_prompt = clean_math_text(problem.prompt.strip())
            cursor_y = self._draw_math_text(
                canvas,
                f"{idx}. {cleaned_prompt}",
                cursor_y,
            )
            if problem.answer:
                # Clean LaTeX notation and format answer
                cleaned_answer = clean_math_text(problem.answer)
                cursor_y = self._draw_math_text(
                    canvas,
                    f"   Answer: {cleaned_answer}",
                    cursor_y,
                )
            
            # Draw answer diagram if present (for "create a diagram" type problems)
            if problem.answer_diagram:
                cursor_y -= self.line_height * 0.5
                cursor_y = self._draw_diagram(canvas, problem.answer_diagram, cursor_y)
            
            if problem.solution_steps:
                cursor_y = self._draw_wrapped_text(
                    canvas,
                    "   Steps:",
                    cursor_y,
                )
                for step in problem.solution_steps:
                    # Clean LaTeX notation and format for middle school
                    cleaned_step = clean_math_text(step)
                    cursor_y = self._draw_math_text(
                        canvas,
                        f"      - {cleaned_step}",
                        cursor_y,
                        font_size=11,
                    )
            cursor_y -= self.line_height
            if cursor_y <= self.top_margin:
                canvas.showPage()
                cursor_y = self.pagesize[1] - self.top_margin
        return cursor_y

    def _draw_stacked_fraction(
        self,
        canvas: pdf_canvas.Canvas,
        numerator: str,
        denominator: str,
        x: float,
        y: float,
        font_size: int = 12,
        with_parens: bool = False,
    ) -> float:
        """
        Draw a stacked fraction (numerator over denominator with line).
        If with_parens=True, draws tall parentheses around the fraction.
        Returns the width used by the fraction (including parentheses if any).
        """
        canvas.setFont("Helvetica", font_size)
        
        # Calculate widths with generous padding
        num_width = canvas.stringWidth(str(numerator), "Helvetica", font_size)
        den_width = canvas.stringWidth(str(denominator), "Helvetica", font_size)
        frac_width = max(num_width, den_width) + 12  # more padding for clarity
        
        # Calculate fraction vertical bounds
        top_y = y + font_size * 0.5
        bottom_y = y - font_size * 1.1
        frac_height = top_y - bottom_y
        
        current_x = x
        paren_width = font_size * 0.4  # slightly wider parens
        
        # Draw opening parenthesis if needed
        if with_parens:
            self._draw_tall_paren(canvas, "(", current_x, y, frac_height, font_size)
            current_x += paren_width + 3  # gap after open paren
        
        # Draw numerator (centered above line)
        num_x = current_x + (frac_width - num_width) / 2
        canvas.drawString(num_x, y + font_size * 0.35, str(numerator))
        
        # Draw fraction line (slightly thicker for visibility)
        line_y = y
        canvas.setLineWidth(1.2)
        canvas.line(current_x, line_y, current_x + frac_width, line_y)
        canvas.setLineWidth(1)
        
        # Draw denominator (centered below line)
        den_x = current_x + (frac_width - den_width) / 2
        canvas.drawString(den_x, y - font_size * 0.85, str(denominator))
        
        current_x += frac_width
        
        # Draw closing parenthesis if needed
        if with_parens:
            current_x += 3  # gap before close paren
            self._draw_tall_paren(canvas, ")", current_x, y, frac_height, font_size)
            current_x += paren_width
        
        total_width = current_x - x
        return total_width

    def _draw_pdf_table(
        self,
        canvas: pdf_canvas.Canvas,
        table_data: dict,
        cursor_y: float,
        font_size: int = 11,
    ) -> float:
        """
        Draw a proper PDF table with headers and rows.
        Returns the new cursor_y position after the table.
        """
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        if not headers:
            return cursor_y
        
        # Calculate column widths based on content
        num_cols = len(headers)
        col_widths = []
        for i in range(num_cols):
            max_width = canvas.stringWidth(str(headers[i]), "Helvetica-Bold", font_size)
            for row in rows:
                if i < len(row):
                    cell_width = canvas.stringWidth(str(row[i]), "Helvetica", font_size)
                    max_width = max(max_width, cell_width)
            col_widths.append(max_width + 20)  # Add padding
        
        # Ensure minimum width for readability and cap maximum
        col_widths = [max(60, min(w, 140)) for w in col_widths]
        
        # Convert "?" placeholders to fill-in blanks for worksheets
        processed_rows = []
        for row in rows:
            processed_row = []
            for cell in row:
                if cell == '?' or cell == '':
                    processed_row.append('____')  # Fill-in-the-blank style
                else:
                    processed_row.append(cell)
            processed_rows.append(processed_row)
        
        # Build table data: headers + processed rows
        data = [headers] + processed_rows
        
        # Create table with wider minimum column widths
        table = Table(data, colWidths=col_widths)
        
        # Style the table
        style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), font_size),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Data rows styling
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), font_size),
            
            # Grid - slightly thicker for better visibility
            ('GRID', (0, 0), (-1, -1), 1.2, colors.black),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ])
        table.setStyle(style)
        
        # Calculate table size
        table_width, table_height = table.wrap(0, 0)
        
        # Check if we need a new page
        if cursor_y - table_height < self.top_margin:
            canvas.showPage()
            cursor_y = self.pagesize[1] - self.top_margin
            canvas.setFont("Helvetica", 12)
        
        # Draw the table
        draw_y = cursor_y - table_height
        table.drawOn(canvas, self.left_margin, draw_y)
        
        # Return new cursor position
        return cursor_y - table_height - self.line_height * 0.5

    def _draw_tall_paren(
        self,
        canvas: pdf_canvas.Canvas,
        paren: str,
        x: float,
        y: float,
        height: float,
        font_size: int = 12,
    ) -> float:
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
        
        canvas.setFont("Helvetica", paren_font_size)
        canvas.drawString(x, paren_y, paren)
        
        # Reset font
        canvas.setFont("Helvetica", font_size)
        
        # Return width of the parenthesis character
        paren_width = canvas.stringWidth(paren, "Helvetica", paren_font_size)
        return paren_width

    def _draw_math_text(
        self,
        canvas: pdf_canvas.Canvas,
        text: str,
        cursor_y: float,
        font_size: int = 12,
    ) -> float:
        """
        Draw text with proper math formatting and wrapping:
        - Tables rendered as actual PDF tables
        - Fractions rendered as stacked (numerator over denominator)
        - Parenthesized fractions get tall parentheses
        - Multiplication shown as · (dot)
        - Long lines wrap properly
        """
        import re
        
        # First check for markdown tables and extract them
        text_before, table_data, text_after = extract_markdown_table(text)
        
        if table_data:
            # We have a table - render text before, then table, then text after
            if text_before:
                cursor_y = self._draw_text_segment(canvas, text_before, cursor_y, font_size)
                cursor_y -= self.line_height * 0.5
            
            # Draw the table
            cursor_y = self._draw_pdf_table(canvas, table_data, cursor_y, font_size=11)
            
            # Draw text after the table
            if text_after:
                cursor_y -= self.line_height * 0.3
                cursor_y = self._draw_text_segment(canvas, text_after, cursor_y, font_size)
            
            return cursor_y
        
        # No table - process normally
        return self._draw_text_segment(canvas, text, cursor_y, font_size)

    def _draw_text_segment(
        self,
        canvas: pdf_canvas.Canvas,
        text: str,
        cursor_y: float,
        font_size: int = 12,
    ) -> float:
        """
        Draw a text segment (without tables) with math formatting.
        """
        import re
        
        # Format for middle school (· instead of *, etc.)
        text = format_for_middle_school(text)
        
        # Check for fractions
        fractions = extract_fractions(text)
        
        if not fractions:
            # No fractions, use regular wrapped text
            return self._draw_wrapped_text(canvas, text, cursor_y, font_size=font_size)
        
        # Has fractions - need smart wrapping
        canvas.setFont("Helvetica", font_size)
        max_width = self.pagesize[0] - self.left_margin - self.right_margin
        
        # Build segments: each segment is either text or a fraction
        segments = []
        last_end = 0
        for start, end, numerator, denominator, has_parens in fractions:
            # Text before fraction
            if start > last_end:
                segments.append(('text', text[last_end:start]))
            # The fraction itself
            segments.append(('fraction', numerator, denominator, has_parens))
            last_end = end
        # Text after last fraction
        if last_end < len(text):
            segments.append(('text', text[last_end:]))
        
        # Calculate width of each segment
        # Use larger gaps for better readability
        FRACTION_GAP = 10  # pixels gap after each fraction
        
        def get_segment_width(seg):
            if seg[0] == 'text':
                return canvas.stringWidth(seg[1], "Helvetica", font_size)
            else:  # fraction
                num, den, has_parens = seg[1], seg[2], seg[3]
                num_w = canvas.stringWidth(str(num), "Helvetica", font_size)
                den_w = canvas.stringWidth(str(den), "Helvetica", font_size)
                frac_w = max(num_w, den_w) + 12  # padding inside fraction
                if has_parens:
                    frac_w += font_size * 0.6 + 10  # parentheses width
                return frac_w + FRACTION_GAP  # gap after
        
        # Render segments with wrapping
        current_x = self.left_margin
        has_fraction_on_line = False
        
        for seg in segments:
            seg_width = get_segment_width(seg)
            
            # Check if we need to wrap
            if current_x + seg_width > self.left_margin + max_width and current_x > self.left_margin:
                # Wrap to new line
                cursor_y -= self.line_height * (1.8 if has_fraction_on_line else 1.0)
                current_x = self.left_margin
                has_fraction_on_line = False
                
                # Check for page break
                if cursor_y <= self.top_margin:
                    canvas.showPage()
                    cursor_y = self.pagesize[1] - self.top_margin
                    canvas.setFont("Helvetica", font_size)
            
            if seg[0] == 'text':
                # For text segments, we may need to wrap within the segment
                words = seg[1].split()
                for i, word in enumerate(words):
                    word_with_space = word + (' ' if i < len(words) - 1 else '')
                    word_width = canvas.stringWidth(word_with_space, "Helvetica", font_size)
                    
                    if current_x + word_width > self.left_margin + max_width and current_x > self.left_margin:
                        cursor_y -= self.line_height * (1.8 if has_fraction_on_line else 1.0)
                        current_x = self.left_margin
                        has_fraction_on_line = False
                        
                        if cursor_y <= self.top_margin:
                            canvas.showPage()
                            cursor_y = self.pagesize[1] - self.top_margin
                            canvas.setFont("Helvetica", font_size)
                    
                    # Check if this is an operator that should be vertically centered
                    # (when appearing between fractions)
                    stripped_word = word_with_space.strip()
                    is_operator = stripped_word in ['•', '÷', '+', '-', '=']
                    if is_operator and has_fraction_on_line:
                        # The fraction line is at cursor_y - that's our vertical center
                        # To center a character, we need baseline at: center - (char_height * 0.4)
                        # For most characters, visual center is ~40% up from baseline
                        
                        if stripped_word == '÷':
                            # Division symbol - BIGGER and centered
                            big_font = font_size * 1.6
                            canvas.setFont("Helvetica", big_font)
                            # Center of ÷ is about 45% up from baseline
                            operator_y = cursor_y - big_font * 0.45
                            canvas.drawString(current_x, operator_y, stripped_word)
                            current_x += canvas.stringWidth(stripped_word, "Helvetica", big_font)
                            canvas.setFont("Helvetica", font_size)
                        else:
                            # Bullet (•) and other operators - centered
                            # Center of • is about 50% up from baseline (it's a middle dot)
                            operator_y = cursor_y - font_size * 0.5
                            canvas.drawString(current_x, operator_y, stripped_word)
                            current_x += canvas.stringWidth(stripped_word, "Helvetica", font_size)
                        # Add spacing after operator
                        current_x += font_size * 0.3
                    else:
                        canvas.drawString(current_x, cursor_y, word_with_space)
                        current_x += word_width
            else:
                # Draw fraction with generous spacing
                num, den, has_parens = seg[1], seg[2], seg[3]
                frac_width = self._draw_stacked_fraction(
                    canvas, num, den, current_x, cursor_y, font_size, with_parens=has_parens
                )
                current_x += frac_width + FRACTION_GAP
                has_fraction_on_line = True
        
        return cursor_y - self.line_height * (1.8 if has_fraction_on_line else 1.0)

    def _draw_wrapped_text(
        self,
        canvas: pdf_canvas.Canvas,
        text: str,
        cursor_y: float,
        *,
        font: str = "Helvetica",
        font_size: int = 12,
    ) -> float:
        # Apply middle school formatting (× for multiplication, etc.)
        text = format_for_middle_school(text)
        
        canvas.setFont(font, font_size)
        max_width = self.pagesize[0] - self.left_margin - self.right_margin
        words = []
        line = ""
        for word in text.split():
            prospective = f"{line} {word}".strip()
            if canvas.stringWidth(prospective, font, font_size) <= max_width:
                line = prospective
            else:
                words.append(line)
                line = word
        if line:
            words.append(line)

        for part in words:
            if cursor_y <= self.top_margin:
                canvas.showPage()
                cursor_y = self.pagesize[1] - self.top_margin
                canvas.setFont(font, font_size)
            canvas.drawString(self.left_margin, cursor_y, part)
            cursor_y -= self.line_height
        return cursor_y


if __name__ == "__main__":
    sample_payload = WorksheetDocumentPayload(
        concept="Ratios",
        grade=7,
        difficulty="hard",
        num_problems=2,
        problems=[
            WorksheetDocumentProblem(prompt="A class has 12 girls and 9 boys. Write the ratio of girls to boys.", answer="4:3"),
            WorksheetDocumentProblem(prompt="Simplify the ratio 8:12.", answer="2:3"),
        ],
    )
    builder = PdfBuilder()
    path = builder.build(sample_payload, "sample_output.pdf")
    print(f"Wrote {path}")

