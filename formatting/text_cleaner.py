"""
text_cleaner.py

Utilities for cleaning and formatting mathematical text for display in worksheets.
Removes LaTeX notation and converts to readable plain text with Unicode symbols.
"""

import re


def extract_markdown_table(text: str):
    """
    Extract markdown table from text and return structured data.
    
    Returns:
        tuple: (text_before, table_data, text_after) where table_data is 
               {'headers': [...], 'rows': [[...], ...]} or None if no table
    """
    if not text:
        return text, None, ""
    
    # Check if this looks like a markdown table
    if '|' not in text or text.count('|') < 4:
        return text, None, ""
    
    # Check for separator-based table pattern first
    separator_match = re.search(r'\|[\s]*[-:]+[\s]*\|[\s]*[-:]*[\s]*\|', text)
    
    if separator_match:
        return _extract_single_line_table(text, separator_match)
    
    # Check for simple pipe-delimited table (no separator row)
    # Pattern: "text. | Header1 | Header2 | val1 | val2 | val3 | val4 | more text"
    return _extract_simple_pipe_table(text)


def _extract_simple_pipe_table(text: str):
    """
    Extract a simple pipe-delimited table without separator row.
    Handles formats like: "Complete the table. | x | y | -2 | -1 | 0 | 1 | 2 | Is this linear?"
    """
    # Find the span of pipe-delimited content
    # Look for pattern: | something | something | ... |
    pipe_match = re.search(r'\|[^|]+\|(?:[^|]*\|)+', text)
    
    if not pipe_match:
        return text, None, ""
    
    pipe_start = pipe_match.start()
    pipe_end = pipe_match.end()
    
    # Find text before and after
    text_before = text[:pipe_start].strip()
    pipe_content = text[pipe_start:pipe_end]
    text_after = text[pipe_end:].strip()
    
    # Clean up text_before - remove trailing punctuation if needed
    text_before = text_before.rstrip('.')
    if text_before:
        text_before += '.'
    
    # Parse the pipe content
    cells = [c.strip() for c in pipe_content.split('|') if c.strip()]
    
    if len(cells) < 4:  # Need at least 2 headers + 2 values
        return text, None, ""
    
    # Detect headers vs data
    # Headers are typically text (x, y, Time, Distance, etc.)
    # Data are typically numbers or placeholders (?, ___)
    
    # Strategy: First two cells that look like headers become headers
    # Rest become data paired into rows
    
    headers = []
    data_start = 0
    
    for i, cell in enumerate(cells):
        if _looks_like_header(cell):
            headers.append(cell)
            data_start = i + 1
            if len(headers) >= 2:
                break
        else:
            # First non-header cell means headers are done
            break
    
    if len(headers) < 2:
        # Try assuming first 2 cells are headers
        if len(cells) >= 2:
            headers = cells[:2]
            data_start = 2
        else:
            return text, None, ""
    
    # Build rows from remaining data
    num_cols = len(headers)
    data_cells = cells[data_start:]
    
    rows = []
    for i in range(0, len(data_cells), num_cols):
        row = data_cells[i:i + num_cols]
        # Pad with ? if row is incomplete
        while len(row) < num_cols:
            row.append('?')
        rows.append(row)
    
    if rows:
        table_data = {'headers': headers, 'rows': rows}
        return text_before, table_data, text_after
    
    return text, None, ""


def _looks_like_header(cell: str) -> bool:
    """Determine if a cell looks like a table header."""
    cell = cell.strip()
    
    # Empty is not a header
    if not cell:
        return False
    
    # Pure numbers are not headers
    if re.match(r'^-?\d+\.?\d*$', cell):
        return False
    
    # Placeholders are not headers
    if cell in ['?', '___', '____', '_', '']:
        return False
    
    # Contains letters - likely a header
    if re.search(r'[a-zA-Z]', cell):
        return True
    
    # Short single characters could be variables (x, y)
    if len(cell) == 1:
        return True
    
    return False


def _extract_single_line_table(text: str, separator_match):
    """Extract a table that's been compressed into a single line."""
    sep_start = separator_match.start()
    sep_end = separator_match.end()
    
    # Find where headers start - look for pattern before the separator
    # Headers are typically: "text text | header1 | header2 |---|---|"
    before_sep = text[:sep_start]
    after_sep = text[sep_end:]
    
    # Find where table headers start by looking for the first | before separator
    # that follows non-table text
    header_start = 0
    # Look for pattern like "table below. x y" which precedes the actual table headers
    # Find the last sentence-ending punctuation before the table
    for i in range(sep_start - 1, -1, -1):
        if text[i] in '.?!:':
            header_start = i + 1
            break
    
    text_before = text[:header_start].strip()
    header_section = text[header_start:sep_start].strip()
    
    # Parse headers - split by |
    headers = [h.strip() for h in header_section.split('|') if h.strip()]
    
    if not headers or len(headers) < 2:
        # Might be headers are just space-separated like "x y"
        # Try to detect simple headers
        words = header_section.split()
        if len(words) >= 2 and all(len(w) <= 10 for w in words):
            headers = words
    
    if not headers:
        return text, None, ""
    
    num_cols = len(headers)
    
    # Parse data after separator
    # Pattern: | val1 | val2 | | val3 | val4 | ... followed by text
    # Find where the table data ends (look for text that's not part of table)
    
    # First, let's find all the table values
    remaining = after_sep
    
    # Split by | and build rows
    parts = remaining.split('|')
    
    rows = []
    current_row = []
    text_after_parts = []
    in_table = True
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        if in_table:
            # Check if this looks like table data or continuation text
            # Table data is typically short numbers or "?" or empty
            if _looks_like_table_cell(part, current_row, num_cols):
                cell_value = part if part else "?"
                current_row.append(cell_value)
                
                if len(current_row) >= num_cols:
                    rows.append(current_row)
                    current_row = []
            else:
                # This might be the start of non-table text
                # Collect remaining parts as text_after
                in_table = False
                if part:
                    text_after_parts.append(part)
        else:
            if part:
                text_after_parts.append(part)
    
    # Handle any remaining partial row
    if current_row:
        while len(current_row) < num_cols:
            current_row.append("?")
        rows.append(current_row)
    
    text_after = ' '.join(text_after_parts).strip()
    
    # Clean up text_after - remove leading punctuation and clean formatting
    text_after = re.sub(r'^[\s\|]+', '', text_after)
    
    if headers and rows:
        table_data = {'headers': headers, 'rows': rows}
        return text_before, table_data, text_after
    
    return text, None, ""


def _looks_like_table_cell(part: str, current_row: list, num_cols: int) -> bool:
    """Determine if a string looks like a table cell value."""
    if not part:
        return True  # Empty cell is valid
    
    # Typical table cells are: numbers, ?, short text, negative numbers
    part = part.strip()
    
    # Check if it's a number (possibly negative or decimal)
    if re.match(r'^-?\d+\.?\d*$', part):
        return True
    
    # Check if it's a placeholder
    if part in ['?', '___', '____', '_', '']:
        return True
    
    # Check if it's very short (likely a variable or value)
    if len(part) <= 3:
        return True
    
    # If we're mid-row, be more lenient
    if current_row and len(current_row) < num_cols:
        return len(part) <= 10
    
    return False


def _clean_markdown_tables(text: str) -> str:
    """
    Clean up markdown table syntax - simplified version that just removes table syntax
    if we can't properly parse it.
    """
    if not text:
        return text
    
    # Remove table separator patterns
    text = re.sub(r'\|[\s]*[-:]+[\s]*\|[\s]*[-:]*[\s]*\|', ' ', text)
    text = re.sub(r'\|[\s]*[-:]+[\s]*\|', ' ', text)
    
    # Clean up multiple pipes
    text = re.sub(r'\|\s*\|', ' | ', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_math_text(text: str) -> str:
    """
    Converts LaTeX-style math notation to readable plain text with Unicode symbols.
    Also cleans up markdown table syntax.
    
    Examples:
        "Use \\( \\pi \\approx 3.14 \\)" -> "Use π ≈ 3.14"
        "\\frac{3}{4}" -> "3/4"
        "x^2 + y^2" -> "x² + y²"
    """
    if not text:
        return text
    
    # Clean up markdown table syntax - convert to readable format
    text = _clean_markdown_tables(text)
    
    # Fix corrupted Unicode escape sequences (e.g., "u00f7" -> "÷")
    # These can appear when JSON escape sequences get partially stripped
    unicode_fixes = {
        'u00f7': '÷',  # division
        'u00d7': '×',  # multiplication
        'u2212': '−',  # minus sign
        'u00b2': '²',  # squared
        'u00b3': '³',  # cubed
        'u00b0': '°',  # degree
        'u03c0': 'π',  # pi
        'u2248': '≈',  # approximately
        'u2264': '≤',  # less than or equal
        'u2265': '≥',  # greater than or equal
        'u2260': '≠',  # not equal
        'u00b1': '±',  # plus/minus
        'u221e': '∞',  # infinity
        'u221a': '√',  # square root
        'u2220': '∠',  # angle
    }
    
    for escaped, char in unicode_fixes.items():
        # Match both "u00f7" and "\u00f7" patterns
        text = text.replace(f'\\{escaped}', char)
        text = text.replace(escaped, char)
    
    # Remove LaTeX delimiters
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', text)
    
    # Convert common LaTeX commands to Unicode
    latex_to_unicode = {
        r'\\pi': 'π',
        r'\\approx': '≈',
        r'\\times': '×',
        r'\\div': '÷',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\pm': '±',
        r'\\infty': '∞',
        r'\\degree': '°',
        r'\\cdot': '·',
        r'\\sqrt': '√',
        r'\\angle': '∠',
        r'\\triangle': '△',
        r'\\parallel': '∥',
        r'\\perp': '⊥',
    }
    
    for latex, unicode_char in latex_to_unicode.items():
        text = re.sub(latex + r'(?!\w)', unicode_char, text)
    
    # Convert \frac{a}{b} to a/b
    # Match \frac{numerator}{denominator}
    def replace_frac(match):
        num = match.group(1)
        denom = match.group(2)
        return f"{num}/{denom}"
    
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, text)
    
    # Convert superscripts (simple cases like x^2, x^{10})
    def replace_superscript(match):
        base = match.group(1) if match.group(1) else ''
        exp = match.group(2) or match.group(3)
        
        # Unicode superscripts for common exponents
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '-': '⁻', '+': '⁺', '=': '⁼', '(': '⁽', ')': '⁾',
            'n': 'ⁿ',
        }
        
        # Try to convert to Unicode superscript
        if len(exp) <= 2 and all(c in superscript_map for c in exp):
            sup = ''.join(superscript_map[c] for c in exp)
            return f"{base}{sup}"
        else:
            # Fall back to caret notation for complex exponents
            return f"{base}^({exp})"
    
    # Match x^2 or x^{10}
    text = re.sub(r'(\w?)?\^(\d)(?!\d)', replace_superscript, text)
    text = re.sub(r'(\w?)?\^\{([^}]+)\}', replace_superscript, text)
    
    # Convert subscripts (simple cases like x_1, x_{10})
    def replace_subscript(match):
        base = match.group(1) if match.group(1) else ''
        sub = match.group(2) or match.group(3)
        
        subscript_map = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            '-': '₋', '+': '₊', '=': '₌', '(': '₍', ')': '₎',
        }
        
        if len(sub) <= 2 and all(c in subscript_map for c in sub):
            subscript = ''.join(subscript_map[c] for c in sub)
            return f"{base}{subscript}"
        else:
            return f"{base}_({sub})"
    
    text = re.sub(r'(\w?)?_(\d)(?!\d)', replace_subscript, text)
    text = re.sub(r'(\w?)?_\{([^}]+)\}', replace_subscript, text)
    
    # Clean up any remaining backslashes from LaTeX commands
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def format_for_middle_school(text: str) -> str:
    """
    Format mathematical text for middle school readability (grades 5-8).
    
    - Replaces asterisk (*) with visible bullet (•) for multiplication (avoids confusion with variable x)
    - Ensures division uses ÷ symbol
    - Adds proper spacing for readability
    
    Examples:
        "3 * 4" -> "3  •  4"
        "12 / 3" -> "12  ÷  3"
    """
    if not text:
        return text
    
    # First apply standard cleaning
    text = clean_math_text(text)
    
    # Replace asterisk with bullet point (• is more visible than · for multiplication)
    # Add extra spacing around operators for clarity
    text = text.replace(' * ', '  •  ')
    text = text.replace('*', '  •  ')
    
    # Also convert any × or · to bullet for consistency and visibility
    text = text.replace('×', '  •  ')
    text = text.replace('·', '  •  ')
    
    # Replace slash division with division sign (for simple expressions, not fractions)
    # Only replace when surrounded by spaces (to preserve fractions like 3/4)
    text = text.replace(' / ', '  ÷  ')
    
    # Add space after colons for better readability (e.g., "Multiply: 3/4" -> "Multiply:  3/4")
    text = text.replace(': ', ':  ')
    
    # Clean up any triple+ spaces
    while '   ' in text:
        text = text.replace('   ', '  ')
    
    return text


def extract_fractions(text: str) -> list:
    """
    Extract fractions from text for special rendering.
    Parentheses are NOT needed when there's an operator (•, ÷) in the expression.
    
    Returns list of tuples: [(start_pos, end_pos, numerator, denominator, has_parens), ...]
    """
    fractions = []
    # Pattern to match fractions, optionally wrapped in parentheses
    # Group 1: optional opening paren
    # Group 2: numerator (may include negative sign)
    # Group 3: denominator
    # Group 4: optional closing paren
    pattern = r'(\()?(-?\d+)/(\d+)(\))?'
    
    # Check if text has operators - if so, no parentheses needed
    has_operators = any(op in text for op in ['•', '÷', '×', '*'])
    
    for match in re.finditer(pattern, text):
        numerator = match.group(2)
        
        # Don't show parentheses when there are operators in the expression
        # The operators make it clear these are separate terms
        has_parens = False
        
        fractions.append((
            match.start(),
            match.end(),
            numerator,
            match.group(3),  # denominator
            has_parens
        ))
    
    return fractions


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Use \\( \\pi \\approx 3.14 \\)",
        "The area is \\frac{3}{4} square meters",
        "Solve x^2 + y^2 = 25",
        "Calculate \\sqrt{16} \\times 2",
        "The angle is 90\\degree",
        "Find \\frac{22}{7} \\times r^2",
    ]
    
    print("Math Text Cleaner Test Cases:")
    print("=" * 60)
    for test in test_cases:
        cleaned = clean_math_text(test)
        print(f"Input:  {test}")
        print(f"Output: {cleaned}")
        print("-" * 60)

