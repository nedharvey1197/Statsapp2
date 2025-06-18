from typing import List, Tuple, Union, Any
from fpdf import FPDF
from io import BytesIO

def init_pdf(title: str = "Statistical Analysis Report") -> FPDF:
    """Initialize a new PDF document with a title.
    
    Args:
        title: The title to display at the top of the PDF
        
    Returns:
        FPDF: A configured PDF document
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, str(title), ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    return pdf

def add_section_title(pdf: FPDF, title: str) -> None:
    """Add a section title to the PDF.
    
    Args:
        pdf: The PDF document to modify
        title: The section title to add
    """
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, str(title), ln=True)
    pdf.set_font("Arial", "", 10)

def add_text_block(pdf: FPDF, text: str) -> None:
    """Add a block of text to the PDF.
    
    Args:
        pdf: The PDF document to modify
        text: The text to add
    """
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, str(text))

def add_key_value_table(pdf: FPDF, data: List[Tuple[str, Any]], col_widths: Tuple[int, int] = (80, 110)) -> None:
    """Add a key-value table to the PDF.
    
    Args:
        pdf: The PDF document to modify
        data: List of (key, value) pairs to display
        col_widths: Tuple of (key_column_width, value_column_width)
    """
    for key, value in data:
        pdf.cell(col_widths[0], 8, str(key), border=1)
        pdf.cell(col_widths[1], 8, str(value), border=1, ln=True)

def export_pdf_to_bytes(pdf: FPDF) -> bytes:
    """Convert a PDF document to bytes for download.
    
    Args:
        pdf: The PDF document to convert
        
    Returns:
        bytes: The PDF document as bytes
    """
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
