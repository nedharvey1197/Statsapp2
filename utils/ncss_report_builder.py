"""
NCSS-Style Report Builder

This module provides functions to build NCSS-style reports from the standard
data structures and generate various output formats (PDF, HTML, UI).

Functions:
    build_ncss_pdf: Build PDF report from NCSS data structures
    display_ncss_report_in_streamlit: Display report in Streamlit UI
    build_ncss_html: Build HTML report from NCSS data structures
    create_ncss_report: Create new NCSS report with standard sections
    add_sas_results_to_report: Add SAS results to NCSS report

Classes:
    NCSSPDFBuilder: Custom PDF builder for NCSS-style reports

Example:
    # Create a report
    report = create_ncss_report("My Analysis")
    
    # Add SAS results
    add_sas_results_to_report(report, sas_results)
    
    # Generate PDF
    pdf_bytes = build_ncss_pdf(report)
    
    # Display in Streamlit
    display_ncss_report_in_streamlit(report)
"""

import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
from fpdf import FPDF

from .ncss_report_structures import (
    NCSSReport, NCSSSection, NCSSTable, NCSSPlot, 
    SectionType, create_model_summary_section, create_anova_section,
    create_estimates_section, create_diagnostics_section, create_plots_section
)


class NCSSPDFBuilder(FPDF):
    """
    Custom PDF builder for NCSS-style reports.
    
    This class extends FPDF to provide NCSS-specific formatting
    for tables, plots, and sections.
    """
    
    def __init__(self):
        """Initialize the PDF builder with NCSS formatting."""
        super().__init__(orientation='L')  # Landscape orientation
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)
    
    def chapter_title(self, title: str) -> None:
        """
        Add a chapter title to the PDF.
        
        Args:
            title: The title text
        """
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, title, ln=True, align='C')
        self.ln(5)
    
    def section_title(self, title: str) -> None:
        """
        Add a section title to the PDF.
        
        Args:
            title: The section title text
        """
        self.set_font("Arial", 'B', 14)
        self.cell(0, 8, title, ln=True)
        self.ln(2)
    
    def add_table(self, table: NCSSTable) -> None:
        """
        Add a table to the PDF with proper formatting.
        
        Args:
            table: The NCSSTable to add
        """
        # Add table title
        self.set_font("Arial", 'B', 12)
        self.cell(0, 6, table.title, ln=True)
        self.ln(2)
        
        # Improved column width logic
        min_widths = {
            'DF': 12, 'N': 12, 'Numerator DF': 14, 'Denominator DF': 18, 'p-value': 18, 'P-Value': 18, 'ProbF': 18, 'Probt': 18,
            'Alpha': 14, 'StdErr': 18, 'Std Error': 18, 't-Value': 18, 'F-Value': 18
        }
        max_widths = {
            'Effect': 38, 'Comparison': 44, 'Name': 44, 'Model Term': 38
        }
        default_min = 18
        default_max = 32
        col_widths = []
        for col in table.columns:
            if col in min_widths:
                col_widths.append(min_widths[col])
            elif col in max_widths:
                col_widths.append(max_widths[col])
            else:
                # Estimate width based on column name and data
                max_width = len(str(col))
                for row in table.rows[:10]:
                    if col in row:
                        max_width = max(max_width, len(str(row[col])))
                col_widths.append(min(max(max_width * 2, default_min), default_max))
        # Adjust total width to fit page (landscape width ~277mm)
        total_width = sum(col_widths)
        if total_width > 260:
            scale = 260 / total_width
            col_widths = [w * scale for w in col_widths]
        # Add header
        self.set_font("Arial", 'B', 10)
        for i, col in enumerate(table.columns):
            self.cell(col_widths[i], 6, str(col), border=1)
        self.ln()
        # Add data rows (formatted)
        self.set_font("Arial", size=8)
        formatted_rows = format_ncss_table_rows(table.rows, table.columns)
        for row in formatted_rows:
            for i, col in enumerate(table.columns):
                value = str(row.get(col, '')) if col in row else ''
                self.cell(col_widths[i], 5, value, border=1)
            self.ln()
        self.ln(5)
    
    def add_plot(self, plot: NCSSPlot) -> None:
        """
        Add a plot to the PDF.
        Embeds the actual image if available.
        Args:
            plot: The NCSSPlot to add
        """
        # Add plot title
        self.set_font("Arial", 'B', 12)
        self.cell(0, 6, plot.title, ln=True)
        self.ln(2)
        # Add plot description if available
        if plot.description:
            self.set_font("Arial", size=10)
            self.multi_cell(0, 4, plot.description)
            self.ln(2)
        # Embed image if available
        if hasattr(plot, 'image_bytes') and plot.image_bytes:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                tmp_img.write(plot.image_bytes)
                tmp_img.flush()
                self.image(tmp_img.name, w=96)  # width in mm (20% smaller than 120)
            self.ln(5)
        else:
            # Add placeholder for plot
            self.set_font("Arial", size=10)
            self.cell(0, 6, f"[Plot: {plot.title}]", ln=True)
            self.ln(5)


def format_ncss_table_rows(rows, columns):
    """
    Format a list of table rows (list of dicts) according to NCSS reporting standards.
    Args:
        rows: List of dicts (table rows)
        columns: List of column names
    Returns:
        List of dicts with formatted values as strings
    """
    import re
    formatted_rows = []
    # Define column type heuristics
    pval_cols = [c for c in columns if re.search(r"p[-_ ]?val|prob|adjp|probt|p-value", c, re.I)]
    int_cols = [c for c in columns if re.search(r"df|n$|subjects?$", c, re.I)]
    float_cols = [c for c in columns if re.search(r"mean|estimate|std|stderr|se|ci|lower|upper|difference|value|f-value|t-value|alpha|component|variance|covparm", c, re.I) and c not in pval_cols + int_cols]

    for row in rows:
        formatted_row = {}
        for col in columns:
            val = row.get(col, "")
            # Try to format numbers
            try:
                if col in pval_cols:
                    v = float(val)
                    if v < 0.001:
                        formatted_row[col] = "<0.001"
                    else:
                        formatted_row[col] = f"{v:.3f}"
                elif col in int_cols:
                    v = float(val)
                    if abs(v - int(v)) < 1e-6:
                        formatted_row[col] = str(int(round(v)))
                    else:
                        formatted_row[col] = f"{v:.2f}"
                elif col in float_cols:
                    v = float(val)
                    formatted_row[col] = f"{v:.2f}"
                else:
                    formatted_row[col] = str(val)
            except Exception:
                formatted_row[col] = str(val)
        formatted_rows.append(formatted_row)
    return formatted_rows


def build_ncss_pdf(report: NCSSReport, extra_plots: Optional[list] = None) -> bytes:
    """
    Build a PDF report from NCSS data structures.
    
    Creates a professional PDF report with proper NCSS formatting,
    including tables, plots, and metadata.
    
    Args:
        report: NCSS report to convert to PDF
        extra_plots: Optional list of NCSSPlot to add to diagnostics section
    
    Returns:
        bytes: PDF content as bytes
        
    Raises:
        Exception: If PDF generation fails
    """
    try:
        pdf = NCSSPDFBuilder()
        
        # Add title page
        pdf.chapter_title(report.title)
        pdf.ln(10)
        
        # Add metadata if available
        if report.metadata:
            pdf.set_font("Arial", size=10)
            for key, value in report.metadata.items():
                pdf.cell(0, 5, f"{key}: {value}", ln=True)
            pdf.ln(10)
        
        # Add sections
        for section in report.sections:
            pdf.section_title(section.title)
            
            # Add section text if available
            if section.text:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 4, section.text)
                pdf.ln(2)
            
            # Add tables
            for table in section.tables:
                pdf.add_table(table)
            
            # Add plots
            for plot in section.plots:
                pdf.add_plot(plot)
            
            # Add extra plots to diagnostics section
            if extra_plots and section.section_type == SectionType.DIAGNOSTICS:
                for plot in extra_plots:
                    pdf.add_plot(plot)
            
            # Add section notes if available
            if section.notes:
                pdf.set_font("Arial", 'I', 9)
                pdf.multi_cell(0, 4, f"Note: {section.notes}")
                pdf.ln(2)
            
            pdf.ln(5)
        
        return bytes(pdf.output(dest='S'))
    except Exception as e:
        raise Exception(f"Failed to generate PDF: {e}")


def display_ncss_report_in_streamlit(report: NCSSReport) -> None:
    """
    Display NCSS report in Streamlit UI.
    
    Creates a structured display of the report with proper formatting
    for tables, plots, and metadata.
    
    Args:
        report: NCSS report to display
    """
    # Title
    st.title(report.title)
    
    # Metadata
    if report.metadata:
        with st.expander("Analysis Information"):
            for key, value in report.metadata.items():
                st.write(f"**{key}:** {value}")
    
    # Sections
    for section in report.sections:
        st.header(section.title)
        
        # Section text
        if section.text:
            st.write(section.text)
        
        # Tables
        for table in section.tables:
            st.subheader(table.title)
            df = table.to_dataframe()
            st.dataframe(df, use_container_width=True)
            
            # Table notes
            if table.notes:
                st.caption(f"Note: {table.notes}")
        
        # Plots
        for plot in section.plots:
            st.subheader(plot.title)
            
            # Display plot
            st.image(plot.image_bytes, caption=plot.description)
        
        # Section notes
        if section.notes:
            st.info(f"**Note:** {section.notes}")


def build_ncss_html(report: NCSSReport) -> str:
    """
    Build an HTML report from NCSS data structures.
    
    Creates a complete HTML report with embedded plots and
    proper NCSS styling.
    
    Args:
        report: NCSS report to convert to HTML
    
    Returns:
        str: Complete HTML content
        
    Raises:
        Exception: If HTML generation fails
    """
    try:
        html_parts = []
        
        # Start HTML
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>""" + report.title + """</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .title { text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                .section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
                .table-title { font-size: 14px; font-weight: bold; margin-bottom: 5px; }
                .ncss-table { border-collapse: collapse; width: 100%; margin-bottom: 15px; }
                .ncss-table th, .ncss-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .ncss-table th { background-color: #f2f2f2; }
                .plot { margin-bottom: 15px; }
                .plot img { max-width: 80%; height: auto; }  /* 20% smaller than 100% */
                .notes { font-style: italic; color: #666; margin-top: 5px; }
            </style>
        </head>
        <body>
        """)
        
        # Title
        html_parts.append(f'<div class="title">{report.title}</div>')
        
        # Metadata
        if report.metadata:
            html_parts.append('<div class="section">')
            html_parts.append('<div class="section-title">Analysis Information</div>')
            for key, value in report.metadata.items():
                html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
            html_parts.append('</div>')
        
        # Sections
        for section in report.sections:
            html_parts.append('<div class="section">')
            html_parts.append(f'<div class="section-title">{section.title}</div>')
            
            # Section text
            if section.text:
                html_parts.append(f'<p>{section.text}</p>')
            
            # Tables
            for table in section.tables:
                html_parts.append(f'<div class="table-title">{table.title}</div>')
                # Format rows before rendering
                formatted_rows = format_ncss_table_rows(table.rows, table.columns)
                df = pd.DataFrame(formatted_rows)
                html_parts.append(df.to_html(index=False, classes='ncss-table'))
                if table.notes:
                    html_parts.append(f'<div class="notes">Note: {table.notes}</div>')
            
            # Plots
            for plot in section.plots:
                html_parts.append('<div class="plot">')
                html_parts.append(f'<div class="table-title">{plot.title}</div>')
                html_parts.append(f'<img src="data:image/png;base64,{plot.get_image_base64()}" alt="{plot.title}">')
                if plot.description:
                    html_parts.append(f'<p>{plot.description}</p>')
                html_parts.append('</div>')
            
            # Section notes
            if section.notes:
                html_parts.append(f'<div class="notes"><strong>Note:</strong> {section.notes}</div>')
            
            html_parts.append('</div>')
        
        # End HTML
        html_parts.append("""
        </body>
        </html>
        """)
        
        return '\n'.join(html_parts)
    except Exception as e:
        raise Exception(f"Failed to generate HTML: {e}")


def create_ncss_report(title: str, metadata: Optional[Dict[str, Any]] = None) -> NCSSReport:
    """
    Create a new NCSS report with standard sections.
    
    This function creates a report with all the standard NCSS sections
    (Model Summary, ANOVA, Estimates, Diagnostics, Plots) ready to be
    populated with data.
    
    Args:
        title: Report title
        metadata: Optional metadata dictionary
    
    Returns:
        NCSSReport: New NCSS report with standard sections
    """
    report = NCSSReport(title=title, metadata=metadata or {})
    
    # Add standard sections
    report.add_section(create_model_summary_section())
    report.add_section(create_anova_section())
    report.add_section(create_estimates_section())
    report.add_section(create_diagnostics_section())
    report.add_section(create_plots_section())
    
    return report


def add_sas_results_to_report(report: NCSSReport, sas_results: Dict[str, Any]) -> None:
    """
    Add SAS results to an NCSS report.
    
    This function takes SAS output datasets and adds them to the
    appropriate sections of the NCSS report.
    
    Args:
        report: NCSS report to add results to
        sas_results: Dictionary containing SAS output tables and results
        
    Raises:
        ValueError: If report is None or sas_results is invalid
    """
    if report is None:
        raise ValueError("Report cannot be None")
    
    if not isinstance(sas_results, dict):
        raise ValueError("sas_results must be a dictionary")
    
    # Add model summary
    if 'model_summary' in sas_results:
        model_section = report.get_section(SectionType.MODEL_SUMMARY)
        if model_section and 'model_summary' in sas_results:
            table = NCSSTable(
                title="Model Information",
                columns=list(sas_results['model_summary'].columns),
                rows=sas_results['model_summary'].to_dict('records')
            )
            model_section.add_table(table)
    
    # Add ANOVA results
    if 'anova' in sas_results:
        anova_section = report.get_section(SectionType.ANOVA)
        if anova_section:
            table = NCSSTable(
                title="Analysis of Variance",
                columns=list(sas_results['anova'].columns),
                rows=sas_results['anova'].to_dict('records')
            )
            anova_section.add_table(table)
    
    # Add parameter estimates
    estimates_df = None
    if 'estimates' in sas_results and sas_results['estimates'] is not None:
        estimates_df = sas_results['estimates']
    elif 'solution' in sas_results and sas_results['solution'] is not None:
        estimates_df = sas_results['solution']
    if estimates_df is not None:
        estimates_section = report.get_section(SectionType.ESTIMATES)
        if estimates_section:
            table = NCSSTable(
                title="Parameter Estimates",
                columns=list(estimates_df.columns),
                rows=estimates_df.to_dict('records')
            )
            estimates_section.add_table(table)
    
    # Add LS Means
    if 'lsmeans' in sas_results:
        estimates_section = report.get_section(SectionType.ESTIMATES)
        if estimates_section:
            table = NCSSTable(
                title="Least Squares Means",
                columns=list(sas_results['lsmeans'].columns),
                rows=sas_results['lsmeans'].to_dict('records')
            )
            estimates_section.add_table(table)

    # Add Pairwise Comparisons
    if 'diffs' in sas_results:
        estimates_section = report.get_section(SectionType.ESTIMATES)
        if estimates_section:
            table = NCSSTable(
                title="Pairwise Comparisons",
                columns=list(sas_results['diffs'].columns),
                rows=sas_results['diffs'].to_dict('records')
            )
            estimates_section.add_table(table)

    # Add diagnostics
    if 'diagnostics' in sas_results:
        diagnostics_section = report.get_section(SectionType.DIAGNOSTICS)
        if diagnostics_section:
            for diagnostic_name, diagnostic_data in sas_results['diagnostics'].items():
                if isinstance(diagnostic_data, pd.DataFrame):
                    table = NCSSTable(
                        title=diagnostic_name,
                        columns=list(diagnostic_data.columns),
                        rows=diagnostic_data.to_dict('records')
                    )
                    diagnostics_section.add_table(table)

    # Add Variance Components
    if 'covparms' in sas_results:
        diagnostics_section = report.get_section(SectionType.DIAGNOSTICS)
        if diagnostics_section:
            table = NCSSTable(
                title="Variance Components",
                columns=list(sas_results['covparms'].columns),
                rows=sas_results['covparms'].to_dict('records')
            )
            diagnostics_section.add_table(table) 