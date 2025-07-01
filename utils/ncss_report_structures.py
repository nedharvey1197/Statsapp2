"""
NCSS-Style Report Data Structures

This module defines the standard data structures for NCSS-style statistical reports.
All analysis scripts and utilities should use these structures for input/output
to ensure consistency and durability across different modeling approaches.

Classes:
    SectionType: Enumeration of standard NCSS report section types
    NCSSTable: Standard table structure with DataFrame conversion
    NCSSPlot: Standard plot structure with base64 encoding
    NCSSSection: Standard section structure containing tables and plots
    NCSSReport: Main container for NCSS-style reports

Factory Functions:
    create_model_summary_section: Creates a standard model summary section
    create_anova_section: Creates a standard ANOVA section
    create_estimates_section: Creates a standard parameter estimates section
    create_diagnostics_section: Creates a standard diagnostics section
    create_plots_section: Creates a standard plots section
    create_run_summary_section: Creates a standard Run Summary section
    create_key_results_section: Creates a standard Key Results section

Example:
    # Create a report
    report = NCSSReport(title="My Analysis")
    
    # Add a section
    anova_section = create_anova_section()
    report.add_section(anova_section)
    
    # Add a table to the section
    table = NCSSTable(
        title="ANOVA Results",
        columns=["Source", "DF", "F-Value", "P-Value"],
        rows=[{"Source": "Treatment", "DF": 2, "F-Value": 5.67, "P-Value": 0.001}]
    )
    anova_section.add_table(table)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import pandas as pd
import base64


class SectionType(Enum):
    """
    Standard NCSS report section types.
    
    These define the standard sections that appear in NCSS-style statistical reports.
    """
    TITLE = "title"
    RUN_SUMMARY = "run_summary"
    KEY_RESULTS = "key_results"
    MODEL_SUMMARY = "model_summary"
    ANOVA = "anova"
    ESTIMATES = "estimates"
    DIAGNOSTICS = "diagnostics"
    PLOTS = "plots"
    NOTES = "notes"


@dataclass
class NCSSTable:
    """
    Standard table structure for NCSS reports.
    
    Attributes:
        title: The title of the table
        columns: List of column names
        rows: List of dictionaries representing table rows
        notes: Optional notes about the table
    
    Methods:
        to_dataframe: Convert table to pandas DataFrame
        to_html: Convert table to HTML string
    """
    title: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    notes: Optional[str] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert table to pandas DataFrame for easier manipulation.
        
        Returns:
            pandas.DataFrame: The table data as a DataFrame
        """
        return pd.DataFrame(self.rows)
    
    def to_html(self) -> str:
        """
        Convert table to HTML string with NCSS styling.
        
        Returns:
            str: HTML table string with 'ncss-table' CSS class
        """
        df = self.to_dataframe()
        return df.to_html(index=False, classes='ncss-table')


@dataclass
class NCSSPlot:
    """
    Standard plot structure for NCSS reports.
    
    Attributes:
        title: The title of the plot
        image_bytes: Plot image as bytes
        description: Description of the plot
        plot_type: Type of plot (e.g., "residual", "qq", "histogram")
    
    Methods:
        get_image_base64: Convert image bytes to base64 for HTML display
    """
    title: str
    image_bytes: bytes
    description: str = ""
    plot_type: str = ""
    
    def get_image_base64(self) -> str:
        """
        Convert image bytes to base64 for HTML display.
        
        Returns:
            str: Base64 encoded image string
        """
        return base64.b64encode(self.image_bytes).decode('utf-8')


@dataclass
class NCSSSection:
    """
    Standard section structure for NCSS reports.
    
    Attributes:
        title: The title of the section
        section_type: Type of section (from SectionType enum)
        tables: List of tables in this section
        plots: List of plots in this section
        text: Optional text content for the section
        notes: Optional notes about the section
    
    Methods:
        add_table: Add a table to this section
        add_plot: Add a plot to this section
    """
    title: str
    section_type: SectionType
    tables: List[NCSSTable] = field(default_factory=list)
    plots: List[NCSSPlot] = field(default_factory=list)
    text: str = ""
    notes: Optional[str] = None
    
    def add_table(self, table: NCSSTable) -> None:
        """
        Add a table to this section.
        
        Args:
            table: The NCSSTable to add
        """
        self.tables.append(table)
    
    def add_plot(self, plot: NCSSPlot) -> None:
        """
        Add a plot to this section.
        
        Args:
            plot: The NCSSPlot to add
        """
        self.plots.append(plot)


@dataclass
class NCSSReport:
    """
    Main container for NCSS-style reports.
    
    Attributes:
        title: The title of the report
        sections: List of sections in the report
        metadata: Dictionary of metadata about the report
    
    Methods:
        add_section: Add a section to the report
        get_section: Get a section by type
        get_all_tables: Get all tables from all sections
        get_all_plots: Get all plots from all sections
    """
    title: str
    sections: List[NCSSSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: NCSSSection) -> None:
        """
        Add a section to the report.
        
        Args:
            section: The NCSSSection to add
        """
        self.sections.append(section)
    
    def get_section(self, section_type: SectionType) -> Optional[NCSSSection]:
        """
        Get a section by type.
        
        Args:
            section_type: The type of section to find
            
        Returns:
            NCSSSection or None: The section if found, None otherwise
        """
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def get_all_tables(self) -> List[NCSSTable]:
        """
        Get all tables from all sections.
        
        Returns:
            List[NCSSTable]: All tables in the report
        """
        tables = []
        for section in self.sections:
            tables.extend(section.tables)
        return tables
    
    def get_all_plots(self) -> List[NCSSPlot]:
        """
        Get all plots from all sections.
        
        Returns:
            List[NCSSPlot]: All plots in the report
        """
        plots = []
        for section in self.sections:
            plots.extend(section.plots)
        return plots


# Factory functions for common NCSS structures
def create_model_summary_section() -> NCSSSection:
    """
    Create a standard model summary section.
    
    Returns:
        NCSSSection: A model summary section
    """
    return NCSSSection(
        title="Model Summary",
        section_type=SectionType.MODEL_SUMMARY
    )


def create_anova_section() -> NCSSSection:
    """
    Create a standard ANOVA section.
    
    Returns:
        NCSSSection: An ANOVA section
    """
    return NCSSSection(
        title="Analysis of Variance",
        section_type=SectionType.ANOVA
    )


def create_estimates_section() -> NCSSSection:
    """
    Create a standard parameter estimates section.
    
    Returns:
        NCSSSection: A parameter estimates section
    """
    return NCSSSection(
        title="Parameter Estimates",
        section_type=SectionType.ESTIMATES
    )


def create_diagnostics_section() -> NCSSSection:
    """
    Create a standard diagnostics section.
    
    Returns:
        NCSSSection: A diagnostics section
    """
    return NCSSSection(
        title="Model Diagnostics",
        section_type=SectionType.DIAGNOSTICS
    )


def create_plots_section() -> NCSSSection:
    """
    Create a standard plots section.
    
    Returns:
        NCSSSection: A plots section
    """
    return NCSSSection(
        title="Diagnostic Plots",
        section_type=SectionType.PLOTS
    )


def create_run_summary_section() -> NCSSSection:
    """
    Create a standard Run Summary section.
    Returns:
        NCSSSection: A Run Summary section
    """
    return NCSSSection(
        title="Run Summary",
        section_type=SectionType.RUN_SUMMARY
    )


def create_key_results_section() -> NCSSSection:
    """
    Create a standard Key Results section.
    Returns:
        NCSSSection: A Key Results section
    """
    return NCSSSection(
        title="Summary of Key Results",
        section_type=SectionType.KEY_RESULTS
    ) 