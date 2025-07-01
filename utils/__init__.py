"""
NCSS-Style Utilities Package

This package provides utilities for creating NCSS-style statistical reports
with consistent data structures and output formats.
"""

from .ncss_report_structures import (
    NCSSReport, NCSSSection, NCSSTable, NCSSPlot, SectionType,
    create_model_summary_section, create_anova_section, create_estimates_section,
    create_diagnostics_section, create_plots_section,
    create_run_summary_section, create_key_results_section
)

from .ncss_plot_utils import (
    create_residual_plot, create_qq_plot, create_histogram_plot,
    create_leverage_plot, create_all_diagnostic_plots, create_correlation_plot
)

from .ncss_export_utils import (
    export_table_to_csv, export_table_to_excel, export_report_to_excel,
    export_report_to_json, export_tables_to_csv_batch, create_data_dictionary,
    export_data_dictionary, create_regulatory_summary, export_regulatory_summary
)

from .ncss_report_builder import (
    build_ncss_pdf, display_ncss_report_in_streamlit, build_ncss_html,
    create_ncss_report, add_sas_results_to_report, format_ncss_table_rows
)

__all__ = [
    # Data structures
    'NCSSReport', 'NCSSSection', 'NCSSTable', 'NCSSPlot', 'SectionType',
    'create_model_summary_section', 'create_anova_section', 'create_estimates_section',
    'create_diagnostics_section', 'create_plots_section',
    'create_run_summary_section', 'create_key_results_section',
    
    # Plot utilities
    'create_residual_plot', 'create_qq_plot', 'create_histogram_plot',
    'create_leverage_plot', 'create_all_diagnostic_plots', 'create_correlation_plot',
    
    # Export utilities
    'export_table_to_csv', 'export_table_to_excel', 'export_report_to_excel',
    'export_report_to_json', 'export_tables_to_csv_batch', 'create_data_dictionary',
    'export_data_dictionary', 'create_regulatory_summary', 'export_regulatory_summary',
    
    # Report builder
    'build_ncss_pdf', 'display_ncss_report_in_streamlit', 'build_ncss_html',
    'create_ncss_report', 'add_sas_results_to_report',
    # Formatting utility
    'format_ncss_table_rows'
] 