# Set page config must be the VERY FIRST Streamlit command
import streamlit as st
st.set_page_config(page_title="Simple GLM (One-Way ANOVA) - Manager Version", layout="wide")


import pandas as pd
import logging
import os
from datetime import datetime
import json
from typing import Dict, Any, List

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT UTILITIES AND MANAGERS
# =============================================================================

# NCSS Report Generation Utilities
# These handle the creation of professional statistical reports in NCSS format
from utils.ncss_report_structures import (
    NCSSReport, NCSSSection, NCSSTable, SectionType
)
from utils.ncss_plot_utils import create_all_diagnostic_plots
from utils.ncss_report_builder import (
    build_ncss_pdf, format_ncss_table_rows
)
from utils.ncss_export_utils import export_report_to_excel

# Output Management Utilities
# Handle folder creation and file organization for analysis outputs
from utils.output_folder_utils import get_or_create_analysis_folder, create_analysis_output_folder

# SAS Analysis Manager (NEW - Core Improvement)
# Centralized SAS connection, execution, error handling, and resource management
from utils.sas_analysis_manager import (
    run_simple_glm_analysis, convert_to_legacy_format, AnalysisType,
    create_sas_manager
)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_example_data():
    """
    Load the simple GLM example data for demonstration purposes.
    
    ROLE IN FLOW: 
    - Provides sample data when user chooses "Use Example Data"
    - Ensures consistent testing and demonstration environment
    - Handles file loading errors gracefully
    
    Returns:
        pd.DataFrame: Sample data with Treatment and TumorSize columns
        None: If loading fails (logged error)
    """
    try:
        df = pd.read_csv("data/simple_example.csv")
        return df
    except Exception as e:
        logger.error(f"Failed to load example data: {e}")
        return None


# =============================================================================
# CORE ANALYSIS FUNCTION (REFACTORED TO USE MANAGER)
# =============================================================================

def run_simple_glm_analysis_with_manager(data_file):
    import os
    from datetime import datetime
    import pandas as pd
    import saspy
    from utils.sas_analysis_manager import run_simple_glm_analysis, AnalysisResult
    import streamlit as st
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Folder setup (local and SAS)
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    daily_folder = datetime.now().strftime('%Y-%m-%d')
    daily_dir = os.path.join(logs_dir, daily_folder)
    os.makedirs(daily_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%H-%M-%S')
    session_dir = os.path.join(daily_dir, f'simple_glm_manager_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    output_folder = os.path.join(session_dir, 'output')
    os.makedirs(output_folder, exist_ok=True)

    # SAS path with your ODA user ID
    user_id = 'u64261399'  # Replace with your actual SAS ODA user ID (e.g., u6123456)
    sas_base_path = f'~/{daily_folder}/simple_glm_manager_{timestamp}'
    pdf_filename = f'sas_direct_report_{timestamp}.pdf'
    rtf_filename = f'sas_direct_report_{timestamp}.rtf'
    pdf_report_path = f'~/{daily_folder}/simple_glm_manager_{timestamp}/{pdf_filename}'
    rtf_report_path = f'~/{daily_folder}/simple_glm_manager_{timestamp}/{rtf_filename}'

    # Test write permissions (local)
    test_file = os.path.join(session_dir, 'test_write.tmp')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"Write permissions confirmed for: {session_dir}")
    except Exception as e:
        logger.error(f"Write permission error for {session_dir}: {e}")
        st.error(f"Cannot write to output directory: {session_dir}")
        return None

    # Load and validate data
    try:
        data = pd.read_csv(data_file)
        logger.info(f"Loaded data from file: {data.shape}")
        if 'Treatment' not in data.columns or 'TumorSize' not in data.columns:
            logger.error("Missing required columns: Treatment, TumorSize")
            st.error("Data must contain 'Treatment' and 'TumorSize' columns")
            return None
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        st.error(f"Failed to load data: {e}")
        return None

    # SAS code with original intent and path support
    sas_code = f"""
    options errors=20 nodate nonumber;
    ods trace on;
    libname outdir '{sas_base_path}';
    proc printto log='{sas_base_path}/sas_debug.log'; run;
    data _null_;
        file '{sas_base_path}/test_sas.txt';
        put "SAS write test";
    run;
    proc printto; run;
    ods pdf file='{pdf_report_path}' style=journal;
    ods rtf file='{rtf_report_path}' style=journal;
    ods output 
        FitStatistics=work.fitstats 
        OverallANOVA=work.anova 
        LSMeans=work.lsmeans 
        LSMeanDiffCL=work.diffs 
        ParameterEstimates=work.coeffs
        NObs=work.nobs
        ClassLevels=work.classlevels;
    proc glm data=work.testdata plots=diagnostics;
        class Treatment;
        model TumorSize = Treatment / solution;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
        output out=work.residuals r=resid p=pred;
    run;
    proc univariate data=work.residuals normal;
        var resid;
        ods output TestsForNormality=work.normtests;
    run;
    ods listing;
    proc report data=work.lsmeans nowd;
        column Treatment LSMean;
        define Treatment / display "Treatment";
        define LSMean / display "LSMean";
        title "LSMeans";
    run;
    proc report data=work.diffs nowd;
        column Treatment _Treatment Difference LowerCL UpperCL;  /* Corrected _Treatment_ to _Treatment */
        define Treatment / display "Treatment i";
        define _Treatment / display "Treatment j";
        define Difference / display "Difference";
        define LowerCL / display "Lower CI";
        define UpperCL / display "Upper CI";
        title "Pairwise Comparisons";
    run;
    ods pdf close;
    ods rtf close;
    ods trace off;
    quit;
    """

    # Execute analysis with session persistence
    logger.info("Initializing SAS session")
    sas = saspy.SASsession(cfgname='oda')
    try:
        logger.info("Transferring data to SAS")
        sas.df2sd(data, 'testdata')
        logger.info("Data transferred to SAS table 'work.testdata'")

        logger.info("Running SAS analysis")
        analysis_result = run_simple_glm_analysis(
            data_file=data_file,
            sas_code=sas_code,
            session_dir=session_dir,
            output_folder=output_folder,
            data=data
        )

        if not analysis_result.success:
            logger.error(f"Analysis failed: {analysis_result.error_message}")
            st.error(f"Analysis failed: {analysis_result.error_message}")
            return None

        logger.info("Converting results to legacy format")
        results = convert_to_legacy_format(analysis_result)
        results['data_info'] = data_info
        results['direct_reports'] = {
            'pdf': os.path.join(session_dir, pdf_filename),
            'rtf': os.path.join(session_dir, rtf_filename)
        }
    except Exception as e:
        logger.error(f"Session error: {e}")
        st.error(f"Session error: {e}")
        return None
    finally:
        logger.info("Closing SAS session")
        sas.endsas()
        logger.info("SAS session closed successfully")

    logger.info("Analysis completed successfully")
    return results




# =============================================================================
# REPORT GENERATION FUNCTIONS
# =============================================================================

def create_ncss_report_from_sas_results(sas_results, title="Simple GLM (One-Way ANOVA)"):
    """
    Create NCSS-style report from SAS analysis results.
    
    ROLE IN FLOW:
    - Transforms raw SAS results into structured NCSS report format
    - Creates UI-displayable sections with tables and explanations
    - Provides clinical interpretation and statistical definitions
    - Maintains professional formatting for export options
    
    This function is UNCHANGED from the original - demonstrates backward compatibility
    The manager refactoring only affects the analysis execution, not report generation
    
    Args:
        sas_results (dict): Analysis results from SAS (legacy format)
        title (str): Report title
        
    Returns:
        NCSSReport: Structured report object for UI display and export
        None: If no results available
    """
    logger.info("Starting NCSS report creation...")
    if not sas_results:
        logger.error("No SAS results available to create report")
        st.error("No SAS results available to create report")
        return None
    
    # =============================================================================
    # REPORT METADATA AND INITIALIZATION
    # =============================================================================
    
    # Set up report metadata for professional documentation
    metadata = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'software_version': 'SAS 9.4',
        'analysis_type': 'One-Way ANOVA',
        'model': 'PROC GLM'
    }
    report = NCSSReport(title=title, metadata=metadata)

    # =============================================================================
    # RUN SUMMARY SECTION (Model Overview and Metadata)
    # =============================================================================
    
    # Create run summary section with model overview information
    run_summary_section = NCSSSection(title="Run Summary", section_type=SectionType.RUN_SUMMARY)
    
    # Add model metadata from manager (analysis type, timestamp, etc.)
    model_state = sas_results.get('model_state', {})
    if model_state:
        # Use organized summary for NCSS report as well
        sas_manager = create_sas_manager()
        if hasattr(sas_manager, 'get_organized_run_summary'):
            organized_summary = sas_manager.get_organized_run_summary(model_state)
            
            # Create separate tables for each category
            for category, items in organized_summary.items():
                if category != 'System Information' and items:  # Skip system info in NCSS report
                    meta_rows = [{"Parameter": label, "Value": value} for label, value in items]
                    meta_table = NCSSTable(
                        title=f"{category}",
                        columns=["Parameter", "Value"],
                        rows=meta_rows
                    )
                    run_summary_section.add_table(meta_table)
        else:
            # Fallback to original format
            meta_rows = [{"Parameter": k, "Value": v} for k, v in model_state.items()]
            meta_table = NCSSTable(
                title="Key Statistics and Model Metadata",
                columns=["Parameter", "Value"],
                rows=meta_rows
            )
            run_summary_section.add_table(meta_table)
    
    # Add SAS-generated fit statistics (preserve natural SAS structure)
    fitstats = sas_results.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        # Use SAS column names directly for NCSS compatibility
        fitstats_table = NCSSTable(
            title="Fit Statistics",
            columns=list(fitstats.columns),
            rows=fitstats.to_dict('records')
        )
        run_summary_section.add_table(fitstats_table)
    
    # Add class levels information (treatment groups)
    classlevels = sas_results.get('classlevels')
    if classlevels is not None and not classlevels.empty:
        class_table = NCSSTable(
            title="Class Level Information",
            columns=list(classlevels.columns),
            rows=classlevels.to_dict('records')
        )
        run_summary_section.add_table(class_table)
    
    report.add_section(run_summary_section)
    
    # Report Definitions section (NCSS standard)
    definitions_section = NCSSSection(title="Report Definitions", section_type=SectionType.NOTES)
    definitions_text = """
    R-Square: The proportion of variance in the response variable explained by the model.
    Adjusted R-Square: R-square adjusted for the number of parameters in the model.
    Root MSE: The square root of the mean squared error, a measure of model fit.
    F-Value: The F-statistic for testing the null hypothesis that all group means are equal.
    P-Value: The probability of observing the F-statistic or more extreme under the null hypothesis.
    Least Squares Means: Model-based estimates of group means, adjusted for other factors.
    Standard Error: The standard error of the least squares mean.
    t-Value: The t-statistic for testing pairwise differences between groups.
    Bonferroni P-Value: P-value adjusted for multiple comparisons using Bonferroni method.
    Convergence: 'Normal (GLM)' indicates that the analysis completed successfully.
    """
    definitions_section.text = definitions_text
    report.add_section(definitions_section)

    # ANOVA section - preserve SAS natural structure
    anova_section = NCSSSection(title="Analysis of Variance", section_type=SectionType.ANOVA)
    anova = sas_results.get('anova')
    if anova is not None and not anova.empty:
        # Use SAS column names directly (they're already NCSS-compatible)
        anova_table = NCSSTable(
            title="ANOVA Results",
            columns=list(anova.columns),
            rows=anova.to_dict('records')
        )
        anova_section.add_table(anova_table)
    report.add_section(anova_section)

    # LS Means section
    lsmeans_section = NCSSSection(title="Least Squares Means", section_type=SectionType.ESTIMATES)
    lsmeans = sas_results.get('lsmeans')
    if lsmeans is not None and not lsmeans.empty:
        lsmeans_table = NCSSTable(
            title="Least Squares Means",
            columns=list(lsmeans.columns),
            rows=lsmeans.to_dict('records')
        )
        lsmeans_section.add_table(lsmeans_table)
    report.add_section(lsmeans_section)

    # Pairwise Comparisons section
    comparisons_section = NCSSSection(title="Pairwise Comparisons", section_type=SectionType.ESTIMATES)
    diffs = sas_results.get('diffs')
    if diffs is not None and not diffs.empty:
        comparisons_table = NCSSTable(
            title="Pairwise Comparisons",
            columns=list(diffs.columns),
            rows=diffs.to_dict('records')
        )
        comparisons_section.add_table(comparisons_table)
    report.add_section(comparisons_section)

    # Parameter Estimates section
    estimates_section = NCSSSection(title="Parameter Estimates", section_type=SectionType.ESTIMATES)
    coeffs = sas_results.get('coeffs')
    if coeffs is not None and not coeffs.empty:
        estimates_table = NCSSTable(
            title="Parameter Estimates",
            columns=list(coeffs.columns),
            rows=coeffs.to_dict('records')
        )
        estimates_section.add_table(estimates_table)
    report.add_section(estimates_section)

    # Diagnostics section (Normality)
    diagnostics_section = NCSSSection(title="Diagnostics", section_type=SectionType.DIAGNOSTICS)
    normtests = sas_results.get('normtests')
    if normtests is not None and not normtests.empty:
        norm_table = NCSSTable(
            title="Normality Tests",
            columns=list(normtests.columns),
            rows=normtests.to_dict('records')
        )
        diagnostics_section.add_table(norm_table)
    report.add_section(diagnostics_section)

    # Plots section (diagnostic plots)
    plots_section = NCSSSection(title="Diagnostic Plots", section_type=SectionType.PLOTS)
    residuals = sas_results.get('residuals')
    predicted = sas_results.get('predicted', [])
    if residuals is not None and len(residuals) > 0:
        # Convert to numpy arrays only for the plotting function if needed
        try:
            import numpy as np
            residuals_array = np.array(residuals)
            predicted_array = np.array(predicted) if predicted else np.zeros_like(residuals_array)
            plots = create_all_diagnostic_plots(residuals_array, predicted_array)
            for plot in plots:
                plots_section.add_plot(plot)
        except ImportError:
            st.warning("Numpy not available for plotting. Skipping diagnostic plots.")
    report.add_section(plots_section)

    return report


# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================

def main():
    """
    Main Streamlit application function for Simple GLM analysis.
    
    ROLE IN FLOW:
    - Entry point for the Streamlit web application
    - Manages user interface and user interactions
    - Orchestrates the complete analysis workflow
    - Handles session state management for persistence
    - Provides multiple export options for results
    
    APPLICATION FLOW:
    1. Setup: Initialize page config and session state
    2. Data Input: Allow user to upload data or use example data
    3. Analysis: Execute SAS analysis using the manager
    4. Results Display: Show both direct SAS reports and NCSS UI reports
    5. Export: Provide multiple download options (PDF, RTF, Excel)
    6. Debug: Optional debug information for troubleshooting
    
    KEY FEATURES:
    - Side-by-side comparison of original vs manager version
    - Real-time data preview and validation
    - Professional report generation and export
    - Comprehensive error handling and user feedback
    """
    
    # =============================================================================
    # SESSION STATE INITIALIZATION
    # =============================================================================
    
    # Initialize session state for persistent data across interactions
    if 'ncss_report' not in st.session_state:
        st.session_state['ncss_report'] = None
    if 'sas_results' not in st.session_state:
        st.session_state['sas_results'] = None
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'direct_reports' not in st.session_state:
        st.session_state['direct_reports'] = None
    
    # =============================================================================
    # APPLICATION HEADER AND VERSION COMPARISON
    # =============================================================================
    
    st.title("Simple GLM (One-Way ANOVA) - Manager Version")
    with st.expander("ℹ️ Version & Info", expanded=False):
        st.markdown("""
        **FLOW OVERVIEW:**
        1. **Data Loading:** User uploads data or uses example data
        2. **Analysis Execution:** SAS Analysis Manager handles SAS connection, execution, and error handling
        3. **Results Processing:** Convert manager results to legacy format for compatibility
        4. **Report Generation:** Create NCSS-style reports for UI display and export
        5. **Export Options:** Provide multiple output formats (PDF, RTF, Excel)

        **KEY IMPROVEMENTS OVER ORIGINAL:**
        - Centralized error handling via SAS Analysis Manager
        - Automated resource cleanup and session management
        - Enhanced logging and debugging capabilities
        - Modular design for future expansion
        - Maintains full backward compatibility with existing report functions

        **Entry point for the Streamlit application:**
        This file demonstrates the refactored approach using the SAS Analysis Manager:
        - Centralized error handling and resource management
        - Maintains full backward compatibility with existing report functions
        - Enhanced logging and debugging capabilities
        - Modular design for future expansion
        
        **To run:**
        ```bash
        streamlit run Simple_GLM_SAS_ncss_manager.py
        ```
        """)

    # =============================================================================
    # DATA INPUT SECTION
    # =============================================================================
    data_option = st.radio(
            "Choose your data source:",
            ["Use Example Data", "Upload My Own Data"],
            index=0
        )
    with st.expander("📊 Data Selection & Overview", expanded=False):
        # Allow user to choose between example data or file upload
        
        if data_option == "Upload My Own Data":
            uploaded_file = st.file_uploader("Upload CSV data file", type=['csv'])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                data.to_csv("temp_data.csv", index=False)
                st.success(f"File uploaded: {uploaded_file.name}")
            else:
                st.info("Please upload a CSV file to continue")
                st.stop()
        else:
            data = load_example_data()
            if data is not None:
                data.to_csv("temp_data.csv", index=False)
            else:
                st.error("Failed to load example data")
                st.stop()
        st.subheader("📋 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Observations", len(data))
        with col2:
            st.metric("Groups", len(data['Treatment'].unique()))
        st.dataframe(data, use_container_width=True)
    
    # =============================================================================
    # ANALYSIS EXECUTION SECTION
    # =============================================================================
    
    st.subheader("📈 Analysis Results")
    
    # Main analysis button - triggers the manager-based analysis
    if st.button("🚀 Run SAS Analysis (Manager Version)", type="primary"):
        try:
            with st.spinner("Running SAS analysis with Manager..."):
                sas_results = run_simple_glm_analysis_with_manager("temp_data.csv")
                if sas_results is None:
                    st.error("❌ Analysis returned None - check SAS connection and data")
                    return
        except Exception as e:
            st.error(f"❌ Analysis failed with exception: {str(e)}")
            logger.error(f"Analysis exception: {e}")
            return
        
        if sas_results:
            st.success("✅ Analysis completed successfully!")
            
            # =============================================================================
            # RUN SUMMARY SECTION (Dynamic Organized Display)
            # =============================================================================
            
            # Display organized run summary using the new manager method
            if 'model_state' in sas_results and sas_results['model_state']:
                st.header("📊 Run Summary")
                st.info("Organized summary of model information, experiment variables, and results")
                
                # Use the new dynamic organized summary display
                sas_manager = create_sas_manager()
                if hasattr(sas_manager, 'format_run_summary_for_display'):
                    run_summary_markdown = sas_manager.format_run_summary_for_display(sas_results['model_state'])
                    st.markdown(run_summary_markdown)
                else:
                    # Fallback to simple table if new method not available
                    model_state = sas_results['model_state']
                    summary_data = []
                    for key, value in model_state.items():
                        if key not in ['analysis_type', 'timestamp', 'SAS datasets_found', 'SAS total_datasets']:
                            summary_data.append([key.replace('_', ' ').title(), str(value)])
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                        st.dataframe(summary_df, use_container_width=True)
                    else:
                        st.info("No detailed summary information available.")
            
            # Store direct report paths
            if 'direct_reports' in sas_results:
                st.session_state['direct_reports'] = sas_results['direct_reports']
            
            # Create NCSS report for UI display (using captured data)
            report = create_ncss_report_from_sas_results(sas_results)
            if report is not None:
                st.session_state['ncss_report'] = report
                st.session_state['sas_results'] = sas_results
                st.session_state['analysis_completed'] = True
                # Store the output folder for reuse in exports
                if 'output_folder' in sas_results:
                    st.session_state['output_folder'] = sas_results['output_folder']
                st.rerun()
            else:
                st.error("Failed to create report from analysis results")
        else:
            st.error("❌ Analysis failed. Check the SAS log for details.")
            for key in ['ncss_report', 'sas_results', 'analysis_completed', 'direct_reports']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # =============================================================================
    # RESULTS DISPLAY SECTION
    # =============================================================================
    
    # Show results if analysis is completed
    if st.session_state.get('analysis_completed', False):
        st.success("Analysis completed successfully!")
        
        # =============================================================================
        # SAS LOG ANALYSIS SECTION (Error/Warning Display) - PROMINENT PLACEMENT
        # =============================================================================
        
        # Display SAS log analysis results prominently at the top
        sas_results = st.session_state.get('sas_results', {})
        if 'log_analysis' in sas_results and sas_results['log_analysis']:
            log_analysis = sas_results['log_analysis']
            
            # Create SAS Manager to get error summary
            sas_manager = create_sas_manager()
            error_summary = sas_manager.log_analyzer.get_error_summary(log_analysis)
            
            # Display errors prominently above the expander
            if log_analysis.has_errors:
                st.error("❌ **SAS Errors Detected:**")
                
            if log_analysis.has_warnings:
                    st.warning("⚠️ **SAS Warnings Detected:**")
            
            # Create dynamic expander title based on log status
            if log_analysis.has_errors:
                expander_title = "🔍 SAS Log Analysis - ERRORS"
            elif log_analysis.has_warnings:
                expander_title = "🔍 SAS Log Analysis - WARNINGS"
            else:
                expander_title = "🔍 SAS Log Analysis"
            
            # Display log analysis results prominently
            with st.expander(expander_title, expanded=False):
                st.markdown("**SAS Log Analysis Results:**")
                for error in log_analysis.error_messages:
                    st.error(f"  - {error}")
                
                for warning in log_analysis.warning_messages:
                    st.warning(f"  - {warning}")
                
                if not log_analysis.has_errors and not log_analysis.has_warnings:
                    st.success("✅ **No errors or warnings detected in SAS log**")
                
                # Show additional log information
                if log_analysis.notes:
                    st.info("**Additional Information:**")
                    for note in log_analysis.notes:
                        st.info(f"  - {note}")
                
                # Show execution time if available
                if log_analysis.execution_time:
                    st.metric("SAS Execution Time", f"{log_analysis.execution_time:.2f} seconds")
                
                # Show convergence status if available
                if log_analysis.convergence_status:
                    st.info(f"**Convergence Status:** {log_analysis.convergence_status}")
                
                # Add button to show full SAS log
                if 'sas_log' in sas_results and sas_results['sas_log']:
                    if st.button("📋 Show Full SAS Log", key="show_full_log_session"):
                        st.code(sas_results['sas_log'], language='text')
        
        # =============================
        # KEY RESULTS SECTION
        # =============================
        sas_results = st.session_state.get('sas_results', {})
        model_state = sas_results.get('model_state', {})
        key_results = model_state.get('key_results', {})
        if key_results:
            st.header("⭐ Key Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{key_results.get('R2', 'N/A'):.4f}" if key_results.get('R2') is not None else "N/A")
                st.metric("Adj R²", f"{key_results.get('Adj_R2', 'N/A'):.4f}" if key_results.get('Adj_R2') is not None else "N/A")
            with col2:
                st.metric("Treatment F", f"{key_results.get('Treatment_F', 'N/A'):.4f}" if key_results.get('Treatment_F') is not None else "N/A")
                st.metric("Treatment P", f"{key_results.get('Treatment_P', 'N/A'):.6g}" if key_results.get('Treatment_P') is not None else "N/A")
            with col3:
                sig = key_results.get('Treatment_Significant')
                st.metric("Significant?", "✅" if sig else ("❌" if sig is not None else "N/A"))
            # LSMeans table
            if 'LSMeans' in key_results and key_results['LSMeans']:
                st.write("**Least Squares Means (Treatment Groups):**")
                lsmeans_df = pd.DataFrame(key_results['LSMeans'])
                st.dataframe(lsmeans_df, use_container_width=True)

        # =============================
        # DIRECT SAS REPORTS (Professional Format)
        # =============================
        with st.expander("📄 Direct SAS Reports (Professional Format)", expanded=False):
            st.info("These are the original SAS-generated reports with native formatting and no translation errors.")
            direct_reports = st.session_state['direct_reports']
            col1, col2 = st.columns(2)
            with col1:
                if 'pdf' in direct_reports and os.path.exists(direct_reports['pdf']):
                    with open(direct_reports['pdf'], 'rb') as f:
                        st.download_button(
                            label="📄 Download SAS PDF Report",
                            data=f.read(),
                            file_name=f"sas_direct_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="sas_pdf_download"
                        )
                    st.write("**Professional PDF report** with native SAS formatting")
                else:
                    st.warning("PDF report not available")
            with col2:
                if 'rtf' in direct_reports and os.path.exists(direct_reports['rtf']):
                    with open(direct_reports['rtf'], 'rb') as f:
                        st.download_button(
                            label="📄 Download SAS RTF Report",
                            data=f.read(),
                            file_name=f"sas_direct_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rtf",
                            mime="application/rtf",
                            key="sas_rtf_download"
                        )
                    st.write("**RTF report** for Word compatibility")
                else:
                    st.warning("RTF report not available")
        
        # =============================================================================
        # NCSS UI REPORT (Interactive Display)
        # =============================================================================
        with st.expander("📊 NCSS-Style Report (UI Display)", expanded=False):
            st.info("This is the internal data captured for UI display - may have translation artifacts.")
            report = st.session_state['ncss_report']
            for section in report.sections:
                st.subheader(section.title)
                if section.text:
                    st.write(section.text)
                for table in section.tables:
                    st.write(f"**{table.title}**")
                    formatted_rows = format_ncss_table_rows(table.rows, table.columns)
                    df = pd.DataFrame([{k: str(v) for k, v in row.items()} for row in formatted_rows])
                    st.dataframe(df, use_container_width=True)
                for plot in section.plots:
                    st.write(f"**{plot.title}**")
                    st.image(plot.image_bytes, caption=plot.description, width=400)
                if section.notes:
                    st.info(f"**Note:** {section.notes}")
        
        # Debug: Show raw SAS results
        with st.expander("🔍 Debug: Raw SAS Results", expanded=False):
            st.header("🔍 Debug: Raw SAS Results")
            if 'sas_results' in st.session_state and st.session_state['sas_results'] is not None:
                sas_results = st.session_state['sas_results']
                for key, value in sas_results.items():
                    if key not in ['direct_reports', 'session_dir']:  # Keep sas_log for debugging
                        st.write(f"**{key}:**")
                        if key == 'sas_log':
                            # Display SAS log in a code block for better readability
                            st.code(value, language='text')
                        elif isinstance(value, pd.DataFrame) and not value.empty:
                            st.dataframe(value, use_container_width=True)
                        elif isinstance(value, list) and len(value) > 0:
                            st.write(f"List with {len(value)} items: {value[:5]}...")
                        else:
                            st.write(f"Value: {value}")
            else:
                st.write("No SAS results available for debugging")
        
        # =============================================================================
        # EXPORT OPTIONS SECTION
        # =============================================================================
        
        st.header("📥 Export Options")
        # Provide multiple export formats in organized columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Direct SAS Reports**")
            if 'direct_reports' in st.session_state and st.session_state['direct_reports']:
                direct_reports = st.session_state['direct_reports']
                if 'pdf' in direct_reports and os.path.exists(direct_reports['pdf']):
                    with open(direct_reports['pdf'], 'rb') as f:
                        st.download_button(
                            label="📄 SAS PDF Report",
                            data=f.read(),
                            file_name=f"sas_direct_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="direct_pdf_download"
                        )
                if 'rtf' in direct_reports and os.path.exists(direct_reports['rtf']):
                    with open(direct_reports['rtf'], 'rb') as f:
                        st.download_button(
                            label="📄 SAS RTF Report",
                            data=f.read(),
                            file_name=f"sas_direct_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rtf",
                            mime="application/rtf",
                            key="direct_rtf_download"
                        )
            else:
                st.write("No direct reports available")
        
        with col2:
            st.write("**NCSS-Style Reports**")
            if st.button("📄 Generate NCSS PDF", key="ncss_pdf_btn"):
                try:
                    report = st.session_state['ncss_report']
                    if report is None:
                        st.error("No report available. Please run the analysis first.")
                    else:
                        # Use existing output folder from session state
                        output_folder = st.session_state.get('output_folder')
                        if not output_folder:
                            st.error("No output folder found. Please run the analysis first.")
                            return
                        
                        pdf_file = os.path.join(output_folder, f"ncss_glm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                        
                        # Generate PDF and save to file
                        pdf_bytes = build_ncss_pdf(report)
                        with open(pdf_file, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        st.download_button(
                            label="💾 Download NCSS PDF",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_file),
                            mime="application/pdf",
                            key="ncss_pdf_download"
                        )
                        st.success(f"✅ NCSS PDF saved to: {output_folder}")
                except Exception as e:
                    st.error(f"Failed to generate NCSS PDF: {e}")
                    logger.error(f"NCSS PDF generation error: {e}")
        
        with col3:
            st.write("**Excel Export**")
            if st.button("📊 Export to Excel", key="excel_btn"):
                try:
                    report = st.session_state['ncss_report']
                    # Use existing output folder from session state
                    output_folder = st.session_state.get('output_folder')
                    if not output_folder:
                        st.error("No output folder found. Please run the analysis first.")
                        return
                    
                    excel_file = os.path.join(output_folder, f"ncss_glm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                    export_report_to_excel(report, excel_file)
                    with open(excel_file, 'rb') as f:
                        st.download_button(
                            label="💾 Download Excel Report",
                            data=f.read(),
                            file_name=os.path.basename(excel_file),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_download"
                        )
                    st.success(f"✅ Excel report saved to: {output_folder}")
                except Exception as e:
                    st.error(f"Failed to export Excel: {e}")
        
        with col4:
            st.write("**Reset**")
            if st.button("🔄 Reset Analysis", key="reset_btn"):
                # Clear session state
                for key in ['ncss_report', 'sas_results', 'analysis_completed', 'direct_reports', 'output_folder']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
   
    main() 