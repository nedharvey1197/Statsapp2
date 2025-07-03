"""
Simple GLM (One-Way ANOVA) Analysis with NCSS-Style Reporting (SAS-based)

This script demonstrates how to use the NCSS utilities for
simple group comparison (one-way ANOVA) using SAS PROC GLM.
"""

import streamlit as st
import pandas as pd
import saspy
import logging
import os
from datetime import datetime
import json
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import NCSS utilities
from utils.ncss_report_structures import (
    NCSSReport, NCSSSection, NCSSTable, SectionType
)
from utils.ncss_plot_utils import create_all_diagnostic_plots
from utils.ncss_report_builder import (
    build_ncss_pdf, format_ncss_table_rows
)
from utils.ncss_export_utils import export_report_to_excel
from utils.output_folder_utils import get_or_create_analysis_folder, create_analysis_output_folder


def load_example_data():
    """Load the simple GLM example data"""
    try:
        df = pd.read_csv("data/simple_example.csv")
        return df
    except Exception as e:
        logger.error(f"Failed to load example data: {e}")
        return None


def cleanup_sas_session(sas_session=None):
    """Clean up SAS session and datasets"""
    try:
        if sas_session is not None:
            # Clean up datasets
            for name in ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'residuals', 'normtests', 'testdata']:
                if sas_session.exist(name):
                    sas_session.sasdata(name).delete()
            sas_session.endsas()
            logger.info("SAS session cleaned up successfully")
            return True
    except Exception as e:
        logger.warning(f"Failed to clean up SAS session: {e}")
        return False


def setup_sas_connection():
    """Setup SAS connection using the same configuration as working versions"""
    try:
        sas = saspy.SASsession(cfgname='oda')
        logger.info("SAS connection established successfully")
        return sas
    except Exception as e:
        logger.error(f"Failed to connect to SAS: {e}")
        st.error(f"Failed to connect to SAS: {e}")
        return None


def run_simple_glm_analysis_with_manager(data_file):
    # Folder setup
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    daily_folder = datetime.now().strftime('%Y-%m-%d')
    daily_dir = os.path.join(logs_dir, daily_folder)
    os.makedirs(daily_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%H-%M-%S')
    session_dir = os.path.join(daily_dir, f'simple_glm_manager_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    output_folder = create_analysis_output_folder('simple_glm')

    # Sanitize ODS file paths
    pdf_filename = f'sas_direct_report_{timestamp}.pdf'
    rtf_filename = f'sas_direct_report_{timestamp}.rtf'
    pdf_report_path = os.path.join(session_dir, pdf_filename).replace('\\', '/')
    rtf_report_path = os.path.join(session_dir, rtf_filename).replace('\\', '/')

    # Test write permissions
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
        logger.info(f"Loaded data: {data.shape}")
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

    # SAS code with integrated proc report and debugging
    sas_code = f"""
    /* Set working directory */
    libname outdir '{session_dir}' replace;
    proc datasets lib=work; delete testdata; run;

    /* Debug: Verify directory access */
    proc printto log="{session_dir}/sas_debug.log"; run;
    data _null_;
        file "{session_dir}/test_sas.txt";
        put "SAS write test";
    run;
    proc printto; run;

    /* Enable ODS TRACE */
    ods trace on;
    ods pdf file="{pdf_report_path}" style=journal;
    ods rtf file="{rtf_report_path}" style=journal;
    options errors=1;
    ods graphics on;

    /* Debug: Verify work.testdata */
    proc contents data=work.testdata; run;

    /* Capture ODS tables */
    ods output 
        FitStatistics=work.fitstats 
        OverallANOVA=work.anova 
        LSMeans=work.lsmeans 
        Diff=work.diffs 
        ParameterEstimates=work.coeffs
        NObs=work.nobs
        ClassLevels=work.classlevels;

    /* Run PROC GLM */
    proc glm data=work.testdata plots=diagnostics;
        class Treatment;
        model TumorSize = Treatment / solution;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
        output out=work.residuals r=resid p=pred;
    run;

    /* Residual normality tests */
    proc univariate data=work.residuals normal;
        var resid;
        ods output TestsForNormality=work.normtests;
    run;

    /* Debug: Verify datasets for proc report */
    proc contents data=work.fitstats; run;
    proc contents data=work.diffs; run;

    /* PROC REPORT for fitstats */
    proc report data=work.fitstats;
        column Label Value;
        define Label / display "Parameter";
        define Value / display "Value";
        title "Fit Statistics";
    run;

    /* PROC REPORT for diffs */
    proc report data=work.diffs;
        column Treatment _Treatment_ Estimate StdErr tValue Probt;
        define Treatment / display "Treatment i";
        define _Treatment_ / display "Treatment j";
        define Estimate / display "Difference";
        define StdErr / display "Std Error";
        define tValue / display "t Value";
        define Probt / display "Pr > |t|";
        title "Pairwise Comparisons";
    run;

    /* Close ODS */
    ods pdf close;
    ods rtf close;
    ods trace off;
    ods graphics off;
    quit;
    """

    # Execute analysis
    logger.info("Running PROC GLM analysis with SAS Analysis Manager")
    analysis_result = run_simple_glm_analysis(
        data_file=data_file,
        sas_code=sas_code,
        session_dir=session_dir,
        output_folder=output_folder,
        data=data
    )

    # Check analysis result
    if not analysis_result.success:
        logger.error(f"Analysis failed: {analysis_result.error_message}")
        st.error(f"Analysis failed: {analysis_result.error_message}")
        return None

    # Verify ODS output
    if not os.path.exists(pdf_report_path):
        logger.error(f"PDF not generated: {pdf_report_path}")
        st.error(f"PDF not generated: {pdf_report_path}")
    if not os.path.exists(rtf_report_path):
        logger.error(f"RTF not generated: {rtf_report_path}")
        st.error(f"RTF not generated: {rtf_report_path}")

    # Convert results
    results = convert_to_legacy_format(analysis_result)
    results['data_info'] = data_info
    results['sas_code'] = sas_code
    results['direct_reports'] = {'pdf': pdf_report_path, 'rtf': rtf_report_path}

    logger.info("Analysis completed successfully with SAS Analysis Manager")
    return results


def create_ncss_report_from_sas_results(sas_results, title="Simple GLM (One-Way ANOVA)"):
    """Create NCSS report from SAS results"""
    logger.info("Starting NCSS report creation...")
    if not sas_results:
        logger.error("No SAS results available to create report")
        st.error("No SAS results available to create report")
        return None
    metadata = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'software_version': 'SAS 9.4',
        'analysis_type': 'One-Way ANOVA',
        'model': 'PROC GLM'
    }
    report = NCSSReport(title=title, metadata=metadata)

    # Run Summary section
    run_summary_section = NCSSSection(title="Run Summary", section_type=SectionType.RUN_SUMMARY)
    
    # Add model metadata
    model_state = sas_results.get('model_state', {})
    if model_state:
        meta_rows = [{"Parameter": k, "Value": v} for k, v in model_state.items()]
        meta_table = NCSSTable(
            title="Model Metadata",
            columns=["Parameter", "Value"],
            rows=meta_rows
        )
        run_summary_section.add_table(meta_table)
    
    # Add SAS-generated fit statistics (preserve natural structure)
    fitstats = sas_results.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        # Use SAS column names directly for NCSS compatibility
        fitstats_table = NCSSTable(
            title="Fit Statistics",
            columns=list(fitstats.columns),
            rows=fitstats.to_dict('records')
        )
        run_summary_section.add_table(fitstats_table)
    
    # Add class levels information
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


def extract_complete_metadata_simple_glm(datasets, data=None):
    """Extract complete metadata from PROC GLM results using consistent approach"""
    
    def get_fit_statistics():
        """Extract fit statistics from FitStatistics table"""
        stats = {}
        if datasets and 'fitstats' in datasets and datasets['fitstats'] is not None:
            fitstats = datasets['fitstats']
            if not fitstats.empty:
                logger.info(f"Extracting fit statistics from PROC GLM fitstats table (shape: {fitstats.shape})")
                logger.info(f"  Columns: {list(fitstats.columns)}")
                # SAS GLM fitstats has different column names
                for _, row in fitstats.iterrows():
                    logger.info(f"  Row data: {row.to_dict()}")
                    # Check for R-Square
                    if 'RSquare' in fitstats.columns:
                        stats['R-Square'] = str(row.get('RSquare', ''))
                        logger.info(f"    Found R-Square: {stats['R-Square']}")
                    # Check for Root MSE
                    if 'RootMSE' in fitstats.columns:
                        stats['Root MSE'] = str(row.get('RootMSE', ''))
                        logger.info(f"    Found Root MSE: {stats['Root MSE']}")
                    # Check for CV
                    if 'CV' in fitstats.columns:
                        stats['CV'] = str(row.get('CV', ''))
                        logger.info(f"    Found CV: {stats['CV']}")
                    # Check for DepMean
                    if 'DepMean' in fitstats.columns:
                        stats['Dependent Mean'] = str(row.get('DepMean', ''))
                        logger.info(f"    Found Dependent Mean: {stats['Dependent Mean']}")
        logger.info(f"Final fit statistics extracted: {stats}")
        return stats
    
    def get_observation_count():
        """Extract observation count from NObs table"""
        if datasets and 'nobs' in datasets and datasets['nobs'] is not None:
            nobs_info = datasets['nobs']
            if not nobs_info.empty:
                for _, row in nobs_info.iterrows():
                    desc = row.get('Label', '')
                    value = row.get('N', '')
                    if 'Number of Observations' in desc or 'N' in desc:
                        return str(value)
        return str(len(data)) if data is not None else 'Not available'
    
    def get_group_count():
        """Extract group count from ClassLevels table"""
        if datasets and 'classlevels' in datasets and datasets['classlevels'] is not None:
            class_info = datasets['classlevels']
            if not class_info.empty:
                return str(len(class_info))
        return str(len(data['Treatment'].unique())) if data is not None and 'Treatment' in data else 'Not available'
    
    # Extract fit statistics
    fit_stats = get_fit_statistics()
    
    # Build complete model_state with all items
    model_state = {
        'Model/Method': 'PROC GLM (One-Way ANOVA)',
        'Dataset name': 'testdata',
        'Response variable': 'TumorSize',
        'Group variable': 'Treatment',
        'Number of Observations': get_observation_count(),
        'Number of Groups': get_group_count(),
        'Convergence': 'Normal (GLM)',
        'AIC': 'Not computed (GLM)',
        'Log-Likelihood': 'Not computed (GLM)',
    }
    
    # Add fit statistics
    model_state.update(fit_stats)
    
    return model_state


def log_sas_dataset_summary(dataset_name: str, dataset: pd.DataFrame, log_file_path: str, indent: str = "  ") -> None:
    """
    Log detailed information about a SAS dataset
    
    Args:
        dataset_name: Name of the dataset
        dataset: The pandas DataFrame
        log_file_path: Path to the log file
        indent: Indentation string
    """
    with open(log_file_path, 'a') as f:
        f.write(f"{indent}{dataset_name}:\n")
        if dataset is not None and not dataset.empty:
            f.write(f"{indent}  Shape: {dataset.shape}\n")
            f.write(f"{indent}  Columns: {list(dataset.columns)}\n")
            f.write(f"{indent}  Sample Data:\n")
            for i, row in dataset.head(3).iterrows():
                f.write(f"{indent}    Row {i}: {row.to_dict()}\n")
        else:
            f.write(f"{indent}  Status: NOT FOUND or EMPTY\n")
        f.write("\n")


def log_extraction_mapping(expected_datasets: List[str], actual_datasets: Dict[str, pd.DataFrame], 
                          log_file_path: str) -> None:
    """
    Log the mapping between expected and actual datasets
    
    Args:
        expected_datasets: List of expected dataset names
        actual_datasets: Dictionary of actual datasets
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write("EXTRACTION MAPPING SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        found_count = 0
        for name in expected_datasets:
            dataset = actual_datasets.get(name)
            if dataset is not None and not dataset.empty:
                f.write(f"✓ {name}: {dataset.shape}\n")
                found_count += 1
            else:
                f.write(f"✗ {name}: NOT FOUND\n")
        
        f.write(f"\nSummary: {found_count}/{len(expected_datasets)} datasets found\n\n")


def log_metadata_extraction(metadata_name: str, extraction_result: Any, source_info: str, 
                           log_file_path: str) -> None:
    """
    Log metadata extraction results
    
    Args:
        metadata_name: Name of the metadata item
        extraction_result: The extracted value
        source_info: Information about the source
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write(f"Metadata Extraction: {metadata_name}\n")
        f.write(f"  Source: {source_info}\n")
        f.write(f"  Result: {extraction_result}\n")
        f.write(f"  Status: {'✓ SUCCESS' if extraction_result != 'Not available' else '✗ FAILED'}\n\n")


def log_model_state_construction(model_state: Dict[str, Any], log_file_path: str) -> None:
    """
    Log the final model state construction
    
    Args:
        model_state: The final model state dictionary
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write("FINAL MODEL STATE:\n")
        f.write("-" * 20 + "\n")
        
        available_count = 0
        missing_count = 0
        
        for key, value in model_state.items():
            if value != "Not available":
                f.write(f"✓ {key}: {value}\n")
                available_count += 1
            else:
                f.write(f"✗ {key}: Not available\n")
                missing_count += 1
        
        f.write(f"\nModel State Summary: {available_count} available, {missing_count} missing\n\n")


def create_comprehensive_log_simple_glm(session_dir: str, data_info: Dict[str, Any], 
                                       datasets: Dict[str, pd.DataFrame], 
                                       model_state: Dict[str, Any], 
                                       sas_code: str = None) -> str:
    """
    Create a comprehensive log file with all SAS data and extraction information for Simple GLM
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        datasets: Dictionary of SAS datasets
        model_state: Final model state
        sas_code: The SAS code that was executed
    
    Returns:
        Path to the comprehensive log file
    """
    log_file_path = os.path.join(session_dir, 'comprehensive_analysis_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write("COMPREHENSIVE ANALYSIS LOG - SIMPLE GLM\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Directory: {session_dir}\n")
        f.write(f"Model: PROC GLM (One-Way ANOVA)\n")
        f.write(f"Analysis Type: Direct ODS Output with Internal Data Capture\n\n")
        
        # Log the SAS code that was executed
        if sas_code:
            f.write("=== SAS CODE EXECUTED ===\n")
            f.write(sas_code)
            f.write("\n\n")
        
        # Input data information
        f.write("=== INPUT DATA ===\n")
        f.write(f"Shape: {data_info.get('shape', 'Unknown')}\n")
        f.write(f"Columns: {data_info.get('columns', [])}\n")
        f.write(f"Data Types: {data_info.get('dtypes', {})}\n\n")
        
        # PROC GLM datasets
        f.write("=== PROC GLM DATASETS ===\n")
        expected_datasets = [
            'fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs',
            'residuals', 'normtests', 'nobs', 'classlevels'
        ]
        
        for dataset_name in expected_datasets:
            dataset = datasets.get(dataset_name)
            log_sas_dataset_summary(dataset_name, dataset, log_file_path)
        
        log_extraction_mapping(expected_datasets, datasets, log_file_path)
        
        # Model state
        log_model_state_construction(model_state, log_file_path)
        
        # Summary statistics
        f.write(f"\n=== SUMMARY STATISTICS ===\n")
        f.write(f"Total PROC GLM datasets: {len(datasets)}\n")
        f.write(f"Total model state items: {len(model_state)}\n")
        
        # Machine-readable summary
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "session_dir": session_dir,
            "model": "PROC GLM (One-Way ANOVA)",
            "input_data": data_info,
            "proc_glm_datasets": {name: {"shape": df.shape if df is not None else None} 
                                for name, df in datasets.items()},
            "model_state": model_state
        }
        
        f.write(f"\nJSON Complete Summary: {json.dumps(summary, indent=2)}\n")
    
    return log_file_path


def create_data_flow_summary_simple_glm(session_dir: str, data_info: Dict[str, Any], 
                                       datasets: Dict[str, pd.DataFrame], 
                                       model_state: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of the data flow and extraction results for Simple GLM
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        datasets: Dictionary of SAS datasets
        model_state: Final model state
    
    Returns:
        Path to the summary report file
    """
    summary_file_path = os.path.join(session_dir, 'data_flow_summary.txt')
    
    with open(summary_file_path, 'w') as f:
        f.write("DATA FLOW SUMMARY REPORT - SIMPLE GLM\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: PROC GLM (One-Way ANOVA)\n\n")
        
        # Input Data Summary
        f.write("1. INPUT DATA\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {data_info.get('shape', 'Unknown')}\n")
        f.write(f"Columns: {', '.join(data_info.get('columns', []))}\n")
        f.write(f"Data Types:\n")
        for col, dtype in data_info.get('dtypes', {}).items():
            f.write(f"  {col}: {dtype}\n")
        f.write("\n")
        
        # PROC GLM Results Summary
        f.write("2. PROC GLM DATASETS\n")
        f.write("-" * 22 + "\n")
        expected_datasets = ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs',
                           'residuals', 'normtests', 'nobs', 'classlevels']
        
        found_count = 0
        for name in expected_datasets:
            dataset = datasets.get(name)
            if dataset is not None and not dataset.empty:
                f.write(f"✓ {name}: {dataset.shape}\n")
                found_count += 1
            else:
                f.write(f"✗ {name}: NOT FOUND\n")
        
        f.write(f"\nPROC GLM Summary: {found_count}/{len(expected_datasets)} datasets found\n\n")
        
        # Key Statistics Summary
        f.write("3. KEY STATISTICS EXTRACTION\n")
        f.write("-" * 30 + "\n")
        
        # Fit Statistics
        fitstats = datasets.get('fitstats')
        if fitstats is not None and not fitstats.empty:
            f.write("Fit Statistics (from PROC GLM):\n")
            for _, row in fitstats.iterrows():
                if 'RSquare' in fitstats.columns:
                    f.write(f"  R-Square: {row.get('RSquare', 'N/A')}\n")
                if 'RootMSE' in fitstats.columns:
                    f.write(f"  Root MSE: {row.get('RootMSE', 'N/A')}\n")
                if 'CV' in fitstats.columns:
                    f.write(f"  CV: {row.get('CV', 'N/A')}\n")
        else:
            f.write("Fit Statistics: NOT AVAILABLE\n")
        
        f.write("\n")
        
        # ANOVA Summary
        anova = datasets.get('anova')
        if anova is not None and not anova.empty:
            f.write("ANOVA Results:\n")
            for _, row in anova.iterrows():
                source = row.get('Source', 'N/A')
                f_value = row.get('FValue', 'N/A')
                prob_f = row.get('ProbF', 'N/A')
                f.write(f"  {source}: F={f_value}, p={prob_f}\n")
        else:
            f.write("ANOVA Results: NOT AVAILABLE\n")
        
        f.write("\n")
        
        # Model State Summary
        f.write("4. FINAL MODEL STATE\n")
        f.write("-" * 20 + "\n")
        available_count = 0
        missing_count = 0
        
        for key, value in model_state.items():
            if value != "Not available":
                f.write(f"✓ {key}: {value}\n")
                available_count += 1
            else:
                f.write(f"✗ {key}: Not available\n")
                missing_count += 1
        
        f.write(f"\nModel State Summary: {available_count} available, {missing_count} missing\n\n")
        
        # Data Flow Diagram
        f.write("5. DATA FLOW DIAGRAM\n")
        f.write("-" * 20 + "\n")
        f.write("Input CSV → SAS WORK → ODS Tables → Python DataFrames → Model State → NCSS Report\n")
        f.write("     ↓           ↓           ↓              ↓              ↓            ↓\n")
        f.write(f"  {data_info.get('shape', 'Unknown')}    testdata     {len(datasets)} tables    {len(model_state)} items    UI/Excel\n")
        f.write("\n")
        
        # Issues and Recommendations
        f.write("6. POTENTIAL ISSUES & RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Check for missing datasets
        missing_datasets = [name for name in expected_datasets 
                          if datasets.get(name) is None or datasets[name].empty]
        if missing_datasets:
            f.write(f"Missing datasets: {', '.join(missing_datasets)}\n")
            f.write("Recommendation: Check SAS log for errors in PROC GLM\n\n")
        else:
            f.write("All expected datasets found ✓\n\n")
        
        # Check for missing metadata
        missing_metadata = [key for key, value in model_state.items() 
                          if value == "Not available"]
        if missing_metadata:
            f.write(f"Missing metadata: {', '.join(missing_metadata)}\n")
            f.write("Recommendation: Review metadata extraction functions\n\n")
        else:
            f.write("All metadata extracted successfully ✓\n\n")
    
    return summary_file_path


def create_machine_readable_log_simple_glm(session_dir: str, data_info: Dict[str, Any], 
                                          datasets: Dict[str, pd.DataFrame], 
                                          model_state: Dict[str, Any]) -> str:
    """
    Create a machine-readable JSON log for Simple GLM analysis
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        datasets: Dictionary of SAS datasets
        model_state: Final model state
    
    Returns:
        Path to the JSON log file
    """
    json_log_path = os.path.join(session_dir, 'machine_readable_log.json')
    
    def prepare_dataset_summary(datasets_dict):
        """Prepare dataset summary for JSON serialization"""
        summary = {}
        for name, dataset in datasets_dict.items():
            if dataset is not None and not dataset.empty:
                summary[name] = {
                    "shape": list(dataset.shape),
                    "columns": list(dataset.columns),
                    "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
                    "sample_data": dataset.head(3).to_dict('records'),
                    "row_count": len(dataset),
                    "column_count": len(dataset.columns)
                }
            else:
                summary[name] = {
                    "status": "missing_or_empty",
                    "shape": None,
                    "columns": None,
                    "dtypes": None,
                    "sample_data": None,
                    "row_count": 0,
                    "column_count": 0
                }
        return summary
    
    # Create comprehensive JSON log
    log_data = {
        "analysis_info": {
            "timestamp": datetime.now().isoformat(),
            "session_directory": session_dir,
            "analysis_type": "simple_glm_one_way_anova",
            "model": "PROC GLM",
            "software": "SAS 9.4 + Python"
        },
        "input_data": {
            "shape": data_info.get('shape'),
            "columns": data_info.get('columns'),
            "dtypes": data_info.get('dtypes'),
            "row_count": data_info.get('shape', (0, 0))[0] if data_info.get('shape') else 0,
            "column_count": data_info.get('shape', (0, 0))[1] if data_info.get('shape') else 0
        },
        "sas_analysis": {
            "proc_glm_datasets": prepare_dataset_summary(datasets),
            "expected_datasets": ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'residuals', 'normtests', 'nobs', 'classlevels']
        },
        "extraction_results": {
            "model_state": model_state,
            "available_metadata": {k: v for k, v in model_state.items() if v != "Not available"},
            "missing_metadata": {k: v for k, v in model_state.items() if v == "Not available"},
            "metadata_count": {
                "total": len(model_state),
                "available": len([v for v in model_state.values() if v != "Not available"]),
                "missing": len([v for v in model_state.values() if v == "Not available"])
            }
        },
        "data_quality_metrics": {
            "proc_glm_success_rate": len([ds for ds in datasets.values() if ds is not None and not ds.empty]) / 9,
            "metadata_success_rate": len([v for v in model_state.values() if v != "Not available"]) / len(model_state),
            "total_datasets_found": len([ds for ds in datasets.values() if ds is not None and not ds.empty])
        },
        "key_statistics": {
            "fit_statistics": {},
            "anova_results": {},
            "model_info": {}
        }
    }
    
    # Extract key statistics for easy access
    fitstats = datasets.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        for _, row in fitstats.iterrows():
            if 'RSquare' in fitstats.columns:
                log_data["key_statistics"]["fit_statistics"]["R_Square"] = str(row.get('RSquare', ''))
            if 'RootMSE' in fitstats.columns:
                log_data["key_statistics"]["fit_statistics"]["Root_MSE"] = str(row.get('RootMSE', ''))
            if 'CV' in fitstats.columns:
                log_data["key_statistics"]["fit_statistics"]["CV"] = str(row.get('CV', ''))
    
    anova = datasets.get('anova')
    if anova is not None and not anova.empty:
        log_data["key_statistics"]["anova_results"] = anova.to_dict('records')
    
    # Write JSON log
    with open(json_log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    return json_log_path


def main():
    st.set_page_config(page_title="Simple GLM (One-Way ANOVA) - Direct ODS Approach", layout="wide")
    
    # Initialize session state properly
    if 'ncss_report' not in st.session_state:
        st.session_state['ncss_report'] = None
    if 'sas_results' not in st.session_state:
        st.session_state['sas_results'] = None
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'direct_reports' not in st.session_state:
        st.session_state['direct_reports'] = None
    
    # Clean up SAS session
    cleanup_sas_session(sas if 'sas' in globals() else None)
    
    st.title("Simple GLM (One-Way ANOVA) - Direct ODS Approach")
    st.write("This app demonstrates direct ODS output for professional reporting with traceable internal data capture.")
    st.info("🆕 **New Approach**: Generates professional SAS reports directly while capturing data for UI display")
    
    # Comparison section
    with st.expander("🔍 **Approach Comparison**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**❌ Original Approach (Translation)**")
            st.write("- Manual ODS table extraction")
            st.write("- Python variable translation")
            st.write("- Potential data corruption")
            st.write("- Formatting loss")
            st.write("- Translation errors")
        
        with col2:
            st.write("**✅ Direct ODS Approach**")
            st.write("- Native SAS report generation")
            st.write("- Direct PDF/RTF output")
            st.write("- Traceable data capture")
            st.write("- Professional formatting")
            st.write("- No translation errors")
    
    st.subheader("📊 Data Selection")
    data_option = st.radio(
        "Choose your data source:",
        ["Use Example Data", "Upload My Own Data"],
        index=0
    )
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
    st.subheader("📈 Analysis Results")
    
    # Debug: Show session state info
    st.write(f"Debug - analysis_completed: {st.session_state.get('analysis_completed', 'Not set')}")
    st.write(f"Debug - ncss_report exists: {'ncss_report' in st.session_state}")
    
    # Show the Run Analysis button
    if st.button("🚀 Run SAS Analysis (Direct ODS)", type="primary"):
        sas = None
        try:
            with st.spinner("Running SAS analysis with direct ODS output..."):
                cleanup_sas_session(sas if 'sas' in locals() else None)
                sas = setup_sas_connection()
                if sas is None:
                    st.error("SAS connection failed. Please check your SAS configuration.")
                    st.stop()
                try:
                    sas_results = run_simple_glm_analysis_direct(sas, "temp_data.csv")
                    if sas_results is None:
                        st.error("❌ Analysis returned None - check SAS connection and data")
                        return
                except Exception as e:
                    st.error(f"❌ Analysis failed with exception: {str(e)}")
                    logger.error(f"Analysis exception: {e}")
                    return
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error: {e}")
            return
        finally:
            # Ensure SAS session is cleaned up
            if sas is not None:
                try:
                    cleanup_sas_session(sas)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up SAS session in main: {cleanup_error}")
        
        if sas_results:
            st.success("✅ Analysis completed successfully!")
            
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
    
    # Show results if analysis is completed
    if st.session_state.get('analysis_completed', False):
        st.success("Analysis completed successfully!")
        
        # Display direct ODS reports section
        if 'direct_reports' in st.session_state and st.session_state['direct_reports']:
            st.header("📄 Direct SAS Reports (Professional Format)")
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
        
        # Display the NCSS report (for UI display)
        if 'ncss_report' in st.session_state and st.session_state['ncss_report'] is not None:
            st.header("📊 NCSS-Style Report (UI Display)")
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
        if st.checkbox("🔍 Show Debug Information"):
            st.header("🔍 Debug: Raw SAS Results")
            if 'sas_results' in st.session_state and st.session_state['sas_results'] is not None:
                sas_results = st.session_state['sas_results']
                for key, value in sas_results.items():
                    if key not in ['sas_log', 'direct_reports', 'session_dir']:  # Skip the log to avoid cluttering
                        st.write(f"**{key}:**")
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            st.dataframe(value, use_container_width=True)
                        elif isinstance(value, list) and len(value) > 0:
                            st.write(f"List with {len(value)} items: {value[:5]}...")
                        else:
                            st.write(f"Value: {value}")
            else:
                st.write("No SAS results available for debugging")
        
        st.header("📥 Export Options")
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
                # Clean up SAS session when user resets
                try:
                    cleanup_sas_session(sas if 'sas' in locals() else None)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up SAS session on reset: {cleanup_error}")
                
                # Clear session state
                for key in ['ncss_report', 'sas_results', 'analysis_completed', 'direct_reports', 'output_folder']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main() 