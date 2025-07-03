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
from typing import Dict, Any

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
            sas_session.end()
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


def run_simple_glm_analysis(sas, data_file):
    """Run simple GLM (one-way ANOVA) analysis using SAS"""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_dir = os.path.join(logs_dir, f'simple_glm_ncss_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    log_file_path = os.path.join(session_dir, 'sas_log.txt')

    try:
        data = pd.read_csv(data_file)
        logger.info(f"Loaded data: {data.shape}")
        
        # Prepare data info for comprehensive logging
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        
        upload_data = data.copy()
        if 'Treatment' in upload_data.columns:
            upload_data['Treatment'] = upload_data['Treatment'].astype(str)
        sas_df = sas.df2sd(upload_data, table='testdata')
        logger.info("Data transferred to SAS")

        with open(log_file_path, 'w') as f:
            f.write(f"SAS Analysis Log - Simple GLM NCSS Style\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"Data Shape: {data.shape}\n")
            f.write(f"Variables: {list(data.columns)}\n\n")

        sas_code = """
        /* Enable ODS TRACE to identify available tables */
        ods trace on;
        
        /* Comprehensive ODS OUTPUT for NCSS-style reporting */
        /* Note: PROC GLM table names may vary - we'll capture what's available */
        ods output 
            FitStatistics=fitstats 
            OverallANOVA=anova 
            LSMeans=lsmeans 
            Diff=diffs 
            ParameterEstimates=coeffs
            NObs=nobs
            ClassLevels=classlevels;
        
        /* Enable ODS Graphics for plots */
        ods graphics on;
        
        proc glm data=work.testdata plots=(means diagnostics);
            class Treatment;
            model TumorSize = Treatment / solution;
            lsmeans Treatment / stderr pdiff cl adjust=bon;
            output out=residuals r=resid p=pred;
        run;
        
        /* Residual normality tests */
        proc univariate data=residuals normal;
            var resid;
            ods output TestsForNormality=normtests;
        run;
        
        /* Print some key tables to verify they exist */
        proc print data=fitstats;
        run;
        proc print data=anova;
        run;
        proc print data=lsmeans;
        run;
        
        ods trace off;
        ods graphics off;
        quit;
        """
        logger.info("Running PROC GLM analysis")
        result = sas.submit(sas_code)
        with open(log_file_path, 'a') as f:
            f.write("SAS Log:\n" + result['LOG'])

        # Extract ODS tables with comprehensive logging and debugging
        datasets = {}
        dataset_names = ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'residuals', 'normtests', 'nobs', 'classlevels']
        logger.info("Starting SAS ODS table extraction...")
        
        # First, let's see what datasets are actually available
        try:
            all_datasets = sas.sasdata2dataframe('_all_')
            logger.info(f"Available datasets: {all_datasets}")
        except Exception as e:
            logger.warning(f"Could not get list of all datasets: {e}")
        
        for name in dataset_names:
            try:
                if sas.exist(name):
                    datasets[name] = sas.sasdata2dataframe(name)
                    logger.info(f"Extracted {name}: {datasets[name].shape if datasets[name] is not None else 'None'}")
                    if datasets[name] is not None and not datasets[name].empty:
                        logger.info(f"  Columns: {list(datasets[name].columns)}")
                        # Log first few rows for debugging
                        logger.info(f"  First few rows: {datasets[name].head().to_dict('records')}")
                    else:
                        logger.warning(f"Dataset {name} is empty or None")
                else:
                    datasets[name] = None
                    logger.warning(f"Dataset {name} not found in SAS")
            except Exception as e:
                logger.error(f"Failed to extract {name}: {e}")
                datasets[name] = None
        
        # Check for essential datasets
        essential_datasets = ['fitstats', 'anova', 'lsmeans']
        missing_essential = [name for name in essential_datasets if datasets.get(name) is None or datasets[name].empty]
        if missing_essential:
            logger.error(f"Missing essential datasets: {missing_essential}")
            st.error(f"Critical error: Missing essential datasets: {missing_essential}")
            # Log the SAS log for debugging
            logger.error(f"SAS Log: {result['LOG']}")
            return None

        # Extract residuals and predicted values as lists
        if datasets['residuals'] is not None and 'resid' in datasets['residuals'].columns:
            residuals = datasets['residuals']['resid'].tolist()
            # Get predicted values if available, otherwise use empty list
            if 'pred' in datasets['residuals'].columns:
                predicted = datasets['residuals']['pred'].tolist()
            else:
                predicted = []
        else:
            residuals = []
            predicted = []

        # Extract complete metadata using consistent approach
        model_state = extract_complete_metadata_simple_glm(datasets, data)
        
        # Create comprehensive log with all data and extraction information
        logger.info("Creating comprehensive analysis log...")
        comprehensive_log_path = create_comprehensive_log_simple_glm(session_dir, data_info, datasets, model_state)
        logger.info(f"Comprehensive log created: {comprehensive_log_path}")
        
        # Create data flow summary
        logger.info("Creating data flow summary...")
        summary_path = create_data_flow_summary_simple_glm(session_dir, data_info, datasets, model_state)
        logger.info(f"Data flow summary created: {summary_path}")
        
        # Create machine-readable JSON log
        logger.info("Creating machine-readable JSON log...")
        json_log_path = create_machine_readable_log_simple_glm(session_dir, data_info, datasets, model_state)
        logger.info(f"Machine-readable log created: {json_log_path}")

        results = {
            'anova': datasets.get('anova'),
            'lsmeans': datasets.get('lsmeans'),
            'diffs': datasets.get('diffs'),
            'coeffs': datasets.get('coeffs'),
            'fitstats': datasets.get('fitstats'),
            'residuals': residuals,
            'predicted': predicted,
            'normtests': datasets.get('normtests'),
            'nobs': datasets.get('nobs'),
            'classlevels': datasets.get('classlevels'),
            'sas_log': result['LOG'],
            'model_state': model_state
        }
        logger.info("Analysis completed successfully")
        return results
    except Exception as e:
        logger.error(f"SAS analysis failed: {e}")
        st.error(f"SAS analysis failed: {e}")
        # Clean up SAS session on error
        try:
            cleanup_sas_session(sas)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up SAS session on error: {cleanup_error}")
        return None
    finally:
        # Ensure SAS session is always cleaned up
        try:
            cleanup_sas_session(sas)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up SAS session in finally block: {cleanup_error}")


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


def create_comprehensive_log_simple_glm(session_dir, data_info, datasets, model_state):
    """Create a comprehensive log file with all SAS data and extraction information"""
    log_file_path = os.path.join(session_dir, 'comprehensive_analysis_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write("COMPREHENSIVE ANALYSIS LOG - Simple GLM\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Directory: {session_dir}\n\n")
        
        # Input data information
        f.write("=== INPUT DATA ===\n")
        f.write(f"Shape: {data_info.get('shape', 'Unknown')}\n")
        f.write(f"Columns: {data_info.get('columns', [])}\n")
        f.write(f"Data Types: {data_info.get('dtypes', {})}\n\n")
        
        # PROC GLM datasets
        f.write("=== PROC GLM DATASETS ===\n")
        expected_datasets = ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'residuals', 'normtests', 'modelinfo', 'nobs', 'classlevels']
        
        for dataset_name in expected_datasets:
            dataset = datasets.get(dataset_name)
            if dataset is not None and not dataset.empty:
                f.write(f"\n  === SAS Dataset: {dataset_name} ===\n")
                f.write(f"  Shape: {dataset.shape}\n")
                f.write(f"  Columns: {list(dataset.columns)}\n")
                f.write(f"  Sample Data (first 3 rows):\n")
                f.write(dataset.head(3).to_string())
                f.write(f"\n")
            else:
                f.write(f"\n  === SAS Dataset: {dataset_name} ===\n")
                f.write(f"  Status: NOT FOUND OR EMPTY\n")
        
        # Model state
        f.write(f"\n=== FINAL MODEL STATE ===\n")
        f.write(f"Total metadata items: {len(model_state)}\n")
        for key, value in model_state.items():
            f.write(f"  {key}: {value}\n")
    
    return log_file_path


def create_data_flow_summary_simple_glm(session_dir, data_info, datasets, model_state):
    """Create a human-readable summary of the data flow and extraction results"""
    summary_file_path = os.path.join(session_dir, 'data_flow_summary.txt')
    
    with open(summary_file_path, 'w') as f:
        f.write("DATA FLOW SUMMARY REPORT - Simple GLM\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Input Data Summary
        f.write("1. INPUT DATA\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {data_info.get('shape', 'Unknown')}\n")
        f.write(f"Columns: {', '.join(data_info.get('columns', []))}\n\n")
        
        # PROC GLM Results Summary
        f.write("2. PROC GLM DATASETS\n")
        f.write("-" * 25 + "\n")
        expected_datasets = ['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'residuals', 'normtests', 'modelinfo', 'nobs', 'classlevels']
        
        found_count = 0
        for name in expected_datasets:
            dataset = datasets.get(name)
            if dataset is not None and not dataset.empty:
                f.write(f"✓ {name}: {dataset.shape}\n")
                found_count += 1
            else:
                f.write(f"✗ {name}: NOT FOUND\n")
        
        f.write(f"\nPROC GLM Summary: {found_count}/{len(expected_datasets)} datasets found\n\n")
        
        # Model State Summary
        f.write("3. FINAL MODEL STATE\n")
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
        f.write("4. DATA FLOW DIAGRAM\n")
        f.write("-" * 20 + "\n")
        f.write("Input CSV → SAS WORK → ODS Tables → Python DataFrames → Model State → NCSS Report\n")
        f.write("     ↓           ↓           ↓              ↓              ↓            ↓\n")
        f.write(f"  {data_info.get('shape', 'Unknown')}    testdata      {len(datasets)} tables    {len(model_state)} items    UI/Excel\n")
    
    return summary_file_path


def create_machine_readable_log_simple_glm(session_dir, data_info, datasets, model_state):
    """Create a machine-readable JSON log for automated processing"""
    json_log_path = os.path.join(session_dir, 'machine_readable_log.json')
    
    # Prepare dataset summaries for JSON
    def prepare_dataset_summary(datasets_dict):
        summary = {}
        for name, dataset in datasets_dict.items():
            if dataset is not None and not dataset.empty:
                summary[name] = {
                    "shape": dataset.shape,
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
            "proc_glm_success_rate": len([ds for ds in datasets.values() if ds is not None and not ds.empty]) / 9,  # Updated count
            "metadata_success_rate": len([v for v in model_state.values() if v != "Not available"]) / len(model_state),
            "total_datasets_found": len([ds for ds in datasets.values() if ds is not None and not ds.empty])
        }
    }
    
    # Write JSON log
    with open(json_log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    return json_log_path


def main():
    st.set_page_config(page_title="Simple GLM (One-Way ANOVA) - NCSS Style", layout="wide")
    
    # Initialize session state properly
    if 'ncss_report' not in st.session_state:
        st.session_state['ncss_report'] = None
    if 'sas_results' not in st.session_state:
        st.session_state['sas_results'] = None
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    
    # Clean up SAS session
    cleanup_sas_session(sas if 'sas' in globals() else None)
    
    st.title("Simple GLM (One-Way ANOVA) with NCSS-Style Reporting")
    st.write("This app demonstrates the NCSS utilities for simple group comparison using SAS.")
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
    if st.button("🚀 Run SAS Analysis", type="primary"):
        sas = None
        try:
            with st.spinner("Running SAS analysis..."):
                cleanup_sas_session(sas if 'sas' in locals() else None)
                sas = setup_sas_connection()
                if sas is None:
                    st.error("SAS connection failed. Please check your SAS configuration.")
                    st.stop()
                try:
                    sas_results = run_simple_glm_analysis(sas, "temp_data.csv")
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
            report = create_ncss_report_from_sas_results(sas_results)
            if report is not None:
                st.session_state['ncss_report'] = report
                st.session_state['sas_results'] = sas_results
                st.session_state['analysis_completed'] = True
                st.rerun()
            else:
                st.error("Failed to create report from analysis results")
        else:
            st.error("❌ Analysis failed. Check the SAS log for details.")
            for key in ['ncss_report', 'sas_results', 'analysis_completed']:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Show results if analysis is completed
    if st.session_state.get('analysis_completed', False):
        st.success("Analysis completed successfully!")
        
        # Display the NCSS report
        if 'ncss_report' in st.session_state and st.session_state['ncss_report'] is not None:
            report = st.session_state['ncss_report']
            st.header("📊 NCSS-Style Report")
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
                    if key != 'sas_log':  # Skip the log to avoid cluttering
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
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Generate PDF Report", key="pdf_btn"):
                try:
                    report = st.session_state['ncss_report']
                    if report is None:
                        st.error("No report available. Please run the analysis first.")
                    else:
                        pdf_bytes = build_ncss_pdf(report)
                        st.download_button(
                            label="💾 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"simple_glm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="pdf_download"
                        )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                    logger.error(f"PDF generation error: {e}")
        with col2:
            if st.button("📊 Export to Excel", key="excel_btn"):
                try:
                    report = st.session_state['ncss_report']
                    excel_file = f"simple_glm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    export_report_to_excel(report, excel_file)
                    with open(excel_file, 'rb') as f:
                        st.download_button(
                            label="💾 Download Excel Report",
                            data=f.read(),
                            file_name=excel_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="excel_download"
                        )
                except Exception as e:
                    st.error(f"Failed to export Excel: {e}")
        with col3:
            if st.button("🔄 Reset Analysis", key="reset_btn"):
                # Clean up SAS session when user resets
                try:
                    cleanup_sas_session(sas if 'sas' in locals() else None)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up SAS session on reset: {cleanup_error}")
                
                # Clear session state
                for key in ['ncss_report', 'sas_results', 'analysis_completed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main() 