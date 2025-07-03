"""
Repeated Measures Analysis with NCSS-Style Reporting

This script demonstrates how to use the new NCSS utilities for
repeated measures analysis with consistent, durable data structures.
"""

import streamlit as st
import pandas as pd
import numpy as np
import saspy
import logging
import os
from datetime import datetime
import seaborn as sns
from scipy import stats
import io
import matplotlib.pyplot as plt
import tempfile
import json
from typing import Dict, List, Any, Optional
from utils.output_folder_utils import create_analysis_output_folder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_example_data():
    """Load the repeated measures example data"""
    try:
        df = pd.read_csv("data/repeated_example.csv")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

# Import the new NCSS utilities
from utils.ncss_report_structures import (
    NCSSReport, NCSSSection, NCSSTable, NCSSPlot, SectionType,
    create_model_summary_section, create_anova_section, create_estimates_section,
    create_diagnostics_section, create_plots_section,
    create_run_summary_section, create_key_results_section
)
from utils.ncss_plot_utils import create_all_diagnostic_plots
from utils.ncss_report_builder import (
    build_ncss_pdf, display_ncss_report_in_streamlit, create_ncss_report,
    add_sas_results_to_report, format_ncss_table_rows
)
from utils.ncss_export_utils import (
    export_report_to_excel, export_report_to_json
)
from utils.output_folder_utils import get_or_create_analysis_folder


def cleanup_sas_session(sas_session=None):
    """Clean up SAS session and datasets strategically"""
    try:
        if sas_session is not None:
            # Clean up GLMM datasets first
            glmm_dataset_names = [
                'glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
                'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms',
                'glmm_optinfo', 'glmm_nobs', 'glmm_parameterestimates'
            ]
            for name in glmm_dataset_names:
                if sas_session.exist(name):
                    sas_session.sasdata(name).delete()
                    logger.debug(f"Cleaned up GLMM dataset: {name}")
            
            # End SAS session
            sas_session.endsas()
            logger.info("SAS session cleaned up successfully")
            return True
    except Exception as e:
        logger.warning(f"Failed to clean up SAS session: {e}")
        return False

def setup_sas_connection():
    """Setup SAS connection using the same configuration as working versions"""
    try:
        # Use 'oda' configuration like the working versions
        sas = saspy.SASsession(cfgname='oda')
        logger.info("SAS connection established successfully")
        return sas
    except Exception as e:
        logger.error(f"Failed to connect to SAS: {e}")
        st.error(f"Failed to connect to SAS: {e}")
        return None


def run_repeated_measures_analysis_direct(sas, data_file):
    """Run repeated measures analysis using direct ODS output approach"""
    
    # Create daily folder structure for logs
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create daily folder (YYYY-MM-DD format)
    daily_folder = datetime.now().strftime('%Y-%m-%d')
    daily_dir = os.path.join(logs_dir, daily_folder)
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)
    
    # Create session-specific folder within daily folder
    timestamp = datetime.now().strftime('%H-%M-%S')
    session_dir = os.path.join(daily_dir, f'repeated_direct_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    log_file_path = os.path.join(session_dir, 'sas_log.txt')
    
    # Create analysis-specific output folder for exports
    output_folder = create_analysis_output_folder('repeated_measures')
    
    # Define output file paths - use absolute paths for SAS
    pdf_report_path = os.path.abspath(os.path.join(session_dir, 'sas_direct_report.pdf'))
    rtf_report_path = os.path.abspath(os.path.join(session_dir, 'sas_direct_report.rtf'))
    
    try:
        # Load and prepare data
        data = pd.read_csv(data_file)
        logger.info(f"Loaded data: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        
        # Prepare data for SAS (same as working version)
        upload_data = data.copy()
        if 'Treatment' in upload_data.columns:
            upload_data['Treatment'] = upload_data['Treatment'].astype(str)
        if 'Week' in upload_data.columns:
            upload_data['Week'] = upload_data['Week'].astype(int)
        if 'Dog' in upload_data.columns:
            upload_data['Dog'] = upload_data['Dog'].astype(str)
        
        # Transfer data to SAS
        sas_df = sas.df2sd(upload_data, table='repeated_data')
        logger.info("Data transferred to SAS")
        
        # Prepare data info for comprehensive logging
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        
        # Save initial log
        with open(log_file_path, 'w') as f:
            f.write(f"SAS Analysis Log - Repeated Measures Direct ODS Approach\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"Data Shape: {data.shape}\n")
            f.write(f"Variables: {list(data.columns)}\n\n")
        
        # Direct ODS approach: Generate professional reports AND capture datasets
        sas_code = f"""
        /* Enable ODS TRACE to identify available tables */
        ods trace on;
        
        /* Generate direct PDF report with professional formatting */
        ods pdf file="{pdf_report_path}" style=journal;
        
        /* Generate RTF report for Word compatibility */
        ods rtf file="{rtf_report_path}" style=journal;
        
        /* Enable ODS Graphics for plots */
        ods graphics on;
        
        /* Capture ODS tables for internal use (traceable fidelity) */
        ods output 
            FitStatistics=fitstats 
            LSMeans=lsmeans 
            Diffs=diffs 
            CovParms=covparms 
            Tests3=tests3
            SolutionF=solution
            ConvergenceStatus=convergence
            ModelInfo=modelinfo
            IterHistory=iterhistory
            Dimensions=dimensions
            NObs=nobs
            InfoCrit=infocrit;
        
        /* Run PROC MIXED with comprehensive output */
        proc mixed data=work.repeated_data method=reml itdetails;
            class Treatment Week Dog;
            model TumorSize = Treatment|Week / solution ddfm=kr outp=resid;
            repeated Week / subject=Dog type=AR(1);
            lsmeans Treatment*Week / diff cl adjust=bon;
        run;
        
        /* Generate summary report using PROC REPORT */
        proc report data=fitstats;
            where Description in ("Log Likelihood", "-2 Log Likelihood", "AIC (smaller is better)", "AICC (smaller is better)", "BIC (smaller is better)");
            column Description Value;
            define Description / display "Parameter";
            define Value / display "Value";
            title "Run Summary - Fit Statistics";
        run;
        
        proc report data=tests3;
            column Effect NumDF DenDF FValue ProbF;
            define Effect / display "Effect";
            define NumDF / display "Num DF";
            define DenDF / display "Den DF";
            define FValue / display "F Value";
            define ProbF / display "Pr > F";
            title "Type 3 Tests of Fixed Effects";
        run;
        
        proc report data=lsmeans;
            column Treatment Week Estimate StdErr DF tValue Probt;
            define Treatment / display "Treatment";
            define Week / display "Week";
            define Estimate / display "Estimate";
            define StdErr / display "Std Error";
            define DF / display "DF";
            define tValue / display "t Value";
            define Probt / display "Pr > |t|";
            title "Least Squares Means";
        run;
        
        proc report data=covparms;
            column CovParm Subject Estimate StdErr ZValue Probt;
            define CovParm / display "Covariance Parameter";
            define Subject / display "Subject";
            define Estimate / display "Estimate";
            define StdErr / display "Std Error";
            define ZValue / display "Z Value";
            define Probt / display "Pr Z";
            title "Covariance Parameter Estimates";
        run;
        
        /* Close ODS destinations */
        ods pdf close;
        ods rtf close;
        ods trace off;
        ods graphics off;
        quit;
        """
        
        logger.info("Running PROC MIXED analysis with direct ODS output")
        result = sas.submit(sas_code)
        
        # Save SAS log
        with open(log_file_path, 'a') as f:
            f.write("SAS Log:\n")
            f.write("-" * 30 + "\n")
            f.write(result['LOG'])
        
        # Extract results with comprehensive logging
        datasets = {}
        dataset_names = [
            'fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
            'solution', 'convergence', 'modelinfo', 'iterhistory',
            'dimensions', 'nobs', 'infocrit'
        ]
        
        logger.info("Starting SAS dataset extraction for internal use...")
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
        essential_datasets = ['fitstats', 'tests3', 'lsmeans']
        missing_essential = [name for name in essential_datasets if datasets.get(name) is None or datasets[name].empty]
        if missing_essential:
            logger.error(f"Missing essential datasets: {missing_essential}")
            st.error(f"Critical error: Missing essential datasets: {missing_essential}")
            # Log the SAS log for debugging
            logger.error(f"SAS Log: {result['LOG']}")
            return None
        
        # Check if direct ODS reports were generated
        direct_reports = {}
        if os.path.exists(pdf_report_path):
            direct_reports['pdf'] = pdf_report_path
            logger.info(f"Direct PDF report generated: {pdf_report_path}")
        else:
            logger.warning(f"Direct PDF report not found: {pdf_report_path}")
            
        if os.path.exists(rtf_report_path):
            direct_reports['rtf'] = rtf_report_path
            logger.info(f"Direct RTF report generated: {rtf_report_path}")
        else:
            logger.warning(f"Direct RTF report not found: {rtf_report_path}")
        
        # Extract residuals data (from 'resid' dataset)
        if sas.exist('resid'):
            resid_df = sas.sd2df('resid')
            logger.info(f"Extracted resid: {resid_df.shape}")
            datasets['residuals_data'] = resid_df
        else:
            datasets['residuals_data'] = None
            logger.warning("Dataset 'resid' not found in SAS session.")
        
        # --- Build model_state dictionary using both PROC MIXED and PROC GLMM ---
        
        # Run PROC GLMM to get missing metadata
        logger.info("Running PROC GLMM to extract missing metadata")
        glmm_datasets = run_glmm_metadata_analysis(sas, data_file)
        
        # Extract complete metadata from both procedures
        model_state = extract_complete_metadata(datasets, glmm_datasets, data)
        
        # Create comprehensive log with all data and extraction information
        logger.info("Creating comprehensive analysis log...")
        comprehensive_log_path = create_comprehensive_log(session_dir, data_info, datasets, glmm_datasets, model_state, sas_code)
        logger.info(f"Comprehensive log created: {comprehensive_log_path}")
        
        # Create data flow summary
        logger.info("Creating data flow summary...")
        summary_path = create_data_flow_summary(session_dir, data_info, datasets, glmm_datasets, model_state)
        logger.info(f"Data flow summary created: {summary_path}")
        
        # Create machine-readable JSON log
        logger.info("Creating machine-readable JSON log...")
        json_log_path = create_machine_readable_log(session_dir, data_info, datasets, glmm_datasets, model_state)
        logger.info(f"Machine-readable log created: {json_log_path}")
        
        # Add additional metadata to match reference report format
        if 'iterhistory' in datasets and datasets['iterhistory'] is not None:
            iter_df = datasets['iterhistory']
            if not iter_df.empty:
                # Get iteration counts
                total_iterations = len(iter_df)
                model_state['Fisher Iterations'] = f"{total_iterations} of a possible 20"
                model_state['Newton Iterations'] = f"{total_iterations} of a possible 100"
        
        # Add run time if available from GLMM
        if glmm_datasets and 'glmm_convergence' in glmm_datasets:
            conv_df = glmm_datasets['glmm_convergence']
            if conv_df is not None and not conv_df.empty:
                # Look for run time in convergence table
                for col in conv_df.columns:
                    if 'time' in col.lower() or 'run' in col.lower():
                        model_state['Run Time (Seconds)'] = str(conv_df[col].iloc[0])
                        break
        
        # --- End model_state ---
        # Map to expected names
        results = {
            'solution': datasets.get('solution'),
            'anova': datasets.get('tests3'),
            'lsmeans': datasets.get('lsmeans'),
            'diffs': datasets.get('diffs'),
            'covparms': datasets.get('covparms'),
            'fitstats': datasets.get('fitstats'),
            'residuals_data': datasets.get('residuals_data'),
            'sas_log': result['LOG'],
            'model_state': model_state,  # Add model_state to results
            'direct_reports': direct_reports,  # Paths to direct ODS reports
            'session_dir': session_dir,  # Session directory for file access
            'output_folder': output_folder  # Analysis-specific output folder for exports
        }
        
        logger.info("Analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"SAS analysis failed: {e}")
        st.error(f"SAS analysis failed: {e}")
        return None


def create_ncss_report_from_sas_results(sas_results, title="Repeated Measures Analysis"):
    """Create NCSS report from SAS results"""
    logger.info("Starting NCSS report creation...")
    logger.info(f"SAS results keys: {list(sas_results.keys()) if sas_results else 'None'}")
    
    if not sas_results:
        logger.error("No SAS results available to create report")
        st.error("No SAS results available to create report")
        return None
    
    # Create base report
    logger.info("Creating base NCSS report...")
    metadata = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'software_version': 'SAS 9.4',
        'analysis_type': 'Repeated Measures Mixed Model',
        'model': 'PROC MIXED with AR(1) covariance structure'
    }
    report = NCSSReport(title=title, metadata=metadata)
    logger.info(f"Base report created with title: {title}")
    
    # 1. Run Summary section (NCSS standard)
    run_summary_section = NCSSSection(title="Run Summary", section_type=SectionType.RUN_SUMMARY)
    # Add model_state as metadata table at the top
    model_state = sas_results.get('model_state', {})
    if model_state:
        meta_rows = [{"Parameter": k, "Value": v} for k, v in model_state.items()]
        meta_table = NCSSTable(
            title="Key Statistics and Model Metadata",
            columns=["Parameter", "Value"],
            rows=meta_rows
        )
        run_summary_section.add_table(meta_table)
    # Add fitstats table
    fitstats = sas_results.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        summary_rows = [(row.get('Descr', ''), str(row.get('Value', ''))) for _, row in fitstats.iterrows()]
        summary_table = NCSSTable(
            title="Fit Data Summary Table",
            columns=["Parameter", "Value"],
            rows=[{"Parameter": k, "Value": v} for k, v in summary_rows]
        )
        run_summary_section.add_table(summary_table)
    report.add_section(run_summary_section)
    
    # 2. Report Definitions section (NCSS standard)
    definitions_section = NCSSSection(title="Report Definitions", section_type=SectionType.NOTES)
    definitions_text = """
    Solution Type: The optimization technique used (e.g., Newton-Raphson, Quasi-Newton).
    Fisher Iterations: The number of iterations in the Fisher Scoring portion of the maximization. (Pending future improvements to parse SAS log for detailed iteration breakdown).
    Newton Iterations: The number of iterations in the Newton-Raphson portion of the maximization. (Pending future improvements to parse SAS log for detailed iteration breakdown).
    Max Retries: The number of times that the variance/covariance parameters could be reset during each iteration. (Pending future improvements to parse SAS log).
    Lambda: The parameter used in the Newton-Raphson process to specify the amount of change in parameters between iterations. (Pending future improvements to parse SAS log).
    Log-Likelihood: The log of the likelihood of the data given the variance/covariance parameter estimates.
    AIC: The Akaike Information Criterion for use in comparing model covariance structures.
    Convergence: 'Normal' indicates that convergence was reached before the limit.
    Run Time: The amount of time it took to reach convergence.
    """
    definitions_section.text = definitions_text
    report.add_section(definitions_section)
    
    # 3. Repeated Component Parameter Estimates (R Matrix) section (NCSS standard)
    r_matrix_section = NCSSSection(title="Repeated Component Parameter Estimates (R Matrix)", section_type=SectionType.DIAGNOSTICS)
    covparms = sas_results.get('covparms')
    if covparms is not None and not covparms.empty:
        r_matrix_table = NCSSTable(
            title="Variance Components",
            columns=list(covparms.columns),
            rows=covparms.to_dict('records')
        )
        r_matrix_section.add_table(r_matrix_table)
    report.add_section(r_matrix_section)
    
    # 4. ANOVA section (NCSS standard)
    anova_section = NCSSSection(title="Analysis of Variance", section_type=SectionType.ANOVA)
    anova = sas_results.get('anova')  # tests3 from SAS
    if anova is not None and not anova.empty:
        anova_table = NCSSTable(
            title="ANOVA Results",
            columns=list(anova.columns),
            rows=anova.to_dict('records')
        )
        anova_section.add_table(anova_table)
    report.add_section(anova_section)
    
    # 5. Individual Comparison Hypothesis Test Results section (NCSS standard)
    comparisons_section = NCSSSection(title="Individual Comparison Hypothesis Test Results", section_type=SectionType.ESTIMATES)
    diffs = sas_results.get('diffs')
    if diffs is not None and not diffs.empty:
        comparisons_table = NCSSTable(
            title="Pairwise Comparisons",
            columns=list(diffs.columns),
            rows=diffs.to_dict('records')
        )
        comparisons_section.add_table(comparisons_table)
    report.add_section(comparisons_section)
    
    # 6. Least Squares (Adjusted) Means section (NCSS standard)
    lsmeans_section = NCSSSection(title="Least Squares (Adjusted) Means", section_type=SectionType.ESTIMATES)
    lsmeans = sas_results.get('lsmeans')
    if lsmeans is not None and not lsmeans.empty:
        lsmeans_table = NCSSTable(
            title="Least Squares Means",
            columns=list(lsmeans.columns),
            rows=lsmeans.to_dict('records')
        )
        lsmeans_section.add_table(lsmeans_table)
    report.add_section(lsmeans_section)
    
    # 7. Parameter Estimates section (from solution table)
    estimates_section = NCSSSection(title="Parameter Estimates", section_type=SectionType.ESTIMATES)
    solution = sas_results.get('solution')
    if solution is not None and not solution.empty:
        estimates_table = NCSSTable(
            title="Parameter Estimates",
            columns=list(solution.columns),
            rows=solution.to_dict('records')
        )
        estimates_section.add_table(estimates_table)
    report.add_section(estimates_section)
    
    # 8. Plots sections (NCSS standard)
    plots_section = NCSSSection(title="Diagnostic Plots", section_type=SectionType.PLOTS)
    # Add diagnostic plots if residuals are available
    if 'residuals_data' in sas_results and sas_results['residuals_data'] is not None and not sas_results['residuals_data'].empty:
        residuals_data = sas_results['residuals_data']
        if 'Resid' in residuals_data.columns:
            residuals = residuals_data['Resid'].values
            try:
                original_data = pd.read_csv("temp_data.csv")
                if 'TumorSize' in original_data.columns and len(residuals) == len(original_data):
                    predicted = original_data['TumorSize'].values - residuals
                else:
                    predicted = np.zeros_like(residuals)
            except:
                predicted = np.zeros_like(residuals)
            plots = create_all_diagnostic_plots(residuals, predicted)
            for plot in plots:
                plots_section.add_plot(plot)
    report.add_section(plots_section)

    return report


def run_glmm_metadata_analysis(sas, data_file):
    """Run PROC GLMM to extract missing metadata for NCSS reporting"""
    try:
        # Load data if not already loaded
        data = pd.read_csv(data_file)
        
        # Prepare data for SAS (same as main analysis)
        upload_data = data.copy()
        if 'Treatment' in upload_data.columns:
            upload_data['Treatment'] = upload_data['Treatment'].astype(str)
        if 'Week' in upload_data.columns:
            upload_data['Week'] = upload_data['Week'].astype(int)
        if 'Dog' in upload_data.columns:
            upload_data['Dog'] = upload_data['Dog'].astype(str)
        
        # Transfer data to SAS if not already there
        if not sas.exist('repeated_data'):
            sas_df = sas.df2sd(upload_data, table='repeated_data')
            logger.info("Data transferred to SAS for GLMM analysis")
        
        # SAS code for PROC GLMM to extract metadata
        glmm_code = """
        /* PROC GLMM for metadata extraction */
        proc glimmix data=work.repeated_data;
            class Treatment Week Dog;
            model TumorSize = Treatment|Week / solution ddfm=kr;
            random Week / subject=Dog type=ar(1);
            ods output 
                FitStatistics=glmm_fitstats 
                ConvergenceStatus=glmm_convergence
                ModelInfo=glmm_modelinfo
                IterHistory=glmm_iterhistory
                Dimensions=glmm_dimensions
                CovParms=glmm_covparms
                OptInfo=glmm_optinfo
                NObs=glmm_nobs
                ParameterEstimates=glmm_parameterestimates
            ;
        run;
        quit;
        """
        
        logger.info("Running PROC GLMM for metadata extraction")
        glmm_result = sas.submit(glmm_code)
        
        # Extract GLMM results with comprehensive logging
        glmm_datasets = {}
        glmm_dataset_names = [
            'glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
            'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms',
            'glmm_optinfo', 'glmm_nobs', 'glmm_parameterestimates'
        ]
        
        logger.info("Starting GLMM dataset extraction...")
        for name in glmm_dataset_names:
            try:
                if sas.exist(name):
                    glmm_datasets[name] = sas.sasdata2dataframe(name)
                    logger.info(f"Extracted GLMM {name}: {glmm_datasets[name].shape if glmm_datasets[name] is not None else 'None'}")
                else:
                    glmm_datasets[name] = None
                    logger.warning(f"GLMM dataset {name} not found")
            except Exception as e:
                logger.error(f"Failed to extract GLMM {name}: {e}")
                glmm_datasets[name] = None
        
        return glmm_datasets
        
    except Exception as e:
        logger.error(f"GLMM metadata analysis failed: {e}")
        return None


def extract_complete_metadata(mixed_datasets, glmm_datasets, data=None):
    """Extract complete metadata from both PROC MIXED and PROC GLIMMIX results"""
    
    def get_run_time():
        """Extract run time from Dimensions table"""
        # Try PROC MIXED first
        if mixed_datasets and 'dimensions' in mixed_datasets and mixed_datasets['dimensions'] is not None:
            dims = mixed_datasets['dimensions']
            if not dims.empty and 'RealTime' in dims.columns:
                return str(dims['RealTime'].iloc[0])
        # Try PROC GLIMMIX
        if glmm_datasets and 'glmm_dimensions' in glmm_datasets and glmm_datasets['glmm_dimensions'] is not None:
            dims = glmm_datasets['glmm_dimensions']
            if not dims.empty and 'RealTime' in dims.columns:
                return str(dims['RealTime'].iloc[0])
        return 'Not available'
    
    def get_convergence_status():
        """Extract convergence status from ConvergenceStatus table"""
        # Try PROC MIXED first
        if mixed_datasets and 'convergence' in mixed_datasets and mixed_datasets['convergence'] is not None:
            conv = mixed_datasets['convergence']
            if not conv.empty and 'Status' in conv.columns:
                status = conv['Status'].iloc[0]
                return 'Normal' if status == 0.0 else f'Status: {status}'
        # Try PROC GLIMMIX
        if glmm_datasets and 'glmm_convergence' in glmm_datasets and glmm_datasets['glmm_convergence'] is not None:
            conv = glmm_datasets['glmm_convergence']
            if not conv.empty and 'Status' in conv.columns:
                status = conv['Status'].iloc[0]
                return 'Normal' if status == 0.0 else f'Status: {status}'
        return 'Not available'
    
    def get_solution_type():
        """Extract solution type from ModelInfo/OptInfo tables"""
        # Try PROC GLIMMIX OptInfo first (more detailed)
        if glmm_datasets and 'glmm_optinfo' in glmm_datasets and glmm_datasets['glmm_optinfo'] is not None:
            optinfo = glmm_datasets['glmm_optinfo']
            if not optinfo.empty:
                for _, row in optinfo.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    if 'Technique' in desc or 'Method' in desc:
                        return str(value)
        # Try PROC MIXED ModelInfo
        if mixed_datasets and 'modelinfo' in mixed_datasets and mixed_datasets['modelinfo'] is not None:
            modelinfo = mixed_datasets['modelinfo']
            if not modelinfo.empty:
                for _, row in modelinfo.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    if 'OptTech' in desc or 'Method' in desc:
                        return str(value)
        return 'REML'  # Default
    
    def get_fit_statistics():
        """Extract fit statistics from FitStatistics table"""
        stats = {}
        
        # Try PROC MIXED first
        if mixed_datasets and 'fitstats' in mixed_datasets and mixed_datasets['fitstats'] is not None:
            fitstats = mixed_datasets['fitstats']
            if not fitstats.empty:
                logger.info(f"Extracting fit statistics from PROC MIXED fitstats table (shape: {fitstats.shape})")
                for _, row in fitstats.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    logger.info(f"  Checking row: Descr='{desc}', Value='{value}'")
                    if '-2 Res Log Likelihood' in desc:
                        stats['-2 Log-Likelihood'] = str(value)
                        logger.info(f"    Found -2 Log-Likelihood: {value}")
                    elif 'Log Likelihood' in desc and '-2' not in desc:
                        stats['Log-Likelihood'] = str(value)
                        logger.info(f"    Found Log-Likelihood: {value}")
                    elif 'AIC (Smaller is Better)' in desc:
                        stats['AIC'] = str(value)
                        logger.info(f"    Found AIC: {value}")
        
        # Try PROC GLIMMIX
        if glmm_datasets and 'glmm_fitstats' in glmm_datasets and glmm_datasets['glmm_fitstats'] is not None:
            fitstats = glmm_datasets['glmm_fitstats']
            if not fitstats.empty:
                logger.info(f"Extracting fit statistics from PROC GLIMMIX glmm_fitstats table (shape: {fitstats.shape})")
                for _, row in fitstats.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    logger.info(f"  Checking row: Descr='{desc}', Value='{value}'")
                    if '-2 Res Log Likelihood' in desc:
                        stats['-2 Log-Likelihood'] = str(value)
                        logger.info(f"    Found -2 Log-Likelihood: {value}")
                    elif 'Log Likelihood' in desc and '-2' not in desc:
                        stats['Log-Likelihood'] = str(value)
                        logger.info(f"    Found Log-Likelihood: {value}")
                    elif 'AIC  (smaller is better)' in desc:
                        stats['AIC'] = str(value)
                        logger.info(f"    Found AIC: {value}")
        
        logger.info(f"Final fit statistics extracted: {stats}")
        return stats
    
    def get_iteration_count():
        """Get total iteration count from IterHistory"""
        # Try PROC MIXED first
        if mixed_datasets and 'iterhistory' in mixed_datasets and mixed_datasets['iterhistory'] is not None:
            iterhistory = mixed_datasets['iterhistory']
            if not iterhistory.empty:
                return str(len(iterhistory))
        # Try PROC GLIMMIX
        if glmm_datasets and 'glmm_iterhistory' in glmm_datasets and glmm_datasets['glmm_iterhistory'] is not None:
            iterhistory = glmm_datasets['glmm_iterhistory']
            if not iterhistory.empty:
                return str(len(iterhistory))
        return 'Not available'
    
    # Extract fit statistics
    fit_stats = get_fit_statistics()
    
    # Build complete model_state with all items (including unavailable ones)
    model_state = {
        'Model/Method': 'PROC MIXED (Repeated Measures Mixed Model)',
        'Dataset name': 'repeated_data',
        'Response variable': 'TumorSize',
        'Subject variable': 'Dog',
        'Repeated variable': 'Week',
        'Fixed Model': 'Treatment|Week',
        'Random Model': 'Not applicable for this analysis',
        'Repeated Pattern': 'AR(1)',
        'Number of Rows': len(data) if data is not None else 'Not available',
        'Number of Subjects': len(data['Dog'].unique()) if data is not None and 'Dog' in data else 'Not available',
        'Solution Type': get_solution_type(),
        'Fisher Iterations': 'Not available - pending future improvements to parse SAS log',
        'Newton Iterations': 'Not available - pending future improvements to parse SAS log',
        'Max Retries': 'Not available - pending future improvements to parse SAS log',
        'Lambda': 'Not available - pending future improvements to parse SAS log',
        'Log-Likelihood': fit_stats.get('Log-Likelihood', 'Not available'),
        '-2 Log-Likelihood': fit_stats.get('-2 Log-Likelihood', 'Not available'),
        'AIC': fit_stats.get('AIC', 'Not available'),
        'Convergence': get_convergence_status(),
        'Run Time (Seconds)': get_run_time(),
    }
    
    return model_state


def log_sas_dataset_summary(dataset_name: str, dataset: pd.DataFrame, log_file_path: str, indent: str = "  ") -> None:
    """
    Log comprehensive summary of a SAS dataset in both human and machine-readable formats
    
    Args:
        dataset_name: Name of the SAS dataset
        dataset: The pandas DataFrame
        log_file_path: Path to the log file
        indent: Indentation for formatting
    """
    with open(log_file_path, 'a') as f:
        f.write(f"\n{indent}=== SAS Dataset: {dataset_name} ===\n")
        
        if dataset is None:
            f.write(f"{indent}Status: NOT FOUND\n")
            return
        
        if dataset.empty:
            f.write(f"{indent}Status: EMPTY\n")
            return
        
        # Human-readable summary
        f.write(f"{indent}Shape: {dataset.shape}\n")
        f.write(f"{indent}Columns: {list(dataset.columns)}\n")
        f.write(f"{indent}Data Types:\n")
        for col in dataset.columns:
            f.write(f"{indent}  {col}: {dataset[col].dtype}\n")
        
        # Sample data (first 3 rows)
        f.write(f"{indent}Sample Data (first 3 rows):\n")
        f.write(dataset.head(3).to_string())
        f.write(f"\n")
        
        # Machine-readable summary (JSON)
        summary = {
            "dataset_name": dataset_name,
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            "sample_data": dataset.head(3).to_dict('records')
        }
        f.write(f"{indent}JSON Summary: {json.dumps(summary, indent=2)}\n")

def log_extraction_mapping(expected_datasets: List[str], actual_datasets: Dict[str, pd.DataFrame], 
                          log_file_path: str) -> None:
    """
    Log the mapping between expected and actual SAS datasets
    
    Args:
        expected_datasets: List of expected dataset names
        actual_datasets: Dictionary of actual datasets
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write(f"\n=== EXTRACTION MAPPING SUMMARY ===\n")
        
        # Create mapping table
        f.write(f"Expected vs Actual Dataset Status:\n")
        f.write(f"{'Expected':<20} {'Status':<15} {'Shape':<15} {'Notes':<30}\n")
        f.write(f"{'-'*20} {'-'*15} {'-'*15} {'-'*30}\n")
        
        for expected in expected_datasets:
            if expected in actual_datasets:
                dataset = actual_datasets[expected]
                if dataset is not None:
                    status = "FOUND"
                    shape = str(dataset.shape)
                    notes = "OK"
                else:
                    status = "NULL"
                    shape = "N/A"
                    notes = "Dataset exists but is None"
            else:
                status = "MISSING"
                shape = "N/A"
                notes = "Dataset not found in SAS"
            
            f.write(f"{expected:<20} {status:<15} {shape:<15} {notes:<30}\n")
        
        # Machine-readable summary
        mapping_summary = {
            "expected_datasets": expected_datasets,
            "actual_datasets": list(actual_datasets.keys()),
            "missing_datasets": [ds for ds in expected_datasets if ds not in actual_datasets],
            "null_datasets": [ds for ds, df in actual_datasets.items() if df is None],
            "found_datasets": [ds for ds, df in actual_datasets.items() if df is not None and not df.empty]
        }
        f.write(f"\nJSON Mapping Summary: {json.dumps(mapping_summary, indent=2)}\n")

def log_metadata_extraction(metadata_name: str, extraction_result: Any, source_info: str, 
                           log_file_path: str) -> None:
    """
    Log metadata extraction results
    
    Args:
        metadata_name: Name of the metadata item
        extraction_result: The extracted value
        source_info: Information about the source (e.g., "from fitstats table")
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write(f"\n  Metadata: {metadata_name}\n")
        f.write(f"    Source: {source_info}\n")
        f.write(f"    Value: {extraction_result}\n")
        f.write(f"    Type: {type(extraction_result).__name__}\n")

def log_model_state_construction(model_state: Dict[str, Any], log_file_path: str) -> None:
    """
    Log the final model state construction
    
    Args:
        model_state: The final model state dictionary
        log_file_path: Path to the log file
    """
    with open(log_file_path, 'a') as f:
        f.write(f"\n=== FINAL MODEL STATE ===\n")
        
        # Human-readable summary
        f.write(f"Total metadata items: {len(model_state)}\n")
        f.write(f"Available items: {len([v for v in model_state.values() if v != 'Not available'])}\n")
        f.write(f"Missing items: {len([v for v in model_state.values() if v == 'Not available'])}\n")
        
        f.write(f"\nMetadata Items:\n")
        for key, value in model_state.items():
            status = "✓" if value != "Not available" else "✗"
            f.write(f"  {status} {key}: {value}\n")
        
        # Machine-readable summary
        model_summary = {
            "total_items": len(model_state),
            "available_items": {k: v for k, v in model_state.items() if v != "Not available"},
            "missing_items": {k: v for k, v in model_state.items() if v == "Not available"}
        }
        f.write(f"\nJSON Model State Summary: {json.dumps(model_summary, indent=2)}\n")

def create_comprehensive_log(session_dir: str, data_info: Dict[str, Any], 
                           sas_datasets: Dict[str, pd.DataFrame], 
                           glmm_datasets: Dict[str, pd.DataFrame],
                           model_state: Dict[str, Any], 
                           sas_code: str = None) -> str:
    """
    Create a comprehensive log file with all SAS data and extraction information
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        sas_datasets: Dictionary of SAS datasets
        glmm_datasets: Dictionary of GLMM datasets
        model_state: Final model state
        sas_code: The SAS code that was executed
    
    Returns:
        Path to the comprehensive log file
    """
    log_file_path = os.path.join(session_dir, 'comprehensive_analysis_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write("COMPREHENSIVE ANALYSIS LOG - REPEATED MEASURES\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Directory: {session_dir}\n")
        f.write(f"Model: PROC MIXED (Repeated Measures)\n")
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
        
        # PROC MIXED datasets
        f.write("=== PROC MIXED DATASETS ===\n")
        expected_mixed_datasets = [
            'fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
            'solution', 'convergence', 'modelinfo', 'iterhistory',
            'dimensions', 'nobs', 'infocrit'
        ]
        
        for dataset_name in expected_mixed_datasets:
            dataset = sas_datasets.get(dataset_name)
            log_sas_dataset_summary(dataset_name, dataset, log_file_path)
        
        log_extraction_mapping(expected_mixed_datasets, sas_datasets, log_file_path)
        
        # PROC GLIMMIX datasets
        f.write("\n=== PROC GLIMMIX DATASETS ===\n")
        expected_glmm_datasets = [
            'glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
            'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms',
            'glmm_optinfo', 'glmm_nobs', 'glmm_parameterestimates'
        ]
        
        for dataset_name in expected_glmm_datasets:
            dataset = glmm_datasets.get(dataset_name)
            log_sas_dataset_summary(dataset_name, dataset, log_file_path)
        
        log_extraction_mapping(expected_glmm_datasets, glmm_datasets, log_file_path)
        
        # Model state
        log_model_state_construction(model_state, log_file_path)
        
        # Summary statistics
        f.write(f"\n=== SUMMARY STATISTICS ===\n")
        f.write(f"Total PROC MIXED datasets: {len(sas_datasets)}\n")
        f.write(f"Total PROC GLIMMIX datasets: {len(glmm_datasets)}\n")
        f.write(f"Total model state items: {len(model_state)}\n")
        
        # Machine-readable summary
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "session_dir": session_dir,
            "input_data": data_info,
            "proc_mixed_datasets": {name: {"shape": df.shape if df is not None else None} 
                                  for name, df in sas_datasets.items()},
            "proc_glmm_datasets": {name: {"shape": df.shape if df is not None else None} 
                                 for name, df in glmm_datasets.items()},
            "model_state": model_state
        }
        
        f.write(f"\nJSON Complete Summary: {json.dumps(summary, indent=2)}\n")
    
    return log_file_path

def create_data_flow_summary(session_dir: str, data_info: Dict[str, Any], 
                           sas_datasets: Dict[str, pd.DataFrame], 
                           glmm_datasets: Dict[str, pd.DataFrame],
                           model_state: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of the data flow and extraction results
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        sas_datasets: Dictionary of SAS datasets
        glmm_datasets: Dictionary of GLMM datasets
        model_state: Final model state
    
    Returns:
        Path to the summary report file
    """
    summary_file_path = os.path.join(session_dir, 'data_flow_summary.txt')
    
    with open(summary_file_path, 'w') as f:
        f.write("DATA FLOW SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Input Data Summary
        f.write("1. INPUT DATA\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {data_info.get('shape', 'Unknown')}\n")
        f.write(f"Columns: {', '.join(data_info.get('columns', []))}\n")
        f.write(f"Data Types:\n")
        for col, dtype in data_info.get('dtypes', {}).items():
            f.write(f"  {col}: {dtype}\n")
        f.write("\n")
        
        # PROC MIXED Results Summary
        f.write("2. PROC MIXED DATASETS\n")
        f.write("-" * 25 + "\n")
        expected_mixed = ['fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
                         'solution', 'convergence', 'modelinfo', 'iterhistory',
                         'dimensions', 'nobs', 'infocrit']
        
        found_count = 0
        for name in expected_mixed:
            dataset = sas_datasets.get(name)
            if dataset is not None and not dataset.empty:
                f.write(f"✓ {name}: {dataset.shape}\n")
                found_count += 1
            else:
                f.write(f"✗ {name}: NOT FOUND\n")
        
        f.write(f"\nPROC MIXED Summary: {found_count}/{len(expected_mixed)} datasets found\n\n")
        
        # PROC GLIMMIX Results Summary
        f.write("3. PROC GLIMMIX DATASETS\n")
        f.write("-" * 26 + "\n")
        expected_glmm = ['glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
                        'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms',
                        'glmm_optinfo', 'glmm_nobs', 'glmm_parameterestimates']
        
        found_count = 0
        for name in expected_glmm:
            dataset = glmm_datasets.get(name)
            if dataset is not None and not dataset.empty:
                f.write(f"✓ {name}: {dataset.shape}\n")
                found_count += 1
            else:
                f.write(f"✗ {name}: NOT FOUND\n")
        
        f.write(f"\nPROC GLIMMIX Summary: {found_count}/{len(expected_glmm)} datasets found\n\n")
        
        # Key Statistics Summary
        f.write("4. KEY STATISTICS EXTRACTION\n")
        f.write("-" * 30 + "\n")
        
        # Fit Statistics
        fitstats = sas_datasets.get('fitstats')
        if fitstats is not None and not fitstats.empty:
            f.write("Fit Statistics (from PROC MIXED):\n")
            for _, row in fitstats.iterrows():
                desc = row.get('Descr', '')
                value = row.get('Value', '')
                f.write(f"  {desc}: {value}\n")
        else:
            f.write("Fit Statistics: NOT AVAILABLE\n")
        
        f.write("\n")
        
        # Model State Summary
        f.write("5. FINAL MODEL STATE\n")
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
        f.write("6. DATA FLOW DIAGRAM\n")
        f.write("-" * 20 + "\n")
        f.write("Input CSV → SAS WORK → ODS Tables → Python DataFrames → Model State → NCSS Report\n")
        f.write("     ↓           ↓           ↓              ↓              ↓            ↓\n")
        f.write(f"  {data_info.get('shape', 'Unknown')}    repeated_data  {len(sas_datasets)} tables    {len(model_state)} items    UI/Excel\n")
        f.write("\n")
        
        # Issues and Recommendations
        f.write("7. POTENTIAL ISSUES & RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Check for missing datasets
        missing_mixed = [name for name in expected_mixed if name not in sas_datasets or sas_datasets[name] is None]
        missing_glmm = [name for name in expected_glmm if name not in glmm_datasets or glmm_datasets[name] is None]
        
        if missing_mixed:
            f.write(f"Missing PROC MIXED datasets: {', '.join(missing_mixed)}\n")
            f.write("  → Check SAS log for warnings about these tables\n")
            f.write("  → Verify ODS OUTPUT statement includes these tables\n")
        
        if missing_glmm:
            f.write(f"Missing PROC GLIMMIX datasets: {', '.join(missing_glmm)}\n")
            f.write("  → Check SAS log for warnings about these tables\n")
            f.write("  → Verify ODS OUTPUT statement includes these tables\n")
        
        # Check for missing metadata
        missing_metadata = [key for key, value in model_state.items() if value == "Not available"]
        if missing_metadata:
            f.write(f"Missing metadata items: {', '.join(missing_metadata)}\n")
            f.write("  → These may not be available in SAS ODS tables\n")
            f.write("  → Consider parsing SAS log for these values\n")
        
        if not missing_mixed and not missing_glmm and not missing_metadata:
            f.write("✓ All expected data extracted successfully\n")
        
        f.write("\n")
        
        # Machine-readable summary
        f.write("8. MACHINE-READABLE SUMMARY (JSON)\n")
        f.write("-" * 40 + "\n")
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "input_data": data_info,
            "proc_mixed_status": {
                "expected": expected_mixed,
                "found": [name for name in expected_mixed if name in sas_datasets and sas_datasets[name] is not None],
                "missing": missing_mixed
            },
            "proc_glmm_status": {
                "expected": expected_glmm,
                "found": [name for name in expected_glmm if name in glmm_datasets and glmm_datasets[name] is not None],
                "missing": missing_glmm
            },
            "model_state_status": {
                "total_items": len(model_state),
                "available": {k: v for k, v in model_state.items() if v != "Not available"},
                "missing": missing_metadata
            }
        }
        f.write(json.dumps(summary, indent=2))
    
    return summary_file_path

def create_machine_readable_log(session_dir: str, data_info: Dict[str, Any], 
                               sas_datasets: Dict[str, pd.DataFrame], 
                               glmm_datasets: Dict[str, pd.DataFrame],
                               model_state: Dict[str, Any]) -> str:
    """
    Create a machine-readable JSON log for automated processing
    
    Args:
        session_dir: Directory for this session's logs
        data_info: Information about the input data
        sas_datasets: Dictionary of SAS datasets
        glmm_datasets: Dictionary of GLMM datasets
        model_state: Final model state
    
    Returns:
        Path to the JSON log file
    """
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
            "analysis_type": "repeated_measures_mixed_model",
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
            "proc_mixed_datasets": prepare_dataset_summary(sas_datasets),
            "proc_glmm_datasets": prepare_dataset_summary(glmm_datasets),
            "expected_datasets": {
                "proc_mixed": ['fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
                              'solution', 'convergence', 'modelinfo', 'iterhistory',
                              'dimensions', 'nobs', 'infocrit'],
                "proc_glmm": ['glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
                             'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms',
                             'glmm_optinfo', 'glmm_nobs', 'glmm_parameterestimates']
            }
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
            "proc_mixed_success_rate": len([ds for ds in sas_datasets.values() if ds is not None and not ds.empty]) / 12,
            "proc_glmm_success_rate": len([ds for ds in glmm_datasets.values() if ds is not None and not ds.empty]) / 9,
            "metadata_success_rate": len([v for v in model_state.values() if v != "Not available"]) / len(model_state),
            "total_datasets_found": len([ds for ds in list(sas_datasets.values()) + list(glmm_datasets.values()) if ds is not None and not ds.empty])
        },
        "key_statistics": {
            "fit_statistics": {},
            "convergence_info": {},
            "model_info": {}
        }
    }
    
    # Extract key statistics for easy access
    fitstats = sas_datasets.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        for _, row in fitstats.iterrows():
            desc = row.get('Descr', '')
            value = row.get('Value', '')
            log_data["key_statistics"]["fit_statistics"][desc] = value
    
    convergence = sas_datasets.get('convergence')
    if convergence is not None and not convergence.empty:
        log_data["key_statistics"]["convergence_info"] = convergence.to_dict('records')
    
    modelinfo = sas_datasets.get('modelinfo')
    if modelinfo is not None and not modelinfo.empty:
        log_data["key_statistics"]["model_info"] = modelinfo.to_dict('records')
    
    # Write JSON log
    with open(json_log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    return json_log_path

def main():
    st.set_page_config(page_title="Repeated Measures Analysis - Direct ODS Approach", layout="wide")
    
    # Clean up any existing SAS sessions on Streamlit rerun
    cleanup_sas_session(sas if 'sas' in globals() else None)
    
    # Debug session state at startup
    logger.info(f"Session state at startup - analysis_completed: {st.session_state.get('analysis_completed', False)}")
    
    # Only reset session state if we're not in the middle of an analysis
    # Don't reset if analysis was just completed
    if 'analysis_completed' not in st.session_state or not st.session_state.get('analysis_completed', False):
        logger.info("Resetting session state (analysis not completed)")
        for key in ['ncss_report', 'sas_results', 'direct_reports']:
            if key in st.session_state:
                del st.session_state[key]
    else:
        logger.info("Preserving session state (analysis completed)")
    
    st.title("Repeated Measures Analysis - Direct ODS Approach")
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
    
    # Initialize session state
    if 'ncss_report' not in st.session_state:
        st.session_state['ncss_report'] = None
    if 'sas_results' not in st.session_state:
        st.session_state['sas_results'] = None
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'direct_reports' not in st.session_state:
        st.session_state['direct_reports'] = None
    
    # Data selection
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
        # Use example data
        data = load_example_data()
        if data is not None:
            data.to_csv("temp_data.csv", index=False)
        else:
            st.error("Failed to load example data")
            st.stop()
    
    # Display data info
    st.subheader("📋 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", len(data))
    with col2:
        st.metric("Subjects (Dogs)", len(data['Dog'].unique()))
    with col3:
        st.metric("Treatments", len(data['Treatment'].unique()))
    st.dataframe(data, use_container_width=True)
    
    # Analysis section
    st.subheader("📈 Analysis Results")
    
    # Run analysis button
    logger.info(f"Analysis button check - analysis_completed: {st.session_state.get('analysis_completed', False)}")
    if st.button("🚀 Run SAS Analysis (Direct ODS)", type="primary") or st.session_state.get('analysis_completed', False):
        if not st.session_state.get('analysis_completed', False):
            with st.spinner("Running SAS analysis with direct ODS output..."):
                # Clean up any existing SAS sessions before starting new analysis
                cleanup_sas_session(sas if 'sas' in locals() else None)
                
                # Setup SAS connection only when button is pressed
                sas = setup_sas_connection()
                
                if sas is None:
                    st.error("SAS connection failed. Please check your SAS configuration.")
                    st.stop()
                
                try:
                    sas_results = run_repeated_measures_analysis_direct(sas, "temp_data.csv")
                    if sas_results is None:
                        st.error("❌ Analysis returned None - check SAS connection and data")
                        return
                except Exception as e:
                    st.error(f"❌ Analysis failed with exception: {str(e)}")
                    logger.error(f"Analysis exception: {e}")
                    return
            
            if sas_results:
                st.success("✅ Analysis completed successfully!")
                logger.info("Analysis completed, starting post-processing...")
                
                # Store direct report paths
                if 'direct_reports' in sas_results:
                    st.session_state['direct_reports'] = sas_results['direct_reports']
                
                # Create NCSS report
                logger.info("Creating NCSS report...")
                report = create_ncss_report_from_sas_results(sas_results)
                logger.info(f"NCSS report created: {report is not None}")
                
                if report is not None:
                    logger.info("Storing report in session state...")
                    # Store report in session state
                    st.session_state['ncss_report'] = report
                    st.session_state['sas_results'] = sas_results
                    st.session_state['analysis_completed'] = True
                    # Store the output folder for reuse in exports
                    if 'output_folder' in sas_results:
                        st.session_state['output_folder'] = sas_results['output_folder']
                    logger.info("Session state updated successfully")
                    
                    logger.info("Triggering UI refresh...")
                    st.rerun()  # Refresh to show results
                else:
                    logger.error("Failed to create NCSS report")
                    st.error("Failed to create report from analysis results")
            else:
                st.error("❌ Analysis failed. Check the SAS log for details.")
                # Clear any previous results
                for key in ['ncss_report', 'sas_results', 'analysis_completed', 'direct_reports']:
                    if key in st.session_state:
                        del st.session_state[key]
        else:
            # Analysis completed - display results
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
            
            # Download options
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
                            
                            pdf_file = os.path.join(output_folder, f"ncss_repeated_measures_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                            
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
                        
                        excel_file = os.path.join(output_folder, f"ncss_repeated_measures_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
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
                    cleanup_sas_session(sas if 'sas' in locals() else None)
                    
                    # Clear session state
                    for key in ['ncss_report', 'sas_results', 'analysis_completed', 'direct_reports', 'output_folder']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()


if __name__ == "__main__":
    main() 