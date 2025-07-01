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
    export_report_to_excel
)


def cleanup_sas_session(sas_session=None):
    """Clean up SAS session and datasets strategically"""
    try:
        if sas_session is not None:
            # Clean up GLMM datasets first
            glmm_dataset_names = [
                'glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
                'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms'
            ]
            for name in glmm_dataset_names:
                if sas_session.exist(name):
                    sas_session.sasdata(name).delete()
                    logger.debug(f"Cleaned up GLMM dataset: {name}")
            
            # End SAS session
            sas_session.end()
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


def run_repeated_measures_analysis(sas, data_file):
    """Run repeated measures analysis using SAS with proper logging"""
    
    # Create logs directory
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_dir = os.path.join(logs_dir, f'repeated_ncss_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    log_file_path = os.path.join(session_dir, 'sas_log.txt')
    
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
        
        # Save initial log
        with open(log_file_path, 'w') as f:
            f.write(f"SAS Analysis Log - NCSS Style\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"Data Shape: {data.shape}\n")
            f.write(f"Variables: {list(data.columns)}\n\n")
        
        # SAS code for repeated measures analysis with comprehensive ODS output
        sas_code = """
        /* Set ODS to capture all diagnostic information */
        ods listing close;
        ods html5 (id=saspy_internal) file=_tomods1 options(bitmap_mode='inline') device=svg style=HTMLBlue;
        ods graphics on / outputfmt=png;
        
        proc mixed data=work.repeated_data;
            class Treatment Week Dog;
            model TumorSize = Treatment|Week / solution ddfm=kr outp=resid;
            repeated Week / subject=Dog type=AR(1);
            lsmeans Treatment*Week / diff cl adjust=bon;
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
                OptimizationInfo=optimizationinfo
                Dimensions=dimensions
                Residuals=residuals
            ;
        run;
        quit;
        
        /* Reopen listing */
        ods listing;
        """
        
        logger.info("Running PROC MIXED analysis")
        result = sas.submit(sas_code)
        
        # Save SAS log
        with open(log_file_path, 'a') as f:
            f.write("SAS Log:\n")
            f.write("-" * 30 + "\n")
            f.write(result['LOG'])
        
        # Extract results
        datasets = {}
        dataset_names = [
            'fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
            'solution', 'convergence', 'modelinfo', 'iterhistory',
            'optimizationinfo', 'dimensions', 'residuals'
        ]
        
        for name in dataset_names:
            try:
                if sas.exist(name):
                    datasets[name] = sas.sasdata2dataframe(name)
                    logger.info(f"Extracted {name}: {datasets[name].shape if datasets[name] is not None else 'None'}")
                else:
                    datasets[name] = None
                    logger.warning(f"Dataset {name} not found")
            except Exception as e:
                logger.error(f"Failed to extract {name}: {e}")
                datasets[name] = None
        
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
            'model_state': model_state  # Add model_state to results
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
            title="Model Metadata",
            columns=["Parameter", "Value"],
            rows=meta_rows
        )
        run_summary_section.add_table(meta_table)
    # Add fitstats table
    fitstats = sas_results.get('fitstats')
    if fitstats is not None and not fitstats.empty:
        summary_rows = [(row.get('Descr', ''), str(row.get('Value', ''))) for _, row in fitstats.iterrows()]
        summary_table = NCSSTable(
            title="Run Summary",
            columns=["Parameter", "Value"],
            rows=[{"Parameter": k, "Value": v} for k, v in summary_rows]
        )
        run_summary_section.add_table(summary_table)
    report.add_section(run_summary_section)
    
    # 2. Report Definitions section (NCSS standard)
    definitions_section = NCSSSection(title="Report Definitions", section_type=SectionType.NOTES)
    definitions_text = """
    Likelihood Type: The likelihood equation that was solved.
    Fixed Model: The model entered as the fixed component of the mixed model.
    Random Model: The model entered as the random component of the mixed model.
    Repeated Pattern: The repeated component structure in the mixed model.
    Solution Type: The method used for finding the (restricted) maximum likelihood solution.
    Fisher Iterations: The number of iterations in the Fisher Scoring portion of the maximization.
    Newton Iterations: The number of iterations in the Newton-Raphson portion of the maximization.
    Retries: The number of times that the variance/covariance parameters could be reset during each iteration.
    Lambda: The parameter used in the Newton-Raphson process to specify the amount of change in parameters between iterations.
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
            ;
        run;
        quit;
        """
        
        logger.info("Running PROC GLMM for metadata extraction")
        glmm_result = sas.submit(glmm_code)
        
        # Extract GLMM results
        glmm_datasets = {}
        glmm_dataset_names = [
            'glmm_fitstats', 'glmm_convergence', 'glmm_modelinfo', 
            'glmm_iterhistory', 'glmm_dimensions', 'glmm_covparms'
        ]
        
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
    """Extract complete metadata from both PROC MIXED and PROC GLMM results"""
    
    # Helper functions for extracting values
    def get_glmm_convergence_value(key):
        if glmm_datasets and 'glmm_convergence' in glmm_datasets and glmm_datasets['glmm_convergence'] is not None:
            convergence = glmm_datasets['glmm_convergence']
            if not convergence.empty:
                if key == 'Convergence':
                    status = convergence.iloc[0].get('Status', 0)
                    return 'Normal' if status == 0.0 else f'Status: {status}'
                elif key == 'Solution Type':
                    reason = convergence.iloc[0].get('Reason', '')
                    if 'Newton' in reason:
                        return 'Newton-Raphson'
                    elif 'Fisher' in reason:
                        return 'Fisher Scoring'
                    else:
                        return 'REML'
        return 'Missing from SAS output'
    
    def get_glmm_iterhistory_value(key):
        if glmm_datasets and 'glmm_iterhistory' in glmm_datasets and glmm_datasets['glmm_iterhistory'] is not None:
            iterhistory = glmm_datasets['glmm_iterhistory']
            if not iterhistory.empty:
                if key == 'Fisher Iterations':
                    # Count Fisher iterations
                    if 'IterationType' in iterhistory.columns:
                        fisher_count = len(iterhistory[iterhistory['IterationType'] == 'Fisher'])
                        return str(fisher_count) if fisher_count > 0 else '0'
                    elif 'Type' in iterhistory.columns:
                        fisher_count = len(iterhistory[iterhistory['Type'] == 'Fisher'])
                        return str(fisher_count) if fisher_count > 0 else '0'
                    else:
                        return str(len(iterhistory))
                elif key == 'Newton Iterations':
                    # Count Newton iterations
                    if 'IterationType' in iterhistory.columns:
                        newton_count = len(iterhistory[iterhistory['IterationType'] == 'Newton'])
                        return str(newton_count) if newton_count > 0 else '0'
                    elif 'Type' in iterhistory.columns:
                        newton_count = len(iterhistory[iterhistory['Type'] == 'Newton'])
                        return str(newton_count) if newton_count > 0 else '0'
                    else:
                        return '0'
                elif key == 'Max Retries':
                    if 'Retries' in iterhistory.columns:
                        return str(iterhistory['Retries'].max())
                    elif 'Retry' in iterhistory.columns:
                        return str(iterhistory['Retry'].max())
                    else:
                        return '0'
        return 'Not available in this output'
    
    def get_glmm_optimizationinfo_value(key):
        # Try to get optimization info from available GLMM tables and PROC MIXED tables
        if key == 'Lambda':
            # First try GLMM IterHistory
            if glmm_datasets and 'glmm_iterhistory' in glmm_datasets and glmm_datasets['glmm_iterhistory'] is not None:
                iterhistory = glmm_datasets['glmm_iterhistory']
                if not iterhistory.empty and 'Lambda' in iterhistory.columns:
                    return str(iterhistory['Lambda'].iloc[-1])  # Get last iteration
            # Check GLMM ModelInfo for lambda
            if glmm_datasets and 'glmm_modelinfo' in glmm_datasets and glmm_datasets['glmm_modelinfo'] is not None:
                modelinfo = glmm_datasets['glmm_modelinfo']
                if not modelinfo.empty:
                    for _, row in modelinfo.iterrows():
                        desc = row.get('Descr', '')
                        value = row.get('Value', '')
                        if 'Lambda' in desc:
                            return str(value)
            # Try PROC MIXED tables as fallback
            if mixed_datasets and 'iterhistory' in mixed_datasets and mixed_datasets['iterhistory'] is not None:
                iterhistory = mixed_datasets['iterhistory']
                if not iterhistory.empty and 'Lambda' in iterhistory.columns:
                    return str(iterhistory['Lambda'].iloc[-1])
            return 'Not available in this output'
        elif key == 'Run Time (Seconds)':
            # Try GLMM ModelInfo first
            if glmm_datasets and 'glmm_modelinfo' in glmm_datasets and glmm_datasets['glmm_modelinfo'] is not None:
                modelinfo = glmm_datasets['glmm_modelinfo']
                if not modelinfo.empty:
                    for _, row in modelinfo.iterrows():
                        desc = row.get('Descr', '')
                        value = row.get('Value', '')
                        if 'Time' in desc or 'Elapsed' in desc:
                            return str(value)
            # Try PROC MIXED ModelInfo
            if mixed_datasets and 'modelinfo' in mixed_datasets and mixed_datasets['modelinfo'] is not None:
                modelinfo = mixed_datasets['modelinfo']
                if not modelinfo.empty:
                    for _, row in modelinfo.iterrows():
                        desc = row.get('Descr', '')
                        value = row.get('Value', '')
                        if 'Time' in desc or 'Elapsed' in desc:
                            return str(value)
            # Estimate from iteration count (use GLMM if available, otherwise PROC MIXED)
            if glmm_datasets and 'glmm_iterhistory' in glmm_datasets and glmm_datasets['glmm_iterhistory'] is not None:
                iterhistory = glmm_datasets['glmm_iterhistory']
                if not iterhistory.empty:
                    iterations = len(iterhistory)
                    estimated_time = iterations * 0.1
                    return f"{estimated_time:.2f} (estimated from GLMM)"
            elif mixed_datasets and 'iterhistory' in mixed_datasets and mixed_datasets['iterhistory'] is not None:
                iterhistory = mixed_datasets['iterhistory']
                if not iterhistory.empty:
                    iterations = len(iterhistory)
                    estimated_time = iterations * 0.1
                    return f"{estimated_time:.2f} (estimated from PROC MIXED)"
            return 'Not available in this output'
        return 'Not available in this output'
    
    def get_glmm_fitstats_value(key):
        if glmm_datasets and 'glmm_fitstats' in glmm_datasets and glmm_datasets['glmm_fitstats'] is not None:
            fitstats = glmm_datasets['glmm_fitstats']
            if not fitstats.empty:
                for _, row in fitstats.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    if key == 'Log-Likelihood' and 'Log Likelihood' in desc:
                        return value
                    elif key == '-2 Log-Likelihood' and '-2 Res Log Likelihood' in desc:
                        return value
                    elif key == 'AIC' and 'AIC' in desc and 'AICC' not in desc:
                        return value
        return 'Missing from SAS output'
    
    # Build complete model_state with data from both procedures
    model_state = {
        'Model/Method': 'PROC MIXED (Repeated Measures GLMM)',
        'Dataset name': 'repeated_data',
        'Response variable': 'TumorSize',
        'Subject variable': 'Dog',
        'Repeated variable': 'Week',
        'Fixed Model': 'Treatment|Week',
        'Random Model': 'Not applicable for this analysis',
        'Repeated Pattern': 'AR(1)',
        'Number of Rows': len(data) if data is not None else 'Not available',
        'Number of Subjects': len(data['Dog'].unique()) if data is not None and 'Dog' in data else 'Not available',
        'Solution Type': get_glmm_convergence_value('Solution Type'),
        'Fisher Iterations': get_glmm_iterhistory_value('Fisher Iterations'),
        'Newton Iterations': get_glmm_iterhistory_value('Newton Iterations'),
        'Max Retries': get_glmm_iterhistory_value('Max Retries'),
        'Lambda': get_glmm_optimizationinfo_value('Lambda'),
        'Log-Likelihood': get_glmm_fitstats_value('Log-Likelihood'),
        '-2 Log-Likelihood': get_glmm_fitstats_value('-2 Log-Likelihood'),
        'AIC': get_glmm_fitstats_value('AIC'),
        'Convergence': get_glmm_convergence_value('Convergence'),
        'Run Time (Seconds)': get_glmm_optimizationinfo_value('Run Time (Seconds)'),
    }
    
    return model_state


def main():
    st.set_page_config(page_title="Repeated Measures Analysis - NCSS Style", layout="wide")
    
    # Clean up any existing SAS sessions on Streamlit rerun
    cleanup_sas_session(sas if 'sas' in globals() else None)
    
    # Debug session state at startup
    logger.info(f"Session state at startup - analysis_completed: {st.session_state.get('analysis_completed', False)}")
    
    # Only reset session state if we're not in the middle of an analysis
    # Don't reset if analysis was just completed
    if 'analysis_completed' not in st.session_state or not st.session_state.get('analysis_completed', False):
        logger.info("Resetting session state (analysis not completed)")
        for key in ['ncss_report', 'sas_results']:
            if key in st.session_state:
                del st.session_state[key]
    else:
        logger.info("Preserving session state (analysis completed)")
    
    st.title("Repeated Measures Analysis with NCSS-Style Reporting")
    st.write("This app demonstrates the new NCSS utilities for consistent, durable report generation.")
    
    # Initialize session state
    if 'ncss_report' not in st.session_state:
        st.session_state['ncss_report'] = None
    if 'sas_results' not in st.session_state:
        st.session_state['sas_results'] = None
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    
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
    
    # --- Display NCSS report sections ---
    logger.info(f"Checking session state: ncss_report exists: {'ncss_report' in st.session_state}")
    logger.info(f"ncss_report value: {st.session_state.get('ncss_report') is not None}")
    logger.info(f"analysis_completed: {st.session_state.get('analysis_completed', False)}")
    
    if 'ncss_report' in st.session_state and st.session_state['ncss_report'] is not None:
        report = st.session_state['ncss_report']
        logger.info(f"Displaying report with {len(report.sections)} sections")
        # Display all NCSS sections in order
        for section in report.sections:
            st.header(section.title)
            if section.text:
                st.write(section.text)
            for table in section.tables:
                st.subheader(table.title)
                # Apply NCSS formatting to table rows
                formatted_rows = format_ncss_table_rows(table.rows, table.columns)
                # Convert all values to strings to avoid ArrowTypeError
                df = pd.DataFrame([{k: str(v) for k, v in row.items()} for row in formatted_rows])
                st.dataframe(df, use_container_width=True)
            for plot in section.plots:
                st.subheader(plot.title)
                # Display plot with 20% smaller size and not full width
                st.image(plot.image_bytes, caption=plot.description, width=400)
            if section.notes:
                st.info(f"**Note:** {section.notes}")
    else:
        st.info("No report available yet. Please run the analysis.")
    # --- End NCSS report display ---
    
    # Analysis section
    st.subheader("📈 Analysis Results")
    
    # Run analysis button
    logger.info(f"Analysis button check - analysis_completed: {st.session_state.get('analysis_completed', False)}")
    if st.button("🚀 Run SAS Analysis", type="primary") or st.session_state.get('analysis_completed', False):
        if not st.session_state.get('analysis_completed', False):
            with st.spinner("Running SAS analysis..."):
                # Clean up any existing SAS sessions before starting new analysis
                cleanup_sas_session(sas if 'sas' in locals() else None)
                
                # Setup SAS connection only when button is pressed
                sas = setup_sas_connection()
                
                if sas is None:
                    st.error("SAS connection failed. Please check your SAS configuration.")
                    st.stop()
                
                try:
                    sas_results = run_repeated_measures_analysis(sas, "temp_data.csv")
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
                    logger.info("Session state updated successfully")
                    
                    logger.info("Triggering UI refresh...")
                    st.rerun()  # Refresh to show results
                else:
                    logger.error("Failed to create NCSS report")
                    st.error("Failed to create report from analysis results")
            else:
                st.error("❌ Analysis failed. Check the SAS log for details.")
                # Clear any previous results
                for key in ['ncss_report', 'sas_results', 'analysis_completed']:
                    if key in st.session_state:
                        del st.session_state[key]
        else:
            # Analysis completed - results are displayed in the NCSS report sections above
            logger.info("Analysis already completed, displaying results...")
            st.success("Analysis completed successfully!")
            
            # Download options
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
                                file_name=f"repeated_measures_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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
                        excel_file = f"repeated_measures_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
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
                    cleanup_sas_session(sas if 'sas' in locals() else None)
                    
                    # Clear session state
                    for key in ['ncss_report', 'sas_results', 'analysis_completed']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()


if __name__ == "__main__":
    main() 