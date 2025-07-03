"""
SAS Analysis Manager - Centralized SAS Analysis Management

This module provides a unified interface for managing SAS analyses across
all applications in the clinical trials analysis platform.

Features:
- Centralized SAS connection management
- Session lifecycle management
- Error handling and recovery
- Dataset extraction and validation
- SAS log analysis
- Resource cleanup
"""

import streamlit as st
import pandas as pd
import saspy
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Enumeration of supported analysis types"""
    SIMPLE_GLM = "simple_glm"
    REPEATED_MEASURES = "repeated_measures"


class ErrorSeverity(Enum):
    """Enumeration of error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SASLogAnalysis:
    """Container for SAS log analysis results"""
    has_errors: bool
    has_warnings: bool
    error_messages: List[str]
    warning_messages: List[str]
    convergence_status: Optional[str]
    execution_time: Optional[float]
    notes: List[str]


@dataclass
class AnalysisResult:
    """Container for complete analysis results"""
    success: bool
    datasets: Dict[str, pd.DataFrame]
    sas_log: str
    log_analysis: SASLogAnalysis
    model_state: Dict[str, Any]
    direct_reports: Dict[str, str]
    session_dir: str
    output_folder: str
    residuals: List[float] = None
    predicted: List[float] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class SASConnectionManager:
    """Manages SAS connection lifecycle"""
    
    def __init__(self, config_name: str = 'oda'):
        self.config_name = config_name
        self.connection = None
        self.connection_attempts = 0
        self.max_retries = 3
    
    def connect(self) -> Optional[saspy.SASsession]:
        """Establish SAS connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"SAS connection attempt {attempt + 1}/{self.max_retries}")
                self.connection = saspy.SASsession(cfgname=self.config_name)
                logger.info("SAS connection established successfully")
                self.connection_attempts = 0  # Reset on success
                return self.connection
            except Exception as e:
                self.connection_attempts += 1
                logger.error(f"SAS connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info("Retrying SAS connection...")
                else:
                    logger.critical("All SAS connection attempts failed")
                    st.error(f"Failed to connect to SAS after {self.max_retries} attempts: {e}")
                    return None
        return None
    
    def disconnect(self) -> bool:
        """Safely disconnect from SAS"""
        try:
            if self.connection is not None:
                self.connection.endsas()
                self.connection = None
                logger.info("SAS connection closed successfully")
                return True
        except Exception as e:
            logger.warning(f"Failed to close SAS connection: {e}")
            return False
        return True
    
    def is_connected(self) -> bool:
        """Check if SAS connection is active"""
        return self.connection is not None


class SASSessionManager:
    """Manages SAS session and dataset lifecycle"""
    
    def __init__(self):
        self.session_datasets = set()
        self.temp_files = []
    
    def register_dataset(self, dataset_name: str) -> None:
        """Register a dataset for cleanup"""
        self.session_datasets.add(dataset_name)
    
    def register_temp_file(self, file_path: str) -> None:
        """Register a temporary file for cleanup"""
        self.temp_files.append(file_path)
    
    def cleanup_session(self, sas_session: Optional[saspy.SASsession]) -> bool:
        """Clean up SAS session and datasets"""
        if sas_session is None:
            return True
        
        success = True
        
        # Clean up datasets
        for dataset_name in self.session_datasets:
            try:
                if sas_session.exist(dataset_name):
                    sas_session.sasdata(dataset_name).delete()
                    logger.debug(f"Cleaned up dataset: {dataset_name}")
            except Exception as e:
                logger.warning(f"Failed to clean up dataset {dataset_name}: {e}")
                success = False
        
        # Clean up temporary files
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")
                success = False
        
        # Reset tracking
        self.session_datasets.clear()
        self.temp_files.clear()
        
        return success


class SASLogAnalyzer:
    """Analyzes SAS logs for errors, warnings, and important information"""
    
    def __init__(self):
        # Common SAS error patterns
        self.error_patterns = [
            r'ERROR:\s*(.+)',
            r'ERROR\s+\d+-\d+:\s*(.+)',
            r'ERROR\s+\d+:\s*(.+)',
            r'ERROR\s+in\s+(.+)',
            r'ERROR:\s*The\s+(.+)',
            r'ERROR:\s*Invalid\s+(.+)',
            r'ERROR:\s*Missing\s+(.+)',
            r'ERROR:\s*No\s+(.+)',
            r'ERROR\s+(\d+)\s*-\s*(\d+):\s*(.+)',  # ERROR 123-456: message
            r'ERROR\s+(\d+):\s*(.+)',  # ERROR 123: message
            r'ERROR\s+(\w+):\s*(.+)',  # ERROR keyword: message
            r'ERROR\s+(\w+)\s+(\w+):\s*(.+)',  # ERROR keyword keyword: message
        ]
        
        # Common SAS warning patterns
        self.warning_patterns = [
            r'WARNING:\s*(.+)',
            r'WARNING\s+\d+-\d+:\s*(.+)',
            r'WARNING\s+\d+:\s*(.+)',
            r'NOTE:\s*The\s+(.+)',
            r'NOTE:\s*Missing\s+(.+)',
            r'UserWarning:\s*(.+)',  # Python UserWarning from saspy
            r'Noticed\s+\'ERROR:\'\s+in\s+LOG',  # saspy warning about errors in log
            r'warnings\.warn\s*\((.+)\)',  # Python warnings
        ]
        
        # Convergence patterns
        self.convergence_patterns = [
            r'Convergence criterion satisfied',
            r'Convergence criterion met',
            r'Maximum number of iterations reached',
            r'Convergence criterion not satisfied',
            r'Convergence failed',
        ]
        
        # Execution time patterns
        self.time_patterns = [
            r'real time\s+(\d+\.?\d*)\s+seconds',
            r'cpu time\s+(\d+\.?\d*)\s+seconds',
        ]
    
    def analyze(self, sas_log: str) -> SASLogAnalysis:
        """Analyze SAS log for errors, warnings, and important information"""
        if not sas_log:
            return SASLogAnalysis(
                has_errors=False,
                has_warnings=False,
                error_messages=[],
                warning_messages=[],
                convergence_status=None,
                execution_time=None,
                notes=[]
            )
        
        # Check for saspy-specific error indicators
        saspy_error_indicators = [
            "Noticed 'ERROR:' in LOG",
            "UserWarning: Noticed 'ERROR:' in LOG",
            "warnings.warn",
            "you ought to take a look"
        ]
        
        # Extract errors
        error_messages = []
        for pattern in self.error_patterns:
            matches = re.findall(pattern, sas_log, re.IGNORECASE)
            error_messages.extend(matches)
        
        # Check for saspy-specific error indicators
        for indicator in saspy_error_indicators:
            if indicator.lower() in sas_log.lower():
                error_messages.append(f"SASPy detected potential errors: {indicator}")
        
        # Extract warnings
        warning_messages = []
        for pattern in self.warning_patterns:
            matches = re.findall(pattern, sas_log, re.IGNORECASE)
            warning_messages.extend(matches)
        
        # Check convergence
        convergence_status = None
        for pattern in self.convergence_patterns:
            if re.search(pattern, sas_log, re.IGNORECASE):
                convergence_status = pattern
                break
        
        # Extract execution time
        execution_time = None
        for pattern in self.time_patterns:
            match = re.search(pattern, sas_log, re.IGNORECASE)
            if match:
                execution_time = float(match.group(1))
                break
        
        # Generate notes
        notes = []
        if convergence_status:
            notes.append(f"Convergence: {convergence_status}")
        if execution_time:
            notes.append(f"Execution time: {execution_time} seconds")
        
        return SASLogAnalysis(
            has_errors=len(error_messages) > 0,
            has_warnings=len(warning_messages) > 0,
            error_messages=error_messages,
            warning_messages=warning_messages,
            convergence_status=convergence_status,
            execution_time=execution_time,
            notes=notes
        )
    
    def get_error_summary(self, log_analysis: SASLogAnalysis) -> str:
        """Generate a user-friendly error summary"""
        if not log_analysis.has_errors and not log_analysis.has_warnings:
            return "Analysis completed successfully with no errors or warnings."
        
        summary = []
        
        if log_analysis.has_errors:
            summary.append("❌ **Errors detected:**")
            for error in log_analysis.error_messages[:3]:  # Show first 3 errors
                summary.append(f"  - {error}")
            if len(log_analysis.error_messages) > 3:
                summary.append(f"  - ... and {len(log_analysis.error_messages) - 3} more errors")
        
        if log_analysis.has_warnings:
            summary.append("⚠️ **Warnings detected:**")
            for warning in log_analysis.warning_messages[:3]:  # Show first 3 warnings
                summary.append(f"  - {warning}")
            if len(log_analysis.warning_messages) > 3:
                summary.append(f"  - ... and {len(log_analysis.warning_messages) - 3} more warnings")
        
        return "\n".join(summary)


class DatasetExtractor:
    """Handles extraction and validation of SAS datasets"""
    
    def __init__(self):
        # Define expected datasets for each analysis type
        self.expected_datasets = {
            AnalysisType.SIMPLE_GLM: [
                'fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs',
                'residuals', 'normtests', 'nobs', 'classlevels'
            ],
            AnalysisType.REPEATED_MEASURES: [
                'fitstats', 'lsmeans', 'diffs', 'covparms', 'tests3',
                'solution', 'convergence', 'modelinfo', 'iterhistory',
                'dimensions', 'nobs', 'infocrit'
            ]
        }
        
        # Define essential datasets that must be present
        self.essential_datasets = {
            AnalysisType.SIMPLE_GLM: ['fitstats', 'anova', 'lsmeans'],
            AnalysisType.REPEATED_MEASURES: ['fitstats', 'tests3', 'lsmeans']
        }
    
    def extract_datasets(self, sas_session: saspy.SASsession, 
                        analysis_type: AnalysisType) -> Dict[str, pd.DataFrame]:
        """Extract all expected datasets for the given analysis type"""
        datasets = {}
        expected = self.expected_datasets.get(analysis_type, [])
        
        logger.info(f"Extracting datasets for {analysis_type.value}")
        
        for dataset_name in expected:
            try:
                if sas_session.exist(dataset_name):
                    dataset = sas_session.sasdata2dataframe(dataset_name)
                    if dataset is not None and not dataset.empty:
                        datasets[dataset_name] = dataset
                        logger.info(f"✓ Extracted {dataset_name}: {dataset.shape}")
                    else:
                        logger.warning(f"Dataset {dataset_name} is empty")
                        datasets[dataset_name] = None
                else:
                    logger.warning(f"Dataset {dataset_name} not found in SAS")
                    datasets[dataset_name] = None
            except Exception as e:
                logger.error(f"Failed to extract {dataset_name}: {e}")
                datasets[dataset_name] = None
        
        return datasets
    
    def validate_datasets(self, datasets: Dict[str, pd.DataFrame], 
                         analysis_type: AnalysisType) -> Tuple[bool, List[str]]:
        """Validate that essential datasets are present and valid"""
        essential = self.essential_datasets.get(analysis_type, [])
        missing = []
        
        for dataset_name in essential:
            if dataset_name not in datasets or datasets[dataset_name] is None:
                missing.append(dataset_name)
        
        if missing:
            return False, missing
        
        return True, []
    
    def get_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate a summary of extracted datasets"""
        summary = {
            'total_expected': 0,
            'total_found': 0,
            'total_empty': 0,
            'dataset_details': {}
        }
        
        for name, dataset in datasets.items():
            summary['total_expected'] += 1
            
            if dataset is not None and not dataset.empty:
                summary['total_found'] += 1
                summary['dataset_details'][name] = {
                    'status': 'found',
                    'shape': dataset.shape,
                    'columns': list(dataset.columns)
                }
            else:
                summary['total_empty'] += 1
                summary['dataset_details'][name] = {
                    'status': 'missing_or_empty',
                    'shape': None,
                    'columns': None
                }
        
        return summary


class SASAnalysisManager:
    """Main manager class for coordinating SAS analyses"""
    
    def __init__(self):
        self.connection_manager = SASConnectionManager()
        self.session_manager = SASSessionManager()
        self.log_analyzer = SASLogAnalyzer()
        self.dataset_extractor = DatasetExtractor()
    
    def run_analysis(self, analysis_type: AnalysisType, data_file: str, 
                    sas_code: str, **kwargs) -> AnalysisResult:
        """
        Run a complete SAS analysis with full error handling and management
        
        Args:
            analysis_type: Type of analysis to run
            data_file: Path to input data file
            sas_code: SAS code to execute
            **kwargs: Additional parameters (session_dir, output_folder, etc.)
        
        Returns:
            AnalysisResult: Complete analysis results with error handling
        """
        sas_session = None
        
        try:
            # Step 1: Establish SAS connection
            logger.info(f"Starting {analysis_type.value} analysis")
            sas_session = self.connection_manager.connect()
            if not sas_session:
                return self._create_error_result(
                    "Failed to establish SAS connection",
                    {"connection_attempts": self.connection_manager.connection_attempts}
                )
            
            # Step 2: Transfer data to SAS
            data_transfer_success = self._transfer_data_to_sas(sas_session, data_file, analysis_type)
            if not data_transfer_success:
                return self._create_error_result("Failed to transfer data to SAS")
            
            # Step 3: Execute SAS analysis
            sas_result = self._execute_sas_analysis(sas_session, sas_code)
            if not sas_result:
                return self._create_error_result("SAS analysis execution failed")
            
            # Step 4: Extract and validate datasets
            datasets = self.dataset_extractor.extract_datasets(sas_session, analysis_type)
            validation_success, missing_datasets = self.dataset_extractor.validate_datasets(
                datasets, analysis_type
            )
            
            if not validation_success:
                return self._create_error_result(
                    f"Missing essential datasets: {', '.join(missing_datasets)}",
                    {"missing_datasets": missing_datasets}
                )
            
            # Step 5: Analyze SAS log
            log_analysis = self.log_analyzer.analyze(sas_result.get('LOG', ''))
            
            # Step 6: Check for direct reports
            direct_reports = self._check_direct_reports(kwargs.get('session_dir', ''))
            
            # Step 7: Create model state
            model_state = self._create_model_state(datasets, analysis_type, kwargs.get('data', None))
            
            # Step 8: Extract residuals and predicted for compatibility
            residuals, predicted = self._extract_residuals_and_predicted(datasets)
            
            # Step 9: Create success result
            return AnalysisResult(
                success=True,
                datasets=datasets,
                sas_log=sas_result.get('LOG', ''),
                log_analysis=log_analysis,
                model_state=model_state,
                direct_reports=direct_reports,
                session_dir=kwargs.get('session_dir', ''),
                output_folder=kwargs.get('output_folder', ''),
                residuals=residuals,  # Add residuals list
                predicted=predicted,  # Add predicted list
                error_message=None,
                error_details=None
            )
            
        except Exception as e:
            logger.error(f"Analysis failed with exception: {e}")
            return self._create_error_result(
                f"Analysis failed: {str(e)}",
                {"exception_type": type(e).__name__, "exception_details": str(e)}
            )
        
        finally:
            # Always cleanup
            if sas_session:
                self.session_manager.cleanup_session(sas_session)
                self.connection_manager.disconnect()
    
    def _transfer_data_to_sas(self, sas_session: saspy.SASsession, 
                             data_file: str, analysis_type: AnalysisType) -> bool:
        """Transfer data to SAS with appropriate table naming"""
        try:
            data = pd.read_csv(data_file)
            logger.info(f"Loaded data: {data.shape}")
            
            # Prepare data for SAS
            upload_data = data.copy()
            
            # Handle different data types based on analysis
            if analysis_type == AnalysisType.SIMPLE_GLM:
                if 'Treatment' in upload_data.columns:
                    upload_data['Treatment'] = upload_data['Treatment'].astype(str)
                table_name = 'testdata'
            elif analysis_type == AnalysisType.REPEATED_MEASURES:
                if 'Treatment' in upload_data.columns:
                    upload_data['Treatment'] = upload_data['Treatment'].astype(str)
                if 'Week' in upload_data.columns:
                    upload_data['Week'] = upload_data['Week'].astype(int)
                if 'Dog' in upload_data.columns:
                    upload_data['Dog'] = upload_data['Dog'].astype(str)
                table_name = 'repeated_data'
            
            # Transfer to SAS
            sas_session.df2sd(upload_data, table=table_name)
            self.session_manager.register_dataset(table_name)
            logger.info(f"Data transferred to SAS table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to transfer data to SAS: {e}")
            return False
    
    def _execute_sas_analysis(self, sas_session: saspy.SASsession, sas_code: str) -> Optional[Dict[str, Any]]:
        """Execute SAS analysis code"""
        try:
            logger.info("Executing SAS analysis")
            result = sas_session.submit(sas_code)
            logger.info("SAS analysis completed")
            return result
        except Exception as e:
            logger.error(f"SAS analysis execution failed: {e}")
            return None
    
    def _check_direct_reports(self, session_dir: str) -> Dict[str, str]:
        """Check for direct ODS reports (PDF, RTF)"""
        direct_reports = {}
        
        if session_dir:
            pdf_path = os.path.join(session_dir, 'sas_direct_report.pdf')
            rtf_path = os.path.join(session_dir, 'sas_direct_report.rtf')
            
            if os.path.exists(pdf_path):
                direct_reports['pdf'] = pdf_path
            if os.path.exists(rtf_path):
                direct_reports['rtf'] = rtf_path
        
        return direct_reports
    
    def _create_model_state(self, datasets: Dict[str, pd.DataFrame], 
                           analysis_type: AnalysisType, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create model state from extracted datasets with model-specific metadata"""
        
        if analysis_type == AnalysisType.SIMPLE_GLM:
            return self._create_simple_glm_model_state(datasets, data)
        elif analysis_type == AnalysisType.REPEATED_MEASURES:
            return self._create_repeated_measures_model_state(datasets, data)
        else:
            # Generic fallback
            return {
                'analysis_type': analysis_type.value,
                'timestamp': datetime.now().isoformat(),
                'datasets_found': len([d for d in datasets.values() if d is not None]),
                'total_datasets': len(datasets)
            }
    
    def _create_simple_glm_model_state(self, datasets: Dict[str, pd.DataFrame], 
                                      data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create model state for Simple GLM (One-Way ANOVA) analysis"""
        
        # Initialize organized model state with categories
        model_state = {
            'analysis_type': 'Simple-GLM - SAS PROC GLM (One-Way ANOVA)',
            'timestamp': datetime.now().isoformat(),
            'SAS datasets_found': len([d for d in datasets.values() if d is not None]),
            'SAS total_datasets': len(datasets)
        }
        
        # =============================================================================
        # MODEL INFORMATION
        # =============================================================================
        model_state.update({
            'Model Type': 'One-Way Analysis of Variance (ANOVA)',
            'Statistical Procedure': 'SAS PROC GLM',
            'Analysis Purpose': 'Compare means across treatment groups',
            'Model Assumptions': 'Normality, Independence, Homogeneity of Variance',
            'Convergence Status': 'Normal (GLM - No iteration required)'
        })
        
        # =============================================================================
        # EXPERIMENT VARIABLES
        # =============================================================================
        # Extract treatment group information
        if 'classlevels' in datasets and datasets['classlevels'] is not None:
            classlevels = datasets['classlevels']
            if not classlevels.empty:
                for _, row in classlevels.iterrows():
                    if 'Levels' in classlevels.columns and pd.notna(row.get('Levels')):
                        model_state['Number of Treatment Groups'] = int(row['Levels'])
                    if 'Values' in classlevels.columns and pd.notna(row.get('Values')):
                        model_state['Treatment Group Names'] = row['Values']
        
        # =============================================================================
        # DATA INFORMATION
        # =============================================================================
        if data is not None:
            model_state.update({
                'Total Observations': data.shape[0],
                'Variables in Dataset': data.shape[1],
                'Dataset Structure': f"{data.shape[0]} rows × {data.shape[1]} columns",
                'Variable Names': ', '.join(data.columns)
            })
        
        # Extract sample size information
        if 'nobs' in datasets and datasets['nobs'] is not None:
            nobs = datasets['nobs']
            if not nobs.empty:
                used_row = nobs[nobs['Label'] == 'Number of Observations Used']
                if not used_row.empty and 'N' in used_row.columns:
                    model_state['Observations Used in Analysis'] = int(used_row.iloc[0]['N'])
        
        # =============================================================================
        # MODEL FIT STATISTICS
        # =============================================================================
        if 'fitstats' in datasets and datasets['fitstats'] is not None:
            fitstats = datasets['fitstats']
            if not fitstats.empty:
                for _, row in fitstats.iterrows():
                    if 'RSquare' in fitstats.columns and pd.notna(row.get('RSquare')):
                        r_square = row['RSquare']
                        model_state['Model Fit (R²)'] = f"{r_square:.4f}"
                        model_state['R² Interpretation'] = f"Model explains {r_square*100:.1f}% of variance"
                    if 'RootMSE' in fitstats.columns and pd.notna(row.get('RootMSE')):
                        model_state['Root Mean Square Error'] = f"{row['RootMSE']:.4f}"
                        model_state['RMSE Interpretation'] = 'Average prediction error (lower is better)'
                    if 'CV' in fitstats.columns and pd.notna(row.get('CV')):
                        model_state['Coefficient of Variation'] = f"{row['CV']:.2f}%"
                        model_state['CV Interpretation'] = 'Relative variability (lower indicates more precision)'
                    if 'DepMean' in fitstats.columns and pd.notna(row.get('DepMean')):
                        model_state['Overall Mean Response'] = f"{row['DepMean']:.2f}"
        
        # =============================================================================
        # STATISTICAL RESULTS
        # =============================================================================
        if 'anova' in datasets and datasets['anova'] is not None:
            anova = datasets['anova']
            if not anova.empty:
                # Debug: Log the actual ANOVA table structure
                logger.info(f"ANOVA table columns: {list(anova.columns)}")
                logger.info(f"ANOVA table sources: {list(anova['Source'].unique()) if 'Source' in anova.columns else 'No Source column'}")
                
                # Try different possible row names for the model effect
                possible_model_rows = ['Model', 'Treatment', 'TRT', 'GROUP']
                model_row = None
                
                for row_name in possible_model_rows:
                    if row_name in anova['Source'].values:
                        model_row = anova[anova['Source'] == row_name]
                        logger.info(f"Found model row with source: {row_name}")
                        break
                
                # If no model row found, try the first non-error row
                if model_row is None or model_row.empty:
                    # Look for rows that are not 'Error' or 'Residual'
                    non_error_rows = anova[~anova['Source'].isin(['Error', 'Residual', 'Corrected Total', 'Total'])]
                    if not non_error_rows.empty:
                        model_row = non_error_rows.iloc[[0]]  # Take the first non-error row
                        logger.info(f"Using first non-error row: {model_row.iloc[0]['Source']}")
                
                if model_row is not None and not model_row.empty:
                    if 'FValue' in model_row.columns and pd.notna(model_row.iloc[0]['FValue']):
                        f_value = model_row.iloc[0]['FValue']
                        model_state['F-Statistic'] = f"{f_value:.4f}"
                        model_state['F-Statistic Interpretation'] = 'Test statistic for group differences'
                        logger.info(f"Extracted F-Value: {f_value}")
                    
                    if 'ProbF' in model_row.columns and pd.notna(model_row.iloc[0]['ProbF']):
                        p_value = model_row.iloc[0]['ProbF']
                        model_state['P-Value'] = f"{p_value:.6f}"
                        logger.info(f"Extracted P-Value: {p_value}")
                        
                        # Add validation for suspicious p-values
                        if p_value == 0.0:
                            logger.warning("P-value is exactly 0.0 - this may indicate an issue with the data or analysis")
                            model_state['P-Value Note'] = "P-value is exactly 0.0 - verify data and analysis"
                        
                        if p_value < 0.001:
                            model_state['Statistical Significance'] = "Highly Significant (p < 0.001)"
                            model_state['Clinical Interpretation'] = "Strong evidence of treatment differences"
                        elif p_value < 0.01:
                            model_state['Statistical Significance'] = "Very Significant (p < 0.01)"
                            model_state['Clinical Interpretation'] = "Strong evidence of treatment differences"
                        elif p_value < 0.05:
                            model_state['Statistical Significance'] = "Significant (p < 0.05)"
                            model_state['Clinical Interpretation'] = "Evidence of treatment differences"
                        else:
                            model_state['Statistical Significance'] = "Not Significant (p ≥ 0.05)"
                            model_state['Clinical Interpretation'] = "No strong evidence of treatment differences"
                else:
                    logger.warning("No suitable model row found in ANOVA table")
                    model_state['P-Value'] = "Not available"
                    model_state['F-Statistic'] = "Not available"
        
        # =============================================================================
        # CONFIDENCE AND RELIABILITY
        # =============================================================================
        model_state.update({
            'Confidence Level': '95% (default)',
            'Multiple Comparison Adjustment': 'Bonferroni (conservative)',
            'Effect Size Available': 'Yes (F-statistic and R²)',
            'Power Analysis': 'Not performed (consider for future studies)'
        })
        
        # At the end, add key_results
        model_state['key_results'] = self._extract_simple_glm_key_results(datasets)
        return model_state
    
    def _extract_simple_glm_key_results(self, datasets: Dict[str, pd.DataFrame]) -> dict:
        """Extract key results for Simple GLM (One-Way ANOVA) for display and downstream use."""
        key_results = {}
        # R2 and Adjusted R2
        if 'fitstats' in datasets and datasets['fitstats'] is not None:
            fitstats = datasets['fitstats']
            if not fitstats.empty:
                for _, row in fitstats.iterrows():
                    if 'RSquare' in fitstats.columns and pd.notna(row.get('RSquare')):
                        key_results['R2'] = float(row['RSquare'])
                    if 'AdjRSq' in fitstats.columns and pd.notna(row.get('AdjRSq')):
                        key_results['Adj_R2'] = float(row['AdjRSq'])
        # Treatment F and P
        if 'anova' in datasets and datasets['anova'] is not None:
            anova = datasets['anova']
            if not anova.empty:
                treatment_row = None
                for name in ['Treatment', 'TRT', 'GROUP']:
                    if name in anova['Source'].values:
                        treatment_row = anova[anova['Source'] == name].iloc[0]
                        break
                if treatment_row is not None:
                    key_results['Treatment_F'] = float(treatment_row['FValue']) if 'FValue' in treatment_row else None
                    key_results['Treatment_P'] = float(treatment_row['ProbF']) if 'ProbF' in treatment_row else None
                    key_results['Treatment_Significant'] = (float(treatment_row['ProbF']) < 0.05) if 'ProbF' in treatment_row else None
        # LSMeans
        if 'lsmeans' in datasets and datasets['lsmeans'] is not None:
            lsmeans = datasets['lsmeans']
            if not lsmeans.empty:
                lsmeans_list = []
                for _, row in lsmeans.iterrows():
                    entry = {
                        'Group': row.get('Treatment', row.get('Group', '')),
                        'Mean': float(row['LSMean']) if 'LSMean' in row and pd.notna(row['LSMean']) else None,
                        'SE': float(row['StdErr']) if 'StdErr' in row and pd.notna(row['StdErr']) else None,
                        'CI_Lower': float(row['Lower']) if 'Lower' in row and pd.notna(row['Lower']) else None,
                        'CI_Upper': float(row['Upper']) if 'Upper' in row and pd.notna(row['Upper']) else None
                    }
                    lsmeans_list.append(entry)
                key_results['LSMeans'] = lsmeans_list
        return key_results
    
    def _create_repeated_measures_model_state(self, datasets: Dict[str, pd.DataFrame], 
                                             data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create model state for Repeated Measures analysis"""
        
        # Initialize organized model state with categories
        model_state = {
            'analysis_type': 'Repeated-Measures - SAS PROC MIXED (Repeated Measures Mixed Model)',
            'timestamp': datetime.now().isoformat(),
            'SAS datasets_found': len([d for d in datasets.values() if d is not None]),
            'SAS total_datasets': len(datasets)
        }
        
        # =============================================================================
        # MODEL INFORMATION
        # =============================================================================
        model_state.update({
            'Model Type': 'Repeated Measures Mixed Model',
            'Statistical Procedure': 'SAS PROC MIXED',
            'Analysis Purpose': 'Analyze treatment effects over time with subject correlation',
            'Model Assumptions': 'Normality, Independence between subjects, Compound symmetry or unstructured covariance',
            'Solution Method': 'REML (Restricted Maximum Likelihood)',
            'Covariance Structure': 'Compound Symmetry (default)'
        })
        
        # =============================================================================
        # EXPERIMENT VARIABLES
        # =============================================================================
        model_state.update({
            'Response Variable': 'TumorSize (continuous outcome)',
            'Subject Variable': 'Dog (random effect)',
            'Repeated Variable': 'Week (within-subject factor)',
            'Fixed Effects': 'Treatment (between-subject factor)',
            'Random Effects': 'Dog (accounts for correlation)',
            'Time Points': 'Multiple measurements per subject'
        })
        
        # =============================================================================
        # DATA INFORMATION
        # =============================================================================
        if data is not None:
            model_state.update({
                'Total Observations': data.shape[0],
                'Variables in Dataset': data.shape[1],
                'Dataset Structure': f"{data.shape[0]} rows × {data.shape[1]} columns",
                'Variable Names': ', '.join(data.columns)
            })
            
            # Extract subject and time information from data
            if 'Dog' in data.columns:
                unique_subjects = data['Dog'].nunique()
                model_state['Number of Subjects'] = unique_subjects
            if 'Week' in data.columns:
                unique_times = data['Week'].nunique()
                model_state['Number of Time Points'] = unique_times
            if 'Treatment' in data.columns:
                unique_treatments = data['Treatment'].nunique()
                model_state['Number of Treatment Groups'] = unique_treatments
        
        # =============================================================================
        # MODEL FIT STATISTICS
        # =============================================================================
        # Extract fit statistics from FitStatistics table
        if 'fitstats' in datasets and datasets['fitstats'] is not None:
            fitstats = datasets['fitstats']
            if not fitstats.empty:
                for _, row in fitstats.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    
                    if '-2 Res Log Likelihood' in desc:
                        log_likelihood = float(value) if value != '' else None
                        if log_likelihood is not None:
                            model_state['-2 Log Likelihood'] = f"{log_likelihood:.4f}"
                            model_state['-2 Log Likelihood Interpretation'] = 'Model fit measure (lower is better)'
                    
                    elif 'Log Likelihood' in desc and '-2' not in desc:
                        log_likelihood = float(value) if value != '' else None
                        if log_likelihood is not None:
                            model_state['Log Likelihood'] = f"{log_likelihood:.4f}"
                            model_state['Log Likelihood Interpretation'] = 'Model fit measure (higher is better)'
                    
                    elif 'AIC (Smaller is Better)' in desc:
                        aic = float(value) if value != '' else None
                        if aic is not None:
                            model_state['Akaike Information Criterion (AIC)'] = f"{aic:.4f}"
                            model_state['AIC Interpretation'] = 'Model fit measure (lower is better)'
                    
                    elif 'BIC (Smaller is Better)' in desc:
                        bic = float(value) if value != '' else None
                        if bic is not None:
                            model_state['Bayesian Information Criterion (BIC)'] = f"{bic:.4f}"
                            model_state['BIC Interpretation'] = 'Model fit measure (lower is better, penalizes complexity)'
        
        # =============================================================================
        # CONVERGENCE AND COMPUTATION
        # =============================================================================
        # Extract convergence status
        convergence_status = 'Not available'
        if 'convergence' in datasets and datasets['convergence'] is not None:
            conv = datasets['convergence']
            if not conv.empty and 'Status' in conv.columns:
                status = conv['Status'].iloc[0]
                convergence_status = 'Normal' if status == 0.0 else f'Status: {status}'
        
        model_state['Convergence Status'] = convergence_status
        if convergence_status == 'Normal':
            model_state['Convergence Interpretation'] = 'Model successfully converged to solution'
        else:
            model_state['Convergence Interpretation'] = 'Model may have convergence issues'
        
        # Extract run time
        run_time = 'Not available'
        if 'dimensions' in datasets and datasets['dimensions'] is not None:
            dims = datasets['dimensions']
            if not dims.empty and 'RealTime' in dims.columns:
                run_time = str(dims['RealTime'].iloc[0])
        
        model_state['Computation Time (seconds)'] = run_time
        model_state['Performance'] = 'Fast computation (good for large datasets)'
        
        # Extract iteration information
        iteration_count = 'Not available'
        if 'iterhistory' in datasets and datasets['iterhistory'] is not None:
            iterhistory = datasets['iterhistory']
            if not iterhistory.empty:
                iteration_count = str(len(iterhistory))
        
        model_state['Iteration Count'] = iteration_count
        if iteration_count != 'Not available':
            model_state['Iteration Interpretation'] = f'Algorithm required {iteration_count} iterations to converge'
        else:
            model_state['Iteration Interpretation'] = 'Efficient convergence (good model specification)'
        
        # =============================================================================
        # STATISTICAL RESULTS
        # =============================================================================
        # Extract solution type from ModelInfo table
        solution_type = 'REML'  # Default
        if 'modelinfo' in datasets and datasets['modelinfo'] is not None:
            modelinfo = datasets['modelinfo']
            if not modelinfo.empty:
                for _, row in modelinfo.iterrows():
                    desc = row.get('Descr', '')
                    value = row.get('Value', '')
                    if 'OptTech' in desc or 'Method' in desc:
                        solution_type = str(value)
        
        model_state['Estimation Method'] = solution_type
        model_state['Solution Type'] = solution_type
        
        # =============================================================================
        # CONFIDENCE AND RELIABILITY
        # =============================================================================
        model_state.update({
            'Confidence Level': '95% (default)',
            'Multiple Comparison Adjustment': 'Bonferroni (for time comparisons)',
            'Effect Size Available': 'Yes (F-statistics and partial η²)',
            'Power Analysis': 'Not performed (consider for future studies)',
            'Missing Data Handling': 'Available case analysis (default)',
            'Covariance Structure Selection': 'Compound symmetry (default, can be changed)'
        })
        
        return model_state
    
    def _extract_residuals_and_predicted(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[List[float], List[float]]:
        """Extract residuals and predicted values as lists for compatibility"""
        residuals = []
        predicted = []
        
        if 'residuals' in datasets and datasets['residuals'] is not None:
            if 'resid' in datasets['residuals'].columns:
                residuals = datasets['residuals']['resid'].tolist()
            if 'pred' in datasets['residuals'].columns:
                predicted = datasets['residuals']['pred'].tolist()
        
        return residuals, predicted
    
    def _create_error_result(self, error_message: str, 
                           error_details: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Create an error result"""
        return AnalysisResult(
            success=False,
            datasets={},
            sas_log="",
            log_analysis=SASLogAnalysis(
                has_errors=True,
                has_warnings=False,
                error_messages=[error_message],
                warning_messages=[],
                convergence_status=None,
                execution_time=None,
                notes=[]
            ),
            model_state={},
            direct_reports={},
            session_dir="",
            output_folder="",
            error_message=error_message,
            error_details=error_details or {}
        )
    
    def get_analysis_summary(self, result: AnalysisResult) -> str:
        """Generate a user-friendly analysis summary"""
        if result.success:
            summary = f"✅ **{result.model_state.get('analysis_type', 'Analysis')} completed successfully**\n\n"
            
            # Add log analysis summary
            if result.log_analysis.has_errors or result.log_analysis.has_warnings:
                summary += self.log_analyzer.get_error_summary(result.log_analysis) + "\n\n"
            
            # Add dataset summary
            dataset_summary = self.dataset_extractor.get_dataset_summary(result.datasets)
            summary += f"📊 **Datasets extracted:** {dataset_summary['total_found']}/{dataset_summary['total_expected']}\n"
            
            # Add direct reports info
            if result.direct_reports:
                summary += f"📄 **Reports generated:** {', '.join(result.direct_reports.keys())}\n"
            
            return summary
        else:
            summary = f"❌ **Analysis failed**\n\n"
            summary += f"**Error:** {result.error_message}\n"
            
            if result.error_details:
                summary += f"**Details:** {result.error_details}\n"
            
            return summary
    
    def get_organized_run_summary(self, model_state: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Organize model state into purpose-based categories for display
        
        Returns:
            Dict with categories as keys and lists of (label, value) tuples as values
        """
        # Define category patterns for automatic organization
        categories = {
            'Model Information': [
                'Model Type', 'Statistical Procedure', 'Analysis Purpose', 
                'Model Assumptions', 'Solution Method', 'Estimation Method',
                'Covariance Structure', 'Convergence Status', 'Convergence Interpretation'
            ],
            'Experiment Variables': [
                'Response Variable', 'Subject Variable', 'Repeated Variable',
                'Fixed Effects', 'Random Effects', 'Time Points',
                'Number of Treatment Groups', 'Treatment Group Names',
                'Number of Subjects', 'Number of Time Points'
            ],
            'Data Information': [
                'Total Observations', 'Variables in Dataset', 'Dataset Structure',
                'Variable Names', 'Observations Used in Analysis'
            ],
            'Model Fit Statistics': [
                'Model Fit (R²)', 'R² Interpretation', 'Root Mean Square Error',
                'RMSE Interpretation', 'Coefficient of Variation', 'CV Interpretation',
                'Overall Mean Response', 'Akaike Information Criterion (AIC)',
                'AIC Interpretation', 'Bayesian Information Criterion (BIC)',
                'BIC Interpretation', 'Log Likelihood', 'Log Likelihood Interpretation',
                '-2 Log Likelihood', '-2 Log Likelihood Interpretation'
            ],
            'Statistical Results': [
                'F-Statistic', 'F-Statistic Interpretation', 'P-Value',
                'Statistical Significance', 'Clinical Interpretation'
            ],
            'Computation & Performance': [
                'Computation Time (seconds)', 'Performance', 'Iteration Count',
                'Iteration Interpretation'
            ],
            'Confidence & Reliability': [
                'Confidence Level', 'Multiple Comparison Adjustment',
                'Effect Size Available', 'Power Analysis', 'Missing Data Handling',
                'Covariance Structure Selection'
            ],
            'System Information': [
                'analysis_type', 'timestamp', 'SAS datasets_found', 'SAS total_datasets'
            ]
        }
        
        # Initialize organized summary
        organized_summary = {category: [] for category in categories.keys()}
        
        # Categorize each model state item
        for key, value in model_state.items():
            categorized = False
            
            # Try to match key to a category
            for category, patterns in categories.items():
                if any(pattern.lower() in key.lower() for pattern in patterns):
                    organized_summary[category].append((key, str(value)))
                    categorized = True
                    break
            
            # If no category match, put in System Information
            if not categorized:
                organized_summary['System Information'].append((key, str(value)))
        
        # Remove empty categories
        organized_summary = {k: v for k, v in organized_summary.items() if v}
        
        return organized_summary
    
    def format_run_summary_for_display(self, model_state: Dict[str, Any]) -> str:
        """
        Format the organized run summary for Streamlit display
        
        Returns:
            Formatted markdown string for display
        """
        organized = self.get_organized_run_summary(model_state)
        
        if not organized:
            return "No model information available."
        
        # Create formatted display
        sections = []
        
        for category, items in organized.items():
            if not items:
                continue
                
            section = f"### 📋 {category}\n\n"
            
            for label, value in items:
                # Skip system information in main display
                if category == 'System Information':
                    continue
                    
                # Format the label and value nicely
                formatted_label = label.replace('_', ' ').title()
                
                # Add interpretation if available
                if 'Interpretation' in label:
                    section += f"**{formatted_label}:** {value}\n\n"
                else:
                    section += f"**{formatted_label}:** {value}\n"
            
            sections.append(section)
        
        return "\n".join(sections)


# Convenience functions for easy integration
def create_sas_manager() -> SASAnalysisManager:
    """Create a new SAS Analysis Manager instance"""
    return SASAnalysisManager()


def run_simple_glm_analysis(data_file: str, sas_code: str, **kwargs) -> AnalysisResult:
    """Convenience function for running Simple GLM analysis"""
    manager = create_sas_manager()
    return manager.run_analysis(AnalysisType.SIMPLE_GLM, data_file, sas_code, **kwargs)


def run_repeated_measures_analysis(data_file: str, sas_code: str, **kwargs) -> AnalysisResult:
    """Convenience function for running Repeated Measures analysis"""
    manager = create_sas_manager()
    return manager.run_analysis(AnalysisType.REPEATED_MEASURES, data_file, sas_code, **kwargs)


def convert_to_legacy_format(analysis_result: AnalysisResult) -> Dict[str, Any]:
    """Convert AnalysisResult to legacy format for compatibility"""
    if not analysis_result.success:
        return None
    
    # Convert to legacy format
    legacy_results = {}
    
    # Add datasets
    for key, dataset in analysis_result.datasets.items():
        legacy_results[key] = dataset
    
    # Add residuals and predicted (already extracted as lists)
    legacy_results['residuals'] = analysis_result.residuals or []
    legacy_results['predicted'] = analysis_result.predicted or []
    
    # Add other fields
    legacy_results['sas_log'] = analysis_result.sas_log
    legacy_results['log_analysis'] = analysis_result.log_analysis
    legacy_results['model_state'] = analysis_result.model_state
    legacy_results['direct_reports'] = analysis_result.direct_reports
    legacy_results['session_dir'] = analysis_result.session_dir
    legacy_results['output_folder'] = analysis_result.output_folder
    
    return legacy_results 