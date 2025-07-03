"""
SAS Integrity Wrapper - Based on Superior Process Concept

This implements the user's process concept:
1. HTML as "Model Report of Record" (zero translation loss)
2. Minimal transformation pipeline (direct ODS OUTPUT)
3. Generic wrapper for any SAS procedure
4. SHA-256 checksums for integrity verification
5. Session isolation and provenance tracking

Key Innovation: HTML preserves exact ODS formatting, print-ready for stakeholders
"""

import saspy
import hashlib
import json
import datetime
import pathlib
import uuid
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelExecutionManifest:
    """Provenance manifest for model execution"""
    run_id: str
    model_name: str
    timestamp: str
    ods_report: str
    tables: List[str]
    log_sha256: str
    session_id: str
    model_type: str
    sas_log: str = ""  # Raw SAS log content
    
    # Additional metadata
    input_data_shape: Optional[tuple] = None
    execution_time: Optional[str] = None
    sas_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def save(self, output_dir: pathlib.Path):
        """Save manifest to JSON file"""
        manifest_path = output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved manifest to: {manifest_path}")


class SASIntegrityWrapper:
    """
    Generic wrapper for SAS model execution with integrity preservation
    
    Key Features:
    - HTML "Model Report of Record" (zero translation loss)
    - Direct ODS OUTPUT capture (minimal transformation)
    - SHA-256 checksums for integrity verification
    - Session isolation and provenance tracking
    - Generic design for any SAS procedure
    """
    
    def __init__(self, output_base_dir: str = './sas_runs'):
        self.output_base_dir = pathlib.Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.sas_session: Optional[saspy.SASsession] = None
        self.current_run_id: Optional[str] = None
        self.current_output_dir: Optional[pathlib.Path] = None
        
        logger.info(f"Initialized SAS Integrity Wrapper: {output_base_dir}")
    
    def bootstrap_session(self) -> bool:
        """
        Bootstrap clean SAS ODA session with ODA best practices
        
        Returns:
            bool: True if session created successfully
        """
        try:
            # Create clean ODA session with HTML results
            self.sas_session = saspy.SASsession(results='HTML')
            
            # Generate unique run ID
            self.current_run_id = uuid.uuid4().hex[:12]
            
            # Get ODA user ID and set up proper paths
            user_id = self._get_oda_user_id()
            if user_id:
                home_dir = f"/home/{user_id}"
                logger.info(f"ODA home directory: {home_dir}")
                
                # Test write access to ODA home directory
                test_code = f"""
                data _null_;
                    file "{home_dir}/test_write.tmp";
                    put "test";
                run;
                """
                result = self.sas_session.submit(test_code)
                if 'ERROR:' in result.get('LOG', ''):
                    logger.warning("ODA home directory may not be writable - using local output directory")
                    home_dir = None
                else:
                    logger.info("ODA home directory is writable")
            else:
                logger.warning("Could not determine ODA user ID - using local output directory")
                home_dir = None
            
            # Create local output directory (more reliable than ~ paths)
            self.current_output_dir = self.output_base_dir / self.current_run_id
            self.current_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Bootstrap successful - Run ID: {self.current_run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            return False
    
    def execute_model(self, sas_code: str, tables: List[str], 
                     model_name: str, model_type: str = "unknown", 
                     report_style: str = "statistical") -> Optional[ModelExecutionManifest]:
        """
        Execute SAS model with integrity preservation
        
        Args:
            sas_code: Full SAS program (with %SYSFUNC etc. already resolved)
            tables: List of ODS table names to capture (e.g., ["FitStatistics", "OverallANOVA"])
            model_name: Logical name for provenance (e.g., "Simple_GLM")
            model_type: Type of model (e.g., "glm", "mixed", "logistic")
            report_style: ODS HTML style for professional formatting 
                         ("statistical", "journal", "minimal", "analysis")
        
        Returns:
            ModelExecutionManifest: Complete provenance manifest
        """
        if not self.sas_session:
            logger.error("SAS session not initialized. Call bootstrap_session() first.")
            return None
        
        try:
            logger.info(f"Executing model: {model_name} (Run ID: {self.current_run_id})")
            
            # Step 1: Define HTML report file (Model Report of Record) - Use relative path
            html_file = str(self.current_output_dir / f"report_{self.current_run_id}.html")
            
            # Step 2: Wrap SAS code with ODS and integrity features
            wrapped_code = f"""
                options errors=20 nodate nonumber;
                
                /* Professional HTML report generation with ODA best practices */
                ods trace on;
                ods html file="report_{self.current_run_id}.html" 
                         style={report_style} 
                         gtitle gfootnote
                         contents="report_{self.current_run_id}_toc.html"
                         frame="report_{self.current_run_id}_frame.html";
                
                /* Enable graphics for diagnostic plots */
                ods graphics on / width=800px height=600px outputfmt=png;
                
                /* ODS OUTPUT statements BEFORE procedures (working approach) */
                {''.join([f'ods output {t}=work.{t.lower()};' for t in tables])}
                
                {sas_code}               /* <-- your model lives here */
                
                ods html close;
                ods trace off;
            """
            
            # Step 4: Execute the wrapped code
            logger.info("Submitting SAS code...")
            result = self.sas_session.submit(wrapped_code)
            log = result.get('LOG', '')
            
            # Step 5: Create provenance manifest
            manifest = ModelExecutionManifest(
                run_id=self.current_run_id,
                model_name=model_name,
                model_type=model_type,
                timestamp=datetime.datetime.utcnow().isoformat(),
                ods_report=html_file,
                tables=tables,
                log_sha256=hashlib.sha256(log.encode()).hexdigest(),
                session_id=str(id(self.sas_session)),
                sas_log=log,  # Store the actual SAS log content
                execution_time=datetime.datetime.now().isoformat(),
                sas_version=self.sas_session.symget('SYSVER') if self.sas_session else "unknown"
            )
            
            # Step 6: Save manifest
            manifest.save(self.current_output_dir)
            
            # Step 7: Copy HTML files from SAS working directory to our output directory
            self._copy_html_files_to_output_dir()
            
            logger.info(f"✓ Model execution completed - manifest written to {self.current_run_id}")
            return manifest
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            return None
    
    def download_results(self, manifest: ModelExecutionManifest) -> Dict[str, pd.DataFrame]:
        """
        Download results as pandas DataFrames
        
        Args:
            manifest: Model execution manifest
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of table names to DataFrames
        """
        if not self.sas_session:
            logger.error("SAS session not available")
            return {}
        
        dfs = {}
        
        try:
            for table_name in manifest.tables:
                logger.info(f"Downloading table: {table_name}")
                
                # Get DataFrame directly from SAS WORK library
                sas_data = self.sas_session.sasdata(table_name.lower(), libref='WORK')
                df = sas_data.to_df()
                
                if df is not None and not df.empty:
                    dfs[table_name] = df
                    
                    # Save as CSV with integrity preservation
                    csv_path = self.current_output_dir / f"{table_name}.csv"
                    df.to_csv(csv_path, index=False)
                    
                    logger.info(f"Saved {table_name}: {df.shape} -> {csv_path}")
                else:
                    logger.warning(f"Table {table_name} is empty or None")
                    dfs[table_name] = pd.DataFrame()
            
            return dfs
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {}
    
    def create_archive(self, manifest: ModelExecutionManifest, 
                      dataframes: Dict[str, pd.DataFrame]) -> str:
        """
        Create one-click viewer & archive
        
        Args:
            manifest: Model execution manifest
            dataframes: Downloaded DataFrames
            
        Returns:
            str: Path to created archive
        """
        try:
            from zipfile import ZipFile
            
            archive_path = self.current_output_dir / f"SAS_{self.current_run_id}.zip"
            
            with ZipFile(archive_path, "w") as z:
                # Add manifest
                z.write(self.current_output_dir / 'manifest.json', arcname='manifest.json')
                
                # Add HTML Model Report of Record (with frame and contents)
                html_path = pathlib.Path(manifest.ods_report)
                if html_path.exists():
                    z.write(html_path, arcname='Model_Report.html')
                    
                    # Add frame and contents files if they exist
                    frame_path = html_path.parent / f"report_{self.current_run_id}_frame.html"
                    contents_path = html_path.parent / f"report_{self.current_run_id}_toc.html"
                    
                    if frame_path.exists():
                        z.write(frame_path, arcname='Model_Report_Frame.html')
                    if contents_path.exists():
                        z.write(contents_path, arcname='Model_Report_TOC.html')
                
                # Add CSV files
                for table_name in manifest.tables:
                    csv_path = self.current_output_dir / f"{table_name}.csv"
                    if csv_path.exists():
                        z.write(csv_path, arcname=f"{table_name}.csv")
            
            logger.info(f"Created archive: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Archive creation failed: {e}")
            return ""
    
    def cleanup_session(self):
        """Clean up SAS session"""
        if self.sas_session:
            try:
                self.sas_session.endsas()
                logger.info("SAS session cleaned up")
            except Exception as e:
                logger.warning(f"Session cleanup warning: {e}")
        
        self.sas_session = None
        self.current_run_id = None
        self.current_output_dir = None
    
    def _copy_html_files_to_output_dir(self):
        """
        Copy HTML files from SAS WORK directory to our output directory using saspy download
        """
        try:
            # Files to download from SAS WORK directory
            html_files = [
                f"report_{self.current_run_id}.html",
                f"report_{self.current_run_id}_frame.html", 
                f"report_{self.current_run_id}_toc.html"
            ]
            
            for html_file in html_files:
                try:
                    # Use saspy download to get file from SAS WORK directory
                    local_path = str(self.current_output_dir / html_file)
                    self.sas_session.download(html_file, local_path)
                    logger.info(f"Downloaded {html_file} to {local_path}")
                except Exception as download_error:
                    logger.warning(f"Could not download {html_file}: {download_error}")
                    
        except Exception as e:
            logger.error(f"Failed to download HTML files: {e}")
    
    def _get_oda_user_id(self) -> Optional[str]:
        """
        Get ODA user ID by parsing the SAS working directory
        """
        try:
            if not self.sas_session:
                logger.error("SAS session not available")
                return None
            
            # Get the SAS working directory
            result = self.sas_session.submit('data _null_; pwd = getoption("work"); put "WORK: " pwd; run;')
            log = result.get('LOG', '')
            
            # Parse user ID from working directory path
            import re
            work_match = re.search(r'/saswork/SAS_work[0-9A-F]+_([^/]+)\.oda\.sas\.com', log)
            
            if work_match:
                user_id = work_match.group(1)
                logger.info(f"Parsed ODA user ID from working directory: {user_id}")
                return user_id
            else:
                # Fallback to known user ID if parsing fails
                known_user_id = "u64261399"
                logger.warning(f"Could not parse user ID from working directory, using fallback: {known_user_id}")
                return known_user_id
                
        except Exception as e:
            logger.error(f"Failed to get ODA user ID: {e}")
            # Final fallback
            return "u64261399"


# Convenience functions for common model types
def execute_simple_glm(data: pd.DataFrame, treatment_col: str, response_col: str, 
                      wrapper: SASIntegrityWrapper) -> Optional[Dict[str, Any]]:
    """
    Execute Simple GLM (One-Way ANOVA) with integrity preservation
    
    Args:
        data: Input DataFrame
        treatment_col: Treatment group column name
        response_col: Response variable column name
        wrapper: SASIntegrityWrapper instance
        
    Returns:
        Dict containing manifest, dataframes, and archive path
    """
    
    # Transfer data to SAS
    wrapper.sas_session.df2sd(data, 'testdata')
    
    # Simple GLM SAS code - raw procedures only (ODS OUTPUT handled by execute_model)
    sas_code = f"""
    proc printto log="sas_debug_{wrapper.current_run_id}.log"; run;
    
    proc glm data=work.testdata plots=diagnostics;
        class {treatment_col};
        model {response_col} = {treatment_col} / solution;
        lsmeans {treatment_col} / stderr pdiff cl adjust=bon;
        output out=work.residuals r=resid p=pred;
    run;
    
    proc univariate data=work.residuals normal;
        var resid;
    run;
    quit;
    
    proc printto; run;
    """
    
    # ODS tables to capture
    tables_needed = [
        "FitStatistics", "OverallANOVA", "LSMeans",
        "LSMeanDiffCL", "ParameterEstimates",
        "NObs", "ClassLevels", "TestsForNormality"
    ]
    
    # Execute model
    manifest = wrapper.execute_model(sas_code, tables_needed, "Simple_GLM", "glm")
    
    if manifest:
        # Download results
        dataframes = wrapper.download_results(manifest)
        
        # Create archive
        archive_path = wrapper.create_archive(manifest, dataframes)
        
        return {
            'manifest': manifest,
            'dataframes': dataframes,
            'archive_path': archive_path,
            'success': True
        }
    else:
        return {'success': False, 'error': 'Model execution failed'}


def execute_repeated_measures(data: pd.DataFrame, treatment_col: str, response_col: str,
                            time_col: str, subject_col: str,
                            wrapper: SASIntegrityWrapper) -> Optional[Dict[str, Any]]:
    """
    Execute Repeated Measures analysis with integrity preservation
    
    Args:
        data: Input DataFrame (long format)
        treatment_col: Treatment group column name
        response_col: Response variable column name
        time_col: Time variable column name
        subject_col: Subject ID column name
        wrapper: SASIntegrityWrapper instance
        
    Returns:
        Dict containing manifest, dataframes, and archive path
    """
    
    # Transfer data to SAS
    wrapper.sas_session.df2sd(data, 'repeated_data')
    
    # Repeated Measures SAS code - raw procedures only (ODS OUTPUT handled by execute_model)
    sas_code = f"""
    proc printto log="sas_debug_{wrapper.current_run_id}.log"; run;
    
    proc mixed data=work.repeated_data method=reml itdetails;
        class {treatment_col} {time_col} {subject_col};
        model {response_col} = {treatment_col}|{time_col} / solution ddfm=kr outp=resid;
        repeated {time_col} / subject={subject_col} type=AR(1);
        lsmeans {treatment_col}*{time_col} / diff cl adjust=bon;
    run;
    
    proc printto; run;
    """
    
    # ODS tables to capture
    tables_needed = [
        "FitStatistics", "LSMeans", "Diffs", "CovParms",
        "Tests3", "SolutionF", "ConvergenceStatus", "ModelInfo",
        "IterHistory", "Dimensions", "NObs", "InfoCrit"
    ]
    
    # Execute model
    manifest = wrapper.execute_model(sas_code, tables_needed, "Repeated_Measures", "mixed")
    
    if manifest:
        # Download results
        dataframes = wrapper.download_results(manifest)
        
        # Create archive
        archive_path = wrapper.create_archive(manifest, dataframes)
        
        return {
            'manifest': manifest,
            'dataframes': dataframes,
            'archive_path': archive_path,
            'success': True
        }
    else:
        return {'success': False, 'error': 'Model execution failed'}


# Global wrapper instance
integrity_wrapper = SASIntegrityWrapper() 