"""
SAS Model Executor - Minimal Transformation Pipeline

This module implements a minimal transformation pipeline that:
1. Executes SAS models with session-level coordination
2. Collects raw outputs with zero manipulation
3. Provides structured return for UI generation and downstream analysis
4. Is ready for future expansion (provenance, compliance, etc.)

Architecture Principle: Minimal data flow from SAS to calling function
"""

import saspy
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelSpecification:
    """Specification for SAS model execution"""
    model_type: str  # 'simple_glm', 'repeated_measures', etc.
    sas_code: str    # Complete SAS code to execute
    input_data: pd.DataFrame
    output_datasets: List[str]  # Expected output dataset names
    report_formats: List[str]   # ['pdf', 'rtf', 'html']
    session_dir: str
    model_name: str = "model"
    
    def __post_init__(self):
        """Validate specification"""
        if not self.sas_code.strip():
            raise ValueError("SAS code cannot be empty")
        if self.input_data.empty:
            raise ValueError("Input data cannot be empty")
        if not self.output_datasets:
            raise ValueError("Must specify expected output datasets")


@dataclass
class RawSASOutput:
    """Raw SAS outputs with zero manipulation"""
    # Raw datasets exactly as extracted from SAS
    datasets: Dict[str, pd.DataFrame]
    
    # Direct ODS reports (file paths)
    reports: Dict[str, str]
    
    # SAS log and metadata
    sas_log: str
    execution_time: str
    session_id: str
    
    # Model metadata
    model_type: str
    model_name: str
    
    # Integrity verification
    dataset_checksums: Dict[str, str]
    
    def __post_init__(self):
        """Calculate checksums for integrity verification"""
        if not hasattr(self, 'dataset_checksums'):
            self.dataset_checksums = {}
            for name, dataset in self.datasets.items():
                if dataset is not None and not dataset.empty:
                    self.dataset_checksums[name] = self._calculate_checksum(dataset)
    
    def _calculate_checksum(self, dataset: pd.DataFrame) -> str:
        """Calculate SHA256 checksum of dataset content"""
        content_str = dataset.to_string()
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def verify_integrity(self) -> Dict[str, bool]:
        """Verify dataset integrity"""
        integrity_status = {}
        for name, dataset in self.datasets.items():
            if dataset is not None and not dataset.empty:
                current_checksum = self._calculate_checksum(dataset)
                original_checksum = self.dataset_checksums.get(name, '')
                integrity_status[name] = current_checksum == original_checksum
            else:
                integrity_status[name] = True
        return integrity_status


@dataclass
class ModelResult:
    """Complete model execution result"""
    success: bool
    raw_output: Optional[RawSASOutput]
    error_message: Optional[str]
    execution_metadata: Dict[str, Any]
    
    # Future-ready fields (for expansion)
    provenance_data: Optional[Dict[str, Any]] = None
    compliance_metadata: Optional[Dict[str, Any]] = None


class SASSessionCoordinator:
    """Coordinates SAS session for model execution"""
    
    def __init__(self, session_id: str, session_dir: str):
        self.session_id = session_id
        self.session_dir = session_dir
        self.sas_session: Optional[saspy.SASsession] = None
        self.is_locked = False
        self.execution_count = 0
        
        # Create session directory
        os.makedirs(session_dir, exist_ok=True)
        
        logger.info(f"Initialized SAS session coordinator: {session_id}")
    
    def lock_session(self) -> bool:
        """Lock session for model execution"""
        if self.is_locked:
            logger.warning(f"Session {self.session_id} is already locked")
            return False
        
        try:
            # Initialize SAS session
            self.sas_session = saspy.SASsession(cfgname='oda')
            self.is_locked = True
            self.execution_count = 0
            
            logger.info(f"Locked SAS session: {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to lock SAS session: {e}")
            return False
    
    def unlock_session(self):
        """Unlock and cleanup session"""
        if self.sas_session:
            try:
                self.sas_session.endsas()
                logger.info(f"Unlocked SAS session: {self.session_id}")
            except Exception as e:
                logger.warning(f"Error during session cleanup: {e}")
        
        self.sas_session = None
        self.is_locked = False
    
    def execute_model(self, model_spec: ModelSpecification) -> ModelResult:
        """Execute model within locked session"""
        if not self.is_locked or not self.sas_session:
            raise RuntimeError("Session must be locked before model execution")
        
        self.execution_count += 1
        execution_id = f"{self.session_id}_exec_{self.execution_count}"
        
        logger.info(f"Executing model: {model_spec.model_name} (Execution: {execution_id})")
        
        try:
            # Step 1: Transfer input data to SAS
            logger.info("Transferring input data to SAS")
            self.sas_session.df2sd(model_spec.input_data, 'input_data')
            
            # Step 2: Execute SAS code
            logger.info("Executing SAS code")
            result = self.sas_session.submit(model_spec.sas_code)
            
            # Step 3: Collect raw outputs (zero manipulation)
            logger.info("Collecting raw outputs")
            raw_output = self._collect_raw_outputs(model_spec, result, execution_id)
            
            # Step 4: Create model result
            model_result = ModelResult(
                success=True,
                raw_output=raw_output,
                error_message=None,
                execution_metadata={
                    'execution_id': execution_id,
                    'session_id': self.session_id,
                    'execution_count': self.execution_count,
                    'model_type': model_spec.model_type,
                    'model_name': model_spec.model_name,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Model execution successful: {execution_id}")
            return model_result
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            return ModelResult(
                success=False,
                raw_output=None,
                error_message=str(e),
                execution_metadata={
                    'execution_id': execution_id,
                    'session_id': self.session_id,
                    'execution_count': self.execution_count,
                    'model_type': model_spec.model_type,
                    'model_name': model_spec.model_name,
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def _collect_raw_outputs(self, model_spec: ModelSpecification, 
                           sas_result: Dict[str, Any], execution_id: str) -> RawSASOutput:
        """Collect raw outputs with zero manipulation"""
        
        # Collect datasets exactly as they exist in SAS
        datasets = {}
        for dataset_name in model_spec.output_datasets:
            try:
                if self.sas_session.exist(dataset_name):
                    dataset = self.sas_session.sasdata2dataframe(dataset_name)
                    datasets[dataset_name] = dataset
                    logger.info(f"Collected dataset: {dataset_name} ({dataset.shape if dataset is not None else 'None'})")
                else:
                    logger.warning(f"Expected dataset not found: {dataset_name}")
                    datasets[dataset_name] = None
            except Exception as e:
                logger.error(f"Failed to collect dataset {dataset_name}: {e}")
                datasets[dataset_name] = None
        
        # Collect reports (file paths)
        reports = {}
        for report_format in model_spec.report_formats:
            report_path = os.path.join(self.session_dir, f"{execution_id}.{report_format}")
            if os.path.exists(report_path):
                reports[report_format] = report_path
                logger.info(f"Found report: {report_path}")
            else:
                logger.warning(f"Expected report not found: {report_path}")
        
        # Create raw output container
        raw_output = RawSASOutput(
            datasets=datasets,
            reports=reports,
            sas_log=sas_result.get('LOG', ''),
            execution_time=datetime.now().isoformat(),
            session_id=self.session_id,
            model_type=model_spec.model_type,
            model_name=model_spec.model_name
        )
        
        return raw_output


class SASModelExecutor:
    """Main executor for SAS models with minimal transformation"""
    
    def __init__(self):
        self.active_sessions: Dict[str, SASSessionCoordinator] = {}
        logger.info("Initialized SAS Model Executor")
    
    def execute_model(self, model_spec: ModelSpecification, 
                     session_id: Optional[str] = None) -> ModelResult:
        """
        Execute SAS model with minimal transformation
        
        Args:
            model_spec: Complete model specification
            session_id: Optional session ID (auto-generated if not provided)
        
        Returns:
            ModelResult with raw outputs and metadata
        """
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get or create session coordinator
        if session_id not in self.active_sessions:
            session_dir = os.path.join(model_spec.session_dir, session_id)
            self.active_sessions[session_id] = SASSessionCoordinator(session_id, session_dir)
        
        session_coordinator = self.active_sessions[session_id]
        
        try:
            # Lock session for execution
            if not session_coordinator.lock_session():
                return ModelResult(
                    success=False,
                    raw_output=None,
                    error_message="Failed to lock SAS session",
                    execution_metadata={'session_id': session_id}
                )
            
            # Execute model
            result = session_coordinator.execute_model(model_spec)
            
            return result
            
        finally:
            # Always unlock session after execution
            session_coordinator.unlock_session()
    
    def execute_multiple_models(self, model_specs: List[ModelSpecification], 
                              session_id: Optional[str] = None) -> List[ModelResult]:
        """
        Execute multiple models in the same session
        
        This is useful when a single analysis requires multiple SAS calls
        but you want to maintain session-level coordination
        """
        
        if not session_id:
            session_id = f"multi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get or create session coordinator
        if session_id not in self.active_sessions:
            # Use first model's session directory
            session_dir = os.path.join(model_specs[0].session_dir, session_id)
            self.active_sessions[session_id] = SASSessionCoordinator(session_id, session_dir)
        
        session_coordinator = self.active_sessions[session_id]
        results = []
        
        try:
            # Lock session for all executions
            if not session_coordinator.lock_session():
                return [ModelResult(
                    success=False,
                    raw_output=None,
                    error_message="Failed to lock SAS session",
                    execution_metadata={'session_id': session_id}
                ) for _ in model_specs]
            
            # Execute each model
            for model_spec in model_specs:
                result = session_coordinator.execute_model(model_spec)
                results.append(result)
                
                # Stop if any model fails
                if not result.success:
                    break
            
            return results
            
        finally:
            # Always unlock session after all executions
            session_coordinator.unlock_session()
    
    def cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].unlock_session()
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    def cleanup_all_sessions(self):
        """Clean up all active sessions"""
        for session_id in list(self.active_sessions.keys()):
            self.cleanup_session(session_id)


# Global executor instance
model_executor = SASModelExecutor() 