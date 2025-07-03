"""
Data Integrity Manager - Preserves Raw SAS Outputs

This module ensures that raw SAS analysis outputs maintain their integrity
while providing UI-friendly displays. It implements a dual-output system:

1. RAW DATA PATH: Preserves original SAS datasets for subsequent analysis
2. UI DISPLAY PATH: Provides formatted displays for user interface

This solves the data integrity loss problem in the transformation pipeline.
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawSASOutput:
    """Container for raw SAS outputs with guaranteed integrity"""
    
    # Original SAS datasets (preserved exactly as extracted)
    datasets: Dict[str, pd.DataFrame]
    
    # Direct ODS reports (PDF/RTF files)
    direct_reports: Dict[str, str]
    
    # SAS log and metadata
    sas_log: str
    model_state: Dict[str, Any]
    
    # Analysis metadata
    analysis_type: str
    timestamp: str
    session_dir: str
    
    # Integrity checksums
    dataset_checksums: Dict[str, str]
    
    def __post_init__(self):
        """Calculate checksums for integrity verification"""
        if not hasattr(self, 'dataset_checksums'):
            self.dataset_checksums = {}
            for name, dataset in self.datasets.items():
                if dataset is not None and not dataset.empty:
                    # Create a hash of the dataset content
                    content_hash = self._calculate_dataset_hash(dataset)
                    self.dataset_checksums[name] = content_hash
    
    def _calculate_dataset_hash(self, dataset: pd.DataFrame) -> str:
        """Calculate a hash of dataset content for integrity verification"""
        import hashlib
        
        # Convert to string representation for hashing
        content_str = dataset.to_string()
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def verify_integrity(self) -> Dict[str, bool]:
        """Verify that datasets haven't been modified"""
        integrity_status = {}
        
        for name, dataset in self.datasets.items():
            if dataset is not None and not dataset.empty:
                current_hash = self._calculate_dataset_hash(dataset)
                original_hash = self.dataset_checksums.get(name, '')
                integrity_status[name] = current_hash == original_hash
            else:
                integrity_status[name] = True  # Empty datasets are always "intact"
        
        return integrity_status
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a dataset with integrity verification"""
        if name not in self.datasets:
            logger.warning(f"Dataset '{name}' not found in raw outputs")
            return None
        
        dataset = self.datasets[name]
        if dataset is None or dataset.empty:
            return dataset
        
        # Verify integrity
        current_hash = self._calculate_dataset_hash(dataset)
        original_hash = self.dataset_checksums.get(name, '')
        
        if current_hash != original_hash:
            logger.error(f"Dataset '{name}' integrity check failed!")
            logger.error(f"Original hash: {original_hash}")
            logger.error(f"Current hash: {current_hash}")
            raise ValueError(f"Dataset '{name}' has been modified - integrity compromised")
        
        return dataset.copy()  # Return copy to prevent modification
    
    def save_raw_outputs(self, output_dir: str) -> str:
        """Save raw outputs to disk for later use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets as CSV files
        datasets_dir = output_path / "raw_datasets"
        datasets_dir.mkdir(exist_ok=True)
        
        for name, dataset in self.datasets.items():
            if dataset is not None and not dataset.empty:
                dataset_path = datasets_dir / f"{name}.csv"
                dataset.to_csv(dataset_path, index=False)
                logger.info(f"Saved raw dataset: {dataset_path}")
        
        # Save metadata
        metadata = {
            'analysis_type': self.analysis_type,
            'timestamp': self.timestamp,
            'session_dir': self.session_dir,
            'model_state': self.model_state,
            'dataset_checksums': self.dataset_checksums,
            'direct_reports': self.direct_reports
        }
        
        metadata_path = output_path / "raw_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save SAS log
        log_path = output_path / "sas_log.txt"
        with open(log_path, 'w') as f:
            f.write(self.sas_log)
        
        logger.info(f"Saved raw outputs to: {output_path}")
        return str(output_path)


@dataclass
class UIDisplayOutput:
    """Container for UI-friendly display outputs"""
    
    # Formatted tables for display
    display_tables: Dict[str, pd.DataFrame]
    
    # Formatted text for UI
    display_text: Dict[str, str]
    
    # Display metadata
    display_metadata: Dict[str, Any]
    
    # Note: These are display-only and should NOT be used for analysis


class DataIntegrityManager:
    """Manages data integrity while providing UI displays"""
    
    def __init__(self):
        self.raw_outputs: Optional[RawSASOutput] = None
        self.ui_outputs: Optional[UIDisplayOutput] = None
    
    def store_raw_outputs(self, analysis_result: Dict[str, Any]) -> RawSASOutput:
        """Store raw SAS outputs with integrity protection"""
        
        # Extract raw datasets (preserve exactly as from SAS)
        raw_datasets = {}
        for key, value in analysis_result.items():
            if isinstance(value, pd.DataFrame):
                raw_datasets[key] = value.copy()  # Make copy to prevent modification
        
        # Create raw output container
        self.raw_outputs = RawSASOutput(
            datasets=raw_datasets,
            direct_reports=analysis_result.get('direct_reports', {}),
            sas_log=analysis_result.get('sas_log', ''),
            model_state=analysis_result.get('model_state', {}),
            analysis_type=analysis_result.get('analysis_type', 'unknown'),
            timestamp=analysis_result.get('timestamp', datetime.now().isoformat()),
            session_dir=analysis_result.get('session_dir', '')
        )
        
        logger.info(f"Stored raw outputs with {len(raw_datasets)} datasets")
        return self.raw_outputs
    
    def create_ui_displays(self, preserve_original: bool = True) -> UIDisplayOutput:
        """Create UI-friendly displays from raw outputs"""
        
        if self.raw_outputs is None:
            raise ValueError("No raw outputs available. Call store_raw_outputs() first.")
        
        # Create display tables (formatted for UI)
        display_tables = {}
        for name, dataset in self.raw_outputs.datasets.items():
            if dataset is not None and not dataset.empty:
                # Create display copy with formatting
                display_df = self._format_for_display(dataset)
                display_tables[name] = display_df
        
        # Create display text
        display_text = self._create_display_text()
        
        # Create display metadata
        display_metadata = self._create_display_metadata()
        
        self.ui_outputs = UIDisplayOutput(
            display_tables=display_tables,
            display_text=display_text,
            display_metadata=display_metadata
        )
        
        logger.info(f"Created UI displays for {len(display_tables)} tables")
        return self.ui_outputs
    
    def _format_for_display(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Format dataset for UI display (loses precision but improves readability)"""
        
        display_df = dataset.copy()
        
        # Format numeric columns for display
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                # Format p-values
                if any(keyword in col.lower() for keyword in ['pval', 'prob', 'adjp']):
                    display_df[col] = display_df[col].apply(self._format_pvalue)
                # Format other floats
                else:
                    display_df[col] = display_df[col].apply(self._format_float)
            # Keep other data types as-is
        
        return display_df
    
    def _format_pvalue(self, value: float) -> str:
        """Format p-value for display"""
        try:
            if pd.isna(value):
                return "NA"
            if value < 0.001:
                return "<0.001"
            else:
                return f"{value:.3f}"
        except:
            return str(value)
    
    def _format_float(self, value: float) -> str:
        """Format float for display"""
        try:
            if pd.isna(value):
                return "NA"
            return f"{value:.2f}"
        except:
            return str(value)
    
    def _create_display_text(self) -> Dict[str, str]:
        """Create formatted text for UI display"""
        if not self.raw_outputs:
            return {}
        
        text = {}
        
        # Create summary text
        model_state = self.raw_outputs.model_state
        text['summary'] = f"""
        Analysis Type: {self.raw_outputs.analysis_type}
        Timestamp: {self.raw_outputs.timestamp}
        Datasets: {len(self.raw_outputs.datasets)}
        """
        
        return text
    
    def _create_display_metadata(self) -> Dict[str, Any]:
        """Create display metadata"""
        if not self.raw_outputs:
            return {}
        
        return {
            'analysis_type': self.raw_outputs.analysis_type,
            'timestamp': self.raw_outputs.timestamp,
            'dataset_count': len(self.raw_outputs.datasets),
            'integrity_status': self.raw_outputs.verify_integrity()
        }
    
    def get_raw_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get raw dataset with integrity verification"""
        if self.raw_outputs is None:
            return None
        return self.raw_outputs.get_dataset(name)
    
    def get_display_table(self, name: str) -> Optional[pd.DataFrame]:
        """Get UI-formatted table"""
        if self.ui_outputs is None:
            return None
        return self.ui_outputs.display_tables.get(name)
    
    def export_raw_data(self, output_dir: str) -> str:
        """Export raw data for subsequent analysis"""
        if self.raw_outputs is None:
            raise ValueError("No raw outputs to export")
        
        return self.raw_outputs.save_raw_outputs(output_dir)
    
    def get_integrity_report(self) -> Dict[str, Any]:
        """Get integrity status report"""
        if self.raw_outputs is None:
            return {'status': 'no_data'}
        
        integrity_status = self.raw_outputs.verify_integrity()
        all_intact = all(integrity_status.values())
        
        return {
            'status': 'intact' if all_intact else 'compromised',
            'dataset_integrity': integrity_status,
            'total_datasets': len(integrity_status),
            'intact_datasets': sum(integrity_status.values()),
            'compromised_datasets': sum(1 for v in integrity_status.values() if not v)
        }


# Global instance for easy access
integrity_manager = DataIntegrityManager() 