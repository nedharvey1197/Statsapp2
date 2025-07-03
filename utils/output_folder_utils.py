"""
Output Folder Utilities

This module provides functions to automatically create analysis-specific subfolders
in the outputs directory for organized file downloads and exports.

Functions:
    create_analysis_output_folder: Create a timestamped subfolder for analysis outputs
    get_analysis_output_path: Get the full path for an analysis output file
    cleanup_old_outputs: Clean up old output folders (optional)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def create_analysis_output_folder(analysis_type: str, base_output_dir: str = "outputs") -> str:
    """
    Create a timestamped subfolder in the outputs directory for analysis-specific files.
    
    Args:
        analysis_type: Type of analysis (e.g., 'simple_glm', 'repeated_measures', 'trial_design')
        base_output_dir: Base output directory (default: "outputs")
        
    Returns:
        str: Path to the created subfolder
        
    Example:
        folder_path = create_analysis_output_folder('simple_glm')
        # Returns: "outputs/simple_glm_20250101_143022"
    """
    # Ensure base output directory exists
    base_path = Path(base_output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{analysis_type}_{timestamp}"
    
    # Create the analysis-specific subfolder
    analysis_folder = base_path / folder_name
    analysis_folder.mkdir(exist_ok=True)
    
    print(f"Created analysis output folder: {analysis_folder}")
    return str(analysis_folder)


def get_analysis_output_path(analysis_type: str, filename: str, base_output_dir: str = "outputs") -> str:
    """
    Get the full path for an analysis output file, creating the subfolder if needed.
    
    Args:
        analysis_type: Type of analysis (e.g., 'simple_glm', 'repeated_measures', 'trial_design')
        filename: Name of the output file
        base_output_dir: Base output directory (default: "outputs")
        
    Returns:
        str: Full path to the output file
        
    Example:
        file_path = get_analysis_output_path('simple_glm', 'report.pdf')
        # Returns: "outputs/simple_glm_20250101_143022/report.pdf"
    """
    # Create the analysis subfolder
    folder_path = create_analysis_output_folder(analysis_type, base_output_dir)
    
    # Return the full file path
    return os.path.join(folder_path, filename)


def get_or_create_analysis_folder(analysis_type: str, session_folder: Optional[str] = None, base_output_dir: str = "outputs") -> str:
    """
    Get an existing analysis folder or create a new one.
    If session_folder is provided, use that; otherwise create a new timestamped folder.
    
    Args:
        analysis_type: Type of analysis (e.g., 'simple_glm', 'repeated_measures', 'trial_design')
        session_folder: Optional existing session folder path
        base_output_dir: Base output directory (default: "outputs")
        
    Returns:
        str: Path to the analysis folder
    """
    if session_folder and os.path.exists(session_folder):
        # Use existing session folder
        return session_folder
    
    # Create new timestamped folder
    return create_analysis_output_folder(analysis_type, base_output_dir)


def cleanup_old_outputs(base_output_dir: str = "outputs", days_to_keep: int = 30) -> None:
    """
    Clean up old output folders older than specified days.
    
    Args:
        base_output_dir: Base output directory (default: "outputs")
        days_to_keep: Number of days to keep output folders (default: 30)
        
    Note: This function is optional and can be called periodically to manage disk space.
    """
    import shutil
    from datetime import timedelta
    
    base_path = Path(base_output_dir)
    if not base_path.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                # Try to parse folder name for timestamp
                folder_name = folder.name
                if '_' in folder_name:
                    # Extract timestamp part (last part after underscore)
                    timestamp_part = folder_name.split('_')[-2:]  # Get last two parts
                    if len(timestamp_part) == 2:
                        timestamp_str = f"{timestamp_part[0]}_{timestamp_part[1]}"
                        folder_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        if folder_date < cutoff_date:
                            shutil.rmtree(folder)
                            print(f"Removed old output folder: {folder}")
            except (ValueError, IndexError):
                # Skip folders that don't match the expected naming pattern
                continue


def list_analysis_outputs(analysis_type: Optional[str] = None, base_output_dir: str = "outputs") -> list:
    """
    List all analysis output folders, optionally filtered by analysis type.
    
    Args:
        analysis_type: Optional analysis type filter
        base_output_dir: Base output directory (default: "outputs")
        
    Returns:
        list: List of output folder paths
    """
    base_path = Path(base_output_dir)
    if not base_path.exists():
        return []
    
    output_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir():
            if analysis_type is None or folder.name.startswith(analysis_type):
                output_folders.append(str(folder))
    
    return sorted(output_folders, reverse=True)  # Most recent first 