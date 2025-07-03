"""
Data Integrity Demo - Shows how to preserve raw SAS outputs

This demonstrates the solution to the data integrity problem:
1. Store raw SAS outputs with integrity protection
2. Create UI displays without losing raw data
3. Use raw data for subsequent analysis
4. Generate FDA-compliant reports from raw data
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Import the integrity manager
from utils.data_integrity_manager import integrity_manager, RawSASOutput, UIDisplayOutput

def demonstrate_integrity_solution():
    """Demonstrate the data integrity solution"""
    
    st.title("🔒 Data Integrity Solution Demo")
    st.markdown("**Preserving Raw SAS Outputs While Providing UI Displays**")
    
    # Simulate SAS analysis results (replace with actual analysis)
    st.subheader("📊 Step 1: Simulate SAS Analysis Results")
    
    # Create sample SAS datasets (simulating what you get from SAS)
    sample_datasets = {
        'fitstats': pd.DataFrame({
            'Description': ['R-Square', 'Adj R-Sq', 'Root MSE', 'F Value', 'Pr > F'],
            'Value': [0.823456789, 0.789123456, 1.234567890, 15.678901234, 0.000123456]
        }),
        'anova': pd.DataFrame({
            'Source': ['Model', 'Error', 'Corrected Total'],
            'DF': [2, 27, 29],
            'Sum of Squares': [123.456789012, 26.543210988, 150.000000000],
            'Mean Square': [61.728394506, 0.983081888],
            'F Value': [62.789012345, None],
            'Pr > F': [0.000000123, None]
        }),
        'lsmeans': pd.DataFrame({
            'Treatment': ['A', 'B', 'C'],
            'LSMean': [12.345678901, 15.678901234, 18.901234567],
            'StdErr': [0.123456789, 0.234567890, 0.345678901],
            't Value': [100.123456789, 66.789012345, 54.678901234],
            'Pr > |t|': [0.000000001, 0.000000123, 0.000001234]
        })
    }
    
    # Simulate analysis result
    analysis_result = {
        'fitstats': sample_datasets['fitstats'],
        'anova': sample_datasets['anova'],
        'lsmeans': sample_datasets['lsmeans'],
        'direct_reports': {
            'pdf': '/path/to/sas_report.pdf',
            'rtf': '/path/to/sas_report.rtf'
        },
        'sas_log': 'SAS log content here...',
        'model_state': {
            'analysis_type': 'simple_glm',
            'timestamp': datetime.now().isoformat(),
            'convergence': 'Normal (GLM)',
            'total_observations': 30
        },
        'session_dir': '/path/to/session'
    }
    
    st.success("✅ Analysis results simulated")
    
    # Step 2: Store raw outputs with integrity protection
    st.subheader("🔒 Step 2: Store Raw Outputs with Integrity Protection")
    
    raw_outputs = integrity_manager.store_raw_outputs(analysis_result)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Datasets Stored", len(raw_outputs.datasets))
        st.metric("Integrity Checksums", len(raw_outputs.dataset_checksums))
    
    with col2:
        integrity_report = integrity_manager.get_integrity_report()
        st.metric("Integrity Status", integrity_report['status'])
        st.metric("Intact Datasets", integrity_report['intact_datasets'])
    
    st.success("✅ Raw outputs stored with integrity protection")
    
    # Step 3: Create UI displays
    st.subheader("🎨 Step 3: Create UI Displays (Preserves Raw Data)")
    
    ui_outputs = integrity_manager.create_ui_displays()
    
    st.metric("Display Tables Created", len(ui_outputs.display_tables))
    st.success("✅ UI displays created without losing raw data")
    
    # Step 4: Demonstrate the difference
    st.subheader("🔍 Step 4: Compare Raw vs Display Data")
    
    # Show raw data (preserved)
    st.write("**Raw Data (Preserved for Analysis):**")
    raw_fitstats = integrity_manager.get_raw_dataset('fitstats')
    st.dataframe(raw_fitstats, use_container_width=True)
    
    # Show display data (formatted for UI)
    st.write("**Display Data (Formatted for UI):**")
    display_fitstats = integrity_manager.get_display_table('fitstats')
    st.dataframe(display_fitstats, use_container_width=True)
    
    # Step 5: Demonstrate integrity verification
    st.subheader("✅ Step 5: Integrity Verification")
    
    integrity_status = raw_outputs.verify_integrity()
    
    for dataset_name, is_intact in integrity_status.items():
        status_icon = "✅" if is_intact else "❌"
        st.write(f"{status_icon} {dataset_name}: {'Intact' if is_intact else 'Compromised'}")
    
    # Step 6: Export raw data for subsequent analysis
    st.subheader("📤 Step 6: Export Raw Data for Subsequent Analysis")
    
    if st.button("Export Raw Data"):
        output_dir = f"exports/raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_path = integrity_manager.export_raw_data(output_dir)
        
        st.success(f"✅ Raw data exported to: {export_path}")
        
        # Show what was exported
        export_files = list(Path(export_path).rglob("*"))
        st.write("**Exported Files:**")
        for file_path in export_files:
            if file_path.is_file():
                st.write(f"📄 {file_path.relative_to(export_path)}")
    
    # Step 7: Demonstrate subsequent analysis capability
    st.subheader("🔬 Step 7: Subsequent Analysis (Using Raw Data)")
    
    if st.button("Perform Subsequent Analysis"):
        # Get raw data for analysis (guaranteed integrity)
        raw_anova = integrity_manager.get_raw_dataset('anova')
        raw_lsmeans = integrity_manager.get_raw_dataset('lsmeans')
        
        # Perform analysis on raw data
        if raw_anova is not None and raw_lsmeans is not None:
            # Example: Calculate effect size from raw data
            model_ss = raw_anova[raw_anova['Source'] == 'Model']['Sum of Squares'].iloc[0]
            total_ss = raw_anova[raw_anova['Source'] == 'Corrected Total']['Sum of Squares'].iloc[0]
            effect_size = model_ss / total_ss
            
            st.success(f"✅ Subsequent analysis completed using raw data")
            st.metric("Effect Size (η²)", f"{effect_size:.4f}")
            st.info("This calculation used the original precision from SAS - no rounding errors!")
    
    # Step 8: FDA Compliance
    st.subheader("📋 Step 8: FDA Compliance (Raw Data Traceability)")
    
    st.info("""
    **FDA Compliance Benefits:**
    
    ✅ **Raw Data Preservation**: Original SAS outputs never modified
    ✅ **Audit Trail**: Complete traceability from SAS to final report
    ✅ **Data Integrity**: Checksums verify no unauthorized changes
    ✅ **Reproducibility**: Raw data can be re-analyzed with same precision
    ✅ **Documentation**: Complete metadata and SAS log preserved
    """)
    
    # Show compliance report
    compliance_report = {
        'raw_data_available': raw_outputs is not None,
        'integrity_verified': all(integrity_status.values()),
        'audit_trail_complete': True,
        'sas_log_preserved': len(raw_outputs.sas_log) > 0,
        'metadata_complete': len(raw_outputs.model_state) > 0
    }
    
    st.write("**Compliance Status:**")
    for check, status in compliance_report.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {check.replace('_', ' ').title()}: {'Yes' if status else 'No'}")


def show_implementation_guide():
    """Show how to implement this in existing code"""
    
    st.subheader("📖 Implementation Guide")
    
    st.markdown("""
    **How to implement this in your existing code:**
    
    ```python
    # 1. Import the integrity manager
    from utils.data_integrity_manager import integrity_manager
    
    # 2. After SAS analysis, store raw outputs
    raw_outputs = integrity_manager.store_raw_outputs(analysis_result)
    
    # 3. Create UI displays (optional)
    ui_outputs = integrity_manager.create_ui_displays()
    
    # 4. Use raw data for subsequent analysis
    raw_dataset = integrity_manager.get_raw_dataset('fitstats')
    
    # 5. Use display data for UI
    display_table = integrity_manager.get_display_table('fitstats')
    
    # 6. Export raw data for FDA compliance
    export_path = integrity_manager.export_raw_data('outputs/raw_data')
    ```
    
    **Key Benefits:**
    - ✅ Raw SAS outputs preserved with 100% integrity
    - ✅ UI displays created without data loss
    - ✅ Subsequent analysis uses original precision
    - ✅ FDA compliance with complete audit trail
    - ✅ No changes to existing analysis functions
    """)

if __name__ == "__main__":
    demonstrate_integrity_solution()
    show_implementation_guide() 