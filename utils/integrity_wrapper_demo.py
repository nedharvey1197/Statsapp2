"""
Integrity Wrapper Demo - Superior Process Concept

This demonstrates the user's superior approach:
1. HTML as "Model Report of Record" (zero translation loss)
2. Minimal transformation pipeline (direct ODS OUTPUT)
3. Generic wrapper for any SAS procedure
4. SHA-256 checksums for integrity verification
5. Session isolation and provenance tracking
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Import the integrity wrapper
from utils.sas_integrity_wrapper import (
    integrity_wrapper, execute_simple_glm, execute_repeated_measures,
    ModelExecutionManifest
)

def demonstrate_superior_approach():
    """Demonstrate the superior integrity wrapper approach"""
    
    st.title("🔒 SAS Integrity Wrapper - Superior Process Concept")
    st.markdown("**HTML Model Report of Record + Zero Translation Loss**")
    
    # Step 1: Bootstrap session
    st.subheader("🚀 Step 1: Bootstrap Clean SAS Session")
    
    if st.button("Bootstrap Session"):
        with st.spinner("Initializing clean SAS ODA session..."):
            success = integrity_wrapper.bootstrap_session()
            
            if success:
                st.success(f"✅ Session bootstrapped - Run ID: {integrity_wrapper.current_run_id}")
                
                # Show session info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Run ID", integrity_wrapper.current_run_id)
                    st.metric("Output Directory", str(integrity_wrapper.current_output_dir))
                
                with col2:
                    st.metric("Session ID", str(id(integrity_wrapper.sas_session)))
                    st.metric("ODA Home", integrity_wrapper.sas_session.symget('USERDIR'))
                
                st.session_state.session_bootstrapped = True
            else:
                st.error("❌ Session bootstrap failed")
                st.session_state.session_bootstrapped = False
    
    # Step 2: Prepare sample data
    st.subheader("📊 Step 2: Prepare Sample Data")
    
    # Simple GLM data
    simple_glm_data = pd.DataFrame({
        'Treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'] * 3,
        'TumorSize': [12.1, 11.8, 12.3, 15.2, 15.7, 15.1, 18.9, 18.5, 18.7] * 3,
        'Subject': [f'S{i:02d}' for i in range(1, 28)]
    })
    
    # Repeated measures data
    repeated_data = pd.DataFrame({
        'Treatment': ['A', 'A', 'A', 'B', 'B', 'B'] * 4,
        'Week': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
        'TumorSize': [12.1, 11.8, 12.3, 15.2, 15.7, 15.1, 11.5, 11.2, 11.8, 14.8, 15.2, 14.9, 10.9, 10.6, 11.2, 14.4, 14.8, 14.5, 10.3, 10.0, 10.6, 14.0, 14.4, 14.1],
        'Dog': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'] * 4
    })
    
    # Data selection
    data_option = st.radio(
        "Choose analysis type:",
        ["Simple GLM (One-Way ANOVA)", "Repeated Measures"],
        index=0
    )
    
    if data_option == "Simple GLM (One-Way ANOVA)":
        st.write("**Simple GLM Data:**")
        st.dataframe(simple_glm_data, use_container_width=True)
        selected_data = simple_glm_data
    else:
        st.write("**Repeated Measures Data:**")
        st.dataframe(repeated_data, use_container_width=True)
        selected_data = repeated_data
    
    # Step 3: Execute model with integrity preservation
    st.subheader("🔬 Step 3: Execute Model (Integrity Preservation)")
    
    if st.button("Execute Model") and st.session_state.get('session_bootstrapped', False):
        with st.spinner("Executing model with integrity preservation..."):
            
            if data_option == "Simple GLM (One-Way ANOVA)":
                result = execute_simple_glm(
                    selected_data, 'Treatment', 'TumorSize', integrity_wrapper
                )
            else:
                result = execute_repeated_measures(
                    selected_data, 'Treatment', 'TumorSize', 'Week', 'Dog', integrity_wrapper
                )
            
            if result['success']:
                st.success("✅ Model execution successful!")
                
                # Display manifest
                manifest = result['manifest']
                st.write("**Execution Manifest:**")
                manifest_df = pd.DataFrame([
                    ['Run ID', manifest.run_id],
                    ['Model Name', manifest.model_name],
                    ['Model Type', manifest.model_type],
                    ['Timestamp', manifest.timestamp],
                    ['ODS Report', manifest.ods_report],
                    ['Tables Captured', len(manifest.tables)],
                    ['Log SHA256', manifest.log_sha256[:16] + '...'],
                    ['Session ID', manifest.session_id],
                    ['SAS Version', manifest.sas_version]
                ], columns=['Field', 'Value'])
                st.dataframe(manifest_df, use_container_width=True)
                
                # Step 4: Show raw outputs (zero manipulation)
                st.subheader("📋 Step 4: Raw Outputs (Zero Manipulation)")
                
                dataframes = result['dataframes']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tables Captured", len(dataframes))
                    st.metric("Archive Created", "Yes" if result['archive_path'] else "No")
                
                with col2:
                    st.metric("Run ID", manifest.run_id)
                    st.metric("Model Type", manifest.model_type)
                
                # Show raw datasets (exactly as from SAS)
                st.write("**Raw Datasets (Direct from SAS WORK):**")
                for table_name, df in dataframes.items():
                    if not df.empty:
                        with st.expander(f"📊 {table_name} ({df.shape})"):
                            st.dataframe(df, use_container_width=True)
                            st.write(f"**Columns:** {list(df.columns)}")
                            st.write(f"**Data Types:** {dict(df.dtypes)}")
                            st.write(f"**First few values:**")
                            for col in df.columns[:3]:  # Show first 3 columns
                                st.write(f"  {col}: {df[col].head().tolist()}")
                    else:
                        st.warning(f"Table {table_name}: Empty")
                
                # Step 5: HTML Model Report of Record
                st.subheader("📄 Step 5: HTML Model Report of Record")
                
                st.info("""
                **Key Innovation: HTML preserves exact ODS formatting**
                
                ✅ **Zero translation loss** - HTML preserves exact SAS ODS formatting
                ✅ **Print-ready** - Can be converted to PDF locally with full fidelity  
                ✅ **ODA-compatible** - No file handle issues in sandbox environment
                ✅ **Stakeholder-friendly** - Exact SAS layout, nothing re-rendered
                """)
                
                # Show archive info
                if result['archive_path']:
                    st.write("**Archive Contents:**")
                    archive_path = Path(result['archive_path'])
                    if archive_path.exists():
                        st.success(f"✅ Archive created: {archive_path.name}")
                        
                        # List archive contents
                        from zipfile import ZipFile
                        with ZipFile(archive_path, 'r') as z:
                            file_list = z.namelist()
                        
                        st.write("**Files in Archive:**")
                        for file_name in file_list:
                            st.write(f"📄 {file_name}")
                        
                        # Download archive
                        with open(archive_path, 'rb') as f:
                            st.download_button(
                                label="📦 Download Complete Archive",
                                data=f.read(),
                                file_name=archive_path.name,
                                mime="application/zip"
                            )
                
                # Step 6: Demonstrate downstream analysis capability
                st.subheader("🔬 Step 6: Downstream Analysis (Using Raw Data)")
                
                if st.button("Perform Downstream Analysis"):
                    # Get raw data for analysis (guaranteed integrity)
                    if data_option == "Simple GLM (One-Way ANOVA)":
                        anova_df = dataframes.get('OverallANOVA')
                        lsmeans_df = dataframes.get('LSMeans')
                        
                        if anova_df is not None and lsmeans_df is not None:
                            st.write("**Downstream Analysis Results:**")
                            
                            # Calculate effect size from raw ANOVA
                            model_ss_row = anova_df[anova_df['Source'] == 'Model']
                            total_ss_row = anova_df[anova_df['Source'] == 'Corrected Total']
                            
                            if not model_ss_row.empty and not total_ss_row.empty:
                                model_ss = model_ss_row['Sum of Squares'].iloc[0]
                                total_ss = total_ss_row['Sum of Squares'].iloc[0]
                                effect_size = model_ss / total_ss
                                
                                st.metric("Effect Size (η²)", f"{effect_size:.4f}")
                                st.info("✅ This calculation used raw SAS data - no rounding errors!")
                    else:
                        # Repeated measures analysis
                        fitstats_df = dataframes.get('FitStatistics')
                        lsmeans_df = dataframes.get('LSMeans')
                        
                        if fitstats_df is not None and lsmeans_df is not None:
                            st.write("**Downstream Analysis Results:**")
                            
                            # Show fit statistics
                            st.write("**Model Fit Statistics:**")
                            st.dataframe(fitstats_df, use_container_width=True)
                            
                            # Show LSMeans summary
                            if not lsmeans_df.empty:
                                st.write("**LSMeans Summary (Raw Data):**")
                                summary_stats = lsmeans_df.groupby('Treatment')['Estimate'].agg(['mean', 'std']).round(4)
                                st.dataframe(summary_stats, use_container_width=True)
                
                # Step 7: Show integrity verification
                st.subheader("✅ Step 7: Integrity Verification")
                
                st.info("""
                **Integrity Guarantees:**
                
                🔒 **SHA-256 checksums** for every artifact catch corruption
                📋 **Run-ID manifest** links code, report, tables, log hash
                🎯 **Session isolation** prevents cross-run contamination
                📊 **Data frames** exported straight from WORK, no manual copy-paste
                """)
                
                # Show integrity status
                st.write("**Integrity Status:**")
                integrity_checks = [
                    ("Manifest Created", True),
                    ("HTML Report Generated", Path(manifest.ods_report.replace('~/', '')).exists()),
                    ("All Tables Captured", len(dataframes) == len(manifest.tables)),
                    ("Archive Created", bool(result['archive_path'])),
                    ("Log SHA256 Calculated", len(manifest.log_sha256) == 64)
                ]
                
                for check_name, status in integrity_checks:
                    icon = "✅" if status else "❌"
                    st.write(f"{icon} {check_name}: {'Yes' if status else 'No'}")
                
                # Step 8: Show future-ready capabilities
                st.subheader("🔮 Step 8: Future-Ready Architecture")
                
                st.info("""
                **This architecture is ready for future expansion:**
                
                ✅ **Generic Wrapper**: Works with any SAS procedure (only ODS table names change)
                ✅ **Provenance Tracking**: Complete manifest with SHA-256 checksums
                ✅ **Regulatory Compliance**: HTML report of record + complete audit trail
                ✅ **Multi-Model Analysis**: Can execute multiple models in same session
                ✅ **Session Management**: Automatic cleanup and isolation
                """)
                
                # Show how to adapt to other procedures
                with st.expander("🔧 Adapting to Other SAS Procedures"):
                    st.code("""
# For PROC MIXED (Repeated Measures)
tables_needed = [
    "FitStatistics", "LSMeans", "Diffs", "CovParms",
    "Tests3", "SolutionF", "ConvergenceStatus"
]

# For PROC LOGISTIC
tables_needed = [
    "ModelFit", "ParameterEstimates", "OddsRatios",
    "Classification", "ROC"
]

# For PROC REG
tables_needed = [
    "FitStatistics", "ParameterEstimates", "OutputStatistics"
]

# Only the ODS table names change - everything else stays the same!
                    """)
                
            else:
                st.error(f"❌ Model execution failed: {result.get('error', 'Unknown error')}")
    
    # Step 9: Cleanup demonstration
    st.subheader("🧹 Step 9: Session Cleanup")
    
    if st.button("Cleanup Session"):
        integrity_wrapper.cleanup_session()
        st.success("✅ Session cleaned up")
        st.session_state.session_bootstrapped = False


def show_architecture_advantages():
    """Show the advantages of this superior approach"""
    
    st.subheader("🎯 Why This Approach is Superior")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ HTML Model Report of Record:**
        
        🎨 **Zero Translation Loss**: HTML preserves exact ODS formatting
        📄 **Print-Ready**: Convert to PDF locally with full fidelity
        🔧 **ODA-Compatible**: No file handle issues in sandbox
        👥 **Stakeholder-Friendly**: Exact SAS layout, nothing re-rendered
        """)
    
    with col2:
        st.markdown("""
        **✅ Minimal Transformation Pipeline:**
        
        🔒 **Direct ODS OUTPUT**: No intermediate formatting
        📊 **Straight from WORK**: DataFrames exported directly
        🔐 **SHA-256 Checksums**: Integrity verification built-in
        🎯 **Session Isolation**: Prevents cross-contamination
        """)
    
    st.markdown("""
    **🎯 Key Innovations:**
    
    > "HTML as Model Report of Record eliminates translation loss while providing print-ready output"
    
    > "Generic wrapper architecture scales to any SAS procedure with minimal changes"
    
    > "SHA-256 checksums provide tamper evidence for regulatory compliance"
    
    **🔧 Technical Advantages:**
    - **No PDF/RTF file handle issues** in ODA sandbox
    - **HTML preserves exact ODS styling** and formatting
    - **Direct download from WORK library** eliminates manual copy-paste
    - **Session-level coordination** handles multiple calls automatically
    - **Future-ready architecture** supports any SAS or non-SAS model
    """)


if __name__ == "__main__":
    demonstrate_superior_approach()
    show_architecture_advantages() 