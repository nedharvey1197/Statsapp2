"""
Model Executor Demo - Minimal Transformation Pipeline

This demonstrates the architecture that:
1. Executes SAS models with session-level coordination
2. Collects raw outputs with zero manipulation
3. Provides structured return for UI generation and downstream analysis
4. Is ready for future expansion
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Import the model executor
from utils.sas_model_executor import (
    model_executor, ModelSpecification, ModelResult, RawSASOutput
)

def demonstrate_minimal_transformation():
    """Demonstrate the minimal transformation pipeline"""
    
    st.title("🔧 SAS Model Executor - Minimal Transformation Pipeline")
    st.markdown("**Session-Level Coordination with Zero Data Manipulation**")
    
    # Step 1: Create sample data
    st.subheader("📊 Step 1: Prepare Model Specification")
    
    # Sample data (simulating your clinical trial data)
    sample_data = pd.DataFrame({
        'Treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'] * 3,
        'TumorSize': [12.1, 11.8, 12.3, 15.2, 15.7, 15.1, 18.9, 18.5, 18.7] * 3,
        'Subject': [f'S{i:02d}' for i in range(1, 28)]
    })
    
    st.write("**Sample Data:**")
    st.dataframe(sample_data, use_container_width=True)
    
    # Step 2: Create model specification
    st.subheader("🔧 Step 2: Create Model Specification")
    
    # SAS code for simple GLM
    sas_code = """
    /* Simple GLM Analysis */
    ods output 
        FitStatistics=fitstats 
        OverallANOVA=anova 
        LSMeans=lsmeans 
        LSMeanDiffCL=diffs 
        ParameterEstimates=coeffs
        NObs=nobs
        ClassLevels=classlevels;
    
    proc glm data=input_data plots=diagnostics;
        class Treatment;
        model TumorSize = Treatment / solution;
        lsmeans Treatment / stderr pdiff cl adjust=bon;
        output out=residuals r=resid p=pred;
    run;
    
    proc univariate data=residuals normal;
        var resid;
        ods output TestsForNormality=normtests;
    run;
    """
    
    # Create model specification
    model_spec = ModelSpecification(
        model_type='simple_glm',
        sas_code=sas_code,
        input_data=sample_data,
        output_datasets=['fitstats', 'anova', 'lsmeans', 'diffs', 'coeffs', 'nobs', 'classlevels', 'normtests'],
        report_formats=['pdf', 'rtf'],
        session_dir='demo_sessions',
        model_name='Simple GLM Demo'
    )
    
    st.success("✅ Model specification created")
    
    # Step 3: Execute model with minimal transformation
    st.subheader("🚀 Step 3: Execute Model (Minimal Transformation)")
    
    if st.button("Execute Model"):
        with st.spinner("Executing SAS model..."):
            # Execute model
            result = model_executor.execute_model(model_spec)
            
            if result.success:
                st.success("✅ Model execution successful!")
                
                # Display execution metadata
                st.write("**Execution Metadata:**")
                metadata_df = pd.DataFrame([
                    ['Session ID', result.execution_metadata['session_id']],
                    ['Execution ID', result.execution_metadata['execution_id']],
                    ['Model Type', result.execution_metadata['model_type']],
                    ['Model Name', result.execution_metadata['model_name']],
                    ['Timestamp', result.execution_metadata['timestamp']]
                ], columns=['Field', 'Value'])
                st.dataframe(metadata_df, use_container_width=True)
                
                # Step 4: Show raw outputs (zero manipulation)
                st.subheader("📋 Step 4: Raw Outputs (Zero Manipulation)")
                
                raw_output = result.raw_output
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Datasets Collected", len(raw_output.datasets))
                    st.metric("Reports Generated", len(raw_output.reports))
                
                with col2:
                    st.metric("Session ID", raw_output.session_id)
                    st.metric("Model Type", raw_output.model_type)
                
                # Show raw datasets (exactly as from SAS)
                st.write("**Raw Datasets (Direct from SAS):**")
                for name, dataset in raw_output.datasets.items():
                    if dataset is not None and not dataset.empty:
                        with st.expander(f"📊 {name} ({dataset.shape})"):
                            st.dataframe(dataset, use_container_width=True)
                            st.write(f"**Columns:** {list(dataset.columns)}")
                            st.write(f"**Data Types:** {dict(dataset.dtypes)}")
                    else:
                        st.warning(f"Dataset {name}: Not available")
                
                # Step 5: Integrity verification
                st.subheader("✅ Step 5: Integrity Verification")
                
                integrity_status = raw_output.verify_integrity()
                
                for dataset_name, is_intact in integrity_status.items():
                    status_icon = "✅" if is_intact else "❌"
                    st.write(f"{status_icon} {dataset_name}: {'Intact' if is_intact else 'Compromised'}")
                
                # Step 6: Demonstrate downstream analysis capability
                st.subheader("🔬 Step 6: Downstream Analysis (Using Raw Data)")
                
                if st.button("Perform Downstream Analysis"):
                    # Get raw data for analysis (guaranteed integrity)
                    raw_anova = raw_output.datasets.get('anova')
                    raw_lsmeans = raw_output.datasets.get('lsmeans')
                    
                    if raw_anova is not None and raw_lsmeans is not None:
                        # Perform analysis on raw data (no manipulation)
                        st.write("**Downstream Analysis Results:**")
                        
                        # Example: Calculate effect size from raw ANOVA
                        model_ss_row = raw_anova[raw_anova['Source'] == 'Model']
                        total_ss_row = raw_anova[raw_anova['Source'] == 'Corrected Total']
                        
                        if not model_ss_row.empty and not total_ss_row.empty:
                            model_ss = model_ss_row['Sum of Squares'].iloc[0]
                            total_ss = total_ss_row['Sum of Squares'].iloc[0]
                            effect_size = model_ss / total_ss
                            
                            st.metric("Effect Size (η²)", f"{effect_size:.4f}")
                            st.info("✅ This calculation used raw SAS data - no rounding errors!")
                        
                        # Example: Calculate confidence intervals from raw LSMeans
                        if not raw_lsmeans.empty:
                            st.write("**LSMeans Summary (Raw Data):**")
                            summary_stats = raw_lsmeans.groupby('Treatment')['LSMean'].agg(['mean', 'std']).round(4)
                            st.dataframe(summary_stats, use_container_width=True)
                
                # Step 7: Show future-ready capabilities
                st.subheader("🔮 Step 7: Future-Ready Architecture")
                
                st.info("""
                **This architecture is ready for future expansion:**
                
                ✅ **Provenance Tracking**: Add provenance_data to ModelResult
                ✅ **Compliance Metadata**: Add compliance_metadata to ModelResult  
                ✅ **Multi-Model Execution**: Use execute_multiple_models()
                ✅ **Session Coordination**: Multiple SAS calls in same session
                ✅ **Audit Trail**: Complete execution metadata captured
                ✅ **Regulatory Compliance**: Raw data preserved for FDA submission
                """)
                
                # Show how to add future capabilities
                with st.expander("🔧 Future Capabilities Example"):
                    st.code("""
# Add provenance tracking (future)
result.provenance_data = {
    'source_data_hash': 'sha256_hash',
    'transformation_steps': ['step1', 'step2'],
    'environment_config': {'sas_version': '9.4', 'platform': 'linux'}
}

# Add compliance metadata (future)
result.compliance_metadata = {
    'fda_compliant': True,
    'audit_trail_complete': True,
    'data_integrity_verified': True
}

# Execute multiple models in same session (future)
model_specs = [model1_spec, model2_spec, model3_spec]
results = model_executor.execute_multiple_models(model_specs, session_id='multi_analysis')
                    """)
                
            else:
                st.error(f"❌ Model execution failed: {result.error_message}")
    
    # Step 8: Cleanup demonstration
    st.subheader("🧹 Step 8: Session Cleanup")
    
    if st.button("Cleanup All Sessions"):
        model_executor.cleanup_all_sessions()
        st.success("✅ All sessions cleaned up")


def show_architecture_benefits():
    """Show the benefits of this architecture"""
    
    st.subheader("🎯 Architecture Benefits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Immediate Benefits:**
        
        🔒 **Zero Data Manipulation**: Raw SAS outputs preserved exactly
        🔧 **Session Coordination**: Multiple calls handled at session level
        📊 **Structured Returns**: Ready for UI generation and downstream analysis
        🎯 **Minimal Transformation**: Only what's needed, nothing more
        """)
    
    with col2:
        st.markdown("""
        **🔮 Future-Ready Benefits:**
        
        📋 **Provenance Tracking**: Ready for complete traceability
        📊 **Compliance Metadata**: Ready for FDA/regulatory compliance
        🔄 **Multi-Model Analysis**: Ready for complex analytical workflows
        🎛️ **Session Management**: Ready for advanced session coordination
        """)
    
    st.markdown("""
    **🎯 Key Architecture Principle:**
    
    > "Minimal data flow from SAS to calling function, with session-level coordination for multiple calls"
    
    This ensures:
    - **Data integrity** is preserved at every step
    - **Session management** is handled automatically
    - **Future expansion** is built into the architecture
    - **Regulatory compliance** is achievable without redesign
    """)


if __name__ == "__main__":
    demonstrate_minimal_transformation()
    show_architecture_benefits() 