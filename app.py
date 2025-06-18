import streamlit as st
import pandas as pd
from io import BytesIO

from analysis.simple_glm import run_simple_analysis
from analysis.repeated_glmm import run_repeated_analysis

# Page config
st.set_page_config(page_title="Clinical Analysis Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("Clinical Copilot Analytics")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Simple One-Way ANOVA", "Repeated Measures GLMM"])
use_example = st.sidebar.checkbox("Use Example Dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")

# Header
st.title("📊 Clinical Trial Analysis Dashboard")
st.markdown("""
This tool helps biostatisticians and clinical teams evaluate simple and repeated-measure study designs.  
Choose an analysis type and upload your data, or use a preloaded example.
""")

# Run button
if st.button("🔍 Run Analysis"):
    if use_example:
        if analysis_type == "Simple One-Way ANOVA":
            file_path = "data/simple_example.csv"
        else:
            file_path = "data/repeated_example.csv"
        df = pd.read_csv(file_path)
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Please upload a CSV or select 'Use Example Dataset'")
        st.stop()

    if analysis_type == "Simple One-Way ANOVA":
        result = run_simple_analysis(df)
    else:
        result = run_repeated_analysis(df)

    # Show results
    st.subheader("📈 Key Results")
    st.dataframe(result["summary_table"])
    
    st.subheader("📉 Diagnostic Plot(s)")
    for fig in result["plots"]:
        st.pyplot(fig)

    # Download report
    st.download_button(
        label="📥 Download NCSS-style PDF Report",
        data=result["pdf_bytes"],
        file_name="clinical_analysis_report.pdf",
        mime="application/pdf"
    )

    # Show statistical notes
    with st.expander("ℹ️ Statistical Notes & Justifications", expanded=False):
        st.markdown(result["notes"])
