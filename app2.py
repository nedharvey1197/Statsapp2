import streamlit as st
import pandas as pd
from analysis.simple_glm import run_simple_analysis
from analysis.repeated_glmm import run_repeated_analysis

st.set_page_config(page_title="Clinical Dashboard", layout="wide")

st.title("📊 Clinical Trial Analysis Dashboard")

# Select analysis type
analysis_type = st.radio("Choose analysis type", ["One-Way ANOVA", "Repeated Measures GLMM"])

# Upload or use example data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
use_example = st.checkbox("Use example dataset", value=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_example:
    if analysis_type == "One-Way ANOVA":
        df = pd.read_csv("data/simple_example.csv")
    else:
        df = pd.read_csv("data/repeated_example.csv")
else:
    st.warning("Upload a file or use example data to proceed.")
    st.stop()

# Run the selected analysis
if analysis_type == "One-Way ANOVA":
    result = run_simple_analysis(df)
else:
    result = run_repeated_analysis(df)

# Display model summary
st.subheader("📄 Model Summary Table")
st.dataframe(result["summary_table"])

# Show diagnostic plots
st.subheader("📈 Diagnostic Plots")
for plot in result["plots"]:
    st.pyplot(plot)

# Show notes
st.subheader("🧠 Notes & Assumptions")
st.markdown(result["notes"])

# Allow PDF download
st.download_button(
    label="📥 Download PDF Report",
    data=result["pdf_bytes"],
    file_name="clinical_analysis_report.pdf",
    mime="application/pdf"
)