import streamlit as st
import webbrowser
import os
from datetime import datetime

# Set up page configuration
st.set_page_config(
    page_title="Clinical Trial Analysis Hub",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX (extracted from copilotapp.py)
st.markdown("""
<style>
    /* Increase font sizes */
    .main .block-container {
        font-size: 16px;
    }
    
    .stMarkdown {
        font-size: 16px;
    }
    
    .stRadio > label {
        font-size: 16px !important;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stSelectbox > label {
        font-size: 16px !important;
    }
    
    .stCheckbox > label {
        font-size: 16px !important;
    }
    
    .stFileUploader > label {
        font-size: 16px !important;
    }
    
    /* Custom styling for tool boxes */
    .tool-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .tool-box:hover {
        background-color: #f8f9fa;
        border-color: #1f77b4;
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .tool-box.simple-glm {
        border-left: 6px solid #2ca02c;
    }
    
    .tool-box.repeated-measures {
        border-left: 6px solid #ff7f0e;
    }
    
    .tool-box.trial-design {
        border-left: 6px solid #d62728;
    }
    
    .tool-box h3 {
        color: #333333;
        margin-bottom: 15px;
        font-size: 20px;
    }
    
    .tool-box p {
        color: #555555;
        margin-bottom: 10px;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .tool-box ul {
        color: #555555;
        font-size: 14px;
        margin-left: 20px;
    }
    
    .tool-box li {
        margin-bottom: 6px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #2ca02c;
    }
    
    .status-offline {
        background-color: #d62728;
    }
    
    /* Better spacing */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Better metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.current_tool = "Data Analysis"

# App configuration
APP_CONFIG = {
    "Simple GLM Direct ODS": {
        "port": 8502,
        "description": "One-way ANOVA using PROC GLM with direct ODS output",
        "features": ["Professional SAS reports", "NCSS-style UI", "PDF/RTF export", "Comprehensive logging"],
        "status": "online"
    },
    "Repeated Measures": {
        "port": 8503,
        "description": "Repeated measures analysis using PROC MIXED with AR(1) covariance",
        "features": ["Mixed model analysis", "Time series data", "Subject-level plots", "Variance components"],
        "status": "online"
    },
    "Trial Design Wizard": {
        "port": 8504,
        "description": "Clinical trial design and model selection wizard",
        "features": ["Study design configuration", "Sample size planning", "Model recommendation", "Scenario management"],
        "status": "online"
    }
}

def check_app_status(port):
    """Check if an app is running on the specified port"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def launch_app(app_name):
    """Launch the specified app"""
    if app_name in APP_CONFIG:
        port = APP_CONFIG[app_name]["port"]
        url = f"http://localhost:{port}"
        
        # Check if app is already running
        if check_app_status(port):
            st.success(f"✅ {app_name} is already running!")
            webbrowser.open(url)
        else:
            st.info(f"🚀 Launching {app_name}...")
            st.info(f"📱 The app will open in a new browser tab at: {url}")
            st.info("💡 If the app doesn't open automatically, please click the link above.")
            
            # Try to open in browser
            try:
                webbrowser.open(url)
            except:
                st.warning("Could not automatically open browser. Please manually navigate to the URL above.")

def main():
    # Header
    st.title("🧪 Clinical Trial Analysis Hub")
    st.markdown("**Unified platform for clinical trial data analysis and study design**")
    
    # Sidebar navigation
    st.sidebar.title("🔧 Available Tools")
    
    # Tool selection
    selected_tool = st.sidebar.selectbox(
        "Choose Tool:",
        list(APP_CONFIG.keys()),
        index=list(APP_CONFIG.keys()).index(st.session_state.current_tool)
    )
    
    st.session_state.current_tool = selected_tool
    
    # Main content area
    if selected_tool == "Simple GLM Direct ODS":
        display_simple_glm_section()
    elif selected_tool == "Repeated Measures":
        display_repeated_measures_section()
    elif selected_tool == "Trial Design Wizard":
        display_trial_design_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        Clinical Trial Analysis Hub | Built with Streamlit and SAS | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

def display_simple_glm_section():
    """Display Simple GLM Direct ODS section"""
    st.markdown("""
    <div class="tool-box simple-glm">
        <h3>🟢 Simple GLM Direct ODS Analysis</h3>
        <p><strong>Purpose:</strong> One-way ANOVA analysis using SAS PROC GLM with direct ODS output for professional reporting.</p>
        <p><strong>Best for:</strong> Cross-sectional studies, treatment group comparisons, regulatory submissions</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Professional SAS-generated PDF/RTF reports</li>
            <li>NCSS-style user interface for data exploration</li>
            <li>Comprehensive logging with daily folder organization</li>
            <li>Direct ODS output (no translation errors)</li>
            <li>Bonferroni-adjusted pairwise comparisons</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Status and launch
    col1, col2 = st.columns([1, 2])
    with col1:
        status = APP_CONFIG["Simple GLM Direct ODS"]["status"]
        status_class = "status-online" if status == "online" else "status-offline"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span class="status-indicator {status_class}"></span>
            <span>Status: {status.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Launch Simple GLM Direct ODS", use_container_width=True):
            launch_app("Simple GLM Direct ODS")
    
    # Additional information
    with st.expander("📋 Analysis Details", expanded=False):
        st.markdown("""
        **Statistical Model:** PROC GLM (General Linear Model)
        
        **Analysis Type:** One-way Analysis of Variance (ANOVA)
        
        **Data Requirements:**
        - Response variable (continuous)
        - Treatment group variable (categorical)
        - Balanced or unbalanced designs supported
        
        **Outputs:**
        - ANOVA table with F-tests
        - Least squares means with standard errors
        - Pairwise comparisons with Bonferroni adjustment
        - Diagnostic plots (residuals, normality)
        - Professional PDF/RTF reports
        - Comprehensive analysis logs
        """)

def display_repeated_measures_section():
    """Display Repeated Measures section"""
    st.markdown("""
    <div class="tool-box repeated-measures">
        <h3>🟡 Repeated Measures Analysis</h3>
        <p><strong>Purpose:</strong> Mixed model analysis for repeated measures designs using SAS PROC MIXED with AR(1) covariance structure.</p>
        <p><strong>Best for:</strong> Longitudinal studies, crossover designs, time series data</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>PROC MIXED with REML estimation</li>
            <li>AR(1) covariance structure for repeated measures</li>
            <li>Treatment × Time interaction analysis</li>
            <li>Subject-level trajectory plots</li>
            <li>Variance component estimation</li>
            <li>Professional SAS reports with direct ODS output</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Status and launch
    col1, col2 = st.columns([1, 2])
    with col1:
        status = APP_CONFIG["Repeated Measures"]["status"]
        status_class = "status-online" if status == "online" else "status-offline"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span class="status-indicator {status_class}"></span>
            <span>Status: {status.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Launch Repeated Measures Analysis", use_container_width=True):
            launch_app("Repeated Measures")
    
    # Additional information
    with st.expander("📋 Analysis Details", expanded=False):
        st.markdown("""
        **Statistical Model:** PROC MIXED (Mixed Model)
        
        **Analysis Type:** Repeated Measures Mixed Model
        
        **Data Requirements:**
        - Response variable (continuous)
        - Treatment group variable (categorical)
        - Time variable (numeric)
        - Subject ID variable (categorical)
        - Long format data (one row per observation)
        
        **Outputs:**
        - Type 3 tests of fixed effects
        - Least squares means with standard errors
        - Variance component estimates
        - Pairwise comparisons with Bonferroni adjustment
        - Subject-level trajectory plots
        - Professional PDF/RTF reports
        - Comprehensive analysis logs
        """)

def display_trial_design_section():
    """Display Trial Design Wizard section"""
    st.markdown("""
    <div class="tool-box trial-design">
        <h3>🔴 Trial Design Wizard</h3>
        <p><strong>Purpose:</strong> Clinical trial design and model selection wizard for planning studies and choosing appropriate statistical methods.</p>
        <p><strong>Best for:</strong> Study planning, protocol development, statistical methodology selection</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Study design configuration (outcome type, repeated measures, etc.)</li>
            <li>Sample size planning and justification</li>
            <li>Model recommendation engine</li>
            <li>Population and endpoint selection</li>
            <li>Scenario management and comparison</li>
            <li>Regulatory compliance guidance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Status and launch
    col1, col2 = st.columns([1, 2])
    with col1:
        status = APP_CONFIG["Trial Design Wizard"]["status"]
        status_class = "status-online" if status == "online" else "status-offline"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span class="status-indicator {status_class}"></span>
            <span>Status: {status.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Launch Trial Design Wizard", use_container_width=True):
            launch_app("Trial Design Wizard")
    
    # Additional information
    with st.expander("📋 Wizard Details", expanded=False):
        st.markdown("""
        **Design Options:**
        - Outcome types: Continuous, Binary, Count, Time-to-Event
        - Study designs: Cross-sectional, Repeated measures, Crossover
        - Subject grouping: Single site, Multi-site, Clustered
        - Covariates: Baseline measures, Demographics, Confounders
        
        **Planning Features:**
        - Sample size calculation
        - Power analysis
        - Effect size estimation
        - Population stratification
        - Endpoint definition
        - Statistical methodology selection
        
        **Outputs:**
        - Recommended statistical model
        - Sample size justification
        - Study design summary
        - Scenario comparisons
        - Regulatory guidance
        """)

if __name__ == "__main__":
    main() 