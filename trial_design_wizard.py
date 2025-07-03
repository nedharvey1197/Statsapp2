import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Set up page configuration
st.set_page_config(
    page_title="Clinical Trial Design Wizard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX
st.markdown("""
<style>
    /* Increase font sizes */
    .main .block-container {
        font-size: 16px;
    }
    
    .stMarkdown {
        font-size: 16px;
    }
    
    .stButton > button {
        background-color: #d62728;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #b71c1c;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Design configuration boxes */
    .design-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .design-box:hover {
        background-color: #f8f9fa;
        border-color: #d62728;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .design-box h3 {
        color: #333333;
        margin-bottom: 15px;
        font-size: 18px;
        border-bottom: 2px solid #d62728;
        padding-bottom: 8px;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9800;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .recommendation-box h3 {
        color: #e65100;
        margin-bottom: 15px;
        font-size: 20px;
    }
    
    /* Scenario box */
    .scenario-box {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #9c27b0;
        margin: 10px 0;
    }
    
    .scenario-box h4 {
        color: #6a1b9a;
        margin-bottom: 10px;
    }
    
    /* Better spacing */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #d62728;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'wizard_initialized' not in st.session_state:
    st.session_state.wizard_initialized = True
    st.session_state.scenarios = []
    st.session_state.current_scenario = None

def select_model(trial_params):
    """
    Model selection logic extracted from deprecated analysis.model_selector
    """
    outcome_type = trial_params.get('outcome_type', 'continuous')
    repeated_measures = trial_params.get('repeated_measures', False)
    grouped_subjects = trial_params.get('grouped_subjects', False)
    covariates = trial_params.get('covariates', False)
    
    # Model selection logic
    if outcome_type == 'continuous':
        if repeated_measures:
            if grouped_subjects:
                model = "PROC MIXED with nested random effects"
                description = "Mixed model for repeated measures with nested subject grouping (e.g., sites within regions)"
            else:
                model = "PROC MIXED with AR(1) covariance"
                description = "Mixed model for repeated measures with autoregressive covariance structure"
        else:
            if grouped_subjects:
                model = "PROC GLM with nested ANOVA"
                description = "General linear model with nested analysis of variance for grouped subjects"
            else:
                model = "PROC GLM (One-way ANOVA)"
                description = "General linear model for simple between-group comparisons"
    
    elif outcome_type == 'binary':
        if repeated_measures:
            model = "PROC GLIMMIX with logistic link"
            description = "Generalized linear mixed model for binary repeated measures"
        else:
            model = "PROC LOGISTIC"
            description = "Logistic regression for binary outcomes"
    
    elif outcome_type == 'count':
        if repeated_measures:
            model = "PROC GLIMMIX with Poisson/Negative Binomial"
            description = "Generalized linear mixed model for count data with repeated measures"
        else:
            model = "PROC GENMOD with Poisson/Negative Binomial"
            description = "Generalized linear model for count data"
    
    elif outcome_type == 'time-to-event':
        model = "PROC PHREG"
        description = "Cox proportional hazards regression for survival analysis"
    
    else:
        model = "PROC GLM"
        description = "General linear model (default recommendation)"
    
    return model, description

def calculate_sample_size(trial_params):
    """
    Basic sample size calculation (placeholder for more sophisticated logic)
    """
    outcome_type = trial_params.get('outcome_type', 'continuous')
    sample_size = trial_params.get('sample_size', 100)
    
    # Basic adjustments based on design complexity
    if trial_params.get('repeated_measures', False):
        # Reduce sample size for repeated measures (more power per subject)
        adjusted_size = int(sample_size * 0.7)
    elif trial_params.get('grouped_subjects', False):
        # Increase sample size for grouped subjects (account for clustering)
        adjusted_size = int(sample_size * 1.3)
    else:
        adjusted_size = sample_size
    
    # Adjust for covariates
    if trial_params.get('covariates', False):
        adjusted_size = int(adjusted_size * 1.1)
    
    return max(adjusted_size, 30)  # Minimum sample size

def main():
    # Header
    st.title("🧪 Clinical Trial Design Wizard")
    st.markdown("**Plan your clinical trial and select the optimal statistical methodology**")
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Design Steps")
    step = st.sidebar.radio(
        "Current Step:",
        ["1. Study Design", "2. Sample Size", "3. Model Selection", "4. Scenarios", "5. Summary"]
    )
    
    # Main content based on step
    if step == "1. Study Design":
        display_study_design_step()
    elif step == "2. Sample Size":
        display_sample_size_step()
    elif step == "3. Model Selection":
        display_model_selection_step()
    elif step == "4. Scenarios":
        display_scenarios_step()
    elif step == "5. Summary":
        display_summary_step()

def display_study_design_step():
    """Display study design configuration"""
    st.markdown("""
    <div class="design-box">
        <h3>📋 Study Design Configuration</h3>
        <p>Configure the basic structure of your clinical trial to determine the appropriate statistical methodology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Outcome Type
    st.subheader("🎯 Primary Outcome")
    outcome_type = st.selectbox(
        "Outcome Type:",
        ["Continuous", "Binary", "Count", "Time-to-Event"],
        help="The type of primary endpoint you're measuring"
    )
    
    # Study Design Features
    st.subheader("📊 Study Design Features")
    col1, col2 = st.columns(2)
    
    with col1:
        repeated_measures = st.checkbox(
            "Repeated Measures Over Time",
            help="Subjects are measured at multiple time points"
        )
        
        grouped_subjects = st.checkbox(
            "Grouped Subjects (e.g., Sites)",
            help="Subjects are nested within groups (sites, centers, etc.)"
        )
    
    with col2:
        covariates = st.checkbox(
            "Include Covariates",
            help="Account for baseline measures, demographics, or other factors"
        )
        
        crossover = st.checkbox(
            "Crossover Design",
            help="Subjects receive multiple treatments in sequence",
            disabled=not repeated_measures
        )
    
    # Population and Endpoints
    st.subheader("👥 Population and Endpoints")
    col1, col2 = st.columns(2)
    
    with col1:
        population = st.multiselect(
            "Target Population:",
            ["Adults (18-65)", "Elderly (65+)", "Pediatric (<18)", "Geriatric (75+)", "Special Populations"],
            default=["Adults (18-65)"],
            help="Select all applicable population groups"
        )
    
    with col2:
        endpoints = st.multiselect(
            "Primary Endpoints:",
            ["Tumor Size", "Response Rate", "Survival", "Quality of Life", "Safety", "Pharmacokinetics"],
            help="Select primary endpoints for your study"
        )
    
    # Store in session state
    st.session_state.trial_params = {
        'outcome_type': outcome_type.lower(),
        'repeated_measures': repeated_measures,
        'grouped_subjects': grouped_subjects,
        'covariates': covariates,
        'crossover': crossover,
        'population': population,
        'endpoints': endpoints
    }
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next: Sample Size →", type="primary"):
            st.sidebar.radio("Current Step:", ["1. Study Design", "2. Sample Size", "3. Model Selection", "4. Scenarios", "5. Summary"], index=1)
            st.rerun()

def display_sample_size_step():
    """Display sample size planning"""
    st.markdown("""
    <div class="design-box">
        <h3>📏 Sample Size Planning</h3>
        <p>Determine the appropriate sample size for your study based on statistical power requirements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample size inputs
    st.subheader("📊 Sample Size Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider(
            "Target Sample Size:",
            min_value=30,
            max_value=1000,
            value=100,
            step=10,
            help="Total number of subjects to enroll"
        )
        
        power = st.slider(
            "Statistical Power:",
            min_value=0.80,
            max_value=0.95,
            value=0.90,
            step=0.05,
            help="Probability of detecting a true effect"
        )
    
    with col2:
        alpha = st.slider(
            "Significance Level (α):",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Type I error rate"
        )
        
        effect_size = st.selectbox(
            "Expected Effect Size:",
            ["Small (0.1)", "Medium (0.3)", "Large (0.5)", "Very Large (0.8)"],
            index=1,
            help="Expected magnitude of treatment effect"
        )
    
    # Calculate adjusted sample size
    trial_params = st.session_state.get('trial_params', {})
    trial_params.update({
        'sample_size': sample_size,
        'power': power,
        'alpha': alpha,
        'effect_size': effect_size
    })
    
    adjusted_size = calculate_sample_size(trial_params)
    
    # Display results
    st.subheader("📈 Sample Size Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Sample Size", sample_size)
    
    with col2:
        st.metric("Adjusted Sample Size", adjusted_size)
    
    with col3:
        difference = adjusted_size - sample_size
        st.metric("Adjustment", f"{difference:+d}", delta_color="normal" if difference == 0 else ("inverse" if difference > 0 else "normal"))
    
    # Justification
    with st.expander("📋 Sample Size Justification", expanded=True):
        st.markdown(f"""
        **Design Adjustments:**
        - **Base sample size:** {sample_size} subjects
        - **Repeated measures adjustment:** {'Applied (70% of base)' if trial_params.get('repeated_measures') else 'Not applicable'}
        - **Grouped subjects adjustment:** {'Applied (130% of base)' if trial_params.get('grouped_subjects') else 'Not applicable'}
        - **Covariates adjustment:** {'Applied (110% of base)' if trial_params.get('covariates') else 'Not applicable'}
        
        **Final recommendation:** {adjusted_size} subjects
        
        **Power analysis:** {power*100:.0f}% power to detect {effect_size} effect size at α = {alpha}
        """)
    
    # Store updated params
    st.session_state.trial_params = trial_params
    st.session_state.adjusted_sample_size = adjusted_size
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next: Model Selection →", type="primary"):
            st.sidebar.radio("Current Step:", ["1. Study Design", "2. Sample Size", "3. Model Selection", "4. Scenarios", "5. Summary"], index=2)
            st.rerun()

def display_model_selection_step():
    """Display model selection and recommendation"""
    st.markdown("""
    <div class="design-box">
        <h3>🔬 Statistical Model Selection</h3>
        <p>Based on your study design, we recommend the optimal statistical methodology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    trial_params = st.session_state.get('trial_params', {})
    
    # Get model recommendation
    recommended_model, model_description = select_model(trial_params)
    
    # Display recommendation
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>🎯 Recommended Statistical Model</h3>
        <p><strong>Model:</strong> {recommended_model}</p>
        <p><strong>Description:</strong> {model_description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model details
    st.subheader("📊 Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Features:**")
        if trial_params.get('repeated_measures'):
            st.markdown("✅ Handles repeated measures")
            st.markdown("✅ Accounts for within-subject correlation")
        else:
            st.markdown("✅ Between-subject comparisons")
            st.markdown("✅ Independent observations")
        
        if trial_params.get('grouped_subjects'):
            st.markdown("✅ Nested random effects")
            st.markdown("✅ Accounts for clustering")
        
        if trial_params.get('covariates'):
            st.markdown("✅ Covariate adjustment")
            st.markdown("✅ Improved precision")
    
    with col2:
        st.markdown("**SAS Implementation:**")
        if "MIXED" in recommended_model:
            st.markdown("```sas\nPROC MIXED data=study_data;\n  class treatment subject time;\n  model outcome = treatment time treatment*time;\n  repeated time / subject=subject type=AR(1);\nrun;\n```")
        elif "GLIMMIX" in recommended_model:
            st.markdown("```sas\nPROC GLIMMIX data=study_data;\n  class treatment subject time;\n  model outcome(event='1') = treatment time;\n  random time / subject=subject;\nrun;\n```")
        else:
            st.markdown("```sas\nPROC GLM data=study_data;\n  class treatment;\n  model outcome = treatment;\n  lsmeans treatment / diff cl adjust=bon;\nrun;\n```")
    
    # Alternative models
    st.subheader("🔄 Alternative Models")
    with st.expander("View Alternative Approaches", expanded=False):
        alternatives = []
        
        if trial_params.get('outcome_type') == 'continuous':
            if trial_params.get('repeated_measures'):
                alternatives.append({
                    "model": "PROC GLM with repeated measures",
                    "description": "Classical repeated measures ANOVA",
                    "pros": "Simple, widely understood",
                    "cons": "Assumes compound symmetry, less flexible"
                })
                alternatives.append({
                    "model": "PROC GENMOD with GEE",
                    "description": "Generalized estimating equations",
                    "pros": "Robust to covariance misspecification",
                    "cons": "Less efficient than mixed models"
                })
            else:
                alternatives.append({
                    "model": "PROC ANOVA",
                    "description": "Classical analysis of variance",
                    "pros": "Simple, parametric",
                    "cons": "Assumes normality, equal variances"
                })
        
        for alt in alternatives:
            st.markdown(f"""
            **{alt['model']}**
            - {alt['description']}
            - ✅ {alt['pros']}
            - ⚠️ {alt['cons']}
            """)
    
    # Store recommendation
    st.session_state.recommended_model = recommended_model
    st.session_state.model_description = model_description
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next: Scenarios →", type="primary"):
            st.sidebar.radio("Current Step:", ["1. Study Design", "2. Sample Size", "3. Model Selection", "4. Scenarios", "5. Summary"], index=3)
            st.rerun()

def display_scenarios_step():
    """Display scenario management"""
    st.markdown("""
    <div class="design-box">
        <h3>📋 Scenario Management</h3>
        <p>Create and compare different study design scenarios to optimize your trial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current scenario
    trial_params = st.session_state.get('trial_params', {})
    recommended_model = st.session_state.get('recommended_model', 'Unknown')
    
    st.subheader("💾 Save Current Scenario")
    scenario_name = st.text_input(
        "Scenario Name:",
        value=f"Scenario_{len(st.session_state.scenarios) + 1}",
        help="Give this scenario a descriptive name"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("💾 Save Scenario", type="primary"):
            scenario = {
                'name': scenario_name,
                'params': trial_params.copy(),
                'model': recommended_model,
                'sample_size': st.session_state.get('adjusted_sample_size', trial_params.get('sample_size', 100)),
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.scenarios.append(scenario)
            st.success(f"✅ Scenario '{scenario_name}' saved!")
            st.rerun()
    
    # Display saved scenarios
    if st.session_state.scenarios:
        st.subheader("📚 Saved Scenarios")
        
        for i, scenario in enumerate(st.session_state.scenarios):
            with st.expander(f"📋 {scenario['name']} (Saved: {scenario['timestamp'][:19]})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Design Parameters:**")
                    st.write(f"Outcome: {scenario['params'].get('outcome_type', 'Unknown')}")
                    st.write(f"Repeated Measures: {scenario['params'].get('repeated_measures', False)}")
                    st.write(f"Grouped Subjects: {scenario['params'].get('grouped_subjects', False)}")
                    st.write(f"Sample Size: {scenario['sample_size']}")
                
                with col2:
                    st.markdown("**Model:**")
                    st.write(scenario['model'])
                    
                    # Action buttons
                    if st.button(f"🔄 Load {scenario['name']}", key=f"load_{i}"):
                        st.session_state.trial_params = scenario['params'].copy()
                        st.session_state.recommended_model = scenario['model']
                        st.session_state.adjusted_sample_size = scenario['sample_size']
                        st.success(f"✅ Loaded scenario '{scenario['name']}'")
                        st.rerun()
                    
                    if st.button(f"🗑️ Delete {scenario['name']}", key=f"delete_{i}"):
                        st.session_state.scenarios.pop(i)
                        st.success(f"✅ Deleted scenario '{scenario['name']}'")
                        st.rerun()
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next: Summary →", type="primary"):
            st.sidebar.radio("Current Step:", ["1. Study Design", "2. Sample Size", "3. Model Selection", "4. Scenarios", "5. Summary"], index=4)
            st.rerun()

def display_summary_step():
    """Display final summary and export options"""
    st.markdown("""
    <div class="design-box">
        <h3>📄 Trial Design Summary</h3>
        <p>Review your complete trial design and export the specifications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    trial_params = st.session_state.get('trial_params', {})
    recommended_model = st.session_state.get('recommended_model', 'Unknown')
    model_description = st.session_state.get('model_description', 'No description available')
    adjusted_sample_size = st.session_state.get('adjusted_sample_size', trial_params.get('sample_size', 100))
    
    # Summary display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Study Design")
        st.markdown(f"""
        **Primary Outcome:** {trial_params.get('outcome_type', 'Unknown').title()}
        
        **Study Features:**
        - Repeated Measures: {'✅ Yes' if trial_params.get('repeated_measures') else '❌ No'}
        - Grouped Subjects: {'✅ Yes' if trial_params.get('grouped_subjects') else '❌ No'}
        - Covariates: {'✅ Yes' if trial_params.get('covariates') else '❌ No'}
        - Crossover: {'✅ Yes' if trial_params.get('crossover') else '❌ No'}
        
        **Target Population:** {', '.join(trial_params.get('population', ['Not specified']))}
        
        **Primary Endpoints:** {', '.join(trial_params.get('endpoints', ['Not specified']))}
        """)
    
    with col2:
        st.subheader("📊 Statistical Design")
        st.markdown(f"""
        **Recommended Model:** {recommended_model}
        
        **Sample Size:** {adjusted_sample_size} subjects
        
        **Power Analysis:**
        - Power: {trial_params.get('power', 0.9)*100:.0f}%
        - Alpha: {trial_params.get('alpha', 0.05)}
        - Effect Size: {trial_params.get('effect_size', 'Medium')}
        
        **Model Description:** {model_description}
        """)
    
    # Export options
    st.subheader("📤 Export Design")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        design_summary = {
            'trial_params': trial_params,
            'recommended_model': recommended_model,
            'model_description': model_description,
            'adjusted_sample_size': adjusted_sample_size,
            'scenarios': st.session_state.scenarios,
            'export_date': datetime.now().isoformat()
        }
        
        json_str = json.dumps(design_summary, indent=2)
        st.download_button(
            label="📄 Export as JSON",
            data=json_str,
            file_name=f"trial_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as text report
        report_text = f"""
CLINICAL TRIAL DESIGN SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STUDY DESIGN
Primary Outcome: {trial_params.get('outcome_type', 'Unknown').title()}
Repeated Measures: {trial_params.get('repeated_measures', False)}
Grouped Subjects: {trial_params.get('grouped_subjects', False)}
Covariates: {trial_params.get('covariates', False)}
Crossover: {trial_params.get('crossover', False)}

Target Population: {', '.join(trial_params.get('population', ['Not specified']))}
Primary Endpoints: {', '.join(trial_params.get('endpoints', ['Not specified']))}

STATISTICAL DESIGN
Recommended Model: {recommended_model}
Sample Size: {adjusted_sample_size} subjects
Power: {trial_params.get('power', 0.9)*100:.0f}%
Alpha: {trial_params.get('alpha', 0.05)}
Effect Size: {trial_params.get('effect_size', 'Medium')}

Model Description: {model_description}

SCENARIOS
Total scenarios saved: {len(st.session_state.scenarios)}
        """
        
        st.download_button(
            label="📄 Export as Text",
            data=report_text,
            file_name=f"trial_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col3:
        # Reset design
        if st.button("🔄 Reset Design", type="secondary"):
            for key in ['trial_params', 'recommended_model', 'model_description', 'adjusted_sample_size', 'scenarios']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Design reset! Start over from Step 1.")
            st.rerun()
    
    # Next steps
    st.subheader("🚀 Next Steps")
    st.markdown("""
    1. **Review the design** with your team and stakeholders
    2. **Validate assumptions** about effect sizes and variability
    3. **Consider regulatory requirements** for your target population
    4. **Plan data collection** based on the recommended sample size
    5. **Prepare statistical analysis plan** using the recommended model
    6. **Launch analysis** using our Clinical Trial Analysis Hub
    """)
    
    # Link back to master app
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Ready to analyze your data?</p>
        <a href="http://localhost:8500" target="_blank">
            <button style="background-color: #1f77b4; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;">
                🧪 Return to Analysis Hub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 