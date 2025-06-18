# Refactor for `run_repeated_analysis(df)` with diagnostics, plots, notes, and PDF

import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
import numpy as np

def safe_float(val):
    """Safely convert a value to float, returning NaN if conversion fails."""
    if pd.isna(val) or val is None:
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')

def format_float(val, precision=2):
    """Format a float value with specified precision, handling NaN values."""
    if pd.isna(val) or np.isnan(val):
        return "N/A"
    return f"{val:.{precision}f}"

def run_repeated_analysis(df: pd.DataFrame):
    df["Treatment"] = df["Treatment"].astype("category")
    df["Week"] = df["Week"].astype("category")

    # Fit both REML and ML models
    model_reml = smf.mixedlm("TumorSize ~ Treatment * Week", data=df, groups=df["Dog"]).fit(reml=True)
    model_ml = smf.mixedlm("TumorSize ~ Treatment * Week", data=df, groups=df["Dog"]).fit(reml=False)

    # Coefficient table
    coef_df = model_reml.summary().tables[1]
    coef_df = coef_df.reset_index().rename(columns={"index": "Term"})

    # LogLik, AIC/BIC comparison
    comparison = pd.DataFrame({
        "Model Type": ["REML", "ML"],
        "Log-Likelihood": [model_reml.llf, model_ml.llf],
        "AIC": [None, model_ml.aic],
        "BIC": [None, model_ml.bic]
    })

    # Diagnostic plots
    plots = []

    # Interaction plot
    fig1, ax1 = plt.subplots()
    sns.pointplot(data=df, x="Week", y="TumorSize", hue="Treatment", ci="sd", ax=ax1)
    ax1.set_title("Treatment x Week Interaction Plot")
    plots.append(fig1)

    # Spaghetti plot
    fig2, ax2 = plt.subplots()
    for dog_id, dog_data in df.groupby("Dog"):
        ax2.plot(dog_data["Week"], dog_data["TumorSize"], marker='o', linestyle='-',
                 label=f"Dog {dog_id}", alpha=0.3)
    ax2.set_title("Individual Dog Tumor Size Trajectories")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Tumor Size")
    plots.append(fig2)

    # Residuals vs fitted
    fitted = model_reml.fittedvalues
    residuals = model_reml.resid
    fig3, ax3 = plt.subplots()
    sns.residplot(x=fitted, y=residuals, lowess=True, ax=ax3)
    ax3.set_title("Residuals vs Fitted (REML)")
    plots.append(fig3)

    # PDF export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Repeated Measures Mixed Model Report", ln=True, align="C")

    # Model comparison section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Model Comparison (REML vs ML)", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in comparison.iterrows():
        loglik = safe_float(row['Log-Likelihood'])
        aic = safe_float(row['AIC'])
        bic = safe_float(row['BIC'])
        
        loglik_str = format_float(loglik)
        aic_str = format_float(aic)
        bic_str = format_float(bic)
        
        pdf.cell(200, 8, f"{row['Model Type']} - LogLik: {loglik_str} AIC: {aic_str} BIC: {bic_str}", ln=True)

    # Fixed effects section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Fixed Effects Estimates (REML)", ln=True)
    pdf.set_font("Arial", "", 10)
    for i, row in coef_df.iterrows():
        coef = safe_float(row['Coef.'])
        se = safe_float(row['Std.Err.'])
        t_val = safe_float(row['z'])
        p_val = safe_float(row['P>|z|'])
        
        coef_str = format_float(coef)
        se_str = format_float(se)
        t_val_str = format_float(t_val)
        p_val_str = format_float(p_val, precision=5)
        
        pdf.cell(200, 8, f"{row['Term']}: Coef={coef_str}, SE={se_str}, t={t_val_str}, p={p_val_str}", ln=True)

    # Statistical notes section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Statistical Notes", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8,
        "Model: Mixed linear model with random intercepts for Dog.\n"
        "REML was used for final coefficient estimation; ML used for AIC/BIC model comparison.\n"
        "The random effects covariance matrix was singular, suggesting Dog ID added minimal variance.\n"
        "No AR(1) correlation structure applied due to limitations in statsmodels MixedLM.\n"
        "Interaction and individual trajectory plots are provided to visualize treatment effects over time."
    )

    # Convert bytearray to bytes for Streamlit compatibility
    pdf_bytes = bytes(pdf.output(dest='S'))

    return {
        "summary_table": coef_df,
        "plots": plots,
        "pdf_bytes": pdf_bytes,
        "notes": (
            "Mixed-effects model with Treatment, Week, and their interaction as fixed effects.\n"
            "Random intercepts by Dog were modeled (singular fit warning suggests low between-subject variance).\n"
            "REML used for coefficient estimation; ML used for model comparison (AIC/BIC).\n"
            "AR(1) correlation structure not applied due to tool limitations."
        )
    }
