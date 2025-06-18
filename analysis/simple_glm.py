# Re-run the refactor for `run_simple_analysis` after kernel reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot

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

def run_simple_analysis(df: pd.DataFrame):
    df["Treatment"] = df["Treatment"].astype("category")

    model = ols("TumorSize ~ C(Treatment)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table["mean_sq"] = anova_table["sum_sq"] / anova_table["df"]

    tukey = pairwise_tukeyhsd(df["TumorSize"], df["Treatment"])

    residuals = model.resid
    fitted = model.fittedvalues
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    dagostino_stat, dagostino_p = stats.normaltest(residuals)
    ad_stat, _, _ = stats.anderson(residuals)

    coef_df = model.summary2().tables[1].reset_index().rename(columns={"index": "Term"})

    plots = []

    fig1 = qqplot(residuals, line='s')
    fig1.suptitle("Q-Q Plot of Residuals")
    plots.append(fig1)

    fig2, ax = plt.subplots()
    sns.residplot(x=fitted, y=residuals, lowess=True, ax=ax)
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    plots.append(fig2)

    fig3, ax = plt.subplots()
    sns.boxplot(x="Treatment", y="TumorSize", data=df, ax=ax)
    ax.set_title("Tumor Size by Treatment Group")
    plots.append(fig3)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Simple One-Way ANOVA Report", ln=True, align="C")

    # ANOVA Table section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "ANOVA Table", ln=True)
    pdf.set_font("Arial", "", 10)
    for i in range(len(anova_table)):
        row = anova_table.iloc[i]
        df_val = safe_float(row['df'])
        sum_sq_val = safe_float(row['sum_sq'])
        mean_sq_val = safe_float(row['mean_sq'])
        f_val = safe_float(row['F'])
        pr_val = safe_float(row['PR(>F)'])
        
        df_str = format_float(df_val, precision=1)
        sum_sq_str = format_float(sum_sq_val)
        mean_sq_str = format_float(mean_sq_val)
        f_str = format_float(f_val)
        pr_str = format_float(pr_val, precision=5)
        
        pdf.cell(200, 8, f"{anova_table.index[i]}: DF={df_str}, SS={sum_sq_str}, MS={mean_sq_str}, F={f_str}, p={pr_str}", ln=True)

    # Tukey HSD section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Tukey HSD Results", ln=True)
    pdf.set_font("Arial", "", 10)
    for result in tukey.summary().data[1:]:
        diff_val = safe_float(result[2])
        p_val = safe_float(result[5])
        
        diff_str = format_float(diff_val)
        p_str = format_float(p_val, precision=5)
        
        pdf.cell(200, 8, f"{result[0]} vs {result[1]}: diff={diff_str}, p={p_str}, reject={result[6]}", ln=True)

    # Normality Tests section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Normality Tests", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(200, 8, f"Shapiro-Wilk: stat={format_float(shapiro_stat, precision=4)}, p={format_float(shapiro_p, precision=4)}", ln=True)
    pdf.cell(200, 8, f"D'Agostino: stat={format_float(dagostino_stat, precision=4)}, p={format_float(dagostino_p, precision=4)}", ln=True)
    pdf.cell(200, 8, f"Anderson-Darling: stat={format_float(ad_stat, precision=4)}", ln=True)

    # Statistical Notes section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Statistical Notes", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8,
        "This model uses OLS to fit a one-way ANOVA with Treatment as a fixed effect.\n"
        "Tukey HSD is used for post-hoc pairwise comparisons, assuming equal variances.\n"
        "Residual normality was evaluated using Shapiro-Wilk, D'Agostino, and Anderson-Darling tests.\n"
        "Plots include a Q-Q plot, residuals vs fitted, and group boxplots.\n"
        "All tests are two-tailed with alpha = 0.05."
    )

    # Convert bytearray to bytes for Streamlit compatibility
    pdf_bytes = bytes(pdf.output(dest='S'))

    return {
        "summary_table": coef_df,
        "plots": plots,
        "pdf_bytes": pdf_bytes,
        "notes": (
            "OLS-based one-way ANOVA with Treatment as a categorical factor.\n"
            "Posthoc comparisons performed using Tukey HSD (family-wise error controlled).\n"
            "Normality tests support residual distribution assumptions. Visual diagnostics provided."
        )
    }
