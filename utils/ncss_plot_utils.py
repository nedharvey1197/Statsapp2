"""
NCSS-Style Plot Utilities

This module provides functions to generate standard diagnostic plots
for NCSS-style reports. All plots are returned as NCSSPlot objects
to maintain consistency with the report structure.

Functions:
    create_residual_plot: Create residual vs fitted values plot
    create_qq_plot: Create normal Q-Q plot for residuals
    create_histogram_plot: Create histogram with normal curve overlay
    create_leverage_plot: Create leverage plot for influential observations
    create_all_diagnostic_plots: Create all standard diagnostic plots
    create_correlation_plot: Create correlation matrix heatmap
    create_means_by_treatment_plot: Create lineplot of TumorSize by Week, colored by Treatment

Example:
    # Create diagnostic plots
    plots = create_all_diagnostic_plots(residuals, fitted_values)
    
    # Add to report section
    for plot in plots:
        diagnostics_section.add_plot(plot)
"""

import io
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

from .ncss_report_structures import NCSSPlot


def create_residual_plot(residuals: np.ndarray, fitted_values: np.ndarray, 
                        title: str = "Residual Plot") -> NCSSPlot:
    """
    Create a standard residual plot (residuals vs fitted values).
    
    This plot helps identify patterns in residuals, heteroscedasticity,
    and model adequacy.
    
    Args:
        residuals: Array of residuals from the model
        fitted_values: Array of fitted/predicted values
        title: Plot title (default: "Residual Plot")
    
    Returns:
        NCSSPlot: Plot object containing the residual plot
        
    Raises:
        ValueError: If arrays have different lengths or contain NaN values
    """
    if len(residuals) != len(fitted_values):
        raise ValueError("Residuals and fitted values must have the same length")
    
    if np.any(np.isnan(residuals)) or np.any(np.isnan(fitted_values)):
        raise ValueError("Arrays cannot contain NaN values")
    
    fig, ax = plt.subplots(figsize=(8, 4.8))  # 20% smaller than (10, 6)
    
    # Create residual plot
    ax.scatter(fitted_values, residuals, alpha=0.6, color='blue', s=30)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add trend line
    z = np.polyfit(fitted_values, residuals, 1)
    p = np.poly1d(z)
    ax.plot(fitted_values, p(fitted_values), "r--", alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Fitted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return NCSSPlot(
        title=title,
        image_bytes=img_buffer.getvalue(),
        description="Residuals plotted against fitted values to check for patterns, heteroscedasticity, and model adequacy",
        plot_type="residual"
    )


def create_qq_plot(residuals: np.ndarray, title: str = "Normal Q-Q Plot") -> NCSSPlot:
    """
    Create a normal Q-Q plot for residuals.
    
    This plot assesses the normality of residuals by comparing
    their quantiles to those of a normal distribution.
    
    Args:
        residuals: Array of residuals from the model
        title: Plot title (default: "Normal Q-Q Plot")
    
    Returns:
        NCSSPlot: Plot object containing the Q-Q plot
        
    Raises:
        ValueError: If residuals contain NaN values
    """
    if np.any(np.isnan(residuals)):
        raise ValueError("Residuals cannot contain NaN values")
    
    fig, ax = plt.subplots(figsize=(6.4, 6.4))  # 20% smaller than (8, 8)
    
    # Create Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return NCSSPlot(
        title=title,
        image_bytes=img_buffer.getvalue(),
        description="Normal Q-Q plot to assess normality of residuals",
        plot_type="qq"
    )


def create_histogram_plot(residuals: np.ndarray, title: str = "Residuals Histogram") -> NCSSPlot:
    """
    Create a histogram of residuals with normal distribution overlay.
    
    This plot provides a visual assessment of the distribution
    of residuals compared to a normal distribution.
    
    Args:
        residuals: Array of residuals from the model
        title: Plot title (default: "Residuals Histogram")
    
    Returns:
        NCSSPlot: Plot object containing the histogram
        
    Raises:
        ValueError: If residuals contain NaN values
    """
    if np.any(np.isnan(residuals)):
        raise ValueError("Residuals cannot contain NaN values")
    
    fig, ax = plt.subplots(figsize=(8, 4.8))  # 20% smaller than (10, 6)
    
    # Create histogram
    ax.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add normal curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal distribution')
    
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return NCSSPlot(
        title=title,
        image_bytes=img_buffer.getvalue(),
        description="Histogram of residuals with normal distribution overlay",
        plot_type="histogram"
    )


def create_leverage_plot(residuals: np.ndarray, leverage: np.ndarray, 
                        title: str = "Leverage Plot") -> NCSSPlot:
    """
    Create a leverage plot (residuals vs leverage).
    
    This plot helps identify influential observations by showing
    the relationship between residuals and leverage values.
    
    Args:
        residuals: Array of residuals from the model
        leverage: Array of leverage values
        title: Plot title (default: "Leverage Plot")
    
    Returns:
        NCSSPlot: Plot object containing the leverage plot
        
    Raises:
        ValueError: If arrays have different lengths or contain NaN values
    """
    if len(residuals) != len(leverage):
        raise ValueError("Residuals and leverage must have the same length")
    
    if np.any(np.isnan(residuals)) or np.any(np.isnan(leverage)):
        raise ValueError("Arrays cannot contain NaN values")
    
    fig, ax = plt.subplots(figsize=(8, 4.8))  # 20% smaller than (10, 6)
    
    # Create leverage plot
    ax.scatter(leverage, residuals, alpha=0.6, color='green', s=30)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add Cook's distance contours (simplified)
    # This is a basic implementation - could be enhanced
    ax.set_xlabel('Leverage', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return NCSSPlot(
        title=title,
        image_bytes=img_buffer.getvalue(),
        description="Leverage plot to identify influential observations",
        plot_type="leverage"
    )


def create_all_diagnostic_plots(residuals: np.ndarray, fitted_values: np.ndarray,
                               leverage: Optional[np.ndarray] = None) -> List[NCSSPlot]:
    """
    Create all standard diagnostic plots for a model.
    
    This function generates the complete set of diagnostic plots
    typically used to assess model adequacy and assumptions.
    
    Args:
        residuals: Array of residuals from the model
        fitted_values: Array of fitted/predicted values
        leverage: Optional array of leverage values for leverage plot
    
    Returns:
        List[NCSSPlot]: List of diagnostic plot objects
        
    Raises:
        ValueError: If input arrays have issues (see individual plot functions)
    """
    plots = []
    
    # Residual plot
    plots.append(create_residual_plot(residuals, fitted_values))
    
    # Q-Q plot
    plots.append(create_qq_plot(residuals))
    
    # Histogram
    plots.append(create_histogram_plot(residuals))
    
    # Leverage plot (if leverage values provided)
    if leverage is not None:
        plots.append(create_leverage_plot(residuals, leverage))
    
    return plots


def create_correlation_plot(data: np.ndarray, variable_names: List[str],
                           title: str = "Correlation Matrix") -> NCSSPlot:
    """
    Create a correlation matrix heatmap.
    
    This plot shows the correlation structure between variables
    in the dataset, useful for identifying multicollinearity.
    
    Args:
        data: 2D array of data (observations x variables)
        variable_names: List of variable names
        title: Plot title (default: "Correlation Matrix")
    
    Returns:
        NCSSPlot: Plot object containing the correlation heatmap
        
    Raises:
        ValueError: If data dimensions don't match variable names or contain NaN values
    """
    if data.shape[1] != len(variable_names):
        raise ValueError("Number of variables must match number of variable names")
    
    if np.any(np.isnan(data)):
        raise ValueError("Data cannot contain NaN values")
    
    fig, ax = plt.subplots(figsize=(8, 6.4))  # 20% smaller than (10, 8)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Add text annotations
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Set labels
    ax.set_xticks(range(len(variable_names)))
    ax.set_yticks(range(len(variable_names)))
    ax.set_xticklabels(variable_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(variable_names, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return NCSSPlot(
        title=title,
        image_bytes=img_buffer.getvalue(),
        description="Correlation matrix heatmap of variables",
        plot_type="correlation"
    )


def create_means_by_treatment_plot(data: 'pd.DataFrame', title: str = "Treatment Means by Week") -> NCSSPlot:
    """
    Create a lineplot of TumorSize by Week, colored by Treatment, as an NCSSPlot.
    Args:
        data: DataFrame with columns 'Week', 'TumorSize', 'Treatment'
        title: Plot title
    Returns:
        NCSSPlot: Plot object containing the means-by-treatment plot
    """
    # Ensure correct types
    plot_data = data.copy()
    plot_data['Week'] = pd.to_numeric(plot_data['Week'], errors='coerce')
    plot_data = plot_data.dropna(subset=['Week'])
    plot_data['Treatment'] = plot_data['Treatment'].astype(str)
    fig, ax = plt.subplots(figsize=(6.4, 4))  # 20% smaller than (8, 5)
    sns.lineplot(data=plot_data, x='Week', y='TumorSize', hue='Treatment', marker='o', ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Week', fontsize=10)
    ax.set_ylabel('Tumor Size', fontsize=10)
    ax.legend(title='Treatment')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return NCSSPlot(
        title=title,
        image_bytes=buf.getvalue(),
        description="Lineplot of Tumor Size by Week and Treatment",
        plot_type="means_by_treatment"
    ) 