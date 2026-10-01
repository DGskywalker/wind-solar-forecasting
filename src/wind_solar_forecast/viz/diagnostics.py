"""Calibration diagnostics and probabilistic forecast uncertainty fan charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wind_solar_forecast.models.calibration import compute_pit_values, compute_reliability_table


def create_reliability_diagram(
    predicted_quantiles: np.ndarray,
    nominal_quantiles: list[float],
    y_true: np.ndarray,
    model_name: str = "Ensemble"
) -> go.Figure:
    """Creates an interactive Plotly reliability diagram comparing nominal vs empirical quantile coverage."""
    rel_df = compute_reliability_table(predicted_quantiles, nominal_quantiles, y_true)
    ece = rel_df["ece_mean"].iloc[0]

    fig = go.Figure()

    # Ideal 1:1 line
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 1.0],
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="Perfect Calibration"
    ))

    # Empirical line
    fig.add_trace(go.Scatter(
        x=rel_df["nominal_quantile"],
        y=rel_df["empirical_coverage"],
        mode="lines+markers",
        marker=dict(size=8, color="#1f77b4"),
        line=dict(color="#1f77b4", width=2.5),
        name=f"{model_name} (ECE: {ece:.3f})"
    ))

    fig.update_layout(
        title=f"Reliability Diagram: {model_name}",
        xaxis_title="Nominal Quantile Level",
        yaxis_title="Empirical Coverage Rate",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
        template="plotly_white",
        width=600,
        height=500
    )
    return fig


def create_pit_histogram(
    predicted_quantiles: np.ndarray,
    nominal_quantiles: list[float],
    y_true: np.ndarray,
    model_name: str = "Ensemble",
    n_bins: int = 10
) -> go.Figure:
    """Creates a Probability Integral Transform (PIT) histogram with horizontal Uniform reference line."""
    pit_vals = compute_pit_values(predicted_quantiles, nominal_quantiles, y_true)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pit_vals,
        nbinsx=n_bins,
        histnorm="probability density",
        marker_color="#2ca02c",
        opacity=0.75,
        name="Empirical PIT"
    ))

    # Uniform reference line (density = 1.0)
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[1.0, 1.0],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Ideal Uniform(0,1)"
    ))

    fig.update_layout(
        title=f"Probability Integral Transform (PIT): {model_name}",
        xaxis_title="PIT Value",
        yaxis_title="Probability Density",
        xaxis=dict(range=[0.0, 1.0]),
        template="plotly_white",
        width=600,
        height=500
    )
    return fig


def create_forecast_fan_chart(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: list[float],
    title: str = "Probabilistic Generation Forecast with Prediction Intervals"
) -> go.Figure:
    """Creates multi-quantile fan chart with shaded confidence bands (90%, 50%) and actual values."""
    fig = go.Figure()

    # Outer band (q05 to q95) -> 90% interval
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=q_preds[:, -1],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="q95"
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=q_preds[:, 0],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.15)",
        name="90% Prediction Interval"
    ))

    # Inner band (q25 to q75) -> 50% interquartile range
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=q_preds[:, 4],  # q75
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="q75"
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=q_preds[:, 2],  # q25
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.35)",
        name="50% Interquartile Range"
    ))

    # Median forecast (q50)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=q_preds[:, 3],  # q50
        mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        name="Median Forecast (q50)"
    ))

    # Observed actuals
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_true,
        mode="lines+markers",
        marker=dict(size=4, color="black"),
        line=dict(color="black", width=1.5),
        name="Actual Realization"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Delivery Time (UTC)",
        yaxis_title="Capacity Factor",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def save_diagnostic_figures(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: list[float],
    output_dir: str = "reports/figures"
) -> None:
    """Saves static publication-quality PNG diagnostic figures to reports/figures/."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Reliability plot
    rel_df = compute_reliability_table(q_preds, quantiles, y_true)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(rel_df["nominal_quantile"], rel_df["empirical_coverage"], "o-", color="#1f77b4", label=f"ECE: {rel_df['ece_mean'].iloc[0]:.3f}")
    plt.xlabel("Nominal Quantile Level")
    plt.ylabel("Empirical Coverage Rate")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "reliability_diagram.png", dpi=300)
    plt.close()

    # 2. PIT histogram
    pit = compute_pit_values(q_preds, quantiles, y_true)
    plt.figure(figsize=(6, 5))
    plt.hist(pit, bins=10, density=True, color="#2ca02c", alpha=0.7, edgecolor="k")
    plt.axhline(1.0, color="r", linestyle="--", label="Uniform(0,1)")
    plt.xlabel("PIT Value")
    plt.ylabel("Density")
    plt.title("Probability Integral Transform (PIT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "pit_histogram.png", dpi=300)
    plt.close()
