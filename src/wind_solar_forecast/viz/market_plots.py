"""Market visualization: surprise vs delta-price regressions, merit-order curves, and equity curves."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_price_impact_scatter(
    df: pd.DataFrame,
    surprise_col: str = "surprise_mw",
    price_delta_col: str = "delta_price_eur_mwh",
    regime_col: str = "regime",
    title: str = "Renewable Generation Surprise vs Intraday-to-Day-Ahead Price Impact"
) -> go.Figure:
    """Creates an interactive scatter plot of generation surprise vs price delta colored by regime."""
    plot_df = df.copy()
    if regime_col not in plot_df.columns:
        if "is_dunkelflaute" in plot_df.columns and "is_storm" in plot_df.columns:
            conditions = [
                plot_df["is_dunkelflaute"] == 1,
                plot_df["is_storm"] == 1,
                plot_df.get("is_heat_dome", 0) == 1,
            ]
            choices = ["Dunkelflaute", "Storm Cutout", "Heat Dome"]
            plot_df[regime_col] = np.select(conditions, choices, default="Normal Operation")
        else:
            plot_df[regime_col] = "Normal Operation"

    fig = px.scatter(
        plot_df,
        x=surprise_col,
        y=price_delta_col,
        color=regime_col,
        trendline="ols",
        title=title,
        labels={
            surprise_col: "Renewable Generation Surprise (MW)",
            price_delta_col: "Price Delta: Intraday - Day-Ahead (EUR/MWh)",
            regime_col: "Market Regime"
        },
        template="plotly_white",
        opacity=0.75
    )
    return fig


def create_merit_order_figure(
    residual_load_gw: np.ndarray,
    prices_eur_mwh: np.ndarray,
    spline_x_gw: np.ndarray,
    spline_y_price: np.ndarray,
    marginal_slope: np.ndarray
) -> go.Figure:
    """Creates a dual-axis Plotly figure of the non-linear merit-order curve and marginal slope."""
    fig = go.Figure()

    # Raw scatter points
    fig.add_trace(go.Scatter(
        x=residual_load_gw,
        y=prices_eur_mwh,
        mode="markers",
        marker=dict(size=4, color="gray", opacity=0.35),
        name="Historical Cleared Prices"
    ))

    # Fitted merit-order spline
    fig.add_trace(go.Scatter(
        x=spline_x_gw,
        y=spline_y_price,
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        name="Merit-Order Spline P(R)"
    ))

    fig.update_layout(
        title="Empirical Merit-Order Power Price Curve",
        xaxis_title="Residual Electrical Load (GW)",
        yaxis_title="Power Price (EUR/MWh)",
        template="plotly_white"
    )
    return fig


def create_backtest_equity_curve(
    timestamps: list[Any] | np.ndarray,
    strategy_equity: list[float],
    benchmark_naive: list[float] | None = None,
    benchmark_nwp: list[float] | None = None
) -> go.Figure:
    """Creates cumulative PnL equity curve plot comparing strategy against naive and NWP benchmarks."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=strategy_equity,
        mode="lines",
        line=dict(color="#2ca02c", width=2.5),
        name="Forecast Error Arbitrage Strategy"
    ))

    if benchmark_nwp is not None:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=benchmark_nwp,
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            name="Raw NWP-Direct Benchmark"
        ))

    if benchmark_naive is not None:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=benchmark_naive,
            mode="lines",
            line=dict(color="#7f7f7f", width=1.5, dash="dot"),
            name="Naive Benchmark"
        ))

    fig.update_layout(
        title="Cumulative Trading PnL (Net of Frictions: 1.70 EUR/MWh)",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL (EUR)",
        template="plotly_white"
    )
    return fig


def save_market_figures(
    df: pd.DataFrame,
    equity_curve: list[float],
    output_dir: str = "reports/figures"
) -> None:
    """Saves static publication-quality PNG market impact and equity figures."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Price impact scatter
    plt.figure(figsize=(7, 5))
    if "surprise_mw" in df.columns and "delta_price_eur_mwh" in df.columns:
        plt.scatter(df["surprise_mw"], df["delta_price_eur_mwh"], alpha=0.5, c="#1f77b4", edgecolors="none")
        # Add regression line
        clean = df[["surprise_mw", "delta_price_eur_mwh"]].dropna()
        if len(clean) > 2:
            m, b = np.polyfit(clean["surprise_mw"], clean["delta_price_eur_mwh"], 1)
            x_vals = np.linspace(clean["surprise_mw"].min(), clean["surprise_mw"].max(), 100)
            plt.plot(x_vals, m * x_vals + b, "r-", label=f"Slope: {m*1000:.2f} EUR/GWh")
            plt.legend()
    plt.xlabel("Renewable Generation Surprise (MW)")
    plt.ylabel("Intraday - Day-Ahead Delta Price (EUR/MWh)")
    plt.title("Price Impact of Forecast Errors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "price_impact_scatter.png", dpi=300)
    plt.close()

    # 2. Equity curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(equity_curve, color="#2ca02c", linewidth=2, label="Arbitrage Strategy (Net PnL)")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Trade Hours")
    plt.ylabel("Cumulative PnL (EUR)")
    plt.title("Trading Backtest Equity Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "backtest_equity_curve.png", dpi=300)
    plt.close()
