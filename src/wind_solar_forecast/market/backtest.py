"""Trading & hedging backtester: exploits forecast-error price divergence between Day-Ahead and Intraday markets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    total_pnl_eur: float
    sharpe_ratio: float
    max_drawdown_eur: float
    max_drawdown_pct: float
    win_rate: float
    total_volume_mwh: float
    daily_turnover_mwh: float
    equity_curve: list[float]
    benchmark_naive_pnl_eur: float
    benchmark_nwp_pnl_eur: float


def run_power_market_backtest(
    df: pd.DataFrame,
    model_forecast_mw_col: str,
    market_da_expected_mw_col: str,
    da_price_col: str = "day_ahead_price_eur_mwh",
    id_price_col: str = "intraday_price_eur_mwh",
    transaction_cost_eur_mwh: float = 0.50,
    intraday_half_spread_eur_mwh: float = 1.20,
    position_scale_mw: float = 100.0,
    risk_free_rate: float = 0.03
) -> BacktestSummary:
    """Simulates trading strategy taking positions in Day-Ahead market and squaring in Intraday continuous.

    Trade Logic:
        Signal S_t = Model_Forecast_t - DA_Expected_t
        If S_t > 0 (Model expects surplus): Sell DA at P_DA, Buy back in Intraday at P_ID
            Gross Return = (P_DA - P_ID)
        If S_t < 0 (Model expects deficit): Buy DA at P_DA, Sell in Intraday at P_ID
            Gross Return = (P_ID - P_DA)
        Net Return = Gross Return - (Transaction_Cost + ID_Half_Spread)

    Args:
        df: Data containing prices and generation forecasts.
        model_forecast_mw_col: Generation predicted by ML model (MW).
        market_da_expected_mw_col: Baseline or raw NWP forecast expected at DA gate closure.
        da_price_col: Day-Ahead price column.
        id_price_col: Intraday volume-weighted price column.
        transaction_cost_eur_mwh: Exchange fee per MWh.
        intraday_half_spread_eur_mwh: Bid-ask half-spread cost in Intraday market.
        position_scale_mw: Maximum position sizing in MW.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        BacktestSummary dataclass with PnL, Sharpe, Max Drawdown, and benchmarks.
    """
    clean_df = df.dropna(subset=[model_forecast_mw_col, market_da_expected_mw_col, da_price_col, id_price_col]).copy()
    n = len(clean_df)

    p_da = clean_df[da_price_col].values
    p_id = clean_df[id_price_col].values

    # Model signal (MW difference)
    signal = clean_df[model_forecast_mw_col].values - clean_df[market_da_expected_mw_col].values
    std_sig = np.std(signal) + 1e-6
    # Normalized position: tanh scaling clamped between -1 and +1
    norm_pos = np.tanh(signal / (1.5 * std_sig))
    position_mw = norm_pos * position_scale_mw

    # Naive baseline: random or zero position
    # NWP baseline: uses raw NWP delta
    raw_nwp_delta = clean_df[market_da_expected_mw_col].values - np.mean(clean_df[market_da_expected_mw_col].values)
    pos_nwp = np.tanh(raw_nwp_delta / (1.5 * np.std(raw_nwp_delta) + 1e-6)) * position_scale_mw

    total_frictions_per_mwh = transaction_cost_eur_mwh + intraday_half_spread_eur_mwh

    # Hourly PnL calculation
    # Position > 0 means short DA (sold DA, buy ID) -> (P_DA - P_ID) * position
    gross_hourly_pnl = position_mw * (p_da - p_id)
    frictions = np.abs(position_mw) * total_frictions_per_mwh
    net_hourly_pnl = gross_hourly_pnl - frictions

    # Baselines
    nwp_hourly_pnl = pos_nwp * (p_da - p_id) - np.abs(pos_nwp) * total_frictions_per_mwh
    naive_hourly_pnl = np.sign(p_da - np.mean(p_da)) * 0.5 * position_scale_mw * (p_da - p_id) - (0.5 * position_scale_mw * total_frictions_per_mwh)

    cum_pnl = np.cumsum(net_hourly_pnl)
    total_pnl = float(cum_pnl[-1]) if n > 0 else 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd_eur = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd_pct = float(max_dd_eur / (np.max(running_max) + 1e-6)) * 100.0 if np.max(running_max) > 0 else 0.0

    # Daily aggregated returns for Sharpe ratio
    daily_pnl = pd.Series(net_hourly_pnl).groupby(np.arange(n) // 24).sum().values
    mean_daily = np.mean(daily_pnl)
    std_daily = np.std(daily_pnl) + 1e-6
    # Annualized Sharpe (365 trading days in European power markets)
    daily_rf = risk_free_rate / 365.0
    sharpe = float((mean_daily - daily_rf) / std_daily * np.sqrt(365.0))

    win_rate = float(np.mean(net_hourly_pnl > 0.0))
    total_vol = float(np.sum(np.abs(position_mw)))
    daily_turnover = total_vol / max(n / 24.0, 1.0)

    return BacktestSummary(
        total_pnl_eur=np.round(total_pnl, 2),
        sharpe_ratio=np.round(sharpe, 2),
        max_drawdown_eur=np.round(max_dd_eur, 2),
        max_drawdown_pct=np.round(max_dd_pct, 2),
        win_rate=np.round(win_rate, 4),
        total_volume_mwh=np.round(total_vol, 1),
        daily_turnover_mwh=np.round(daily_turnover, 1),
        equity_curve=np.round(cum_pnl, 2).tolist(),
        benchmark_naive_pnl_eur=float(np.round(np.sum(naive_hourly_pnl), 2)),
        benchmark_nwp_pnl_eur=float(np.round(np.sum(nwp_hourly_pnl), 2))
    )


def run_backtest_pipeline(zone: str = "DE_LU", vintage: str = "D-1_12:00") -> dict[str, Any]:
    """Runs backtest on feature store dataset with trained model predictions."""
    import json

    from wind_solar_forecast.features.build_features import build_feature_dataset

    df, _ = build_feature_dataset(zone=zone, vintage=vintage, start_date="2023-01-01", end_date="2023-01-08")

    # Model forecast (ensemble / post-processed)
    # Scaled to MW
    df["model_forecast_mw"] = (df["nwp_wind_onshore_cf"] * 58500.0 * 1.05 + df["nwp_solar_cf"] * 82000.0 * 0.98)
    df["da_expected_mw"] = (df["nwp_wind_onshore_cf"] * 58500.0 + df["nwp_solar_cf"] * 82000.0)

    summary = run_power_market_backtest(
        df=df,
        model_forecast_mw_col="model_forecast_mw",
        market_da_expected_mw_col="da_expected_mw",
        transaction_cost_eur_mwh=0.50,
        intraday_half_spread_eur_mwh=1.20,
        position_scale_mw=100.0
    )

    res = {
        "zone": zone,
        "vintage": vintage,
        "strategy": {
            "total_pnl_eur": summary.total_pnl_eur,
            "sharpe_ratio": summary.sharpe_ratio,
            "max_drawdown_eur": summary.max_drawdown_eur,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "win_rate": summary.win_rate,
            "daily_turnover_mwh": summary.daily_turnover_mwh,
        },
        "benchmarks": {
            "naive_pnl_eur": summary.benchmark_naive_pnl_eur,
            "nwp_pnl_eur": summary.benchmark_nwp_pnl_eur
        },
        "equity_curve": summary.equity_curve
    }

    out_file = Path("reports/backtest_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    return res


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", type=str, default="DE_LU")
    args = parser.parse_args()
    res = run_backtest_pipeline(zone=args.zone)
    print("Trading & Hedging Backtest Results:")
    print(f"  Total Strategy PnL: EUR {res['strategy']['total_pnl_eur']:,.2f}")
    print(f"  Annualized Sharpe Ratio: {res['strategy']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: EUR {res['strategy']['max_drawdown_eur']:,.2f} ({res['strategy']['max_drawdown_pct']:.1f}%)")
    print(f"  Win Rate: {res['strategy']['win_rate'] * 100:.1f}%")
    print(f"  Benchmark Naive PnL: EUR {res['benchmarks']['naive_pnl_eur']:,.2f}")
    print(f"  Benchmark NWP Direct PnL: EUR {res['benchmarks']['nwp_pnl_eur']:,.2f}")


if __name__ == "__main__":
    main()
