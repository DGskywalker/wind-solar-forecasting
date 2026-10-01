"""Unit tests for market econometrics: OLS Two-Way FE, 2SLS IV, Double ML, and trading backtest."""

import numpy as np
import pandas as pd
import pytest

from wind_solar_forecast.market.backtest import run_power_market_backtest
from wind_solar_forecast.market.causality import estimate_double_machine_learning
from wind_solar_forecast.market.merit_order import MeritOrderModel
from wind_solar_forecast.market.price_impact import (
    compute_generation_surprise,
    estimate_2sls_instrumental_variables,
    estimate_ols_fixed_effects,
    estimate_quantile_price_impact,
)


@pytest.fixture
def market_econometric_data() -> pd.DataFrame:
    """Constructs synthetic power market panel data with endogenous surprise and weather instrument."""
    np.random.seed(42)
    n = 240  # 10 days
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    # Exogenous weather forecast error (NWP error in m/s)
    ws_error = np.random.normal(0, 1.5, n)

    # Actual and forecast generation (MW)
    # Wind speed error directly drives generation surprise (first stage)
    gen_surprise_mw = 1200.0 * ws_error + np.random.normal(0, 300, n)
    actual_gen_mw = np.maximum(8000.0 + gen_surprise_mw, 0.0)
    forecast_gen_mw = np.maximum(8000.0, 0.0)

    # Market fundamentals
    load_mw = 45000.0 + 8000.0 * np.sin(2 * np.pi * dates.hour / 24.0) + np.random.normal(0, 1000, n)
    residual_load_mw = load_mw - actual_gen_mw
    gas_ttf = 35.0 + np.random.normal(0, 2.0, n)
    carbon_eua = 75.0 + np.random.normal(0, 1.5, n)

    # True price impact coefficient: -0.0035 EUR/(MWh * MW) = -3.5 EUR/(MWh * GW)
    # Generation surprise reduces power price
    delta_p = -0.0035 * gen_surprise_mw + 0.0004 * (load_mw - 45000.0) + np.random.normal(0, 5.0, n)
    da_price = 80.0 + 0.001 * (residual_load_mw - 35000.0) + np.random.normal(0, 3.0, n)
    id_price = da_price + delta_p

    return pd.DataFrame({
        "valid_time": dates,
        "actual_generation_mw": actual_gen_mw,
        "forecast_generation_mw": forecast_gen_mw,
        "surprise_mw": gen_surprise_mw,
        "wind_speed_forecast_error": ws_error,
        "total_load_mw": load_mw,
        "residual_load_mw": residual_load_mw,
        "gas_ttf_eur_mwh": gas_ttf,
        "eua_carbon_eur_ton": carbon_eua,
        "day_ahead_price_eur_mwh": da_price,
        "intraday_price_eur_mwh": id_price,
        "delta_price_eur_mwh": delta_p,
        "temperature_2m_c": np.random.normal(12.0, 5.0, n),
    })


def test_generation_surprise(market_econometric_data: pd.DataFrame) -> None:
    """Tests surprise calculation, capacity normalization, and z-scoring."""
    df = market_econometric_data
    res = compute_generation_surprise(df["actual_generation_mw"].values, df["forecast_generation_mw"].values, installed_capacity_mw=25000.0)
    assert len(res) == len(df)
    assert pytest.approx(np.mean(res["surprise_z"]), abs=0.1) == 0.0
    assert pytest.approx(np.std(res["surprise_z"]), abs=0.1) == 1.0


def test_ols_fixed_effects(market_econometric_data: pd.DataFrame) -> None:
    """Tests OLS with two-way fixed effects and negative price impact coefficient."""
    df = market_econometric_data
    res = estimate_ols_fixed_effects(df, hac_lags=12)
    assert res.coefficient < 0.0, "Surprise coefficient should be negative (excess gen reduces price)!"
    assert res.p_value < 0.05
    assert res.n_obs == len(df)


def test_2sls_instrumental_variables(market_econometric_data: pd.DataFrame) -> None:
    """Tests 2SLS IV estimation and verifies strong instrument F-statistic."""
    df = market_econometric_data
    res = estimate_2sls_instrumental_variables(df)
    assert res.coefficient < 0.0
    assert res.diagnostics["first_stage_f_statistic"] > 10.0, "Instrument must satisfy Stock-Yogo strong IV threshold!"
    assert res.diagnostics["is_instrument_strong"] is True


def test_quantile_price_impact(market_econometric_data: pd.DataFrame) -> None:
    """Tests quantile price impact estimation across percentiles."""
    df = market_econometric_data
    res = estimate_quantile_price_impact(df, quantiles=[0.10, 0.50, 0.90])
    assert 0.10 in res and 0.50 in res and 0.90 in res
    assert res[0.50]["coefficient"] < 0.0


def test_merit_order_spline(market_econometric_data: pd.DataFrame) -> None:
    """Tests non-linear merit-order spline fitting and positive slope."""
    df = market_econometric_data
    mo = MeritOrderModel()
    mo.fit(df["residual_load_mw"].values, df["day_ahead_price_eur_mwh"].values)

    slopes = mo.marginal_slope(df["residual_load_mw"].values)
    assert (slopes >= 0.0).all(), "Merit order slope dP/dR must be non-negative!"

    sens = mo.propagate_forecast_error(
        df["residual_load_mw"].values,
        df["surprise_mw"].values,
        df["delta_price_eur_mwh"].values
    )
    assert sens.mean_marginal_slope_eur_gwh >= 0.0


def test_double_machine_learning(market_econometric_data: pd.DataFrame) -> None:
    """Tests Double ML causal estimation with cross-fitting."""
    df = market_econometric_data
    dml_res = estimate_double_machine_learning(df, n_folds=3)
    assert dml_res.theta_effect < 0.0
    assert dml_res.n_obs == len(df)
    assert dml_res.ci_95[0] < dml_res.theta_effect < dml_res.ci_95[1]


def test_trading_backtest(market_econometric_data: pd.DataFrame) -> None:
    """Tests backtest strategy PnL accounting and friction deduction."""
    df = market_econometric_data
    # Construct model forecast column slightly better than naive
    df["model_forecast"] = df["forecast_generation_mw"] + 0.8 * df["surprise_mw"]
    df["da_expected"] = df["forecast_generation_mw"]

    bt = run_power_market_backtest(
        df=df,
        model_forecast_mw_col="model_forecast",
        market_da_expected_mw_col="da_expected",
        transaction_cost_eur_mwh=0.50,
        intraday_half_spread_eur_mwh=1.20
    )

    assert len(bt.equity_curve) == len(df)
    assert bt.total_volume_mwh > 0.0
    assert 0.0 <= bt.win_rate <= 1.0
