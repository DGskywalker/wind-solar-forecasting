"""Price impact econometrics: Surprise formulation, OLS with Two-Way Fixed Effects, and 2SLS Instrumental Variables."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg


@dataclass(frozen=True)
class RegressionSpecResult:
    spec_name: str
    coefficient: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_95: tuple[float, float]
    r_squared: float
    n_obs: int
    diagnostics: dict[str, float]


def compute_generation_surprise(
    actual_generation_mw: np.ndarray,
    forecast_generation_mw: np.ndarray,
    installed_capacity_mw: float
) -> pd.DataFrame:
    """Computes raw, capacity-normalized, and z-scored generation forecast surprises.

    S_t = Actual_t - Forecast_t
    Positive surprise -> excess generation -> downward price pressure.
    """
    raw_surprise_mw = actual_generation_mw - forecast_generation_mw
    cf_surprise = raw_surprise_mw / max(installed_capacity_mw, 1.0)
    std_s = np.std(cf_surprise) + 1e-6
    z_surprise = (cf_surprise - np.mean(cf_surprise)) / std_s

    return pd.DataFrame({
        "surprise_mw": np.round(raw_surprise_mw, 1),
        "surprise_cf": np.round(cf_surprise, 4),
        "surprise_z": np.round(z_surprise, 4)
    })


def estimate_ols_fixed_effects(
    df: pd.DataFrame,
    dep_var: str = "delta_price_eur_mwh",
    surprise_var: str = "surprise_mw",
    control_vars: list[str] | None = None,
    hac_lags: int = 24
) -> RegressionSpecResult:
    """Estimates OLS with Hour and Month Fixed Effects using Newey-West HAC standard errors.

    Model:
        Delta_Price_t = beta * S_t + gamma * Controls_t + mu_hour + eta_month + epsilon_t
    """
    data = df.copy()
    if control_vars is None:
        control_vars = ["gas_ttf_eur_mwh", "eua_carbon_eur_ton", "total_load_mw"]

    # One-hot encode Hour and Month fixed effects
    data["hour"] = pd.to_datetime(data["valid_time"], utc=True).dt.hour
    data["month"] = pd.to_datetime(data["valid_time"], utc=True).dt.month

    # Drop first dummy to avoid dummy variable trap
    hour_dummies = pd.get_dummies(data["hour"], prefix="fe_h", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(data["month"], prefix="fe_m", drop_first=True, dtype=float)

    reg_df = pd.concat([data[[dep_var, surprise_var] + control_vars], hour_dummies, month_dummies], axis=1).dropna()

    y = reg_df[dep_var].values
    X_vars = [surprise_var] + control_vars + list(hour_dummies.columns) + list(month_dummies.columns)
    X = sm.add_constant(reg_df[X_vars].values)

    # Fit OLS with Newey-West HAC covariance
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # Index 1 is surprise_var (after constant)
    coef = float(model.params[1])
    se = float(model.bse[1])
    t_stat = float(model.tvalues[1])
    p_val = float(model.pvalues[1])
    ci = (float(model.conf_int()[1, 0]), float(model.conf_int()[1, 1]))

    return RegressionSpecResult(
        spec_name="OLS_TwoWay_FE_HAC",
        coefficient=coef,
        std_error=se,
        t_statistic=t_stat,
        p_value=p_val,
        ci_95=ci,
        r_squared=float(model.rsquared),
        n_obs=len(y),
        diagnostics={"f_statistic": float(model.fvalue or 0.0), "hac_max_lags": float(hac_lags)}
    )


def estimate_2sls_instrumental_variables(
    df: pd.DataFrame,
    dep_var: str = "delta_price_eur_mwh",
    endog_var: str = "surprise_mw",
    instrument_var: str = "wind_speed_forecast_error",
    control_vars: list[str] | None = None
) -> RegressionSpecResult:
    """Estimates Two-Stage Least Squares (2SLS) using exogenous NWP meteorological errors as instruments.

    First Stage:
        S_t = pi * (WS_actual - WS_forecast)_t + delta * Controls_t + v_t
    Second Stage:
        Delta_Price_t = beta_IV * S_hat_t + gamma * Controls_t + epsilon_t

    Mitigates endogeneity from unobserved load shocks or simultaneous bidding.
    """
    if control_vars is None:
        control_vars = ["gas_ttf_eur_mwh", "eua_carbon_eur_ton", "total_load_mw"]

    data = df[[dep_var, endog_var, instrument_var] + control_vars].dropna()

    y = data[dep_var].values
    D = data[endog_var].values
    Z = data[instrument_var].values
    W = data[control_vars].values

    # Stage 1: Regress endogenous D on Instrument Z and Controls W
    X_stage1 = sm.add_constant(np.column_stack([Z, W]))
    stage1_model = sm.OLS(D, X_stage1).fit()
    D_hat = stage1_model.fittedvalues

    # First-stage F-statistic on instrument (Stock-Yogo weak IV benchmark > 10)
    # F-stat on Z (index 1)
    f_stat_iv = float(stage1_model.tvalues[1] ** 2)

    # Stage 2: Regress y on D_hat and Controls W
    X_stage2 = sm.add_constant(np.column_stack([D_hat, W]))
    stage2_model = sm.OLS(y, X_stage2).fit()

    coef_iv = float(stage2_model.params[1])
    # 2SLS standard error correction for fitted values
    residuals_iv = y - sm.add_constant(np.column_stack([D, W])) @ stage2_model.params
    sigma2_iv = np.sum(residuals_iv ** 2) / max(len(y) - X_stage2.shape[1], 1)
    cov_iv = sigma2_iv * np.linalg.pinv(X_stage2.T @ X_stage2)
    se_iv = float(np.sqrt(max(cov_iv[1, 1], 1e-9)))
    t_stat_iv = coef_iv / (se_iv + 1e-9)
    p_val_iv = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat_iv))))
    ci_iv = (coef_iv - 1.96 * se_iv, coef_iv + 1.96 * se_iv)

    return RegressionSpecResult(
        spec_name="2SLS_Instrumental_Variables",
        coefficient=coef_iv,
        std_error=se_iv,
        t_statistic=t_stat_iv,
        p_value=p_val_iv,
        ci_95=ci_iv,
        r_squared=float(stage2_model.rsquared),
        n_obs=len(y),
        diagnostics={"first_stage_f_statistic": f_stat_iv, "is_instrument_strong": bool(f_stat_iv > 10.0)}
    )


def estimate_quantile_price_impact(
    df: pd.DataFrame,
    dep_var: str = "delta_price_eur_mwh",
    surprise_var: str = "surprise_mw",
    quantiles: list[float] | None = None
) -> dict[float, dict[str, float]]:
    """Quantile regression showing asymmetric price sensitivity across price distribution percentiles."""
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    data = df[[dep_var, surprise_var, "total_load_mw"]].dropna()
    y = data[dep_var].values
    load_gw = data["total_load_mw"].values / 1000.0
    surprise_gw = data[surprise_var].values / 1000.0
    X = sm.add_constant(np.column_stack([surprise_gw, load_gw]))

    results = {}
    for q in quantiles:
        mod = QuantReg(y, X).fit(q=q, max_iter=3000)
        # Rescale coefficient back to per-MW
        results[q] = {
            "coefficient": float(mod.params[1] / 1000.0),
            "std_error": float(mod.bse[1] / 1000.0),
            "p_value": float(mod.pvalues[1]),
            "ci_low": float(mod.conf_int()[1, 0] / 1000.0),
            "ci_high": float(mod.conf_int()[1, 1] / 1000.0)
        }

    return results


def run_all_price_impact_regressions(zone: str = "DE_LU", vintage: str = "D-1_12:00") -> dict[str, Any]:
    """Loads feature store data and executes OLS-FE, 2SLS-IV, Quantile, and DML models."""
    import json
    from pathlib import Path

    from wind_solar_forecast.features.build_features import build_feature_dataset

    df, _ = build_feature_dataset(zone=zone, vintage=vintage, start_date="2023-01-01", end_date="2023-01-08")
    installed_cap = 67000.0  # MW DE_LU benchmark

    # Compute surprises
    pred_gen = (df["nwp_wind_onshore_cf"] * 58500.0 + df["nwp_solar_cf"] * 82000.0).values
    actual_gen = (df["wind_onshore_mw"] + df["solar_pv_mw"]).values
    surprise_df = compute_generation_surprise(actual_gen, pred_gen, installed_cap)

    reg_data = pd.concat([df, surprise_df], axis=1)
    reg_data["delta_price_eur_mwh"] = reg_data["intraday_price_eur_mwh"] - reg_data["day_ahead_price_eur_mwh"]
    reg_data["wind_speed_forecast_error"] = (
        (reg_data["wind_speed_100m"] if "wind_speed_100m" in reg_data.columns else reg_data["forecast_wind_speed_100m"])
        - reg_data["forecast_wind_speed_100m"]
    )

    ols_res = estimate_ols_fixed_effects(reg_data, hac_lags=12)
    iv_res = estimate_2sls_instrumental_variables(reg_data)
    q_res = estimate_quantile_price_impact(reg_data)

    from wind_solar_forecast.market.causality import estimate_double_machine_learning
    dml_res = estimate_double_machine_learning(reg_data, n_folds=3)

    summary = {
        "zone": zone,
        "vintage": vintage,
        "ols_fe": {
            "coefficient_eur_mwh_per_mw": np.round(ols_res.coefficient, 6),
            "std_error": np.round(ols_res.std_error, 6),
            "t_statistic": np.round(ols_res.t_statistic, 3),
            "p_value": np.round(ols_res.p_value, 5),
            "ci_95": [np.round(ols_res.ci_95[0], 6), np.round(ols_res.ci_95[1], 6)],
            "r_squared": np.round(ols_res.r_squared, 4),
        },
        "iv_2sls": {
            "coefficient_eur_mwh_per_mw": np.round(iv_res.coefficient, 6),
            "std_error": np.round(iv_res.std_error, 6),
            "t_statistic": np.round(iv_res.t_statistic, 3),
            "p_value": np.round(iv_res.p_value, 5),
            "ci_95": [np.round(iv_res.ci_95[0], 6), np.round(iv_res.ci_95[1], 6)],
            "first_stage_f_statistic": np.round(iv_res.diagnostics["first_stage_f_statistic"], 2),
        },
        "double_ml": {
            "theta_effect_eur_mwh_per_gw": np.round(dml_res.theta_effect, 4),
            "std_error": np.round(dml_res.std_error, 4),
            "p_value": np.round(dml_res.p_value, 5),
            "ci_95": [np.round(dml_res.ci_95[0], 4), np.round(dml_res.ci_95[1], 4)],
        },
        "quantile_regressions": q_res
    }

    out_file = Path("reports/market_impact_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", type=str, default="DE_LU")
    args = parser.parse_args()
    res = run_all_price_impact_regressions(zone=args.zone)
    print("Market Econometric Results Summary:")
    print(f"  OLS-FE Beta: {res['ols_fe']['coefficient_eur_mwh_per_mw']} EUR/MWh per MW (p={res['ols_fe']['p_value']})")
    print(f"  2SLS-IV Beta: {res['iv_2sls']['coefficient_eur_mwh_per_mw']} EUR/MWh per MW (F={res['iv_2sls']['first_stage_f_statistic']})")
    print(f"  Double ML Theta: {res['double_ml']['theta_effect_eur_mwh_per_gw']} EUR/MWh per GW (95% CI: {res['double_ml']['ci_95']})")


if __name__ == "__main__":
    main()
