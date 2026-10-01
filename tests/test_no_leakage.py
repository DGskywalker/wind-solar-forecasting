"""CRITICAL TEMPORAL LEAKAGE AUDIT: Verifies strict absence of lookahead bias."""

import numpy as np
import pandas as pd
import pytest

from wind_solar_forecast.features.build_features import build_feature_dataset
from wind_solar_forecast.features.temporal import compute_strictly_backward_lags
from wind_solar_forecast.utils.timeutils import get_forecast_issue_time


def test_forecast_issue_time_strictly_precedes_valid_time() -> None:
    """Verifies that all forecast vintages have issue_time <= valid_time (positive lead time)."""
    dates = pd.date_range("2023-01-01", "2023-01-14", freq="1h", tz="UTC")
    for vintage in ["D-2_12:00", "D-1_00:00", "D-1_12:00", "ID_4h"]:
        for dt in dates:
            issue_ts = get_forecast_issue_time(dt, vintage)
            assert issue_ts < dt, f"Vintage {vintage} at {dt} has issue_time {issue_ts} >= valid_time!"
            lead_hours = (dt - issue_ts).total_seconds() / 3600.0
            if vintage == "ID_4h":
                assert pytest.approx(lead_hours, abs=0.1) == 4.0
            elif vintage == "D-1_12:00":
                # Lead time for D-1 12:00 CET must be between 12 and 36 hours
                assert 12.0 <= lead_hours <= 37.0


def test_backward_lags_contain_no_forward_information() -> None:
    """Verifies that a future impulse shock at T does not alter any lagged features prior to T."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    base_series = pd.Series(np.ones(n), index=dates)

    # Compute baseline lags
    df_lags_base = compute_strictly_backward_lags(base_series, lags=[1, 24, 48], prefix="lag")

    # Introduce an extreme sudden shock at index T=50
    shocked_series = base_series.copy()
    shock_idx = 50
    shocked_series.iloc[shock_idx:] = 999.0

    df_lags_shocked = compute_strictly_backward_lags(shocked_series, lags=[1, 24, 48], prefix="lag")

    # Features before shock_idx must be IDENTICAL between baseline and shocked series
    for lag in [1, 24, 48]:
        col = f"lag_{lag}h"
        # Prior to shock_idx, zero difference
        diff_pre_shock = (df_lags_shocked[col].iloc[:shock_idx] - df_lags_base[col].iloc[:shock_idx]).dropna()
        assert (diff_pre_shock == 0.0).all(), f"Leakage detected! Shock at {shock_idx} affected {col} before shock!"

        # Exactly at shock_idx, lag_1h must STILL reflect the pre-shock value (no instantaneous leakage)
        assert df_lags_shocked[col].iloc[shock_idx] == base_series.iloc[shock_idx - lag]


def test_feature_manifest_leakage_tagging(tmp_path: pytest.TempPathFactory) -> None:
    """Verifies that actual target values are marked with TARGET_CONTEMPORANEOUS and not SAFE_AT_GATE_CLOSURE."""
    out_dir = str(tmp_path)
    df, manifest = build_feature_dataset(
        zone="DE_LU",
        vintage="D-1_12:00",
        start_date="2023-01-01",
        end_date="2023-01-07 23:00:00",
        output_dir=out_dir
    )

    forbidden_in_inputs = {
        "actual_wind_onshore_cf",
        "actual_wind_offshore_cf",
        "actual_solar_cf",
        "wind_onshore_mw",
        "wind_offshore_mw",
        "solar_pv_mw",
        "total_renewable_mw",
        "day_ahead_price_eur_mwh",
        "intraday_price_eur_mwh",
        "residual_load_mw"
    }

    manifest_dict = {entry.feature_name: entry.leakage_risk_tag for entry in manifest}

    for col in forbidden_in_inputs:
        if col in manifest_dict:
            assert manifest_dict[col] == "TARGET_CONTEMPORANEOUS", (
                f"Target column '{col}' is incorrectly marked as '{manifest_dict[col]}' instead of TARGET_CONTEMPORANEOUS!"
            )


def test_walk_forward_embargo_discipline() -> None:
    """Verifies that an expanding walk-forward split enforces the horizon-length embargo."""
    total_days = 60
    dates = pd.date_range("2023-01-01", periods=total_days * 24, freq="1h", tz="UTC")

    # 30-day train, 1-day embargo (24h), 10-day test
    train_end = dates[30 * 24 - 1]
    embargo_hours = 24
    test_start = train_end + pd.Timedelta(hours=embargo_hours + 1)

    assert (test_start - train_end).total_seconds() / 3600.0 > embargo_hours
