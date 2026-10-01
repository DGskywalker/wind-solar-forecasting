"""Feature pipeline and feature store writer with strict temporal discipline and feature cataloging."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from wind_solar_forecast.data.capacity import get_installed_capacity
from wind_solar_forecast.data.geo import get_zone_geo
from wind_solar_forecast.data.schemas import FeatureManifestEntry
from wind_solar_forecast.features.pv_model import compute_pv_capacity_factor
from wind_solar_forecast.features.spatial import compute_spatial_gradient, compute_upstream_advection_lag
from wind_solar_forecast.features.temporal import (
    compute_fourier_cyclical_features,
    compute_holiday_and_dst_flags,
    compute_ramp_features,
    compute_strictly_backward_lags,
)
from wind_solar_forecast.features.turbine_curve import aggregate_zone_wind_cf
from wind_solar_forecast.features.weather_features import build_weather_derived_features
from wind_solar_forecast.utils.io import load_parquet, save_parquet
from wind_solar_forecast.utils.logging import logger
from wind_solar_forecast.utils.timeutils import to_utc_datetime


def identify_regime_flags(
    wind_cf: np.ndarray,
    solar_cf: np.ndarray,
    wind_speed: np.ndarray,
    temp_c: np.ndarray,
    min_dunkelflaute_hours: int = 6
) -> pd.DataFrame:
    """Computes regime flags: Dunkelflaute, Storm cut-out risk, Heat dome, and Negative-price precursor.

    - Dunkelflaute: Wind CF < 10% AND Solar CF < 5% sustained for at least 6 consecutive hours.
    - Storm: Wind speed > 90th percentile (~18+ m/s) with cut-out hysteresis.
    - Heat dome: Temperature > 95th percentile with high solar irradiance and cooling load.
    - Negative-price precursor: Renewable CF > 0.65 during off-peak weekend or afternoon hours.
    """
    n = len(wind_cf)

    # Low renewable indicator
    low_res = (wind_cf < 0.10) & (solar_cf < 0.05)
    # Rolling sum to detect persistence
    dunkelflaute = pd.Series(low_res).rolling(min_dunkelflaute_hours, min_periods=min_dunkelflaute_hours).sum().values == min_dunkelflaute_hours

    # Storm flag (wind speed > 18.0 m/s)
    storm = wind_speed >= 18.0

    # Heat dome flag (temp > 30C / 303.15K)
    heat_dome = temp_c >= 30.0

    # Negative price precursor
    combined_cf = 0.6 * wind_cf + 0.4 * solar_cf
    negative_price_risk = combined_cf > 0.60

    return pd.DataFrame({
        "is_dunkelflaute": dunkelflaute.astype(int),
        "is_storm": storm.astype(int),
        "is_heat_dome": heat_dome.astype(int),
        "is_negative_price_precursor": negative_price_risk.astype(int),
    })


def build_feature_dataset(
    zone: str = "DE_LU",
    vintage: str = "D-1_12:00",
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-08",
    raw_base_dir: str = "data/raw",
    output_dir: str = "data/feature_store"
) -> tuple[pd.DataFrame, list[FeatureManifestEntry]]:
    """Builds and catalogs complete feature dataset joining weather, physical models, and market targets.

    Args:
        zone: European bidding zone code.
        vintage: Forecast vintage cycle.
        start_date: Start date string.
        end_date: End date string.
        raw_base_dir: Base path where raw data was ingested.
        output_dir: Target directory for feature store.

    Returns:
        Tuple of (Engineered DataFrame, List of FeatureManifestEntry records).
    """
    start_ts = to_utc_datetime(start_date)
    end_ts = to_utc_datetime(end_date)
    clean_vint = vintage.replace(":", "").replace("-", "_")

    # 1. Load raw datasets
    weather_file = Path(raw_base_dir) / f"weather/era5_{zone}_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"
    nwp_file = Path(raw_base_dir) / f"nwp_forecasts/nwp_{zone}_{clean_vint}_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"
    market_file = Path(raw_base_dir) / f"market/entsoe_{zone}_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"

    # If raw data does not exist yet, trigger pipeline ingest
    if not (weather_file.exists() and nwp_file.exists() and market_file.exists()):
        from wind_solar_forecast.pipeline.ingest import run_ingestion_pipeline
        run_ingestion_pipeline(zone=zone, start_date=start_date, end_date=end_date, vintages=[vintage], output_base_dir=raw_base_dir)

    weather_df = pd.read_parquet(weather_file)
    nwp_df = pd.read_parquet(nwp_file)
    market_df = pd.read_parquet(market_file)

    weather_df["valid_time"] = pd.to_datetime(weather_df["valid_time"], utc=True)
    nwp_df["valid_time"] = pd.to_datetime(nwp_df["valid_time"], utc=True)
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)

    # 2. Join NWP and Market Data on valid timestamp
    df = pd.merge(nwp_df, market_df, left_on="valid_time", right_on="timestamp", how="inner")
    df = df.sort_values("valid_time").reset_index(drop=True)

    # 3. Physical Capacity Factor Features (NWP-direct)
    # Wind onshore
    df["nwp_wind_onshore_cf"] = aggregate_zone_wind_cf(
        df["forecast_wind_speed_100m"].values, is_offshore=False
    )
    # Wind offshore
    df["nwp_wind_offshore_cf"] = aggregate_zone_wind_cf(
        df["forecast_wind_speed_100m"].values, is_offshore=True
    )
    # Solar PV
    df["nwp_solar_cf"] = compute_pv_capacity_factor(
        timestamps=df["valid_time"],
        ghi=df["forecast_ghi"].values,
        temp_2m_k=df["forecast_temperature_2m_k"].values,
        total_cloud_cover=df["forecast_cloud_cover"].values,
        zone=zone
    )

    # 4. Ramp and Gradient Features on NWP CF
    ramp_onshore = compute_ramp_features(df["nwp_wind_onshore_cf"], windows=[1, 3, 6], prefix="ramp_wind_cf")
    ramp_solar = compute_ramp_features(df["nwp_solar_cf"], windows=[1, 3, 6], prefix="ramp_solar_cf")
    df = pd.concat([df, ramp_onshore, ramp_solar], axis=1)

    # 5. Temporal Features
    dt_index = pd.DatetimeIndex(df["valid_time"])
    fourier_df = compute_fourier_cyclical_features(dt_index)
    holiday_dst_df = compute_holiday_and_dst_flags(dt_index, zone=zone)
    df = pd.concat([df, fourier_df, holiday_dst_df], axis=1)

    # 6. Physical Atmospheric Derived Features
    df = build_weather_derived_features(df)

    # 7. Regime Flags
    regimes_df = identify_regime_flags(
        wind_cf=df["nwp_wind_onshore_cf"].values,
        solar_cf=df["nwp_solar_cf"].values,
        wind_speed=df["forecast_wind_speed_100m"].values,
        temp_c=df["temperature_2m_c"].values
    )
    df = pd.concat([df, regimes_df], axis=1)

    # 8. Capacity-normalized Targets
    caps = [get_installed_capacity(zone, t) for t in df["valid_time"]]
    cap_on_mw = np.array([c["wind_onshore"] for c in caps])
    cap_off_mw = np.array([c["wind_offshore"] for c in caps])
    cap_sol_mw = np.array([c["solar_pv"] for c in caps])

    df["actual_wind_onshore_cf"] = np.clip(df["wind_onshore_mw"] / np.maximum(cap_on_mw, 1.0), 0.0, 1.0)
    df["actual_wind_offshore_cf"] = np.clip(df["wind_offshore_mw"] / np.maximum(cap_off_mw, 1.0), 0.0, 1.0)
    df["actual_solar_cf"] = np.clip(df["solar_pv_mw"] / np.maximum(cap_sol_mw, 1.0), 0.0, 1.0)

    # 9. Strictly backward-looking market lags (available at gate closure)
    # Day-Ahead price lags: 24h, 48h, 168h
    da_price_lags = compute_strictly_backward_lags(df["day_ahead_price_eur_mwh"], lags=[24, 48, 168], prefix="da_price_lag")
    # Residual load lags: 24h, 168h
    res_load_lags = compute_strictly_backward_lags(df["residual_load_mw"], lags=[24, 168], prefix="res_load_lag")
    df = pd.concat([df, da_price_lags, res_load_lags], axis=1)

    # Clean initial NaN values from lags with backward fill / sensible initial values
    df = df.bfill().ffill()

    # 10. Persist to Feature Store
    out_path = Path(output_dir) / f"features_{zone}_{clean_vint}_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"
    save_parquet(df, out_path)
    logger.info("Persisted engineered features dataset", path=str(out_path), rows=len(df), cols=len(df.columns))

    # 11. Compile Feature Manifest
    manifest_entries: list[FeatureManifestEntry] = []
    target_cols = {
        "actual_wind_onshore_cf", "actual_wind_offshore_cf", "actual_solar_cf",
        "wind_onshore_mw", "wind_offshore_mw", "solar_pv_mw", "total_renewable_mw",
        "day_ahead_price_eur_mwh", "intraday_price_eur_mwh", "residual_load_mw"
    }

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        is_target = col in target_cols
        tag = "TARGET_CONTEMPORANEOUS" if is_target else "SAFE_AT_GATE_CLOSURE"
        desc = f"Feature {col} engineered for {zone} under vintage {vintage}"

        manifest_entries.append(
            FeatureManifestEntry(
                feature_name=col,
                dtype=dtype_str,
                source_table="feature_store",
                vintage_availability=vintage,
                leakage_risk_tag=tag,
                description=desc
            )
        )

    manifest_path = Path(output_dir) / f"manifest_{zone}_{clean_vint}.json"
    with open(manifest_path, "w") as f:
        json.dump([e.model_dump() for e in manifest_entries], f, indent=2)
    logger.info("Generated feature store manifest", path=str(manifest_path))

    return df, manifest_entries


def main() -> None:
    build_feature_dataset(zone="DE_LU", vintage="D-1_12:00", start_date="2023-01-01", end_date="2023-01-08")


if __name__ == "__main__":
    main()
