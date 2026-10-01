"""Unit and integration tests for data ingestion, schemas, and coordinate transformations."""

import pandas as pd
import pytest

from wind_solar_forecast.data.capacity import get_installed_capacity
from wind_solar_forecast.data.geo import compute_latitude_weights, get_zone_geo
from wind_solar_forecast.data.schemas import (
    ForecastVintage,
    MarketPriceRecord,
    NWPForecastPoint,
    PowerGenerationRecord,
    WeatherObservation,
)
from wind_solar_forecast.pipeline.ingest import run_ingestion_pipeline


def test_latitude_weighting() -> None:
    """Tests that cosine latitude weights sum to 1 and decrease with increasing latitude."""
    lats = [0.0, 30.0, 60.0]
    weights = compute_latitude_weights(lats)
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert weights[0] > weights[1] > weights[2]


def test_zone_geography() -> None:
    """Tests geography metadata completeness for key European zones."""
    for zone in ["DE_LU", "FR", "ES", "GB", "NL", "DK_1", "PL"]:
        geo = get_zone_geo(zone)
        assert geo.lat_min < geo.centroid_lat < geo.lat_max
        assert geo.lon_min < geo.centroid_lon < geo.lon_max
        assert 20.0 <= geo.pv_tilt_deg <= 50.0


def test_capacity_interpolation() -> None:
    """Tests continuous capacity interpolation over calendar years."""
    t1 = pd.Timestamp("2022-01-01", tz="UTC")
    t2 = pd.Timestamp("2023-01-01", tz="UTC")
    cap1 = get_installed_capacity("DE_LU", t1)
    cap2 = get_installed_capacity("DE_LU", t2)
    assert cap2["solar_pv"] > cap1["solar_pv"]
    assert cap2["wind_onshore"] >= cap1["wind_onshore"]


def test_schema_contracts() -> None:
    """Tests Pydantic validation contracts on weather, generation, and market records."""
    obs = WeatherObservation(
        valid_time="2023-01-01T12:00:00Z",
        zone="DE_LU",
        wind_speed_100m=8.5,
        wind_direction_100m=245.0,
        wind_speed_10m=5.2,
        surface_solar_radiation_downwards=420.0,
        surface_net_solar_radiation=340.0,
        temperature_2m_k=285.15,
        total_cloud_cover=0.35,
        surface_pressure_pa=101200.0,
    )
    assert obs.wind_speed_100m == 8.5

    nwp = NWPForecastPoint(
        issue_time="2022-12-31T12:00:00Z",
        valid_time="2023-01-01T12:00:00Z",
        zone="DE_LU",
        vintage=ForecastVintage.D_1_1200,
        horizon_hours=24.0,
        forecast_wind_speed_100m=9.1,
        forecast_ghi=410.0,
        forecast_temperature_2m_k=284.5,
        forecast_cloud_cover=0.40,
    )
    assert nwp.horizon_hours == 24.0


def test_one_week_sample_ingest_de_lu(tmp_path: pytest.TempPathFactory) -> None:
    """Executes a 1-week data ingestion sample for DE_LU and verifies schema completeness."""
    out_dir = str(tmp_path)
    res = run_ingestion_pipeline(
        zone="DE_LU",
        start_date="2023-01-01",
        end_date="2023-01-07 23:00:00",
        vintages=["D-1_12:00", "ID_4h"],
        output_base_dir=out_dir
    )

    weather_df = res["weather"]
    nwp_df = res["nwp"]
    market_df = res["market"]

    # 7 days * 24 hours = 168 rows
    assert len(weather_df) == 168
    assert len(market_df) == 168
    assert len(nwp_df) == 168 * 2  # 2 vintages

    # Physical checks
    assert (weather_df["wind_speed_100m"] >= 0.0).all()
    assert (weather_df["surface_solar_radiation_downwards"] >= 0.0).all()
    assert (market_df["wind_onshore_mw"] >= 0.0).all()
    assert (market_df["solar_pv_mw"] >= 0.0).all()
    assert (market_df["total_load_mw"] > 0.0).all()

    # Residual load accounting identity
    total_renewable = market_df["wind_onshore_mw"] + market_df["wind_offshore_mw"] + market_df["solar_pv_mw"]
    assert pytest.approx((market_df["total_load_mw"] - total_renewable).values, abs=1.0) == market_df["residual_load_mw"].values
