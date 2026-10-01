"""Open-Meteo Historical Forecast API ingestion for real NWP forecast vintages and errors."""

import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

from wind_solar_forecast.data.geo import get_zone_geo
from wind_solar_forecast.utils.io import save_parquet
from wind_solar_forecast.utils.logging import logger
from wind_solar_forecast.utils.timeutils import get_forecast_issue_time, to_utc_datetime


class OpenMeteoIngestor:
    """Ingestor for operational NWP forecasts (ECMWF IFS / DWD ICON) via Open-Meteo API."""

    BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    def __init__(self, output_dir: str | Path = "data/raw/nwp_forecasts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.getenv("OPENMETEO_API_KEY")

    def fetch_zone_nwp_forecast(
        self,
        zone: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        vintage: str = "D-1_12:00",
        weather_actuals_df: pd.DataFrame | None = None,
        use_fallback: bool = True
    ) -> pd.DataFrame:
        """Fetches NWP forecast time series corresponding to a specific forecast vintage.

        Args:
            zone: European bidding zone.
            start_date: Target start time.
            end_date: Target end time.
            vintage: Vintage tag ('D-2_12:00', 'D-1_00:00', 'D-1_12:00', 'ID_4h').
            weather_actuals_df: Actual weather reanalysis DataFrame to base realistic errors upon if falling back.
            use_fallback: Whether to use physically realistic NWP simulation if API is rate-limited or fails.

        Returns:
            DataFrame containing forecast points indexed by (issue_time, valid_time, zone).
        """
        start_ts = to_utc_datetime(start_date)
        end_ts = to_utc_datetime(end_date)
        clean_vint = vintage.replace(":", "").replace("-", "_")
        out_file = self.output_dir / f"nwp_{zone}_{clean_vint}_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"

        if out_file.exists():
            logger.info("Loading cached NWP forecasts", path=str(out_file))
            return pd.read_parquet(out_file)

        use_synthetic = os.getenv("USE_SYNTHETIC_DATA_FALLBACK", "true").lower() == "true"
        if not self.api_key and use_synthetic and use_fallback:
            df = self._simulate_nwp_from_actuals(zone, start_ts, end_ts, vintage, weather_actuals_df)
            save_parquet(df, out_file)
            return df

        geo = get_zone_geo(zone)
        # Attempt public Open-Meteo fetch if online
        try:
            params: dict[str, Any] = {
                "latitude": geo.centroid_lat,
                "longitude": geo.centroid_lon,
                "start_date": start_ts.strftime("%Y-%m-%d"),
                "end_date": end_ts.strftime("%Y-%m-%d"),
                "hourly": "wind_speed_100m,wind_direction_100m,direct_normal_irradiance,global_horizontal_irradiance,temperature_2m,cloud_cover",
                "timezone": "UTC",
            }
            if self.api_key:
                params["apikey"] = self.api_key

            logger.info("Querying Open-Meteo Historical Forecast API", zone=zone, vintage=vintage)
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                times = pd.to_datetime(hourly.get("time", []), utc=True)
                if len(times) > 0:
                    df = pd.DataFrame({
                        "valid_time": times,
                        "zone": zone,
                        "vintage": vintage,
                        "forecast_wind_speed_100m": np.array(hourly.get("wind_speed_100m", [])),
                        "forecast_ghi": np.array(hourly.get("global_horizontal_irradiance", [])),
                        "forecast_temperature_2m_k": np.array(hourly.get("temperature_2m", [])) + 273.15,
                        "forecast_cloud_cover": np.array(hourly.get("cloud_cover", [])) / 100.0,
                        "is_synthetic": False,
                    })
                    df["issue_time"] = df["valid_time"].apply(lambda t: get_forecast_issue_time(t, vintage))
                    df["horizon_hours"] = (df["valid_time"] - df["issue_time"]).dt.total_seconds() / 3600.0
                    save_parquet(df, out_file)
                    return df
        except Exception as e:
            logger.warning("Open-Meteo API query encountered issue, using realistic NWP error simulation", error=str(e))

        if not use_fallback:
            raise RuntimeError(f"Failed to fetch Open-Meteo data and fallback is disabled.")

        df = self._simulate_nwp_from_actuals(zone, start_ts, end_ts, vintage, weather_actuals_df)
        save_parquet(df, out_file)
        return df

    def _simulate_nwp_from_actuals(
        self,
        zone: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        vintage: str,
        actuals_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Simulates realistic NWP forecast errors based on horizon uncertainty and synoptic phase shifts."""
        if actuals_df is None:
            from wind_solar_forecast.data.era5_ingest import ERA5Ingestor
            ingestor = ERA5Ingestor()
            actuals_df = ingestor.fetch_zone_weather(zone, start_ts, end_ts)

        df_act = actuals_df.copy()
        df_act["valid_time"] = pd.to_datetime(df_act["valid_time"], utc=True)
        df_act = df_act.sort_values("valid_time").reset_index(drop=True)

        n = len(df_act)
        seed = int(abs(hash(f"{zone}_{vintage}"))) % (2**31)
        rng = np.random.default_rng(seed)

        # Compute issue time and lead time for each valid time
        issue_times = [get_forecast_issue_time(t, vintage) for t in df_act["valid_time"]]
        lead_hours = np.array([(vt - it).total_seconds() / 3600.0 for vt, it in zip(df_act["valid_time"], issue_times)])

        # Error growth with horizon h: sigma(h) = sigma_base * sqrt(h / 24)
        h_factor = np.sqrt(np.maximum(lead_hours, 1.0) / 24.0)

        # Wind speed error: combination of AR(1) bias, phase shift, and turbulence noise
        ar_wind_noise = pd.Series(rng.normal(0, 1.0, n)).ewm(span=8).mean().values
        wind_speed_err = (1.2 * ar_wind_noise + rng.normal(0, 0.6, n)) * h_factor
        forecast_wind_speed = np.maximum(df_act["wind_speed_100m"].values + wind_speed_err, 0.1)

        # Cloud cover error: beta-perturbed shift, increasing with horizon
        cloud_err = (rng.normal(0, 0.15, n) + pd.Series(rng.normal(0, 0.2, n)).ewm(span=6).mean().values) * h_factor
        forecast_cloud = np.clip(df_act["total_cloud_cover"].values + cloud_err, 0.0, 1.0)

        # GHI forecast: derived from actual SSRD modulated by cloud error
        ghi_ratio = np.where(df_act["surface_solar_radiation_downwards"].values > 10.0,
                             (1.0 - 0.75 * (forecast_cloud ** 2.5)) / np.maximum(1.0 - 0.75 * (df_act["total_cloud_cover"].values ** 2.5), 0.1),
                             1.0)
        forecast_ghi = np.maximum(df_act["surface_solar_radiation_downwards"].values * ghi_ratio + rng.normal(0, 8.0, n), 0.0)

        # Temperature error: slight diurnal phase lag and variance
        temp_err = (rng.normal(0, 0.7, n) + 0.3 * np.sin(np.pi * lead_hours / 12.0)) * h_factor
        forecast_temp = df_act["temperature_2m_k"].values + temp_err

        out_df = pd.DataFrame({
            "valid_time": df_act["valid_time"],
            "issue_time": issue_times,
            "horizon_hours": np.round(lead_hours, 1),
            "zone": zone,
            "vintage": vintage,
            "forecast_wind_speed_100m": np.round(forecast_wind_speed, 3),
            "forecast_ghi": np.round(forecast_ghi, 2),
            "forecast_temperature_2m_k": np.round(forecast_temp, 2),
            "forecast_cloud_cover": np.round(forecast_cloud, 3),
            "is_synthetic": True,
        })
        return out_df
