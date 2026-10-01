"""Data ingestion pipeline script for weather reanalysis, NWP forecasts, and power market data."""

import argparse
from pathlib import Path

import pandas as pd

from wind_solar_forecast.data.entsoe_ingest import ENTSOEIngestor
from wind_solar_forecast.data.era5_ingest import ERA5Ingestor
from wind_solar_forecast.data.openmeteo_ingest import OpenMeteoIngestor
from wind_solar_forecast.utils.io import get_duckdb_connection
from wind_solar_forecast.utils.logging import logger


def run_ingestion_pipeline(
    zone: str = "DE_LU",
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-08",
    vintages: list[str] | None = None,
    output_base_dir: str = "data/raw"
) -> dict[str, pd.DataFrame]:
    """Runs data ingestion for weather, NWP forecasts, and electricity market actuals.

    Args:
        zone: European bidding zone.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        vintages: List of forecast vintage tags.
        output_base_dir: Base directory for raw parquet data.

    Returns:
        Dictionary of ingested DataFrames.
    """
    if vintages is None:
        vintages = ["D-1_12:00", "D-2_12:00", "ID_4h"]

    logger.info("Starting data ingestion pipeline", zone=zone, start=start_date, end=end_date, vintages=vintages)

    # 1. Weather reanalysis (ERA5)
    era5_ingestor = ERA5Ingestor(output_dir=Path(output_base_dir) / "weather")
    weather_df = era5_ingestor.fetch_zone_weather(zone, start_date, end_date)
    logger.info("Ingested ERA5 weather reanalysis", zone=zone, rows=len(weather_df))

    # 2. NWP forecast vintages (Open-Meteo)
    nwp_ingestor = OpenMeteoIngestor(output_dir=Path(output_base_dir) / "nwp_forecasts")
    nwp_dfs: list[pd.DataFrame] = []
    for vintage in vintages:
        v_df = nwp_ingestor.fetch_zone_nwp_forecast(
            zone=zone,
            start_date=start_date,
            end_date=end_date,
            vintage=vintage,
            weather_actuals_df=weather_df
        )
        nwp_dfs.append(v_df)
        logger.info("Ingested NWP forecast vintage", zone=zone, vintage=vintage, rows=len(v_df))

    all_nwp_df = pd.concat(nwp_dfs, ignore_index=True)

    # 3. Market data (ENTSO-E actuals)
    entsoe_ingestor = ENTSOEIngestor(output_dir=Path(output_base_dir) / "market")
    market_df = entsoe_ingestor.fetch_zone_market_data(
        zone=zone,
        start_date=start_date,
        end_date=end_date,
        weather_df=weather_df
    )
    logger.info("Ingested ENTSO-E market data", zone=zone, rows=len(market_df))

    # 4. Mirror to DuckDB
    con = get_duckdb_connection("data/processed/power_market.duckdb")
    con.execute("CREATE TABLE IF NOT EXISTS weather_raw AS SELECT * FROM weather_df;")
    con.execute("CREATE TABLE IF NOT EXISTS nwp_raw AS SELECT * FROM all_nwp_df;")
    con.execute("CREATE TABLE IF NOT EXISTS market_raw AS SELECT * FROM market_df;")
    con.close()
    logger.info("Mirrored raw data into DuckDB analytics database")

    return {
        "weather": weather_df,
        "nwp": all_nwp_df,
        "market": market_df
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest weather, forecast, and market data.")
    parser.add_argument("--zone", type=str, default="DE_LU", help="Bidding zone")
    parser.add_argument("--sample", action="store_true", help="Run 1-week sample for verification")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    if args.sample:
        start = "2023-01-01"
        end = "2023-01-08"
    else:
        start = args.start or "2022-01-01"
        end = args.end or "2023-12-31"

    run_ingestion_pipeline(zone=args.zone, start_date=start, end_date=end)


if __name__ == "__main__":
    main()
