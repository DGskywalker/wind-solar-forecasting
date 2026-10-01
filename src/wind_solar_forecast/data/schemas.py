"""Pydantic v2 data contracts and validation schemas for weather, generation, and market feeds."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class TechnologyType(str, Enum):
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    SOLAR_PV = "solar_pv"


class ForecastVintage(str, Enum):
    D_2_1200 = "D-2_12:00"
    D_1_0000 = "D-1_00:00"
    D_1_1200 = "D-1_12:00"
    ID_4H = "ID_4h"


class WeatherObservation(BaseModel):
    """Schema for atmospheric reanalysis observations (ERA5)."""
    model_config = ConfigDict(extra="ignore")

    valid_time: datetime = Field(..., description="Target validity time in UTC")
    zone: str = Field(..., description="European bidding zone code, e.g. DE_LU")
    wind_speed_100m: float = Field(..., ge=0.0, le=100.0, description="100m wind speed in m/s")
    wind_direction_100m: float = Field(..., ge=0.0, le=360.0, description="Wind direction in degrees [0, 360)")
    wind_speed_10m: float = Field(..., ge=0.0, le=80.0, description="10m wind speed in m/s")
    surface_solar_radiation_downwards: float = Field(..., ge=0.0, le=1400.0, description="GHI in W/m²")
    surface_net_solar_radiation: float = Field(..., ge=0.0, le=1400.0, description="Net surface solar in W/m²")
    temperature_2m_k: float = Field(..., ge=220.0, le=335.0, description="2m air temperature in Kelvin")
    total_cloud_cover: float = Field(..., ge=0.0, le=1.0, description="Fractional cloud cover [0, 1]")
    surface_pressure_pa: float = Field(..., ge=80000.0, le=110000.0, description="Surface barometric pressure in Pa")


class NWPForecastPoint(BaseModel):
    """Schema for Numerical Weather Prediction forecast points."""
    model_config = ConfigDict(extra="ignore")

    issue_time: datetime = Field(..., description="Timestamp when model run was published (vintage cut-off)")
    valid_time: datetime = Field(..., description="Validity timestamp in UTC")
    zone: str = Field(..., description="Bidding zone")
    vintage: ForecastVintage = Field(..., description="Forecast vintage cycle")
    horizon_hours: float = Field(..., ge=0.0, description="Lead time in hours (valid_time - issue_time)")
    forecast_wind_speed_100m: float = Field(..., ge=0.0, le=100.0)
    forecast_ghi: float = Field(..., ge=0.0, le=1400.0)
    forecast_temperature_2m_k: float = Field(..., ge=220.0, le=335.0)
    forecast_cloud_cover: float = Field(..., ge=0.0, le=1.0)


class PowerGenerationRecord(BaseModel):
    """Schema for actual generation per production type from ENTSO-E."""
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(..., description="Valid hour start in UTC")
    zone: str = Field(..., description="Bidding zone code")
    wind_onshore_mw: float = Field(..., ge=0.0, description="Onshore wind generation in MW")
    wind_offshore_mw: float = Field(default=0.0, ge=0.0, description="Offshore wind generation in MW")
    solar_pv_mw: float = Field(..., ge=0.0, description="Solar PV generation in MW")
    total_load_mw: float = Field(..., ge=0.0, description="Total grid electrical load in MW")


class MarketPriceRecord(BaseModel):
    """Schema for wholesale electricity prices and fundamentals."""
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(..., description="Valid hour start in UTC")
    zone: str = Field(..., description="Bidding zone code")
    day_ahead_price_eur_mwh: float = Field(..., description="Day-Ahead clearing price in EUR/MWh")
    intraday_price_eur_mwh: float = Field(..., description="Volume-weighted Intraday price in EUR/MWh")
    gas_ttf_eur_mwh: float = Field(default=35.0, ge=0.0, description="Dutch TTF natural gas price in EUR/MWh")
    eua_carbon_eur_ton: float = Field(default=75.0, ge=0.0, description="EU ETS carbon allowance in EUR/tCO2")


class InstalledCapacityRecord(BaseModel):
    """Schema for installed nameplate generation capacity."""
    model_config = ConfigDict(extra="ignore")

    year: int = Field(..., ge=2015, le=2035)
    zone: str = Field(..., description="Bidding zone")
    wind_onshore_gw: float = Field(..., ge=0.0)
    wind_offshore_gw: float = Field(default=0.0, ge=0.0)
    solar_pv_gw: float = Field(..., ge=0.0)


class FeatureManifestEntry(BaseModel):
    """Schema for cataloging feature store variables, lineage, and leakage safety."""
    model_config = ConfigDict(extra="ignore")

    feature_name: str
    dtype: str
    source_table: str
    vintage_availability: str
    leakage_risk_tag: str = Field(..., description="'SAFE_AT_GATE_CLOSURE' or 'TARGET_CONTEMPORANEOUS'")
    description: str
