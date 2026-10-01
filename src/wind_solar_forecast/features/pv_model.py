"""PVLib-based solar PV capacity factor physical model with POA transposition and thermal derating."""

import numpy as np
import pandas as pd
import pvlib

from wind_solar_forecast.data.geo import get_zone_geo


def compute_pv_capacity_factor(
    timestamps: pd.DatetimeIndex,
    ghi: np.ndarray,
    temp_2m_k: np.ndarray,
    total_cloud_cover: np.ndarray,
    zone: str = "DE_LU"
) -> np.ndarray:
    """Calculates physical Solar PV Capacity Factor (CF) using plane-of-array transposition and cell temperature.

    Args:
        timestamps: Array or index of UTC timestamps.
        ghi: Global Horizontal Irradiance in W/m².
        temp_2m_k: 2m ambient air temperature in Kelvin.
        total_cloud_cover: Cloud fraction [0, 1].
        zone: Bidding zone code (for latitude/longitude and standard tilt/azimuth).

    Returns:
        Array of solar capacity factors in [0.0, 1.0].
    """
    geo = get_zone_geo(zone)
    dt_index = pd.DatetimeIndex(timestamps)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize("UTC")

    # Solar position kinematics
    solpos = pvlib.solarposition.get_solarposition(
        time=dt_index,
        latitude=geo.centroid_lat,
        longitude=geo.centroid_lon
    )
    apparent_zenith = solpos["apparent_zenith"].values
    apparent_azimuth = solpos["azimuth"].values

    # Clearsky DNI/DHI partition (Erbs or simple Perez approximation)
    cos_zenith = np.maximum(np.cos(np.deg2rad(apparent_zenith)), 0.0)
    dni_est = np.where(cos_zenith > 0.05, np.maximum(ghi - 0.25 * ghi, 0.0) / (cos_zenith + 0.05), 0.0)
    dhi_est = np.maximum(ghi - dni_est * cos_zenith, 0.0)

    # Transposition to Plane of Array (POA)
    surface_tilt = geo.pv_tilt_deg
    surface_azimuth = geo.pv_azimuth_deg

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=apparent_zenith,
        solar_azimuth=apparent_azimuth,
        dni=dni_est,
        ghi=ghi,
        dhi=dhi_est
    )
    raw_poa = poa["poa_global"]
    if hasattr(raw_poa, "values"):
        raw_poa = raw_poa.values
    poa_global = np.maximum(np.nan_to_num(raw_poa, nan=0.0), 0.0)

    # Thermal derating: cell temperature (Faiman model)
    temp_c = temp_2m_k - 273.15
    # Cell temp rises above ambient with irradiance: Tcell = Tamb + (POA / (u0 + u1*ws))
    # Standard open rack: u0 ~ 25 W/(m²K)
    cell_temp = temp_c + (poa_global / 25.0)

    # Power output with standard monocrystalline silicon temp coeff (-0.4% per deg C above 25C)
    gamma_pmp = -0.004
    thermal_derate = 1.0 + gamma_pmp * (cell_temp - 25.0)

    # AC capacity factor (Reference STC = 1000 W/m², DC/AC inverter sizing ratio 1.20)
    dc_power_ratio = (poa_global / 1000.0) * np.clip(thermal_derate, 0.70, 1.15)
    ac_cf = np.clip(dc_power_ratio, 0.0, 1.0)

    # Zero out night hours where zenith >= 90
    ac_cf = np.where(apparent_zenith >= 90.0, 0.0, ac_cf)
    return np.round(np.clip(ac_cf, 0.0, 0.98), 4)
