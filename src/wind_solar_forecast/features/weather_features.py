"""Atmospheric feature extraction: shear exponents, air density, wind power density, and humidity."""

import numpy as np
import pandas as pd


def compute_air_density(pressure_pa: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """Computes moist-air density using the ideal gas law: rho = P / (R_spec * T).

    Args:
        pressure_pa: Barometric pressure in Pascal.
        temp_k: Air temperature in Kelvin.

    Returns:
        Air density in kg/m³.
    """
    R_SPECIFIC = 287.058  # J/(kg*K) for dry air
    return pressure_pa / (R_SPECIFIC * temp_k)


def compute_wind_power_density(wind_speed: np.ndarray, air_density: np.ndarray) -> np.ndarray:
    """Computes instantaneous wind power density: WPD = 0.5 * rho * v^3.

    Args:
        wind_speed: Wind speed in m/s.
        air_density: Air density in kg/m³.

    Returns:
        Wind power density in W/m².
    """
    return 0.5 * air_density * (wind_speed ** 3.0)


def compute_wind_shear_exponent(wind_speed_100m: np.ndarray, wind_speed_10m: np.ndarray) -> np.ndarray:
    """Computes vertical wind shear power-law exponent alpha: v2/v1 = (z2/z1)^alpha.

    Args:
        wind_speed_100m: Wind speed at 100m in m/s.
        wind_speed_10m: Wind speed at 10m in m/s.

    Returns:
        Power-law shear exponent alpha (typically 0.10 to 0.40).
    """
    ratio = np.maximum(wind_speed_100m, 0.1) / np.maximum(wind_speed_10m, 0.1)
    height_ratio = 100.0 / 10.0
    alpha = np.log(ratio) / np.log(height_ratio)
    return np.clip(alpha, 0.0, 0.60)


def build_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes full suite of physical atmospheric features from raw meteorological columns.

    Args:
        df: DataFrame containing meteorological reanalysis or forecast columns.

    Returns:
        DataFrame enriched with physical weather features.
    """
    out = df.copy()

    ws100_col = "forecast_wind_speed_100m" if "forecast_wind_speed_100m" in out.columns else "wind_speed_100m"
    temp_col = "forecast_temperature_2m_k" if "forecast_temperature_2m_k" in out.columns else "temperature_2m_k"
    p_col = "surface_pressure_pa" if "surface_pressure_pa" in out.columns else None

    # Air density
    if p_col and p_col in out.columns:
        p = out[p_col].values
    else:
        p = np.full(len(out), 101325.0)

    rho = compute_air_density(p, out[temp_col].values)
    out["air_density_kg_m3"] = np.round(rho, 3)

    # Wind power density
    wpd = compute_wind_power_density(out[ws100_col].values, rho)
    out["wind_power_density_w_m2"] = np.round(wpd, 1)

    # Wind shear exponent if 10m is available
    if "wind_speed_10m" in out.columns:
        out["wind_shear_alpha"] = np.round(
            compute_wind_shear_exponent(out[ws100_col].values, out["wind_speed_10m"].values), 3
        )

    # Temperature in Celsius
    out["temperature_2m_c"] = np.round(out[temp_col] - 273.15, 2)

    # Wind direction sine and cosine if direction exists
    if "wind_direction_100m" in out.columns:
        rad = np.deg2rad(out["wind_direction_100m"].values)
        out["wind_dir_sin"] = np.round(np.sin(rad), 4)
        out["wind_dir_cos"] = np.round(np.cos(rad), 4)

    return out
