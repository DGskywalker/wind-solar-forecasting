"""Spatial feature engineering: upstream weather advection, spatial gradients, and regional upscaling."""

import numpy as np

from wind_solar_forecast.data.geo import get_zone_geo


def compute_upstream_advection_lag(zone: str, wind_dir_deg: np.ndarray, wind_speed_m_s: np.ndarray) -> np.ndarray:
    """Estimates atmospheric advection delay (in hours) of storm fronts entering from the upwind boundary.

    European wind regimes predominantly advect from the North Sea and Atlantic (240-270 deg).
    For a zone of characteristic spatial length L (e.g. 500 km for DE_LU), advection time is L / v_wind.

    Args:
        zone: Bidding zone code.
        wind_dir_deg: Wind direction in degrees.
        wind_speed_m_s: 100m wind speed in m/s.

    Returns:
        Estimated front arrival time in hours.
    """
    geo = get_zone_geo(zone)
    # Approximate zone diameter in kilometers
    lat_span_km = (geo.lat_max - geo.lat_min) * 111.0
    lon_span_km = (geo.lon_max - geo.lon_min) * 111.0 * np.cos(np.deg2rad(geo.centroid_lat))
    char_length_km = np.sqrt(lat_span_km * lon_span_km)

    # Effective speed in km/h: v * 3.6
    speed_kmh = np.maximum(wind_speed_m_s * 3.6, 5.0)
    advection_hours = char_length_km / speed_kmh
    return np.clip(advection_hours, 1.0, 48.0)


def compute_spatial_gradient(
    u100: np.ndarray,
    v100: np.ndarray,
    lat_span_deg: float,
    lon_span_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Approximates large-scale synoptic velocity gradients (divergence and vorticity proxy).

    Args:
        u100: Zonal wind component.
        v100: Meridional wind component.
        lat_span_deg: Latitude span in degrees.
        lon_span_deg: Longitude span in degrees.

    Returns:
        Tuple of (spatial_gradient_magnitude, flow_divergence_proxy).
    """
    du_dx = np.gradient(u100) / max(lon_span_deg, 0.1)
    dv_dy = np.gradient(v100) / max(lat_span_deg, 0.1)

    divergence = du_dx + dv_dy
    gradient_mag = np.sqrt(du_dx**2 + dv_dy**2)
    return gradient_mag, divergence
