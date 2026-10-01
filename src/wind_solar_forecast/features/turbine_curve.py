"""Turbine power curve models and regional portfolio aggregation functions."""

import numpy as np


def vestas_v112_power_curve(wind_speed: np.ndarray | float) -> np.ndarray:
    """Computes normalized capacity factor for Vestas V112-3.45 MW onshore turbine.

    Specifications:
        Cut-in: 3.0 m/s
        Rated: 12.0 m/s
        Cut-out: 25.0 m/s

    Args:
        wind_speed: Wind speed at 100m hub height in m/s.

    Returns:
        Capacity factor in [0.0, 1.0].
    """
    ws = np.asarray(wind_speed, dtype=float)
    cf = np.zeros_like(ws)

    # Region II (cubic aerodynamic power extraction)
    mask_ramp = (ws >= 3.0) & (ws < 12.0)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / (12.0 - 3.0)) ** 3.0

    # Region III (rated power output)
    mask_rated = (ws >= 12.0) & (ws <= 25.0)
    cf[mask_rated] = 1.0

    # Region IV (high-wind cut-out shutdown)
    cf[ws > 25.0] = 0.0

    return np.clip(cf, 0.0, 1.0)


def siemens_sg_80_167_power_curve(wind_speed: np.ndarray | float) -> np.ndarray:
    """Computes normalized capacity factor for Siemens Gamesa SG 8.0-167 DD offshore turbine.

    Specifications:
        Cut-in: 3.0 m/s
        Rated: 12.5 m/s
        Cut-out: 25.0 m/s (with storm ramp-down ride-through to 28.0 m/s)

    Args:
        wind_speed: Wind speed at 120-130m hub height in m/s.

    Returns:
        Capacity factor in [0.0, 1.0].
    """
    ws = np.asarray(wind_speed, dtype=float)
    cf = np.zeros_like(ws)

    # Region II
    mask_ramp = (ws >= 3.0) & (ws < 12.5)
    cf[mask_ramp] = ((ws[mask_ramp] - 3.0) / (12.5 - 3.0)) ** 2.9

    # Region III
    mask_rated = (ws >= 12.5) & (ws <= 25.0)
    cf[mask_rated] = 1.0

    # High wind storm ride-through (linear derating from 25 m/s to 28 m/s)
    mask_storm = (ws > 25.0) & (ws <= 28.0)
    cf[mask_storm] = 1.0 - (ws[mask_storm] - 25.0) / 3.0

    cf[ws > 28.0] = 0.0
    return np.clip(cf, 0.0, 1.0)


def aggregate_zone_wind_cf(
    wind_speeds: np.ndarray,
    is_offshore: bool = False,
    spatial_dispersion_sigma: float = 1.5
) -> np.ndarray:
    """Applies regional smoothing convolution representing portfolio dispersion across turbines.

    Due to spatial dispersion across hundreds of wind parks in a bidding zone,
    aggregate power curves exhibit softer transitions than single-turbine curves
    (Norgaard & Holttinen 2004).

    Args:
        wind_speeds: Gridded or zonal mean 100m wind speed in m/s.
        is_offshore: Whether evaluating offshore (SG 8.0-167) or onshore (Vestas V112).
        spatial_dispersion_sigma: Standard deviation of local wind speed variability within the zone.

    Returns:
        Smoothed zonal wind capacity factor array.
    """
    ws = np.asarray(wind_speeds, dtype=float)
    curve_func = siemens_sg_80_167_power_curve if is_offshore else vestas_v112_power_curve

    # Fast 5-point Gauss-Hermite integration around mean speed
    offsets = np.array([-1.8, -0.9, 0.0, 0.9, 1.8]) * spatial_dispersion_sigma
    weights = np.array([0.06, 0.24, 0.40, 0.24, 0.06])

    smoothed_cf = np.zeros_like(ws)
    for offset, weight in zip(offsets, weights):
        perturbed_ws = np.maximum(ws + offset, 0.0)
        smoothed_cf += weight * curve_func(perturbed_ws)

    return np.clip(smoothed_cf, 0.0, 1.0)
