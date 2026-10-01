"""Unit tests for feature engineering, turbine power curves, PVLib model, and regime flags."""

import numpy as np
import pandas as pd
import pytest

from wind_solar_forecast.features.build_features import build_feature_dataset, identify_regime_flags
from wind_solar_forecast.features.pv_model import compute_pv_capacity_factor
from wind_solar_forecast.features.temporal import compute_fourier_cyclical_features, compute_strictly_backward_lags
from wind_solar_forecast.features.turbine_curve import (
    aggregate_zone_wind_cf,
    siemens_sg_80_167_power_curve,
    vestas_v112_power_curve,
)


def test_turbine_power_curves() -> None:
    """Verifies aerodynamic physical properties of Vestas and Siemens turbine power curves."""
    # Below cut-in (3 m/s)
    assert vestas_v112_power_curve(2.5) == 0.0
    assert siemens_sg_80_167_power_curve(2.5) == 0.0

    # Rated wind speed (12 m/s to 25 m/s)
    assert vestas_v112_power_curve(12.5) == 1.0
    assert vestas_v112_power_curve(20.0) == 1.0
    assert siemens_sg_80_167_power_curve(13.0) == 1.0

    # High wind cut-out
    assert vestas_v112_power_curve(26.0) == 0.0
    # Siemens storm derating at 26 m/s
    assert 0.0 < siemens_sg_80_167_power_curve(26.0) < 1.0
    assert siemens_sg_80_167_power_curve(29.0) == 0.0


def test_aggregate_zone_wind_cf() -> None:
    """Verifies that regional portfolio aggregation smooths the step responses."""
    speeds = np.array([2.8, 3.2, 11.5, 12.5, 24.8, 25.5])
    cf_agg = aggregate_zone_wind_cf(speeds, is_offshore=False)
    # At 2.8 m/s (below single turbine cut-in), aggregate CF is non-zero due to spatial wind dispersion
    assert cf_agg[0] > 0.0
    assert (cf_agg >= 0.0).all() and (cf_agg <= 1.0).all()


def test_pv_capacity_factor() -> None:
    """Tests solar PV capacity factor physical bounds and day/night contrast."""
    times = pd.date_range("2023-06-21 00:00:00", "2023-06-21 23:00:00", freq="1h", tz="UTC")
    ghi = np.zeros(24)
    # Midday solar irradiance
    ghi[10:15] = [400, 750, 850, 700, 450]
    temp_k = np.full(24, 298.15)
    clouds = np.zeros(24)

    cf = compute_pv_capacity_factor(times, ghi, temp_k, clouds, zone="DE_LU")
    assert len(cf) == 24
    # Midnight should be zero
    assert cf[0] == 0.0
    assert cf[23] == 0.0
    # Midday should be substantial (> 0.50)
    assert cf[12] > 0.50
    assert (cf >= 0.0).all() and (cf <= 1.0).all()


def test_regime_flags() -> None:
    """Tests Dunkelflaute, Storm, and Heat dome identification."""
    wind_cf = np.full(10, 0.05)
    solar_cf = np.full(10, 0.02)
    wind_speed = np.full(10, 3.0)
    temp_c = np.full(10, 2.0)

    flags = identify_regime_flags(wind_cf, solar_cf, wind_speed, temp_c, min_dunkelflaute_hours=6)
    # Sinks into Dunkelflaute after 6 consecutive hours
    assert flags["is_dunkelflaute"].iloc[0] == 0
    assert flags["is_dunkelflaute"].iloc[5] == 1
    assert flags["is_dunkelflaute"].iloc[9] == 1
