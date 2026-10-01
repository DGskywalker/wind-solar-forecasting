"""Temporal feature engineering: Fourier harmonics, holiday calendars, DST awareness, and backward lags."""

import numpy as np
import pandas as pd

from wind_solar_forecast.utils.timeutils import is_dst_transition


def get_fixed_holidays(year: int) -> set[tuple[int, int]]:
    """Common fixed European national public holidays (month, day)."""
    return {
        (1, 1),   # New Year's Day
        (5, 1),   # Labour Day
        (8, 15),  # Assumption of Mary
        (10, 3),  # German Unity Day / early autumn
        (11, 1),  # All Saints' Day
        (12, 25), # Christmas Day
        (12, 26), # St. Stephen's / Boxing Day
    }


def compute_fourier_cyclical_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Computes continuous diurnal and annual Fourier harmonic representations.

    Args:
        timestamps: DatetimeIndex of timestamps.

    Returns:
        DataFrame with sin/cos components for hour, day of week, and day of year.
    """
    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    doy = timestamps.dayofyear.values

    # Diurnal (24h period)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # Weekly (7 day period)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # Annual (365.25 day period)
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame({
        "hour_sin": np.round(hour_sin, 4),
        "hour_cos": np.round(hour_cos, 4),
        "dow_sin": np.round(dow_sin, 4),
        "dow_cos": np.round(dow_cos, 4),
        "doy_sin": np.round(doy_sin, 4),
        "doy_cos": np.round(doy_cos, 4),
        "is_weekend": (dow >= 5).astype(int),
    })


def compute_holiday_and_dst_flags(timestamps: pd.DatetimeIndex, zone: str = "DE_LU") -> pd.DataFrame:
    """Extracts public holiday flags and Daylight Saving Time transition markers."""
    holidays_flag = []
    dst_flag = []

    for ts in timestamps:
        fixed = get_fixed_holidays(ts.year)
        is_hol = 1 if (ts.month, ts.day) in fixed else 0
        holidays_flag.append(is_hol)
        dst_flag.append(is_dst_transition(ts, zone))

    return pd.DataFrame({
        "is_holiday": np.array(holidays_flag, dtype=int),
        "dst_transition_flag": np.array(dst_flag, dtype=int),
    })


def compute_strictly_backward_lags(
    series: pd.Series,
    lags: list[int] | None = None,
    prefix: str = "lag"
) -> pd.DataFrame:
    """Constructs strictly backward-looking lag features to ensure zero future leakage.

    Args:
        series: Pandas Series indexed by time.
        lags: List of lag periods in hours (e.g. [1, 2, 3, 24, 48, 168]).
        prefix: Column naming prefix.

    Returns:
        DataFrame containing lagged columns.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]

    df_lags = pd.DataFrame(index=series.index)
    for lag in lags:
        df_lags[f"{prefix}_{lag}h"] = series.shift(lag)

    return df_lags


def compute_ramp_features(series: pd.Series, windows: list[int] | None = None, prefix: str = "ramp") -> pd.DataFrame:
    """Computes backward-looking power ramp rate changes: delta_y_t = y_t - y_{t-k}."""
    if windows is None:
        windows = [1, 3, 6]

    df_ramps = pd.DataFrame(index=series.index)
    for w in windows:
        df_ramps[f"{prefix}_{w}h"] = series.diff(w)

    return df_ramps
