"""Regime-specific analysis: Dunkelflaute duration, wind ramp rates, heat domes, and negative prices."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DunkelflauteEvent:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: int
    mean_wind_cf: float
    mean_solar_cf: float
    energy_deficit_mwh: float


def detect_dunkelflaute_events(
    df: pd.DataFrame,
    installed_capacity_mw: float,
    wind_cf_col: str = "actual_wind_onshore_cf",
    solar_cf_col: str = "actual_solar_cf",
    time_col: str = "valid_time",
    min_duration_hours: int = 6
) -> list[DunkelflauteEvent]:
    """Identifies and characterizes contiguous Dunkelflaute events (wind CF < 10% & solar CF < 5%)."""
    wind = df[wind_cf_col].values
    solar = df[solar_cf_col].values
    times = pd.to_datetime(df[time_col], utc=True).values

    is_low = (wind < 0.10) & (solar < 0.05)
    events: list[DunkelflauteEvent] = []

    in_event = False
    start_idx = 0

    for i in range(len(df)):
        if is_low[i] and not in_event:
            in_event = True
            start_idx = i
        elif not is_low[i] and in_event:
            in_event = False
            duration = i - start_idx
            if duration >= min_duration_hours:
                sub_w = wind[start_idx:i]
                sub_s = solar[start_idx:i]
                # Baseline expected generation ~ 25% capacity
                shortfall = np.sum(0.25 - (sub_w + sub_s)) * installed_capacity_mw
                events.append(
                    DunkelflauteEvent(
                        start_time=pd.Timestamp(times[start_idx]),
                        end_time=pd.Timestamp(times[i - 1]),
                        duration_hours=duration,
                        mean_wind_cf=float(np.mean(sub_w)),
                        mean_solar_cf=float(np.mean(sub_s)),
                        energy_deficit_mwh=float(max(shortfall, 0.0))
                    )
                )

    if in_event and (len(df) - start_idx >= min_duration_hours):
        duration = len(df) - start_idx
        events.append(
            DunkelflauteEvent(
                start_time=pd.Timestamp(times[start_idx]),
                end_time=pd.Timestamp(times[-1]),
                duration_hours=duration,
                mean_wind_cf=float(np.mean(wind[start_idx:])),
                mean_solar_cf=float(np.mean(solar[start_idx:])),
                energy_deficit_mwh=float(max(np.sum(0.25 - (wind[start_idx:] + solar[start_idx:])) * installed_capacity_mw, 0.0))
            )
        )

    return events


def analyze_wind_ramps(
    actual_cf: np.ndarray,
    forecast_cf: np.ndarray,
    ramp_threshold: float = 0.15
) -> dict[str, float]:
    """Quantifies ramp-rate detection accuracy and phase shift under fast meteorological frontal passages."""
    actual_ramp_1h = np.diff(actual_cf)
    forecast_ramp_1h = np.diff(forecast_cf)

    large_up_ramps = actual_ramp_1h > ramp_threshold
    large_down_ramps = actual_ramp_1h < -ramp_threshold

    captured_up = np.sum(large_up_ramps & (forecast_ramp_1h > 0.08))
    total_up = np.sum(large_up_ramps)
    hit_rate_up = float(captured_up / total_up) if total_up > 0 else 1.0

    captured_down = np.sum(large_down_ramps & (forecast_ramp_1h < -0.08))
    total_down = np.sum(large_down_ramps)
    hit_rate_down = float(captured_down / total_down) if total_down > 0 else 1.0

    return {
        "total_up_ramps": int(total_up),
        "total_down_ramps": int(total_down),
        "up_ramp_hit_rate": np.round(hit_rate_up, 4),
        "down_ramp_hit_rate": np.round(hit_rate_down, 4),
        "ramp_rmse": float(np.sqrt(np.mean((actual_ramp_1h - forecast_ramp_1h) ** 2)))
    }
