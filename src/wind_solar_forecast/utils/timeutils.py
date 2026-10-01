"""Time utilities for UTC-to-local market conversion, DST transitions, and forecast vintage tracking."""

from datetime import datetime

import pandas as pd


def to_utc_datetime(dt: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Converts any date/time representation into a tz-aware UTC pd.Timestamp.

    Args:
        dt: Input timestamp or ISO date string.

    Returns:
        pd.Timestamp localized/converted to UTC.
    """
    ts = pd.to_datetime(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_market_tz(ts: pd.Timestamp, zone: str = "DE_LU") -> pd.Timestamp:
    """Converts UTC timestamp to the local bidding zone timezone (CET/CEST or GMT/BST).

    Args:
        ts: UTC pd.Timestamp.
        zone: European bidding zone code (e.g., 'DE_LU', 'GB', 'ES').

    Returns:
        Timestamp in zone's local civil time.
    """
    if zone in ("GB", "IE_SEM"):
        target_tz = "Europe/London"
    elif zone in ("PT",):
        target_tz = "Europe/Lisbon"
    elif zone in ("DK_1", "DK_2", "SE_3"):
        target_tz = "Europe/Copenhagen"
    else:
        target_tz = "Europe/Berlin"

    utc_ts = to_utc_datetime(ts)
    return utc_ts.tz_convert(target_tz)


def is_dst_transition(ts: pd.Timestamp, zone: str = "DE_LU") -> int:
    """Identifies daylight saving transition days (+1 for 23h spring forward, -1 for 25h fall back, 0 otherwise).

    Args:
        ts: Timestamp to evaluate.
        zone: Bidding zone code.

    Returns:
        Integer indicator: +1 (spring forward), -1 (fall back), 0 (standard day).
    """
    local_ts = to_market_tz(ts, zone)
    start_of_day = local_ts.normalize()
    end_of_day = start_of_day + pd.Timedelta(days=1)
    diff_hours = (end_of_day - start_of_day).total_seconds() / 3600.0

    if diff_hours < 24.0:
        return 1
    if diff_hours > 24.0:
        return -1
    return 0


def get_forecast_issue_time(
    target_time: pd.Timestamp,
    vintage: str = "D-1_12:00"
) -> pd.Timestamp:
    """Computes the exact forecast issue time (vintage cut-off) in UTC.

    For Day-Ahead (D-1_12:00), forecasts are issued at or before 12:00 CET on D-1.
    For D-2_12:00, issued at 12:00 CET on D-2.
    For ID_4h, issued exactly 4 hours prior to target delivery time.

    Args:
        target_time: Delivery/valid timestamp in UTC.
        vintage: Forecast vintage identifier ('D-2_12:00', 'D-1_00:00', 'D-1_12:00', 'ID_4h').

    Returns:
        Issue timestamp in UTC.
    """
    target_utc = to_utc_datetime(target_time)

    if vintage == "ID_4h":
        return target_utc - pd.Timedelta(hours=4)

    # Local midnight of delivery day in Europe/Berlin
    local_target = target_utc.tz_convert("Europe/Berlin")
    delivery_date = local_target.floor("D")

    if vintage == "D-1_12:00":
        # 12:00 CET/CEST on D-1 (day before delivery date)
        issue_local = delivery_date - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    elif vintage == "D-1_00:00":
        # 00:00 CET/CEST on D-1
        issue_local = delivery_date - pd.Timedelta(days=1)
    elif vintage == "D-2_12:00":
        # 12:00 CET/CEST on D-2
        issue_local = delivery_date - pd.Timedelta(days=2) + pd.Timedelta(hours=12)
    else:
        # Default fallback: 24h prior
        return target_utc - pd.Timedelta(hours=24)

    return issue_local.tz_convert("UTC")
