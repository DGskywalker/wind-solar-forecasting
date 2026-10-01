"""Installed capacity time series and interpolation per European bidding zone."""

import pandas as pd

# Historical and operational nameplate installed capacity benchmarks (in GW)
# Sources: ENTSO-E SO&AF, IRENA Renewable Capacity Statistics, Agora Energiewende
CAPACITY_BENCHMARKS: dict[str, dict[int, dict[str, float]]] = {
    "DE_LU": {
        2021: {"wind_onshore": 53.8, "wind_offshore": 7.8, "solar_pv": 59.2},
        2022: {"wind_onshore": 55.9, "wind_offshore": 8.1, "solar_pv": 67.4},
        2023: {"wind_onshore": 58.5, "wind_offshore": 8.5, "solar_pv": 82.0},
        2024: {"wind_onshore": 61.0, "wind_offshore": 8.9, "solar_pv": 95.0},
    },
    "FR": {
        2021: {"wind_onshore": 18.7, "wind_offshore": 0.0, "solar_pv": 13.2},
        2022: {"wind_onshore": 20.1, "wind_offshore": 0.5, "solar_pv": 15.8},
        2023: {"wind_onshore": 21.8, "wind_offshore": 1.5, "solar_pv": 18.5},
        2024: {"wind_onshore": 23.2, "wind_offshore": 2.0, "solar_pv": 21.0},
    },
    "ES": {
        2021: {"wind_onshore": 28.1, "wind_offshore": 0.0, "solar_pv": 15.3},
        2022: {"wind_onshore": 29.8, "wind_offshore": 0.0, "solar_pv": 20.1},
        2023: {"wind_onshore": 30.8, "wind_offshore": 0.0, "solar_pv": 25.5},
        2024: {"wind_onshore": 32.0, "wind_offshore": 0.0, "solar_pv": 30.0},
    },
    "GB": {
        2021: {"wind_onshore": 14.2, "wind_offshore": 11.3, "solar_pv": 13.9},
        2022: {"wind_onshore": 14.6, "wind_offshore": 13.8, "solar_pv": 14.5},
        2023: {"wind_onshore": 15.0, "wind_offshore": 14.7, "solar_pv": 15.6},
        2024: {"wind_onshore": 15.5, "wind_offshore": 15.5, "solar_pv": 16.8},
    },
    "NL": {
        2021: {"wind_onshore": 5.3, "wind_offshore": 2.5, "solar_pv": 14.9},
        2022: {"wind_onshore": 6.1, "wind_offshore": 2.6, "solar_pv": 19.6},
        2023: {"wind_onshore": 6.8, "wind_offshore": 4.7, "solar_pv": 23.9},
        2024: {"wind_onshore": 7.4, "wind_offshore": 5.5, "solar_pv": 27.5},
    },
    "DK_1": {
        2021: {"wind_onshore": 4.4, "wind_offshore": 1.7, "solar_pv": 1.7},
        2022: {"wind_onshore": 4.6, "wind_offshore": 2.3, "solar_pv": 2.6},
        2023: {"wind_onshore": 4.8, "wind_offshore": 2.3, "solar_pv": 3.7},
        2024: {"wind_onshore": 5.0, "wind_offshore": 2.7, "solar_pv": 4.5},
    },
    "PL": {
        2021: {"wind_onshore": 7.1, "wind_offshore": 0.0, "solar_pv": 7.7},
        2022: {"wind_onshore": 8.2, "wind_offshore": 0.0, "solar_pv": 12.2},
        2023: {"wind_onshore": 9.8, "wind_offshore": 0.0, "solar_pv": 17.1},
        2024: {"wind_onshore": 10.6, "wind_offshore": 0.0, "solar_pv": 20.0},
    },
}


def get_installed_capacity(zone: str, dt: pd.Timestamp) -> dict[str, float]:
    """Retrieves installed capacity (in MW) for a given zone and timestamp with linear interpolation.

    Args:
        zone: Bidding zone code.
        dt: Target timestamp.

    Returns:
        Dictionary mapping technology ('wind_onshore', 'wind_offshore', 'solar_pv') to capacity in MW.
    """
    benchmarks = CAPACITY_BENCHMARKS.get(zone)
    if not benchmarks:
        # Generic default scale
        return {"wind_onshore": 10000.0, "wind_offshore": 2000.0, "solar_pv": 12000.0}

    year = dt.year
    frac_year = year + (dt.dayofyear - 1) / 365.25

    sorted_years = sorted(benchmarks.keys())
    if frac_year <= sorted_years[0]:
        base = benchmarks[sorted_years[0]]
        return {k: v * 1000.0 for k, v in base.items()}
    if frac_year >= sorted_years[-1]:
        base = benchmarks[sorted_years[-1]]
        return {k: v * 1000.0 for k, v in base.items()}

    # Interpolate between y0 and y1
    y0 = int(frac_year)
    y1 = y0 + 1
    weight = frac_year - y0

    caps0 = benchmarks.get(y0, benchmarks[sorted_years[0]])
    caps1 = benchmarks.get(y1, benchmarks[sorted_years[-1]])

    result = {}
    for tech in ["wind_onshore", "wind_offshore", "solar_pv"]:
        gw = (1.0 - weight) * caps0.get(tech, 0.0) + weight * caps1.get(tech, 0.0)
        result[tech] = gw * 1000.0  # Convert GW to MW

    return result
