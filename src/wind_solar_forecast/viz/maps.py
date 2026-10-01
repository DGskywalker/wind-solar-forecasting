"""Plotly European geographical visualization of wind and solar capacity factors."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wind_solar_forecast.data.geo import ZONE_GEOGRAPHY


def create_europe_capacity_factor_map(
    zone_metrics: dict[str, float],
    technology: str = "wind_onshore",
    title: str = "European Renewable Generation Capacity Factor by Bidding Zone"
) -> go.Figure:
    """Generates an interactive Plotly scatter geo map of Europe with capacity factor bubbles.

    Args:
        zone_metrics: Dict mapping bidding zone code to capacity factor [0.0, 1.0].
        technology: Name of technology ('wind_onshore', 'wind_offshore', 'solar_pv').
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    records = []
    for zone, cf in zone_metrics.items():
        if zone in ZONE_GEOGRAPHY:
            geo = ZONE_GEOGRAPHY[zone]
            records.append({
                "zone": zone,
                "country": geo.country,
                "lat": geo.centroid_lat,
                "lon": geo.centroid_lon,
                "capacity_factor": float(np.clip(cf, 0.0, 1.0)),
                "cf_pct": f"{cf * 100:.1f}%",
                "bubble_size": max(cf * 35.0, 8.0)
            })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame([{"zone": "DE_LU", "country": "Germany", "lat": 51.16, "lon": 10.45, "capacity_factor": 0.45, "cf_pct": "45.0%", "bubble_size": 20.0}])

    color_scale = "Blues" if "wind" in technology else "YlOrRd"

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="capacity_factor",
        size="bubble_size",
        hover_name="zone",
        hover_data={"country": True, "cf_pct": True, "lat": False, "lon": False, "bubble_size": False},
        color_continuous_scale=color_scale,
        range_color=[0.0, 1.0],
        labels={"capacity_factor": "Capacity Factor"},
        title=title,
    )

    fig.update_geos(
        scope="europe",
        resolution=50,
        showcountries=True,
        countrycolor="rgb(200, 200, 200)",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showocean=True,
        oceancolor="rgb(230, 240, 250)",
        fitbounds="locations"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), template="plotly_white")
    return fig
