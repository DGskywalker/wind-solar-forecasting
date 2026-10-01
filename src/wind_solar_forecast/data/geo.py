"""Geographical definitions, bounding boxes, bidding zone definitions, and spatial aggregation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BiddingZoneGeo:
    zone: str
    country: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    centroid_lat: float
    centroid_lon: float
    pv_tilt_deg: float
    pv_azimuth_deg: float
    offshore_eligible: bool


ZONE_GEOGRAPHY: dict[str, BiddingZoneGeo] = {
    "DE_LU": BiddingZoneGeo(
        zone="DE_LU",
        country="Germany",
        lat_min=47.27,
        lat_max=55.06,
        lon_min=5.87,
        lon_max=15.04,
        centroid_lat=51.1657,
        centroid_lon=10.4515,
        pv_tilt_deg=35.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "FR": BiddingZoneGeo(
        zone="FR",
        country="France",
        lat_min=42.33,
        lat_max=51.09,
        lon_min=-4.79,
        lon_max=8.23,
        centroid_lat=46.6034,
        centroid_lon=1.8883,
        pv_tilt_deg=32.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "ES": BiddingZoneGeo(
        zone="ES",
        country="Spain",
        lat_min=36.00,
        lat_max=43.79,
        lon_min=-9.30,
        lon_max=3.32,
        centroid_lat=40.4637,
        centroid_lon=-3.7492,
        pv_tilt_deg=30.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=False,
    ),
    "GB": BiddingZoneGeo(
        zone="GB",
        country="United Kingdom",
        lat_min=49.90,
        lat_max=58.70,
        lon_min=-7.50,
        lon_max=1.80,
        centroid_lat=55.3781,
        centroid_lon=-3.4360,
        pv_tilt_deg=38.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "NL": BiddingZoneGeo(
        zone="NL",
        country="Netherlands",
        lat_min=50.75,
        lat_max=53.55,
        lon_min=3.36,
        lon_max=7.22,
        centroid_lat=52.1326,
        centroid_lon=5.2913,
        pv_tilt_deg=35.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "BE": BiddingZoneGeo(
        zone="BE",
        country="Belgium",
        lat_min=49.50,
        lat_max=51.51,
        lon_min=2.54,
        lon_max=6.41,
        centroid_lat=50.8503,
        centroid_lon=4.3517,
        pv_tilt_deg=35.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "DK_1": BiddingZoneGeo(
        zone="DK_1",
        country="Denmark",
        lat_min=54.55,
        lat_max=57.75,
        lon_min=8.08,
        lon_max=10.62,
        centroid_lat=56.2639,
        centroid_lon=9.5018,
        pv_tilt_deg=40.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "DK_2": BiddingZoneGeo(
        zone="DK_2",
        country="Denmark",
        lat_min=54.50,
        lat_max=56.20,
        lon_min=10.90,
        lon_max=12.70,
        centroid_lat=55.4038,
        centroid_lon=11.7914,
        pv_tilt_deg=40.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "PL": BiddingZoneGeo(
        zone="PL",
        country="Poland",
        lat_min=49.00,
        lat_max=54.83,
        lon_min=14.12,
        lon_max=24.15,
        centroid_lat=51.9194,
        centroid_lon=19.1451,
        pv_tilt_deg=35.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=False,
    ),
    "IT_NORD": BiddingZoneGeo(
        zone="IT_NORD",
        country="Italy",
        lat_min=44.00,
        lat_max=47.10,
        lon_min=6.60,
        lon_max=13.90,
        centroid_lat=45.4642,
        centroid_lon=9.1900,
        pv_tilt_deg=30.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=False,
    ),
    "SE_3": BiddingZoneGeo(
        zone="SE_3",
        country="Sweden",
        lat_min=58.00,
        lat_max=61.00,
        lon_min=11.00,
        lon_max=19.00,
        centroid_lat=59.3293,
        centroid_lon=18.0686,
        pv_tilt_deg=42.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "AT": BiddingZoneGeo(
        zone="AT",
        country="Austria",
        lat_min=46.37,
        lat_max=49.02,
        lon_min=9.53,
        lon_max=17.16,
        centroid_lat=47.5162,
        centroid_lon=14.5501,
        pv_tilt_deg=35.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=False,
    ),
    "PT": BiddingZoneGeo(
        zone="PT",
        country="Portugal",
        lat_min=36.96,
        lat_max=42.15,
        lon_min=-9.50,
        lon_max=-6.19,
        centroid_lat=39.3999,
        centroid_lon=-8.2245,
        pv_tilt_deg=32.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
    "IE_SEM": BiddingZoneGeo(
        zone="IE_SEM",
        country="Ireland",
        lat_min=51.40,
        lat_max=55.40,
        lon_min=-10.60,
        lon_max=-5.40,
        centroid_lat=53.4129,
        centroid_lon=-8.2439,
        pv_tilt_deg=36.0,
        pv_azimuth_deg=180.0,
        offshore_eligible=True,
    ),
}


def compute_latitude_weights(lats: np.ndarray) -> np.ndarray:
    """Computes cosine-latitude weights for area-preserving spatial averaging.

    w_i = cos(lat_i * pi / 180) / sum(cos(lat_j * pi / 180))

    Args:
        lats: Array of latitude coordinates in degrees.

    Returns:
        Normalized spatial weights summing to 1.0.
    """
    weights = np.cos(np.deg2rad(lats))
    weights = np.maximum(weights, 0.0)
    return weights / np.sum(weights)


def get_zone_geo(zone: str) -> BiddingZoneGeo:
    """Retrieves spatial metadata for a given European bidding zone."""
    if zone not in ZONE_GEOGRAPHY:
        raise KeyError(f"Unknown bidding zone: {zone}. Supported: {list(ZONE_GEOGRAPHY.keys())}")
    return ZONE_GEOGRAPHY[zone]
