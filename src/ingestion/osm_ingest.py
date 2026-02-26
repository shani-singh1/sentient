from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Tuple

import requests

from .metadata import append_metadata, build_metadata
from .paths import RAW_ROOT, ensure_dir

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Parse `min_lon,min_lat,max_lon,max_lat` into floats.
    """

    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def build_query(bbox: Tuple[float, float, float, float]) -> str:
    """
    Build a simple Overpass query for road geometries.

    We request all highway ways within the bounding box.
    """

    min_lon, min_lat, max_lon, max_lat = bbox
    return f"""
    [out:json][timeout:180];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._;>;);
    out body;
    """


def fetch_osm_roads(city: str, bbox: Tuple[float, float, float, float], start_date: dt.date, end_date: dt.date) -> Path:
    """
    Fetch OSM road network for the ROI and persist raw JSON.

    OSM is effectively "timeless" for this purpose, but we still log
    the requested time window for reproducibility.
    """

    query = build_query(bbox)
    response = requests.post(OVERPASS_URL, data={"data": query})
    response.raise_for_status()

    year = start_date.year
    month = start_date.month
    out_dir = ensure_dir(RAW_ROOT / "osm" / f"{year:04d}" / f"{month:02d}")
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    safe_city = city.lower().replace(" ", "_")
    out_path = out_dir / f"osm_roads_{safe_city}_{timestamp}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(response.json(), f)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OpenStreetMap road network for a given city/ROI.")
    parser.add_argument("--city", required=True, help="Human-readable city name (for metadata only).")
    parser.add_argument(
        "--bbox",
        required=True,
        help="Bounding box as 'min_lon,min_lat,max_lon,max_lat' in WGS84.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD). Used for logging only.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD). Used for logging only.",
    )

    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    out_file = fetch_osm_roads(args.city, bbox, start_date, end_date)

    meta = build_metadata(
        source="osm_roads",
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        files=[out_file],
        crs="EPSG:4326",  # OSM/Overpass uses WGS84
        notes="Raw OSM road network from Overpass API.",
    )
    append_metadata(meta)


if __name__ == "__main__":
    main()

