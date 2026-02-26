from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Tuple

import cdsapi

from .metadata import append_metadata, build_metadata
from .paths import RAW_ROOT, ensure_dir, get_secret


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def month_range(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples between start and end."""
    ym_start = start.year * 12 + (start.month - 1)
    ym_end = end.year * 12 + (end.month - 1)
    out: list[tuple[int, int]] = []
    for ym in range(ym_start, ym_end + 1):
        year, month = divmod(ym, 12)
        out.append((year, month + 1))
    return out


def build_cds_client() -> cdsapi.Client:
    """
    Build a CDS API client using the CDS_API_KEY environment variable if present.

    This avoids the need for a ~/.cdsapirc file when running inside this project.
    """

    api_key = get_secret("CDS_API_KEY")
    api_url = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")

    if api_key:
        return cdsapi.Client(url=api_url, key=api_key)
    # Fallback to standard configuration discovery (~/.cdsapirc or CDSAPI_* env)
    return cdsapi.Client()


def fetch_era5_monthly(
    city: str,
    bbox: Tuple[float, float, float, float],
    start_date: dt.date,
    end_date: dt.date,
    variables: list[str],
) -> list[Path]:
    """
    Download ERA5 monthly means to NetCDF files.

    We deliberately do not post-process; raw NetCDF is stored.
    """

    client = build_cds_client()
    west, south, east, north = bbox

    out_files: list[Path] = []
    for year, month in month_range(start_date, end_date):
        target_dir = ensure_dir(RAW_ROOT / "era5" / f"{year:04d}" / f"{month:02d}")
        target = target_dir / f"era5_{city.lower().replace(' ', '_')}_{year:04d}{month:02d}.nc"

        # ERA5 API expects N,W,S,E
        area = [north, west, south, east]

        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "time": "00:00",
                "format": "netcdf",
                "area": area,
            },
            str(target),
        )
        out_files.append(target)

    return out_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5 monthly climate data for a city/ROI.")
    parser.add_argument("--city", required=True, help="Human-readable city name (for metadata only).")
    parser.add_argument(
        "--bbox",
        required=True,
        help="Bounding box as 'min_lon,min_lat,max_lon,max_lat' in WGS84.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD). Will be truncated to month.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD). Will be truncated to month.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["total_precipitation", "2m_temperature"],
        help="ERA5 variable names to request.",
    )

    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    files = fetch_era5_monthly(args.city, bbox, start_date, end_date, args.variables)

    meta = build_metadata(
        source="era5",
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        files=files,
        crs="EPSG:4326",
        notes="ERA5 monthly means via CDS API.",
    )
    append_metadata(meta)


if __name__ == "__main__":
    main()

