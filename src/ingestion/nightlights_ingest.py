from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Tuple

from .gee_utils import bbox_to_ee_geometry, export_image_to_geotiff, init_ee
from .metadata import append_metadata, build_metadata
from .paths import RAW_ROOT, ensure_dir


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def month_range(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    ym_start = start.year * 12 + (start.month - 1)
    ym_end = end.year * 12 + (end.month - 1)
    out: list[tuple[int, int]] = []
    for ym in range(ym_start, ym_end + 1):
        year, month = divmod(ym, 12)
        out.append((year, month + 1))
    return out


def _month_start_end(year: int, month: int) -> tuple[str, str]:
    start = dt.date(year, month, 1)
    if month == 12:
        end = dt.date(year + 1, 1, 1)
    else:
        end = dt.date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def fetch_nightlights_monthly(
    city: str,
    bbox: Tuple[float, float, float, float],
    start_date: dt.date,
    end_date: dt.date,
    scale: int,
    skip_existing: bool = True,
) -> list[Path]:
    ee = init_ee()
    region = bbox_to_ee_geometry(ee, bbox)
    city_key = city.lower().replace(" ", "_")
    out_files: list[Path] = []

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    months_list = month_range(start_date, end_date)
    total = len(months_list)
    for idx, (year, month) in enumerate(months_list, 1):
        ym = f"{year:04d}-{month:02d}"
        print(f"[Nightlights] {ym} ({idx}/{total}) ...", flush=True)
        start_iso, end_iso = _month_start_end(year, month)
        collection = viirs.filterBounds(region).filterDate(start_iso, end_iso).select(["avg_rad"])
        count = int(collection.size().getInfo())
        if count <= 0:
            print(f"[Nightlights] {ym} skip (no scenes)", flush=True)
            continue

        out_dir = ensure_dir(RAW_ROOT / "nightlights" / f"{year:04d}" / f"{month:02d}")
        out_path = out_dir / f"viirs_monthly_{city_key}_{year:04d}{month:02d}.tif"
        if skip_existing and out_path.exists():
            print(f"[Nightlights] {ym} skip (exists)", flush=True)
            out_files.append(out_path)
            continue
        monthly = collection.mean().clip(region).toFloat()
        export_image_to_geotiff(ee, monthly, region, out_path, scale=scale, crs="EPSG:4326")
        out_files.append(out_path)
        print(f"[Nightlights] {ym} done", flush=True)

    return out_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VIIRS monthly night-light GeoTIFFs via GEE.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat in EPSG:4326")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download even if output exists")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    files = fetch_nightlights_monthly(
        args.city, bbox, start_date, end_date, args.scale, skip_existing=not args.no_skip_existing
    )
    meta = build_metadata(
        source="nightlights",
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        files=files,
        crs="EPSG:4326",
        notes=f"VIIRS monthly night-light composites from GEE (scale={args.scale}m).",
    )
    append_metadata(meta)
    print(f"wrote {len(files)} nightlights monthly GeoTIFFs")


if __name__ == "__main__":
    main()
