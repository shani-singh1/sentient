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


def fetch_sentinel2_monthly(
    city: str,
    bbox: Tuple[float, float, float, float],
    start_date: dt.date,
    end_date: dt.date,
    max_cloud: int,
    scale: int,
    skip_existing: bool = True,
) -> list[Path]:
    ee = init_ee()
    region = bbox_to_ee_geometry(ee, bbox)
    city_key = city.lower().replace(" ", "_")
    out_files: list[Path] = []

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    months_list = month_range(start_date, end_date)
    total = len(months_list)
    for idx, (year, month) in enumerate(months_list, 1):
        ym = f"{year:04d}-{month:02d}"
        print(f"[Sentinel2] {ym} ({idx}/{total}) ...", flush=True)
        start_iso, end_iso = _month_start_end(year, month)
        monthly_col = s2.filterDate(start_iso, end_iso).select(["B4", "B3", "B2", "B8"])
        count = int(monthly_col.size().getInfo())
        if count <= 0:
            print(f"[Sentinel2] {ym} skip (no scenes)", flush=True)
            continue

        out_dir = ensure_dir(RAW_ROOT / "sentinel2" / f"{year:04d}" / f"{month:02d}")
        out_path = out_dir / f"sentinel2_compact_{city_key}_{year:04d}{month:02d}.tif"
        if skip_existing and out_path.exists():
            print(f"[Sentinel2] {ym} skip (exists)", flush=True)
            out_files.append(out_path)
            continue
        monthly = monthly_col.median().clip(region).toFloat()
        export_image_to_geotiff(ee, monthly, region, out_path, scale=scale, crs="EPSG:4326")
        out_files.append(out_path)
        print(f"[Sentinel2] {ym} done", flush=True)

    return out_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Sentinel-2 monthly compact GeoTIFFs via GEE.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat in EPSG:4326")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-cloud", type=int, default=40)
    parser.add_argument("--scale", type=int, default=20)
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download even if output exists")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    files = fetch_sentinel2_monthly(
        args.city, bbox, start_date, end_date, args.max_cloud, args.scale,
        skip_existing=not args.no_skip_existing,
    )
    meta = build_metadata(
        source="sentinel2",
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        files=files,
        crs="EPSG:4326",
        notes=f"Compact Sentinel-2 monthly median composites from GEE (max_cloud={args.max_cloud}, scale={args.scale}m).",
    )
    append_metadata(meta)
    print(f"wrote {len(files)} sentinel2 monthly GeoTIFFs")


if __name__ == "__main__":
    main()

