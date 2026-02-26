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


def year_range(start: dt.date, end: dt.date) -> list[int]:
    return list(range(start.year, end.year + 1))


def fetch_population_annual(
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

    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop")
    years_list = year_range(start_date, end_date)
    total = len(years_list)
    for idx, year in enumerate(years_list, 1):
        print(f"[Population] {year} ({idx}/{total}) ...", flush=True)
        yearly = (
            worldpop.filterBounds(region)
            .filterDate(f"{year:04d}-01-01", f"{year + 1:04d}-01-01")
            .select(["population"])
            .mean()
        )
        if yearly.bandNames().size().getInfo() <= 0:
            print(f"[Population] {year} skip (no data)", flush=True)
            continue

        out_dir = ensure_dir(RAW_ROOT / "population" / f"{year:04d}" / "01")
        out_path = out_dir / f"worldpop_{city_key}_{year:04d}.tif"
        if skip_existing and out_path.exists():
            print(f"[Population] {year} skip (exists)", flush=True)
            out_files.append(out_path)
            continue
        export_image_to_geotiff(ee, yearly.clip(region).toFloat(), region, out_path, scale=scale, crs="EPSG:4326")
        out_files.append(out_path)
        print(f"[Population] {year} done", flush=True)

    return out_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest annual population raster via GEE WorldPop.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat in EPSG:4326")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download even if output exists")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    files = fetch_population_annual(
        args.city, bbox, start_date, end_date, args.scale, skip_existing=not args.no_skip_existing
    )
    meta = build_metadata(
        source="population",
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        files=files,
        crs="EPSG:4326",
        notes=f"WorldPop population rasters from GEE (scale={args.scale}m).",
    )
    append_metadata(meta)
    print(f"wrote {len(files)} population annual GeoTIFFs")


if __name__ == "__main__":
    main()
