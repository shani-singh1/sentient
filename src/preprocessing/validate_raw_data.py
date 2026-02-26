from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from statistics import median
from typing import Any

import rasterio

from src.common.paths import RAW_ROOT, RESULTS_ROOT, ensure_dir


MONTHLY_SOURCES = ["era5", "sentinel1", "sentinel2", "landsat", "nightlights"]
ANNUAL_SOURCES = ["population"]
STATIC_SOURCES = ["osm"]


def month_range(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    ym_start = start.year * 12 + (start.month - 1)
    ym_end = end.year * 12 + (end.month - 1)
    out: list[tuple[int, int]] = []
    for ym in range(ym_start, ym_end + 1):
        year, month = divmod(ym, 12)
        out.append((year, month + 1))
    return out


def expected_monthly_files(city_key: str, year: int, month: int) -> dict[str, Path]:
    return {
        "era5": RAW_ROOT / "era5" / f"{year:04d}" / f"{month:02d}" / f"era5_{city_key}_{year:04d}{month:02d}.nc",
        "sentinel1": RAW_ROOT / "sentinel1" / f"{year:04d}" / f"{month:02d}" / f"sentinel1_compact_{city_key}_{year:04d}{month:02d}.tif",
        "sentinel2": RAW_ROOT / "sentinel2" / f"{year:04d}" / f"{month:02d}" / f"sentinel2_compact_{city_key}_{year:04d}{month:02d}.tif",
        "landsat": RAW_ROOT / "landsat" / f"{year:04d}" / f"{month:02d}" / f"landsat_composite_{city_key}_{year:04d}{month:02d}.tif",
        "nightlights": RAW_ROOT / "nightlights" / f"{year:04d}" / f"{month:02d}" / f"viirs_monthly_{city_key}_{year:04d}{month:02d}.tif",
    }


def expected_annual_files(city_key: str, year: int) -> dict[str, Path]:
    return {
        "population": RAW_ROOT / "population" / f"{year:04d}" / "01" / f"worldpop_{city_key}_{year:04d}.tif",
    }


def _raster_summary(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        return {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "crs": ds.crs.to_string() if ds.crs else None,
            "xres": float(ds.res[0]),
            "yres": float(ds.res[1]),
        }


def validate(
    city: str,
    start_date: dt.date,
    end_date: dt.date,
    expected_crs: str,
    fail_on_warning: bool,
    allow_missing_monthly: set[str],
) -> dict[str, Any]:
    city_key = city.lower().replace(" ", "_")
    months = month_range(start_date, end_date)
    years = list(range(start_date.year, end_date.year + 1))

    errors: list[str] = []
    warnings: list[str] = []
    raster_checks: dict[str, list[dict[str, Any]]] = {k: [] for k in ["sentinel1", "sentinel2", "landsat", "nightlights", "population"]}
    existing_counts: dict[str, int] = {k: 0 for k in MONTHLY_SOURCES + ANNUAL_SOURCES + STATIC_SOURCES}

    # Monthly and ERA5 expected files
    for year, month in months:
        for source, path in expected_monthly_files(city_key, year, month).items():
            if path.exists():
                existing_counts[source] += 1
                if path.suffix.lower() == ".tif":
                    try:
                        raster_checks[source].append(_raster_summary(path))
                    except Exception as exc:
                        errors.append(f"{source} raster unreadable: {path} ({exc})")
            else:
                msg = f"missing {source} file: {path}"
                if source in allow_missing_monthly:
                    warnings.append(msg)
                else:
                    errors.append(msg)

    # Annual population
    for year in years:
        for source, path in expected_annual_files(city_key, year).items():
            if path.exists():
                existing_counts[source] += 1
                try:
                    raster_checks[source].append(_raster_summary(path))
                except Exception as exc:
                    errors.append(f"{source} raster unreadable: {path} ({exc})")
            else:
                errors.append(f"missing {source} file: {path}")

    # OSM static presence check
    osm_files = sorted((RAW_ROOT / "osm").rglob(f"osm_roads_{city_key}_*.json")) if (RAW_ROOT / "osm").exists() else []
    if osm_files:
        existing_counts["osm"] = len(osm_files)
    else:
        errors.append(f"missing osm roads file under {RAW_ROOT / 'osm'}")

    # CRS and shape/resolution consistency checks
    for source, entries in raster_checks.items():
        if not entries:
            continue
        widths = [e["width"] for e in entries]
        heights = [e["height"] for e in entries]
        xres = [abs(e["xres"]) for e in entries]
        yres = [abs(e["yres"]) for e in entries]
        crs_values = sorted({e["crs"] for e in entries})

        if any(c != expected_crs for c in crs_values):
            errors.append(f"{source} CRS mismatch: expected {expected_crs}, found {crs_values}")

        median_w = median(widths)
        median_h = median(heights)
        median_x = median(xres)
        median_y = median(yres)

        for idx, w in enumerate(widths):
            if abs(w - median_w) > max(10, int(0.15 * median_w)):
                warnings.append(f"{source} width outlier at index {idx}: {w} vs median {median_w}")
        for idx, h in enumerate(heights):
            if abs(h - median_h) > max(10, int(0.15 * median_h)):
                warnings.append(f"{source} height outlier at index {idx}: {h} vs median {median_h}")
        for idx, r in enumerate(xres):
            if abs(r - median_x) > max(0.00001, 0.35 * median_x):
                warnings.append(f"{source} xres outlier at index {idx}: {r} vs median {median_x}")
        for idx, r in enumerate(yres):
            if abs(r - median_y) > max(0.00001, 0.35 * median_y):
                warnings.append(f"{source} yres outlier at index {idx}: {r} vs median {median_y}")

    if fail_on_warning and warnings:
        errors.extend([f"[warning-as-error] {w}" for w in warnings])

    status = "passed" if not errors else "failed"
    return {
        "city": city,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "status": status,
        "expected_crs": expected_crs,
        "existing_counts": existing_counts,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw data completeness and raster consistency before preprocessing.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--expected-crs", default="EPSG:4326")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument(
        "--allow-missing-monthly",
        default="landsat",
        help="Comma-separated monthly sources that may have documented data gaps (default: landsat).",
    )
    parser.add_argument("--output", default=str(RESULTS_ROOT / "raw_data_validation.json"))
    args = parser.parse_args()

    allow_missing = {s.strip() for s in args.allow_missing_monthly.split(",") if s.strip()}
    payload = validate(
        city=args.city,
        start_date=dt.date.fromisoformat(args.start_date),
        end_date=dt.date.fromisoformat(args.end_date),
        expected_crs=args.expected_crs,
        fail_on_warning=args.fail_on_warning,
        allow_missing_monthly=allow_missing,
    )
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
