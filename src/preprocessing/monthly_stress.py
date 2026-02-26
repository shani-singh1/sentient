from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.common.paths import PROCESSED_ROOT, RAW_ROOT, ensure_dir


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
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


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def load_osm_road_stats(city: str, start: dt.date, end: dt.date) -> dict[str, float]:
    src_root = RAW_ROOT / "osm"
    if not src_root.exists():
        return {"road_way_count": 0.0, "road_length_km": 0.0}

    city_key = city.lower().replace(" ", "_")
    files = sorted(src_root.rglob(f"osm_roads_{city_key}_*.json"))
    if not files:
        return {"road_way_count": 0.0, "road_length_km": 0.0}

    # Prefer an OSM snapshot within the requested date window; fallback to latest.
    preferred: list[Path] = []
    for path in files:
        try:
            timestamp = path.stem.rsplit("_", 1)[-1]
            stamp = dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").date()
            if start <= stamp <= end:
                preferred.append(path)
        except Exception:
            continue
    latest = preferred[-1] if preferred else files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])

    nodes: dict[int, tuple[float, float]] = {}
    ways: list[list[int]] = []

    for el in elements:
        if el.get("type") == "node" and "id" in el and "lat" in el and "lon" in el:
            nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
        elif el.get("type") == "way" and isinstance(el.get("nodes"), list):
            ways.append([int(nid) for nid in el["nodes"]])

    total_m = 0.0
    for node_ids in ways:
        for i in range(1, len(node_ids)):
            a = nodes.get(node_ids[i - 1])
            b = nodes.get(node_ids[i])
            if a is None or b is None:
                continue
            total_m += _haversine_m(a[0], a[1], b[0], b[1])

    return {"road_way_count": float(len(ways)), "road_length_km": total_m / 1000.0}


def load_era5_monthly(city: str, year: int, month: int) -> dict[str, float | None]:
    nc_path = RAW_ROOT / "era5" / f"{year:04d}" / f"{month:02d}" / f"era5_{city.lower().replace(' ', '_')}_{year:04d}{month:02d}.nc"
    if not nc_path.exists():
        return {
            "era5_total_precipitation_mean": None,
            "era5_total_precipitation_sum": None,
            "era5_2m_temperature_mean": None,
        }

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required for ERA5 preprocessing. Install with: pip install xarray netcdf4") from exc

    ds = _open_era5_dataset(xr, nc_path)
    out: dict[str, float | None] = {
        "era5_total_precipitation_mean": None,
        "era5_total_precipitation_sum": None,
        "era5_2m_temperature_mean": None,
    }

    if "tp" in ds.variables:
        tp = ds["tp"]
        out["era5_total_precipitation_mean"] = float(tp.mean().values)
        out["era5_total_precipitation_sum"] = float(tp.sum().values)
    elif "total_precipitation" in ds.variables:
        tp = ds["total_precipitation"]
        out["era5_total_precipitation_mean"] = float(tp.mean().values)
        out["era5_total_precipitation_sum"] = float(tp.sum().values)

    if "t2m" in ds.variables:
        t2m = ds["t2m"]
        out["era5_2m_temperature_mean"] = float(t2m.mean().values)
    elif "2m_temperature" in ds.variables:
        t2m = ds["2m_temperature"]
        out["era5_2m_temperature_mean"] = float(t2m.mean().values)

    ds.close()
    return out


def _open_era5_dataset(xr: object, path: Path) -> object:
    with path.open("rb") as f:
        magic = f.read(4)

    if magic == b"PK\x03\x04":
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".nc", ".netcdf"))]
            if not members:
                raise RuntimeError(f"ZIP ERA5 file has no .nc member: {path}")
            member = members[0]
            with zf.open(member, "r") as src, tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(src.read())
                tmp_path = Path(tmp.name)
        ds = xr.open_dataset(tmp_path, engine="netcdf4")
        ds.load()
        return ds

    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return xr.open_dataset(path, engine="scipy")


def monthly_tif_path(source: str, prefix: str, city: str, year: int, month: int) -> Path:
    return RAW_ROOT / source / f"{year:04d}" / f"{month:02d}" / f"{prefix}_{city.lower().replace(' ', '_')}_{year:04d}{month:02d}.tif"


def population_path(city: str, year: int) -> Path:
    return RAW_ROOT / "population" / f"{year:04d}" / "01" / f"worldpop_{city.lower().replace(' ', '_')}_{year:04d}.tif"


def month_has_overlap(city: str, year: int, month: int) -> bool:
    era5 = RAW_ROOT / "era5" / f"{year:04d}" / f"{month:02d}" / f"era5_{city.lower().replace(' ', '_')}_{year:04d}{month:02d}.nc"
    s1 = monthly_tif_path("sentinel1", "sentinel1_compact", city, year, month)
    s2 = monthly_tif_path("sentinel2", "sentinel2_compact", city, year, month)
    ls = monthly_tif_path("landsat", "landsat_composite", city, year, month)
    nl = monthly_tif_path("nightlights", "viirs_monthly", city, year, month)
    pop = population_path(city, year)
    return era5.exists() and s1.exists() and s2.exists() and ls.exists() and nl.exists() and pop.exists()


def _tile_bounds(h: int, w: int, tile_y: int, tile_x: int, grid_size: int) -> tuple[int, int, int, int]:
    r0 = (h * tile_y) // grid_size
    r1 = (h * (tile_y + 1)) // grid_size
    c0 = (w * tile_x) // grid_size
    c1 = (w * (tile_x + 1)) // grid_size
    return r0, r1, c0, c1


def _array_stats(arr: np.ndarray) -> dict[str, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"mean": 0.0, "p90": 0.0, "frac_pos": 0.0}
    return {
        "mean": float(np.mean(valid)),
        "p90": float(np.percentile(valid, 90)),
        "frac_pos": float(np.mean(valid > 0)),
    }


def _read_tile(path: Path, tile_y: int, tile_x: int, grid_size: int) -> np.ndarray:
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        r0, r1, c0, c1 = _tile_bounds(h, w, tile_y, tile_x, grid_size)
        win = Window(col_off=c0, row_off=r0, width=max(1, c1 - c0), height=max(1, r1 - r0))
        arr = ds.read(window=win, masked=True).astype(np.float32)
    return np.ma.filled(arr, np.nan)


def _band_or_nan(arr: np.ndarray, band_index: int) -> np.ndarray:
    if arr.ndim < 3 or band_index >= arr.shape[0]:
        if arr.ndim >= 3 and arr.shape[0] > 0:
            shape = arr[0].shape
        else:
            shape = (1, 1)
        return np.full(shape, np.nan, dtype=np.float32)
    return arr[band_index]


def _landsat_thermal_kelvin(thermal: np.ndarray) -> np.ndarray:
    valid = thermal[np.isfinite(thermal)]
    if valid.size == 0:
        return thermal
    if float(np.nanmedian(valid)) > 1000:
        return thermal * 0.00341802 + 149.0
    return thermal


def build_month_tile_rows(city: str, year: int, month: int, road: dict[str, float], era5: dict[str, float | None], grid_size: int) -> list[dict[str, float | int | str | None]]:
    s1_path = monthly_tif_path("sentinel1", "sentinel1_compact", city, year, month)
    s2_path = monthly_tif_path("sentinel2", "sentinel2_compact", city, year, month)
    ls_path = monthly_tif_path("landsat", "landsat_composite", city, year, month)
    nl_path = monthly_tif_path("nightlights", "viirs_monthly", city, year, month)
    pop_path = population_path(city, year)

    rows: list[dict[str, float | int | str | None]] = []
    for ty in range(grid_size):
        for tx in range(grid_size):
            tile_id = f"tile_{ty:02d}_{tx:02d}"

            s1 = _band_or_nan(_read_tile(s1_path, ty, tx, grid_size), 0)
            s2 = _read_tile(s2_path, ty, tx, grid_size)
            ls = _read_tile(ls_path, ty, tx, grid_size)
            nl = _band_or_nan(_read_tile(nl_path, ty, tx, grid_size), 0)
            pop = _band_or_nan(_read_tile(pop_path, ty, tx, grid_size), 0)

            red = _band_or_nan(s2, 0)
            green = _band_or_nan(s2, 1)
            blue = _band_or_nan(s2, 2)
            nir = _band_or_nan(s2, 3)
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)

            s1_stats = _array_stats(s1)
            ndvi_stats = _array_stats(ndvi)
            ndwi_stats = _array_stats(ndwi)

            thermal = _landsat_thermal_kelvin(_band_or_nan(ls, 3))
            thermal_stats = _array_stats(thermal)

            night_stats = _array_stats(nl)
            pop_stats = _array_stats(pop)

            row: dict[str, float | int | str | None] = {
                "city": city,
                "year": year,
                "month": month,
                "month_id": f"{year:04d}-{month:02d}",
                "tile_id": tile_id,
                "road_way_count": road["road_way_count"],
                "road_length_km": road["road_length_km"],
                "s1_backscatter_mean": s1_stats["mean"],
                "s1_backscatter_p90": s1_stats["p90"],
                "s1_flood_fraction": float(np.mean(np.isfinite(s1) & (s1 < -17.0))),
                "s2_ndvi_mean": ndvi_stats["mean"],
                "s2_ndwi_mean": ndwi_stats["mean"],
                "s2_green_p90": float(_array_stats(green)["p90"]),
                "landsat_thermal_mean_k": thermal_stats["mean"],
                "landsat_heat_exposure_fraction": float(np.mean(np.isfinite(thermal) & (thermal > 305.0))),
                "nightlights_mean": night_stats["mean"],
                "nightlights_p90": night_stats["p90"],
                "population_mean": pop_stats["mean"],
                "population_p90": pop_stats["p90"],
                **era5,
            }
            rows.append(row)

    return rows


def write_monthly_processed(city: str, records: list[dict[str, float | int | str | None]]) -> list[Path]:
    out_files: list[Path] = []
    city_dir = ensure_dir(PROCESSED_ROOT / city.lower().replace(" ", "_"))

    by_month: dict[str, list[dict[str, float | int | str | None]]] = {}
    for row in records:
        key = f"{int(row['year']):04d}_{int(row['month']):02d}"
        by_month.setdefault(key, []).append(row)

    for ym, rows in sorted(by_month.items()):
        out_path = city_dir / f"{ym}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        out_files.append(out_path)

    return out_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build monthly processed stress records from raw ingestion outputs.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--grid-size", type=int, default=4, help="Square grid size for tile-level monthly stress summaries.")
    args = parser.parse_args()

    city = args.city
    _ = parse_bbox(args.bbox)
    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)

    road = load_osm_road_stats(city, start, end)
    if road["road_way_count"] <= 0:
        raise RuntimeError("OSM road data not found. Run src.ingestion.osm_ingest first.")

    rows: list[dict[str, float | int | str | None]] = []
    for year, month in month_range(start, end):
        if not month_has_overlap(city, year, month):
            continue

        era5 = load_era5_monthly(city, year, month)
        rows.extend(build_month_tile_rows(city, year, month, road, era5, args.grid_size))

    if not rows:
        raise RuntimeError(
            "No overlapping months found across era5, sentinel1 compact, sentinel2 compact, landsat, nightlights, and population."
        )

    out_files = write_monthly_processed(city, rows)
    tile_count = len({str(r["tile_id"]) for r in rows})
    print(f"wrote {len(out_files)} monthly parquet files with {tile_count} tiles under {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
