"""
Build road-level risk by overlaying OSM roads on tile risk.
Output: ranked roads with geometry for map display.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.common.paths import RAW_ROOT, RESULTS_ROOT, ensure_dir


BBOX = [77.45, 12.8, 77.75, 13.1]  # west, south, east, north
GRID_SIZE = 4


def point_to_tile(lon: float, lat: float) -> str:
    west, south, east, north = BBOX
    j = max(0, min(GRID_SIZE - 1, int((lon - west) / (east - west) * GRID_SIZE)))
    i = max(0, min(GRID_SIZE - 1, int((north - lat) / (north - south) * GRID_SIZE)))
    return f"tile_{i:02d}_{j:02d}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2
    a += math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build road-level risk from OSM + tile risk.")
    parser.add_argument("--city", default="Bengaluru")
    parser.add_argument("--risk-scores", default=str(RESULTS_ROOT / "risk_scores.parquet"))
    parser.add_argument("--osm", default=None, help="Path to OSM JSON; auto-detect if not set")
    parser.add_argument("--max-roads", type=int, default=500)
    parser.add_argument("--min-length-m", type=float, default=50)
    args = parser.parse_args()

    city_key = args.city.lower().replace(" ", "_")
    osm_path = args.osm
    if not osm_path:
        osm_dir = RAW_ROOT / "osm"
        files = sorted(osm_dir.rglob(f"osm_roads_{city_key}_*.json"))
        if not files:
            raise FileNotFoundError(f"No OSM files for {city_key}")
        osm_path = files[-1]
    osm_path = Path(osm_path)

    risk_df = pd.read_parquet(args.risk_scores)
    tile_risk = risk_df.groupby("tile_id", as_index=False)["risk_score"].mean()
    tile_risk_map = dict(zip(tile_risk["tile_id"].astype(str), tile_risk["risk_score"]))

    with open(osm_path, encoding="utf-8") as f:
        osm = json.load(f)

    nodes: dict[int, tuple[float, float]] = {}
    for el in osm.get("elements", []):
        if el.get("type") == "node" and "id" in el and "lat" in el and "lon" in el:
            nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))

    roads: list[dict] = []
    for el in osm.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("ref") or "Unnamed Road"
        highway = tags.get("highway", "road")
        node_ids = el.get("nodes") or []
        if len(node_ids) < 2:
            continue

        path: list[tuple[float, float]] = []
        length_m = 0.0
        tiles_seen: set[str] = set()
        for i, nid in enumerate(node_ids):
            coord = nodes.get(int(nid))
            if coord is None:
                continue
            lat, lon = coord
            path.append((lon, lat))
            if i > 0:
                prev = nodes.get(int(node_ids[i - 1]))
                if prev:
                    length_m += haversine_m(prev[0], prev[1], lat, lon)
            tiles_seen.add(point_to_tile(lon, lat))

        if length_m < args.min_length_m or len(path) < 2:
            continue

        risks = [tile_risk_map[t] for t in tiles_seen if t in tile_risk_map]
        risk_score = float(sum(risks) / len(risks)) if risks else 0.0

        roads.append({
            "name": str(name),
            "highway": str(highway),
            "path": path,
            "length_m": round(length_m, 1),
            "risk_score": round(risk_score, 4),
            "risk_pct": round(risk_score * 100, 1),
        })

    # Assign risk level and zone (quadrant) for geographic spread
    west, south, east, north = BBOX
    for r in roads:
        if r["risk_score"] >= 0.7:
            r["risk_level"] = "High"
        elif r["risk_score"] >= 0.4:
            r["risk_level"] = "Medium"
        else:
            r["risk_level"] = "Low"
        # Zone from centroid (NW, NE, SW, SE)
        cx = sum(p[0] for p in r["path"]) / len(r["path"])
        cy = sum(p[1] for p in r["path"]) / len(r["path"])
        zone_x = "E" if cx > (west + east) / 2 else "W"
        zone_y = "N" if cy > (south + north) / 2 else "S"
        r["zone"] = zone_y + zone_x

    # Stratified sampling: ensure High, Medium, Low from across the city
    named = [r for r in roads if r["name"] != "Unnamed Road"]
    unnamed = [r for r in roads if r["name"] == "Unnamed Road"]
    high = [r for r in named + unnamed if r["risk_level"] == "High"]
    med = [r for r in named + unnamed if r["risk_level"] == "Medium"]
    low = [r for r in named + unnamed if r["risk_level"] == "Low"]
    # Sort each by risk (high desc, low asc for variety)
    high = sorted(high, key=lambda r: r["risk_score"], reverse=True)
    med = sorted(med, key=lambda r: r["risk_score"], reverse=True)
    low = sorted(low, key=lambda r: r["risk_score"], reverse=True)
    # Take ~40% high (priority), ~35% medium, ~25% low for geographic/risk diversity
    n_high = min(len(high), max(args.max_roads // 2, args.max_roads - 150))
    n_med = min(len(med), (args.max_roads - n_high) // 2)
    n_low = min(len(low), args.max_roads - n_high - n_med)
    roads = (high[:n_high] + med[:n_med] + low[:n_low])
    # Final sort: high first, then medium, then low; within tier by risk
    roads = sorted(roads, key=lambda r: (-(1 if r["risk_level"] == "High" else 0.5 if r["risk_level"] == "Medium" else 0), -r["risk_score"]))

    for i, r in enumerate(roads):
        r["rank"] = i + 1

    out_dir = ensure_dir(RESULTS_ROOT)
    out_path = out_dir / "road_risk_ranking.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"roads": roads, "source": str(osm_path), "tile_risk_source": args.risk_scores}, f, indent=2)

    print(f"Wrote {len(roads)} roads to {out_path}")


if __name__ == "__main__":
    main()
