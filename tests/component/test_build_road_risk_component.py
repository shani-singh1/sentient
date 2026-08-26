"""Component tests for src/features/build_road_risk.py: the CLI entry point.

Builds a tiny synthetic OSM extract, tile risk score table, and feature
dataset, then runs build_road_risk.main() end to end and inspects the two
artifacts it writes: road_risk_ranking.json and dashboard.json.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features import build_road_risk as brr

pytestmark = pytest.mark.component

CITY = "testcity"
BBOX = [10.0, 10.0, 10.2, 10.2]  # small synthetic bounding box, degrees
MONTHS = [f"2024-{m:02d}" for m in range(1, 9)]  # 8 months


def _osm_payload() -> dict:
    # Two ways forming an "L" shape, long enough to clear min-length-m, each
    # crossing a couple of tiles so tile-to-road aggregation has real work to do.
    nodes = [
        {"type": "node", "id": 1, "lat": 10.19, "lon": 10.01},
        {"type": "node", "id": 2, "lat": 10.15, "lon": 10.01},
        {"type": "node", "id": 3, "lat": 10.11, "lon": 10.01},
        {"type": "node", "id": 4, "lat": 10.05, "lon": 10.15},
        {"type": "node", "id": 5, "lat": 10.02, "lon": 10.18},
    ]
    ways = [
        {"type": "way", "id": 100, "nodes": [1, 2, 3], "tags": {"name": "North Main Road", "highway": "primary"}},
        {"type": "way", "id": 101, "nodes": [4, 5], "tags": {"highway": "residential"}},  # Unnamed Road
    ]
    return {"elements": nodes + ways}


def _tile_ids_for_bbox() -> list[str]:
    # The 4x4 grid over BBOX; used to synthesize tile-level risk scores.
    return [f"{CITY}__tile_{i:02d}_{j:02d}" for i in range(4) for j in range(4)]


@pytest.fixture()
def road_risk_workspace(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    results_root = tmp_path / "results"
    features_root = tmp_path / "features"
    monkeypatch.setattr(brr, "RAW_ROOT", raw_root)
    monkeypatch.setattr(brr, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(brr, "FEATURES_ROOT", features_root)
    monkeypatch.setitem(brr.BBOX_BY_CITY, CITY, BBOX)

    osm_dir = raw_root / "osm"
    osm_dir.mkdir(parents=True)
    osm_path = osm_dir / f"osm_roads_{CITY}_20240101T000000.json"
    osm_path.write_text(json.dumps(_osm_payload()), encoding="utf-8")

    tiles = _tile_ids_for_bbox()
    rng_seed = 0
    risk_rows = []
    dataset_rows = []
    for month_i, month in enumerate(MONTHS):
        for tile_i, tile_id in enumerate(tiles):
            rng_seed += 1
            # Tile 0 trends upward over time (will approach the critical band);
            # every other tile stays low and flat.
            risk = 0.1 + 0.09 * month_i if tile_i == 0 else 0.05 + (rng_seed % 3) * 0.01
            risk_rows.append({"tile_id": tile_id, "target_month": month, "risk_score": risk, "model_name": "test_model"})
            dataset_rows.append(
                {
                    "tile_id": tile_id,
                    "target_month": month,
                    "stress_accum_rain_3m": risk * 2,
                    "stress_accum_flood_3m": risk,
                    "stress_accum_heat_3m": risk * 0.5,
                    "s2_ndwi_mean_lag0": 0.1,
                    "nightlights_mean_lag0": 0.2,
                    "population_mean_lag0": 0.3,
                    "s2_ndvi_mean_trend": -0.01,
                }
            )

    results_root.mkdir(parents=True)
    risk_path = results_root / "risk_scores.parquet"
    pd.DataFrame(risk_rows).to_parquet(risk_path, index=False)

    features_root.mkdir(parents=True)
    pd.DataFrame(dataset_rows).to_parquet(features_root / "dataset.parquet", index=False)

    return {"results_root": results_root, "risk_path": risk_path}


def _run_main(monkeypatch, max_roads: int = 10) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["build_road_risk.py", "--city", "Testcity", "--max-roads", str(max_roads), "--min-length-m", "10"],
    )
    brr.main()


def test_main_writes_road_ranking_with_expected_road_count(road_risk_workspace, monkeypatch):
    _run_main(monkeypatch)

    ranking = json.loads((road_risk_workspace["results_root"] / "road_risk_ranking.json").read_text(encoding="utf-8"))
    assert len(ranking["roads"]) == 2  # both synthetic ways clear the length threshold
    names = {r["name"] for r in ranking["roads"]}
    assert names == {"North Main Road", "Unnamed Road"}


def test_every_road_has_a_valid_tier_and_bounded_priority_score(road_risk_workspace, monkeypatch):
    _run_main(monkeypatch)

    ranking = json.loads((road_risk_workspace["results_root"] / "road_risk_ranking.json").read_text(encoding="utf-8"))
    for road in ranking["roads"]:
        assert road["risk_level"] in {"High", "Medium", "Low"}
        assert 0 <= road["priority"] <= 100
        assert road["rank"] >= 1
        assert road["city_rank"] >= 1


def test_dashboard_payload_has_one_tile_series_per_grid_cell(road_risk_workspace, monkeypatch):
    _run_main(monkeypatch)

    dashboard = json.loads((road_risk_workspace["results_root"] / "dashboard.json").read_text(encoding="utf-8"))
    city_payload = dashboard["cities"][CITY]
    assert len(city_payload["tiles"]) == 16  # 4x4 grid
    assert dashboard["months"] == MONTHS
    for tile in city_payload["tiles"]:
        assert len(tile["series"]) == len(MONTHS)


def test_dashboard_summary_percent_high_is_between_zero_and_hundred(road_risk_workspace, monkeypatch):
    _run_main(monkeypatch)

    dashboard = json.loads((road_risk_workspace["results_root"] / "dashboard.json").read_text(encoding="utf-8"))
    summary = dashboard["cities"][CITY]["summary"]
    assert 0.0 <= summary["pct_high"] <= 100.0
    assert summary["roads_analyzed"] == 2


def test_road_crossing_the_trending_tile_gets_a_months_to_critical_projection(road_risk_workspace, monkeypatch):
    _run_main(monkeypatch)

    dashboard = json.loads((road_risk_workspace["results_root"] / "dashboard.json").read_text(encoding="utf-8"))
    roads = dashboard["roads"]
    # North Main Road runs through the north-west corner, which is tile 0 -
    # the one engineered to trend upward toward the critical band.
    north_main = next(r for r in roads if r["name"] == "North Main Road")
    assert any(v is not None for v in north_main["trend"])
    assert north_main["action"]  # a recommended action string is always present


def test_main_raises_for_city_with_no_configured_bounding_box(road_risk_workspace, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_road_risk.py", "--city", "Nowhereland", "--max-roads", "5"])

    with pytest.raises(ValueError):
        brr.main()
