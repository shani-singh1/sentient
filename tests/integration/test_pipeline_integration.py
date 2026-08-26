"""Integration test for the full offline pipeline: dataset build through
road-level dashboard generation, chained together against synthetic data.

Real satellite data is never required to run this suite. A single
module-scoped fixture builds a small, deterministic "processed monthly
stress" tree and then drives it through every production stage:

    build_dataset.main()
      -> train_models.train_one(mode="baseline")   (a fast real model, the
         same Ridge-on-lag0-features configuration the production baseline
         uses; the full 10-model production sweep is covered by unit tests
         on build_model() and is far too slow to run per test invocation)
      -> score_risk.main()
      -> evaluate.main()
      -> build_road_risk.main()

Each test function below asserts a different contract at the boundary
between two stages, proving the schemas the real pipeline depends on
actually line up end to end.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate as ev
from src.features import build_dataset as bd
from src.features import build_road_risk as brr
from src.inference import score_risk as sr
from src.training import train_models as tm

pytestmark = pytest.mark.integration

CITY = "pipelinecity"
TILES = ["tile_00_00", "tile_00_01", "tile_01_00"]
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)] + ["2024-01"]  # 13 contiguous months
BBOX = [0.0, 0.0, 4.0, 4.0]

# One point per tile, chosen with the exact point_to_tile formula in mind
# (see build_road_risk.point_to_tile) so each way lands in a known tile.
TILE_POINTS = {
    "tile_00_00": (0.5, 3.5),
    "tile_00_01": (1.5, 3.5),
    "tile_01_00": (0.5, 2.5),
}


def _synthetic_processed_frame() -> dict[str, pd.DataFrame]:
    """One dataframe per month, matching the schema monthly_stress.py writes."""
    rng = np.random.default_rng(42)
    by_month: dict[str, list[dict]] = {m: [] for m in MONTHS}
    for month_i, month in enumerate(MONTHS):
        seasonal = 1.0 + 0.5 * np.sin(month_i / 12.0 * 2 * np.pi)
        for tile_i, tile_id in enumerate(TILES):
            tile_bias = tile_i * 0.3
            noise = rng.normal(scale=0.02)
            row = {
                "month_id": month,
                "tile_id": tile_id,
                "road_way_count": 10.0 + tile_i,
                "road_length_km": 5.0 + tile_i,
                "s1_backscatter_mean": -10.0 + noise,
                "s1_backscatter_p90": -5.0 + noise,
                "s1_flood_fraction": max(0.0, 0.05 + tile_bias * seasonal + noise),
                "s2_ndvi_mean": 0.4 - tile_bias * 0.1,
                "s2_ndwi_mean": 0.1 + noise,
                "s2_green_p90": 0.3 + noise,
                "landsat_thermal_mean_k": 300.0 + seasonal,
                "landsat_heat_exposure_fraction": max(0.0, 0.1 + tile_bias * (1.5 - seasonal) + noise),
                "nightlights_mean": 2.0 + tile_i,
                "nightlights_p90": 4.0 + tile_i,
                "population_mean": 100.0 + tile_i * 10,
                "population_p90": 200.0 + tile_i * 10,
                "era5_total_precipitation_mean": max(0.0, 0.02 * seasonal + noise * 0.01),
                "era5_total_precipitation_sum": max(0.0, 0.6 * seasonal + noise * 0.1 + tile_bias),
                "era5_2m_temperature_mean": 295.0 + seasonal,
            }
            by_month[month].append(row)
    return {m: pd.DataFrame(rows) for m, rows in by_month.items()}


def _osm_payload() -> dict:
    nodes = []
    ways = []
    node_id = 1
    for way_id, (tile_id, (lon, lat)) in enumerate(TILE_POINTS.items(), start=500):
        a = {"type": "node", "id": node_id, "lat": lat, "lon": lon}
        b = {"type": "node", "id": node_id + 1, "lat": lat - 0.3, "lon": lon + 0.3}
        nodes.extend([a, b])
        ways.append(
            {
                "type": "way",
                "id": way_id,
                "nodes": [node_id, node_id + 1],
                "tags": {"name": f"Road over {tile_id}", "highway": "residential"},
            }
        )
        node_id += 2
    return {"elements": nodes + ways}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("pipeline_integration")
    processed_root = workspace / "processed"
    features_root = workspace / "features"
    results_root = workspace / "results"
    models_root = workspace / "models"
    raw_root = workspace / "raw"
    for d in (processed_root, features_root, results_root, models_root, raw_root / "osm"):
        d.mkdir(parents=True)

    mp = pytest.MonkeyPatch()
    mp.setattr(bd, "PROCESSED_ROOT", processed_root)
    mp.setattr(bd, "FEATURES_ROOT", features_root)
    mp.setattr(sr, "FEATURES_ROOT", features_root)
    mp.setattr(sr, "RESULTS_ROOT", results_root)
    mp.setattr(ev, "FEATURES_ROOT", features_root)
    mp.setattr(ev, "MODELS_ROOT", models_root)
    mp.setattr(ev, "RESULTS_ROOT", results_root)
    mp.setattr(brr, "RAW_ROOT", raw_root)
    mp.setattr(brr, "RESULTS_ROOT", results_root)
    mp.setattr(brr, "FEATURES_ROOT", features_root)
    mp.setitem(brr.BBOX_BY_CITY, CITY, BBOX)

    try:
        # Stage 1: synthetic processed monthly stress, one parquet per month.
        city_dir = processed_root / CITY
        city_dir.mkdir(parents=True)
        for month, frame in _synthetic_processed_frame().items():
            frame.to_parquet(city_dir / f"{month.replace('-', '_')}.parquet", index=False)

        # Stage 2: feature dataset.
        mp.setattr("sys.argv", ["build_dataset.py", "--city", "Pipelinecity", "--window-size", "3", "--train-fraction", "0.6"])
        bd.main()
        dataset_path = features_root / "dataset.parquet"

        # Stage 3: a single fast real model (not the full production sweep).
        # Tree ensembles need dozens of rows before they can split at all;
        # Ridge on the baseline feature set stays well-behaved at this scale.
        df = pd.read_parquet(dataset_path)
        train_df, val_df = tm.time_split(df, val_fraction=0.3)
        trained = tm.train_one(train_df, val_df, "baseline")
        model_path = models_root / "best_model.joblib"
        joblib.dump({"model": trained["model"], "features": trained["features"], "model_name": "baseline"}, model_path)

        # Stage 4: score every row in the dataset.
        scores_path = results_root / "risk_scores.parquet"
        mp.setattr(
            "sys.argv",
            ["score_risk.py", "--dataset", str(dataset_path), "--model", str(model_path), "--output", str(scores_path)],
        )
        sr.main()

        # Stage 5: offline evaluation against the proxy target.
        mp.setattr(
            "sys.argv",
            ["evaluate.py", "--dataset", str(dataset_path), "--predictions", str(scores_path)],
        )
        ev.main()

        # Stage 6: OSM extract + road-level dashboard.
        osm_path = raw_root / "osm" / f"osm_roads_{CITY}_20240101T000000.json"
        osm_path.write_text(json.dumps(_osm_payload()), encoding="utf-8")
        mp.setattr(
            "sys.argv",
            [
                "build_road_risk.py",
                "--city",
                "Pipelinecity",
                "--risk-scores",
                str(scores_path),
                "--max-roads",
                "10",
                "--min-length-m",
                "10",
            ],
        )
        brr.main()

        yield {
            "dataset_path": dataset_path,
            "scores_path": scores_path,
            "results_root": results_root,
            "model_name": trained["model"],
        }
    finally:
        mp.undo()


def test_dataset_stage_produces_rows_for_every_tile(pipeline):
    df = pd.read_parquet(pipeline["dataset_path"])
    # 13 months, window_size 3 -> 10 windows per tile, 3 tiles = 30 rows.
    assert len(df) == 30
    assert set(df["tile_id"].str.split("__").str[1].unique()) == set(TILES)


def test_score_risk_stage_normalizes_scores_into_unit_interval(pipeline):
    scores = pd.read_parquet(pipeline["scores_path"])
    assert scores["risk_score"].min() >= 0.0
    assert scores["risk_score"].max() <= 1.0
    assert scores["risk_score"].max() == pytest.approx(1.0)  # min-max scaling hits both ends
    assert (scores["model_name"] == "baseline").all()


def test_score_risk_stage_preserves_row_count_from_dataset(pipeline):
    dataset = pd.read_parquet(pipeline["dataset_path"])
    scores = pd.read_parquet(pipeline["scores_path"])
    assert len(scores) == len(dataset)


def test_evaluate_stage_writes_row_count_matching_scored_rows(pipeline):
    payload = json.loads((pipeline["results_root"] / "evaluation.json").read_text(encoding="utf-8"))
    scores = pd.read_parquet(pipeline["scores_path"])
    assert payload["row_count"] == len(scores)
    assert "spearman_risk_vs_target_proxy" in payload
    assert isinstance(payload["overfit_guardrails"], list)


def test_road_risk_stage_produces_a_road_per_synthetic_way(pipeline):
    ranking = json.loads((pipeline["results_root"] / "road_risk_ranking.json").read_text(encoding="utf-8"))
    assert len(ranking["roads"]) == len(TILE_POINTS)
    for road in ranking["roads"]:
        assert road["city"] == CITY


def test_dashboard_stage_trend_series_length_matches_scored_months(pipeline):
    scores = pd.read_parquet(pipeline["scores_path"])
    n_months = scores["target_month"].nunique()
    dashboard = json.loads((pipeline["results_root"] / "dashboard.json").read_text(encoding="utf-8"))
    assert len(dashboard["months"]) == n_months
    for road in dashboard["roads"]:
        assert len(road["trend"]) == n_months


def test_dashboard_stage_every_road_has_a_recommended_action(pipeline):
    dashboard = json.loads((pipeline["results_root"] / "dashboard.json").read_text(encoding="utf-8"))
    for road in dashboard["roads"]:
        assert isinstance(road["action"], str) and road["action"]
