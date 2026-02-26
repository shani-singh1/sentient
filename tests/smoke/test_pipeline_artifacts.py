from __future__ import annotations

import json

import pandas as pd


def test_dataset_exists_and_has_core_columns() -> None:
    df = pd.read_parquet("data/features/dataset.parquet")
    required = {"tile_id", "time_window", "target_month", "target_proxy"}
    assert required.issubset(set(df.columns))


def test_risk_scores_exist_for_at_least_one_track() -> None:
    candidates = [
        "data/results/risk_scores_cnn_temporal.parquet",
        "data/results/risk_scores.parquet",
    ]
    loaded = None
    for path in candidates:
        try:
            loaded = pd.read_parquet(path)
            break
        except Exception:
            continue
    assert loaded is not None
    assert {"tile_id", "risk_score"}.issubset(set(loaded.columns))


def test_evaluation_json_exists() -> None:
    with open("data/results/evaluation.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "row_count" in payload
