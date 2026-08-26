"""Component tests for src/features/build_dataset.py: the CLI entry point itself.

These tests exercise `main()` end to end against a synthetic processed-data
tree (no real satellite data required) and check the artifacts it writes to
disk: dataset.parquet, normalization_stats.json, and dataset_manifest.json.
The underlying pure functions (normalize_features, build_windows) already
have dedicated unit tests; this file verifies the CLI wiring around them.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features import build_dataset as bd

pytestmark = pytest.mark.component


def _write_processed_city(processed_root, city: str, months: list[str], tiles: list[str]) -> None:
    city_dir = processed_root / bd.city_key(city)
    city_dir.mkdir(parents=True, exist_ok=True)
    for i, month_id in enumerate(months):
        rows = []
        for tile_id in tiles:
            row = {"month_id": month_id, "tile_id": tile_id}
            for col in bd.FEATURE_COLUMNS:
                row[col] = float(i + 1) + hash((col, tile_id)) % 5 * 0.01
            rows.append(row)
        pd.DataFrame(rows).to_parquet(city_dir / f"{month_id.replace('-', '_')}.parquet", index=False)


@pytest.fixture()
def dataset_workspace(tmp_path, monkeypatch):
    processed_root = tmp_path / "processed"
    features_root = tmp_path / "features"
    monkeypatch.setattr(bd, "PROCESSED_ROOT", processed_root)
    monkeypatch.setattr(bd, "FEATURES_ROOT", features_root)

    months = [f"2024-{m:02d}" for m in range(1, 7)]  # 6 contiguous months
    tiles = ["tile_00_00", "tile_00_01"]
    _write_processed_city(processed_root, "Testcity", months, tiles)

    return {"processed_root": processed_root, "features_root": features_root, "months": months, "tiles": tiles}


def test_main_writes_dataset_parquet_with_expected_row_count(dataset_workspace, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["build_dataset.py", "--city", "Testcity", "--window-size", "3", "--train-fraction", "0.6"],
    )

    bd.main()

    out_path = dataset_workspace["features_root"] / "dataset.parquet"
    assert out_path.exists()
    out = pd.read_parquet(out_path)
    # 6 months, window_size 3 -> (6 - 3) = 3 windows per tile, 2 tiles = 6 rows.
    assert len(out) == 6
    assert {"tile_id", "target_month", "target_proxy", "dataset_version"}.issubset(out.columns)


def test_main_output_matches_calling_the_underlying_functions_directly(dataset_workspace, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["build_dataset.py", "--city", "Testcity", "--window-size", "3", "--train-fraction", "0.6"],
    )
    bd.main()
    persisted = pd.read_parquet(dataset_workspace["features_root"] / "dataset.parquet")

    # Recompute independently through the same (separately unit-tested) pure
    # functions to confirm main() did not lose or corrupt data in transit.
    df = bd.load_processed_monthlies("Testcity")
    df = df.sort_values(["date", "city", "tile_id"]).reset_index(drop=True)
    split_idx = max(1, int(len(df) * 0.6))
    split_idx = min(split_idx, len(df) - 1)
    cutoff = df["date"].sort_values().iloc[split_idx - 1]
    train_mask = df["date"] <= cutoff
    normalized, _ = bd.normalize_features(df, train_mask, bd.FEATURE_COLUMNS)
    expected = bd.build_windows(normalized, 3)

    assert len(persisted) == len(expected)
    pd.testing.assert_series_equal(
        persisted["target_proxy"].sort_values().reset_index(drop=True),
        expected["target_proxy"].sort_values().reset_index(drop=True),
        check_exact=False,
    )


def test_main_writes_normalization_stats_for_every_feature_column(dataset_workspace, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_dataset.py", "--city", "Testcity", "--window-size", "3"])

    bd.main()

    stats = json.loads((dataset_workspace["features_root"] / "normalization_stats.json").read_text(encoding="utf-8"))
    assert set(bd.FEATURE_COLUMNS).issubset(stats.keys())
    for col_stats in stats.values():
        assert "mean" in col_stats and "std" in col_stats


def test_main_writes_manifest_with_row_count_matching_dataset(dataset_workspace, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_dataset.py", "--city", "Testcity", "--window-size", "3"])

    bd.main()

    manifest = json.loads((dataset_workspace["features_root"] / "dataset_manifest.json").read_text(encoding="utf-8"))
    out = pd.read_parquet(dataset_workspace["features_root"] / "dataset.parquet")
    assert manifest["rows"] == len(out)
    assert manifest["cities"] == ["testcity"]
    assert manifest["window_size"] == 3


def test_main_rejects_window_size_below_two(dataset_workspace, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_dataset.py", "--city", "Testcity", "--window-size", "1"])

    with pytest.raises(ValueError):
        bd.main()


def test_main_rejects_train_fraction_out_of_open_unit_interval(dataset_workspace, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["build_dataset.py", "--city", "Testcity", "--window-size", "3", "--train-fraction", "1.5"],
    )

    with pytest.raises(ValueError):
        bd.main()
