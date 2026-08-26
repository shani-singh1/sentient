"""Unit tests for the pure functions in src/features/build_dataset.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_dataset as bd

pytestmark = pytest.mark.unit


def test_city_key_lowercases_and_replaces_spaces_with_underscore() -> None:
    assert bd.city_key("Bengaluru Metro") == "bengaluru_metro"


def test_city_key_is_idempotent_on_already_normalized_input() -> None:
    assert bd.city_key("mumbai") == "mumbai"


def test_normalize_features_zscores_using_only_train_rows() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
    train_mask = pd.Series([True, True, True, False])

    out, stats = bd.normalize_features(df, train_mask, ["x"])

    expected_mean = np.mean([1.0, 2.0, 3.0])
    expected_std = np.std([1.0, 2.0, 3.0], ddof=1)
    assert stats["x"]["mean"] == pytest.approx(expected_mean)
    assert stats["x"]["std"] == pytest.approx(expected_std)
    # The held-out outlier is normalized with train statistics, not its own.
    assert out["x"].iloc[3] == pytest.approx((100.0 - expected_mean) / expected_std)


def test_normalize_features_forces_unit_std_for_constant_column() -> None:
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    train_mask = pd.Series([True, True, True])

    out, stats = bd.normalize_features(df, train_mask, ["x"])

    assert stats["x"]["std"] == 1.0
    assert (out["x"] == 0.0).all()


def test_normalize_features_creates_missing_column_as_zero_before_scaling() -> None:
    df = pd.DataFrame({"other": [1.0, 2.0]})
    train_mask = pd.Series([True, True])

    out, stats = bd.normalize_features(df, train_mask, ["missing_col"])

    assert "missing_col" in out.columns
    assert stats["missing_col"]["mean"] == 0.0


def _synthetic_monthly_frame(tile_id: str, city: str, months: list[str], base: float = 1.0) -> pd.DataFrame:
    rows = []
    for i, month_id in enumerate(months):
        row = {"city": city, "tile_id": tile_id, "month_id": month_id, "date": pd.Timestamp(month_id + "-01")}
        for col in bd.FEATURE_COLUMNS:
            row[col] = base + i * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_windows_computes_target_proxy_from_next_month_only() -> None:
    months = ["2024-01", "2024-02", "2024-03", "2024-04"]
    df = _synthetic_monthly_frame("city__tile_00_00", "city", months)

    out = bd.build_windows(df, window_size=3)

    assert len(out) == 1
    next_row = df.iloc[3]
    expected = (
        float(next_row["era5_total_precipitation_sum"])
        + 2.0 * float(next_row["landsat_heat_exposure_fraction"])
        + 3.0 * float(next_row["s1_flood_fraction"])
    )
    assert out.iloc[0]["target_proxy"] == pytest.approx(expected)
    assert out.iloc[0]["target_month"] == "2024-04"
    assert out.iloc[0]["sample_month"] == "2024-03"


def test_build_windows_raises_when_no_contiguous_window_exists() -> None:
    # A gap between Jan and Mar means no 3-month contiguous window can end before it.
    months = ["2024-01", "2024-03", "2024-04"]
    df = _synthetic_monthly_frame("city__tile_00_00", "city", months)

    with pytest.raises(ValueError):
        bd.build_windows(df, window_size=3)


def test_build_windows_skips_tiles_shorter_than_the_window() -> None:
    months = ["2024-01", "2024-02"]
    df = _synthetic_monthly_frame("city__tile_00_00", "city", months)

    with pytest.raises(ValueError):
        bd.build_windows(df, window_size=3)


def test_build_windows_one_hot_encodes_every_city_present() -> None:
    months = ["2024-01", "2024-02", "2024-03", "2024-04"]
    df_a = _synthetic_monthly_frame("alpha__tile_00_00", "alpha", months)
    df_b = _synthetic_monthly_frame("beta__tile_00_00", "beta", months, base=5.0)
    df = pd.concat([df_a, df_b], ignore_index=True)

    out = bd.build_windows(df, window_size=3)

    assert set(out.columns) & {"city_alpha", "city_beta"} == {"city_alpha", "city_beta"}
    assert (out["city_alpha"] + out["city_beta"] == 1.0).all()


def test_build_windows_lag_columns_reflect_reverse_chronological_order() -> None:
    months = ["2024-01", "2024-02", "2024-03", "2024-04"]
    df = _synthetic_monthly_frame("city__tile_00_00", "city", months)

    out = bd.build_windows(df, window_size=3)

    # lag0 is the most recent observed month (sample_month = 2024-03), lag2 is the oldest (2024-01).
    row = out.iloc[0]
    assert row["road_way_count_lag0"] == pytest.approx(1.0 + 2 * 0.1)
    assert row["road_way_count_lag2"] == pytest.approx(1.0)
