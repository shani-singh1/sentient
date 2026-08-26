"""Unit tests for src/training/train_models.py: metrics, selection, and model factory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.training import train_models as tm

pytestmark = pytest.mark.unit


def test_spearman_rank_correlation_perfect_positive() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert tm.spearman_rank_correlation(y_true, y_pred) == pytest.approx(1.0)


def test_spearman_rank_correlation_perfect_negative() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    assert tm.spearman_rank_correlation(y_true, y_pred) == pytest.approx(-1.0)


def test_spearman_rank_correlation_zero_when_predictions_are_constant() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    assert tm.spearman_rank_correlation(y_true, y_pred) == 0.0


def test_spearman_rank_correlation_zero_for_fewer_than_two_points() -> None:
    assert tm.spearman_rank_correlation(np.array([1.0]), np.array([1.0])) == 0.0


def test_top_decile_lift_is_at_maximum_for_a_perfect_ranking() -> None:
    # 40 points; true quartile-75 event exactly matches predicted top decile.
    y_true = np.arange(40, dtype=float)
    y_pred = np.arange(40, dtype=float)
    assert tm.top_decile_lift(y_true, y_pred) == pytest.approx(4.0)


def test_top_decile_lift_zero_when_targets_contain_nan() -> None:
    # NaN propagates through the quantile and comparison, zeroing every event
    # flag; the function must guard against dividing by that zero baseline.
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert tm.top_decile_lift(y_true, y_pred) == 0.0


def test_top_decile_lift_zero_for_empty_arrays() -> None:
    assert tm.top_decile_lift(np.array([]), np.array([])) == 0.0


def test_metrics_payload_contains_expected_keys_and_float_types() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    payload = tm.metrics_payload(y_true, y_pred)

    assert set(payload.keys()) == {"mae", "rmse", "r2", "spearman", "top_decile_lift"}
    assert all(isinstance(v, float) for v in payload.values())
    assert payload["mae"] >= 0.0
    assert payload["rmse"] >= 0.0


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_lag0": [1.0, 2.0],
            "x_lag1": [3.0, 4.0],
            "stress_accum_rain_3m": [0.5, 0.6],
            "x_trend": [0.1, 0.2],
            "target_proxy": [0.0, 0.0],
            "target_month": ["2024-01", "2024-02"],
            "tile_id": ["a", "b"],
            "time_window": ["w1", "w2"],
            "imagery_reference": ["r", "r"],
            "sample_month": ["2023-12", "2024-01"],
            "dataset_version": ["v1", "v1"],
            "city": ["bengaluru", "bengaluru"],
        }
    )


def test_select_features_baseline_keeps_only_lag0_and_accumulator_columns() -> None:
    feats = tm.select_features(_feature_frame(), "baseline")
    assert set(feats) == {"x_lag0", "stress_accum_rain_3m"}


def test_select_features_temporal_mode_excludes_metadata_columns() -> None:
    feats = tm.select_features(_feature_frame(), "temporal_rf")
    excluded = {"target_proxy", "target_month", "tile_id", "time_window", "imagery_reference", "sample_month", "dataset_version", "city"}
    assert excluded.isdisjoint(feats)
    assert {"x_lag0", "x_lag1", "x_trend", "stress_accum_rain_3m"}.issubset(feats)


def test_time_split_produces_strict_temporal_ordering() -> None:
    months = [f"2024-{m:02d}" for m in range(1, 11)]
    df = pd.DataFrame({"target_month": months, "value": range(10)}).sample(frac=1.0, random_state=1)

    train_df, val_df = tm.time_split(df, val_fraction=0.3)

    assert pd.to_datetime(train_df["target_month"] + "-01").max() <= pd.to_datetime(val_df["target_month"] + "-01").min()
    assert len(val_df) == pytest.approx(3, abs=1)


def test_build_model_baseline_is_ridge_regression() -> None:
    assert isinstance(tm.build_model("baseline"), Ridge)


def test_build_model_temporal_rf_is_random_forest() -> None:
    assert isinstance(tm.build_model("temporal_rf"), RandomForestRegressor)


def test_build_model_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        tm.build_model("not_a_real_mode")


def test_blend_regressor_predicts_the_mean_of_its_base_estimators() -> None:
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])
    blend = tm.BlendRegressor(estimators=[DummyRegressor(strategy="mean"), DummyRegressor(strategy="median")])

    blend.fit(x, y)
    preds = blend.predict(x)

    # Both dummies collapse to the constant 20.0 here, so the blend must too.
    np.testing.assert_allclose(preds, np.full(3, 20.0))


def test_time_stack_regressor_produces_predictions_for_every_row() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.01, size=30)

    stack = tm.TimeStackRegressor(
        estimators=[("a", DummyRegressor(strategy="mean")), ("b", Ridge(alpha=1.0))],
        meta_fraction=0.3,
    )
    stack.fit(x, y)
    preds = stack.predict(x)

    assert preds.shape == (30,)
    assert isinstance(stack.meta_, Ridge)
    assert len(stack.fitted_) == 2
