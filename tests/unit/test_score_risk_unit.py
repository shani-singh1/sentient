"""Unit tests for the pure functions in src/inference/score_risk.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.inference import score_risk as sr

pytestmark = pytest.mark.unit


def test_normalize_0_1_scales_into_unit_range() -> None:
    series = pd.Series([0.0, 5.0, 10.0])
    out = sr.normalize_0_1(series)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_0_1_returns_zeros_for_constant_series() -> None:
    series = pd.Series([7.0, 7.0, 7.0])
    out = sr.normalize_0_1(series)
    assert (out == 0.0).all()


def test_normalize_0_1_preserves_series_index() -> None:
    series = pd.Series([1.0, 2.0], index=[10, 20])
    out = sr.normalize_0_1(series)
    assert list(out.index) == [10, 20]


def test_zone_from_tile_buckets_2x2_groups() -> None:
    assert sr.zone_from_tile("tile_01_02") == "zone_0_1"


def test_zone_from_tile_strips_city_prefix_first() -> None:
    assert sr.zone_from_tile("bengaluru__tile_03_03") == "zone_1_1"


def test_zone_from_tile_falls_back_to_default_zone_on_malformed_input() -> None:
    assert sr.zone_from_tile("not-a-tile-id") == "zone_0_0"
