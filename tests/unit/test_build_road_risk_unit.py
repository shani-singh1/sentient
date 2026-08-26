"""Unit tests for the pure functions in src/features/build_road_risk.py."""
from __future__ import annotations

import pytest

from src.features import build_road_risk as brr

pytestmark = pytest.mark.unit

BBOX = [77.45, 12.8, 77.75, 13.1]


def test_city_key_lowercases_and_replaces_spaces() -> None:
    assert brr.city_key("Bengaluru City") == "bengaluru_city"


def test_point_to_tile_northwest_corner_is_tile_00() -> None:
    west, south, east, north = BBOX
    tile = brr.point_to_tile(west + 1e-6, north - 1e-6, BBOX, "bengaluru")
    assert tile == "bengaluru__tile_00_00"


def test_point_to_tile_southeast_corner_is_last_tile() -> None:
    west, south, east, north = BBOX
    tile = brr.point_to_tile(east - 1e-6, south + 1e-6, BBOX, "bengaluru")
    assert tile == f"bengaluru__tile_{brr.GRID_SIZE - 1:02d}_{brr.GRID_SIZE - 1:02d}"


def test_point_to_tile_clamps_points_outside_bbox() -> None:
    west, south, east, north = BBOX
    tile = brr.point_to_tile(west - 5.0, north + 5.0, BBOX, "bengaluru")
    assert tile == "bengaluru__tile_00_00"


def test_haversine_m_zero_distance_for_same_point() -> None:
    assert brr.haversine_m(12.9, 77.6, 12.9, 77.6) == pytest.approx(0.0, abs=1e-6)


def test_haversine_m_matches_known_reference_distance() -> None:
    # Roughly Bengaluru city center to Electronic City, about 18 km.
    d = brr.haversine_m(12.9716, 77.5946, 12.8452, 77.6602)
    assert 15_000 < d < 21_000


def test_tile_center_returns_point_inside_bbox() -> None:
    center = brr._tile_center("bengaluru__tile_00_00")
    assert center is not None
    lon, lat = center
    west, south, east, north = BBOX
    assert west <= lon <= east
    assert south <= lat <= north


def test_tile_center_returns_none_for_unknown_city() -> None:
    assert brr._tile_center("atlantis__tile_00_00") is None


def test_tile_center_returns_none_for_malformed_tile_id() -> None:
    assert brr._tile_center("not-a-valid-id") is None


def test_months_to_critical_is_zero_when_already_at_threshold() -> None:
    series = [0.1, 0.2, 0.3, 0.4, 0.75]
    assert brr._months_to_critical(series, threshold=0.7) == 0


def test_months_to_critical_extrapolates_rising_trend() -> None:
    series = [0.10, 0.20, 0.30, 0.40]
    # slope is 0.1/month, current 0.40, threshold 0.70 -> ceil(0.30/0.10) = 3
    assert brr._months_to_critical(series, threshold=0.70) == 3


def test_months_to_critical_returns_none_for_flat_series() -> None:
    series = [0.3, 0.3, 0.3, 0.3]
    assert brr._months_to_critical(series, threshold=0.9) is None


def test_months_to_critical_returns_none_with_insufficient_history() -> None:
    series = [None, None, 0.5]
    assert brr._months_to_critical(series, threshold=0.9) is None


def test_months_to_critical_returns_none_when_projection_exceeds_24_months() -> None:
    series = [0.10, 0.101, 0.102, 0.103]
    assert brr._months_to_critical(series, threshold=0.99) is None


def test_months_to_critical_ignores_leading_none_values() -> None:
    series = [None, 0.10, 0.20, 0.30, 0.40]
    assert brr._months_to_critical(series, threshold=0.70) == 3
