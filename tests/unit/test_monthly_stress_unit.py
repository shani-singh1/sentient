"""Unit tests for the pure helper functions in src/preprocessing/monthly_stress.py."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.preprocessing import monthly_stress as ms

pytestmark = pytest.mark.unit


def test_parse_bbox_returns_four_floats_in_order() -> None:
    assert ms.parse_bbox("77.45,12.8,77.75,13.1") == (77.45, 12.8, 77.75, 13.1)


def test_parse_bbox_tolerates_surrounding_whitespace() -> None:
    assert ms.parse_bbox(" 1 , 2 , 3 , 4 ") == (1.0, 2.0, 3.0, 4.0)


def test_parse_bbox_rejects_wrong_number_of_parts() -> None:
    with pytest.raises(ValueError):
        ms.parse_bbox("1,2,3")


def test_month_range_within_single_month() -> None:
    assert ms.month_range(dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == [(2024, 1)]


def test_month_range_spans_a_year_boundary() -> None:
    result = ms.month_range(dt.date(2023, 11, 15), dt.date(2024, 2, 3))
    assert result == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_month_range_start_equals_end_date() -> None:
    assert ms.month_range(dt.date(2022, 6, 1), dt.date(2022, 6, 1)) == [(2022, 6)]


def test_haversine_m_zero_for_identical_points() -> None:
    assert ms._haversine_m(12.9, 77.6, 12.9, 77.6) == pytest.approx(0.0, abs=1e-6)


def test_haversine_m_one_degree_latitude_is_about_111_km() -> None:
    distance = ms._haversine_m(0.0, 0.0, 1.0, 0.0)
    assert distance == pytest.approx(111_195, rel=0.01)


def test_array_stats_all_nan_returns_zeroed_payload() -> None:
    arr = np.full((3, 3), np.nan)
    assert ms._array_stats(arr) == {"mean": 0.0, "p90": 0.0, "frac_pos": 0.0}


def test_array_stats_computes_mean_p90_and_positive_fraction() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0])

    stats = ms._array_stats(arr)

    assert stats["mean"] == pytest.approx(np.mean(arr))
    assert stats["p90"] == pytest.approx(np.percentile(arr, 90))
    assert stats["frac_pos"] == pytest.approx(5.0 / 6.0)


def test_array_stats_ignores_nan_values_in_the_mix() -> None:
    arr = np.array([10.0, np.nan, 20.0, np.nan])
    stats = ms._array_stats(arr)
    assert stats["mean"] == pytest.approx(15.0)


def test_landsat_thermal_kelvin_rescales_raw_digital_numbers() -> None:
    raw = np.array([10000.0, 20000.0, 30000.0])
    expected = raw * 0.00341802 + 149.0

    result = ms._landsat_thermal_kelvin(raw)

    np.testing.assert_allclose(result, expected)


def test_landsat_thermal_kelvin_passes_through_values_already_in_kelvin() -> None:
    kelvin = np.array([295.0, 300.0, 305.0])

    result = ms._landsat_thermal_kelvin(kelvin)

    np.testing.assert_array_equal(result, kelvin)


def test_landsat_thermal_kelvin_handles_all_nan_input() -> None:
    arr = np.full(4, np.nan)
    result = ms._landsat_thermal_kelvin(arr)
    assert np.all(np.isnan(result))


def test_tile_bounds_partitions_extent_without_gap_at_first_tile() -> None:
    r0, r1, c0, c1 = ms._tile_bounds(h=100, w=100, tile_y=0, tile_x=0, grid_size=4)
    assert (r0, c0) == (0, 0)
    assert (r1, c1) == (25, 25)


def test_tile_bounds_last_tile_reaches_the_far_edge() -> None:
    r0, r1, c0, c1 = ms._tile_bounds(h=100, w=100, tile_y=3, tile_x=3, grid_size=4)
    assert r1 == 100
    assert c1 == 100


def test_band_or_nan_extracts_requested_band() -> None:
    arr = np.stack([np.zeros((2, 2)), np.ones((2, 2)) * 7, np.ones((2, 2)) * 9])
    band = ms._band_or_nan(arr, 1)
    np.testing.assert_array_equal(band, np.full((2, 2), 7.0))


def test_band_or_nan_returns_nan_filled_when_band_index_out_of_range() -> None:
    arr = np.zeros((2, 3, 3))
    band = ms._band_or_nan(arr, 5)
    assert band.shape == (3, 3)
    assert np.all(np.isnan(band))


def test_monthly_tif_path_uses_lowercased_underscored_city() -> None:
    path = ms.monthly_tif_path("landsat", "landsat_composite", "Bengaluru", 2024, 3)
    assert path.name == "landsat_composite_bengaluru_202403.tif"
    assert path.parent.name == "03"
    assert path.parent.parent.name == "2024"


def test_population_path_returns_exact_year_file_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "RAW_ROOT", tmp_path)
    exact_dir = tmp_path / "population" / "2022" / "01"
    exact_dir.mkdir(parents=True)
    (exact_dir / "worldpop_bengaluru_2022.tif").write_bytes(b"x")

    result = ms.population_path("Bengaluru", 2022)

    assert result == exact_dir / "worldpop_bengaluru_2022.tif"


def test_population_path_falls_back_to_latest_earlier_year(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "RAW_ROOT", tmp_path)
    for year in (2018, 2020):
        d = tmp_path / "population" / str(year) / "01"
        d.mkdir(parents=True)
        (d / f"worldpop_bengaluru_{year}.tif").write_bytes(b"x")

    result = ms.population_path("Bengaluru", 2023)

    assert result.name == "worldpop_bengaluru_2020.tif"


def test_month_has_overlap_false_when_no_raw_data_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "RAW_ROOT", tmp_path)
    assert ms.month_has_overlap("Bengaluru", 2024, 1) is False
