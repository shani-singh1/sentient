"""Unit tests for the pure helper functions in src/preprocessing/validate_raw_data.py."""
from __future__ import annotations

import datetime as dt

import pytest

from src.preprocessing import validate_raw_data as vrd

pytestmark = pytest.mark.unit


def test_month_range_spans_multiple_years() -> None:
    result = vrd.month_range(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
    assert result[0] == (2020, 1)
    assert result[-1] == (2021, 12)
    assert len(result) == 24


def test_expected_monthly_files_has_one_path_per_monthly_source() -> None:
    files = vrd.expected_monthly_files("bengaluru", 2024, 5)
    assert set(files.keys()) == {"era5", "sentinel1", "sentinel2", "landsat", "nightlights"}
    assert files["era5"].name == "era5_bengaluru_202405.nc"
    assert files["sentinel1"].name == "sentinel1_compact_bengaluru_202405.tif"


def test_expected_monthly_files_pads_single_digit_months() -> None:
    files = vrd.expected_monthly_files("mumbai", 2024, 3)
    for path in files.values():
        assert "03" in path.parts


def test_expected_annual_files_targets_january_worldpop_raster() -> None:
    files = vrd.expected_annual_files("hyderabad", 2021)
    assert files["population"].name == "worldpop_hyderabad_2021.tif"
    assert files["population"].parent.name == "01"


def test_population_fallback_file_returns_none_without_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vrd, "RAW_ROOT", tmp_path)
    assert vrd.population_fallback_file("bengaluru", 2024) is None


def test_population_fallback_file_picks_latest_year_not_after_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vrd, "RAW_ROOT", tmp_path)
    for year in (2019, 2021, 2025):
        d = tmp_path / "population" / str(year) / "01"
        d.mkdir(parents=True)
        (d / f"worldpop_bengaluru_{year}.tif").write_bytes(b"x")

    result = vrd.population_fallback_file("bengaluru", 2022)

    assert result is not None
    assert result.name == "worldpop_bengaluru_2021.tif"
