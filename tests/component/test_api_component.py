"""Component tests for src/api/main.py: helper functions and error handling.

These exercise the API module's internal helpers and edge cases (missing
files, malformed inputs, config fallbacks) in isolation. Happy-path
end-to-end coverage against the real shipped data lives in
tests/integration/test_api_integration.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import main as api

pytestmark = pytest.mark.component


def test_tile_to_center_maps_first_tile_near_northwest_corner():
    bbox = [77.45, 12.8, 77.75, 13.1]
    lon, lat = api._tile_to_center("tile_00_00", bbox, grid_size=4)
    west, south, east, north = bbox
    assert west < lon < east
    assert south < lat < north
    assert lon < (west + east) / 2  # west half
    assert lat > (south + north) / 2  # north half


def test_tile_to_center_falls_back_to_bbox_centroid_on_malformed_id():
    bbox = [0.0, 0.0, 10.0, 10.0]
    lon, lat = api._tile_to_center("garbage-id", bbox, grid_size=4)
    assert (lon, lat) == (5.0, 5.0)


def test_load_config_returns_documented_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "CONFIG_PATH", tmp_path / "does_not_exist.json")
    cfg = api._load_config()
    assert cfg["bbox"] == "77.45,12.8,77.75,13.1"
    assert cfg["grid_size"] == 4


def test_load_scores_raises_file_not_found_when_no_scores_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        api._load_scores("tabular")


def test_risk_roads_endpoint_returns_404_when_ranking_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS_ROOT", tmp_path)
    client = TestClient(api.app)

    response = client.get("/risk/roads")

    assert response.status_code == 404
    assert "build_road_risk" in response.json()["detail"]


def test_dashboard_endpoint_returns_404_when_payload_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS_ROOT", tmp_path)
    client = TestClient(api.app)

    response = client.get("/dashboard")

    assert response.status_code == 404


def test_risk_latest_endpoint_returns_404_when_scores_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS_ROOT", tmp_path)
    client = TestClient(api.app)

    response = client.get("/risk/latest")

    assert response.status_code == 404


def test_risk_latest_rejects_limit_above_maximum(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS_ROOT", tmp_path)
    client = TestClient(api.app)

    response = client.get("/risk/latest?limit=999999")

    assert response.status_code == 422
