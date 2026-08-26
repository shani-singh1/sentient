"""Integration tests for src/api/main.py against the real, committed,
already-deployed data artifacts (data/features, data/results) and the real
static frontend files. No mocking of paths: this is what a client actually
receives from the running service.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)


def test_root_serves_the_command_center_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SENTIENT" in response.text
    assert "app.js" in response.text


def test_static_assets_are_served_alongside_the_api():
    js = client.get("/app.js")
    css = client.get("/styles.css")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_unknown_deep_path_returns_a_real_404_not_the_spa_shell():
    response = client.get("/this/route/does/not/exist")
    assert response.status_code == 404
    assert "application/json" in response.headers["content-type"]


def test_metadata_endpoint_reports_city_and_grid_configuration():
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["city"]
    assert payload["grid_size"] > 0
    assert "," in payload["bbox"]


def test_risk_latest_ranks_results_in_descending_risk_order():
    response = client.get("/risk/latest?limit=25")
    assert response.status_code == 200
    rows = response.json()
    assert 0 < len(rows) <= 25
    scores = [r["risk_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


@pytest.mark.parametrize("by", ["tile", "zone", "row"])
def test_risk_ranking_supports_every_aggregation_mode(by):
    response = client.get(f"/risk/ranking?by={by}&limit=10")
    assert response.status_code == 200
    rows = response.json()
    assert 0 < len(rows) <= 10
    assert all("risk_score" in r and "rank" in r for r in rows)


def test_risk_by_zone_returns_one_row_per_zone_sorted_descending():
    response = client.get("/risk/by_zone")
    assert response.status_code == 200
    rows = response.json()
    zone_ids = [r["zone_id"] for r in rows]
    assert len(zone_ids) == len(set(zone_ids))
    risks = [r["zone_risk"] for r in rows]
    assert risks == sorted(risks, reverse=True)


def test_risk_roads_endpoint_reports_a_total_at_least_as_large_as_the_page():
    response = client.get("/risk/roads?limit=15")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["roads"]) <= 15
    assert payload["total"] >= len(payload["roads"])


def test_dashboard_endpoint_contains_every_deployed_city():
    response = client.get("/dashboard")
    assert response.status_code == 200
    payload = response.json()
    for city in ("bengaluru", "mumbai", "hyderabad"):
        assert city in payload["cities"]
        city_payload = payload["cities"][city]
        assert "summary" in city_payload
        assert "tiles" in city_payload
        assert "stress_series" in city_payload
    assert len(payload["roads"]) > 0


def test_dashboard_city_cuts_are_internally_consistent():
    response = client.get("/dashboard")
    payload = response.json()
    for city, cuts in payload["city_cuts"].items():
        assert cuts["high"] >= cuts["medium"]


def test_risk_heatmap_points_fall_within_a_plausible_india_bounding_box():
    response = client.get("/risk/heatmap?limit=20")
    assert response.status_code == 200
    points = response.json()
    assert 0 < len(points) <= 20
    for p in points:
        assert 65.0 < p["lon"] < 90.0
        assert 6.0 < p["lat"] < 35.0


def test_risk_heatmap_with_nonexistent_month_returns_empty_list_not_an_error():
    response = client.get("/risk/heatmap?target_month=1999-01")
    assert response.status_code == 200
    assert response.json() == []
