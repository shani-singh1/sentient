"""Smoke test for Road Risk API. Run: python scripts/smoke_test_api.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def main() -> None:
    r = client.get("/metadata")
    assert r.status_code == 200, f"metadata: {r.status_code}"
    print("GET /metadata OK")

    r = client.get("/risk/ranking?limit=5&model=tabular")
    assert r.status_code == 200, f"ranking: {r.status_code}"
    data = r.json()
    assert isinstance(data, list)
    print(f"GET /risk/ranking OK ({len(data)} items)")

    r = client.get("/risk/heatmap?model=tabular&limit=10")
    assert r.status_code == 200
    print(f"GET /risk/heatmap OK ({len(r.json())} points)")

    r = client.get("/risk/by_zone?model=tabular")
    assert r.status_code == 200
    print(f"GET /risk/by_zone OK ({len(r.json())} zones)")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
