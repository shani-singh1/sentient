from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import requests

from .paths import ensure_dir, get_secret


def init_ee() -> "object":
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("earthengine-api is required. Install with: pip install earthengine-api") from exc

    project_id = get_secret("GEE_PROJECT_ID") or os.getenv("GEE_PROJECT_ID")

    if project_id:
        ee.Initialize(project=project_id)
    else:
        ee.Initialize()
    return ee


def bbox_to_ee_geometry(ee: "object", bbox: Tuple[float, float, float, float]) -> "object":
    west, south, east, north = bbox
    return ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)


def export_image_to_geotiff(
    ee: "object",
    image: "object",
    region: "object",
    out_path: Path,
    scale: int,
    crs: str = "EPSG:4326",
) -> Path:
    ensure_dir(out_path.parent)

    payload = {
        "scale": scale,
        "region": region,
        "crs": crs,
        "format": "GEO_TIFF",
    }
    retries = 5
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            url = image.getDownloadURL(payload)
            with requests.get(url, stream=True, timeout=600) as resp:
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"temporary status={resp.status_code}", response=resp)
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return out_path
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(60, 2 ** attempt))

    assert last_error is not None
    raise last_error
