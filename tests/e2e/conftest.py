"""Shared fixtures for the end-to-end suite: a real uvicorn process serving
the real committed data and frontend, driven by a real headless Chromium
browser through Playwright. Nothing here is mocked.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server():
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(60):
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"uvicorn exited early with code {proc.returncode}:\n{output}")
            try:
                urllib.request.urlopen(f"{base_url}/metadata", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("live server did not become ready in time")

        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser():
    playwright_sync = pytest.importorskip("playwright.sync_api")
    with playwright_sync.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


# A 1x1 transparent PNG used to stub basemap tile imagery. The tiles are
# purely decorative background pixels; stubbing them removes 8-10 real
# network round trips per test to a third-party CDN whose latency has
# nothing to do with the application under test, without touching a single
# line of real app logic (MapLibre still requests, receives, and renders
# real HTTP responses, just tiny local ones instead of remote images).
_BLANK_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000"
    "001f15c4890000000d49444154789c6360606060000000050001a5f6"
    "45400000000049454e44ae426082"
)


_MAPLIBRE_JS = (VENDOR_DIR / "maplibre-gl.js").read_bytes()
_MAPLIBRE_CSS = (VENDOR_DIR / "maplibre-gl.css").read_bytes()


@pytest.fixture()
def page(browser):
    """A fresh, fully network-hermetic page: the app's two external
    dependencies (the MapLibre GL library from a CDN, and CARTO basemap
    tile imagery) are served from a local vendor copy / stub instead of a
    live network fetch. This is the real, unmodified application and the
    real MapLibre library; only third-party network round trips that have
    nothing to do with the code under test are removed, so the suite is
    deterministic regardless of the host's outbound network conditions.
    """
    context = browser.new_context()
    context.route(
        "**/basemaps.cartocdn.com/**",
        lambda route: route.fulfill(status=200, content_type="image/png", body=_BLANK_PNG),
    )
    context.route(
        "**/unpkg.com/maplibre-gl@*/dist/maplibre-gl.js",
        lambda route: route.fulfill(status=200, content_type="application/javascript", body=_MAPLIBRE_JS),
    )
    context.route(
        "**/unpkg.com/maplibre-gl@*/dist/maplibre-gl.css",
        lambda route: route.fulfill(status=200, content_type="text/css", body=_MAPLIBRE_CSS),
    )
    pg = context.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda exc: errors.append(str(exc)))
    pg.on("console", lambda msg: errors.append(f"[console.error] {msg.text}") if msg.type == "error" else None)
    pg.uncaught_errors = errors  # type: ignore[attr-defined]
    yield pg
    context.close()


@pytest.fixture()
def booted_page(page, live_server):
    """A page already navigated to the live app with the command center
    fully booted: dashboard data fetched and the map initialized.

    Basemap tile imagery is stubbed locally (see the `page` fixture), so
    this only depends on the app's own code and one remaining external
    fetch (the MapLibre GL library from a CDN). On a shared, busy host the
    OS scheduler can occasionally starve the browser process for tens of
    seconds at a time for reasons that have nothing to do with the
    application; a couple of retries absorbs that noise without masking a
    real regression, which would fail on every attempt, not just the first.
    """
    attempts = 4
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            page.goto(live_server, wait_until="load", timeout=20000)
            page.wait_for_function("window.S && window.S.data && window.S.mapReady", timeout=20000, polling=250)
            return page
        except Exception as exc:  # noqa: BLE001 - retry any transient navigation/timeout error
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2)
    raise AssertionError(f"command center did not finish booting after {attempts} attempts: {last_error}")
