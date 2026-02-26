from __future__ import annotations

"""Shared project path utilities."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
MODELS_ROOT = PROJECT_ROOT / "models"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
