from __future__ import annotations

from pathlib import Path
import os


def project_root() -> Path:
    """
    Return the assumed project root.

    We resolve two levels up from this file:
    `.../sentient/src/ingestion/paths.py` -> `.../sentient`.
    """

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = project_root()

RAW_ROOT: Path = PROJECT_ROOT / "data" / "raw"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_secret(name: str) -> str | None:
    """
    Return a secret value from the environment or from a local .env file.

    This avoids having to export environment variables manually while keeping
    secrets out of logs and source control.
    """

    value = os.getenv(name)
    if value:
        return value

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == name:
                return val.strip()
    except Exception:
        # Fail silently; callers will handle missing secrets.
        return None

    return None

