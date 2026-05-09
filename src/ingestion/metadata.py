from __future__ import annotations

import csv
import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .paths import RAW_ROOT, ensure_dir


@dataclass
class IngestionMetadata:
    """Minimal, source-agnostic ingestion log schema."""

    timestamp_utc: str
    source: str
    city: str
    start_date: str
    end_date: str
    bbox_west: float
    bbox_south: float
    bbox_east: float
    bbox_north: float
    crs: Optional[str] = None
    files: List[str] = field(default_factory=list)
    notes: Optional[str] = None


def _metadata_log_path() -> Path:
    return ensure_dir(RAW_ROOT) / "ingestion_metadata.csv"


def append_metadata(entry: IngestionMetadata) -> None:
    """
    Append a metadata entry to the central CSV log.

    Columns are stable so downstream modules can read them easily.
    """

    path = _metadata_log_path()
    is_new = not path.exists()
    fieldnames = list(asdict(entry).keys())

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(entry))


def build_metadata(
    *,
    source: str,
    city: str,
    start_date: dt.date,
    end_date: dt.date,
    bbox: Iterable[float],
    files: Iterable[Path],
    crs: Optional[str] = None,
    notes: Optional[str] = None,
) -> IngestionMetadata:
    west, south, east, north = list(bbox)
    return IngestionMetadata(
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        source=source,
        city=city,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        bbox_west=west,
        bbox_south=south,
        bbox_east=east,
        bbox_north=north,
        crs=crs,
        files=[str(p) for p in files],
        notes=notes,
    )

