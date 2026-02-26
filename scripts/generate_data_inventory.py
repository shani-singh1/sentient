from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_path_parts(path: Path, root: Path) -> tuple[str | None, str | None]:
    rel = path.relative_to(root)
    if len(rel.parts) >= 3:
        return rel.parts[0], rel.parts[1]
    return None, None


def build_inventory(root: Path) -> dict[str, object]:
    inventory: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(root),
        "totals": {},
        "by_kind": {},
        "by_source_year": {},
    }

    data_root = root / "data"
    files = [p for p in data_root.rglob("*") if p.is_file()] if data_root.exists() else []

    totals = {
        "file_count": len(files),
        "total_size_bytes": sum(p.stat().st_size for p in files),
    }
    inventory["totals"] = totals

    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "size_bytes": 0})
    by_source_year: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "size_bytes": 0}))

    for file_path in files:
        size = file_path.stat().st_size
        ext = file_path.suffix.lower() or "<no_ext>"
        by_kind[ext]["count"] += 1
        by_kind[ext]["size_bytes"] += size

        if "raw" in file_path.parts:
            raw_idx = file_path.parts.index("raw")
            if len(file_path.parts) > raw_idx + 2:
                source = file_path.parts[raw_idx + 1]
                year = file_path.parts[raw_idx + 2]
                by_source_year[source][year]["count"] += 1
                by_source_year[source][year]["size_bytes"] += size

    inventory["by_kind"] = dict(sorted(by_kind.items(), key=lambda item: item[0]))
    inventory["by_source_year"] = {
        source: dict(sorted(years.items(), key=lambda item: item[0]))
        for source, years in sorted(by_source_year.items(), key=lambda item: item[0])
    }
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic data inventory manifest.")
    parser.add_argument("--output", default="data/results/data_inventory_manifest.json")
    args = parser.parse_args()

    root = project_root()
    manifest = build_inventory(root)
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote inventory manifest: {out_path}")


if __name__ == "__main__":
    main()
