from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.paths import FEATURES_ROOT, RESULTS_ROOT, ensure_dir


def normalize_0_1(series: pd.Series) -> pd.Series:
    mn = float(series.min())
    mx = float(series.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def zone_from_tile(tile_id: str) -> str:
    try:
        if "__" in tile_id:
            tile_id = tile_id.split("__", 1)[1]
        _, y, x = tile_id.split("_")
        yi = int(y)
        xi = int(x)
        return f"zone_{yi // 2}_{xi // 2}"
    except Exception:
        return "zone_0_0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and produce risk scores.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--output", default=str(RESULTS_ROOT / "risk_scores.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    art = joblib.load(args.model)
    model = art["model"]
    features = art["features"]
    model_name = str(art.get("model_name", "tabular"))

    x = df[features].to_numpy(dtype=float)
    pred = model.predict(x)

    base_cols = ["tile_id", "time_window", "target_month", "sample_month"]
    if "city" in df.columns:
        base_cols.insert(0, "city")
    out = df[base_cols].copy()
    out["zone_id"] = out["tile_id"].astype(str).map(zone_from_tile)
    out["road_segment_id"] = out["tile_id"].astype(str) + "_roads"
    out["model_track"] = "tabular"
    out["model_name"] = model_name
    out["prediction_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out["raw_risk"] = pred
    out["risk_score"] = normalize_0_1(out["raw_risk"])
    if "dataset_version" in df.columns:
        out["dataset_version"] = df["dataset_version"].astype(str)

    by_tile = out.groupby("tile_id", as_index=False)["risk_score"].mean().rename(columns={"risk_score": "tile_risk"})
    out = out.merge(by_tile, on="tile_id", how="left")
    by_zone = out.groupby("zone_id", as_index=False)["risk_score"].mean().rename(columns={"risk_score": "zone_risk"})
    out = out.merge(by_zone, on="zone_id", how="left")

    out_dir = ensure_dir(RESULTS_ROOT)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = out_dir / out_path.name
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
