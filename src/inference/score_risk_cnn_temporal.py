from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.paths import FEATURES_ROOT, RESULTS_ROOT, ensure_dir
from src.models.cnn_temporal import CNNLSTMRegressor, CNNTCNRegressor
from src.models.cnn_temporal_data import ImageSequenceDataset, zone_from_tile


def normalize_0_1(series: pd.Series) -> pd.Series:
    mn = float(series.min())
    mx = float(series.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def build_model(model_type: str, in_channels: int, aux_dim: int) -> torch.nn.Module:
    if model_type == "cnn_lstm":
        return CNNLSTMRegressor(in_channels=in_channels, aux_dim=aux_dim)
    if model_type == "cnn_tcn":
        return CNNTCNRegressor(in_channels=in_channels, aux_dim=aux_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with CNN+temporal model and produce risk scores.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--model", default="models/cnn_temporal_best.pt")
    parser.add_argument("--city", default="Bengaluru")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=str(RESULTS_ROOT / "risk_scores_cnn_temporal.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    ds = ImageSequenceDataset(df, city=args.city, grid_size=args.grid_size, out_size=args.image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.model, map_location="cpu")
    model = build_model(str(ckpt["model_type"]), int(ckpt["in_channels"]), int(ckpt.get("aux_dim", ds.aux_dim)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for x_seq, x_aux, _ in loader:
            p = model(x_seq, x_aux)
            preds.append(p.detach().cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    base_cols = ["tile_id", "time_window", "target_month", "sample_month"]
    if "city" in df.columns:
        base_cols.insert(0, "city")
    out = df[base_cols].copy()
    out["zone_id"] = out["tile_id"].astype(str).map(zone_from_tile)
    out["road_segment_id"] = out["tile_id"].astype(str) + "_roads"
    out["model_track"] = "cnn_temporal"
    out["model_name"] = str(ckpt["model_type"])
    out["prediction_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out["raw_risk"] = y_pred
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
