from __future__ import annotations

import argparse
import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.paths import FEATURES_ROOT, MODELS_ROOT, ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def time_split(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_month"] + "-01")
    df = df.sort_values("target_date").reset_index(drop=True)

    split_idx = max(1, int(len(df) * (1.0 - val_fraction)))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def select_features(df: pd.DataFrame, mode: str) -> list[str]:
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "target_proxy",
            "target_month",
            "tile_id",
            "time_window",
            "imagery_reference",
            "sample_month",
            "target_date",
            "dataset_version",
            "city",
        }
    ]
    if mode == "baseline":
        return [c for c in numeric_cols if c.endswith("_lag0") or c.startswith("stress_accum_")]
    if mode in {"temporal_gb", "temporal_rf"}:
        return numeric_cols
    raise ValueError(f"unsupported mode: {mode}")


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def top_decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    pred_threshold = float(np.quantile(y_pred, 0.9))
    events = (y_true >= float(np.quantile(y_true, 0.75))).astype(float)
    top_mask = y_pred >= pred_threshold
    baseline = float(np.mean(events))
    if baseline <= 0:
        return 0.0
    return float(np.mean(events[top_mask]) / baseline)


def metrics_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman_rank_correlation(y_true, y_pred),
        "top_decile_lift": top_decile_lift(y_true, y_pred),
    }


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame, mode: str) -> dict[str, object]:
    feats = select_features(train_df, mode)
    x_train = train_df[feats].to_numpy(dtype=float)
    y_train = train_df["target_proxy"].to_numpy(dtype=float)

    x_val = val_df[feats].to_numpy(dtype=float)
    y_val = val_df["target_proxy"].to_numpy(dtype=float)

    if mode == "baseline":
        model = Ridge(alpha=1.0)
    elif mode == "temporal_gb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=42,
        )
    elif mode == "temporal_rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    model.fit(x_train, y_train)
    pred = model.predict(x_val)

    metrics = {
        "model": mode,
        "feature_count": len(feats),
        **metrics_payload(y_val, pred),
        "validation_rows": int(len(val_df)),
    }

    return {
        "model": model,
        "features": feats,
        "metrics": metrics,
        "pred": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and temporal regression models.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = pd.read_parquet(args.dataset)
    train_df, val_df = time_split(df, args.val_fraction)

    baseline = train_one(train_df, val_df, "baseline")
    temporal_gb = train_one(train_df, val_df, "temporal_gb")
    temporal_rf = train_one(train_df, val_df, "temporal_rf")

    out_dir = ensure_dir(MODELS_ROOT)
    joblib.dump({"model": baseline["model"], "features": baseline["features"], "model_name": "baseline"}, out_dir / "baseline_model.joblib")
    joblib.dump({"model": temporal_gb["model"], "features": temporal_gb["features"], "model_name": "temporal_gb"}, out_dir / "temporal_gb_model.joblib")
    joblib.dump({"model": temporal_rf["model"], "features": temporal_rf["features"], "model_name": "temporal_rf"}, out_dir / "temporal_rf_model.joblib")

    candidates = [
        ("baseline", baseline),
        ("temporal_gb", temporal_gb),
        ("temporal_rf", temporal_rf),
    ]
    best_name, best_obj = max(
        candidates,
        key=lambda x: (
            float(x[1]["metrics"]["top_decile_lift"]),
            float(x[1]["metrics"]["spearman"]),
            float(x[1]["metrics"]["r2"]),
        ),
    )
    joblib.dump({"model": best_obj["model"], "features": best_obj["features"], "model_name": best_name}, out_dir / "best_model.joblib")

    val_target = val_df["target_proxy"].to_numpy(dtype=float)
    yearly_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_obj in [("baseline", baseline), ("temporal_gb", temporal_gb), ("temporal_rf", temporal_rf)]:
        preds = np.asarray(model_obj["pred"], dtype=float)
        by_year: dict[str, dict[str, float]] = {}
        for year in sorted(val_df["target_date"].dt.year.unique()):
            mask = val_df["target_date"].dt.year == year
            if int(mask.sum()) == 0:
                continue
            by_year[str(int(year))] = metrics_payload(val_target[mask.to_numpy()], preds[mask.to_numpy()])
        yearly_metrics[model_name] = by_year

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "seed": args.seed,
        "baseline": baseline["metrics"],
        "temporal_gb": temporal_gb["metrics"],
        "temporal_rf": temporal_rf["metrics"],
        "validation_metrics_by_year": yearly_metrics,
        "best_model": best_name,
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
