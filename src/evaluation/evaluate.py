from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import FEATURES_ROOT, MODELS_ROOT, RESULTS_ROOT, ensure_dir


def spearman_rank_correlation(a: pd.Series, b: pd.Series) -> float:
    ra = a.rank(method="average")
    rb = b.rank(method="average")
    if float(ra.std()) == 0.0 or float(rb.std()) == 0.0:
        return 0.0
    corr = float(np.corrcoef(ra, rb)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def top_decile_lift(scores: pd.Series, events: pd.Series) -> float:
    threshold = scores.quantile(0.9)
    top_mask = scores >= threshold
    baseline = float(events.mean())
    if baseline <= 0:
        return 0.0
    return float(events[top_mask].mean() / baseline)


def metrics_block(scores: pd.Series, target_proxy: pd.Series) -> dict[str, float]:
    events = (target_proxy >= target_proxy.quantile(0.75)).astype(float)
    return {
        "spearman_risk_vs_target_proxy": spearman_rank_correlation(scores, target_proxy),
        "top_decile_lift": top_decile_lift(scores, events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation using proxy future stress events.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--predictions", default=str(RESULTS_ROOT / "risk_scores.parquet"))
    parser.add_argument("--training-metrics", default=str(MODELS_ROOT / "training_metrics.json"))
    args = parser.parse_args()

    ds = pd.read_parquet(args.dataset)
    pred = pd.read_parquet(args.predictions)

    merged = pred.merge(ds[["tile_id", "time_window", "target_proxy"]], on=["tile_id", "time_window"], how="left")
    merged["event_proxy"] = (merged["target_proxy"] >= merged["target_proxy"].quantile(0.75)).astype(float)
    merged["target_date"] = pd.to_datetime(merged["target_month"] + "-01", errors="coerce")

    overall = metrics_block(merged["risk_score"], merged["target_proxy"])
    by_year: dict[str, dict[str, float]] = {}
    for year in sorted(merged["target_date"].dt.year.dropna().unique()):
        mask = merged["target_date"].dt.year == year
        if int(mask.sum()) == 0:
            continue
        by_year[str(int(year))] = metrics_block(merged.loc[mask, "risk_score"], merged.loc[mask, "target_proxy"])

    overfit_flags: list[str] = []
    if overall["spearman_risk_vs_target_proxy"] > 0.98:
        overfit_flags.append("Very high proxy correlation; verify no target leakage in features.")
    if overall["top_decile_lift"] > 8.0:
        overfit_flags.append("Very high lift; verify proxy definition and split strategy.")

    eval_payload = {
        "row_count": int(len(merged)),
        **overall,
        "metrics_by_year": by_year,
        "overfit_guardrails": overfit_flags,
    }

    tm_path = Path(args.training_metrics)
    if tm_path.exists():
        eval_payload["training_metrics"] = json.loads(tm_path.read_text(encoding="utf-8"))

    out_dir = ensure_dir(RESULTS_ROOT)
    out_path = out_dir / "evaluation.json"
    out_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    print(json.dumps(eval_payload, indent=2))


if __name__ == "__main__":
    main()
