from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.paths import FEATURES_ROOT, MODELS_ROOT, ensure_dir
from src.models.cnn_temporal import CNNLSTMRegressor, CNNTCNRegressor
from src.models.cnn_temporal_data import ImageSequenceDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_split_three_way(df: pd.DataFrame, val_fraction: float, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["target_date"] = pd.to_datetime(data["target_month"] + "-01")
    data = data.sort_values("target_date").reset_index(drop=True)

    n = len(data)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = max(1, n - n_val - n_test)

    train = data.iloc[:n_train].copy()
    val = data.iloc[n_train : n_train + n_val].copy()
    test = data.iloc[n_train + n_val :].copy()

    if len(test) == 0:
        test = val.tail(1).copy()
        val = val.iloc[:-1].copy()
    if len(val) == 0:
        val = train.tail(1).copy()
        train = train.iloc[:-1].copy()

    return train, val, test


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    r2 = 0.0 if var <= 1e-12 else float(1.0 - np.mean((y_true - y_pred) ** 2) / var)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman_rank_correlation(y_true, y_pred),
        "top_decile_lift": top_decile_lift(y_true, y_pred),
    }


def metrics_by_year(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    years = sorted(pd.to_datetime(dates).dt.year.unique())
    for year in years:
        mask = pd.to_datetime(dates).dt.year == year
        if int(mask.sum()) == 0:
            continue
        out[str(int(year))] = eval_metrics(y_true[mask.to_numpy()], y_pred[mask.to_numpy()])
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for x_seq, x_aux, y in loader:
        x_seq = x_seq.to(device)
        x_aux = x_aux.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(x_seq, x_aux)
        loss = loss_fn(pred, y)

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(float(loss.item()))
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.float32)
    return float(np.mean(losses) if losses else 0.0), y_true, y_pred


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_state = None
    best_val_r2 = -1e9
    best_val_metrics: dict[str, float] = {}
    no_improve = 0

    for _ in range(epochs):
        run_epoch(model, train_loader, loss_fn, device, optimizer=opt)
        _, y_val, p_val = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        val_metrics = eval_metrics(y_val, p_val)
        scheduler.step(val_metrics["r2"])

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    _, y_test, p_test = run_epoch(model, test_loader, loss_fn, device, optimizer=None)
    test_metrics = eval_metrics(y_test, p_test)

    return {
        "name": model_name,
        "model": model,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "test_pred": p_test,
        "test_true": y_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN + temporal models (LSTM and TCN) on image sequences.")
    parser.add_argument("--dataset", default=str(FEATURES_ROOT / "dataset.parquet"))
    parser.add_argument("--city", default="Bengaluru")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df = pd.read_parquet(args.dataset)
    train_df, val_df, test_df = time_split_three_way(df, args.val_fraction, args.test_fraction)

    train_ds = ImageSequenceDataset(train_df, city=args.city, grid_size=args.grid_size, out_size=args.image_size, augment=True)
    val_ds = ImageSequenceDataset(val_df, city=args.city, grid_size=args.grid_size, out_size=args.image_size, augment=False)
    test_ds = ImageSequenceDataset(test_df, city=args.city, grid_size=args.grid_size, out_size=args.image_size, augment=False)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = CNNLSTMRegressor(in_channels=train_ds.in_channels, aux_dim=train_ds.aux_dim).to(device)
    tcn = CNNTCNRegressor(in_channels=train_ds.in_channels, aux_dim=train_ds.aux_dim).to(device)

    out_lstm = train_model(
        "cnn_lstm",
        lstm,
        train_loader,
        val_loader,
        test_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        device,
    )
    out_tcn = train_model(
        "cnn_tcn",
        tcn,
        train_loader,
        val_loader,
        test_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        device,
    )

    candidates = [out_lstm, out_tcn]
    best = max(candidates, key=lambda x: float(x["val_metrics"]["r2"]))

    out_dir = ensure_dir(MODELS_ROOT)
    torch.save(
        {
            "state_dict": out_lstm["model"].state_dict(),
            "model_type": "cnn_lstm",
            "in_channels": train_ds.in_channels,
            "aux_dim": train_ds.aux_dim,
        },
        out_dir / "cnn_lstm.pt",
    )
    torch.save(
        {
            "state_dict": out_tcn["model"].state_dict(),
            "model_type": "cnn_tcn",
            "in_channels": train_ds.in_channels,
            "aux_dim": train_ds.aux_dim,
        },
        out_dir / "cnn_tcn.pt",
    )
    torch.save(
        {
            "state_dict": best["model"].state_dict(),
            "model_type": best["name"],
            "in_channels": train_ds.in_channels,
            "aux_dim": train_ds.aux_dim,
        },
        out_dir / "cnn_temporal_best.pt",
    )

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "seed": args.seed,
        "cnn_lstm": {
            "val": out_lstm["val_metrics"],
            "test": out_lstm["test_metrics"],
            "test_metrics_by_year": metrics_by_year(test_df["target_date"], out_lstm["test_true"], out_lstm["test_pred"]),
        },
        "cnn_tcn": {
            "val": out_tcn["val_metrics"],
            "test": out_tcn["test_metrics"],
            "test_metrics_by_year": metrics_by_year(test_df["target_date"], out_tcn["test_true"], out_tcn["test_pred"]),
        },
        "best_model": best["name"],
    }
    (out_dir / "cnn_temporal_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
