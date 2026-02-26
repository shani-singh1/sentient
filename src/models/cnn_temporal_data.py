from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import Dataset

from src.common.paths import RAW_ROOT


AUX_COLUMNS = [
    c
    for c in [
        "stress_accum_rain_3m",
        "stress_accum_heat_3m",
        "stress_accum_flood_3m",
        "road_way_count_lag0",
        "road_length_km_lag0",
        "s1_backscatter_mean_lag0",
        "s1_flood_fraction_lag0",
        "s2_ndvi_mean_lag0",
        "s2_ndwi_mean_lag0",
        "landsat_thermal_mean_k_lag0",
        "landsat_heat_exposure_fraction_lag0",
        "nightlights_mean_lag0",
        "population_mean_lag0",
        "era5_total_precipitation_sum_lag0",
        "era5_2m_temperature_mean_lag0",
    ]
]


def month_list(start_month: str, end_month: str) -> list[str]:
    start = dt.date.fromisoformat(start_month + "-01")
    end = dt.date.fromisoformat(end_month + "-01")
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(f"{cur.year:04d}-{cur.month:02d}")
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return out


def parse_tile(tile_id: str) -> tuple[int, int]:
    _, y, x = tile_id.split("_")
    return int(y), int(x)


def zone_from_tile(tile_id: str) -> str:
    try:
        y, x = parse_tile(tile_id)
        return f"zone_{y // 2}_{x // 2}"
    except Exception:
        return "zone_0_0"


def tile_window(h: int, w: int, tile_y: int, tile_x: int, grid_size: int) -> Window:
    r0 = (h * tile_y) // grid_size
    r1 = (h * (tile_y + 1)) // grid_size
    c0 = (w * tile_x) // grid_size
    c1 = (w * (tile_x + 1)) // grid_size
    return Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0))


def raster_path(source: str, city: str, ym: str) -> Path:
    y, m = ym.split("-")
    if source == "sentinel1":
        return RAW_ROOT / "sentinel1" / y / m / f"sentinel1_compact_{city}_{y}{m}.tif"
    if source == "sentinel2":
        return RAW_ROOT / "sentinel2" / y / m / f"sentinel2_compact_{city}_{y}{m}.tif"
    if source == "landsat":
        return RAW_ROOT / "landsat" / y / m / f"landsat_composite_{city}_{y}{m}.tif"
    if source == "nightlights":
        return RAW_ROOT / "nightlights" / y / m / f"viirs_monthly_{city}_{y}{m}.tif"
    if source == "population":
        return RAW_ROOT / "population" / y / "01" / f"worldpop_{city}_{y}.tif"
    raise ValueError(source)


def read_tile_tensor(path: Path, tile_y: int, tile_x: int, grid_size: int, out_size: int, n_channels: int = 1) -> torch.Tensor:
    """Read raster tile and return tensor with exactly n_channels (pad/trim as needed)."""
    if not path.exists():
        return torch.zeros(n_channels, out_size, out_size, dtype=torch.float32)

    try:
        with rasterio.open(path) as ds:
            win = tile_window(ds.height, ds.width, tile_y, tile_x, grid_size)
            arr = ds.read(window=win, masked=True).astype(np.float32)
    except Exception:
        return torch.zeros(n_channels, out_size, out_size, dtype=torch.float32)
    arr = np.ma.filled(arr, np.nan)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)

    # Ensure exactly n_channels for stack consistency across months
    if t.shape[0] < n_channels:
        pad = torch.zeros(n_channels - t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype)
        t = torch.cat([t, pad], dim=0)
    else:
        t = t[:n_channels]

    mean = t.mean(dim=(1, 2), keepdim=True)
    std = t.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    t = (t - mean) / std
    return t


class ImageSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, city: str, grid_size: int, out_size: int = 32, augment: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.city = city.lower().replace(" ", "_")
        self.grid_size = grid_size
        self.out_size = out_size
        self.augment = augment
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    @property
    def in_channels(self) -> int:
        return 11

    @property
    def aux_dim(self) -> int:
        return len([c for c in AUX_COLUMNS if c in self.df.columns])

    def __len__(self) -> int:
        return len(self.df)

    def _load_month_tile(self, ym: str, tile_id: str) -> torch.Tensor:
        key = (ym, tile_id)
        if key in self._cache:
            return self._cache[key]

        ty, tx = parse_tile(tile_id)
        # Use n_channels to ensure consistent tensor shapes across months (fixes stack mismatch)
        s1 = read_tile_tensor(raster_path("sentinel1", self.city, ym), ty, tx, self.grid_size, self.out_size, n_channels=1)
        s2 = read_tile_tensor(raster_path("sentinel2", self.city, ym), ty, tx, self.grid_size, self.out_size, n_channels=4)
        ls = read_tile_tensor(raster_path("landsat", self.city, ym), ty, tx, self.grid_size, self.out_size, n_channels=4)
        nl = read_tile_tensor(raster_path("nightlights", self.city, ym), ty, tx, self.grid_size, self.out_size, n_channels=1)
        pop = read_tile_tensor(raster_path("population", self.city, ym), ty, tx, self.grid_size, self.out_size, n_channels=1)

        x = torch.cat([s1, s2, ls, nl, pop], dim=0)
        self._cache[key] = x
        return x

    def _augment_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return x_seq
        if torch.rand(1).item() < 0.5:
            x_seq = torch.flip(x_seq, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x_seq = torch.flip(x_seq, dims=[-2])
        return x_seq

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        start_month, end_month = str(row["time_window"]).split("__")
        months = month_list(start_month, end_month)

        seq = [self._load_month_tile(ym, str(row["tile_id"])) for ym in months]
        x_seq = torch.stack(seq, dim=0)
        x_seq = self._augment_seq(x_seq)

        aux_vals: list[float] = []
        for c in AUX_COLUMNS:
            if c in row.index:
                v = row[c]
                aux_vals.append(float(0.0 if pd.isna(v) else v))
        x_aux = torch.tensor(aux_vals, dtype=torch.float32)

        y = torch.tensor(float(row["target_proxy"]), dtype=torch.float32)
        return x_seq, x_aux, y
