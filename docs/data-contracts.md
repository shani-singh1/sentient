# Data Contracts

This page defines key files consumed across modules so teams can safely change internals without breaking downstream components.

## 1) Config Contract

One file per city, for example `config/pipeline.bengaluru.2020_2024.json`

Required keys:
- `city` (string)
- `bbox` (string: `west,south,east,north`)
- `start_date`, `end_date` (ISO date)
- `window_size` (int)
- `grid_size` (int)
- `sentinel1_scale` (int)
- `sentinel2_scale` (int)
- `sentinel2_max_cloud` (int)

## 2) Core Model Dataset

File: `data/features/dataset.parquet`

Expected to contain identifiers and target/risk inputs used by training and scoring, such as:
- `tile_id`
- `zone_id` (if available)
- temporal features and stress aggregates
- target/risk-related columns used by train/inference modules

## 3) Risk Score Outputs

### Tabular output

File: `data/results/risk_scores.parquet`

Common fields consumed by API/UI:
- `tile_id`
- `zone_id` (optional)
- `road_segment_id` (optional)
- `risk_score`
- `target_month` (optional)
- `model_track`, `model_name` (optional metadata)

### CNN-temporal output

File: `data/results/risk_scores_cnn_temporal.parquet`

Expected to expose at least:
- `tile_id`
- `risk_score`

## 4) Road Ranking Output

File: `data/results/road_risk_ranking.json`

Top-level schema:
- `roads`: list of road objects
- `sources`: map of city to path to its OSM source file
- `tile_risk_source`: path to risk parquet source

Road object fields:
- `id` (string, `<city>_<rank>`)
- `rank` (int, position in the displayed stratified sample)
- `city_rank` (int, position by risk within the city, ignoring sampling)
- `priority` (int, 0..100, percentile against every analyzed road in the city)
- `name` (string)
- `highway` (string)
- `city` (string)
- `zone` (`NW` | `NE` | `SW` | `SE`)
- `path` (list of `[lon, lat]`)
- `length_m` (float)
- `risk_score` (float, 0..1)
- `risk_pct` (float, 0..100)
- `risk_level` (`High` | `Medium` | `Low`, relative percentile cuts per city: top 15% / next 35% / rest)

Risk tiers are **relative within each city**, not fixed absolute thresholds. Cut points live in `dashboard.json`'s top-level `city_cuts` object (`{city: {high, medium}}`).

## 5) Dashboard Payload

File: `data/results/dashboard.json` (about 1.8 MB, loaded once by the Command Center frontend)

Top-level schema:
- `generated_at_utc`, `model_name`, `city_cuts`, `months` (sorted list of `YYYY-MM` strings)
- `cities`: map of city key to `{label, bbox, summary, tiles, stress_series}`
  - `summary`: plain-language snapshot stats (`critical_now`, `entering_critical_6m`, `monsoon_stress_yoy_pct`, `risk_concentration`, `worst_zone`, `total_length_km`, ...)
  - `tiles`: list of `{id, lon, lat, series}`, one monthly risk value per entry in `months`
  - `stress_series`: `{rain, flood, heat}`, each a monthly 0-100 index aligned to `months`
- `roads`: every road from `road_risk_ranking.json` plus `trend` (monthly risk series), `months_to_critical`, `drivers` (top 3 stress drivers with plain-language labels and severity scores), and `action` (recommended next step)

## 6) Validation and inventory reports

- `data/results/raw_data_validation.json`
- `data/results/data_inventory_manifest.json`
- `data/results/evaluation.json`

These files are read by `/metadata` and should remain JSON-serializable and backward compatible.

## Contract Change Guidelines

- Additive changes are preferred over renames/removals.
- If removing/renaming columns, update:
  - API handlers in `src/api/main.py`
  - Frontend mappings in `src/frontend/web/app.js`
  - The corresponding rows in `tests/integration/` and `tests/component/`
