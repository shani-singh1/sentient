# Data Contracts

This page defines key files consumed across modules so teams can safely change internals without breaking downstream components.

## 1) Config Contract

File: `config/pipeline.bengaluru.2020_2024.json`

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
- `source`: path to OSM source file
- `tile_risk_source`: path to risk parquet source

Road object fields:
- `rank` (int)
- `name` (string)
- `highway` (string)
- `path` (list of `[lon, lat]`)
- `length_m` (float)
- `risk_score` (float, 0..1)
- `risk_pct` (float, 0..100)
- `risk_level` (`High` | `Medium` | `Low`)
- `zone` (`NW` | `NE` | `SW` | `SE`, if generated)

## 5) Validation and inventory reports

- `data/results/raw_data_validation.json`
- `data/results/data_inventory_manifest.json`
- `data/results/evaluation.json`

These files are read by `/metadata` and should remain JSON-serializable and backward compatible.

## Contract Change Guidelines

- Additive changes are preferred over renames/removals.
- If removing/renaming columns, update:
  - API handlers in `src/api/main.py`
  - Frontend mappings in `src/frontend/app.py`
  - Runbook contract references in `RUNBOOK.md`
