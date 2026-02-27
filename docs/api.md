# API

Backend service is implemented in `src/api/main.py` using FastAPI.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
```

Default base URL: `http://127.0.0.1:8000`

## Endpoints

## `GET /metadata`

Returns operational metadata:
- config values (`city`, `bbox`, `grid_size`)
- optional embedded reports:
  - data inventory manifest
  - raw data validation
  - evaluation

## `GET /risk/latest`

Query:
- `limit` (1..2000)
- `model` (`tabular` or `cnn_temporal`)

Returns latest top risk rows with rank and map centers (`lon`, `lat`) when `tile_id` exists.

## `GET /risk/ranking`

Query:
- `limit` (1..2000)
- `model` (`tabular` or `cnn_temporal`)
- `by` (`tile`, `zone`, `row`)

Returns aggregated or row-level risk ranking.

## `GET /risk/by_zone`

Query:
- `model` (`tabular` or `cnn_temporal`)

Returns zone-level average risk and ranking.

## `GET /risk/heatmap`

Query:
- `model` (`tabular` or `cnn_temporal`)
- `target_month` (optional)
- `limit` (1..5000)

Returns map-ready points: `tile_id`, `lon`, `lat`, `risk_score`.

## `GET /risk/roads`

Query:
- `limit` (1..500)

Returns road-level ranking and geometry from `road_risk_ranking.json`.

## Model Selection Behavior

- `model=tabular` -> reads `data/results/risk_scores.parquet`
- `model=cnn_temporal` (or `cnn`) -> reads `data/results/risk_scores_cnn_temporal.parquet`

## API Reliability Notes

- API is read-only and file-backed (no DB required).
- Missing expected artifacts return HTTP `404`.
- Missing required fields for zone aggregation returns HTTP `422`.
