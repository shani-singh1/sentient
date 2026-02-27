# Architecture

## System Context

Sentient transforms city-scale environmental stress signals into road-level risk intelligence for prioritizing inspections and preventive maintenance.

## High-Level Components

### 1) Data ingestion layer

- Location: `src/ingestion/`
- Pulls source-specific monthly or snapshot datasets:
  - `sentinel1_ingest.py`
  - `sentinel2_ingest.py`
  - `landsat_ingest.py`
  - `era5_ingest.py`
  - `nightlights_ingest.py`
  - `population_ingest.py`
  - `osm_ingest.py`

### 2) Data quality + preprocessing layer

- Location: `src/preprocessing/`
- `validate_raw_data.py`: checks presence/coverage and writes validation report.
- `monthly_stress.py`: converts raw sources into monthly stress features by grid tile.

### 3) Feature engineering layer

- Location: `src/features/`
- `build_dataset.py`: assembles model-ready dataset with temporal windows.
- `build_road_risk.py`: projects tile risk onto OSM road geometries and ranks roads.

### 4) Modeling layer

- Location: `src/training/`, `src/models/`
- Tabular track: `train_models.py`
- CNN-temporal track: `train_cnn_temporal.py`
- CNN data shaping utilities: `src/models/cnn_temporal_data.py`

### 5) Inference and evaluation layer

- Location: `src/inference/`, `src/evaluation/`
- Tabular inference: `score_risk.py`
- CNN inference: `score_risk_cnn_temporal.py`
- Evaluation: `evaluate.py`

### 6) Serving layer

- API: `src/api/main.py` (FastAPI)
- Frontend: `src/frontend/app.py` (Streamlit + PyDeck + Plotly)

## Runtime Artifacts

- Raw data: `data/raw/`
- Processed monthly data: `data/processed/`
- Feature tables: `data/features/`
- Results: `data/results/`
- Models and metrics: `models/`

## Configuration Model

Primary config file: `config/pipeline.bengaluru.2020_2024.json`

Key parameters:
- `city`
- `bbox`
- `start_date`, `end_date`
- `grid_size`
- `window_size`
- source-specific scales/cloud constraints

## Design Notes

- **Model redundancy:** CNN-temporal primary with tabular fallback.
- **Operational traceability:** reports and manifests in `data/results/`.
- **Decision focus:** road-level ranking output designed for inspection planning, not just tile heatmaps.
