# SENTIENT Documentation

SENTIENT is an end-to-end geospatial ML system that predicts road infrastructure risk for Bengaluru, Mumbai, and Hyderabad from multi-source satellite and climate signals, then presents it through the Command Center, an interactive map-first decision interface for municipal commissioners and maintenance departments.

## What This Project Delivers

- Ingests monthly data from Sentinel-1/2, Landsat, ERA5, Nightlights, Population, and OSM across three cities.
- Builds tile-month stress features and temporal windows for model training (99 features per row).
- Trains a 10-model tabular sweep (Ridge through a temporal-holdout stack) with automatic selection, plus an independent CNN-temporal track (LSTM / TCN) as a second opinion.
- Produces risk scores and road-level rankings, with stress drivers, trend projections, and recommended actions.
- Serves everything through one FastAPI service: the JSON API and the Command Center frontend, at `http://localhost:8000`.

## End-to-End Flow

1. **Ingestion** (`src/ingestion/`) pulls and stores raw datasets.
2. **Validation + preprocessing** (`src/preprocessing/`) checks completeness and builds monthly stress.
3. **Feature engineering** (`src/features/build_dataset.py`) creates training-ready tabular windows.
4. **Training** (`src/training/`) fits the tabular sweep and the CNN-temporal track.
5. **Inference + evaluation** (`src/inference/`, `src/evaluation/`) scores risk and evaluates outputs.
6. **Road risk assembly** (`src/features/build_road_risk.py`) overlays risk on OSM roads and builds the dashboard payload.
7. **Serving** (`src/api/main.py`) exposes the API and the Command Center frontend (`src/frontend/web/`).

## Who Should Read What

- **New developer onboarding:** Start with `Setup` then `Architecture`.
- **ML engineer:** Read `Pipeline`, `Data Contracts`, and `Key Algorithms`.
- **Backend/frontend engineer:** Read `API` and `Frontend`.
- **Operator/SRE:** Read `Operations` and `Pipeline`.
- **QA:** Read `Testing`.

## Quick Links

- Full product and engineering reference: [`README.md`](https://github.com/shani-singh1/sentient/blob/main/README.md)
- Pseudocode for the core logic: `Key Algorithms`
- Every test case, input, and expected output: `Testing`
- Pipeline entrypoint: `scripts/run_pipeline.ps1`
- Serve entrypoint: `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000`
