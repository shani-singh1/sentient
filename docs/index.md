# Sentient Documentation

Sentient is an end-to-end geospatial ML system that identifies road infrastructure risk in Bengaluru using multi-source satellite and climate signals, then presents actionable rankings and map views for decision-makers.

## What This Project Delivers

- Ingests monthly data from Sentinel-1/2, Landsat, ERA5, Nightlights, Population, and OSM.
- Builds tile-month stress features and temporal windows for model training.
- Trains two tracks:
  - CNN-temporal primary model.
  - Tabular fallback model.
- Produces risk scores and road-level rankings for inspections.
- Serves outputs via FastAPI and a decision-focused Streamlit dashboard.

## End-to-End Flow

1. **Ingestion** (`src/ingestion/`) pulls and stores raw datasets.
2. **Validation + preprocessing** (`src/preprocessing/`) checks completeness and builds monthly stress.
3. **Feature engineering** (`src/features/build_dataset.py`) creates training-ready tabular windows.
4. **Training** (`src/training/`) fits tabular and CNN-temporal models.
5. **Inference + evaluation** (`src/inference/`, `src/evaluation/`) scores risk and evaluates outputs.
6. **Road risk assembly** (`src/features/build_road_risk.py`) overlays risk on OSM roads.
7. **Serving** (`src/api/main.py`, `src/frontend/app.py`) exposes APIs and a UI.

## Who Should Read What

- **New developer onboarding:** Start with `Setup` then `Architecture`.
- **ML engineer:** Read `Pipeline`, `Data Contracts`, and `Operations`.
- **Backend/frontend engineer:** Read `API` and `Frontend`.
- **Operator/SRE:** Read `Operations` and `Pipeline`.

## Quick Links

- Runbook: `RUNBOOK.md`
- Pipeline entrypoint: `scripts/run_pipeline.ps1`
- API entrypoint: `scripts/run_api.ps1`
- Frontend entrypoint: `scripts/run_frontend.ps1`
