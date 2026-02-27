# Pipeline

## Canonical Orchestration

Primary entrypoint:

`scripts/run_pipeline.ps1`

Parameters:
- `-ConfigPath` (default: `config/pipeline.bengaluru.2020_2024.json`)
- `-ModelTrack` (`cnn`, `tabular`, `both`)

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -ConfigPath "config/pipeline.bengaluru.2020_2024.json" -ModelTrack both
```

## Stage-by-Stage Breakdown

### Stage A: Ingestion

Runs these modules:
- `src.ingestion.era5_ingest`
- `src.ingestion.osm_ingest`
- `src.ingestion.population_ingest`
- `src.ingestion.landsat_ingest`
- `src.ingestion.nightlights_ingest`
- `src.ingestion.sentinel1_ingest`
- `src.ingestion.sentinel2_ingest`

Output root: `data/raw/`

### Stage B: Data quality and preprocessing

- `src.preprocessing.validate_raw_data`
- `src.preprocessing.monthly_stress`

Outputs:
- Validation report: `data/results/raw_data_validation.json`
- Monthly processed tables: `data/processed/<city>/YYYY_MM.parquet`

### Stage C: Dataset assembly

- `src.features.build_dataset`

Outputs:
- `data/features/dataset.parquet`
- `data/features/normalization_stats.json`
- `data/features/dataset_manifest.json`

### Stage D: Model training + scoring

#### Tabular track

- `src.training.train_models`
- `src.inference.score_risk`
- `src.evaluation.evaluate`

Artifacts:
- `models/best_model.joblib`
- `models/training_metrics.json`
- `data/results/risk_scores.parquet`
- `data/results/evaluation.json`

#### CNN-temporal track

- `src.training.train_cnn_temporal`
- `src.inference.score_risk_cnn_temporal`

Artifacts:
- `models/cnn_temporal_best.pt`
- `models/cnn_temporal_metrics.json`
- `data/results/risk_scores_cnn_temporal.parquet`

### Stage E: Road-level ranking and inventory

- `src.features.build_road_risk --max-roads 300`
- `scripts/generate_data_inventory.py`

Artifacts:
- `data/results/road_risk_ranking.json`
- `data/results/data_inventory_manifest.json`

## Recommended Execution Modes

- **Production-like run:** `-ModelTrack both`
- **Fast fallback run:** `-ModelTrack tabular`
- **Research on deep model only:** `-ModelTrack cnn`

## Operational Notes

- Re-runs are common in geospatial pipelines; prefer resumable ingest modes where supported.
- If risk map appears geographically skewed, regenerate road ranking and verify risk level mix.
- Keep config, model artifacts, and output naming aligned to avoid API/frontend mismatches.
