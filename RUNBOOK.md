# Pipeline Runbook

## 1) Install dependencies
```powershell
python -m pip install -r requirements.txt
```

## 2) Configure credentials
Required by source:
- ERA5: `CDS_API_KEY`
- Earth Engine: `GEE_PROJECT_ID` and local `earthengine authenticate`

Use `.env` in project root or environment variables.

## 3) Scope and config
This runbook targets Bengaluru 2020-01-01 to 2024-12-31 using:
- `config/pipeline.bengaluru.2020_2024.json`

## 4) End-to-end execution
Run everything (CNN primary + tabular fallback):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -ConfigPath "config/pipeline.bengaluru.2020_2024.json" -ModelTrack both
```

Run only CNN track:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -ConfigPath "config/pipeline.bengaluru.2020_2024.json" -ModelTrack cnn
```

Run only tabular fallback:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -ConfigPath "config/pipeline.bengaluru.2020_2024.json" -ModelTrack tabular
```

## 5) Manual stage commands (optional)
```powershell
python -m src.ingestion.era5_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31
python -m src.ingestion.osm_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31
python -m src.ingestion.population_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31
python -m src.ingestion.landsat_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31
python -m src.ingestion.nightlights_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31
python -m src.ingestion.sentinel1_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31 --scale 20
python -m src.ingestion.sentinel2_ingest --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31 --max-cloud 40 --scale 20
python -m src.preprocessing.validate_raw_data --city Bengaluru --start-date 2020-01-01 --end-date 2024-12-31
python -m src.preprocessing.monthly_stress --city Bengaluru --bbox "77.45,12.8,77.75,13.1" --start-date 2020-01-01 --end-date 2024-12-31 --grid-size 4
python -m src.features.build_dataset --city Bengaluru --window-size 3
python -m src.training.train_cnn_temporal --dataset data/features/dataset.parquet --city Bengaluru --grid-size 4
python -m src.inference.score_risk_cnn_temporal --dataset data/features/dataset.parquet --model models/cnn_temporal_best.pt --city Bengaluru --grid-size 4
python -m src.training.train_models --dataset data/features/dataset.parquet --val-fraction 0.3
python -m src.inference.score_risk --dataset data/features/dataset.parquet --model models/best_model.joblib
python -m src.evaluation.evaluate --dataset data/features/dataset.parquet --predictions data/results/risk_scores.parquet --training-metrics models/training_metrics.json
python scripts/generate_data_inventory.py
python -m src.features.build_road_risk --max-roads 300
```

## 6) Outputs
- Processed tile-month stress summaries: `data/processed/<city>/YYYY_MM.parquet`
- Feature dataset: `data/features/dataset.parquet`
- Normalization stats: `data/features/normalization_stats.json`
- Tabular models: `models/baseline_model.joblib`, `models/temporal_gb_model.joblib`, `models/temporal_rf_model.joblib`, `models/best_model.joblib`
- CNN models: `models/cnn_lstm.pt`, `models/cnn_tcn.pt`, `models/cnn_temporal_best.pt`
- Risk scores (tabular): `data/results/risk_scores.parquet`
- Risk scores (cnn temporal): `data/results/risk_scores_cnn_temporal.parquet`
- Evaluation: `data/results/evaluation.json`
- Inventory manifest: `data/results/data_inventory_manifest.json`

## 7) API and frontend

**Run API** (read-only, serves risk scores):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
```
API base: http://127.0.0.1:8000
- `GET /metadata` – config, inventory, evaluation
- `GET /risk/latest?limit=100&model=tabular` – top risk rows
- `GET /risk/ranking?limit=100&model=tabular&by=tile` – ranked by tile/zone/row
- `GET /risk/by_zone?model=tabular` – risk aggregated by zone
- `GET /risk/heatmap?model=tabular` – points with lon, lat, risk_score for map

**Run frontend**:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frontend.ps1
```
Frontend: http://localhost:8501 – risk heatmap, ranked segments, model comparison (tabular vs CNN)
