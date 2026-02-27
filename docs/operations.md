# Operations and Troubleshooting

## Run Profiles

- **Full run (recommended):**
  - `scripts/run_pipeline.ps1 -ModelTrack both`
- **Fast recovery run:**
  - `scripts/run_pipeline.ps1 -ModelTrack tabular`
- **Model research run:**
  - `scripts/run_pipeline.ps1 -ModelTrack cnn`

## Monitoring Outputs

Check these artifacts after each run:
- `data/results/raw_data_validation.json`
- `data/features/dataset_manifest.json`
- `models/training_metrics.json`
- `models/cnn_temporal_metrics.json`
- `data/results/evaluation.json`
- `data/results/road_risk_ranking.json`

## Common Failures

## 1) Credential and provider access

Symptoms:
- ingestion modules fail early

Actions:
- verify `CDS_API_KEY`, `GEE_PROJECT_ID`
- run `earthengine authenticate`

## 2) Missing or stale outputs

Symptoms:
- API returns 404 on risk endpoints
- frontend reports missing road risk file

Actions:
- re-run inference stage
- run `python -m src.features.build_road_risk --max-roads 300`

## 3) Inconsistent data contracts

Symptoms:
- API aggregation errors (`422`)
- frontend missing columns

Actions:
- inspect parquet columns
- update API/frontend field mappings together

## 4) Skewed road map distribution

Symptoms:
- roads shown from only one side/zone

Actions:
- regenerate ranking JSON from latest scores
- verify level distribution (`High`, `Medium`, `Low`)

## Deployment Notes

- API and frontend are file-backed and can run independently.
- Use `docker-compose.yml` / `Dockerfile` for containerized environments.
- Keep `config` and model artifact names stable across deployments.

## Suggested Enhancements

- Add CI checks for schema compatibility of risk output files.
- Add automated tests for API endpoint field contracts.
- Add release tags to model artifact versions for rollback safety.
