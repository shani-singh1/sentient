# Setup

## Prerequisites

- Python 3.12+
- Windows PowerShell (project scripts are PowerShell-first)
- Network access for data providers (only needed to re-ingest raw satellite data; the repo ships with processed data and a trained model already in place)
- Credentials for re-ingestion only:
  - ERA5 (`CDS_API_KEY`)
  - Google Earth Engine (`GEE_PROJECT_ID`, plus local authentication)

## 1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For running the test suite (optional):

```powershell
python -m pip install -r requirements-test.txt
python -m playwright install chromium
```

For this documentation site (optional):

```powershell
python -m pip install mkdocs mkdocs-material
```

## 2) Configure credentials

Set environment variables (or use local env injection):

- `CDS_API_KEY`
- `GEE_PROJECT_ID`

Authenticate Earth Engine once on machine:

```powershell
earthengine authenticate
```

## 3) Validate project config

One config per city, used by the run scripts:

`config/pipeline.bengaluru.2020_2024.json`, `config/pipeline.mumbai.2020_2024.json`, `config/pipeline.hyderabad.2020_2024.json`

## 4) Run the app

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` for the Command Center.

## 5) First smoke run (recommended)

Run API smoke checks after generating outputs:

```powershell
python scripts/smoke_test_api.py
python -m pytest tests/smoke -q
```

## 6) Run documentation UI locally

The Command Center already uses port 8000, so serve docs on a different port:

```powershell
mkdocs serve -a 127.0.0.1:8001
```

Then open `http://127.0.0.1:8001`.

## Common Setup Issues

- Missing parquet support:
  - Ensure `pyarrow` is installed from `requirements.txt`
- Credential failures during re-ingestion:
  - Confirm env vars are visible in current shell session
- Port 8000 already in use:
  - Another instance of the app (or the docs site) is likely running; stop it or pick a different `--port`
