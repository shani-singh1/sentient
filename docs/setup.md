# Setup

## Prerequisites

- Python 3.10+
- Windows PowerShell (project scripts are PowerShell-first)
- Network access for data providers
- Credentials for:
  - ERA5 (`CDS_API_KEY`)
  - Google Earth Engine (`GEE_PROJECT_ID`, plus local authentication)

## 1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For documentation site (optional but recommended):

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

Default config used by run scripts:

`config/pipeline.bengaluru.2020_2024.json`

## 4) First smoke run (recommended)

Run API smoke checks after generating outputs:

```powershell
python scripts/smoke_test_api.py
```

## 5) Run documentation UI locally

```powershell
mkdocs serve
```

Then open:

`http://127.0.0.1:8000`

## Common Setup Issues

- `streamlit` command not found:
  - Use `python -m streamlit run src/frontend/app.py --server.port 8501`
- Missing parquet support:
  - Ensure `pyarrow` is installed from `requirements.txt`
- Credential failures:
  - Confirm env vars are visible in current shell session
