# Run Road Risk API (read-only, serves data/results)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Starting Road Risk API on http://127.0.0.1:8000"
Write-Host "Endpoints: GET /metadata, /risk/latest, /risk/ranking, /risk/by_zone, /risk/heatmap"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
