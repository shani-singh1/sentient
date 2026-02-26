# Run Road Risk frontend (Streamlit)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Starting Road Risk frontend on http://localhost:8501"
python -m streamlit run src/frontend/app.py --server.port 8501
