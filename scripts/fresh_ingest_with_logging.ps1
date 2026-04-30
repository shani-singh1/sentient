param(
  [string]$City = "Bengaluru",
  [string]$Bbox = "77.45,12.8,77.75,13.1",
  [string]$StartDate = "2020-01-01",
  [string]$EndDate = "2024-12-31",
  [switch]$ForceClean,
  [switch]$SkipSentinel1 = $false,
  [int]$Sentinel2Scale = 30,
  [switch]$RunEra5 = $false,
  [switch]$RunOsm = $false
)

$ErrorActionPreference = "Stop"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "data/logs"
New-Item -Path $logDir -ItemType Directory -Force | Out-Null
$logFile = Join-Path $logDir "fresh_ingest_$ts.log"

function Append-LogLine {
  param([string]$Text)
  $line = $Text + [Environment]::NewLine
  for ($i = 0; $i -lt 5; $i++) {
    try {
      [System.IO.File]::AppendAllText($logFile, $line)
      return
    } catch {
      Start-Sleep -Milliseconds 200
    }
  }
  throw "Unable to append to log file: $logFile"
}

function Write-Log {
  param([string]$Message)
  $line = "$(Get-Date -Format s) $Message"
  Write-Host $line
  Append-LogLine $line
}

function Run-Step {
  param([string]$Name, [string]$Cmd, [int]$StepNum = 0, [int]$StepTotal = 0)
  $prefix = ""
  if ($StepTotal -gt 0) { $prefix = "[$StepNum/$StepTotal] " }
  Write-Log "${prefix}START $Name"
  Write-Log "CMD $Cmd"
  $previousErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = Invoke-Expression $Cmd 2>&1
  } finally {
    $ErrorActionPreference = $previousErrorAction
  }
  if ($output) {
    $output | ForEach-Object {
      $txt = "$_"
      Write-Host $txt
      Append-LogLine $txt
    }
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Log "FAIL $Name (exit=$LASTEXITCODE)"
    exit $LASTEXITCODE
  }
  Write-Log "${prefix}DONE $Name"
}

if ($ForceClean) {
  Write-Log "Cleaning stale raw source data for fresh ingestion."
  $sources = @("sentinel1", "sentinel2", "landsat", "nightlights", "population")
  foreach ($source in $sources) {
    foreach ($year in 2020..2024) {
      $target = Join-Path "data/raw/$source" "$year"
      if (Test-Path $target) {
        Remove-Item -Path $target -Recurse -Force
        Write-Log "Deleted $target"
      } else {
        Write-Log "Skip missing $target"
      }
    }
  }
}

$step = 0
$stepsPerYear = 4
if (-not $SkipSentinel1) { $stepsPerYear = 5 }
$stepTotal = 5 * $stepsPerYear
if ($RunEra5) { $stepTotal++ }
if ($RunOsm) { $stepTotal++ }

if ($RunEra5) {
  $step++
  Run-Step "ERA5 ingestion" "python -m src.ingestion.era5_ingest --city $City --bbox '$Bbox' --start-date $StartDate --end-date $EndDate" -StepNum $step -StepTotal $stepTotal
}

if ($RunOsm) {
  $step++
  Run-Step "OSM ingestion" "python -m src.ingestion.osm_ingest --city $City --bbox '$Bbox' --start-date $StartDate --end-date $EndDate" -StepNum $step -StepTotal $stepTotal
}

foreach ($year in 2020..2024) {
  if (-not $SkipSentinel1) {
    $step++
    Run-Step "Sentinel1 $year" "python -m src.ingestion.sentinel1_ingest --city $City --bbox '$Bbox' --start-date $year-01-01 --end-date $year-12-31 --scale 20" -StepNum $step -StepTotal $stepTotal
  }
  $step++
  Run-Step "Sentinel2 $year" "python -m src.ingestion.sentinel2_ingest --city $City --bbox '$Bbox' --start-date $year-01-01 --end-date $year-12-31 --max-cloud 40 --scale $Sentinel2Scale" -StepNum $step -StepTotal $stepTotal
  $step++
  Run-Step "Landsat $year" "python -m src.ingestion.landsat_ingest --city $City --bbox '$Bbox' --start-date $year-01-01 --end-date $year-12-31 --max-cloud 50 --scale 30" -StepNum $step -StepTotal $stepTotal
  $step++
  Run-Step "Nightlights $year" "python -m src.ingestion.nightlights_ingest --city $City --bbox '$Bbox' --start-date $year-01-01 --end-date $year-12-31 --scale 500" -StepNum $step -StepTotal $stepTotal
  $step++
  Run-Step "Population $year" "python -m src.ingestion.population_ingest --city $City --bbox '$Bbox' --start-date $year-01-01 --end-date $year-12-31 --scale 100" -StepNum $step -StepTotal $stepTotal
}

Write-Log "Fresh ingestion complete. Log file: $logFile"
