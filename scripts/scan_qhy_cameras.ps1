# Step 1: scan for QHY cameras via qhyccd.dll (no Alpaca server required).
# Run from POLITE repo root on the observatory Windows PC:
#
#   .\scripts\scan_qhy_cameras.ps1
#
# Optional DLL override:
#   $env:QHYCCD_DLL = "C:\Program Files\QHYCCD\AllInOne\sdk\x64\qhyccd.dll"
#   .\scripts\scan_qhy_cameras.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Scan = Join-Path $Root "qhy_alpaca\scripts\scan_qhy_cameras.py"

if (-not (Test-Path $Scan)) {
    Write-Error "Missing $Scan � git pull the latest POLITE repo."
}

if (Test-Path $VenvPy) {
    $Python = $VenvPy
} else {
    $Python = "python"
}

Write-Host "Close EZCAP and all QHY apps before scanning."
Write-Host ""
& $Python $Scan
exit $LASTEXITCODE
