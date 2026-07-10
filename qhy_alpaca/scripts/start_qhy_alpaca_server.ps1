# Start the POLITE QHY268M Alpaca camera server (SDK-direct, bypasses ASCOM COM).
# Run from the observatory Windows PC in PowerShell:
#
#   cd C:\path\to\POLITE\qhy_alpaca
#   .\scripts\start_qhy_alpaca_server.ps1
#
# Prerequisites:
#   - EZCAP and all QHY apps CLOSED (USB exclusive access)
#   - Python 3.13 venv with POLITE deps OR a dedicated venv here
#   - qhyccd.dll discoverable (see scan_qhy_cameras.py)
#
# Camera Alpaca endpoint: http://localhost:11112  (device 0)
# EFW + Pyxis stay on ASCOM Remote Server: http://localhost:11111

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Src = Join-Path $Root "src"
$Config = Join-Path $Root "config.windows.yaml"

if (-not (Test-Path $Config)) {
    Write-Error "Missing $Config"
}

$env:QHYCCD_ALPACA_CONFIG = $Config
$env:PYTHONPATH = $Src

# Prefer POLITE repo venv (qhy_alpaca/ sits at the repo root).
$PoliteRoot = (Resolve-Path (Join-Path $Root "..")).Path
$VenvPy = Join-Path $PoliteRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    $Python = $VenvPy
} else {
    $Python = "python"
}

Write-Host "POLITE QHY Alpaca server"
Write-Host "  config: $Config"
Write-Host "  python: $Python"
Write-Host "  endpoint: http://localhost:11112 (camera device 0)"
Write-Host ""
Write-Host "Close EZCAP before connecting. Ctrl+C to stop."
Write-Host ""

Set-Location $Root
& $Python (Join-Path $Src "main.py")
