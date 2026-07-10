# QHY268M SDK-direct bring-up for POLITE first light (observatory Windows PC).
#
# Usage (from POLITE repo root):
#   .\scripts\qhy268_bringup.ps1 -Step all          # guided full sequence
#   .\scripts\qhy268_bringup.ps1 -Step scan         # step 1 only
#   .\scripts\qhy268_bringup.ps1 -Step smoke-camera # step 4 only
#
# Prerequisites:
#   - git pull (this repo)
#   - .\.venv\Scripts\Activate.ps1
#   - .\scripts\install_qhy_alpaca_deps.ps1  (once)
#   - EZCAP and all QHY apps CLOSED before scan/connect
#   - ASCOM Remote Server running on :11111 for EFW + Pyxis (steps 5-6)

param(
    [ValidateSet("all", "scan", "server", "verify-server", "smoke-camera", "smoke-full", "night")]
    [string]$Step = "all",
    [string]$OutFits = "logs\qhy_smoke.fits",
    [double]$Exposure = 2.0,
    [string]$NightPlan = "night_plans\20260709.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

function Require-Venv {
    if (-not (Test-Path $VenvPy)) {
        Write-Error "Missing $VenvPy — create the venv per before_observations_checklist.md"
    }
}

function Step-Scan {
    Write-Host "`n=== Step 1: SDK scan ===" -ForegroundColor Cyan
    Write-Host "Close EZCAP first. Press Enter to continue or Ctrl+C to abort."
    Read-Host | Out-Null
    & (Join-Path $Root "scripts\scan_qhy_cameras.ps1")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nScan failed. Try:" -ForegroundColor Yellow
        Write-Host '  $env:QHYCCD_DLL = "C:\Program Files\QHYCCD\AllInOne\sdk\x64\qhyccd.dll"'
        Write-Host "  .\scripts\scan_qhy_cameras.ps1"
        exit $LASTEXITCODE
    }
}

function Step-Server {
    Write-Host "`n=== Step 2: Start camera Alpaca server ===" -ForegroundColor Cyan
    Write-Host "Open a NEW PowerShell window and run:"
    Write-Host "  cd $Root"
    Write-Host "  .\scripts\start_qhy_alpaca_server.ps1"
    Write-Host ""
    Write-Host "Leave that window running. Press Enter here when the server is up."
    Read-Host | Out-Null
}

function Step-VerifyServer {
    Write-Host "`n=== Step 3: Verify camera server ===" -ForegroundColor Cyan
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:11112/management/apiversions" -UseBasicParsing -TimeoutSec 5
        Write-Host "OK: $($r.Content)"
    } catch {
        Write-Error "Camera server not reachable on :11112. Start scripts\start_qhy_alpaca_server.ps1 first."
    }
    Write-Host "ASCOM Remote Server should still expose EFW + Pyxis on :11111 (unchanged)."
}

function Step-SmokeCamera {
    Write-Host "`n=== Step 4: Camera-only smoke test ===" -ForegroundColor Cyan
    Require-Venv
    $outFits = if ([System.IO.Path]::IsPathRooted($OutFits)) { $OutFits } else { Join-Path $Root $OutFits }
    $outDir = Split-Path -Parent $outFits
    if ($outDir -and -not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }
    & $VenvPy (Join-Path $Root "scripts\qhy_alpaca_smoke_test.py") `
        --exposure $Exposure --out $outFits
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Wrote $outFits" -ForegroundColor Green
}

function Step-SmokeFull {
    Write-Host "`n=== Step 5: Full chain smoke test ===" -ForegroundColor Cyan
    Write-Host "Requires PWI4 (:8220) + ASCOM Remote Server (:11111 EFW/Pyxis)."
    Require-Venv
    & $VenvPy (Join-Path $Root "scripts\observatory_smoke_test.py")
    exit $LASTEXITCODE
}

function Step-Night {
    Write-Host "`n=== Step 6: Night run ===" -ForegroundColor Cyan
    Require-Venv
    $plan = Join-Path $Root $NightPlan
    if (-not (Test-Path $plan)) {
        Write-Error "Missing night plan: $plan"
    }
    & $VenvPy (Join-Path $Root "scripts\plan_night.py") $plan --run
    exit $LASTEXITCODE
}

Set-Location $Root

switch ($Step) {
    "scan"          { Step-Scan; break }
    "server"        { Step-Server; break }
    "verify-server" { Step-VerifyServer; break }
    "smoke-camera"  { Step-VerifyServer; Step-SmokeCamera; break }
    "smoke-full"    { Step-VerifyServer; Step-SmokeFull; break }
    "night"         { Step-VerifyServer; Step-Night; break }
    "all" {
        Step-Scan
        Step-Server
        Step-VerifyServer
        Step-SmokeCamera
        Write-Host "`nSteps 5-6 are manual when ready:" -ForegroundColor Yellow
        Write-Host "  .\scripts\qhy268_bringup.ps1 -Step smoke-full"
        Write-Host "  .\scripts\qhy268_bringup.ps1 -Step night"
        break
    }
}

Write-Host "`nDone ($Step)." -ForegroundColor Green
