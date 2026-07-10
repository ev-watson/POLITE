# Step 2: start the SDK-direct QHY268M Alpaca server on port 11112.
# Run in a dedicated PowerShell window (leave it running):
#
#   .\scripts\start_qhy_alpaca_server.ps1
#
# Verify from another window:
#   curl http://localhost:11112/management/apiversions

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Inner = Join-Path $Root "third_party\alpaca-qhyccd-camera\scripts\start_qhy_alpaca_server.ps1"

if (-not (Test-Path $Inner)) {
    Write-Error "Missing $Inner — git pull the latest POLITE repo."
}

& $Inner
