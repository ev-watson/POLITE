# Launcher shim. The implementation lives in
# qhy_alpaca/scripts/start_qhy_alpaca_server.ps1; this file only resolves the
# repo root and invokes it, so the server can be started from the familiar
# scripts/ directory. Do not "deduplicate" the two — they are not a fork.
#
# Step 2: start the SDK-direct QHY268M Alpaca server on port 11112.
# Run in a dedicated PowerShell window (leave it running):
#
#   .\scripts\launch_qhy_server.ps1
#
# Verify from another window:
#   curl http://localhost:11112/management/apiversions

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Inner = Join-Path $Root "qhy_alpaca\scripts\start_qhy_alpaca_server.ps1"

if (-not (Test-Path $Inner)) {
    Write-Error "Missing $Inner - git pull the latest POLITE repo."
}

& $Inner
