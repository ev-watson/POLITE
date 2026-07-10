# Install Python packages for the SDK-direct QHY Alpaca camera server.
# Run once on the observatory Windows PC from the POLITE repo root:
#
#   .\scripts\install_qhy_alpaca_deps.ps1
#
# Uses the repo .venv when present.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Req = Join-Path $Root "qhy_alpaca\requirements.txt"

if (Test-Path $VenvPy) {
    $Python = $VenvPy
    $Pip = Join-Path $Root ".venv\Scripts\pip.exe"
} else {
    $Python = "python"
    $Pip = "pip"
}

if (-not (Test-Path $Req)) {
    Write-Error "Missing $Req � git pull the latest POLITE repo."
}

Write-Host "Installing QHY Alpaca server dependencies into:"
Write-Host "  $Python"
& $Pip install -r $Req
Write-Host "Done."
