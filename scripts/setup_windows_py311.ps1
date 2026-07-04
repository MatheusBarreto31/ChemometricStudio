param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Ensuring PowerShell can run local activation scripts..."
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

Write-Host "[2/4] Ensuring Python 3.11 is installed..."
$has311 = py -0p | Select-String "-V:3.11"
if (-not $has311) {
    winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
}

Write-Host "[3/4] Creating virtual environment at $VenvPath ..."
if (-not (Test-Path $VenvPath)) {
    py -3.11 -m venv $VenvPath
}

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "[4/4] Installing dependencies from requirements.txt ..."
& $pythonExe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\\$VenvPath\\Scripts\\Activate.ps1"
Write-Host "Run app with:  .\\$VenvPath\\Scripts\\python.exe launcher.py"