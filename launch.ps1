<#
.SYNOPSIS
  One-click launcher for MedicalAI - Light Weight
.DESCRIPTION
  Checks for Python, sets up venv, installs deps, and launches the app.
  Double-click this file or run: powershell -File launch.ps1
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Host.UI.RawUI.WindowTitle = "MedicalAI - Light Weight"

# ── Check Python ──
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        if ($v -match "Python 3\.(1[0-9]|[0-9]+)") {
            $python = $cmd
            break
        }
    } catch {}
}
if (-not $python) {
    Write-Host "Python 3.10+ is required but not found." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Using: $(& $python --version)" -ForegroundColor Green

# ── Virtual Environment ──
$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $python -m venv $venvPath
    if (-not $?) { throw "Failed to create venv" }
}

# ── Activate ──
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

# ── Install Dependencies ──
$reqPath = Join-Path $ProjectRoot "requirements.txt"
if (Test-Path $reqPath) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    pip install -q -r $reqPath 2>&1 | Out-Null
    if (-not $?) {
        Write-Host "Retrying with full output..." -ForegroundColor Yellow
        pip install -r $reqPath
    }
}

# ── Check / Generate Default Models ──
$defaultClassifier = Join-Path $ProjectRoot "models\default\fusion_classifier.onnx"
$checkpoint = Join-Path $ProjectRoot "checkpoints\fusion_model.pth"
$onnxFull = Join-Path $ProjectRoot "checkpoints\onnx_full\fusion_full.onnx"

if ((-not (Test-Path $checkpoint)) -and (-not (Test-Path $onnxFull)) -and (-not (Test-Path $defaultClassifier))) {
    Write-Host "No models found. Generating default models..." -ForegroundColor Yellow
    python setup_default.py
    Write-Host "Default models generated. The app will work immediately." -ForegroundColor Green
}

if (Test-Path $checkpoint) {
    Write-Host "Trained model found." -ForegroundColor Green
} elseif (Test-Path $onnxFull) {
    Write-Host "ONNX pipeline found." -ForegroundColor Green
} elseif (Test-Path $defaultClassifier) {
    Write-Host "Default model found. Train a proper model for accurate results." -ForegroundColor Yellow
}

# ── Ask how to launch ──
Write-Host ""
Write-Host "MedicalAI - Light Weight" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host "1) Web UI (recommended - opens in browser)"
Write-Host "2) Command-line interface (CLI)"
Write-Host ""

$choice = Read-Host "Select (1 or 2)"

if ($choice -eq "2") {
    Write-Host "Launching CLI..." -ForegroundColor Green
    python run.py
} else {
    Write-Host "Launching web UI..." -ForegroundColor Green
    python web_ui.py
}

# ── Keep window open on crash ──
if (-not $?) {
    Write-Host "App exited with error. Press Enter to close." -ForegroundColor Red
    Read-Host
}
