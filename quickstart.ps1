# Quick Start Guide for AutoML-Insight

Write-Host "🤖 AutoML-Insight Quick Start" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Generate demo datasets
Write-Host ""
Write-Host "Generating demo datasets..." -ForegroundColor Yellow
python generate_demo_data.py

# Create necessary directories
Write-Host ""
Write-Host "Creating result directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "results/logs" | Out-Null
New-Item -ItemType Directory -Force -Path "results/runs" | Out-Null
New-Item -ItemType Directory -Force -Path "results/reports" | Out-Null

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To launch the dashboard, run:" -ForegroundColor Cyan
Write-Host "  streamlit run app/main.py" -ForegroundColor White
Write-Host ""
Write-Host "To run an experiment, run:" -ForegroundColor Cyan
Write-Host "  python experiments/run_experiment.py" -ForegroundColor White
Write-Host ""
