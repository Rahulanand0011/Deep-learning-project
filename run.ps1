Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Chat with PDF application..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

& "$ScriptDir\.venv\Scripts\streamlit.exe" run "$ScriptDir\app.py"
