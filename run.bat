@echo off
title Chat with PDF
echo ========================================================
echo Starting Chat with PDF application...
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
