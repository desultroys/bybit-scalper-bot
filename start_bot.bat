@echo off
chcp 65001 >nul
title Bybit Scalper - Web Dashboard

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [HATA] Once setup.bat calistirin!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo ================================================
echo   Bybit Scalper Web Dashboard baslatiliyor...
echo   Tarayicide acin: http://localhost:8080
echo ================================================
echo.

start "" "http://localhost:8080"
python server.py

pause
