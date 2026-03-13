@echo off
chcp 65001 >nul
title BingX Scalper - Kurulum

echo.
echo ================================================
echo   BINGX SCALPER BOT - WINDOWS KURULUM
echo ================================================
echo.

python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [HATA] Python bulunamadi!
    echo Python 3.10+ indirin: https://www.python.org/downloads/
    echo Kurulum sirasinda "Add Python to PATH" secenegini isaretleyin!
    pause
    exit /b 1
)
echo [OK] Python bulundu.

python -m pip install --upgrade pip --quiet

IF NOT EXIST "venv" (
    echo [*] Sanal ortam olusturuluyor...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [*] Paketler kuruluyor...
pip install aiohttp websockets numpy pandas scikit-learn fastapi "uvicorn[standard]" python-dotenv requests --quiet
echo [OK] Paketler kuruldu.

pip install TA-Lib --quiet 2>nul || echo [UYARI] TA-Lib olmadan numpy fallback kullanilacak.

IF NOT EXIST ".env" (
    copy .env.example .env >nul
    echo [OK] .env olusturuldu - API keylerini dashboard'dan girin.
)

IF NOT EXIST "logs" mkdir logs

echo.
echo ================================================
echo   KURULUM TAMAMLANDI!
echo   Baslatmak icin: start_bot.bat
echo   Web: http://localhost:8080
echo ================================================
echo.
pause
