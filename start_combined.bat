@echo off
echo ============================================
echo  HİBRİT SİNYAL — combined_signal.py
echo  KURAL Skoru + Fibonacci S/R + Claude AI
echo ============================================
echo.

cd /d %~dp0

echo Gerekli paketler kontrol ediliyor...
python -m pip install python-dotenv anthropic requests numpy -q
echo Paketler hazir.
echo.

echo Hibrit bot baslatiliyor (her 15dk, BTC-USDT)...
start "HİBRİT - combined_signal" cmd /k "python -u combined_signal.py --interval 15 --symbol BTC-USDT"

echo.
echo Pencere acildi. Log dosyasi: logs/combined_signals.json
echo.
pause
