@echo off
echo ========================================
echo  KURAL + BOT KARSILASTIRMA TESTI
echo  Her iki pencere de acik kalsin!
echo ========================================
echo.

cd /d %~dp0

echo Gerekli paketler kontrol ediliyor...
python -m pip install python-dotenv anthropic requests numpy -q
echo Paketler hazir.
echo.

echo [1/2] Kural Tabanlı Sinyal başlatılıyor (her 60dk)...
start "KURAL - rule_signal" cmd /k "python -u rule_signal.py --interval 15"

timeout /t 3 /nobreak > nul

echo [2/2] AI Bot başlatılıyor (her 60dk, dry-run)...
start "AI BOT - ai_trader" cmd /k "python -u ai_trader.py --dry-run --interval 15 --min-confidence 60"

echo.
echo Her iki pencere de arkaplanda calisiyor.
echo Yarin aksamüzeri bu konusmaya geri gel, karsilastirma yapacagiz.
echo.
pause
