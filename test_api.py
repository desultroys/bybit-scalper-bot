"""
test_api.py - API key'ini test et
Kullanim: python test_api.py
"""
import hashlib, hmac, json, time, sys
try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# .env dosyasini oku
import os
from pathlib import Path

def load_env():
    env = {}
    for f in [".env", "../.env"]:
        if Path(f).exists():
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env

env = load_env()
API_KEY    = env.get("BYBIT_API_KEY", "")
API_SECRET = env.get("BYBIT_API_SECRET", "")

print("=" * 50)
print("  BYBIT API KEY TEST")
print("=" * 50)
print()

# Key kontrol
if not API_KEY or API_KEY == "your_api_key_here":
    print("[HATA] .env dosyasinda API_KEY bulunamadi!")
    print("  .env dosyasini acip su formatta olduğunu kontrol et:")
    print("  BYBIT_API_KEY=senin_key_buraya")
    sys.exit(1)

if not API_SECRET or API_SECRET == "your_api_secret_here":
    print("[HATA] .env dosyasinda API_SECRET bulunamadi!")
    sys.exit(1)

print(f"API Key : {API_KEY[:8]}...{API_KEY[-4:]}  (uzunluk: {len(API_KEY)})")
print(f"Secret  : {'*' * 20}  (uzunluk: {len(API_SECRET)})")
print()

# Bybit sunucu saatini al
BASE = "https://api-testnet.bybit.com"
print("[1] Sunucu saati kontrol ediliyor...")
try:
    r = requests.get(BASE + "/v5/market/time", timeout=10)
    server_ts = int(r.json()["result"]["timeNano"]) // 1_000_000
    local_ts  = int(time.time() * 1000)
    diff      = server_ts - local_ts
    print(f"    Sunucu: {server_ts} | Yerel: {local_ts} | Fark: {diff}ms")
    if abs(diff) > 5000:
        print(f"    [UYARI] Saat farki {diff}ms — cok fazla ama otomatik duzeltiliyor")
    else:
        print("    [OK] Saat farki normal")
except Exception as e:
    print(f"    [HATA] Sunucuya ulasilamadi: {e}")
    sys.exit(1)

# Imzali istek
print()
print("[2] Imzali API istegi gonderiliyor...")
recv_window = 20000  # 20 saniye - genis pencere
timestamp   = str(server_ts)  # sunucu saatini kullan
params      = {"accountType": "UNIFIED", "coin": "USDT"}
query       = "accountType=UNIFIED&coin=USDT"
payload     = timestamp + API_KEY + str(recv_window) + query
signature   = hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

headers = {
    "X-BAPI-API-KEY":     API_KEY,
    "X-BAPI-TIMESTAMP":   timestamp,
    "X-BAPI-SIGN":        signature,
    "X-BAPI-RECV-WINDOW": str(recv_window),
}

try:
    r = requests.get(
        BASE + "/v5/account/wallet-balance",
        headers=headers, params=params, timeout=10
    )
    data = r.json()
    code = data.get("retCode")
    msg  = data.get("retMsg", "")

    if code == 0:
        print("    [OK] API KEY GECERLI!")
        try:
            coins = data["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == "USDT":
                    print(f"    USDT Bakiye: {c.get('availableToWithdraw','?')}")
        except Exception:
            print("    Bakiye bilgisi alinamadi ama key gecerli")
    elif code == 10003 or "invalid" in msg.lower():
        print(f"    [HATA] API key gecersiz (kod:{code})")
        print()
        print("    COZUM ONERILERI:")
        print("    1. testnet.bybit.com adresinden key aldığına emin ol")
        print("       (bybit.com degil - FARKLI site!)")
        print("    2. Key kopyalarken bosluk girmedığine emin ol")
        print("    3. Key'in Trade + Position izinleri var mi?")
        print("    4. Key'i sil, yenisini olustur")
    elif code == 10004 or "timestamp" in msg.lower():
        print(f"    [HATA] Timestamp sorunu (kod:{code}): {msg}")
        print("    Windows saatini internet ile senkronize et:")
        print("    Gorev Cubugu > Saat > Tarih/Saat ayarla > Simdi senkronize et")
    else:
        print(f"    [HATA] kod:{code} mesaj:{msg}")
    
    print()
    print(f"    Ham yanit: {json.dumps(data, ensure_ascii=False)[:300]}")

except Exception as e:
    print(f"    [HATA] {e}")

print()
print("[3] Key Bilgisi Ozeti:")
print(f"    Key uzunlugu: {len(API_KEY)} (normal: 18-20 karakter)")
print(f"    Secret uzunlugu: {len(API_SECRET)} (normal: 36-40 karakter)")
print(f"    Bosluk var mi? Key:{'EVET!' if ' ' in API_KEY else 'Hayir'}  Secret:{'EVET!' if ' ' in API_SECRET else 'Hayir'}")
print()
input("Devam etmek icin Enter'a basin...")
