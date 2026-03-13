# 🤖 Bybit USDT Perpetual Scalping Bot

> ⚠️ **UYARI**: Bu bot gerçek para kullanır. Kripto futures trading yüksek risk içerir.  
> Kaldıraç kullanımı sermayenizi hızla eritebilir. Yalnızca kaybetmeyi göze aldığınız  
> miktarla işlem yapın. Bu yazılım "olduğu gibi" sunulur, kâr garantisi yoktur.

---

## 📁 Dosya Yapısı

```
bybit_scalper/
├── main.py                 # Ana giriş noktası — asyncio orchestrator
├── config.py               # Tüm parametreler (kaldıraç, SL/TP, risk vb.)
├── models.py               # Veri yapıları (OrderBook, Signal, Position...)
├── websocket_handler.py    # Bybit V5 WebSocket — orderbook delta/snapshot
├── coinglass_fetcher.py    # Coinglass likidasyon heatmap API
├── signal_generator.py     # RSI, EMA, VWAP, ATR + sinyal üretimi
├── trade_executor.py       # Bybit REST API — emir verme, trailing stop
├── ml_optimizer.py         # RandomForest + PyTorch policy network
├── trade_logger.py         # JSONL trade logu + performance raporu
├── requirements.txt
└── .env.example
```

---

## 🚀 Kurulum

### 1. Python ortamı
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. TA-Lib (C kütüphanesi gerektirir)
```bash
# Ubuntu/Debian
sudo apt-get install -y libta-lib-dev
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# TA-Lib kurulmazsa numpy fallback otomatik kullanılır
```

### 3. API Anahtarları

#### Bybit Testnet
1. https://testnet.bybit.com → kayıt ol
2. API Management → Create API Key
3. İzinler: **Trade + Position** ✅, IP kısıtlaması önerilir

#### Coinglass (opsiyonel)
1. https://www.coinglass.com → üye ol → Free tier API key al
2. Limit: ~10 req/dk (scalping için yeterli)

```bash
cp .env.example .env
nano .env   # değerleri gir
```

### 4. Çalıştır

```bash
# .env'i yükle ve testnet'te başlat
source .env && python main.py

# Veya doğrudan env ile
BYBIT_API_KEY="xxx" BYBIT_API_SECRET="yyy" USE_TESTNET=true python main.py

# Trade loglarından backtest raporu üret
python main.py --backtest
```

---

## ⚙️ Parametreler (config.py)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `leverage` | 15x | Sabit kaldıraç |
| `stop_loss_pct` | %0.4 | Stop-loss yüzdesi |
| `take_profit_pct` | %1.0 | Take-profit yüzdesi |
| `trailing_stop_pct` | %0.3 | Trailing stop |
| `max_risk_per_trade_pct` | %2 | Trade başına max risk |
| `max_daily_drawdown_pct` | %10 | Günlük max drawdown → dur |
| `max_daily_pnl_cap_pct` | %5 | Günlük kâr hedefi → dur |
| `atr_skip_threshold_pct` | %1 | ATR bu değeri geçerse trade atla |
| `bid_wall_threshold_usdt` | 500K$ | Bid wall tanıma eşiği |
| `liquidation_cluster_min_usd` | 2M$ | Min likidasyon cluster büyüklüğü |
| `ml_retrain_every_n_trades` | 50 | Her N trade'de ML yeniden eğit |

---

## 📊 Sinyal Mantığı

### Long Koşulları (4/4 = STRONG, 3/4 = MEDIUM)
1. 📗 Orderbook'ta güçlü **bid wall** (≥500K USDT)
2. 💧 Fiyata ±%0.5 mesafede **long likidasyon cluster'ı** (alım baskısı yaratır)
3. 📈 **RSI(14) > 50** (momentum pozitif)
4. ➕ **EMA(9) > EMA(21)** (bullish crossover)

### Short Koşulları
1. 📕 **Ask wall** (≥500K USDT)
2. 💧 **Short likidasyon cluster'ı** yakında
3. 📉 **RSI(14) < 50**
4. ➖ **EMA(9) < EMA(21)**

### Filtreler
- ATR > %1 → çok volatil → trade atla
- Spread > %0.05 → çok geniş → atla
- Zayıf sinyal (2/4) → işlem yapma

---

## 🧠 ML Optimizasyonu

Her 50 trade sonrası:
1. Son 200 trade'i al
2. **RandomForest** (100 ağaç) ile eğit
3. Feature importance hesapla (RSI, EMA, orderbook imbalance, likidasyon proximity)
4. Sinyal skoru çarpanı üret [0.5 – 1.5x]
5. **PyTorch** policy network (opsiyonel) eğit

Model `logs/ml_model.pkl`'e kaydedilir ve yeniden başlatmada yüklenir.

---

## 📈 Log Dosyaları

```
logs/
├── bot.log              # Genel log (rotasyonlu, max 50MB)
├── trades.jsonl         # Her trade → bir satır JSON
└── performance.json     # Canlı performance raporu
```

### trades.jsonl örneği
```json
{"ts":"2024-01-15T10:23:45","symbol":"BTCUSDT","side":"Buy","entry":42150.5,
 "exit":42573.1,"pnl_usdt":18.42,"pnl_pct":1.0,"exit_reason":"take_profit","is_win":true}
```

---

## ⚠️ Risk Yönetimi Özeti

| Kural | Değer |
|-------|-------|
| Pozisyon başına max risk | %2 sermaye |
| Günlük max drawdown | %10 → bot durur |
| Günlük kâr hedefi | %5 → bot durur |
| Max açık pozisyon | 2 |
| ATR filtresi | >%1 → trade yok |
| Trailing stop | %0.3 |

---

## 🔄 Mainnet'e Geçiş

1. Testnet'te en az **100 trade** deneyin
2. Win rate > %55, Sharpe > 1.0 hedefleyin  
3. `.env`'de `USE_TESTNET=false` yapın
4. İlk hafta **düşük sermaye** (100-200 USDT) ile başlayın
5. `config.py`'de `leverage`'i **10x**'e düşürün

---

## 🐛 Sorun Giderme

**TA-Lib ImportError**: `pip install TA-Lib --no-build-isolation` deneyin veya numpy fallback otomatik devreye girer.

**WebSocket bağlantı kopuyor**: Testnet'te instabil olabilir. `ws_reconnect_delay=3` yapın.

**Coinglass 429 hatası**: Rate limit — `liquidation_update_interval=15` yapın.

**"Yeterli fiyat geçmişi yok"**: Bot başladıktan 30 saniye sonra otomatik düzelir.
