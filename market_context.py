"""
market_context.py — Piyasa verisi toplayıcı
Claude'a gönderilecek yapılandırılmış bağlamı hazırlar.
Kaynaklar: BingX OHLCV, orderbook, funding; CoinGlass likidasyon
"""
import time, os, json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone


BINGX_BASE = "https://open-api.bingx.com"


def _get(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── OHLCV ──────────────────────────────────────────────────────────
def fetch_candles(symbol="BTC-USDT", interval="15m", limit=100) -> list[dict]:
    raw = _get(f"{BINGX_BASE}/openApi/swap/v3/quote/klines", {
        "symbol": symbol, "interval": interval, "limit": str(limit)
    })
    candles = raw.get("data", []) if isinstance(raw, dict) else raw
    result = []
    for c in candles:
        try:
            if isinstance(c, dict):
                result.append({
                    "ts": int(c.get("time", c.get("t", 0))),
                    "o": float(c.get("open", 0)),
                    "h": float(c.get("high", 0)),
                    "l": float(c.get("low", 0)),
                    "c": float(c.get("close", 0)),
                    "v": float(c.get("volume", 0)),
                })
        except:
            pass
    return sorted(result, key=lambda x: x["ts"])


# ── Orderbook ──────────────────────────────────────────────────────
def fetch_orderbook(symbol="BTC-USDT", depth=20) -> dict:
    raw = _get(f"{BINGX_BASE}/openApi/swap/v2/quote/depth", {
        "symbol": symbol, "limit": str(depth)
    })
    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    bids = data.get("bids", [])
    asks = data.get("asks", [])

    def parse_levels(levels):
        result = []
        for lv in levels[:depth]:
            try:
                if isinstance(lv, list):
                    result.append({"p": float(lv[0]), "q": float(lv[1])})
                elif isinstance(lv, dict):
                    result.append({"p": float(lv.get("price", 0)), "q": float(lv.get("quantity", 0))})
            except:
                pass
        return result

    parsed_bids = parse_levels(bids)
    parsed_asks = parse_levels(asks)

    bid_vol = sum(l["p"] * l["q"] for l in parsed_bids)
    ask_vol = sum(l["p"] * l["q"] for l in parsed_asks)
    total = bid_vol + ask_vol + 1e-10
    imbalance = (bid_vol - ask_vol) / total  # +1 = saf alıcı, -1 = saf satıcı

    # Büyük duvarlar (top 3)
    top_bids = sorted(parsed_bids, key=lambda x: x["q"], reverse=True)[:3]
    top_asks = sorted(parsed_asks, key=lambda x: x["q"], reverse=True)[:3]

    return {
        "bid_vol_usd": round(bid_vol, 0),
        "ask_vol_usd": round(ask_vol, 0),
        "imbalance": round(imbalance, 3),  # >0.2 = alıcı baskısı
        "top_bids": top_bids,
        "top_asks": top_asks,
        "best_bid": parsed_bids[0]["p"] if parsed_bids else 0,
        "best_ask": parsed_asks[0]["p"] if parsed_asks else 0,
    }


# ── Funding Rate ───────────────────────────────────────────────────
def fetch_funding(symbol="BTC-USDT") -> dict:
    raw = _get(f"{BINGX_BASE}/openApi/swap/v2/quote/premiumIndex", {
        "symbol": symbol
    })
    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    if isinstance(data, list) and data:
        data = data[0]
    try:
        fr = float(data.get("lastFundingRate", data.get("fundingRate", 0)))
        mark = float(data.get("markPrice", 0))
        index = float(data.get("indexPrice", 0))
    except:
        fr, mark, index = 0, 0, 0
    return {
        "funding_rate": round(fr * 100, 4),  # % cinsinden
        "mark_price": mark,
        "index_price": index,
        "basis_pct": round((mark - index) / (index + 1e-10) * 100, 4) if index else 0,
    }


# ── Smart Money Concepts ──────────────────────────────────────────
def compute_smart_money(candles: list[dict]) -> dict:
    """
    ICT / Smart Money kavramları:
    - Swing High/Low  : 3-bar pivot noktaları
    - BOS             : Break of Structure (trend onayı)
    - Liquidity Sweep : Stop avı hareketleri
    - Order Block (OB): Kurumsal giriş bölgeleri
    - Fair Value Gap  : Fiyat imbalance boşlukları
    - Premium/Discount: Range içinde fiyat konumu
    """
    if len(candles) < 20:
        return {}

    highs  = [c["h"] for c in candles]
    lows   = [c["l"] for c in candles]
    closes = [c["c"] for c in candles]
    opens  = [c["o"] for c in candles]
    price  = closes[-1]

    n = len(candles)
    lookback = min(50, n - 2)

    # ── Swing High / Low (3-bar pivot) ────────────────────────────
    swing_highs = []
    swing_lows  = []
    for i in range(1, lookback + 1):
        idx = n - 1 - i  # 0-tabanlı, ortadaki mum
        if idx < 1 or idx >= n - 1:
            continue
        if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx + 1]:
            swing_highs.append(highs[idx])
        if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx + 1]:
            swing_lows.append(lows[idx])

    last_sh = swing_highs[0] if swing_highs else max(highs[-20:])
    last_sl = swing_lows[0]  if swing_lows  else min(lows[-20:])

    # ── Break of Structure ─────────────────────────────────────────
    bos_bull = price > last_sh   # Bullish BOS: son swing high kırıldı
    bos_bear = price < last_sl   # Bearish BOS: son swing low kırıldı

    # ── Liquidity Sweep (son 5 mum) ────────────────────────────────
    # High sweep: wick geçti ama kırmızı kapandı → short fırsatı
    # Low  sweep: wick geçti ama yeşil kapandı  → long fırsatı
    liq_sweep = "yok"
    for i in range(-5, 0):
        h = highs[i]; l = lows[i]; c = closes[i]; o = opens[i]
        if h > last_sh and c < last_sh:
            liq_sweep = f"SELL_SIDE_SWEEP @{last_sh:.0f}"
            break
        if l < last_sl and c > last_sl:
            liq_sweep = f"BUY_SIDE_SWEEP @{last_sl:.0f}"
            break

    # ── Order Blocks (son 30 mum) ──────────────────────────────────
    atr_approx = float(np.mean([highs[i] - lows[i] for i in range(-14, 0)]))
    bull_ob = None
    bear_ob = None
    for i in range(max(-30, -(n - 1)), -1):
        next_body = abs(closes[i + 1] - opens[i + 1])
        if next_body < atr_approx * 1.5:
            continue
        # Sonraki mum güçlü yukarı → önceki kırmızı = Bullish OB
        if closes[i + 1] > opens[i + 1] and closes[i] < opens[i] and bull_ob is None:
            bull_ob = {"high": round(opens[i], 2), "low": round(lows[i], 2)}
        # Sonraki mum güçlü aşağı → önceki yeşil = Bearish OB
        if closes[i + 1] < opens[i + 1] and closes[i] > opens[i] and bear_ob is None:
            bear_ob = {"high": round(highs[i], 2), "low": round(closes[i], 2)}

    in_bull_ob = bool(bull_ob and bull_ob["low"] <= price <= bull_ob["high"])
    in_bear_ob = bool(bear_ob and bear_ob["low"] <= price <= bear_ob["high"])

    # ── Fair Value Gap (son 20 mum) ────────────────────────────────
    # Bullish FVG: highs[i-1] < lows[i+1]  → boşluk destek
    # Bearish FVG: lows[i-1]  > highs[i+1] → boşluk direnç
    bull_fvgs = []
    bear_fvgs = []
    for i in range(max(-20, -(n - 1)), -1):
        if highs[i - 1] < lows[i + 1]:
            bull_fvgs.append({"top": round(lows[i + 1], 2), "bot": round(highs[i - 1], 2)})
        if lows[i - 1] > highs[i + 1]:
            bear_fvgs.append({"top": round(lows[i - 1], 2), "bot": round(highs[i + 1], 2)})

    # Fiyatın altındaki en yakın bullish FVG (destek)
    nearest_bull_fvg = None
    below = [f for f in bull_fvgs if f["top"] < price]
    if below:
        nearest_bull_fvg = max(below, key=lambda x: x["top"])

    # Fiyatın üstündeki en yakın bearish FVG (direnç)
    nearest_bear_fvg = None
    above = [f for f in bear_fvgs if f["bot"] > price]
    if above:
        nearest_bear_fvg = min(above, key=lambda x: x["bot"])

    # ── Premium / Discount ─────────────────────────────────────────
    rng_high = max(highs[-50:])
    rng_low  = min(lows[-50:])
    rng_mid  = (rng_high + rng_low) / 2
    pd_pct   = (price - rng_mid) / (rng_high - rng_low + 1e-10) * 100

    result: dict = {
        "last_swing_high":      round(last_sh, 2),
        "last_swing_low":       round(last_sl, 2),
        "bos_bullish":          bos_bull,
        "bos_bearish":          bos_bear,
        "liq_sweep":            liq_sweep,
        "range_high":           round(rng_high, 2),
        "range_low":            round(rng_low, 2),
        "premium_discount_pct": round(float(pd_pct), 1),
        "in_premium":           pd_pct > 10,
        "in_discount":          pd_pct < -10,
        "in_bull_ob":           in_bull_ob,
        "in_bear_ob":           in_bear_ob,
    }
    if bull_ob:
        result["bull_ob"] = bull_ob
    if bear_ob:
        result["bear_ob"] = bear_ob
    if nearest_bull_fvg:
        result["bull_fvg"] = nearest_bull_fvg
    if nearest_bear_fvg:
        result["bear_fvg"] = nearest_bear_fvg

    return result


# ── Likidasyon Verisi (OI + L/S Ratio) ────────────────────────────
def fetch_liquidation_data(symbol="BTC-USDT", current_price: float = 0) -> dict:
    """
    BingX Open Interest + Long/Short Ratio kullanarak tahmini
    likidasyon bölgelerini hesaplar.
    """
    # Open Interest
    oi_raw = _get(f"{BINGX_BASE}/openApi/swap/v2/quote/openInterest", {"symbol": symbol})
    oi_data = oi_raw.get("data", {}) if isinstance(oi_raw, dict) else {}

    # Long/Short Ratio (son 5dk)
    ls_raw = _get(f"{BINGX_BASE}/openApi/swap/v2/quote/globalLongShortAccountRatio", {
        "symbol": symbol, "period": "5m", "limit": "1"
    })
    ls_list = ls_raw.get("data", []) if isinstance(ls_raw, dict) else []
    ls_data = ls_list[0] if ls_list else {}

    try:
        oi_usd = float(oi_data.get("openInterest", 0))
    except:
        oi_usd = 0

    try:
        long_r  = float(ls_data.get("longAccount",  ls_data.get("longRatio",  0.5)))
        short_r = float(ls_data.get("shortAccount", ls_data.get("shortRatio", 0.5)))
        # Bazı endpoint'ler 0-1 aralığında, bazıları 0-100
        if long_r > 1:
            long_r /= 100; short_r /= 100
    except:
        long_r, short_r = 0.5, 0.5

    long_oi  = oi_usd * long_r
    short_oi = oi_usd * short_r

    # Tahmini likidasyon seviyeleri (10x ve 20x kaldıraç)
    liq = {}
    if current_price:
        liq = {
            "long_liq_10x":  round(current_price * 0.90, 0),   # -%10
            "long_liq_20x":  round(current_price * 0.95, 0),   # -%5
            "short_liq_10x": round(current_price * 1.10, 0),   # +%10
            "short_liq_20x": round(current_price * 1.05, 0),   # +%5
        }

    if long_r > 0.58:
        dominant = "long_agir"      # Long taraf aşırı kalabalık → short cascade riski
    elif short_r > 0.58:
        dominant = "short_agir"     # Short taraf aşırı kalabalık → long cascade riski
    else:
        dominant = "dengeli"

    return {
        "oi_usd":      round(oi_usd, 0),
        "long_pct":    round(long_r * 100, 1),
        "short_pct":   round(short_r * 100, 1),
        "long_oi_usd": round(long_oi, 0),
        "short_oi_usd":round(short_oi, 0),
        "dominant":    dominant,
        **liq,
    }


# ── Teknik göstergeler (son mumdan) ───────────────────────────────
def compute_technicals(candles: list[dict]) -> dict:
    if len(candles) < 50:
        return {}
    c = np.array([x["c"] for x in candles])
    h = np.array([x["h"] for x in candles])
    l = np.array([x["l"] for x in candles])
    v = np.array([x["v"] for x in candles])

    def ema(a, p):
        e = np.empty(len(a)); e[0] = a[0]; k = 2/(p+1)
        for i in range(1, len(a)): e[i] = k*a[i] + (1-k)*e[i-1]
        return e

    def rsi(a, p=14):
        d = np.diff(a)
        g = np.where(d > 0, d, 0.); l_ = np.where(d < 0, -d, 0.)
        ag = g[:p].mean(); al = l_[:p].mean()
        for i in range(p, len(a)-1):
            ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l_[i])/p
        return 100 - 100/(1 + ag/(al+1e-10))

    def atr(h_, l_, c_, p=14):
        tr = max(h_[-1]-l_[-1], abs(h_[-1]-c_[-2]), abs(l_[-1]-c_[-2]))
        return tr

    e9  = ema(c, 9)[-1]
    e21 = ema(c, 21)[-1]
    e50 = ema(c, 50)[-1]
    e200= ema(c, 200)[-1] if len(c) >= 200 else ema(c, len(c))[-1]
    rsi_val = rsi(c)
    atr_val = atr(h, l, c)
    price = c[-1]

    # Hacim trendi
    vol_avg = v[-20:].mean()
    vol_ratio = v[-1] / (vol_avg + 1e-10)

    # Momentum
    mom3 = (c[-1] - c[-4]) / c[-4] * 100
    mom8 = (c[-1] - c[-9]) / c[-9] * 100

    # Son 3 mum rengi
    candle_colors = []
    for i in range(-3, 0):
        candle_colors.append("yesil" if candles[i]["c"] >= candles[i]["o"] else "kirmizi")

    # VWAP (son 50 mum, günlük reset yok — periyodik VWAP)
    tp = (h + l + c) / 3  # typical price
    vwap = float(np.sum(tp[-50:] * v[-50:]) / (np.sum(v[-50:]) + 1e-10))
    vwap_dist_pct = (price - vwap) / vwap * 100  # + = üstünde, - = altında

    # CVD — Cumulative Volume Delta (son 50 mum)
    # Her mum: bullish → +hacim, bearish → -hacim
    o = np.array([x["o"] for x in candles])
    delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    cvd = float(np.sum(delta[-50:]))
    # CVD trendi: son 10 vs önceki 10
    cvd_recent = float(np.sum(delta[-10:]))
    cvd_prev   = float(np.sum(delta[-20:-10]))
    cvd_trend  = "yukari" if cvd_recent > cvd_prev else "asagi"

    return {
        "price": round(price, 2),
        "ema9": round(e9, 2),
        "ema21": round(e21, 2),
        "ema50": round(e50, 2),
        "ema200": round(e200, 2),
        "rsi": round(rsi_val, 1),
        "atr": round(atr_val, 2),
        "atr_pct": round(atr_val / price * 100, 3),
        "vol_ratio": round(vol_ratio, 2),
        "mom3_pct": round(mom3, 3),
        "mom8_pct": round(mom8, 3),
        "last_3_candles": candle_colors,
        "above_ema200": price > e200,
        "above_ema50": price > e50,
        "ema9_above_21": e9 > e21,
        "vwap": round(vwap, 2),
        "vwap_dist_pct": round(vwap_dist_pct, 3),
        "above_vwap": price > vwap,
        "cvd": round(cvd, 2),
        "cvd_trend": cvd_trend,
        "cvd_recent": round(cvd_recent, 2),
    }


# ── Ana toplayıcı ──────────────────────────────────────────────────
def get_full_context(symbol="BTC-USDT", interval="15m") -> dict:
    """Claude'a gönderilecek tam piyasa bağlamını döndürür."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    candles = fetch_candles(symbol, interval, limit=220)
    ob      = fetch_orderbook(symbol, depth=20)
    funding = fetch_funding(symbol)
    tech    = compute_technicals(candles)
    smc     = compute_smart_money(candles)
    price   = tech.get("price", 0)
    liq     = fetch_liquidation_data(symbol, current_price=float(price) if price else 0)

    # Son 5 mum özeti
    last5 = []
    for c in candles[-5:]:
        ts = datetime.fromtimestamp(c["ts"]/1000, tz=timezone.utc).strftime("%H:%M")
        direction = "yukari" if c["c"] >= c["o"] else "asagi"
        body_pct = abs(c["c"] - c["o"]) / (c["h"] - c["l"] + 1e-10) * 100
        last5.append(f"{ts} {direction} body%{body_pct:.0f} vol:{c['v']:.1f}")

    return {
        "timestamp": now,
        "symbol": symbol,
        "interval": interval,
        "technicals": tech,
        "orderbook": ob,
        "funding": funding,
        "smart_money": smc,
        "liquidations": liq,
        "last_5_candles": last5,
    }


if __name__ == "__main__":
    ctx = get_full_context()
    print(json.dumps(ctx, indent=2, ensure_ascii=False))
