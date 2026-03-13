"""
coinglass_fetcher.py — Coinglass API'den likidasyon heatmap verileri çek.
Her CFG.liquidation_update_interval saniyede bir güncelle.
"""

from __future__ import annotations
import asyncio
import time
from typing import TYPE_CHECKING, List

import aiohttp

from config import CFG, COINGLASS_API_KEY, COINGLASS_BASE
from models import LiquidationCluster
from logger import log

if TYPE_CHECKING:
    from models import BotState


# Coinglass sembol eşleştirme (BTCUSDT → BTC)
_SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
}

# ─── Ücretsiz tier rate limit: ~10 req/dk ─────────────────────────
_RATE_LIMIT_DELAY = 7.0  # saniye (güvenli margin)


async def _fetch_liquidation_levels(
    session: aiohttp.ClientSession,
    symbol: str,
) -> List[LiquidationCluster]:
    """
    Coinglass liquidation heatmap endpoint'ini çağır.
    Büyük likidasyon clusterlarını döndür.

    Endpoint: GET /public/v2/futures/liquidation-order/chart
    Parametreler: symbol, time_type (0=1h), ex (Bybit)
    """
    coin = _SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
    url = f"{COINGLASS_BASE}/futures/liquidation-order/chart"

    headers = {
        "coinglassSecret": COINGLASS_API_KEY,
        "User-Agent": "ScalpBot/1.0",
    }
    params = {
        "symbol": coin,
        "ex": "Bybit",
        "time_type": "0",   # 1 saatlik dilim
    }

    clusters = []
    try:
        async with session.get(
            url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 429:
                log.warning("⚠️ Coinglass rate limit — bekleniyor...")
                await asyncio.sleep(30)
                return []
            if resp.status != 200:
                log.warning(f"Coinglass HTTP {resp.status} for {symbol}")
                return []

            data = await resp.json()

            # API cevap formatı:
            # data.data → { priceList: [...], longLiqList: [...], shortLiqList: [...] }
            body = data.get("data", {})
            prices      = body.get("priceList", [])
            long_liqs   = body.get("longLiqList", [])   # bu fiyattan LONG'lar likide olur
            short_liqs  = body.get("shortLiqList", [])

            threshold = CFG.liquidation_cluster_min_usd

            for i, price in enumerate(prices):
                try:
                    p = float(price)
                    ll = float(long_liqs[i])  if i < len(long_liqs)  else 0.0
                    sl = float(short_liqs[i]) if i < len(short_liqs) else 0.0

                    if ll >= threshold:
                        clusters.append(LiquidationCluster(
                            price=p, total_usd=ll, side="long", symbol=symbol
                        ))
                    if sl >= threshold:
                        clusters.append(LiquidationCluster(
                            price=p, total_usd=sl, side="short", symbol=symbol
                        ))
                except (ValueError, TypeError):
                    continue

    except aiohttp.ClientError as e:
        log.error(f"Coinglass bağlantı hatası ({symbol}): {e}")
    except Exception as e:
        log.error(f"Coinglass genel hata ({symbol}): {e}")

    log.debug(f"💧 {symbol}: {len(clusters)} likidasyon cluster'ı yüklendi")
    return clusters


async def liquidation_fetch_loop(state: "BotState"):
    """
    Her CFG.liquidation_update_interval saniyede tüm semboller için
    likidasyon verilerini güncelle.
    """
    if not COINGLASS_API_KEY:
        log.warning("⚠️ COINGLASS_API_KEY yok — likidasyon verisi simüle edilecek")
        await _simulate_liquidation_loop(state)
        return

    async with aiohttp.ClientSession() as session:
        while state.is_running:
            for sym in CFG.symbols:
                if not state.is_running:
                    break
                clusters = await _fetch_liquidation_levels(session, sym)
                state.liquidation_clusters[sym] = clusters
                await asyncio.sleep(_RATE_LIMIT_DELAY / len(CFG.symbols))

            await asyncio.sleep(
                max(1, CFG.liquidation_update_interval - _RATE_LIMIT_DELAY)
            )

    log.info("Likidasyon fetch loop sonlandı.")


async def _simulate_liquidation_loop(state: "BotState"):
    """
    Coinglass API anahtarı yoksa likidasyon verilerini
    fiyat bazlı simüle et (test amaçlı).
    """
    import random
    log.info("🎭 Likidasyon simülasyon modu aktif")

    while state.is_running:
        for sym in CFG.symbols:
            ob = state.orderbooks.get(sym)
            if ob is None or ob.mid_price == 0:
                continue

            mid = ob.mid_price
            clusters = []

            # Fiyatın üstünde ve altında rastgele cluster'lar üret
            for offset_pct in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
                price = mid * (1 + offset_pct / 100)
                usd   = random.uniform(2_000_000, 15_000_000)
                side  = "long" if offset_pct < 0 else "short"
                clusters.append(LiquidationCluster(
                    price=price, total_usd=usd, side=side, symbol=sym
                ))

            state.liquidation_clusters[sym] = clusters

        await asyncio.sleep(CFG.liquidation_update_interval)


def find_nearby_clusters(
    state: "BotState",
    symbol: str,
    current_price: float,
    proximity_pct: float | None = None,
) -> dict:
    """
    Mevcut fiyata ±proximity_pct% mesafedeki cluster'ları döndür.

    Returns:
        {
            "long_clusters":  [yakındaki long likidasyon seviyeleri],
            "short_clusters": [yakındaki short likidasyon seviyeleri],
            "nearest_long":   en yakın long cluster fiyatı veya None,
            "nearest_short":  en yakın short cluster fiyatı veya None,
        }
    """
    if proximity_pct is None:
        proximity_pct = CFG.liquidation_proximity_pct

    all_clusters = state.liquidation_clusters.get(symbol, [])
    lower = current_price * (1 - proximity_pct / 100)
    upper = current_price * (1 + proximity_pct / 100)

    long_near  = [c for c in all_clusters if c.side == "long"  and lower <= c.price <= upper]
    short_near = [c for c in all_clusters if c.side == "short" and lower <= c.price <= upper]

    nearest_long  = min(long_near,  key=lambda c: abs(c.price - current_price), default=None)
    nearest_short = min(short_near, key=lambda c: abs(c.price - current_price), default=None)

    return {
        "long_clusters":  long_near,
        "short_clusters": short_near,
        "nearest_long":   nearest_long.price if nearest_long  else None,
        "nearest_short":  nearest_short.price if nearest_short else None,
        "total_long_usd":  sum(c.total_usd for c in long_near),
        "total_short_usd": sum(c.total_usd for c in short_near),
    }
