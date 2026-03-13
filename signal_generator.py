"""
signal_generator.py — Teknik analiz + sinyal üretimi
RSI, EMA crossover, VWAP, ATR, Orderbook imbalance + Likidasyon proximity
"""

from __future__ import annotations
import asyncio
from collections import deque
from typing import Dict, Deque, Optional, TYPE_CHECKING

import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # TA-Lib yüklü değilse numpy ile hesapla

from config import CFG
from models import Side, Signal, SignalStrength
from coinglass_fetcher import find_nearby_clusters
from logger import log

if TYPE_CHECKING:
    from models import BotState


# ─── Fiyat geçmişi deque'leri (her sembol için) ───────────────────
_price_history:  Dict[str, Deque[float]] = {}
_volume_history: Dict[str, Deque[float]] = {}
_high_history:   Dict[str, Deque[float]] = {}
_low_history:    Dict[str, Deque[float]] = {}

MAX_HIST = 200  # son 200 bar tut


def _init_history(symbol: str):
    if symbol not in _price_history:
        _price_history[symbol]  = deque(maxlen=MAX_HIST)
        _volume_history[symbol] = deque(maxlen=MAX_HIST)
        _high_history[symbol]   = deque(maxlen=MAX_HIST)
        _low_history[symbol]    = deque(maxlen=MAX_HIST)


def update_price_history(symbol: str, close: float, volume: float = 0.0,
                          high: float = 0.0, low: float = 0.0):
    """Her yeni fiyat tick'inde çağrıl."""
    _init_history(symbol)
    _price_history[symbol].append(close)
    _volume_history[symbol].append(volume)
    _high_history[symbol].append(high if high >= close else close)
    _low_history[symbol].append(low  if low  <= close else close)


# ─── Göstergeler ──────────────────────────────────────────────────

def _calc_rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI hesapla — TA-Lib varsa kullan, yoksa numpy fallback."""
    if len(prices) < period + 1:
        return 50.0  # nötr

    if TALIB_AVAILABLE:
        rsi = talib.RSI(prices, timeperiod=period)
        val = rsi[-1]
        return float(val) if not np.isnan(val) else 50.0

    # Numpy fallback
    deltas = np.diff(prices[-(period * 2):])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """EMA hesapla."""
    if TALIB_AVAILABLE:
        return talib.EMA(prices, timeperiod=period)

    alpha = 2.0 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _calc_atr(highs: np.ndarray, lows: np.ndarray,
              closes: np.ndarray, period: int = 14) -> float:
    """ATR hesapla, son bar için yüzde olarak döndür."""
    if len(closes) < period + 1:
        return 0.0

    if TALIB_AVAILABLE:
        atr = talib.ATR(highs, lows, closes, timeperiod=period)
        val = atr[-1]
    else:
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
            tr_list.append(tr)
        val = float(np.mean(tr_list[-period:]))

    if np.isnan(val) or closes[-1] == 0:
        return 0.0
    return float(val) / closes[-1] * 100.0  # yüzde olarak


def _calc_vwap(prices: np.ndarray, volumes: np.ndarray,
               period: int = 20) -> float:
    """Son N barda VWAP."""
    n = min(period, len(prices), len(volumes))
    if n == 0:
        return prices[-1] if len(prices) > 0 else 0.0
    p = prices[-n:]
    v = volumes[-n:]
    if v.sum() == 0:
        return float(p.mean())
    return float(np.sum(p * v) / np.sum(v))


# ─── Ana sinyal üretici ───────────────────────────────────────────

def generate_signal(
    symbol: str,
    state: "BotState",
    ml_score_override: Optional[float] = None,
) -> Optional[Signal]:
    """
    Tüm koşulları değerlendir ve sinyal üret.

    Long koşulları:
      1. Bid wall tespit edildi (≥ 500K USDT)
      2. Fiyat yakınında LONG likidasyon cluster'ı var (alımı destekler)
      3. RSI > 50
      4. EMA(9) > EMA(21) — bullish crossover

    Short koşulları:
      1. Ask wall tespit edildi
      2. SHORT likidasyon cluster'ı yakında
      3. RSI < 50
      4. EMA(9) < EMA(21)

    Ek filtreler:
      - ATR > %1 → volatilite çok yüksek, trade atla
      - Spread > %0.05 → spread çok geniş, atla
    """
    _init_history(symbol)

    ob = state.orderbooks.get(symbol)
    if ob is None or len(ob.bids) == 0:
        return None

    current_price = ob.mid_price
    if current_price == 0:
        return None

    prices  = np.array(list(_price_history[symbol]),  dtype=float)
    volumes = np.array(list(_volume_history[symbol]), dtype=float)
    highs   = np.array(list(_high_history[symbol]),   dtype=float)
    lows    = np.array(list(_low_history[symbol]),    dtype=float)

    if len(prices) < 30:
        log.debug(f"{symbol}: Yeterli fiyat geçmişi yok ({len(prices)}/30)")
        return None

    # ─── 1. ATR Filtresi ──────────────────────────────────────────
    atr_pct = _calc_atr(highs, lows, prices, CFG.atr_period)
    if atr_pct > CFG.atr_skip_threshold_pct:
        log.debug(f"{symbol}: ATR=%{atr_pct:.2f} > %{CFG.atr_skip_threshold_pct} — trade atlanıyor")
        return None

    # ─── 2. Spread Filtresi ───────────────────────────────────────
    if ob.spread_pct > 0.05:
        log.debug(f"{symbol}: Spread=%{ob.spread_pct:.4f} çok geniş")
        return None

    # ─── 3. RSI ───────────────────────────────────────────────────
    rsi = _calc_rsi(prices, CFG.rsi_period)

    # ─── 4. EMA Crossover ─────────────────────────────────────────
    ema_fast_arr = _calc_ema(prices, CFG.ema_fast)
    ema_slow_arr = _calc_ema(prices, CFG.ema_slow)
    ema_bullish  = ema_fast_arr[-1] > ema_slow_arr[-1]  # EMA(9) > EMA(21)
    ema_cross    = ema_fast_arr[-1]  # debug için

    # ─── 5. VWAP ──────────────────────────────────────────────────
    vwap = _calc_vwap(prices, volumes, CFG.vwap_period)
    above_vwap = current_price > vwap

    # ─── 6. Orderbook ─────────────────────────────────────────────
    bid_wall = ob.top_bid_wall(CFG.bid_wall_threshold_usdt)
    ask_wall = ob.top_ask_wall(CFG.ask_wall_threshold_usdt)
    imbalance = ob.imbalance_ratio(depth=10)

    # ─── 7. Likidasyon Cluster'ları ───────────────────────────────
    liq = find_nearby_clusters(state, symbol, current_price)
    long_liq_nearby  = liq["nearest_long"]  is not None
    short_liq_nearby = liq["nearest_short"] is not None
    liq_proximity = min(
        abs(liq["nearest_long"]  - current_price) / current_price * 100 if long_liq_nearby  else 999,
        abs(liq["nearest_short"] - current_price) / current_price * 100 if short_liq_nearby else 999,
    )

    # ─── LONG SİNYALİ ─────────────────────────────────────────────
    long_conditions = {
        "bid_wall":    bid_wall is not None,
        "liq_cluster": long_liq_nearby,
        "rsi_ok":      rsi > 50,
        "ema_bullish": ema_bullish,
    }
    long_score = sum(long_conditions.values())

    # ─── SHORT SİNYALİ ────────────────────────────────────────────
    short_conditions = {
        "ask_wall":     ask_wall is not None,
        "liq_cluster":  short_liq_nearby,
        "rsi_ok":       rsi < 50,
        "ema_bearish":  not ema_bullish,
    }
    short_score = sum(short_conditions.values())

    log.debug(
        f"{symbol} | RSI={rsi:.1f} | EMA={'↑' if ema_bullish else '↓'} | "
        f"Imb={imbalance:.2f} | ATR%={atr_pct:.2f} | "
        f"Long={long_score}/4 | Short={short_score}/4"
    )

    # ML override (varsa) — base skoru ağırlıklandır
    ml_multiplier = ml_score_override if ml_score_override else 1.0

    # En az 3/4 koşul gerekli
    def _score_to_strength(score: int) -> SignalStrength:
        if score == 4: return SignalStrength.STRONG
        if score == 3: return SignalStrength.MEDIUM
        if score == 2: return SignalStrength.WEAK
        return SignalStrength.NONE

    if long_score >= 3 and long_score > short_score:
        strength = _score_to_strength(long_score)
        if strength == SignalStrength.WEAK:
            return None  # Zayıf sinyal → işlem yapma

        reason = (
            f"LONG | RSI={rsi:.1f} | EMA={'bullish'} | "
            f"BidWall={bid_wall:.1f if bid_wall else 'none'} | "
            f"LiqCluster={'yakın' if long_liq_nearby else 'uzak'} | "
            f"Imb={imbalance:.2f}"
        )
        return Signal(
            symbol=symbol, side=Side.LONG, strength=strength,
            price=current_price, reason=reason,
            score=long_score / 4.0 * ml_multiplier,
            rsi=rsi, ema_cross=ema_bullish,
            orderbook_ok=imbalance > CFG.orderbook_imbalance_ratio,
            liquidation_ok=long_liq_nearby,
            atr_ok=True,
        )

    elif short_score >= 3 and short_score > long_score:
        strength = _score_to_strength(short_score)
        if strength == SignalStrength.WEAK:
            return None

        reason = (
            f"SHORT | RSI={rsi:.1f} | EMA={'bearish'} | "
            f"AskWall={ask_wall:.1f if ask_wall else 'none'} | "
            f"LiqCluster={'yakın' if short_liq_nearby else 'uzak'}"
        )
        return Signal(
            symbol=symbol, side=Side.SHORT, strength=strength,
            price=current_price, reason=reason,
            score=short_score / 4.0 * ml_multiplier,
            rsi=rsi, ema_cross=not ema_bullish,
            orderbook_ok=imbalance < (1 / CFG.orderbook_imbalance_ratio),
            liquidation_ok=short_liq_nearby,
            atr_ok=True,
        )

    return None  # Sinyal yok


async def signal_loop(
    state: "BotState",
    signal_queue: asyncio.Queue,
    ml_model=None,
):
    """
    Her sembol için periyodik sinyal üret ve queue'ya koy.
    ml_model: MLOptimizer nesnesi (None ise ML filtresi atlanır)
    """
    while state.is_running:
        for sym in CFG.symbols:
            ob = state.orderbooks.get(sym)
            if ob and ob.mid_price > 0:
                # Fiyat geçmişini güncelle (mid price kullan)
                update_price_history(
                    sym,
                    close=ob.mid_price,
                    volume=sum(lvl.size for lvl in ob.bids[:5] + ob.asks[:5]),
                    high=ob.best_ask,
                    low=ob.best_bid,
                )

            # ML skorunu al (model hazırsa)
            ml_score = None
            if ml_model is not None and ml_model.is_ready:
                ml_score = ml_model.get_signal_multiplier(sym, state)

            sig = generate_signal(sym, state, ml_score_override=ml_score)
            if sig is not None and sig.side != Side.NONE:
                can_trade, reason = state.can_trade()
                if can_trade:
                    await signal_queue.put(sig)
                    log.info(
                        f"📊 SİNYAL: {sig.symbol} {sig.side.value} "
                        f"[{sig.strength.value}] score={sig.score:.2f} | {sig.reason}"
                    )
                else:
                    log.debug(f"Trade engellendi: {reason}")

        await asyncio.sleep(1.0)  # 1 saniyede bir sinyal üret
