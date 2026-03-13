"""
models.py — Veri yapıları: OrderBook, Trade, Signal, Position, BotState
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Side(str, Enum):
    LONG  = "Buy"
    SHORT = "Sell"
    NONE  = "None"


class SignalStrength(str, Enum):
    STRONG = "STRONG"   # tüm koşullar sağlandı
    MEDIUM = "MEDIUM"   # 3/4 koşul sağlandı
    WEAK   = "WEAK"     # 2/4 koşul sağlandı
    NONE   = "NONE"


@dataclass
class OrderBookLevel:
    price: float
    size: float

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)   # büyükten küçüğe
    asks: List[OrderBookLevel] = field(default_factory=list)   # küçükten büyüğe
    ts_ms: int = 0                                              # son güncelleme (ms)
    seq: int = 0                                                # delta sequence

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_pct(self) -> float:
        if self.best_bid == 0:
            return 0.0
        return (self.best_ask - self.best_bid) / self.best_bid * 100.0

    def top_bid_wall(self, threshold_usdt: float) -> Optional[float]:
        """En büyük bid wall fiyatını döndür (threshold üstünde)."""
        for lvl in self.bids:
            if lvl.notional >= threshold_usdt:
                return lvl.price
        return None

    def top_ask_wall(self, threshold_usdt: float) -> Optional[float]:
        """En büyük ask wall fiyatını döndür."""
        for lvl in self.asks:
            if lvl.notional >= threshold_usdt:
                return lvl.price
        return None

    def imbalance_ratio(self, depth: int = 10) -> float:
        """
        Order flow imbalance: bid_volume / ask_volume (ilk N seviye).
        > 1.5 → alıcı baskısı, < 0.67 → satıcı baskısı
        """
        bid_vol = sum(lvl.size for lvl in self.bids[:depth])
        ask_vol = sum(lvl.size for lvl in self.asks[:depth])
        if ask_vol == 0:
            return 1.0
        return bid_vol / ask_vol


@dataclass
class LiquidationCluster:
    price: float
    total_usd: float
    side: str           # "long" veya "short" (hangi pozisyonlar likide olur)
    symbol: str
    ts: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.ts


@dataclass
class Signal:
    symbol: str
    side: Side
    strength: SignalStrength
    price: float
    reason: str
    score: float = 0.0      # ML skoru (0-1)
    ts: float = field(default_factory=time.time)

    # Alt sinyal bileşenleri (debugging için)
    rsi: float = 0.0
    ema_cross: bool = False
    orderbook_ok: bool = False
    liquidation_ok: bool = False
    atr_ok: bool = True


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    size: float             # coin miktarı
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    open_ts: float = field(default_factory=time.time)
    order_id: str = ""
    highest_price: float = 0.0  # trailing stop için
    lowest_price: float = float("inf")

    @property
    def unrealized_pnl(self, current_price: float = 0) -> float:
        if current_price == 0:
            return 0.0
        if self.side == Side.LONG:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    @property
    def notional_usdt(self) -> float:
        return self.entry_price * self.size


@dataclass
class TradeRecord:
    """Her tamamlanan trade'in logu."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    pnl_usdt: float
    pnl_pct: float
    duration_sec: float
    exit_reason: str        # "take_profit" | "stop_loss" | "trailing" | "manual"
    open_ts: float
    close_ts: float
    signal_score: float
    # Sinyal bileşenleri
    rsi_at_entry: float = 0.0
    ema_cross_at_entry: bool = False
    ob_imbalance_at_entry: float = 0.0
    liq_proximity_at_entry: float = 0.0
    atr_at_entry: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl_usdt > 0


@dataclass
class BotState:
    """Bot'un anlık durumu — tüm modüller bu nesneyi paylaşır."""
    capital_usdt: float
    initial_capital: float
    daily_pnl_usdt: float = 0.0
    daily_start_capital: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    positions: Dict[str, Position] = field(default_factory=dict)
    orderbooks: Dict[str, OrderBook] = field(default_factory=dict)
    liquidation_clusters: Dict[str, List[LiquidationCluster]] = field(default_factory=dict)
    trade_history: List[TradeRecord] = field(default_factory=list)
    is_running: bool = True
    halt_reason: str = ""
    last_ml_retrain_at: int = 0  # kaçıncı trade'de son kez retrain yapıldı

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100.0

    @property
    def total_drawdown_pct(self) -> float:
        return (self.initial_capital - self.capital_usdt) / self.initial_capital * 100.0

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_capital == 0:
            return 0.0
        return self.daily_pnl_usdt / self.daily_start_capital * 100.0

    def can_trade(self) -> Tuple[bool, str]:
        """Risk limitlerini kontrol et."""
        if not self.is_running:
            return False, self.halt_reason
        if self.total_drawdown_pct >= 10.0:
            return False, f"Max drawdown aşıldı: %{self.total_drawdown_pct:.1f}"
        if self.daily_pnl_pct <= -10.0:
            return False, f"Günlük loss limiti: %{self.daily_pnl_pct:.1f}"
        if self.daily_pnl_pct >= 5.0:
            return False, f"Günlük kâr hedefi doldu: %{self.daily_pnl_pct:.1f}"
        if len(self.positions) >= 2:
            return False, "Max açık pozisyon sayısına ulaşıldı"
        return True, ""
