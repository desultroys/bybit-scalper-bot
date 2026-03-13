"""
config.py - BingX Perpetual Futures Scalping Bot
"""
import os
from dataclasses import dataclass, field
from typing import List

BINGX_API_KEY    = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "")
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY", "")

USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

# BingX endpoints
BINGX_REST    = "https://open-api.bingx.com"
BINGX_WS      = "wss://open-api-ws.bingx.com/market"

# BingX testnet (demo hesap - gerçek API key ile çalışır)
BINGX_REST_DEMO = "https://open-api.bingx.com"  # BingX demo/live aynı endpoint
BINGX_WS_DEMO   = "wss://open-api-ws.bingx.com/market"

COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"


@dataclass
class BotConfig:
    symbols: List[str]   = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    primary_symbol: str  = "BTC-USDT"

    initial_capital_usdt: float = 1000.0
    max_risk_per_trade_pct: float = 2.0
    leverage: int  = 15
    max_leverage: int = 40
    min_leverage: int = 10

    stop_loss_pct:    float = 0.4
    take_profit_pct:  float = 1.0
    trailing_stop_pct: float = 0.3

    max_daily_drawdown_pct: float = 10.0
    max_daily_pnl_cap_pct:  float = 5.0
    max_open_positions: int = 2

    rsi_period: int  = 14
    ema_fast:   int  = 9
    ema_slow:   int  = 21
    atr_period: int  = 14
    atr_skip_threshold_pct: float = 1.0
    vwap_period: int = 20

    orderbook_depth: int = 20
    bid_wall_threshold_usdt: float = 500_000
    ask_wall_threshold_usdt: float = 500_000
    orderbook_imbalance_ratio: float = 1.5

    liquidation_proximity_pct: float = 0.5
    liquidation_update_interval: int = 8
    liquidation_cluster_min_usd: float = 2_000_000

    ml_retrain_every_n_trades: int = 50
    ml_lookback_trades: int = 200
    ml_min_samples: int = 30

    ws_reconnect_delay: int = 5
    log_dir: str = "logs"
    trade_log_file: str = "logs/trades.jsonl"
    performance_log_file: str = "logs/performance.json"


CFG = BotConfig()
