"""
main.py — Bot giriş noktası. Tüm async loop'ları başlatır.

Kullanım:
  # Testnet (varsayılan)
  export BYBIT_API_KEY="..."
  export BYBIT_API_SECRET="..."
  export COINGLASS_API_KEY="..."   # opsiyonel
  python main.py

  # Mainnet
  USE_TESTNET=false python main.py
"""

from __future__ import annotations
import asyncio
import os
import signal
import sys
import time
from typing import Optional

# ─── Paket kontrolü ───────────────────────────────────────────────
MISSING_PACKAGES = []
try: import websockets
except ImportError: MISSING_PACKAGES.append("websockets")
try: import aiohttp
except ImportError: MISSING_PACKAGES.append("aiohttp")
try: import numpy
except ImportError: MISSING_PACKAGES.append("numpy")
try: import pandas
except ImportError: MISSING_PACKAGES.append("pandas")

if MISSING_PACKAGES:
    print(f"❌ Eksik paketler: {', '.join(MISSING_PACKAGES)}")
    print("   pip install " + " ".join(MISSING_PACKAGES))
    sys.exit(1)

# ─── Kendi modüller ───────────────────────────────────────────────
from config import CFG, TESTNET, BYBIT_API_KEY, BYBIT_API_SECRET
from models import BotState
from logger import log
from websocket_handler import orderbook_ws_loop
from coinglass_fetcher import liquidation_fetch_loop
from signal_generator import signal_loop
from trade_executor import trade_executor_loop
from ml_optimizer import MLOptimizer, ml_optimizer_loop
from trade_logger import trade_logger_loop, print_dashboard

# Global state (tüm modüller paylaşır)
STATE: Optional[BotState] = None
ML_OPT: Optional[MLOptimizer] = None


def _banner():
    net_str = "🔴 TESTNET" if TESTNET else "🟢 MAINNET ⚠️"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         BYBIT USDT PERPETUAL SCALPING BOT v1.0              ║
║  Çiftler  : {', '.join(CFG.symbols):<45}║
║  Sermaye  : {CFG.initial_capital_usdt:.0f} USDT{'':<40}║
║  Kaldıraç : {CFG.leverage}x{'':<48}║
║  Ağ       : {net_str:<48}║
╚══════════════════════════════════════════════════════════════╝
    """)


def _check_env():
    """Zorunlu ortam değişkenlerini kontrol et."""
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        log.error(
            "❌ BYBIT_API_KEY ve BYBIT_API_SECRET env değişkenleri gerekli!\n"
            "   export BYBIT_API_KEY='your_key'\n"
            "   export BYBIT_API_SECRET='your_secret'"
        )
        sys.exit(1)

    if TESTNET:
        log.info("✅ Testnet modu — gerçek para kullanılmıyor")
    else:
        log.warning(
            "⚠️  MAINNET MODU AKTİF — GERÇEK PARA KULLANILACAK!\n"
            "   5 saniye içinde Ctrl+C ile iptal edebilirsiniz..."
        )
        time.sleep(5)


async def _dashboard_loop(state: BotState):
    """Her 30 saniyede canlı dashboard göster."""
    while state.is_running:
        await asyncio.sleep(30)
        print_dashboard(state)


async def _grace_shutdown(state: BotState):
    """Graceful shutdown: açık pozisyonları kapat."""
    log.info("🛑 Graceful shutdown başlatıldı...")
    state.is_running = False

    if state.positions:
        log.warning(f"⚠️  {len(state.positions)} açık pozisyon var. Manuel kapatma gerekebilir.")
        for sym, pos in state.positions.items():
            log.warning(f"   → {sym} {pos.side.value} @ {pos.entry_price:.2f}")

    log.info("Bot durduruldu. Logs: ./logs/")


async def main():
    global STATE, ML_OPT

    _banner()
    _check_env()

    # ─── State başlat ──────────────────────────────────────────────
    STATE = BotState(
        capital_usdt=CFG.initial_capital_usdt,
        initial_capital=CFG.initial_capital_usdt,
        daily_start_capital=CFG.initial_capital_usdt,
    )
    ML_OPT = MLOptimizer()

    # ─── Queue'lar ────────────────────────────────────────────────
    signal_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    trade_log_queue: asyncio.Queue = asyncio.Queue()

    # ─── Ctrl+C handler ───────────────────────────────────────────
    loop = asyncio.get_event_loop()

    def _handle_sigint():
        log.info("SIGINT alındı...")
        asyncio.create_task(_grace_shutdown(STATE))

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    loop.add_signal_handler(signal.SIGTERM, _handle_sigint)

    log.info("🚀 Bot başlatılıyor...")

    # ─── Tüm async görevleri başlat ───────────────────────────────
    tasks = [
        asyncio.create_task(orderbook_ws_loop(STATE),         name="ws_orderbook"),
        asyncio.create_task(liquidation_fetch_loop(STATE),    name="liq_fetcher"),
        asyncio.create_task(signal_loop(STATE, signal_queue, ML_OPT), name="signal_gen"),
        asyncio.create_task(trade_executor_loop(STATE, signal_queue, trade_log_queue), name="trade_exec"),
        asyncio.create_task(ml_optimizer_loop(STATE, ML_OPT), name="ml_opt"),
        asyncio.create_task(trade_logger_loop(STATE, trade_log_queue), name="trade_logger"),
        asyncio.create_task(_dashboard_loop(STATE),            name="dashboard"),
    ]

    log.info(f"✅ {len(tasks)} görev başlatıldı. Sinyal bekleniyor...")

    # ─── Görevleri çalıştır ───────────────────────────────────────
    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in done:
            if task.exception():
                log.error(f"❌ Görev hata verdi [{task.get_name()}]: {task.exception()}")

    except asyncio.CancelledError:
        log.info("Görevler iptal edildi.")
    finally:
        # Tüm görevleri iptal et
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("🔚 Bot tamamen durduruldu.")


# ─── Backtest yardımcısı (testnet'ten önce çalıştır) ──────────────

def run_quick_backtest():
    """
    JSONL trade loglarından basit backtest raporu üret.
    python main.py --backtest
    """
    import json

    log_file = CFG.trade_log_file
    if not os.path.exists(log_file):
        print(f"Trade logu bulunamadı: {log_file}")
        return

    trades = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                trades.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    if not trades:
        print("Kayıtlı trade yok.")
        return

    wins   = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if not t["is_win"]]
    pnls   = [t["pnl_usdt"] for t in trades]
    total  = sum(pnls)

    print(f"""
╔══════════════════════════════════════╗
║       BACKTEST / TRADE RAPORU       ║
╠══════════════════════════════════════╣
  Toplam Trade  : {len(trades)}
  Win Rate      : %{len(wins)/len(trades)*100:.1f}
  Toplam PNL    : {total:+.2f} USDT
  Ortalama Win  : {sum(t['pnl_usdt'] for t in wins)/max(len(wins),1):+.2f} USDT
  Ortalama Loss : {sum(t['pnl_usdt'] for t in losses)/max(len(losses),1):+.2f} USDT
  En İyi Trade  : {max(pnls):+.2f} USDT
  En Kötü Trade : {min(pnls):+.2f} USDT
╚══════════════════════════════════════╝
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--backtest":
        run_quick_backtest()
    else:
        asyncio.run(main())
