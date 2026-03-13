"""
trade_logger.py — Trade loglarını JSONL dosyasına yaz.
Performance istatistiklerini hesapla ve JSON olarak kaydet.
"""

from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

from config import CFG
from models import TradeRecord
from logger import log

if TYPE_CHECKING:
    from models import BotState


os.makedirs(CFG.log_dir, exist_ok=True)


def _record_to_dict(record: TradeRecord) -> dict:
    """TradeRecord → JSON-serializable dict."""
    d = {
        "ts":          time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.close_ts)),
        "symbol":      record.symbol,
        "side":        record.side,
        "entry":       round(record.entry_price, 4),
        "exit":        round(record.exit_price, 4),
        "size":        record.size,
        "leverage":    record.leverage,
        "pnl_usdt":    round(record.pnl_usdt, 4),
        "pnl_pct":     round(record.pnl_pct, 4),
        "duration_sec":round(record.duration_sec, 1),
        "exit_reason": record.exit_reason,
        "is_win":      record.is_win,
        "rsi":         round(record.rsi_at_entry, 2),
        "ema_cross":   record.ema_cross_at_entry,
        "ob_imbalance":round(record.ob_imbalance_at_entry, 4),
        "signal_score":round(record.signal_score, 4),
    }
    return d


async def trade_logger_loop(
    state: "BotState",
    trade_log_queue: asyncio.Queue,
):
    """
    Trade log queue'dan kayıtları al, JSONL'e yaz.
    Her 10 trade'de performance.json güncelle.
    """
    write_count = 0

    while state.is_running or not trade_log_queue.empty():
        try:
            record: TradeRecord = await asyncio.wait_for(
                trade_log_queue.get(), timeout=2.0
            )
        except asyncio.TimeoutError:
            continue

        # ─── JSONL append ─────────────────────────────────────────
        try:
            d = _record_to_dict(record)
            with open(CFG.trade_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(d) + "\n")
            write_count += 1
        except Exception as e:
            log.error(f"Trade log yazma hatası: {e}")

        # ─── Periyodik performance raporu ─────────────────────────
        if write_count % 10 == 0:
            _write_performance_report(state)

    _write_performance_report(state)  # kapanışta son kez yaz
    log.info("Trade logger loop sonlandı.")


def _write_performance_report(state: "BotState"):
    """Performance istatistiklerini JSON'a yaz."""
    try:
        history = state.trade_history
        if not history:
            return

        pnls = [r.pnl_usdt for r in history]
        wins = [r for r in history if r.is_win]
        losses = [r for r in history if not r.is_win]

        # Sharpe yaklaşımı (basit)
        pnl_arr = __import__("numpy").array(pnls)
        sharpe = (pnl_arr.mean() / (pnl_arr.std() + 1e-9)) * (252 ** 0.5) if len(pnls) > 1 else 0.0

        # Max drawdown
        cumsum = __import__("numpy").cumsum(pnls)
        peak = __import__("numpy").maximum.accumulate(cumsum)
        drawdown = (cumsum - peak)
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Ortalama ödül/risk
        avg_win_pnl  = sum(r.pnl_usdt for r in wins)  / max(len(wins), 1)
        avg_loss_pnl = sum(r.pnl_usdt for r in losses) / max(len(losses), 1)
        rr_ratio = abs(avg_win_pnl / avg_loss_pnl) if avg_loss_pnl != 0 else 0.0

        report = {
            "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_trades":    state.total_trades,
            "win_rate_pct":    round(state.win_rate, 2),
            "total_pnl_usdt":  round(sum(pnls), 4),
            "capital_usdt":    round(state.capital_usdt, 4),
            "daily_pnl_usdt":  round(state.daily_pnl_usdt, 4),
            "daily_pnl_pct":   round(state.daily_pnl_pct, 2),
            "drawdown_pct":    round(state.total_drawdown_pct, 2),
            "max_dd_usdt":     round(max_dd, 4),
            "sharpe_approx":   round(float(sharpe), 4),
            "avg_win_usdt":    round(avg_win_pnl, 4),
            "avg_loss_usdt":   round(avg_loss_pnl, 4),
            "rr_ratio":        round(rr_ratio, 2),
            "avg_duration_sec":round(sum(r.duration_sec for r in history) / len(history), 1),
            "by_symbol": _per_symbol_stats(history),
        }

        with open(CFG.performance_log_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    except Exception as e:
        log.error(f"Performance raporu yazılamadı: {e}")


def _per_symbol_stats(history: list) -> dict:
    """Her sembol için ayrı istatistik."""
    symbols = set(r.symbol for r in history)
    result = {}
    for sym in symbols:
        sym_records = [r for r in history if r.symbol == sym]
        wins = sum(1 for r in sym_records if r.is_win)
        result[sym] = {
            "trades":   len(sym_records),
            "wins":     wins,
            "win_rate": round(wins / len(sym_records) * 100, 1),
            "total_pnl": round(sum(r.pnl_usdt for r in sym_records), 4),
        }
    return result


def print_dashboard(state: "BotState"):
    """Terminale canlı dashboard çıktısı ver."""
    pos_str = ", ".join(
        f"{sym}:{p.side.value}@{p.entry_price:.2f}"
        for sym, p in state.positions.items()
    ) or "Yok"

    print(
        f"\n{'='*60}\n"
        f"  💹 BYBIT SCALPER DASHBOARD — {time.strftime('%H:%M:%S')}\n"
        f"{'='*60}\n"
        f"  Sermaye      : {state.capital_usdt:.2f} USDT\n"
        f"  Günlük PNL   : {state.daily_pnl_usdt:+.2f} USDT (%{state.daily_pnl_pct:+.1f})\n"
        f"  Toplam Trade : {state.total_trades}\n"
        f"  Win Rate     : %{state.win_rate:.1f}\n"
        f"  Drawdown     : %{state.total_drawdown_pct:.2f}\n"
        f"  Açık Pozisyon: {pos_str}\n"
        f"{'='*60}"
    )
