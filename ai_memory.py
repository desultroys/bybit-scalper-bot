"""
ai_memory.py — Claude'un ticaret hafızası ve öğrenme sistemi.

Her karar kaydedilir: bağlam, gerekçe, sonuç.
Claude bir sonraki analizde bu hafızayı okur ve öğrenir.
"""
import json, os
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path("logs/ai_memory.json")
MAX_TRADES_IN_MEMORY = 200  # Hafızada tutulacak max işlem sayısı


def load_memory() -> dict:
    """Hafızayı yükle. Yoksa boş yapı döndür."""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "learned_patterns": [],  # Claude'un keşfettiği pattern'ler
        "market_notes": [],      # Genel piyasa gözlemleri
        "trades": [],            # Trade geçmişi
    }


def save_memory(memory: dict):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    # Çok eski trade'leri temizle
    if len(memory["trades"]) > MAX_TRADES_IN_MEMORY:
        memory["trades"] = memory["trades"][-MAX_TRADES_IN_MEMORY:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def log_decision(memory: dict, decision: dict, context: dict) -> str:
    """
    Yeni bir karar kaydeder.
    decision: {"action": "long"|"short"|"wait", "confidence": 0-100,
                "sl_pct": float, "tp_pct": float, "reasoning": str,
                "key_factors": [...], "risks": [...]}
    Döndürür: trade_id
    """
    trade_id = f"T{len(memory['trades'])+1:04d}_{datetime.now(timezone.utc).strftime('%m%d%H%M')}"
    entry = {
        "id": trade_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "symbol": context.get("symbol", "BTC-USDT"),
        "price_at_decision": context.get("technicals", {}).get("price", 0),
        "action": decision.get("action", "wait"),
        "confidence": decision.get("confidence", 0),
        "sl_pct": decision.get("sl_pct", 0),
        "tp_pct": decision.get("tp_pct", 0),
        "reasoning": decision.get("reasoning", ""),
        "key_factors": decision.get("key_factors", []),
        "risks": decision.get("risks", []),
        "outcome": None,  # Sonra doldurulacak
        "pnl_pct": None,
        "exit_price": None,
        "exit_reason": None,
    }
    memory["trades"].append(entry)
    memory["total_trades"] += 1
    save_memory(memory)
    return trade_id


def log_outcome(memory: dict, trade_id: str, outcome: dict):
    """
    Bir trade'in sonucunu kaydeder ve istatistikleri günceller.
    outcome: {"result": "win"|"loss"|"be", "pnl_pct": float,
              "exit_price": float, "exit_reason": str}
    """
    for trade in memory["trades"]:
        if trade["id"] == trade_id:
            trade["outcome"] = outcome.get("result")
            trade["pnl_pct"] = outcome.get("pnl_pct", 0)
            trade["exit_price"] = outcome.get("exit_price", 0)
            trade["exit_reason"] = outcome.get("exit_reason", "")

            pnl = outcome.get("pnl_pct", 0)
            memory["total_pnl"] += pnl
            if pnl > 0:
                memory["wins"] += 1
            elif pnl < 0:
                memory["losses"] += 1
            break
    save_memory(memory)


def add_learned_pattern(memory: dict, pattern: str):
    """Claude'un keşfettiği bir pattern'i kaydet."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory["learned_patterns"].append({"date": timestamp, "pattern": pattern})
    # En fazla 50 pattern tut
    if len(memory["learned_patterns"]) > 50:
        memory["learned_patterns"] = memory["learned_patterns"][-50:]
    save_memory(memory)


def format_memory_for_claude(memory: dict) -> str:
    """
    Claude'a gönderilecek hafıza özetini hazırlar.
    Claude bu özeti okuyarak önceki kararlarından öğrenir.
    """
    total = memory["total_trades"]
    wins = memory["wins"]
    losses = memory["losses"]
    wr = wins / max(total, 1) * 100

    lines = [
        f"=== GECMiS PERFORMANS ({total} islem) ===",
        f"Win Rate: %{wr:.1f} | W:{wins} L:{losses} | Toplam PnL: %{memory['total_pnl']:.1f}",
        "",
    ]

    # Son 10 kapalı trade
    closed = [t for t in memory["trades"] if t["outcome"] is not None][-10:]
    if closed:
        lines.append("--- Son 10 islem ---")
        for t in closed:
            outcome_str = "✓ WIN" if t["outcome"] == "win" else ("✗ LOSS" if t["outcome"] == "loss" else "= BE")
            px  = t['price_at_decision'] or 0
            pnl = t['pnl_pct'] or 0
            lines.append(
                f"{t['timestamp'][:10]} {t['action']:5s} @{px:.0f} "
                f"conf:{t['confidence']}% -> {outcome_str} {pnl:+.2f}% "
                f"[{t['exit_reason']}]"
            )
        lines.append("")

    # Öğrenilen pattern'ler
    patterns = memory["learned_patterns"][-10:]
    if patterns:
        lines.append("--- Ogrenilen Patternler ---")
        for p in patterns:
            lines.append(f"• {p['date']}: {p['pattern']}")
        lines.append("")

    # Açık pozisyon varsa
    open_trades = [t for t in memory["trades"] if t["outcome"] is None and t["action"] != "wait"]
    if open_trades:
        last_open = open_trades[-1]
        lines.append(f"--- ACIK POZISYON: {last_open['action'].upper()} @ {last_open['price_at_decision']:.0f} ---")
        lines.append(f"  Giris: {last_open['timestamp']}")
        lines.append(f"  SL:%{last_open['sl_pct']} TP:%{last_open['tp_pct']}")
        lines.append("")

    return "\n".join(lines)
