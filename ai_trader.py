"""
ai_trader.py — Claude AI Karar Motoru
======================================
Kullanim:
  python ai_trader.py --dry-run          # simülasyon (para riski yok)
  python ai_trader.py --dry-run --once   # tek analiz
  python ai_trader.py --interval 5       # her 5dk analiz
  python ai_trader.py --min-confidence 70
  python ai_trader.py --report           # hafıza istatistikleri
  python ai_trader.py                    # CANLI mod

Gereksinimler: .env dosyasinda ANTHROPIC_API_KEY olmali
"""
import os, json, time, argparse, sys, io
from datetime import datetime, timezone
from dotenv import load_dotenv

# Windows terminal unicode fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BINGX_API_KEY     = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET  = os.getenv("BINGX_API_SECRET", "")

# ── Modüller ───────────────────────────────────────────────────────
from market_context import get_full_context
from ai_memory import (load_memory, save_memory, log_decision,
                       log_outcome, add_learned_pattern,
                       format_memory_for_claude)

# ── Sabitler ──────────────────────────────────────────────────────
MODEL           = "claude-haiku-4-5-20251001"   # Ucuz + hızlı
INTERVAL_MIN    = 15                             # Kaç dakikada bir analiz
MIN_CONFIDENCE  = 65                             # Bu altı = "bekle"
MAX_LEVERAGE    = 20                             # Güvenli kaldıraç (20x)
RISK_PCT        = 1.5                            # Trade başına risk %


# ── Claude API ────────────────────────────────────────────────────
def call_claude(prompt: str, system: str) -> str:
    """Claude API'yi çağırır, yanıt döndürür."""
    try:
        import anthropic
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
        import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


# ── Sistem Promptu ────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen BTC/USDT vadeli işlem uzmanısın. Piyasayı analiz edip net kararlar verirsin.

KARAR KURALLARI:
1. Her analiz sonunda kesin bir karar ver: LONG, SHORT veya BEKLE
2. Geçmiş performansına bakarak yanlış pattern'lerden kaçın
3. Yüksek hacimli likidasyon bölgelerinde dikkatli ol
4. Funding rate yüksekse (%0.1+) o yöne karşı pozisyon alma
5. Orderbook imbalance >0.3 ise güçlü sinyal, <0.15 ise zayıf

ÇIKTI FORMATI (tam olarak bu JSON formatında yanıt ver):
{
  "action": "long" | "short" | "wait",
  "confidence": 0-100,
  "sl_pct": 0.5-2.0,
  "tp_pct": 1.5-6.0,
  "reasoning": "Kısa Türkçe açıklama (2-3 cümle)",
  "key_factors": ["faktör1", "faktör2", "faktör3"],
  "risks": ["risk1", "risk2"],
  "learned_note": "Bu karardan öğrenilen şey (pattern veya null)"
}

Başka hiçbir şey yazma. Sadece JSON."""


# ── Karar Alma ────────────────────────────────────────────────────
def analyze_and_decide(symbol: str, memory: dict) -> dict | None:
    """Piyasayı analiz et ve karar ver."""
    print(f"\n[AI] {symbol} analiz ediliyor...")

    # 1. Piyasa verisi
    ctx = get_full_context(symbol, "15m")
    tech = ctx.get("technicals", {})
    ob   = ctx.get("orderbook", {})
    fund = ctx.get("funding", {})
    smc  = ctx.get("smart_money", {})
    liqd = ctx.get("liquidations", {})

    price = tech.get("price", 0)
    if not price:
        print("  [HATA] Fiyat verisi alınamadı")
        return None

    # 2. Hafıza özeti
    memory_summary = format_memory_for_claude(memory)

    # 3. Prompt hazırla
    def _f(v, default=0): return v if v is not None else default

    prompt = f"""
ZAMAN: {ctx['timestamp']}
SEMBOL: {symbol} | ARALIK: 15dk

=== TEKNiK GÖSTERGELER ===
Fiyat: ${price:,.2f}
EMA9/21/50/200: {_f(tech.get('ema9')):.0f} / {_f(tech.get('ema21')):.0f} / {_f(tech.get('ema50')):.0f} / {_f(tech.get('ema200')):.0f}
RSI(14): {_f(tech.get('rsi'), 50):.1f}
ATR: ${_f(tech.get('atr')):.1f} (%{_f(tech.get('atr_pct')):.3f})
Momentum 3b: %{_f(tech.get('mom3_pct')):+.3f} | 8b: %{_f(tech.get('mom8_pct')):+.3f}
Hacim oranı: {_f(tech.get('vol_ratio'), 1):.2f}x (ortalamaya göre)
Son 3 mum: {' | '.join(tech.get('last_3_candles', []))}
EMA200 üstü: {tech.get('above_ema200', False)} | EMA50 üstü: {tech.get('above_ema50', False)}
VWAP(50): {_f(tech.get('vwap')):.0f} | Fiyat VWAP farkı: %{_f(tech.get('vwap_dist_pct')):+.3f} | VWAP üstü: {tech.get('above_vwap', False)}
CVD(50): {_f(tech.get('cvd')):+.0f} | CVD trendi: {tech.get('cvd_trend', '?')} | Son 10 delta: {_f(tech.get('cvd_recent')):+.0f}

=== ORDERBOOK ===
Alıcı hacim: ${_f(ob.get('bid_vol_usd')):,.0f} | Satıcı hacim: ${_f(ob.get('ask_vol_usd')):,.0f}
Dengesizlik: {_f(ob.get('imbalance')):+.3f} (>0 alıcı baskısı, <0 satıcı)
En iyi bid: {_f(ob.get('best_bid')):.2f} | ask: {_f(ob.get('best_ask')):.2f}

=== FUNDING & BASIS ===
Funding Rate: %{_f(fund.get('funding_rate')):.4f} (8 saatlik)
Mark/Index farkı: %{_f(fund.get('basis_pct')):.4f}

=== SMART MONEY (ICT) ===
Swing High/Low: {_f(smc.get('last_swing_high')):.0f} / {_f(smc.get('last_swing_low')):.0f}
BOS Bullish: {smc.get('bos_bullish', False)} | BOS Bearish: {smc.get('bos_bearish', False)}
Liquidity Sweep: {smc.get('liq_sweep', 'yok')}
Premium/Discount: %{_f(smc.get('premium_discount_pct')):+.1f} (+ = premium/pahali, - = discount/ucuz)
Bullish OB (destek): {smc.get('bull_ob', 'yok')} | Icinde: {smc.get('in_bull_ob', False)}
Bearish OB (direnc): {smc.get('bear_ob', 'yok')} | Icinde: {smc.get('in_bear_ob', False)}
Bullish FVG (destek): {smc.get('bull_fvg', 'yok')}
Bearish FVG (direnc): {smc.get('bear_fvg', 'yok')}

=== LİKİDASYON & AÇIK POZİSYON ===
Toplam OI: ${_f(liqd.get('oi_usd')):,.0f} | Long: %{_f(liqd.get('long_pct')):.1f} Short: %{_f(liqd.get('short_pct')):.1f}
Durum: {liqd.get('dominant','?')} (long_agir=uzun kalabalık→short cascade riski, short_agir=short kalabalık→long squeeze riski)
Tahmini Likidasyon Seviyeleri (kaldıraç varsayımı):
  Long liq @10x: ${_f(liqd.get('long_liq_10x')):,.0f} | @20x: ${_f(liqd.get('long_liq_20x')):,.0f}
  Short liq @10x: ${_f(liqd.get('short_liq_10x')):,.0f} | @20x: ${_f(liqd.get('short_liq_20x')):,.0f}

=== SON 5 MUM ===
{chr(10).join(ctx.get('last_5_candles', []))}

{memory_summary}

Şimdi analiz yap ve karar ver:"""

    # 4. Claude'u çağır
    try:
        response = call_claude(prompt, SYSTEM_PROMPT)
        # JSON parse
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw.strip())
    except Exception as e:
        print(f"  [HATA] Claude yanıtı parse edilemedi: {e}")
        print(f"  Yanıt: {response[:200] if 'response' in dir() else 'yok'}")
        return None

    decision["price"] = price
    decision["timestamp"] = ctx["timestamp"]
    return decision


# ── Trade Executor (BingX) ────────────────────────────────────────
def execute_trade(decision: dict, symbol: str, capital: float, dry_run: bool = False) -> str | None:
    """Kararı BingX'te uygular. dry_run=True ise sadece simüle eder."""
    action = decision.get("action", "wait")
    if action == "wait":
        return None

    sl_pct = decision.get("sl_pct", 1.0)
    tp_pct = decision.get("tp_pct", 3.0)
    confidence = decision.get("confidence", 0)

    # Pozisyon büyüklüğü: risk bazlı
    price = decision.get("price", 0)
    sl_dist = price * sl_pct / 100
    risk_usd = capital * RISK_PCT / 100
    size = risk_usd / sl_dist
    notional = size * price
    leverage = min(MAX_LEVERAGE, int(notional / capital) + 1)

    if dry_run:
        print(f"\n  [DRY RUN] {action.upper()} @ ${price:,.2f}")
        print(f"  Size: {size:.4f} | Notional: ${notional:,.0f} | Leverage: {leverage}x")
        print(f"  SL: %{sl_pct} (${price*(1-sl_pct/100) if action=='long' else price*(1+sl_pct/100):,.0f})")
        print(f"  TP: %{tp_pct} (${price*(1+tp_pct/100) if action=='long' else price*(1-tp_pct/100):,.0f})")
        print(f"  Güven: %{confidence}")
        return "DRY_RUN"

    # Gerçek işlem — trade_executor.py üzerinden
    try:
        from trade_executor import TradeExecutor
        executor = TradeExecutor(
            api_key=BINGX_API_KEY,
            api_secret=BINGX_API_SECRET,
            testnet=(os.getenv("USE_TESTNET", "false").lower() == "true")
        )
        side = "BUY" if action == "long" else "SELL"
        result = executor.open_position(
            symbol=symbol,
            side=side,
            size=round(size, 4),
            leverage=leverage,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
        )
        return result.get("orderId", "OK")
    except Exception as e:
        print(f"  [HATA] Trade execute edilemedi: {e}")
        return None


# ── Sanal Pozisyon Takibi (dry-run için) ─────────────────────────
def check_virtual_position(memory: dict, symbol: str, open_trade_id: str) -> bool:
    """
    Dry-run modunda açık sanal pozisyonun SL/TP'ye ulaşıp ulaşmadığını kontrol eder.
    Pozisyon kapandıysa True döner.
    """
    trade = next((t for t in memory["trades"] if t["id"] == open_trade_id), None)
    if not trade:
        return True  # Bulunamadı, kapalı say

    entry  = trade["price_at_decision"]
    sl_pct = trade["sl_pct"] / 100
    tp_pct = trade["tp_pct"] / 100
    side   = trade["action"]

    if side == "long":
        sl_price = entry * (1 - sl_pct)
        tp_price = entry * (1 + tp_pct)
    else:
        sl_price = entry * (1 + sl_pct)
        tp_price = entry * (1 - tp_pct)

    # Güncel fiyat
    try:
        from market_context import fetch_candles
        candles = fetch_candles(symbol, "15m", limit=5)
        if not candles:
            return False
        current_high = max(c["h"] for c in candles[-3:])
        current_low  = min(c["l"] for c in candles[-3:])
        current_px   = candles[-1]["c"]
    except:
        return False

    result = None
    exit_px = None

    if side == "long":
        if current_low <= sl_price:
            result = "loss"; exit_px = sl_price
        elif current_high >= tp_price:
            result = "win"; exit_px = tp_price
    else:
        if current_high >= sl_price:
            result = "loss"; exit_px = sl_price
        elif current_low <= tp_price:
            result = "win"; exit_px = tp_price

    if result:
        pnl_pct = ((exit_px - entry) / entry * 100) if side == "long" else ((entry - exit_px) / entry * 100)
        log_outcome(memory, open_trade_id, {
            "result": result,
            "pnl_pct": round(pnl_pct, 3),
            "exit_price": exit_px,
            "exit_reason": "take_profit" if result == "win" else "stop_loss",
        })
        emoji = "[WIN]" if result == "win" else "[LOSS]"
        print(f"  {emoji} Pozisyon kapandi: {side.upper()} @ {entry:.0f} -> {exit_px:.0f} | PnL: %{pnl_pct:+.2f}")
        return True

    # Hâlâ açık
    if side == "long":
        unrealized = (current_px - entry) / entry * 100
    else:
        unrealized = (entry - current_px) / entry * 100
    print(f"  [ACIK] {side.upper()} @ {entry:.0f} | Simdi: {current_px:.0f} | Kar/Zarar: %{unrealized:+.2f} | SL:{sl_price:.0f} TP:{tp_price:.0f}")
    return False


# ── Rapor ─────────────────────────────────────────────────────────
def show_report(memory: dict):
    """Hafıza istatistiklerini gösterir."""
    total  = memory["total_trades"]
    wins   = memory["wins"]
    losses = memory["losses"]
    waits  = sum(1 for t in memory["trades"] if t.get("outcome") == "wait")
    wr     = wins / max(wins + losses, 1) * 100

    print("\n" + "="*55)
    print("  CLAUDE AI TRADER — PERFORMANS RAPORU")
    print("="*55)
    print(f"  Toplam karar : {total}")
    print(f"  Islem acilan : {wins + losses}  (Bekle: {waits})")
    print(f"  Win / Loss   : {wins}W / {losses}L")
    print(f"  Win Rate     : %{wr:.1f}")
    print(f"  Toplam PnL   : %{memory['total_pnl']:+.2f}")

    patterns = memory.get("learned_patterns", [])
    if patterns:
        print(f"\n  Ogrenilen {len(patterns)} pattern:")
        for p in patterns[-5:]:
            print(f"    [{p['date']}] {p['pattern'][:80]}")

    closed = list(t for t in memory["trades"] if t.get("outcome") not in (None, "wait"))[-5:]
    if closed:
        print(f"\n  Son {len(closed)} islem:")
        for t in closed:
            icon = "W" if t["outcome"] == "win" else "L"
            print(f"    [{icon}] {t['timestamp'][:10]} {t['action']:5s} @{t['price_at_decision']:.0f} -> %{t.get('pnl_pct', 0):+.2f}")
    print("="*55)


# ── Ana Döngü ────────────────────────────────────────────────────
def run(symbol: str = "BTC-USDT", dry_run: bool = False, once: bool = False):
    if not ANTHROPIC_API_KEY:
        print("[HATA] ANTHROPIC_API_KEY .env dosyasinda eksik!")
        return

    print(f"""
=== Claude AI Trader ===
Sembol : {symbol}
Model  : {MODEL}
Mod    : {'DRY RUN' if dry_run else 'CANLI'} | Aralik: {INTERVAL_MIN}dk
MinConf: %{MIN_CONFIDENCE}
========================
""")

    memory = load_memory()
    show_report(memory)

    open_trade_id = None
    open_trades = [t for t in memory["trades"]
                   if t["outcome"] is None and t["action"] != "wait"]
    if open_trades:
        open_trade_id = open_trades[-1]["id"]
        print(f"[INFO] Onceki acik pozisyon: {open_trade_id}")

    while True:
        now = datetime.now(timezone.utc)
        print(f"\n{'='*50}")
        print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Analiz basliyor...")

        # Açık sanal pozisyon varsa önce onu kontrol et
        if open_trade_id and dry_run:
            closed = check_virtual_position(memory, symbol, open_trade_id)
            if closed:
                open_trade_id = None

        try:
            decision = analyze_and_decide(symbol, memory)
        except Exception as e:
            print(f"[HATA] Analiz hatasi: {e}")
            decision = None

        if decision:
            action = decision.get("action", "wait")
            conf   = decision.get("confidence", 0)

            print(f"\n  KARAR: {action.upper()} | Guven: %{conf}")
            print(f"  Gerekcce: {decision.get('reasoning', '')}")
            print(f"  Faktorler: {' | '.join(decision.get('key_factors', []))}")
            print(f"  Riskler: {' | '.join(decision.get('risks', []))}")

            note = decision.get("learned_note")
            if note and note != "null":
                add_learned_pattern(memory, note)
                print(f"  [HAFIZA] Pattern kaydedildi: {note[:80]}")

            if action != "wait" and conf >= MIN_CONFIDENCE:
                if open_trade_id:
                    print(f"  [UYARI] Acik pozisyon var: {open_trade_id} — yeni islem atlanıyor")
                else:
                    trade_id = log_decision(memory, decision, {
                        "symbol": symbol,
                        "technicals": {"price": decision.get("price", 0)}
                    })
                    # SL/TP fiyatlarını hafızaya yaz (takip için)
                    for t in memory["trades"]:
                        if t["id"] == trade_id:
                            t["sl_pct"] = decision.get("sl_pct", 1.0)
                            t["tp_pct"] = decision.get("tp_pct", 3.0)
                    save_memory(memory)
                    print(f"  [HAFIZA] Karar kaydedildi: {trade_id}")

                    capital  = float(os.getenv("BOT_CAPITAL", "1000"))
                    order_id = execute_trade(decision, symbol, capital, dry_run)
                    if order_id:
                        open_trade_id = trade_id
                        print(f"  [ISLEM] Acildi: {order_id}")
            else:
                print(f"  >> {'Bekleniyor' if action == 'wait' else f'Guven %{conf} < %{MIN_CONFIDENCE} esigi'}")
                log_decision(memory, decision, {
                    "symbol": symbol,
                    "technicals": {"price": decision.get("price", 0)}
                })
                if memory["trades"]:
                    memory["trades"][-1]["outcome"] = "wait"
                    save_memory(memory)

        if once:
            print("\n[INFO] --once modu, cikiliyor.")
            break

        next_check = INTERVAL_MIN * 60
        next_time  = datetime.fromtimestamp(
            time.time() + next_check, tz=timezone.utc
        ).strftime("%H:%M UTC")
        print(f"\n  Sonraki analiz: {next_time} ({INTERVAL_MIN} dk)")
        time.sleep(next_check)


# ── Argümanlar ────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Claude AI Trader")
    p.add_argument("--symbol",         default="BTC-USDT")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--once",           action="store_true")
    p.add_argument("--interval",       type=int,   default=15)
    p.add_argument("--min-confidence", type=int,   default=65)
    p.add_argument("--report",         action="store_true", help="Sadece rapor goster, cikis yap")
    a = p.parse_args()

    INTERVAL_MIN   = a.interval
    MIN_CONFIDENCE = a.min_confidence

    if a.report:
        memory = load_memory()
        show_report(memory)
    else:
        run(symbol=a.symbol, dry_run=a.dry_run, once=a.once)
