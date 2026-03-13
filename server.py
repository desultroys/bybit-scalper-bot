"""
server.py - FastAPI web sunucusu (BingX)
http://localhost:8080
"""
from __future__ import annotations
import asyncio, json, os, sys, time
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse
    import uvicorn
    from pydantic import BaseModel
except ImportError:
    print("[HATA] fastapi/uvicorn yuklu degil: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="BingX Scalper Dashboard", version="2.0")
bot_task: Optional[asyncio.Task] = None
bot_state = None
ws_clients: list = []
backtest_status: dict = {"running": False, "progress": "", "result": None, "error": ""}


class BotSettings(BaseModel):
    bingx_api_key:     str   = ""
    bingx_api_secret:  str   = ""
    coinglass_api_key: str   = ""
    use_testnet:       bool  = True
    leverage:          int   = 15
    stop_loss_pct:     float = 0.4
    take_profit_pct:   float = 1.0
    trailing_stop_pct: float = 0.3
    max_risk_pct:      float = 2.0
    initial_capital:   float = 1000.0
    symbols:           list  = ["BTC-USDT", "ETH-USDT"]


def _load_env() -> dict:
    env = {}
    if Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _save_env(s: BotSettings):
    lines = [
        "BINGX_API_KEY="     + s.bingx_api_key,
        "BINGX_API_SECRET="  + s.bingx_api_secret,
        "COINGLASS_API_KEY=" + s.coinglass_api_key,
        "USE_TESTNET="       + ("true" if s.use_testnet else "false"),
        "BOT_LEVERAGE="      + str(s.leverage),
        "BOT_SL_PCT="        + str(s.stop_loss_pct),
        "BOT_TP_PCT="        + str(s.take_profit_pct),
        "BOT_TS_PCT="        + str(s.trailing_stop_pct),
        "BOT_RISK_PCT="      + str(s.max_risk_pct),
        "BOT_CAPITAL="       + str(s.initial_capital),
        "BOT_SYMBOLS="       + ",".join(s.symbols),
    ]
    Path(".env").write_text("\n".join(lines), encoding="utf-8")
    os.environ.update({
        "BINGX_API_KEY":     s.bingx_api_key,
        "BINGX_API_SECRET":  s.bingx_api_secret,
        "COINGLASS_API_KEY": s.coinglass_api_key,
        "USE_TESTNET":       "true" if s.use_testnet else "false",
        "BOT_LEVERAGE":      str(s.leverage),
        "BOT_SL_PCT":        str(s.stop_loss_pct),
        "BOT_TP_PCT":        str(s.take_profit_pct),
        "BOT_TS_PCT":        str(s.trailing_stop_pct),
        "BOT_RISK_PCT":      str(s.max_risk_pct),
        "BOT_CAPITAL":       str(s.initial_capital),
        "BOT_SYMBOLS":       ",".join(s.symbols),
    })


async def _run_bot():
    global bot_state
    try:
        import importlib, config as cm, models as mm
        importlib.reload(cm)
        from models import BotState
        from config import CFG
        from websocket_handler import orderbook_ws_loop
        from coinglass_fetcher import liquidation_fetch_loop
        from signal_generator import signal_loop
        from trade_executor import trade_executor_loop
        from ml_optimizer import MLOptimizer, ml_optimizer_loop
        from trade_logger import trade_logger_loop

        bot_state = BotState(
            capital_usdt=CFG.initial_capital_usdt,
            initial_capital=CFG.initial_capital_usdt,
            daily_start_capital=CFG.initial_capital_usdt,
        )
        ml_opt = MLOptimizer()
        sq: asyncio.Queue = asyncio.Queue(maxsize=50)
        tlq: asyncio.Queue = asyncio.Queue()
        await asyncio.gather(
            orderbook_ws_loop(bot_state),
            liquidation_fetch_loop(bot_state),
            signal_loop(bot_state, sq, ml_opt),
            trade_executor_loop(bot_state, sq, tlq),
            ml_optimizer_loop(bot_state, ml_opt),
            trade_logger_loop(bot_state, tlq),
            return_exceptions=True,
        )
    except Exception as e:
        import traceback
        print("Bot hatasi:", e)
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    f = Path("dashboard.html")
    return HTMLResponse(f.read_text(encoding="utf-8") if f.exists() else "<h1>dashboard.html bulunamadi</h1>")


@app.get("/api/settings")
async def get_settings():
    env = _load_env()
    return {
        "bingx_api_key":    env.get("BINGX_API_KEY", ""),
        "bingx_api_secret": "***" if env.get("BINGX_API_SECRET") else "",
        "coinglass_api_key": env.get("COINGLASS_API_KEY", ""),
        "use_testnet":      env.get("USE_TESTNET", "true").lower() == "true",
        "leverage":         int(os.getenv("BOT_LEVERAGE", "15")),
        "stop_loss_pct":    float(os.getenv("BOT_SL_PCT", "0.4")),
        "take_profit_pct":  float(os.getenv("BOT_TP_PCT", "1.0")),
        "trailing_stop_pct": float(os.getenv("BOT_TS_PCT", "0.3")),
        "max_risk_pct":     float(os.getenv("BOT_RISK_PCT", "2.0")),
        "initial_capital":  float(os.getenv("BOT_CAPITAL", "1000")),
        "symbols":          os.getenv("BOT_SYMBOLS", "BTC-USDT,ETH-USDT").split(","),
        "exchange":         "BingX",
    }


@app.post("/api/settings")
async def save_settings(s: BotSettings):
    _save_env(s)
    return {"ok": True, "message": "Ayarlar kaydedildi"}


@app.post("/api/bot/start")
async def start_bot():
    global bot_task
    if bot_task and not bot_task.done():
        return {"ok": False, "message": "Bot zaten calisiyor"}
    env = _load_env()
    key    = os.getenv("BINGX_API_KEY")    or env.get("BINGX_API_KEY", "")
    secret = os.getenv("BINGX_API_SECRET") or env.get("BINGX_API_SECRET", "")
    if not key or key in ("", "your_api_key_here"):
        return {"ok": False, "message": "BINGX_API_KEY ayarlanmamis! Ayarlar sekmesinden girin."}
    if not secret or secret in ("", "your_api_secret_here"):
        return {"ok": False, "message": "BINGX_API_SECRET ayarlanmamis! Ayarlar sekmesinden girin."}
    os.environ["BINGX_API_KEY"]    = key
    os.environ["BINGX_API_SECRET"] = secret
    bot_task = asyncio.create_task(_run_bot())
    return {"ok": True, "message": "Bot baslatildi"}


@app.post("/api/bot/stop")
async def stop_bot():
    global bot_state, bot_task
    if bot_state:
        bot_state.is_running = False
    if bot_task and not bot_task.done():
        bot_task.cancel()
    bot_task = None
    return {"ok": True, "message": "Bot durduruldu"}


@app.get("/api/bot/status")
async def bot_status():
    running = bot_task is not None and not bot_task.done()
    if bot_state:
        return {
            "running": running,
            "capital": round(bot_state.capital_usdt, 2),
            "initial_capital": round(bot_state.initial_capital, 2),
            "daily_pnl": round(bot_state.daily_pnl_usdt, 2),
            "daily_pnl_pct": round(bot_state.daily_pnl_pct, 2),
            "total_trades": bot_state.total_trades,
            "win_rate": round(bot_state.win_rate, 1),
            "drawdown_pct": round(bot_state.total_drawdown_pct, 2),
            "open_positions": [
                {"symbol": sym, "side": p.side.value,
                 "entry": round(p.entry_price, 2), "size": p.size,
                 "sl": round(p.stop_loss, 2), "tp": round(p.take_profit, 2)}
                for sym, p in bot_state.positions.items()
            ],
            "orderbooks": {
                sym: {"bid": round(ob.best_bid, 2), "ask": round(ob.best_ask, 2),
                      "mid": round(ob.mid_price, 2), "imbalance": round(ob.imbalance_ratio(), 3)}
                for sym, ob in bot_state.orderbooks.items() if ob.mid_price > 0
            },
        }
    return {"running": running, "capital": 0, "initial_capital": 0,
            "total_trades": 0, "win_rate": 0, "daily_pnl": 0,
            "daily_pnl_pct": 0, "drawdown_pct": 0,
            "open_positions": [], "orderbooks": {}}


@app.get("/api/trades")
async def get_trades(limit: int = 200):
    f = Path("logs/trades.jsonl")
    if not f.exists():
        return {"trades": [], "total": 0}
    trades = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try: trades.append(json.loads(line))
        except: pass
    trades.reverse()
    return {"trades": trades[:limit], "total": len(trades)}


@app.get("/api/performance")
async def get_performance():
    f = Path("logs/performance.json")
    if not f.exists(): return {}
    try: return json.loads(f.read_text(encoding="utf-8"))
    except: return {}


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    f = Path("logs/bot.log")
    if not f.exists(): return {"lines": []}
    try:
        all_lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": all_lines[-lines:]}
    except: return {"lines": []}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await bot_status()
            await ws.send_json(data)
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# --- BACKTEST ---

def _make_json_safe(obj):
    """Tum numpy/pandas tiplerini JSON-uyumlu Python tiplerine donustur."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(i) for i in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_make_json_safe(i) for i in obj.tolist()]
    if obj != obj:  # NaN check
        return None
    return obj


@app.post("/api/backtest/run")
async def run_backtest_api(params: dict):
    global backtest_status
    if backtest_status["running"]:
        return {"ok": False, "message": "Backtest zaten calisiyor"}
    backtest_status = {"running": True, "progress": "Baslatiliyor...", "result": None, "error": ""}

    async def _run():
        global backtest_status
        try:
            import importlib, backtest as bt
            importlib.reload(bt)
            symbol   = params.get("symbol", "BTC-USDT")
            days     = int(params.get("days", 90))
            interval = int(params.get("interval", 15))
            capital  = float(params.get("capital", 1000))
            leverage = int(params.get("leverage", 15))
            sl       = float(params.get("sl", 0.4))
            tp       = float(params.get("tp", 1.0))
            trailing = float(params.get("trailing", 0.3))
            backtest_status["progress"] = symbol + " verisi cekiliyor..."
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: bt.run(symbol, days, interval, capital, leverage, sl, tp, trailing)
            )
            # Numpy tiplerini JSON-uyumlu Python tiplerine cevir
            result = _make_json_safe(result)
            backtest_status["result"]   = result
            backtest_status["progress"] = "Tamamlandi"
        except Exception as e:
            backtest_status["error"]    = str(e)
            backtest_status["progress"] = "Hata: " + str(e)
        finally:
            backtest_status["running"] = False

    asyncio.create_task(_run())
    return {"ok": True, "message": "Backtest baslatildi"}


@app.get("/api/backtest/status")
async def get_backtest_status():
    return backtest_status


@app.get("/api/backtest/result/{symbol}")
async def get_backtest_result(symbol: str):
    sym = symbol.upper().replace("-", "")
    if (backtest_status.get("result") or {}).get("symbol","").replace("-","").upper() == sym:
        return backtest_status["result"]
    for fname in [f"logs/backtest_{sym}.json", f"logs/backtest_{symbol}.json"]:
        if Path(fname).exists():
            try: return json.loads(Path(fname).read_text(encoding="utf-8"))
            except: pass
    return {}


if __name__ == "__main__":
    print("================================================")
    print("  BingX Scalper Web Dashboard")
    print("  Tarayicide acin: http://localhost:8080")
    print("================================================")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
