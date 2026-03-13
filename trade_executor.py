"""
trade_executor.py - BingX Perpetual Futures REST API
Imzalama: HMAC-SHA256, query string bazli
Dok: https://bingx-api.github.io/docs/swapV2/trade-api.html
"""
from __future__ import annotations
import asyncio, hashlib, hmac, json, time, urllib.parse
from typing import Optional, TYPE_CHECKING
import aiohttp
import json
from config import CFG, USE_TESTNET, BINGX_API_KEY, BINGX_API_SECRET, BINGX_REST
from models import Side, Position, Signal, TradeRecord
from logger import log

if TYPE_CHECKING:
    from models import BotState

BASE = BINGX_REST
_TIME_OFFSET_MS = 0


async def _sync_time(session: aiohttp.ClientSession):
    """BingX sunucu saatiyle senkronize et."""
    global _TIME_OFFSET_MS
    try:
        async with session.get(BASE + "/openApi/swap/v2/server/time",
                               timeout=aiohttp.ClientTimeout(total=5)) as r:
            d = await r.json()
            server_ts = int(d.get("data", {}).get("serverTime", time.time()*1000))
            _TIME_OFFSET_MS = server_ts - int(time.time() * 1000)
            log.info(f"BingX zaman sync. Fark: {_TIME_OFFSET_MS}ms")
    except Exception as e:
        log.warning(f"Zaman sync basarisiz: {e}")
        _TIME_OFFSET_MS = 0


def _sign(params: dict) -> dict:
    """BingX imzalama: timestamp ekle, tüm parametreleri sırala, HMAC-SHA256."""
    params["timestamp"] = str(int(time.time() * 1000) + _TIME_OFFSET_MS)
    # BingX: parametreleri alfabetik sırala, & ile birleştir
    sorted_params = sorted(params.items())
    query = "&".join(f"{k}={v}" for k, v in sorted_params)
    sig = hmac.new(
        BINGX_API_SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    params["signature"] = sig
    return params


def _headers() -> dict:
    return {
        "X-BX-APIKEY": BINGX_API_KEY,
        "Content-Type": "application/json",
    }


async def _get(session: aiohttp.ClientSession, path: str, params: dict) -> dict:
    params = _sign(params)
    url = BASE + path
    try:
        async with session.get(url, params=params, headers=_headers(),
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.json()
    except Exception as e:
        log.error(f"GET {path} hatasi: {e}")
        return {"code": -1, "msg": str(e)}


async def _post(session: aiohttp.ClientSession, path: str, params: dict) -> dict:
    params = _sign(params)
    # BingX: parametreler query string'de (imza dahil)
    qs  = urllib.parse.urlencode(params)
    url = BASE + path + "?" + qs
    try:
        async with session.post(url, headers=_headers(),
                                timeout=aiohttp.ClientTimeout(total=10)) as r:
            text = await r.text()
            try:
                return json.loads(text)
            except Exception:
                return {"code": -1, "msg": text[:200]}
    except Exception as e:
        log.error(f"POST {path} hatasi: {e}")
        return {"code": -1, "msg": str(e)}


async def set_leverage(session: aiohttp.ClientSession, symbol: str, leverage: int):
    """Kaldirec ayarla - BingX swap v2."""
    # BingX: long ve short ayri ayarlanir
    for side in ["LONG", "SHORT"]:
        params = {
            "symbol":       symbol,
            "side":         side,
            "leverage":     str(leverage),
            "recvWindow":   "5000",
        }
        r = await _post(session, "/openApi/swap/v2/trade/leverage", params)
        code = r.get("code", -1)
        if code == 0:
            log.info(f"Kaldirec ayarlandi: {symbol} {side} {leverage}x")
        elif code in (80012, 80014):
            log.info(f"Kaldirec zaten {leverage}x: {symbol} {side}")
        else:
            log.warning(f"Kaldirec hatasi {symbol} {side}: {r.get('msg',r)}")
        await asyncio.sleep(0.15)


async def get_wallet_balance(session: aiohttp.ClientSession) -> float:
    """USDT bakiyesini al."""
    r = await _get(session, "/openApi/swap/v2/user/balance", {})
    try:
        balance = r["data"]["balance"]
        return float(balance.get("availableMargin", 0))
    except (KeyError, TypeError):
        log.warning(f"Bakiye alinamadi: {r.get('msg', 'bilinmiyor')}")
        return 0.0


def _calc_qty(capital: float, price: float, leverage: int,
              risk_pct: float, sl_pct: float) -> float:
    """Risk bazli pozisyon buyuklugu."""
    risk_usdt = capital * risk_pct / 100
    sl_dist   = price * sl_pct / 100
    if sl_dist == 0:
        return 0.0
    qty = risk_usdt / sl_dist
    max_qty = (capital * leverage) / price
    qty = min(qty, max_qty)
    # BTC icin 0.001, ETH icin 0.01 minimum
    min_qty = 0.001 if "BTC" in "symbol" else 0.01
    return round(max(qty, min_qty), 4)


async def place_order(session: aiohttp.ClientSession, symbol: str,
                      side: Side, qty: float,
                      stop_loss: float, take_profit: float) -> Optional[str]:
    """
    BingX market order + SL + TP.
    positionSide: LONG veya SHORT (one-way mode icin BOTH da kullanilabilir)
    """
    side_str = "BUY" if side == Side.LONG else "SELL"
    pos_side = "LONG" if side == Side.LONG else "SHORT"

    params = {
        "symbol":       symbol,
        "side":         side_str,
        "positionSide": pos_side,
        "type":         "MARKET",
        "quantity":     str(qty),
        "stopLoss":     json.dumps({"type": "MARK_PRICE", "stopPrice": str(round(stop_loss, 2)), "workingType": "MARK_PRICE"}),
        "takeProfit":   json.dumps({"type": "MARK_PRICE", "stopPrice": str(round(take_profit, 2)), "workingType": "MARK_PRICE"}),
    }

    log.info(f"EMIR: {symbol} {side_str} qty={qty} SL={stop_loss:.2f} TP={take_profit:.2f}")
    r = await _post(session, "/openApi/swap/v2/trade/order", params)

    if r.get("code") == 0:
        order_id = str(r.get("data", {}).get("order", {}).get("orderId", ""))
        log.info(f"Emir kabul edildi: {order_id}")
        return order_id
    else:
        log.error(f"Emir reddedildi: {r.get('msg')} (kod:{r.get('code')})")
        return None


async def close_position(session: aiohttp.ClientSession, symbol: str,
                         side: Side, qty: float, reason: str = "") -> bool:
    """Pozisyonu kapat (reduce-only market order)."""
    close_side = "SELL" if side == Side.LONG else "BUY"
    pos_side   = "LONG" if side == Side.LONG else "SHORT"

    params = {
        "symbol":       symbol,
        "side":         close_side,
        "positionSide": pos_side,
        "type":         "MARKET",
        "quantity":     str(qty),
        "reduceOnly":   "true",
    }
    log.info(f"POZ KAPANIYOR: {symbol} {reason}")
    r = await _post(session, "/openApi/swap/v2/trade/order", params)
    if r.get("code") == 0:
        log.info(f"Pozisyon kapatildi: {symbol}")
        return True
    else:
        log.error(f"Kapatma hatasi: {r.get('msg')}")
        return False


async def trade_executor_loop(state: "BotState", signal_queue: asyncio.Queue,
                               trade_log_queue: asyncio.Queue):
    async with aiohttp.ClientSession() as session:
        # Zaman sync
        await _sync_time(session)

        # Kaldirec ayarla
        for sym in CFG.symbols:
            await set_leverage(session, sym, CFG.leverage)

        # Bakiye
        balance = await get_wallet_balance(session)
        if balance > 0:
            state.capital_usdt = balance
            state.daily_start_capital = balance
            log.info(f"Baslangic bakiyesi: {balance:.2f} USDT")
        else:
            log.warning("Bakiye alinamadi, varsayilan kullaniliyor")
            state.daily_start_capital = state.capital_usdt

        while state.is_running:
            # Sinyal
            try:
                signal: Signal = signal_queue.get_nowait()
            except asyncio.QueueEmpty:
                signal = None

            if signal is not None:
                await _process_signal(session, signal, state, trade_log_queue)

            # Acik pozisyonlari izle
            await _monitor_positions(session, state, trade_log_queue)

            # Periyodik bakiye guncelle
            if state.total_trades % 10 == 0 and state.total_trades > 0:
                bal = await get_wallet_balance(session)
                if bal > 0:
                    state.capital_usdt = bal

            await asyncio.sleep(0.5)

    log.info("Trade executor loop sonlandi.")


async def _process_signal(session, signal: Signal, state: "BotState",
                           trade_log_queue: asyncio.Queue):
    symbol = signal.symbol
    if symbol in state.positions:
        return

    can_trade, reason = state.can_trade()
    if not can_trade:
        log.warning(f"Trade yapilamaz: {reason}")
        return

    price = signal.price
    sl_pct = CFG.stop_loss_pct / 100
    tp_pct = CFG.take_profit_pct / 100

    if signal.side == Side.LONG:
        sl = price * (1 - sl_pct)
        tp = price * (1 + tp_pct)
    else:
        sl = price * (1 + sl_pct)
        tp = price * (1 - tp_pct)

    qty = _calc_qty(state.capital_usdt, price, CFG.leverage,
                    CFG.max_risk_per_trade_pct, CFG.stop_loss_pct)
    if qty <= 0:
        return

    order_id = await place_order(session, symbol, signal.side, qty, sl, tp)
    if order_id:
        pos = Position(
            symbol=symbol, side=signal.side,
            entry_price=price, size=qty, leverage=CFG.leverage,
            stop_loss=sl, take_profit=tp,
            trailing_stop_pct=CFG.trailing_stop_pct,
            order_id=order_id, highest_price=price, lowest_price=price,
        )
        state.positions[symbol] = pos
        log.info(f"POZ ACILDI: {symbol} {signal.side.value} {qty} @ {price:.2f}")


async def _monitor_positions(session, state: "BotState",
                              trade_log_queue: asyncio.Queue):
    for symbol, pos in list(state.positions.items()):
        ob = state.orderbooks.get(symbol)
        if not ob or ob.mid_price == 0:
            continue

        current = ob.mid_price
        exit_reason = None

        if pos.side == Side.LONG:
            pos.highest_price = max(pos.highest_price, current)
            trail_sl = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
            if trail_sl > pos.stop_loss:
                pos.stop_loss = trail_sl
            if current <= pos.stop_loss:
                exit_reason = "stop_loss"
            elif current >= pos.take_profit:
                exit_reason = "take_profit"
        else:
            pos.lowest_price = min(pos.lowest_price, current)
            trail_sl = pos.lowest_price * (1 + pos.trailing_stop_pct / 100)
            if trail_sl < pos.stop_loss:
                pos.stop_loss = trail_sl
            if current >= pos.stop_loss:
                exit_reason = "stop_loss"
            elif current <= pos.take_profit:
                exit_reason = "take_profit"

        if exit_reason:
            ok = await close_position(session, symbol, pos.side, pos.size, exit_reason)
            if ok:
                await _record_trade(pos, current, exit_reason, state, trade_log_queue)
                del state.positions[symbol]


async def _record_trade(pos: Position, exit_price: float, exit_reason: str,
                         state: "BotState", trade_log_queue: asyncio.Queue):
    if pos.side == Side.LONG:
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 * pos.leverage
    else:
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100 * pos.leverage

    pnl_usdt = pos.notional_usdt * (pnl_pct / 100)
    # BingX taker komisyonu: %0.05
    commission = pos.notional_usdt * 0.05 / 100 * 2
    pnl_usdt -= commission

    import time as _time
    record = TradeRecord(
        symbol=pos.symbol, side=pos.side.value,
        entry_price=pos.entry_price, exit_price=exit_price,
        size=pos.size, leverage=pos.leverage,
        pnl_usdt=round(pnl_usdt, 4), pnl_pct=round(pnl_pct, 2),
        duration_sec=_time.time() - pos.open_ts,
        exit_reason=exit_reason,
        open_ts=pos.open_ts, close_ts=_time.time(),
        signal_score=0.0,
    )

    state.capital_usdt += pnl_usdt
    state.daily_pnl_usdt += pnl_usdt
    state.total_trades += 1
    if record.is_win:
        state.winning_trades += 1
    state.trade_history.append(record)

    emoji = "✅" if record.is_win else "❌"
    log.info(f"{emoji} TRADE: {pos.symbol} {pos.side.value} | "
             f"PNL: {pnl_usdt:+.2f} USDT (%{pnl_pct:+.1f}) | "
             f"{exit_reason} | WinRate: %{state.win_rate:.1f}")
    await trade_log_queue.put(record)
