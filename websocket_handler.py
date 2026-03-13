"""
websocket_handler.py - BingX Perpetual Futures WebSocket
BingX format: dataType = "BTC-USDT@depth20@500ms"
"""
from __future__ import annotations
import asyncio, json, gzip, time, uuid
from typing import TYPE_CHECKING
import websockets
from websockets.exceptions import ConnectionClosed
from config import CFG, BINGX_WS
from models import OrderBook, OrderBookLevel
from logger import log

if TYPE_CHECKING:
    from models import BotState


def _parse_levels(raw: list) -> list:
    levels = []
    for item in raw:
        try:
            if isinstance(item, (list, tuple)):
                price, size = float(item[0]), float(item[1])
            elif isinstance(item, dict):
                price = float(item.get("p", item.get("price", 0)))
                size  = float(item.get("v", item.get("qty", 0)))
            else:
                continue
            if size > 0:
                levels.append(OrderBookLevel(price=price, size=size))
        except (ValueError, TypeError):
            continue
    return levels


async def orderbook_ws_loop(state: "BotState"):
    for sym in CFG.symbols:
        if sym not in state.orderbooks:
            state.orderbooks[sym] = OrderBook(symbol=sym)

    while state.is_running:
        try:
            log.info(f"BingX WS baglaniliyor: {BINGX_WS}")
            async with websockets.connect(
                BINGX_WS, ping_interval=20, ping_timeout=15,
                close_timeout=5, max_size=10*1024*1024,
            ) as ws:
                for sym in CFG.symbols:
                    await ws.send(json.dumps({
                        "id": str(uuid.uuid4())[:8],
                        "reqType": "sub",
                        "dataType": f"{sym}@depth20@500ms",
                    }))
                    log.info(f"Subscribe: {sym}@depth20@500ms")
                    await asyncio.sleep(0.2)

                async for raw in ws:
                    if not state.is_running:
                        break
                    try:
                        if isinstance(raw, bytes):
                            try: raw = gzip.decompress(raw).decode()
                            except: raw = raw.decode()
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if msg.get("ping"):
                        await ws.send(json.dumps({"pong": msg["ping"]}))
                        continue
                    if msg.get("code") == 0:
                        continue

                    dt = msg.get("dataType", "")
                    if "@depth" not in dt:
                        continue

                    symbol = dt.split("@")[0]
                    ob = state.orderbooks.get(symbol)
                    if not ob:
                        continue

                    data = msg.get("data", {})
                    bids_raw = data.get("bids", [])
                    asks_raw = data.get("asks", [])
                    if bids_raw or asks_raw:
                        ob.bids = sorted(_parse_levels(bids_raw), key=lambda x: x.price, reverse=True)
                        ob.asks = sorted(_parse_levels(asks_raw), key=lambda x: x.price)
                        ob.ts_ms = int(time.time() * 1000)

        except ConnectionClosed as e:
            log.warning(f"WS kapandi: {e}. Yeniden baglaniliyor...")
            await asyncio.sleep(CFG.ws_reconnect_delay)
        except Exception as e:
            log.error(f"WS hatasi: {e}")
            await asyncio.sleep(CFG.ws_reconnect_delay)
    log.info("WebSocket loop sonlandi.")
