"""
rule_signal.py — Kural Tabanlı Sinyal Motoru
=============================================
Backtest.py'deki aynı score() mantığını canlı veriye uygular.
Her 60 dakikada bir BingX'ten veri çeker, sinyal üretir, sanal P&L takip eder.
$1000 sermaye | 20x kaldıraç | %2 risk/trade | SL:%1.5 TP:%4.5

Çalıştırma:
  python rule_signal.py
  python rule_signal.py --interval 60
  python rule_signal.py --once
"""
import sys, io, os, json, time, argparse
from datetime import datetime, timezone

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import numpy as np
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'requests', '-q'])
    import numpy as np
    import requests

# ── Ayarlar ───────────────────────────────────────────────────────
SYMBOL       = 'BTC-USDT'
INTERVAL     = '15m'
CAPITAL      = 1000.0
LEVERAGE     = 20
RISK_PCT     = 2.0
SL_PCT       = 1.5
TP_PCT       = 4.5
MIN_SCORE    = 9
LOG_FILE     = 'logs/rule_signals.json'
LOOP_MIN     = 60

# ── Veri çekme ────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = 220) -> list:
    urls = [
        'https://open-api.bingx.com/openApi/swap/v3/quote/klines',
        'https://open-api.bingx.com/openApi/swap/v2/quote/klines',
    ]
    for url in urls:
        try:
            r = requests.get(url, params={
                'symbol': symbol, 'interval': interval, 'limit': str(limit)
            }, timeout=15)
            raw = r.json()
            if isinstance(raw, dict):
                data = raw.get('data', [])
            else:
                data = raw
            if not data:
                continue
            rows = []
            for c in data:
                if isinstance(c, dict):
                    rows.append({
                        't': str(c.get('time', c.get('openTime', ''))),
                        'o': float(c.get('open', 0)),
                        'h': float(c.get('high', 0)),
                        'l': float(c.get('low', 0)),
                        'c': float(c.get('close', 0)),
                        'v': float(c.get('volume', 1)),
                    })
                elif isinstance(c, list) and len(c) >= 5:
                    rows.append({'t': str(c[0]), 'o': float(c[1]), 'h': float(c[2]),
                                 'l': float(c[3]), 'c': float(c[4]),
                                 'v': float(c[5]) if len(c) > 5 else 1.0})
            if rows:
                rows.sort(key=lambda x: int(x['t']))
                return rows
        except Exception as e:
            print(f'  [HATA] klines: {e}')
    return []


def fetch_orderbook(symbol: str) -> float:
    """Orderbook imbalance döndürür."""
    try:
        r = requests.get('https://open-api.bingx.com/openApi/swap/v2/quote/depth',
                         params={'symbol': symbol, 'limit': '20'}, timeout=8)
        d = r.json().get('data', {})
        bids = d.get('bids', [])[:10]
        asks = d.get('asks', [])[:10]
        bv = sum(float(b[1]) for b in bids)
        av = sum(float(a[1]) for a in asks)
        return (bv - av) / (bv + av + 1e-10)
    except:
        return 0.0


# ── Gösterge hesaplama ────────────────────────────────────────────
def _ema(arr, n):
    e = np.zeros(len(arr)); e[0] = arr[0]; k = 2 / (n + 1)
    for i in range(1, len(arr)):
        e[i] = arr[i] * k + e[i-1] * (1 - k)
    return e

def _rsi(arr, n=14):
    d = np.diff(arr)
    g = np.where(d > 0, d, 0.0); ls = np.where(d < 0, -d, 0.0)
    ag = np.convolve(g, np.ones(n)/n, 'valid')
    al = np.convolve(ls, np.ones(n)/n, 'valid')
    return float(100 - 100 / (1 + ag[-1] / (al[-1] + 1e-10)))

def _atr(h, l, c, n=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    return float(tr[-n:].mean())

def _adx(h, l, c, n=14):
    pd_ = np.zeros(len(c)); md_ = np.zeros(len(c)); tr = np.zeros(len(c))
    for i in range(1, len(c)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        u = h[i]-h[i-1]; dn = l[i-1]-l[i]
        pd_[i] = u if u > dn and u > 0 else 0
        md_[i] = dn if dn > u and dn > 0 else 0
    dx = np.zeros(len(c))
    for i in range(n, len(c)):
        at = tr[i-n:i].mean() + 1e-10
        pv = pd_[i-n:i].sum()/at*100; mv = md_[i-n:i].sum()/at*100
        dx[i] = abs(pv-mv)/(pv+mv+1e-10)*100
    s = np.zeros(len(c)); s[n] = dx[n]
    for i in range(n+1, len(c)): s[i] = (s[i-1]*(n-1)+dx[i])/n
    return float(s[-1])

def compute_indicators(candles: list) -> dict | None:
    if len(candles) < 215:
        return None
    o = np.array([c['o'] for c in candles], dtype=float)
    h = np.array([c['h'] for c in candles], dtype=float)
    l = np.array([c['l'] for c in candles], dtype=float)
    c = np.array([c['c'] for c in candles], dtype=float)
    v = np.array([c_['v'] for c_ in candles], dtype=float)

    e9  = _ema(c, 9);  e21 = _ema(c, 21)
    e50 = _ema(c, 50); e200= _ema(c, 200)
    rsi = _rsi(c, 14)
    atr = _atr(h, l, c, 14)
    adx = _adx(h, l, c, 14)
    atr_pct = atr / c[-1] * 100

    vm = np.convolve(v, np.ones(20)/20, 'full')[:len(v)]
    vratio = float(v[-1] / (vm[-1] + 1e-10))

    body     = float(c[-1] - o[-1])
    body_pct = float(abs(body) / (h[-1] - l[-1] + 1e-10))

    # Orderbook proxy
    bv_ = np.where(c >= o, v, v * abs(c-o)/(h-l+1e-10))
    av_ = np.where(c <  o, v, v * abs(c-o)/(h-l+1e-10))
    bvm = np.convolve(bv_, np.ones(10)/10, 'full')[:len(v)]
    avm = np.convolve(av_, np.ones(10)/10, 'full')[:len(v)]
    ob_imb = float((bvm[-1]-avm[-1])/(bvm[-1]+avm[-1]+1e-10))

    delta5  = float(np.sum((bv_-av_)[-5:]))

    # Likidasyon kümeleri
    LB = 40; px = c[-1]
    rh = h[-LB-1:-1]; rl = l[-LB-1:-1]
    liq_s = float(np.sum((rh > px*1.005) & (rh < px*1.03)) / LB)
    liq_l = float(np.sum((rl < px*0.995) & (rl > px*0.97)) / LB)

    # VWAP
    tp = (h + l + c) / 3
    vwap = float(np.sum(tp[-20:]*v[-20:]) / (np.sum(v[-20:]) + 1e-10))

    # Bollinger
    bm = c[-20:].mean(); bs = c[-20:].std()
    bb_pct = float((c[-1] - (bm - 2*bs)) / (4*bs + 1e-10))

    # Momentum
    mom3 = float(c[-1] - c[-4]) if len(c) >= 4 else 0.0
    mom8 = float(c[-1] - c[-9]) if len(c) >= 9 else 0.0

    # CVD50
    delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    cvd50 = float(np.sum(delta[-50:]))

    # Premium/Discount
    rh50 = h[-50:].max(); rl50 = l[-50:].min()
    pd_pct = float((c[-1] - (rh50+rl50)/2) / (rh50-rl50+1e-10) * 100)

    # Swing H/L & BOS (edge-triggered, 3-bar window)
    sh_arr = np.full(len(c), np.nan)
    sl_arr = np.full(len(c), np.nan)
    for i in range(1, len(c)-1):
        if h[i] > h[i-1] and h[i] > h[i+1]: sh_arr[i] = h[i]
        if l[i] < l[i-1] and l[i] < l[i+1]: sl_arr[i] = l[i]

    def ffill(arr):
        out = arr.copy()
        last = np.nan
        for i in range(len(out)):
            if not np.isnan(out[i]): last = out[i]
            else: out[i] = last
        return out

    last_sh = ffill(sh_arr)
    last_sl = ffill(sl_arr)

    # Edge-triggered: geçen bar kırmadı, bu bar kırdı
    def bos_edge(close_arr, level_arr):
        flags = np.zeros(len(close_arr), dtype=int)
        for i in range(1, len(close_arr)):
            if (not np.isnan(level_arr[i]) and not np.isnan(level_arr[i-1])
                    and close_arr[i] > level_arr[i]
                    and close_arr[i-1] <= level_arr[i-1]):
                flags[i] = 1
        # 3 bar geçerlilik
        out = np.zeros(len(close_arr), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                out[i:i+3] = 1
        return out

    bos_bull_arr = bos_edge(c, last_sh)
    bos_bear_arr = bos_edge(c, last_sl)  # bear: close < last_sl edge

    # Bear BOS: close < last_sl
    def bos_bear_edge(close_arr, level_arr):
        flags = np.zeros(len(close_arr), dtype=int)
        for i in range(1, len(close_arr)):
            if (not np.isnan(level_arr[i]) and not np.isnan(level_arr[i-1])
                    and close_arr[i] < level_arr[i]
                    and close_arr[i-1] >= level_arr[i-1]):
                flags[i] = 1
        out = np.zeros(len(close_arr), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                out[i:i+3] = 1
        return out

    bos_bear_arr = bos_bear_edge(c, last_sl)

    return {
        'price':    float(c[-1]),
        'open':     float(o[-1]),
        'e9':       float(e9[-1]),
        'e21':      float(e21[-1]),
        'e50':      float(e50[-1]),
        'e200':     float(e200[-1]),
        'rsi':      rsi,
        'adx':      adx,
        'atr_pct':  atr_pct,
        'vratio':   vratio,
        'body':     body,
        'body_pct': body_pct,
        'ob_imb':   ob_imb,
        'delta5':   delta5,
        'liq_s':    liq_s,
        'liq_l':    liq_l,
        'vwap':     vwap,
        'bb_pct':   bb_pct,
        'mom3':     mom3,
        'mom8':     mom8,
        'cvd50':    cvd50,
        'pd_pct':   pd_pct,
        'bos_bull': int(bos_bull_arr[-1]),
        'bos_bear': int(bos_bear_arr[-1]),
    }


# ── Skor hesaplama (backtest.py ile özdeş) ────────────────────────
def score(ind: dict) -> tuple[int, int]:
    if ind['atr_pct'] > 2.0: return 0, 0
    if ind['adx']     < 25:  return 0, 0
    if ind['vratio']  < 1.2: return 0, 0

    bos_bull = ind['bos_bull'] == 1
    bos_bear = ind['bos_bear'] == 1
    if not bos_bull and not bos_bear: return 0, 0
    if bos_bull and bos_bear:         return 0, 0

    allow_long  = bos_bull and not bos_bear
    allow_short = bos_bear and not bos_bull

    # EMA50 vs EMA200 yön filtresi
    if allow_long  and ind['e50'] <= ind['e200']: return 0, 0
    if allow_short and ind['e50'] >= ind['e200']: return 0, 0

    # Mum gövdesi doğrulaması
    if allow_long  and (ind['body'] < 0 or ind['body_pct'] < 0.35): return 0, 0
    if allow_short and (ind['body'] > 0 or ind['body_pct'] < 0.35): return 0, 0

    ls = 0; ss = 0; c = ind['price']

    if c > ind['e200']:          ls += 2
    else:                        ss += 2
    if ind['e50'] > ind['e200']: ls += 1
    else:                        ss += 1

    c9 = ind['e9']; c21 = ind['e21']
    if c9 > c21: ls += 1
    else:        ss += 1

    rsi = ind['rsi']
    if 45 < rsi < 68: ls += 1
    if 32 < rsi < 55: ss += 1
    if rsi < 35:      ls += 1
    if rsi > 65:      ss += 1

    ob = ind['ob_imb']
    if ob >  0.20: ls += 1
    if ob < -0.20: ss += 1

    if ind['delta5'] > 0: ls += 1
    if ind['delta5'] < 0: ss += 1

    if ind['liq_s'] > 0.25: ls += 1
    if ind['liq_l'] > 0.25: ss += 1

    if c > ind['vwap']: ls += 1
    else:               ss += 1

    bb = ind['bb_pct']
    if bb < 0.20 and ind['body'] > 0: ls += 1
    if bb > 0.80 and ind['body'] < 0: ss += 1

    if ind['mom3'] > 0 and ind['mom8'] > 0 and c9 > c21: ls += 1
    if ind['mom3'] < 0 and ind['mom8'] < 0 and c9 < c21: ss += 1

    if ind['cvd50'] > 0: ls += 1
    if ind['cvd50'] < 0: ss += 1

    pd_val = ind['pd_pct']
    if pd_val < -10: ls += 1
    if pd_val >  10: ss += 1

    if bos_bull: ls += 1
    if bos_bear: ss += 1

    if allow_long:  return int(ls), 0
    if allow_short: return 0, int(ss)
    return 0, 0


# ── Log yönetimi ──────────────────────────────────────────────────
def load_log() -> dict:
    os.makedirs('logs', exist_ok=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'capital': CAPITAL, 'leverage': LEVERAGE,
        'wins': 0, 'losses': 0, 'total_pnl_pct': 0.0,
        'signals': [], 'open_position': None
    }

def save_log(log: dict):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ── Sanal pozisyon takibi ─────────────────────────────────────────
def check_position(log: dict, candles: list) -> bool:
    pos = log.get('open_position')
    if not pos:
        return False

    hi = max(c['h'] for c in candles[-3:])
    lo = min(c['l'] for c in candles[-3:])
    px = candles[-1]['c']

    side   = pos['side']
    entry  = pos['entry']
    sl     = pos['sl']
    tp     = pos['tp']

    result = None; exit_px = None

    if side == 'long':
        if lo <= sl:   result = 'loss'; exit_px = sl
        elif hi >= tp: result = 'win';  exit_px = tp
    else:
        if hi >= sl:   result = 'loss'; exit_px = sl
        elif lo <= tp: result = 'win';  exit_px = tp

    if result:
        pnl_pct = ((exit_px-entry)/entry*100) if side=='long' else ((entry-exit_px)/entry*100)
        pnl_pct_lev = pnl_pct * LEVERAGE
        capital_change = log['capital'] * (pnl_pct_lev * RISK_PCT / 100 / SL_PCT)
        log['capital'] = round(log['capital'] + capital_change * (1 if result=='win' else -1), 2)
        if result == 'win': log['wins'] += 1
        else:               log['losses'] += 1
        log['total_pnl_pct'] = round(log['total_pnl_pct'] + pnl_pct_lev * RISK_PCT / 100 / SL_PCT * 100, 2)

        icon = 'WIN' if result == 'win' else 'LOSS'
        print(f'  [{icon}] {side.upper()} @{entry:,.0f} -> {exit_px:,.0f} | PnL: %{pnl_pct_lev:+.2f}')

        # Sinyal kaydına sonuç ekle
        for s in log['signals']:
            if s.get('id') == pos.get('signal_id'):
                s['result']  = result
                s['exit_px'] = exit_px
                s['pnl_pct'] = round(pnl_pct_lev, 2)
                break

        log['open_position'] = None
        return True

    # Hâlâ açık
    unr = (px-entry)/entry*100 if side=='long' else (entry-px)/entry*100
    unr_lev = unr * LEVERAGE
    print(f'  [ACIK] {side.upper()} @{entry:,.0f} | Sim: {px:,.0f} | Gerçekleşmemiş: %{unr_lev:+.2f} | Sermaye: ${log["capital"]:,.2f}')
    return False


# ── Tek analiz döngüsü ────────────────────────────────────────────
def run_once(log: dict) -> dict:
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print(f'\n{"="*55}')
    print(f'[KURAL] {now_str}')

    # 1. Veri çek
    candles = fetch_klines(SYMBOL, INTERVAL, 220)
    if len(candles) < 215:
        print('  [HATA] Yeterli mum yok')
        return log

    # 2. Açık pozisyon kontrolü
    if log.get('open_position'):
        check_position(log, candles)

    # 3. Yeni sinyal yok mu?
    if log.get('open_position'):
        print('  [UYARI] Açık pozisyon var, yeni sinyal bekleniyor')
        save_log(log)
        return log

    # 4. Göstergeler
    ind = compute_indicators(candles)
    if not ind:
        print('  [HATA] Gösterge hesplanamadı')
        return log

    # 5. Live orderbook imbalance (daha doğru)
    live_ob = fetch_orderbook(SYMBOL)
    ind['ob_imb'] = live_ob  # canlı değerle override

    # 6. Skor
    ls, ss = score(ind)
    price = ind['price']

    print(f'  Fiyat: {price:,.2f} | EMA9: {ind["e9"]:,.0f} | EMA200: {ind["e200"]:,.0f}')
    print(f'  RSI: {ind["rsi"]:.1f} | ADX: {ind["adx"]:.1f} | ATR%: {ind["atr_pct"]:.3f}')
    print(f'  BOS Bull: {bool(ind["bos_bull"])} | BOS Bear: {bool(ind["bos_bear"])}')
    print(f'  EMA50 vs EMA200: {"BULL" if ind["e50"]>ind["e200"] else "BEAR"}')
    print(f'  OB Imb: {live_ob:+.4f} | CVD50: {ind["cvd50"]:,.0f}')
    print(f'  Skor: LONG={ls} SHORT={ss} (min={MIN_SCORE})')

    side   = None
    reason = 'skor düşük'

    if ls >= MIN_SCORE and ls >= ss + 2:
        side = 'long'; reason = f'LONG skoru {ls}'
    elif ss >= MIN_SCORE and ss >= ls + 2:
        side = 'short'; reason = f'SHORT skoru {ss}'

    signal_id = f'{now_str.replace(" ","_").replace(":","")}'

    signal = {
        'id':       signal_id,
        'ts':       now_str,
        'price':    price,
        'action':   side or 'wait',
        'ls':       ls, 'ss': ss,
        'rsi':      round(ind['rsi'], 1),
        'adx':      round(ind['adx'], 1),
        'atr_pct':  round(ind['atr_pct'], 3),
        'e50_vs_e200': 'bull' if ind['e50'] > ind['e200'] else 'bear',
        'bos_bull': bool(ind['bos_bull']),
        'bos_bear': bool(ind['bos_bear']),
        'ob_imb':   round(live_ob, 4),
        'cvd50':    round(ind['cvd50'], 0),
        'result':   None, 'exit_px': None, 'pnl_pct': None,
        'capital_after': log['capital'],
    }
    log['signals'].append(signal)

    if side:
        sl_price = price * (1 - SL_PCT/100) if side == 'long' else price * (1 + SL_PCT/100)
        tp_price = price * (1 + TP_PCT/100) if side == 'long' else price * (1 - TP_PCT/100)
        log['open_position'] = {
            'signal_id': signal_id,
            'side':  side,
            'entry': price,
            'sl':    round(sl_price, 2),
            'tp':    round(tp_price, 2),
        }
        print(f'\n  >>> SINYAL: {side.upper()} | Giriş: {price:,.2f}')
        print(f'  >>> SL: {sl_price:,.2f} (-{SL_PCT}%) | TP: {tp_price:,.2f} (+{TP_PCT}%)')
        print(f'  >>> Sermaye: ${log["capital"]:,.2f} | Kaldıraç: {LEVERAGE}x')
    else:
        print(f'  >>> BEKLE ({reason})')

    w = log['wins']; l = log['losses']
    wr = w / max(w+l, 1) * 100
    print(f'  Stats: {w}W/{l}L ({wr:.0f}%) | Sermaye: ${log["capital"]:,.2f}')

    save_log(log)
    return log


# ── Ana döngü ─────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int, default=LOOP_MIN)
    p.add_argument('--once',     action='store_true')
    a = p.parse_args()

    print(f"""
╔══════════════════════════════════════╗
║  KURAL TABANLI SİNYAL — rule_signal  ║
║  Sembol : {SYMBOL:<26}║
║  Sermaye: ${CAPITAL:<4.0f} | Kaldıraç: {LEVERAGE}x      ║
║  SL:%{SL_PCT:<4} TP:%{TP_PCT:<4} MinSkor:{MIN_SCORE}         ║
║  Aralık : {a.interval} dakika                  ║
╚══════════════════════════════════════╝
""")

    log = load_log()
    print(f'Önceki kayıt: {len(log["signals"])} sinyal | '
          f'{log["wins"]}W/{log["losses"]}L | Sermaye: ${log["capital"]:,.2f}')

    while True:
        log = run_once(log)

        if a.once:
            print('\n[--once] Çıkılıyor.')
            break

        nxt = datetime.fromtimestamp(
            time.time() + a.interval * 60, tz=timezone.utc
        ).strftime('%H:%M UTC')
        print(f'\n  Sonraki kontrol: {nxt} ({a.interval} dk)')
        time.sleep(a.interval * 60)


if __name__ == '__main__':
    main()
