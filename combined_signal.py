"""
combined_signal.py — Hibrit Sinyal Motoru (KURAL + Fibonacci + Claude AI)
=========================================================================
KURAL skoru sinyal kalitesini ölçer,
Fibonacci S/R seviyeleri SL/TP yerleşimini belirler,
Claude AI son kararı verir.

Çalıştırma:
  python combined_signal.py
  python combined_signal.py --interval 15
  python combined_signal.py --once
  python combined_signal.py --symbol BTC-USDT
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

from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Ayarlar ───────────────────────────────────────────────────────
SYMBOL      = 'BTC-USDT'
INTERVAL    = '15m'
CAPITAL     = 1000.0
LEVERAGE    = 20
RISK_PCT    = 2.0          # Trade başına risk %
MIN_SCORE   = 6            # Combined: EMA filtresi yok → biraz düşük eşik
MIN_CONF    = 60           # Claude güven eşiği
LOG_FILE    = 'logs/combined_signals.json'
LOOP_MIN    = 15
BE_AT_PCT   = 50.0         # TP yolunun %50'sinde breakeven

# Hard gate — Claude çağrılmaz
ADX_MIN     = 25
ATR_MAX_PCT = 2.0
VRATIO_MIN  = 1.2          # Hacim filtresi (sadece yumuşak uyarı, hard değil)

MODEL = "claude-haiku-4-5-20251001"


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
            data = raw.get('data', []) if isinstance(raw, dict) else raw
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


def fetch_orderbook_imbalance(symbol: str) -> float:
    try:
        r = requests.get('https://open-api.bingx.com/openApi/swap/v2/quote/depth',
                         params={'symbol': symbol, 'limit': '20'}, timeout=8)
        d = r.json().get('data', {})
        bv = sum(float(b[1]) for b in d.get('bids', [])[:10])
        av = sum(float(a[1]) for a in d.get('asks', [])[:10])
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

    bv_ = np.where(c >= o, v, v * abs(c-o)/(h-l+1e-10))
    av_ = np.where(c <  o, v, v * abs(c-o)/(h-l+1e-10))
    bvm = np.convolve(bv_, np.ones(10)/10, 'full')[:len(v)]
    avm = np.convolve(av_, np.ones(10)/10, 'full')[:len(v)]
    ob_imb = float((bvm[-1]-avm[-1])/(bvm[-1]+avm[-1]+1e-10))

    delta5  = float(np.sum((bv_-av_)[-5:]))

    LB = 40; px = c[-1]
    rh = h[-LB-1:-1]; rl = l[-LB-1:-1]
    liq_s = float(np.sum((rh > px*1.005) & (rh < px*1.03)) / LB)
    liq_l = float(np.sum((rl < px*0.995) & (rl > px*0.97)) / LB)

    tp = (h + l + c) / 3
    vwap = float(np.sum(tp[-20:]*v[-20:]) / (np.sum(v[-20:]) + 1e-10))

    bm = c[-20:].mean(); bs = c[-20:].std()
    bb_pct = float((c[-1] - (bm - 2*bs)) / (4*bs + 1e-10))

    mom3 = float(c[-1] - c[-4]) if len(c) >= 4 else 0.0
    mom8 = float(c[-1] - c[-9]) if len(c) >= 9 else 0.0

    delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    cvd50 = float(np.sum(delta[-50:]))

    rh50 = h[-50:].max(); rl50 = l[-50:].min()
    pd_pct = float((c[-1] - (rh50+rl50)/2) / (rh50-rl50+1e-10) * 100)

    sh_arr = np.full(len(c), np.nan)
    sl_arr = np.full(len(c), np.nan)
    for i in range(1, len(c)-1):
        if h[i] > h[i-1] and h[i] > h[i+1]: sh_arr[i] = h[i]
        if l[i] < l[i-1] and l[i] < l[i+1]: sl_arr[i] = l[i]

    def ffill(arr):
        out = arr.copy(); last = np.nan
        for i in range(len(out)):
            if not np.isnan(out[i]): last = out[i]
            else: out[i] = last
        return out

    last_sh = ffill(sh_arr)
    last_sl = ffill(sl_arr)

    def bos_bull_edge(close_arr, level_arr):
        flags = np.zeros(len(close_arr), dtype=int)
        for i in range(1, len(close_arr)):
            if (not np.isnan(level_arr[i]) and not np.isnan(level_arr[i-1])
                    and close_arr[i] > level_arr[i]
                    and close_arr[i-1] <= level_arr[i-1]):
                flags[i] = 1
        out = np.zeros(len(close_arr), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1: out[i:i+5] = 1
        return out

    def bos_bear_edge(close_arr, level_arr):
        flags = np.zeros(len(close_arr), dtype=int)
        for i in range(1, len(close_arr)):
            if (not np.isnan(level_arr[i]) and not np.isnan(level_arr[i-1])
                    and close_arr[i] < level_arr[i]
                    and close_arr[i-1] >= level_arr[i-1]):
                flags[i] = 1
        out = np.zeros(len(close_arr), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1: out[i:i+5] = 1
        return out

    bos_bull_arr = bos_bull_edge(c, last_sh)
    bos_bear_arr = bos_bear_edge(c, last_sl)

    return {
        'price':    float(c[-1]),
        'open':     float(o[-1]),
        'highs':    h,
        'lows':     l,
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


# ── Fibonacci Hesaplama ───────────────────────────────────────────
def compute_fibonacci(highs: np.ndarray, lows: np.ndarray, price: float) -> dict:
    """
    Son 100 barın swing high/low'undan Fibonacci seviyeleri hesaplar.
    Hem retracement (destek/direnç) hem de extension seviyeleri döndürür.
    """
    n = min(100, len(highs))
    sh = float(highs[-n:].max())
    sl = float(lows[-n:].min())
    rng = sh - sl

    # Retracement seviyeleri (swing_low=0.0, swing_high=1.0)
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {r: round(sl + r * rng, 2) for r in ratios}

    # Yukarı extension (LONG TP için — swing_high üstü)
    ext_up = {
        1.272: round(sh + 0.272 * rng, 2),
        1.414: round(sh + 0.414 * rng, 2),
        1.618: round(sh + 0.618 * rng, 2),
    }
    # Aşağı extension (SHORT TP için — swing_low altı)
    ext_dn = {
        -0.236: round(sl - 0.236 * rng, 2),
        -0.382: round(sl - 0.382 * rng, 2),
        -0.618: round(sl - 0.618 * rng, 2),
    }

    # Destek (fiyat altındaki en yakın seviye)
    supports    = {r: v for r, v in levels.items() if v < price}
    resistances = {r: v for r, v in levels.items() if v > price}

    nearest_sup = max(supports.values(),    default=sl)
    nearest_res = min(resistances.values(), default=sh)

    # Fib oranı ile etiket
    all_named = {**levels, **ext_up, **ext_dn}
    def fib_label(price_level):
        for r, v in all_named.items():
            if abs(v - price_level) < 1:
                return f'Fib {r:.3f}'
        return 'Fib'

    # Tüm seviyeleri düz liste (prompt için) — aşağıdan yukarıya sıralı
    all_levels = sorted(
        [(r, v, 'ext-dn') for r, v in ext_dn.items()] +
        [(r, v, 'ret')    for r, v in levels.items()] +
        [(r, v, 'ext-up') for r, v in ext_up.items()],
        key=lambda x: x[1]
    )

    return {
        'swing_high':         round(sh, 2),
        'swing_low':          round(sl, 2),
        'range':              round(rng, 2),
        'levels':             levels,
        'ext_up':             ext_up,
        'ext_dn':             ext_dn,
        'nearest_support':    round(nearest_sup, 2),
        'nearest_resistance': round(nearest_res, 2),
        'all_levels':         all_levels,
        'fib_label':          fib_label,
    }


# ── KURAL Skoru (EMA50/200 yön filtresi YUMUŞAK) ─────────────────
def score_combined(ind: dict) -> tuple[int, int, str]:
    """
    Backtest score() ile özdeş ama EMA50/200 yön hard block kaldırıldı.
    Returns (long_score, short_score, block_reason)
    block_reason != '' ise hard gate tetiklendi.
    """
    if ind['atr_pct'] > ATR_MAX_PCT:
        return 0, 0, f'ATR yüksek ({ind["atr_pct"]:.2f}% > {ATR_MAX_PCT}%)'
    if ind['adx'] < ADX_MIN:
        return 0, 0, f'ADX düşük ({ind["adx"]:.1f} < {ADX_MIN})'
    # Hacim: sadece skor etkisi (hard block yok)

    bos_bull = ind['bos_bull'] == 1
    bos_bear = ind['bos_bear'] == 1
    if not bos_bull and not bos_bear:
        return 0, 0, 'BOS yok'
    if bos_bull and bos_bear:
        return 0, 0, 'BOS çakışma'

    allow_long  = bos_bull and not bos_bear
    allow_short = bos_bear and not bos_bull

    # Mum gövdesi doğrulaması (hâlâ hard)
    if allow_long  and (ind['body'] < 0 or ind['body_pct'] < 0.35):
        return 0, 0, 'Mum gövdesi yön uyumsuz (LONG)'
    if allow_short and (ind['body'] > 0 or ind['body_pct'] < 0.35):
        return 0, 0, 'Mum gövdesi yön uyumsuz (SHORT)'

    ls = 0; ss = 0; p = ind['price']

    # EMA200 üstü/altı (artık sadece skor, hard block değil)
    if p > ind['e200']:          ls += 2
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

    if p > ind['vwap']: ls += 1
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

    if allow_long:  return int(ls), 0, ''
    if allow_short: return 0, int(ss), ''
    return 0, 0, ''


# ── Claude API ────────────────────────────────────────────────────
def call_claude(prompt: str, system: str) -> str:
    try:
        import anthropic
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'anthropic', '-q'])
        import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


SYSTEM_PROMPT = """Sen BTC/USDT vadeli işlem uzmanısın. Piyasayı KURAL skoru + Fibonacci seviyeleriyle analiz edip net kararlar verirsin.

FIBONACCI KURALLARI (ÖNEMLİ):
- SL ve TP'yi YÜZDELİK olarak değil, FİBONACCİ FİYAT SEVİYELERİNDE belirle
- SHORT için: SL = minimum Fib 0.382 seviyesi (Fib 0.236 SL olarak YASAK — çok dar), TP = ext-dn seviyeleri (tercihen -0.618)
- LONG  için: SL = maksimum Fib 0.618 seviyesi (Fib 0.786 veya üstü SL olarak YASAK — çok dar), TP = ext-up seviyeleri (tercihen 1.414)
- SL entry'den en az %0.8 uzakta olmalı, TP/SL oranı en az 2:1 olmalı
- Eğer uygun fib seviyesi yoksa BEKLE kararı ver

KURAL SKORU YORUMU:
- 10+ : Çok güçlü sinyal
- 8-9 : Güçlü sinyal
- 6-7 : Orta sinyal (dikkatli ol)
- <6  : Zayıf (genel olarak beklemeyi tercih et)

ÇIKTI FORMATI (tam olarak bu JSON formatında yanıt ver):
{
  "action": "long" | "short" | "wait",
  "confidence": 0-100,
  "sl_price": <fibonacci seviye fiyatı veya null>,
  "tp_price": <fibonacci seviye fiyatı veya null>,
  "sl_fib_ratio": <0.0-1.0 gibi fib oranı veya null>,
  "tp_fib_ratio": <fib oranı veya null>,
  "reasoning": "Kısa Türkçe açıklama (2-3 cümle, fib seviyelerini belirt)",
  "key_factors": ["faktör1", "faktör2", "faktör3"],
  "risks": ["risk1", "risk2"]
}

Başka hiçbir şey yazma. Sadece JSON."""


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


# ── Sanal Pozisyon Takibi ─────────────────────────────────────────
def check_position(log: dict, candles: list) -> bool:
    pos = log.get('open_position')
    if not pos:
        return False

    hi = max(c['h'] for c in candles[-3:])
    lo = min(c['l'] for c in candles[-3:])
    px = candles[-1]['c']

    side  = pos['side']
    entry = pos['entry']
    sl    = pos['sl']
    tp    = pos['tp']
    be_triggered = pos.get('be_triggered', False)

    # Breakeven kontrolü (TP yolunun %BE_AT_PCT'inde SL'yi entry'e taşı)
    if not be_triggered:
        be_trigger_pct = BE_AT_PCT / 100.0
        tp_path = abs(tp - entry)
        if side == 'long':
            be_level = entry + tp_path * be_trigger_pct
            if px >= be_level:
                pos['sl'] = entry
                pos['be_triggered'] = True
                sl = entry
                print(f'  [BE] Breakeven tetiklendi: SL → {entry:,.2f}')
        else:
            be_level = entry - tp_path * be_trigger_pct
            if px <= be_level:
                pos['sl'] = entry
                pos['be_triggered'] = True
                sl = entry
                print(f'  [BE] Breakeven tetiklendi: SL → {entry:,.2f}')

    result = None; exit_px = None
    if side == 'long':
        if lo <= sl:   result = 'loss'; exit_px = sl
        elif hi >= tp: result = 'win';  exit_px = tp
    else:
        if hi >= sl:   result = 'loss'; exit_px = sl
        elif lo <= tp: result = 'win';  exit_px = tp

    if result:
        pnl_pct     = ((exit_px-entry)/entry*100) if side=='long' else ((entry-exit_px)/entry*100)
        sl_dist_pct = abs(entry - pos.get('sl_original', sl)) / entry * 100
        if sl_dist_pct < 0.01: sl_dist_pct = 1.0  # fallback
        risk_ratio  = RISK_PCT / sl_dist_pct
        pnl_lev     = pnl_pct * LEVERAGE
        capital_change = log['capital'] * abs(pnl_lev) / 100 * risk_ratio
        log['capital'] = round(log['capital'] + capital_change * (1 if result=='win' else -1), 2)
        if result == 'win': log['wins'] += 1
        else:               log['losses'] += 1
        log['total_pnl_pct'] = round(log['total_pnl_pct'] + pnl_lev * risk_ratio, 4)

        icon = 'WIN' if result == 'win' else 'LOSS'
        print(f'  [{icon}] {side.upper()} @{entry:,.0f} → {exit_px:,.0f} | PnL: %{pnl_lev:+.2f} (levered)')

        for s in log['signals']:
            if s.get('id') == pos.get('signal_id'):
                s['result']  = result
                s['exit_px'] = exit_px
                s['pnl_pct'] = round(pnl_lev, 2)
                break

        log['open_position'] = None
        return True

    unr = (px-entry)/entry*100 if side=='long' else (entry-px)/entry*100
    print(f'  [AÇIK] {side.upper()} @{entry:,.0f} | Şimdi: {px:,.0f} | '
          f'Gerçekleşmemiş: %{unr*LEVERAGE:+.2f} | SL: {sl:,.0f} | BE: {be_triggered}')
    return False


# ── Tek analiz döngüsü ────────────────────────────────────────────
def run_once(log: dict, symbol: str, tf: str = INTERVAL) -> dict:
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print(f'\n{"="*60}')
    print(f'[HİBRİT] {now_str} | {symbol} | {tf}')

    # 1. Veri çek
    candles = fetch_klines(symbol, tf, 220)
    if len(candles) < 215:
        print('  [HATA] Yeterli mum yok')
        return log

    # 2. Açık pozisyon kontrolü
    if log.get('open_position'):
        check_position(log, candles)

    if log.get('open_position'):
        print('  [BİLGİ] Açık pozisyon var, yeni sinyal bekleniyor')
        save_log(log)
        return log

    # 3. Göstergeler
    ind = compute_indicators(candles)
    if not ind:
        print('  [HATA] Gösterge hesplanamadı')
        return log

    # 4. Canlı orderbook imbalance
    live_ob = fetch_orderbook_imbalance(symbol)
    ind['ob_imb'] = live_ob

    # 5. KURAL skoru (hard gate dahil)
    ls, ss, block_reason = score_combined(ind)
    price = ind['price']

    ema_yön = 'BULL' if ind['e50'] > ind['e200'] else 'BEAR'
    print(f'  Fiyat: {price:,.2f} | EMA50/200: {ema_yön} | RSI: {ind["rsi"]:.1f}')
    print(f'  ADX: {ind["adx"]:.1f} | ATR%: {ind["atr_pct"]:.3f} | Hacim: {ind["vratio"]:.2f}x')
    print(f'  BOS Bull: {bool(ind["bos_bull"])} | BOS Bear: {bool(ind["bos_bear"])}')
    print(f'  OB: {live_ob:+.4f} | CVD50: {ind["cvd50"]:+,.0f} | PD%: {ind["pd_pct"]:+.1f}')
    print(f'  KURAL Skor: LONG={ls} SHORT={ss} (min={MIN_SCORE})')

    signal_id = now_str.replace(' ', '_').replace(':', '')

    # Hard gate tetiklendiyse → BEKLE (Claude çağrılmaz)
    if block_reason:
        print(f'  [HARD GATE] {block_reason} → BEKLE (Claude çağrılmadı)')
        signal = {
            'id': signal_id, 'ts': now_str, 'price': price,
            'action': 'wait', 'reason': block_reason,
            'ls': ls, 'ss': ss,
            'rsi': round(ind['rsi'], 1), 'adx': round(ind['adx'], 1),
            'atr_pct': round(ind['atr_pct'], 3),
            'ema_yön': ema_yön,
            'bos_bull': bool(ind['bos_bull']), 'bos_bear': bool(ind['bos_bear']),
            'ob_imb': round(live_ob, 4), 'cvd50': round(ind['cvd50'], 0),
            'claude_called': False,
            'result': None, 'exit_px': None, 'pnl_pct': None,
            'capital_after': log['capital'],
        }
        log['signals'].append(signal)
        save_log(log)
        w = log['wins']; l = log['losses']
        print(f'  Stats: {w}W/{l}L | Sermaye: ${log["capital"]:,.2f}')
        return log

    # Skor eşiği kontrolü
    dominant_score = max(ls, ss)
    if dominant_score < MIN_SCORE:
        print(f'  [SKOR] {dominant_score} < {MIN_SCORE} → BEKLE (Claude çağrılmadı)')
        signal = {
            'id': signal_id, 'ts': now_str, 'price': price,
            'action': 'wait', 'reason': f'Skor düşük ({dominant_score})',
            'ls': ls, 'ss': ss,
            'rsi': round(ind['rsi'], 1), 'adx': round(ind['adx'], 1),
            'atr_pct': round(ind['atr_pct'], 3),
            'ema_yön': ema_yön,
            'bos_bull': bool(ind['bos_bull']), 'bos_bear': bool(ind['bos_bear']),
            'ob_imb': round(live_ob, 4), 'cvd50': round(ind['cvd50'], 0),
            'claude_called': False,
            'result': None, 'exit_px': None, 'pnl_pct': None,
            'capital_after': log['capital'],
        }
        log['signals'].append(signal)
        save_log(log)
        w = log['wins']; l = log['losses']
        print(f'  Stats: {w}W/{l}L | Sermaye: ${log["capital"]:,.2f}')
        return log

    # 6. Fibonacci hesapla
    fib = compute_fibonacci(ind['highs'], ind['lows'], price)

    print(f'\n  [FİB] Swing: {fib["swing_low"]:,.0f} → {fib["swing_high"]:,.0f} (range: {fib["range"]:,.0f})')
    for r, v, typ in fib['all_levels']:
        marker = ' ◄ FİYAT' if abs(v - price) < fib['range'] * 0.03 else ''
        sup_res = 'destek' if v < price else 'direnç'
        print(f'    Fib {r:.3f} [{typ}]: {v:,.2f} ({sup_res}){marker}')
    print(f'  En yakın destek: {fib["nearest_support"]:,.2f} | Direnç: {fib["nearest_resistance"]:,.2f}')

    # 7. Claude için prompt
    direction_hint = 'LONG yönlü' if ls > 0 else 'SHORT yönlü'
    fib_lines = '\n'.join(
        f'  Fib {r:+.3f} [{typ}]: {v:,.2f} — {"SHORT TP hedefi" if typ=="ext-dn" else ("LONG TP hedefi" if typ=="ext-up" else ("destek" if v < price else "direnç"))}'
        for r, v, typ in fib['all_levels']
    )

    prompt = f"""
ZAMAN: {now_str}
SEMBOL: {symbol} | ARALIK: 15dk

=== KURAL SKORU ===
LONG skoru : {ls}/13 | SHORT skoru: {ss}/13
Dominant yön: {direction_hint} (min eşik: {MIN_SCORE})
EMA50/200 yönü: {ema_yön} ({"LONG için avantajlı" if ema_yön=="BULL" else "SHORT için avantajlı"})
BOS Bullish: {bool(ind['bos_bull'])} | BOS Bearish: {bool(ind['bos_bear'])}

=== TEKNİK GÖSTERGELER ===
Fiyat: ${price:,.2f}
EMA9/21/50/200: {ind['e9']:,.0f} / {ind['e21']:,.0f} / {ind['e50']:,.0f} / {ind['e200']:,.0f}
RSI(14): {ind['rsi']:.1f}
ADX: {ind['adx']:.1f} | ATR%: {ind['atr_pct']:.3f}
Hacim oranı: {ind['vratio']:.2f}x
OB İmbalans: {live_ob:+.4f}
CVD50: {ind['cvd50']:+,.0f}
Momentum 3b: {ind['mom3']:+.1f} | 8b: {ind['mom8']:+.1f}
VWAP: {ind['vwap']:,.0f} | Fiyat üstünde: {price > ind['vwap']}
Bollinger%: {ind['bb_pct']:.3f} (0=alt band, 1=üst band)
Premium/Discount: %{ind['pd_pct']:+.1f}
Liq Short cluster: {ind['liq_s']:.2f} | Liq Long cluster: {ind['liq_l']:.2f}

=== FİBONACCİ SEVİYELERİ (Son 100 bar) ===
Swing Low: {fib['swing_low']:,.2f} | Swing High: {fib['swing_high']:,.2f}
Mevcut Fiyat: {price:,.2f}

{fib_lines}

En yakın DESTEK : {fib['nearest_support']:,.2f}
En yakın DİRENÇ : {fib['nearest_resistance']:,.2f}
SHORT TP seçenekleri (swing_low altı): {', '.join(f'Fib {r:+.3f}={v:,.0f}' for r,v,_ in fib['all_levels'] if _ == 'ext-dn')}
LONG  TP seçenekleri (swing_high üstü): {', '.join(f'Fib {r:+.3f}={v:,.0f}' for r,v,_ in fib['all_levels'] if _ == 'ext-up')}

GÖREV:
1. Yukarıdaki KURAL skoru ve Fibonacci seviyelerini değerlendir
2. Eğer işlem açacaksan, SL ve TP'yi FİBONACCİ FİYAT SEVİYELERİNDE belirle
3. SHORT için:
   - SL = MİNİMUM Fib 0.382 seviyesi (Fib 0.236 YASAK — çok dar, noise içinde kalır)
   - TP = ext-dn seviyeleri, TERCİHEN Fib -0.618 (derin extension, R:R≈3:1 hedefle)
4. LONG için:
   - SL = MAKSİMUM Fib 0.618 seviyesi (Fib 0.786 ve üstü YASAK — çok dar)
   - TP = ext-up seviyeleri, TERCİHEN Fib 1.414 (R:R≈3:1 hedefle)
5. SL entry'den en az %0.8 uzakta olmalı — daha yakınsa BEKLE
6. TP/SL oranı en az 2:1 olmadıkça BEKLE tercih et

Şimdi analiz yap ve karar ver:"""

    print(f'\n  [CLAUDE] Analiz yapılıyor...')
    try:
        response = call_claude(prompt, SYSTEM_PROMPT)
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        decision = json.loads(raw.strip())
    except Exception as e:
        print(f'  [HATA] Claude yanıtı parse edilemedi: {e}')
        save_log(log)
        return log

    action     = decision.get('action', 'wait')
    confidence = decision.get('confidence', 0)
    sl_price   = decision.get('sl_price')
    tp_price   = decision.get('tp_price')
    reasoning  = decision.get('reasoning', '')
    sl_fib     = decision.get('sl_fib_ratio', '?')
    tp_fib     = decision.get('tp_fib_ratio', '?')

    print(f'  [CLAUDE] Karar: {action.upper()} | Güven: %{confidence}')
    print(f'  Gerekçe: {reasoning}')
    if sl_price: print(f'  SL: {sl_price:,.2f} (Fib {sl_fib}) | TP: {tp_price:,.2f} (Fib {tp_fib})')

    # Güven eşiği
    if action != 'wait' and confidence < MIN_CONF:
        print(f'  [DÜŞÜK GÜVEN] %{confidence} < %{MIN_CONF} → BEKLE')
        action = 'wait'

    # SL/TP geçerlilik kontrolü
    if action in ('long', 'short') and (not sl_price or not tp_price):
        print('  [HATA] SL veya TP fiyatı eksik → BEKLE')
        action = 'wait'

    if action in ('long', 'short'):
        # Minimum R:R kontrolü
        if action == 'long':
            sl_dist = price - sl_price
            tp_dist = tp_price - price
        else:
            sl_dist = sl_price - price
            tp_dist = price - tp_price

        if sl_dist <= 0 or tp_dist <= 0:
            print(f'  [HATA] SL/TP yönü yanlış → BEKLE')
            action = 'wait'
        elif sl_dist / price < 0.008:
            print(f'  [SL DAR] SL mesafesi %{sl_dist/price*100:.2f} < %0.8 → BEKLE')
            action = 'wait'
        elif tp_dist / (sl_dist + 1e-10) < 1.5:
            print(f'  [R:R] {tp_dist/sl_dist:.2f}:1 < 1.5:1 → BEKLE')
            action = 'wait'

    signal = {
        'id':          signal_id,
        'ts':          now_str,
        'price':       price,
        'action':      action,
        'reason':      reasoning if action != 'wait' else decision.get('reasoning', ''),
        'ls':          ls, 'ss': ss,
        'rsi':         round(ind['rsi'], 1),
        'adx':         round(ind['adx'], 1),
        'atr_pct':     round(ind['atr_pct'], 3),
        'ema_yön':     ema_yön,
        'bos_bull':    bool(ind['bos_bull']),
        'bos_bear':    bool(ind['bos_bear']),
        'ob_imb':      round(live_ob, 4),
        'cvd50':       round(ind['cvd50'], 0),
        'confidence':  confidence,
        'sl_price':    sl_price,
        'tp_price':    tp_price,
        'sl_fib':      sl_fib,
        'tp_fib':      tp_fib,
        'fib_swing_high': fib['swing_high'],
        'fib_swing_low':  fib['swing_low'],
        'claude_called': True,
        'result':      None, 'exit_px': None, 'pnl_pct': None,
        'capital_after': log['capital'],
    }
    log['signals'].append(signal)

    if action in ('long', 'short'):
        log['open_position'] = {
            'signal_id':   signal_id,
            'side':        action,
            'entry':       price,
            'sl':          round(sl_price, 2),
            'sl_original': round(sl_price, 2),
            'tp':          round(tp_price, 2),
            'be_triggered': False,
        }
        rr = abs(tp_price - price) / abs(sl_price - price)
        print(f'\n  >>> SİNYAL: {action.upper()} | Giriş: {price:,.2f}')
        print(f'  >>> SL: {sl_price:,.2f} (Fib {sl_fib}) | TP: {tp_price:,.2f} (Fib {tp_fib})')
        print(f'  >>> R:R = {rr:.2f}:1 | Sermaye: ${log["capital"]:,.2f} | {LEVERAGE}x')
    else:
        print(f'  >>> BEKLE')

    w = log['wins']; l = log['losses']
    wr = w / max(w+l, 1) * 100
    print(f'  Stats: {w}W/{l}L ({wr:.0f}%) | Sermaye: ${log["capital"]:,.2f}')

    save_log(log)
    return log


# ── Ana döngü ─────────────────────────────────────────────────────
def main():
    global ADX_MIN
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int,   default=LOOP_MIN)
    p.add_argument('--once',     action='store_true')
    p.add_argument('--symbol',   default=SYMBOL)
    p.add_argument('--tf',       default=INTERVAL, help='Mum zaman dilimi: 1m,3m,5m,15m,1h...')
    p.add_argument('--adx-min',  type=float, default=ADX_MIN, help='ADX hard gate esigi')
    a = p.parse_args()
    ADX_MIN = a.adx_min

    print(f"""
╔══════════════════════════════════════════════╗
║  HİBRİT SİNYAL — combined_signal.py          ║
║  KURAL Skoru + Fibonacci S/R + Claude AI      ║
║  Sembol : {a.symbol:<34}║
║  TF: {a.tf:<5} | Sermaye: ${CAPITAL:<4.0f} | {LEVERAGE}x          ║
║  Hard Gate: ADX<{ADX_MIN} veya ATR>{ATR_MAX_PCT}%         ║
║  Min Skor: {MIN_SCORE} | Min Güven: %{MIN_CONF}             ║
║  Aralık  : {a.interval} dakika                       ║
╚══════════════════════════════════════════════╝
""")


    if not ANTHROPIC_API_KEY:
        print('[HATA] ANTHROPIC_API_KEY bulunamadı! .env dosyasını kontrol et.')
        sys.exit(1)

    log = load_log()
    print(f'Önceki kayıt: {len(log["signals"])} sinyal | '
          f'{log["wins"]}W/{log["losses"]}L | Sermaye: ${log["capital"]:,.2f}')

    while True:
        log = run_once(log, a.symbol, a.tf)

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
