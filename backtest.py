"""
backtest.py v6 — BingX Scalper
Özellikler:
  - Sabit %SL/%TP (ulaşılabilir hedefler)
  - Çok daha az ama kaliteli sinyal (score >= 7/10)
  - Basit trailing: TP'nin %50'sine ulaşınca SL = breakeven
  - Orderbook hacim imbalance simülasyonu
  - Likidasyon cluster simülasyonu
  - Erken çıkış: karda güçlü ters sinyal
  - Tüm numpy tipleri JSON-safe
"""
from __future__ import annotations
import argparse, json, math, os, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class Cfg:
    symbol:         str   = "BTC-USDT"
    interval_min:   int   = 15
    days:           int   = 90
    capital:        float = 1000.0
    leverage:       int   = 15
    risk_pct:       float = 2.0       # Trade başına max risk
    sl_pct:         float = 0.40      # Stop loss %
    tp_pct:         float = 1.20      # Take profit % (R:R = 3:1)
    be_at_pct:      float = 50.0      # TP'nin bu %'sine ulaşınca SL = breakeven (50 = %50)
    commission:     float = 0.05      # BingX taker
    slippage:       float = 0.02
    min_score:      int   = 7         # 10 üzerinden (az ama kaliteli sinyal)
    min_adx:        float = 22.0
    max_atr_pct:    float = 2.0
    early_pct:      float = 0.30      # Bu kârdan sonra ters sinyal = çık

IVMAP = {1:"1m",3:"3m",5:"5m",15:"15m",30:"30m",60:"1h",240:"4h",1440:"1d"}

# ── Veri Çekme ────────────────────────────────────────────────────
def fetch(symbol: str, interval_min: int, days: int) -> pd.DataFrame:
    try: import requests
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable,"-m","pip","install","requests","-q"])
        import requests

    iv   = IVMAP.get(interval_min, "15m")
    step = interval_min * 60 * 1000
    end  = int(time.time() * 1000)
    start= end - days * 86400 * 1000
    LIMIT= 1000

    print(f"\n[VERİ] {symbol} {iv} {days}gün çekiliyor...")
    urls = [
        "https://open-api.bingx.com/openApi/swap/v3/quote/klines",
        "https://open-api.bingx.com/openApi/swap/v2/quote/klines",
    ]

    rows = []
    for url in urls:
        rows=[]; cursor=start; fails=0
        while cursor < end:
            wend = min(cursor + LIMIT*step, end)
            try:
                r   = requests.get(url, params={
                    "symbol":symbol,"interval":iv,
                    "startTime":str(cursor),"endTime":str(wend),"limit":str(LIMIT)
                }, timeout=20)
                raw = r.json()
            except Exception as e:
                fails+=1
                if fails>=3: break
                time.sleep(2); continue

            if isinstance(raw,dict):
                if raw.get("code",0) not in (0,None,""):
                    fails+=1
                    if fails>=3: break
                    cursor=wend+step; continue
                candles=raw.get("data",[])
            elif isinstance(raw,list): candles=raw
            else: cursor=wend+step; continue

            if not candles: cursor=wend+step; continue

            for c in candles:
                try:
                    if isinstance(c,dict):
                        ts=int(c.get("time",c.get("openTime",c.get("t",0))))
                        row=[ts,float(c.get("open",0)),float(c.get("high",0)),
                             float(c.get("low",0)),float(c.get("close",0)),
                             float(c.get("volume",1))]
                    elif isinstance(c,list) and len(c)>=5:
                        row=[int(c[0]),float(c[1]),float(c[2]),
                             float(c[3]),float(c[4]),
                             float(c[5]) if len(c)>5 else 1.0]
                    else: continue
                    if row[4]>0: rows.append(row)
                except: continue

            if not rows: cursor=wend+step; continue
            last=max(r[0] for r in rows[-len(candles):])
            cursor=last+step
            time.sleep(0.12)

        if rows:
            print(f"  OK: {len(rows)} mum")
            break

    if not rows:
        print("  [HATA] Veri alınamadı!")
        return pd.DataFrame()

    df=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"])
    df["ts"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    df=df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df=df[df["close"]>0]
    print(f"  {len(df)} mum | {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")
    return df

# ── Göstergeler ───────────────────────────────────────────────────
def _ema(a,p):
    e=np.empty(len(a)); e[0]=a[0]; k=2/(p+1)
    for i in range(1,len(a)): e[i]=k*a[i]+(1-k)*e[i-1]
    return e

def _rsi(a,p=14):
    r=np.full(len(a),50.0); d=np.diff(a)
    g=np.where(d>0,d,0.); l=np.where(d<0,-d,0.)
    ag=g[:p].mean(); al=l[:p].mean()
    for i in range(p,len(a)-1):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+l[i])/p
        r[i+1]=100-100/(1+ag/(al+1e-10))
    return r

def _atr(h,l,c,p=14):
    a=np.zeros(len(c))
    for i in range(1,len(c)):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        a[i]=(a[i-1]*(p-1)+tr)/p if i>=p else tr
    return a

def _adx(h,l,c,p=14):
    pd_=np.zeros(len(c)); md_=np.zeros(len(c)); tr=np.zeros(len(c))
    for i in range(1,len(c)):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        u=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pd_[i]=u if u>dn and u>0 else 0
        md_[i]=dn if dn>u and dn>0 else 0
    dx=np.zeros(len(c))
    for i in range(p,len(c)):
        at=tr[i-p:i].mean()+1e-10
        pv=pd_[i-p:i].sum()/at*100; mv=md_[i-p:i].sum()/at*100
        dx[i]=abs(pv-mv)/(pv+mv+1e-10)*100
    s=np.zeros(len(c)); s[p]=dx[p]
    for i in range(p+1,len(c)): s[i]=(s[i-1]*(p-1)+dx[i])/p
    return s

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    c=df.close.values; h=df.high.values
    l=df.low.values;   v=df.volume.values

    df["e9"]  =_ema(c,9);   df["e21"]=_ema(c,21)
    df["e50"] =_ema(c,50);  df["e200"]=_ema(c,200)
    df["rsi"] =_rsi(c,14);  df["atr"] =_atr(h,l,c,14)
    df["adx"] =_adx(h,l,c,14)
    df["atr_pct"]=df["atr"]/df["close"]*100

    # Hacim
    vm=pd.Series(v).rolling(20,min_periods=1).mean().values
    df["vratio"]=v/(vm+1e-10)

    # Mum gövdesi
    rng=h-l+1e-10
    df["body"]=c-df.open.values
    df["body_pct"]=abs(c-df.open.values)/rng

    # Orderbook proxy: kayan hacim imbalance (10 bar)
    bv=np.where(c>=df.open.values, v, v*abs(c-df.open.values)/rng)
    av=np.where(c< df.open.values, v, v*abs(c-df.open.values)/rng)
    bvm=pd.Series(bv).rolling(10,min_periods=1).mean().values
    avm=pd.Series(av).rolling(10,min_periods=1).mean().values
    df["ob_imb"]=(bvm-avm)/(bvm+avm+1e-10)

    # Kümülatif delta (son 5 bar)
    df["delta5"]=pd.Series(bv-av).rolling(5,min_periods=1).sum()

    # Likidasyon cluster: yakın swing yoğunluğu (40 bar)
    LB=40; ls=np.zeros(len(c)); ll=np.zeros(len(c))
    for i in range(LB,len(c)):
        p=c[i]
        rh=h[i-LB:i]; rl=l[i-LB:i]
        ls[i]=np.sum((rh>p*1.005)&(rh<p*1.03))/LB
        ll[i]=np.sum((rl<p*0.995)&(rl>p*0.97))/LB
    df["liq_s"]=ls; df["liq_l"]=ll

    # Bollinger (20,2)
    bm=pd.Series(c).rolling(20,min_periods=1).mean()
    bs=pd.Series(c).rolling(20,min_periods=1).std().fillna(0)
    df["bb_pct"]=(pd.Series(c)-(bm-2*bs))/(4*bs+1e-10)

    # VWAP kayan (20)
    df["vwap"]=(pd.Series(c*v).rolling(20,min_periods=1).sum()/
                pd.Series(v).rolling(20,min_periods=1).sum())

    # Momentum
    df["mom3"]=pd.Series(c).diff(3).fillna(0)
    df["mom8"]=pd.Series(c).diff(8).fillna(0)

    # ── SMC Eklentileri ───────────────────────────────────────────
    # CVD kümülatif (50 bar) — kurumsal yön
    df["cvd50"]=pd.Series(bv-av).rolling(50,min_periods=1).sum()

    # Premium / Discount (50-bar range) — aşırı uzaklaşma filtresi
    rh50=pd.Series(h).rolling(50,min_periods=1).max()
    rl50=pd.Series(l).rolling(50,min_periods=1).min()
    df["pd_pct"]=(pd.Series(c)-(rh50+rl50)/2)/(rh50-rl50+1e-10)*100

    # Swing High/Low (3-bar pivot, lookahead yok — i-1 konumunda pivot)
    sh_arr=np.full(len(c),np.nan); sl_arr=np.full(len(c),np.nan)
    for i in range(1,len(c)-1):
        if h[i]>h[i-1] and h[i]>h[i+1]: sh_arr[i]=h[i]
        if l[i]<l[i-1] and l[i]<l[i+1]: sl_arr[i]=l[i]
    df["last_sh"]=pd.Series(sh_arr).ffill()
    df["last_sl"]=pd.Series(sl_arr).ffill()

    # BOS: sadece kırılma ANINDA tetikle (edge-triggered, sürekli değil)
    # Bir önceki barda fiyat swing seviyesinin ötesinde DEĞİLDİ, bu barda geçti
    close_s  = pd.Series(c)
    sh_s     = df["last_sh"]
    sl_s     = df["last_sl"]
    df["bos_bull"] = ((close_s > sh_s) & (close_s.shift(1) <= sh_s.shift(1))).astype(int)
    df["bos_bear"] = ((close_s < sl_s) & (close_s.shift(1) >= sl_s.shift(1))).astype(int)
    # 3 bar boyunca geçerliliğini koru (kırılma sonrası giriş fırsatı)
    df["bos_bull"] = df["bos_bull"].rolling(3, min_periods=1).max().astype(int)
    df["bos_bear"] = df["bos_bear"].rolling(3, min_periods=1).max().astype(int)

    return df

# ── Sinyal Skoru (0-10) ───────────────────────────────────────────
def score(df: pd.DataFrame, i: int) -> Tuple[int,int]:
    if i < 215: return 0,0
    r=df.iloc[i]; p1=df.iloc[i-1]; p2=df.iloc[i-2]

    # Geçmez filtreler
    if float(r.atr_pct) > 2.0: return 0,0   # Aşırı volatil
    if float(r.adx)     < 25:  return 0,0   # Trend yok (eşik yükseltildi)
    if float(r.vratio)  < 1.2: return 0,0   # Hacim onayı yok

    # SMC Hard Filtre: BOS yönü = işlem yönü
    bos_bull = int(r.bos_bull) == 1
    bos_bear = int(r.bos_bear) == 1
    if not bos_bull and not bos_bear: return 0,0
    # BOS yönüne ters skoru sıfırla — sadece yapı yönüne gir
    allow_long  = bos_bull and not bos_bear
    allow_short = bos_bear and not bos_bull
    # Her iki BOS aynı anda true ise (geçiş anı) işlem yok
    if bos_bull and bos_bear: return 0,0

    # EMA yapısı onayı: LONG için EMA50>EMA200 (bull trend), SHORT için EMA50<EMA200 (bear trend)
    e50_v = float(r.e50); e200_v = float(r.e200)
    if allow_long  and e50_v <= e200_v: return 0,0  # Bear trend'de LONG yok
    if allow_short and e50_v >= e200_v: return 0,0  # Bull trend'de SHORT yok

    # Mum gövdesi doğrulaması: yönle uyumlu güçlü mum zorunlu
    body      = float(r.close) - float(r.open)
    body_pct  = float(r.body_pct)
    if allow_long  and (body < 0 or body_pct < 0.35): return 0,0  # Yeşil + güçlü gövde
    if allow_short and (body > 0 or body_pct < 0.35): return 0,0  # Kırmızı + güçlü gövde

    ls=0; ss=0; c=float(r.close)

    # 1. EMA200 ana trend (2 puan — en önemli)
    if c > float(r.e200): ls+=2
    else:                  ss+=2

    # 2. EMA50 trend uyumu
    if float(r.e50) > float(r.e200): ls+=1
    else:                              ss+=1

    # 3. EMA9/21 kesişim veya uyum
    c9=float(r.e9); c21=float(r.e21)
    p9=float(p1.e9); p21=float(p1.e21)
    if   c9>c21 and p9<=p21: ls+=2   # Taze yukarı kesişim
    elif c9>c21:              ls+=1   # Yukarı trend devam
    if   c9<c21 and p9>=p21: ss+=2   # Taze aşağı kesişim
    elif c9<c21:              ss+=1   # Aşağı trend devam

    # 4. RSI bölgesi
    rsi=float(r.rsi)
    if 45 < rsi < 68: ls+=1   # Momentum bölgesi, aşırı alım değil
    if 32 < rsi < 55: ss+=1
    if rsi < 35:       ls+=1   # Aşırı satım dönüşü
    if rsi > 65:       ss+=1   # Aşırı alım dönüşü

    # 5. Orderbook imbalance
    ob=float(r.ob_imb)
    if ob > 0.20: ls+=1    # Alıcı baskısı
    if ob < -0.20: ss+=1   # Satıcı baskısı

    # 6. Kümülatif delta
    if float(r.delta5) > 0: ls+=1
    if float(r.delta5) < 0: ss+=1

    # 7. Likidasyon cluster
    if float(r.liq_s) > 0.25: ls+=1  # Yukarıda short liq = pump
    if float(r.liq_l) > 0.25: ss+=1  # Aşağıda long liq = dump

    # 8. VWAP
    if c > float(r.vwap): ls+=1
    else:                   ss+=1

    # 9. Bollinger
    bb=float(r.bb_pct)
    if bb < 0.20 and float(r.body)>0: ls+=1  # Alt bant destek + yeşil mum
    if bb > 0.80 and float(r.body)<0: ss+=1  # Üst bant direnç + kırmızı mum

    # 10. Momentum uyumu
    if float(r.mom3)>0 and float(r.mom8)>0 and c9>c21: ls+=1
    if float(r.mom3)<0 and float(r.mom8)<0 and c9<c21: ss+=1

    # 11. CVD kümülatif (50 bar) — kurumsal akış
    if float(r.cvd50) > 0: ls+=1
    if float(r.cvd50) < 0: ss+=1

    # 12. Premium / Discount — sadece ucuzda long, pahalıda short
    pd_val=float(r.pd_pct)
    if pd_val < -10: ls+=1   # Discount zone: long için ideal
    if pd_val >  10: ss+=1   # Premium zone: short için ideal

    # 13. BOS (Break of Structure) — trend yön onayı
    if int(r.bos_bull): ls+=1
    if int(r.bos_bear): ss+=1

    # BOS yönü dışındaki skoru sıfırla
    if allow_long:  return int(ls), 0
    if allow_short: return 0, int(ss)
    return 0, 0

# ── Pozisyon Veri Sınıfı ─────────────────────────────────────────
@dataclass
class Pos:
    side:str; entry:float; sl:float; tp:float
    sl0:float; tp0:float; size:float; idx:int; ts:str
    ls:int=0; ss:int=0; be_set:bool=False

@dataclass
class Trade:
    side:str; entry:float; ex:float
    sl0:float; tp0:float
    pnl:float; pnl_pct:float
    reason:str; bars:int
    ts_open:str; ts_close:str
    ls:int=0; ss:int=0
    @property
    def win(self): return self.pnl > 0

# ── Ana Döngü ─────────────────────────────────────────────────────
def run_bt(df: pd.DataFrame, cfg: Cfg) -> dict:
    capital=cfg.capital; trades=[]; equity=[]; monthly={}
    pos: Optional[Pos] = None
    cooldown = 0  # SL sonrası bekleme sayacı (bar)
    COOLDOWN_BARS = 4  # SL'den sonra 4 bar bekle

    sl_dist_pct  = cfg.sl_pct / 100
    tp_dist_pct  = cfg.tp_pct / 100
    be_trigger   = cfg.be_at_pct / 100  # TP yolunun bu kadarına ulaşınca BE

    for i in range(len(df)):
        r=df.iloc[i]
        px=float(r.close); hi=float(r.high); lo=float(r.low)
        ts=str(r.ts)[:16]; mk=str(r.ts)[:7]

        equity.append(round(capital,4))
        if mk not in monthly: monthly[mk]={"s":capital,"e":capital,"n":0}

        # ── Açık pozisyon yönetimi ────────────────────────────────
        if pos:
            ex=None; reason=""

            if pos.side=="long":
                # Breakeven: TP yolunun %40'ına ulaşınca SL = entry
                if not pos.be_set:
                    be_price = pos.entry + (pos.tp - pos.entry)*be_trigger
                    if hi >= be_price:
                        pos.sl = pos.entry * 1.0001
                        pos.be_set = True

                sl_hit = lo <= pos.sl
                tp_hit = hi >= pos.tp
                if sl_hit and tp_hit:
                    # Aynı barda ikisi de — mum yönüne göre karar ver
                    body = float(r.close) - float(r.open)
                    if body >= 0:  # Yeşil mum: önce TP
                        ex=pos.tp; reason="take_profit"
                    else:          # Kırmızı mum: önce SL
                        ex=pos.sl; reason="stop_loss"
                elif tp_hit: ex=pos.tp; reason="take_profit"
                elif sl_hit: ex=pos.sl; reason="stop_loss"

            else:  # short
                if not pos.be_set:
                    be_price = pos.entry - (pos.entry - pos.tp)*be_trigger
                    if lo <= be_price:
                        pos.sl = pos.entry * 0.9999
                        pos.be_set = True

                sl_hit = hi >= pos.sl
                tp_hit = lo <= pos.tp
                if sl_hit and tp_hit:
                    body = float(r.close) - float(r.open)
                    if body <= 0:  # Kırmızı mum: önce TP (short için iyi)
                        ex=pos.tp; reason="take_profit"
                    else:          # Yeşil mum: önce SL
                        ex=pos.sl; reason="stop_loss"
                elif tp_hit: ex=pos.tp; reason="take_profit"
                elif sl_hit: ex=pos.sl; reason="stop_loss"

            # Erken çıkış: kârda iken güçlü ters sinyal
            if ex is None:
                cur_p = ((px-pos.entry)/pos.entry*100*cfg.leverage
                         if pos.side=="long"
                         else (pos.entry-px)/pos.entry*100*cfg.leverage)
                if cur_p >= cfg.early_pct:
                    ls2,ss2=score(df,i)
                    if pos.side=="long"  and ss2>=cfg.min_score and ss2>ls2+2:
                        ex=px; reason="early_exit"
                    elif pos.side=="short" and ls2>=cfg.min_score and ls2>ss2+2:
                        ex=px; reason="early_exit"

            if ex is not None:
                slip = ex*cfg.slippage/100
                aex  = ex-slip if pos.side=="long" else ex+slip

                # Doğru PnL: size * fiyat farkı
                if pos.side=="long":
                    raw_pnl = pos.size*(aex-pos.entry)
                    pp = (aex-pos.entry)/pos.entry*100*cfg.leverage
                else:
                    raw_pnl = pos.size*(pos.entry-aex)
                    pp = (pos.entry-aex)/pos.entry*100*cfg.leverage

                comm = pos.size*pos.entry*cfg.commission/100*2
                pu   = raw_pnl - comm
                capital = max(capital+pu, 0.01)
                monthly[mk]["n"]+=1; monthly[mk]["e"]=capital

                trades.append(Trade(
                    side=pos.side, entry=round(pos.entry,2), ex=round(aex,2),
                    sl0=round(pos.sl0,2), tp0=round(pos.tp0,2),
                    pnl=round(float(pu),4), pnl_pct=round(float(pp),2),
                    reason=reason, bars=i-pos.idx,
                    ts_open=pos.ts, ts_close=ts,
                    ls=pos.ls, ss=pos.ss,
                ))
                pos=None
                if reason == "stop_loss":
                    cooldown = COOLDOWN_BARS  # SL sonrası cooldown başlat

        # Cooldown: her barda pos yoksa azalt
        if pos is None and cooldown > 0:
            cooldown -= 1

        # ── Yeni sinyal ──────────────────────────────────────────
        if pos is None and cooldown == 0 and capital > 10:
            ls,ss = score(df,i)
            side  = None

            # Daha katı filtre: min_score VE rakipten en az 2 puan önde
            if ls >= cfg.min_score and ls >= ss+2: side="long"
            elif ss >= cfg.min_score and ss >= ls+2: side="short"

            if side:
                slip  = px*cfg.slippage/100
                entry = px+slip if side=="long" else px-slip

                # Sabit % SL/TP
                if side=="long":
                    sl = entry*(1 - sl_dist_pct)
                    tp = entry*(1 + tp_dist_pct)
                else:
                    sl = entry*(1 + sl_dist_pct)
                    tp = entry*(1 - tp_dist_pct)

                # Pozisyon büyüklüğü: risk bazlı
                sl_d = abs(entry-sl)
                ru   = capital*cfg.risk_pct/100
                size = min(ru/sl_d, capital*cfg.leverage/entry)
                size = round(max(size,0.001),6)

                pos = Pos(side=side,entry=entry,sl=sl,tp=tp,
                          sl0=sl,tp0=tp,size=size,idx=i,ts=ts,
                          ls=ls,ss=ss,be_set=False)

        monthly[mk]["e"]=capital

    # Son pozisyonu kapat
    if pos:
        lp=float(df.iloc[-1].close)
        if pos.side=="long":
            rp=pos.size*(lp-pos.entry); pp=(lp-pos.entry)/pos.entry*100*cfg.leverage
        else:
            rp=pos.size*(pos.entry-lp); pp=(pos.entry-lp)/pos.entry*100*cfg.leverage
        comm=pos.size*pos.entry*cfg.commission/100*2
        pu=rp-comm; capital=max(capital+pu,0.01)
        trades.append(Trade(
            side=pos.side,entry=round(pos.entry,2),ex=round(lp,2),
            sl0=round(pos.sl0,2),tp0=round(pos.tp0,2),
            pnl=round(float(pu),4),pnl_pct=round(float(pp),2),
            reason="end",bars=len(df)-1-pos.idx,
            ts_open=pos.ts,ts_close=str(df.iloc[-1].ts)[:16],
            ls=pos.ls,ss=pos.ss,
        ))

    return _stats(trades,equity,monthly,cfg,capital)

# ── İstatistikler ────────────────────────────────────────────────
def _f(v):
    try:
        import numpy as _np
        if isinstance(v,_np.bool_): return bool(v)
        if isinstance(v,_np.integer): return int(v)
        if isinstance(v,_np.floating): return float(v)
    except: pass
    if isinstance(v,float) and (v!=v or abs(v)==float('inf')): return None
    return v

def _stats(trades,equity,monthly,cfg,final):
    n=len(trades)
    if n==0:
        return {"error":"Sinyal üretilemedi. min_score çok yüksek veya ADX filtresi çok kısıtlayıcı."}

    wins=[t for t in trades if t.win]
    loss=[t for t in trades if not t.win]
    pnls=[float(t.pnl) for t in trades]
    ret =(final-cfg.capital)/cfg.capital*100

    eq  =np.array(equity,dtype=float)
    pk  =np.maximum.accumulate(eq)
    dd  =(eq-pk)/(pk+1e-10)*100
    mdd =float(dd.min())

    gp=sum(t.pnl for t in wins if t.pnl>0)
    gl=abs(sum(t.pnl for t in loss if t.pnl<0))
    pf=gp/(gl+1e-10)

    pa =np.array(pnls)
    sh =float(pa.mean()/(pa.std()+1e-10)*math.sqrt(252))
    cal=ret/(abs(mdd)+1e-10)
    aw =float(sum(t.pnl for t in wins)/max(len(wins),1))
    al =float(sum(t.pnl for t in loss)/max(len(loss),1))
    rr =abs(aw/(al+1e-10))

    sw=sl2=mw=ml=0
    for t in trades:
        if t.win: sw+=1;sl2=0;mw=max(mw,sw)
        else:     sl2+=1;sw=0;ml=max(ml,sl2)

    exits={}
    for t in trades: exits[t.reason]=exits.get(t.reason,0)+1

    step=max(1,len(equity)//500)
    ec=[{"i":i,"v":round(float(equity[i]),2)} for i in range(0,len(equity),step)]
    ml_=[{"month":k,"pnl":round(float(v["e"]-v["s"]),2),
           "pct":round(float((v["e"]-v["s"])/(v["s"]+1e-10)*100),2),
           "trades":int(v["n"])} for k,v in sorted(monthly.items())]
    tl=[{"ts_open":t.ts_open,"ts_close":t.ts_close,"side":t.side,
          "entry":float(t.entry),"exit":float(t.ex),
          "sl":float(t.sl0),"tp":float(t.tp0),
          "pnl_usdt":float(t.pnl),"pnl_pct":float(t.pnl_pct),
          "exit_reason":t.reason,"is_win":bool(t.win),
          "long_score":int(t.ls),"short_score":int(t.ss)}
         for t in reversed(trades[-100:])]

    return {
        "symbol":cfg.symbol,"interval":str(cfg.interval_min)+"dk","days":cfg.days,
        "initial_capital":float(cfg.capital),"final_capital":round(float(final),2),
        "total_pnl":round(float(final-cfg.capital),2),
        "total_return_pct":round(float(ret),2),
        "total_trades":int(n),
        "win_trades":int(len(wins)),"loss_trades":int(len(loss)),
        "win_rate":round(float(len(wins)/n*100),2),
        "avg_win_usdt":round(aw,4),"avg_loss_usdt":round(al,4),
        "rr_ratio":round(rr,3),"profit_factor":round(float(pf),3),
        "tp_exits":int(exits.get("take_profit",0)),
        "sl_exits":int(exits.get("stop_loss",0)),
        "early_exits":int(exits.get("early_exit",0)),
        "max_win_streak":int(mw),"max_loss_streak":int(ml),
        "max_drawdown_pct":round(float(mdd),2),
        "max_drawdown_usdt":round(float((eq-pk).min()),2),
        "sharpe":round(float(sh),3),"calmar":round(float(cal),3),
        "monthly":ml_,"equity_curve":ec,"trades":tl,
        "exit_breakdown":{k:int(v) for k,v in exits.items()},
        "config":{"leverage":int(cfg.leverage),
                  "sl_pct":float(cfg.sl_pct),"tp_pct":float(cfg.tp_pct),
                  "min_score":int(cfg.min_score)},
    }

def _report(r):
    if "error" in r: print("[HATA]",r["error"]); return
    s="+" if r["total_return_pct"]>=0 else ""
    eb=r.get("exit_breakdown",{})
    print("\n"+"="*62)
    print(f"  BACKTEST v6 | {r['symbol']} {r['interval']} | {r['days']} gün")
    print(f"  SL:%{r['config']['sl_pct']}  TP:%{r['config']['tp_pct']}  MinSkor:{r['config']['min_score']}/10")
    print("="*62)
    print(f"  Başlangıç  : {r['initial_capital']:.2f} USDT")
    print(f"  Final      : {r['final_capital']:.2f} USDT")
    print(f"  Getiri     : {s}{r['total_pnl']:.2f} ({s}{r['total_return_pct']:.2f}%)")
    print(f"  Win Rate   : %{r['win_rate']:.1f} ({r['win_trades']}W/{r['loss_trades']}L)")
    print(f"  Profit Fac : {r['profit_factor']:.3f}")
    print(f"  Sharpe     : {r['sharpe']:.3f}")
    print(f"  Max DD     : {r['max_drawdown_pct']:.2f}%")
    print(f"  R:R        : {r['rr_ratio']:.2f}:1")
    print(f"  Çıkış      : TP={eb.get('take_profit',0)} SL={eb.get('stop_loss',0)} Erken={eb.get('early_exit',0)}")
    print("-"*62)
    for m in r["monthly"]:
        s2="+" if m["pnl"]>=0 else ""
        bar="#"*min(int(abs(m["pct"])*2),30)
        print(f"  {m['month']}  {s2}{m['pnl']:>9.2f} ({s2}{m['pct']:>5.1f}%)  {bar}")
    print("="*62)

def run(symbol="BTC-USDT",days=90,interval=15,capital=1000.,
        leverage=15,sl=0.4,tp=1.2,min_score=7,trailing=None,save=True) -> dict:
    cfg=Cfg(symbol=symbol,interval_min=interval,days=days,
            capital=capital,leverage=leverage,sl_pct=sl,tp_pct=tp,
            min_score=min_score)
    print(f"[BACKTEST v6] {symbol} | SL:%{sl} TP:%{tp} R:R={tp/sl:.1f}:1 | Score>={cfg.min_score}/10")
    df=fetch(symbol,interval,days)
    if df.empty: return {"error":"Veri alınamadı"}
    print("  Göstergeler hesaplanıyor...")
    df=indicators(df)
    print("  Simüle ediliyor...")
    r=run_bt(df,cfg)
    _report(r)
    if save and "error" not in r:
        os.makedirs("logs",exist_ok=True)
        fn=f"logs/backtest_{symbol.replace('-','')}.json"
        with open(fn,"w",encoding="utf-8") as f: json.dump(r,f,ensure_ascii=False,indent=2)
        print(f"  Kaydedildi: {fn}")
    return r

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",   default="BTC-USDT")
    p.add_argument("--days",     type=int,   default=90)
    p.add_argument("--interval", type=int,   default=15)
    p.add_argument("--capital",  type=float, default=1000)
    p.add_argument("--leverage", type=int,   default=15)
    p.add_argument("--sl",        type=float, default=0.4)
    p.add_argument("--tp",        type=float, default=1.2)
    p.add_argument("--min-score", type=int,   default=7)
    p.add_argument("--both",      action="store_true")
    a=p.parse_args()
    ms = a.min_score
    if a.both:
        for s in ["BTC-USDT","ETH-USDT"]: run(s,a.days,a.interval,a.capital,a.leverage,a.sl,a.tp,ms)
    else:
        run(a.symbol,a.days,a.interval,a.capital,a.leverage,a.sl,a.tp,ms)
