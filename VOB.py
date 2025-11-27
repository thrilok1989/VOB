# deepseek_full_dhan.py
"""
DeepSeek — Full upgrade for Dhan API (single-file Streamlit app)
- Uses Dhan (if available) for intraday + option chain
- Full Gamma/Vega exposures, walls, sequences
- Institutional OI engine (ATM ±2)
- Breakout & Reversal confirmation
- VOB detection & plotting
- Safe normalization of fields + caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
from collections import deque
import warnings

warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

# Optional Dhan API import (paid)
try:
    from dhan_data_fetcher import DhanDataFetcher  # your Dhan SDK class
    DHAN_AVAILABLE = True
except Exception as e:
    DHAN_AVAILABLE = False
    DhanDataFetcher = None

# Fallback to yfinance for basic intraday OHLCV if Dhan missing
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# -------------------------
# -------------------------
#  UTILITIES & NORMALIZERS
# -------------------------
# -------------------------

def safe(v, default=0.0):
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

def safe_get(row, key, default=0.0):
    if isinstance(row, (pd.Series, dict)):
        return row.get(key, default) if isinstance(row, dict) else (row[key] if key in row.index else default)
    return default

def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df

def now_ist():
    return datetime.now(IST)

# Normalizer for Dhan option-chain / other providers to canonical column names
def normalize_chain(df_raw: Any) -> pd.DataFrame:
    """
    Accepts:
      - list of dicts
      - pandas DataFrame
      - nested dict structure from Dhan
    Returns canonical columns:
      strikePrice, oi_ce, oi_pe, change_oi_ce, change_oi_pe,
      ltp_ce, ltp_pe, iv_ce, iv_pe, bid_ce, ask_ce, bid_pe, ask_pe,
      gamma_ce, gamma_pe, vega_ce, vega_pe, theta_ce, theta_pe,
      totalTradedVolume_CE -> vol_ce, totalTradedVolume_PE -> vol_pe
    """
    if df_raw is None:
        return pd.DataFrame()
    # If Dhan returns nested structure like {'data': [...]}, try to extract
    if isinstance(df_raw, dict) and 'data' in df_raw and isinstance(df_raw['data'], list):
        df = pd.DataFrame(df_raw['data'])
    elif isinstance(df_raw, list):
        df = pd.DataFrame(df_raw)
    elif isinstance(df_raw, pd.DataFrame):
        df = df_raw.copy()
    elif isinstance(df_raw, dict):
        # If dict of strikes keyed by strike price
        try:
            df = pd.DataFrame(list(df_raw.values()))
        except Exception:
            df = pd.DataFrame([df_raw])
    else:
        df = pd.DataFrame(df_raw)

    # Common rename map
    rename_map = {
        # Dhan-like fields
        "strikePrice": "strikePrice",
        "strike_price": "strikePrice",
        "strike": "strikePrice",

        "openInterest_CE": "oi_ce",
        "openInterest_PE": "oi_pe",
        "openInterestCE": "oi_ce",
        "openInterestPE": "oi_pe",

        "changeinOpenInterest_CE": "change_oi_ce",
        "changeinOpenInterest_PE": "change_oi_pe",
        "changeInOpenInterestCE": "change_oi_ce",
        "changeInOpenInterestPE": "change_oi_pe",

        "lastPrice_CE": "ltp_ce",
        "lastPrice_PE": "ltp_pe",
        "lastPriceCE": "ltp_ce",
        "lastPricePE": "ltp_pe",

        "impliedVolatility_CE": "iv_ce",
        "impliedVolatility_PE": "iv_pe",
        "ivCe": "iv_ce",
        "ivPe": "iv_pe",

        "bidQty_CE": "bid_ce",
        "askQty_CE": "ask_ce",
        "bidQty_PE": "bid_pe",
        "askQty_PE": "ask_pe",

        "Gamma_CE": "gamma_ce",
        "Gamma_PE": "gamma_pe",
        "Vega_CE": "vega_ce",
        "Vega_PE": "vega_pe",
        "Theta_CE": "theta_ce",
        "Theta_PE": "theta_pe",

        "totalTradedVolume_CE": "vol_ce",
        "totalTradedVolume_PE": "vol_pe",
        "totalTradedVolumeCE": "vol_ce",
        "totalTradedVolumePE": "vol_pe"
    }

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure core columns exist and numeric
    core_cols = ["strikePrice", "oi_ce", "oi_pe", "change_oi_ce", "change_oi_pe",
                 "ltp_ce", "ltp_pe", "iv_ce", "iv_pe", "bid_ce", "ask_ce", "bid_pe", "ask_pe",
                 "gamma_ce", "gamma_pe", "vega_ce", "vega_pe", "theta_ce", "theta_pe",
                 "vol_ce", "vol_pe"]
    for c in core_cols:
        if c not in df.columns:
            df[c] = 0.0

    df = ensure_numeric(df, [c for c in core_cols if c != "strikePrice"])
    if "strikePrice" in df.columns:
        df["strikePrice"] = pd.to_numeric(df["strikePrice"], errors="coerce")
    df.dropna(subset=["strikePrice"], inplace=True)
    df.sort_values("strikePrice", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------------
# -------------------------
#  DATA FETCH (DHAN + fallback)
# -------------------------
# -------------------------

def fetch_intraday_dhan(symbol: str, interval: str = "5"):
    """
    Example Dhan call wrapper.
    - symbol: ticker like 'NSE:NIFTY' or 'NSE:RELIANCE' depending on Dhan SDK
    - interval: '1', '5', '15', '60'
    Returns pandas DataFrame with OHLCV indexed by datetime
    """
    if not DHAN_AVAILABLE:
        return pd.DataFrame()
    try:
        fetcher = DhanDataFetcher()
        # Example fetcher signature; adjust if actual sdk differs
        # Many Dhan SDKs accept instrument symbol (like 'NIFTY 50' or 'NSE:NIFTY')
        # We'll attempt multiple common choices
        possible_instruments = [symbol, symbol.replace("^", ""), symbol.replace(".NS", ""), symbol.replace("^NSE", "NSE")]
        for instr in possible_instruments:
            try:
                # fetch_intraday_data(instrument, interval, from_date, to_date)
                now = now_ist()
                to_date = now.strftime("%Y-%m-%d %H:%M:%S")
                from_date = (now - timedelta(days=7)).replace(hour=9, minute=15, second=0).strftime("%Y-%m-%d %H:%M:%S")
                result = fetcher.fetch_intraday_data(instr, interval=interval, from_date=from_date, to_date=to_date)
                if result and result.get("success") and result.get("data"):
                    df = pd.DataFrame(result["data"])
                    # normalize columns (many formats)
                    # unify to Open, High, Low, Close, Volume, Timestamp
                    df.columns = [str(c).capitalize() for c in df.columns]
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        df.set_index('Timestamp', inplace=True)
                    # ensure OHLCV
                    if 'Open' not in df.columns and 'open' in result['data'][0]:
                        df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)
                    for col in ['Open','High','Low','Close','Volume']:
                        if col not in df.columns:
                            df[col] = 0.0
                    df = df[['Open','High','Low','Close','Volume']]
                    df = df.astype(float)
                    return df
            except Exception:
                continue
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def fetch_option_chain_dhan(symbol: str, expiry: str = None) -> pd.DataFrame:
    """
    Wrapper for Dhan option chain fetch.
    - symbol e.g. '^NSEI' or 'RELIANCE.NS' depending on mapping
    - expiry: 'YYYY-MM-DD' or None to pick nearest
    """
    if not DHAN_AVAILABLE:
        return pd.DataFrame()
    try:
        fetcher = DhanDataFetcher()
        # Adjust parameters to your Dhan SDK.
        # Many fetchers provide: fetch_option_chain(instrument, expiryDate)
        # Here we attempt common variants and normalize output.
        possible_instruments = [symbol, symbol.replace("^",""), symbol.replace(".NS","")]
        for instr in possible_instruments:
            try:
                # try fetch_option_chain
                res = fetcher.fetch_option_chain(instr, expiry_date=expiry) if hasattr(fetcher, 'fetch_option_chain') else None
                if not res:
                    # try alternate name
                    res = fetcher.fetch_option_chain(instr) if hasattr(fetcher, 'fetch_option_chain') else None
                if res and isinstance(res, dict) and res.get('success') and res.get('data'):
                    df = pd.DataFrame(res['data'])
                    return normalize_chain(df)
                if isinstance(res, list):
                    return normalize_chain(res)
            except Exception:
                continue
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def fetch_intraday_fallback(symbol: str, period: str = "7d", interval: str = "5m"):
    """Fallback using yfinance if available (intraday limited)"""
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        tick = yf.Ticker(symbol)
        df = tick.history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        df = df[['Open','High','Low','Close','Volume']].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

# Top-level fetchers used by app
def fetch_intraday(symbol: str, interval_minutes: int = 5):
    interval_map = {1:'1', 5:'5', 15:'15', 60:'60'}
    interval = interval_map.get(interval_minutes, '5')
    # try Dhan first
    if DHAN_AVAILABLE:
        df = fetch_intraday_dhan(symbol, interval=interval)
        if not df.empty:
            return df
    # fallback
    return fetch_intraday_fallback(symbol, period="7d", interval=f"{interval}m")

def fetch_option_chain(symbol: str, expiry: str = None):
    if DHAN_AVAILABLE:
        df = fetch_option_chain_dhan(symbol, expiry=expiry)
        if not df.empty:
            return df
    return pd.DataFrame()

# -------------------------
# -------------------------
#  TECHNICAL INDICATORS (fast)
# -------------------------
# -------------------------

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14):
    h = df['High']; l = df['Low']; c = df['Close']
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().fillna(method='bfill')

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def mfi(df: pd.DataFrame, period: int = 10):
    if df['Volume'].sum() == 0:
        return pd.Series([50.0]*len(df), index=df.index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    val = 100 - (100 / (1 + ratio))
    return val.fillna(50)

# -------------------------
# -------------------------
#  VOLUME ORDER BLOCKS (VOB)
# -------------------------
# -------------------------

class VolumeOrderBlocks:
    def __init__(self, sensitivity: int = 5):
        self.len1 = sensitivity
        self.len2 = sensitivity + 13

    def detect(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        if df is None or len(df) < self.len2:
            return [], []
        ema1 = ema(df['Close'], self.len1)
        ema2 = ema(df['Close'], self.len2)
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_dn = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        atr_series = atr(df, period=200)
        bullish = []
        bearish = []
        for i in range(len(df)):
            if cross_up.iloc[i]:
                look = df.iloc[max(0, i-self.len2):i+1]
                if look.empty: continue
                low_idx = look['Low'].idxmin()
                low_price = look.loc[low_idx, 'Low']
                vol_sum = look['Volume'].sum()
                open_p = look.loc[low_idx, 'Open']
                close_p = look.loc[low_idx, 'Close']
                src = min(open_p, close_p)
                if pd.notna(atr_series.iloc[i]):
                    if (src - low_price) < (atr_series.iloc[i] * 0.5):
                        src = low_price + atr_series.iloc[i] * 0.5
                mid = (src + low_price) / 2
                bullish.append({"index": low_idx, "upper": src, "lower": low_price, "mid": mid, "volume": vol_sum})
            if cross_dn.iloc[i]:
                look = df.iloc[max(0, i-self.len2):i+1]
                if look.empty: continue
                high_idx = look['High'].idxmax()
                high_price = look.loc[high_idx, 'High']
                vol_sum = look['Volume'].sum()
                open_p = look.loc[high_idx, 'Open']
                close_p = look.loc[high_idx, 'Close']
                src = max(open_p, close_p)
                if pd.notna(atr_series.iloc[i]):
                    if (high_price - src) < (atr_series.iloc[i] * 0.5):
                        src = high_price - atr_series.iloc[i] * 0.5
                mid = (src + high_price) / 2
                bearish.append({"index": high_idx, "upper": high_price, "lower": src, "mid": mid, "volume": vol_sum})
        # filter overlaps
        if len(atr_series)>0:
            atr_val = float(atr_series.iloc[-1])
        else:
            atr_val = 0.0
        def filt(blocks):
            out = []
            for b in blocks:
                if not any(abs(b['mid'] - ex['mid']) < (atr_val if atr_val>0 else 1e-6) for ex in out):
                    out.append(b)
            return out
        bullish = filt(bullish)
        bearish = filt(bearish)
        return bullish, bearish

    def plot(self, df: pd.DataFrame, bullish: List[Dict], bearish: List[Dict], title:str="VOB"):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        for b in bullish:
            fig.add_hline(y=b['upper'], line_dash='dash', annotation_text='Bull upper', opacity=0.6)
            fig.add_hline(y=b['lower'], line_dash='dash', annotation_text='Bull lower', opacity=0.4)
        for b in bearish:
            fig.add_hline(y=b['upper'], line_dash='dash', annotation_text='Bear upper', opacity=0.6)
            fig.add_hline(y=b['lower'], line_dash='dash', annotation_text='Bear lower', opacity=0.4)
        fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
        return fig

# -------------------------
# -------------------------
#  GAMMA SEQUENCE ANALYZER
# -------------------------
# -------------------------

class GammaSequenceAnalyzer:
    """
    Calculates gamma exposures (CE, PE), net gamma per strike,
    finds sequences and gamma walls (local extrema with thresholds),
    computes zone aggregates (ATM, near OTM etc).
    """

    def __init__(self):
        # thresholds - tuneable
        self.levels = [
            (20000, 'EXTREME_POSITIVE', 100),
            (10000, 'HIGH_POSITIVE', 75),
            (5000, 'MODERATE_POSITIVE', 50),
            (-1000, 'NEUTRAL', 0),
            (-5000, 'MODERATE_NEGATIVE', -50),
            (-10000, 'HIGH_NEGATIVE', -75),
            (-20000, 'EXTREME_NEGATIVE', -100)
        ]

    def calculate_gamma_exposure(self, df_chain: pd.DataFrame) -> pd.DataFrame:
        df = df_chain.copy()
        # Ensure columns exist
        for col in ["gamma_ce", "gamma_pe", "oi_ce", "oi_pe"]:
            if col not in df.columns:
                df[col] = 0.0
        # gamma exposure per side: gamma * OI * contract_size (100)
        df['gamma_exposure_ce'] = df['gamma_ce'] * df['oi_ce'] * 100.0
        df['gamma_exposure_pe'] = df['gamma_pe'] * df['oi_pe'] * 100.0
        df['net_gamma'] = df['gamma_exposure_ce'] + df['gamma_exposure_pe']
        return df

    def profile_total(self, df_chain: pd.DataFrame) -> Dict[str, Any]:
        if df_chain is None or df_chain.empty:
            return {'total_gamma':0,'profile':'NEUTRAL','score':0}
        df = self.calculate_gamma_exposure(df_chain)
        total = float(df['net_gamma'].sum())
        for thresh, name, score in self.levels:
            if total >= thresh:
                return {'total_gamma': total, 'profile': name, 'score': score}
        # if not matched, assume extreme negative
        return {'total_gamma': total, 'profile': 'EXTREME_NEGATIVE', 'score': -100}

    def analyze_zones(self, df_chain: pd.DataFrame, spot: float) -> Dict[str, Any]:
        if df_chain is None or df_chain.empty:
            return {}
        df = self.calculate_gamma_exposure(df_chain)
        strikes = df['strikePrice'].unique()
        strike_step = strikes[1] - strikes[0] if len(strikes)>1 else 50
        zones = {
            'itm_puts': df[df['strikePrice'] < spot - strike_step],
            'near_otm_puts': df[(df['strikePrice'] >= spot - strike_step) & (df['strikePrice'] < spot)],
            'atm': df[abs(df['strikePrice'] - spot) <= strike_step],
            'near_otm_calls': df[(df['strikePrice'] > spot) & (df['strikePrice'] <= spot + strike_step)],
            'otm_calls': df[df['strikePrice'] > spot + strike_step],
        }
        analysis = {}
        for name,zdf in zones.items():
            if not zdf.empty:
                analysis[name] = {
                    'gamma_exposure': float(zdf['net_gamma'].sum()),
                    'oi_sum': float(zdf['oi_ce'].sum() + zdf['oi_pe'].sum()),
                    'strike_range': (float(zdf['strikePrice'].min()), float(zdf['strikePrice'].max()))
                }
        return analysis

    def sequences_and_walls(self, df_chain: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify contiguous sequences of positive/negative net gamma across strikes
        and gamma walls (local extrema with magnitude threshold).
        """
        if df_chain is None or df_chain.empty:
            return {'positive_sequences':[], 'negative_sequences':[], 'walls':[]}

        df = self.calculate_gamma_exposure(df_chain).sort_values('strikePrice').reset_index(drop=True)
        df['net_gamma_change'] = df['net_gamma'].diff().fillna(0)
        # sequences
        pos_seqs = []
        neg_seqs = []
        current = {'sign': None, 'strikes': [], 'total': 0.0}
        for _, row in df.iterrows():
            sign = 1 if row['net_gamma'] >= 0 else -1
            if current['sign'] is None:
                current['sign'] = sign
                current['strikes'] = [row['strikePrice']]
                current['total'] = float(row['net_gamma'])
            elif sign == current['sign']:
                current['strikes'].append(row['strikePrice'])
                current['total'] += float(row['net_gamma'])
            else:
                # push
                seq = {'strikes': current['strikes'], 'total_gamma': current['total'], 'length': len(current['strikes'])}
                if current['sign'] == 1:
                    pos_seqs.append(seq)
                else:
                    neg_seqs.append(seq)
                current = {'sign': sign, 'strikes':[row['strikePrice']], 'total': float(row['net_gamma'])}
        # push last
        if current['sign'] is not None:
            seq = {'strikes': current['strikes'], 'total_gamma': current['total'], 'length': len(current['strikes'])}
            (pos_seqs if current['sign']==1 else neg_seqs).append(seq)

        # gamma walls (local max/min)
        walls = []
        for i in range(1, len(df)-1):
            g = df.loc[i,'net_gamma']; g_prev = df.loc[i-1,'net_gamma']; g_next = df.loc[i+1,'net_gamma']
            # local maxima
            if g > g_prev and g > g_next and g>5000:
                walls.append({'strike': float(df.loc[i,'strikePrice']), 'gamma_exposure': float(g), 'type':'RESISTANCE', 'strength': 'STRONG' if g>10000 else 'MODERATE'})
            if g < g_prev and g < g_next and g < -5000:
                walls.append({'strike': float(df.loc[i,'strikePrice']), 'gamma_exposure': float(g), 'type':'SUPPORT', 'strength': 'STRONG' if g<-10000 else 'MODERATE'})
        return {'positive_sequences': pos_seqs, 'negative_sequences': neg_seqs, 'walls': walls}

# -------------------------
# -------------------------
#  INSTITUTIONAL OI ADVANCED ENGINE
# -------------------------
# -------------------------

class InstitutionalOIAdvanced:
    """
    ATM ±2 strike analysis using master rules + bid/ask + IV + volume signals
    Produces patterns and aggregated score.
    """

    def __init__(self):
        # master table of patterns - simplified mapping
        self.master_rules = {
            'CALL': {
                'Winding_Up_Price_Up': {'bias':'BEARISH','confidence':'HIGH','move':'Selling/Writing'},
                'Winding_Up_Price_Down': {'bias':'BEARISH','confidence':'HIGH','move':'Sellers Dominating'},
                'Unwinding_Down_Price_Up': {'bias':'BULLISH','confidence':'MEDIUM','move':'Short Covering'},
                'Unwinding_Down_Price_Down': {'bias':'MILD_BEARISH','confidence':'LOW','move':'Longs Exiting'}
            },
            'PUT': {
                'Winding_Up_Price_Down': {'bias':'BULLISH','confidence':'HIGH','move':'Selling/Writing'},
                'Winding_Up_Price_Up': {'bias':'BULLISH','confidence':'HIGH','move':'Sellers Dominating'},
                'Unwinding_Down_Price_Down': {'bias':'BEARISH','confidence':'MEDIUM','move':'Short Covering'},
                'Unwinding_Down_Price_Up': {'bias':'MILD_BULLISH','confidence':'LOW','move':'Longs Exiting'}
            }
        }

    def analyze_instrumental_pattern(self, option_type: str, oi_change: float, price_change: float, volume: float, iv_change: float, bid_ask_ratio: float) -> Dict[str,Any]:
        """
        Return pattern dict for single strike side
        """
        oi_action = "Winding_Up" if oi_change > 0 else "Unwinding_Down"
        price_action = "Price_Up" if price_change > 0 else "Price_Down"
        key = f"{oi_action}_{price_action}"
        base = self.master_rules.get(option_type, {}).get(key, {})
        volume_signal = "High" if volume > 1000 else "Low"
        iv_signal = "Rising" if iv_change > 0 else "Falling"
        liquidity_signal = "Bid_Heavy" if bid_ask_ratio > 1.2 else ("Ask_Heavy" if bid_ask_ratio < 0.8 else "Balanced")
        # confidence adjust
        confidence = base.get('confidence','LOW')
        if volume_signal=='High' and abs(iv_change) > 1.0:
            confidence = "VERY_HIGH"
        return {
            'option_type': option_type,
            'pattern_key': key,
            'bias': base.get('bias','NEUTRAL'),
            'institution_move': base.get('move','Unknown'),
            'confidence': confidence,
            'volume_signal': volume_signal,
            'iv_signal': iv_signal,
            'liquidity_signal': liquidity_signal,
            'oi_change': float(oi_change),
            'price_change': float(price_change),
            'volume': float(volume)
        }

    def analyze_atm(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str,Any]:
        df = normalize_chain(df_chain)
        if df is None or df.empty:
            return {'overall_bias':'NEUTRAL','score':0,'patterns':[]}
        # choose strike difference
        strike_diff = float(df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]) if len(df)>1 else 50
        atm_df = df[abs(df['strikePrice'] - spot_price) <= (strike_diff*2)].copy()
        patterns = []
        total_score = 0.0
        count = 0
        for _, row in atm_df.iterrows():
            strike = float(row['strikePrice'])
            # CALL
            oi_ce = float(row.get('oi_ce',0))
            change_ce = float(row.get('change_oi_ce',0))
            ltp_ce = float(row.get('ltp_ce',0))
            prev_ce = float(row.get('prev_lt_ce', ltp_ce))  # prev_lt_ce optional
            iv_ce = float(row.get('iv_ce',0))
            prev_iv_ce = float(row.get('prev_iv_ce', iv_ce))
            bid_ce = float(row.get('bid_ce',1))
            ask_ce = float(row.get('ask_ce',1))
            bidask_ce = bid_ce / ask_ce if ask_ce != 0 else 1
            price_change_ce = ltp_ce - prev_ce

            # analyze CALL
            if change_ce != 0:
                p = self.analyze_instrumental_pattern('CALL', change_ce, price_change_ce, float(row.get('vol_ce',0)), iv_ce - prev_iv_ce, bidask_ce)
                p['strike'] = strike
                patterns.append(p)
                total_score += self._bias_to_score(p['bias'], p['confidence'])
                count += 1

            # PUT
            oi_pe = float(row.get('oi_pe',0))
            change_pe = float(row.get('change_oi_pe',0))
            ltp_pe = float(row.get('ltp_pe',0))
            prev_pe = float(row.get('prev_lt_pe', ltp_pe))
            iv_pe = float(row.get('iv_pe',0))
            prev_iv_pe = float(row.get('prev_iv_pe', iv_pe))
            bid_pe = float(row.get('bid_pe',1))
            ask_pe = float(row.get('ask_pe',1))
            bidask_pe = bid_pe / ask_pe if ask_pe != 0 else 1
            price_change_pe = ltp_pe - prev_pe

            if change_pe != 0:
                p = self.analyze_instrumental_pattern('PUT', change_pe, price_change_pe, float(row.get('vol_pe',0)), iv_pe - prev_iv_pe, bidask_pe)
                p['strike'] = strike
                patterns.append(p)
                total_score += self._bias_to_score(p['bias'], p['confidence'])
                count += 1

        avg_score = (total_score / count) if count>0 else 0.0
        overall_bias = 'BULLISH' if avg_score>0.2 else ('BEARISH' if avg_score < -0.2 else 'NEUTRAL')
        return {'overall_bias': overall_bias, 'score': avg_score*100, 'patterns': patterns, 'strikes_analyzed': len(atm_df)}

    def _bias_to_score(self, bias: str, confidence: str) -> float:
        bias_scores = {'BULLISH':1.0,'MILD_BULLISH':0.5,'NEUTRAL':0.0,'MILD_BEARISH':-0.5,'BEARISH':-1.0}
        conf_mul = {'VERY_HIGH':1.5,'HIGH':1.2,'MEDIUM':1.0,'LOW':0.7}
        return bias_scores.get(bias,0.0) * conf_mul.get(confidence,1.0)

# -------------------------
# -------------------------
#  BREAKOUT & REVERSAL ANALYZER (FULL)
# -------------------------
# -------------------------

class BreakoutReversalAnalyzer:
    def __init__(self):
        # weight config (tunable)
        self.weights = {
            'oi_pattern':25,
            'price_oi_conflict':20,
            'iv_behavior':15,
            'pcr':15,
            'max_pain':10,
            'wall_breakdown':15
        }

    def analyze_breakout_confirmation(self, df_chain: pd.DataFrame, spot_price: float, price_change: float, volume_change: float) -> Dict[str,Any]:
        df = normalize_chain(df_chain)
        if df is None or df.empty:
            return {'breakout_confidence':0,'direction':'UNKNOWN','signals':[]}
        # ATM ±2 selection
        strike_diff = float(df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]) if len(df)>1 else 50
        atm = df[abs(df['strikePrice'] - spot_price) <= (strike_diff*2)].copy()
        if atm.empty:
            return {'breakout_confidence':0,'direction':'UNKNOWN','signals':['No ATM strikes']}

        is_up = price_change > 0
        total_score = 0.0
        max_score = sum(self.weights.values())
        signals = []

        # 1. OI action
        oi_score, oi_signals = self._analyze_oi_pattern(atm, is_up)
        signals.extend(oi_signals); total_score += oi_score

        # 2. Price vs OI conflict
        poc_score, poc_signals = self._analyze_price_oi_conflict(atm, is_up)
        signals.extend(poc_signals); total_score += poc_score

        # 3. IV behavior
        iv_score, iv_signals = self._analyze_iv_behavior(atm, is_up)
        signals.extend(iv_signals); total_score += iv_score

        # 4. PCR trend
        pcr_score, pcr_signals = self._analyze_pcr_trend(df, is_up)
        signals.extend(pcr_signals); total_score += pcr_score

        # 5. Max pain movement
        mp_score, mp_signals = self._analyze_max_pain(df, spot_price, is_up)
        signals.extend(mp_signals); total_score += mp_score

        # 6. OI wall breakdown
        wall_score, wall_signals = self._analyze_wall_breakdown(atm, is_up)
        signals.extend(wall_signals); total_score += wall_score

        confidence = (total_score / max_score) * 100 if max_score > 0 else 0
        breakout_type = "REAL_BREAKOUT" if confidence >= 60 else ("WEAK_BREAKOUT" if confidence >= 30 else "FAKE_BREAKOUT")
        return {'breakout_confidence': confidence, 'direction': 'UP' if is_up else 'DOWN', 'breakout_type': breakout_type, 'signals': signals, 'total_score': total_score, 'max_score': max_score}

    def _analyze_oi_pattern(self, atm: pd.DataFrame, is_upside: bool):
        # Score out of weight['oi_pattern']
        weight = self.weights['oi_pattern']
        signals=[]
        score=0.0
        total_ce_change = float(atm['change_oi_ce'].sum())
        total_pe_change = float(atm['change_oi_pe'].sum())
        # Upside wants CE OI decreasing and PE OI increasing
        if is_upside:
            if total_ce_change < 0:
                signals.append("CE OI decreasing (call shorts covering)"); score += weight*0.4
            else:
                signals.append("CE OI not decreasing")
            if total_pe_change > 0:
                signals.append("PE OI increasing (put writing)"); score += weight*0.6
            else:
                signals.append("PE OI not increasing")
        else:
            if total_ce_change > 0:
                signals.append("CE OI increasing (call writing)"); score += weight*0.4
            else:
                signals.append("CE OI not increasing")
            if total_pe_change < 0:
                signals.append("PE OI decreasing (put shorts covering)"); score += weight*0.6
            else:
                signals.append("PE OI not decreasing")
        return score, signals

    def _analyze_price_oi_conflict(self, atm: pd.DataFrame, is_upside: bool):
        weight = self.weights['price_oi_conflict']
        signals=[]; score=0.0
        # dominant strike by OI
        if is_upside:
            dom = atm.loc[atm['oi_ce'].idxmax()]
            if float(dom['change_oi_ce']) < 0:
                signals.append("Clean upside: price ↑ + CE OI ↓"); score += weight
            else:
                signals.append("Price ↑ but CE OI not ↓ (possible seller wall)")
        else:
            dom = atm.loc[atm['oi_pe'].idxmax()]
            if float(dom['change_oi_pe']) < 0:
                signals.append("Clean downside: price ↓ + PE OI ↓"); score += weight
            else:
                signals.append("Price ↓ but PE OI not ↓ (possible defender)")
        return score, signals

    def _analyze_iv_behavior(self, atm: pd.DataFrame, is_upside: bool):
        weight = self.weights['iv_behavior']
        signals=[]; score=0.0
        avg_ce_iv = float(atm['iv_ce'].mean())
        avg_pe_iv = float(atm['iv_pe'].mean())
        # Upside: CE IV should rise
        if is_upside:
            if avg_ce_iv > avg_pe_iv:
                signals.append(f"CE IV ({avg_ce_iv:.1f}) > PE IV ({avg_pe_iv:.1f})"); score += weight*0.6
            else:
                signals.append("CE IV not > PE IV")
            if avg_pe_iv < 20:
                signals.append("Low PE IV indicates little hedging"); score += weight*0.4
        else:
            if avg_pe_iv > avg_ce_iv:
                signals.append(f"PE IV ({avg_pe_iv:.1f}) > CE IV ({avg_ce_iv:.1f})"); score += weight*0.6
            else:
                signals.append("PE IV not > CE IV")
            if avg_ce_iv < 20:
                signals.append("Low CE IV indicates little call hedging"); score += weight*0.4
        return score, signals

    def _analyze_pcr_trend(self, df_chain: pd.DataFrame, is_upside: bool):
        weight = self.weights['pcr']
        df = normalize_chain(df_chain)
        total_ce = float(df['oi_ce'].sum())
        total_pe = float(df['oi_pe'].sum())
        pcr = (total_pe / total_ce) if total_ce>0 else 0.0
        signals=[]; score=0.0
        if is_upside:
            if pcr > 0.8:
                signals.append(f"PCR {pcr:.2f} > 0.8 (bullish)"); score += weight
            elif pcr > 0.6:
                signals.append(f"PCR {pcr:.2f} neutral"); score += weight*0.5
            else:
                signals.append(f"PCR {pcr:.2f} low (bearish)")
        else:
            if pcr < 0.7:
                signals.append(f"PCR {pcr:.2f} < 0.7 (bearish)"); score += weight
            elif pcr < 0.9:
                signals.append(f"PCR {pcr:.2f} neutral"); score += weight*0.5
            else:
                signals.append(f"PCR {pcr:.2f} > 0.9 (bullish)")
        return score, signals

    def _analyze_max_pain(self, df_chain: pd.DataFrame, spot: float, is_upside: bool):
        # Simplified max pain: largest combined OI side
        weight = self.weights['max_pain']
        df = normalize_chain(df_chain)
        ce_max_strike = float(df.loc[df['oi_ce'].idxmax()]['strikePrice']) if not df.empty else spot
        pe_max_strike = float(df.loc[df['oi_pe'].idxmax()]['strikePrice']) if not df.empty else spot
        signals=[]; score=0.0
        if is_upside:
            if pe_max_strike > spot:
                signals.append(f"Max PE OI at {pe_max_strike} above spot (supports upside)"); score += weight
            else:
                signals.append(f"Max PE OI at {pe_max_strike} below spot")
        else:
            if ce_max_strike < spot:
                signals.append(f"Max CE OI at {ce_max_strike} below spot (supports downside)"); score += weight
            else:
                signals.append(f"Max CE OI at {ce_max_strike} above spot")
        return score, signals

    def _analyze_wall_breakdown(self, atm: pd.DataFrame, is_upside: bool):
        weight = self.weights['wall_breakdown']
        signals=[]; score=0.0
        ce_oi_change = float(atm['change_oi_ce'].sum())
        pe_oi_change = float(atm['change_oi_pe'].sum())
        if is_upside:
            if ce_oi_change < 0:
                signals.append("CE OI wall breaking"); score += weight*0.6
            if pe_oi_change > 0:
                signals.append("PE OI building (writing)"); score += weight*0.4
        else:
            if pe_oi_change < 0:
                signals.append("PE OI wall breaking"); score += weight*0.6
            if ce_oi_change > 0:
                signals.append("CE OI building (writing)"); score += weight*0.4
        return score, signals

# -------------------------
# -------------------------
#  BiasAnalysisPro (All indicators plus adaptive weighting)
# -------------------------
# -------------------------

class BiasAnalysisPro:
    def __init__(self, config: Dict[str,Any] = None):
        # set default config
        self.config = {
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'bias_strength': 60,
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,
            'divergence_threshold': 60
        }
        if config:
            self.config.update(config)

    def analyze(self, symbol: str = "^NSEI"):
        df = fetch_intraday(symbol, interval_minutes=5)
        if df is None or df.empty or len(df) < 50:
            return {'success':False, 'error':'Insufficient intraday data'}
        nowprice = float(df['Close'].iloc[-1])

        res = []
        # Volume Delta
        up_vol = df.loc[df['Close'] > df['Open'], 'Volume'].sum()
        down_vol = df.loc[df['Close'] < df['Open'], 'Volume'].sum()
        vol_delta = up_vol - down_vol
        vol_bias = 'BULLISH' if vol_delta > 0 else ('BEARISH' if vol_delta < 0 else 'NEUTRAL')
        res.append({'indicator':'Volume Delta','bias':vol_bias,'value':vol_delta,'weight':1.0,'category':'fast'})

        # HVP
        hvp_b, hvp_s, ph, pl = VolumeOrderBlocks(sensitivity=5).detect(df)  # reuse
        hvp_bias = 'BULLISH' if hvp_b else ('BEARISH' if hvp_s else 'NEUTRAL')
        res.append({'indicator':'HVP','bias':hvp_bias,'value':f'Highs:{ph} Lows:{pl}','weight':1.0,'category':'fast'})

        # VOB via EMA cross
        vob_b, vob_s, ema1, ema2 = self._vob_signal(df)
        vob_bias = 'BULLISH' if vob_b else ('BEARISH' if vob_s else 'NEUTRAL')
        res.append({'indicator':'VOB','bias':vob_bias,'value':f'EMA5:{ema1:.2f} EMA18:{ema2:.2f}','weight':1.0,'category':'fast'})

        # Order Blocks (ema5/18)
        ema5 = ema(df['Close'],5); ema18 = ema(df['Close'],18)
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        ob_bias = 'BULLISH' if cross_up else ('BEARISH' if cross_dn else 'NEUTRAL')
        res.append({'indicator':'OrderBlocks','bias':ob_bias,'value':f'{ema5.iloc[-1]:.2f}/{ema18.iloc[-1]:.2f}','weight':1.0,'category':'fast'})

        # RSI
        rsi_v = float(rsi(df['Close'], self.config['rsi_period']).iloc[-1])
        rsi_bias = 'BULLISH' if rsi_v>50 else 'BEARISH'
        res.append({'indicator':'RSI','bias':rsi_bias,'value':rsi_v,'weight':1.0,'category':'fast'})

        # DMI (approx)
        plus_di, minus_di, adx_val = self._calc_dmi(df)
        dmi_bias = 'BULLISH' if plus_di > minus_di else 'BEARISH'
        res.append({'indicator':'DMI','bias':dmi_bias,'value':f'+DI:{plus_di:.1f} -DI:{minus_di:.1f}','weight':1.0,'category':'fast'})

        # VIDYA simplified
        vid = ema(df['Close'], 10)
        vid_bias = 'BULLISH' if df['Close'].iloc[-1] > vid.iloc[-1] else ('BEARISH' if df['Close'].iloc[-1] < vid.iloc[-1] else 'NEUTRAL')
        res.append({'indicator':'VIDYA','bias':vid_bias,'value':float(vid.iloc[-1]),'weight':1.0,'category':'fast'})

        # MFI
        mfi_v = float(mfi(df, self.config['mfi_period']).iloc[-1])
        mfi_bias = 'BULLISH' if mfi_v>50 else 'BEARISH'
        res.append({'indicator':'MFI','bias':mfi_bias,'value':mfi_v,'weight':1.0,'category':'fast'})

        # Aggregate FAST only (we keep medium/slow optional)
        fast_total = len([x for x in res if x['category']=='fast'])
        fast_bull = sum(1 for x in res if x['category']=='fast' and x['bias']=='BULLISH')
        fast_bear = sum(1 for x in res if x['category']=='fast' and x['bias']=='BEARISH')

        fast_weight = self.config['normal_fast_weight']
        medium_weight = self.config['normal_medium_weight']
        slow_weight = 0.0  # we will not use slow by default here

        bullish_signals = fast_bull * fast_weight
        bearish_signals = fast_bear * fast_weight
        total_signals = fast_total * fast_weight if fast_total>0 else 1.0

        bullish_bias_pct = (bullish_signals / total_signals) * 100.0
        bearish_bias_pct = (bearish_signals / total_signals) * 100.0

        overall_bias = 'BULLISH' if bullish_bias_pct >= self.config['bias_strength'] else ('BEARISH' if bearish_bias_pct >= self.config['bias_strength'] else 'NEUTRAL')
        return {'success':True, 'symbol': symbol, 'current_price': nowprice, 'bias_results': res, 'overall_bias': overall_bias, 'bullish_pct': bullish_bias_pct, 'bearish_pct': bearish_bias_pct}

    def _vob_signal(self, df):
        b, s = False, False
        try:
            b, s = False, False
            ema1 = ema(df['Close'], 5).iloc[-1]
            ema2 = ema(df['Close'], 18).iloc[-1]
            prev1 = ema(df['Close'], 5).iloc[-2]
            prev2 = ema(df['Close'], 18).iloc[-2]
            cross_up = (prev1 <= prev2) and (ema1 > ema2)
            cross_dn = (prev1 >= prev2) and (ema1 < ema2)
            b, s = bool(cross_up), bool(cross_dn)
            return b, s, ema1, ema2
        except Exception:
            return False, False, float('nan'), float('nan')

    def _calc_dmi(self, df):
        high = df['High']; low = df['Low']; close = df['Close']
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move>down_move)&(up_move>0),0).rolling(window=self.config['dmi_period']).mean().iloc[-1]
        minus_dm = down_move.where((down_move>up_move)&(down_move>0),0).rolling(window=self.config['dmi_period']).mean().iloc[-1]
        return float(safe(plus_dm,0.0)), float(safe(minus_dm,0.0)), 0.0

# -------------------------
# -------------------------
#  STREAMLIT APP & UI
# -------------------------
# -------------------------

st.set_page_config(page_title="DeepSeek — Full (Dhan)", layout="wide")
st.title("DeepSeek — Full (Dhan) — Gamma / Institutional / VOB / Breakout")

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Symbol (Dhan instrument)", options=["NIFTY", "BANKNIFTY", "RELIANCE", "INFY", "^NSEI"], index=0)
    interval_m = st.selectbox("Intraday interval (minutes)", options=[1,5,15,60], index=1)
    sensitivity = st.slider("VOB sensitivity (len)", min_value=3, max_value=9, value=5)
    expiry_input = st.text_input("Expiry (YYYY-MM-DD) or leave blank to auto", value="")
    run_btn = st.button("Run Full Analysis")

# caching heavy functions
@st.cache_data(ttl=20)
def cached_intraday(sym, interval_m):
    return fetch_intraday(sym, interval_m)

@st.cache_data(ttl=20)
def cached_option_chain(sym, expiry):
    return fetch_option_chain(sym, expiry)

@st.cache_data(ttl=20)
def cached_gamma_and_institutional(df_chain, spot):
    g = GammaSequenceAnalyzer()
    gamma_profile = g.profile_total(df_chain)
    gamma_zones = g.analyze_zones(df_chain, spot)
    seqs = g.sequences_and_walls(df_chain)
    inst = InstitutionalOIAdvanced()
    inst_res = inst.analyze_atm(df_chain, spot)
    return gamma_profile, gamma_zones, seqs, inst_res

if run_btn:
    st.info(f"Fetching intraday for {symbol} ...")
    df_intraday = cached_intraday(symbol, interval_m)
    if df_intraday is None or df_intraday.empty:
        st.error("No intraday data found (Dhan or fallback). Check symbol mapping to Dhan instrument.")
    else:
        st.success(f"Fetched {len(df_intraday)} candles. Current price: {df_intraday['Close'].iloc[-1]:.2f}")
        # Bias analysis
        biaser = BiasAnalysisPro()
        bias_res = biaser.analyze(symbol)
        st.subheader("Bias Analysis (fast indicators)")
        st.write("Overall Bias:", bias_res.get('overall_bias'))
        st.dataframe(pd.DataFrame(bias_res['bias_results']))
        # VOB detect and plot
        vob = VolumeOrderBlocks(sensitivity=sensitivity)
        bulls, bears = vob.detect(df_intraday)
        st.subheader("Volume Order Blocks (VOB)")
        if bulls or bears:
            st.write("Bullish blocks:", len(bulls), "Bearish blocks:", len(bears))
            fig_vob = vob.plot(df_intraday, bulls, bears, title=f"VOB — {symbol}")
            st.plotly_chart(fig_vob, use_container_width=True)
        else:
            st.write("No VOB blocks detected on this timeframe.")
        # option chain
        st.subheader("Option Chain / Institutional Analysis")
        df_chain = cached_option_chain(symbol, expiry_input if expiry_input else None)
        if df_chain is None or df_chain.empty:
            st.warning("No option chain fetched. Ensure Dhan integration and instrument mapping are correct. You can paste option chain JSON below.")
            uploaded = st.file_uploader("Upload option chain CSV/JSON (for offline testing)", type=['csv','json','txt'])
            pasted = st.text_area("Or paste option-chain JSON (list of strike dicts)")
            df_chain_local = None
            if uploaded:
                try:
                    if uploaded.name.endswith('.csv'):
                        df_chain_local = pd.read_csv(uploaded)
                    else:
                        txt = uploaded.getvalue().decode('utf-8')
                        try:
                            df_chain_local = pd.read_json(txt)
                        except Exception:
                            df_chain_local = pd.read_csv(pd.compat.StringIO(txt))
                except Exception as e:
                    st.error(f"Could not parse uploaded file: {e}")
            elif pasted:
                try:
                    parsed = json.loads(pasted)
                    df_chain_local = pd.DataFrame(parsed)
                except Exception as e:
                    st.error(f"Could not parse pasted JSON: {e}")
            if df_chain_local is not None:
                st.success("Using pasted/uploaded option chain for analysis.")
                df_chain = normalize_chain(df_chain_local)
            else:
                df_chain = pd.DataFrame()  # empty - we will skip institutional
        else:
            st.write(f"Fetched option chain ({len(df_chain)} strikes). Showing top rows:")
            st.dataframe(df_chain.head(10))

        # Proceed if option chain available
        if df_chain is not None and not df_chain.empty:
            spot = float(st.number_input("Spot price (detected from intraday)", value=float(df_intraday['Close'].iloc[-1])))
            # Gamma & institutional cached
            gamma_profile, gamma_zones, seqs, inst_res = cached_gamma_and_institutional(df_chain, spot)
            st.metric("Gamma Profile", f"{gamma_profile.get('profile','NEUTRAL')} ({gamma_profile.get('score',0)})")
            st.write("Gamma Zones:")
            st.json(gamma_zones)
            st.write("Gamma sequences & walls:")
            st.json(seqs)
            st.write("Institutional ATM ±2 analysis:")
            st.json(inst_res)

            # Breakout / reversal quick-check
            st.subheader("Breakout / Reversal Confirmation")
            price_change = float(st.number_input("Price change (session) example", value=0.0))
            volume_change = float(st.number_input("Volume change (session) example", value=0.0))
            br = BreakoutReversalAnalyzer()
            br_res = br.analyze_breakout_confirmation(df_chain, spot, price_change, volume_change)
            st.write("Breakout check:", br_res)

            # show top gamma exposures
            df_gamma = GammaSequenceAnalyzer().calculate_gamma_exposure(df_chain)
            df_gamma['net_gamma'] = df_gamma['net_gamma'].astype(float)
            st.subheader("Top Gamma exposures")
            if not df_gamma.empty:
                top_pos = df_gamma.sort_values('net_gamma', ascending=False).head(10)
                top_neg = df_gamma.sort_values('net_gamma', ascending=True).head(10)
                st.write("Top positive net gamma")
                st.dataframe(top_pos[['strikePrice','gamma_exposure_ce','gamma_exposure_pe','net_gamma']])
                st.write("Top negative net gamma")
                st.dataframe(top_neg[['strikePrice','gamma_exposure_ce','gamma_exposure_pe','net_gamma']])
            # download option chain & gamma
            csv = df_chain.to_csv(index=False)
            st.download_button("Download normalized option chain CSV", csv, file_name=f"{symbol}_option_chain.csv", mime="text/csv")

st.markdown("---")
st.caption("DeepSeek full Dhan edition — modular logic in single file. Tell me if your Dhan SDK uses other function names and I'll patch the fetch wrappers (`fetch_intraday_dhan`, `fetch_option_chain_dhan`) to match your SDK exactly.")
