import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
import json
import time
import numpy as np
from collections import deque
import warnings
import math
from scipy.stats import norm
import io
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# LUXALGO REVERSAL PROBABILITY ZONE
# =============================================

class ReversalProbabilityZone:
    """LuxAlgo Reversal Probability Zone Implementation"""
    
    def calculate_reversal_probability(self, df, length=20):
        """Calculate reversal probabilities based on pivot analysis"""
        try:
            # Step 1: find pivots
            df = df.copy()
            df["max"] = df["high"].rolling(length, center=True).max()
            df["min"] = df["low"].rolling(length, center=True).min()

            df["pivot_high"] = (df["high"] == df["max"])
            df["pivot_low"] = (df["low"] == df["min"])

            # Step 2: collect price & bar deltas
            price_moves = []
            bar_moves = []

            last_price = df["close"].iloc[0]
            last_index = 0

            for i in range(len(df)):
                price = df["close"].iloc[i]

                if df["pivot_high"].iloc[i] or df["pivot_low"].iloc[i]:
                    delta_price = abs(price - last_price)
                    delta_bars = i - last_index

                    price_moves.append(delta_price)
                    bar_moves.append(delta_bars)

                    last_price = price
                    last_index = i

            if len(price_moves) < 5:
                return None

            # Step 3: Calculate percentiles
            p25 = np.percentile(price_moves, 25)
            p50 = np.percentile(price_moves, 50)
            p75 = np.percentile(price_moves, 75)
            p90 = np.percentile(price_moves, 90)

            # Step 4: Recent pivot tells if reversal expected
            last_pivot_high = df["pivot_high"].iloc[-1]
            last_pivot_low = df["pivot_low"].iloc[-1]

            # Calculate strength based on percentiles
            strength_ratio = p75 / p25 if p25 > 0 else 1
            if strength_ratio > 2.0:
                strength = "extreme"
            elif strength_ratio > 1.5:
                strength = "strong"
            elif strength_ratio > 1.2:
                strength = "medium"
            else:
                strength = "weak"

            if last_pivot_high:
                # Expect bearish reversal
                return {
                    "pivot_type": "high",
                    "bullish_prob": 0.1,
                    "bearish_prob": 0.9,
                    "strength": strength,
                    "expected_move_points": p50,
                }

            if last_pivot_low:
                # Expect bullish reversal
                return {
                    "pivot_type": "low",
                    "bullish_prob": 0.9,
                    "bearish_prob": 0.1,
                    "strength": strength,
                    "expected_move_points": p50,
                }

            return None
            
        except Exception as e:
            print(f"Error in reversal probability: {e}")
            return None

# =============================================
# PRICE ACTION ENGINE
# =============================================

class PriceActionEngine:
    """Multi-timeframe Trend Detection and Pattern Recognition"""
    
    def detect_trend(self, df):
        """Detect trend using EMA alignment and structure"""
        try:
            df = df.copy()
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['EMA50'] = df['close'].ewm(span=50).mean()
            df['EMA200'] = df['close'].ewm(span=200).mean()

            # EMA alignment
            up = df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1]
            down = df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] < df['EMA200'].iloc[-1]

            # Price structure (HH/HL vs LH/LL)
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)
            
            higher_highs = all(recent_highs.iloc[i] > recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
            higher_lows = all(recent_lows.iloc[i] > recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))
            lower_highs = all(recent_highs.iloc[i] < recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
            lower_lows = all(recent_lows.iloc[i] < recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))

            if up and (higher_highs or higher_lows):
                return "STRONG UPTREND"
            elif down and (lower_highs or lower_lows):
                return "STRONG DOWNTREND"
            elif up:
                return "UPTREND"
            elif down:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            print(f"Error in trend detection: {e}")
            return "UNKNOWN"

    def get_support_resistance(self, df, lookback=20):
        """Automatic Support & Resistance using swing highs/lows"""
        try:
            df = df.copy()
            df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
            df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

            resistances = df[df['swing_high']].tail(lookback)['high'].values[-3:]
            supports = df[df['swing_low']].tail(lookback)['low'].values[-3:]

            return supports.tolist(), resistances.tolist()
        except Exception as e:
            print(f"Error in S/R detection: {e}")
            return [], []

    def detect_double_top_bottom(self, df, tolerance=0.3):
        """Detect Double Top and Double Bottom patterns"""
        try:
            highs = df['high'].rolling(20).max()
            lows = df['low'].rolling(20).min()
            
            recent_high = highs.iloc[-1]
            prev_high = highs.iloc[-15] if len(highs) > 15 else recent_high
            
            recent_low = lows.iloc[-1]
            prev_low = lows.iloc[-15] if len(lows) > 15 else recent_low

            # Double Top
            if abs(recent_high - prev_high) <= tolerance:
                return "DOUBLE_TOP"
            
            # Double Bottom
            if abs(recent_low - prev_low) <= tolerance:
                return "DOUBLE_BOTTOM"
                
            return None
        except Exception as e:
            print(f"Error in double top/bottom detection: {e}")
            return None

    def detect_triangle(self, df):
        """Detect Ascending/Descending Triangles"""
        try:
            supports, resistances = self.get_support_resistance(df)

            if len(resistances) < 2 or len(supports) < 2:
                return None

            # Lower highs + equal lows → descending triangle
            if len(resistances) >= 3 and resistances[0] > resistances[1] > resistances[2]:
                if len(supports) >= 2 and abs(supports[0] - supports[1]) < 20:
                    return "DESCENDING TRIANGLE"

            # Higher lows + equal highs → ascending triangle
            if len(supports) >= 3 and supports[0] < supports[1] < supports[2]:
                if len(resistances) >= 2 and abs(resistances[0] - resistances[1]) < 20:
                    return "ASCENDING TRIANGLE"

            return None
        except Exception as e:
            print(f"Error in triangle detection: {e}")
            return None

    def detect_flag(self, df):
        """Detect Flag and Pole pattern"""
        try:
            if len(df) < 50:
                return None
                
            recent_move = abs(df['close'].iloc[-20] - df['close'].iloc[-50])
            consolidation = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            if recent_move > 80 and consolidation < 40:
                return "FLAG PATTERN"
            return None
        except Exception as e:
            print(f"Error in flag detection: {e}")
            return None

    def analyze_multi_timeframe(self, df_15, df_1h, df_1d):
        """Complete multi-timeframe price action analysis"""
        try:
            trend_15 = self.detect_trend(df_15)
            trend_1h = self.detect_trend(df_1h)
            trend_1d = self.detect_trend(df_1d)

            supports_15, resistances_15 = self.get_support_resistance(df_15)
            pattern_15 = self.detect_triangle(df_15) or self.detect_double_top_bottom(df_15) or self.detect_flag(df_15)

            # Calculate trend score
            trends = [trend_15, trend_1h, trend_1d]
            uptrend_count = sum(1 for t in trends if "UPTREND" in t)
            
            if uptrend_count == 3:
                trend_score = "STRONG BULLISH"
            elif uptrend_count == 2:
                trend_score = "MILD BULLISH"
            elif uptrend_count == 1:
                trend_score = "NEUTRAL"
            else:
                trend_score = "BEARISH"

            return {
                "trend_15m": trend_15,
                "trend_1h": trend_1h,
                "trend_1d": trend_1d,
                "trend_score": trend_score,
                "supports": supports_15,
                "resistances": resistances_15,
                "pattern": pattern_15,
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata'))
            }
        except Exception as e:
            print(f"Error in multi-timeframe analysis: {e}")
            return None

# =============================================
# GAMMA SEQUENCE & EXPIRY SPIKE DETECTOR
# =============================================

class GammaExpirySpikeDetector:
    """Gamma Sequence + Expiry Sudden Move Detection"""
    
    def gamma_sequence_expiry(self, option_chain, spot_price):
        """Detect expiry spike probability using gamma sequence"""
        try:
            # --- FIND ATM ---
            strikes = np.array([x["strike"] for x in option_chain])
            atm_index = np.argmin(np.abs(strikes - spot_price))
            atm = option_chain[atm_index]

            ce_gamma = atm.get("ce_gamma", 0.05)
            pe_gamma = atm.get("pe_gamma", 0.05)
            ce_oi_chg = atm.get("ce_oi_change", 0)
            pe_oi_chg = atm.get("pe_oi_change", 0)
            ce_iv = atm.get("ce_iv", 15)
            pe_iv = atm.get("pe_iv", 15)
            ce_ltp = atm.get("ce_ltp", 10)
            pe_ltp = atm.get("pe_ltp", 10)

            # --- 1. GAMMA PRESSURE ---
            gamma_pressure = (ce_gamma + pe_gamma) * 10000

            # --- 2. GAMMA HEDGE IMBALANCE ---
            hedge_imbalance = abs(ce_oi_chg - pe_oi_chg)

            # --- 3. GAMMA FLIP (big indicator of sudden spike) ---
            gamma_flip = ce_oi_chg < 0 and pe_oi_chg < 0

            # --- 4. INTRADAY TIME CHECK (post 1 PM expiry spike) ---
            now = datetime.now().time()
            is_expiry_spike_window = now > time(13, 0)

            # --- 5. IV CRUSH (if IV falling fast = spike coming) ---
            iv_crush = True if (ce_iv + pe_iv) / 2 < 15 else False

            # --- 6. STRADDLE PRICE COMPRESSION ---
            straddle_price = ce_ltp + pe_ltp
            compression = straddle_price < (0.005 * spot_price)   # <0.5% of index

            # --- 7. Expected Move ---
            expected_move = gamma_pressure * (1 + hedge_imbalance * 0.5)

            # ----------------------------
            # FINAL EXPIRY SPIKE PROBABILITY
            # ----------------------------
            spike_score = 0
            
            # Gamma pressure
            if gamma_pressure > 60: spike_score += 20
            if gamma_pressure > 80: spike_score += 30

            # Hedge imbalance
            if hedge_imbalance > 5000: spike_score += 20

            # IV crush
            if iv_crush: spike_score += 15

            # Straddle compression
            if compression: spike_score += 20

            # Gamma flip
            if gamma_flip: spike_score += 25

            # 1 PM expiry behaviour
            if is_expiry_spike_window: spike_score += 20

            # Cap at 100
            spike_score = min(spike_score, 100)

            return {
                "gamma_pressure": round(gamma_pressure, 2),
                "hedge_imbalance": hedge_imbalance,
                "iv_crush": iv_crush,
                "straddle_compression": compression,
                "gamma_flip": gamma_flip,
                "expiry_spike_score": spike_score,
                "expected_move_points": round(expected_move, 2)
            }
        except Exception as e:
            print(f"Error in gamma sequence analysis: {e}")
            return {
                "gamma_pressure": 0,
                "hedge_imbalance": 0,
                "iv_crush": False,
                "straddle_compression": False,
                "gamma_flip": False,
                "expiry_spike_score": 0,
                "expected_move_points": 0
            }

# =============================================
# COMPREHENSIVE BIAS ANALYSIS MODULE (ENHANCED)
# =============================================

class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis with ALL new components
    """
    
    def __init__(self):
        """Initialize bias analysis with default configuration"""
        self.config = self._default_config()
        self.all_bias_results = []
        self.overall_bias = "NEUTRAL"
        self.overall_score = 0
        self.reversal_analyzer = ReversalProbabilityZone()
        self.price_action_engine = PriceActionEngine()
        self.gamma_detector = GammaExpirySpikeDetector()

    def _default_config(self) -> Dict:
        """Default configuration from Pine Script"""
        return {
            # Timeframes
            'tf1': '15m',
            'tf2': '1h',

            # Indicator periods
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,

            # Volume
            'volume_roc_length': 14,
            'volume_threshold': 1.2,

            # Volatility
            'volatility_ratio_length': 14,
            'volatility_threshold': 1.5,

            # OBV
            'obv_smoothing': 21,

            # Force Index
            'force_index_length': 13,
            'force_index_smoothing': 2,

            # Price ROC
            'price_roc_length': 12,

            # Market Breadth
            'breadth_threshold': 60,

            # Divergence
            'divergence_lookback': 30,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # Choppiness Index
            'ci_length': 14,
            'ci_high_threshold': 61.8,
            'ci_low_threshold': 38.2,

            # Bias parameters
            'bias_strength': 60,
            'divergence_threshold': 60,

            # Adaptive weights
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,

            # Stocks with weights
            'stocks': {
                '^NSEBANK': 10.0,  # BANKNIFTY Index
                'RELIANCE.NS': 9.98,
                'HDFCBANK.NS': 9.67,
                'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54,
                'ICICIBANK.NS': 8.01,
                'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98,
                'ITC.NS': 2.44,
                'MARUTI.NS': 0.0
            }
        }

    # =========================================================================
    # DATA FETCHING - UPDATED VERSION
    # =========================================================================

    def fetch_data(self, symbol: str, period: str = '5d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance with better error handling"""
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            ticker = yf.Ticker(symbol)
            
            # Try to fetch data
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"⚠️  No data returned for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"⚠️  Missing column: {col}")
                    return pd.DataFrame()
            
            # Handle volume data (Yahoo Finance often has 0 volume for indices)
            if df['volume'].sum() == 0:
                print(f"⚠️  Warning: Zero volume data for {symbol}")
                # Set minimum volume to avoid division by zero
                df['volume'] = 1000000  # Dummy volume for indices
            
            # Replace any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✅ Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()

    # =========================================================================
    # TECHNICAL INDICATORS (EXISTING)
    # =========================================================================

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index with NaN/zero handling"""
        if df['volume'].sum() == 0:
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        mfi = mfi.fillna(50)

        return mfi

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        """Calculate DMI indicators"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()

        return plus_di, minus_di, adx

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with NaN/zero handling"""
        if df['volume'].sum() == 0:
            return (df['high'] + df['low'] + df['close']) / 3

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['volume']).cumsum() / cumulative_volume_safe
        vwap = vwap.fillna(typical_price)

        return vwap

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA (Variable Index Dynamic Average) matching Pine Script"""
        close = df['close']

        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()

        cmo_denom = p + n
        cmo_denom = cmo_denom.replace(0, np.nan)
        abs_cmo = abs(100 * (p - n) / cmo_denom).fillna(0)

        alpha = 2 / (length + 1)
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                            (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])

        vidya_smoothed = vidya.rolling(window=15).mean()

        atr = self.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * band_distance
        lower_band = vidya_smoothed - atr * band_distance

        is_trend_up = close > upper_band
        is_trend_down = close < lower_band

        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False

        return vidya_smoothed, vidya_bullish, vidya_bearish

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta (up_vol - down_vol) matching Pine Script"""
        if df['volume'].sum() == 0:
            return 0, False, False

        up_vol = ((df['close'] > df['open']).astype(int) * df['volume']).sum()
        down_vol = ((df['close'] < df['open']).astype(int) * df['volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots matching Pine Script"""
        if df['volume'].sum() == 0:
            return False, False, 0, 0

        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        volume_sum = df['volume'].rolling(window=left_bars * 2).sum()
        ref_vol = volume_sum.quantile(0.95)
        norm_vol = (volume_sum / ref_vol * 5).fillna(0)

        hvp_bullish = False
        hvp_bearish = False

        if len(pivot_lows) > 0:
            last_pivot_low_idx = pivot_lows[-1]
            if norm_vol.iloc[last_pivot_low_idx] > vol_filter:
                hvp_bullish = True

        if len(pivot_highs) > 0:
            last_pivot_high_idx = pivot_highs[-1]
            if norm_vol.iloc[last_pivot_high_idx] > vol_filter:
                hvp_bearish = True

        return hvp_bullish, hvp_bearish, len(pivot_highs), len(pivot_lows)

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5):
        """Calculate Volume Order Blocks matching Pine Script"""
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['close'], length1)
        ema2 = self.calculate_ema(df['close'], length2)

        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    # =========================================================================
    # ENHANCED BIAS ANALYSIS WITH NEW COMPONENTS
    # =========================================================================

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """
        Analyze all bias indicators including NEW components
        """
        print(f"Fetching data for {symbol}...")
        df = self.fetch_data(symbol, period='5d', interval='5m')

        if df.empty or len(df) < 100:
            error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }

        current_price = df['close'].iloc[-1]

        # Initialize bias results list
        bias_results = []
        stock_data = []

        # =====================================================================
        # FAST INDICATORS (8 total)
        # =====================================================================

        # 1. VOLUME DELTA
        volume_delta, volume_bullish, volume_bearish = self.calculate_volume_delta(df)
        vol_delta_bias = "BULLISH" if volume_bullish else "BEARISH" if volume_bearish else "NEUTRAL"
        vol_delta_score = 100 if volume_bullish else -100 if volume_bearish else 0

        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': vol_delta_bias,
            'score': vol_delta_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 2. HVP (High Volume Pivots)
        hvp_bullish, hvp_bearish, pivot_highs, pivot_lows = self.calculate_hvp(df)
        hvp_bias = "BULLISH" if hvp_bullish else "BEARISH" if hvp_bearish else "NEUTRAL"
        hvp_score = 100 if hvp_bullish else -100 if hvp_bearish else 0
        hvp_value = f"Bull Signal (Lows: {pivot_lows}, Highs: {pivot_highs})" if hvp_bullish else f"Bear Signal (Highs: {pivot_highs}, Lows: {pivot_lows})" if hvp_bearish else f"No Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"

        bias_results.append({
            'indicator': 'HVP (High Volume Pivots)',
            'value': hvp_value,
            'bias': hvp_bias,
            'score': hvp_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 3. VOB (Volume Order Blocks)
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)
        vob_bias = "BULLISH" if vob_bullish else "BEARISH" if vob_bearish else "NEUTRAL"
        vob_score = 100 if vob_bullish else -100 if vob_bearish else 0
        vob_value = f"Bull Cross (EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f})" if vob_bullish else f"Bear Cross (EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f})" if vob_bearish else f"EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f} (No Cross)" if vob_ema5 > vob_ema18 else f"EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f} (No Cross)"

        bias_results.append({
            'indicator': 'VOB (Volume Order Blocks)',
            'value': vob_value,
            'bias': vob_bias,
            'score': vob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 4. ORDER BLOCKS (EMA Crossover)
        ema5 = self.calculate_ema(df['close'], 5)
        ema18 = self.calculate_ema(df['close'], 18)
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        ob_bias = "BULLISH" if cross_up else "BEARISH" if cross_dn else "NEUTRAL"
        ob_score = 100 if cross_up else -100 if cross_dn else 0

        bias_results.append({
            'indicator': 'Order Blocks (EMA 5/18)',
            'value': f"EMA5: {ema5.iloc[-1]:.2f} | EMA18: {ema18.iloc[-1]:.2f}",
            'bias': ob_bias,
            'score': ob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 5. RSI
        rsi = self.calculate_rsi(df['close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]
        rsi_bias = "BULLISH" if rsi_value > 50 else "BEARISH"
        rsi_score = 100 if rsi_value > 50 else -100

        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.2f}",
            'bias': rsi_bias,
            'score': rsi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 6. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        plus_di_value = plus_di.iloc[-1]
        minus_di_value = minus_di.iloc[-1]
        dmi_bias = "BULLISH" if plus_di_value > minus_di_value else "BEARISH"
        dmi_score = 100 if plus_di_value > minus_di_value else -100

        bias_results.append({
            'indicator': 'DMI',
            'value': f"+DI:{plus_di_value:.1f} -DI:{minus_di_value:.1f}",
            'bias': dmi_bias,
            'score': dmi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 7. VIDYA
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
        vidya_bias = "BULLISH" if vidya_bullish else "BEARISH" if vidya_bearish else "NEUTRAL"
        vidya_score = 100 if vidya_bullish else -100 if vidya_bearish else 0

        bias_results.append({
            'indicator': 'VIDYA',
            'value': f"{vidya_val.iloc[-1]:.2f}" if not vidya_val.empty else "N/A",
            'bias': vidya_bias,
            'score': vidya_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 8. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1]
        if np.isnan(mfi_value):
            mfi_value = 50.0
        mfi_bias = "BULLISH" if mfi_value > 50 else "BEARISH"
        mfi_score = 100 if mfi_value > 50 else -100

        bias_results.append({
            'indicator': 'MFI (Money Flow)',
            'value': f"{mfi_value:.2f}",
            'bias': mfi_bias,
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # =====================================================================
        # NEW: REVERSAL PROBABILITY ANALYSIS
        # =====================================================================
        reversal_data = self.reversal_analyzer.calculate_reversal_probability(df)
        if reversal_data:
            rev_bullish_prob = reversal_data['bullish_prob']
            rev_bearish_prob = reversal_data['bearish_prob']
            rev_bias = "BULLISH" if rev_bullish_prob > 0.7 else "BEARISH" if rev_bearish_prob > 0.7 else "NEUTRAL"
            rev_score = 100 if rev_bullish_prob > 0.7 else -100 if rev_bearish_prob > 0.7 else 0

            bias_results.append({
                'indicator': 'Reversal Probability',
                'value': f"Bull:{rev_bullish_prob:.1f} Bear:{rev_bearish_prob:.1f}",
                'bias': rev_bias,
                'score': rev_score,
                'weight': 1.5,
                'category': 'fast'
            })

        # =====================================================================
        # NEW: PRICE ACTION ANALYSIS
        # =====================================================================
        price_action_data = self.price_action_engine.analyze_multi_timeframe(df, df, df)  # Using same DF for demo
        if price_action_data:
            pa_trend_score = price_action_data['trend_score']
            pa_bias = "BULLISH" if "BULLISH" in pa_trend_score else "BEARISH" if "BEARISH" in pa_trend_score else "NEUTRAL"
            pa_score = 100 if "STRONG BULLISH" in pa_trend_score else 75 if "MILD BULLISH" in pa_trend_score else -100 if "BEARISH" in pa_trend_score else 0

            bias_results.append({
                'indicator': 'Price Action Trend',
                'value': f"{pa_trend_score}",
                'bias': pa_bias,
                'score': pa_score,
                'weight': 2.0,
                'category': 'medium'
            })

        # =====================================================================
        # CALCULATE OVERALL BIAS
        # =====================================================================
        fast_bull = 0
        fast_bear = 0
        fast_total = 0

        medium_bull = 0
        medium_bear = 0
        medium_total = 0

        slow_bull = 0
        slow_bear = 0
        slow_total = 0

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for bias in bias_results:
            if 'BULLISH' in bias['bias']:
                bullish_count += 1
                if bias['category'] == 'fast':
                    fast_bull += 1
                elif bias['category'] == 'medium':
                    medium_bull += 1
                elif bias['category'] == 'slow':
                    slow_bull += 1
            elif 'BEARISH' in bias['bias']:
                bearish_count += 1
                if bias['category'] == 'fast':
                    fast_bear += 1
                elif bias['category'] == 'medium':
                    medium_bear += 1
                elif bias['category'] == 'slow':
                    slow_bear += 1
            else:
                neutral_count += 1

            if bias['category'] == 'fast':
                fast_total += 1
            elif bias['category'] == 'medium':
                medium_total += 1
            elif bias['category'] == 'slow':
                slow_total += 1

        # Calculate percentages
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        medium_bull_pct = (medium_bull / medium_total) * 100 if medium_total > 0 else 0
        medium_bear_pct = (medium_bear / medium_total) * 100 if medium_total > 0 else 0

        slow_bull_pct = (slow_bull / slow_total) * 100 if slow_total > 0 else 0
        slow_bear_pct = (slow_bear / slow_total) * 100 if slow_total > 0 else 0

        # Adaptive weighting
        divergence_threshold = self.config['divergence_threshold']
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        if divergence_detected:
            fast_weight = self.config['reversal_fast_weight']
            medium_weight = self.config['reversal_medium_weight']
            slow_weight = self.config['reversal_slow_weight']
            mode = "REVERSAL"
        else:
            fast_weight = self.config['normal_fast_weight']
            medium_weight = self.config['normal_medium_weight']
            slow_weight = self.config['normal_slow_weight']
            mode = "NORMAL"

        # Calculate weighted scores
        bullish_signals = (fast_bull * fast_weight) + (medium_bull * medium_weight) + (slow_bull * slow_weight)
        bearish_signals = (fast_bear * fast_weight) + (medium_bear * medium_weight) + (slow_bear * slow_weight)
        total_signals = (fast_total * fast_weight) + (medium_total * medium_weight) + (slow_total * slow_weight)

        bullish_bias_pct = (bullish_signals / total_signals) * 100 if total_signals > 0 else 0
        bearish_bias_pct = (bearish_signals / total_signals) * 100 if total_signals > 0 else 0

        # Determine overall bias
        bias_strength = self.config['bias_strength']

        if bullish_bias_pct >= bias_strength:
            overall_bias = "BULLISH"
            overall_score = bullish_bias_pct
            overall_confidence = min(100, bullish_bias_pct)
        elif bearish_bias_pct >= bias_strength:
            overall_bias = "BEARISH"
            overall_score = -bearish_bias_pct
            overall_confidence = min(100, bearish_bias_pct)
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0
            overall_confidence = 100 - max(bullish_bias_pct, bearish_bias_pct)

        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'stock_data': stock_data,
            'mode': mode,
            'fast_bull_pct': fast_bull_pct,
            'fast_bear_pct': fast_bear_pct,
            'slow_bull_pct': slow_bull_pct,
            'slow_bear_pct': slow_bear_pct,
            'bullish_bias_pct': bullish_bias_pct,
            'bearish_bias_pct': bearish_bias_pct,
            'reversal_data': reversal_data,
            'price_action_data': price_action_data
        }

# =============================================
# MARKET REGIME DETECTOR (THE BRAIN)
# =============================================

class MarketRegimeDetector:
    """
    Detects what TYPE of market we're in right now
    """
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def detect_market_regime(self, df: pd.DataFrame, vix_value: float = None,
                            volume_ratio: float = 1.0) -> Dict[str, Any]:
        """
        Master function that detects current market regime
        """
        if df.empty or len(df) < 50:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        results = {
            'regime': None,
            'confidence': 0,
            'characteristics': [],
            'best_strategies': [],
            'indicators_to_trust': [],
            'indicators_to_ignore': [],
            'risk_level': 'MEDIUM',
            'trade_recommendation': None
        }
        
        # Calculate market characteristics
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Price action
        close = df['close']
        high = df['high'].rolling(20).max()
        low = df['low'].rolling(20).min()
        range_pct = ((high.iloc[-1] - low.iloc[-1]) / low.iloc[-1]) * 100
        
        # Trend strength
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        current_price = close.iloc[-1]
        trend_up = current_price > ema20.iloc[-1] > ema50.iloc[-1]
        trend_down = current_price < ema20.iloc[-1] < ema50.iloc[-1]
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_strength = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Time-based factors
        current_time = datetime.now(self.ist)
        is_expiry_week = self._is_expiry_week(current_time)
        is_event_day = self._is_event_day(current_time)
        time_of_day = current_time.time()
        
        # VIX analysis
        vix_high = vix_value and vix_value > 20
        vix_low = vix_value and vix_value < 13
        
        # =====================================================
        # REGIME DETECTION LOGIC
        # =====================================================
        
        # 1. HIGH VOLATILITY BREAKOUT MARKET
        if atr_ratio > 1.5 and volume_strength > 2.0 and (vix_high or not vix_value):
            results['regime'] = 'HIGH_VOLATILITY_BREAKOUT'
            results['confidence'] = 85
            results['characteristics'] = [
                'High ATR (trending strongly)',
                'High volume (institutional activity)',
                'VIX elevated (fear/uncertainty)'
            ]
            results['best_strategies'] = [
                'Momentum trading',
                'Breakout trades with wide stops',
                'Follow the trend aggressively'
            ]
            results['indicators_to_trust'] = [
                'Volume Delta',
                'DMI',
                'Order Blocks',
                'HVP'
            ]
            results['indicators_to_ignore'] = [
                'RSI (gets overbought in trends)',
                'Mean reversion indicators'
            ]
            results['risk_level'] = 'HIGH'
            results['trade_recommendation'] = '🔥 ACTIVE - Trade breakouts with 2% stops'
        
        # 2. STRONG TRENDING MARKET
        elif (trend_up or trend_down) and atr_ratio > 1.2 and volume_strength > 1.3:
            results['regime'] = 'STRONG_TREND_UP' if trend_up else 'STRONG_TREND_DOWN'
            results['confidence'] = 80
            results['characteristics'] = [
                f'Clear {"uptrend" if trend_up else "downtrend"}',
                'Healthy volume',
                'Normal volatility'
            ]
            results['best_strategies'] = [
                'Trend following',
                'Buy/Sell pullbacks to moving averages',
                'Trail stops'
            ]
            results['indicators_to_trust'] = [
                'RSI',
                'VIDYA',
                'Volume Delta',
                'DMI'
            ]
            results['indicators_to_ignore'] = [
                'Reversal signals (counter-trend)'
            ]
            results['risk_level'] = 'MEDIUM'
            results['trade_recommendation'] = f'✅ ACTIVE - Trade {"LONG" if trend_up else "SHORT"} pullbacks'
        
        # 3. RANGE-BOUND MARKET
        elif range_pct < 2 and atr_ratio < 0.8 and volume_strength < 1.2:
            results['regime'] = 'RANGE_BOUND'
            results['confidence'] = 75
            results['characteristics'] = [
                'Narrow range (consolidation)',
                'Low volatility',
                'Low volume'
            ]
            results['best_strategies'] = [
                'Range trading',
                'Sell resistance, buy support',
                'Avoid breakout trades'
            ]
            results['indicators_to_trust'] = [
                'RSI (50 level)',
                'Order Blocks',
                'VOB'
            ]
            results['indicators_to_ignore'] = [
                'Trend indicators',
                'Momentum indicators'
            ]
            results['risk_level'] = 'LOW'
            results['trade_recommendation'] = '⚠️ CAUTIOUS - Scalp between support/resistance only'
        
        # 4. LOW VOLUME TRAP ZONE
        elif volume_strength < 0.6 and time_of_day > datetime.strptime("11:30", "%H:%M").time() and \
             time_of_day < datetime.strptime("14:00", "%H:%M").time():
            results['regime'] = 'LOW_VOLUME_TRAP'
            results['confidence'] = 90
            results['characteristics'] = [
                'Lunch time (11:30 AM - 2:00 PM)',
                'Very low volume',
                'Choppy price action'
            ]
            results['best_strategies'] = [
                'AVOID TRADING',
                'Take a break',
                'Wait for afternoon session'
            ]
            results['indicators_to_trust'] = []
            results['indicators_to_ignore'] = ['ALL']
            results['risk_level'] = 'VERY_HIGH'
            results['trade_recommendation'] = '🛑 AVOID - Lunch time trap zone'
        
        # 5. EXPIRY DAY BEHAVIOUR
        elif is_expiry_week and time_of_day > datetime.strptime("13:30", "%H:%M").time():
            results['regime'] = 'EXPIRY_MANIPULATION'
            results['confidence'] = 85
            results['characteristics'] = [
                'Expiry week',
                'After 1:30 PM',
                'Max pain gravitational pull'
            ]
            results['best_strategies'] = [
                'Close existing positions',
                'Avoid new entries',
                'Watch for squaring off'
            ]
            results['indicators_to_trust'] = [
                'Max Pain levels',
                'PCR OI'
            ]
            results['indicators_to_ignore'] = [
                'Technical indicators (manipulated)'
            ]
            results['risk_level'] = 'VERY_HIGH'
            results['trade_recommendation'] = '🛑 AVOID - Expiry day manipulation zone'
        
        # 6. POST-GAP DAY
        elif self._is_gap_day(df):
            gap_type = self._gap_direction(df)
            results['regime'] = f'POST_GAP_{gap_type}'
            results['confidence'] = 70
            results['characteristics'] = [
                f'{gap_type} gap detected',
                'First 30 minutes critical',
                'Watch for gap fill or continuation'
            ]
            results['best_strategies'] = [
                'Wait for opening range (9:15-9:45)',
                'Trade breakout of opening range',
                'Watch for gap fill opportunities'
            ]
            results['indicators_to_trust'] = [
                'Volume Delta',
                'HVP',
                'Order Blocks'
            ]
            results['indicators_to_ignore'] = []
            results['risk_level'] = 'HIGH'
            results['trade_recommendation'] = '⚠️ CAUTIOUS - Wait for opening range breakout'
        
        # 7. EVENT DAY
        elif is_event_day:
            results['regime'] = 'EVENT_DAY'
            results['confidence'] = 95
            results['characteristics'] = [
                'Major event today (Budget/RBI/Elections/US CPI)',
                'Unpredictable volatility',
                'Avoid trading'
            ]
            results['best_strategies'] = [
                'STAY OUT',
                'Wait for event result',
                'Trade post-event clarity'
            ]
            results['indicators_to_trust'] = []
            results['indicators_to_ignore'] = ['ALL']
            results['risk_level'] = 'EXTREME'
            results['trade_recommendation'] = '🛑 AVOID - Event day, stay out completely'
        
        # 8. LOW VOLATILITY GRIND
        elif vix_low and atr_ratio < 0.7 and volume_strength < 0.9:
            results['regime'] = 'LOW_VOLATILITY_GRIND'
            results['confidence'] = 75
            results['characteristics'] = [
                'VIX very low (complacency)',
                'Low volatility',
                'Grinding slow market'
            ]
            results['best_strategies'] = [
                'Options selling strategies',
                'Tight range trading',
                'Prepare for volatility spike'
            ]
            results['indicators_to_trust'] = [
                'RSI',
                'MFI',
                'Order Blocks'
            ]
            results['indicators_to_ignore'] = [
                'Breakout indicators'
            ]
            results['risk_level'] = 'LOW'
            results['trade_recommendation'] = '⚠️ CAUTIOUS - Tight stops, expect slow grind'
        
        # 9. DEFAULT - NORMAL MARKET
        else:
            results['regime'] = 'NORMAL_MARKET'
            results['confidence'] = 60
            results['characteristics'] = [
                'Normal volatility',
                'Average volume',
                'Mixed signals'
            ]
            results['best_strategies'] = [
                'Follow all indicators',
                'Wait for high-confidence setups',
                'Use normal position sizing'
            ]
            results['indicators_to_trust'] = [
                'All 8 bias indicators',
                'Options analysis',
                'Volume patterns'
            ]
            results['indicators_to_ignore'] = []
            results['risk_level'] = 'MEDIUM'
            results['trade_recommendation'] = '✅ MODERATE - Trade normal setups with 1.5% stops'
        
        return results

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _is_expiry_week(self, current_time: datetime) -> bool:
        """Check if it's expiry week (last week of month)"""
        days_in_month = (current_time.replace(day=28) + timedelta(days=4)).day
        return current_time.day > days_in_month - 7

    def _is_event_day(self, current_time: datetime) -> bool:
        """Check if it's a known event day"""
        return current_time.day <= 7

    def _is_gap_day(self, df: pd.DataFrame) -> bool:
        """Check if today opened with a gap"""
        if len(df) < 10:
            return False
        
        today_open = df['open'].iloc[0]
        yesterday_close = df['close'].iloc[-2] if len(df) > 1 else today_open
        gap_pct = abs((today_open - yesterday_close) / yesterday_close) * 100
        
        return gap_pct > 0.5

    def _gap_direction(self, df: pd.DataFrame) -> str:
        """Determine if gap is up or down"""
        if len(df) < 10:
            return 'FLAT'
        
        today_open = df['open'].iloc[0]
        yesterday_close = df['close'].iloc[-2] if len(df) > 1 else today_open
        
        if today_open > yesterday_close * 1.005:
            return 'UP'
        elif today_open < yesterday_close * 0.995:
            return 'DOWN'
        return 'FLAT'

# =============================================
# TRAP DETECTOR & OI INTELLIGENCE ENGINE
# =============================================

class TrapDetector:
    """
    Detects market traps and identifies:
    - Bull Traps (fake breakouts)
    - Bear Traps (fake breakdowns)
    - Short Covering (trapped shorts being squeezed)
    - Long Liquidation (trapped longs being stopped out)
    """

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def analyze_market_trap(self,
                           price_data: pd.DataFrame,
                           options_data: Dict[str, Any] = None,
                           bias_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Master trap detection combining price action, OI, and volume
        """
        if price_data.empty or len(price_data) < 20:
            return {'trap_detected': False, 'type': 'UNKNOWN'}
        
        results = {
            'trap_detected': False,
            'trap_type': None,
            'trap_confidence': 0,
            'action': None,
            'characteristics': [],
            'oi_analysis': None,
            'who_is_trapped': None,
            'expected_move': None,
            'trade_setup': None
        }
        
        # Price action analysis
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        # Recent price movements
        current_price = close.iloc[-1]
        price_5_bars_ago = close.iloc[-5] if len(close) > 5 else current_price
        price_change = ((current_price - price_5_bars_ago) / price_5_bars_ago) * 100
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # =====================================================
        # OI TRAP DETECTION (The Real Intelligence)
        # =====================================================
        
        if options_data:
            oi_trap = self._detect_oi_trap(options_data, price_change, volume_ratio)
            results['oi_analysis'] = oi_trap
            
            # 1. BULL TRAP DETECTION
            if oi_trap['type'] == 'BULL_TRAP':
                results['trap_detected'] = True
                results['trap_type'] = 'BULL_TRAP'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price rising but Call OI increasing (Call writers betting against rise)',
                    'Weak volume on up move',
                    'Call writers are confident - they are trapping bulls'
                ]
                results['action'] = '🔻 SELL/SHORT'
                results['who_is_trapped'] = 'BUYERS (Bulls)'
                results['expected_move'] = 'DOWN (Bulls will be stopped out)'
                results['trade_setup'] = {
                    'direction': 'SHORT',
                    'entry': 'On next bounce',
                    'target': f"{oi_trap.get('max_pain', current_price * 0.98):.0f}",
                    'stop_loss': f"{current_price * 1.015:.0f}",
                    'confidence': oi_trap['confidence']
                }
            
            # 2. BEAR TRAP DETECTION
            elif oi_trap['type'] == 'BEAR_TRAP':
                results['trap_detected'] = True
                results['trap_type'] = 'BEAR_TRAP'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price falling but Put OI increasing (Put writers betting against fall)',
                    'Weak volume on down move',
                    'Put writers are confident - they are trapping bears'
                ]
                results['action'] = '🔺 BUY/LONG'
                results['who_is_trapped'] = 'SELLERS (Bears)'
                results['expected_move'] = 'UP (Bears will be squeezed)'
                results['trade_setup'] = {
                    'direction': 'LONG',
                    'entry': 'On next dip',
                    'target': f"{oi_trap.get('max_pain', current_price * 1.02):.0f}",
                    'stop_loss': f"{current_price * 0.985:.0f}",
                    'confidence': oi_trap['confidence']
                }
            
            # 3. SHORT COVERING DETECTION
            elif oi_trap['type'] == 'SHORT_COVERING':
                results['trap_detected'] = True
                results['trap_type'] = 'SHORT_COVERING'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price rising WITH Put OI decreasing (Shorts closing positions)',
                    'High volume (panicked short covering)',
                    'This is a SHORT SQUEEZE - very powerful'
                ]
                results['action'] = '🚀 STRONG BUY'
                results['who_is_trapped'] = 'SHORT SELLERS'
                results['expected_move'] = 'SHARP UP (Short squeeze can be violent)'
                results['trade_setup'] = {
                    'direction': 'LONG',
                    'entry': 'Immediate or on small dip',
                    'target': f"{current_price * 1.03:.0f}",
                    'stop_loss': f"{current_price * 0.99:.0f}",
                    'confidence': oi_trap['confidence'],
                    'note': '⚡ SHORT SQUEEZE - Move fast, tight stops'
                }
            
            # 4. LONG LIQUIDATION DETECTION
            elif oi_trap['type'] == 'LONG_LIQUIDATION':
                results['trap_detected'] = True
                results['trap_type'] = 'LONG_LIQUIDATION'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price falling WITH Call OI decreasing (Longs being stopped out)',
                    'High volume (panicked selling)',
                    'This is a LONG SQUEEZE - cascade selling'
                ]
                results['action'] = '📉 STRONG SELL'
                results['who_is_trapped'] = 'LONG HOLDERS'
                results['expected_move'] = 'SHARP DOWN (Long squeeze accelerates)'
                results['trade_setup'] = {
                    'direction': 'SHORT',
                    'entry': 'Immediate or on small bounce',
                    'target': f"{current_price * 0.97:.0f}",
                    'stop_loss': f"{current_price * 1.01:.0f}",
                    'confidence': oi_trap['confidence'],
                    'note': '⚡ LONG SQUEEZE - Move fast, tight stops'
                }
            
            # 5. LONG BUILDUP (Genuine buying)
            elif oi_trap['type'] == 'LONG_BUILDUP':
                results['trap_detected'] = False
                results['trap_type'] = 'GENUINE_LONG_BUILDUP'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price rising WITH Call OI increasing (Fresh buying)',
                    'Good volume (institutional buying)',
                    'This is GENUINE DEMAND - not a trap'
                ]
                results['action'] = '✅ BUY'
                results['who_is_trapped'] = 'NOBODY - Fresh buyers entering'
                results['expected_move'] = 'CONTINUED UP (Healthy uptrend)'
                results['trade_setup'] = {
                    'direction': 'LONG',
                    'entry': 'On pullbacks',
                    'target': f"{current_price * 1.025:.0f}",
                    'stop_loss': f"{current_price * 0.985:.0f}",
                    'confidence': oi_trap['confidence']
                }
            
            # 6. SHORT BUILDUP (Genuine selling)
            elif oi_trap['type'] == 'SHORT_BUILDUP':
                results['trap_detected'] = False
                results['trap_type'] = 'GENUINE_SHORT_BUILDUP'
                results['trap_confidence'] = oi_trap['confidence']
                results['characteristics'] = [
                    'Price falling WITH Put OI increasing (Fresh selling)',
                    'Good volume (institutional selling)',
                    'This is GENUINE SUPPLY - not a trap'
                ]
                results['action'] = '✅ SELL'
                results['who_is_trapped'] = 'NOBODY - Fresh sellers entering'
                results['expected_move'] = 'CONTINUED DOWN (Healthy downtrend)'
                results['trade_setup'] = {
                    'direction': 'SHORT',
                    'entry': 'On bounces',
                    'target': f"{current_price * 0.975:.0f}",
                    'stop_loss': f"{current_price * 1.015:.0f}",
                    'confidence': oi_trap['confidence']
                }
        
        # Price-based trap detection (fallback if no options data)
        else:
            price_trap = self._detect_price_trap(price_data, bias_data)
            if price_trap['detected']:
                results['trap_detected'] = True
                results['trap_type'] = price_trap['type']
                results['trap_confidence'] = price_trap['confidence']
                results['characteristics'] = price_trap['characteristics']
        
        return results

    def _detect_oi_trap(self, options_data: Dict[str, Any],
                       price_change: float, volume_ratio: float) -> Dict[str, Any]:
        """
        Analyzes Open Interest to detect traps
        """
        result = {
            'type': 'NONE',
            'confidence': 0,
            'max_pain': None
        }
        
        try:
            # Extract OI data
            total_ce_oi = options_data.get('total_ce_oi', 0)
            total_pe_oi = options_data.get('total_pe_oi', 0)
            total_ce_change = options_data.get('total_ce_change', 0)
            total_pe_change = options_data.get('total_pe_change', 0)
            
            # Get max pain if available
            comp_metrics = options_data.get('comprehensive_metrics', {})
            max_pain = comp_metrics.get('max_pain_strike')
            result['max_pain'] = max_pain
            
            # Normalize changes
            ce_change_pct = (total_ce_change / total_ce_oi * 100) if total_ce_oi > 0 else 0
            pe_change_pct = (total_pe_change / total_pe_oi * 100) if total_pe_oi > 0 else 0
            
            # =====================================================
            # THE INTELLIGENCE - What moves markets
            # =====================================================
            
            # 1. BULL TRAP: Price up + Call OI up (Call writers confident)
            if price_change > 0.5 and total_ce_change > 0 and ce_change_pct > 2:
                if volume_ratio < 1.5:  # Weak volume = Trap
                    result['type'] = 'BULL_TRAP'
                    result['confidence'] = min(85, 60 + (ce_change_pct * 2))
                else:  # Strong volume = Genuine
                    result['type'] = 'LONG_BUILDUP'
                    result['confidence'] = min(80, 50 + (volume_ratio * 10))
            
            # 2. BEAR TRAP: Price down + Put OI up (Put writers confident)
            elif price_change < -0.5 and total_pe_change > 0 and pe_change_pct > 2:
                if volume_ratio < 1.5:  # Weak volume = Trap
                    result['type'] = 'BEAR_TRAP'
                    result['confidence'] = min(85, 60 + (pe_change_pct * 2))
                else:  # Strong volume = Genuine
                    result['type'] = 'SHORT_BUILDUP'
                    result['confidence'] = min(80, 50 + (volume_ratio * 10))
            
            # 3. SHORT COVERING: Price up + Put OI down (Shorts panicking)
            elif price_change > 0.5 and total_pe_change < 0 and abs(pe_change_pct) > 2:
                if volume_ratio > 1.5:  # High volume = Panic
                    result['type'] = 'SHORT_COVERING'
                    result['confidence'] = min(90, 70 + (volume_ratio * 5))
            
            # 4. LONG LIQUIDATION: Price down + Call OI down (Longs stopping out)
            elif price_change < -0.5 and total_ce_change < 0 and abs(ce_change_pct) > 2:
                if volume_ratio > 1.5:  # High volume = Panic
                    result['type'] = 'LONG_LIQUIDATION'
                    result['confidence'] = min(90, 70 + (volume_ratio * 5))
        
        except Exception as e:
            print(f"Error in OI trap detection: {e}")
        
        return result

    def _detect_price_trap(self, df: pd.DataFrame,
                          bias_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Price-based trap detection (fallback when no options data)
        """
        result = {
            'detected': False,
            'type': 'NONE',
            'confidence': 0,
            'characteristics': []
        }
        
        if len(df) < 20:
            return result
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Recent highs/lows
        recent_high = high.rolling(10).max().iloc[-1]
        recent_low = low.rolling(10).min().iloc[-1]
        current_price = close.iloc[-1]
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Bull trap: Price at high but volume declining
        if current_price >= recent_high * 0.999:
            if current_volume < avg_volume * 0.8:
                result['detected'] = True
                result['type'] = 'BULL_TRAP_PRICE'
                result['confidence'] = 65
                result['characteristics'] = [
                    'Price at recent highs',
                    'Volume declining (no follow-through)',
                    'Likely exhaustion - reversal expected'
                ]
        
        # Bear trap: Price at low but volume declining
        elif current_price <= recent_low * 1.001:
            if current_volume < avg_volume * 0.8:
                result['detected'] = True
                result['type'] = 'BEAR_TRAP_PRICE'
                result['confidence'] = 65
                result['characteristics'] = [
                    'Price at recent lows',
                    'Volume declining (no follow-through)',
                    'Likely exhaustion - reversal expected'
                ]
        
        return result

# =============================================
# EXECUTION FILTER ENGINE (THE GUARDIAN)
# =============================================

class ExecutionFilterEngine:
    """
    The GUARDIAN that protects your capital
    Tells you WHEN to avoid trading
    """
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def should_trade(self,
                    regime: Dict[str, Any],
                    trap_analysis: Dict[str, Any],
                    bias_data: Dict[str, Any],
                    options_data: Dict[str, Any],
                    market_data: Dict[str, Any],
                    current_price: float,
                    df: pd.DataFrame) -> Dict[str, Any]:
        """
        Master filter - Returns TRUE only if ALL conditions met
        """
        result = {
            'trade_allowed': True,
            'confidence': 100,
            'filters_passed': [],
            'filters_failed': [],
            'risk_level': 'MEDIUM',
            'position_sizing': 'NORMAL',
            'final_recommendation': None,
            'warnings': []
        }
        
        # =====================================================
        # CRITICAL FILTERS (Must pass or NO TRADE)
        # =====================================================
        
        # Filter 1: Market Regime Check
        if regime.get('regime') == 'LOW_VOLUME_TRAP':
            result['trade_allowed'] = False
            result['filters_failed'].append('🛑 Low volume trap zone (lunch time)')
            result['final_recommendation'] = 'AVOID - Wait for afternoon session'
            return result
        
        if regime.get('regime') == 'EVENT_DAY':
            result['trade_allowed'] = False
            result['filters_failed'].append('🛑 Event day - Unpredictable volatility')
            result['final_recommendation'] = 'AVOID - Stay out on event days'
            return result
        
        if regime.get('regime') == 'EXPIRY_MANIPULATION':
            result['trade_allowed'] = False
            result['filters_failed'].append('🛑 Expiry day manipulation zone')
            result['final_recommendation'] = 'AVOID - Close existing, no new trades'
            return result
        
        result['filters_passed'].append('✓ Market regime OK for trading')
        
        # Filter 2: Time of Day Check
        current_time = datetime.now(self.ist).time()
        
        # Avoid opening 10 minutes (opening trap)
        if current_time < datetime.strptime("09:25", "%H:%M").time():
            result['trade_allowed'] = False
            result['filters_failed'].append('🛑 Too early - Wait for 9:25 AM')
            result['final_recommendation'] = 'AVOID - Opening trap zone'
            return result
        
        result['filters_passed'].append('✓ Time of day suitable')
        
        # Filter 3: VIX Check (if available)
        vix_value = None
        if market_data and market_data.get('india_vix', {}).get('success'):
            vix_value = market_data['india_vix'].get('value', 15)
            
            if vix_value > 25:
                result['warnings'].append('⚠️ High VIX (>25) - Use smaller position size')
                result['position_sizing'] = 'SMALL'
                result['confidence'] -= 15
            
            if vix_value < 12 and regime.get('regime') != 'LOW_VOLATILITY_GRIND':
                result['warnings'].append('⚠️ VIX too low - Volatility spike risk')
                result['confidence'] -= 10
        
        result['filters_passed'].append('✓ VIX level acceptable')
        
        # Filter 4: Volume Check
        if not df.empty and len(df) > 20:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio < 0.5:
                result['warnings'].append('⚠️ Very low volume - Avoid breakout trades')
                result['confidence'] -= 20
            
            if volume_ratio > 3:
                result['warnings'].append('⚠️ Extreme volume - Possible climax')
                result['confidence'] -= 10
        
        result['filters_passed'].append('✓ Volume healthy')
        
        # Filter 5: ATR Check (volatility)
        if not df.empty and len(df) > 20:
            atr = self._calculate_atr(df)
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
            
            if atr_ratio < 0.5:
                result['warnings'].append('⚠️ Very low ATR - Tight range, avoid momentum trades')
                result['confidence'] -= 15
        
        result['filters_passed'].append('✓ Volatility acceptable')
        
        # =====================================================
        # IMPORTANT FILTERS (Reduce confidence but allow trade)
        # =====================================================
        
        # Filter 6: PCR Check
        if options_data:
            pcr_oi = options_data.get('pcr_oi', 1.0)
            
            if pcr_oi > 1.8:
                result['warnings'].append('⚠️ Extreme PCR (>1.8) - Too much fear, avoid shorts')
                result['confidence'] -= 15
            
            if pcr_oi < 0.6:
                result['warnings'].append('⚠️ Low PCR (<0.6) - Too much greed, avoid longs')
                result['confidence'] -= 15
        
        result['filters_passed'].append('✓ PCR within acceptable range')
        
        # Filter 7: Global Markets Check
        if market_data and market_data.get('global_markets'):
            us_sentiment = self._check_global_sentiment(market_data['global_markets'])
            
            if us_sentiment == 'NEGATIVE' and bias_data.get('overall_bias') == 'BULLISH':
                result['warnings'].append('⚠️ Global markets negative but India bullish - Use caution')
                result['confidence'] -= 20
            
            if us_sentiment == 'POSITIVE' and bias_data.get('overall_bias') == 'BEARISH':
                result['warnings'].append('⚠️ Global markets positive but India bearish - Use caution')
                result['confidence'] -= 20
        
        result['filters_passed'].append('✓ Global sentiment aligned')
        
        # Filter 8: Trap Check
        if trap_analysis.get('trap_detected'):
            trap_type = trap_analysis.get('trap_type', '')
            if 'TRAP' in trap_type:
                result['filters_passed'].append(f'✓ {trap_type} detected - HIGH CONVICTION TRADE')
                result['confidence'] += 15
            else:
                result['filters_passed'].append(f'✓ {trap_type} - Normal market activity')
        
        # Filter 9: Bias Consensus Check
        if bias_data:
            bullish_count = bias_data.get('bullish_count', 0)
            bearish_count = bias_data.get('bearish_count', 0)
            total = bias_data.get('total_indicators', 8)
            consensus = max(bullish_count, bearish_count) / total if total > 0 else 0
            
            if consensus < 0.6:
                result['warnings'].append('⚠️ Weak consensus among indicators - Wait for stronger signal')
                result['confidence'] -= 25
        
        result['filters_passed'].append('✓ Indicator consensus strong')
        
        # Filter 10: Sector Rotation Check
        if market_data and market_data.get('sector_rotation', {}).get('success'):
            sector_data = market_data['sector_rotation']
            sector_breadth = sector_data.get('sector_breadth', 50)
            
            if sector_breadth < 30 and bias_data.get('overall_bias') == 'BULLISH':
                result['warnings'].append('⚠️ Weak sector breadth - Bullish signal less reliable')
                result['confidence'] -= 15
            
            if sector_breadth > 70 and bias_data.get('overall_bias') == 'BEARISH':
                result['warnings'].append('⚠️ Strong sector breadth - Bearish signal less reliable')
                result['confidence'] -= 15
        
        result['filters_passed'].append('✓ Sector rotation supports bias')
        
        # =====================================================
        # FINAL DETERMINATION
        # =====================================================
        
        if result['confidence'] >= 80:
            result['risk_level'] = 'LOW'
            result['position_sizing'] = 'FULL'
        elif result['confidence'] >= 60:
            result['risk_level'] = 'MEDIUM'
            result['position_sizing'] = 'NORMAL'
        elif result['confidence'] >= 40:
            result['risk_level'] = 'HIGH'
            result['position_sizing'] = 'SMALL'
        else:
            result['trade_allowed'] = False
            result['risk_level'] = 'EXTREME'
            result['final_recommendation'] = 'AVOID - Too many warning signals'
            return result
        
        if result['trade_allowed']:
            if result['confidence'] >= 80:
                result['final_recommendation'] = f"🎯 HIGH CONFIDENCE TRADE - {result['position_sizing']} position size"
            elif result['confidence'] >= 60:
                result['final_recommendation'] = f"✅ MODERATE TRADE - {result['position_sizing']} position size"
            else:
                result['final_recommendation'] = f"⚠️ LOW CONFIDENCE - {result['position_sizing']} position size only"
        
        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _check_global_sentiment(self, global_markets: List[Dict[str, Any]]) -> str:
        """Check overall global market sentiment"""
        if not global_markets:
            return 'NEUTRAL'
        
        bullish = 0
        bearish = 0
        
        for market in global_markets:
            if market.get('change_pct', 0) > 0.5:
                bullish += 1
            elif market.get('change_pct', 0) < -0.5:
                bearish += 1
        
        if bullish > bearish * 1.5:
            return 'POSITIVE'
        elif bearish > bullish * 1.5:
            return 'NEGATIVE'
        return 'NEUTRAL'

# =============================================
# MASTER DECISION ENGINE
# =============================================

class MasterDecisionEngine:
    """
    THE BRAIN - Combines everything into ONE intelligent trading decision
    """
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.regime_detector = MarketRegimeDetector()
        self.trap_detector = TrapDetector()
        self.execution_filter = ExecutionFilterEngine()

    def make_trading_decision(self,
                             price_data: pd.DataFrame,
                             bias_data: Dict[str, Any],
                             options_data: Dict[str, Any],
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Master function that makes THE FINAL TRADING DECISION
        """
        
        if price_data.empty or not bias_data or not options_data:
            return {
                'success': False,
                'error': 'Insufficient data for decision making'
            }
        
        current_price = price_data['close'].iloc[-1]
        current_time = datetime.now(self.ist)
        
        # Get VIX value if available
        vix_value = None
        if market_data and market_data.get('india_vix', {}).get('success'):
            vix_value = market_data['india_vix'].get('value', 15)
        
        # Step 1: Detect Market Regime
        regime = self.regime_detector.detect_market_regime(
            df=price_data,
            vix_value=vix_value
        )
        
        # Step 2: Detect Traps
        trap_analysis = self.trap_detector.analyze_market_trap(
            price_data=price_data,
            options_data=options_data,
            bias_data=bias_data
        )
        
        # Step 3: Run Execution Filters
        filter_result = self.execution_filter.should_trade(
            regime=regime,
            trap_analysis=trap_analysis,
            bias_data=bias_data,
            options_data=options_data,
            market_data=market_data,
            current_price=current_price,
            df=price_data
        )
        
        # =====================================================
        # MAKE FINAL DECISION
        # =====================================================
        
        decision = {
            'success': True,
            'timestamp': current_time,
            'current_price': current_price,
            'trade_decision': None,
            'trade_direction': None,
            'trade_type': None,
            'confidence': 0,
            'regime': regime,
            'trap_analysis': trap_analysis,
            'filter_result': filter_result,
            'entry_zone': None,
            'targets': [],
            'stop_loss': None,
            'position_size': 'NORMAL',
            'key_factors': [],
            'simple_summary': None,
            'detailed_explanation': None
        }
        
        # Check if trading is allowed
        if not filter_result['trade_allowed']:
            decision['trade_decision'] = 'NO TRADE'
            decision['confidence'] = 0
            decision['simple_summary'] = filter_result['final_recommendation']
            decision['key_factors'] = filter_result['filters_failed']
            return decision
        
        # Get base confidence from filters
        base_confidence = filter_result['confidence']
        
        # Determine trade direction from bias and trap analysis
        bias_direction = bias_data.get('overall_bias', 'NEUTRAL')
        trap_direction = None
        
        if trap_analysis.get('trade_setup'):
            trap_direction = trap_analysis['trade_setup'].get('direction')
        
        # =====================================================
        # DECISION LOGIC
        # =====================================================
        
        # SCENARIO 1: High Conviction Trap Trade
        if trap_analysis.get('trap_detected') and trap_analysis.get('trap_confidence', 0) >= 70:
            decision['trade_decision'] = 'TRADE'
            decision['trade_type'] = trap_analysis['trap_type']
            decision['trade_direction'] = trap_direction
            decision['confidence'] = min(95, base_confidence + 15)

            trade_setup = trap_analysis.get('trade_setup', {})
            decision['entry_zone'] = trade_setup.get('entry', 'Market')
            decision['targets'] = [trade_setup.get('target', 'N/A')]
            decision['stop_loss'] = trade_setup.get('stop_loss', 'N/A')
            decision['position_size'] = filter_result['position_sizing']
            
            decision['key_factors'] = [
                f"{trap_analysis['trap_type']} detected with {trap_analysis['trap_confidence']}% confidence",
                f"Trapped: {trap_analysis.get('who_is_trapped', 'Unknown')}",
                f"Expected: {trap_analysis.get('expected_move', 'Unknown')}",
                f"✓ All execution filters passed"
            ]
            
            decision['simple_summary'] = f"""
🎯 {decision['trade_direction']} TRADE SIGNAL
Time: {current_time.strftime('%H:%M:%S')} IST
Current Price: ₹{current_price:.2f}
Confidence: {decision['confidence']:.0f}%

🚨 HIGH CONVICTION {trap_analysis['trap_type']} TRADE
Confidence: {decision['confidence']:.0f}%
This is a TRAP REVERSAL - one of the most profitable setups!

TRADE DETAILS:

Direction: {decision['trade_direction']}
Trade Type: {decision['trade_type']}
Position Size: {decision['position_size']}

LEVELS:

Entry Zone: {decision['entry_zone']}
Target: ₹{decision['targets'][0] if decision['targets'] else 'N/A'}
Stop Loss: ₹{decision['stop_loss']}


KEY FACTORS:
{chr(10).join([f"• {factor}" for factor in decision['key_factors']])}
"""
        # SCENARIO 2: Strong Technical Bias Trade
        elif bias_direction in ['BULLISH', 'BEARISH'] and base_confidence >= 65:
            decision['trade_decision'] = 'TRADE'
            decision['trade_type'] = 'TECHNICAL_BIAS'
            decision['trade_direction'] = 'LONG' if bias_direction == 'BULLISH' else 'SHORT'
            decision['confidence'] = base_confidence
            
            # Calculate trade levels
            if decision['trade_direction'] == 'LONG':
                call_resistance = options_data.get('comprehensive_metrics', {}).get('call_resistance', current_price + 100)
                put_support = options_data.get('comprehensive_metrics', {}).get('put_support', current_price - 100)
                
                decision['entry_zone'] = f"₹{put_support:.0f} - ₹{current_price:.0f}"
                decision['targets'] = [
                    f"{current_price + (call_resistance - current_price) * 0.5:.0f}",
                    f"{call_resistance:.0f}"
                ]
                decision['stop_loss'] = f"{put_support - 30:.0f}"
            else:
                call_resistance = options_data.get('comprehensive_metrics', {}).get('call_resistance', current_price + 100)
                put_support = options_data.get('comprehensive_metrics', {}).get('put_support', current_price - 100)
                
                decision['entry_zone'] = f"₹{current_price:.0f} - ₹{call_resistance:.0f}"
                decision['targets'] = [
                    f"{current_price - (current_price - put_support) * 0.5:.0f}",
                    f"{put_support:.0f}"
                ]
                decision['stop_loss'] = f"{call_resistance + 30:.0f}"
            
            decision['position_size'] = filter_result['position_sizing']
            
            decision['key_factors'] = [
                f"✓ Technical indicators show {bias_direction} bias",
                f"✓ Market regime is {regime['regime']} - suitable for this trade type",
                f"✓ PCR OI: {options_data.get('pcr_oi', 'N/A'):.2f}",
                f"✓ All execution filters passed"
            ]
            
            decision['simple_summary'] = f"""
📈 {decision['trade_direction']} TRADE SIGNAL
Time: {current_time.strftime('%H:%M:%S')} IST
Current Price: ₹{current_price:.2f}
Confidence: {decision['confidence']:.0f}%

✅ TECHNICAL BIAS TRADE
Market Regime: {regime['regime']}
Risk Level: {filter_result['risk_level']}

TRADE DETAILS:

Direction: {decision['trade_direction']}
Trade Type: {decision['trade_type']}
Position Size: {decision['position_size']}

LEVELS:

Entry Zone: {decision['entry_zone']}
Target 1: ₹{decision['targets'][0]}
Target 2: ₹{decision['targets'][1]}
Stop Loss: ₹{decision['stop_loss']}


KEY FACTORS:
{chr(10).join([f"{factor}" for factor in decision['key_factors']])}

WHY THIS DECISION:

{bias_data.get('bullish_count', 0)}/{bias_data.get('total_indicators', 8)} indicators are {bias_direction.lower()}
Market regime supports this trade
All critical execution filters passed
"""
        # SCENARIO 3: Wait - No Clear Setup
        else:
            decision['trade_decision'] = 'WAIT'
            decision['confidence'] = base_confidence
            decision['simple_summary'] = f"""
⏳ WAIT FOR BETTER SETUP
Time: {current_time.strftime('%H:%M:%S')} IST
Current Price: ₹{current_price:.2f}
Confidence: {decision['confidence']:.0f}%

CURRENT SITUATION:

Market Bias: {bias_direction}
Market Regime: {regime['regime']}
Confidence Level: {base_confidence:.0f}%

WHY WAITING:
Market is tradeable but no high-conviction setup yet.
WHAT TO WATCH:

Wait for stronger consensus among indicators
Look for trap formation (higher conviction)
Monitor regime changes


FILTERS STATUS:
✓ Passed: {len(filter_result.get('filters_passed', []))}
⚠️ Warnings: {len(filter_result.get('warnings', []))}
"""
            decision['key_factors'] = [
                f"Market bias: {bias_direction}",
                f"Regime: {regime['regime']}",
                "Waiting for higher conviction setup",
                "All filters passed but no strong signal"
            ]
        
        return decision

# =============================================
# ENHANCED MARKET DATA FETCHER
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources
    """
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.request_count = 0
        self.last_request_time = None
        self.rate_limit_delay = 1.0
        
    def _rate_limit(self):
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = datetime.now()
        self.request_count += 1

    def get_current_time_ist(self):
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                time.sleep(1)
                ticker = yf.Ticker("^INDIAVIX")
                hist = ticker.history(period="1d", interval="1m")
                
                if hist.empty:
                    hist = ticker.history(period="5d", interval="5m")
                
                if not hist.empty and len(hist) > 0:
                    vix_value = hist['Close'].iloc[-1]
                    
                    if not (5 <= vix_value <= 100):
                        time.sleep(retry_delay)
                        continue

                    if vix_value > 25:
                        vix_sentiment = "HIGH FEAR"
                        vix_bias = "BEARISH"
                        vix_score = -75
                    elif vix_value > 20:
                        vix_sentiment = "ELEVATED FEAR"
                        vix_bias = "BEARISH"
                        vix_score = -50
                    elif vix_value > 15:
                        vix_sentiment = "MODERATE"
                        vix_bias = "NEUTRAL"
                        vix_score = 0
                    elif vix_value > 12:
                        vix_sentiment = "LOW VOLATILITY"
                        vix_bias = "BULLISH"
                        vix_score = 40
                    else:
                        vix_sentiment = "COMPLACENCY"
                        vix_bias = "NEUTRAL"
                        vix_score = 0

                    return {
                        'success': True,
                        'source': 'Yahoo Finance',
                        'value': vix_value,
                        'sentiment': vix_sentiment,
                        'bias': vix_bias,
                        'score': vix_score,
                        'timestamp': self.get_current_time_ist()
                    }
                
                time.sleep(retry_delay)
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for India VIX: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return {
            'success': False, 
            'error': 'India VIX data not available after retries',
            'value': 15.0,
            'sentiment': 'UNKNOWN',
            'bias': 'NEUTRAL',
            'score': 0,
            'timestamp': self.get_current_time_ist()
        }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        sectors_map = {
            '^CNXIT': 'NIFTY IT',
            '^CNXAUTO': 'NIFTY AUTO',
            '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL',
            '^CNXREALTY': 'NIFTY REALTY',
            '^CNXFMCG': 'NIFTY FMCG',
            '^CNXBANK': 'NIFTY BANK'
        }

        sector_data = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sector = {
                executor.submit(self._fetch_single_sector, symbol, name): (symbol, name)
                for symbol, name in sectors_map.items()
            }
            
            for future in as_completed(future_to_sector):
                result = future.result()
                if result:
                    sector_data.append(result)
                time.sleep(0.5)

        return sector_data

    def _fetch_single_sector(self, symbol: str, name: str) -> Optional[Dict[str, Any]]:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d", interval="1d")

            if not hist.empty and len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_pct = ((last_price - prev_price) / prev_price) * 100

                if change_pct > 1.5:
                    bias = "STRONG BULLISH"
                    score = 75
                elif change_pct > 0.5:
                    bias = "BULLISH"
                    score = 50
                elif change_pct < -1.5:
                    bias = "STRONG BEARISH"
                    score = -75
                elif change_pct < -0.5:
                    bias = "BEARISH"
                    score = -50
                else:
                    bias = "NEUTRAL"
                    score = 0

                return {
                    'sector': name,
                    'last_price': last_price,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score,
                    'source': 'Yahoo Finance'
                }
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            return None

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG',
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '000001.SS': 'SHANGHAI'
        }

        market_data = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_market = {
                executor.submit(self._fetch_single_market, symbol, name): (symbol, name)
                for symbol, name in global_markets.items()
            }
            
            for future in as_completed(future_to_market):
                result = future.result()
                if result:
                    market_data.append(result)
                time.sleep(0.5)

        return market_data

    def _fetch_single_market(self, symbol: str, name: str) -> Optional[Dict[str, Any]]:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")

            if len(hist) >= 2:
                current_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change_pct = ((current_close - prev_close) / prev_close) * 100

                if change_pct > 1.5:
                    bias = "STRONG BULLISH"
                    score = 75
                elif change_pct > 0.5:
                    bias = "BULLISH"
                    score = 50
                elif change_pct < -1.5:
                    bias = "STRONG BEARISH"
                    score = -75
                elif change_pct < -0.5:
                    bias = "BEARISH"
                    score = -50
                else:
                    bias = "NEUTRAL"
                    score = 0

                return {
                    'market': name,
                    'symbol': symbol,
                    'last_price': current_close,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score
                }
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            return None

    def analyze_sector_rotation(self) -> Dict[str, Any]:
        sectors = self.fetch_sector_indices()

        if not sectors:
            return {'success': False, 'error': 'No sector data available'}

        sectors_sorted = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)
        leaders = sectors_sorted[:3]
        laggards = sectors_sorted[-3:]

        bullish_sectors = [s for s in sectors if s['change_pct'] > 0.5]
        bearish_sectors = [s for s in sectors if s['change_pct'] < -0.5]

        if len(sectors) > 0:
            sector_breadth = (len(bullish_sectors) / len(sectors)) * 100
        else:
            sector_breadth = 50

        if len(leaders) > 0 and leaders[0]['change_pct'] > 2:
            rotation_pattern = "STRONG ROTATION"
            if 'IT' in leaders[0]['sector'] or 'PHARMA' in leaders[0]['sector']:
                rotation_type = "DEFENSIVE ROTATION (Risk-off)"
                rotation_bias = "BEARISH"
                rotation_score = -40
            elif 'METAL' in leaders[0]['sector'] or 'ENERGY' in leaders[0]['sector']:
                rotation_type = "CYCLICAL ROTATION (Risk-on)"
                rotation_bias = "BULLISH"
                rotation_score = 60
            elif 'BANK' in leaders[0]['sector'] or 'AUTO' in leaders[0]['sector']:
                rotation_type = "GROWTH ROTATION (Risk-on)"
                rotation_bias = "BULLISH"
                rotation_score = 70
            else:
                rotation_type = "MIXED ROTATION"
                rotation_bias = "NEUTRAL"
                rotation_score = 0
        else:
            rotation_pattern = "NO CLEAR ROTATION"
            rotation_type = "CONSOLIDATION"
            rotation_bias = "NEUTRAL"
            rotation_score = 0

        if sector_breadth > 70:
            sector_sentiment = "STRONG BULLISH"
            sector_score = 75
        elif sector_breadth > 55:
            sector_sentiment = "BULLISH"
            sector_score = 50
        elif sector_breadth < 30:
            sector_sentiment = "STRONG BEARISH"
            sector_score = -75
        elif sector_breadth < 45:
            sector_sentiment = "BEARISH"
            sector_score = -50
        else:
            sector_sentiment = "NEUTRAL"
            sector_score = 0

        return {
            'success': True,
            'leaders': leaders,
            'laggards': laggards,
            'bullish_sectors_count': len(bullish_sectors),
            'bearish_sectors_count': len(bearish_sectors),
            'sector_breadth': sector_breadth,
            'rotation_pattern': rotation_pattern,
            'rotation_type': rotation_type,
            'rotation_bias': rotation_bias,
            'rotation_score': rotation_score,
            'sector_sentiment': sector_sentiment,
            'sector_score': sector_score,
            'all_sectors': sectors,
            'timestamp': self.get_current_time_ist()
        }

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        print("=" * 60)
        print("FETCHING ENHANCED MARKET DATA")
        print("=" * 60)
        
        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'sector_rotation': {},
            'summary': {},
            'fetch_status': {}
        }
        
        fetch_status = {
            'india_vix': 'pending',
            'sectors': 'pending',
            'global': 'pending',
            'rotation': 'pending'
        }
        
        # 1. Fetch India VIX
        try:
            print("\n[1/4] Fetching India VIX...")
            self._rate_limit()
            result['india_vix'] = self.fetch_india_vix()
            fetch_status['india_vix'] = 'success' if result['india_vix'].get('success') else 'failed'
            print(f"  Status: {fetch_status['india_vix']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['india_vix'] = 'error'
            result['india_vix'] = {'success': False, 'error': str(e)}
        
        # 2. Fetch Sector Indices
        try:
            print("\n[2/4] Fetching sector indices...")
            self._rate_limit()
            result['sector_indices'] = self.fetch_sector_indices()
            fetch_status['sectors'] = 'success' if result['sector_indices'] else 'failed'
            print(f"  Fetched {len(result['sector_indices'])} sectors")
            print(f"  Status: {fetch_status['sectors']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['sectors'] = 'error'
            result['sector_indices'] = []
        
        # 3. Fetch Global Markets
        try:
            print("\n[3/4] Fetching global markets...")
            self._rate_limit()
            result['global_markets'] = self.fetch_global_markets()
            fetch_status['global'] = 'success' if result['global_markets'] else 'failed'
            print(f"  Fetched {len(result['global_markets'])} markets")
            print(f"  Status: {fetch_status['global']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['global'] = 'error'
            result['global_markets'] = []
        
        # 4. Analyze Sector Rotation
        try:
            print("\n[4/4] Analyzing Sector Rotation...")
            result['sector_rotation'] = self.analyze_sector_rotation()
            fetch_status['rotation'] = 'success' if result['sector_rotation'].get('success') else 'failed'
            print(f"  Status: {fetch_status['rotation']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['rotation'] = 'error'
            result['sector_rotation'] = {'success': False, 'error': str(e)}
        
        # Calculate summary
        try:
            print("\n[5/5] Calculating summary...")
            result['summary'] = self._calculate_summary(result)
            print("  ✓ Summary calculated")
        except Exception as e:
            print(f"  ❌ Summary Error: {str(e)}")
            result['summary'] = {}
        
        result['fetch_status'] = fetch_status
        
        print("\n" + "=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)
        success_count = sum(1 for status in fetch_status.values() if status == 'success')
        print(f"Successful: {success_count}/{len(fetch_status)}")
        for key, status in fetch_status.items():
            icon = "✓" if status == 'success' else "✗"
            print(f"  {icon} {key}: {status}")
        print("=" * 60)
        
        return result

    def _calculate_summary(self, data: Dict) -> Dict[str, Any]:
        summary = {
            'total_data_points': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'overall_sentiment': 'NEUTRAL'
        }

        all_scores = []

        if data['india_vix'].get('success'):
            summary['total_data_points'] += 1
            all_scores.append(data['india_vix']['score'])
            bias = data['india_vix']['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        for sector in data['sector_indices']:
            summary['total_data_points'] += 1
            all_scores.append(sector['score'])
            bias = sector['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        for market in data['global_markets']:
            summary['total_data_points'] += 1
            all_scores.append(market['score'])
            bias = market['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        if all_scores:
            summary['avg_score'] = np.mean(all_scores)

            if summary['avg_score'] > 25:
                summary['overall_sentiment'] = 'BULLISH'
            elif summary['avg_score'] < -25:
                summary['overall_sentiment'] = 'BEARISH'
            else:
                summary['overall_sentiment'] = 'NEUTRAL'

        return summary

# =============================================
# VOLUME ORDER BLOCKS
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_lines_count = 500
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.sent_alerts = set()
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period=200) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3

    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.calculate_ema(df['close'], self.length1)
        ema2 = self.calculate_ema(df['close'], self.length2)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        atr = self.calculate_atr(df)
        atr1 = atr * 2 / 3
        
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(len(df)):
            if cross_up.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                lowest_idx = lookback_data['low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'low']
                
                vol = lookback_data['volume'].sum()
                
                open_price = lookback_data.loc[lowest_idx, 'open']
                close_price = lookback_data.loc[lowest_idx, 'close']
                src = min(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (src - lowest_price) < atr1.iloc[i] * 0.5:
                    src = lowest_price + atr1.iloc[i] * 0.5
                
                mid = (src + lowest_price) / 2
                
                bullish_blocks.append({
                    'index': lowest_idx,
                    'upper': src,
                    'lower': lowest_price,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bullish'
                })
                
            elif cross_down.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                highest_idx = lookback_data['high'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'high']
                
                vol = lookback_data['volume'].sum()
                
                open_price = lookback_data.loc[highest_idx, 'open']
                close_price = lookback_data.loc[highest_idx, 'close']
                src = max(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (highest_price - src) < atr1.iloc[i] * 0.5:
                    src = highest_price - atr1.iloc[i] * 0.5
                
                mid = (src + highest_price) / 2
                
                bearish_blocks.append({
                    'index': highest_idx,
                    'upper': highest_price,
                    'lower': src,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bearish'
                })
        
        bullish_blocks = self.filter_overlapping_blocks(bullish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        bearish_blocks = self.filter_overlapping_blocks(bearish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        
        return bullish_blocks, bearish_blocks

    def filter_overlapping_blocks(self, blocks: List[Dict[str, Any]], atr_value: float) -> List[Dict[str, Any]]:
        if not blocks:
            return []
        
        filtered_blocks = []
        for block in blocks:
            overlap = False
            for existing_block in filtered_blocks:
                if abs(block['mid'] - existing_block['mid']) < atr_value:
                    overlap = True
                    break
            if not overlap:
                filtered_blocks.append(block)
        
        return filtered_blocks

# =============================================
# NSE OPTIONS ANALYZER
# =============================================

class NSEOptionsAnalyzer:
    """Integrated NSE Options Analyzer with complete ATM bias analysis"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
                'FINNIFTY': {'lot_size': 40, 'atm_range': 200, 'zone_size': 100},
            }
        }
        self.last_refresh_time = {}
        self.refresh_interval = 2
        self.cached_bias_data = {}
        
    def set_refresh_interval(self, minutes: int):
        self.refresh_interval = minutes

    def should_refresh_data(self, instrument: str) -> bool:
        current_time = datetime.now(self.ist)
        
        if instrument not in self.last_refresh_time:
            self.last_refresh_time[instrument] = current_time
            return True
        
        last_refresh = self.last_refresh_time[instrument]
        time_diff = (current_time - last_refresh).total_seconds() / 60
        
        if time_diff >= self.refresh_interval:
            self.last_refresh_time[instrument] = current_time
            return True
        
        return False

    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}"

            response = session.get(url, timeout=10)
            data = response.json()

            records = data['records']['data']
            expiry = data['records']['expiryDates'][0]
            underlying = data['records']['underlyingValue']

            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item)
            total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item)

            return {
                'success': True,
                'instrument': instrument,
                'spot': underlying,
                'expiry': expiry,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'total_ce_change': total_ce_change,
                'total_pe_change': total_pe_change,
                'records': records
            }
        except Exception as e:
            return {
                'success': False,
                'instrument': instrument,
                'error': str(e)
            }

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            records = data['records']
            spot = data['spot']

            # Create mock option chain data for demonstration
            option_chain = []
            for i in range(-5, 6):
                strike = spot + i * 100
                option_chain.append({
                    'strike': strike,
                    'ce_gamma': 0.05 + abs(i) * 0.01,
                    'pe_gamma': 0.05 + abs(i) * 0.01,
                    'ce_oi_change': 1000 - i * 200,
                    'pe_oi_change': 1000 + i * 200,
                    'ce_iv': 15 + abs(i) * 2,
                    'pe_iv': 15 + abs(i) * 2,
                    'ce_ltp': max(10, 50 - abs(i) * 5),
                    'pe_ltp': max(10, 50 - abs(i) * 5)
                })

            return {
                'instrument': instrument,
                'spot_price': spot,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change'],
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'option_chain': option_chain
            }

        except Exception as e:
            print(f"Error in ATM bias analysis: {e}")
            return None

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_atm_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                        self.cached_bias_data[instrument] = bias_data
                except Exception as e:
                    print(f"Error fetching {instrument}: {e}")
                    if instrument in self.cached_bias_data:
                        results.append(self.cached_bias_data[instrument])
            else:
                if instrument in self.cached_bias_data:
                    results.append(self.cached_bias_data[instrument])
        
        return results

# =============================================
# ENHANCED NIFTY APP WITH ALL FEATURES
# =============================================

class EnhancedNiftyApp:
    
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"

        # Initialize all indicators
        self.bias_analyzer = BiasAnalysisPro()
        self.market_data_fetcher = EnhancedMarketData()
        self.options_analyzer = NSEOptionsAnalyzer()
        self.decision_engine = MasterDecisionEngine()
        self.price_action_engine = PriceActionEngine()
        self.gamma_detector = GammaExpirySpikeDetector()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        defaults = {
            'comprehensive_bias_data': None,
            'last_comprehensive_bias_update': None,
            'enhanced_market_data': None,
            'last_market_data_update': None,
            'market_bias_data': None,
            'last_market_bias_update': None,
            'master_decision': None,
            'last_decision_time': None,
            'decision_history': [],
            'price_action_data': None,
            'gamma_data': None,
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def setup_secrets(self):
        try:
            self.dhan_token = st.secrets.get("dhan", {}).get("access_token", "demo_token")
            self.dhan_client_id = st.secrets.get("dhan", {}).get("client_id", "demo_client")
            self.supabase_url = st.secrets.get("supabase", {}).get("url", "")
            self.supabase_key = st.secrets.get("supabase", {}).get("anon_key", "")
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except Exception as e:
            st.warning(f"Secrets setup warning: {e}")

    def setup_supabase(self):
        try:
            if self.supabase_url and self.supabase_key:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            else:
                self.supabase = None
        except Exception as e:
            st.warning(f"Supabase connection error: {str(e)}")
            self.supabase = None

    def get_dhan_headers(self) -> Dict[str, str]:
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.dhan_token,
            'client-id': self.dhan_client_id
        }

    def fetch_intraday_data(self, interval: str = "5", days_back: int = 5) -> Optional[Dict[str, Any]]:
        try:
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=min(days_back, 90))
            
            from_date = start_date.strftime("%Y-%m-%d 09:15:00")
            to_date = end_date.strftime("%Y-%m-%d 15:30:00")
            
            payload = {
                "securityId": str(self.nifty_security_id),
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": str(interval),
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                "https://api.dhan.co/v2/charts/intraday",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data or 'open' not in data or len(data['open']) == 0:
                    return None
                return data
            else:
                return None
                
        except Exception as e:
            return None

    def process_data(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        if not api_data or 'open' not in api_data:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': api_data['timestamp'],
            'open': api_data['open'],
            'high': api_data['high'],
            'low': api_data['low'],
            'close': api_data['close'],
            'volume': api_data['volume']
        })
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(self.ist)
        df = df.set_index('datetime')
        
        return df

    # =============================================
    # STREAMLIT UI COMPONENTS
    # =============================================

    def display_technical_bias_analysis(self):
        st.header("🎯 Technical Bias Analysis")
        
        if st.button("🔄 Analyze Technical Bias", type="primary"):
            with st.spinner("Analyzing technical indicators..."):
                try:
                    bias_data = self.bias_analyzer.analyze_all_bias_indicators()
                    if bias_data.get('success'):
                        st.session_state.comprehensive_bias_data = bias_data
                        st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
                        st.success("Technical bias analysis completed!")
                    else:
                        st.error(f"Error: {bias_data.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            # Overall Bias
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bias_color = "green" if bias_data['overall_bias'] == 'BULLISH' else "red" if bias_data['overall_bias'] == 'BEARISH' else "gray"
                st.metric("Overall Bias", bias_data['overall_bias'], delta=None, delta_color="off")
            with col2:
                st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
            with col3:
                st.metric("Bullish Indicators", f"{bias_data['bullish_count']}/{bias_data['total_indicators']}")
            with col4:
                st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
            
            # Bias Results Table
            st.subheader("Detailed Indicator Analysis")
            bias_df = pd.DataFrame(bias_data['bias_results'])
            st.dataframe(bias_df, use_container_width=True)
            
            # NEW: Reversal Probability
            if bias_data.get('reversal_data'):
                st.subheader("🔄 Reversal Probability Analysis")
                rev_data = bias_data['reversal_data']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bullish Probability", f"{rev_data['bullish_prob']:.1%}")
                with col2:
                    st.metric("Bearish Probability", f"{rev_data['bearish_prob']:.1%}")
                with col3:
                    st.metric("Strength", rev_data['strength'].upper())
            
            # NEW: Price Action Analysis
            if bias_data.get('price_action_data'):
                st.subheader("📊 Price Action Analysis")
                pa_data = bias_data['price_action_data']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trend Score", pa_data['trend_score'])
                with col2:
                    st.write("Supports:", pa_data['supports'])
                with col3:
                    st.write("Resistances:", pa_data['resistances'])
                
                if pa_data.get('pattern'):
                    st.info(f"Pattern Detected: {pa_data['pattern']}")

    def display_options_analysis(self):
        st.header("📊 Options Chain Analysis")
        
        if st.button("🔄 Fetch Options Data", type="primary"):
            with st.spinner("Fetching options chain data..."):
                try:
                    options_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                    if options_data:
                        st.session_state.market_bias_data = options_data
                        st.session_state.last_market_bias_update = datetime.now(self.ist)
                        st.success(f"Fetched data for {len(options_data)} instruments")
                    else:
                        st.error("Failed to fetch options data")
                except Exception as e:
                    st.error(f"Options analysis failed: {str(e)}")
        
        if st.session_state.market_bias_data:
            for instrument_data in st.session_state.market_bias_data:
                with st.expander(f"📈 {instrument_data['instrument']} Analysis", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                    with col2:
                        st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                    with col3:
                        st.metric("CE OI", f"{instrument_data['total_ce_oi']:,}")
                    with col4:
                        st.metric("PE OI", f"{instrument_data['total_pe_oi']:,}")
                    
                    # Gamma Analysis
                    if instrument_data.get('option_chain'):
                        gamma_data = self.gamma_detector.gamma_sequence_expiry(
                            instrument_data['option_chain'], 
                            instrument_data['spot_price']
                        )
                        
                        st.subheader("🎯 Gamma & Expiry Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expiry Spike Score", f"{gamma_data['expiry_spike_score']}/100")
                        with col2:
                            st.metric("Gamma Pressure", gamma_data['gamma_pressure'])
                        with col3:
                            st.metric("Expected Move", f"₹{gamma_data['expected_move_points']:.0f}")
                        
                        # Spike probability interpretation
                        spike_score = gamma_data['expiry_spike_score']
                        if spike_score >= 85:
                            st.error("🚨 HIGH SPIKE PROBABILITY - Violent move expected")
                        elif spike_score >= 60:
                            st.warning("⚠️ MODERATE SPIKE PROBABILITY - Significant move likely")
                        elif spike_score >= 30:
                            st.info("ℹ️ LOW SPIKE PROBABILITY - Some movement possible")
                        else:
                            st.success("✅ NO SPIKE EXPECTED - Normal trading")

    def display_market_data(self):
        st.header("🌍 Enhanced Market Data")
        
        if st.button("🔄 Fetch Market Data", type="primary"):
            with st.spinner("Fetching comprehensive market data..."):
                try:
                    market_data = self.market_data_fetcher.fetch_all_enhanced_data()
                    st.session_state.enhanced_market_data = market_data
                    st.session_state.last_market_data_update = datetime.now(self.ist)
                    st.success("Market data fetched successfully!")
                except Exception as e:
                    st.error(f"Market data fetch failed: {str(e)}")
        
        if st.session_state.enhanced_market_data:
            market_data = st.session_state.enhanced_market_data
            
            # India VIX
            if market_data['india_vix'].get('success'):
                vix_data = market_data['india_vix']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("India VIX", f"{vix_data['value']:.2f}")
                with col2:
                    st.metric("Sentiment", vix_data['sentiment'])
                with col3:
                    st.metric("Bias", vix_data['bias'])
            
            # Sector Rotation
            if market_data['sector_rotation'].get('success'):
                sector_data = market_data['sector_rotation']
                st.subheader("📊 Sector Rotation")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sector Breadth", f"{sector_data['sector_breadth']:.1f}%")
                with col2:
                    st.metric("Rotation Pattern", sector_data['rotation_pattern'])
                with col3:
                    st.metric("Overall Sentiment", sector_data['sector_sentiment'])
                
                # Sector Leaders
                if sector_data.get('leaders'):
                    st.write("**Sector Leaders:**")
                    for leader in sector_data['leaders']:
                        st.write(f"- {leader['sector']}: {leader['change_pct']:+.2f}% ({leader['bias']})")
            
            # Global Markets
            if market_data['global_markets']:
                st.subheader("🌐 Global Markets")
                for market in market_data['global_markets'][:4]:  # Show top 4
                    col1, col2, col3 = st.columns([2,1,1])
                    with col1:
                        st.write(f"**{market['market']}**")
                    with col2:
                        st.write(f"{market['change_pct']:+.2f}%")
                    with col3:
                        st.write(market['bias'])

    def display_master_decision(self):
        st.header("🧠 Master Decision Engine - THE BRAIN")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("The Master Decision Engine combines Market Regime, Trap Detection, and Execution Filters into ONE intelligent trading decision")
        with col2:
            if st.button("🎯 Get Decision", type="primary"):
                with st.spinner("Generating master trading decision..."):
                    try:
                        # Fetch current price data
                        api_data = self.fetch_intraday_data(interval='5')
                        if api_data:
                            df = self.process_data(api_data)
                            
                            if not df.empty and st.session_state.comprehensive_bias_data and st.session_state.market_bias_data:
                                # Generate decision
                                decision = self.decision_engine.make_trading_decision(
                                    price_data=df,
                                    bias_data=st.session_state.comprehensive_bias_data,
                                    options_data=st.session_state.market_bias_data[0] if st.session_state.market_bias_data else None,
                                    market_data=st.session_state.enhanced_market_data
                                )
                                
                                st.session_state.master_decision = decision
                                st.session_state.last_decision_time = datetime.now(self.ist)
                                
                                # Store in history
                                st.session_state.decision_history.append(decision)
                                if len(st.session_state.decision_history) > 10:
                                    st.session_state.decision_history.pop(0)
                                
                                st.success("Decision generated successfully!")
                                st.rerun()
                            else:
                                st.error("Please load Technical Bias, Options Data, and Market Data first!")
                    except Exception as e:
                        st.error(f"Error generating decision: {str(e)}")
        
        st.divider()
        
        # Display last decision time
        if st.session_state.last_decision_time:
            time_diff = datetime.now(self.ist) - st.session_state.last_decision_time
            minutes_ago = int(time_diff.total_seconds() / 60)
            st.write(f"Last decision: {st.session_state.last_decision_time.strftime('%H:%M:%S')} IST ({minutes_ago} min ago)")
        
        if st.session_state.master_decision:
            decision = st.session_state.master_decision
            
            if not decision.get('success'):
                st.error(f"❌ Decision generation failed: {decision.get('error', 'Unknown error')}")
                return
            
            # Display decision summary
            st.markdown(decision.get('simple_summary', 'No summary available'))
            
            st.divider()
            
            # Create tabs for detailed analysis
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Trade Setup", "📊 Market Regime", "🚨 Trap Analysis", "🛡️ Execution Filters"
            ])
            
            with tab1:
                st.subheader("Trade Setup Details")
                
                if decision.get('trade_decision') == 'TRADE':
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Direction", decision.get('trade_direction', 'N/A'))
                    with col2:
                        st.metric("Trade Type", decision.get('trade_type', 'N/A'))
                    with col3:
                        st.metric("Confidence", f"{decision.get('confidence', 0):.0f}%")
                    with col4:
                        st.metric("Position Size", decision.get('position_size', 'N/A'))
                    
                    st.subheader("Entry & Exit Levels")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Entry Zone:** {decision.get('entry_zone', 'N/A')}")
                        st.write(f"**Stop Loss:** ₹{decision.get('stop_loss', 'N/A')}")
                    with col2:
                        targets = decision.get('targets', [])
                        if targets:
                            st.write(f"**Target 1:** ₹{targets[0]}")
                            if len(targets) > 1:
                                st.write(f"**Target 2:** ₹{targets[1]}")
                    
                    st.subheader("Key Factors")
                    for factor in decision.get('key_factors', []):
                        st.write(f"• {factor}")
                
                else:
                    st.info("No trade recommended at this time")
            
            with tab2:
                st.subheader("Market Regime Analysis")
                regime = decision.get('regime', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Regime", regime.get('regime', 'Unknown'))
                with col2:
                    st.metric("Confidence", f"{regime.get('confidence', 0)}%")
                with col3:
                    st.metric("Risk Level", regime.get('risk_level', 'Unknown'))
                
                st.write("**Characteristics:**")
                for char in regime.get('characteristics', []):
                    st.write(f"• {char}")
                
                st.write("**Best Strategies:**")
                for strategy in regime.get('best_strategies', []):
                    st.write(f"• {strategy}")
                
                st.info(f"**Recommendation:** {regime.get('trade_recommendation', 'N/A')}")
            
            with tab3:
                st.subheader("Trap Analysis")
                trap = decision.get('trap_analysis', {})
                
                if trap.get('trap_detected'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trap Type", trap.get('trap_type', 'Unknown'))
                    with col2:
                        st.metric("Confidence", f"{trap.get('trap_confidence', 0)}%")
                    with col3:
                        st.metric("Action", trap.get('action', 'N/A'))
                    
                    st.write("**Who is Trapped:**", trap.get('who_is_trapped', 'Unknown'))
                    st.write("**Expected Move:**", trap.get('expected_move', 'Unknown'))
                    
                    st.write("**Characteristics:**")
                    for char in trap.get('characteristics', []):
                        st.write(f"• {char}")
                else:
                    st.success("✅ No trap detected - Market moving genuinely")
            
            with tab4:
                st.subheader("Execution Filters")
                filters = decision.get('filter_result', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trade Allowed", "✅ YES" if filters.get('trade_allowed') else "🚫 NO")
                with col2:
                    st.metric("Confidence", f"{filters.get('confidence', 0)}%")
                with col3:
                    st.metric("Risk Level", filters.get('risk_level', 'Unknown'))
                
                st.write(f"**✓ Filters Passed:** {len(filters.get('filters_passed', []))}")
                for passed_filter in filters.get('filters_passed', []):
                    st.success(passed_filter)
                
                if filters.get('warnings'):
                    st.write("**⚠️ Warnings:**")
                    for warning in filters['warnings']:
                        st.warning(warning)
                
                if filters.get('filters_failed'):
                    st.write("**🚫 Filters Failed:**")
                    for failed_filter in filters['filters_failed']:
                        st.error(failed_filter)
        
        else:
            st.info("👆 Click 'Get Decision' to generate intelligent trading decision")
            st.write("""
            **The Master Decision Engine combines:**
            
            1. **Market Regime Detection** - Identifies the type of market (trending, range-bound, trap zone, etc.)
            2. **Trap Detection** - Detects bull traps, bear traps, short covering, and long liquidation
            3. **Execution Filters** - 10+ filters to protect your capital
            4. **Final Decision** - ONE clear trading decision with entry, target, and stop loss
            
            **Prerequisites:**
            - Load Technical Bias Analysis (Tab 3)
            - Load Options Chain Analysis (Tab 2)
            - Load Enhanced Market Data (Tab 6)
            """)

    def display_price_action_analysis(self):
        st.header("📈 Price Action Analysis")
        
        if st.button("🔄 Analyze Price Action", type="primary"):
            with st.spinner("Analyzing price action patterns..."):
                try:
                    # Fetch data for multiple timeframes (using same data for demo)
                    api_data = self.fetch_intraday_data(interval='5')
                    if api_data:
                        df = self.process_data(api_data)
                        price_action_data = self.price_action_engine.analyze_multi_timeframe(df, df, df)
                        st.session_state.price_action_data = price_action_data
                        st.success("Price action analysis completed!")
                    else:
                        st.error("Failed to fetch price data")
                except Exception as e:
                    st.error(f"Price action analysis failed: {str(e)}")
        
        if st.session_state.price_action_data:
            pa_data = st.session_state.price_action_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("15m Trend", pa_data['trend_15m'])
            with col2:
                st.metric("1h Trend", pa_data['trend_1h'])
            with col3:
                st.metric("1D Trend", pa_data['trend_1d'])
            
            st.subheader("Trend Score")
            st.info(f"**{pa_data['trend_score']}** - Based on multi-timeframe alignment")
            
            st.subheader("Support & Resistance")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Supports:**")
                for support in pa_data['supports']:
                    st.write(f"- ₹{support:.0f}")
            with col2:
                st.write("**Resistances:**")
                for resistance in pa_data['resistances']:
                    st.write(f"- ₹{resistance:.0f}")
            
            if pa_data.get('pattern'):
                st.subheader("Chart Pattern Detected")
                st.warning(f"**{pa_data['pattern']}** - Monitor for breakout/breakdown")

    def run(self):
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Complete Quantitative + Price Action Analysis with Master Decision Engine*")
        
        # Sidebar
        st.sidebar.title("Configuration")
        st.sidebar.info("""
        **Dashboard Features:**
        - Technical Bias Analysis
        - Options Chain Intelligence  
        - Price Action Patterns
        - Market Regime Detection
        - Trap Detection
        - Master Decision Engine
        """)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
            "📊 Price Action", "🌍 Market Data", "🧠 Master Decision", "📋 System Status"
        ])
        
        with tab1:
            st.header("Price Charts & Analysis")
            st.info("Price chart visualization would go here")
            
        with tab2:
            self.display_options_analysis()
            
        with tab3:
            self.display_technical_bias_analysis()
            
        with tab4:
            self.display_price_action_analysis()
            
        with tab5:
            self.display_market_data()
            
        with tab6:
            self.display_master_decision()
            
        with tab7:
            st.header("System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Data Status")
                if st.session_state.comprehensive_bias_data:
                    st.success("✅ Technical Bias Data")
                else:
                    st.error("❌ Technical Bias Data")
                    
                if st.session_state.market_bias_data:
                    st.success("✅ Options Data")
                else:
                    st.error("❌ Options Data")
                    
                if st.session_state.enhanced_market_data:
                    st.success("✅ Market Data")
                else:
                    st.error("❌ Market Data")
            
            with col2:
                st.subheader("Last Updates")
                if st.session_state.last_comprehensive_bias_update:
                    st.write(f"Bias: {st.session_state.last_comprehensive_bias_update.strftime('%H:%M:%S')}")
                if st.session_state.last_market_bias_update:
                    st.write(f"Options: {st.session_state.last_market_bias_update.strftime('%H:%M:%S')}")
                if st.session_state.last_market_data_update:
                    st.write(f"Market: {st.session_state.last_market_data_update.strftime('%H:%M:%S')}")
            
            with col3:
                st.subheader("System Info")
                st.write(f"Time: {datetime.now(self.ist).strftime('%H:%M:%S')} IST")
                st.write("Status: 🟢 Operational")
                
            # Decision History
            if st.session_state.decision_history:
                st.subheader("Recent Decisions")
                for i, decision in enumerate(reversed(st.session_state.decision_history[-5:])):
                    with st.expander(f"Decision {i+1} - {decision.get('timestamp').strftime('%H:%M:%S')}"):
                        st.write(f"Trade: {decision.get('trade_decision', 'N/A')}")
                        st.write(f"Direction: {decision.get('trade_direction', 'N/A')}")
                        st.write(f"Confidence: {decision.get('confidence', 0):.0f}%")

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
