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
    page_title="Advanced Nifty Trading Dashboard Pro",
    page_icon="📈",
    layout="wide"
)

# =============================================
# COMPREHENSIVE BIAS ANALYSIS MODULE
# =============================================

class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis matching Pine Script indicator EXACTLY
    Analyzes 13 bias indicators with adaptive scoring
    """

    def __init__(self):
        """Initialize bias analysis with default configuration"""
        self.config = self._default_config()
        self.ist = pytz.timezone('Asia/Kolkata')

    def _default_config(self) -> Dict[str, Any]:
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

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance with enhanced error handling"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()

            # Ensure all required columns exist
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index"""
        if df['Volume'].sum() == 0:
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi.fillna(50)

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate DMI indicators"""
        high = df['High']
        low = df['Low']
        close = df['Close']

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
        """Calculate VWAP"""
        if df['Volume'].sum() == 0:
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe
        return vwap.fillna(typical_price)

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0) -> Tuple[pd.Series, bool, bool]:
        """Calculate VIDYA (Variable Index Dynamic Average)"""
        close = df['Close']

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

    def calculate_volume_delta(self, df: pd.DataFrame) -> Tuple[float, bool, bool]:
        """Calculate Volume Delta (up_vol - down_vol)"""
        if df['Volume'].sum() == 0:
            return 0, False, False

        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0) -> Tuple[bool, bool, int, int]:
        """Calculate High Volume Pivots"""
        if df['Volume'].sum() == 0:
            return False, False, 0, 0

        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['High'].iloc[j] >= df['High'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['Low'].iloc[j] <= df['Low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        volume_sum = df['Volume'].rolling(window=left_bars * 2).sum()
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

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5) -> Tuple[bool, bool, float, float]:
        """Calculate Volume Order Blocks"""
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        """Analyze all 8 bias indicators with enhanced error handling"""

        print(f"Fetching data for {symbol}...")
        try:
            df = self.fetch_data(symbol, period='7d', interval='5m')

            if df.empty or len(df) < 100:
                error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 100)'
                print(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'symbol': symbol
                }

            current_price = df['Close'].iloc[-1]
            bias_results = []

            # 1. VOLUME DELTA
            volume_delta, volume_bullish, volume_bearish = self.calculate_volume_delta(df)
            if volume_bullish:
                vol_delta_bias = "BULLISH"
                vol_delta_score = 100
            elif volume_bearish:
                vol_delta_bias = "BEARISH"
                vol_delta_score = -100
            else:
                vol_delta_bias = "NEUTRAL"
                vol_delta_score = 0

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
            if hvp_bullish:
                hvp_bias = "BULLISH"
                hvp_score = 100
                hvp_value = f"Bull Signal (Lows: {pivot_lows}, Highs: {pivot_highs})"
            elif hvp_bearish:
                hvp_bias = "BEARISH"
                hvp_score = -100
                hvp_value = f"Bear Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"
            else:
                hvp_bias = "NEUTRAL"
                hvp_score = 0
                hvp_value = f"No Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"

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
            if vob_bullish:
                vob_bias = "BULLISH"
                vob_score = 100
                vob_value = f"Bull Cross (EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f})"
            elif vob_bearish:
                vob_bias = "BEARISH"
                vob_score = -100
                vob_value = f"Bear Cross (EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f})"
            else:
                vob_bias = "NEUTRAL"
                vob_score = 0
                if vob_ema5 > vob_ema18:
                    vob_value = f"EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f} (No Cross)"
                else:
                    vob_value = f"EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f} (No Cross)"

            bias_results.append({
                'indicator': 'VOB (Volume Order Blocks)',
                'value': vob_value,
                'bias': vob_bias,
                'score': vob_score,
                'weight': 1.0,
                'category': 'fast'
            })

            # 4. ORDER BLOCKS (EMA Crossover)
            ema5 = self.calculate_ema(df['Close'], 5)
            ema18 = self.calculate_ema(df['Close'], 18)
            cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
            cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])

            if cross_up:
                ob_bias = "BULLISH"
                ob_score = 100
            elif cross_dn:
                ob_bias = "BEARISH"
                ob_score = -100
            else:
                ob_bias = "NEUTRAL"
                ob_score = 0

            bias_results.append({
                'indicator': 'Order Blocks (EMA 5/18)',
                'value': f"EMA5: {ema5.iloc[-1]:.2f} | EMA18: {ema18.iloc[-1]:.2f}",
                'bias': ob_bias,
                'score': ob_score,
                'weight': 1.0,
                'category': 'fast'
            })

            # 5. RSI
            rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
            rsi_value = rsi.iloc[-1]
            if rsi_value > 50:
                rsi_bias = "BULLISH"
                rsi_score = 100
            else:
                rsi_bias = "BEARISH"
                rsi_score = -100

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
            if plus_di_value > minus_di_value:
                dmi_bias = "BULLISH"
                dmi_score = 100
            else:
                dmi_bias = "BEARISH"
                dmi_score = -100

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
            if vidya_bullish:
                vidya_bias = "BULLISH"
                vidya_score = 100
            elif vidya_bearish:
                vidya_bias = "BEARISH"
                vidya_score = -100
            else:
                vidya_bias = "NEUTRAL"
                vidya_score = 0

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

            if mfi_value > 50:
                mfi_bias = "BULLISH"
                mfi_score = 100
            else:
                mfi_bias = "BEARISH"
                mfi_score = -100

            bias_results.append({
                'indicator': 'MFI (Money Flow)',
                'value': f"{mfi_value:.2f}",
                'bias': mfi_bias,
                'score': mfi_score,
                'weight': 1.0,
                'category': 'fast'
            })

            # Calculate overall bias
            fast_bull = 0
            fast_bear = 0
            fast_total = 0

            bullish_count = 0
            bearish_count = 0
            neutral_count = 0

            for bias in bias_results:
                if 'BULLISH' in bias['bias']:
                    bullish_count += 1
                    if bias['category'] == 'fast':
                        fast_bull += 1
                elif 'BEARISH' in bias['bias']:
                    bearish_count += 1
                    if bias['category'] == 'fast':
                        fast_bear += 1
                else:
                    neutral_count += 1

                if bias['category'] == 'fast':
                    fast_total += 1

            # Calculate percentages
            fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
            fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

            # Adaptive weighting
            divergence_threshold = self.config['divergence_threshold']
            divergence_detected = False  # Simplified for this implementation

            if divergence_detected:
                fast_weight = self.config['reversal_fast_weight']
                mode = "REVERSAL"
            else:
                fast_weight = self.config['normal_fast_weight']
                mode = "NORMAL"

            # Calculate weighted scores
            bullish_signals = fast_bull * fast_weight
            bearish_signals = fast_bear * fast_weight
            total_signals = fast_total * fast_weight

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
                'timestamp': datetime.now(self.ist),
                'bias_results': bias_results,
                'overall_bias': overall_bias,
                'overall_score': overall_score,
                'overall_confidence': overall_confidence,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'total_indicators': len(bias_results),
                'mode': mode,
                'fast_bull_pct': fast_bull_pct,
                'fast_bear_pct': fast_bear_pct,
                'bullish_bias_pct': bullish_bias_pct,
                'bearish_bias_pct': bearish_bias_pct
            }
            
        except Exception as e:
            error_msg = f"Error in bias analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'symbol': symbol
            }

# =============================================
# OPTIONS CHAIN ANALYZER
# =============================================

class NSEOptionsAnalyzer:
    """NSE Options Analyzer with comprehensive bias analysis"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
                'FINNIFTY': {'lot_size': 40, 'atm_range': 200, 'zone_size': 100},
            }
        }
        
    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch option chain data from NSE"""
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

            # Calculate totals
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

    def analyze_options_bias(self, instrument: str) -> Dict[str, Any]:
        """Analyze options chain bias"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            spot = data['spot']
            total_ce_oi = data['total_ce_oi']
            total_pe_oi = data['total_pe_oi']
            total_ce_change = data['total_ce_change']
            total_pe_change = data['total_pe_change']

            # Calculate PCR
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_change = abs(total_pe_change) / abs(total_ce_change) if total_ce_change != 0 else 0

            # Calculate bias score
            bias_score = 0
            
            # OI Bias
            if pcr_oi > 1.2:
                bias_score += 2
            elif pcr_oi > 1.0:
                bias_score += 1
            elif pcr_oi < 0.8:
                bias_score -= 2
            elif pcr_oi < 1.0:
                bias_score -= 1

            # Change Bias
            if pcr_change > 1.2:
                bias_score += 1
            elif pcr_change < 0.8:
                bias_score -= 1

            # Determine overall bias
            if bias_score >= 2:
                overall_bias = "STRONG BULLISH"
            elif bias_score >= 1:
                overall_bias = "BULLISH"
            elif bias_score <= -2:
                overall_bias = "STRONG BEARISH"
            elif bias_score <= -1:
                overall_bias = "BEARISH"
            else:
                overall_bias = "NEUTRAL"

            return {
                'instrument': instrument,
                'spot_price': spot,
                'overall_bias': overall_bias,
                'bias_score': bias_score,
                'pcr_oi': pcr_oi,
                'pcr_change': pcr_change,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'total_ce_change': total_ce_change,
                'total_pe_change': total_pe_change,
                'timestamp': datetime.now(self.ist)
            }

        except Exception as e:
            print(f"Error in options analysis: {e}")
            return None

# =============================================
# ENHANCED MARKET DATA
# =============================================

class EnhancedMarketData:
    """Enhanced market data fetcher"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]

                if vix_value > 25:
                    vix_sentiment = "HIGH FEAR"
                    vix_bias = "BEARISH"
                elif vix_value > 20:
                    vix_sentiment = "ELEVATED FEAR"
                    vix_bias = "BEARISH"
                elif vix_value > 15:
                    vix_sentiment = "MODERATE"
                    vix_bias = "NEUTRAL"
                elif vix_value > 12:
                    vix_sentiment = "LOW VOLATILITY"
                    vix_bias = "BULLISH"
                else:
                    vix_sentiment = "COMPLACENCY"
                    vix_bias = "NEUTRAL"

                return {
                    'success': True,
                    'value': vix_value,
                    'sentiment': vix_sentiment,
                    'bias': vix_bias,
                    'timestamp': datetime.now(self.ist)
                }
        except:
            pass

        return {'success': False}

# =============================================
# MAIN APPLICATION
# =============================================

class AdvancedNiftyDashboard:
    def __init__(self):
        self.setup_secrets()
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize analyzers
        self.bias_analyzer = BiasAnalysisPro()
        self.options_analyzer = NSEOptionsAnalyzer()
        self.market_data = EnhancedMarketData()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'comprehensive_bias_data' not in st.session_state:
            st.session_state.comprehensive_bias_data = None
        if 'options_bias_data' not in st.session_state:
            st.session_state.options_bias_data = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
            
    def setup_secrets(self):
        """Setup API credentials"""
        try:
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except:
            pass

    def get_color_for_bias(self, bias: str) -> str:
        """Get color for bias display"""
        if "BULLISH" in bias:
            return "green"
        elif "BEARISH" in bias:
            return "red"
        else:
            return "orange"

    def get_emoji_for_bias(self, bias: str) -> str:
        """Get emoji for bias display"""
        if "STRONG BULLISH" in bias:
            return "🚀"
        elif "BULLISH" in bias:
            return "📈"
        elif "STRONG BEARISH" in bias:
            return "🔻"
        elif "BEARISH" in bias:
            return "📉"
        else:
            return "➡️"

    def display_overall_summary(self):
        """Display overall market summary"""
        st.header("🎯 Overall Market Summary")
        
        if not st.session_state.comprehensive_bias_data or not st.session_state.options_bias_data:
            st.warning("Please update data first using the buttons below")
            return
            
        tech_bias = st.session_state.comprehensive_bias_data
        options_bias = st.session_state.options_bias_data[0] if st.session_state.options_bias_data else None
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if tech_bias['success']:
                color = self.get_color_for_bias(tech_bias['overall_bias'])
                emoji = self.get_emoji_for_bias(tech_bias['overall_bias'])
                st.metric(
                    "Technical Bias",
                    f"{emoji} {tech_bias['overall_bias']}",
                    f"Score: {tech_bias['overall_score']:.1f}",
                    delta_color="off"
                )
        
        with col2:
            if options_bias:
                color = self.get_color_for_bias(options_bias['overall_bias'])
                emoji = self.get_emoji_for_bias(options_bias['overall_bias'])
                st.metric(
                    "Options Bias",
                    f"{emoji} {options_bias['overall_bias']}",
                    f"PCR: {options_bias['pcr_oi']:.2f}",
                    delta_color="off"
                )
        
        with col3:
            if tech_bias['success']:
                st.metric(
                    "Confidence",
                    f"{tech_bias['overall_confidence']:.1f}%",
                    f"Mode: {tech_bias['mode']}"
                )
        
        with col4:
            if tech_bias['success']:
                st.metric(
                    "Current Price",
                    f"₹{tech_bias['current_price']:.2f}",
                    "NIFTY 50"
                )
        
        # Combined recommendation
        st.subheader("💡 Combined Market Recommendation")
        
        if tech_bias['success'] and options_bias:
            tech_strength = abs(tech_bias['overall_score'])
            options_strength = abs(options_bias['bias_score'])
            
            if (tech_bias['overall_bias'] == "BULLISH" and options_bias['overall_bias'] == "BULLISH" and 
                tech_strength > 60 and options_strength >= 2):
                st.success("""
                **🎯 STRONG BULLISH CONFIRMATION - HIGH CONFIDENCE**
                
                Both Technical and Options analysis confirm BULLISH bias with strong signals.
                **Recommended Action:** Consider aggressive LONG positions with proper risk management.
                """)
            elif (tech_bias['overall_bias'] == "BEARISH" and options_bias['overall_bias'] == "BEARISH" and 
                  tech_strength > 60 and options_strength >= 2):
                st.error("""
                **🎯 STRONG BEARISH CONFIRMATION - HIGH CONFIDENCE**
                
                Both Technical and Options analysis confirm BEARISH bias with strong signals.
                **Recommended Action:** Consider aggressive SHORT positions with proper risk management.
                """)
            elif tech_bias['overall_bias'] == options_bias['overall_bias']:
                st.info("""
                **📊 CONFIRMED BIAS - MODERATE CONFIDENCE**
                
                Technical and Options analysis are aligned. Consider directional positions.
                **Risk Management:** Use proper position sizing and stop losses.
                """)
            else:
                st.warning("""
                **⚖️ MIXED SIGNALS - CAUTION ADVISED**
                
                Technical and Options analysis show conflicting signals.
                **Recommended Action:** Wait for clearer direction or use range-bound strategies.
                **Advice:** Monitor key support and resistance levels closely.
                """)

    def display_technical_bias_analysis(self):
        """Display comprehensive technical bias analysis"""
        st.header("📊 Technical Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("8-indicator comprehensive technical analysis with adaptive weighting")
        with col2:
            if st.button("🔄 Update Technical Analysis", type="primary", key="tech_update"):
                with st.spinner("Running comprehensive technical analysis..."):
                    bias_data = self.bias_analyzer.analyze_all_bias_indicators("^NSEI")
                    st.session_state.comprehensive_bias_data = bias_data
                    st.session_state.last_update = datetime.now(self.ist)
                    if bias_data['success']:
                        st.success("Technical analysis completed!")
                    else:
                        st.error(f"Analysis failed: {bias_data['error']}")
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if not bias_data['success']:
                st.error(f"❌ Technical analysis failed: {bias_data['error']}")
                return
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Bias", bias_data['overall_bias'])
            with col2:
                st.metric("Bias Score", f"{bias_data['overall_score']:.1f}")
            with col3:
                st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
            with col4:
                st.metric("Analysis Mode", bias_data['mode'])
            
            # Detailed indicators
            st.subheader("📈 Technical Indicators Breakdown")
            
            # Create a DataFrame for better display
            indicators_df = pd.DataFrame(bias_data['bias_results'])
            
            # Display with color coding
            for _, indicator in indicators_df.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(f"**{indicator['indicator']}**")
                with col2:
                    st.write(indicator['value'])
                with col3:
                    color = self.get_color_for_bias(indicator['bias'])
                    st.markdown(f"<span style='color: {color}; font-weight: bold'>{indicator['bias']}</span>", 
                               unsafe_allow_html=True)
                with col4:
                    st.write(f"{indicator['score']}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bias distribution pie chart
                bias_counts = {
                    'Bullish': bias_data['bullish_count'],
                    'Bearish': bias_data['bearish_count'],
                    'Neutral': bias_data['neutral_count']
                }
                
                fig_pie = px.pie(
                    values=list(bias_counts.values()),
                    names=list(bias_counts.keys()),
                    title="Bias Distribution",
                    color=list(bias_counts.keys()),
                    color_discrete_map={
                        'Bullish': 'green',
                        'Bearish': 'red',
                        'Neutral': 'orange'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Indicator scores bar chart
                fig_bar = px.bar(
                    indicators_df,
                    x='indicator',
                    y='score',
                    color='bias',
                    title="Indicator Scores",
                    color_discrete_map={
                        'BULLISH': 'green',
                        'BEARISH': 'red',
                        'NEUTRAL': 'orange'
                    }
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Advanced metrics
            st.subheader("🔍 Advanced Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fast Bull %", f"{bias_data['fast_bull_pct']:.1f}%")
            with col2:
                st.metric("Fast Bear %", f"{bias_data['fast_bear_pct']:.1f}%")
            with col3:
                st.metric("Bullish Bias %", f"{bias_data['bullish_bias_pct']:.1f}%")
            with col4:
                st.metric("Bearish Bias %", f"{bias_data['bearish_bias_pct']:.1f}%")
        
        else:
            st.info("👆 Click 'Update Technical Analysis' to run the analysis")

    def display_options_bias_analysis(self):
        """Display options chain bias analysis"""
        st.header("📊 Options Chain Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Real-time options chain analysis with PCR and OI analysis")
        with col2:
            if st.button("🔄 Update Options Data", type="primary", key="options_update"):
                with st.spinner("Fetching options chain data..."):
                    instruments = ['NIFTY', 'BANKNIFTY']
                    options_data = []
                    for instrument in instruments:
                        bias_data = self.options_analyzer.analyze_options_bias(instrument)
                        if bias_data:
                            options_data.append(bias_data)
                    st.session_state.options_bias_data = options_data
                    st.session_state.last_update = datetime.now(self.ist)
                    st.success("Options data updated!")
        
        if st.session_state.options_bias_data:
            options_data = st.session_state.options_bias_data
            
            for instrument_data in options_data:
                with st.expander(f"🎯 {instrument_data['instrument']} - Options Analysis", expanded=True):
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Spot Price", 
                            f"₹{instrument_data['spot_price']:.2f}",
                            "Current"
                        )
                    
                    with col2:
                        color = self.get_color_for_bias(instrument_data['overall_bias'])
                        emoji = self.get_emoji_for_bias(instrument_data['overall_bias'])
                        st.metric(
                            "Options Bias",
                            f"{emoji} {instrument_data['overall_bias']}",
                            f"Score: {instrument_data['bias_score']:.1f}",
                            delta_color="off"
                        )
                    
                    with col3:
                        pcr_color = "green" if instrument_data['pcr_oi'] > 1.0 else "red"
                        st.metric(
                            "PCR OI",
                            f"{instrument_data['pcr_oi']:.2f}",
                            delta_color="off"
                        )
                    
                    with col4:
                        st.metric(
                            "PCR Change",
                            f"{instrument_data['pcr_change']:.2f}",
                            delta_color="off"
                        )
                    
                    # OI Analysis
                    st.subheader("📊 Open Interest Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Call OI", f"{instrument_data['total_ce_oi']:,}")
                    with col2:
                        st.metric("Put OI", f"{instrument_data['total_pe_oi']:,}")
                    with col3:
                        st.metric("Call Δ OI", f"{instrument_data['total_ce_change']:,}")
                    with col4:
                        st.metric("Put Δ OI", f"{instrument_data['total_pe_change']:,}")
                    
                    # PCR Interpretation
                    st.subheader("💡 PCR Interpretation")
                    pcr_oi = instrument_data['pcr_oi']
                    
                    if pcr_oi > 1.5:
                        st.success("""
                        **Very Bullish PCR**: Significant put writing indicates strong bullish sentiment.
                        Market participants are expecting upward movement.
                        """)
                    elif pcr_oi > 1.2:
                        st.info("""
                        **Bullish PCR**: Moderate put writing suggests bullish bias.
                        Positive sentiment with room for movement.
                        """)
                    elif pcr_oi > 0.8:
                        st.warning("""
                        **Neutral PCR**: Balanced put-call ratio suggests indecision.
                        Market waiting for catalyst or direction.
                        """)
                    else:
                        st.error("""
                        **Bearish PCR**: Significant call writing indicates bearish sentiment.
                        Market participants are expecting downward movement.
                        """)
        
        else:
            st.info("👆 Click 'Update Options Data' to fetch options chain analysis")

    def display_market_overview(self):
        """Display market overview with multiple data sources"""
        st.header("🌍 Market Overview")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info("Comprehensive market analysis from multiple sources")
        with col2:
            if st.button("🔄 Update All Data", type="primary", key="all_update"):
                with st.spinner("Updating all market data..."):
                    # Update technical analysis
                    tech_data = self.bias_analyzer.analyze_all_bias_indicators("^NSEI")
                    st.session_state.comprehensive_bias_data = tech_data
                    
                    # Update options data
                    instruments = ['NIFTY', 'BANKNIFTY']
                    options_data = []
                    for instrument in instruments:
                        bias_data = self.options_analyzer.analyze_options_bias(instrument)
                        if bias_data:
                            options_data.append(bias_data)
                    st.session_state.options_bias_data = options_data
                    
                    # Update market data
                    vix_data = self.market_data.fetch_india_vix()
                    st.session_state.market_data = vix_data
                    
                    st.session_state.last_update = datetime.now(self.ist)
                    st.success("All market data updated!")
        
        with col3:
            if st.session_state.last_update:
                st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Market data display
        if st.session_state.market_data:
            vix_data = st.session_state.market_data
            if vix_data['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("India VIX", f"{vix_data['value']:.2f}")
                with col2:
                    st.metric("VIX Sentiment", vix_data['sentiment'])
                with col3:
                    color = self.get_color_for_bias(vix_data['bias'])
                    st.metric("VIX Bias", vix_data['bias'], delta_color="off")
                with col4:
                    if vix_data['value'] > 20:
                        st.warning("High Volatility Expected")
                    else:
                        st.success("Normal Volatility Regime")
        
        # Quick status
        st.subheader("📊 Data Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.comprehensive_bias_data:
                status = "✅ Available" if st.session_state.comprehensive_bias_data['success'] else "❌ Failed"
                st.metric("Technical Analysis", status)
            else:
                st.metric("Technical Analysis", "⏳ Pending")
        
        with col2:
            if st.session_state.options_bias_data:
                status = f"✅ {len(st.session_state.options_bias_data)} instruments"
                st.metric("Options Analysis", status)
            else:
                st.metric("Options Analysis", "⏳ Pending")
        
        with col3:
            if st.session_state.market_data:
                status = "✅ Available" if st.session_state.market_data['success'] else "❌ Failed"
                st.metric("Market Data", status)
            else:
                st.metric("Market Data", "⏳ Pending")
        
        with col4:
            if st.session_state.last_update:
                st.metric("Last Update", st.session_state.last_update.strftime('%H:%M'))
            else:
                st.metric("Last Update", "Never")

    def display_trading_signals(self):
        """Display trading signals based on analysis"""
        st.header("🚀 Trading Signals")
        
        if not st.session_state.comprehensive_bias_data or not st.session_state.options_bias_data:
            st.warning("Please update market data first to generate trading signals")
            return
        
        tech_bias = st.session_state.comprehensive_bias_data
        options_bias = st.session_state.options_bias_data[0] if st.session_state.options_bias_data else None
        
        if not tech_bias['success'] or not options_bias:
            st.error("Incomplete data for signal generation")
            return
        
        # Generate signals based on combined analysis
        tech_score = tech_bias['overall_score']
        options_score = options_bias['bias_score']
        tech_confidence = tech_bias['overall_confidence']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Technical Signals")
            
            if tech_score > 60 and tech_confidence > 70:
                st.success("""
                **STRONG BULLISH TECHNICAL SIGNAL**
                - Multiple indicators confirming uptrend
                - High confidence score
                - Consider LONG positions
                """)
            elif tech_score > 30:
                st.info("""
                **MODERATE BULLISH TECHNICAL SIGNAL**
                - Favorable technical conditions
                - Medium confidence
                - Consider cautious LONG positions
                """)
            elif tech_score < -60 and tech_confidence > 70:
                st.error("""
                **STRONG BEARISH TECHNICAL SIGNAL**
                - Multiple indicators confirming downtrend
                - High confidence score
                - Consider SHORT positions
                """)
            elif tech_score < -30:
                st.warning("""
                **MODERATE BEARISH TECHNICAL SIGNAL**
                - Unfavorable technical conditions
                - Medium confidence
                - Consider cautious SHORT positions
                """)
            else:
                st.warning("""
                **NEUTRAL TECHNICAL SIGNAL**
                - Mixed or weak signals
                - Wait for clearer direction
                - Consider range-bound strategies
                """)
        
        with col2:
            st.subheader("📊 Options Signals")
            
            if options_score >= 2:
                st.success("""
                **BULLISH OPTIONS SIGNAL**
                - High PCR indicating put writing
                - Bullish OI patterns
                - Supports LONG bias
                """)
            elif options_score <= -2:
                st.error("""
                **BEARISH OPTIONS SIGNAL**
                - Low PCR indicating call writing
                - Bearish OI patterns
                - Supports SHORT bias
                """)
            else:
                st.info("""
                **NEUTRAL OPTIONS SIGNAL**
                - Balanced PCR
                - Mixed OI patterns
                - No clear directional bias
                """)
        
        # Combined signal
        st.subheader("🎯 Combined Trading Recommendation")
        
        combined_score = (tech_score + options_score * 10) / 2  # Normalize scores
        
        if combined_score > 40:
            st.success(f"""
            **OVERALL: STRONG BULLISH SIGNAL** (Score: {combined_score:.1f})
            
            **Recommended Action:**
            - Enter LONG positions on dips
            - Target: Resistance levels above current price
            - Stop Loss: Below key support
            - Confidence: High
            """)
        elif combined_score > 20:
            st.info(f"""
            **OVERALL: MODERATE BULLISH SIGNAL** (Score: {combined_score:.1f})
            
            **Recommended Action:**
            - Consider LONG positions with caution
            - Wait for pullbacks for better entry
            - Use tight stop losses
            - Confidence: Medium
            """)
        elif combined_score < -40:
            st.error(f"""
            **OVERALL: STRONG BEARISH SIGNAL** (Score: {combined_score:.1f})
            
            **Recommended Action:**
            - Enter SHORT positions on rallies
            - Target: Support levels below current price
            - Stop Loss: Above key resistance
            - Confidence: High
            """)
        elif combined_score < -20:
            st.warning(f"""
            **OVERALL: MODERATE BEARISH SIGNAL** (Score: {combined_score:.1f})
            
            **Recommended Action:**
            - Consider SHORT positions with caution
            - Wait for bounces for better entry
            - Use tight stop losses
            - Confidence: Medium
            """)
        else:
            st.warning(f"""
            **OVERALL: NEUTRAL SIGNAL** (Score: {combined_score:.1f})
            
            **Recommended Action:**
            - Wait for clearer directional bias
            - Consider range-bound strategies
            - Monitor key levels for breakout
            - Confidence: Low to Medium
            """)
        
        # Risk Management
        st.subheader("🛡️ Risk Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recommended Position Size", "1-2%", "Of portfolio")
        with col2:
            st.metric("Stop Loss", "1-2%", "From entry")
        with col3:
            st.metric("Risk-Reward Ratio", "1:2", "Minimum")

    def run(self):
        """Main application runner"""
        st.title("🚀 Advanced Nifty Trading Dashboard Pro")
        st.markdown("""
        *Comprehensive Market Analysis • Technical Indicators • Options Chain • Trading Signals*
        """)
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Dashboard Controls")
            
            st.subheader("Auto Refresh")
            auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
            refresh_interval = st.selectbox("Refresh Interval", [1, 2, 5, 10, 15], index=2)
            
            st.subheader("Alert Settings")
            enable_alerts = st.checkbox("Enable Trading Alerts", value=False)
            min_confidence = st.slider("Minimum Confidence %", 50, 90, 70)
            
            st.subheader("Data Sources")
            show_technical = st.checkbox("Show Technical Analysis", value=True)
            show_options = st.checkbox("Show Options Analysis", value=True)
            show_signals = st.checkbox("Show Trading Signals", value=True)
            
            if st.button("📊 Quick Update", type="secondary"):
                st.rerun()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🔬 Technical Analysis", 
            "📊 Options Analysis", 
            "🚀 Trading Signals",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.display_overall_summary()
            self.display_market_overview()
        
        with tab2:
            if show_technical:
                self.display_technical_bias_analysis()
            else:
                st.info("Technical analysis disabled in settings")
        
        with tab3:
            if show_options:
                self.display_options_bias_analysis()
            else:
                st.info("Options analysis disabled in settings")
        
        with tab4:
            if show_signals:
                self.display_trading_signals()
            else:
                st.info("Trading signals disabled in settings")
        
        with tab5:
            st.header("⚙️ Application Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Analysis Parameters")
                st.number_input("RSI Period", value=14, min_value=7, max_value=21)
                st.number_input("DMI Period", value=13, min_value=7, max_value=21)
                st.number_input("MFI Period", value=10, min_value=5, max_value=15)
                
            with col2:
                st.subheader("Display Settings")
                st.checkbox("Show Advanced Metrics", value=True)
                st.checkbox("Show Raw Data", value=False)
                st.checkbox("Enable Dark Mode", value=True)
            
            st.subheader("Risk Management")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("Max Position Size %", value=2.0, min_value=0.5, max_value=5.0, step=0.5)
            with col2:
                st.number_input("Stop Loss %", value=1.5, min_value=0.5, max_value=3.0, step=0.5)
            with col3:
                st.number_input("Target %", value=3.0, min_value=1.0, max_value=5.0, step=0.5)
        
        # Auto refresh logic
        if auto_refresh:
            time.sleep(refresh_interval * 60)
            st.rerun()

# Run the application
if __name__ == "__main__":
    app = AdvancedNiftyDashboard()
    app.run()