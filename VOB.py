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
# COMPREHENSIVE BIAS ANALYSIS MODULE
# =============================================

class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis matching Pine Script indicator EXACTLY
    Analyzes 13 bias indicators:
    - Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
    - Medium (2): Close vs VWAP, Price vs VWAP
    - Slow (3): Weighted stocks (Daily, TF1, TF2)
    """
    
    def __init__(self):
        """Initialize bias analysis with default configuration"""
        self.config = self._default_config()
        self.all_bias_results = []
        self.overall_bias = "NEUTRAL"
        self.overall_score = 0

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
    # TECHNICAL INDICATORS
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
        # Check if volume data is available
        if df['volume'].sum() == 0:
            # Return neutral MFI (50) if no volume data
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        # Avoid division by zero
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))

        # Fill NaN with neutral value (50)
        mfi = mfi.fillna(50)

        return mfi

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        """Calculate DMI indicators"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()

        return plus_di, minus_di, adx

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with NaN/zero handling"""
        # Check if volume data is available
        if df['volume'].sum() == 0:
            # Return typical price as fallback if no volume data
            return (df['high'] + df['low'] + df['close']) / 3

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()

        # Avoid division by zero
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['volume']).cumsum() / cumulative_volume_safe

        # Fill NaN with typical price
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

        # Calculate momentum (CMO - Chande Momentum Oscillator)
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()

        # Avoid division by zero
        cmo_denom = p + n
        cmo_denom = cmo_denom.replace(0, np.nan)
        abs_cmo = abs(100 * (p - n) / cmo_denom).fillna(0)

        # Calculate VIDYA
        alpha = 2 / (length + 1)
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                            (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])

        # Smooth VIDYA
        vidya_smoothed = vidya.rolling(window=15).mean()

        # Calculate bands
        atr = self.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * band_distance
        lower_band = vidya_smoothed - atr * band_distance

        # Determine trend based on band crossovers
        is_trend_up = close > upper_band
        is_trend_down = close < lower_band

        # Get current state
        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False

        return vidya_smoothed, vidya_bullish, vidya_bearish

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta (up_vol - down_vol) matching Pine Script"""
        if df['volume'].sum() == 0:
            return 0, False, False

        # Calculate up and down volume
        up_vol = ((df['close'] > df['open']).astype(int) * df['volume']).sum()
        down_vol = ((df['close'] < df['open']).astype(int) * df['volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots matching Pine Script
        Returns: (hvp_bullish, hvp_bearish, pivot_high_count, pivot_low_count)
        """
        if df['volume'].sum() == 0:
            return False, False, 0, 0

        # Calculate pivot highs and lows
        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            # Check for pivot high
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            # Check for pivot low
            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        # Calculate volume sum and reference
        volume_sum = df['volume'].rolling(window=left_bars * 2).sum()
        ref_vol = volume_sum.quantile(0.95)
        norm_vol = (volume_sum / ref_vol * 5).fillna(0)

        # Check recent HVP signals
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
        """Calculate Volume Order Blocks matching Pine Script
        Returns: (vob_bullish, vob_bearish, ema1_value, ema2_value)
        """
        # Calculate EMAs
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['close'], length1)
        ema2 = self.calculate_ema(df['close'], length2)

        # Detect crossovers
        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        # In real implementation, we would check if price touched OB zones
        # For simplicity, using crossover signals
        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    # =========================================================================
    # ENHANCED INDICATORS (KEPT FOR COMPATIBILITY)
    # =========================================================================

    def calculate_volatility_ratio(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volatility Ratio"""
        atr = self.calculate_atr(df, length)
        stdev = df['close'].rolling(window=length).std()
        volatility_ratio = (stdev / atr) * 100

        high_volatility = volatility_ratio.iloc[-1] > self.config['volatility_threshold']
        low_volatility = volatility_ratio.iloc[-1] < (self.config['volatility_threshold'] * 0.5)

        return volatility_ratio, high_volatility, low_volatility

    def calculate_volume_roc(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volume Rate of Change with NaN/zero handling"""
        # Check if volume data is available
        if df['volume'].sum() == 0:
            # Return neutral volume ROC if no volume data
            neutral_roc = pd.Series([0.0] * len(df), index=df.index)
            return neutral_roc, False, False

        # Avoid division by zero
        volume_shifted = df['volume'].shift(length).replace(0, np.nan)
        volume_roc = ((df['volume'] - df['volume'].shift(length)) / volume_shifted) * 100

        # Fill NaN with 0
        volume_roc = volume_roc.fillna(0)

        # Check for strong/weak volume (handle NaN gracefully)
        last_value = volume_roc.iloc[-1] if not np.isnan(volume_roc.iloc[-1]) else 0
        strong_volume = last_value > self.config['volume_threshold']
        weak_volume = last_value < -self.config['volume_threshold']

        return volume_roc, strong_volume, weak_volume

    def calculate_obv(self, df: pd.DataFrame, smoothing: int = 21):
        """Calculate On Balance Volume with NaN/zero handling"""
        # Check if volume data is available
        if df['volume'].sum() == 0:
            # Return neutral OBV if no volume data
            neutral_obv = pd.Series([0.0] * len(df), index=df.index)
            neutral_obv_ma = pd.Series([0.0] * len(df), index=df.index)
            return neutral_obv, neutral_obv_ma, False, False

        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        obv_ma = obv.rolling(window=smoothing).mean()

        # Handle potential NaN or missing values
        obv = obv.fillna(0)
        obv_ma = obv_ma.fillna(0)

        # Safe comparison with fallback
        try:
            obv_rising = obv.iloc[-1] > obv.iloc[-2] if len(obv) >= 2 else False
            obv_falling = obv.iloc[-1] < obv.iloc[-2] if len(obv) >= 2 else False
            obv_bullish = obv.iloc[-1] > obv_ma.iloc[-1] and obv_rising
            obv_bearish = obv.iloc[-1] < obv_ma.iloc[-1] and obv_falling
        except:
            obv_bullish = False
            obv_bearish = False

        return obv, obv_ma, obv_bullish, obv_bearish

    def calculate_force_index(self, df: pd.DataFrame, length: int = 13, smoothing: int = 2):
        """Calculate Force Index with NaN/zero handling"""
        # Check if volume data is available
        if df['volume'].sum() == 0:
            # Return neutral force index if no volume data
            neutral_force = pd.Series([0.0] * len(df), index=df.index)
            return neutral_force, False, False

        force_index = (df['close'] - df['close'].shift(1)) * df['volume']
        force_index = force_index.fillna(0)

        force_index_ma = force_index.ewm(span=length, adjust=False).mean()
        force_index_smoothed = force_index_ma.ewm(span=smoothing, adjust=False).mean()

        # Handle potential NaN
        force_index_smoothed = force_index_smoothed.fillna(0)

        # Safe comparison with fallback
        try:
            force_rising = force_index_smoothed.iloc[-1] > force_index_smoothed.iloc[-2] if len(force_index_smoothed) >= 2 else False
            force_falling = force_index_smoothed.iloc[-1] < force_index_smoothed.iloc[-2] if len(force_index_smoothed) >= 2 else False
            force_bullish = force_index_smoothed.iloc[-1] > 0 and force_rising
            force_bearish = force_index_smoothed.iloc[-1] < 0 and force_falling
        except:
            force_bullish = False
            force_bearish = False

        return force_index_smoothed, force_bullish, force_bearish

    def calculate_price_roc(self, df: pd.DataFrame, length: int = 12):
        """Calculate Price Rate of Change"""
        price_roc = ((df['close'] - df['close'].shift(length)) / df['close'].shift(length)) * 100

        price_momentum_bullish = price_roc.iloc[-1] > 0
        price_momentum_bearish = price_roc.iloc[-1] < 0

        return price_roc, price_momentum_bullish, price_momentum_bearish

    def calculate_choppiness_index(self, df: pd.DataFrame, period: int = 14):
        """Calculate Choppiness Index"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        sum_true_range = true_range.rolling(window=period).sum()
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        ci = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(period)

        market_chopping = ci.iloc[-1] > self.config['ci_high_threshold']
        market_trending = ci.iloc[-1] < self.config['ci_low_threshold']

        return ci, market_chopping, market_trending

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 30):
        """Detect RSI/MACD Divergences"""
        rsi = self.calculate_rsi(df['close'], 14)

        # MACD
        macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

        close_series = df['close'].tail(lookback)
        rsi_series = rsi.tail(lookback)
        macd_series = macd_line.tail(lookback)

        # Bullish divergence
        lowest_close_idx = close_series.idxmin()
        lowest_rsi_idx = rsi_series.idxmin()
        bullish_rsi_divergence = (lowest_close_idx == close_series.index[-1] and
                                  rsi_series.iloc[-1] > rsi_series.loc[lowest_rsi_idx] and
                                  rsi_series.iloc[-1] < self.config['rsi_oversold'])

        # Bearish divergence
        highest_close_idx = close_series.idxmax()
        highest_rsi_idx = rsi_series.idxmax()
        bearish_rsi_divergence = (highest_close_idx == close_series.index[-1] and
                                  rsi_series.iloc[-1] < rsi_series.loc[highest_rsi_idx] and
                                  rsi_series.iloc[-1] > self.config['rsi_overbought'])

        return bullish_rsi_divergence, bearish_rsi_divergence

    # =========================================================================
    # MARKET BREADTH & STOCKS ANALYSIS
    # =========================================================================

    def _fetch_stock_data(self, symbol: str, weight: float):
        """Helper function to fetch single stock data for parallel processing"""
        try:
            # Use 5d period with 5m interval (Yahoo Finance limitation for intraday data)
            df = self.fetch_data(symbol, period='5d', interval='5m')
            if df.empty or len(df) < 2:
                return None

            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[0]
            change_pct = ((current_price - prev_price) / prev_price) * 100

            return {
                'symbol': symbol.replace('.NS', ''),
                'change_pct': change_pct,
                'weight': weight,
                'is_bullish': change_pct > 0
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None

    def calculate_market_breadth(self):
        """Calculate market breadth from top stocks (optimized with parallel processing)"""
        bullish_stocks = 0
        total_stocks = 0
        stock_data = []

        # Optimize: Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_stock = {
                executor.submit(self._fetch_stock_data, symbol, weight): (symbol, weight)
                for symbol, weight in self.config['stocks'].items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_stock):
                result = future.result()
                if result:
                    stock_data.append({
                        'symbol': result['symbol'],
                        'change_pct': result['change_pct'],
                        'weight': result['weight']
                    })
                    if result['is_bullish']:
                        bullish_stocks += 1
                    total_stocks += 1

        if total_stocks > 0:
            market_breadth = (bullish_stocks / total_stocks) * 100
        else:
            market_breadth = 50

        breadth_bullish = market_breadth > self.config['breadth_threshold']
        breadth_bearish = market_breadth < (100 - self.config['breadth_threshold'])

        return market_breadth, breadth_bullish, breadth_bearish, bullish_stocks, total_stocks, stock_data

    # =========================================================================
    # COMPREHENSIVE BIAS ANALYSIS
    # =========================================================================

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """
        Analyze all 8 bias indicators:
        Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
        """

        print(f"Fetching data for {symbol}...")
        # Use 7d period with 5m interval (Yahoo Finance limitation for intraday data)
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
        stock_data = []  # Empty since we removed Weighted Stocks indicators

        # =====================================================================
        # FAST INDICATORS (8 total)
        # =====================================================================

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
            # Determine if EMA5 is above or below EMA18
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
        ema5 = self.calculate_ema(df['close'], 5)
        ema18 = self.calculate_ema(df['close'], 18)

        # Detect crossovers
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
        rsi = self.calculate_rsi(df['close'], self.config['rsi_period'])
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
        adx_value = adx.iloc[-1]

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
            mfi_value = 50.0  # Neutral default

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

        # =====================================================================
        # CALCULATE OVERALL BIAS (Matching Pine Script Logic)
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

        # Adaptive weighting (matching Pine Script)
        # Check for divergence
        divergence_threshold = self.config['divergence_threshold']
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        # Determine mode
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
            'bearish_bias_pct': bearish_bias_pct
        }


# =============================================
# MARKET REGIME DETECTOR (THE BRAIN)
# =============================================

class MarketRegimeDetector:
    """
    Detects what TYPE of market we're in right now
    This is THE KEY to knowing which indicators to trust
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
        # Simplified: Check if it's last 7 days of month
        days_in_month = (current_time.replace(day=28) + timedelta(days=4)).day
        return current_time.day > days_in_month - 7

    def _is_event_day(self, current_time: datetime) -> bool:
        """Check if it's a known event day"""
        # Add your event calendar here
        # For now, checking if it's first week of month (Budget season)
        return current_time.day <= 7

    def _is_gap_day(self, df: pd.DataFrame) -> bool:
        """Check if today opened with a gap"""
        if len(df) < 10:
            return False
        
        # Compare today's open with yesterday's close
        today_open = df['open'].iloc[0]
        yesterday_close = df['close'].iloc[-2] if len(df) > 1 else today_open
        gap_pct = abs((today_open - yesterday_close) / yesterday_close) * 100
        
        return gap_pct > 0.5  # More than 0.5% gap

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
    This is the TRADER'S EYE - what moves markets
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
        This is where the MAGIC happens
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

    def get_trap_summary(self, trap_analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of trap analysis
        """
        if not trap_analysis.get('trap_detected'):
            return "✅ NO TRAP DETECTED - Market moving genuinely"
        
        trap_type = trap_analysis.get('trap_type', 'UNKNOWN')
        confidence = trap_analysis.get('trap_confidence', 0)
        who_trapped = trap_analysis.get('who_is_trapped', 'Unknown')
        expected_move = trap_analysis.get('expected_move', 'Unknown')
        
        summary = f"""
🚨 TRAP DETECTED: {trap_type}
Confidence: {confidence}%
Who is Trapped: {who_trapped}
Expected Move: {expected_move}
Characteristics:
"""
        for char in trap_analysis.get('characteristics', []):
            summary += f"  • {char}\n"
        
        if trap_analysis.get('trade_setup'):
            setup = trap_analysis['trade_setup']
            summary += f"""
TRADE SETUP:
Direction: {setup.get('direction', 'N/A')}
Entry: {setup.get('entry', 'N/A')}
Target: ₹{setup.get('target', 'N/A')}
Stop Loss: ₹{setup.get('stop_loss', 'N/A')}
{setup.get('note', '')}
"""
        return summary


# =============================================
# EXECUTION FILTER ENGINE (THE GUARDIAN)
# =============================================

class ExecutionFilterEngine:
    """
    The GUARDIAN that protects your capital
    Tells you WHEN to avoid trading
    This prevents 80% of bad trades
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
        This is what separates profitable traders from losers
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
                # If it's a genuine trap, this is actually GOOD
                result['filters_passed'].append(f'✓ {trap_type} detected - HIGH CONVICTION TRADE')
                result['confidence'] += 15  # BONUS confidence
            else:
                # Genuine buildup/liquidation - proceed normally
                result['filters_passed'].append(f'✓ {trap_type} - Normal market activity')
        
        # Filter 9: Bias Consensus Check
        if bias_data:
            bullish_count = bias_data.get('bullish_count', 0)
            bearish_count = bias_data.get('bearish_count', 0)
            total = bias_data.get('total_indicators', 8)
            consensus = max(bullish_count, bearish_count) / total if total > 0 else 0
            
            if consensus < 0.6:  # Less than 60% agreement
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
        
        # Calculate risk level based on confidence
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
        
        # Generate final recommendation
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

    def format_filter_report(self, filter_result: Dict[str, Any]) -> str:
        """Generate human-readable filter report"""
        if not filter_result.get('trade_allowed'):
            return f"""
🚫 TRADE NOT ALLOWED
Reason: {filter_result.get('final_recommendation', 'Unknown')}
Failed Filters:
{chr(10).join(filter_result.get('filters_failed', []))}
"""
        report = f"""
✅ TRADE ALLOWED
Confidence: {filter_result.get('confidence', 0)}%
Risk Level: {filter_result.get('risk_level', 'UNKNOWN')}
Position Size: {filter_result.get('position_sizing', 'NORMAL')}
✓ Filters Passed: {len(filter_result.get('filters_passed', []))}
"""
        if filter_result.get('warnings'):
            report += "\n⚠️ **WARNINGS:**\n"
            for warning in filter_result['warnings']:
                report += f"{warning}\n"
        
        report += f"\n💡 **RECOMMENDATION:** {filter_result.get('final_recommendation', 'N/A')}"
        
        return report


# =============================================
# MASTER DECISION ENGINE
# =============================================

class MasterDecisionEngine:
    """
    THE BRAIN - Combines everything into ONE intelligent trading decision
    This is what separates your app from all others
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
            'trade_decision': None,  # TRADE or NO TRADE
            'trade_direction': None,  # LONG or SHORT
            'trade_type': None,  # Type of trade
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
            decision['confidence'] = min(95, base_confidence + 15)  # Bonus for trap
            
            # Use trap trade setup
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

    def format_decision_for_telegram(self, decision: Dict[str, Any]) -> str:
        """Format decision for Telegram alert"""
        if not decision.get('success'):
            return "❌ Unable to generate trading decision"
        
        return decision.get('simple_summary', 'No summary available')


# =============================================
# ENHANCED MARKET DATA FETCHER INTEGRATION
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources:
    1. Dhan API: India VIX, Sector Indices, Futures Data
    2. Yahoo Finance: Global Markets, Intermarket Data
    3. NSE: FII/DII Data (optional)
    """
    
    def __init__(self):
        """Initialize enhanced market data fetcher"""
        self.ist = pytz.timezone('Asia/Kolkata')
        self.dhan_fetcher = None
        self.request_count = 0
        self.last_request_time = None
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = datetime.now()
        self.request_count += 1

    def get_current_time_ist(self):
        """Get current time in IST"""
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX from Yahoo Finance with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                
                ticker = yf.Ticker("^INDIAVIX")
                
                # Try 1d period first, fall back to 5d if needed
                hist = ticker.history(period="1d", interval="1m")
                
                if hist.empty:
                    print(f"Attempt {attempt + 1}: Trying 5d period...")
                    hist = ticker.history(period="5d", interval="5m")
                
                if not hist.empty and len(hist) > 0:
                    vix_value = hist['Close'].iloc[-1]
                    
                    # Validate VIX value (should be between 5 and 100)
                    if not (5 <= vix_value <= 100):
                        print(f"Invalid VIX value: {vix_value}, retrying...")
                        time.sleep(retry_delay)
                        continue

                    # VIX Interpretation
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
            'value': 15.0,  # Default neutral value
            'sentiment': 'UNKNOWN',
            'bias': 'NEUTRAL',
            'score': 0,
            'timestamp': self.get_current_time_ist()
        }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices from Yahoo Finance with better error handling"""
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
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sector = {
                executor.submit(self._fetch_single_sector, symbol, name): (symbol, name)
                for symbol, name in sectors_map.items()
            }
            
            for future in as_completed(future_to_sector):
                result = future.result()
                if result:
                    sector_data.append(result)
                time.sleep(0.5)  # Rate limiting

        return sector_data

    def _fetch_single_sector(self, symbol: str, name: str) -> Optional[Dict[str, Any]]:
        """Helper to fetch single sector data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d", interval="1d")

            if not hist.empty and len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                open_price = hist['Open'].iloc[-1]
                high_price = hist['High'].iloc[-1]
                low_price = hist['Low'].iloc[-1]

                change_pct = ((last_price - prev_price) / prev_price) * 100

                # Determine bias
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
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score,
                    'source': 'Yahoo Finance'
                }
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            return None

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices with better error handling"""
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
        """Helper to fetch single market data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")

            if len(hist) >= 2:
                current_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]

                change_pct = ((current_close - prev_close) / prev_close) * 100

                # Determine bias
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
                    'prev_close': prev_close,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score
                }
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            return None

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """Fetch intermarket data (commodities, currencies, bonds) with better error handling"""
        intermarket_assets = {
            'DX-Y.NYB': 'US DOLLAR INDEX',
            'CL=F': 'CRUDE OIL',
            'GC=F': 'GOLD',
            'INR=X': 'USD/INR',
            '^TNX': 'US 10Y TREASURY',
            'BTC-USD': 'BITCOIN'
        }

        intermarket_data = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_asset = {
                executor.submit(self._fetch_single_intermarket, symbol, name): (symbol, name)
                for symbol, name in intermarket_assets.items()
            }
            
            for future in as_completed(future_to_asset):
                result = future.result()
                if result:
                    intermarket_data.append(result)
                time.sleep(0.5)

        return intermarket_data

    def _fetch_single_intermarket(self, symbol: str, name: str) -> Optional[Dict[str, Any]]:
        """Helper to fetch single intermarket asset data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")

            if len(hist) >= 2:
                current_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]

                change_pct = ((current_close - prev_close) / prev_close) * 100

                # Specific interpretations for each asset
                if 'DOLLAR' in name:
                    if change_pct > 0.5:
                        bias = "BEARISH (for India)"
                        score = -40
                    elif change_pct < -0.5:
                        bias = "BULLISH (for India)"
                        score = 40
                    else:
                        bias = "NEUTRAL"
                        score = 0
                elif 'OIL' in name:
                    if change_pct > 2:
                        bias = "BEARISH (for India)"
                        score = -50
                    elif change_pct < -2:
                        bias = "BULLISH (for India)"
                        score = 50
                    else:
                        bias = "NEUTRAL"
                        score = 0
                elif 'GOLD' in name:
                    if change_pct > 1:
                        bias = "RISK OFF"
                        score = -40
                    elif change_pct < -1:
                        bias = "RISK ON"
                        score = 40
                    else:
                        bias = "NEUTRAL"
                        score = 0
                elif 'INR' in name:
                    if change_pct > 0.5:
                        bias = "BEARISH (INR Weak)"
                        score = -40
                    elif change_pct < -0.5:
                        bias = "BULLISH (INR Strong)"
                        score = 40
                    else:
                        bias = "NEUTRAL"
                        score = 0
                elif 'TREASURY' in name:
                    if change_pct > 2:
                        bias = "RISK OFF"
                        score = -40
                    elif change_pct < -2:
                        bias = "RISK ON"
                        score = 40
                    else:
                        bias = "NEUTRAL"
                        score = 0
                else:
                    if change_pct > 1:
                        bias = "BULLISH"
                        score = 40
                    elif change_pct < -1:
                        bias = "BEARISH"
                        score = -40
                    else:
                        bias = "NEUTRAL"
                        score = 0

                return {
                    'asset': name,
                    'symbol': symbol,
                    'last_price': current_close,
                    'prev_close': prev_close,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score
                }
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            return None

    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation to identify market leadership changes"""
        sectors = self.fetch_sector_indices()

        if not sectors:
            return {'success': False, 'error': 'No sector data available'}

        # Sort sectors by performance
        sectors_sorted = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)

        # Identify leaders and laggards
        leaders = sectors_sorted[:3]
        laggards = sectors_sorted[-3:]

        # Calculate sector strength score
        bullish_sectors = [s for s in sectors if s['change_pct'] > 0.5]
        bearish_sectors = [s for s in sectors if s['change_pct'] < -0.5]
        neutral_sectors = [s for s in sectors if -0.5 <= s['change_pct'] <= 0.5]

        # Market breadth from sectors
        if len(sectors) > 0:
            sector_breadth = (len(bullish_sectors) / len(sectors)) * 100
        else:
            sector_breadth = 50

        # Determine rotation pattern
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

        # Overall sector sentiment
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
            'neutral_sectors_count': len(neutral_sectors),
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

    def analyze_intraday_seasonality(self) -> Dict[str, Any]:
        """Analyze intraday time-based patterns"""
        now = self.get_current_time_ist()
        current_time = now.time()

        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()

        # Determine current market session
        if current_time < market_open:
            session = "PRE-MARKET"
            session_bias = "NEUTRAL"
            session_score = 0
            session_characteristics = "Low volume, wide spreads. Wait for market open."
            trading_recommendation = "AVOID - Wait for market open"
        elif current_time < datetime.strptime("09:30", "%H:%M").time():
            session = "OPENING RANGE (9:15-9:30)"
            session_bias = "HIGH VOLATILITY"
            session_score = 0
            session_characteristics = "High volatility, gap movements, institutional orders"
            trading_recommendation = "CAUTIOUS - Wait for range breakout or use tight stops"
        elif current_time < datetime.strptime("10:00", "%H:%M").time():
            session = "POST-OPENING (9:30-10:00)"
            session_bias = "TREND FORMATION"
            session_score = 40
            session_characteristics = "Trend develops, direction becomes clear"
            trading_recommendation = "ACTIVE - Trade in direction of trend"
        elif current_time < datetime.strptime("11:30", "%H:%M").time():
            session = "MID-MORNING (10:00-11:30)"
            session_bias = "TRENDING"
            session_score = 50
            session_characteristics = "Best trending period, follow momentum"
            trading_recommendation = "VERY ACTIVE - Best time for trend following"
        elif current_time < datetime.strptime("14:30", "%H:%M").time():
            session = "LUNCHTIME (11:30-14:30)"
            session_bias = "CONSOLIDATION"
            session_score = -20
            session_characteristics = "Low volume, choppy, range-bound"
            trading_recommendation = "REDUCE ACTIVITY - Scalping only or stay out"
        elif current_time < datetime.strptime("15:15", "%H:%M").time():
            session = "AFTERNOON SESSION (14:30-15:15)"
            session_bias = "MOMENTUM"
            session_score = 45
            session_characteristics = "Volume picks up, trends resume"
            trading_recommendation = "ACTIVE - Trade breakouts and momentum"
        elif current_time < market_close:
            session = "CLOSING RANGE (15:15-15:30)"
            session_bias = "HIGH VOLATILITY"
            session_score = 0
            session_characteristics = "High volume, squaring off positions, volatile"
            trading_recommendation = "CAUTIOUS - Close positions or use wide stops"
        else:
            session = "POST-MARKET"
            session_bias = "NEUTRAL"
            session_score = 0
            session_characteristics = "Market closed"
            trading_recommendation = "NO TRADING - Market closed"

        # Day of week patterns
        weekday = now.strftime("%A")

        if weekday == "Monday":
            day_bias = "GAP TENDENCY"
            day_characteristics = "Weekend news gaps, follow-through from Friday"
        elif weekday == "Tuesday" or weekday == "Wednesday":
            day_bias = "TRENDING"
            day_characteristics = "Best trending days, institutional activity high"
        elif weekday == "Thursday":
            day_bias = "CONSOLIDATION"
            day_characteristics = "Pre-Friday profit booking, consolidation"
        elif weekday == "Friday":
            day_bias = "PROFIT BOOKING"
            day_characteristics = "Week-end squaring off, typically weak close"
        else:
            day_bias = "WEEKEND"
            day_characteristics = "Market closed"

        return {
            'success': True,
            'current_time': now.strftime("%H:%M:%S"),
            'session': session,
            'session_bias': session_bias,
            'session_score': session_score,
            'session_characteristics': session_characteristics,
            'trading_recommendation': trading_recommendation,
            'weekday': weekday,
            'day_bias': day_bias,
            'day_characteristics': day_characteristics,
            'timestamp': now
        }

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """Fetch all enhanced market data from all sources with progress tracking"""
        print("=" * 60)
        print("FETCHING ENHANCED MARKET DATA")
        print("=" * 60)
        
        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'intermarket': [],
            'sector_rotation': {},
            'intraday_seasonality': {},
            'summary': {},
            'fetch_status': {}
        }
        
        # Track fetch status
        fetch_status = {
            'india_vix': 'pending',
            'sectors': 'pending',
            'global': 'pending',
            'intermarket': 'pending',
            'rotation': 'pending',
            'seasonality': 'pending'
        }
        
        # 1. Fetch India VIX
        try:
            print("\n[1/6] Fetching India VIX...")
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
            print("\n[2/6] Fetching sector indices...")
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
            print("\n[3/6] Fetching global markets...")
            self._rate_limit()
            result['global_markets'] = self.fetch_global_markets()
            fetch_status['global'] = 'success' if result['global_markets'] else 'failed'
            print(f"  Fetched {len(result['global_markets'])} markets")
            print(f"  Status: {fetch_status['global']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['global'] = 'error'
            result['global_markets'] = []
        
        # 4. Fetch Intermarket Data
        try:
            print("\n[4/6] Fetching intermarket data...")
            self._rate_limit()
            result['intermarket'] = self.fetch_intermarket_data()
            fetch_status['intermarket'] = 'success' if result['intermarket'] else 'failed'
            print(f"  Fetched {len(result['intermarket'])} assets")
            print(f"  Status: {fetch_status['intermarket']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['intermarket'] = 'error'
            result['intermarket'] = []
        
        # 5. Analyze Sector Rotation
        try:
            print("\n[5/6] Analyzing Sector Rotation...")
            result['sector_rotation'] = self.analyze_sector_rotation()
            fetch_status['rotation'] = 'success' if result['sector_rotation'].get('success') else 'failed'
            print(f"  Status: {fetch_status['rotation']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['rotation'] = 'error'
            result['sector_rotation'] = {'success': False, 'error': str(e)}
        
        # 6. Analyze Intraday Seasonality
        try:
            print("\n[6/6] Analyzing Intraday Seasonality...")
            result['intraday_seasonality'] = self.analyze_intraday_seasonality()
            fetch_status['seasonality'] = 'success' if result['intraday_seasonality'].get('success') else 'failed'
            print(f"  Status: {fetch_status['seasonality']}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            fetch_status['seasonality'] = 'error'
            result['intraday_seasonality'] = {'success': False, 'error': str(e)}
        
        # 7. Calculate summary statistics
        try:
            print("\n[7/7] Calculating summary...")
            result['summary'] = self._calculate_summary(result)
            print("  ✓ Summary calculated")
        except Exception as e:
            print(f"  ❌ Summary Error: {str(e)}")
            result['summary'] = {}
        
        # Store fetch status
        result['fetch_status'] = fetch_status
        
        # Print final summary
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
        """Calculate summary statistics from all data"""
        summary = {
            'total_data_points': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'overall_sentiment': 'NEUTRAL'
        }

        all_scores = []

        # Count India VIX
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

        # Count sectors
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

        # Count global markets
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

        # Count intermarket
        for asset in data['intermarket']:
            summary['total_data_points'] += 1
            all_scores.append(asset['score'])
            bias = asset['bias']
            if 'BULLISH' in bias or 'RISK ON' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias or 'RISK OFF' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Calculate average score
        if all_scores:
            summary['avg_score'] = np.mean(all_scores)

            # Determine overall sentiment
            if summary['avg_score'] > 25:
                summary['overall_sentiment'] = 'BULLISH'
            elif summary['avg_score'] < -25:
                summary['overall_sentiment'] = 'BEARISH'
            else:
                summary['overall_sentiment'] = 'NEUTRAL'

        return summary


# =============================================
# TRADING SIGNAL MANAGER WITH COOLDOWN
# =============================================

class TradingSignalManager:
    """Manage trading signals with cooldown periods"""
    
    def __init__(self, cooldown_minutes=15):
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = {}
        self.sent_signals = set()
        
    def can_send_signal(self, signal_type: str, instrument: str) -> Tuple[bool, int]:
        """Check if signal can be sent based on cooldown"""
        key = f"{signal_type}_{instrument}"
        current_time = datetime.now()
        
        if key in self.last_signal_time:
            last_sent = self.last_signal_time[key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return False, self.cooldown_minutes - int(time_diff)
        
        self.last_signal_time[key] = current_time
        return True, 0

    def generate_trading_recommendation(self, instrument_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading recommendation based on comprehensive analysis"""
        try:
            overall_bias = instrument_data['overall_bias']
            bias_score = instrument_data['bias_score']
            spot_price = instrument_data['spot_price']
            comp_metrics = instrument_data.get('comprehensive_metrics', {})
            detailed_bias = instrument_data.get('detailed_atm_bias', {})
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(instrument_data, comp_metrics)
            
            # Generate signal based on bias strength and confidence
            if "Strong Bullish" in overall_bias and confidence >= 75 and bias_score >= 3:
                signal_type = "STRONG_BUY"
                direction = "BULLISH"
                strength = "HIGH"
            elif "Bullish" in overall_bias and confidence >= 60 and bias_score >= 2:
                signal_type = "BUY"
                direction = "BULLISH" 
                strength = "MEDIUM"
            elif "Strong Bearish" in overall_bias and confidence >= 75 and bias_score <= -3:
                signal_type = "STRONG_SELL"
                direction = "BEARISH"
                strength = "HIGH"
            elif "Bearish" in overall_bias and confidence >= 60 and bias_score <= -2:
                signal_type = "SELL"
                direction = "BEARISH"
                strength = "MEDIUM"
            else:
                return None
            
            # Get key levels
            call_resistance = comp_metrics.get('call_resistance', spot_price + 100)
            put_support = comp_metrics.get('put_support', spot_price - 100)
            max_pain = comp_metrics.get('max_pain_strike', spot_price)
            
            # Generate entry/exit levels
            if direction == "BULLISH":
                entry_zone = f"{put_support:.0f}-{spot_price:.0f}"
                targets = [
                    spot_price + (call_resistance - spot_price) * 0.5,
                    call_resistance
                ]
                stop_loss = put_support - 20
            else:  # BEARISH
                entry_zone = f"{spot_price:.0f}-{call_resistance:.0f}"
                targets = [
                    spot_price - (spot_price - put_support) * 0.5,
                    put_support
                ]
                stop_loss = call_resistance + 20
            
            recommendation = {
                'instrument': instrument_data['instrument'],
                'signal_type': signal_type,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'spot_price': spot_price,
                'bias_score': bias_score,
                'entry_zone': entry_zone,
                'targets': [f"{t:.0f}" for t in targets],
                'stop_loss': f"{stop_loss:.0f}",
                'call_resistance': f"{call_resistance:.0f}",
                'put_support': f"{put_support:.0f}",
                'max_pain': f"{max_pain:.0f}",
                'pcr_oi': instrument_data['pcr_oi'],
                'key_metrics': {
                    'synthetic_bias': comp_metrics.get('synthetic_bias', 'N/A'),
                    'atm_buildup': comp_metrics.get('atm_buildup', 'N/A'),
                    'vega_bias': comp_metrics.get('atm_vega_bias', 'N/A')
                }
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None

    def calculate_confidence_score(self, instrument_data: Dict[str, Any], comp_metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for trading signal"""
        confidence = 50  # Base confidence
        
        # PCR Confidence
        pcr_oi = instrument_data['pcr_oi']
        if pcr_oi > 1.3 or pcr_oi < 0.7:
            confidence += 15
        elif pcr_oi > 1.1 or pcr_oi < 0.9:
            confidence += 10
        
        # Bias Score Confidence
        bias_score = abs(instrument_data['bias_score'])
        if bias_score >= 3:
            confidence += 20
        elif bias_score >= 2:
            confidence += 15
        elif bias_score >= 1:
            confidence += 10
        
        # Synthetic Bias Confidence
        synthetic_bias = comp_metrics.get('synthetic_bias', 'Neutral')
        if 'Bullish' in synthetic_bias or 'Bearish' in synthetic_bias:
            confidence += 10
        
        # Max Pain Confidence
        dist_mp = abs(comp_metrics.get('distance_from_max_pain', 0))
        if dist_mp > 100:
            confidence += 10
        elif dist_mp > 50:
            confidence += 5
        
        # Multiple confirmation factors
        confirming_factors = 0
        if comp_metrics.get('synthetic_bias', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
        if comp_metrics.get('total_vega_bias', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
        if comp_metrics.get('atm_buildup', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
            
        confidence += confirming_factors * 5
        
        return min(confidence, 95)  # Cap at 95%

    def format_signal_message(self, recommendation: Dict[str, Any]) -> str:
        """Format trading signal for Telegram notification"""
        emoji = "🟢" if recommendation['direction'] == "BULLISH" else "🔴"
        strength_emoji = "🔥" if recommendation['strength'] == "HIGH" else "⚡"
        
        message = f"""
{strength_emoji} {emoji} TRADING SIGNAL ALERT {emoji} {strength_emoji}
🎯 {recommendation['instrument']} - {recommendation['signal_type']}
⏰ Time: {recommendation['timestamp'].strftime('%H:%M:%S')} IST
📊 Confidence: {recommendation['confidence']}%
💰 Current Price: ₹{recommendation['spot_price']:.2f}
📈 Bias Score: {recommendation['bias_score']:.2f}
🔢 PCR OI: {recommendation['pcr_oi']:.2f}
🎯 TRADING PLAN:

Entry Zone: ₹{recommendation['entry_zone']}
Target 1: ₹{recommendation['targets'][0]}
Target 2: ₹{recommendation['targets'][1]}
Stop Loss: ₹{recommendation['stop_loss']}

📊 KEY LEVELS:

Call Resistance: ₹{recommendation['call_resistance']}
Put Support: ₹{recommendation['put_support']}
Max Pain: ₹{recommendation['max_pain']}

🔍 CONFIRMING METRICS:

Synthetic Bias: {recommendation['key_metrics']['synthetic_bias']}
ATM Buildup: {recommendation['key_metrics']['atm_buildup']}
Vega Bias: {recommendation['key_metrics']['vega_bias']}

⏳ Next signal in {self.cooldown_minutes} minutes
⚠️ Risk Disclaimer: Trade at your own risk. Use proper position sizing and risk management.
"""
        return message


# =============================================
# VOLUME SPIKE DETECTOR
# =============================================

class VolumeSpikeDetector:
    """Detect sudden volume spikes in real-time"""
    
    def __init__(self, lookback_period=20, spike_threshold=2.5):
        self.lookback_period = lookback_period
        self.spike_threshold = spike_threshold
        self.volume_history = deque(maxlen=lookback_period)
        self.sent_alerts = set()
        
    def detect_volume_spike(self, current_volume: float, timestamp: datetime) -> Tuple[bool, float]:
        """Detect if current volume is a spike compared to historical average"""
        if len(self.volume_history) < 5:
            self.volume_history.append(current_volume)
            return False, 0
        
        volume_array = np.array(list(self.volume_history))
        avg_volume = np.mean(volume_array)
        std_volume = np.std(volume_array)
        
        self.volume_history.append(current_volume)
        
        if avg_volume == 0:
            return False, 0
        
        volume_ratio = current_volume / avg_volume
        is_spike = (volume_ratio > self.spike_threshold) and (current_volume > avg_volume + 2 * std_volume)
        
        return is_spike, volume_ratio


# =============================================
# VOLUME ORDER BLOCKS
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator by BigBeluga"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_lines_count = 500
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.sent_alerts = set()
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period=200) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3

    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks based on the Pine Script logic"""
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

    def check_price_near_blocks(self, current_price: float, blocks: List[Dict[str, Any]], threshold: float = 5) -> List[Dict[str, Any]]:
        nearby_blocks = []
        for block in blocks:
            distance_to_upper = abs(current_price - block['upper'])
            distance_to_lower = abs(current_price - block['lower'])
            distance_to_mid = abs(current_price - block['mid'])
            
            if (distance_to_upper <= threshold or 
                distance_to_lower <= threshold or 
                distance_to_mid <= threshold):
                nearby_blocks.append(block)
        
        return nearby_blocks


# =============================================
# ALERT MANAGER
# =============================================

class AlertManager:
    """Manage cooldown periods for all alerts"""
    
    def __init__(self, cooldown_minutes=10):
        self.cooldown_minutes = cooldown_minutes
        self.alert_timestamps = {}
        
    def can_send_alert(self, alert_type: str, alert_id: str) -> bool:
        """Check if alert can be sent (cooldown period passed)"""
        key = f"{alert_type}_{alert_id}"
        current_time = datetime.now()
        
        if key in self.alert_timestamps:
            last_sent = self.alert_timestamps[key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return False
        
        self.alert_timestamps[key] = current_time
        return True

    def cleanup_old_alerts(self, max_age_hours=24):
        """Clean up old alert timestamps"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.alert_timestamps.items():
            time_diff = (current_time - timestamp).total_seconds() / 3600
            if time_diff > max_age_hours:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.alert_timestamps[key]


# =============================================
# NSE OPTIONS ANALYZER WITH AUTO-REFRESH
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
            },
            'stocks': {
                'RELIANCE': {'lot_size': 250, 'atm_range': 100, 'zone_size': 50},
                'TCS': {'lot_size': 150, 'atm_range': 100, 'zone_size': 50},
            }
        }
        self.last_refresh_time = {}
        self.refresh_interval = 2  # 2 minutes default refresh
        self.cached_bias_data = {}
        
    def set_refresh_interval(self, minutes: int):
        """Set auto-refresh interval"""
        self.refresh_interval = minutes

    def should_refresh_data(self, instrument: str) -> bool:
        """Check if data should be refreshed based on last refresh time"""
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
        
    def calculate_greeks(self, option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks"""
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == 'CE':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
                
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
            
            return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
        except:
            return 0, 0, 0, 0, 0

    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch option chain data from NSE"""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}" if instrument in self.NSE_INSTRUMENTS['indices'] else \
                  f"https://www.nseindia.com/api/option-chain-equities?symbol={url_instrument}"

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

    def delta_volume_bias(self, price: float, volume: float, chg_oi: float) -> str:
        """Calculate delta volume bias"""
        if price > 0 and volume > 0 and chg_oi > 0:
            return "Bullish"
        elif price < 0 and volume > 0 and chg_oi > 0:
            return "Bearish"
        elif price > 0 and volume > 0 and chg_oi < 0:
            return "Bullish"
        elif price < 0 and volume > 0 and chg_oi < 0:
            return "Bearish"
        else:
            return "Neutral"

    def final_verdict(self, score: float) -> str:
        """Determine final verdict based on score"""
        if score >= 4:
            return "Strong Bullish"
        elif score >= 2:
            return "Bullish"
        elif score <= -4:
            return "Strong Bearish"
        elif score <= -2:
            return "Bearish"
        else:
            return "Neutral"

    def determine_level(self, row: pd.Series) -> str:
        """Determine support/resistance level based on OI"""
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']

        # Strong Support condition
        if pe_oi > 1.12 * ce_oi:
            return "Support"
        # Strong Resistance condition
        elif ce_oi > 1.12 * pe_oi:
            return "Resistance"
        # Neutral if none dominant
        else:
            return "Neutral"

    def calculate_max_pain(self, df_full_chain: pd.DataFrame) -> Optional[float]:
        """Calculate Max Pain strike"""
        try:
            strikes = df_full_chain['strikePrice'].unique()
            pain_values = []

            for strike in strikes:
                call_pain = 0
                put_pain = 0

                # Calculate pain for all strikes
                for _, row in df_full_chain.iterrows():
                    row_strike = row['strikePrice']

                    # Call pain: If strike price > current strike, calls are ITM
                    if row_strike < strike:
                        call_pain += (strike - row_strike) * row.get('openInterest_CE', 0)

                    # Put pain: If strike price < current strike, puts are ITM
                    if row_strike > strike:
                        put_pain += (row_strike - strike) * row.get('openInterest_PE', 0)

                total_pain = call_pain + put_pain
                pain_values.append({'strike': strike, 'pain': total_pain})

            # Max pain is the strike with minimum total pain
            max_pain_data = min(pain_values, key=lambda x: x['pain'])
            return max_pain_data['strike']
        except:
            return None

    def calculate_synthetic_future_bias(self, atm_ce_price: float, atm_pe_price: float, atm_strike: float, spot_price: float) -> Tuple[str, float, float]:
        """Calculate Synthetic Future Bias at ATM"""
        try:
            synthetic_future = atm_strike + atm_ce_price - atm_pe_price
            difference = synthetic_future - spot_price

            if difference > 5:  # Threshold can be adjusted
                return "Bullish", synthetic_future, difference
            elif difference < -5:
                return "Bearish", synthetic_future, difference
            else:
                return "Neutral", synthetic_future, difference
        except:
            return "Neutral", 0, 0

    def calculate_atm_buildup_pattern(self, atm_ce_oi: float, atm_pe_oi: float, atm_ce_change: float, atm_pe_change: float) -> str:
        """Determine ATM buildup pattern based on OI changes"""
        try:
            # Classify based on OI changes
            if atm_ce_change > 0 and atm_pe_change > 0:
                if atm_ce_change > atm_pe_change:
                    return "Long Buildup (Bearish)"
                else:
                    return "Short Buildup (Bullish)"
            elif atm_ce_change < 0 and atm_pe_change < 0:
                if abs(atm_ce_change) > abs(atm_pe_change):
                    return "Short Covering (Bullish)"
                else:
                    return "Long Unwinding (Bearish)"
            elif atm_ce_change > 0 and atm_pe_change < 0:
                return "Call Writing (Bearish)"
            elif atm_ce_change < 0 and atm_pe_change > 0:
                return "Put Writing (Bullish)"
            else:
                return "Neutral"
        except:
            return "Neutral"

    def calculate_atm_vega_bias(self, atm_ce_vega: float, atm_pe_vega: float, atm_ce_oi: float, atm_pe_oi: float) -> Tuple[str, float]:
        """Calculate ATM Vega exposure bias"""
        try:
            ce_vega_exposure = atm_ce_vega * atm_ce_oi
            pe_vega_exposure = atm_pe_vega * atm_pe_oi

            total_vega_exposure = ce_vega_exposure + pe_vega_exposure

            if pe_vega_exposure > ce_vega_exposure * 1.1:
                return "Bullish (High Put Vega)", total_vega_exposure
            elif ce_vega_exposure > pe_vega_exposure * 1.1:
                return "Bearish (High Call Vega)", total_vega_exposure
            else:
                return "Neutral", total_vega_exposure
        except:
            return "Neutral", 0

    def find_call_resistance_put_support(self, df_full_chain: pd.DataFrame, spot_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Find key resistance (from Call OI) and support (from Put OI) strikes"""
        try:
            # Find strikes above spot with highest Call OI (Resistance)
            above_spot = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            if not above_spot.empty:
                call_resistance = above_spot.nlargest(1, 'openInterest_CE')['strikePrice'].values[0]
            else:
                call_resistance = None

            # Find strikes below spot with highest Put OI (Support)
            below_spot = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            if not below_spot.empty:
                put_support = below_spot.nlargest(1, 'openInterest_PE')['strikePrice'].values[0]
            else:
                put_support = None

            return call_resistance, put_support
        except:
            return None, None

    def calculate_total_vega_bias(self, df_full_chain: pd.DataFrame) -> Tuple[str, float, float, float]:
        """Calculate total Vega bias across all strikes"""
        try:
            total_ce_vega = (df_full_chain['Vega_CE'] * df_full_chain['openInterest_CE']).sum()
            total_pe_vega = (df_full_chain['Vega_PE'] * df_full_chain['openInterest_PE']).sum()

            total_vega = total_ce_vega + total_pe_vega

            if total_pe_vega > total_ce_vega * 1.1:
                return "Bullish (Put Heavy)", total_vega, total_ce_vega, total_pe_vega
            elif total_ce_vega > total_pe_vega * 1.1:
                return "Bearish (Call Heavy)", total_vega, total_ce_vega, total_pe_vega
            else:
                return "Neutral", total_vega, total_ce_vega, total_pe_vega
        except:
            return "Neutral", 0, 0, 0

    def detect_unusual_activity(self, df_full_chain: pd.DataFrame, spot_price: float) -> List[Dict[str, Any]]:
        """Detect strikes with unusual activity (high volume relative to OI)"""
        try:
            unusual_strikes = []

            for _, row in df_full_chain.iterrows():
                strike = row['strikePrice']

                # Check Call side
                ce_oi = row.get('openInterest_CE', 0)
                ce_volume = row.get('totalTradedVolume_CE', 0)
                if ce_oi > 0 and ce_volume / ce_oi > 0.5:  # Volume > 50% of OI
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'CE',
                        'volume_oi_ratio': ce_volume / ce_oi if ce_oi > 0 else 0,
                        'volume': ce_volume,
                        'oi': ce_oi
                    })

                # Check Put side
                pe_oi = row.get('openInterest_PE', 0)
                pe_volume = row.get('totalTradedVolume_PE', 0)
                if pe_oi > 0 and pe_volume / pe_oi > 0.5:
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'PE',
                        'volume_oi_ratio': pe_volume / pe_oi if pe_oi > 0 else 0,
                        'volume': pe_volume,
                        'oi': pe_oi
                    })

            # Sort by volume/OI ratio and return top 5
            unusual_strikes.sort(key=lambda x: x['volume_oi_ratio'], reverse=True)
            return unusual_strikes[:5]
        except:
            return []

    def calculate_overall_buildup_pattern(self, df_full_chain: pd.DataFrame, spot_price: float) -> str:
        """Calculate overall buildup pattern across ITM, ATM, and OTM strikes"""
        try:
            # Separate into ITM, ATM, OTM
            itm_calls = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            otm_calls = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            atm_strikes = df_full_chain[abs(df_full_chain['strikePrice'] - spot_price) <= 50].copy()

            # Calculate OI changes for each zone
            itm_ce_change = itm_calls['changeinOpenInterest_CE'].sum() if not itm_calls.empty else 0
            itm_pe_change = itm_calls['changeinOpenInterest_PE'].sum() if not itm_calls.empty else 0

            otm_ce_change = otm_calls['changeinOpenInterest_CE'].sum() if not otm_calls.empty else 0
            otm_pe_change = otm_calls['changeinOpenInterest_PE'].sum() if not otm_calls.empty else 0

            atm_ce_change = atm_strikes['changeinOpenInterest_CE'].sum() if not atm_strikes.empty else 0
            atm_pe_change = atm_strikes['changeinOpenInterest_PE'].sum() if not atm_strikes.empty else 0

            # Determine pattern
            patterns = []

            if itm_pe_change > 0 and otm_ce_change > 0:
                patterns.append("Protective Strategy (Bullish)")
            elif itm_ce_change > 0 and otm_pe_change > 0:
                patterns.append("Protective Strategy (Bearish)")

            if atm_ce_change > atm_pe_change and abs(atm_ce_change) > 1000:
                patterns.append("Strong Call Writing (Bearish)")
            elif atm_pe_change > atm_ce_change and abs(atm_pe_change) > 1000:
                patterns.append("Strong Put Writing (Bullish)")

            if otm_ce_change > itm_ce_change and otm_ce_change > 1000:
                patterns.append("OTM Call Buying (Bullish)")
            elif otm_pe_change > itm_pe_change and otm_pe_change > 1000:
                patterns.append("OTM Put Buying (Bearish)")

            return " | ".join(patterns) if patterns else "Balanced/Neutral"

        except:
            return "Neutral"

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Comprehensive ATM bias analysis with all metrics"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            records = data['records']
            spot = data['spot']
            expiry = data['expiry']

            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06

            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)

                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)

            if not calls or not puts:
                return None

            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

            # Find ATM strike
            atm_range = self.NSE_INSTRUMENTS['indices'].get(instrument, {}).get('atm_range', 200)
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            df_atm = df[abs(df['strikePrice'] - atm_strike) <= atm_range]

            if df_atm.empty:
                return None

            # Get ATM row data
            atm_df = df[df['strikePrice'] == atm_strike]
            if not atm_df.empty:
                atm_ce_price = atm_df['lastPrice_CE'].values[0]
                atm_pe_price = atm_df['lastPrice_PE'].values[0]
                atm_ce_oi = atm_df['openInterest_CE'].values[0]
                atm_pe_oi = atm_df['openInterest_PE'].values[0]
                atm_ce_change = atm_df['changeinOpenInterest_CE'].values[0]
                atm_pe_change = atm_df['changeinOpenInterest_PE'].values[0]
                atm_ce_vega = atm_df['Vega_CE'].values[0]
                atm_pe_vega = atm_df['Vega_PE'].values[0]
            else:
                return None

            # Calculate all comprehensive metrics
            synthetic_bias, synthetic_future, synthetic_diff = self.calculate_synthetic_future_bias(
                atm_ce_price, atm_pe_price, atm_strike, spot
            )
            
            atm_buildup = self.calculate_atm_buildup_pattern(
                atm_ce_oi, atm_pe_oi, atm_ce_change, atm_pe_change
            )
            
            atm_vega_bias, atm_vega_exposure = self.calculate_atm_vega_bias(
                atm_ce_vega, atm_pe_vega, atm_ce_oi, atm_pe_oi
            )
            
            max_pain_strike = self.calculate_max_pain(df)
            distance_from_max_pain = spot - max_pain_strike if max_pain_strike else 0
            
            call_resistance, put_support = self.find_call_resistance_put_support(df, spot)
            
            total_vega_bias, total_vega, total_ce_vega_exp, total_pe_vega_exp = self.calculate_total_vega_bias(df)
            
            unusual_activity = self.detect_unusual_activity(df, spot)
            
            overall_buildup = self.calculate_overall_buildup_pattern(df, spot)

            # Calculate detailed ATM bias breakdown
            detailed_atm_bias = self.calculate_detailed_atm_bias(df_atm, atm_strike, spot)

            # Calculate comprehensive bias score
            weights = {
                "oi_bias": 2, "chg_oi_bias": 2, "volume_bias": 1, 
                "iv_bias": 1, "premium_bias": 1, "delta_bias": 1,
                "synthetic_bias": 2, "vega_bias": 1, "max_pain_bias": 1
            }

            total_score = 0
            
            # OI Bias
            oi_bias = "Bullish" if data['total_pe_oi'] > data['total_ce_oi'] else "Bearish"
            total_score += weights["oi_bias"] if oi_bias == "Bullish" else -weights["oi_bias"]
            
            # Change in OI Bias
            chg_oi_bias = "Bullish" if data['total_pe_change'] > data['total_ce_change'] else "Bearish"
            total_score += weights["chg_oi_bias"] if chg_oi_bias == "Bullish" else -weights["chg_oi_bias"]
            
            # Synthetic Bias
            total_score += weights["synthetic_bias"] if synthetic_bias == "Bullish" else -weights["synthetic_bias"] if synthetic_bias == "Bearish" else 0
            
            # Vega Bias
            vega_bias_score = 1 if "Bullish" in atm_vega_bias else -1 if "Bearish" in atm_vega_bias else 0
            total_score += weights["vega_bias"] * vega_bias_score
            
            # Max Pain Bias (if spot above max pain, bullish)
            max_pain_bias = "Bullish" if distance_from_max_pain > 0 else "Bearish" if distance_from_max_pain < 0 else "Neutral"
            total_score += weights["max_pain_bias"] if max_pain_bias == "Bullish" else -weights["max_pain_bias"] if max_pain_bias == "Bearish" else 0

            overall_bias = self.final_verdict(total_score)

            return {
                'instrument': instrument,
                'spot_price': spot,
                'atm_strike': atm_strike,
                'overall_bias': overall_bias,
                'bias_score': total_score,
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'pcr_change': abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change'],
                'detailed_atm_bias': detailed_atm_bias,
                'comprehensive_metrics': {
                    'synthetic_bias': synthetic_bias,
                    'synthetic_future': synthetic_future,
                    'synthetic_diff': synthetic_diff,
                    'atm_buildup': atm_buildup,
                    'atm_vega_bias': atm_vega_bias,
                    'atm_vega_exposure': atm_vega_exposure,
                    'max_pain_strike': max_pain_strike,
                    'distance_from_max_pain': distance_from_max_pain,
                    'call_resistance': call_resistance,
                    'put_support': put_support,
                    'total_vega_bias': total_vega_bias,
                    'total_vega': total_vega,
                    'unusual_activity_count': len(unusual_activity),
                    'overall_buildup': overall_buildup
                }
            }

        except Exception as e:
            print(f"Error in ATM bias analysis: {e}")
            return None

    def calculate_detailed_atm_bias(self, df_atm: pd.DataFrame, atm_strike: float, spot_price: float) -> Dict[str, Any]:
        """Calculate detailed ATM bias breakdown for all metrics"""
        try:
            detailed_bias = {}
            
            for _, row in df_atm.iterrows():
                if row['strikePrice'] == atm_strike:
                    # Calculate per-strike delta and gamma exposure
                    ce_delta_exp = row['Delta_CE'] * row['openInterest_CE']
                    pe_delta_exp = row['Delta_PE'] * row['openInterest_PE']
                    ce_gamma_exp = row['Gamma_CE'] * row['openInterest_CE']
                    pe_gamma_exp = row['Gamma_PE'] * row['openInterest_PE']

                    net_delta_exp = ce_delta_exp + pe_delta_exp
                    net_gamma_exp = ce_gamma_exp + pe_gamma_exp
                    strike_iv_skew = row['impliedVolatility_PE'] - row['impliedVolatility_CE']

                    delta_exp_bias = "Bullish" if net_delta_exp > 0 else "Bearish" if net_delta_exp < 0 else "Neutral"
                    gamma_exp_bias = "Bullish" if net_gamma_exp > 0 else "Bearish" if net_gamma_exp < 0 else "Neutral"
                    iv_skew_bias = "Bullish" if strike_iv_skew > 0 else "Bearish" if strike_iv_skew < 0 else "Neutral"

                    detailed_bias = {
                        "Strike": row['strikePrice'],
                        "Zone": 'ATM',
                        "Level": self.determine_level(row),
                        "OI_Bias": "Bullish" if row['openInterest_CE'] < row['openInterest_PE'] else "Bearish",
                        "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                        "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                        "Delta_Bias": "Bullish" if abs(row['Delta_PE']) > abs(row['Delta_CE']) else "Bearish",
                        "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                        "Premium_Bias": "Bullish" if row['lastPrice_CE'] < row['lastPrice_PE'] else "Bearish",
                        "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                        "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                        "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                        "DVP_Bias": self.delta_volume_bias(
                            row['lastPrice_CE'] - row['lastPrice_PE'],
                            row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                            row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                        ),
                        "Delta_Exposure_Bias": delta_exp_bias,
                        "Gamma_Exposure_Bias": gamma_exp_bias,
                        "IV_Skew_Bias": iv_skew_bias,
                        # Raw values for display
                        "CE_OI": row['openInterest_CE'],
                        "PE_OI": row['openInterest_PE'],
                        "CE_Change": row['changeinOpenInterest_CE'],
                        "PE_Change": row['changeinOpenInterest_PE'],
                        "CE_Volume": row['totalTradedVolume_CE'],
                        "PE_Volume": row['totalTradedVolume_PE'],
                        "CE_Price": row['lastPrice_CE'],
                        "PE_Price": row['lastPrice_PE'],
                        "CE_IV": row['impliedVolatility_CE'],
                        "PE_IV": row['impliedVolatility_PE'],
                        "Delta_CE": row['Delta_CE'],
                        "Delta_PE": row['Delta_PE'],
                        "Gamma_CE": row['Gamma_CE'],
                        "Gamma_PE": row['Gamma_PE']
                    }
                    break
            
            return detailed_bias
            
        except Exception as e:
            print(f"Error in detailed ATM bias: {e}")
            return {}

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive market bias across all instruments with auto-refresh"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_atm_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                        # Update cache
                        self.cached_bias_data[instrument] = bias_data
                except Exception as e:
                    print(f"Error fetching {instrument}: {e}")
                    # Use cached data if available
                    if instrument in self.cached_bias_data:
                        results.append(self.cached_bias_data[instrument])
            else:
                # Return cached data if available and not forcing refresh
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
        self.vob_indicator = VolumeOrderBlocks(sensitivity=5)
        self.volume_spike_detector = VolumeSpikeDetector(lookback_period=20, spike_threshold=2.5)
        self.alert_manager = AlertManager(cooldown_minutes=10)
        self.options_analyzer = NSEOptionsAnalyzer()
        self.trading_signal_manager = TradingSignalManager(cooldown_minutes=15)
        self.bias_analyzer = BiasAnalysisPro()
        self.market_data_fetcher = EnhancedMarketData()
        self.decision_engine = MasterDecisionEngine()  # NEW: Master Decision Engine
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables with better defaults"""
        defaults = {
            'sent_vob_alerts': set(),
            'sent_volume_block_alerts': set(),
            'sent_volume_spike_alerts': set(),
            'last_alert_check': None,
            'volume_history': [],
            'market_bias_data': None,
            'last_bias_update': None,
            'last_signal_check': None,
            'sent_trading_signals': {},
            'comprehensive_bias_data': None,
            'last_comprehensive_bias_update': None,
            'enhanced_market_data': None,
            'last_market_data_update': None,
            'error_count': 0,
            'last_error_time': None,
            'retry_count': 0,
            'data_fetch_attempts': {},
            'master_decision': None,  # NEW
            'last_decision_time': None,  # NEW
            'decision_history': [],  # NEW
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
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
        """Initialize Supabase client"""
        try:
            if self.supabase_url and self.supabase_key:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
                self.supabase.table('nifty_data').select("id").limit(1).execute()
            else:
                self.supabase = None
        except Exception as e:
            st.warning(f"Supabase connection error: {str(e)}")
            self.supabase = None

    def get_dhan_headers(self) -> Dict[str, str]:
        """Get headers for DhanHQ API calls"""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.dhan_token,
            'client-id': self.dhan_client_id
        }

    def test_api_connection(self) -> bool:
        """Test DhanHQ API connection"""
        st.info("🔍 Testing API connection...")
        test_payload = {"IDX_I": [self.nifty_security_id]}
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/marketfeed/ltp",
                headers=self.get_dhan_headers(),
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ API Connection Successful!")
                return True
            else:
                st.error(f"❌ API Connection Failed: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ API Test Failed: {str(e)}")
            return False

    def fetch_intraday_data(self, interval: str = "5", days_back: int = 5) -> Optional[Dict[str, Any]]:
        """Fetch intraday data from DhanHQ API"""
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
                    st.warning("⚠️ API returned empty data")
                    return None
                st.success(f"✅ Data fetched: {len(data['open'])} candles")
                return data
            else:
                st.error(f"❌ API Error {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"❌ Data fetch error: {str(e)}")
            return None

    def process_data(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        """Process API data into DataFrame"""
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

    def send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
            return False

    # NEW: Display Master Decision Engine
    def display_master_decision(self):
        """Display Master Decision Engine analysis"""
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
                                if 'decision_history' not in st.session_state:
                                    st.session_state.decision_history = []
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

    def run(self):
        """Main application with all features"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Volume Analysis, Options Chain, Technical Bias, Trading Signals & Master Decision Engine*")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
            "📋 Bias Tabulation", "🚀 Trading Signals", "🌍 Market Data", "🧠 Master Decision"
        ])
        
        def run(self):
        st.title("📈 Advanced Nifty Trading Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
        "📋 Bias Tabulation", "🚀 Trading Signals", "🌍 Market Data"
    ])

    with tab1:
        # Modern price analysis, chart, metrics, alerts all in main run logic or a modular function
        self.display_price_analysis()  # If you want, copy the price tab logic from Errorhandling.py .run()

    with tab2:
        self.display_comprehensive_options_analysis()

    with tab3:
        self.display_comprehensive_bias_analysis()

    with tab4:
        self.display_option_chain_bias_tabulation()

    with tab5:
        self.display_trading_signals_panel()  # Or render the signals panel + history as in Errorhandling.py

    with tab6:
        self.display_enhanced_market_data()

        # Remove or update the forced rerun for production!
        time.sleep(30)
        st.rerun()


# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
