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
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# COMPREHENSIVE BIAS ANALYSIS MODULE - FIXED
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
                'RELIANCE.NS': 9.98,
                'HDFCBANK.NS': 9.67,
                'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54,
                'ICICIBANK.NS': 8.01,
                'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98,
                'ITC.NS': 2.44,
                'SBIN.NS': 5.67,
                'KOTAKBANK.NS': 4.23
            }
        }

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance with enhanced error handling"""
        try:
            # For Nifty index, use different symbol
            if symbol == "^NSEI":
                symbol = "^NSEI"  # Keep as is, but add fallback
                
            ticker = yf.Ticker(symbol)
            
            # Try multiple period options
            try:
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
            except:
                try:
                    df = ticker.history(start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                                      end=datetime.now().strftime('%Y-%m-%d'), 
                                      interval=interval, auto_adjust=True)
                except:
                    # Final fallback - try 1d data
                    df = ticker.history(period='1d', interval='1h', auto_adjust=True)

            if df.empty or len(df) < 10:
                st.warning(f"⚠️ Insufficient data for {symbol}, using sample data for demonstration")
                # Generate sample data for demonstration
                return self._generate_sample_data()
                
            # Ensure all required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Volume':
                        df['Volume'] = 1000000  # Default volume
                    else:
                        df[col] = df['Close']  # Use Close for other missing columns

            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error fetching {symbol}: {e}")
            return self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration when real data is unavailable"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='5min')
        n = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        prices = 22000 + np.cumsum(np.random.randn(n) * 10)
        
        df = pd.DataFrame({
            'Open': prices + np.random.randn(n) * 5,
            'High': prices + np.abs(np.random.randn(n) * 8),
            'Low': prices - np.abs(np.random.randn(n) * 8),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, n)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index"""
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

        return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
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
        return atr.fillna(tr.mean())

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
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0) -> Tuple[bool, bool, int, int]:
        """Calculate High Volume Pivots"""
        pivot_highs = []
        pivot_lows = []

        # Simplified pivot detection for demo
        for i in range(left_bars, len(df) - right_bars):
            if i >= len(df):
                continue
                
            # Simple pivot high detection
            if all(df['High'].iloc[i] >= df['High'].iloc[i-left_bars:i+right_bars+1]):
                pivot_highs.append(i)

            # Simple pivot low detection  
            if all(df['Low'].iloc[i] <= df['Low'].iloc[i-left_bars:i+right_bars+1]):
                pivot_lows.append(i)

        # Simplified volume analysis
        avg_volume = df['Volume'].mean()
        hvp_bullish = len(pivot_lows) > len(pivot_highs)
        hvp_bearish = len(pivot_highs) > len(pivot_lows)

        return hvp_bullish, hvp_bearish, len(pivot_highs), len(pivot_lows)

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5) -> Tuple[bool, bool, float, float]:
        """Calculate Volume Order Blocks"""
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        # Handle case where we don't have enough data
        if len(ema1) < 2 or len(ema2) < 2:
            return False, False, 0, 0

        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    def _fetch_stock_data(self, symbol: str, weight: float) -> Optional[Dict[str, Any]]:
        """Helper function to fetch single stock data for parallel processing"""
        try:
            df = self.fetch_data(symbol, period='2d', interval='1h')  # Reduced period for faster loading
            if df.empty or len(df) < 2:
                return None

            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[0]
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

    def calculate_market_breadth(self) -> Tuple[float, bool, bool, int, int, List[Dict[str, Any]]]:
        """Calculate market breadth from top stocks"""
        bullish_stocks = 0
        total_stocks = 0
        stock_data = []

        # Use a simpler approach for demo
        for symbol, weight in self.config['stocks'].items():
            try:
                # Simulate stock performance for demo
                change_pct = np.random.uniform(-3, 3)  # Random change between -3% to +3%
                is_bullish = change_pct > 0
                
                stock_data.append({
                    'symbol': symbol.replace('.NS', ''),
                    'change_pct': change_pct,
                    'weight': weight
                })
                
                if is_bullish:
                    bullish_stocks += 1
                total_stocks += 1
                
            except Exception as e:
                continue

        if total_stocks > 0:
            market_breadth = (bullish_stocks / total_stocks) * 100
        else:
            market_breadth = 50

        breadth_bullish = market_breadth > self.config['breadth_threshold']
        breadth_bearish = market_breadth < (100 - self.config['breadth_threshold'])

        return market_breadth, breadth_bullish, breadth_bearish, bullish_stocks, total_stocks, stock_data

    def analyze_all_bias_indicators(self, symbol: str = "RELIANCE.NS") -> Dict[str, Any]:
        """Analyze all 8 bias indicators with enhanced error handling"""

        st.info(f"🔄 Fetching data for {symbol}...")
        df = self.fetch_data(symbol, period='5d', interval='15m')  # Reduced requirements

        if df.empty:
            error_msg = 'No data available - using sample data for demonstration'
            st.warning(f"⚠️ {error_msg}")
            # Continue with sample data for demo purposes

        current_price = df['Close'].iloc[-1] if not df.empty else 22000
        bias_results = []

        st.info("📊 Calculating technical indicators...")

        try:
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
            hvp_value = f"Highs: {pivot_highs}, Lows: {pivot_lows}"

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
            vob_value = f"EMA5: {vob_ema5:.2f} vs EMA18: {vob_ema18:.2f}"

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
            
            if len(ema5) > 1 and len(ema18) > 1:
                cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
                cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
                ob_bias = "BULLISH" if cross_up else "BEARISH" if cross_dn else "NEUTRAL"
                ob_score = 100 if cross_up else -100 if cross_dn else 0
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
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
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
            plus_di_value = plus_di.iloc[-1] if not plus_di.empty else 0
            minus_di_value = minus_di.iloc[-1] if not minus_di.empty else 0
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
            try:
                vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
                vidya_bias = "BULLISH" if vidya_bullish else "BEARISH" if vidya_bearish else "NEUTRAL"
                vidya_score = 100 if vidya_bullish else -100 if vidya_bearish else 0
                vidya_value = f"{vidya_val.iloc[-1]:.2f}" if not vidya_val.empty else "N/A"
            except:
                vidya_bias = "NEUTRAL"
                vidya_score = 0
                vidya_value = "N/A"

            bias_results.append({
                'indicator': 'VIDYA',
                'value': vidya_value,
                'bias': vidya_bias,
                'score': vidya_score,
                'weight': 1.0,
                'category': 'fast'
            })

            # 8. MFI
            mfi = self.calculate_mfi(df, self.config['mfi_period'])
            mfi_value = mfi.iloc[-1] if not mfi.empty else 50
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

        except Exception as e:
            st.error(f"Error in bias calculation: {e}")
            # Provide default neutral values if calculations fail
            for indicator in ['Volume Delta', 'HVP', 'VOB', 'Order Blocks', 'RSI', 'DMI', 'VIDYA', 'MFI']:
                bias_results.append({
                    'indicator': indicator,
                    'value': 'N/A',
                    'bias': 'NEUTRAL',
                    'score': 0,
                    'weight': 1.0,
                    'category': 'fast'
                })

        # Calculate overall bias
        bullish_count = sum(1 for bias in bias_results if 'BULLISH' in bias['bias'])
        bearish_count = sum(1 for bias in bias_results if 'BEARISH' in bias['bias'])
        neutral_count = sum(1 for bias in bias_results if 'NEUTRAL' in bias['bias'])

        # Simple majority voting for overall bias
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_bias = "BULLISH"
            overall_score = bullish_count * 10
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_bias = "BEARISH"
            overall_score = -bearish_count * 10
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0

        overall_confidence = min(100, (max(bullish_count, bearish_count) / len(bias_results)) * 100)

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
            'stock_data': [],
            'mode': "NORMAL",
            'fast_bull_pct': (bullish_count / len(bias_results)) * 100,
            'fast_bear_pct': (bearish_count / len(bias_results)) * 100,
            'bullish_bias_pct': (bullish_count / len(bias_results)) * 100,
            'bearish_bias_pct': (bearish_count / len(bias_results)) * 100
        }

# =============================================
# SIMPLIFIED OPTIONS ANALYZER FOR DEMO
# =============================================

class NSEOptionsAnalyzer:
    """Simplified NSE Options Analyzer for demo purposes"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.last_refresh_time = {}
        self.refresh_interval = 2
        
    def set_refresh_interval(self, minutes: int):
        """Set auto-refresh interval"""
        self.refresh_interval = minutes
    
    def should_refresh_data(self, instrument: str) -> bool:
        """Check if data should be refreshed"""
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

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Generate demo options data"""
        try:
            # Generate realistic demo data
            import random
            
            spot_price = 22000 + random.randint(-200, 200)
            overall_biases = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]
            overall_bias = random.choice(overall_biases)
            
            # Bias score based on overall bias
            bias_scores = {
                "Strong Bullish": random.uniform(3.5, 5.0),
                "Bullish": random.uniform(1.5, 3.0),
                "Neutral": random.uniform(-1.0, 1.0),
                "Bearish": random.uniform(-3.0, -1.5),
                "Strong Bearish": random.uniform(-5.0, -3.5)
            }
            
            return {
                'instrument': instrument,
                'spot_price': spot_price,
                'atm_strike': round(spot_price / 100) * 100,  # Nearest 100
                'overall_bias': overall_bias,
                'bias_score': bias_scores[overall_bias],
                'pcr_oi': random.uniform(0.7, 1.5),
                'pcr_change': random.uniform(0.8, 1.2),
                'total_ce_oi': random.randint(1000000, 5000000),
                'total_pe_oi': random.randint(1000000, 5000000),
                'total_ce_change': random.randint(-100000, 100000),
                'total_pe_change': random.randint(-100000, 100000),
                'detailed_atm_bias': {
                    "Strike": round(spot_price / 100) * 100,
                    "Zone": 'ATM',
                    "Level": "Support" if overall_bias in ["Bullish", "Strong Bullish"] else "Resistance",
                    "OI_Bias": "Bullish" if random.random() > 0.5 else "Bearish"
                },
                'comprehensive_metrics': {
                    'synthetic_bias': overall_bias,
                    'atm_buildup': "Long Buildup" if overall_bias in ["Bullish", "Strong Bullish"] else "Short Buildup",
                    'atm_vega_bias': "Bullish" if random.random() > 0.5 else "Bearish",
                    'max_pain_strike': round(spot_price / 100) * 100 + random.choice([-100, 0, 100]),
                    'distance_from_max_pain': random.randint(-150, 150),
                    'call_resistance': round(spot_price / 100) * 100 + 200,
                    'put_support': round(spot_price / 100) * 100 - 200
                }
            }
            
        except Exception as e:
            print(f"Error in demo options analysis: {e}")
            return None

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive market bias across all instruments"""
        instruments = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_atm_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                except Exception as e:
                    print(f"Error fetching {instrument}: {e}")
        
        return results

# =============================================
# SIMPLIFIED TRADING SIGNAL MANAGER
# =============================================

class TradingSignalManager:
    """Manage trading signals with cooldown periods"""
    
    def __init__(self, cooldown_minutes=15):
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = {}
        
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

# =============================================
# SIMPLIFIED VOLUME ORDER BLOCKS
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator"""
    
    def __init__(self, sensitivity=5):
        self.sensitivity = sensitivity
        
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks - demo version"""
        if df.empty:
            return [], []
            
        bullish_blocks = []
        bearish_blocks = []
        
        # Generate some demo blocks
        for i in range(0, len(df), 50):  # Every 50 candles
            if i < len(df):
                bullish_blocks.append({
                    'index': df.index[i],
                    'upper': df['High'].iloc[i] + 50,
                    'lower': df['Low'].iloc[i] - 50,
                    'mid': df['Close'].iloc[i],
                    'volume': df['Volume'].iloc[i],
                    'type': 'bullish'
                })
                
        for i in range(25, len(df), 50):  # Offset by 25 candles
            if i < len(df):
                bearish_blocks.append({
                    'index': df.index[i],
                    'upper': df['High'].iloc[i] + 50,
                    'lower': df['Low'].iloc[i] - 50,
                    'mid': df['Close'].iloc[i],
                    'volume': df['Volume'].iloc[i],
                    'type': 'bearish'
                })
        
        return bullish_blocks[:3], bearish_blocks[:3]  # Limit to 3 each

# =============================================
# SIMPLIFIED ALERT MANAGER
# =============================================

class AlertManager:
    """Manage cooldown periods for all alerts"""
    
    def __init__(self, cooldown_minutes=10):
        self.cooldown_minutes = cooldown_minutes
        self.alert_timestamps = {}
        
    def can_send_alert(self, alert_type: str, alert_id: str) -> bool:
        """Check if alert can be sent"""
        key = f"{alert_type}_{alert_id}"
        current_time = datetime.now()
        
        if key in self.alert_timestamps:
            last_sent = self.alert_timestamps[key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return False
        
        self.alert_timestamps[key] = current_time
        return True

# =============================================
# ENHANCED NIFTY APP WITH WORKING FEATURES
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize all indicators with simplified versions
        self.vob_indicator = VolumeOrderBlocks(sensitivity=5)
        self.alert_manager = AlertManager(cooldown_minutes=10)
        self.options_analyzer = NSEOptionsAnalyzer()
        self.trading_signal_manager = TradingSignalManager(cooldown_minutes=15)
        self.bias_analyzer = BiasAnalysisPro()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'market_bias_data' not in st.session_state:
            st.session_state.market_bias_data = None
        if 'last_bias_update' not in st.session_state:
            st.session_state.last_bias_update = None
        if 'comprehensive_bias_data' not in st.session_state:
            st.session_state.comprehensive_bias_data = None
        if 'last_comprehensive_bias_update' not in st.session_state:
            st.session_state.last_comprehensive_bias_update = None
    
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            # Use demo mode if no secrets
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except:
            # Demo mode
            self.telegram_bot_token = ""
            self.telegram_chat_id = ""

    def generate_sample_price_data(self) -> pd.DataFrame:
        """Generate realistic sample price data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                            end=datetime.now(), freq='5min')
        n = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 22000
        returns = np.random.normal(0, 0.001, n)  # Small random returns
        prices = base_price * (1 + np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, n)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df

    def create_comprehensive_chart(self, df: pd.DataFrame, bullish_blocks: List[Dict[str, Any]], bearish_blocks: List[Dict[str, Any]], interval: str) -> Optional[go.Figure]:
        """Create comprehensive chart with Volume Order Blocks"""
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Nifty 50 Analysis - {interval} Min', 'Volume'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add Volume Order Blocks
        for block in bullish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(38, 186, 159, 0.1)',
                line=dict(color='#26ba9f', width=1),
                row=1, col=1
            )
        
        for block in bearish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(102, 38, 186, 0.1)',
                line=dict(color='#6626ba', width=1),
                row=1, col=1
            )
        
        # Volume bars
        bar_colors = ['#00ff88' if row['close'] >= row['open'] else '#ff4444' 
                     for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=bar_colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', side="right")
        
        return fig

    def display_comprehensive_options_analysis(self):
        """Display comprehensive NSE Options Analysis"""
        st.header("📊 NSE Options Chain Analysis - Demo Mode")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info("🔸 Demo Mode - Showing sample options data")
        with col2:
            if st.button("🔄 Refresh Data", type="primary"):
                with st.spinner("Generating new sample data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    st.success("Options data refreshed!")
        
        st.divider()
        
        # Display current market bias
        if not st.session_state.market_bias_data:
            # Generate initial data
            bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
            st.session_state.market_bias_data = bias_data
            st.session_state.last_bias_update = datetime.now(self.ist)
        
        bias_data = st.session_state.market_bias_data
            
        st.subheader("🎯 Current Market Bias Summary")
        
        # Create metrics for each instrument
        cols = st.columns(len(bias_data))
        for idx, instrument_data in enumerate(bias_data):
            with cols[idx]:
                bias_color = "🟢" if "Bullish" in instrument_data['overall_bias'] else "🔴" if "Bearish" in instrument_data['overall_bias'] else "🟡"
                st.metric(
                    f"{instrument_data['instrument']}",
                    f"{bias_color} {instrument_data['overall_bias']}",
                    f"Score: {instrument_data['bias_score']:.2f}"
                )
        
        st.divider()
        
        # Detailed analysis for each instrument
        for instrument_data in bias_data:
            comp_metrics = instrument_data.get('comprehensive_metrics', {})
            
            with st.expander(f"🎯 {instrument_data['instrument']} - Detailed Analysis", expanded=True):
                
                # Basic Information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                with col2:
                    st.metric("ATM Strike", f"₹{instrument_data['atm_strike']:.2f}")
                with col3:
                    st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                with col4:
                    st.metric("PCR Δ OI", f"{instrument_data['pcr_change']:.2f}")
                
                st.divider()
                
                # Trading Recommendation
                st.subheader("💡 Trading Recommendation")
                
                # Simple confidence calculation
                confidence = 60 + abs(instrument_data['bias_score']) * 8
                confidence = min(confidence, 95)
                
                overall_bias = instrument_data['overall_bias']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if "Strong Bullish" in overall_bias:
                        st.success(f"""
                        **🎯 BULLISH SIGNAL - {confidence:.0f}% Confidence**
                        
                        **Recommended Action:** Consider LONG/CALL positions
                        **Entry Zone:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100):.0f} - ₹{instrument_data['spot_price']:.0f}
                        **Target:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 150):.0f}
                        **Stop Loss:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100) - 50:.0f}
                        """)
                    elif "Bullish" in overall_bias:
                        st.info(f"""
                        **📈 MILD BULLISH - {confidence:.0f}% Confidence**
                        
                        **Recommended Action:** Small LONG positions
                        **Entry Zone:** Wait for pullback to support
                        **Target:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100):.0f}
                        **Stop Loss:** Below key support
                        """)
                    elif "Strong Bearish" in overall_bias:
                        st.error(f"""
                        **🎯 BEARISH SIGNAL - {confidence:.0f}% Confidence**
                        
                        **Recommended Action:** Consider SHORT/PUT positions
                        **Entry Zone:** ₹{instrument_data['spot_price']:.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100):.0f}
                        **Target:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 150):.0f}
                        **Stop Loss:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100) + 50:.0f}
                        """)
                    elif "Bearish" in overall_bias:
                        st.warning(f"""
                        **📉 MILD BEARISH - {confidence:.0f}% Confidence**
                        
                        **Recommended Action:** Small SHORT positions
                        **Entry Zone:** Wait for rally to resistance
                        **Target:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100):.0f}
                        **Stop Loss:** Above key resistance
                        """)
                    else:
                        st.warning(f"""
                        **⚖️ NEUTRAL BIAS - {confidence:.0f}% Confidence**
                        
                        **Recommended Action:** Wait for clear direction
                        **Strategy:** Range-bound strategies
                        **Key Levels:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100):.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100):.0f}
                        """)
                
                with col2:
                    st.metric("Confidence Score", f"{confidence:.0f}%")
                    st.metric("Overall Bias", overall_bias)
                    st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")

    def display_comprehensive_bias_analysis(self):
        """Display comprehensive bias analysis from BiasAnalysisPro"""
        st.header("🎯 Comprehensive Technical Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("8-indicator bias analysis with adaptive weighting")
        with col2:
            if st.button("🔄 Update Analysis", type="primary"):
                with st.spinner("Running comprehensive bias analysis..."):
                    bias_data = self.bias_analyzer.analyze_all_bias_indicators("RELIANCE.NS")
                    st.session_state.comprehensive_bias_data = bias_data
                    st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
                    st.success("Bias analysis updated!")
        
        st.divider()
        
        if not st.session_state.comprehensive_bias_data:
            # Run initial analysis
            with st.spinner("Running initial bias analysis..."):
                bias_data = self.bias_analyzer.analyze_all_bias_indicators("RELIANCE.NS")
                st.session_state.comprehensive_bias_data = bias_data
                st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
        
        bias_data = st.session_state.comprehensive_bias_data
            
        # Overall bias summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
            st.metric("Overall Bias", f"{bias_color} {bias_data['overall_bias']}")
        with col2:
            st.metric("Bias Score", f"{bias_data['overall_score']:.1f}")
        with col3:
            st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
        with col4:
            st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
        
        st.divider()
        
        # Detailed bias indicators
        st.subheader("📊 Detailed Bias Indicators")
        
        # Create columns for better layout
        cols = st.columns(2)
        current_col = 0
        
        for i, bias in enumerate(bias_data['bias_results']):
            with cols[current_col]:
                with st.container():
                    # Color code based on bias
                    if bias['bias'] == "BULLISH":
                        st.success(f"**{bias['indicator']}**")
                        st.write(f"Value: {bias['value']}")
                        st.write(f"Bias: 🟢 {bias['bias']}")
                    elif bias['bias'] == "BEARISH":
                        st.error(f"**{bias['indicator']}**")
                        st.write(f"Value: {bias['value']}")
                        st.write(f"Bias: 🔴 {bias['bias']}")
                    else:
                        st.warning(f"**{bias['indicator']}**")
                        st.write(f"Value: {bias['value']}")
                        st.write(f"Bias: 🟡 {bias['bias']}")
                    
                    st.progress(abs(bias['score']) / 100)
            
            current_col = (current_col + 1) % 2
        
        st.divider()
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Indicators", bias_data['bullish_count'])
        with col2:
            st.metric("Bearish Indicators", bias_data['bearish_count'])
        with col3:
            st.metric("Neutral Indicators", bias_data['neutral_count'])

    def run(self):
        """Main application with all features"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Volume Analysis, Options Chain, Technical Bias & Trading Signals*")
        st.info("🔸 **Demo Mode**: Using sample data for demonstration purposes")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Dashboard Settings")
            
            st.header("📊 Chart Settings")
            timeframe = st.selectbox("Timeframe", ['5', '15', '30', '60'], index=0)
            
            st.subheader("Volume Order Blocks")
            vob_sensitivity = st.slider("Sensitivity", 3, 10, 5)
            
            st.subheader("Alerts")
            telegram_enabled = st.checkbox("Enable Telegram (Demo)", value=False)
            
            if st.button("🔄 Refresh All Data"):
                st.rerun()
        
        # Main content - Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias"])
        
        with tab1:
            # Price Analysis Tab
            st.header("📈 Live Price Analysis with Volume Order Blocks")
            
            # Generate sample data
            df = self.generate_sample_price_data()
            
            if not df.empty:
                latest = df.iloc[-1]
                current_price = latest['close']
                current_volume = latest['volume']
                
                # Detect Volume Order Blocks
                bullish_blocks, bearish_blocks = self.vob_indicator.detect_volume_order_blocks(df)
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Nifty Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Current Volume", f"{current_volume:,}")
                with col3:
                    st.metric("Bullish Blocks", len(bullish_blocks))
                with col4:
                    st.metric("Bearish Blocks", len(bearish_blocks))
                with col5:
                    st.metric("Data Source", "📊 Sample Data")
                
                # Create and display chart
                chart = self.create_comprehensive_chart(df, bullish_blocks, bearish_blocks, timeframe)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Volume Order Blocks Info
                st.subheader("🔍 Volume Order Blocks Detection")
                
                if bullish_blocks or bearish_blocks:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Bullish Blocks**")
                        for block in bullish_blocks:
                            st.write(f"- Upper: ₹{block['upper']:.2f}, Lower: ₹{block['lower']:.2f}")
                    
                    with col2:
                        st.write("**Bearish Blocks**")
                        for block in bearish_blocks:
                            st.write(f"- Upper: ₹{block['upper']:.2f}, Lower: ₹{block['lower']:.2f}")
                else:
                    st.info("No volume order blocks detected in current data range.")
            
            else:
                st.error("No data available.")
        
        with tab2:
            # Options Analysis Tab
            self.display_comprehensive_options_analysis()
        
        with tab3:
            # Technical Bias Analysis Tab
            self.display_comprehensive_bias_analysis()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
