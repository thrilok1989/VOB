import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import math
from scipy.stats import norm
import plotly.express as px
from collections import deque
import hashlib

warnings.filterwarnings('ignore')

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# Telegram Configuration - REPLACE WITH YOUR ACTUAL CREDENTIALS
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Replace with your bot token
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"  # Replace with your chat ID

# Import Dhan API for Indian indices volume data
try:
    from dhan_data_fetcher import DhanDataFetcher
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    print("Warning: Dhan API not available. Volume data may be missing for Indian indices.")


class TelegramNotifier:
    """Telegram notification system for bias alerts"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.sent_alerts = set()  # Track sent alerts to avoid duplicates
        
    def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            if not self.bot_token or self.bot_token == "YOUR_TELEGRAM_BOT_TOKEN":
                print("Telegram bot token not configured")
                return False
                
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def send_bias_alert(self, analysis_type: str, bias: str, confidence: float, details: str = "") -> bool:
        """Send bias alert to Telegram"""
        timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        
        emoji = "🟢" if "BULL" in bias.upper() else "🔴" if "BEAR" in bias.upper() else "🟡"
        
        message = f"""
<b>{emoji} {analysis_type} BIAS ALERT</b>
📊 <b>Bias:</b> {bias}
🎯 <b>Confidence:</b> {confidence:.1f}%
⏰ <b>Time:</b> {timestamp}

{details}
        """.strip()
        
        # Create alert signature to avoid duplicates
        alert_signature = f"{analysis_type}_{bias}_{datetime.now().strftime('%Y%m%d%H')}"
        
        if alert_signature not in self.sent_alerts:
            success = self.send_message(message)
            if success:
                self.sent_alerts.add(alert_signature)
                # Clean old alerts (keep only last 24 hours)
                current_time = datetime.now()
                self.sent_alerts = {alert for alert in self.sent_alerts 
                                  if alert.split('_')[-1] >= (current_time - timedelta(hours=24)).strftime('%Y%m%d%H')}
            return success
        return False
    
    def send_overall_bias_alert(self, biases: Dict[str, str], overall_bias: str, score: float) -> bool:
        """Send overall bias alert when all components agree"""
        timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        
        emoji = "🟢" if "BULL" in overall_bias.upper() else "🔴" if "BEAR" in overall_bias.upper() else "🟡"
        
        bias_details = "\n".join([f"• {k}: {v}" for k, v in biases.items()])
        
        message = f"""
<b>🚨 COMPLETE BIAS CONFIRMATION ALERT</b>

{emoji} <b>OVERALL BIAS:</b> {overall_bias}
🎯 <b>Score:</b> {score:.1f}
⏰ <b>Time:</b> {timestamp}

<b>Component Biases:</b>
{bias_details}

<b>Action:</b> All analysis components confirm {overall_bias} bias. Consider taking positions accordingly.
        """.strip()
        
        alert_signature = f"OVERALL_{overall_bias}_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        if alert_signature not in self.sent_alerts:
            success = self.send_message(message)
            if success:
                self.sent_alerts.add(alert_signature)
            return success
        return False


# Initialize Telegram notifier
telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)


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
    # DATA FETCHING
    # =========================================================================

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Dhan API (for Indian indices) or Yahoo Finance (for others)
        Note: Yahoo Finance limits intraday data - use 7d max for 5m interval
        """
        # Check if this is an Indian index that needs Dhan API
        indian_indices = {'^NSEI': 'NIFTY', '^BSESN': 'SENSEX', '^NSEBANK': 'BANKNIFTY'}

        if symbol in indian_indices and DHAN_AVAILABLE:
            try:
                # Use Dhan API for Indian indices to get proper volume data
                dhan_instrument = indian_indices[symbol]
                fetcher = DhanDataFetcher()

                # Convert interval to Dhan API format (1, 5, 15, 25, 60)
                interval_map = {'1m': '1', '5m': '5', '15m': '15', '1h': '60'}
                dhan_interval = interval_map.get(interval, '5')

                # Calculate date range for historical data (7 days) - Use IST timezone
                now_ist = datetime.now(IST)
                to_date = now_ist.strftime('%Y-%m-%d %H:%M:%S')
                from_date = (now_ist - timedelta(days=7)).replace(hour=9, minute=15, second=0).strftime('%Y-%m-%d %H:%M:%S')

                # Fetch intraday data with 7 days historical range
                result = fetcher.fetch_intraday_data(dhan_instrument, interval=dhan_interval, from_date=from_date, to_date=to_date)

                if result.get('success') and result.get('data') is not None:
                    df = result['data']

                    # Ensure column names are lowercase for consistency
                    df.columns = [col.lower() for col in df.columns]

                    # Set timestamp as index
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)

                    # Ensure volume column exists and has valid data
                    if 'volume' not in df.columns:
                        df['volume'] = 0
                    else:
                        # Replace NaN volumes with 0
                        df['volume'] = df['volume'].fillna(0)

                    if not df.empty:
                        print(f"✅ Fetched {len(df)} candles for {symbol} from Dhan API with volume data (from {from_date} to {to_date})")
                        return df
                    else:
                        print(f"⚠️  Warning: Empty data from Dhan API for {symbol}, falling back to yfinance")
                else:
                    print(f"Warning: Dhan API failed for {symbol}: {result.get('error')}, falling back to yfinance")
            except Exception as e:
                print(f"Error fetching from Dhan API for {symbol}: {e}, falling back to yfinance")

        # Fallback to Yahoo Finance for non-Indian indices or if Dhan fails
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()

            # Ensure column names are lowercase for consistency
            df.columns = [col.lower() for col in df.columns]

            # Ensure volume column exists (even if it's zeros for indices)
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                # Replace NaN volumes with 0
                df['volume'] = df['volume'].fillna(0)

            # Warn if volume is all zeros (common for Yahoo Finance indices)
            if df['volume'].sum() == 0 and symbol in indian_indices:
                print(f"⚠️  Warning: Volume data is zero for {symbol} from Yahoo Finance")

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
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
    # COMPREHENSIVE BIAS ANALYSIS - WITH CORRECTED LOGIC
    # =========================================================================

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """
        Analyze all 8 bias indicators with CORRECTED overall bias calculation
        """

        print(f"Fetching data for {symbol}...")
        # Use 7d period with 5m interval (Yahoo Finance limitation for intraday data)
        df = self.fetch_data(symbol, period='7d', interval='5m')

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
        # CORRECTED OVERALL BIAS CALCULATION
        # =====================================================================
        
        # Count biases by category
        fast_bull = sum(1 for b in bias_results if b['category'] == 'fast' and 'BULLISH' in b['bias'])
        fast_bear = sum(1 for b in bias_results if b['category'] == 'fast' and 'BEARISH' in b['bias'])
        fast_total = sum(1 for b in bias_results if b['category'] == 'fast')
        
        # Calculate weighted scores with CORRECTED logic
        total_weighted_score = 0
        total_possible_score = 0
        
        for bias in bias_results:
            weight = bias['weight']
            score = bias['score']
            
            # Only count non-neutral biases significantly
            if abs(score) > 0:
                total_weighted_score += score * weight
                total_possible_score += 100 * weight  # Maximum possible score for this indicator
        
        # Calculate overall bias percentage
        if total_possible_score > 0:
            overall_score_percentage = (total_weighted_score / total_possible_score) * 100
        else:
            overall_score_percentage = 0
        
        # CORRECTED: Determine overall bias based on STRONG consensus, not just one strong signal
        bias_strength = self.config['bias_strength']
        
        # Check for strong consensus (majority of indicators agree)
        total_indicators = len(bias_results)
        bullish_indicators = sum(1 for b in bias_results if 'BULLISH' in b['bias'])
        bearish_indicators = sum(1 for b in bias_results if 'BEARISH' in b['bias'])
        
        bullish_percentage = (bullish_indicators / total_indicators) * 100
        bearish_percentage = (bearish_indicators / total_indicators) * 100
        
        # REQUIRE STRONG CONSENSUS: At least 60% of indicators must agree
        if bullish_percentage >= 60 and overall_score_percentage > 0:
            overall_bias = "BULLISH"
            overall_score = overall_score_percentage
        elif bearish_percentage >= 60 and overall_score_percentage < 0:
            overall_bias = "BEARISH" 
            overall_score = overall_score_percentage
        else:
            # If no strong consensus, use weighted score with higher threshold
            if overall_score_percentage >= bias_strength:
                overall_bias = "BULLISH"
                overall_score = overall_score_percentage
            elif overall_score_percentage <= -bias_strength:
                overall_bias = "BEARISH"
                overall_score = overall_score_percentage
            else:
                overall_bias = "NEUTRAL"
                overall_score = 0

        overall_confidence = min(100, abs(overall_score_percentage))

        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(IST),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': bullish_indicators,
            'bearish_count': bearish_indicators,
            'neutral_count': total_indicators - (bullish_indicators + bearish_indicators),
            'total_indicators': total_indicators,
            'stock_data': stock_data,
            'bullish_percentage': bullish_percentage,
            'bearish_percentage': bearish_percentage,
            'weighted_score_percentage': overall_score_percentage
        }


# =============================================
# VOLUME ORDER BLOCKS (FROM SECOND APP)
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
# STREAMLIT APP UI
# =============================================

# Initialize all analyzers
analysis = BiasAnalysisPro()
vob_indicator = VolumeOrderBlocks(sensitivity=5)

# Streamlit UI setup
st.set_page_config(page_title="Bias Analysis Pro - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Bias Analysis Pro — Complete Single-file App")
st.markdown(
    "This Streamlit app wraps the **BiasAnalysisPro** engine (Pine → Python) and shows bias summary, "
    "price action, and volume order blocks analysis."
)

# Sidebar inputs
st.sidebar.header("Data & Symbol")
symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

# Auto-refresh configuration
st.sidebar.header("Auto-Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=2)

# Telegram configuration
st.sidebar.header("Telegram Alerts")
telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=False)
if telegram_enabled:
    st.sidebar.info("Alerts will be sent when all analysis components agree on bias direction")

# Shared state storage
if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'last_symbol' not in st.session_state:
    st.session_state['last_symbol'] = None
if 'fetch_time' not in st.session_state:
    st.session_state['fetch_time'] = None
if 'overall_nifty_bias' not in st.session_state:
    st.session_state.overall_nifty_bias = "NEUTRAL"
if 'overall_nifty_score' not in st.session_state:
    st.session_state.overall_nifty_score = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# Function to generate unique element IDs
def generate_element_id(prefix: str, content: str) -> str:
    """Generate unique element ID using hash of content"""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{prefix}_{content_hash}"

def run_complete_analysis():
    """Run complete analysis for all tabs with Telegram notifications"""
    st.session_state['last_symbol'] = symbol_input
    
    # Technical Analysis
    with st.spinner("Fetching data and running technical analysis..."):
        df_fetched = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
        st.session_state['last_df'] = df_fetched
        st.session_state['fetch_time'] = datetime.now(IST)

    if df_fetched is None or df_fetched.empty:
        st.error("No data fetched. Check symbol or network.")
        return False

    # Run bias analysis
    with st.spinner("Running full bias analysis..."):
        result = analysis.analyze_all_bias_indicators(symbol_input)
        st.session_state['last_result'] = result
        
        # Send technical bias alert if strong
        if result.get('success') and result.get('overall_confidence', 0) >= 70 and telegram_enabled:
            telegram_notifier.send_bias_alert(
                "TECHNICAL ANALYSIS", 
                result['overall_bias'],
                result['overall_confidence'],
                f"Symbol: {result['symbol']}\nPrice: ₹{result['current_price']:.2f}"
            )
    
    # Calculate overall Nifty bias
    if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
        tech_result = st.session_state['last_result']
        tech_score = tech_result.get('overall_score', 0)
        
        if tech_score > 15:
            st.session_state.overall_nifty_bias = "BULLISH"
        elif tech_score < -15:
            st.session_state.overall_nifty_bias = "BEARISH"
        else:
            st.session_state.overall_nifty_bias = "NEUTRAL"
        st.session_state.overall_nifty_score = tech_score
    
    return True

# Refresh button
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
        if run_complete_analysis():
            st.sidebar.success("Analysis refreshed!")
            st.session_state.analysis_count += 1
with col2:
    st.sidebar.metric("Auto-Refresh", "ON" if auto_refresh else "OFF")
    st.sidebar.metric("Run Count", st.session_state.analysis_count)

# Auto-refresh logic
if auto_refresh:
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_refresh).total_seconds() / 60
    
    if time_diff >= refresh_interval:
        with st.spinner("Auto-refreshing analysis..."):
            if run_complete_analysis():
                st.session_state.last_refresh = current_time
                st.session_state.analysis_count += 1
                st.rerun()

# Display refresh status
if st.session_state.fetch_time:
    time_since_refresh = (datetime.now(IST) - st.session_state.fetch_time).total_seconds() / 60
    st.sidebar.write(f"Last refresh: {st.session_state.fetch_time.strftime('%H:%M:%S')} IST")
    st.sidebar.write(f"Minutes since: {time_since_refresh:.1f}m")

# Display overall Nifty bias prominently
st.sidebar.markdown("---")
st.sidebar.header("Overall Nifty Bias")
if st.session_state.overall_nifty_bias:
    bias_color = "🟢" if st.session_state.overall_nifty_bias == "BULLISH" else "🔴" if st.session_state.overall_nifty_bias == "BEARISH" else "🟡"
    st.sidebar.metric(
        "NIFTY 50 Bias",
        f"{bias_color} {st.session_state.overall_nifty_bias}",
        f"Score: {st.session_state.overall_nifty_score:.1f}"
    )

# Enhanced tabs with selected features
tabs = st.tabs([
    "Overall Bias", "Bias Summary", "Price Action"
])

# OVERALL BIAS TAB
with tabs[0]:
    st.header("🎯 Overall Nifty Bias Analysis")
    
    if not st.session_state.overall_nifty_bias:
        st.info("No analysis run yet. Click **Refresh Analysis** to get started.")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display overall bias with color coding
            if st.session_state.overall_nifty_bias == "BULLISH":
                st.success(f"## 🟢 OVERALL NIFTY BIAS: BULLISH")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bullish")
            elif st.session_state.overall_nifty_bias == "BEARISH":
                st.error(f"## 🔴 OVERALL NIFTY BIAS: BEARISH")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bearish", delta_color="inverse")
            else:
                st.warning(f"## 🟡 OVERALL NIFTY BIAS: NEUTRAL")
                st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}")
        
        st.markdown("---")
        
        # Trading Recommendation
        st.subheader("📈 Trading Recommendation")
        
        if st.session_state.overall_nifty_bias == "BULLISH":
            st.success("""
            **Recommended Action:** Consider LONG positions
            - Look for buying opportunities on dips
            - Support levels are likely to hold
            - Target resistance levels for profit booking
            """)
        elif st.session_state.overall_nifty_bias == "BEARISH":
            st.error("""
            **Recommended Action:** Consider SHORT positions  
            - Look for selling opportunities on rallies
            - Resistance levels are likely to hold
            - Target support levels for profit booking
            """)
        else:
            st.warning("""
            **Recommended Action:** Wait for clearer direction
            - Market is in consolidation phase
            - Consider range-bound strategies
            - Wait for breakout confirmation
            """)

# BIAS SUMMARY TAB
with tabs[1]:
    st.subheader("Technical Bias Summary")
    if st.session_state['last_result'] is None:
        st.info("No analysis run yet. Click **Refresh Analysis** to get started.")
    else:
        res = st.session_state['last_result']
        if not res.get('success', False):
            st.error(f"Analysis failed: {res.get('error')}")
        else:
            st.markdown(f"**Symbol:** `{res['symbol']}`")
            st.markdown(f"**Timestamp (IST):** {res['timestamp']}")
            st.metric("Current Price", f"{res['current_price']:.2f}")
            st.metric("Technical Bias", res['overall_bias'], delta=f"Confidence: {res['overall_confidence']:.1f}%")

            # Show bias results table
            bias_table = pd.DataFrame(res['bias_results'])
            # Reorder columns for nicer view
            cols_order = ['indicator', 'value', 'bias', 'score', 'weight', 'category']
            bias_table = bias_table[cols_order]
            bias_table.columns = [c.capitalize() for c in bias_table.columns]
            st.subheader("Indicator-level Biases")
            st.dataframe(bias_table, use_container_width=True)

            # Summary stats
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Bullish Indicators", int(res['bullish_count']))
            col2.metric("Bearish Indicators", int(res['bearish_count']))
            col3.metric("Neutral Indicators", int(res['neutral_count']))

# PRICE ACTION TAB
with tabs[2]:
    st.header("📈 Price Action Analysis")
    
    if st.session_state['last_df'] is None:
        st.info("No data loaded yet. Click **Refresh Analysis** to get started.")
    else:
        df = st.session_state['last_df']
        
        # Create price action chart with volume order blocks
        st.subheader("Price Chart with Volume Order Blocks")
        
        # Detect volume order blocks
        bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
        
        # Create the chart with unique ID
        chart_id = generate_element_id("price_chart", f"{symbol_input}_{period_input}_{interval_input}")
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price with Volume Order Blocks", "Volume"),
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
                name='Price',
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
        bar_colors = ['#00ff88' if row['close'] >= row['open'] else '#ff4444' for _, row in df.iterrows()]
        
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
        
        st.plotly_chart(fig, use_container_width=True, key=chart_id)
        
        # Volume Order Blocks Summary
        st.subheader("Volume Order Blocks Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Blocks", len(bullish_blocks))
            if bullish_blocks:
                latest_bullish = bullish_blocks[-1]
                st.write(f"Latest Bullish Block:")
                st.write(f"- Upper: ₹{latest_bullish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bullish['lower']:.2f}")
                st.write(f"- Volume: {latest_bullish['volume']:,.0f}")
        
        with col2:
            st.metric("Bearish Blocks", len(bearish_blocks))
            if bearish_blocks:
                latest_bearish = bearish_blocks[-1]
                st.write(f"Latest Bearish Block:")
                st.write(f"- Upper: ₹{latest_bearish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bearish['lower']:.2f}")
                st.write(f"- Volume: {latest_bearish['volume']:,.0f}")

# Run initial analysis automatically when app starts
if st.session_state.analysis_count == 0:
    with st.spinner("Running initial analysis..."):
        if run_complete_analysis():
            st.session_state.analysis_count = 1
            st.rerun()

# Footer with refresh status
st.markdown("---")
refresh_status = "🟢 Auto-refresh ACTIVE" if auto_refresh else "🟡 Auto-refresh INACTIVE"
st.caption(f"BiasAnalysisPro — {refresh_status} | Interval: {refresh_interval} min | Run Count: {st.session_state.analysis_count}")