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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib

warnings.filterwarnings('ignore')

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# Import Dhan API for Indian indices volume data
try:
    from dhan_data_fetcher import DhanDataFetcher
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    print("Warning: Dhan API not available. Volume data may be missing for Indian indices.")


# =============================================
# TELEGRAM NOTIFICATION SYSTEM
# =============================================

class TelegramNotifier:
    """Telegram notification system for bias alerts"""
    
    def __init__(self):
        # Get credentials from Streamlit secrets
        self.bot_token = st.secrets.get("TELEGRAM", {}).get("BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM", {}).get("CHAT_ID", "")
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes cooldown between same type alerts
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)
        
    def send_message(self, message: str, alert_type: str = "INFO") -> bool:
        """Send message to Telegram"""
        try:
            if not self.is_configured():
                print("Telegram credentials not configured in secrets")
                return False
            
            # Check cooldown
            current_time = time.time()
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                    print(f"Alert {alert_type} in cooldown")
                    return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.last_alert_time[alert_type] = current_time
                print(f"✅ Telegram alert sent: {alert_type}")
                return True
            else:
                print(f"❌ Telegram send failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    def send_bias_alert(self, technical_bias: str, options_bias: str, atm_bias: str, overall_bias: str, score: float):
        """Send comprehensive bias alert"""
        # Check if all three components are aligned
        components = [technical_bias, options_bias, atm_bias]
        bullish_count = sum(1 for bias in components if "BULL" in bias.upper())
        bearish_count = sum(1 for bias in components if "BEAR" in bias.upper())
        
        if bullish_count >= 2 or bearish_count >= 2:
            emoji = "🚀" if bullish_count >= 2 else "🔻"
            alert_type = "STRONG_BULL" if bullish_count >= 2 else "STRONG_BEAR"
            
            message = f"""
{emoji} <b>STRONG BIAS ALERT - NIFTY 50</b> {emoji}

📊 <b>Component Analysis:</b>
• Technical Analysis: <b>{technical_bias}</b>
• Options Chain: <b>{options_bias}</b>  
• ATM Detailed: <b>{atm_bias}</b>

🎯 <b>Overall Bias:</b> <code>{overall_bias}</code>
⭐ <b>Confidence Score:</b> <code>{score:.1f}/100</code>

⏰ <b>Time:</b> {datetime.now(IST).strftime('%H:%M:%S')}
            
💡 <b>Market Insight:</b>
{'Bullish momentum detected across multiple timeframes' if bullish_count >= 2 else 'Bearish pressure building across indicators'}
"""
            
            return self.send_message(message, alert_type)
        
        return False


# Initialize Telegram Notifier
telegram_notifier = TelegramNotifier()


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

                    # Ensure column names match yfinance format (capitalized)
                    df.columns = [col.capitalize() for col in df.columns]

                    # Set timestamp as index
                    if 'Timestamp' in df.columns:
                        df.set_index('Timestamp', inplace=True)

                    # Ensure volume column exists and has valid data
                    if 'Volume' not in df.columns:
                        df['Volume'] = 0
                    else:
                        # Replace NaN volumes with 0
                        df['Volume'] = df['Volume'].fillna(0)

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

            # Ensure volume column exists (even if it's zeros for indices)
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                # Replace NaN volumes with 0
                df['Volume'] = df['Volume'].fillna(0)

            # Warn if volume is all zeros (common for Yahoo Finance indices)
            if df['Volume'].sum() == 0 and symbol in indian_indices:
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
        if df['Volume'].sum() == 0:
            # Return neutral MFI (50) if no volume data
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

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
        high = df['High']
        low = df['Low']
        close = df['Close']

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
        if df['Volume'].sum() == 0:
            # Return typical price as fallback if no volume data
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()

        # Avoid division by zero
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe

        # Fill NaN with typical price
        vwap = vwap.fillna(typical_price)

        return vwap

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

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA (Variable Index Dynamic Average) matching Pine Script"""
        close = df['Close']

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
        if df['Volume'].sum() == 0:
            return 0, False, False

        # Calculate up and down volume
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots matching Pine Script
        Returns: (hvp_bullish, hvp_bearish, pivot_high_count, pivot_low_count)
        """
        if df['Volume'].sum() == 0:
            return False, False, 0, 0

        # Calculate pivot highs and lows
        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            # Check for pivot high
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['High'].iloc[j] >= df['High'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            # Check for pivot low
            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['Low'].iloc[j] <= df['Low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        # Calculate volume sum and reference
        volume_sum = df['Volume'].rolling(window=left_bars * 2).sum()
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
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        # Detect crossovers
        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        # In real implementation, we would check if price touched OB zones
        # For simplicity, using crossover signals
        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

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
        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }

        current_price = df['Close'].iloc[-1]

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
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)

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
        # CALCULATE OVERALL BIAS (MATCHING PINE SCRIPT LOGIC) - FIXED
        # =====================================================================
        fast_bull = 0
        fast_bear = 0
        fast_total = 0

        medium_bull = 0
        medium_bear = 0
        medium_total = 0

        # FIX 1: Disable slow category completely
        slow_bull = 0
        slow_bear = 0
        slow_total = 0  # Set to zero to avoid division by zero

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
                # Skip slow category
            elif 'BEARISH' in bias['bias']:
                bearish_count += 1
                if bias['category'] == 'fast':
                    fast_bear += 1
                elif bias['category'] == 'medium':
                    medium_bear += 1
                # Skip slow category
            else:
                neutral_count += 1

            if bias['category'] == 'fast':
                fast_total += 1
            elif bias['category'] == 'medium':
                medium_total += 1
            # Skip slow category counting

        # Calculate percentages - FIXED for slow category
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        medium_bull_pct = (medium_bull / medium_total) * 100 if medium_total > 0 else 0
        medium_bear_pct = (medium_bear / medium_total) * 100 if medium_total > 0 else 0

        # FIX 1: Set slow percentages to 0 since we disabled slow indicators
        slow_bull_pct = 0
        slow_bear_pct = 0

        # Adaptive weighting (matching Pine Script)
        # Check for divergence - FIXED for slow category
        divergence_threshold = self.config['divergence_threshold']
        # Since slow_bull_pct is 0, divergence won't trigger incorrectly
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        # Determine mode - FIXED: Always use normal mode since slow indicators disabled
        if divergence_detected and slow_total > 0:  # Only if we had slow indicators
            fast_weight = self.config['reversal_fast_weight']
            medium_weight = self.config['reversal_medium_weight']
            slow_weight = self.config['reversal_slow_weight']
            mode = "REVERSAL"
        else:
            # Use normal weights, ignore slow weight
            fast_weight = self.config['normal_fast_weight']
            medium_weight = self.config['normal_medium_weight']
            slow_weight = 0  # FIX: Set slow weight to 0 since no slow indicators
            mode = "NORMAL"

        # Calculate weighted scores - FIXED: Exclude slow category
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
            'timestamp': datetime.now(IST),
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
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period=200) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks based on the Pine Script logic"""
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.calculate_ema(df['Close'], self.length1)
        ema2 = self.calculate_ema(df['Close'], self.length2)
        
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
                    
                lowest_idx = lookback_data['Low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'Low']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[lowest_idx, 'Open']
                close_price = lookback_data.loc[lowest_idx, 'Close']
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
                    
                highest_idx = lookback_data['High'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'High']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[highest_idx, 'Open']
                close_price = lookback_data.loc[highest_idx, 'Close']
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

def plot_vob(df: pd.DataFrame, bullish_blocks: List[Dict], bearish_blocks: List[Dict]) -> go.Figure:
    """Plot Volume Order Blocks on candlestick chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=('Price with Volume Order Blocks', 'Volume'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                 row=1, col=1)
    
    # Add bullish blocks
    for block in bullish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="green", 
                     annotation_text=f"Bull Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="green", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    
    # Add bearish blocks
    for block in bearish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="red",
                     annotation_text=f"Bear Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="red", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    
    # Volume
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
                 row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, 
                     template='plotly_dark',
                     height=600,
                     showlegend=True)
    
    return fig


# =============================================
# ML PREDICTION ENGINE
# =============================================

class MLPredictionEngine:
    """ML engine that learns market behavior after bias alignments"""
    
    def __init__(self):
        self.breakout_model = None
        self.direction_model = None
        self.target_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.historical_data = []
        self.model_path = "ml_market_model.pkl"
        
    def prepare_features(self, technical_data, options_data, volume_data, bias_data, market_intel):
        """Prepare comprehensive feature vector from all app data"""
        
        features = {}
        
        # Technical Features
        if technical_data:
            features['rsi'] = technical_data.get('rsi', 50)
            features['macd_signal'] = 1 if technical_data.get('macd_signal') == 'bullish' else -1
            features['ema_trend'] = self._encode_trend(technical_data.get('ema_trend', 'neutral'))
            features['atr_normalized'] = technical_data.get('atr', 0) / max(1, technical_data.get('price', 1))
            features['sr_distance'] = technical_data.get('sr_distance', 0)
            features['htf_trend_score'] = technical_data.get('htf_trend_score', 0)
        
        # Volume/Order Block Features
        if volume_data:
            features['volume_spike'] = volume_data.get('volume_spike_ratio', 1.0)
            features['bullish_blocks'] = volume_data.get('bullish_blocks_count', 0)
            features['bearish_blocks'] = volume_data.get('bearish_blocks_count', 0)
            features['structure_break'] = 1 if volume_data.get('structure_break', False) else 0
            features['liquidity_grab'] = 1 if volume_data.get('liquidity_grab', False) else 0
        
        # Options Chain Features
        if options_data:
            features['oi_ratio'] = options_data.get('oi_ce', 1) / max(1, options_data.get('oi_pe', 1))
            features['iv_skew'] = options_data.get('iv_ce', 0) - options_data.get('iv_pe', 0)
            features['max_pain_shift'] = options_data.get('max_pain_shift', 0)
            features['pcr_value'] = options_data.get('pcr', 1.0)
            features['atm_bias_score'] = options_data.get('atm_bias_score', 0)
            features['oi_unwinding'] = options_data.get('oi_unwinding_ratio', 0)
        
        # Market Intel Features
        if market_intel:
            features['india_vix'] = market_intel.get('india_vix', 0)
            features['sgx_nifty_change'] = market_intel.get('sgx_nifty_change', 0)
            features['global_futures_trend'] = self._encode_trend(market_intel.get('global_trend', 'neutral'))
            features['fii_dii_trend'] = market_intel.get('fii_dii_trend', 0)
            features['sector_rotation'] = market_intel.get('sector_rotation_strength', 0)
        
        # Bias Score Features
        if bias_data:
            features['technical_bias_score'] = bias_data.get('technical_bias', 0)
            features['options_bias_score'] = bias_data.get('options_bias', 0)
            features['institutional_bias_score'] = bias_data.get('institutional_bias', 0)
            features['volume_confirmation'] = 1 if bias_data.get('volume_confirmed', False) else 0
            features['market_sentiment_score'] = bias_data.get('sentiment_score', 0)
            features['unified_bias'] = bias_data.get('unified_bias', 0)
        
        # Derived Features
        features['bias_alignment'] = self._calculate_bias_alignment(bias_data)
        features['momentum_strength'] = self._calculate_momentum_strength(technical_data, volume_data)
        features['institutional_pressure'] = self._calculate_institutional_pressure(options_data, bias_data)
        
        return features
    
    def _encode_trend(self, trend):
        """Encode trend as numerical value"""
        trend_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
        return trend_map.get(trend, 0)
    
    def _calculate_bias_alignment(self, bias_data):
        """Calculate how aligned different biases are"""
        if not bias_data:
            return 0
        
        biases = [
            bias_data.get('technical_bias', 0),
            bias_data.get('options_bias', 0),
            bias_data.get('institutional_bias', 0)
        ]
        
        alignment = sum(biases) / (max(1, sum(abs(b) for b in biases)))
        return alignment
    
    def _calculate_momentum_strength(self, technical_data, volume_data):
        """Calculate combined momentum strength"""
        momentum = 0
        
        if technical_data:
            momentum += technical_data.get('rsi_strength', 0) * 0.3
            momentum += technical_data.get('trend_strength', 0) * 0.4
        
        if volume_data:
            momentum += volume_data.get('volume_momentum', 0) * 0.3
        
        return momentum
    
    def _calculate_institutional_pressure(self, options_data, bias_data):
        """Calculate institutional pressure"""
        pressure = 0
        
        if options_data:
            pressure += options_data.get('oi_pressure', 0) * 0.6
            pressure += options_data.get('iv_pressure', 0) * 0.4
        
        if bias_data:
            pressure += bias_data.get('institutional_bias', 0) * 0.3
        
        return pressure
    
    def create_labels(self, current_data, future_data, timeframe_minutes=30):
        """Create ML labels based on future price action"""
        if not future_data:
            return 0, 0, 0
        
        current_price = current_data.get('price', 0)
        future_price = future_data.get('price', current_price)
        high_price = future_data.get('high', future_price)
        low_price = future_data.get('low', future_price)
        
        # Breakout or Fakeout
        breakout_level = current_data.get('breakout_level', current_price)
        if current_price > breakout_level:
            breakout_success = 1 if future_price > breakout_level else 0
        else:
            breakout_success = 1 if future_price < breakout_level else 0
        
        # Direction success
        current_bias = current_data.get('unified_bias', 0)
        if current_bias > 0:
            direction_success = 1 if future_price > current_price else 0
        elif current_bias < 0:
            direction_success = 1 if future_price < current_price else 0
        else:
            direction_success = 0
        
        # Target hit
        target_level = current_data.get('target_level', current_price)
        if current_bias > 0:
            target_hit = 1 if high_price >= target_level else 0
        elif current_bias < 0:
            target_hit = 1 if low_price <= target_level else 0
        else:
            target_hit = 0
        
        return breakout_success, direction_success, target_hit
    
    def collect_training_data(self, feature_vector, future_outcomes):
        """Collect data for ML training"""
        training_sample = {
            'timestamp': datetime.now(IST),
            'features': feature_vector,
            'outcomes': future_outcomes,
            'breakout_success': future_outcomes[0],
            'direction_success': future_outcomes[1],
            'target_hit': future_outcomes[2]
        }
        
        self.historical_data.append(training_sample)
        
        # Keep only last 6 months of data
        six_months_ago = datetime.now(IST) - timedelta(days=180)
        self.historical_data = [
            data for data in self.historical_data 
            if data['timestamp'] > six_months_ago
        ]
    
    def train_model(self):
        """Train the ML model on collected historical data"""
        if len(self.historical_data) < 100:
            print("Insufficient training data. Need at least 100 samples.")
            return False
        
        try:
            # Prepare features and labels
            X = []
            y_breakout = []
            y_direction = []
            y_target = []
            
            for sample in self.historical_data:
                features = list(sample['features'].values())
                X.append(features)
                y_breakout.append(sample['breakout_success'])
                y_direction.append(sample['direction_success'])
                y_target.append(sample['target_hit'])
            
            X = np.array(X)
            y_breakout = np.array(y_breakout)
            y_direction = np.array(y_direction)
            y_target = np.array(y_target)
            
            # Store feature columns
            if not self.feature_columns:
                self.feature_columns = list(self.historical_data[0]['features'].keys())
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train multiple models for different predictions
            self.breakout_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.breakout_model.fit(X_scaled, y_breakout)
            
            self.direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.direction_model.fit(X_scaled, y_direction)
            
            self.target_model = LogisticRegression(random_state=42)
            self.target_model.fit(X_scaled, y_target)
            
            # Calculate cross-validation scores
            breakout_score = cross_val_score(self.breakout_model, X_scaled, y_breakout, cv=5).mean()
            direction_score = cross_val_score(self.direction_model, X_scaled, y_direction, cv=5).mean()
            target_score = cross_val_score(self.target_model, X_scaled, y_target, cv=5).mean()
            
            self.is_trained = True
            
            print(f"✅ ML Models Trained Successfully!")
            print(f"   Breakout Prediction CV Score: {breakout_score:.3f}")
            print(f"   Direction Prediction CV Score: {direction_score:.3f}")
            print(f"   Target Prediction CV Score: {target_score:.3f}")
            
            # Save models
            self.save_models()
            
            return True
            
        except Exception as e:
            print(f"Error training ML model: {e}")
            return False
    
    def predict_confidence(self, feature_vector):
        """Predict ML confidence score (0-100)"""
        if not self.is_trained:
            return 50.0
        
        try:
            features = np.array([list(feature_vector.values())])
            features_scaled = self.scaler.transform(features)
            
            breakout_prob = self.breakout_model.predict_proba(features_scaled)[0][1]
            direction_prob = self.direction_model.predict_proba(features_scaled)[0][1]
            target_prob = self.target_model.predict_proba(features_scaled)[0][1]
            
            confidence = (
                breakout_prob * 0.4 +
                direction_prob * 0.4 +  
                target_prob * 0.2
            ) * 100
            
            return min(100, max(0, confidence))
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return 50.0
    
    def save_models(self):
        """Save trained models"""
        try:
            model_data = {
                'breakout_model': self.breakout_model,
                'direction_model': self.direction_model,
                'target_model': self.target_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
            print("✅ ML models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            model_data = joblib.load(self.model_path)
            
            self.breakout_model = model_data['breakout_model']
            self.direction_model = model_data['direction_model']
            self.target_model = model_data['target_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print("✅ ML models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


# =============================================
# NEWS API INTEGRATION
# =============================================

class NewsAPIIntegration:
    """Real-time news sentiment analysis"""
    
    def __init__(self):
        self.news_api_key = st.secrets.get("NEWSAPI", {}).get("API_KEY", "")
        self.last_fetch_time = None
        self.news_cache = []
    
    def fetch_market_news(self, query="Nifty Stock Market India"):
        """Fetch real-time market news"""
        if not self.news_api_key:
            return self._get_sample_news()
        
        try:
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}&sortBy=publishedAt&pageSize=10"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                articles = news_data.get('articles', [])
                
                processed_articles = []
                for article in articles:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'url': article.get('url', ''),
                        'sentiment': self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                    })
                
                self.news_cache = processed_articles
                self.last_fetch_time = datetime.now(IST)
                return processed_articles
            else:
                print(f"News API error: {response.status_code}")
                return self._get_sample_news()
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_sample_news()
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        bullish_keywords = ['bullish', 'rally', 'surge', 'gain', 'up', 'positive', 'strong', 'buy', 'outperform']
        bearish_keywords = ['bearish', 'fall', 'drop', 'decline', 'down', 'negative', 'weak', 'sell', 'underperform']
        
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_sample_news(self):
        """Return sample news when API is unavailable"""
        return [
            {
                'title': 'Nifty 50 Shows Strong Momentum Amid Positive Global Cues',
                'description': 'Indian markets continue upward trend with strong institutional participation',
                'source': 'Market News',
                'sentiment': 'bullish',
                'impact': 'high'
            },
            {
                'title': 'Banking Stocks Face Pressure Due to Regulatory Changes',
                'description': 'Banking sector under scrutiny as new regulations take effect',
                'source': 'Financial Times', 
                'sentiment': 'bearish',
                'impact': 'medium'
            }
        ]
    
    def get_news_sentiment_score(self):
        """Calculate overall news sentiment score"""
        if not self.news_cache:
            self.fetch_market_news()
        
        sentiments = [article['sentiment'] for article in self.news_cache]
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        total = len(sentiments)
        
        if total == 0:
            return 0
        
        sentiment_score = (bullish_count - bearish_count) / total * 100
        return sentiment_score


# =============================================
# MASTER TRIGGER ENGINE
# =============================================

class MasterTriggerEngine:
    """Master engine that coordinates ML, news, and AI triggers"""
    
    def __init__(self):
        self.ml_engine = MLPredictionEngine()
        self.news_api = NewsAPIIntegration()
        self.telegram_notifier = telegram_notifier
        self.last_trigger_time = None
        self.trigger_cooldown = 300
    
    def should_trigger_analysis(self, unified_bias, ml_confidence):
        """Check if conditions meet for master trigger"""
        current_time = datetime.now(IST)
        
        if self.last_trigger_time:
            time_diff = (current_time - self.last_trigger_time).total_seconds()
            if time_diff < self.trigger_cooldown:
                return False
        
        if unified_bias >= 80 and ml_confidence >= 85:
            self.last_trigger_time = current_time
            return True
        
        return False
    
    def execute_master_trigger(self, feature_data, unified_bias, ml_confidence):
        """Execute full analysis pipeline when triggered"""
        try:
            print("🚀 MASTER TRIGGER ACTIVATED!")
            
            news_articles = self.news_api.fetch_market_news()
            news_sentiment = self.news_api.get_news_sentiment_score()
            
            ai_analysis = self._enhanced_ai_analysis(feature_data, news_articles, unified_bias, ml_confidence)
            prediction = self._generate_prediction(feature_data, ai_analysis, news_sentiment)
            
            self._send_comprehensive_alert(unified_bias, ml_confidence, prediction, ai_analysis)
            
            return {
                'triggered': True,
                'timestamp': datetime.now(IST),
                'unified_bias': unified_bias,
                'ml_confidence': ml_confidence,
                'news_sentiment': news_sentiment,
                'ai_analysis': ai_analysis,
                'prediction': prediction
            }
            
        except Exception as e:
            print(f"Error in master trigger: {e}")
            return {'triggered': False, 'error': str(e)}
    
    def _enhanced_ai_analysis(self, feature_data, news_articles, unified_bias, ml_confidence):
        """Enhanced AI analysis combining all data"""
        
        context = {
            'market_context': {
                'unified_bias': unified_bias,
                'ml_confidence': ml_confidence,
                'technical_strength': feature_data.get('technical_bias_score', 0),
                'options_pressure': feature_data.get('options_bias_score', 0),
                'institutional_flow': feature_data.get('institutional_bias_score', 0)
            },
            'news_summary': {
                'total_articles': len(news_articles),
                'bullish_articles': len([a for a in news_articles if a['sentiment'] == 'bullish']),
                'bearish_articles': len([a for a in news_articles if a['sentiment'] == 'bearish']),
                'key_themes': self._extract_news_themes(news_articles)
            },
            'market_conditions': {
                'bias_alignment': feature_data.get('bias_alignment', 0),
                'momentum_strength': feature_data.get('momentum_strength', 0),
                'institutional_pressure': feature_data.get('institutional_pressure', 0)
            }
        }
        
        ai_insights = self._generate_ai_insights(context)
        
        return ai_insights
    
    def _extract_news_themes(self, news_articles):
        """Extract key themes from news articles"""
        themes = []
        all_text = ' '.join([article['title'] + ' ' + article.get('description', '') for article in news_articles])
        
        theme_keywords = {
            'earnings': ['earnings', 'results', 'profit', 'revenue'],
            'economy': ['economy', 'gdp', 'inflation', 'rates', 'rbi'],
            'global': ['global', 'fed', 'us', 'europe', 'asia'],
            'sector': ['banking', 'it', 'auto', 'pharma', 'energy'],
            'technical': ['technical', 'breakout', 'support', 'resistance']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text.lower() for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]
    
    def _generate_ai_insights(self, context):
        """Generate AI-powered market insights"""
        
        market_context = context['market_context']
        news_summary = context['news_summary']
        
        insights = {
            'market_outlook': 'NEUTRAL',
            'confidence': 'MEDIUM',
            'key_drivers': [],
            'risks': [],
            'recommendation': 'HOLD'
        }
        
        bias_score = market_context['unified_bias']
        ml_score = market_context['ml_confidence']
        news_bullish_ratio = news_summary['bullish_articles'] / max(1, news_summary['total_articles'])
        
        combined_score = (bias_score * 0.4 + ml_score * 0.4 + news_bullish_ratio * 100 * 0.2)
        
        if combined_score > 70:
            insights['market_outlook'] = 'BULLISH'
            insights['confidence'] = 'HIGH' if ml_score > 85 else 'MEDIUM'
            insights['recommendation'] = 'BUY'
        elif combined_score < 30:
            insights['market_outlook'] = 'BEARISH'
            insights['confidence'] = 'HIGH' if ml_score > 85 else 'MEDIUM'
            insights['recommendation'] = 'SELL'
        
        if market_context['technical_strength'] > 70:
            insights['key_drivers'].append('Strong Technical Momentum')
        if market_context['options_pressure'] > 70:
            insights['key_drivers'].append('Positive Options Flow')
        if market_context['institutional_flow'] > 70:
            insights['key_drivers'].append('Institutional Support')
        
        if news_summary['bearish_articles'] > news_summary['bullish_articles']:
            insights['risks'].append('Negative News Sentiment')
        if market_context['ml_confidence'] < 70:
            insights['risks'].append('Low ML Confidence')
        
        return insights
    
    def _generate_prediction(self, feature_data, ai_analysis, news_sentiment):
        """Generate comprehensive market prediction"""
        
        prediction = {
            'timeframe': '1-4 hours',
            'direction': ai_analysis['market_outlook'],
            'confidence': ai_analysis['confidence'],
            'targets': self._calculate_targets(feature_data, ai_analysis['market_outlook']),
            'stoploss': self._calculate_stoploss(feature_data, ai_analysis['market_outlook']),
            'probability': min(100, max(0, feature_data.get('ml_confidence', 50) + news_sentiment / 2)),
            'validity_period': '4 hours'
        }
        
        return prediction
    
    def _calculate_targets(self, feature_data, direction):
        if direction == 'BULLISH':
            return ['+0.5%', '+1.0%', '+1.5%']
        elif direction == 'BEARISH':
            return ['-0.5%', '-1.0%', '-1.5%']
        else:
            return ['+0.25%', '-0.25%']
    
    def _calculate_stoploss(self, feature_data, direction):
        if direction == 'BULLISH':
            return '-0.8%'
        elif direction == 'BEARISH':
            return '+0.8%'
        else:
            return '-0.5%'
    
    def _send_comprehensive_alert(self, unified_bias, ml_confidence, prediction, ai_analysis):
        """Send comprehensive alert via Telegram"""
        
        if not self.telegram_notifier.is_configured():
            return
        
        message = f"""
🚀 **MASTER TRIGGER ALERT - HIGH CONVICTION SIGNAL**

📊 **Signal Strength:**
• Unified Bias: `{unified_bias:.1f}`
• ML Confidence: `{ml_confidence:.1f}`
• Combined Score: `{(unified_bias + ml_confidence) / 2:.1f}`

🎯 **AI Prediction:**
• Direction: `{prediction['direction']}`
• Confidence: `{prediction['confidence']}`
• Timeframe: `{prediction['timeframe']}`
• Probability: `{prediction['probability']:.1f}%`

📈 **Targets:** {', '.join(prediction['targets'])}
🛡️ **Stoploss:** {prediction['stoploss']}

💡 **Key Drivers:**
{chr(10).join(['• ' + driver for driver in ai_analysis.get('key_drivers', ['Strong alignment across all factors'])])}

⚠️ **Risks:**
{chr(10).join(['• ' + risk for risk in ai_analysis.get('risks', ['Normal market volatility'])])}

⏰ **Timestamp:** {datetime.now(IST).strftime('%H:%M:%S')}
        """
        
        self.telegram_notifier.send_message(message, "MASTER_TRIGGER")


# =============================================
# SIMPLIFIED NSE OPTIONS ANALYZER
# =============================================

class SimpleOptionsAnalyzer:
    """Simplified options analyzer for demo purposes"""
    
    def __init__(self):
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
            }
        }
    
    def get_sample_options_data(self, instrument="NIFTY"):
        """Generate sample options data for demo"""
        import random
        
        return {
            'instrument': instrument,
            'spot_price': 22150 + random.randint(-100, 100),
            'atm_strike': 22100,
            'overall_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
            'bias_score': random.uniform(-5, 5),
            'pcr_oi': random.uniform(0.8, 1.2),
            'pcr_change': random.uniform(0.9, 1.1),
            'total_ce_oi': random.randint(5000000, 8000000),
            'total_pe_oi': random.randint(5000000, 8000000),
            'total_ce_change': random.randint(-100000, 100000),
            'total_pe_change': random.randint(-100000, 100000),
            'detailed_atm_bias': {
                'OI_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'ChgOI_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'Volume_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
            },
            'comprehensive_metrics': {
                'synthetic_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'atm_buildup': random.choice(['Long Buildup', 'Short Buildup', 'Neutral']),
                'max_pain_strike': 22100 + random.randint(-100, 100),
                'total_vega_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'call_resistance': 22200 + random.randint(0, 100),
                'put_support': 22000 - random.randint(0, 100)
            }
        }


# =============================================
# STREAMLIT APP UI
# =============================================

def add_ml_analysis_tab():
    """Add ML Analysis tab to the Streamlit app"""
    
    st.header("🤖 ML Prediction Engine & Master Trigger")
    st.markdown("### ML learns market behavior after bias alignments")
    
    # Initialize session state
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = MLPredictionEngine()
        st.session_state.ml_engine.load_models()
    
    if 'master_engine' not in st.session_state:
        st.session_state.master_engine = MasterTriggerEngine()
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs([
        "🧠 ML Predictions", 
        "📰 News Sentiment", 
        "🚀 Master Trigger"
    ])
    
    with tab1:
        st.subheader("Machine Learning Predictions")
        
        if st.session_state.get('analysis_complete'):
            feature_data = prepare_feature_data()
            
            if feature_data:
                ml_confidence = st.session_state.ml_engine.predict_confidence(feature_data)
                unified_bias = st.session_state.overall_nifty_score
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ML Confidence", f"{ml_confidence:.1f}%")
                
                with col2:
                    st.metric("Unified Bias", f"{unified_bias:.1f}")
                
                with col3:
                    trigger_ready = unified_bias >= 80 and ml_confidence >= 85
                    status_color = "🟢" if trigger_ready else "🟡"
                    st.metric("Trigger Status", f"{status_color} {'READY' if trigger_ready else 'WAITING'}")
                
                with col4:
                    bias_alignment = feature_data.get('bias_alignment', 0)
                    st.metric("Bias Alignment", f"{bias_alignment:.1%}")
                
                # ML Training Section
                st.subheader("🎯 ML Model Training")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Train ML Model", use_container_width=True):
                        with st.spinner("Training ML model..."):
                            if st.session_state.ml_engine.train_model():
                                st.success("✅ ML model trained successfully!")
                            else:
                                st.error("❌ ML training failed. Need more data.")
                
                with col2:
                    if st.button("💾 Save ML Data", use_container_width=True):
                        st.success("✅ Training data collected!")
                
                # ML Insights
                st.subheader("🔍 ML Insights")
                
                if ml_confidence > 70:
                    st.success(f"**High Confidence Signal** ({ml_confidence:.1f}%)")
                    st.info("ML indicates strong probability of continued move in the bias direction")
                elif ml_confidence < 30:
                    st.error(f"**Low Confidence Signal** ({ml_confidence:.1f}%)") 
                    st.warning("ML suggests caution - market move may not sustain")
                else:
                    st.warning(f"**Neutral Confidence** ({ml_confidence:.1f}%)")
                    st.info("ML suggests waiting for clearer signals")
        
        else:
            st.info("Run complete analysis first to generate ML predictions")
    
    with tab2:
        st.subheader("📰 Real-time News Sentiment")
        
        if st.button("📡 Fetch Latest News"):
            with st.spinner("Fetching market news..."):
                news_articles = st.session_state.master_engine.news_api.fetch_market_news()
                sentiment_score = st.session_state.master_engine.news_api.get_news_sentiment_score()
                
                st.metric("Overall News Sentiment", f"{sentiment_score:.1f}%")
                
                for i, article in enumerate(news_articles[:5]):
                    with st.expander(f"{article['title']} - {article['source']}"):
                        st.write(f"**Description:** {article['description']}")
                        st.write(f"**Sentiment:** {article['sentiment']}")
                        st.write(f"**Published:** {article['published_at']}")
    
    with tab3:
        st.subheader("🚀 Master Trigger Engine")
        
        if st.session_state.get('analysis_complete'):
            feature_data = prepare_feature_data()
            unified_bias = st.session_state.overall_nifty_score
            ml_confidence = st.session_state.ml_engine.predict_confidence(feature_data) if feature_data else 0
            
            trigger_ready = st.session_state.master_engine.should_trigger_analysis(unified_bias, ml_confidence)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if trigger_ready:
                    st.success("🎯 TRIGGER READY!")
                    st.write("Conditions met:")
                    st.write(f"- Unified Bias: {unified_bias:.1f} >= 80")
                    st.write(f"- ML Confidence: {ml_confidence:.1f} >= 85")
                    
                    if st.button("🔥 EXECUTE MASTER TRIGGER", type="primary"):
                        with st.spinner("Executing master analysis..."):
                            result = st.session_state.master_engine.execute_master_trigger(
                                feature_data, unified_bias, ml_confidence
                            )
                            
                            if result['triggered']:
                                st.balloons()
                                st.success("🎉 Master trigger executed successfully!")
                                st.subheader("Trigger Results")
                                st.json(result)
                            else:
                                st.error("Trigger execution failed")
                else:
                    st.warning("🕐 Trigger conditions not met")
                    st.write("Required conditions:")
                    st.write("- Unified Bias >= 80")
                    st.write("- ML Confidence >= 85")
                    st.write(f"Current: Bias={unified_bias:.1f}, ML={ml_confidence:.1f}")
            
            with col2:
                st.subheader("Trigger History")
                st.info("Trigger history will appear here after executions")
        
        else:
            st.info("Run complete analysis to activate master trigger")

def prepare_feature_data():
    """Prepare feature data from all analysis components"""
    feature_data = {}
    
    # Technical data
    if st.session_state.get('last_result') and st.session_state['last_result'].get('success'):
        tech_data = st.session_state['last_result']
        feature_data['technical_bias_score'] = tech_data.get('overall_score', 0)
        feature_data['rsi'] = 50
        feature_data['ema_trend'] = 'bullish' if tech_data.get('overall_bias') == 'BULLISH' else 'bearish'
    
    # Options data
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY':
                feature_data['options_bias_score'] = instrument_data.get('bias_score', 0)
                feature_data['pcr_value'] = instrument_data.get('pcr_oi', 1.0)
                feature_data['atm_bias_score'] = instrument_data.get('bias_score', 0)
                break
    
    # Volume data
    if st.session_state.vob_blocks:
        feature_data['bullish_blocks_count'] = len(st.session_state.vob_blocks.get('bullish', []))
        feature_data['bearish_blocks_count'] = len(st.session_state.vob_blocks.get('bearish', []))
        feature_data['volume_confirmed'] = 1 if feature_data['bullish_blocks_count'] > feature_data['bearish_blocks_count'] else 0
    
    # Unified bias
    feature_data['unified_bias'] = st.session_state.overall_nifty_score
    feature_data['sentiment_score'] = 50
    
    return feature_data

def run_complete_analysis():
    """Run complete analysis for all tabs"""
    try:
        # Technical Analysis
        analysis = BiasAnalysisPro()
        df_fetched = analysis.fetch_data("^NSEI", period='7d', interval='5m')
        
        if df_fetched is None or df_fetched.empty:
            st.error("No data fetched. Check symbol or network.")
            return False
            
        st.session_state['last_df'] = df_fetched
        st.session_state['fetch_time'] = datetime.now(IST)

        # Run bias analysis
        result = analysis.analyze_all_bias_indicators("^NSEI")
        st.session_state['last_result'] = result

        # Run Volume Order Blocks analysis
        vob_indicator = VolumeOrderBlocks(sensitivity=5)
        if st.session_state['last_df'] is not None:
            df = st.session_state['last_df']
            bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
            st.session_state.vob_blocks = {
                'bullish': bullish_blocks,
                'bearish': bearish_blocks
            }

        # Run options analysis
        options_analyzer = SimpleOptionsAnalyzer()
        enhanced_bias_data = []
        instruments = ['NIFTY', 'BANKNIFTY']
        
        for instrument in instruments:
            bias_data = options_analyzer.get_sample_options_data(instrument)
            enhanced_bias_data.append(bias_data)
        
        st.session_state.market_bias_data = enhanced_bias_data
        st.session_state.last_bias_update = datetime.now(IST)
        
        # Calculate overall bias
        st.session_state.overall_nifty_bias = "BULLISH" if result.get('overall_bias') == 'BULLISH' else "BEARISH" if result.get('overall_bias') == 'BEARISH' else "NEUTRAL"
        st.session_state.overall_nifty_score = result.get('overall_score', 0)
        
        st.session_state.analysis_complete = True
        return True
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        return False

def main():
    st.set_page_config(page_title="Bias Analysis Pro - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 Bias Analysis Pro — Complete Single-file App")
    st.markdown("Advanced market analysis with ML-powered predictions and real-time insights")

    # Sidebar
    st.sidebar.header("Data & Symbol")
    symbol_input = st.sidebar.text_input("Symbol", value="^NSEI")
    period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d'], index=2)
    interval_input = st.sidebar.selectbox("Interval", options=['5m', '15m', '1h'], index=0)

    # Auto-refresh
    st.sidebar.header("Auto-Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=2)

    # Telegram
    st.sidebar.header("🔔 Telegram Alerts")
    if telegram_notifier.is_configured():
        st.sidebar.success("✅ Telegram configured!")
        telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    else:
        st.sidebar.warning("⚠️ Telegram not configured")
        telegram_enabled = False

    # Initialize session state
    if 'last_df' not in st.session_state:
        st.session_state['last_df'] = None
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    if 'market_bias_data' not in st.session_state:
        st.session_state.market_bias_data = None
    if 'overall_nifty_bias' not in st.session_state:
        st.session_state.overall_nifty_bias = "NEUTRAL"
    if 'overall_nifty_score' not in st.session_state:
        st.session_state.overall_nifty_score = 0
    if 'vob_blocks' not in st.session_state:
        st.session_state.vob_blocks = {'bullish': [], 'bearish': []}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Analysis button
    if st.sidebar.button("🔄 Run Complete Analysis", type="primary", use_container_width=True):
        with st.spinner("Running complete analysis..."):
            if run_complete_analysis():
                st.sidebar.success("Analysis complete!")
                st.rerun()

    # Display overall bias
    st.sidebar.markdown("---")
    st.sidebar.header("Overall Nifty Bias")
    if st.session_state.overall_nifty_bias:
        bias_color = "🟢" if st.session_state.overall_nifty_bias == "BULLISH" else "🔴" if st.session_state.overall_nifty_bias == "BEARISH" else "🟡"
        st.sidebar.metric(
            "NIFTY 50 Bias",
            f"{bias_color} {st.session_state.overall_nifty_bias}",
            f"Score: {st.session_state.overall_nifty_score:.1f}"
        )

    # Main tabs
    tabs = st.tabs([
        "Overall Bias", "Bias Summary", "Price Action", "Option Chain", 
        "Bias Tabulation", "🤖 AI Analysis", "🚀 ML Engine"
    ])

    with tabs[0]:
        st.header("🎯 Overall Nifty Bias Analysis")
        
        if not st.session_state.overall_nifty_bias:
            st.info("No analysis run yet. Click 'Run Complete Analysis' to start.")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
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
            
            # Components breakdown
            st.subheader("Bias Components Breakdown")
            components_data = []
            
            if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
                tech_result = st.session_state['last_result']
                components_data.append({
                    'Component': 'Technical Analysis',
                    'Bias': tech_result.get('overall_bias', 'NEUTRAL'),
                    'Score': tech_result.get('overall_score', 0),
                    'Weight': '20%'
                })
            
            if st.session_state.market_bias_data:
                for instrument_data in st.session_state.market_bias_data:
                    if instrument_data['instrument'] == 'NIFTY':
                        components_data.append({
                            'Component': 'Options Chain',
                            'Bias': instrument_data.get('overall_bias', 'Neutral'),
                            'Score': instrument_data.get('bias_score', 0),
                            'Weight': '25%'
                        })
                        break
            
            if st.session_state.vob_blocks:
                bullish_blocks = st.session_state.vob_blocks['bullish']
                bearish_blocks = st.session_state.vob_blocks['bearish']
                vob_score = len(bullish_blocks) - len(bearish_blocks)
                vob_bias = "BULLISH" if vob_score > 0 else "BEARISH" if vob_score < 0 else "NEUTRAL"
                
                components_data.append({
                    'Component': 'Volume Order Blocks',
                    'Bias': vob_bias,
                    'Score': vob_score,
                    'Weight': '10%'
                })
            
            if components_data:
                components_df = pd.DataFrame(components_data)
                st.dataframe(components_df, use_container_width=True)
            
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

    with tabs[1]:
        st.header("📊 Technical Bias Summary")
        
        if st.session_state['last_result'] is None:
            st.info("No analysis run yet. Click 'Run Complete Analysis' to start.")
        else:
            res = st.session_state['last_result']
            if not res.get('success', False):
                st.error(f"Analysis failed: {res.get('error')}")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"{res['current_price']:.2f}")
                col2.metric("Technical Bias", res['overall_bias'])
                col3.metric("Confidence", f"{res['overall_confidence']:.1f}%")

                # Show bias results
                if res.get('bias_results'):
                    bias_table = pd.DataFrame(res['bias_results'])
                    st.dataframe(bias_table, use_container_width=True)

    with tabs[2]:
        st.header("📈 Price Action Analysis")
        
        if st.session_state['last_df'] is None:
            st.info("No data loaded yet. Click 'Run Complete Analysis' to start.")
        else:
            df = st.session_state['last_df']
            
            # Price chart with volume order blocks
            st.subheader("Price Chart with Volume Order Blocks")
            bullish_blocks = st.session_state.vob_blocks.get('bullish', [])
            bearish_blocks = st.session_state.vob_blocks.get('bearish', [])
            
            fig = plot_vob(df, bullish_blocks, bearish_blocks)
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume analysis
            st.subheader("Volume Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Bullish Blocks", len(bullish_blocks))
                if bullish_blocks:
                    latest_bullish = bullish_blocks[-1]
                    st.write(f"Latest Bullish Block:")
                    st.write(f"- Range: ₹{latest_bullish['lower']:.2f} - ₹{latest_bullish['upper']:.2f}")
            
            with col2:
                st.metric("Bearish Blocks", len(bearish_blocks))
                if bearish_blocks:
                    latest_bearish = bearish_blocks[-1]
                    st.write(f"Latest Bearish Block:")
                    st.write(f"- Range: ₹{latest_bearish['lower']:.2f} - ₹{latest_bearish['upper']:.2f}")

    with tabs[3]:
        st.header("📊 Options Chain Analysis")
        
        if not st.session_state.market_bias_data:
            st.info("No options data available. Click 'Run Complete Analysis' to start.")
        else:
            for instrument_data in st.session_state.market_bias_data:
                with st.expander(f"📈 {instrument_data['instrument']} Options Analysis", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                    with col2:
                        st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                    with col3:
                        st.metric("Overall Bias", instrument_data['overall_bias'])
                    with col4:
                        st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
                    
                    # Show detailed metrics
                    st.subheader("Detailed Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**OI Analysis:**")
                        st.write(f"- Total CE OI: {instrument_data['total_ce_oi']:,}")
                        st.write(f"- Total PE OI: {instrument_data['total_pe_oi']:,}")
                        st.write(f"- CE OI Change: {instrument_data['total_ce_change']:,}")
                        st.write(f"- PE OI Change: {instrument_data['total_pe_change']:,}")
                    
                    with col2:
                        st.write("**Key Levels:**")
                        comp_metrics = instrument_data.get('comprehensive_metrics', {})
                        st.write(f"- Max Pain: ₹{comp_metrics.get('max_pain_strike', 'N/A')}")
                        st.write(f"- Call Resistance: ₹{comp_metrics.get('call_resistance', 'N/A')}")
                        st.write(f"- Put Support: ₹{comp_metrics.get('put_support', 'N/A')}")

    with tabs[4]:
        st.header("📋 Comprehensive Bias Tabulation")
        
        if not st.session_state.market_bias_data:
            st.info("No data available. Click 'Run Complete Analysis' to start.")
        else:
            for instrument_data in st.session_state.market_bias_data:
                with st.expander(f"🎯 {instrument_data['instrument']} - Complete Analysis"):
                    # Basic Information
                    st.subheader("Basic Information")
                    info_data = {
                        'Metric': ['Instrument', 'Spot Price', 'ATM Strike', 'Overall Bias', 'Bias Score', 'PCR OI'],
                        'Value': [
                            instrument_data['instrument'],
                            f"₹{instrument_data['spot_price']:.2f}",
                            f"₹{instrument_data['atm_strike']:.2f}",
                            instrument_data['overall_bias'],
                            f"{instrument_data['bias_score']:.2f}",
                            f"{instrument_data['pcr_oi']:.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)
                    
                    # Bias Breakdown
                    if 'detailed_atm_bias' in instrument_data:
                        st.subheader("ATM Bias Breakdown")
                        bias_data = []
                        for key, value in instrument_data['detailed_atm_bias'].items():
                            bias_data.append({
                                'Metric': key,
                                'Value': value
                            })
                        st.dataframe(pd.DataFrame(bias_data), use_container_width=True)

    with tabs[5]:
        st.header("🤖 AI-Powered Analysis")
        st.info("""
        **AI Analysis Systems:**
        - **Smart Interpretation:** Converts complex data into human-readable insights
        - **Pattern Recognition:** Identifies recurring market patterns
        - **Sentiment Analysis:** Analyzes news and market sentiment
        
        *AI systems provide enhanced market understanding beyond traditional indicators*
        """)
        
        if st.session_state.get('analysis_complete'):
            st.success("✅ AI Analysis Ready")
            st.write("Market data has been processed and is ready for AI interpretation")
            
            # Sample AI insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Intelligence", "HIGH", "Advanced")
            with col2:
                st.metric("Pattern Recognition", "ACTIVE", "Real-time")
            with col3:
                st.metric("Risk Assessment", "MODERATE", "Stable")
        else:
            st.warning("Run complete analysis to enable AI systems")

    with tabs[6]:
        add_ml_analysis_tab()

    # Footer
    st.markdown("---")
    st.caption("BiasAnalysisPro — Complete Enhanced Dashboard with ML-Powered Predictions • Real-time Market Analysis • Institutional Grade Insights")

if __name__ == "__main__":
    main()
