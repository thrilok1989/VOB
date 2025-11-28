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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# FIX 5: Add plotting function for VOB
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
# TRADING SIGNAL ENGINE FUNCTIONS
# =============================================

# ================================================================
# 1. CONSOLIDATION DETECTION
# ================================================================
def is_consolidation(df, vob_zones):
    last_15 = df.tail(15)

    # Condition A — Tight Range < 0.35%
    price_range = last_15['high'].max() - last_15['low'].min()
    tight_range = price_range < (df.iloc[-1]['close'] * 0.0035)

    # Condition B — Small bodies vs wicks
    body = abs(last_15['close'] - last_15['open']).mean()
    wick = ((last_15['high'] - last_15['low']) - body).mean()
    small_bodies = body < (0.5 * wick)

    # Condition C — Flat EMA20
    df['ema20'] = df['close'].ewm(span=20).mean()
    ema_slope = df['ema20'].iloc[-1] - df['ema20'].iloc[-5]
    ema_flat = abs(ema_slope) < (df.iloc[-1]['close'] * 0.0007)

    # Condition D — Inside VOB trap zone
    inside_vob = df.iloc[-1]['low'] < vob_zones['high'] and df.iloc[-1]['high'] > vob_zones['low']

    conditions_true = sum([tight_range, small_bodies, ema_flat, inside_vob])
    return conditions_true >= 2

# ================================================================
# 2. SUPPORT & RESISTANCE CALCULATION
# ================================================================
def find_swing_levels(df, lookback=5):
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df)-lookback):
        high_slice = df['high'].iloc[i-lookback:i+lookback+1]
        low_slice = df['low'].iloc[i-lookback:i+lookback+1]

        if df['high'].iloc[i] == high_slice.max():
            swing_highs.append((df['time'].iloc[i], df['high'].iloc[i]))
        if df['low'].iloc[i] == low_slice.min():
            swing_lows.append((df['time'].iloc[i], df['low'].iloc[i]))

    return swing_highs, swing_lows

def calculate_sr(df):
    swing_highs, swing_lows = find_swing_levels(df)
    last_highs = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:5]
    last_lows = sorted(swing_lows, key=lambda x: x[0], reverse=True)[:5]
    resistance_levels = [h[1] for h in last_highs]
    support_levels = [l[1] for l in last_lows]
    return support_levels, resistance_levels

# ================================================================
# 3. TREND DETECTION
# ================================================================
def detect_trend(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema100"] = df["close"].ewm(span=100).mean()
    last = df.iloc[-1]

    if last["close"] > last["ema20"] > last["ema50"]:
        return "Strong Uptrend"
    if last["close"] < last["ema20"] < last["ema50"]:
        return "Strong Downtrend"
    if last["ema20"] > last["ema50"] > last["ema100"]:
        return "Uptrend"
    if last["ema20"] < last["ema50"] < last["ema100"]:
        return "Downtrend"
    return "Sideways"

# ================================================================
# 4. MULTI-TIMEFRAME ANALYSIS
# ================================================================
def process_timeframe(tf_name, df):
    trend = detect_trend(df)
    supports, resistances = calculate_sr(df)
    return {
        "timeframe": tf_name,
        "trend": trend,
        "supports": supports,
        "resistances": resistances
    }

def multi_tf_analysis(data_dict):
    results = {}
    for tf_name, df in data_dict.items():
        results[tf_name] = process_timeframe(tf_name, df)
    return results

# ================================================================
# 5. BREAKOUT / BREAKDOWN / REVERSAL DETECTION
# ================================================================
def is_breakout(df, resistances):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if not resistances:
        return False
    res = min(resistances, key=lambda x: abs(x - last['close']))
    breakout = last['close'] > res and last['close'] > last['open'] and last['volume'] > prev['volume']
    return breakout

def is_breakdown(df, supports):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if not supports:
        return False
    sup = min(supports, key=lambda x: abs(x - last['close']))
    breakdown = last['close'] < sup and last['close'] < last['open'] and last['volume'] > prev['volume']
    return breakdown

def is_reversal(df, supports, resistances, vob_zones):
    last = df.iloc[-1]

    # Bullish Reversal
    for s in supports:
        if abs(last['low'] - s) < (last['close'] * 0.0015) and last['close'] > last['open']:
            if not (last['low'] < vob_zones['high'] and last['high'] > vob_zones['low']):
                return "BULL_REVERSAL"

    # Bearish Reversal
    for r in resistances:
        if abs(last['high'] - r) < (last['close'] * 0.0015) and last['close'] < last['open']:
            if not (last['low'] < vob_zones['high'] and last['high'] > vob_zones['low']):
                return "BEAR_REVERSAL"
    return None

# ================================================================
# 6. MASTER SIGNAL ENGINE
# ================================================================
def generate_signal(df, vob_zones, app_bias, tf_analysis, bot_token, chat_id):
    if is_consolidation(df, vob_zones):
        print("NO SIGNAL — Market in consolidation")
        return None

    current_tf = tf_analysis["5m"]
    trend = current_tf["trend"]
    supports = current_tf["supports"]
    resistances = current_tf["resistances"]

    # 1. Trend + Bias Signal
    if app_bias == "BULL" and "Up" in trend:
        send_telegram("📈 BUY SIGNAL — Trend + Bias Confirmed", bot_token, chat_id)
        return "BUY"
    if app_bias == "BEAR" and "Down" in trend:
        send_telegram("📉 SELL SIGNAL — Trend + Bias Confirmed", bot_token, chat_id)
        return "SELL"

    # 2. Breakout / Breakdown
    if is_breakout(df, resistances):
        send_telegram("🚀 BREAKOUT BUY — Resistance Cleared", bot_token, chat_id)
        return "BREAKOUT_BUY"
    if is_breakdown(df, supports):
        send_telegram("⚠️ BREAKDOWN SELL — Support Broken", bot_token, chat_id)
        return "BREAKDOWN_SELL"

    # 3. Reversal
    rev = is_reversal(df, supports, resistances, vob_zones)
    if rev == "BULL_REVERSAL":
        send_telegram("🔄 BULLISH REVERSAL — Support Rejected", bot_token, chat_id)
        return "REVERSAL_BUY"
    if rev == "BEAR_REVERSAL":
        send_telegram("🔄 BEARISH REVERSAL — Resistance Rejected", bot_token, chat_id)
        return "REVERSAL_SELL"

    print("NO SIGNAL — No breakout/breakdown/reversal")
    return None

# ================================================================
# 7. TELEGRAM ALERTS
# ================================================================
def send_telegram(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except:
        pass


# =============================================
# GAMMA SEQUENCE ANALYZER
# =============================================

class GammaSequenceAnalyzer:
    """Comprehensive Gamma Sequence Analysis for Institutional Bias Detection"""
    
    def __init__(self):
        self.gamma_levels = {
            'EXTREME_POSITIVE': {'threshold': 10000, 'bias': 'STRONG_BULLISH', 'score': 100},
            'HIGH_POSITIVE': {'threshold': 5000, 'bias': 'BULLISH', 'score': 75},
            'MODERATE_POSITIVE': {'threshold': 1000, 'bias': 'MILD_BULLISH', 'score': 50},
            'NEUTRAL': {'threshold': -1000, 'bias': 'NEUTRAL', 'score': 0},
            'MODERATE_NEGATIVE': {'threshold': -5000, 'bias': 'MILD_BEARISH', 'score': -50},
            'HIGH_NEGATIVE': {'threshold': -10000, 'bias': 'BEARISH', 'score': -75},
            'EXTREME_NEGATIVE': {'threshold': -20000, 'bias': 'STRONG_BEARISH', 'score': -100}
        }
    
    def calculate_gamma_exposure(self, df_chain: pd.DataFrame) -> pd.DataFrame:
        """Calculate Gamma exposure for all strikes"""
        df = df_chain.copy()
        
        # Calculate Gamma exposure
        df['gamma_exposure_ce'] = df['Gamma_CE'] * df['openInterest_CE'] * 100  # Multiply by 100 for contract size
        df['gamma_exposure_pe'] = df['Gamma_PE'] * df['openInterest_PE'] * 100
        df['net_gamma_exposure'] = df['gamma_exposure_ce'] + df['gamma_exposure_pe']
        
        # Calculate Gamma profile
        df['gamma_profile'] = df['net_gamma_exposure'].apply(self._get_gamma_profile)
        
        return df
    
    def _get_gamma_profile(self, gamma_exposure: float) -> str:
        """Get Gamma profile based on exposure level"""
        for level, config in self.gamma_levels.items():
            if gamma_exposure >= config['threshold']:
                return level
        return 'EXTREME_NEGATIVE'
    
    def analyze_gamma_sequence_bias(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Comprehensive Gamma sequence bias analysis"""
        try:
            df_with_gamma = self.calculate_gamma_exposure(df_chain)
            
            # Analyze different strike zones
            analysis = {
                'total_gamma_exposure': df_with_gamma['net_gamma_exposure'].sum(),
                'gamma_bias': self._calculate_overall_gamma_bias(df_with_gamma),
                'zones': self._analyze_gamma_zones(df_with_gamma, spot_price),
                'sequence': self._analyze_gamma_sequence(df_with_gamma),
                'walls': self._find_gamma_walls(df_with_gamma),
                'profile': self._get_gamma_profile(df_with_gamma['net_gamma_exposure'].sum())
            }
            
            # Calculate comprehensive Gamma score
            analysis['gamma_score'] = self._calculate_gamma_score(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error in Gamma sequence analysis: {e}")
            return {'gamma_bias': 'NEUTRAL', 'gamma_score': 0, 'error': str(e)}
    
    def _calculate_overall_gamma_bias(self, df: pd.DataFrame) -> str:
        """Calculate overall Gamma bias"""
        total_gamma = df['net_gamma_exposure'].sum()
        
        if total_gamma > 10000:
            return "STRONG_BULLISH"
        elif total_gamma > 5000:
            return "BULLISH"
        elif total_gamma > 1000:
            return "MILD_BULLISH"
        elif total_gamma < -10000:
            return "STRONG_BEARISH"
        elif total_gamma < -5000:
            return "BEARISH"
        elif total_gamma < -1000:
            return "MILD_BEARISH"
        else:
            return "NEUTRAL"
    
    def _analyze_gamma_zones(self, df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Analyze Gamma across different price zones"""
        strike_diff = df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]
        
        zones = {
            'itm_puts': df[df['strikePrice'] < spot_price - strike_diff].copy(),
            'near_otm_puts': df[(df['strikePrice'] >= spot_price - strike_diff) & (df['strikePrice'] < spot_price)].copy(),
            'atm': df[abs(df['strikePrice'] - spot_price) <= strike_diff].copy(),
            'near_otm_calls': df[(df['strikePrice'] > spot_price) & (df['strikePrice'] <= spot_price + strike_diff)].copy(),
            'otm_calls': df[df['strikePrice'] > spot_price + strike_diff].copy()
        }
        
        zone_analysis = {}
        for zone_name, zone_data in zones.items():
            if not zone_data.empty:
                zone_analysis[zone_name] = {
                    'gamma_exposure': zone_data['net_gamma_exposure'].sum(),
                    'bias': self._calculate_overall_gamma_bias(zone_data),
                    'strike_range': f"{zone_data['strikePrice'].min():.0f}-{zone_data['strikePrice'].max():.0f}"
                }
        
        return zone_analysis
    
    def _analyze_gamma_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Gamma sequence patterns"""
        # Sort by strike price
        df_sorted = df.sort_values('strikePrice')
        
        # Calculate Gamma changes between strikes
        df_sorted['gamma_change'] = df_sorted['net_gamma_exposure'].diff()
        
        # Identify sequences
        positive_sequences = []
        negative_sequences = []
        current_sequence = []
        
        for _, row in df_sorted.iterrows():
            if not current_sequence:
                current_sequence.append(row)
                continue
                
            current_gamma = row['net_gamma_exposure']
            prev_gamma = current_sequence[-1]['net_gamma_exposure']
            
            if (current_gamma >= 0 and prev_gamma >= 0) or (current_gamma < 0 and prev_gamma < 0):
                current_sequence.append(row)
            else:
                if current_sequence:
                    seq_gamma = sum([x['net_gamma_exposure'] for x in current_sequence])
                    if seq_gamma >= 0:
                        positive_sequences.append({
                            'strikes': [x['strikePrice'] for x in current_sequence],
                            'total_gamma': seq_gamma,
                            'length': len(current_sequence)
                        })
                    else:
                        negative_sequences.append({
                            'strikes': [x['strikePrice'] for x in current_sequence],
                            'total_gamma': seq_gamma,
                            'length': len(current_sequence)
                        })
                current_sequence = [row]
        
        return {
            'positive_sequences': positive_sequences,
            'negative_sequences': negative_sequences,
            'longest_positive_seq': max([seq['length'] for seq in positive_sequences]) if positive_sequences else 0,
            'longest_negative_seq': max([seq['length'] for seq in negative_sequences]) if negative_sequences else 0
        }
    
    def _find_gamma_walls(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find significant Gamma walls/resistance levels"""
        # Find local maxima/minima in Gamma exposure
        gamma_walls = []
        
        for i in range(1, len(df) - 1):
            current_gamma = df.iloc[i]['net_gamma_exposure']
            prev_gamma = df.iloc[i-1]['net_gamma_exposure']
            next_gamma = df.iloc[i+1]['net_gamma_exposure']
            
            # Gamma wall (local maximum with high positive Gamma)
            if current_gamma > prev_gamma and current_gamma > next_gamma and current_gamma > 5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'RESISTANCE',
                    'strength': 'STRONG' if current_gamma > 10000 else 'MODERATE'
                })
            
            # Gamma vacuum (local minimum with high negative Gamma)
            elif current_gamma < prev_gamma and current_gamma < next_gamma and current_gamma < -5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'SUPPORT',
                    'strength': 'STRONG' if current_gamma < -10000 else 'MODERATE'
                })
        
        return sorted(gamma_walls, key=lambda x: abs(x['gamma_exposure']), reverse=True)[:5]  # Top 5
    
    def _calculate_gamma_score(self, analysis: Dict) -> float:
        """Calculate comprehensive Gamma score from -100 to 100"""
        base_score = self.gamma_levels.get(analysis['profile'], {}).get('score', 0)
        
        # Adjust score based on sequence analysis
        seq_analysis = analysis.get('sequence', {})
        pos_seqs = len(seq_analysis.get('positive_sequences', []))
        neg_seqs = len(seq_analysis.get('negative_sequences', []))
        
        if pos_seqs > neg_seqs:
            sequence_bonus = 10
        elif neg_seqs > pos_seqs:
            sequence_bonus = -10
        else:
            sequence_bonus = 0
        
        # Adjust score based on Gamma walls
        walls = analysis.get('walls', [])
        resistance_walls = len([w for w in walls if w['type'] == 'RESISTANCE'])
        support_walls = len([w for w in walls if w['type'] == 'SUPPORT'])
        
        if resistance_walls > support_walls:
            walls_penalty = -5
        elif support_walls > resistance_walls:
            walls_penalty = 5
        else:
            walls_penalty = 0
        
        final_score = base_score + sequence_bonus + walls_penalty
        return max(-100, min(100, final_score))


# FIX 3: Add caching for Gamma analysis
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def cached_gamma_analysis(_analyzer, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Cached Gamma analysis to improve performance"""
    return _analyzer.analyze_gamma_sequence_bias(df_chain, spot_price)


# =============================================
# INSTITUTIONAL OI ADVANCED ANALYZER
# =============================================

class InstitutionalOIAdvanced:
    """Advanced Institutional OI Analysis with Gamma Sequencing"""
    
    def __init__(self):
        self.master_table_rules = {
            'CALL': {
                'Winding_Up_Price_Up': {'bias': 'BEARISH', 'institution_move': 'Selling/Writing', 'confidence': 'HIGH'},
                'Winding_Up_Price_Down': {'bias': 'BEARISH', 'institution_move': 'Sellers Dominating', 'confidence': 'HIGH'},
                'Unwinding_Down_Price_Up': {'bias': 'BULLISH', 'institution_move': 'Short Covering', 'confidence': 'MEDIUM'},
                'Unwinding_Down_Price_Down': {'bias': 'MILD_BEARISH', 'institution_move': 'Longs Exiting', 'confidence': 'LOW'}
            },
            'PUT': {
                'Winding_Up_Price_Down': {'bias': 'BULLISH', 'institution_move': 'Selling/Writing', 'confidence': 'HIGH'},
                'Winding_Up_Price_Up': {'bias': 'BULLISH', 'institution_move': 'Sellers Dominating', 'confidence': 'HIGH'},
                'Unwinding_Down_Price_Down': {'bias': 'BEARISH', 'institution_move': 'Short Covering', 'confidence': 'MEDIUM'},
                'Unwinding_Down_Price_Up': {'bias': 'MILD_BULLISH', 'institution_move': 'Longs Exiting', 'confidence': 'LOW'}
            }
        }
        self.gamma_analyzer = GammaSequenceAnalyzer()
    
    def calculate_institutional_oi_bias(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive bias from institutional OI patterns"""
        if not patterns:
            return {
                'overall_bias': 'NEUTRAL',
                'bias_score': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'neutral_patterns': 0,
                'confidence': 'LOW',
                'dominant_move': 'No patterns detected'
            }
        
        # Count patterns by bias
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_score = 0
        pattern_count = len(patterns)
        
        # Institution moves tracking
        institution_moves = {}
        
        for pattern in patterns:
            bias = pattern.get('bias', 'NEUTRAL')
            confidence = pattern.get('confidence', 'LOW')
            institution_move = pattern.get('institution_move', 'Unknown')
            
            # Track institution moves
            if institution_move not in institution_moves:
                institution_moves[institution_move] = 0
            institution_moves[institution_move] += 1
            
            # Calculate score based on bias and confidence
            if 'BULL' in bias.upper():
                bullish_count += 1
                score = 1.0
            elif 'BEAR' in bias.upper():
                bearish_count += 1
                score = -1.0
            else:
                neutral_count += 1
                score = 0.0
            
            # Apply confidence multiplier
            confidence_multipliers = {
                'VERY_HIGH': 1.5,
                'HIGH': 1.2,
                'MEDIUM': 1.0,
                'LOW': 0.7
            }
            multiplier = confidence_multipliers.get(confidence, 1.0)
            total_score += score * multiplier
        
        # Calculate average score
        avg_score = total_score / pattern_count if pattern_count > 0 else 0
        
        # Determine overall bias
        if avg_score > 0.2:
            overall_bias = "BULLISH"
            bias_score = min(100, avg_score * 100)
        elif avg_score < -0.2:
            overall_bias = "BEARISH" 
            bias_score = max(-100, avg_score * 100)
        else:
            overall_bias = "NEUTRAL"
            bias_score = 0
        
        # Determine confidence level
        bias_strength = max(bullish_count, bearish_count) / pattern_count if pattern_count > 0 else 0
        if bias_strength >= 0.7:
            confidence_level = "VERY_HIGH"
        elif bias_strength >= 0.6:
            confidence_level = "HIGH"
        elif bias_strength >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Find dominant institution move
        dominant_move = max(institution_moves.items(), key=lambda x: x[1])[0] if institution_moves else "No dominant move"
        
        return {
            'overall_bias': overall_bias,
            'bias_score': bias_score,
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'neutral_patterns': neutral_count,
            'total_patterns': pattern_count,
            'confidence': confidence_level,
            'dominant_move': dominant_move,
            'institution_moves': institution_moves,
            'bias_strength': bias_strength * 100
        }

    def analyze_institutional_oi_pattern(self, option_type: str, oi_change: float, price_change: float, 
                                       volume: float, iv_change: float, bid_ask_ratio: float) -> Dict:
        """Analyze institutional OI patterns based on master table rules"""
        
        # Determine OI action
        oi_action = "Winding_Up" if oi_change > 0 else "Unwinding_Down"
        
        # Determine price action
        price_action = "Price_Up" if price_change > 0 else "Price_Down"
        
        # Determine pattern key
        pattern_key = f"{oi_action}_{price_action}"
        
        # Get base pattern
        base_pattern = self.master_table_rules.get(option_type, {}).get(pattern_key, {})
        
        if not base_pattern:
            return {'bias': 'NEUTRAL', 'confidence': 'LOW', 'pattern': 'Unknown'}
        
        # Enhance with volume and IV analysis
        volume_signal = "High" if volume > 1000 else "Low"
        iv_signal = "Rising" if iv_change > 0 else "Falling"
        
        # Bid/Ask analysis
        liquidity_signal = "Bid_Heavy" if bid_ask_ratio > 1.2 else "Ask_Heavy" if bid_ask_ratio < 0.8 else "Balanced"
        
        # Adjust confidence based on volume and IV
        confidence = base_pattern['confidence']
        if volume_signal == "High" and abs(iv_change) > 1.0:
            confidence = "VERY_HIGH"
        
        return {
            'option_type': option_type,
            'bias': base_pattern['bias'],
            'institution_move': base_pattern['institution_move'],
            'confidence': confidence,
            'pattern': pattern_key,
            'volume_signal': volume_signal,
            'iv_signal': iv_signal,
            'liquidity_signal': liquidity_signal,
            'oi_change': oi_change,
            'price_change': price_change,
            'volume': volume
        }
    
    def analyze_atm_institutional_footprint(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Comprehensive institutional footprint analysis for ATM ±2 strikes"""
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            # Get ATM ±2 strikes
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_range = strike_diff * 2
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= atm_range].copy()
            
            if atm_strikes.empty:
                return {
                    'overall_bias': 'NEUTRAL', 
                    'score': 0, 
                    'patterns': [],
                    'institutional_bias_analysis': {
                        'overall_bias': 'NEUTRAL',
                        'bias_score': 0,
                        'bullish_patterns': 0,
                        'bearish_patterns': 0,
                        'neutral_patterns': 0,
                        'confidence': 'LOW',
                        'dominant_move': 'No patterns detected'
                    }
                }
            
            patterns = []
            total_score = 0
            pattern_count = 0
            
            for _, strike_data in atm_strikes.iterrows():
                strike = strike_data['strikePrice']
                
                # Analyze CALL side
                if pd.notna(strike_data.get('change_oi_ce')):
                    ce_pattern = self.analyze_institutional_oi_pattern(
                        option_type='CALL',
                        oi_change=strike_data.get('change_oi_ce', 0),
                        price_change=strike_data.get('ltp_ce', 0) - strike_data.get('previousClose_CE', strike_data.get('ltp_ce', 0)),
                        volume=strike_data.get('volume_ce', 0),
                        iv_change=strike_data.get('iv_ce', 0) - strike_data.get('previousIV_CE', strike_data.get('iv_ce', 0)),
                        bid_ask_ratio=strike_data.get('bid_ce', 1) / max(1, strike_data.get('ask_ce', 1))
                    )
                    ce_pattern['strike'] = strike
                    ce_pattern['option_type'] = 'CE'
                    patterns.append(ce_pattern)
                    
                    # Convert bias to score
                    bias_score = self._bias_to_score(ce_pattern['bias'], ce_pattern['confidence'])
                    total_score += bias_score
                    pattern_count += 1
                
                # Analyze PUT side
                if pd.notna(strike_data.get('change_oi_pe')):
                    pe_pattern = self.analyze_institutional_oi_pattern(
                        option_type='PUT',
                        oi_change=strike_data.get('change_oi_pe', 0),
                        price_change=strike_data.get('ltp_pe', 0) - strike_data.get('previousClose_PE', strike_data.get('ltp_pe', 0)),
                        volume=strike_data.get('volume_pe', 0),
                        iv_change=strike_data.get('iv_pe', 0) - strike_data.get('previousIV_PE', strike_data.get('iv_pe', 0)),
                        bid_ask_ratio=strike_data.get('bid_pe', 1) / max(1, strike_data.get('ask_pe', 1))
                    )
                    pe_pattern['strike'] = strike
                    pe_pattern['option_type'] = 'PE'
                    patterns.append(pe_pattern)
                    
                    # Convert bias to score
                    bias_score = self._bias_to_score(pe_pattern['bias'], pe_pattern['confidence'])
                    total_score += bias_score
                    pattern_count += 1
            
            # Calculate overall bias
            if pattern_count > 0:
                avg_score = total_score / pattern_count
                if avg_score > 0.2:
                    overall_bias = "BULLISH"
                elif avg_score < -0.2:
                    overall_bias = "BEARISH"
                else:
                    overall_bias = "NEUTRAL"
            else:
                overall_bias = "NEUTRAL"
                avg_score = 0
            
            # Calculate comprehensive institutional bias
            institutional_bias_analysis = self.calculate_institutional_oi_bias(patterns)
            
            # Add Gamma sequencing analysis with caching
            gamma_analysis = cached_gamma_analysis(self.gamma_analyzer, df_chain, spot_price)
            
            return {
                'overall_bias': overall_bias,
                'score': avg_score * 100,  # Convert to percentage
                'patterns': patterns,
                'institutional_bias_analysis': institutional_bias_analysis,
                'gamma_analysis': gamma_analysis,
                'strikes_analyzed': len(atm_strikes),
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            print(f"Error in institutional footprint analysis: {e}")
            return {
                'overall_bias': 'NEUTRAL', 
                'score': 0, 
                'patterns': [],
                'institutional_bias_analysis': {
                    'overall_bias': 'NEUTRAL',
                    'bias_score': 0,
                    'bullish_patterns': 0,
                    'bearish_patterns': 0,
                    'neutral_patterns': 0,
                    'confidence': 'LOW',
                    'dominant_move': 'Analysis error'
                }
            }
    
    def _bias_to_score(self, bias: str, confidence: str) -> float:
        """Convert bias and confidence to numerical score"""
        bias_scores = {
            'BULLISH': 1.0,
            'MILD_BULLISH': 0.5,
            'NEUTRAL': 0.0,
            'MILD_BEARISH': -0.5,
            'BEARISH': -1.0
        }
        
        confidence_multipliers = {
            'VERY_HIGH': 1.5,
            'HIGH': 1.2,
            'MEDIUM': 1.0,
            'LOW': 0.7
        }
        
        base_score = bias_scores.get(bias, 0.0)
        multiplier = confidence_multipliers.get(confidence, 1.0)
        
        return base_score * multiplier


# FIX 4: Add column normalization function
def normalize_chain_columns(df_chain: pd.DataFrame) -> pd.DataFrame:
    """Normalize option chain column names to handle different API formats"""
    df = df_chain.copy()
    
    # Define column mapping for different API formats
    column_mapping = {
        # Standardize to our expected column names
        'changeinOpenInterest_CE': 'change_oi_ce',
        'changeinOpenInterest_PE': 'change_oi_pe',
        'openInterest_CE': 'oi_ce', 
        'openInterest_PE': 'oi_pe',
        'impliedVolatility_CE': 'iv_ce',
        'impliedVolatility_PE': 'iv_pe',
        'lastPrice_CE': 'ltp_ce',
        'lastPrice_PE': 'ltp_pe',
        'totalTradedVolume_CE': 'volume_ce',
        'totalTradedVolume_PE': 'volume_pe',
        'bidQty_CE': 'bid_ce',
        'askQty_CE': 'ask_ce',
        'bidQty_PE': 'bid_pe', 
        'askQty_PE': 'ask_pe',
        
        # Alternative column names (common in different APIs)
        'CE_changeinOpenInterest': 'change_oi_ce',
        'PE_changeinOpenInterest': 'change_oi_pe',
        'CE_openInterest': 'oi_ce',
        'PE_openInterest': 'oi_pe',
        'CE_impliedVolatility': 'iv_ce',
        'PE_impliedVolatility': 'iv_pe',
        'CE_lastPrice': 'ltp_ce',
        'PE_lastPrice': 'ltp_pe',
        'CE_totalTradedVolume': 'volume_ce',
        'PE_totalTradedVolume': 'volume_pe',
        'CE_bidQty': 'bid_ce',
        'CE_askQty': 'ask_ce',
        'PE_bidQty': 'bid_pe',
        'PE_askQty': 'ask_pe',
        
        # Very short column names
        'chg_oi_ce': 'change_oi_ce',
        'chg_oi_pe': 'change_oi_pe',
        'oi_ce': 'oi_ce',
        'oi_pe': 'oi_pe',
        'iv_ce': 'iv_ce', 
        'iv_pe': 'iv_pe',
        'ltp_ce': 'ltp_ce',
        'ltp_pe': 'ltp_pe',
        'vol_ce': 'volume_ce',
        'vol_pe': 'volume_pe'
    }
    
    # Rename columns that exist in the dataframe
    existing_columns = set(df.columns)
    rename_dict = {}
    
    for old_col, new_col in column_mapping.items():
        if old_col in existing_columns:
            rename_dict[old_col] = new_col
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Create missing columns with default values
    required_columns = ['change_oi_ce', 'change_oi_pe', 'oi_ce', 'oi_pe', 'iv_ce', 'iv_pe', 
                       'ltp_ce', 'ltp_pe', 'volume_ce', 'volume_pe', 'bid_ce', 'ask_ce', 'bid_pe', 'ask_pe']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns
    
    # Calculate Greeks if not present (simplified calculation)
    if 'Gamma_CE' not in df.columns:
        df['Gamma_CE'] = 0.01  # Simplified gamma value
        df['Gamma_PE'] = 0.01
    
    return df


# =============================================
# BREAKOUT & REVERSAL CONFIRMATION ANALYZER
# =============================================

class BreakoutReversalAnalyzer:
    """Institutional Breakout & Reversal Confirmation System"""
    
    def __init__(self):
        self.breakout_threshold = 0.6  # 60% confidence for real breakout
        self.reversal_threshold = 0.7   # 70% confidence for reversal
        
    def analyze_breakout_confirmation(self, df_chain: pd.DataFrame, spot_price: float, 
                                    price_change: float, volume_change: float) -> Dict[str, Any]:
        """
        Comprehensive breakout confirmation analysis
        Returns confidence score 0-100 for breakout validity
        """
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            # Get ATM ±2 strikes for analysis
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= strike_diff * 2].copy()
            
            if atm_strikes.empty:
                return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
            
            # Determine breakout direction
            is_upside_breakout = price_change > 0
            direction = "UP" if is_upside_breakout else "DOWN"
            
            signals = []
            total_score = 0
            max_score = 0
            
            # 1. OI Change Analysis (25 points)
            oi_analysis = self._analyze_oi_pattern(atm_strikes, is_upside_breakout)
            signals.extend(oi_analysis['signals'])
            total_score += oi_analysis['score']
            max_score += 25
            
            # 2. Price vs OI Conflict (20 points)
            conflict_analysis = self._analyze_price_oi_conflict(atm_strikes, is_upside_breakout)
            signals.extend(conflict_analysis['signals'])
            total_score += conflict_analysis['score']
            max_score += 20
            
            # 3. IV Behavior Analysis (15 points)
            iv_analysis = self._analyze_iv_behavior(atm_strikes, is_upside_breakout)
            signals.extend(iv_analysis['signals'])
            total_score += iv_analysis['score']
            max_score += 15
            
            # 4. PCR Trend Analysis (15 points)
            pcr_analysis = self._analyze_pcr_trend(df_chain, is_upside_breakout)
            signals.extend(pcr_analysis['signals'])
            total_score += pcr_analysis['score']
            max_score += 15
            
            # 5. Max Pain Movement (10 points)
            max_pain_analysis = self._analyze_max_pain_movement(df_chain, spot_price, is_upside_breakout)
            signals.extend(max_pain_analysis['signals'])
            total_score += max_pain_analysis['score']
            max_score += 10
            
            # 6. Strike OI Wall Breakdown (15 points)
            wall_analysis = self._analyze_oi_wall_breakdown(atm_strikes, is_upside_breakout)
            signals.extend(wall_analysis['signals'])
            total_score += wall_analysis['score']
            max_score += 15
            
            # Calculate final confidence
            breakout_confidence = (total_score / max_score) * 100 if max_score > 0 else 0
            
            # Determine breakout type
            if breakout_confidence >= 60:
                breakout_type = "REAL_BREAKOUT"
            elif breakout_confidence >= 30:
                breakout_type = "WEAK_BREAKOUT"
            else:
                breakout_type = "FAKE_BREAKOUT"
            
            return {
                'breakout_confidence': breakout_confidence,
                'direction': direction,
                'breakout_type': breakout_type,
                'signals': signals,
                'total_score': total_score,
                'max_score': max_score
            }
            
        except Exception as e:
            print(f"Error in breakout analysis: {e}")
            return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
    
    def analyze_reversal_confirmation(self, df_chain: pd.DataFrame, spot_price: float,
                                   price_action: Dict) -> Dict[str, Any]:
        """
        Comprehensive reversal confirmation analysis
        Returns confidence score 0-100 for reversal validity
        """
        try:
            # FIX 4: Normalize column names first
            df_chain = normalize_chain_columns(df_chain)
            
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= strike_diff * 2].copy()
            
            if atm_strikes.empty:
                return {'reversal_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
            
            # Determine if we're looking for top or bottom reversal
            is_top_reversal = price_action.get('has_upper_wick', False) or price_action.get('is_overbought', False)
            direction = "TOP_REVERSAL" if is_top_reversal else "BOTTOM_REVERSAL"
            
            signals = []
            total_score = 0
            max_score = 0
            
            # 1. OI Divergence Analysis (30 points)
            divergence_analysis = self._analyze_oi_divergence(atm_strikes, is_top_reversal)
            signals.extend(divergence_analysis['signals'])
            total_score += divergence_analysis['score']
            max_score += 30
            
            # 2. IV Crash Detection (20 points)
            iv_crash_analysis = self._analyze_iv_crash(atm_strikes)
            signals.extend(iv_crash_analysis['signals'])
            total_score += iv_crash_analysis['score']
            max_score += 20
            
            # 3. Writer Defense Analysis (25 points)
            defense_analysis = self._analyze_writer_defense(df_chain, spot_price, is_top_reversal)
            signals.extend(defense_analysis['signals'])
            total_score += defense_analysis['score']
            max_score += 25
            
            # 4. Opposite OI Build (15 points)
            opposite_oi_analysis = self._analyze_opposite_oi_build(atm_strikes, is_top_reversal)
            signals.extend(opposite_oi_analysis['signals'])
            total_score += opposite_oi_analysis['score']
            max_score += 15
            
            # 5. PCR Extremes (10 points)
            pcr_extreme_analysis = self._analyze_pcr_extremes(df_chain)
            signals.extend(pcr_extreme_analysis['signals'])
            total_score += pcr_extreme_analysis['score']
            max_score += 10
            
            # Calculate final confidence
            reversal_confidence = (total_score / max_score) * 100 if max_score > 0 else 0
            
            # Determine reversal strength
            if reversal_confidence >= 70:
                reversal_type = "STRONG_REVERSAL"
            elif reversal_confidence >= 50:
                reversal_type = "MODERATE_REVERSAL"
            else:
                reversal_type = "WEAK_REVERSAL"
            
            return {
                'reversal_confidence': reversal_confidence,
                'direction': direction,
                'reversal_type': reversal_type,
                'signals': signals,
                'total_score': total_score,
                'max_score': max_score
            }
            
        except Exception as e:
            print(f"Error in reversal analysis: {e}")
            return {'reversal_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
    
    def _analyze_oi_pattern(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze OI patterns for breakout confirmation"""
        signals = []
        score = 0
        
        # Calculate total OI changes
        total_ce_oi_change = atm_strikes['change_oi_ce'].sum()
        total_pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_upside:
            # Upside breakout: CE OI should decrease, PE OI should increase
            if total_ce_oi_change < 0:
                signals.append("✅ CE OI decreasing (call sellers running)")
                score += 10
            else:
                signals.append("❌ CE OI increasing (call writers active)")
            
            if total_pe_oi_change > 0:
                signals.append("✅ PE OI increasing (put writers entering)")
                score += 15
            else:
                signals.append("❌ PE OI decreasing (no put writing)")
        else:
            # Downside breakout: CE OI should increase, PE OI should decrease
            if total_ce_oi_change > 0:
                signals.append("✅ CE OI increasing (call writers attacking)")
                score += 10
            else:
                signals.append("❌ CE OI decreasing (no call writing)")
            
            if total_pe_oi_change < 0:
                signals.append("✅ PE OI decreasing (put sellers exiting)")
                score += 15
            else:
                signals.append("❌ PE OI increasing (put writers defending)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_price_oi_conflict(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze price vs OI conflict"""
        signals = []
        score = 0
        
        # Get dominant strike
        dominant_strike = atm_strikes.loc[atm_strikes['oi_ce'].idxmax()] if is_upside else atm_strikes.loc[atm_strikes['oi_pe'].idxmax()]
        
        if is_upside:
            # Clean upside: Price ↑, CE OI ↓
            if dominant_strike['change_oi_ce'] < 0:
                signals.append("✅ Clean breakout: Price ↑ + CE OI ↓ (short covering)")
                score += 20
            else:
                signals.append("❌ Fake breakout: Price ↑ + CE OI ↑ (sellers building wall)")
        else:
            # Clean downside: Price ↓, PE OI ↓
            if dominant_strike['change_oi_pe'] < 0:
                signals.append("✅ Clean breakdown: Price ↓ + PE OI ↓ (put covering)")
                score += 20
            else:
                signals.append("❌ Fake breakdown: Price ↓ + PE OI ↑ (put writers defending)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_iv_behavior(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze IV behavior for breakout confirmation"""
        signals = []
        score = 0
        
        avg_ce_iv = atm_strikes['iv_ce'].mean()
        avg_pe_iv = atm_strikes['iv_pe'].mean()
        
        if is_upside:
            # Upside: CE IV should rise slightly, PE IV stable/fall
            if avg_ce_iv > avg_pe_iv:
                signals.append("✅ CE IV > PE IV (upside momentum)")
                score += 10
            else:
                signals.append("❌ CE IV <= PE IV (weak upside)")
            
            if avg_pe_iv < 20:  # Low PE IV indicates no put buying pressure
                signals.append("✅ Low PE IV (no put hedging)")
                score += 5
        else:
            # Downside: PE IV should rise, CE IV stable/fall
            if avg_pe_iv > avg_ce_iv:
                signals.append("✅ PE IV > CE IV (downside momentum)")
                score += 10
            else:
                signals.append("❌ PE IV <= CE IV (weak downside)")
            
            if avg_ce_iv < 20:  # Low CE IV indicates no call buying pressure
                signals.append("✅ Low CE IV (no call hedging)")
                score += 5
        
        return {'signals': signals, 'score': score}
    
    def _analyze_pcr_trend(self, df_chain: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze PCR trend for breakout direction"""
        signals = []
        score = 0
        
        total_ce_oi = df_chain['oi_ce'].sum()
        total_pe_oi = df_chain['oi_pe'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if is_upside:
            if pcr > 0.8:
                signals.append(f"✅ PCR {pcr:.2f} > 0.8 (bullish bias)")
                score += 10
            elif pcr > 0.6:
                signals.append(f"⚠️ PCR {pcr:.2f} neutral")
                score += 5
            else:
                signals.append(f"❌ PCR {pcr:.2f} < 0.6 (bearish bias)")
        else:
            if pcr < 0.7:
                signals.append(f"✅ PCR {pcr:.2f} < 0.7 (bearish bias)")
                score += 10
            elif pcr < 0.9:
                signals.append(f"⚠️ PCR {pcr:.2f} neutral")
                score += 5
            else:
                signals.append(f"❌ PCR {pcr:.2f} > 0.9 (bullish bias)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_max_pain_movement(self, df_chain: pd.DataFrame, spot_price: float, is_upside: bool) -> Dict:
        """Analyze max pain movement"""
        signals = []
        score = 0
        
        # Simple max pain approximation (you can enhance this)
        max_ce_oi_strike = df_chain.loc[df_chain['oi_ce'].idxmax()]['strikePrice']
        max_pe_oi_strike = df_chain.loc[df_chain['oi_pe'].idxmax()]['strikePrice']
        
        if is_upside:
            if max_pe_oi_strike > spot_price:
                signals.append(f"✅ Max PE OI at ₹{max_pe_oi_strike:.0f} (above spot)")
                score += 10
            else:
                signals.append(f"❌ Max PE OI at ₹{max_pe_oi_strike:.0f} (below spot)")
        else:
            if max_ce_oi_strike < spot_price:
                signals.append(f"✅ Max CE OI at ₹{max_ce_oi_strike:.0f} (below spot)")
                score += 10
            else:
                signals.append(f"❌ Max CE OI at ₹{max_ce_oi_strike:.0f} (above spot)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_oi_wall_breakdown(self, atm_strikes: pd.DataFrame, is_upside: bool) -> Dict:
        """Analyze OI wall breakdown"""
        signals = []
        score = 0
        
        if is_upside:
            # Upside: CE OI should unwind, PE OI should build
            ce_oi_change = atm_strikes['change_oi_ce'].sum()
            pe_oi_change = atm_strikes['change_oi_pe'].sum()
            
            if ce_oi_change < 0:
                signals.append("✅ CE OI wall breaking (unwinding)")
                score += 10
            if pe_oi_change > 0:
                signals.append("✅ PE OI building (fresh writing)")
                score += 5
        else:
            # Downside: PE OI should unwind, CE OI should build
            ce_oi_change = atm_strikes['change_oi_ce'].sum()
            pe_oi_change = atm_strikes['change_oi_pe'].sum()
            
            if pe_oi_change < 0:
                signals.append("✅ PE OI wall breaking (unwinding)")
                score += 10
            if ce_oi_change > 0:
                signals.append("✅ CE OI building (fresh writing)")
                score += 5
        
        return {'signals': signals, 'score': score}
    
    def _analyze_oi_divergence(self, atm_strikes: pd.DataFrame, is_top_reversal: bool) -> Dict:
        """Analyze OI divergence for reversal detection"""
        signals = []
        score = 0
        
        ce_oi_change = atm_strikes['change_oi_ce'].sum()
        pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_top_reversal:
            # Top reversal: Price ↑ but CE OI ↑ (sellers loading)
            if ce_oi_change > 0 and pe_oi_change < 0:
                signals.append("✅ OI Divergence: CE OI ↑ + PE OI ↓ (sellers loading)")
                score += 30
            else:
                signals.append("❌ No clear OI divergence pattern")
        else:
            # Bottom reversal: Price ↓ but PE OI ↑ (put writers loading)
            if pe_oi_change > 0 and ce_oi_change < 0:
                signals.append("✅ OI Divergence: PE OI ↑ + CE OI ↓ (put writers loading)")
                score += 30
            else:
                signals.append("❌ No clear OI divergence pattern")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_iv_crash(self, atm_strikes: pd.DataFrame) -> Dict:
        """Analyze IV crash for reversal detection"""
        signals = []
        score = 0
        
        # Check if IV is collapsing (you might want historical comparison)
        avg_iv = (atm_strikes['iv_ce'].mean() + atm_strikes['iv_pe'].mean()) / 2
        
        if avg_iv < 15:
            signals.append(f"✅ IV Crash: Avg IV {avg_iv:.1f}% (smart money exiting)")
            score += 20
        elif avg_iv < 20:
            signals.append(f"⚠️ Moderate IV: {avg_iv:.1f}%")
            score += 10
        else:
            signals.append(f"❌ High IV: {avg_iv:.1f}% (volatility present)")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_writer_defense(self, df_chain: pd.DataFrame, spot_price: float, is_top_reversal: bool) -> Dict:
        """Analyze writer defense at key strikes"""
        signals = []
        score = 0
        
        if is_top_reversal:
            # Find resistance strike with high CE OI
            above_spot = df_chain[df_chain['strikePrice'] > spot_price]
            if not above_spot.empty:
                resistance_strike = above_spot.nlargest(1, 'oi_ce')
                if not resistance_strike.empty:
                    strike = resistance_strike['strikePrice'].values[0]
                    oi = resistance_strike['oi_ce'].values[0]
                    if oi > 1000000:  # 1M+ OI indicates strong resistance
                        signals.append(f"✅ Writer Defense: ₹{strike:.0f} CE OI {oi:,.0f}")
                        score += 25
        else:
            # Find support strike with high PE OI
            below_spot = df_chain[df_chain['strikePrice'] < spot_price]
            if not below_spot.empty:
                support_strike = below_spot.nlargest(1, 'oi_pe')
                if not support_strike.empty:
                    strike = support_strike['strikePrice'].values[0]
                    oi = support_strike['oi_pe'].values[0]
                    if oi > 1000000:  # 1M+ OI indicates strong support
                        signals.append(f"✅ Writer Defense: ₹{strike:.0f} PE OI {oi:,.0f}")
                        score += 25
        
        if score == 0:
            signals.append("❌ No strong writer defense detected")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_opposite_oi_build(self, atm_strikes: pd.DataFrame, is_top_reversal: bool) -> Dict:
        """Analyze opposite OI build for reversal"""
        signals = []
        score = 0
        
        ce_oi_change = atm_strikes['change_oi_ce'].sum()
        pe_oi_change = atm_strikes['change_oi_pe'].sum()
        
        if is_top_reversal:
            # Top reversal: New CE writing happening during uptrend
            if ce_oi_change > 10000:  # Significant CE writing
                signals.append(f"✅ Opposite OI: CE writing {ce_oi_change:,.0f} (reversal signal)")
                score += 15
            else:
                signals.append("❌ No significant opposite OI build")
        else:
            # Bottom reversal: New PE writing happening during downtrend
            if pe_oi_change > 10000:  # Significant PE writing
                signals.append(f"✅ Opposite OI: PE writing {pe_oi_change:,.0f} (reversal signal)")
                score += 15
            else:
                signals.append("❌ No significant opposite OI build")
        
        return {'signals': signals, 'score': score}
    
    def _analyze_pcr_extremes(self, df_chain: pd.DataFrame) -> Dict:
        """Analyze PCR extremes for reversal signals"""
        signals = []
        score = 0
        
        total_ce_oi = df_chain['oi_ce'].sum()
        total_pe_oi = df_chain['oi_pe'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if pcr > 1.4:
            signals.append(f"✅ PCR Extreme: {pcr:.2f} > 1.4 (overbought, reversal likely)")
            score += 10
        elif pcr < 0.5:
            signals.append(f"✅ PCR Extreme: {pcr:.2f} < 0.5 (oversold, reversal likely)")
            score += 10
        else:
            signals.append(f"⚠️ PCR Normal: {pcr:.2f}")
            score += 5
        
        return {'signals': signals, 'score': score}

    def get_breakout_reversal_score(self, df_chain: pd.DataFrame, spot_price: float, 
                                  price_action: Dict) -> Dict[str, Any]:
        """
        Combined breakout/reversal scoring system (0-100)
        """
        # Analyze breakout first
        price_change = price_action.get('price_change', 0)
        volume_change = price_action.get('volume_change', 0)
        
        breakout_analysis = self.analyze_breakout_confirmation(df_chain, spot_price, price_change, volume_change)
        reversal_analysis = self.analyze_reversal_confirmation(df_chain, spot_price, price_action)
        
        # Determine overall market state
        breakout_confidence = breakout_analysis.get('breakout_confidence', 0)
        reversal_confidence = reversal_analysis.get('reversal_confidence', 0)
        
        if breakout_confidence >= 60 and reversal_confidence < 50:
            market_state = "STRONG_BREAKOUT"
            overall_score = breakout_confidence
        elif reversal_confidence >= 70 and breakout_confidence < 40:
            market_state = "STRONG_REVERSAL"
            overall_score = reversal_confidence
        elif breakout_confidence >= 40 and reversal_confidence >= 50:
            market_state = "CONFLICT_ZONE"
            overall_score = (breakout_confidence + reversal_confidence) / 2
        else:
            market_state = "NEUTRAL_CHOPPY"
            overall_score = max(breakout_confidence, reversal_confidence)
        
        return {
            'overall_score': overall_score,
            'market_state': market_state,
            'breakout_analysis': breakout_analysis,
            'reversal_analysis': reversal_analysis,
            'trading_signal': self._generate_trading_signal(market_state, overall_score)
        }
    
    def _generate_trading_signal(self, market_state: str, score: float) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        if market_state == "STRONG_BREAKOUT":
            if score >= 70:
                return {'action': 'STRONG_BUY', 'confidence': 'HIGH', 'message': 'Real breakout confirmed'}
            else:
                return {'action': 'MODERATE_BUY', 'confidence': 'MEDIUM', 'message': 'Breakout likely'}
        
        elif market_state == "STRONG_REVERSAL":
            if score >= 75:
                return {'action': 'STRONG_SELL', 'confidence': 'HIGH', 'message': 'Reversal confirmed'}
            else:
                return {'action': 'MODERATE_SELL', 'confidence': 'MEDIUM', 'message': 'Reversal likely'}
        
        elif market_state == "CONFLICT_ZONE":
            return {'action': 'WAIT', 'confidence': 'LOW', 'message': 'Market in conflict, wait for clarity'}
        
        else:  # NEUTRAL_CHOPPY
            return {'action': 'RANGE_TRADE', 'confidence': 'LOW', 'message': 'Market choppy, trade ranges'}

# =============================================
# ENHANCED MARKET DATA FETCHER INTEGRATION
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources
    """

    def __init__(self):
        """Initialize enhanced market data fetcher"""
        self.dhan_fetcher = None
        if DHAN_AVAILABLE:
            try:
                self.dhan_fetcher = DhanDataFetcher()
            except Exception as e:
                print(f"Dhan API not available: {e}")

    # =========================================================================
    # DHAN API DATA FETCHING
    # =========================================================================

    def fetch_india_vix(self) -> Dict[str, Any]:
        """
        Fetch India VIX from Dhan API

        Returns:
            Dict with VIX data and sentiment
        """
        if not self.dhan_fetcher:
            return self._fetch_india_vix_yfinance()

        try:
            data = self.dhan_fetcher.fetch_ohlc_data(['INDIAVIX'])

            if data.get('INDIAVIX', {}).get('success'):
                vix_data = data['INDIAVIX']
                vix_value = vix_data.get('last_price', 0)

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
                    'source': 'Dhan API',
                    'value': vix_value,
                    'sentiment': vix_sentiment,
                    'bias': vix_bias,
                    'score': vix_score,
                    'timestamp': datetime.now(IST)
                }
            else:
                return self._fetch_india_vix_yfinance()
        except Exception as e:
            return self._fetch_india_vix_yfinance()

    def _fetch_india_vix_yfinance(self) -> Dict[str, Any]:
        """Fallback: Fetch India VIX from Yahoo Finance"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]

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
                    'timestamp': datetime.now(IST)
                }
        except Exception as e:
            pass

        return {'success': False, 'error': 'India VIX data not available'}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """
        Fetch all sector indices from Dhan API

        Returns:
            List of sector data with performance and bias
        """
        sectors = ['NIFTY_IT', 'NIFTY_AUTO', 'NIFTY_PHARMA', 'NIFTY_METAL',
                   'NIFTY_REALTY', 'NIFTY_ENERGY', 'NIFTY_FMCG']

        sector_data = []

        if self.dhan_fetcher:
            try:
                # Fetch all sectors in one call
                data = self.dhan_fetcher.fetch_ohlc_data(sectors)

                for sector in sectors:
                    if data.get(sector, {}).get('success'):
                        sector_info = data[sector]

                        last_price = sector_info.get('last_price', 0)
                        open_price = sector_info.get('open', last_price)

                        # Calculate change %
                        if open_price > 0:
                            change_pct = ((last_price - open_price) / open_price) * 100
                        else:
                            change_pct = 0

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

                        sector_data.append({
                            'sector': sector.replace('NIFTY_', 'NIFTY '),
                            'last_price': last_price,
                            'open': open_price,
                            'high': sector_info.get('high', 0),
                            'low': sector_info.get('low', 0),
                            'change_pct': change_pct,
                            'bias': bias,
                            'score': score,
                            'source': 'Dhan API'
                        })
            except Exception as e:
                print(f"Dhan sector fetch error: {e}")

        # Fallback to Yahoo Finance if Dhan failed
        if not sector_data:
            sector_data = self._fetch_sector_indices_yfinance()

        return sector_data

    def _fetch_sector_indices_yfinance(self) -> List[Dict[str, Any]]:
        """Fallback: Fetch sector indices from Yahoo Finance"""
        sectors_map = {
            '^CNXIT': 'NIFTY IT',
            '^CNXAUTO': 'NIFTY AUTO',
            '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL',
            '^CNXREALTY': 'NIFTY REALTY',
            '^CNXFMCG': 'NIFTY FMCG'
        }

        sector_data = []

        for symbol, name in sectors_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[0]
                    high_price = hist['High'].max()
                    low_price = hist['Low'].min()

                    change_pct = ((last_price - open_price) / open_price) * 100

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

                    sector_data.append({
                        'sector': name,
                        'last_price': last_price,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return sector_data

    # =========================================================================
    # YAHOO FINANCE DATA FETCHING
    # =========================================================================

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch global market indices from Yahoo Finance

        Returns:
            List of global market data with bias
        """
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

        for symbol, name in global_markets.items():
            try:
                ticker = yf.Ticker(symbol)
                # Get last 2 days to calculate change
                hist = ticker.history(period="2d")

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

                    market_data.append({
                        'market': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """
        Fetch intermarket data (commodities, currencies, bonds)

        Returns:
            List of intermarket data with bias
        """
        intermarket_assets = {
            'DX-Y.NYB': 'US DOLLAR INDEX',
            'CL=F': 'CRUDE OIL',
            'GC=F': 'GOLD',
            'INR=X': 'USD/INR',
            '^TNX': 'US 10Y TREASURY',
            'BTC-USD': 'BITCOIN'
        }

        intermarket_data = []

        for symbol, name in intermarket_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]

                    change_pct = ((current_close - prev_close) / prev_close) * 100

                    # Specific interpretations for each asset
                    if 'DOLLAR' in name:
                        # Strong dollar = bearish for emerging markets
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
                        # High oil = bearish for India (import dependent)
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
                        # Gold up = risk-off sentiment
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
                        # USD/INR up = INR weakening = bearish
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
                        # Yields up = risk-off
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
                        # Generic bias
                        if change_pct > 1:
                            bias = "BULLISH"
                            score = 40
                        elif change_pct < -1:
                            bias = "BEARISH"
                            score = -40
                        else:
                            bias = "NEUTRAL"
                            score = 0

                    intermarket_data.append({
                        'asset': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return intermarket_data

    # =========================================================================
    # COMPREHENSIVE DATA FETCH
    # =========================================================================

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """
        Fetch all enhanced market data from all sources

        Returns:
            Dict containing all market data organized by category
        """
        print("Fetching enhanced market data...")

        result = {
            'timestamp': datetime.now(IST),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'intermarket': [],
            'gamma_squeeze': {},
            'sector_rotation': {},
            'intraday_seasonality': {},
            'summary': {}
        }

        # 1. Fetch India VIX
        print("  - Fetching India VIX...")
        result['india_vix'] = self.fetch_india_vix()

        # 2. Fetch Sector Indices
        print("  - Fetching sector indices...")
        result['sector_indices'] = self.fetch_sector_indices()

        # 3. Fetch Global Markets
        print("  - Fetching global markets...")
        result['global_markets'] = self.fetch_global_markets()

        # 4. Fetch Intermarket Data
        print("  - Fetching intermarket data...")
        result['intermarket'] = self.fetch_intermarket_data()

        # 5. Detect Gamma Squeeze
        print("  - Analyzing Gamma Squeeze...")
        result['gamma_squeeze'] = self.detect_gamma_squeeze('NIFTY')

        # 6. Analyze Sector Rotation
        print("  - Analyzing Sector Rotation...")
        result['sector_rotation'] = self.analyze_sector_rotation()

        # 7. Analyze Intraday Seasonality
        print("  - Analyzing Intraday Seasonality...")
        result['intraday_seasonality'] = self.analyze_intraday_seasonality()

        # 8. Calculate summary statistics
        result['summary'] = self._calculate_summary(result)

        print("✓ Enhanced market data fetch completed!")

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

    # =========================================================================
    # GAMMA SQUEEZE DETECTION
    # =========================================================================

    def detect_gamma_squeeze(self, instrument: str = 'NIFTY') -> Dict[str, Any]:
        """
        Detect gamma squeeze potential from option chain data

        A gamma squeeze occurs when market makers need to hedge large gamma exposure,
        potentially causing rapid price movements.

        Args:
            instrument: Instrument name (NIFTY, BANKNIFTY, etc.)

        Returns:
            Dict with gamma squeeze analysis
        """
        # Check if option chain data is available
        if 'market_bias_data' not in st.session_state:
            return {'success': False, 'error': 'Option chain data not available'}

        market_bias_data = st.session_state.market_bias_data
        if not market_bias_data:
            return {'success': False, 'error': f'No option data for {instrument}'}

        try:
            # Find the instrument data
            instrument_data = None
            for data in market_bias_data:
                if data.get('instrument') == instrument:
                    instrument_data = data
                    break

            if not instrument_data:
                return {'success': False, 'error': f'No data found for {instrument}'}

            spot = instrument_data.get('spot_price', 0)

            # Get gamma data from institutional analysis if available
            gamma_exposure = 0
            if 'institutional_analysis' in instrument_data:
                gamma_analysis = instrument_data['institutional_analysis'].get('gamma_analysis', {})
                gamma_exposure = gamma_analysis.get('total_gamma_exposure', 0)

            # Gamma squeeze risk levels
            if abs(gamma_exposure) > 1000000:  # High gamma exposure
                if gamma_exposure > 0:
                    squeeze_risk = "HIGH UPSIDE RISK"
                    squeeze_bias = "BULLISH GAMMA SQUEEZE"
                    squeeze_score = 80
                    interpretation = "Large positive gamma → MMs will buy on dips, sell on rallies (resistance to movement)"
                else:
                    squeeze_risk = "HIGH DOWNSIDE RISK"
                    squeeze_bias = "BEARISH GAMMA SQUEEZE"
                    squeeze_score = -80
                    interpretation = "Large negative gamma → MMs will sell on dips, buy on rallies (amplified movement)"
            elif abs(gamma_exposure) > 500000:
                if gamma_exposure > 0:
                    squeeze_risk = "MODERATE UPSIDE RISK"
                    squeeze_bias = "BULLISH"
                    squeeze_score = 50
                    interpretation = "Moderate positive gamma → Some resistance to downward movement"
                else:
                    squeeze_risk = "MODERATE DOWNSIDE RISK"
                    squeeze_bias = "BEARISH"
                    squeeze_score = -50
                    interpretation = "Moderate negative gamma → Some amplification of movement"
            else:
                squeeze_risk = "LOW"
                squeeze_bias = "NEUTRAL"
                squeeze_score = 0
                interpretation = "Low gamma exposure → Normal market conditions"

            return {
                'success': True,
                'instrument': instrument,
                'spot': spot,
                'gamma_exposure': gamma_exposure,
                'squeeze_risk': squeeze_risk,
                'squeeze_bias': squeeze_bias,
                'squeeze_score': squeeze_score,
                'interpretation': interpretation,
                'timestamp': datetime.now(IST)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # SECTOR ROTATION MODEL
    # =========================================================================

    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """
        Analyze sector rotation to identify market leadership changes

        Returns:
            Dict with sector rotation analysis
        """
        sectors = self.fetch_sector_indices()

        if not sectors:
            return {'success': False, 'error': 'No sector data available'}

        # Sort sectors by performance
        sectors_sorted = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)

        # Identify leaders and laggards
        leaders = sectors_sorted[:3]  # Top 3 performing sectors
        laggards = sectors_sorted[-3:]  # Bottom 3 performing sectors

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
            'timestamp': datetime.now(IST)
        }

    # =========================================================================
    # INTRADAY SEASONALITY
    # =========================================================================

    def analyze_intraday_seasonality(self) -> Dict[str, Any]:
        """
        Analyze intraday time-based patterns

        Common patterns:
        - Opening 15 minutes: High volatility
        - 10:00-11:00 AM: Post-opening trend
        - 11:00-14:30: Lunchtime lull
        - 14:30-15:30: Closing rally/selloff

        Returns:
            Dict with intraday seasonality analysis
        """
        now = datetime.now(IST)
        current_time = now.time()
        current_hour = now.hour
        current_minute = now.minute

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


# ============================================================================
# STREAMLIT TAB FOR ENHANCED MARKET DATA
# ============================================================================

def add_enhanced_market_data_tab():
    """Add Enhanced Market Data tab to the Streamlit app"""
    
    st.header("🌐 Enhanced Market Data Dashboard")
    st.markdown("### Comprehensive Market Intelligence from Multiple Sources")
    
    # Initialize enhanced market data fetcher
    if 'enhanced_market_fetcher' not in st.session_state:
        st.session_state.enhanced_market_fetcher = EnhancedMarketData()
    
    if 'enhanced_market_data' not in st.session_state:
        st.session_state.enhanced_market_data = None
    
    # Refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔄 Refresh All Market Data", type="primary", use_container_width=True, key="refresh_market_data_btn"):
            with st.spinner("Fetching comprehensive market data from all sources..."):
                st.session_state.enhanced_market_data = st.session_state.enhanced_market_fetcher.fetch_all_enhanced_data()
                st.success("Market data refreshed!")
    
    with col2:
        st.metric("Auto-Refresh", "ON" if st.session_state.get('auto_refresh', False) else "OFF")
    
    with col3:
        if st.session_state.enhanced_market_data:
            last_update = st.session_state.enhanced_market_data['timestamp']
            st.caption(f"Last: {last_update.strftime('%H:%M:%S')}")
    
    # Display data if available
    if st.session_state.enhanced_market_data:
        data = st.session_state.enhanced_market_data
        
        # Summary Section
        st.subheader("📊 Market Summary")
        summary = data['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data Points", summary['total_data_points'])
        with col2:
            st.metric("Bullish Signals", summary['bullish_count'])
        with col3:
            st.metric("Bearish Signals", summary['bearish_count'])
        with col4:
            sentiment_color = "🟢" if summary['overall_sentiment'] == 'BULLISH' else "🔴" if summary['overall_sentiment'] == 'BEARISH' else "🟡"
            st.metric("Overall Sentiment", f"{sentiment_color} {summary['overall_sentiment']}")
        
        # Create tabs for different data categories
        market_tabs = st.tabs([
            "🇮🇳 Indian Markets", 
            "🌍 Global Markets", 
            "🔄 Sector Rotation", 
            "⏰ Intraday Timing",
            "γ Gamma Analysis"
        ])
        
        # Tab 1: Indian Markets
        with market_tabs[0]:
            st.subheader("Indian Market Analysis")
            
            # India VIX
            if data['india_vix'].get('success'):
                vix_data = data['india_vix']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("India VIX", f"{vix_data['value']:.2f}")
                with col2:
                    st.metric("Sentiment", vix_data['sentiment'])
                with col3:
                    bias_color = "🟢" if "BULLISH" in vix_data['bias'] else "🔴" if "BEARISH" in vix_data['bias'] else "🟡"
                    st.metric("Bias", f"{bias_color} {vix_data['bias']}")
                with col4:
                    st.metric("Source", vix_data['source'])
            
            # Sector Indices
            st.subheader("Sector Performance")
            if data['sector_indices']:
                sectors_df = pd.DataFrame(data['sector_indices'])
                
                # Color formatting for sectors
                def color_sector_bias(val):
                    if 'BULLISH' in val:
                        return 'background-color: #90EE90'
                    elif 'BEARISH' in val:
                        return 'background-color: #FFB6C1'
                    else:
                        return 'background-color: #FFFFE0'
                
                styled_sectors = sectors_df.style.map(color_sector_bias, subset=['bias'])
                st.dataframe(styled_sectors, use_container_width=True)
        
        # Tab 2: Global Markets
        with market_tabs[1]:
            st.subheader("Global Market Analysis")
            
            # Global Markets
            if data['global_markets']:
                global_df = pd.DataFrame(data['global_markets'])
                st.dataframe(global_df, use_container_width=True)
            
            # Intermarket Data
            st.subheader("Intermarket Analysis")
            if data['intermarket']:
                intermarket_df = pd.DataFrame(data['intermarket'])
                st.dataframe(intermarket_df, use_container_width=True)
        
        # Tab 3: Sector Rotation
        with market_tabs[2]:
            st.subheader("Sector Rotation Analysis")
            
            if data['sector_rotation'].get('success'):
                rotation = data['sector_rotation']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sector Breadth", f"{rotation['sector_breadth']:.1f}%")
                with col2:
                    st.metric("Rotation Pattern", rotation['rotation_pattern'])
                with col3:
                    st.metric("Sector Sentiment", rotation['sector_sentiment'])
                
                # Leaders and Laggards
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🏆 Sector Leaders")
                    if rotation['leaders']:
                        for leader in rotation['leaders']:
                            st.write(f"**{leader['sector']}**: {leader['change_pct']:.2f}%")
                
                with col2:
                    st.subheader("📉 Sector Laggards")
                    if rotation['laggards']:
                        for laggard in rotation['laggards']:
                            st.write(f"**{laggard['sector']}**: {laggard['change_pct']:.2f}%")
        
        # Tab 4: Intraday Timing
        with market_tabs[3]:
            st.subheader("Intraday Seasonality & Timing")
            
            if data['intraday_seasonality'].get('success'):
                seasonality = data['intraday_seasonality']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Session", seasonality['session'])
                with col2:
                    st.metric("Session Bias", seasonality['session_bias'])
                with col3:
                    st.metric("Weekday", seasonality['weekday'])
                
                st.info(f"**Session Characteristics**: {seasonality['session_characteristics']}")
                st.warning(f"**Trading Recommendation**: {seasonality['trading_recommendation']}")
                st.info(f"**Day Pattern**: {seasonality['day_characteristics']}")
        
        # Tab 5: Gamma Analysis
        with market_tabs[4]:
            st.subheader("Gamma Squeeze Analysis")
            
            if data['gamma_squeeze'].get('success'):
                gamma = data['gamma_squeeze']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Instrument", gamma['instrument'])
                with col2:
                    st.metric("Gamma Exposure", f"{gamma['gamma_exposure']:,.0f}")
                with col3:
                    risk_color = "🔴" if "HIGH" in gamma['squeeze_risk'] else "🟡" if "MODERATE" in gamma['squeeze_risk'] else "🟢"
                    st.metric("Squeeze Risk", f"{risk_color} {gamma['squeeze_risk']}")
                with col4:
                    bias_color = "🟢" if "BULLISH" in gamma['squeeze_bias'] else "🔴" if "BEARISH" in gamma['squeeze_bias'] else "🟡"
                    st.metric("Gamma Bias", f"{bias_color} {gamma['squeeze_bias']}")
                
                st.info(f"**Interpretation**: {gamma['interpretation']}")
    
    else:
        st.info("Click 'Refresh All Market Data' to load comprehensive market intelligence")
        
        # Quick data preview
        if st.button("Quick Preview - India VIX Only", key="quick_preview_btn"):
            with st.spinner("Fetching India VIX..."):
                vix_data = st.session_state.enhanced_market_fetcher.fetch_india_vix()
                if vix_data.get('success'):
                    st.metric("India VIX", f"{vix_data['value']:.2f}", vix_data['sentiment'])
                else:
                    st.error("Failed to fetch India VIX data")


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Bias Analysis Pro - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 Bias Analysis Pro — Complete Single-file App")
    st.markdown(
        "This Streamlit app wraps the **BiasAnalysisPro** engine (Pine → Python) and shows bias summary, "
        "price action, option chain analysis, and bias tabulation."
    )

    # Initialize all analyzers
    analysis = BiasAnalysisPro()
    options_analyzer = NSEOptionsAnalyzer()
    vob_indicator = VolumeOrderBlocks(sensitivity=5)
    gamma_analyzer = GammaSequenceAnalyzer()
    institutional_analyzer = InstitutionalOIAdvanced()
    breakout_analyzer = BreakoutReversalAnalyzer()

    # Sidebar inputs
    st.sidebar.header("Data & Symbol")
    symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
    period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
    interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

    # Auto-refresh configuration
    st.sidebar.header("Auto-Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
    st.session_state.auto_refresh = auto_refresh  # Store in session state
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=1)

    # Telegram Configuration
    st.sidebar.header("🔔 Telegram Alerts")
    if telegram_notifier.is_configured():
        st.sidebar.success("✅ Telegram configured via secrets!")
        telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    else:
        st.sidebar.warning("⚠️ Telegram not configured")
        st.sidebar.info("Add to .streamlit/secrets.toml:")
        st.sidebar.code("""
    [TELEGRAM]
    BOT_TOKEN = "your_bot_token_here"
    CHAT_ID = "your_chat_id_here"
    """)
        telegram_enabled = False

    # Shared state storage
    if 'last_df' not in st.session_state:
        st.session_state['last_df'] = None
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    if 'last_symbol' not in st.session_state:
        st.session_state['last_symbol'] = None
    if 'fetch_time' not in st.session_state:
        st.session_state['fetch_time'] = None
    if 'market_bias_data' not in st.session_state:
        st.session_state.market_bias_data = None
    if 'last_bias_update' not in st.session_state:
        st.session_state.last_bias_update = None
    if 'overall_nifty_bias' not in st.session_state:
        st.session_state.overall_nifty_bias = "NEUTRAL"
    if 'overall_nifty_score' not in st.session_state:
        st.session_state.overall_nifty_score = 0
    if 'atm_detailed_bias' not in st.session_state:
        st.session_state.atm_detailed_bias = None
    if 'vob_blocks' not in st.session_state:
        st.session_state.vob_blocks = {'bullish': [], 'bearish': []}
    if 'last_telegram_alert' not in st.session_state:
        st.session_state.last_telegram_alert = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Initialize session state for auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0

    # Enhanced tabs with all features including new Market Data tab
    tabs = st.tabs([
        "Overall Bias", "Bias Summary", "Price Action", "Option Chain", 
        "Bias Tabulation", "🌐 Market Data", "🤖 AI Analysis"
    ])

    # OVERALL BIAS TAB
    with tabs[0]:
        st.header("🎯 Overall Nifty Bias Analysis")
        
        if not st.session_state.overall_nifty_bias:
            st.info("No analysis run yet. Analysis runs automatically every minute...")
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
            
            # Breakdown of bias components
            st.subheader("Bias Components Breakdown")
            
            components_data = []
            
            # Technical Analysis Component
            if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
                tech_result = st.session_state['last_result']
                components_data.append({
                    'Component': 'Technical Analysis',
                    'Bias': tech_result.get('overall_bias', 'NEUTRAL'),
                    'Score': tech_result.get('overall_score', 0),
                    'Weight': '20%',
                    'Confidence': f"{tech_result.get('overall_confidence', 0):.1f}%"
                })
            
            # Options Analysis Component
            if st.session_state.market_bias_data:
                for instrument_data in st.session_state.market_bias_data:
                    if instrument_data['instrument'] == 'NIFTY':
                        components_data.append({
                            'Component': 'Options Chain Overall',
                            'Bias': instrument_data.get('combined_bias', 'Neutral'),
                            'Score': instrument_data.get('combined_score', 0),
                            'Weight': '25%',
                            'Confidence': 'High' if abs(instrument_data.get('combined_score', 0)) > 2 else 'Medium'
                        })
                        break
            
            # Institutional OI Bias Component
            if st.session_state.market_bias_data:
                for instrument_data in st.session_state.market_bias_data:
                    if instrument_data['instrument'] == 'NIFTY' and 'institutional_analysis' in instrument_data:
                        inst_analysis = instrument_data['institutional_analysis']
                        inst_bias_analysis = inst_analysis.get('institutional_bias_analysis', {})
                        
                        if inst_bias_analysis:
                            components_data.append({
                                'Component': 'Institutional OI Patterns',
                                'Bias': inst_bias_analysis.get('overall_bias', 'NEUTRAL'),
                                'Score': inst_bias_analysis.get('bias_score', 0),
                                'Weight': '15%',
                                'Confidence': inst_bias_analysis.get('confidence', 'LOW')
                            })
                        break
            
            # ATM Detailed Bias Component
            if st.session_state.atm_detailed_bias:
                atm_data = st.session_state.atm_detailed_bias
                components_data.append({
                    'Component': 'ATM Detailed Bias',
                    'Bias': atm_data['bias'],
                    'Score': atm_data['score'],
                    'Weight': '10%',
                    'Confidence': f"{abs(atm_data['score']):.1f}%"
                })
            
            # Volume Analysis Component
            if st.session_state.vob_blocks:
                bullish_blocks = st.session_state.vob_blocks['bullish']
                bearish_blocks = st.session_state.vob_blocks['bearish']
                vob_score = len(bullish_blocks) - len(bearish_blocks)
                vob_bias = "BULLISH" if vob_score > 0 else "BEARISH" if vob_score < 0 else "NEUTRAL"
                
                components_data.append({
                    'Component': 'Volume Order Blocks',
                    'Bias': vob_bias,
                    'Score': vob_score,
                    'Weight': '10%',
                    'Confidence': f"Blocks: {len(bullish_blocks)}B/{len(bearish_blocks)}S"
                })
            
            # Breakout/Reversal Component
            if st.session_state.market_bias_data:
                for instrument_data in st.session_state.market_bias_data:
                    if instrument_data['instrument'] == 'NIFTY' and 'breakout_reversal_analysis' in instrument_data:
                        breakout_data = instrument_data['breakout_reversal_analysis']
                        breakout_score = breakout_data.get('overall_score', 0)
                        market_state = breakout_data.get('market_state', 'NEUTRAL_CHOPPY')
                        
                        components_data.append({
                            'Component': 'Breakout/Reversal',
                            'Bias': market_state.replace('_', ' ').title(),
                            'Score': breakout_score,
                            'Weight': '20%',
                            'Confidence': f"{breakout_score:.1f}%"
                        })
                        break
            
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

    # MARKET DATA TAB
    with tabs[5]:
        add_enhanced_market_data_tab()

    # Footer
    st.markdown("---")
    st.caption("BiasAnalysisPro — Complete Enhanced Dashboard with Auto-Refresh, Overall Nifty Bias Analysis, Institutional Breakout/Reversal Detection, and AI-Powered Analysis.")
    st.caption("🔔 Telegram alerts sent when Technical Analysis, Options Chain, and ATM Detailed Bias are aligned (Bullish/Bearish)")

if __name__ == "__main__":
    main()
