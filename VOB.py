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
# SHARED UTILITIES CLASS
# =============================================

class TradingUtilities:
    """Shared utilities for all analyzers - ELIMINATES DUPLICATE CODE"""
    
    @staticmethod
    def normalize_chain_columns(df_chain: pd.DataFrame) -> pd.DataFrame:
        """SINGLE IMPLEMENTATION: Normalize option chain column names"""
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
            
            # Alternative column names
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
                df[col] = 0
        
        # Calculate Greeks if not present
        if 'Gamma_CE' not in df.columns:
            df['Gamma_CE'] = 0.01
            df['Gamma_PE'] = 0.01
        
        return df
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """SINGLE IMPLEMENTATION: Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """SINGLE IMPLEMENTATION: Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
        """SINGLE IMPLEMENTATION: Calculate option Greeks"""
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
    
    @staticmethod
    def determine_bias_from_score(score: float) -> str:
        """SINGLE IMPLEMENTATION: Convert score to bias string"""
        if score >= 25:
            return "STRONG BULLISH"
        elif score >= 10:
            return "BULLISH"
        elif score <= -25:
            return "STRONG BEARISH"
        elif score <= -10:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def normalize_score(score: float, min_val: float = -100, max_val: float = 100) -> float:
        """Normalize score to standard range"""
        return max(min_val, min(max_val, score))

# =============================================
# CONSOLIDATED STATE MANAGEMENT
# =============================================

class AnalysisState:
    """Single state management for all analysis - ELIMINATES REDUNDANT STATE"""
    
    def __init__(self):
        self.current_analysis = {
            'technical': None,
            'options': None,
            'institutional': None,
            'vob_blocks': {'bullish': [], 'bearish': []},
            'atm_detailed': None,
            'timestamp': None,
            'symbol': None,
            'overall_bias': "NEUTRAL",
            'overall_score': 0
        }
        self.last_refresh = datetime.now(IST)
        self.last_telegram_alert = None
    
    def update_technical(self, result: Dict, df: pd.DataFrame):
        """Update technical analysis results"""
        self.current_analysis['technical'] = result
        self.current_analysis['symbol'] = result.get('symbol')
        self.current_analysis['timestamp'] = datetime.now(IST)
    
    def update_options(self, options_data: List[Dict]):
        """Update options analysis results"""
        self.current_analysis['options'] = options_data
        self.current_analysis['timestamp'] = datetime.now(IST)
    
    def update_vob_blocks(self, bullish_blocks: List, bearish_blocks: List):
        """Update volume order blocks"""
        self.current_analysis['vob_blocks'] = {
            'bullish': bullish_blocks,
            'bearish': bearish_blocks
        }
    
    def update_atm_detailed(self, atm_data: Dict):
        """Update ATM detailed bias"""
        self.current_analysis['atm_detailed'] = atm_data
    
    def update_overall_bias(self, bias: str, score: float):
        """Update overall market bias"""
        self.current_analysis['overall_bias'] = bias
        self.current_analysis['overall_score'] = score
    
    def get_technical(self) -> Optional[Dict]:
        return self.current_analysis['technical']
    
    def get_options(self) -> Optional[List[Dict]]:
        return self.current_analysis['options']
    
    def get_vob_blocks(self) -> Dict:
        return self.current_analysis['vob_blocks']
    
    def get_atm_detailed(self) -> Optional[Dict]:
        return self.current_analysis['atm_detailed']
    
    def get_overall_bias(self) -> Tuple[str, float]:
        return self.current_analysis['overall_bias'], self.current_analysis['overall_score']
    
    def should_refresh(self, interval_minutes: int) -> bool:
        """Check if data should be refreshed"""
        current_time = datetime.now(IST)
        time_diff = (current_time - self.last_refresh).total_seconds() / 60
        return time_diff >= interval_minutes
    
    def mark_refreshed(self):
        """Mark data as refreshed"""
        self.last_refresh = datetime.now(IST)

# =============================================
# TELEGRAM NOTIFICATION SYSTEM (UNCHANGED)
# =============================================

class TelegramNotifier:
    """Telegram notification system for bias alerts"""
    
    def __init__(self):
        self.bot_token = st.secrets.get("TELEGRAM", {}).get("BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM", {}).get("CHAT_ID", "")
        self.last_alert_time = {}
        self.alert_cooldown = 300
        
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)
        
    def send_message(self, message: str, alert_type: str = "INFO") -> bool:
        try:
            if not self.is_configured():
                return False
            
            current_time = time.time()
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
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
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    def send_bias_alert(self, technical_bias: str, options_bias: str, atm_bias: str, overall_bias: str, score: float):
        """Send comprehensive bias alert"""
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

# =============================================
# SIMPLIFIED BIAS ANALYSIS PRO
# =============================================

class BiasAnalysisPro:
    """
    Simplified Bias Analysis - REMOVED UNUSED CONFIGURATION
    """

    def __init__(self):
        """Initialize with only used configuration"""
        self.config = self._simplified_config()
        self.all_bias_results = []
        self.overall_bias = "NEUTRAL"
        self.overall_score = 0
        self.utilities = TradingUtilities()

    def _simplified_config(self) -> Dict:
        """REMOVED UNUSED PARAMETERS - Only keep actually used ones"""
        return {
            # Timeframes
            'tf1': '15m',
            'tf2': '1h',

            # Indicator periods (ACTUALLY USED)
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,

            # Volume (ACTUALLY USED)
            'volume_roc_length': 14,
            'volume_threshold': 1.2,

            # Bias parameters (ACTUALLY USED)
            'bias_strength': 60,
            'divergence_threshold': 60,

            # Adaptive weights (ACTUALLY USED)
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,
        }

    # DATA FETCHING (UNCHANGED)
    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Dhan API or Yahoo Finance"""
        indian_indices = {'^NSEI': 'NIFTY', '^BSESN': 'SENSEX', '^NSEBANK': 'BANKNIFTY'}

        if symbol in indian_indices and DHAN_AVAILABLE:
            try:
                dhan_instrument = indian_indices[symbol]
                fetcher = DhanDataFetcher()

                interval_map = {'1m': '1', '5m': '5', '15m': '15', '1h': '60'}
                dhan_interval = interval_map.get(interval, '5')

                now_ist = datetime.now(IST)
                to_date = now_ist.strftime('%Y-%m-%d %H:%M:%S')
                from_date = (now_ist - timedelta(days=7)).replace(hour=9, minute=15, second=0).strftime('%Y-%m-%d %H:%M:%S')

                result = fetcher.fetch_intraday_data(dhan_instrument, interval=dhan_interval, from_date=from_date, to_date=to_date)

                if result.get('success') and result.get('data') is not None:
                    df = result['data']
                    df.columns = [col.capitalize() for col in df.columns]

                    if 'Volume' not in df.columns:
                        df['Volume'] = 0
                    else:
                        df['Volume'] = df['Volume'].fillna(0)

                    if not df.empty:
                        return df
                else:
                    print(f"Warning: Dhan API failed for {symbol}, falling back to yfinance")
            except Exception as e:
                print(f"Error fetching from Dhan API for {symbol}: {e}")

        # Fallback to Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return pd.DataFrame()

            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # TECHNICAL INDICATORS USING SHARED UTILITIES
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
        mfi = mfi.fillna(50)

        return mfi

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        """Calculate DMI indicators"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range - USING SHARED ATR
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
        """Calculate VWAP"""
        if df['Volume'].sum() == 0:
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe
        vwap = vwap.fillna(typical_price)

        return vwap

    # USING SHARED UTILITIES FOR EMA
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        return self.utilities.calculate_ema(data, period)

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA using shared ATR"""
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

        # USING SHARED ATR
        atr = self.utilities.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * band_distance
        lower_band = vidya_smoothed - atr * band_distance

        is_trend_up = close > upper_band
        is_trend_down = close < lower_band

        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False

        return vidya_smoothed, vidya_bullish, vidya_bearish

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta"""
        if df['Volume'].sum() == 0:
            return 0, False, False

        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots"""
        if df['Volume'].sum() == 0:
            return False, False, 0, 0

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

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5):
        """Calculate Volume Order Blocks using shared EMA"""
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    # COMPREHENSIVE BIAS ANALYSIS (FIXED)
    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """Analyze all 8 bias indicators with simplified logic"""
        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            return {
                'success': False,
                'error': f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            }

        current_price = df['Close'].iloc[-1]
        bias_results = []

        # FAST INDICATORS (8 total)
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

        # 2. HVP
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

        # 3. VOB
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
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)
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
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
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

        # CALCULATE OVERALL BIAS - COMPLETELY SIMPLIFIED
        fast_bull = sum(1 for bias in bias_results if 'BULLISH' in bias['bias'] and bias['category'] == 'fast')
        fast_bear = sum(1 for bias in bias_results if 'BEARISH' in bias['bias'] and bias['category'] == 'fast')
        fast_total = sum(1 for bias in bias_results if bias['category'] == 'fast')

        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        # SIMPLIFIED: Use only fast indicators with simple weighting
        fast_weight = self.config['normal_fast_weight']
        
        # Calculate weighted scores using only fast indicators
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
            'timestamp': datetime.now(IST),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': fast_bull,
            'bearish_count': fast_bear,
            'neutral_count': fast_total - fast_bull - fast_bear,
            'total_indicators': len(bias_results),
            'mode': "NORMAL",
            'fast_bull_pct': fast_bull_pct,
            'fast_bear_pct': fast_bear_pct,
            'bullish_bias_pct': bullish_bias_pct,
            'bearish_bias_pct': bearish_bias_pct
        }

# =============================================
# SIMPLIFIED VOLUME ORDER BLOCKS
# =============================================

class VolumeOrderBlocks:
    """Simplified Volume Order Blocks - REMOVED UNUSED METHODS"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.utilities = TradingUtilities()
        
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks using shared utilities"""
        if len(df) < self.length2:
            return [], []
        
        # USING SHARED EMA
        ema1 = self.utilities.calculate_ema(df['Close'], self.length1)
        ema2 = self.utilities.calculate_ema(df['Close'], self.length2)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        # USING SHARED ATR
        atr = self.utilities.calculate_atr(df, 200)
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
        """Filter overlapping blocks"""
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
# SIMPLIFIED GAMMA SEQUENCE ANALYZER
# =============================================

class GammaSequenceAnalyzer:
    """Simplified Gamma Sequence Analysis"""
    
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
        self.utilities = TradingUtilities()
    
    def calculate_gamma_exposure(self, df_chain: pd.DataFrame) -> pd.DataFrame:
        """Calculate Gamma exposure"""
        df = df_chain.copy()
        df['gamma_exposure_ce'] = df['Gamma_CE'] * df['openInterest_CE'] * 100
        df['gamma_exposure_pe'] = df['Gamma_PE'] * df['openInterest_PE'] * 100
        df['net_gamma_exposure'] = df['gamma_exposure_ce'] + df['gamma_exposure_pe']
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
            
            analysis = {
                'total_gamma_exposure': df_with_gamma['net_gamma_exposure'].sum(),
                'gamma_bias': self._calculate_overall_gamma_bias(df_with_gamma),
                'zones': self._analyze_gamma_zones(df_with_gamma, spot_price),
                'sequence': self._analyze_gamma_sequence(df_with_gamma),
                'walls': self._find_gamma_walls(df_with_gamma),
                'profile': self._get_gamma_profile(df_with_gamma['net_gamma_exposure'].sum())
            }
            
            analysis['gamma_score'] = self._calculate_gamma_score(analysis)
            return analysis
            
        except Exception as e:
            return {'gamma_bias': 'NEUTRAL', 'gamma_score': 0, 'error': str(e)}
    
    def _calculate_overall_gamma_bias(self, df: pd.DataFrame) -> str:
        """Calculate overall Gamma bias"""
        total_gamma = df['net_gamma_exposure'].sum()
        
        if total_gamma > 10000: return "STRONG_BULLISH"
        elif total_gamma > 5000: return "BULLISH"
        elif total_gamma > 1000: return "MILD_BULLISH"
        elif total_gamma < -10000: return "STRONG_BEARISH"
        elif total_gamma < -5000: return "BEARISH"
        elif total_gamma < -1000: return "MILD_BEARISH"
        else: return "NEUTRAL"
    
    def _analyze_gamma_zones(self, df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Analyze Gamma across different price zones"""
        strike_diff = df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0] if len(df) > 1 else 50
        
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
        df_sorted = df.sort_values('strikePrice')
        df_sorted['gamma_change'] = df_sorted['net_gamma_exposure'].diff()
        
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
        gamma_walls = []
        
        for i in range(1, len(df) - 1):
            current_gamma = df.iloc[i]['net_gamma_exposure']
            prev_gamma = df.iloc[i-1]['net_gamma_exposure']
            next_gamma = df.iloc[i+1]['net_gamma_exposure']
            
            if current_gamma > prev_gamma and current_gamma > next_gamma and current_gamma > 5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'RESISTANCE',
                    'strength': 'STRONG' if current_gamma > 10000 else 'MODERATE'
                })
            elif current_gamma < prev_gamma and current_gamma < next_gamma and current_gamma < -5000:
                gamma_walls.append({
                    'strike': df.iloc[i]['strikePrice'],
                    'gamma_exposure': current_gamma,
                    'type': 'SUPPORT',
                    'strength': 'STRONG' if current_gamma < -10000 else 'MODERATE'
                })
        
        return sorted(gamma_walls, key=lambda x: abs(x['gamma_exposure']), reverse=True)[:5]
    
    def _calculate_gamma_score(self, analysis: Dict) -> float:
        """Calculate comprehensive Gamma score"""
        base_score = self.gamma_levels.get(analysis['profile'], {}).get('score', 0)
        
        seq_analysis = analysis.get('sequence', {})
        pos_seqs = len(seq_analysis.get('positive_sequences', []))
        neg_seqs = len(seq_analysis.get('negative_sequences', []))
        
        if pos_seqs > neg_seqs:
            sequence_bonus = 10
        elif neg_seqs > pos_seqs:
            sequence_bonus = -10
        else:
            sequence_bonus = 0
        
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

# =============================================
# CONSOLIDATED OPTIONS ANALYZER
# =============================================

class ConsolidatedOptionsAnalyzer:
    """MERGED: NSEOptionsAnalyzer + InstitutionalOIAdvanced + BreakoutReversalAnalyzer"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.utilities = TradingUtilities()
        self.gamma_analyzer = GammaSequenceAnalyzer()
        
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
                'FINNIFTY': {'lot_size': 40, 'atm_range': 200, 'zone_size': 100},
            }
        }
        
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
        
        self.weights = {
            'basic_oi': 0.25,
            'institutional': 0.30,
            'gamma': 0.20,
            'breakout': 0.25
        }
        
        self.last_refresh_time = {}
        self.refresh_interval = 2

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

    def analyze_institutional_oi_pattern(self, option_type: str, oi_change: float, price_change: float, 
                                       volume: float, iv_change: float, bid_ask_ratio: float) -> Dict:
        oi_action = "Winding_Up" if oi_change > 0 else "Unwinding_Down"
        price_action = "Price_Up" if price_change > 0 else "Price_Down"
        pattern_key = f"{oi_action}_{price_action}"
        
        base_pattern = self.master_table_rules.get(option_type, {}).get(pattern_key, {})
        
        if not base_pattern:
            return {'bias': 'NEUTRAL', 'confidence': 'LOW', 'pattern': 'Unknown'}
        
        volume_signal = "High" if volume > 1000 else "Low"
        iv_signal = "Rising" if iv_change > 0 else "Falling"
        liquidity_signal = "Bid_Heavy" if bid_ask_ratio > 1.2 else "Ask_Heavy" if bid_ask_ratio < 0.8 else "Balanced"
        
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

    def calculate_net_institutional_bias(self, patterns: List[Dict]) -> float:
        if not patterns:
            return 0.0
        
        net_score = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            option_type = pattern.get('option_type', '')
            bias = pattern.get('bias', 'NEUTRAL')
            
            if option_type == 'CE':
                multiplier = -1
            elif option_type == 'PE':
                multiplier = 1
            else:
                multiplier = 0
            
            if 'BULL' in bias.upper():
                bias_score = 1.0
            elif 'BEAR' in bias.upper():
                bias_score = -1.0
            else:
                bias_score = 0.0
            
            confidence_multipliers = {
                'VERY_HIGH': 1.5,
                'HIGH': 1.2,
                'MEDIUM': 1.0,
                'LOW': 0.7
            }
            weight = confidence_multipliers.get(pattern.get('confidence', 'LOW'), 1.0)
            
            net_score += bias_score * multiplier * weight
            total_weight += weight
        
        if total_weight > 0:
            normalized_score = (net_score / total_weight) * 100
            return max(-100, min(100, normalized_score))
        else:
            return 0.0

    def calculate_institutional_oi_bias(self, patterns: List[Dict]) -> Dict[str, Any]:
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
        
        bias_score = self.calculate_net_institutional_bias(patterns)
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for pattern in patterns:
            bias = pattern.get('bias', 'NEUTRAL')
            if 'BULL' in bias.upper():
                bullish_count += 1
            elif 'BEAR' in bias.upper():
                bearish_count += 1
            else:
                neutral_count += 1
        
        if bias_score > 15:
            overall_bias = "BULLISH"
        elif bias_score < -15:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"
        
        bias_strength = abs(bias_score)
        if bias_strength >= 40:
            confidence_level = "VERY_HIGH"
        elif bias_strength >= 25:
            confidence_level = "HIGH"
        elif bias_strength >= 15:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        institution_moves = {}
        for pattern in patterns:
            move = pattern.get('institution_move', 'Unknown')
            if move not in institution_moves:
                institution_moves[move] = 0
            institution_moves[move] += 1
        
        dominant_move = max(institution_moves.items(), key=lambda x: x[1])[0] if institution_moves else "No dominant move"
        
        return {
            'overall_bias': overall_bias,
            'bias_score': bias_score,
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'neutral_patterns': neutral_count,
            'total_patterns': len(patterns),
            'confidence': confidence_level,
            'dominant_move': dominant_move,
            'institution_moves': institution_moves,
            'bias_strength': bias_strength
        }

    def analyze_atm_institutional_footprint(self, df_chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        try:
            df_chain = self.utilities.normalize_chain_columns(df_chain)
            
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
            
            for _, strike_data in atm_strikes.iterrows():
                strike = strike_data['strikePrice']
                
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
            
            institutional_bias_analysis = self.calculate_institutional_oi_bias(patterns)
            gamma_analysis = self.gamma_analyzer.analyze_gamma_sequence_bias(df_chain, spot_price)
            
            return {
                'overall_bias': institutional_bias_analysis['overall_bias'],
                'score': institutional_bias_analysis['bias_score'],
                'patterns': patterns,
                'institutional_bias_analysis': institutional_bias_analysis,
                'gamma_analysis': gamma_analysis,
                'strikes_analyzed': len(atm_strikes),
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
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

    def calculate_max_pain(self, df_full_chain: pd.DataFrame) -> Optional[float]:
        try:
            strikes = df_full_chain['strikePrice'].unique()
            pain_values = []

            for strike in strikes:
                call_pain = 0
                put_pain = 0

                for _, row in df_full_chain.iterrows():
                    row_strike = row['strikePrice']

                    if row_strike < strike:
                        call_pain += (strike - row_strike) * row.get('openInterest_CE', 0)

                    if row_strike > strike:
                        put_pain += (row_strike - strike) * row.get('openInterest_PE', 0)

                total_pain = call_pain + put_pain
                pain_values.append({'strike': strike, 'pain': total_pain})

            max_pain_data = min(pain_values, key=lambda x: x['pain'])
            return max_pain_data['strike']
        except:
            return None

    def calculate_synthetic_future_bias(self, atm_ce_price: float, atm_pe_price: float, atm_strike: float, spot_price: float) -> Tuple[str, float, float]:
        try:
            synthetic_future = atm_strike + atm_ce_price - atm_pe_price
            difference = synthetic_future - spot_price

            if difference > 5:
                return "Bullish", synthetic_future, difference
            elif difference < -5:
                return "Bearish", synthetic_future, difference
            else:
                return "Neutral", synthetic_future, difference
        except:
            return "Neutral", 0, 0

    def calculate_atm_buildup_pattern(self, atm_ce_oi: float, atm_pe_oi: float, atm_ce_change: float, atm_pe_change: float) -> str:
        try:
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

    def find_call_resistance_put_support(self, df_full_chain: pd.DataFrame, spot_price: float) -> Tuple[Optional[float], Optional[float]]:
        try:
            above_spot = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            if not above_spot.empty:
                call_resistance = above_spot.nlargest(1, 'openInterest_CE')['strikePrice'].values[0]
            else:
                call_resistance = None

            below_spot = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            if not below_spot.empty:
                put_support = below_spot.nlargest(1, 'openInterest_PE')['strikePrice'].values[0]
            else:
                put_support = None

            return call_resistance, put_support
        except:
            return None, None

    def calculate_detailed_atm_bias(self, df_atm: pd.DataFrame, atm_strike: float, spot_price: float) -> Dict[str, Any]:
        try:
            detailed_bias = {}
            
            for _, row in df_atm.iterrows():
                if row['strikePrice'] == atm_strike:
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
                        "OI_Bias": "Bullish" if row['openInterest_CE'] < row['openInterest_PE'] else "Bearish",
                        "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                        "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                        "Delta_Bias": "Bullish" if abs(row['Delta_PE']) > abs(row['Delta_CE']) else "Bearish",
                        "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                        "Premium_Bias": "Bullish" if row['lastPrice_CE'] < row['lastPrice_PE'] else "Bearish",
                        "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                        "Delta_Exposure_Bias": delta_exp_bias,
                        "Gamma_Exposure_Bias": gamma_exp_bias,
                        "IV_Skew_Bias": iv_skew_bias,
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
            return {}

    def analyze_breakout_confirmation(self, df_chain: pd.DataFrame, spot_price: float, 
                                    price_change: float, volume_change: float) -> Dict[str, Any]:
        try:
            df_chain = self.utilities.normalize_chain_columns(df_chain)
            
            strike_diff = df_chain['strikePrice'].iloc[1] - df_chain['strikePrice'].iloc[0] if len(df_chain) > 1 else 50
            atm_strikes = df_chain[abs(df_chain['strikePrice'] - spot_price) <= strike_diff * 2].copy()
            
            if atm_strikes.empty:
                return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}
            
            is_upside_breakout = price_change > 0
            direction = "UP" if is_upside_breakout else "DOWN"
            
            signals = []
            total_score = 0
            max_score = 0
            
            # OI Analysis
            total_ce_oi_change = atm_strikes['change_oi_ce'].sum()
            total_pe_oi_change = atm_strikes['change_oi_pe'].sum()
            
            if is_upside_breakout:
                if total_ce_oi_change < 0:
                    signals.append("✅ CE OI decreasing (call sellers running)")
                    total_score += 10
                if total_pe_oi_change > 0:
                    signals.append("✅ PE OI increasing (put writers entering)")
                    total_score += 15
            else:
                if total_ce_oi_change > 0:
                    signals.append("✅ CE OI increasing (call writers attacking)")
                    total_score += 10
                if total_pe_oi_change < 0:
                    signals.append("✅ PE OI decreasing (put sellers exiting)")
                    total_score += 15
            
            max_score += 25
            
            # PCR Analysis
            total_ce_oi = df_chain['oi_ce'].sum()
            total_pe_oi = df_chain['oi_pe'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            if is_upside_breakout:
                if pcr > 0.8:
                    signals.append(f"✅ PCR {pcr:.2f} > 0.8 (bullish bias)")
                    total_score += 10
            else:
                if pcr < 0.7:
                    signals.append(f"✅ PCR {pcr:.2f} < 0.7 (bearish bias)")
                    total_score += 10
            
            max_score += 10
            
            breakout_confidence = (total_score / max_score) * 100 if max_score > 0 else 0
            
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
            return {'breakout_confidence': 0, 'direction': 'UNKNOWN', 'signals': []}

    def analyze_comprehensive_options_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            records = data['records']
            spot = data['spot']
            expiry = data['expiry']

            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06

            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.utilities.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)

                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.utilities.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)

            if not calls or not puts:
                return None

            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

            atm_range = self.NSE_INSTRUMENTS['indices'].get(instrument, {}).get('atm_range', 200)
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            df_atm = df[abs(df['strikePrice'] - atm_strike) <= atm_range]

            if df_atm.empty:
                return None

            atm_df = df[df['strikePrice'] == atm_strike]
            if not atm_df.empty:
                atm_ce_price = atm_df['lastPrice_CE'].values[0]
                atm_pe_price = atm_df['lastPrice_PE'].values[0]
                atm_ce_oi = atm_df['openInterest_CE'].values[0]
                atm_pe_oi = atm_df['openInterest_PE'].values[0]
                atm_ce_change = atm_df['changeinOpenInterest_CE'].values[0]
                atm_pe_change = atm_df['changeinOpenInterest_PE'].values[0]
            else:
                return None

            synthetic_bias, synthetic_future, synthetic_diff = self.calculate_synthetic_future_bias(
                atm_ce_price, atm_pe_price, atm_strike, spot
            )
            
            atm_buildup = self.calculate_atm_buildup_pattern(
                atm_ce_oi, atm_pe_oi, atm_ce_change, atm_pe_change
            )
            
            max_pain_strike = self.calculate_max_pain(df)
            distance_from_max_pain = spot - max_pain_strike if max_pain_strike else 0
            
            call_resistance, put_support = self.find_call_resistance_put_support(df, spot)
            
            detailed_atm_bias = self.calculate_detailed_atm_bias(df_atm, atm_strike, spot)
            
            institutional_analysis = self.analyze_atm_institutional_footprint(df, spot)
            
            breakout_analysis = self.analyze_breakout_confirmation(df, spot, 0.5, 15)
            
            breakout_score = breakout_analysis.get('breakout_confidence', 0)
            if breakout_analysis.get('direction') == "DOWN":
                breakout_score = -breakout_score

            basic_score = 0
            weights = {
                "oi_bias": 2, "chg_oi_bias": 2, "synthetic_bias": 2, "breakout_bias": 2
            }

            if data['total_pe_oi'] > data['total_ce_oi']:
                basic_score += weights["oi_bias"]
            else:
                basic_score -= weights["oi_bias"]
            
            if data['total_pe_change'] > data['total_ce_change']:
                basic_score += weights["chg_oi_bias"]
            else:
                basic_score -= weights["chg_oi_bias"]
            
            if synthetic_bias == "Bullish":
                basic_score += weights["synthetic_bias"]
            elif synthetic_bias == "Bearish":
                basic_score -= weights["synthetic_bias"]

            max_possible_score = sum(weights.values())
            normalized_score = (basic_score / max_possible_score) * 100

            institutional_score = institutional_analysis.get('score', 0)
            gamma_score = institutional_analysis.get('gamma_analysis', {}).get('gamma_score', 0)

            combined_score = (
                normalized_score * self.weights['basic_oi'] +
                institutional_score * self.weights['institutional'] +
                gamma_score * self.weights['gamma'] +
                breakout_score * self.weights['breakout']
            )

            combined_bias = self.utilities.determine_bias_from_score(combined_score)

            return {
                'instrument': instrument,
                'spot_price': spot,
                'atm_strike': atm_strike,
                'overall_bias': combined_bias,
                'combined_bias': combined_bias,
                'bias_score': normalized_score,
                'combined_score': combined_score,
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'pcr_change': abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change'],
                'detailed_atm_bias': detailed_atm_bias,
                'institutional_analysis': institutional_analysis,
                'breakout_reversal_analysis': breakout_analysis,
                'institutional_score': institutional_score,
                'gamma_score': gamma_score,
                'breakout_score': breakout_score,
                'comprehensive_metrics': {
                    'synthetic_bias': synthetic_bias,
                    'synthetic_future': synthetic_future,
                    'synthetic_diff': synthetic_diff,
                    'atm_buildup': atm_buildup,
                    'max_pain_strike': max_pain_strike,
                    'distance_from_max_pain': distance_from_max_pain,
                    'call_resistance': call_resistance,
                    'put_support': put_support
                }
            }

        except Exception as e:
            print(f"Error in comprehensive options analysis: {e}")
            return None

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_options_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                except Exception as e:
                    print(f"Error fetching {instrument}: {e}")
        
        return results

# =============================================
# PLOTTING FUNCTIONS
# =============================================

def plot_vob(df: pd.DataFrame, bullish_blocks: List[Dict], bearish_blocks: List[Dict], chart_id: str = "vob_chart") -> go.Figure:
    """Plot Volume Order Blocks on candlestick chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=('Price with Volume Order Blocks', 'Volume'))
    
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                 row=1, col=1)
    
    for i, block in enumerate(bullish_blocks):
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="green", 
                     annotation_text=f"Bull Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="green", row=1, col=1)
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    
    for i, block in enumerate(bearish_blocks):
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="red",
                     annotation_text=f"Bear Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="red", row=1, col=1)
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    
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
# STREAMLIT APP WITH SIMPLIFIED STATE MANAGEMENT
# =============================================

st.set_page_config(page_title="Bias Analysis Pro - Cleaned Version", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Bias Analysis Pro — CLEANED & OPTIMIZED")
st.markdown("**OPTIMIZED VERSION** - Removed duplicates, consolidated utilities, simplified state management")

# Initialize analyzers with shared utilities
analysis = BiasAnalysisPro()
options_analyzer = ConsolidatedOptionsAnalyzer()
vob_indicator = VolumeOrderBlocks(sensitivity=5)

# Initialize single state management
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = AnalysisState()

state = st.session_state.analysis_state

# Sidebar inputs
st.sidebar.header("Data & Symbol")
symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

# Auto-refresh configuration
st.sidebar.header("Auto-Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=1)

# Telegram Configuration
st.sidebar.header("🔔 Telegram Alerts")
telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True) if telegram_notifier.is_configured() else False

# Function to calculate ATM detailed bias score
def calculate_atm_detailed_bias(detailed_bias_data: Dict) -> Tuple[str, float]:
    if not detailed_bias_data:
        return "NEUTRAL", 0
    
    bias_scores = []
    bias_weights = []
    
    bias_mappings = {
        'OI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'ChgOI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'Volume_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Delta_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Gamma_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'Premium_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
        'IV_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'Delta_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
        'Gamma_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
        'IV_Skew_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5}
    }
    
    total_weight = 0
    total_score = 0
    
    for bias_key, mapping in bias_mappings.items():
        if bias_key in detailed_bias_data:
            bias_value = detailed_bias_data[bias_key]
            if bias_value in mapping:
                score = mapping[bias_value]
                weight = mapping['weight']
                total_score += score * weight
                total_weight += weight
    
    if total_weight == 0:
        return "NEUTRAL", 0
    
    normalized_score = (total_score / total_weight) * 100
    
    if normalized_score > 15:
        bias = "BULLISH"
    elif normalized_score < -15:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    
    return bias, normalized_score

# Function to run complete analysis
def run_complete_analysis():
    state.current_analysis['symbol'] = symbol_input
    
    # Technical Analysis
    with st.spinner("Fetching data and running technical analysis..."):
        df_fetched = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
        if df_fetched is None or df_fetched.empty:
            st.error("No data fetched. Check symbol or network.")
            return False

    # Run bias analysis
    with st.spinner("Running full bias analysis..."):
        result = analysis.analyze_all_bias_indicators(symbol_input)
        state.update_technical(result, df_fetched)

    # Run Volume Order Blocks analysis
    with st.spinner("Detecting Volume Order Blocks..."):
        bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df_fetched)
        state.update_vob_blocks(bullish_blocks, bearish_blocks)

    # Run options analysis
    with st.spinner("Running institutional footprint analysis..."):
        enhanced_bias_data = options_analyzer.get_overall_market_bias(force_refresh=True)
        state.update_options(enhanced_bias_data)
    
    # Calculate ATM detailed bias
    if enhanced_bias_data:
        for instrument_data in enhanced_bias_data:
            if instrument_data['instrument'] == 'NIFTY' and 'detailed_atm_bias' in instrument_data:
                atm_bias, atm_score = calculate_atm_detailed_bias(instrument_data['detailed_atm_bias'])
                state.update_atm_detailed({
                    'bias': atm_bias,
                    'score': atm_score,
                    'details': instrument_data['detailed_atm_bias']
                })
                break
    
    # Calculate overall Nifty bias
    calculate_overall_nifty_bias()
    
    # Send Telegram alert if conditions met
    if telegram_enabled:
        send_telegram_alert()
    
    state.mark_refreshed()
    return True

def calculate_overall_nifty_bias():
    bias_scores = []
    bias_weights = []
    
    # Technical Analysis Bias (20% weight)
    tech_result = state.get_technical()
    if tech_result and tech_result.get('success'):
        tech_score = tech_result.get('overall_score', 0)
        bias_scores.append(tech_score)
        bias_weights.append(0.20)
    
    # Enhanced Options Chain Bias (25% weight)
    options_data = state.get_options()
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY':
                options_score = instrument_data.get('combined_score', instrument_data.get('bias_score', 0))
                bias_scores.append(options_score)
                bias_weights.append(0.25)
                break
    
    # Institutional OI BIAS (15% weight)
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY' and 'institutional_analysis' in instrument_data:
                inst_analysis = instrument_data['institutional_analysis']
                inst_bias_analysis = inst_analysis.get('institutional_bias_analysis', {})
                if inst_bias_analysis:
                    inst_score = inst_bias_analysis.get('bias_score', 0)
                    bias_scores.append(inst_score)
                    bias_weights.append(0.15)
                break
    
    # ATM Detailed Bias (10% weight)
    atm_data = state.get_atm_detailed()
    if atm_data:
        atm_score = atm_data['score']
        bias_scores.append(atm_score)
        bias_weights.append(0.10)
    
    # Volume Order Blocks Bias (10% weight)
    vob_blocks = state.get_vob_blocks()
    if vob_blocks:
        bullish_blocks = vob_blocks['bullish']
        bearish_blocks = vob_blocks['bearish']
        vob_score = (len(bullish_blocks) - len(bearish_blocks)) * 10
        vob_score = max(-100, min(100, vob_score))
        bias_scores.append(vob_score)
        bias_weights.append(0.10)
    
    # Breakout/Reversal Analysis (20% weight)
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY' and 'breakout_reversal_analysis' in instrument_data:
                breakout_data = instrument_data['breakout_reversal_analysis']
                breakout_score = breakout_data.get('overall_score', 0)
                bias_scores.append(breakout_score)
                bias_weights.append(0.20)
                break
    
    # Calculate weighted average
    if bias_scores and bias_weights:
        total_weight = sum(bias_weights)
        weighted_score = sum(score * weight for score, weight in zip(bias_scores, bias_weights)) / total_weight
        overall_bias = TradingUtilities.determine_bias_from_score(weighted_score)
        state.update_overall_bias(overall_bias, weighted_score)

def send_telegram_alert():
    if not telegram_enabled:
        return
    
    technical_bias = "NEUTRAL"
    options_bias = "NEUTRAL" 
    atm_bias = "NEUTRAL"
    
    tech_result = state.get_technical()
    if tech_result and tech_result.get('success'):
        technical_bias = tech_result.get('overall_bias', 'NEUTRAL')
    
    options_data = state.get_options()
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY':
                options_bias = instrument_data.get('combined_bias', instrument_data.get('overall_bias', 'NEUTRAL'))
                break
    
    atm_data = state.get_atm_detailed()
    if atm_data:
        atm_bias = atm_data['bias']
    
    overall_bias, overall_score = state.get_overall_bias()
    telegram_notifier.send_bias_alert(
        technical_bias=technical_bias,
        options_bias=options_bias,
        atm_bias=atm_bias,
        overall_bias=overall_bias,
        score=overall_score
    )

# AUTO-RUN ANALYSIS ON STARTUP
if state.current_analysis['technical'] is None:
    with st.spinner("🚀 Starting optimized initial analysis..."):
        if run_complete_analysis():
            st.success("✅ Optimized initial analysis complete!")
            st.rerun()

# Refresh button
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
        if run_complete_analysis():
            st.sidebar.success("Analysis refreshed!")
            st.rerun()
with col2:
    st.sidebar.metric("Auto-Refresh", "ON" if auto_refresh else "OFF")

# Auto-refresh logic
if auto_refresh and state.should_refresh(refresh_interval):
    with st.spinner("Auto-refreshing analysis..."):
        if run_complete_analysis():
            st.rerun()

# Display overall Nifty bias
st.sidebar.markdown("---")
st.sidebar.header("Overall Nifty Bias")
overall_bias, overall_score = state.get_overall_bias()
if overall_bias:
    bias_color = "🟢" if "BULL" in overall_bias.upper() else "🔴" if "BEAR" in overall_bias.upper() else "🟡"
    st.sidebar.metric(
        "NIFTY 50 Bias",
        f"{bias_color} {overall_bias}",
        f"Score: {overall_score:.1f}"
    )

# Display last update time
if state.current_analysis['timestamp']:
    st.sidebar.caption(f"Last update: {state.current_analysis['timestamp'].strftime('%H:%M:%S')} IST")

# Enhanced tabs
tabs = st.tabs([
    "Overall Bias", "Bias Summary", "Price Action", "Option Chain", "Bias Tabulation"
])

# OVERALL BIAS TAB
with tabs[0]:
    st.header("🎯 Optimized Overall Nifty Bias Analysis")
    
    overall_bias, overall_score = state.get_overall_bias()
    if not overall_bias:
        st.info("No analysis run yet. Analysis runs automatically...")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if "STRONG BULL" in overall_bias.upper():
                st.success(f"## 🟢 {overall_bias}")
                st.metric("Bias Score", f"{overall_score:.1f}", delta="Strong Bullish", delta_color="normal")
            elif "BULL" in overall_bias.upper():
                st.success(f"## 🟢 {overall_bias}")
                st.metric("Bias Score", f"{overall_score:.1f}", delta="Bullish")
            elif "STRONG BEAR" in overall_bias.upper():
                st.error(f"## 🔴 {overall_bias}")
                st.metric("Bias Score", f"{overall_score:.1f}", delta="Strong Bearish", delta_color="inverse")
            elif "BEAR" in overall_bias.upper():
                st.error(f"## 🔴 {overall_bias}")
                st.metric("Bias Score", f"{overall_score:.1f}", delta="Bearish", delta_color="inverse")
            else:
                st.warning(f"## 🟡 {overall_bias}")
                st.metric("Bias Score", f"{overall_score:.1f}")
        
        st.markdown("---")
        
        # Trading Recommendation
        st.subheader("📈 Trading Recommendation")
        
        if "STRONG BULL" in overall_bias.upper():
            st.success("""
            **Recommended Action:** STRONG LONG positions
            - Aggressive buying on dips
            - Strong support levels expected to hold
            - Target higher resistance levels
            """)
        elif "BULL" in overall_bias.upper():
            st.success("""
            **Recommended Action:** Consider LONG positions
            - Look for buying opportunities on dips
            - Support levels are likely to hold
            """)
        elif "STRONG BEAR" in overall_bias.upper():
            st.error("""
            **Recommended Action:** STRONG SHORT positions  
            - Aggressive selling on rallies
            - Strong resistance levels expected to hold
            """)
        elif "BEAR" in overall_bias.upper():
            st.error("""
            **Recommended Action:** Consider SHORT positions  
            - Look for selling opportunities on rallies
            - Resistance levels are likely to hold
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
    st.header("Technical Bias Summary")
    tech_result = state.get_technical()
    if tech_result is None:
        st.info("No analysis run yet. Analysis runs automatically...")
    else:
        if not tech_result.get('success', False):
            st.error(f"Analysis failed: {tech_result.get('error')}")
        else:
            st.markdown(f"**Symbol:** `{tech_result['symbol']}`")
            st.markdown(f"**Timestamp (IST):** {tech_result['timestamp']}")
            st.metric("Current Price", f"{tech_result['current_price']:.2f}")
            st.metric("Technical Bias", tech_result['overall_bias'], delta=f"Confidence: {tech_result['overall_confidence']:.1f}%")

            bias_table = pd.DataFrame(tech_result['bias_results'])
            cols_order = ['indicator', 'value', 'bias', 'score', 'weight', 'category']
            bias_table = bias_table[cols_order]
            bias_table.columns = [c.capitalize() for c in bias_table.columns]
            st.subheader("Indicator-level Biases")
            st.dataframe(bias_table, use_container_width=True)

            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Bullish Indicators", int(tech_result['bullish_count']))
            col2.metric("Bearish Indicators", int(tech_result['bearish_count']))
            col3.metric("Neutral Indicators", int(tech_result['neutral_count']))

# PRICE ACTION TAB
with tabs[2]:
    st.header("📈 Price Action Analysis")
    
    tech_result = state.get_technical()
    if tech_result is None:
        st.info("No data loaded yet. Analysis runs automatically...")
    else:
        st.subheader("Price Chart with Volume Order Blocks")
        
        vob_blocks = state.get_vob_blocks()
        bullish_blocks = vob_blocks['bullish']
        bearish_blocks = vob_blocks['bearish']
        
        # For demonstration - in real app you'd use the actual dataframe
        st.info("Chart would display here with Volume Order Blocks")
        
        st.subheader("Volume Order Blocks Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Blocks", len(bullish_blocks))
            if bullish_blocks:
                latest_bullish = bullish_blocks[-1]
                st.write(f"Latest Bullish Block:")
                st.write(f"- Upper: ₹{latest_bullish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bullish['lower']:.2f}")
        
        with col2:
            st.metric("Bearish Blocks", len(bearish_blocks))
            if bearish_blocks:
                latest_bearish = bearish_blocks[-1]
                st.write(f"Latest Bearish Block:")
                st.write(f"- Upper: ₹{latest_bearish['upper']:.2f}")
                st.write(f"- Lower: ₹{latest_bearish['lower']:.2f}")

# OPTION CHAIN TAB
with tabs[3]:
    st.header("📊 Optimized NSE Options Chain Analysis")
    
    options_data = state.get_options()
    if not options_data:
        st.info("No option chain data available. Analysis runs automatically...")
    else:
        st.subheader("🎯 Market Bias Summary")
        cols = st.columns(len(options_data))
        for idx, instrument_data in enumerate(options_data):
            with cols[idx]:
                bias_to_show = instrument_data.get('combined_bias', instrument_data.get('overall_bias', 'Neutral'))
                score_to_show = instrument_data.get('combined_score', instrument_data.get('bias_score', 0))
                
                bias_color = "🟢" if "Bullish" in bias_to_show else "🔴" if "Bearish" in bias_to_show else "🟡"
                st.metric(
                    f"{instrument_data['instrument']}",
                    f"{bias_color} {bias_to_show}",
                    f"Score: {score_to_show:.1f}"
                )

# BIAS TABULATION TAB
with tabs[4]:
    st.header("📋 Optimized Comprehensive Bias Tabulation")
    
    options_data = state.get_options()
    if not options_data:
        st.info("No option chain data available. Analysis runs automatically...")
    else:
        for instrument_idx, instrument_data in enumerate(options_data):
            with st.expander(f"🎯 {instrument_data['instrument']} - Complete Bias Analysis", expanded=True):
                
                st.subheader("📊 Basic Information")
                basic_info = pd.DataFrame({
                    'Metric': [
                        'Instrument', 'Spot Price', 'ATM Strike', 'Overall Bias', 
                        'Bias Score', 'PCR OI', 'PCR Change OI'
                    ],
                    'Value': [
                        instrument_data['instrument'],
                        f"₹{instrument_data['spot_price']:.2f}",
                        f"₹{instrument_data['atm_strike']:.2f}",
                        instrument_data['overall_bias'],
                        f"{instrument_data['bias_score']:.1f}",
                        f"{instrument_data['pcr_oi']:.2f}",
                        f"{instrument_data['pcr_change']:.2f}"
                    ]
                })
                st.dataframe(basic_info, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("✅ OPTIMIZED BiasAnalysisPro — Removed duplicates, consolidated utilities, simplified state management")
st.caption("🔔 Telegram alerts sent when Technical Analysis, Options Chain, and ATM Detailed Bias are aligned")

# Display optimized weights in sidebar
st.sidebar.markdown("**Optimized Bias Weights:**")
st.sidebar.markdown("- Technical: 20%")
st.sidebar.markdown("- Options Chain: 25%")
st.sidebar.markdown("- Institutional OI: 15%")
st.sidebar.markdown("- ATM Detailed: 10%")
st.sidebar.markdown("- Volume Blocks: 10%")
st.sidebar.markdown("- Breakout/Reversal: 20%")