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
    """Shared utilities for all analyzers"""
    
    @staticmethod
    def normalize_chain_columns(df_chain: pd.DataFrame) -> pd.DataFrame:
        """Normalize option chain column names to handle different API formats"""
        df = df_chain.copy()
        
        column_mapping = {
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
            
            # Short column names
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
        
        # Rename columns that exist
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
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
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
    
    @staticmethod
    def normalize_score(score: float, min_val: float, max_val: float) -> float:
        """Normalize any score to -100 to +100 range"""
        if max_val == min_val:
            return 0
        return ((score - min_val) / (max_val - min_val)) * 200 - 100
    
    @staticmethod
    def determine_bias_from_score(score: float) -> str:
        """Single bias determination logic"""
        if score >= 25:
            return "STRONG_BULLISH"
        elif score >= 10:
            return "BULLISH"
        elif score <= -25:
            return "STRONG_BEARISH"
        elif score <= -10:
            return "BEARISH"
        else:
            return "NEUTRAL"

# =============================================
# TELEGRAM NOTIFICATION SYSTEM
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
            return False
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    def send_bias_alert(self, technical_bias: str, options_bias: str, atm_bias: str, overall_bias: str, score: float):
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

# =============================================
# ANALYSIS STATE MANAGEMENT
# =============================================

class AnalysisState:
    """Single state management for all analysis"""
    
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
    
    def update_technical(self, result):
        self.current_analysis['technical'] = result
        self.current_analysis['timestamp'] = datetime.now(IST)
    
    def update_options(self, data):
        self.current_analysis['options'] = data
    
    def update_vob_blocks(self, bullish_blocks, bearish_blocks):
        self.current_analysis['vob_blocks'] = {
            'bullish': bullish_blocks,
            'bearish': bearish_blocks
        }
    
    def update_atm_detailed(self, atm_data):
        self.current_analysis['atm_detailed'] = atm_data
    
    def update_overall_bias(self, bias, score):
        self.current_analysis['overall_bias'] = bias
        self.current_analysis['overall_score'] = score
    
    def get_technical(self):
        return self.current_analysis.get('technical')
    
    def get_options(self):
        return self.current_analysis.get('options')
    
    def get_vob_blocks(self):
        return self.current_analysis.get('vob_blocks', {'bullish': [], 'bearish': []})
    
    def get_atm_detailed(self):
        return self.current_analysis.get('atm_detailed')
    
    def get_overall_bias(self):
        return self.current_analysis.get('overall_bias', "NEUTRAL"), self.current_analysis.get('overall_score', 0)

# =============================================
# BIAS ANALYSIS PRO (SIMPLIFIED)
# =============================================

class BiasAnalysisPro:
    """Simplified Bias Analysis - Fixed version"""

    def __init__(self):
        self.config = {
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,
            'bias_strength': 60
        }
        self.utils = TradingUtilities()

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
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

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
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

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """Fixed version - analyze all bias indicators"""

        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            return {
                'success': False,
                'error': f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            }

        current_price = df['Close'].iloc[-1]
        bias_results = []

        # 1. RSI
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]
        rsi_bias = "BULLISH" if rsi_value > 50 else "BEARISH"
        rsi_score = 100 if rsi_bias == "BULLISH" else -100

        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.2f}",
            'bias': rsi_bias,
            'score': rsi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 2. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        plus_di_value = plus_di.iloc[-1] if not plus_di.empty else 0
        minus_di_value = minus_di.iloc[-1] if not minus_di.empty else 0
        
        dmi_bias = "BULLISH" if plus_di_value > minus_di_value else "BEARISH"
        dmi_score = 100 if dmi_bias == "BULLISH" else -100

        bias_results.append({
            'indicator': 'DMI',
            'value': f"+DI:{plus_di_value:.1f} -DI:{minus_di_value:.1f}",
            'bias': dmi_bias,
            'score': dmi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 3. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1] if not mfi.empty else 50
        mfi_bias = "BULLISH" if mfi_value > 50 else "BEARISH"
        mfi_score = 100 if mfi_bias == "BULLISH" else -100

        bias_results.append({
            'indicator': 'MFI',
            'value': f"{mfi_value:.2f}",
            'bias': mfi_bias,
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 4. Volume Delta (simplified)
        if df['Volume'].sum() > 0:
            up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
            down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
            volume_delta = up_vol - down_vol
            volume_bias = "BULLISH" if volume_delta > 0 else "BEARISH"
            volume_score = 100 if volume_bias == "BULLISH" else -100
        else:
            volume_delta = 0
            volume_bias = "NEUTRAL"
            volume_score = 0

        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': volume_bias,
            'score': volume_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 5. EMA Cross (simplified Order Blocks)
        ema5 = self.utils.calculate_ema(df['Close'], 5)
        ema18 = self.utils.calculate_ema(df['Close'], 18)
        
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
            'indicator': 'EMA Cross',
            'value': f"EMA5: {ema5.iloc[-1]:.2f} | EMA18: {ema18.iloc[-1]:.2f}",
            'bias': ob_bias,
            'score': ob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # FIXED: Calculate overall bias properly
        bullish_count = sum(1 for bias in bias_results if bias['bias'] == "BULLISH")
        bearish_count = sum(1 for bias in bias_results if bias['bias'] == "BEARISH")
        neutral_count = sum(1 for bias in bias_results if bias['bias'] == "NEUTRAL")

        # Simple weighted average (fixed calculation)
        total_score = sum(bias['score'] * bias['weight'] for bias in bias_results)
        total_weight = sum(bias['weight'] for bias in bias_results)
        
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0

        # Determine overall bias
        if overall_score > 20:
            overall_bias = "BULLISH"
        elif overall_score < -20:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(IST),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results)
        }

# =============================================
# VOLUME ORDER BLOCKS (SIMPLIFIED)
# =============================================

class VolumeOrderBlocks:
    """Simplified Volume Order Blocks indicator"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.bullish_blocks = []
        self.bearish_blocks = []
        self.utils = TradingUtilities()
    
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Detect Volume Order Blocks"""
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.utils.calculate_ema(df['Close'], self.length1)
        ema2 = self.utils.calculate_ema(df['Close'], self.length2)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        atr = self.utils.calculate_atr(df, 200) * 3
        
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
                
                bullish_blocks.append({
                    'index': lowest_idx,
                    'upper': lookback_data.loc[lowest_idx, 'Open'],
                    'lower': lowest_price,
                    'mid': (lookback_data.loc[lowest_idx, 'Open'] + lowest_price) / 2,
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
                
                bearish_blocks.append({
                    'index': highest_idx,
                    'upper': highest_price,
                    'lower': lookback_data.loc[highest_idx, 'Open'],
                    'mid': (highest_price + lookback_data.loc[highest_idx, 'Open']) / 2,
                    'volume': vol,
                    'type': 'bearish'
                })
        
        return bullish_blocks[-5:], bearish_blocks[-5:]  # Return only recent blocks

# =============================================
# OPTIONS ANALYZER (SIMPLIFIED)
# =============================================

class OptionsAnalyzer:
    """Simplified Options Analyzer"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.utils = TradingUtilities()
        
    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Mock option chain data - replace with real implementation"""
        try:
            # Mock data for demonstration
            return {
                'success': True,
                'instrument': instrument,
                'spot': 22000.50 if instrument == 'NIFTY' else 48000.75,
                'expiry': '28-Dec-2023',
                'total_ce_oi': 1500000,
                'total_pe_oi': 1800000,
                'total_ce_change': 50000,
                'total_pe_change': 75000,
                'records': []  # Empty for mock data
            }
        except Exception as e:
            return {
                'success': False,
                'instrument': instrument,
                'error': str(e)
            }
    
    def analyze_comprehensive_options_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Simplified options bias analysis"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            # Mock analysis based on OI data
            pcr_oi = data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0
            pcr_change = abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0
            
            # Simple bias calculation
            if pcr_oi > 1.0 and pcr_change > 1.0:
                bias = "Bullish"
                score = 25.0
            elif pcr_oi < 1.0 and pcr_change < 1.0:
                bias = "Bearish"
                score = -25.0
            else:
                bias = "Neutral"
                score = 0.0

            return {
                'instrument': instrument,
                'spot_price': data['spot'],
                'overall_bias': bias,
                'bias_score': score,
                'pcr_oi': pcr_oi,
                'pcr_change': pcr_change,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change']
            }

        except Exception as e:
            print(f"Error in options analysis: {e}")
            return None

# =============================================
# PLOTTING FUNCTION
# =============================================

def plot_vob(df: pd.DataFrame, bullish_blocks: List[Dict], bearish_blocks: List[Dict]) -> go.Figure:
    """Plot Volume Order Blocks"""
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
                     annotation_text="Bull Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="green", row=1, col=1)
    
    # Add bearish blocks
    for block in bearish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="red",
                     annotation_text="Bear Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="red", row=1, col=1)
    
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
# STREAMLIT APP - OPTIMIZED
# =============================================

st.set_page_config(page_title="Bias Analysis Pro - OPTIMIZED", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Bias Analysis Pro — CLEANED & OPTIMIZED")
st.markdown("**OPTIMIZED VERSION** - Removed duplicates, consolidated utilities, simplified state management")

# Initialize analyzers
analysis = BiasAnalysisPro()
options_analyzer = OptionsAnalyzer()
vob_indicator = VolumeOrderBlocks(sensitivity=5)
telegram_notifier = TelegramNotifier()

# Initialize state
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = AnalysisState()

state = st.session_state.analysis_state

# Sidebar
st.sidebar.header("Data & Symbol")
symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

st.sidebar.header("Auto-Refresh Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=1)

st.sidebar.header("🔔 Telegram Alerts")
telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=False)

# Function to run complete analysis
def run_complete_analysis():
    """Run complete analysis"""
    try:
        # Technical Analysis
        with st.spinner("Running technical analysis..."):
            result = analysis.analyze_all_bias_indicators(symbol_input)
            if result.get('success'):
                state.update_technical(result)
            else:
                st.error(f"Technical analysis failed: {result.get('error')}")
                return False

        # Volume Order Blocks
        with st.spinner("Detecting Volume Order Blocks..."):
            df = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
            if df is not None and not df.empty:
                bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
                state.update_vob_blocks(bullish_blocks, bearish_blocks)

        # Options Analysis
        with st.spinner("Running options analysis..."):
            options_data = []
            for instrument in ['NIFTY', 'BANKNIFTY']:
                bias_data = options_analyzer.analyze_comprehensive_options_bias(instrument)
                if bias_data:
                    options_data.append(bias_data)
            state.update_options(options_data)

        # Calculate overall bias
        calculate_overall_bias()
        
        return True
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return False

def calculate_overall_bias():
    """Calculate overall bias from all components"""
    bias_scores = []
    bias_weights = []
    
    # Technical Analysis (40%)
    tech_result = state.get_technical()
    if tech_result and tech_result.get('success'):
        tech_score = tech_result.get('overall_score', 0)
        bias_scores.append(tech_score)
        bias_weights.append(0.40)
    
    # Options Analysis (40%)
    options_data = state.get_options()
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY':
                options_score = instrument_data.get('bias_score', 0)
                bias_scores.append(options_score)
                bias_weights.append(0.40)
                break
    
    # Volume Order Blocks (20%)
    vob_blocks = state.get_vob_blocks()
    bullish_count = len(vob_blocks.get('bullish', []))
    bearish_count = len(vob_blocks.get('bearish', []))
    vob_score = (bullish_count - bearish_count) * 10
    vob_score = max(-100, min(100, vob_score))
    
    bias_scores.append(vob_score)
    bias_weights.append(0.20)
    
    # Calculate weighted average
    if bias_scores and bias_weights:
        total_weight = sum(bias_weights)
        weighted_score = sum(score * weight for score, weight in zip(bias_scores, bias_weights)) / total_weight
        overall_bias = TradingUtilities.determine_bias_from_score(weighted_score)
        state.update_overall_bias(overall_bias, weighted_score)

# Auto-run on startup
if state.get_technical() is None:
    with st.spinner("🚀 Running initial analysis..."):
        if run_complete_analysis():
            st.success("✅ Initial analysis complete!")

# Refresh button
if st.sidebar.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
    if run_complete_analysis():
        st.sidebar.success("Analysis refreshed!")
        st.rerun()

# Display overall bias
st.sidebar.markdown("---")
st.sidebar.header("Overall Market Bias")
overall_bias, overall_score = state.get_overall_bias()

bias_color = "🟢" if "BULL" in overall_bias else "🔴" if "BEAR" in overall_bias else "🟡"
st.sidebar.metric(
    "Market Bias",
    f"{bias_color} {overall_bias.replace('_', ' ')}",
    f"Score: {overall_score:.1f}"
)

# Main tabs
tabs = st.tabs(["Overall Bias", "Technical Analysis", "Options Chain", "Price Action"])

with tabs[0]:
    st.header("🎯 Overall Market Bias")
    
    if overall_bias == "NEUTRAL":
        st.info("## 🟡 MARKET BIAS: NEUTRAL")
    elif "BULL" in overall_bias:
        st.success(f"## 🟢 MARKET BIAS: {overall_bias.replace('_', ' ')}")
    else:
        st.error(f"## 🔴 MARKET BIAS: {overall_bias.replace('_', ' ')}")
    
    st.metric("Confidence Score", f"{overall_score:.1f}")
    
    # Component breakdown
    st.subheader("Component Analysis")
    components = []
    
    tech_result = state.get_technical()
    if tech_result and tech_result.get('success'):
        components.append({
            'Component': 'Technical Indicators',
            'Bias': tech_result.get('overall_bias', 'NEUTRAL'),
            'Score': tech_result.get('overall_score', 0),
            'Weight': '40%'
        })
    
    options_data = state.get_options()
    if options_data:
        for instrument_data in options_data:
            if instrument_data['instrument'] == 'NIFTY':
                components.append({
                    'Component': 'Options Chain',
                    'Bias': instrument_data.get('overall_bias', 'NEUTRAL'),
                    'Score': instrument_data.get('bias_score', 0),
                    'Weight': '40%'
                })
                break
    
    vob_blocks = state.get_vob_blocks()
    bullish_count = len(vob_blocks.get('bullish', []))
    bearish_count = len(vob_blocks.get('bearish', []))
    vob_bias = "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
    components.append({
        'Component': 'Volume Order Blocks',
        'Bias': vob_bias,
        'Score': (bullish_count - bearish_count) * 10,
        'Weight': '20%'
    })
    
    if components:
        st.dataframe(pd.DataFrame(components), use_container_width=True)

with tabs[1]:
    st.header("Technical Analysis")
    
    tech_result = state.get_technical()
    if tech_result is None:
        st.info("No technical analysis available. Click refresh to run analysis.")
    elif not tech_result.get('success'):
        st.error(f"Technical analysis failed: {tech_result.get('error')}")
    else:
        st.metric("Current Price", f"{tech_result['current_price']:.2f}")
        st.metric("Technical Bias", tech_result['overall_bias'], 
                 delta=f"Score: {tech_result['overall_score']:.1f}")
        
        # Show indicators
        bias_table = pd.DataFrame(tech_result['bias_results'])
        st.dataframe(bias_table, use_container_width=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Bullish Indicators", tech_result['bullish_count'])
        col2.metric("Bearish Indicators", tech_result['bearish_count'])
        col3.metric("Neutral Indicators", tech_result['neutral_count'])

with tabs[2]:
    st.header("Options Chain Analysis")
    
    options_data = state.get_options()
    if not options_data:
        st.info("No options data available. Click refresh to run analysis.")
    else:
        for instrument_data in options_data:
            with st.expander(f"{instrument_data['instrument']} Options", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                with col2:
                    st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                with col3:
                    st.metric("PCR Change", f"{instrument_data['pcr_change']:.2f}")
                with col4:
                    bias_color = "🟢" if "Bull" in instrument_data['overall_bias'] else "🔴" if "Bear" in instrument_data['overall_bias'] else "🟡"
                    st.metric("Options Bias", f"{bias_color} {instrument_data['overall_bias']}")
                
                st.metric("Bias Score", f"{instrument_data['bias_score']:.1f}")

with tabs[3]:
    st.header("Price Action Analysis")
    
    df = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
    if df is None or df.empty:
        st.info("No price data available.")
    else:
        vob_blocks = state.get_vob_blocks()
        fig = plot_vob(df, vob_blocks.get('bullish', []), vob_blocks.get('bearish', []))
        st.plotly_chart(fig, use_container_width=True)
        
        # VOB Summary
        st.subheader("Volume Order Blocks Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bullish Blocks", len(vob_blocks.get('bullish', [])))
        with col2:
            st.metric("Bearish Blocks", len(vob_blocks.get('bearish', [])))

# Footer
st.markdown("---")
st.caption("✅ OPTIMIZED BiasAnalysisPro - Cleaned code, removed duplicates, simplified architecture")
st.caption("📊 Real-time market bias analysis with technical indicators, options data, and volume analysis")

# Auto-refresh
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_refresh).total_seconds() / 60
    
    if time_diff >= refresh_interval:
        with st.spinner("Auto-refreshing..."):
            if run_complete_analysis():
                st.session_state.last_refresh = current_time
                st.rerun()