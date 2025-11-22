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

    def get_current_time_ist(self):
        """Get current time in IST"""
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX from Yahoo Finance"""
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
                    'timestamp': self.get_current_time_ist()
                }
        except Exception as e:
            pass

        return {'success': False, 'error': 'India VIX data not available'}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices from Yahoo Finance"""
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

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices from Yahoo Finance"""
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
        """Fetch intermarket data (commodities, currencies, bonds)"""
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
        """Fetch all enhanced market data from all sources"""
        print("Fetching enhanced market data...")

        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'intermarket': [],
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

        # 5. Analyze Sector Rotation
        print("  - Analyzing Sector Rotation...")
        result['sector_rotation'] = self.analyze_sector_rotation()

        # 6. Analyze Intraday Seasonality
        print("  - Analyzing Intraday Seasonality...")
        result['intraday_seasonality'] = self.analyze_intraday_seasonality()

        # 7. Calculate summary statistics
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
    # DATA FETCHING
    # =========================================================================

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Dhan API (for Indian indices) or Yahoo Finance (for others)
        Note: Yahoo Finance limits intraday data - use 7d max for 5m interval
        """
        # Check if this is an Indian index that needs Dhan API
        indian_indices = {'^NSEI': 'NIFTY', '^BSESN': 'SENSEX', '^NSEBANK': 'BANKNIFTY'}

        if symbol in indian_indices:
            try:
                # Use Dhan API for Indian indices to get proper volume data
                dhan_instrument = indian_indices[symbol]
                
                # For now, fallback to Yahoo Finance since Dhan API import is not available
                # In production, you would use the actual Dhan API here
                print(f"⚠️  Dhan API not available, using Yahoo Finance for {symbol}")
            except Exception as e:
                print(f"Error with Dhan API for {symbol}: {e}, falling back to yfinance")

        # Fallback to Yahoo Finance for all symbols
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
    # ENHANCED INDICATORS (KEPT FOR COMPATIBILITY)
    # =========================================================================

    def calculate_volatility_ratio(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volatility Ratio"""
        atr = self.calculate_atr(df, length)
        stdev = df['Close'].rolling(window=length).std()
        volatility_ratio = (stdev / atr) * 100

        high_volatility = volatility_ratio.iloc[-1] > self.config['volatility_threshold']
        low_volatility = volatility_ratio.iloc[-1] < (self.config['volatility_threshold'] * 0.5)

        return volatility_ratio, high_volatility, low_volatility

    def calculate_volume_roc(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volume Rate of Change with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral volume ROC if no volume data
            neutral_roc = pd.Series([0.0] * len(df), index=df.index)
            return neutral_roc, False, False

        # Avoid division by zero
        volume_shifted = df['Volume'].shift(length).replace(0, np.nan)
        volume_roc = ((df['Volume'] - df['Volume'].shift(length)) / volume_shifted) * 100

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
        if df['Volume'].sum() == 0:
            # Return neutral OBV if no volume data
            neutral_obv = pd.Series([0.0] * len(df), index=df.index)
            neutral_obv_ma = pd.Series([0.0] * len(df), index=df.index)
            return neutral_obv, neutral_obv_ma, False, False

        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
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
        if df['Volume'].sum() == 0:
            # Return neutral force index if no volume data
            neutral_force = pd.Series([0.0] * len(df), index=df.index)
            return neutral_force, False, False

        force_index = (df['Close'] - df['Close'].shift(1)) * df['Volume']
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
        price_roc = ((df['Close'] - df['Close'].shift(length)) / df['Close'].shift(length)) * 100

        price_momentum_bullish = price_roc.iloc[-1] > 0
        price_momentum_bearish = price_roc.iloc[-1] < 0

        return price_roc, price_momentum_bullish, price_momentum_bearish

    def calculate_choppiness_index(self, df: pd.DataFrame, period: int = 14):
        """Calculate Choppiness Index"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        sum_true_range = true_range.rolling(window=period).sum()
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()

        ci = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(period)

        market_chopping = ci.iloc[-1] > self.config['ci_high_threshold']
        market_trending = ci.iloc[-1] < self.config['ci_low_threshold']

        return ci, market_chopping, market_trending

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 30):
        """Detect RSI/MACD Divergences"""
        rsi = self.calculate_rsi(df['Close'], 14)

        # MACD
        macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

        close_series = df['Close'].tail(lookback)
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
{strength_emoji} {emoji} *TRADING SIGNAL ALERT* {emoji} {strength_emoji}

🎯 *{recommendation['instrument']} - {recommendation['signal_type']}*
⏰ Time: {recommendation['timestamp'].strftime('%H:%M:%S')} IST
📊 Confidence: {recommendation['confidence']}%

💰 Current Price: ₹{recommendation['spot_price']:.2f}
📈 Bias Score: {recommendation['bias_score']:.2f}
🔢 PCR OI: {recommendation['pcr_oi']:.2f}

🎯 *TRADING PLAN:*
• Entry Zone: ₹{recommendation['entry_zone']}
• Target 1: ₹{recommendation['targets'][0]}
• Target 2: ₹{recommendation['targets'][1]}
• Stop Loss: ₹{recommendation['stop_loss']}

📊 *KEY LEVELS:*
• Call Resistance: ₹{recommendation['call_resistance']}
• Put Support: ₹{recommendation['put_support']}
• Max Pain: ₹{recommendation['max_pain']}

🔍 *CONFIRMING METRICS:*
• Synthetic Bias: {recommendation['key_metrics']['synthetic_bias']}
• ATM Buildup: {recommendation['key_metrics']['atm_buildup']}
• Vega Bias: {recommendation['key_metrics']['vega_bias']}

⏳ *Next signal in {self.cooldown_minutes} minutes*

⚠️ *Risk Disclaimer: Trade at your own risk. Use proper position sizing and risk management.*
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
        self.market_data_fetcher = EnhancedMarketData()  # NEW: Enhanced market data
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'sent_volume_block_alerts' not in st.session_state:
            st.session_state.sent_volume_block_alerts = set()
        if 'sent_volume_spike_alerts' not in st.session_state:
            st.session_state.sent_volume_spike_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        if 'volume_history' not in st.session_state:
            st.session_state.volume_history = []
        if 'market_bias_data' not in st.session_state:
            st.session_state.market_bias_data = None
        if 'last_bias_update' not in st.session_state:
            st.session_state.last_bias_update = None
        if 'last_signal_check' not in st.session_state:
            st.session_state.last_signal_check = None
        if 'sent_trading_signals' not in st.session_state:
            st.session_state.sent_trading_signals = {}
        if 'comprehensive_bias_data' not in st.session_state:
            st.session_state.comprehensive_bias_data = None
        if 'last_comprehensive_bias_update' not in st.session_state:
            st.session_state.last_comprehensive_bias_update = None
        if 'enhanced_market_data' not in st.session_state:  # NEW: Enhanced market data
            st.session_state.enhanced_market_data = None
        if 'last_market_data_update' not in st.session_state:  # NEW: Enhanced market data
            st.session_state.last_market_data_update = None
    
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

    # NEW: Enhanced Market Data Display Methods
    def display_enhanced_market_data(self):
        """Display comprehensive enhanced market data"""
        st.header("🌍 Enhanced Market Data Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Comprehensive market analysis from multiple sources including India VIX, global markets, sector rotation, and intermarket analysis")
        with col2:
            if st.button("🔄 Update Market Data", type="primary"):
                with st.spinner("Fetching comprehensive market data..."):
                    try:
                        market_data = self.market_data_fetcher.fetch_all_enhanced_data()
                        st.session_state.enhanced_market_data = market_data
                        st.session_state.last_market_data_update = datetime.now(self.ist)
                        st.success("Market data updated successfully!")
                    except Exception as e:
                        st.error(f"Error fetching market data: {str(e)}")
        
        st.divider()
        
        if st.session_state.last_market_data_update:
            st.write(f"Last update: {st.session_state.last_market_data_update.strftime('%H:%M:%S')} IST")
        
        if st.session_state.enhanced_market_data:
            market_data = st.session_state.enhanced_market_data
            
            # Overall Summary
            st.subheader("📊 Market Summary")
            summary = market_data['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Sentiment", summary['overall_sentiment'])
            with col2:
                st.metric("Average Score", f"{summary['avg_score']:.1f}")
            with col3:
                st.metric("Bullish Signals", summary['bullish_count'])
            with col4:
                st.metric("Total Data Points", summary['total_data_points'])
            
            st.divider()
            
            # Create tabs for different market data categories
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🇮🇳 India VIX", "📈 Sector Analysis", "🌍 Global Markets", 
                "🔄 Intermarket", "📊 Sector Rotation", "⏰ Intraday Timing"
            ])
            
            with tab1:
                self.display_india_vix_data(market_data['india_vix'])
            
            with tab2:
                self.display_sector_data(market_data['sector_indices'])
            
            with tab3:
                self.display_global_markets(market_data['global_markets'])
            
            with tab4:
                self.display_intermarket_data(market_data['intermarket'])
            
            with tab5:
                self.display_sector_rotation(market_data['sector_rotation'])
            
            with tab6:
                self.display_intraday_seasonality(market_data['intraday_seasonality'])
            
        else:
            st.info("👆 Click 'Update Market Data' to load comprehensive market analysis")
            st.write("""
            **Data Sources Included:**
            - **India VIX**: Market volatility and fear index
            - **Sector Indices**: Nifty sector performance and rotation
            - **Global Markets**: International market performance
            - **Intermarket Analysis**: Commodities, currencies, bonds
            - **Sector Rotation**: Market leadership analysis
            - **Intraday Seasonality**: Time-based market patterns
            """)

    def display_india_vix_data(self, vix_data: Dict[str, Any]):
        """Display India VIX data"""
        if not vix_data.get('success'):
            st.error("India VIX data not available")
            return
        
        st.subheader("🇮🇳 India VIX - Fear Index")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VIX Value", f"{vix_data['value']:.2f}")
        with col2:
            st.metric("Sentiment", vix_data['sentiment'])
        with col3:
            st.metric("Bias", vix_data['bias'])
        with col4:
            st.metric("Score", vix_data['score'])
        
        # VIX Interpretation
        st.info(f"**Interpretation**: {vix_data['sentiment']} - {self.get_vix_interpretation(vix_data['value'])}")
        st.write(f"**Source**: {vix_data['source']} | **Timestamp**: {vix_data['timestamp'].strftime('%H:%M:%S')}")

    def get_vix_interpretation(self, vix_value: float) -> str:
        """Get VIX interpretation text"""
        if vix_value > 25:
            return "Extreme fear, potential market bottom"
        elif vix_value > 20:
            return "Elevated fear, high volatility expected"
        elif vix_value > 15:
            return "Moderate volatility, normal market conditions"
        elif vix_value > 12:
            return "Low volatility, complacency setting in"
        else:
            return "Very low volatility, potential for spike"

    def display_sector_data(self, sectors: List[Dict[str, Any]]):
        """Display sector indices data"""
        st.subheader("📈 Nifty Sector Performance")
        
        if not sectors:
            st.info("No sector data available")
            return
        
        # Create sector performance table
        sector_df = pd.DataFrame(sectors)
        sector_df = sector_df.sort_values('change_pct', ascending=False)
        
        # Display as metrics
        cols = st.columns(4)
        for idx, sector in enumerate(sector_df.head(8).itertuples()):
            with cols[idx % 4]:
                color = "🟢" if sector.change_pct > 0 else "🔴"
                st.metric(
                    f"{color} {sector.sector}",
                    f"₹{sector.last_price:.0f}",
                    f"{sector.change_pct:+.2f}%"
                )
        
        # Detailed table
        st.subheader("Detailed Sector Analysis")
        display_df = sector_df[['sector', 'last_price', 'change_pct', 'bias', 'score', 'source']].copy()
        st.dataframe(display_df, use_container_width=True)

    def display_global_markets(self, global_markets: List[Dict[str, Any]]):
        """Display global markets data"""
        st.subheader("🌍 Global Market Performance")
        
        if not global_markets:
            st.info("No global market data available")
            return
        
        # Create metrics for major markets
        major_markets = ['S&P 500', 'NASDAQ', 'NIKKEI 225', 'HANG SENG']
        filtered_markets = [m for m in global_markets if m['market'] in major_markets]
        
        cols = st.columns(4)
        for idx, market in enumerate(filtered_markets):
            with cols[idx]:
                color = "🟢" if market['change_pct'] > 0 else "🔴"
                st.metric(
                    f"{color} {market['market']}",
                    f"{market['last_price']:.0f}",
                    f"{market['change_pct']:+.2f}%"
                )
        
        # Detailed table
        st.subheader("All Global Markets")
        market_df = pd.DataFrame(global_markets)
        market_df = market_df.sort_values('change_pct', ascending=False)
        display_df = market_df[['market', 'last_price', 'change_pct', 'bias', 'score']].copy()
        st.dataframe(display_df, use_container_width=True)

    def display_intermarket_data(self, intermarket: List[Dict[str, Any]]):
        """Display intermarket analysis data"""
        st.subheader("🔄 Intermarket Analysis")
        
        if not intermarket:
            st.info("No intermarket data available")
            return
        
        # Create metrics for key intermarket assets
        cols = st.columns(4)
        for idx, asset in enumerate(intermarket):
            with cols[idx % 4]:
                color = "🟢" if "BULLISH" in asset['bias'] or "RISK ON" in asset['bias'] else "🔴"
                st.metric(
                    f"{color} {asset['asset']}",
                    f"{asset['last_price']:.2f}",
                    f"{asset['change_pct']:+.2f}%"
                )
                st.caption(f"Bias: {asset['bias']}")
        
        # Interpretation
        st.subheader("Intermarket Interpretation")
        bullish_count = len([a for a in intermarket if "BULLISH" in a['bias'] or "RISK ON" in a['bias']])
        bearish_count = len([a for a in intermarket if "BEARISH" in a['bias'] or "RISK OFF" in a['bias']])
        
        if bullish_count > bearish_count:
            st.success("Overall intermarket sentiment: **RISK-ON**")
        elif bearish_count > bullish_count:
            st.error("Overall intermarket sentiment: **RISK-OFF**")
        else:
            st.warning("Overall intermarket sentiment: **MIXED**")

    def display_sector_rotation(self, rotation_data: Dict[str, Any]):
        """Display sector rotation analysis"""
        if not rotation_data.get('success'):
            st.info("Sector rotation analysis not available")
            return
        
        st.subheader("📊 Sector Rotation Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sector Breadth", f"{rotation_data['sector_breadth']:.1f}%")
        with col2:
            st.metric("Rotation Pattern", rotation_data['rotation_pattern'])
        with col3:
            st.metric("Sector Sentiment", rotation_data['sector_sentiment'])
        
        # Leaders and Laggards
        st.subheader("🏆 Sector Leaders")
        leaders_df = pd.DataFrame(rotation_data['leaders'])
        if not leaders_df.empty:
            st.dataframe(leaders_df[['sector', 'change_pct', 'bias']], use_container_width=True)
        
        st.subheader("📉 Sector Laggards")
        laggards_df = pd.DataFrame(rotation_data['laggards'])
        if not laggards_df.empty:
            st.dataframe(laggards_df[['sector', 'change_pct', 'bias']], use_container_width=True)
        
        # Rotation Interpretation
        st.info(f"**Rotation Type**: {rotation_data['rotation_type']}")
        st.write(f"**Bullish Sectors**: {rotation_data['bullish_sectors_count']} | "
                f"**Bearish Sectors**: {rotation_data['bearish_sectors_count']} | "
                f"**Neutral Sectors**: {rotation_data['neutral_sectors_count']}")

    def display_intraday_seasonality(self, seasonality_data: Dict[str, Any]):
        """Display intraday seasonality analysis"""
        if not seasonality_data.get('success'):
            st.info("Intraday seasonality analysis not available")
            return
        
        st.subheader("⏰ Intraday Market Timing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Session", seasonality_data['session'])
        with col2:
            st.metric("Session Bias", seasonality_data['session_bias'])
        with col3:
            st.metric("Day of Week", seasonality_data['weekday'])
        
        # Session characteristics
        st.subheader("📋 Session Analysis")
        st.info(f"**Current Time**: {seasonality_data['current_time']}")
        st.write(f"**Session Characteristics**: {seasonality_data['session_characteristics']}")
        st.write(f"**Trading Recommendation**: {seasonality_data['trading_recommendation']}")
        
        # Day patterns
        st.write(f"**Day Pattern**: {seasonality_data['day_bias']} - {seasonality_data['day_characteristics']}")

    def check_trading_signals(self):
        """Check for trading signals and send notifications"""
        if not st.session_state.market_bias_data:
            return
        
        current_time = datetime.now(self.ist)
        
        # Check if we should check for signals (every 2 minutes)
        if (st.session_state.last_signal_check and 
            (current_time - st.session_state.last_signal_check).total_seconds() < 120):
            return
        
        st.session_state.last_signal_check = current_time
        
        signals_sent = []
        
        for instrument_data in st.session_state.market_bias_data:
            # Generate trading recommendation
            recommendation = self.trading_signal_manager.generate_trading_recommendation(instrument_data)
            
            if recommendation:
                instrument = recommendation['instrument']
                signal_type = recommendation['signal_type']
                
                # Check cooldown
                can_send, minutes_remaining = self.trading_signal_manager.can_send_signal(signal_type, instrument)
                
                if can_send:
                    # Format and send message
                    message = self.trading_signal_manager.format_signal_message(recommendation)
                    
                    if self.send_telegram_message(message):
                        signals_sent.append(f"{instrument} {signal_type}")
                        st.success(f"Trading signal sent: {instrument} {signal_type}")
                        
                        # Store in session state
                        signal_key = f"{instrument}_{signal_type}_{current_time.strftime('%Y%m%d_%H%M')}"
                        st.session_state.sent_trading_signals[signal_key] = recommendation
                else:
                    st.info(f"Cooldown active for {instrument}: {minutes_remaining} min remaining")
        
        if signals_sent:
            st.rerun()
    
    def display_trading_signals_panel(self) -> bool:
        """Display panel for trading signals and settings"""
        st.sidebar.header("🎯 Trading Signals")
        
        # Signal settings
        signal_cooldown = st.sidebar.slider(
            "Signal Cooldown (min)", 
            min_value=5, 
            max_value=60, 
            value=15,
            help="Minimum time between trading signals for same instrument"
        )
        
        self.trading_signal_manager.cooldown_minutes = signal_cooldown
        
        options_refresh = st.sidebar.slider(
            "Options Data Refresh (min)",
            min_value=1,
            max_value=10,
            value=2,
            help="How often to refresh options chain data"
        )
        
        self.options_analyzer.set_refresh_interval(options_refresh)
        
        enable_trading_signals = st.sidebar.checkbox(
            "Enable Trading Signals",
            value=True,
            help="Send automated trading recommendations based on options analysis"
        )
        
        # Display recent signals
        if st.session_state.sent_trading_signals:
            st.sidebar.subheader("Recent Signals")
            recent_signals = list(st.session_state.sent_trading_signals.values())[-5:]  # Last 5 signals
            
            for signal in reversed(recent_signals):
                emoji = "🟢" if signal['direction'] == "BULLISH" else "🔴"
                with st.sidebar.expander(f"{emoji} {signal['instrument']} {signal['signal_type']}", expanded=False):
                    st.write(f"Time: {signal['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"Confidence: {signal['confidence']}%")
                    st.write(f"Entry: ₹{signal['entry_zone']}")
                    st.write(f"Targets: ₹{signal['targets'][0]}, ₹{signal['targets'][1]}")
                    st.write(f"SL: ₹{signal['stop_loss']}")
        
        return enable_trading_signals

    def format_market_bias_for_alerts(self) -> str:
        """Format market bias data for Telegram alerts"""
        try:
            bias_data = st.session_state.market_bias_data
            if not bias_data:
                return "Market bias data not available"
            
            message = "📊 COMPREHENSIVE OPTIONS MARKET BIAS:\n\n"
            
            for instrument_data in bias_data:
                message += f"🎯 {instrument_data['instrument']}:\n"
                message += f"   • Spot: ₹{instrument_data['spot_price']:.2f}\n"
                message += f"   • Overall Bias: {instrument_data['overall_bias']} (Score: {instrument_data['bias_score']:.2f})\n"
                message += f"   • PCR OI: {instrument_data['pcr_oi']:.2f} | PCR Δ: {instrument_data['pcr_change']:.2f}\n"
                
                # Add comprehensive metrics
                comp_metrics = instrument_data.get('comprehensive_metrics', {})
                if comp_metrics:
                    message += f"   • Synthetic Bias: {comp_metrics.get('synthetic_bias', 'N/A')}\n"
                    message += f"   • ATM Buildup: {comp_metrics.get('atm_buildup', 'N/A')}\n"
                    message += f"   • Vega Bias: {comp_metrics.get('atm_vega_bias', 'N/A')}\n"
                    message += f"   • Max Pain: {comp_metrics.get('max_pain_strike', 'N/A')} (Dist: {comp_metrics.get('distance_from_max_pain', 0):+.1f})\n"
                    message += f"   • Call Res: {comp_metrics.get('call_resistance', 'N/A')} | Put Sup: {comp_metrics.get('put_support', 'N/A')}\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            return f"Market bias analysis temporarily unavailable"

    def check_volume_block_alerts(self, current_price: float, bullish_blocks: List[Dict[str, Any]], bearish_blocks: List[Dict[str, Any]], threshold: float = 5) -> bool:
        """Check if price is near volume order blocks and send alerts with comprehensive ATM bias"""
        if not bullish_blocks and not bearish_blocks:
            return False
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        # Get comprehensive market bias
        market_bias = self.format_market_bias_for_alerts()
        
        # Check bullish blocks
        nearby_bullish = self.vob_indicator.check_price_near_blocks(current_price, bullish_blocks, threshold)
        for block in nearby_bullish:
            alert_id = f"vol_block_bullish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_block", alert_id):
                message = f"""🚨 PRICE NEAR BULLISH VOLUME ORDER BLOCK!

📊 Nifty 50 Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

🎯 Volume Order Block:
• Type: BULLISH 
• Upper: ₹{block['upper']:.2f}
• Lower: ₹{block['lower']:.2f}
• Mid: ₹{block['mid']:.2f}
• Volume: {block['volume']:,}

📈 Distance to Block: {abs(current_price - block['mid']):.2f} points

{market_bias}

💡 Trading Suggestion:
Consider LONG positions with stop below support

⏳ Next alert in 10 minutes

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Bullish Volume Block alert sent!")
                    alert_sent = True
        
        # Check bearish blocks
        nearby_bearish = self.vob_indicator.check_price_near_blocks(current_price, bearish_blocks, threshold)
        for block in nearby_bearish:
            alert_id = f"vol_block_bearish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_block", alert_id):
                message = f"""🚨 PRICE NEAR BEARISH VOLUME ORDER BLOCK!

📊 Nifty 50 Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

🎯 Volume Order Block:
• Type: BEARISH 
• Upper: ₹{block['upper']:.2f}
• Lower: ₹{block['lower']:.2f}
• Mid: ₹{block['mid']:.2f}
• Volume: {block['volume']:,}

📉 Distance to Block: {abs(current_price - block['mid']):.2f} points

{market_bias}

💡 Trading Suggestion:
Consider SHORT positions with stop above resistance

⏳ Next alert in 10 minutes

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Bearish Volume Block alert sent!")
                    alert_sent = True
        
        return alert_sent

    def check_volume_spike_alerts(self, df: pd.DataFrame) -> bool:
        """Check for sudden volume spikes and send alerts with comprehensive ATM bias"""
        if df.empty or len(df) < 2:
            return False
        
        current_candle = df.iloc[-1]
        previous_candle = df.iloc[-2]
        
        current_volume = current_candle['volume']
        current_time = current_candle.name
        current_price = current_candle['close']
        price_change = current_candle['close'] - previous_candle['close']
        price_change_pct = (price_change / previous_candle['close']) * 100
        
        # Detect volume spike
        is_spike, volume_ratio = self.volume_spike_detector.detect_volume_spike(current_volume, current_time)
        
        if is_spike:
            alert_id = f"volume_spike_{current_time.strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_spike", alert_id):
                # Get comprehensive market bias
                market_bias = self.format_market_bias_for_alerts()
                
                spike_type = "BUYING" if price_change > 0 else "SELLING"
                emoji = "🟢" if price_change > 0 else "🔴"
                
                message = f"""📈 SUDDEN VOLUME SPIKE DETECTED!

{emoji} Nifty 50 Volume Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

📊 Volume Analysis:
• Current Volume: {current_volume:,}
• Volume Ratio: {volume_ratio:.1f}x average
• Price Change: ₹{price_change:+.2f} ({price_change_pct:+.2f}%)

🎯 Spike Type: {spike_type} PRESSURE

{market_bias}

💡 Market Interpretation:
High volume with {spike_type.lower()} pressure indicates 
strong institutional activity

⏳ Next alert in 10 minutes

⚡ Immediate Action:
Watch for breakout/breakdown confirmation!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Volume Spike alert sent! ({volume_ratio:.1f}x average)")
                    return True
        
        return False

    def get_bias_color(self, bias_text: str) -> str:
        """Get color for bias text"""
        if 'Bullish' in str(bias_text):
            return 'bullish'
        elif 'Bearish' in str(bias_text):
            return 'bearish'
        else:
            return 'neutral'

    def get_score_color(self, score: float) -> str:
        """Get color for bias score"""
        if score >= 2:
            return 'bullish'
        elif score <= -2:
            return 'bearish'
        else:
            return 'neutral'

    def get_pcr_color(self, pcr_value: float) -> str:
        """Get color for PCR value"""
        if pcr_value > 1.2:
            return 'bullish'
        elif pcr_value < 0.8:
            return 'bearish'
        else:
            return 'neutral'

    def get_diff_color(self, diff_value: float) -> str:
        """Get color for difference values"""
        if diff_value > 0:
            return 'bullish'
        elif diff_value < 0:
            return 'bearish'
        else:
            return 'neutral'

    def get_change_color(self, change_value: float) -> str:
        """Get color for change values"""
        if change_value > 0:
            return 'bullish'
        elif change_value < 0:
            return 'bearish'
        else:
            return 'neutral'

    def get_color_code(self, color_type: str) -> str:
        """Get hex color code for color type"""
        color_map = {
            'bullish': '#90EE90',  # Light Green
            'bearish': '#FFB6C1',  # Light Red
            'neutral': '#FFFFE0',  # Light Yellow
            'normal': '#FFFFFF'    # White
        }
        return color_map.get(color_type, '#FFFFFF')

    def calculate_confidence_score(self, instrument_data: Dict[str, Any], comp_metrics: Dict[str, Any]) -> float:
        """Calculate confidence score based on multiple factors"""
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
        
        return min(confidence, 100)

    def display_comprehensive_options_analysis(self):
        """Display comprehensive NSE Options Analysis with detailed ATM bias tabulation"""
        st.header("📊 NSE Options Chain Analysis - Auto Refresh")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"Options data auto-refreshes every {self.options_analyzer.refresh_interval} minutes")
        with col2:
            if st.button("🔄 Force Refresh", type="primary"):
                with st.spinner("Force refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    st.success("Options data refreshed!")
        with col3:
            if st.session_state.last_bias_update:
                st.write(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Display current market bias
        if st.session_state.market_bias_data:
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
                detailed_bias = instrument_data.get('detailed_atm_bias', {})
                
                with st.expander(f"🎯 {instrument_data['instrument']} - Detailed ATM Bias Analysis", expanded=True):
                    
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
                    
                    confidence_score = self.calculate_confidence_score(instrument_data, comp_metrics)
                    overall_bias = instrument_data['overall_bias']
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if "Strong Bullish" in overall_bias and confidence_score >= 80:
                            st.success(f"""
                            **🎯 HIGH CONFIDENCE BULLISH SIGNAL - {confidence_score}% Confidence**
                            
                            **Recommended Action:** Aggressive LONG/CALL positions
                            **Entry Zone:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 50):.0f} - ₹{instrument_data['spot_price']:.0f}
                            **Target 1:** ₹{instrument_data['spot_price'] + (comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100) - instrument_data['spot_price']) * 0.5:.0f}
                            **Target 2:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100):.0f}
                            **Stop Loss:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100) - 20:.0f}
                            """)
                        elif "Bullish" in overall_bias:
                            st.info(f"""
                            **📈 BULLISH BIAS - {confidence_score}% Confidence**
                            
                            **Recommended Action:** Consider LONG/CALL positions
                            **Entry Zone:** Wait for pullback to support
                            **Target:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 80):.0f}
                            **Stop Loss:** Below key support
                            """)
                        elif "Strong Bearish" in overall_bias and confidence_score >= 80:
                            st.error(f"""
                            **🎯 HIGH CONFIDENCE BEARISH SIGNAL - {confidence_score}% Confidence**
                            
                            **Recommended Action:** Aggressive SHORT/PUT positions
                            **Entry Zone:** ₹{instrument_data['spot_price']:.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 50):.0f}
                            **Target 1:** ₹{instrument_data['spot_price'] - (instrument_data['spot_price'] - comp_metrics.get('put_support', instrument_data['spot_price'] - 100)) * 0.5:.0f}
                            **Target 2:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100):.0f}
                            **Stop Loss:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100) + 20:.0f}
                            """)
                        elif "Bearish" in overall_bias:
                            st.warning(f"""
                            **📉 BEARISH BIAS - {confidence_score}% Confidence**
                            
                            **Recommended Action:** Consider SHORT/PUT positions
                            **Entry Zone:** Wait for rally to resistance
                            **Target:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 80):.0f}
                            **Stop Loss:** Above key resistance
                            """)
                        else:
                            st.warning(f"""
                            **⚖️ NEUTRAL/UNCLEAR BIAS - {confidence_score}% Confidence**
                            
                            **Recommended Action:** Wait for clear directional bias
                            **Strategy:** Consider range-bound strategies
                            **Key Levels:** Monitor ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 50):.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 50):.0f}
                            """)
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence_score}%")
                        st.metric("Overall Bias", overall_bias)
                        st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
        
        else:
            st.info("👆 Options data will auto-refresh. Click 'Force Refresh' to load immediately.")

    def display_comprehensive_bias_analysis(self):
        """Display comprehensive bias analysis from BiasAnalysisPro with enhanced error handling"""
        st.header("🎯 Comprehensive Technical Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("8-indicator bias analysis with adaptive weighting and market breadth")
        with col2:
            if st.button("🔄 Update Bias Analysis", type="primary"):
                with st.spinner("Running comprehensive bias analysis..."):
                    try:
                        bias_data = self.bias_analyzer.analyze_all_bias_indicators("^NSEI")
                        st.session_state.comprehensive_bias_data = bias_data
                        st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
                        if bias_data['success']:
                            st.success("Bias analysis completed successfully!")
                        else:
                            st.error(f"Bias analysis failed: {bias_data['error']}")
                    except Exception as e:
                        st.error(f"Error during bias analysis: {str(e)}")
        
        st.divider()
        
        # Display last update time
        if st.session_state.last_comprehensive_bias_update:
            st.write(f"Last analysis: {st.session_state.last_comprehensive_bias_update.strftime('%H:%M:%S')} IST")
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if not bias_data['success']:
                st.error(f"❌ Bias analysis failed: {bias_data['error']}")
                
                # Provide alternative data source options
                st.info("💡 **Troubleshooting Tips:**")
                st.write("""
                1. Try using a different symbol (e.g., 'NSEI' instead of '^NSEI')
                2. Check your internet connection
                3. Try again in a few minutes as Yahoo Finance might be temporarily unavailable
                4. Use the Options Chain analysis below which uses NSE data directly
                """)
                
                # Fallback to manual input for testing
                with st.expander("🛠️ Manual Data Input (Testing)"):
                    st.warning("Use this for testing when Yahoo Finance is unavailable")
                    manual_bias = st.selectbox("Manual Bias", ["BULLISH", "BEARISH", "NEUTRAL"])
                    manual_score = st.slider("Manual Score", -100, 100, 0)
                    
                    if st.button("Apply Manual Data"):
                        st.session_state.comprehensive_bias_data = {
                            'success': True,
                            'overall_bias': manual_bias,
                            'overall_score': manual_score,
                            'overall_confidence': 75,
                            'current_price': 22000,
                            'bias_results': [
                                {'indicator': 'RSI', 'value': '55.0', 'bias': manual_bias, 'score': manual_score},
                                {'indicator': 'Volume Delta', 'value': '1000', 'bias': manual_bias, 'score': manual_score},
                                {'indicator': 'DMI', 'value': '+DI:25 -DI:20', 'bias': manual_bias, 'score': manual_score},
                            ],
                            'bullish_count': 3 if manual_bias == "BULLISH" else 0,
                            'bearish_count': 3 if manual_bias == "BEARISH" else 0,
                            'neutral_count': 3 if manual_bias == "NEUTRAL" else 0,
                            'total_indicators': 3
                        }
                        st.rerun()
                
                return
            
            # Overall bias summary
            st.subheader("📊 Overall Market Bias")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
                st.metric(
                    "Overall Bias", 
                    f"{bias_color} {bias_data['overall_bias']}",
                    delta=f"Score: {bias_data['overall_score']:.1f}"
                )
            with col2:
                # Create a gauge chart for bias score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = bias_data['overall_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Bias Score"},
                    gauge = {
                        'axis': {'range': [-100, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-100, -50], 'color': "lightcoral"},
                            {'range': [-50, 0], 'color': "lightyellow"},
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "limegreen"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': bias_data['overall_score']}}
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
            with col4:
                st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
            
            st.divider()
            
            # Detailed bias indicators in a table
            st.subheader("📈 Detailed Technical Indicators")
            
            # Convert bias results to DataFrame for better display
            bias_df = pd.DataFrame(bias_data['bias_results'])
            
            # Add color coding
            def style_bias(val):
                if val == 'BULLISH':
                    return 'color: green; font-weight: bold'
                elif val == 'BEARISH':
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: orange; font-weight: bold'
            
            # Display as styled table
            styled_df = bias_df[['indicator', 'value', 'bias', 'score']].style.applymap(
                style_bias, subset=['bias']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Visual representation
            st.subheader("📊 Bias Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of bias distribution
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
                # Bar chart of indicator scores
                fig_bar = px.bar(
                    bias_df,
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
            
            st.divider()
            
            # Advanced metrics
            st.subheader("🔍 Advanced Analysis Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bullish Indicators", bias_data['bullish_count'])
            with col2:
                st.metric("Bearish Indicators", bias_data['bearish_count'])
            with col3:
                st.metric("Neutral Indicators", bias_data['neutral_count'])
            with col4:
                st.metric("Total Indicators", bias_data['total_indicators'])
            
            # Additional metrics if available
            if 'fast_bull_pct' in bias_data:
                st.subheader("📈 Weighted Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Fast Bull %", f"{bias_data['fast_bull_pct']:.1f}%")
                with col2:
                    st.metric("Fast Bear %", f"{bias_data['fast_bear_pct']:.1f}%")
                with col3:
                    st.metric("Bullish Bias %", f"{bias_data['bullish_bias_pct']:.1f}%")
                with col4:
                    st.metric("Bearish Bias %", f"{bias_data['bearish_bias_pct']:.1f}%")
            
            # Trading recommendation based on bias
            st.divider()
            st.subheader("💡 Trading Recommendation")
            
            bias_strength = abs(bias_data['overall_score'])
            overall_bias = bias_data['overall_bias']
            confidence = bias_data['overall_confidence']
            
            if overall_bias == "BULLISH" and bias_strength > 60 and confidence > 70:
                st.success("""
                **🎯 STRONG BULLISH SIGNAL - HIGH CONFIDENCE**
                
                **Recommended Action:** Consider LONG positions
                **Strategy:** Look for buying opportunities on dips
                **Risk Management:** Use tight stop losses
                **Target:** Expect upward momentum to continue
                """)
            elif overall_bias == "BULLISH":
                st.info("""
                **📈 BULLISH BIAS - MODERATE CONFIDENCE**
                
                **Recommended Action:** Cautious LONG positions
                **Strategy:** Wait for confirmations before entering
                **Risk Management:** Use proper position sizing
                """)
            elif overall_bias == "BEARISH" and bias_strength > 60 and confidence > 70:
                st.error("""
                **🎯 STRONG BEARISH SIGNAL - HIGH CONFIDENCE**
                
                **Recommended Action:** Consider SHORT positions  
                **Strategy:** Look for selling opportunities on rallies
                **Risk Management:** Use tight stop losses
                **Target:** Expect downward momentum to continue
                """)
            elif overall_bias == "BEARISH":
                st.warning("""
                **📉 BEARISH BIAS - MODERATE CONFIDENCE**
                
                **Recommended Action:** Cautious SHORT positions
                **Strategy:** Wait for confirmations before entering
                **Risk Management:** Use proper position sizing
                """)
            else:
                st.warning("""
                **⚖️ NEUTRAL/UNCLEAR BIAS**
                
                **Recommended Action:** Wait for clearer direction
                **Strategy:** Consider range-bound strategies
                **Risk Management:** Reduce position sizes
                **Advice:** Monitor for breakout signals
                """)
        
        else:
            st.info("👆 Click 'Update Bias Analysis' to run comprehensive technical analysis")
            st.write("This analysis uses 8 technical indicators to determine market bias:")
            st.write("""
            - **Volume Delta** - Buying vs Selling pressure
            - **HVP** - High Volume Pivots  
            - **VOB** - Volume Order Blocks
            - **Order Blocks** - EMA crossover signals
            - **RSI** - Momentum indicator
            - **DMI** - Directional Movement Index
            - **VIDYA** - Variable Index Dynamic Average
            - **MFI** - Money Flow Index
            """)

    def display_option_chain_bias_tabulation(self):
        """Display all option chain bias data in comprehensive tabulation"""
        st.header("📋 Comprehensive Option Chain Bias Data")
        
        if not st.session_state.market_bias_data:
            st.info("No option chain data available. Please refresh options analysis first.")
            return
        
        for instrument_data in st.session_state.market_bias_data:
            with st.expander(f"🎯 {instrument_data['instrument']} - Complete Bias Analysis", expanded=True):
                
                # Basic Information Table
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
                        f"{instrument_data['bias_score']:.2f}",
                        f"{instrument_data['pcr_oi']:.2f}",
                        f"{instrument_data['pcr_change']:.2f}"
                    ]
                })
                st.dataframe(basic_info, use_container_width=True, hide_index=True)
                
                # Detailed ATM Bias Table
                if 'detailed_atm_bias' in instrument_data and instrument_data['detailed_atm_bias']:
                    st.subheader("🔍 Detailed ATM Bias Analysis")
                    detailed_bias = instrument_data['detailed_atm_bias']
                    
                    # Create comprehensive table for detailed bias
                    bias_metrics = []
                    bias_values = []
                    bias_signals = []
                    
                    for key, value in detailed_bias.items():
                        if key not in ['Strike', 'Zone', 'CE_OI', 'PE_OI', 'CE_Change', 'PE_Change', 
                                     'CE_Volume', 'PE_Volume', 'CE_Price', 'PE_Price', 'CE_IV', 'PE_IV',
                                     'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE']:
                            bias_metrics.append(key.replace('_', ' ').title())
                            bias_values.append(str(value))
                            
                            # Determine signal strength
                            if 'Bullish' in str(value):
                                bias_signals.append('🟢 Bullish')
                            elif 'Bearish' in str(value):
                                bias_signals.append('🔴 Bearish')
                            else:
                                bias_signals.append('🟡 Neutral')
                    
                    detailed_df = pd.DataFrame({
                        'Metric': bias_metrics,
                        'Value': bias_values,
                        'Signal': bias_signals
                    })
                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Raw values table
                    st.subheader("📈 Raw Option Data")
                    raw_data = []
                    if 'CE_OI' in detailed_bias:
                        raw_data.append(['Call OI', f"{detailed_bias['CE_OI']:,.0f}"])
                        raw_data.append(['Put OI', f"{detailed_bias['PE_OI']:,.0f}"])
                        raw_data.append(['Call OI Change', f"{detailed_bias['CE_Change']:,.0f}"])
                        raw_data.append(['Put OI Change', f"{detailed_bias['PE_Change']:,.0f}"])
                        raw_data.append(['Call Volume', f"{detailed_bias['CE_Volume']:,.0f}"])
                        raw_data.append(['Put Volume', f"{detailed_bias['PE_Volume']:,.0f}"])
                        raw_data.append(['Call Price', f"₹{detailed_bias['CE_Price']:.2f}"])
                        raw_data.append(['Put Price', f"₹{detailed_bias['PE_Price']:.2f}"])
                        raw_data.append(['Call IV', f"{detailed_bias['CE_IV']:.2f}%"])
                        raw_data.append(['Put IV', f"{detailed_bias['PE_IV']:.2f}%"])
                        raw_data.append(['Call Delta', f"{detailed_bias['Delta_CE']:.4f}"])
                        raw_data.append(['Put Delta', f"{detailed_bias['Delta_PE']:.4f}"])
                        raw_data.append(['Call Gamma', f"{detailed_bias['Gamma_CE']:.4f}"])
                        raw_data.append(['Put Gamma', f"{detailed_bias['Gamma_PE']:.4f}"])
                    
                    raw_df = pd.DataFrame(raw_data, columns=['Parameter', 'Value'])
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)
                
                # Comprehensive Metrics Table
                if 'comprehensive_metrics' in instrument_data and instrument_data['comprehensive_metrics']:
                    st.subheader("🎯 Advanced Option Metrics")
                    comp_metrics = instrument_data['comprehensive_metrics']
                    
                    comp_data = []
                    for key, value in comp_metrics.items():
                        if key not in ['total_vega', 'total_ce_vega_exp', 'total_pe_vega_exp']:
                            comp_data.append([
                                key.replace('_', ' ').title(),
                                str(value) if not isinstance(value, (int, float)) else f"{value:.2f}"
                            ])
                    
                    comp_df = pd.DataFrame(comp_data, columns=['Metric', 'Value'])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Visual Analysis
                st.subheader("📊 Visual Bias Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bias Score Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = instrument_data['bias_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{instrument_data['instrument']} Bias Score"},
                        gauge = {
                            'axis': {'range': [-10, 10]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-10, -4], 'color': "lightcoral"},
                                {'range': [-4, -2], 'color': "lightyellow"},
                                {'range': [-2, 2], 'color': "lightgray"},
                                {'range': [2, 4], 'color': "lightgreen"},
                                {'range': [4, 10], 'color': "limegreen"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': instrument_data['bias_score']}}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # PCR Analysis
                    pcr_data = {
                        'Metric': ['PCR OI', 'PCR Change OI'],
                        'Value': [instrument_data['pcr_oi'], instrument_data['pcr_change']],
                        'Interpretation': [
                            'Bullish' if instrument_data['pcr_oi'] > 1.0 else 'Bearish',
                            'Bullish' if instrument_data['pcr_change'] > 1.0 else 'Bearish'
                        ]
                    }
                    pcr_df = pd.DataFrame(pcr_data)
                    st.dataframe(pcr_df, use_container_width=True, hide_index=True)
                
                # Trading Levels
                if 'comprehensive_metrics' in instrument_data:
                    comp_metrics = instrument_data['comprehensive_metrics']
                    st.subheader("🎯 Key Trading Levels")
                    
                    levels_data = []
                    if 'call_resistance' in comp_metrics and comp_metrics['call_resistance']:
                        levels_data.append(['Call Resistance', f"₹{comp_metrics['call_resistance']:.0f}"])
                    if 'put_support' in comp_metrics and comp_metrics['put_support']:
                        levels_data.append(['Put Support', f"₹{comp_metrics['put_support']:.0f}"])
                    if 'max_pain_strike' in comp_metrics and comp_metrics['max_pain_strike']:
                        levels_data.append(['Max Pain', f"₹{comp_metrics['max_pain_strike']:.0f}"])
                    
                    levels_data.append(['Current Spot', f"₹{instrument_data['spot_price']:.0f}"])
                    
                    levels_df = pd.DataFrame(levels_data, columns=['Level', 'Price'])
                    st.dataframe(levels_df, use_container_width=True, hide_index=True)

    def create_comprehensive_chart(self, df: pd.DataFrame, bullish_blocks: List[Dict[str, Any]], bearish_blocks: List[Dict[str, Any]], interval: str) -> Optional[go.Figure]:
        """Create comprehensive chart with Volume Order Blocks"""
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Nifty 50 Analysis - {interval} Min', 'Volume with Spike Detection'),
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
        colors = {'bullish': '#26ba9f', 'bearish': '#6626ba'}
        
        for block in bullish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(38, 186, 159, 0.1)',
                line=dict(color=colors['bullish'], width=1),
                row=1, col=1
            )
        
        for block in bearish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(102, 38, 186, 0.1)',
                line=dict(color=colors['bearish'], width=1),
                row=1, col=1
            )
        
        # Volume bars with spike detection
        bar_colors = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if i < len(df) - 1:
                bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
            else:
                current_volume = row['volume']
                if len(df) > 5:
                    avg_volume = df['volume'].iloc[-6:-1].mean()
                    if current_volume > avg_volume * 2.5:
                        bar_colors.append('#ffeb3b')
                    else:
                        bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
                else:
                    bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
        
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
            height=800,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', side="right")
        
        return fig

    def run(self):
        """Main application with all features"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Volume Analysis, Options Chain, Technical Bias & Trading Signals*")
        
        # Sidebar with trading signals panel
        with st.sidebar:
            st.header("🔧 API Status")
            if st.button("Test API Connection"):
                self.test_api_connection()
            
            # Trading signals settings
            enable_trading_signals = self.display_trading_signals_panel()
            
            st.header("📊 Chart Settings")
            timeframe = st.selectbox("Timeframe", ['1', '3', '5', '15'], index=1)
            
            st.subheader("Volume Order Blocks")
            vob_sensitivity = st.slider("Sensitivity", 3, 10, 5)
            alert_threshold = st.slider("Alert Threshold (points)", 1, 10, 5)
            
            st.subheader("Volume Spike Detection")
            spike_threshold = st.slider("Spike Threshold (x avg)", 2.0, 5.0, 2.5)
            
            st.subheader("Alert Cooldown")
            cooldown_minutes = st.slider("Cooldown (minutes)", 1, 30, 10)
            
            st.subheader("Alerts")
            volume_block_alerts = st.checkbox("Volume Block Alerts", value=True)
            volume_spike_alerts = st.checkbox("Volume Spike Alerts", value=True)
            telegram_enabled = st.checkbox("Enable Telegram", value=bool(self.telegram_bot_token))
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Update cooldown periods
        self.alert_manager.cooldown_minutes = cooldown_minutes
        
        # Main content - Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([  # NEW: Added tab6 for Enhanced Market Data
            "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
            "📋 Bias Tabulation", "🚀 Trading Signals", "🌍 Market Data"  # NEW: Enhanced Market Data tab
        ])
        
        with tab1:
            # Price Analysis Tab
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            # Update detectors
            self.vob_indicator = VolumeOrderBlocks(sensitivity=vob_sensitivity)
            self.volume_spike_detector.spike_threshold = spike_threshold
            
            # Fetch data
            df = pd.DataFrame()
            with st.spinner("Fetching market data..."):
                api_data = self.fetch_intraday_data(interval=timeframe)
                if api_data:
                    df = self.process_data(api_data)
            
            if not df.empty:
                latest = df.iloc[-1]
                current_price = latest['close']
                current_volume = latest['volume']
                
                # Detect Volume Order Blocks
                bullish_blocks, bearish_blocks = self.vob_indicator.detect_volume_order_blocks(df)
                
                # Calculate volume statistics
                if len(df) > 5:
                    avg_vol = df['volume'].iloc[-6:-1].mean()
                    volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
                else:
                    volume_ratio = 0
                
                # Display metrics
                with col1:
                    st.metric("Nifty Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Current Volume", f"{current_volume:,}")
                with col3:
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
                with col4:
                    st.metric("Bullish Blocks", len(bullish_blocks))
                with col5:
                    st.metric("Bearish Blocks", len(bearish_blocks))
                with col6:
                    if (volume_spike_alerts or volume_block_alerts) and telegram_enabled:
                        st.metric("Alerts Status", "✅ Active")
                    else:
                        st.metric("Alerts Status", "❌ Inactive")
                
                # Create and display chart
                chart = self.create_comprehensive_chart(df, bullish_blocks, bearish_blocks, timeframe)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Check for alerts
                alerts_sent = []
                if telegram_enabled:
                    if volume_block_alerts:
                        block_alert = self.check_volume_block_alerts(
                            current_price, bullish_blocks, bearish_blocks, alert_threshold
                        )
                        if block_alert:
                            alerts_sent.append("Volume Block")
                    
                    if volume_spike_alerts:
                        spike_alert = self.check_volume_spike_alerts(df)
                        if spike_alert:
                            alerts_sent.append("Volume Spike")
                
                if alerts_sent:
                    st.success(f"📱 Alerts sent: {', '.join(alerts_sent)} (Cooldown: {cooldown_minutes}min)")
                
                # Real-time volume monitoring
                st.subheader("🔍 Live Volume Monitoring")
                vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
                
                with vol_col1:
                    if len(df) > 5:
                        avg_vol = df['volume'].iloc[-6:-1].mean()
                        st.metric("20-period Avg Volume", f"{avg_vol:,.0f}")
                
                with vol_col2:
                    st.metric("Current Volume", f"{current_volume:,.0f}")
                
                with vol_col3:
                    if volume_ratio > spike_threshold:
                        st.error(f"Volume Spike: {volume_ratio:.1f}x")
                    elif volume_ratio > 1.5:
                        st.warning(f"High Volume: {volume_ratio:.1f}x")
                    else:
                        st.success(f"Normal: {volume_ratio:.1f}x")
                
                with vol_col4:
                    if len(df) > 1:
                        price_change = latest['close'] - df.iloc[-2]['close']
                        st.metric("Price Change", f"₹{price_change:+.2f}")
            
            else:
                st.error("No data available. Please check your API credentials and try again.")
        
        with tab2:
            # Options Analysis Tab with auto-refresh
            self.display_comprehensive_options_analysis()
            
            # Auto-refresh logic
            current_time = datetime.now(self.ist)
            if (st.session_state.last_bias_update is None or 
                (current_time - st.session_state.last_bias_update).total_seconds() > self.options_analyzer.refresh_interval * 60):
                
                with st.spinner("Auto-refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = current_time
                    st.rerun()
        
        with tab3:
            # Technical Bias Analysis Tab
            self.display_comprehensive_bias_analysis()
        
        with tab4:
            # New comprehensive bias tabulation
            self.display_option_chain_bias_tabulation()
        
        with tab5:
            # Trading Signals Tab
            st.header("🚀 Automated Trading Signals")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Trading signals generated from comprehensive options chain analysis")
            with col2:
                if st.button("Check Signals Now", type="primary"):
                    self.check_trading_signals()
            
            st.divider()
            
            # Display current market conditions
            if st.session_state.market_bias_data:
                st.subheader("Current Market Conditions")
                
                for instrument_data in st.session_state.market_bias_data:
                    with st.expander(f"{instrument_data['instrument']} - Signal Readiness", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Bias", instrument_data['overall_bias'])
                        with col2:
                            st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
                        with col3:
                            st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                        with col4:
                            confidence = self.trading_signal_manager.calculate_confidence_score(
                                instrument_data, 
                                instrument_data.get('comprehensive_metrics', {})
                            )
                            st.metric("Signal Confidence", f"{confidence}%")
                        
                        # Generate and display potential signal
                        recommendation = self.trading_signal_manager.generate_trading_recommendation(instrument_data)
                        if recommendation:
                            st.success(f"✅ **{recommendation['signal_type']} Signal Ready**")
                            st.write(f"Strength: {recommendation['strength']} | Confidence: {recommendation['confidence']}%")
                            
                            if enable_trading_signals and telegram_enabled:
                                can_send, minutes_remaining = self.trading_signal_manager.can_send_signal(
                                    recommendation['signal_type'], 
                                    recommendation['instrument']
                                )
                                if can_send:
                                    if st.button(
                                        f"Send {recommendation['instrument']} {recommendation['signal_type']} Signal",
                                        key=f"send_{recommendation['instrument']}"
                                    ):
                                        message = self.trading_signal_manager.format_signal_message(recommendation)
                                        if self.send_telegram_message(message):
                                            st.success(f"Signal sent for {recommendation['instrument']}!")
                                            # Store in session state
                                            current_time = datetime.now(self.ist)
                                            signal_key = f"{recommendation['instrument']}_{recommendation['signal_type']}_{current_time.strftime('%Y%m%d_%H%M%S')}"
                                            st.session_state.sent_trading_signals[signal_key] = recommendation
                                else:
                                    st.warning(f"Cooldown active: {minutes_remaining} minutes remaining")
                        else:
                            st.info("📊 Monitoring market conditions...")
            
            # Display signal history
            if st.session_state.sent_trading_signals:
                st.divider()
                st.subheader("Signal History")
                
                signals_df = pd.DataFrame(list(st.session_state.sent_trading_signals.values()))
                if not signals_df.empty:
                    # Format the dataframe for display
                    display_df = signals_df[['instrument', 'signal_type', 'direction', 'confidence', 'timestamp']].copy()
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                    display_df = display_df.sort_values('timestamp', ascending=False)
                    
                    st.dataframe(display_df, use_container_width=True)
        
        with tab6:  # NEW: Enhanced Market Data Tab
            self.display_enhanced_market_data()
        
        # Check for trading signals automatically
        if enable_trading_signals and telegram_enabled:
            self.check_trading_signals()
        
        # Cleanup and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()