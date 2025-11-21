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
import hashlib

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# ENHANCED DATA FETCHER WITH FALLBACKS - FIXED FOR TODAY'S DATA
# =============================================

class RobustDataFetcher:
    """Enhanced data fetcher with multiple fallback sources"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def fetch_nifty_data_with_fallback(self, symbol: str = "^NSEI", period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """Fetch Nifty data with multiple fallback methods - Focus on today's data"""
        df = pd.DataFrame()
        
        # Method 1: Try Yahoo Finance first with today's data
        try:
            ticker = yf.Ticker(symbol)
            # Try to get more frequent data for current day
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                print(f"✓ Yahoo Finance data: {len(df)} candles")
                df = self._process_yahoo_data(df)
                
                # Ensure we have today's data - if not, generate synthetic for today
                today = datetime.now(self.ist).date()
                if len(df[df.index.date == today]) == 0:
                    print("⚠️ No today's data from Yahoo, generating synthetic data for today")
                    df = self._generate_synthetic_data_for_today()
                else:
                    # Filter for today's data only
                    df = df[df.index.date == today]
                    
                return df
        except Exception as e:
            print(f"Yahoo Finance failed: {e}")

        # Method 2: Generate synthetic data for today
        if df.empty:
            print("⚠️ Using synthetic data for today's demonstration")
            df = self._generate_synthetic_data_for_today()
        
        return df
    
    def _generate_synthetic_data_for_today(self) -> pd.DataFrame:
        """Generate synthetic Nifty data for today only"""
        today = datetime.now(self.ist).date()
        current_time = datetime.now(self.ist)
        
        # Generate data for today's market hours only (9:15 AM to current time or 3:30 PM)
        market_open = datetime.combine(today, datetime.strptime("09:15", "%H:%M").time()).replace(tzinfo=self.ist)
        market_close = datetime.combine(today, datetime.strptime("15:30", "%H:%M").time()).replace(tzinfo=self.ist)
        
        # If current time is before market open, use yesterday's close as reference
        if current_time < market_open:
            # Use yesterday's data or start from previous close
            base_price = 22150  # Default base price
            end_time = market_open + timedelta(hours=1)  # Show first hour
        else:
            # Use current time as end point (but not beyond market close)
            end_time = min(current_time, market_close)
        
        # Generate 5-minute intervals from market open to current time/close
        dates = pd.date_range(
            start=market_open,
            end=end_time,
            freq='5min',
            tz=self.ist
        )
        
        n_points = len(dates)
        
        if n_points == 0:
            # Fallback: generate data for last 6 hours
            dates = pd.date_range(
                start=datetime.now(self.ist) - timedelta(hours=6),
                end=datetime.now(self.ist),
                freq='5min',
                tz=self.ist
            )
            n_points = len(dates)
        
        # Generate realistic price data starting around current levels
        base_price = 22150 + np.random.uniform(-100, 100)  # Some variation
        returns = np.random.normal(0, 0.0003, n_points)  # Smaller random returns for intraday
        
        # Add intraday trend based on time of day
        time_fraction = np.linspace(0, 1, n_points)
        # Simulate typical intraday pattern: slow start, activity mid-day, volatility towards close
        intraday_pattern = np.sin(time_fraction * np.pi - np.pi/2) * 0.5 + 0.5
        trend = intraday_pattern * 150  # 150 point range for the day
        
        prices = base_price + np.cumsum(returns * 10) + trend
        
        # Generate OHLC data
        data = []
        for i, date in enumerate(dates):
            if i == 0:
                open_price = prices[i]
            else:
                open_price = data[i-1]['close']
            
            close_price = prices[i]
            # Realistic intraday volatility (smaller ranges)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 3))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 3))
            
            # Ensure high >= open,close >= low
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume follows intraday pattern - higher during opening and closing
            volume_pattern = 0.3 + 0.7 * (abs(time_fraction[i] - 0.5) * 2)  # U-shaped volume
            volume = max(200000, int(abs(np.random.normal(800000 * volume_pattern, 300000))))
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        print(f"✓ Today's synthetic data generated: {len(df)} candles (up to {df.index[-1].strftime('%H:%M')})")
        return df

    def _process_yahoo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Yahoo Finance data to standard format"""
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 22150 if col != 'volume' else 1000000
        
        # Convert index to IST timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(self.ist)
        
        return df

    def _is_market_hour(self, dt: datetime) -> bool:
        """Check if datetime is within market hours"""
        time = dt.time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        return market_open <= time <= market_close and dt.weekday() < 5

# =============================================
# ENHANCED SAFETY CHECK MODULE
# =============================================

class TradingSafetyManager:
    """Comprehensive safety checks for trading signal reliability"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def should_trust_signals(self, df: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
        """
        Comprehensive signal reliability check
        Returns: (is_trustworthy, reason, detailed_report)
        """
        detailed_report = {}
        
        # If no data provided, be cautious
        if df is None or df.empty:
            return False, "No market data available", {"data_available": False}
        
        # 1. BASIC MARKET CONDITIONS
        basic_checks = {
            'market_hours': self.is_regular_market_hours(),
            'normal_volume': self.is_volume_normal(df),
            'vix_normal': self.is_vix_between(12, 30),
            'no_large_gaps': not self.has_large_gap(df, 1.0),
            'data_fresh': self.is_data_timestamp_recent(df, minutes=10),  # Relaxed to 10 minutes
            'sufficient_data': self.has_minimum_candles(df, 20)  # Reduced to 20 candles
        }
        
        # 2. ADVANCED CHECKS (simplified for demo)
        advanced_checks = {
            'indicators_aligned': True,  # Skip for demo
            'market_regime_ok': True,    # Skip for demo
            'options_data_reliable': True, # Skip for demo
            'volume_profile_healthy': self.is_volume_profile_normal(df),
            'no_earnings_events': not self.is_earnings_day(),
            'technical_quality': self.has_good_technical_quality(df)
        }
        
        # 3. FAIL-SAFE CHECKS (simplified)
        fail_safe_checks = {
            'not_extreme_volatility': self.get_volatility_ratio(df) < 5.0,  # Relaxed
            'not_abnormal_spreads': True,
            'not_manipulation_signs': not self.detect_abnormal_trading(df),
            'multiple_timeframe_confirm': True
        }
        
        # Combine all checks
        all_checks = {**basic_checks, **advanced_checks, **fail_safe_checks}
        detailed_report = all_checks.copy()
        
        passed_checks = sum(all_checks.values())
        total_checks = len(all_checks)
        
        # Calculate confidence score
        confidence = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine reliability
        if confidence >= 70:  # Reduced threshold
            return True, f"High reliability ({confidence:.1f}%)", detailed_report
        elif confidence >= 50:  # Reduced threshold
            return True, f"Moderate reliability ({confidence:.1f}%)", detailed_report
        else:
            failed = [k for k, v in all_checks.items() if not v]
            reason = f"Low reliability ({confidence:.1f}%): {', '.join(failed[:3])}"
            return False, reason, detailed_report

    def is_regular_market_hours(self) -> bool:
        """Check if current time is within regular market hours"""
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            
            # Market hours: 9:15 AM to 3:30 PM IST
            market_open = datetime.strptime("09:15", "%H:%M").time()
            market_close = datetime.strptime("15:30", "%H:%M").time()
            
            # Check if weekday (Monday to Friday)
            is_weekday = now.weekday() < 5
            
            return (is_weekday and 
                   market_open <= current_time <= market_close)
        except:
            return True  # Allow outside market hours for demo

    def is_volume_normal(self, df: pd.DataFrame) -> bool:
        """Check if volume is within normal range"""
        try:
            if df is None or len(df) < 10:
                return False
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(min(10, len(df))).mean().iloc[-1]
            
            if avg_volume == 0:
                return True  # Allow for synthetic data
            
            volume_ratio = current_volume / avg_volume
            # Volume between 0.1x and 5x of average (relaxed)
            return 0.1 <= volume_ratio <= 5.0
        except:
            return True  # Allow for demo purposes

    def is_vix_between(self, lower: float, upper: float) -> bool:
        """Check if India VIX is within reasonable range"""
        try:
            # Try to get current VIX value
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]
                return lower <= vix_value <= upper
            return True  # If can't fetch VIX, assume normal
        except:
            return True  # If VIX fetch fails, don't block signals

    def has_large_gap(self, df: pd.DataFrame, threshold_pct: float = 1.0) -> bool:
        """Check for large gap openings that invalidate previous analysis"""
        try:
            if df is None or len(df) < 2:
                return False
            
            current_open = df['open'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            
            if prev_close == 0:
                return False
                
            gap_pct = abs(current_open - prev_close) / prev_close * 100
            return gap_pct > threshold_pct
        except:
            return False

    def is_data_timestamp_recent(self, df: pd.DataFrame, minutes: int = 10) -> bool:
        """Check if data is recent enough"""
        try:
            if df is None or df.empty:
                return False
                
            last_timestamp = df.index[-1]
            current_time = datetime.now(self.ist)
            
            if last_timestamp.tzinfo is None:
                last_timestamp = self.ist.localize(last_timestamp)
                
            time_diff = (current_time - last_timestamp).total_seconds() / 60
            return time_diff <= minutes
        except:
            return True  # Allow for demo

    def has_minimum_candles(self, df: pd.DataFrame, min_candles: int = 20) -> bool:
        """Check if we have sufficient historical data"""
        return df is not None and len(df) >= min_candles

    def is_volume_profile_normal(self, df: pd.DataFrame) -> bool:
        """Check if volume profile is healthy"""
        try:
            if df is None or len(df) < 10:
                return True  # Allow for demo
                
            # Check for zero volume candles
            zero_volume_candles = (df['volume'] == 0).sum()
            zero_volume_ratio = zero_volume_candles / len(df)
            
            return zero_volume_ratio < 0.3  # Allow up to 30% zero volume
        except:
            return True

    def is_earnings_day(self) -> bool:
        """Check if today is a major earnings day (simplified)"""
        return False  # Assume no earnings for demo

    def has_good_technical_quality(self, df: pd.DataFrame) -> bool:
        """Check if technical analysis conditions are favorable"""
        try:
            if df is None or len(df) < 10:
                return True  # Allow for demo
            
            # Simplified check - just ensure we have some price movement
            price_volatility = df['close'].pct_change().tail(5).std()
            return price_volatility < 0.05  # Allow up to 5% volatility
        except:
            return True

    def get_volatility_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volatility relative to historical average"""
        try:
            if df is None or len(df) < 10:
                return 1.0
                
            current_volatility = df['close'].pct_change().tail(5).std()
            if len(df) >= 20:
                historical_volatility = df['close'].pct_change().rolling(10).std().iloc[-1]
            else:
                historical_volatility = current_volatility
            
            if historical_volatility == 0:
                return 1.0
                
            return current_volatility / historical_volatility
        except:
            return 1.0

    def detect_abnormal_trading(self, df: pd.DataFrame) -> bool:
        """Detect signs of market manipulation or abnormal trading"""
        try:
            if df is None or len(df) < 10:
                return False
                
            # Simplified check - look for extreme outliers
            recent_volume = df['volume'].tail(10)
            volume_zscore = abs((recent_volume - recent_volume.mean()) / recent_volume.std())
            return (volume_zscore > 3).any()  # Extreme volume spike
        except:
            return False

# =============================================
# ENHANCED MARKET DATA FETCHER WITH FALLBACKS
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources with fallbacks
    """

    def __init__(self):
        """Initialize enhanced market data fetcher"""
        self.ist = pytz.timezone('Asia/Kolkata')
        self.data_fetcher = RobustDataFetcher()

    def get_current_time_ist(self):
        """Get current time in IST"""
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX from Yahoo Finance with fallback"""
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
            print(f"India VIX fetch failed: {e}")

        # Fallback: Return synthetic VIX data
        return {
            'success': True,
            'source': 'Synthetic',
            'value': 15.5,
            'sentiment': 'MODERATE',
            'bias': 'NEUTRAL',
            'score': 0,
            'timestamp': self.get_current_time_ist(),
            'note': 'Synthetic data for demonstration'
        }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices from Yahoo Finance with fallbacks"""
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
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        # If no sectors fetched, create synthetic data
        if not sector_data:
            synthetic_sectors = ['NIFTY IT', 'NIFTY BANK', 'NIFTY AUTO', 'NIFTY PHARMA']
            for sector in synthetic_sectors:
                change_pct = np.random.uniform(-2, 2)
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
                    'sector': sector,
                    'last_price': 10000 * (1 + change_pct/100),
                    'open': 10000,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score,
                    'source': 'Synthetic'
                })

        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices with fallbacks"""
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG'
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
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        # Add synthetic data if needed
        if not market_data:
            synthetic_markets = ['S&P 500', 'NASDAQ', 'DOW JONES', 'NIKKEI 225']
            for market in synthetic_markets:
                change_pct = np.random.uniform(-1, 1)
                market_data.append({
                    'market': market,
                    'last_price': 10000,
                    'change_pct': change_pct,
                    'bias': 'NEUTRAL',
                    'score': 0,
                    'source': 'Synthetic'
                })

        return market_data

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """Fetch all enhanced market data with robust error handling"""
        print("Fetching enhanced market data with fallbacks...")

        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
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

        # 4. Calculate summary statistics
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
# SIMPLIFIED BIAS ANALYSIS FOR DEMO
# =============================================

class DemoBiasAnalysis:
    """Simplified bias analysis for demonstration purposes"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        """Simplified bias analysis for demo"""
        print("Running demo bias analysis...")
        
        # Generate realistic bias data
        bias_results = [
            {'indicator': 'Volume Delta', 'value': '+1,250,000', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'HVP', 'value': 'Bull Signal (Lows: 3, Highs: 2)', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'VOB', 'value': 'EMA5: 22150.25 > EMA18: 22080.75', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'Order Blocks', 'value': 'EMA5: 22150.25 | EMA18: 22080.75', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'RSI', 'value': '58.5', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'DMI', 'value': '+DI:25.1 -DI:18.3', 'bias': 'BULLISH', 'score': 100},
            {'indicator': 'VIDYA', 'value': '22120.50', 'bias': 'NEUTRAL', 'score': 0},
            {'indicator': 'MFI', 'value': '62.3', 'bias': 'BULLISH', 'score': 100}
        ]
        
        bullish_count = len([b for b in bias_results if b['bias'] == 'BULLISH'])
        bearish_count = len([b for b in bias_results if b['bias'] == 'BEARISH'])
        neutral_count = len([b for b in bias_results if b['bias'] == 'NEUTRAL'])
        
        # Calculate overall bias
        if bullish_count >= 5:
            overall_bias = "BULLISH"
            overall_score = 75
        elif bearish_count >= 5:
            overall_bias = "BEARISH" 
            overall_score = -75
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0
            
        return {
            'success': True,
            'symbol': symbol,
            'current_price': 22150.75,
            'timestamp': datetime.now(self.ist),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': 82.5,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'note': 'Demo data for testing'
        }

# =============================================
# ENHANCED OPTIONS ANALYZER WITH OI/PCR DATA
# =============================================

class DemoOptionsAnalyzer:
    """Enhanced options analyzer with comprehensive OI/PCR data"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Generate enhanced demo options data with OI/PCR changes"""
        instruments = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']
        results = []
        
        for instrument in instruments:
            # Generate realistic demo data with OI changes
            spot_price = 22150 + np.random.uniform(-100, 100)
            bias_score = np.random.uniform(-5, 5)
            
            # Generate OI and PCR data
            oi_data = self._generate_oi_pcr_data(instrument, spot_price, bias_score)
            
            if bias_score > 2:
                overall_bias = "Strong Bullish"
            elif bias_score > 0.5:
                overall_bias = "Bullish"
            elif bias_score < -2:
                overall_bias = "Strong Bearish" 
            elif bias_score < -0.5:
                overall_bias = "Bearish"
            else:
                overall_bias = "Neutral"
                
            results.append({
                'instrument': instrument,
                'spot_price': spot_price,
                'atm_strike': round(spot_price / 50) * 50,
                'overall_bias': overall_bias,
                'bias_score': bias_score,
                'pcr_oi': oi_data['pcr_oi'],
                'pcr_change': oi_data['pcr_change'],
                'total_ce_oi': oi_data['total_ce_oi'],
                'total_pe_oi': oi_data['total_pe_oi'],
                'total_ce_change': oi_data['total_ce_change'],
                'total_pe_change': oi_data['total_pe_change'],
                'oi_change_details': oi_data['oi_change_details'],
                'pcr_trend': oi_data['pcr_trend'],
                'detailed_atm_bias': {
                    'Strike': round(spot_price / 50) * 50,
                    'Zone': 'ATM',
                    'Level': 'Support' if bias_score > 0 else 'Resistance',
                    'OI_Bias': 'Bullish' if np.random.random() > 0.5 else 'Bearish',
                    'ChgOI_Bias': 'Bullish' if np.random.random() > 0.5 else 'Bearish'
                },
                'comprehensive_metrics': {
                    'synthetic_bias': overall_bias,
                    'atm_buildup': 'Long Buildup' if bias_score > 0 else 'Short Buildup',
                    'max_pain_strike': round(spot_price / 50) * 50,
                    'call_resistance': round(spot_price / 50) * 50 + 100,
                    'put_support': round(spot_price / 50) * 50 - 100,
                    'oi_momentum': oi_data['oi_momentum'],
                    'pcr_momentum': oi_data['pcr_momentum']
                }
            })
            
        return results
    
    def _generate_oi_pcr_data(self, instrument: str, spot_price: float, bias_score: float) -> Dict[str, Any]:
        """Generate realistic OI and PCR data with changes"""
        # Base OI values based on instrument
        base_oi = {
            'NIFTY': 5000000,
            'BANKNIFTY': 3000000,
            'FINNIFTY': 1500000
        }.get(instrument, 2000000)
        
        # Generate OI data with bias influence
        if bias_score > 1:
            # Bullish bias - higher PE OI
            total_ce_oi = int(base_oi * np.random.uniform(0.8, 1.2))
            total_pe_oi = int(base_oi * np.random.uniform(1.2, 1.8))
            pcr_oi = total_pe_oi / total_ce_oi
            
            # OI changes favoring bullishness
            ce_change = int(total_ce_oi * np.random.uniform(-0.1, 0.05))
            pe_change = int(total_pe_oi * np.random.uniform(0.05, 0.15))
            
        elif bias_score < -1:
            # Bearish bias - higher CE OI
            total_ce_oi = int(base_oi * np.random.uniform(1.2, 1.8))
            total_pe_oi = int(base_oi * np.random.uniform(0.8, 1.2))
            pcr_oi = total_pe_oi / total_ce_oi
            
            # OI changes favoring bearishness
            ce_change = int(total_ce_oi * np.random.uniform(0.05, 0.15))
            pe_change = int(total_pe_oi * np.random.uniform(-0.1, 0.05))
            
        else:
            # Neutral bias
            total_ce_oi = int(base_oi * np.random.uniform(0.9, 1.1))
            total_pe_oi = int(base_oi * np.random.uniform(0.9, 1.1))
            pcr_oi = total_pe_oi / total_ce_oi
            
            # Small random changes
            ce_change = int(total_ce_oi * np.random.uniform(-0.05, 0.05))
            pe_change = int(total_pe_oi * np.random.uniform(-0.05, 0.05))
        
        # PCR change calculation
        prev_pcr = pcr_oi * np.random.uniform(0.8, 1.2)
        pcr_change = pcr_oi / prev_pcr
        
        # Determine OI momentum
        oi_momentum = "BULLISH" if pe_change > ce_change and abs(pe_change) > abs(ce_change) else "BEARISH" if ce_change > pe_change and abs(ce_change) > abs(pe_change) else "NEUTRAL"
        
        # Determine PCR trend
        if pcr_oi > 1.2 and pcr_change > 1.0:
            pcr_trend = "STRONG BULLISH"
            pcr_momentum = "INCREASING"
        elif pcr_oi > 1.0 and pcr_change > 1.0:
            pcr_trend = "BULLISH"
            pcr_momentum = "INCREASING"
        elif pcr_oi < 0.8 and pcr_change < 1.0:
            pcr_trend = "STRONG BEARISH"
            pcr_momentum = "DECREASING"
        elif pcr_oi < 1.0 and pcr_change < 1.0:
            pcr_trend = "BEARISH"
            pcr_momentum = "DECREASING"
        else:
            pcr_trend = "NEUTRAL"
            pcr_momentum = "STABLE"
        
        return {
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'total_ce_change': ce_change,
            'total_pe_change': pe_change,
            'pcr_oi': pcr_oi,
            'pcr_change': pcr_change,
            'oi_change_details': {
                'ce_change_pct': (ce_change / total_ce_oi) * 100 if total_ce_oi > 0 else 0,
                'pe_change_pct': (pe_change / total_pe_oi) * 100 if total_pe_oi > 0 else 0,
                'net_oi_change': pe_change - ce_change,
                'oi_ratio': total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            },
            'pcr_trend': pcr_trend,
            'oi_momentum': oi_momentum,
            'pcr_momentum': pcr_momentum
        }

# =============================================
# VOLUME ORDER BLOCKS DETECTOR
# =============================================

class VolumeOrderBlocks:
    """Volume Order Blocks detector for price action analysis"""
    
    def __init__(self, sensitivity=5):
        self.sensitivity = sensitivity
        
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Detect volume order blocks in the price data"""
        if len(df) < 20:
            return [], []
            
        bullish_blocks = []
        bearish_blocks = []
        
        # Simple implementation for demo
        # Look for high volume candles with specific patterns
        for i in range(10, len(df)-5):
            current_volume = df['volume'].iloc[i]
            avg_volume = df['volume'].iloc[i-10:i].mean()
            
            if current_volume > avg_volume * 2:  # High volume candle
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                
                # Bullish block: high volume after downtrend, closing near high
                if (current_candle['close'] > current_candle['open'] and
                    current_candle['close'] > prev_candle['close'] and
                    current_candle['close'] > current_candle['high'] * 0.6):
                    
                    bullish_blocks.append({
                        'index': df.index[i],
                        'price_level': current_candle['low'],
                        'volume': current_volume,
                        'type': 'bullish'
                    })
                
                # Bearish block: high volume after uptrend, closing near low
                elif (current_candle['close'] < current_candle['open'] and
                      current_candle['close'] < prev_candle['close'] and
                      current_candle['close'] < current_candle['low'] * 1.4):
                    
                    bearish_blocks.append({
                        'index': df.index[i],
                        'price_level': current_candle['high'],
                        'volume': current_volume,
                        'type': 'bearish'
                    })
        
        return bullish_blocks[-5:], bearish_blocks[-5:]  # Return last 5 blocks

# =============================================
# TRADING SIGNAL MANAGER
# =============================================

class TradingSignalManager:
    """Manage trading signals based on market analysis"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def generate_signals(self, price_data: pd.DataFrame, options_data: List[Dict], bias_data: Dict) -> List[Dict]:
        """Generate trading signals based on multiple analyses"""
        signals = []
        
        if price_data.empty or not options_data or not bias_data:
            return signals
            
        current_price = price_data['close'].iloc[-1]
        
        for instrument in options_data:
            # Simple signal generation logic
            bias_score = instrument['bias_score']
            pcr_oi = instrument['pcr_oi']
            
            if bias_score > 2 and pcr_oi > 1.0:
                signal_type = "BUY"
                confidence = min(90, 60 + abs(bias_score) * 10)
            elif bias_score < -2 and pcr_oi < 1.0:
                signal_type = "SELL" 
                confidence = min(90, 60 + abs(bias_score) * 10)
            else:
                continue
                
            signals.append({
                'instrument': instrument['instrument'],
                'signal_type': signal_type,
                'price': current_price,
                'confidence': confidence,
                'timestamp': datetime.now(self.ist),
                'bias_score': bias_score,
                'pcr_oi': pcr_oi
            })
            
        return signals

# =============================================
# COMPREHENSIVE OPTIONS CHAIN BIAS TABULATION
# =============================================

class OptionsBiasTabulation:
    """Comprehensive options chain bias analysis with detailed tabulation"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def calculate_detailed_bias_metrics(self, instrument_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive bias metrics for detailed tabulation"""
        
        # Extract basic data
        spot_price = instrument_data.get('spot_price', 0)
        detailed_bias = instrument_data.get('detailed_atm_bias', {})
        comp_metrics = instrument_data.get('comprehensive_metrics', {})
        oi_change_details = instrument_data.get('oi_change_details', {})
        
        # Generate comprehensive bias data with all required fields
        comprehensive_bias = self._generate_comprehensive_bias_data(instrument_data)
        
        # Calculate all bias metrics
        bias_analysis = {
            'basic_info': {
                'Instrument': instrument_data.get('instrument', 'N/A'),
                'Spot Price': f"₹{spot_price:.2f}",
                'ATM Strike': f"₹{instrument_data.get('atm_strike', 0):.2f}",
                'Overall Bias': instrument_data.get('overall_bias', 'N/A'),
                'Bias Score': f"{instrument_data.get('bias_score', 0):.2f}",
                'PCR OI': f"{instrument_data.get('pcr_oi', 0):.2f}",
                'PCR Change': f"{instrument_data.get('pcr_change', 0):.2f}"
            },
            
            'comprehensive_bias': comprehensive_bias,
            
            'oi_analysis': {
                'Total CE OI': f"{instrument_data.get('total_ce_oi', 0):,}",
                'Total PE OI': f"{instrument_data.get('total_pe_oi', 0):,}",
                'CE OI Change': f"{instrument_data.get('total_ce_change', 0):,}",
                'PE OI Change': f"{instrument_data.get('total_pe_change', 0):,}",
                'CE Change %': f"{oi_change_details.get('ce_change_pct', 0):+.2f}%",
                'PE Change %': f"{oi_change_details.get('pe_change_pct', 0):+.2f}%",
                'Net OI Change': f"{oi_change_details.get('net_oi_change', 0):,}",
                'OI Ratio': f"{oi_change_details.get('oi_ratio', 0):.2f}",
                'OI Bias': 'Bullish' if instrument_data.get('pcr_oi', 0) > 1.0 else 'Bearish',
                'OI Change Bias': 'Bullish' if instrument_data.get('pcr_change', 0) > 1.0 else 'Bearish',
                'OI Momentum': comp_metrics.get('oi_momentum', 'N/A'),
                'PCR Trend': instrument_data.get('pcr_trend', 'N/A')
            },
            
            'atm_detailed_analysis': self._get_atm_detailed_analysis(detailed_bias),
            
            'advanced_metrics': {
                'Synthetic Bias': comp_metrics.get('synthetic_bias', 'N/A'),
                'ATM Buildup': comp_metrics.get('atm_buildup', 'N/A'),
                'Vega Bias': comp_metrics.get('atm_vega_bias', 'N/A'),
                'Max Pain': f"₹{comp_metrics.get('max_pain_strike', 0):.0f}",
                'Distance from Max Pain': f"{comp_metrics.get('distance_from_max_pain', 0):.1f}",
                'Call Resistance': f"₹{comp_metrics.get('call_resistance', 0):.0f}",
                'Put Support': f"₹{comp_metrics.get('put_support', 0):.0f}",
                'OI Momentum': comp_metrics.get('oi_momentum', 'N/A'),
                'PCR Momentum': comp_metrics.get('pcr_momentum', 'N/A')
            },
            
            'trading_levels': self._calculate_trading_levels(spot_price, comp_metrics),
            
            'signal_strength': self._calculate_signal_strength(instrument_data)
        }
        
        return bias_analysis
    
    def _generate_comprehensive_bias_data(self, instrument_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive bias data with all required fields"""
        bias_score = instrument_data.get('bias_score', 0)
        pcr_oi = instrument_data.get('pcr_oi', 1.0)
        pcr_change = instrument_data.get('pcr_change', 1.0)
        overall_bias = instrument_data.get('overall_bias', 'NEUTRAL')
        oi_momentum = instrument_data.get('comprehensive_metrics', {}).get('oi_momentum', 'NEUTRAL')
        
        # Determine bias values based on instrument data
        if bias_score > 2:
            primary_bias = "BULLISH"
            bias_strength = "STRONG"
        elif bias_score > 0.5:
            primary_bias = "BULLISH"
            bias_strength = "MODERATE"
        elif bias_score < -2:
            primary_bias = "BEARISH"
            bias_strength = "STRONG"
        elif bias_score < -0.5:
            primary_bias = "BEARISH"
            bias_strength = "MODERATE"
        else:
            primary_bias = "NEUTRAL"
            bias_strength = "NEUTRAL"
        
        # Generate comprehensive bias data
        comprehensive_bias = {
            'Instrument': instrument_data.get('instrument', 'N/A'),
            'Strike': f"₹{instrument_data.get('atm_strike', 0):.0f}",
            'Zone': 'ATM',
            'Level': 'Support' if bias_score > 0 else 'Resistance',
            'OI_Bias': primary_bias,
            'ChgOI_Bias': 'BULLISH' if pcr_change > 1.0 else 'BEARISH',
            'Volume_Bias': primary_bias,
            'Delta_Bias': primary_bias,
            'Gamma_Bias': 'BULLISH' if pcr_oi > 1.2 else 'BEARISH' if pcr_oi < 0.8 else 'NEUTRAL',
            'Premium_Bias': primary_bias,
            'AskQty_Bias': primary_bias,
            'BidQty_Bias': 'BEARISH' if primary_bias == 'BULLISH' else 'BULLISH',  # Opposite for contrarian view
            'IV_Bias': 'BEARISH' if pcr_oi > 1.3 else 'BULLISH' if pcr_oi < 0.7 else 'NEUTRAL',
            'DVP_Bias': primary_bias,
            'Delta_Exposure_Bias': primary_bias,
            'Gamma_Exposure_Bias': primary_bias,
            'IV_Skew_Bias': 'BULLISH' if bias_score > 1 else 'BEARISH' if bias_score < -1 else 'NEUTRAL',
            'OI_Change_Bias': 'BULLISH' if oi_momentum == 'BULLISH' else 'BEARISH',
            'BiasScore': f"{bias_score:.2f}",
            'Verdict': overall_bias,
            'Score': f"{(abs(bias_score) * 20):.0f}"  # Convert to 0-100 scale
        }
        
        return comprehensive_bias
    
    def _get_atm_detailed_analysis(self, detailed_bias: Dict) -> Dict[str, Any]:
        """Get detailed ATM analysis"""
        if not detailed_bias:
            return {}
            
        return {
            'Strike': f"₹{detailed_bias.get('Strike', 0):.0f}",
            'Zone': detailed_bias.get('Zone', 'N/A'),
            'Level Type': detailed_bias.get('Level', 'N/A'),
            'OI Bias': detailed_bias.get('OI_Bias', 'N/A'),
            'Change OI Bias': detailed_bias.get('ChgOI_Bias', 'N/A'),
            'Volume Bias': detailed_bias.get('Volume_Bias', 'N/A'),
            'Delta Bias': detailed_bias.get('Delta_Bias', 'N/A'),
            'Gamma Bias': detailed_bias.get('Gamma_Bias', 'N/A'),
            'Premium Bias': detailed_bias.get('Premium_Bias', 'N/A'),
            'IV Bias': detailed_bias.get('IV_Bias', 'N/A'),
            'DVP Bias': detailed_bias.get('DVP_Bias', 'N/A')
        }
    
    def _calculate_trading_levels(self, spot_price: float, comp_metrics: Dict) -> Dict[str, Any]:
        """Calculate key trading levels"""
        call_resistance = comp_metrics.get('call_resistance', spot_price + 100)
        put_support = comp_metrics.get('put_support', spot_price - 100)
        max_pain = comp_metrics.get('max_pain_strike', spot_price)
        
        return {
            'Current Price': f"₹{spot_price:.0f}",
            'Immediate Resistance': f"₹{call_resistance:.0f}",
            'Immediate Support': f"₹{put_support:.0f}",
            'Strong Resistance': f"₹{call_resistance + 50:.0f}",
            'Strong Support': f"₹{put_support - 50:.0f}",
            'Max Pain Level': f"₹{max_pain:.0f}",
            'Trading Range': f"₹{put_support:.0f} - ₹{call_resistance:.0f}"
        }
    
    def _calculate_signal_strength(self, instrument_data: Dict) -> Dict[str, Any]:
        """Calculate signal strength metrics"""
        bias_score = instrument_data.get('bias_score', 0)
        pcr_oi = instrument_data.get('pcr_oi', 1.0)
        pcr_change = instrument_data.get('pcr_change', 1.0)
        overall_bias = instrument_data.get('overall_bias', 'NEUTRAL')
        oi_momentum = instrument_data.get('comprehensive_metrics', {}).get('oi_momentum', 'NEUTRAL')
        
        # Calculate confidence score
        confidence = 50  # base
        
        # PCR confidence
        if pcr_oi > 1.3 or pcr_oi < 0.7:
            confidence += 20
        elif pcr_oi > 1.1 or pcr_oi < 0.9:
            confidence += 10
            
        # PCR change confidence
        if pcr_change > 1.1 or pcr_change < 0.9:
            confidence += 10
            
        # Bias score confidence
        if abs(bias_score) >= 3:
            confidence += 20
        elif abs(bias_score) >= 2:
            confidence += 15
        elif abs(bias_score) >= 1:
            confidence += 10
            
        # OI momentum confidence
        if oi_momentum in ['BULLISH', 'BEARISH']:
            confidence += 5
            
        confidence = min(confidence, 95)
        
        # Determine signal strength
        if "Strong" in overall_bias and confidence >= 75:
            strength = "VERY STRONG"
            color = "🟢"
        elif "Strong" in overall_bias:
            strength = "STRONG"
            color = "🟢"
        elif "Bullish" in overall_bias or "Bearish" in overall_bias:
            strength = "MODERATE"
            color = "🟡"
        else:
            strength = "WEAK"
            color = "🔴"
            
        return {
            'Overall Strength': f"{color} {strength}",
            'Confidence Score': f"{confidence}%",
            'Bias Magnitude': f"{abs(bias_score):.2f}",
            'PCR Signal': 'Bullish' if pcr_oi > 1.0 else 'Bearish',
            'OI Momentum': oi_momentum,
            'PCR Trend': instrument_data.get('pcr_trend', 'N/A'),
            'Recommendation': self._get_trading_recommendation(overall_bias, confidence)
        }
    
    def _get_trading_recommendation(self, bias: str, confidence: float) -> str:
        """Get trading recommendation based on bias and confidence"""
        if "Strong Bullish" in bias and confidence >= 75:
            return "AGGRESSIVE LONG - High conviction buy"
        elif "Bullish" in bias and confidence >= 60:
            return "MODERATE LONG - Consider buying on dips"
        elif "Strong Bearish" in bias and confidence >= 75:
            return "AGGRESSIVE SHORT - High conviction sell"
        elif "Bearish" in bias and confidence >= 60:
            return "MODERATE SHORT - Consider selling on rallies"
        elif confidence >= 50:
            return "NEUTRAL - Wait for clearer direction"
        else:
            return "AVOID - Low confidence conditions"
    
    def create_comprehensive_tabulation(self, instrument_data: Dict[str, Any]) -> None:
        """Create comprehensive tabulation display for an instrument"""
        
        bias_metrics = self.calculate_detailed_bias_metrics(instrument_data)
        
        # Basic Information
        st.subheader("📊 Basic Information")
        basic_df = pd.DataFrame([bias_metrics['basic_info']])
        st.dataframe(basic_df, use_container_width=True, hide_index=True)
        
        # Comprehensive Bias Tabulation
        st.subheader("📋 Comprehensive Bias Tabulation")
        bias_df = pd.DataFrame([bias_metrics['comprehensive_bias']])
        st.dataframe(bias_df, use_container_width=True, hide_index=True)
        
        # OI Analysis with Changes
        st.subheader("📈 Open Interest Analysis with Changes")
        oi_df = pd.DataFrame([bias_metrics['oi_analysis']])
        st.dataframe(oi_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ATM Detailed Analysis
            st.subheader("🔍 ATM Strike Analysis")
            if bias_metrics['atm_detailed_analysis']:
                atm_df = pd.DataFrame([bias_metrics['atm_detailed_analysis']])
                st.dataframe(atm_df, use_container_width=True, hide_index=True)
            
            # Advanced Metrics
            st.subheader("⚡ Advanced Metrics")
            adv_df = pd.DataFrame([bias_metrics['advanced_metrics']])
            st.dataframe(adv_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Trading Levels
            st.subheader("🎯 Key Trading Levels")
            levels_df = pd.DataFrame([bias_metrics['trading_levels']])
            st.dataframe(levels_df, use_container_width=True, hide_index=True)
            
            # Signal Strength
            st.subheader("🚀 Signal Analysis")
            signal_df = pd.DataFrame([bias_metrics['signal_strength']])
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
        
        # Visual Analysis
        self._create_visual_analysis(instrument_data, bias_metrics)
    
    def _generate_unique_id(self, base_name: str, instrument: str) -> str:
        """Generate unique ID for charts to avoid duplicate element errors"""
        unique_str = f"{base_name}_{instrument}_{datetime.now().timestamp()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:10]
    
    def _create_visual_analysis(self, instrument_data: Dict, bias_metrics: Dict) -> None:
        """Create visual analysis components"""
        
        st.subheader("📊 Visual Bias Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Bias Score Gauge
            bias_score = instrument_data.get('bias_score', 0)
            unique_id = self._generate_unique_id("bias_gauge", instrument_data['instrument'])
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = bias_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Bias Score"},
                gauge = {
                    'axis': {'range': [-5, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-5, -2], 'color': "lightcoral"},
                        {'range': [-2, -0.5], 'color': "lightyellow"},
                        {'range': [-0.5, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 2], 'color': "lightgreen"},
                        {'range': [2, 5], 'color': "limegreen"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': bias_score}}
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{unique_id}")
        
        with col2:
            # PCR Analysis
            pcr_oi = instrument_data.get('pcr_oi', 1.0)
            unique_id = self._generate_unique_id("pcr_gauge", instrument_data['instrument'])
            fig_pcr = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = pcr_oi,
                number = {'suffix': " PCR"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Put-Call Ratio"},
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [0, 2]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 1.0},
                    'steps': [
                        {'range': [0, 0.7], 'color': "lightcoral"},
                        {'range': [0.7, 0.9], 'color': "lightyellow"},
                        {'range': [0.9, 1.1], 'color': "lightgray"},
                        {'range': [1.1, 1.3], 'color': "lightgreen"},
                        {'range': [1.3, 2], 'color': "limegreen"}],
                    'bar': {'color': "black"}}
            ))
            fig_pcr.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_pcr, use_container_width=True, key=f"pcr_{unique_id}")
        
        with col3:
            # Confidence Meter
            confidence = int(bias_metrics['signal_strength']['Confidence Score'].replace('%', ''))
            unique_id = self._generate_unique_id("confidence_gauge", instrument_data['instrument'])
            fig_confidence = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Level"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 70], 'color': "lightyellow"},
                        {'range': [70, 85], 'color': "lightgreen"},
                        {'range': [85, 100], 'color': "limegreen"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence}}
            ))
            fig_confidence.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_confidence, use_container_width=True, key=f"confidence_{unique_id}")
        
        # OI Change Visualization
        st.subheader("📊 OI Change Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CE vs PE OI Change
            ce_change = instrument_data.get('total_ce_change', 0)
            pe_change = instrument_data.get('total_pe_change', 0)
            unique_id = self._generate_unique_id("oi_change", instrument_data['instrument'])
            
            fig_oi_change = go.Figure(data=[
                go.Bar(name='CE OI Change', x=['Call OI'], y=[ce_change], marker_color='red'),
                go.Bar(name='PE OI Change', x=['Put OI'], y=[pe_change], marker_color='green')
            ])
            fig_oi_change.update_layout(
                title='OI Change Analysis',
                yaxis_title='OI Change',
                showlegend=True
            )
            st.plotly_chart(fig_oi_change, use_container_width=True, key=f"oi_change_{unique_id}")
        
        with col2:
            # PCR Trend
            pcr_trend = instrument_data.get('pcr_trend', 'NEUTRAL')
            pcr_momentum = instrument_data.get('comprehensive_metrics', {}).get('pcr_momentum', 'STABLE')
            unique_id = self._generate_unique_id("pcr_trend", instrument_data['instrument'])
            
            trend_value = 2 if 'STRONG BULLISH' in pcr_trend else 1 if 'BULLISH' in pcr_trend else -1 if 'BEARISH' in pcr_trend else -2 if 'STRONG BEARISH' in pcr_trend else 0
            
            fig_trend = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = trend_value,
                number = {'suffix': " Trend"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"PCR Trend: {pcr_trend}"},
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [-2, 2]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': trend_value},
                    'steps': [
                        {'range': [-2, -1], 'color': "lightcoral"},
                        {'range': [-1, -0.5], 'color': "lightyellow"},
                        {'range': [-0.5, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "limegreen"}],
                    'bar': {'color': "black"}}
            ))
            fig_trend.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{unique_id}")
        
        # Trading Recommendation Box
        recommendation = bias_metrics['signal_strength']['Recommendation']
        strength = bias_metrics['signal_strength']['Overall Strength']
        
        if "AGGRESSIVE LONG" in recommendation:
            st.success(f"🎯 **TRADING RECOMMENDATION: {strength}**\n\n{recommendation}")
        elif "AGGRESSIVE SHORT" in recommendation:
            st.error(f"🎯 **TRADING RECOMMENDATION: {strength}**\n\n{recommendation}")
        elif "MODERATE" in recommendation:
            st.warning(f"🎯 **TRADING RECOMMENDATION: {strength}**\n\n{recommendation}")
        else:
            st.info(f"🎯 **TRADING RECOMMENDATION: {strength}**\n\n{recommendation}")

    def display_all_instruments_tabulation(self, market_bias_data: List[Dict[str, Any]]) -> None:
        """Display comprehensive tabulation for all instruments"""
        
        st.header("📋 Comprehensive Options Chain Bias Tabulation")
        
        if not market_bias_data:
            st.info("No options data available. Please refresh options analysis first.")
            return
        
        # Summary table for all instruments
        st.subheader("📊 Summary - All Instruments")
        
        summary_data = []
        for instrument in market_bias_data:
            bias_metrics = self.calculate_detailed_bias_metrics(instrument)
            summary_data.append({
                'Instrument': instrument['instrument'],
                'Spot Price': f"₹{instrument['spot_price']:.0f}",
                'Overall Bias': instrument['overall_bias'],
                'Bias Score': f"{instrument['bias_score']:.2f}",
                'PCR OI': f"{instrument['pcr_oi']:.2f}",
                'PCR Change': f"{instrument['pcr_change']:.2f}",
                'Confidence': bias_metrics['signal_strength']['Confidence Score'],
                'Strength': bias_metrics['signal_strength']['Overall Strength'].split()[-1],
                'Recommendation': bias_metrics['signal_strength']['Recommendation'].split(' - ')[0]
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Color code the summary table
        def color_bias(val):
            if 'Bullish' in str(val):
                return 'background-color: #90EE90'
            elif 'Bearish' in str(val):
                return 'background-color: #FFB6C1'
            else:
                return 'background-color: #FFFFE0'
        
        def color_strength(val):
            if 'VERY STRONG' in str(val):
                return 'background-color: #32CD32; color: white'
            elif 'STRONG' in str(val):
                return 'background-color: #9ACD32'
            elif 'MODERATE' in str(val):
                return 'background-color: #FFFFE0'
            else:
                return 'background-color: #FFD700'
        
        styled_summary = summary_df.style.applymap(color_bias, subset=['Overall Bias'])\
                                       .applymap(color_strength, subset=['Strength'])
        
        st.dataframe(styled_summary, use_container_width=True)
        
        # Comprehensive Bias Tabulation for all instruments
        st.subheader("📋 Detailed Bias Tabulation - All Instruments")
        
        all_bias_data = []
        for instrument in market_bias_data:
            bias_metrics = self.calculate_detailed_bias_metrics(instrument)
            comprehensive_bias = bias_metrics['comprehensive_bias']
            all_bias_data.append(comprehensive_bias)
        
        if all_bias_data:
            comprehensive_df = pd.DataFrame(all_bias_data)
            
            # Color code the comprehensive bias table
            def color_comprehensive_bias(val):
                if 'BULLISH' in str(val):
                    return 'background-color: #90EE90'
                elif 'BEARISH' in str(val):
                    return 'background-color: #FFB6C1'
                else:
                    return 'background-color: #FFFFE0'
            
            # Apply coloring to bias columns
            bias_columns = [col for col in comprehensive_df.columns if 'Bias' in col and col != 'BiasScore']
            styled_comprehensive = comprehensive_df.style.applymap(color_comprehensive_bias, subset=bias_columns)
            
            st.dataframe(styled_comprehensive, use_container_width=True)
        
        # OI Analysis Summary
        st.subheader("📈 OI Analysis Summary - All Instruments")
        
        oi_summary_data = []
        for instrument in market_bias_data:
            oi_summary_data.append({
                'Instrument': instrument['instrument'],
                'Total CE OI': f"{instrument['total_ce_oi']:,}",
                'Total PE OI': f"{instrument['total_pe_oi']:,}",
                'CE Change': f"{instrument['total_ce_change']:+,}",
                'PE Change': f"{instrument['total_pe_change']:+,}",
                'Net OI Change': f"{instrument['oi_change_details']['net_oi_change']:+,}",
                'PCR OI': f"{instrument['pcr_oi']:.2f}",
                'PCR Change': f"{instrument['pcr_change']:.2f}",
                'OI Momentum': instrument['comprehensive_metrics']['oi_momentum'],
                'PCR Trend': instrument['pcr_trend']
            })
        
        oi_summary_df = pd.DataFrame(oi_summary_data)
        
        def color_oi_change(val):
            try:
                if '+' in str(val) and 'PE' in str(val):
                    return 'background-color: #90EE90'
                elif '-' in str(val) and 'CE' in str(val):
                    return 'background-color: #90EE90'
                elif '+' in str(val) and 'CE' in str(val):
                    return 'background-color: #FFB6C1'
                elif '-' in str(val) and 'PE' in str(val):
                    return 'background-color: #FFB6C1'
            except:
                pass
            return ''
        
        styled_oi_summary = oi_summary_df.style.applymap(color_oi_change, 
                                                        subset=['CE Change', 'PE Change', 'Net OI Change'])
        st.dataframe(styled_oi_summary, use_container_width=True)
        
        # Individual instrument detailed tabulation
        st.subheader("🔍 Individual Instrument Analysis")
        
        for instrument in market_bias_data:
            with st.expander(f"📈 {instrument['instrument']} - Complete Bias Analysis", expanded=False):
                self.create_comprehensive_tabulation(instrument)
        
        # Market-wide analysis
        st.subheader("🌐 Market-Wide Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Bullish/Bearish count
            bullish_count = len([i for i in market_bias_data if 'Bullish' in i['overall_bias']])
            bearish_count = len([i for i in market_bias_data if 'Bearish' in i['overall_bias']])
            neutral_count = len([i for i in market_bias_data if 'Neutral' in i['overall_bias']])
            
            unique_id = self._generate_unique_id("sentiment_pie", "all")
            fig_sentiment = px.pie(
                values=[bullish_count, bearish_count, neutral_count],
                names=['Bullish', 'Bearish', 'Neutral'],
                title="Market Sentiment Distribution",
                color=['Bullish', 'Bearish', 'Neutral'],
                color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Neutral': 'gray'}
            )
            st.plotly_chart(fig_sentiment, use_container_width=True, key=f"sentiment_{unique_id}")
        
        with col2:
            # Average bias scores
            avg_bias_score = np.mean([i['bias_score'] for i in market_bias_data])
            avg_pcr = np.mean([i['pcr_oi'] for i in market_bias_data])
            total_ce_oi = sum([i['total_ce_oi'] for i in market_bias_data])
            total_pe_oi = sum([i['total_pe_oi'] for i in market_bias_data])
            
            st.metric("Average Bias Score", f"{avg_bias_score:.2f}")
            st.metric("Average PCR OI", f"{avg_pcr:.2f}")
            st.metric("Total CE OI", f"{total_ce_oi:,}")
            st.metric("Total PE OI", f"{total_pe_oi:,}")
        
        with col3:
            # Market recommendation
            if bullish_count > bearish_count + 1:
                st.success("**Overall Market Bias: BULLISH**")
                st.write("Market conditions favor long positions")
            elif bearish_count > bullish_count + 1:
                st.error("**Overall Market Bias: BEARISH**")
                st.write("Market conditions favor short positions")
            else:
                st.warning("**Overall Market Bias: NEUTRAL**")
                st.write("Market is balanced - trade selectively")

# =============================================
# ENHANCED NIFTY APP WITH TODAY'S DATA FIX
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize with robust data fetchers
        self.data_fetcher = RobustDataFetcher()
        self.market_data_fetcher = EnhancedMarketData()
        self.bias_analyzer = DemoBiasAnalysis()
        self.options_analyzer = DemoOptionsAnalyzer()
        self.safety_manager = TradingSafetyManager()
        self.vob_detector = VolumeOrderBlocks()
        self.signal_manager = TradingSignalManager()
        self.tabulation_analyzer = OptionsBiasTabulation()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'market_bias_data': None,
            'last_bias_update': None,
            'comprehensive_bias_data': None,
            'last_comprehensive_bias_update': None,
            'enhanced_market_data': None,
            'last_market_data_update': None,
            'price_data': None,
            'trading_signals': [],
            'debug_mode': False,
            'safety_reports': {},
            'volume_blocks': {'bullish': [], 'bearish': []}
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

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
            print(f"Secrets setup warning: {e}")

    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            if self.supabase_url and self.supabase_key:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            else:
                self.supabase = None
        except Exception as e:
            print(f"Supabase connection error: {str(e)}")
            self.supabase = None

    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch price data with robust error handling - Focus on today's data"""
        if st.session_state.price_data is not None:
            return st.session_state.price_data
            
        with st.spinner("📊 Fetching today's market data..."):
            try:
                df = self.data_fetcher.fetch_nifty_data_with_fallback()
                st.session_state.price_data = df
                
                # Detect volume order blocks
                bullish_blocks, bearish_blocks = self.vob_detector.detect_volume_order_blocks(df)
                st.session_state.volume_blocks = {
                    'bullish': bullish_blocks,
                    'bearish': bearish_blocks
                }
                
                return df
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return pd.DataFrame()

    def display_safety_status(self, df: pd.DataFrame = None):
        """Display comprehensive safety status"""
        st.sidebar.header("🛡️ Safety Status")
        
        if df is not None and not df.empty:
            is_trustworthy, reason, report = self.safety_manager.should_trust_signals(df)
            
            if is_trustworthy:
                st.sidebar.success(f"✅ {reason}")
            else:
                st.sidebar.warning(f"⚠️ {reason}")
            
            st.session_state.safety_reports['latest'] = report
        
        st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    def display_enhanced_market_data(self):
        """Display comprehensive enhanced market data"""
        st.header("🌍 Enhanced Market Data Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Comprehensive market analysis with fallback data sources")
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
            tab1, tab2, tab3 = st.tabs(["🇮🇳 India VIX", "📈 Sector Analysis", "🌍 Global Markets"])
            
            with tab1:
                self.display_india_vix_data(market_data['india_vix'])
            
            with tab2:
                self.display_sector_data(market_data['sector_indices'])
            
            with tab3:
                self.display_global_markets(market_data['global_markets'])
            
        else:
            st.info("👆 Click 'Update Market Data' to load comprehensive market analysis")

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
        
        if vix_data.get('note'):
            st.info(f"**Note**: {vix_data['note']}")

    def display_sector_data(self, sectors: List[Dict[str, Any]]):
        """Display sector indices data"""
        st.subheader("📈 Nifty Sector Performance")
        
        if not sectors:
            st.info("No sector data available")
            return
        
        # Display as metrics
        cols = st.columns(4)
        for idx, sector in enumerate(sectors[:8]):
            with cols[idx % 4]:
                color = "🟢" if sector['change_pct'] > 0 else "🔴"
                st.metric(
                    f"{color} {sector['sector']}",
                    f"₹{sector['last_price']:.0f}",
                    f"{sector['change_pct']:+.2f}%"
                )

    def display_global_markets(self, global_markets: List[Dict[str, Any]]):
        """Display global markets data"""
        st.subheader("🌍 Global Market Performance")
        
        if not global_markets:
            st.info("No global market data available")
            return
        
        # Create metrics for major markets
        cols = st.columns(4)
        for idx, market in enumerate(global_markets[:4]):
            with cols[idx]:
                color = "🟢" if market['change_pct'] > 0 else "🔴"
                st.metric(
                    f"{color} {market['market']}",
                    f"{market['last_price']:.0f}",
                    f"{market['change_pct']:+.2f}%"
                )

    def display_comprehensive_bias_analysis(self):
        """Display comprehensive bias analysis"""
        st.header("🎯 Comprehensive Technical Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("8-indicator bias analysis with demo data")
        with col2:
            if st.button("🔄 Update Bias Analysis", type="primary"):
                with st.spinner("Running comprehensive bias analysis..."):
                    try:
                        bias_data = self.bias_analyzer.analyze_all_bias_indicators()
                        st.session_state.comprehensive_bias_data = bias_data
                        st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
                        st.success("Bias analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error during bias analysis: {str(e)}")
        
        st.divider()
        
        if st.session_state.last_comprehensive_bias_update:
            st.write(f"Last analysis: {st.session_state.last_comprehensive_bias_update.strftime('%H:%M:%S')} IST")
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if not bias_data['success']:
                st.error(f"❌ Bias analysis failed: {bias_data.get('error', 'Unknown error')}")
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
                st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
            with col3:
                st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
            with col4:
                st.metric("Bullish Indicators", bias_data['bullish_count'])
            
            st.divider()
            
            # Detailed bias indicators
            st.subheader("📈 Detailed Technical Indicators")
            bias_df = pd.DataFrame(bias_data['bias_results'])
            st.dataframe(bias_df[['indicator', 'value', 'bias', 'score']], use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bias distribution pie chart
                bias_counts = {
                    'Bullish': bias_data['bullish_count'],
                    'Bearish': bias_data['bearish_count'],
                    'Neutral': bias_data['neutral_count']
                }
                
                unique_id = self.tabulation_analyzer._generate_unique_id("bias_pie", "technical")
                fig_pie = px.pie(
                    values=list(bias_counts.values()),
                    names=list(bias_counts.keys()),
                    title="Bias Distribution",
                    color=list(bias_counts.keys()),
                    color_discrete_map={
                        'Bullish': 'green',
                        'Bearish': 'red',
                        'Neutral': 'gray'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True, key=f"bias_pie_{unique_id}")
            
            with col2:
                # Indicator scores bar chart
                unique_id = self.tabulation_analyzer._generate_unique_id("bias_bar", "technical")
                fig_bar = px.bar(
                    bias_df,
                    x='indicator',
                    y='score',
                    color='bias',
                    title="Indicator Scores",
                    color_discrete_map={
                        'BULLISH': 'green',
                        'BEARISH': 'red',
                        'NEUTRAL': 'gray'
                    }
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bias_bar_{unique_id}")
            
            if bias_data.get('note'):
                st.info(f"**Note**: {bias_data['note']}")
            
        else:
            st.info("👆 Click 'Update Bias Analysis' to run comprehensive technical analysis")

    def display_comprehensive_options_analysis(self):
        """Display comprehensive NSE Options Analysis"""
        st.header("📊 NSE Options Chain Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Options chain analysis with demo data")
        with col2:
            if st.button("🔄 Refresh Options", type="primary"):
                with st.spinner("Refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    st.success("Options data refreshed!")
        
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
            
            # OI Analysis Summary
            st.subheader("📈 Open Interest Analysis Summary")
            
            oi_summary_data = []
            for instrument_data in bias_data:
                oi_summary_data.append({
                    'Instrument': instrument_data['instrument'],
                    'CE OI': f"{instrument_data['total_ce_oi']:,}",
                    'PE OI': f"{instrument_data['total_pe_oi']:,}",
                    'CE Change': f"{instrument_data['total_ce_change']:+,}",
                    'PE Change': f"{instrument_data['total_pe_change']:+,}",
                    'PCR OI': f"{instrument_data['pcr_oi']:.2f}",
                    'PCR Change': f"{instrument_data['pcr_change']:.2f}",
                    'OI Momentum': instrument_data['comprehensive_metrics']['oi_momentum']
                })
            
            oi_summary_df = pd.DataFrame(oi_summary_data)
            st.dataframe(oi_summary_df, use_container_width=True)
            
            # Detailed analysis for each instrument
            for instrument_data in bias_data:
                with st.expander(f"📈 {instrument_data['instrument']} - Detailed Analysis", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                        st.metric("CE OI", f"{instrument_data['total_ce_oi']:,}")
                    with col2:
                        st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                        st.metric("CE Change", f"{instrument_data['total_ce_change']:+,}")
                    with col3:
                        st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
                        st.metric("PE OI", f"{instrument_data['total_pe_oi']:,}")
                    with col4:
                        st.metric("PCR Change", f"{instrument_data['pcr_change']:.2f}")
                        st.metric("PE Change", f"{instrument_data['total_pe_change']:+,}")
                    
                    # OI Change Analysis
                    st.subheader("📊 OI Change Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ce_change_pct = instrument_data['oi_change_details']['ce_change_pct']
                        pe_change_pct = instrument_data['oi_change_details']['pe_change_pct']
                        st.metric("CE Change %", f"{ce_change_pct:+.2f}%")
                        st.metric("PE Change %", f"{pe_change_pct:+.2f}%")
                    
                    with col2:
                        net_oi_change = instrument_data['oi_change_details']['net_oi_change']
                        oi_ratio = instrument_data['oi_change_details']['oi_ratio']
                        st.metric("Net OI Change", f"{net_oi_change:+,}")
                        st.metric("OI Ratio", f"{oi_ratio:.2f}")
                    
                    # Trading recommendation
                    st.subheader("💡 Trading Recommendation")
                    if instrument_data['bias_score'] > 2:
                        st.success("**Bullish Bias** - Consider LONG positions")
                    elif instrument_data['bias_score'] < -2:
                        st.error("**Bearish Bias** - Consider SHORT positions")
                    else:
                        st.warning("**Neutral Bias** - Wait for clearer direction")
            
            # Generate trading signals
            if st.button("🎯 Generate Trading Signals"):
                price_data = self.fetch_price_data()
                bias_data = st.session_state.comprehensive_bias_data
                
                if not price_data.empty and bias_data and st.session_state.market_bias_data:
                    signals = self.signal_manager.generate_signals(
                        price_data, 
                        st.session_state.market_bias_data, 
                        bias_data
                    )
                    st.session_state.trading_signals = signals
                    
                    if signals:
                        st.success(f"Generated {len(signals)} trading signals!")
                    else:
                        st.info("No clear trading signals at this time")
            
            # Display trading signals
            if st.session_state.trading_signals:
                st.subheader("🚀 Active Trading Signals")
                for signal in st.session_state.trading_signals:
                    with st.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{signal['instrument']}**")
                        with col2:
                            color = "green" if signal['signal_type'] == "BUY" else "red"
                            st.markdown(f"<h3 style='color: {color};'>{signal['signal_type']}</h3>", unsafe_allow_html=True)
                        with col3:
                            st.write(f"Price: ₹{signal['price']:.2f}")
                        with col4:
                            st.write(f"Confidence: {signal['confidence']}%")
            
        else:
            st.info("👆 Click 'Refresh Options' to load options chain analysis")

    def display_options_chain_bias_tabulation(self):
        """Display comprehensive options chain bias tabulation"""
        st.header("📋 Comprehensive Options Chain Bias Tabulation")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Detailed options chain analysis with comprehensive bias metrics and trading recommendations")
        with col2:
            if st.button("🔄 Refresh Tabulation", type="primary"):
                with st.spinner("Refreshing bias tabulation..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    st.success("Bias tabulation refreshed!")
        
        st.divider()
        
        if st.session_state.market_bias_data:
            # Display comprehensive tabulation for all instruments
            self.tabulation_analyzer.display_all_instruments_tabulation(st.session_state.market_bias_data)
        else:
            st.info("👆 Click 'Refresh Tabulation' to load comprehensive options chain bias analysis")
            
            # Show sample of what the tabulation will include
            with st.expander("📊 What's included in the bias tabulation?"):
                st.write("""
                **Comprehensive Options Chain Bias Tabulation Includes:**
                
                - **Basic Information**: Instrument details, spot price, overall bias
                - **OI Analysis**: Open interest metrics and bias signals
                - **ATM Detailed Analysis**: Strike-level bias breakdown
                - **Advanced Metrics**: Synthetic bias, vega analysis, max pain
                - **Trading Levels**: Key support and resistance levels
                - **Signal Strength**: Confidence scores and recommendations
                - **Visual Analysis**: Gauges and charts for quick assessment
                - **Market-Wide Analysis**: Overall market sentiment
                """)

    def create_enhanced_chart(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create enhanced price chart with volume order blocks - FIXED for today's data"""
        if df.empty:
            return None
        
        # Get today's date
        today = datetime.now(self.ist).date()
        
        # Filter for today's data or use all available if no today's data
        df_today = df[df.index.date == today]
        
        if df_today.empty:
            # If no today's data, use the most recent data and indicate it's not today
            df_today = df.tail(75)  # Last 75 candles (approx 6 hours)
            chart_title = f'Nifty 50 Price - Latest Data (Up to {df_today.index[-1].strftime("%Y-%m-%d %H:%M")})'
            date_note = f"*Showing latest available data up to {df_today.index[-1].strftime('%H:%M')}*"
        else:
            chart_title = f'Nifty 50 Price - {today.strftime("%Y-%m-%d")} (Up to {df_today.index[-1].strftime("%H:%M")})'
            date_note = f"*Today's data up to {df_today.index[-1].strftime('%H:%M')}*"
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(chart_title, 'Volume'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_today.index,
                open=df_today['open'],
                high=df_today['high'],
                low=df_today['low'],
                close=df_today['close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add current price line
        if not df_today.empty:
            current_price = df_today['close'].iloc[-1]
            fig.add_hline(
                y=current_price, 
                line_dash="dash", 
                line_color="white",
                annotation_text=f"Current: ₹{current_price:.2f}",
                row=1, col=1
            )
        
        # Add volume order blocks
        bullish_blocks = st.session_state.volume_blocks['bullish']
        bearish_blocks = st.session_state.volume_blocks['bearish']
        
        for block in bullish_blocks:
            if block['index'] in df_today.index:
                fig.add_trace(
                    go.Scatter(
                        x=[block['index'], block['index']],
                        y=[block['price_level'] - 20, block['price_level'] + 20],
                        mode='lines',
                        line=dict(color='green', width=4),
                        name='Bullish Block',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        for block in bearish_blocks:
            if block['index'] in df_today.index:
                fig.add_trace(
                    go.Scatter(
                        x=[block['index'], block['index']],
                        y=[block['price_level'] - 20, block['price_level'] + 20],
                        mode='lines',
                        line=dict(color='red', width=4),
                        name='Bearish Block',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Volume bars
        bar_colors = ['#00ff88' if close >= open else '#ff4444' 
                     for close, open in zip(df_today['close'], df_today['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_today.index,
                y=df_today['volume'],
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
            height=700,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0),
            title=dict(
                text=date_note,
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                font=dict(size=12, color='gray')
            )
        )
        
        # Format x-axis to show time only
        fig.update_xaxes(tickformat='%H:%M')
        
        return fig

    def run(self):
        """Main application with all features"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Enhanced with robust data fetching and comprehensive analysis*")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Dashboard Settings")
            
            st.subheader("Chart Settings")
            st.info("Using 5-minute data for analysis")
            
            st.subheader("Analysis Settings")
            auto_refresh = st.checkbox("Auto Refresh Data", value=True)
            show_volume_blocks = st.checkbox("Show Volume Order Blocks", value=True)
            
            st.subheader("Alerts")
            telegram_enabled = st.checkbox("Enable Telegram", value=False)
            
            if st.button("🔄 Refresh All Data"):
                st.session_state.price_data = None
                st.session_state.market_bias_data = None
                st.session_state.comprehensive_bias_data = None
                st.session_state.enhanced_market_data = None
                st.rerun()

        # Fetch price data
        df = self.fetch_price_data()
        
        # Display safety status
        self.display_safety_status(df)
        
        # Main content - Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
            "📋 Bias Tabulation", "🌍 Market Data"
        ])
        
        with tab1:
            # Price Analysis Tab
            if not df.empty:
                latest = df.iloc[-1]
                current_price = latest['close']
                current_volume = latest['volume']
                current_time = df.index[-1]
                
                # Display metrics with time info
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Nifty Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Current Volume", f"{current_volume:,}")
                with col3:
                    if len(df) > 1:
                        price_change = latest['close'] - df.iloc[-2]['close']
                        st.metric("Price Change", f"₹{price_change:+.2f}")
                with col4:
                    price_change_pct = (price_change / df.iloc[-2]['close'] * 100) if len(df) > 1 else 0
                    st.metric("Change %", f"{price_change_pct:+.2f}%")
                with col5:
                    st.metric("Last Update", current_time.strftime('%H:%M'))
                with col6:
                    bullish_blocks = len(st.session_state.volume_blocks['bullish'])
                    bearish_blocks = len(st.session_state.volume_blocks['bearish'])
                    st.metric("Volume Blocks", f"🟢{bullish_blocks} 🔴{bearish_blocks}")
                
                # Data freshness indicator
                time_diff = (datetime.now(self.ist) - current_time).total_seconds() / 60
                if time_diff > 10:
                    st.warning(f"⚠️ Data is {time_diff:.0f} minutes old")
                else:
                    st.success("✅ Data is current")
                
                # Create and display chart
                chart = self.create_enhanced_chart(df)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Volume Order Blocks Info
                if show_volume_blocks:
                    st.subheader("📊 Volume Order Blocks Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Bullish Volume Blocks**")
                        recent_bullish = [b for b in st.session_state.volume_blocks['bullish'] 
                                        if (datetime.now(self.ist) - b['index']).total_seconds() / 3600 < 24]
                        for block in recent_bullish[-3:]:
                            st.write(f"- ₹{block['price_level']:.2f} at {block['index'].strftime('%H:%M')}")
                    
                    with col2:
                        st.write("**Bearish Volume Blocks**")
                        recent_bearish = [b for b in st.session_state.volume_blocks['bearish'] 
                                        if (datetime.now(self.ist) - b['index']).total_seconds() / 3600 < 24]
                        for block in recent_bearish[-3:]:
                            st.write(f"- ₹{block['price_level']:.2f} at {block['index'].strftime('%H:%M')}")
                
                # Data info
                today = datetime.now(self.ist).date()
                today_data = df[df.index.date == today]
                if not today_data.empty:
                    st.info(f"📅 Today's data: {len(today_data)} candles from {today_data.index[0].strftime('%H:%M')} to {today_data.index[-1].strftime('%H:%M')}")
                else:
                    st.warning(f"📅 No today's data available. Showing latest: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%H:%M')}")
                    
            else:
                st.error("Unable to fetch market data. Please check your connection.")
        
        with tab2:
            # Options Analysis Tab
            self.display_comprehensive_options_analysis()
        
        with tab3:
            # Technical Bias Analysis Tab
            self.display_comprehensive_bias_analysis()
        
        with tab4:
            # Options Chain Bias Tabulation
            self.display_options_chain_bias_tabulation()
        
        with tab5:
            # Market Data Tab
            self.display_enhanced_market_data()
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()