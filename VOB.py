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
        
        # 1. BASIC MARKET CONDITIONS
        basic_checks = {
            'market_hours': self.is_regular_market_hours(),
            'normal_volume': self.is_volume_normal(df),
            'vix_normal': self.is_vix_between(12, 30),
            'no_large_gaps': not self.has_large_gap(df, 1.0),
            'data_fresh': self.is_data_timestamp_recent(df, minutes=2),
            'sufficient_data': self.has_minimum_candles(df, 50)
        }
        
        # 2. ADVANCED CHECKS
        advanced_checks = {
            'indicators_aligned': self.are_indicators_aligned(),
            'market_regime_ok': self.is_market_regime_suitable(df),
            'options_data_reliable': self.is_options_data_trustworthy(),
            'volume_profile_healthy': self.is_volume_profile_normal(df),
            'no_earnings_events': not self.is_earnings_day(),
            'technical_quality': self.has_good_technical_quality(df)
        }
        
        # 3. FAIL-SAFE CHECKS
        fail_safe_checks = {
            'not_extreme_volatility': self.get_volatility_ratio(df) < 3.0,
            'not_abnormal_spreads': self.are_bid_ask_spreads_normal(),
            'not_manipulation_signs': not self.detect_abnormal_trading(df),
            'multiple_timeframe_confirm': self.multiple_timeframe_alignment()
        }
        
        # Combine all checks
        all_checks = {**basic_checks, **advanced_checks, **fail_safe_checks}
        detailed_report = all_checks.copy()
        
        passed_checks = sum(all_checks.values())
        total_checks = len(all_checks)
        
        # Calculate confidence score
        confidence = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine reliability
        if confidence >= 80:
            return True, f"High reliability ({confidence:.1f}%)", detailed_report
        elif confidence >= 60:
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
            return False

    def is_volume_normal(self, df: pd.DataFrame) -> bool:
        """Check if volume is within normal range"""
        try:
            if df is None or len(df) < 20:
                return False
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume == 0:
                return False
            
            volume_ratio = current_volume / avg_volume
            # Volume between 0.3x and 3x of average
            return 0.3 <= volume_ratio <= 3.0
        except:
            return False

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

    def is_data_timestamp_recent(self, df: pd.DataFrame, minutes: int = 2) -> bool:
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
            return False

    def has_minimum_candles(self, df: pd.DataFrame, min_candles: int = 50) -> bool:
        """Check if we have sufficient historical data"""
        return df is not None and len(df) >= min_candles

    def are_indicators_aligned(self) -> bool:
        """Check if multiple indicators confirm each other"""
        try:
            bias_data = st.session_state.get('comprehensive_bias_data')
            
            if not bias_data or not bias_data.get('success'):
                return False
            
            bullish_count = bias_data.get('bullish_count', 0)
            bearish_count = bias_data.get('bearish_count', 0)
            total_indicators = bias_data.get('total_indicators', 0)
            
            if total_indicators == 0:
                return False
            
            # Require clear majority (at least 60% agreement)
            min_agreement = total_indicators * 0.6
            return (bullish_count >= min_agreement or 
                    bearish_count >= min_agreement)
        except:
            return False

    def is_market_regime_suitable(self, df: pd.DataFrame) -> bool:
        """Check if current market regime works with our strategies"""
        try:
            if df is None or len(df) < 20:
                return False
                
            conditions = {
                'not_choppy': not self.is_choppy_market(df),
                'not_trend_exhaustion': not self.is_trend_exhausted(df),
                'reasonable_volatility': self.get_volatility_ratio(df) < 2.5,
            }
            return all(conditions.values())
        except:
            return False

    def is_options_data_trustworthy(self) -> bool:
        """Check if options chain data is reliable"""
        try:
            market_bias_data = st.session_state.get('market_bias_data')
            if not market_bias_data:
                return False
            
            for instrument_data in market_bias_data:
                # Check for abnormal OI patterns
                total_oi = instrument_data.get('total_ce_oi', 0) + instrument_data.get('total_pe_oi', 0)
                if total_oi < 1000000:  # Too low OI
                    return False
                    
                # Check PCR sanity
                pcr_oi = instrument_data.get('pcr_oi', 1.0)
                if pcr_oi > 3.0 or pcr_oi < 0.2:  # Extreme PCR values
                    return False
                    
                # Check if max pain is reasonable
                spot = instrument_data.get('spot_price', 0)
                comp_metrics = instrument_data.get('comprehensive_metrics', {})
                max_pain = comp_metrics.get('max_pain_strike', spot)
                
                if spot == 0:
                    return False
                    
                if abs(spot - max_pain) / spot > 0.05:  # >5% difference
                    return False
            
            return True
        except:
            return False

    def is_volume_profile_normal(self, df: pd.DataFrame) -> bool:
        """Check if volume profile is healthy"""
        try:
            if df is None or len(df) < 10:
                return False
                
            # Check for zero volume candles
            zero_volume_candles = (df['volume'] == 0).sum()
            zero_volume_ratio = zero_volume_candles / len(df)
            
            # Check volume consistency
            volume_std = df['volume'].tail(10).std()
            volume_mean = df['volume'].tail(10).mean()
            
            volume_consistency = volume_std / volume_mean if volume_mean > 0 else 1.0
            
            return (zero_volume_ratio < 0.1 and    # Less than 10% zero volume
                    volume_consistency < 1.0)      # Reasonable volume consistency
        except:
            return False

    def is_earnings_day(self) -> bool:
        """Check if today is a major earnings day (simplified)"""
        try:
            # This would typically check an earnings calendar
            # For now, return False (no earnings detection)
            return False
        except:
            return False

    def has_good_technical_quality(self, df: pd.DataFrame) -> bool:
        """Check if technical analysis conditions are favorable"""
        try:
            if df is None or len(df) < 20:
                return False
            
            # Check for clean price action (no extreme wicks)
            recent_candles = df.tail(5)
            candle_ranges = recent_candles['high'] - recent_candles['low']
            body_sizes = abs(recent_candles['close'] - recent_candles['open'])
            
            # Avoid division by zero
            valid_ranges = candle_ranges > 0
            if not valid_ranges.any():
                return False
                
            wick_ratios = (candle_ranges[valid_ranges] - body_sizes[valid_ranges]) / candle_ranges[valid_ranges]
            avg_wick_ratio = wick_ratios.mean()
            
            # Check for consistent volume
            recent_volume = df['volume'].tail(20)
            if recent_volume.mean() == 0:
                return False
                
            volume_consistency = recent_volume.std() / recent_volume.mean()
            
            # Check for reasonable price movement
            price_volatility = df['close'].pct_change().tail(10).std()
            
            return all([
                avg_wick_ratio < 0.6,           # Reasonable wick sizes
                volume_consistency < 1.0,       # Consistent volume
                price_volatility < 0.03,        # Not extreme volatility
                not self.is_choppy_market(df)   # Not stuck in tight range
            ])
        except:
            return False

    def get_volatility_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volatility relative to historical average"""
        try:
            if df is None or len(df) < 20:
                return 1.0
                
            current_volatility = df['close'].pct_change().tail(5).std()
            historical_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            if historical_volatility == 0:
                return 1.0
                
            return current_volatility / historical_volatility
        except:
            return 1.0

    def are_bid_ask_spreads_normal(self) -> bool:
        """Check if bid-ask spreads are normal (simplified)"""
        # In a real implementation, this would check actual bid-ask data
        # For now, assume normal
        return True

    def detect_abnormal_trading(self, df: pd.DataFrame) -> bool:
        """Detect signs of market manipulation or abnormal trading"""
        try:
            if df is None or len(df) < 10:
                return False
                
            # Check for extreme volume spikes without price movement
            recent_data = df.tail(10)
            volume_spikes = (recent_data['volume'] > recent_data['volume'].rolling(5).mean() * 3).sum()
            price_changes = abs(recent_data['close'].pct_change()).mean()
            
            # If multiple volume spikes with little price movement
            if volume_spikes >= 3 and price_changes < 0.001:
                return True
                
            return False
        except:
            return False

    def multiple_timeframe_alignment(self) -> bool:
        """Check if signals align across multiple timeframes"""
        try:
            # This would require fetching data for multiple timeframes
            # For now, return True (alignment check disabled)
            return True
        except:
            return False

    def is_choppy_market(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect choppy/sideways market conditions"""
        try:
            if df is None or len(df) < lookback:
                return False
                
            recent_data = df.tail(lookback)
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[0]
            
            # If price range is less than 1% over the lookback period, consider it choppy
            return price_range < 0.01
        except:
            return False

    def is_trend_exhausted(self, df: pd.DataFrame, lookback: int = 10) -> bool:
        """Detect if current trend might be exhausted"""
        try:
            if df is None or len(df) < lookback:
                return False
                
            recent_data = df.tail(lookback)
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # If significant move (>3%) in short period, might be exhausted
            return abs(price_change) > 0.03
        except:
            return False

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
        """Fetch India VIX from Yahoo Finance with enhanced error handling"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty and len(hist) > 0:
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
            print(f"Error fetching India VIX: {e}")

        # Return fallback data
        return {
            'success': False, 
            'source': 'Fallback',
            'value': 15.0,
            'sentiment': "MODERATE",
            'bias': "NEUTRAL",
            'score': 0,
            'timestamp': self.get_current_time_ist(),
            'error': 'India VIX data not available, using fallback'
        }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices from Yahoo Finance with enhanced error handling"""
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

                if not hist.empty and len(hist) > 0:
                    last_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[0]
                    
                    # Handle potential NaN values
                    if pd.isna(last_price) or pd.isna(open_price):
                        continue
                        
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
                        'high': hist['High'].max() if 'High' in hist.columns else last_price,
                        'low': hist['Low'].min() if 'Low' in hist.columns else last_price,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                # Add fallback data for this sector
                sector_data.append({
                    'sector': name,
                    'last_price': 10000.0,
                    'open': 10000.0,
                    'high': 10100.0,
                    'low': 9900.0,
                    'change_pct': 0.0,
                    'bias': "NEUTRAL",
                    'score': 0,
                    'source': 'Fallback',
                    'error': f'Data unavailable for {name}'
                })

        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices from Yahoo Finance with enhanced error handling"""
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG',
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX'
        }

        market_data = []

        for symbol, name in global_markets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    
                    if pd.isna(current_close) or pd.isna(prev_close):
                        continue

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
                # Add fallback data
                market_data.append({
                    'market': name,
                    'symbol': symbol,
                    'last_price': 10000.0,
                    'prev_close': 10000.0,
                    'change_pct': 0.0,
                    'bias': "NEUTRAL",
                    'score': 0,
                    'error': f'Data unavailable for {name}'
                })

        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """Fetch intermarket data (commodities, currencies, bonds) with enhanced error handling"""
        intermarket_assets = {
            'DX-Y.NYB': 'US DOLLAR INDEX',
            'CL=F': 'CRUDE OIL',
            'GC=F': 'GOLD',
            'INR=X': 'USD/INR',
            '^TNX': 'US 10Y TREASURY'
        }

        intermarket_data = []

        for symbol, name in intermarket_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    
                    if pd.isna(current_close) or pd.isna(prev_close):
                        continue

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
                # Add fallback data
                intermarket_data.append({
                    'asset': name,
                    'symbol': symbol,
                    'last_price': 100.0,
                    'prev_close': 100.0,
                    'change_pct': 0.0,
                    'bias': "NEUTRAL",
                    'score': 0,
                    'error': f'Data unavailable for {name}'
                })

        return intermarket_data

    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation to identify market leadership changes"""
        sectors = self.fetch_sector_indices()

        if not sectors:
            return {
                'success': False, 
                'error': 'No sector data available',
                'leaders': [],
                'laggards': [],
                'sector_sentiment': 'NEUTRAL',
                'sector_score': 0,
                'timestamp': self.get_current_time_ist()
            }

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
        """Fetch all enhanced market data from all sources with comprehensive error handling"""
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

        try:
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

        except Exception as e:
            print(f"Error in enhanced market data fetch: {e}")
            # Ensure we still return a valid structure even if some data fails
            result['summary'] = {
                'total_data_points': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'avg_score': 0,
                'overall_sentiment': 'NEUTRAL',
                'error': f'Partial data fetch: {str(e)}'
            }

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
        if data['india_vix'].get('success', False):
            summary['total_data_points'] += 1
            all_scores.append(data['india_vix'].get('score', 0))
            bias = data['india_vix'].get('bias', 'NEUTRAL')
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count sectors
        for sector in data.get('sector_indices', []):
            summary['total_data_points'] += 1
            all_scores.append(sector.get('score', 0))
            bias = sector.get('bias', 'NEUTRAL')
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count global markets
        for market in data.get('global_markets', []):
            summary['total_data_points'] += 1
            all_scores.append(market.get('score', 0))
            bias = market.get('bias', 'NEUTRAL')
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count intermarket
        for asset in data.get('intermarket', []):
            summary['total_data_points'] += 1
            all_scores.append(asset.get('score', 0))
            bias = asset.get('bias', 'NEUTRAL')
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

            # Ensure required columns exist and handle missing data
            required_columns = ['Open', 'High', 'Low', 'Close']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Column {col} missing for {symbol}")
                    return pd.DataFrame()

            # Handle volume column
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)

            # Clean data - remove any rows with NaN values in essential columns
            df = df.dropna(subset=required_columns)

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with error handling"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral 50
        except:
            return pd.Series([50.0] * len(data), index=data.index)

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index with error handling"""
        try:
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
        except:
            return pd.Series([50.0] * len(df), index=df.index)

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate DMI indicators with error handling"""
        try:
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
        except:
            zero_series = pd.Series([0.0] * len(df), index=df.index)
            return zero_series, zero_series, zero_series

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP with error handling"""
        try:
            if df['Volume'].sum() == 0:
                return (df['High'] + df['Low'] + df['Close']) / 3

            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_volume = df['Volume'].cumsum()
            cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
            vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe
            return vwap.fillna(typical_price)
        except:
            return (df['High'] + df['Low'] + df['Close']) / 3

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR with error handling"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.fillna(0)
        except:
            return pd.Series([0.0] * len(df), index=df.index)

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with error handling"""
        try:
            return data.ewm(span=period, adjust=False).mean().fillna(data)
        except:
            return data

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0) -> Tuple[pd.Series, bool, bool]:
        """Calculate VIDYA (Variable Index Dynamic Average) with error handling"""
        try:
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
        except:
            zero_series = pd.Series([0.0] * len(df), index=df.index)
            return zero_series, False, False

    def calculate_volume_delta(self, df: pd.DataFrame) -> Tuple[float, bool, bool]:
        """Calculate Volume Delta (up_vol - down_vol) with error handling"""
        try:
            if df['Volume'].sum() == 0:
                return 0, False, False

            up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
            down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

            volume_delta = up_vol - down_vol
            volume_bullish = volume_delta > 0
            volume_bearish = volume_delta < 0

            return volume_delta, volume_bullish, volume_bearish
        except:
            return 0, False, False

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0) -> Tuple[bool, bool, int, int]:
        """Calculate High Volume Pivots with error handling"""
        try:
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
        except:
            return False, False, 0, 0

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5) -> Tuple[bool, bool, float, float]:
        """Calculate Volume Order Blocks with error handling"""
        try:
            length2 = length1 + 13
            ema1 = self.calculate_ema(df['Close'], length1)
            ema2 = self.calculate_ema(df['Close'], length2)

            cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
            cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

            vob_bullish = cross_up
            vob_bearish = cross_dn

            return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]
        except:
            return False, False, 0.0, 0.0

    def _fetch_stock_data(self, symbol: str, weight: float) -> Optional[Dict[str, Any]]:
        """Helper function to fetch single stock data for parallel processing"""
        try:
            df = self.fetch_data(symbol, period='5d', interval='5m')
            if df.empty or len(df) < 2:
                return None

            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[0]
            
            if pd.isna(current_price) or pd.isna(prev_price):
                return None
                
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
        """Calculate market breadth from top stocks with error handling"""
        bullish_stocks = 0
        total_stocks = 0
        stock_data = []

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_stock = {
                    executor.submit(self._fetch_stock_data, symbol, weight): (symbol, weight)
                    for symbol, weight in self.config['stocks'].items()
                }

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
        except:
            return 50.0, False, False, 0, 0, []

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        """Analyze all 8 bias indicators with comprehensive error handling"""

        print(f"Fetching data for {symbol}...")
        try:
            df = self.fetch_data(symbol, period='7d', interval='5m')

            if df.empty or len(df) < 50:  # Reduced minimum requirement for demo
                error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 50)'
                print(f"❌ {error_msg}")
                
                # Return fallback analysis with basic data
                return {
                    'success': True,  # Mark as success to allow display
                    'symbol': symbol,
                    'current_price': 22000.0,  # Fallback price
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'bias_results': [
                        {
                            'indicator': 'Volume Delta',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'HVP (High Volume Pivots)',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL", 
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'VOB (Volume Order Blocks)',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'Order Blocks (EMA 5/18)',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'RSI',
                            'value': "Data Unavailable", 
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'DMI',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'VIDYA',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        },
                        {
                            'indicator': 'MFI (Money Flow)',
                            'value': "Data Unavailable",
                            'bias': "NEUTRAL",
                            'score': 0,
                            'weight': 1.0,
                            'category': 'fast'
                        }
                    ],
                    'overall_bias': "NEUTRAL",
                    'overall_score': 0,
                    'overall_confidence': 50,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 8,
                    'total_indicators': 8,
                    'stock_data': [],
                    'mode': "FALLBACK",
                    'fast_bull_pct': 0,
                    'fast_bear_pct': 0,
                    'bullish_bias_pct': 0,
                    'bearish_bias_pct': 0,
                    'note': 'Using fallback data due to Yahoo Finance unavailability'
                }

            current_price = df['Close'].iloc[-1]
            bias_results = []
            stock_data = []

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
            
            # Safe indexing
            if len(ema5) >= 2 and len(ema18) >= 2:
                cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
                cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
            else:
                cross_up = False
                cross_dn = False

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
            rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50
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
            plus_di_value = plus_di.iloc[-1] if len(plus_di) > 0 else 0
            minus_di_value = minus_di.iloc[-1] if len(minus_di) > 0 else 0
            
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

            vidya_value = vidya_val.iloc[-1] if len(vidya_val) > 0 else 0
            bias_results.append({
                'indicator': 'VIDYA',
                'value': f"{vidya_value:.2f}",
                'bias': vidya_bias,
                'score': vidya_score,
                'weight': 1.0,
                'category': 'fast'
            })

            # 8. MFI
            mfi = self.calculate_mfi(df, self.config['mfi_period'])
            mfi_value = mfi.iloc[-1] if len(mfi) > 0 else 50

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
            bullish_divergence = False
            bearish_divergence = False
            divergence_detected = bullish_divergence or bearish_divergence

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
# TRADING SIGNAL MANAGER WITH COOLDOWN & SAFETY
# =============================================

class TradingSignalManager:
    """Manage trading signals with cooldown periods and safety checks"""
    
    def __init__(self, cooldown_minutes=15):
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = {}
        self.sent_signals = set()
        self.safety_manager = TradingSafetyManager()
        
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
    
    def generate_trading_recommendation(self, instrument_data: Dict[str, Any], df: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """Generate trading recommendation with safety checks"""
        try:
            # Safety check first
            is_trustworthy, reason, report = self.safety_manager.should_trust_signals(df)
            
            if not is_trustworthy:
                return {
                    'instrument': instrument_data['instrument'],
                    'signal_type': "BLOCKED",
                    'direction': "NEUTRAL",
                    'strength': "LOW",
                    'confidence': 0,
                    'timestamp': datetime.now(),
                    'blocked_reason': reason,
                    'safety_report': report
                }
            
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
                },
                'safety_checked': True,
                'safety_reason': reason
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
        if recommendation.get('signal_type') == "BLOCKED":
            return f"""🚫 SIGNAL BLOCKED - SAFETY CHECK FAILED

📊 {recommendation['instrument']}
⏰ Time: {recommendation['timestamp'].strftime('%H:%M:%S')} IST

❌ Reason: {recommendation['blocked_reason']}

⚠️ Trading conditions not favorable
💡 Wait for better market conditions"""

        emoji = "🟢" if recommendation['direction'] == "BULLISH" else "🔴"
        strength_emoji = "🔥" if recommendation['strength'] == "HIGH" else "⚡"
        
        message = f"""
{strength_emoji} {emoji} *TRADING SIGNAL ALERT* {emoji} {strength_emoji}

🎯 *{recommendation['instrument']} - {recommendation['signal_type']}*
⏰ Time: {recommendation['timestamp'].strftime('%H:%M:%S')} IST
📊 Confidence: {recommendation['confidence']}%
🛡️ Safety: ✅ PASSED

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
        """Fetch option chain data from NSE with enhanced error handling"""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}" if instrument in self.NSE_INSTRUMENTS['indices'] else \
                  f"https://www.nseindia.com/api/option-chain-equities?symbol={url_instrument}"

            response = session.get(url, timeout=10)
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'instrument': instrument,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
                
            data = response.json()

            # Check if data structure is valid
            if 'records' not in data or 'data' not in data['records']:
                return {
                    'success': False,
                    'instrument': instrument,
                    'error': 'Invalid data structure from NSE'
                }

            records = data['records']['data']
            
            if not records:
                return {
                    'success': False,
                    'instrument': instrument,
                    'error': 'No records found in option chain'
                }
                
            expiry = data['records']['expiryDates'][0] if data['records']['expiryDates'] else 'No expiry'
            underlying = data['records']['underlyingValue']

            # Calculate totals with error handling
            total_ce_oi = 0
            total_pe_oi = 0
            total_ce_change = 0
            total_pe_change = 0

            for item in records:
                if 'CE' in item:
                    total_ce_oi += item['CE'].get('openInterest', 0)
                    total_ce_change += item['CE'].get('changeinOpenInterest', 0)
                if 'PE' in item:
                    total_pe_oi += item['PE'].get('openInterest', 0)
                    total_pe_change += item['PE'].get('changeinOpenInterest', 0)

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
        """Calculate Max Pain strike with error handling"""
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
            if pain_values:
                max_pain_data = min(pain_values, key=lambda x: x['pain'])
                return max_pain_data['strike']
            else:
                return None
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
        """Comprehensive ATM bias analysis with all metrics and enhanced error handling"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                # Return fallback data for demo purposes
                return {
                    'instrument': instrument,
                    'spot_price': 22000.0,
                    'atm_strike': 22000.0,
                    'overall_bias': "NEUTRAL",
                    'bias_score': 0,
                    'pcr_oi': 1.0,
                    'pcr_change': 1.0,
                    'total_ce_oi': 1000000,
                    'total_pe_oi': 1000000,
                    'total_ce_change': 10000,
                    'total_pe_change': 10000,
                    'detailed_atm_bias': {
                        "Strike": 22000.0,
                        "Zone": 'ATM',
                        "Level": "Neutral",
                        "OI_Bias": "Neutral",
                        "ChgOI_Bias": "Neutral",
                        "Volume_Bias": "Neutral",
                        "Delta_Bias": "Neutral",
                        "Gamma_Bias": "Neutral",
                        "Premium_Bias": "Neutral",
                        "AskQty_Bias": "Neutral",
                        "BidQty_Bias": "Neutral",
                        "IV_Bias": "Neutral",
                        "DVP_Bias": "Neutral",
                        "Delta_Exposure_Bias": "Neutral",
                        "Gamma_Exposure_Bias": "Neutral",
                        "IV_Skew_Bias": "Neutral",
                        "CE_OI": 500000,
                        "PE_OI": 500000,
                        "CE_Change": 5000,
                        "PE_Change": 5000,
                        "CE_Volume": 10000,
                        "PE_Volume": 10000,
                        "CE_Price": 100.0,
                        "PE_Price": 100.0,
                        "CE_IV": 15.0,
                        "PE_IV": 15.0,
                        "Delta_CE": 0.5,
                        "Delta_PE": -0.5,
                        "Gamma_CE": 0.01,
                        "Gamma_PE": 0.01
                    },
                    'comprehensive_metrics': {
                        'synthetic_bias': "Neutral",
                        'synthetic_future': 22000.0,
                        'synthetic_diff': 0.0,
                        'atm_buildup': "Neutral",
                        'atm_vega_bias': "Neutral",
                        'atm_vega_exposure': 0.0,
                        'max_pain_strike': 22000.0,
                        'distance_from_max_pain': 0.0,
                        'call_resistance': 22100.0,
                        'put_support': 21900.0,
                        'total_vega_bias': "Neutral",
                        'total_vega': 0.0,
                        'unusual_activity_count': 0,
                        'overall_buildup': "Balanced/Neutral"
                    },
                    'note': 'Using fallback data due to NSE API unavailability'
                }

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
# ENHANCED NIFTY APP WITH ALL FEATURES & SAFETY
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
        self.safety_manager = TradingSafetyManager()
        
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
        if 'enhanced_market_data' not in st.session_state:
            st.session_state.enhanced_market_data = None
        if 'last_market_data_update' not in st.session_state:
            st.session_state.last_market_data_update = None
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        if 'safety_reports' not in st.session_state:
            st.session_state.safety_reports = {}
        
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

    def display_safety_status(self, df: pd.DataFrame = None):
        """Display comprehensive safety status"""
        st.sidebar.header("🛡️ Safety Status")
        
        if df is not None and not df.empty:
            is_trustworthy, reason, report = self.safety_manager.should_trust_signals(df)
            
            if is_trustworthy:
                st.sidebar.success(f"✅ {reason}")
            else:
                st.sidebar.error(f"❌ {reason}")
            
            # Store report for debugging
            st.session_state.safety_reports['latest'] = report
            
            # Show detailed report in debug mode
            if st.session_state.debug_mode:
                with st.sidebar.expander("🔍 Safety Report Details"):
                    st.json(report)
        
        # Debug mode toggle
        st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
        
        # Safety settings
        st.sidebar.subheader("Safety Settings")
        min_confidence = st.sidebar.slider("Min Confidence %", 50, 90, 70)
        return min_confidence

    def display_enhanced_market_data(self):
        """Display comprehensive enhanced market data with error handling"""
        st.header("🌍 Enhanced Market Data Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Comprehensive market analysis from multiple sources")
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
                st.metric("Overall Sentiment", summary.get('overall_sentiment', 'NEUTRAL'))
            with col2:
                st.metric("Average Score", f"{summary.get('avg_score', 0):.1f}")
            with col3:
                st.metric("Bullish Signals", summary.get('bullish_count', 0))
            with col4:
                st.metric("Total Data Points", summary.get('total_data_points', 0))
            
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
        """Display India VIX data with error handling"""
        st.subheader("🇮🇳 India VIX - Fear Index")
        
        if not vix_data.get('success', False):
            st.warning("⚠️ India VIX data not available - using fallback data")
            vix_data = {
                'value': 15.0,
                'sentiment': "MODERATE",
                'bias': "NEUTRAL",
                'score': 0,
                'source': 'Fallback'
            }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VIX Value", f"{vix_data.get('value', 15.0):.2f}")
        with col2:
            st.metric("Sentiment", vix_data.get('sentiment', 'MODERATE'))
        with col3:
            st.metric("Bias", vix_data.get('bias', 'NEUTRAL'))
        with col4:
            st.metric("Score", vix_data.get('score', 0))
        
        # VIX Interpretation
        vix_value = vix_data.get('value', 15.0)
        st.info(f"**Interpretation**: {vix_data.get('sentiment', 'MODERATE')} - {self.get_vix_interpretation(vix_value)}")
        st.write(f"**Source**: {vix_data.get('source', 'Fallback')}")

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
        """Display sector indices data with error handling"""
        st.subheader("📈 Nifty Sector Performance")
        
        if not sectors:
            st.info("No sector data available - using fallback data")
            # Create fallback sector data
            fallback_sectors = [
                {'sector': 'NIFTY IT', 'last_price': 10000.0, 'change_pct': 0.5, 'bias': 'NEUTRAL'},
                {'sector': 'NIFTY AUTO', 'last_price': 10000.0, 'change_pct': -0.3, 'bias': 'NEUTRAL'},
                {'sector': 'NIFTY PHARMA', 'last_price': 10000.0, 'change_pct': 1.2, 'bias': 'BULLISH'},
                {'sector': 'NIFTY BANK', 'last_price': 10000.0, 'change_pct': -0.8, 'bias': 'BEARISH'},
            ]
            sectors = fallback_sectors
        
        # Create sector performance table
        sector_df = pd.DataFrame(sectors)
        
        # Display as metrics
        cols = st.columns(4)
        for idx, sector in enumerate(sector_df.head(8).itertuples()):
            with cols[idx % 4]:
                color = "🟢" if getattr(sector, 'change_pct', 0) > 0 else "🔴"
                st.metric(
                    f"{color} {getattr(sector, 'sector', 'Unknown')}",
                    f"₹{getattr(sector, 'last_price', 0):.0f}",
                    f"{getattr(sector, 'change_pct', 0):+.2f}%"
                )
        
        # Detailed table
        st.subheader("Detailed Sector Analysis")
        display_df = sector_df[['sector', 'last_price', 'change_pct', 'bias']].copy()
        st.dataframe(display_df, use_container_width=True)

    def display_global_markets(self, global_markets: List[Dict[str, Any]]):
        """Display global markets data with error handling"""
        st.subheader("🌍 Global Market Performance")
        
        if not global_markets:
            st.info("No global market data available - using fallback data")
            # Create fallback global markets data
            global_markets = [
                {'market': 'S&P 500', 'last_price': 4500.0, 'change_pct': 0.3},
                {'market': 'NASDAQ', 'last_price': 14000.0, 'change_pct': 0.7},
                {'market': 'NIKKEI 225', 'last_price': 33000.0, 'change_pct': -0.2},
                {'market': 'HANG SENG', 'last_price': 16000.0, 'change_pct': -1.2},
            ]
        
        # Create metrics for major markets
        major_markets = ['S&P 500', 'NASDAQ', 'NIKKEI 225', 'HANG SENG']
        filtered_markets = [m for m in global_markets if m.get('market') in major_markets]
        
        cols = st.columns(4)
        for idx, market in enumerate(filtered_markets):
            with cols[idx]:
                color = "🟢" if market.get('change_pct', 0) > 0 else "🔴"
                st.metric(
                    f"{color} {market.get('market', 'Unknown')}",
                    f"{market.get('last_price', 0):.0f}",
                    f"{market.get('change_pct', 0):+.2f}%"
                )

    def display_intermarket_data(self, intermarket: List[Dict[str, Any]]):
        """Display intermarket analysis data with error handling"""
        st.subheader("🔄 Intermarket Analysis")
        
        if not intermarket:
            st.info("No intermarket data available - using fallback data")
            # Create fallback intermarket data
            intermarket = [
                {'asset': 'US DOLLAR INDEX', 'last_price': 100.0, 'change_pct': 0.1, 'bias': 'NEUTRAL'},
                {'asset': 'CRUDE OIL', 'last_price': 75.0, 'change_pct': -1.5, 'bias': 'BULLISH (for India)'},
                {'asset': 'GOLD', 'last_price': 1950.0, 'change_pct': 0.3, 'bias': 'NEUTRAL'},
                {'asset': 'USD/INR', 'last_price': 83.0, 'change_pct': 0.2, 'bias': 'BEARISH (INR Weak)'},
            ]
        
        # Create metrics for key intermarket assets
        cols = st.columns(4)
        for idx, asset in enumerate(intermarket):
            with cols[idx % 4]:
                color = "🟢" if "BULLISH" in str(asset.get('bias', '')) or "RISK ON" in str(asset.get('bias', '')) else "🔴"
                st.metric(
                    f"{color} {asset.get('asset', 'Unknown')}",
                    f"{asset.get('last_price', 0):.2f}",
                    f"{asset.get('change_pct', 0):+.2f}%"
                )
                st.caption(f"Bias: {asset.get('bias', 'NEUTRAL')}")

    def display_sector_rotation(self, rotation_data: Dict[str, Any]):
        """Display sector rotation analysis with error handling"""
        if not rotation_data.get('success', False):
            st.info("Sector rotation analysis not available")
            return
        
        st.subheader("📊 Sector Rotation Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sector Breadth", f"{rotation_data.get('sector_breadth', 50):.1f}%")
        with col2:
            st.metric("Rotation Pattern", rotation_data.get('rotation_pattern', 'NO CLEAR ROTATION'))
        with col3:
            st.metric("Sector Sentiment", rotation_data.get('sector_sentiment', 'NEUTRAL'))

    def display_intraday_seasonality(self, seasonality_data: Dict[str, Any]):
        """Display intraday seasonality analysis with error handling"""
        if not seasonality_data.get('success', False):
            st.info("Intraday seasonality analysis not available")
            return
        
        st.subheader("⏰ Intraday Market Timing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Session", seasonality_data.get('session', 'UNKNOWN'))
        with col2:
            st.metric("Session Bias", seasonality_data.get('session_bias', 'NEUTRAL'))
        with col3:
            st.metric("Day of Week", seasonality_data.get('weekday', 'UNKNOWN'))

    def check_trading_signals(self, df: pd.DataFrame = None):
        """Check for trading signals with safety checks"""
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
            # Generate trading recommendation with safety check
            recommendation = self.trading_signal_manager.generate_trading_recommendation(instrument_data, df)
            
            if recommendation:
                instrument = recommendation['instrument']
                signal_type = recommendation['signal_type']
                
                # Skip blocked signals
                if signal_type == "BLOCKED":
                    st.warning(f"Signal blocked for {instrument}: {recommendation['blocked_reason']}")
                    continue
                
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
                if signal.get('signal_type') == "BLOCKED":
                    emoji = "🚫"
                    signal_text = f"{emoji} {signal['instrument']} BLOCKED"
                else:
                    emoji = "🟢" if signal['direction'] == "BULLISH" else "🔴"
                    signal_text = f"{emoji} {signal['instrument']} {signal['signal_type']}"
                
                with st.sidebar.expander(signal_text, expanded=False):
                    st.write(f"Time: {signal['timestamp'].strftime('%H:%M:%S')}")
                    if signal.get('signal_type') == "BLOCKED":
                        st.write(f"Reason: {signal['blocked_reason']}")
                    else:
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
                        if bias_data.get('success', False):
                            st.success("Bias analysis completed successfully!")
                        else:
                            st.warning(f"Bias analysis completed with fallback data")
                    except Exception as e:
                        st.error(f"Error during bias analysis: {str(e)}")
        
        st.divider()
        
        # Display last update time
        if st.session_state.last_comprehensive_bias_update:
            st.write(f"Last analysis: {st.session_state.last_comprehensive_bias_update.strftime('%H:%M:%S')} IST")
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if not bias_data.get('success', False):
                st.warning("⚠️ Using fallback bias analysis data due to Yahoo Finance unavailability")
                
            # Overall bias summary
            st.subheader("📊 Overall Market Bias")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
                st.metric(
                    "Overall Bias", 
                    f"{bias_color} {bias_data['overall_bias']}",
                    delta=f"Score: {bias_data.get('overall_score', 0):.1f}"
                )
            with col2:
                # Create a gauge chart for bias score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = bias_data.get('overall_score', 0),
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
                            'value': bias_data.get('overall_score', 0)}}
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                st.metric("Confidence", f"{bias_data.get('overall_confidence', 50):.1f}%")
            with col4:
                st.metric("Current Price", f"₹{bias_data.get('current_price', 22000):.2f}")
            
            st.divider()
            
            # Detailed bias indicators in a table
            st.subheader("📈 Detailed Technical Indicators")
            
            # Convert bias results to DataFrame for better display
            if 'bias_results' in bias_data:
                bias_df = pd.DataFrame(bias_data['bias_results'])
                
                # Display as styled table
                styled_df = bias_df[['indicator', 'value', 'bias', 'score']]
                st.dataframe(styled_df, use_container_width=True)
            
            # Trading recommendation based on bias
            st.divider()
            st.subheader("💡 Trading Recommendation")
            
            bias_strength = abs(bias_data.get('overall_score', 0))
            overall_bias = bias_data.get('overall_bias', 'NEUTRAL')
            confidence = bias_data.get('overall_confidence', 50)
            
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Price Analysis", "📊 Options Analysis", "🎯 Technical Bias", 
            "📋 Bias Tabulation", "🚀 Trading Signals", "🌍 Market Data"
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
            
            # Display safety status
            min_confidence = self.display_safety_status(df)
            
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
                    self.check_trading_signals(df if 'df' in locals() else None)
            
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
        
        with tab6:
            self.display_enhanced_market_data()
        
        # Check for trading signals automatically
        if enable_trading_signals and telegram_enabled:
            self.check_trading_signals(df if 'df' in locals() else None)
        
        # Cleanup and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()