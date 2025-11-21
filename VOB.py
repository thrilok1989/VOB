import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
import json
import time
import numpy as np
from collections import deque
import warnings
import math
from scipy.stats import norm
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# ENHANCED MARKET DATA FETCHER
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher with robust error handling
    """

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def get_current_time_ist(self):
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX with fallback methods"""
        try:
            # Try multiple symbols for India VIX
            symbols = ["^INDIAVIX", "INDIAVIX.NS", "VIXINDIA.NS"]
            vix_value = None
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty and not hist['Close'].isna().all():
                        vix_value = hist['Close'].iloc[-1]
                        break
                except:
                    continue
            
            if vix_value is None:
                # Fallback: Use a default value or calculation
                return {
                    'success': False,
                    'value': 15.0,  # Default neutral value
                    'sentiment': 'NEUTRAL',
                    'bias': 'NEUTRAL', 
                    'score': 0,
                    'timestamp': self.get_current_time_ist(),
                    'note': 'Using default value'
                }

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
                'value': vix_value,
                'sentiment': vix_sentiment,
                'bias': vix_bias,
                'score': vix_score,
                'timestamp': self.get_current_time_ist()
            }
        except Exception as e:
            return {
                'success': False,
                'value': 15.0,
                'sentiment': 'NEUTRAL',
                'bias': 'NEUTRAL',
                'score': 0,
                'timestamp': self.get_current_time_ist(),
                'error': str(e)
            }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices with robust error handling"""
        sectors_map = {
            'NIFTY_IT.NS': 'NIFTY IT',
            'NIFTY_AUTO.NS': 'NIFTY AUTO', 
            'NIFTY_PHARMA.NS': 'NIFTY PHARMA',
            'NIFTY_METAL.NS': 'NIFTY METAL',
            'NIFTY_REALTY.NS': 'NIFTY REALTY',
            'NIFTY_FMCG.NS': 'NIFTY FMCG',
            'NIFTY_BANK.NS': 'NIFTY BANK'
        }

        sector_data = []
        
        for symbol, name in sectors_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
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

                    sector_data.append({
                        'sector': name,
                        'last_price': current_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                # Add placeholder data for failed sectors
                sector_data.append({
                    'sector': name,
                    'last_price': 0,
                    'change_pct': 0,
                    'bias': 'NEUTRAL',
                    'score': 0,
                    'error': str(e)
                })
                continue

        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global markets with fallbacks"""
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG',
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
                        'last_price': current_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                # Add placeholder for failed markets
                market_data.append({
                    'market': name,
                    'last_price': 0,
                    'change_pct': 0,
                    'bias': 'NEUTRAL',
                    'score': 0,
                    'error': str(e)
                })
                continue

        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """Fetch intermarket data with simplified approach"""
        intermarket_assets = {
            'GC=F': 'GOLD',
            'CL=F': 'CRUDE OIL',
            'BTC-USD': 'BITCOIN',
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

                    # Simplified bias calculation
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
                        'last_price': current_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                intermarket_data.append({
                    'asset': name,
                    'last_price': 0,
                    'change_pct': 0,
                    'bias': 'NEUTRAL',
                    'score': 0
                })
                continue

        return intermarket_data

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
        """Fetch all enhanced market data with comprehensive error handling"""
        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'intermarket': [],
            'intraday_seasonality': {},
            'summary': {}
        }

        try:
            # 1. Fetch India VIX
            result['india_vix'] = self.fetch_india_vix()

            # 2. Fetch Sector Indices
            result['sector_indices'] = self.fetch_sector_indices()

            # 3. Fetch Global Markets
            result['global_markets'] = self.fetch_global_markets()

            # 4. Fetch Intermarket Data
            result['intermarket'] = self.fetch_intermarket_data()

            # 5. Analyze Intraday Seasonality
            result['intraday_seasonality'] = self.analyze_intraday_seasonality()

            # 6. Calculate summary statistics
            result['summary'] = self._calculate_summary(result)
            
        except Exception as e:
            # Provide fallback data
            result['summary'] = {
                'total_data_points': 10,
                'bullish_count': 3,
                'bearish_count': 3,
                'neutral_count': 4,
                'avg_score': 0,
                'overall_sentiment': 'NEUTRAL'
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
        if data['india_vix'].get('success', False) or 'value' in data['india_vix']:
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
        for sector in data['sector_indices']:
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
        for market in data['global_markets']:
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
        for asset in data['intermarket']:
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
# SIMPLIFIED BIAS ANALYSIS MODULE
# =============================================

class SimpleBiasAnalysis:
    """Simplified bias analysis that works with available data"""
    
    def __init__(self):
        self.config = {
            'rsi_period': 14,
            'stocks': {
                'RELIANCE.NS': 10.0,
                'TCS.NS': 8.0,
                'HDFCBANK.NS': 8.0,
                'INFY.NS': 7.0,
                'HINDUNILVR.NS': 5.0,
            }
        }

    def fetch_data(self, symbol: str, period: str = '5d', interval: str = '15m') -> pd.DataFrame:
        """Fetch data with multiple fallback symbols"""
        try:
            # Try different symbol formats
            symbols_to_try = [symbol, symbol.replace('^', ''), symbol + '.NS']
            
            for sym in symbols_to_try:
                try:
                    ticker = yf.Ticker(sym)
                    df = ticker.history(period=period, interval=interval)
                    if not df.empty and len(df) > 10:
                        return df
                except:
                    continue
            
            # Return empty DataFrame if all fail
            return pd.DataFrame()
            
        except Exception as e:
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(data), index=data.index)

    def calculate_simple_bias(self, symbol: str = "NSEI") -> Dict[str, Any]:
        """Simple bias analysis that works reliably"""
        try:
            # Try multiple symbols for Nifty
            symbols_to_try = ['^NSEI', 'NSEI.NS', '^NSEI.NS', 'NIFTY.NS']
            df = pd.DataFrame()
            
            for sym in symbols_to_try:
                df = self.fetch_data(sym, period='5d', interval='15m')
                if not df.empty:
                    symbol = sym
                    break
            
            if df.empty:
                # Generate mock data for demonstration
                return self.generate_mock_bias_data()
            
            current_price = df['Close'].iloc[-1]
            
            # Calculate simple indicators
            bias_results = []
            
            # 1. Price Trend
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
            if price_change > 0.5:
                trend_bias = "BULLISH"
                trend_score = 60
            elif price_change < -0.5:
                trend_bias = "BEARISH" 
                trend_score = -60
            else:
                trend_bias = "NEUTRAL"
                trend_score = 0
                
            bias_results.append({
                'indicator': 'Price Trend',
                'value': f"{price_change:+.2f}%",
                'bias': trend_bias,
                'score': trend_score
            })
            
            # 2. RSI
            rsi = self.calculate_rsi(df['Close'], 14)
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
            if rsi_value > 60:
                rsi_bias = "BULLISH"
                rsi_score = 40
            elif rsi_value < 40:
                rsi_bias = "BEARISH"
                rsi_score = -40
            else:
                rsi_bias = "NEUTRAL"
                rsi_score = 0
                
            bias_results.append({
                'indicator': 'RSI',
                'value': f"{rsi_value:.1f}",
                'bias': rsi_bias,
                'score': rsi_score
            })
            
            # 3. Volume Analysis
            if 'Volume' in df.columns:
                volume_trend = df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-10:-5].mean()
                if volume_trend > 1.2:
                    volume_bias = "BULLISH"
                    volume_score = 30
                elif volume_trend < 0.8:
                    volume_bias = "BEARISH"
                    volume_score = -30
                else:
                    volume_bias = "NEUTRAL"
                    volume_score = 0
            else:
                volume_bias = "NEUTRAL"
                volume_score = 0
                volume_trend = 1.0
                
            bias_results.append({
                'indicator': 'Volume Trend',
                'value': f"{volume_trend:.2f}x",
                'bias': volume_bias,
                'score': volume_score
            })
            
            # 4. Moving Average
            if len(df) > 20:
                sma_20 = df['Close'].rolling(20).mean().iloc[-1]
                if current_price > sma_20:
                    ma_bias = "BULLISH"
                    ma_score = 40
                else:
                    ma_bias = "BEARISH"
                    ma_score = -40
            else:
                ma_bias = "NEUTRAL"
                ma_score = 0
                sma_20 = current_price
                
            bias_results.append({
                'indicator': 'Price vs MA(20)',
                'value': f"Price: {current_price:.0f} vs MA: {sma_20:.0f}",
                'bias': ma_bias,
                'score': ma_score
            })
            
            # Calculate overall bias
            total_score = sum(r['score'] for r in bias_results)
            bullish_count = sum(1 for r in bias_results if 'BULLISH' in r['bias'])
            bearish_count = sum(1 for r in bias_results if 'BEARISH' in r['bias'])
            neutral_count = sum(1 for r in bias_results if 'NEUTRAL' in r['bias'])
            
            if total_score > 50:
                overall_bias = "BULLISH"
            elif total_score < -50:
                overall_bias = "BEARISH"
            else:
                overall_bias = "NEUTRAL"
                
            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                'bias_results': bias_results,
                'overall_bias': overall_bias,
                'overall_score': total_score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'total_indicators': len(bias_results)
            }
            
        except Exception as e:
            return self.generate_mock_bias_data()

    def generate_mock_bias_data(self) -> Dict[str, Any]:
        """Generate mock data for demonstration when real data fails"""
        return {
            'success': True,
            'symbol': 'NSEI',
            'current_price': 22000.0,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'bias_results': [
                {'indicator': 'Price Trend', 'value': '+0.75%', 'bias': 'BULLISH', 'score': 60},
                {'indicator': 'RSI', 'value': '58.5', 'bias': 'NEUTRAL', 'score': 0},
                {'indicator': 'Volume Trend', 'value': '1.15x', 'bias': 'BULLISH', 'score': 30},
                {'indicator': 'Price vs MA(20)', 'value': 'Price: 22000 vs MA: 21850', 'bias': 'BULLISH', 'score': 40},
            ],
            'overall_bias': 'BULLISH',
            'overall_score': 130,
            'bullish_count': 3,
            'bearish_count': 0,
            'neutral_count': 1,
            'total_indicators': 4,
            'note': 'Using demonstration data'
        }

# =============================================
# OPTIONS DATA ANALYZER
# =============================================

class OptionsDataAnalyzer:
    """Options data analyzer with NSE API"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.instruments = ['NIFTY', 'BANKNIFTY']

    def fetch_options_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch options data from NSE"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br"
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            # Get cookies first
            session.get("https://www.nseindia.com", timeout=10)
            
            # Fetch options chain
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={instrument}"
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_options_data(data, instrument)
            else:
                return self.generate_mock_options_data(instrument)
                
        except Exception as e:
            return self.generate_mock_options_data(instrument)

    def process_options_data(self, data: Dict, instrument: str) -> Dict[str, Any]:
        """Process options chain data"""
        try:
            records = data['records']['data']
            spot_price = data['records']['underlyingValue']
            
            # Calculate PCR
            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            # Find max pain
            strikes = list(set(item['strikePrice'] for item in records))
            pain_values = []
            
            for strike in strikes:
                pain = 0
                for item in records:
                    if 'CE' in item and item['strikePrice'] == strike:
                        if strike < spot_price:
                            pain += (spot_price - strike) * item['CE']['openInterest']
                    if 'PE' in item and item['strikePrice'] == strike:
                        if strike > spot_price:
                            pain += (strike - spot_price) * item['PE']['openInterest']
                pain_values.append((strike, pain))
            
            max_pain_strike = min(pain_values, key=lambda x: x[1])[0] if pain_values else spot_price
            
            # Determine bias based on PCR and max pain
            if pcr_oi > 1.2 and spot_price > max_pain_strike:
                bias = "BULLISH"
                score = 75
            elif pcr_oi < 0.8 and spot_price < max_pain_strike:
                bias = "BEARISH"
                score = -75
            elif pcr_oi > 1.0:
                bias = "MILD BULLISH"
                score = 25
            elif pcr_oi < 1.0:
                bias = "MILD BEARISH"
                score = -25
            else:
                bias = "NEUTRAL"
                score = 0
            
            return {
                'success': True,
                'instrument': instrument,
                'spot_price': spot_price,
                'pcr_oi': pcr_oi,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'max_pain': max_pain_strike,
                'bias': bias,
                'score': score,
                'timestamp': datetime.now(self.ist)
            }
            
        except Exception as e:
            return self.generate_mock_options_data(instrument)

    def generate_mock_options_data(self, instrument: str) -> Dict[str, Any]:
        """Generate mock options data"""
        return {
            'success': False,
            'instrument': instrument,
            'spot_price': 22150.0 if instrument == 'NIFTY' else 48500.0,
            'pcr_oi': 1.15,
            'total_ce_oi': 15000000,
            'total_pe_oi': 17250000,
            'max_pain': 22100.0 if instrument == 'NIFTY' else 48400.0,
            'bias': "BULLISH",
            'score': 75,
            'timestamp': datetime.now(self.ist),
            'note': 'Using demonstration data'
        }

    def get_all_options_data(self) -> List[Dict[str, Any]]:
        """Get options data for all instruments"""
        results = []
        for instrument in self.instruments:
            results.append(self.fetch_options_data(instrument))
        return results

# =============================================
# VOLUME ANALYSIS
# =============================================

class VolumeAnalyzer:
    """Volume analysis for price action"""
    
    def __init__(self):
        self.volume_history = deque(maxlen=20)

    def analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if df.empty or len(df) < 10:
            return {'success': False, 'message': 'Insufficient data'}
        
        try:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].iloc[-10:-1].mean()
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume spike detection
            if volume_ratio > 2.0:
                volume_status = "HIGH SPIKE"
                score = 80
            elif volume_ratio > 1.5:
                volume_status = "ELEVATED"
                score = 40
            elif volume_ratio < 0.5:
                volume_status = "LOW"
                score = -40
            else:
                volume_status = "NORMAL"
                score = 0
            
            # Price-Volume correlation
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            if price_change > 0.1 and volume_ratio > 1.2:
                pv_bias = "BULLISH"
                pv_score = 60
            elif price_change < -0.1 and volume_ratio > 1.2:
                pv_bias = "BEARISH"
                pv_score = -60
            else:
                pv_bias = "NEUTRAL"
                pv_score = 0
            
            return {
                'success': True,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_status': volume_status,
                'price_change': price_change,
                'pv_bias': pv_bias,
                'score': score + pv_score
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# =============================================
# MAIN DASHBOARD APPLICATION
# =============================================

class NiftyTradingDashboard:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.setup_session_state()
        
        # Initialize analyzers
        self.market_data = EnhancedMarketData()
        self.bias_analyzer = SimpleBiasAnalysis()
        self.options_analyzer = OptionsDataAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()

    def setup_session_state(self):
        """Initialize session state variables"""
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'bias_data' not in st.session_state:
            st.session_state.bias_data = None
        if 'options_data' not in st.session_state:
            st.session_state.options_data = None
        if 'nifty_data' not in st.session_state:
            st.session_state.nifty_data = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False

    def fetch_nifty_data(self) -> pd.DataFrame:
        """Fetch Nifty price data"""
        try:
            symbols = ['^NSEI', 'NSEI.NS', '^NSEI.NS']
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period='5d', interval='15m')
                    if not df.empty:
                        return df
                except:
                    continue
            
            # Generate mock data if real data fails
            return self.generate_mock_nifty_data()
            
        except Exception as e:
            return self.generate_mock_nifty_data()

    def generate_mock_nifty_data(self) -> pd.DataFrame:
        """Generate mock Nifty data for demonstration"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                             end=datetime.now(), freq='15min')
        dates = [d for d in dates if d.time() >= pd.Timestamp('09:15').time() 
                and d.time() <= pd.Timestamp('15:30').time()]
        
        np.random.seed(42)
        prices = [22000]
        for i in range(1, len(dates)):
            change = np.random.normal(0, 10)
            new_price = prices[-1] + change
            prices.append(max(21500, min(22500, new_price)))
        
        df = pd.DataFrame({
            'Open': [p - np.random.uniform(5, 15) for p in prices],
            'High': [p + np.random.uniform(5, 20) for p in prices],
            'Low': [p - np.random.uniform(5, 20) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(1000000, 5000000) for _ in prices]
        }, index=dates)
        
        return df

    def create_price_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive price chart"""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Nifty 50 Price Action', 'Volume'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if len(df) > 20:
            df['MA20'] = df['Close'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA20'],
                    name='MA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#00ff88' if close >= open_ else '#ff4444' 
                 for close, open_ in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        
        return fig

    def display_market_overview(self):
        """Display market overview section"""
        st.header("🌍 Market Overview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Comprehensive market analysis across multiple dimensions")
        with col2:
            if st.button("🔄 Update All", type="primary"):
                self.refresh_all_data()

        if st.session_state.last_update:
            st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')} IST")

        if st.session_state.market_data:
            data = st.session_state.market_data
            
            # Overall Summary
            st.subheader("📊 Market Summary")
            summary = data['summary']
            
            cols = st.columns(5)
            metrics = [
                ("Overall Sentiment", summary['overall_sentiment'], ""),
                ("Avg Score", f"{summary['avg_score']:.1f}", ""),
                ("Bullish", summary['bullish_count'], ""),
                ("Bearish", summary['bearish_count'], ""),
                ("Neutral", summary['neutral_count'], "")
            ]
            
            for (col, (label, value, delta)) in zip(cols, metrics):
                col.metric(label, value, delta)

            # Detailed sections in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🇮🇳 India VIX", "📈 Sectors", "🌍 Global", "⏰ Timing"
            ])
            
            with tab1:
                self.display_india_vix(data['india_vix'])
            
            with tab2:
                self.display_sectors(data['sector_indices'])
            
            with tab3:
                self.display_global_markets(data['global_markets'])
            
            with tab4:
                self.display_timing(data['intraday_seasonality'])
        else:
            st.info("Click 'Update All' to load market data")

    def display_india_vix(self, vix_data):
        """Display India VIX analysis"""
        st.subheader("🇮🇳 India VIX - Fear Index")
        
        if vix_data.get('success') or 'value' in vix_data:
            cols = st.columns(4)
            with cols[0]:
                st.metric("VIX Value", f"{vix_data['value']:.2f}")
            with cols[1]:
                st.metric("Sentiment", vix_data['sentiment'])
            with cols[2]:
                st.metric("Bias", vix_data['bias'])
            with cols[3]:
                st.metric("Score", vix_data['score'])
            
            # Interpretation
            vix_value = vix_data['value']
            if vix_value > 25:
                st.error("**🚨 High Fear Zone** - Extreme volatility expected. Market likely near bottom.")
            elif vix_value > 20:
                st.warning("**⚠️ Elevated Fear** - Increased volatility. Caution advised.")
            elif vix_value > 15:
                st.info("**✅ Normal Zone** - Typical market conditions.")
            elif vix_value > 12:
                st.success("**😊 Low Volatility** - Complacency setting in. Good for trend following.")
            else:
                st.success("**😴 Very Low VIX** - Extreme complacency. Potential for volatility spike.")
                
        else:
            st.info("India VIX data not available")

    def display_sectors(self, sectors):
        """Display sector performance"""
        st.subheader("📈 Nifty Sector Performance")
        
        if sectors:
            valid_sectors = [s for s in sectors if s.get('last_price', 0) > 0]
            
            if valid_sectors:
                # Top performers
                st.write("**Top Performers:**")
                top_sectors = sorted(valid_sectors, key=lambda x: x['change_pct'], reverse=True)[:4]
                
                cols = st.columns(4)
                for idx, sector in enumerate(top_sectors):
                    with cols[idx]:
                        color = "🟢" if sector['change_pct'] > 0 else "🔴"
                        st.metric(
                            f"{color} {sector['sector']}",
                            f"₹{sector['last_price']:.0f}",
                            f"{sector['change_pct']:+.2f}%"
                        )
                
                # Detailed table
                st.write("**Detailed Analysis:**")
                sector_df = pd.DataFrame(valid_sectors)
                display_df = sector_df[['sector', 'last_price', 'change_pct', 'bias', 'score']]
                st.dataframe(display_df, use_container_width=True)
                
                # Sector strength
                bullish_sectors = len([s for s in valid_sectors if 'BULLISH' in s['bias']])
                total_sectors = len(valid_sectors)
                sector_strength = (bullish_sectors / total_sectors) * 100
                
                st.metric("Sector Strength", f"{sector_strength:.1f}% Bullish")
                
            else:
                st.info("No sector data available")
        else:
            st.info("Sector data not available")

    def display_global_markets(self, markets):
        """Display global markets"""
        st.subheader("🌍 Global Markets")
        
        if markets:
            valid_markets = [m for m in markets if m.get('last_price', 0) > 0]
            
            if valid_markets:
                cols = st.columns(4)
                for idx, market in enumerate(valid_markets[:4]):
                    with cols[idx]:
                        color = "🟢" if market['change_pct'] > 0 else "🔴"
                        st.metric(
                            f"{color} {market['market']}",
                            f"{market['last_price']:.0f}",
                            f"{market['change_pct']:+.2f}%"
                        )
                
                # Global sentiment
                bullish_global = len([m for m in valid_markets if 'BULLISH' in m['bias']])
                total_global = len(valid_markets)
                global_sentiment = (bullish_global / total_global) * 100
                
                if global_sentiment > 60:
                    st.success(f"**Positive Global Sentiment**: {global_sentiment:.1f}% markets bullish")
                elif global_sentiment < 40:
                    st.error(f"**Negative Global Sentiment**: {global_sentiment:.1f}% markets bullish")
                else:
                    st.info(f"**Mixed Global Sentiment**: {global_sentiment:.1f}% markets bullish")
                    
            else:
                st.info("Global market data not available")
        else:
            st.info("Global data not available")

    def display_timing(self, timing_data):
        """Display intraday timing analysis"""
        st.subheader("⏰ Intraday Market Timing")
        
        if timing_data.get('success'):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Current Session", timing_data['session'])
            with cols[1]:
                st.metric("Session Bias", timing_data['session_bias'])
            with cols[2]:
                st.metric("Day of Week", timing_data['weekday'])
            
            st.info(f"**📋 Session Characteristics**: {timing_data['session_characteristics']}")
            st.success(f"**💡 Trading Recommendation**: {timing_data['trading_recommendation']}")
            
            # Additional timing insights
            st.write("**📅 Day-wise Patterns:**")
            st.write(f"- **{timing_data['day_bias']}**: {timing_data['day_characteristics']}")
            
        else:
            st.info("Timing analysis not available")

    def display_technical_analysis(self):
        """Display technical analysis section"""
        st.header("🎯 Technical Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Multi-timeframe technical indicators and bias analysis")
        with col2:
            if st.button("🔄 Update Technicals", type="primary"):
                st.session_state.bias_data = self.bias_analyzer.calculate_simple_bias()
                st.session_state.nifty_data = self.fetch_nifty_data()

        if st.session_state.bias_data and st.session_state.nifty_data:
            bias_data = st.session_state.bias_data
            nifty_df = st.session_state.nifty_data
            
            # Overall bias summary
            st.subheader("📊 Technical Bias Summary")
            
            cols = st.columns(4)
            with cols[0]:
                bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
                st.metric("Overall Bias", f"{bias_color} {bias_data['overall_bias']}")
            with cols[1]:
                st.metric("Bias Score", f"{bias_data['overall_score']:.1f}")
            with cols[2]:
                st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
            with cols[3]:
                st.metric("Indicators", f"{bias_data['bullish_count']}/{bias_data['total_indicators']} Bullish")

            # Price chart
            st.subheader("📈 Price & Volume Analysis")
            chart = self.create_price_chart(nifty_df)
            st.plotly_chart(chart, use_container_width=True)

            # Technical indicators
            st.subheader("🔧 Technical Indicators")
            bias_df = pd.DataFrame(bias_data['bias_results'])
            
            # Style the dataframe
            def color_bias(val):
                if 'BULLISH' in val:
                    return 'color: green; font-weight: bold'
                elif 'BEARISH' in val:
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: orange; font-weight: bold'
            
            styled_df = bias_df.style.applymap(color_bias, subset=['bias'])
            st.dataframe(styled_df, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bias distribution pie chart
                fig_pie = px.pie(
                    values=[bias_data['bullish_count'], bias_data['bearish_count'], bias_data['neutral_count']],
                    names=['Bullish', 'Bearish', 'Neutral'],
                    title="Bias Distribution",
                    color=['Bullish', 'Bearish', 'Neutral'],
                    color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Neutral': 'orange'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Indicator scores bar chart
                fig_bar = px.bar(
                    bias_df,
                    x='indicator',
                    y='score',
                    color='bias',
                    title="Indicator Scores",
                    color_discrete_map={'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'orange'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Trading recommendations
            st.subheader("💡 Trading Insights")
            self.display_trading_insights(bias_data, nifty_df)
            
            if bias_data.get('note'):
                st.info(f"*Note: {bias_data['note']}*")
                
        else:
            st.info("Click 'Update Technicals' to load technical analysis")

    def display_trading_insights(self, bias_data, nifty_df):
        """Display trading insights based on analysis"""
        overall_bias = bias_data['overall_bias']
        bias_score = bias_data['overall_score']
        current_price = bias_data['current_price']
        
        # Calculate support and resistance levels
        if len(nifty_df) > 20:
            support = nifty_df['Low'].tail(20).min()
            resistance = nifty_df['High'].tail(20).max()
        else:
            support = current_price * 0.99
            resistance = current_price * 1.01
        
        if overall_bias == "BULLISH" and bias_score > 100:
            st.success("""
            **🎯 STRONG BULLISH SIGNAL - HIGH CONFIDENCE**
            
            **Recommended Action:**
            - Look for LONG opportunities on dips
            - Entry Zone: ₹{:.0f} - ₹{:.0f}
            - Target 1: ₹{:.0f} (Resistance)
            - Target 2: ₹{:.0f} (Extended)
            - Stop Loss: Below ₹{:.0f} (Support)
            
            **Strategy:** Trend following, buy on dips
            **Risk:** Low to Moderate
            """.format(support, current_price, resistance, resistance + (resistance - support) * 0.5, support))
            
        elif overall_bias == "BULLISH":
            st.info("""
            **📈 BULLISH BIAS - MODERATE CONFIDENCE**
            
            **Recommended Action:**
            - Consider LONG positions with caution
            - Wait for pullbacks to better entry levels
            - Target: ₹{:.0f} (Resistance)
            - Stop Loss: Below key support
            
            **Strategy:** Wait for confirmation, use smaller position size
            **Risk:** Moderate
            """.format(resistance))
            
        elif overall_bias == "BEARISH" and bias_score < -100:
            st.error("""
            **🎯 STRONG BEARISH SIGNAL - HIGH CONFIDENCE**
            
            **Recommended Action:**
            - Look for SHORT opportunities on rallies
            - Entry Zone: ₹{:.0f} - ₹{:.0f}
            - Target 1: ₹{:.0f} (Support)
            - Target 2: ₹{:.0f} (Extended)
            - Stop Loss: Above ₹{:.0f} (Resistance)
            
            **Strategy:** Trend following, sell on rallies
            **Risk:** Low to Moderate
            """.format(current_price, resistance, support, support - (resistance - support) * 0.5, resistance))
            
        elif overall_bias == "BEARISH":
            st.warning("""
            **📉 BEARISH BIAS - MODERATE CONFIDENCE**
            
            **Recommended Action:**
            - Consider SHORT positions with caution
            - Wait for rallies to better entry levels
            - Target: ₹{:.0f} (Support)
            - Stop Loss: Above key resistance
            
            **Strategy:** Wait for confirmation, use smaller position size
            **Risk:** Moderate
            """.format(support))
            
        else:
            st.warning("""
            **⚖️ NEUTRAL/UNCLEAR BIAS**
            
            **Recommended Action:**
            - Wait for clearer directional signals
            - Consider range-bound strategies
            - Key Range: ₹{:.0f} - ₹{:.0f}
            - Breakout above ₹{:.0f} for bullish bias
            - Breakdown below ₹{:.0f} for bearish bias
            
            **Strategy:** Wait for breakout, reduce position sizes
            **Risk:** High (due to uncertainty)
            """.format(support, resistance, resistance, support))

    def display_options_analysis(self):
        """Display options analysis section"""
        st.header("📊 Options Chain Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("NSE Options data analysis for market sentiment")
        with col2:
            if st.button("🔄 Update Options", type="primary"):
                st.session_state.options_data = self.options_analyzer.get_all_options_data()

        if st.session_state.options_data:
            options_data = st.session_state.options_data
            
            st.subheader("🎯 Options Market Sentiment")
            
            # Display each instrument
            for option_data in options_data:
                with st.expander(f"📈 {option_data['instrument']} Analysis", expanded=True):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Spot Price", f"₹{option_data['spot_price']:.2f}")
                    with cols[1]:
                        st.metric("PCR OI", f"{option_data['pcr_oi']:.2f}")
                    with cols[2]:
                        st.metric("Max Pain", f"₹{option_data['max_pain']:.0f}")
                    with cols[3]:
                        bias_color = "🟢" if "BULLISH" in option_data['bias'] else "🔴" if "BEARISH" in option_data['bias'] else "🟡"
                        st.metric("Bias", f"{bias_color} {option_data['bias']}")
                    
                    # PCR Interpretation
                    pcr = option_data['pcr_oi']
                    if pcr > 1.5:
                        pcr_sentiment = "EXTREME BULLISH"
                        pcr_color = "green"
                    elif pcr > 1.2:
                        pcr_sentiment = "BULLISH"
                        pcr_color = "lightgreen"
                    elif pcr > 0.9:
                        pcr_sentiment = "NEUTRAL"
                        pcr_color = "orange"
                    elif pcr > 0.7:
                        pcr_sentiment = "BEARISH"
                        pcr_color = "lightcoral"
                    else:
                        pcr_sentiment = "EXTREME BEARISH"
                        pcr_color = "red"
                    
                    st.write(f"**PCR Sentiment**: :{pcr_color}[{pcr_sentiment}]")
                    
                    # Max Pain Analysis
                    spot = option_data['spot_price']
                    max_pain = option_data['max_pain']
                    pain_distance = ((spot - max_pain) / spot) * 100
                    
                    if pain_distance > 0.5:
                        pain_sentiment = "BULLISH (Above Max Pain)"
                    elif pain_distance < -0.5:
                        pain_sentiment = "BEARISH (Below Max Pain)"
                    else:
                        pain_sentiment = "NEUTRAL (Near Max Pain)"
                    
                    st.write(f"**Max Pain Analysis**: {pain_sentiment} (Distance: {pain_distance:+.2f}%)")
                    
                    # OI Analysis
                    ce_oi = option_data['total_ce_oi']
                    pe_oi = option_data['total_pe_oi']
                    st.write(f"**Open Interest**: Calls: {ce_oi:,} | Puts: {pe_oi:,}")
                    
                    if option_data.get('note'):
                        st.info(f"*Note: {option_data['note']}*")
        else:
            st.info("Click 'Update Options' to load options chain analysis")

    def display_volume_analysis(self):
        """Display volume analysis"""
        st.header("🔍 Volume Analysis")
        
        if st.session_state.nifty_data is not None:
            nifty_df = st.session_state.nifty_data
            volume_analysis = self.volume_analyzer.analyze_volume(nifty_df)
            
            if volume_analysis['success']:
                st.subheader("📊 Volume Insights")
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Current Volume", f"{volume_analysis['current_volume']:,.0f}")
                with cols[1]:
                    st.metric("Avg Volume", f"{volume_analysis['avg_volume']:,.0f}")
                with cols[2]:
                    st.metric("Volume Ratio", f"{volume_analysis['volume_ratio']:.2f}x")
                with cols[3]:
                    st.metric("Volume Status", volume_analysis['volume_status'])
                
                # Volume interpretation
                volume_ratio = volume_analysis['volume_ratio']
                price_change = volume_analysis['price_change']
                pv_bias = volume_analysis['pv_bias']
                
                st.write(f"**Price-Volume Correlation**: {pv_bias} (Price Change: {price_change:+.2f}%)")
                
                if volume_ratio > 2.0:
                    st.warning("""
                    **🚨 HIGH VOLUME SPIKE DETECTED**
                    - Significant institutional activity
                    - Potential breakout/breakdown confirmation
                    - Monitor for follow-through
                    """)
                elif volume_ratio > 1.5:
                    st.info("""
                    **📈 ELEVATED VOLUME**
                    - Above average participation
                    - Validates price movement
                    - Good trending conditions
                    """)
                elif volume_ratio < 0.5:
                    st.info("""
                    **📉 LOW VOLUME**
                    - Light participation
                    - Caution: weak moves may not sustain
                    - Wait for volume confirmation
                    """)
                else:
                    st.success("""
                    **✅ NORMAL VOLUME**
                    - Typical market activity
                    - Healthy price discovery
                    - Continue with planned strategy
                    """)
                    
            else:
                st.info("Volume analysis not available")
        else:
            st.info("Load price data first to analyze volume")

    def refresh_all_data(self):
        """Refresh all data sources"""
        with st.spinner("Updating all market data..."):
            st.session_state.market_data = self.market_data.fetch_all_enhanced_data()
            st.session_state.bias_data = self.bias_analyzer.calculate_simple_bias()
            st.session_state.options_data = self.options_analyzer.get_all_options_data()
            st.session_state.nifty_data = self.fetch_nifty_data()
            st.session_state.last_update = datetime.now(self.ist)

    def run(self):
        """Main application runner"""
        st.title("📈 Nifty Trading Dashboard")
        st.markdown("*Comprehensive Market Analysis & Trading Insights*")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Dashboard Controls")
            
            st.subheader("Data Settings")
            st.session_state.auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
            
            if st.button("🔄 Refresh All Data", type="primary", use_container_width=True):
                self.refresh_all_data()
                st.success("All data refreshed!")
            
            st.divider()
            
            st.subheader("📊 Features")
            st.write("• **Market Overview** - Multi-source analysis")
            st.write("• **Technical Analysis** - Indicators & bias")
            st.write("• **Options Analysis** - PCR & Max Pain")
            st.write("• **Volume Analysis** - Price-volume insights")
            st.write("• **Trading Insights** - Actionable signals")
            
            st.divider()
            
            st.subheader("ℹ️ Info")
            st.write("Data sources: Yahoo Finance, NSE India")
            st.write("Updates: Real-time where available")
            st.write("Note: Some data may be delayed")
            
            if st.session_state.last_update:
                st.write(f"Last refresh: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌍 Market Overview", 
            "🎯 Technical Analysis", 
            "📊 Options Analysis",
            "🔍 Volume Analysis"
        ])
        
        with tab1:
            self.display_market_overview()
            
        with tab2:
            self.display_technical_analysis()
            
        with tab3:
            self.display_options_analysis()
            
        with tab4:
            self.display_volume_analysis()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(30)
            st.rerun()

# Run the application
if __name__ == "__main__":
    app = NiftyTradingDashboard()
    app.run()
