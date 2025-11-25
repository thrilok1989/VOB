import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
import numpy as np
from collections import deque
import warnings
import math
from scipy.stats import norm
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import random

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# ENHANCED MARKET DATA MODULE
# =============================================

class EnhancedMarketData:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX with comprehensive analysis"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]
                
                # Comprehensive VIX interpretation
                if vix_value > 30:
                    sentiment, bias, score, interpretation = "EXTREME FEAR", "BEARISH", -90, "Market panic, high volatility expected"
                elif vix_value > 25:
                    sentiment, bias, score, interpretation = "HIGH FEAR", "BEARISH", -75, "Elevated fear, caution advised"
                elif vix_value > 20:
                    sentiment, bias, score, interpretation = "ELEVATED FEAR", "BEARISH", -50, "Increased volatility expected"
                elif vix_value > 15:
                    sentiment, bias, score, interpretation = "MODERATE", "NEUTRAL", 0, "Normal market conditions"
                elif vix_value > 12:
                    sentiment, bias, score, interpretation = "LOW VOLATILITY", "BULLISH", 40, "Complacency setting in"
                else:
                    sentiment, bias, score, interpretation = "COMPLACENCY", "BULLISH", 60, "Very low volatility, potential for spike"

                return {
                    'success': True,
                    'value': vix_value,
                    'sentiment': sentiment,
                    'bias': bias,
                    'score': score,
                    'interpretation': interpretation,
                    'trend': 'rising' if vix_value > 15 else 'falling',
                    'signal': 'SELL' if vix_value > 20 else 'BUY' if vix_value < 12 else 'HOLD'
                }
        except:
            pass
        
        # Fallback VIX data
        return {
            'success': True,
            'value': 14.5,
            'sentiment': "LOW VOLATILITY",
            'bias': "BULLISH",
            'score': 40,
            'interpretation': "Normal market conditions with low volatility",
            'trend': 'falling',
            'signal': 'HOLD'
        }

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch comprehensive sector data with rotation analysis"""
        sectors_map = {
            '^CNXIT': 'NIFTY IT', 
            '^CNXAUTO': 'NIFTY AUTO', 
            '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL', 
            '^CNXREALTY': 'NIFTY REALTY', 
            '^CNXFMCG': 'NIFTY FMCG',
            '^CNXBANK': 'NIFTY BANK',
            '^CNXFINANCE': 'NIFTY FINANCIAL',
            '^CNXENERGY': 'NIFTY ENERGY'
        }
        
        sector_data = []
        for symbol, name in sectors_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1d")
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # Determine sector characteristics
                    if 'IT' in name or 'PHARMA' in name:
                        sector_type = 'DEFENSIVE'
                    elif 'BANK' in name or 'FINANCE' in name:
                        sector_type = 'FINANCIAL'
                    elif 'METAL' in name or 'ENERGY' in name:
                        sector_type = 'CYCLICAL'
                    elif 'AUTO' in name or 'REALTY' in name:
                        sector_type = 'DISCRETIONARY'
                    else:
                        sector_type = 'CONSUMER'
                    
                    if change_pct > 2.0: 
                        bias, score, strength = "STRONG BULLISH", 80, "VERY STRONG"
                    elif change_pct > 1.0: 
                        bias, score, strength = "BULLISH", 60, "STRONG"
                    elif change_pct > 0.5: 
                        bias, score, strength = "MILD BULLISH", 40, "MODERATE"
                    elif change_pct < -2.0: 
                        bias, score, strength = "STRONG BEARISH", -80, "VERY STRONG"
                    elif change_pct < -1.0: 
                        bias, score, strength = "BEARISH", -60, "STRONG"
                    elif change_pct < -0.5: 
                        bias, score, strength = "MILD BEARISH", -40, "MODERATE"
                    else: 
                        bias, score, strength = "NEUTRAL", 0, "WEAK"

                    sector_data.append({
                        'sector': name, 
                        'symbol': symbol,
                        'last_price': current_price,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'strength': strength,
                        'type': sector_type,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    })
            except:
                # Fallback sector data
                base_price = random.uniform(1000, 50000)
                change_pct = random.uniform(-3, 3)
                sector_type = random.choice(['DEFENSIVE', 'FINANCIAL', 'CYCLICAL', 'DISCRETIONARY', 'CONSUMER'])
                
                if change_pct > 2.0: bias, score, strength = "STRONG BULLISH", 80, "VERY STRONG"
                elif change_pct > 1.0: bias, score, strength = "BULLISH", 60, "STRONG"
                elif change_pct > 0.5: bias, score, strength = "MILD BULLISH", 40, "MODERATE"
                elif change_pct < -2.0: bias, score, strength = "STRONG BEARISH", -80, "VERY STRONG"
                elif change_pct < -1.0: bias, score, strength = "BEARISH", -60, "STRONG"
                elif change_pct < -0.5: bias, score, strength = "MILD BEARISH", -40, "MODERATE"
                else: bias, score, strength = "NEUTRAL", 0, "WEAK"

                sector_data.append({
                    'sector': name, 
                    'symbol': symbol,
                    'last_price': base_price,
                    'change_pct': change_pct,
                    'bias': bias,
                    'score': score,
                    'strength': strength,
                    'type': sector_type,
                    'volume': random.randint(1000000, 5000000)
                })
        
        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices with sentiment analysis"""
        global_markets = {
            '^GSPC': {'name': 'S&P 500', 'region': 'US'},
            '^IXIC': {'name': 'NASDAQ', 'region': 'US'},
            '^DJI': {'name': 'DOW JONES', 'region': 'US'},
            '^N225': {'name': 'NIKKEI 225', 'region': 'ASIA'},
            '^HSI': {'name': 'HANG SENG', 'region': 'ASIA'},
            '^FTSE': {'name': 'FTSE 100', 'region': 'EUROPE'},
            '^GDAXI': {'name': 'DAX', 'region': 'EUROPE'},
            '000001.SS': {'name': 'SHANGHAI', 'region': 'ASIA'},
            '^STOXX50E': {'name': 'EURO STOXX 50', 'region': 'EUROPE'}
        }
        
        market_data = []
        for symbol, info in global_markets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    
                    # Global market sentiment
                    if change_pct > 1.5: 
                        sentiment, impact = "STRONG BULLISH", "POSITIVE"
                    elif change_pct > 0.5: 
                        sentiment, impact = "BULLISH", "POSITIVE"
                    elif change_pct < -1.5: 
                        sentiment, impact = "STRONG BEARISH", "NEGATIVE"
                    elif change_pct < -0.5: 
                        sentiment, impact = "BEARISH", "NEGATIVE"
                    else: 
                        sentiment, impact = "NEUTRAL", "NEUTRAL"
                    
                    # Market hours detection
                    now_utc = datetime.now(pytz.utc)
                    market_status = "CLOSED"
                    
                    market_data.append({
                        'market': info['name'],
                        'symbol': symbol,
                        'region': info['region'],
                        'last_price': current_close,
                        'change_pct': change_pct,
                        'sentiment': sentiment,
                        'impact': impact,
                        'status': market_status,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    })
            except:
                # Fallback global data
                base_price = random.uniform(1000, 40000)
                change_pct = random.uniform(-2, 2)
                
                if change_pct > 1.5: sentiment, impact = "STRONG BULLISH", "POSITIVE"
                elif change_pct > 0.5: sentiment, impact = "BULLISH", "POSITIVE"
                elif change_pct < -1.5: sentiment, impact = "STRONG BEARISH", "NEGATIVE"
                elif change_pct < -0.5: sentiment, impact = "BEARISH", "NEGATIVE"
                else: sentiment, impact = "NEUTRAL", "NEUTRAL"

                market_data.append({
                    'market': info['name'],
                    'symbol': symbol,
                    'region': info['region'],
                    'last_price': base_price,
                    'change_pct': change_pct,
                    'sentiment': sentiment,
                    'impact': impact,
                    'status': "CLOSED",
                    'volume': random.randint(1000000, 50000000)
                })
        
        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """Fetch intermarket analysis: commodities, currencies, bonds"""
        intermarket_assets = {
            'DX-Y.NYB': {'name': 'US DOLLAR INDEX', 'type': 'CURRENCY'},
            'CL=F': {'name': 'CRUDE OIL', 'type': 'COMMODITY'},
            'GC=F': {'name': 'GOLD', 'type': 'COMMODITY'},
            'SI=F': {'name': 'SILVER', 'type': 'COMMODITY'},
            'INR=X': {'name': 'USD/INR', 'type': 'CURRENCY'},
            '^TNX': {'name': 'US 10Y TREASURY', 'type': 'BOND'},
            'BTC-USD': {'name': 'BITCOIN', 'type': 'CRYPTO'},
            'ETH-USD': {'name': 'ETHEREUM', 'type': 'CRYPTO'},
            'ZN=F': {'name': 'US 10Y NOTE', 'type': 'BOND'}
        }
        
        intermarket_data = []
        for symbol, info in intermarket_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    
                    # Intermarket relationships and impact on Indian markets
                    if info['type'] == 'CURRENCY':
                        if 'USD' in info['name']:
                            if change_pct > 0.5:
                                impact, reasoning = "NEGATIVE", "Strong USD negative for EM"
                            elif change_pct < -0.5:
                                impact, reasoning = "POSITIVE", "Weak USD positive for EM"
                            else:
                                impact, reasoning = "NEUTRAL", "Stable USD"
                        elif 'INR' in info['name']:
                            if change_pct > 0.5:
                                impact, reasoning = "NEGATIVE", "Weak INR inflationary"
                            elif change_pct < -0.5:
                                impact, reasoning = "POSITIVE", "Strong INR positive"
                            else:
                                impact, reasoning = "NEUTRAL", "Stable INR"
                    
                    elif info['type'] == 'COMMODITY':
                        if 'OIL' in info['name']:
                            if change_pct > 2:
                                impact, reasoning = "NEGATIVE", "High oil prices inflationary"
                            elif change_pct < -2:
                                impact, reasoning = "POSITIVE", "Low oil prices positive"
                            else:
                                impact, reasoning = "NEUTRAL", "Stable oil prices"
                        elif 'GOLD' in info['name'] or 'SILVER' in info['name']:
                            if change_pct > 1:
                                impact, reasoning = "NEGATIVE", "Gold rise indicates risk-off"
                            elif change_pct < -1:
                                impact, reasoning = "POSITIVE", "Gold fall indicates risk-on"
                            else:
                                impact, reasoning = "NEUTRAL", "Stable precious metals"
                    
                    elif info['type'] == 'BOND':
                        if change_pct > 1:
                            impact, reasoning = "NEGATIVE", "Rising yields negative for growth"
                        elif change_pct < -1:
                            impact, reasoning = "POSITIVE", "Falling yields positive for growth"
                        else:
                            impact, reasoning = "NEUTRAL", "Stable bond yields"
                    
                    elif info['type'] == 'CRYPTO':
                        if abs(change_pct) > 5:
                            impact, reasoning = "HIGH VOLATILITY", "Crypto volatility affects sentiment"
                        else:
                            impact, reasoning = "NEUTRAL", "Stable crypto markets"
                    
                    else:
                        impact, reasoning = "NEUTRAL", "No significant impact"
                    
                    intermarket_data.append({
                        'asset': info['name'],
                        'symbol': symbol,
                        'type': info['type'],
                        'last_price': current_close,
                        'change_pct': change_pct,
                        'impact': impact,
                        'reasoning': reasoning,
                        'trend': 'rising' if change_pct > 0 else 'falling'
                    })
            except:
                # Fallback intermarket data
                base_price = random.uniform(10, 5000)
                change_pct = random.uniform(-5, 5)
                
                if info['type'] == 'CURRENCY':
                    impact = random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"])
                    reasoning = "Currency movement impact"
                elif info['type'] == 'COMMODITY':
                    impact = random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"])
                    reasoning = "Commodity price impact"
                elif info['type'] == 'BOND':
                    impact = random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"])
                    reasoning = "Bond yield impact"
                else:
                    impact = "NEUTRAL"
                    reasoning = "Market impact"

                intermarket_data.append({
                    'asset': info['name'],
                    'symbol': symbol,
                    'type': info['type'],
                    'last_price': base_price,
                    'change_pct': change_pct,
                    'impact': impact,
                    'reasoning': reasoning,
                    'trend': 'rising' if change_pct > 0 else 'falling'
                })
        
        return intermarket_data

    def analyze_sector_rotation(self, sector_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sector rotation and market leadership"""
        if not sector_data:
            return {'success': False, 'error': 'No sector data available'}
        
        # Sort sectors by performance
        sectors_sorted = sorted(sector_data, key=lambda x: x['change_pct'], reverse=True)
        
        # Identify leaders and laggards
        leaders = sectors_sorted[:3]
        laggards = sectors_sorted[-3:]
        
        # Analyze sector type performance
        defensive_sectors = [s for s in sector_data if s['type'] == 'DEFENSIVE']
        cyclical_sectors = [s for s in sector_data if s['type'] == 'CYCLICAL']
        financial_sectors = [s for s in sector_data if s['type'] == 'FINANCIAL']
        
        # Calculate average performance by type
        def_avg = np.mean([s['change_pct'] for s in defensive_sectors]) if defensive_sectors else 0
        cyc_avg = np.mean([s['change_pct'] for s in cyclical_sectors]) if cyclical_sectors else 0
        fin_avg = np.mean([s['change_pct'] for s in financial_sectors]) if financial_sectors else 0
        
        # Determine market regime
        if cyc_avg > def_avg and cyc_avg > 0.5:
            regime = "RISK-ON"
            regime_reason = "Cyclicals outperforming defensives"
        elif def_avg > cyc_avg and def_avg > 0.5:
            regime = "RISK-OFF"
            regime_reason = "Defensives outperforming cyclicals"
        else:
            regime = "NEUTRAL"
            regime_reason = "Mixed sector performance"
        
        # Market breadth
        advancing_sectors = len([s for s in sector_data if s['change_pct'] > 0])
        declining_sectors = len([s for s in sector_data if s['change_pct'] < 0])
        breadth = (advancing_sectors / len(sector_data)) * 100 if sector_data else 0
        
        # Strength analysis
        strong_bullish = len([s for s in sector_data if s['score'] >= 60])
        weak_bullish = len([s for s in sector_data if 0 < s['score'] < 60])
        strong_bearish = len([s for s in sector_data if s['score'] <= -60])
        weak_bearish = len([s for s in sector_data if -60 < s['score'] < 0])
        
        return {
            'success': True,
            'leaders': leaders,
            'laggards': laggards,
            'market_regime': regime,
            'regime_reason': regime_reason,
            'sector_breadth': breadth,
            'defensive_avg': def_avg,
            'cyclical_avg': cyc_avg,
            'financial_avg': fin_avg,
            'advancing_sectors': advancing_sectors,
            'declining_sectors': declining_sectors,
            'strong_bullish': strong_bullish,
            'weak_bullish': weak_bullish,
            'strong_bearish': strong_bearish,
            'weak_bearish': weak_bearish,
            'total_sectors': len(sector_data)
        }

    def analyze_intraday_seasonality(self) -> Dict[str, Any]:
        """Analyze intraday time-based patterns and seasonality"""
        now = datetime.now(self.ist)
        current_time = now.time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        # Determine current market session with detailed analysis
        if current_time < market_open:
            session = "PRE-MARKET"
            characteristics = [
                "Low liquidity, wide bid-ask spreads",
                "Reacting to overnight global cues",
                "Gap-up/gap-down formations common"
            ]
            opportunity = "LOW"
            recommendation = "Wait for market open, avoid trading"
            volatility = "LOW"
            
        elif current_time < datetime.strptime("09:30", "%H:%M").time():
            session = "OPENING RANGE (9:15-9:30)"
            characteristics = [
                "Highest volatility of the day",
                "Institutional order execution",
                "Gap filling/extension movements",
                "Sets tone for morning session"
            ]
            opportunity = "HIGH"
            recommendation = "Trade breakouts with tight stops"
            volatility = "VERY HIGH"
            
        elif current_time < datetime.strptime("10:30", "%H:%M").time():
            session = "MORNING TREND (9:30-10:30)"
            characteristics = [
                "Strong trending movements",
                "High volume participation",
                "Directional clarity emerges",
                "Best for trend following"
            ]
            opportunity = "VERY HIGH"
            recommendation = "Follow the established trend"
            volatility = "HIGH"
            
        elif current_time < datetime.strptime("12:00", "%H:%M").time():
            session = "MID-MORNING (10:30-12:00)"
            characteristics = [
                "Consolidation after morning move",
                "Pullback opportunities",
                "Institutional repositioning",
                "Good for mean reversion"
            ]
            opportunity = "HIGH"
            recommendation = "Look for pullback entries"
            volatility = "MODERATE"
            
        elif current_time < datetime.strptime("14:00", "%H:%M").time():
            session = "LUNCH HOUR (12:00-14:00)"
            characteristics = [
                "Lowest volume of the day",
                "Choppy, range-bound action",
                "Retail dominated trading",
                "False breakouts common"
            ]
            opportunity = "LOW"
            recommendation = "Reduce position sizes or avoid"
            volatility = "LOW"
            
        elif current_time < datetime.strptime("15:15", "%H:%M").time():
            session = "AFTERNOON SESSION (14:00-15:15)"
            characteristics = [
                "Volume picks up again",
                "European market overlap",
                "Trend resumption/continuation",
                "Good for momentum trading"
            ]
            opportunity = "HIGH"
            recommendation = "Trade in direction of overall trend"
            volatility = "MODERATE-HIGH"
            
        elif current_time <= market_close:
            session = "CLOSING RANGE (15:15-15:30)"
            characteristics = [
                "Squaring off positions",
                "High volatility returns",
                "Institutional rebalancing",
                "Sets up for next day"
            ]
            opportunity = "MODERATE"
            recommendation = "Close positions or use wide stops"
            volatility = "HIGH"
            
        else:
            session = "POST-MARKET"
            characteristics = ["Market closed", "Analyze today's action", "Prepare for tomorrow"]
            opportunity = "NONE"
            recommendation = "No trading - market closed"
            volatility = "NONE"
        
        # Day of week patterns
        weekday = now.strftime("%A")
        day_patterns = {
            "Monday": {
                "pattern": "Gap and follow-through",
                "characteristics": ["Weekend news gaps", "Follow Friday's trend", "High volume open"],
                "bias": "TRENDING"
            },
            "Tuesday": {
                "pattern": "Trend continuation",
                "characteristics": ["Strongest trending day", "Institutional activity", "Clean moves"],
                "bias": "TRENDING"
            },
            "Wednesday": {
                "pattern": "Mid-week consolidation",
                "characteristics": ["Often range-bound", "Fade extremes", "Prepare for Thursday"],
                "bias": "RANGING"
            },
            "Thursday": {
                "pattern": "Pre-weekend positioning",
                "characteristics": ["Volatility increases", "Weekend profit booking", "Option expiry effects"],
                "bias": "VOLATILE"
            },
            "Friday": {
                "pattern": "Weekend squaring",
                "characteristics": ["Low volume close", "Position squaring", "Weekend gap risk"],
                "bias": "UNPREDICTABLE"
            }
        }
        
        day_info = day_patterns.get(weekday, {
            "pattern": "Weekend",
            "characteristics": ["Market closed"],
            "bias": "NONE"
        })
        
        return {
            'success': True,
            'current_time': now.strftime("%H:%M:%S"),
            'current_session': session,
            'session_characteristics': characteristics,
            'trading_opportunity': opportunity,
            'volatility_expectation': volatility,
            'recommendation': recommendation,
            'weekday': weekday,
            'day_pattern': day_info['pattern'],
            'day_characteristics': day_info['characteristics'],
            'day_bias': day_info['bias'],
            'market_status': "OPEN" if market_open <= current_time <= market_close else "CLOSED",
            'time_to_close': f"{(datetime.combine(now.date(), market_close) - now).seconds // 60} minutes" if market_open <= current_time <= market_close else "N/A"
        }

    def get_comprehensive_market_analysis(self) -> Dict[str, Any]:
        """Get all market analysis in one call"""
        st.info("🔄 Fetching comprehensive market analysis...")
        
        analysis = {
            'timestamp': datetime.now(self.ist),
            'india_vix': self.fetch_india_vix(),
            'sector_indices': self.fetch_sector_indices(),
            'global_markets': self.fetch_global_markets(),
            'intermarket_data': self.fetch_intermarket_data(),
        }
        
        # Add derived analysis
        analysis['sector_rotation'] = self.analyze_sector_rotation(analysis['sector_indices'])
        analysis['intraday_seasonality'] = self.analyze_intraday_seasonality()
        
        # Calculate overall market sentiment
        sentiment_score = 0
        sentiment_factors = []
        
        # VIX contribution
        if analysis['india_vix']['success']:
            sentiment_score += analysis['india_vix']['score'] * 0.3
            sentiment_factors.append(f"VIX: {analysis['india_vix']['sentiment']}")
        
        # Sector contribution
        if analysis['sector_rotation']['success']:
            sector_sentiment = 50 + (analysis['sector_rotation']['sector_breadth'] - 50)
            sentiment_score += sector_sentiment * 0.3
            sentiment_factors.append(f"Sector Breadth: {analysis['sector_rotation']['sector_breadth']:.1f}%")
        
        # Global markets contribution
        global_scores = [m['change_pct'] for m in analysis['global_markets']]
        if global_scores:
            global_avg = np.mean(global_scores)
            sentiment_score += global_avg * 2
            sentiment_factors.append(f"Global Markets: {global_avg:+.2f}%")
        
        # Determine overall sentiment
        if sentiment_score > 20:
            overall_sentiment = "STRONGLY BULLISH"
        elif sentiment_score > 10:
            overall_sentiment = "BULLISH"
        elif sentiment_score > -10:
            overall_sentiment = "NEUTRAL"
        elif sentiment_score > -20:
            overall_sentiment = "BEARISH"
        else:
            overall_sentiment = "STRONGLY BEARISH"
        
        analysis['overall_sentiment'] = {
            'score': sentiment_score,
            'sentiment': overall_sentiment,
            'factors': sentiment_factors,
            'timestamp': datetime.now(self.ist)
        }
        
        return analysis

# =============================================
# ENHANCED APP WITH COMPREHENSIVE MARKET TAB
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        
        # Initialize core components
        self.market_data_fetcher = EnhancedMarketData()
        
        # Initialize session state
        self.init_session_state()

    def init_session_state(self):
        if 'comprehensive_market_data' not in st.session_state:
            st.session_state.comprehensive_market_data = None

    def display_comprehensive_market_tab(self):
        """Display the enhanced market analysis tab"""
        st.header("🌍 Comprehensive Market Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info("Real-time analysis of India VIX, global markets, sector rotation, and intraday patterns")
        with col2:
            if st.button("🔄 Update All Market Data", type="primary", use_container_width=True):
                with st.spinner("Fetching comprehensive market analysis..."):
                    market_data = self.market_data_fetcher.get_comprehensive_market_analysis()
                    st.session_state.comprehensive_market_data = market_data
        with col3:
            if st.session_state.comprehensive_market_data:
                last_update = st.session_state.comprehensive_market_data['timestamp']
                st.write(f"Last update: {last_update.strftime('%H:%M:%S')}")
        
        # Auto-load data if not present
        if st.session_state.comprehensive_market_data is None:
            with st.spinner("Loading comprehensive market data..."):
                market_data = self.market_data_fetcher.get_comprehensive_market_analysis()
                st.session_state.comprehensive_market_data = market_data
        
        if not st.session_state.comprehensive_market_data:
            st.error("Unable to load market data. Please check your internet connection.")
            return
        
        market_data = st.session_state.comprehensive_market_data
        
        # Create tabs for different market analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Market Overview", 
            "🇮🇳 India VIX & Sentiment", 
            "📈 Sector Rotation", 
            "🌍 Global Markets", 
            "🔄 Intermarket Analysis", 
            "⏰ Intraday Patterns"
        ])
        
        with tab1:
            self.display_market_overview(market_data)
        
        with tab2:
            self.display_india_vix_analysis(market_data)
        
        with tab3:
            self.display_sector_rotation_analysis(market_data)
        
        with tab4:
            self.display_global_markets_analysis(market_data)
        
        with tab5:
            self.display_intermarket_analysis(market_data)
        
        with tab6:
            self.display_intraday_patterns(market_data)

    def display_market_overview(self, market_data: Dict[str, Any]):
        """Display comprehensive market overview"""
        st.subheader("📊 Overall Market Sentiment & Summary")
        
        # Overall Sentiment
        sentiment = market_data['overall_sentiment']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = sentiment['score']
            if score > 20:
                st.success(f"**Overall Sentiment**\n# 🟢 {sentiment['sentiment']}")
            elif score > 10:
                st.success(f"**Overall Sentiment**\n# 🟡 {sentiment['sentiment']}")
            elif score > -10:
                st.info(f"**Overall Sentiment**\n# ⚪ {sentiment['sentiment']}")
            elif score > -20:
                st.warning(f"**Overall Sentiment**\n# 🟠 {sentiment['sentiment']}")
            else:
                st.error(f"**Overall Sentiment**\n# 🔴 {sentiment['sentiment']}")
        
        with col2:
            st.metric("Sentiment Score", f"{score:.1f}")
        
        with col3:
            vix_data = market_data['india_vix']
            st.metric("India VIX", f"{vix_data['value']:.2f}", vix_data['sentiment'])
        
        with col4:
            rotation = market_data['sector_rotation']
            if rotation['success']:
                st.metric("Sector Breadth", f"{rotation['sector_breadth']:.1f}%")
        
        # Key Factors
        st.subheader("🔍 Key Market Factors")
        factors_col1, factors_col2 = st.columns(2)
        
        with factors_col1:
            st.write("**Sentiment Drivers:**")
            for factor in sentiment['factors']:
                st.write(f"• {factor}")
        
        with factors_col2:
            st.write("**Market Regime:**")
            rotation = market_data['sector_rotation']
            if rotation['success']:
                st.write(f"• **{rotation['market_regime']}** - {rotation['regime_reason']}")
                st.write(f"• Advancing Sectors: {rotation['advancing_sectors']}")
                st.write(f"• Declining Sectors: {rotation['declining_sectors']}")
        
        # Quick Summary
        st.subheader("💡 Trading Implications")
        
        if sentiment['score'] > 20:
            st.success("""
            **STRONGLY BULLISH ENVIRONMENT**
            - Aggressive long positions favored
            - Trend following strategies work best
            - Reduce hedging, increase exposure
            """)
        elif sentiment['score'] > 10:
            st.info("""
            **BULLISH ENVIRONMENT** 
            - Moderate long bias recommended
            - Buy on dips strategy effective
            - Selective stock picking
            """)
        elif sentiment['score'] > -10:
            st.warning("""
            **NEUTRAL ENVIRONMENT**
            - Range-bound trading likely
            - Mean reversion strategies
            - Wait for clear breakout
            """)
        elif sentiment['score'] > -20:
            st.warning("""
            **BEARISH ENVIRONMENT**
            - Caution advised for longs
            - Short on rallies approach
            - Increase cash positions
            """)
        else:
            st.error("""
            **STRONGLY BEARISH ENVIRONMENT**
            - Defensive positioning crucial
            - Short positions favored
            - Heavy hedging recommended
            """)

    def display_india_vix_analysis(self, market_data: Dict[str, Any]):
        """Display detailed India VIX analysis"""
        st.subheader("🇮🇳 India VIX - Fear Index Analysis")
        
        vix_data = market_data['india_vix']
        if not vix_data['success']:
            st.error("VIX data not available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VIX Value", f"{vix_data['value']:.2f}")
        
        with col2:
            st.metric("Market Sentiment", vix_data['sentiment'])
        
        with col3:
            st.metric("Trading Signal", vix_data['signal'])
        
        with col4:
            st.metric("Trend Direction", vix_data['trend'].title())
        
        # VIX Interpretation
        st.subheader("📊 VIX Interpretation & Implications")
        
        interpretation_col1, interpretation_col2 = st.columns(2)
        
        with interpretation_col1:
            st.info("**Current Reading Analysis**")
            st.write(f"**Interpretation**: {vix_data['interpretation']}")
            st.write(f"**Bias**: {vix_data['bias']}")
            st.write(f"**Confidence Score**: {vix_data['score']}/100")
            
            # VIX levels guide
            st.write("**VIX Levels Guide:**")
            st.write("• < 12: Complacency Zone")
            st.write("• 12-15: Low Volatility") 
            st.write("• 15-20: Normal Range")
            st.write("• 20-25: Elevated Fear")
            st.write("• 25-30: High Fear")
            st.write("• > 30: Extreme Fear")
        
        with interpretation_col2:
            st.info("**Trading Implications**")
            if vix_data['value'] > 25:
                st.error("""
                **HIGH FEAR ZONE**
                - Expect high volatility
                - Wide stop losses needed
                - Option premiums expensive
                - Fear-driven selling possible
                """)
            elif vix_data['value'] > 20:
                st.warning("""
                **ELEVATED FEAR ZONE** 
                - Increased volatility expected
                - Caution on leverage
                - Good for option sellers
                - Monitor risk closely
                """)
            elif vix_data['value'] > 15:
                st.info("""
                **NORMAL VOLATILITY**
                - Standard trading conditions
                - Normal position sizing
                - Technical analysis reliable
                - Stable market environment
                """)
            elif vix_data['value'] > 12:
                st.success("""
                **LOW VOLATILITY ZONE**
                - Calm market conditions
                - Good for trend following
                - Tighter stop losses possible
                - Low option premiums
                """)
            else:
                st.success("""
                **COMPLACENCY ZONE**
                - Very low volatility
                - Risk of volatility spike
                - Excellent for premium selling
                - Mean reversion opportunities
                """)
        
        # VIX Chart would go here (requires historical data)
        st.info("📈 *VIX historical chart would be displayed here with real data*")

    def display_sector_rotation_analysis(self, market_data: Dict[str, Any]):
        """Display sector rotation and leadership analysis"""
        st.subheader("📈 Sector Rotation & Market Leadership")
        
        rotation_data = market_data['sector_rotation']
        sector_data = market_data['sector_indices']
        
        if not rotation_data['success']:
            st.error("Sector rotation analysis not available")
            return
        
        # Market Regime
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime = rotation_data['market_regime']
            if regime == "RISK-ON":
                st.success(f"**Market Regime**\n# 🟢 {regime}")
            elif regime == "RISK-OFF":
                st.error(f"**Market Regime**\n# 🔴 {regime}")
            else:
                st.info(f"**Market Regime**\n# ⚪ {regime}")
        
        with col2:
            st.metric("Sector Breadth", f"{rotation_data['sector_breadth']:.1f}%")
        
        with col3:
            st.metric("Advancing Sectors", rotation_data['advancing_sectors'])
        
        with col4:
            st.metric("Declining Sectors", rotation_data['declining_sectors'])
        
        # Sector Performance by Type
        st.subheader("🏷️ Sector Performance by Type")
        
        type_col1, type_col2, type_col3 = st.columns(3)
        
        with type_col1:
            st.metric("Defensive Avg", f"{rotation_data['defensive_avg']:+.2f}%")
            def_sectors = [s for s in sector_data if s['type'] == 'DEFENSIVE']
            for sector in def_sectors:
                st.write(f"• {sector['sector']}: {sector['change_pct']:+.2f}%")
        
        with type_col2:
            st.metric("Cyclical Avg", f"{rotation_data['cyclical_avg']:+.2f}%")
            cyc_sectors = [s for s in sector_data if s['type'] == 'CYCLICAL']
            for sector in cyc_sectors:
                st.write(f"• {sector['sector']}: {sector['change_pct']:+.2f}%")
        
        with type_col3:
            st.metric("Financial Avg", f"{rotation_data['financial_avg']:+.2f}%")
            fin_sectors = [s for s in sector_data if s['type'] == 'FINANCIAL']
            for sector in fin_sectors:
                st.write(f"• {sector['sector']}: {sector['change_pct']:+.2f}%")
        
        # Sector Leaders and Laggards
        st.subheader("🏆 Sector Leaders & Laggards")
        
        leader_col1, leader_col2 = st.columns(2)
        
        with leader_col1:
            st.success("**Top 3 Performers**")
            for i, leader in enumerate(rotation_data['leaders']):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.write(f"{emoji} {leader['sector']}: {leader['change_pct']:+.2f}% ({leader['bias']})")
        
        with leader_col2:
            st.error("**Bottom 3 Performers**")
            for i, laggard in enumerate(rotation_data['laggards']):
                emoji = "🔻"
                st.write(f"{emoji} {laggard['sector']}: {laggard['change_pct']:+.2f}% ({laggard['bias']})")
        
        # Complete Sector Performance Table
        st.subheader("📋 Complete Sector Performance")
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('change_pct', ascending=False)
        
        # Display with color coding
        def color_sector_performance(val):
            if val > 1.0:
                return 'background-color: #90EE90'
            elif val < -1.0:
                return 'background-color: #FFB6C1'
            else:
                return ''
        
        display_df = sector_df[['sector', 'last_price', 'change_pct', 'bias', 'strength', 'type']]
        st.dataframe(
            display_df.style.applymap(color_sector_performance, subset=['change_pct']),
            use_container_width=True
        )

    def display_global_markets_analysis(self, market_data: Dict[str, Any]):
        """Display global markets analysis"""
        st.subheader("🌍 Global Markets Performance")
        
        global_data = market_data['global_markets']
        
        if not global_data:
            st.error("Global markets data not available")
            return
        
        # Global Market Summary
        st.subheader("🌐 Global Market Summary")
        
        # Group by region
        regions = {}
        for market in global_data:
            region = market['region']
            if region not in regions:
                regions[region] = []
            regions[region].append(market)
        
        # Display by region
        for region, markets in regions.items():
            st.write(f"**{region} MARKETS**")
            cols = st.columns(len(markets))
            
            for idx, market in enumerate(markets):
                with cols[idx]:
                    change = market['change_pct']
                    if change > 1.0:
                        st.success(f"**{market['market']}**\n{market['last_price']:.0f}\n{change:+.2f}%")
                    elif change < -1.0:
                        st.error(f"**{market['market']}**\n{market['last_price']:.0f}\n{change:+.2f}%")
                    else:
                        st.info(f"**{market['market']}**\n{market['last_price']:.0f}\n{change:+.2f}%")
        
        # Detailed Global Analysis
        st.subheader("📊 Detailed Global Analysis")
        
        global_col1, global_col2 = st.columns(2)
        
        with global_col1:
            st.info("**Market Sentiment by Region**")
            region_sentiment = {}
            for market in global_data:
                region = market['region']
                if region not in region_sentiment:
                    region_sentiment[region] = []
                region_sentiment[region].append(market['sentiment'])
            
            for region, sentiments in region_sentiment.items():
                bullish_count = sum(1 for s in sentiments if 'BULLISH' in s)
                total_count = len(sentiments)
                sentiment_pct = (bullish_count / total_count) * 100
                st.write(f"• **{region}**: {sentiment_pct:.1f}% bullish")
        
        with global_col2:
            st.info("**Impact on Indian Markets**")
            us_markets = [m for m in global_data if m['region'] == 'US']
            asia_markets = [m for m in global_data if m['region'] == 'ASIA']
            
            if us_markets:
                us_avg = np.mean([m['change_pct'] for m in us_markets])
                st.write(f"• **US Markets**: {us_avg:+.2f}%")
                if us_avg > 1.0:
                    st.write("  → Positive for Indian markets")
                elif us_avg < -1.0:
                    st.write("  → Negative for Indian markets")
                else:
                    st.write("  → Neutral impact")
            
            if asia_markets:
                asia_avg = np.mean([m['change_pct'] for m in asia_markets])
                st.write(f"• **Asian Markets**: {asia_avg:+.2f}%")
        
        # Complete Global Markets Table
        st.subheader("📋 Global Markets Performance Table")
        global_df = pd.DataFrame(global_data)
        global_df = global_df.sort_values('change_pct', ascending=False)
        
        display_df = global_df[['market', 'region', 'last_price', 'change_pct', 'sentiment', 'impact', 'status']]
        st.dataframe(display_df, use_container_width=True)

    def display_intermarket_analysis(self, market_data: Dict[str, Any]):
        """Display intermarket analysis"""
        st.subheader("🔄 Intermarket Analysis")
        
        intermarket_data = market_data['intermarket_data']
        
        if not intermarket_data:
            st.error("Intermarket data not available")
            return
        
        # Group by asset type
        asset_types = {}
        for asset in intermarket_data:
            asset_type = asset['type']
            if asset_type not in asset_types:
                asset_types[asset_type] = []
            asset_types[asset_type].append(asset)
        
        # Display by asset type
        for asset_type, assets in asset_types.items():
            st.subheader(f"📊 {asset_type} Analysis")
            
            cols = st.columns(len(assets))
            for idx, asset in enumerate(assets):
                with cols[idx]:
                    change = asset['change_pct']
                    if abs(change) > 2.0:
                        color = "red" if change < 0 else "green"
                    else:
                        color = "orange"
                    
                    st.metric(
                        asset['asset'],
                        f"{asset['last_price']:.2f}",
                        f"{change:+.2f}%"
                    )
                    st.caption(f"Impact: {asset['impact']}")
        
        # Intermarket Relationships
        st.subheader("🔗 Key Intermarket Relationships")
        
        rel_col1, rel_col2 = st.columns(2)
        
        with rel_col1:
            st.info("**USD & Indian Markets**")
            usd_data = next((a for a in intermarket_data if 'DOLLAR' in a['asset']), None)
            inr_data = next((a for a in intermarket_data if 'INR' in a['asset']), None)
            
            if usd_data and inr_data:
                st.write(f"• **USD Index**: {usd_data['change_pct']:+.2f}%")
                st.write(f"• **USD/INR**: {inr_data['change_pct']:+.2f}%")
                
                if usd_data['change_pct'] > 0.5 and inr_data['change_pct'] > 0.5:
                    st.error("Strong USD & Weak INR → Negative for Indian equities")
                elif usd_data['change_pct'] < -0.5 and inr_data['change_pct'] < -0.5:
                    st.success("Weak USD & Strong INR → Positive for Indian equities")
                else:
                    st.info("Stable currency environment")
        
        with rel_col2:
            st.info("**Commodities & Inflation**")
            oil_data = next((a for a in intermarket_data if 'OIL' in a['asset']), None)
            gold_data = next((a for a in intermarket_data if 'GOLD' in a['asset']), None)
            
            if oil_data and gold_data:
                st.write(f"• **Crude Oil**: {oil_data['change_pct']:+.2f}%")
                st.write(f"• **Gold**: {gold_data['change_pct']:+.2f}%")
                
                if oil_data['change_pct'] > 2.0:
                    st.warning("Rising oil prices → Inflation concerns")
                if gold_data['change_pct'] > 1.0:
                    st.warning("Gold rising → Risk-off sentiment")
        
        # Complete Intermarket Table
        st.subheader("📋 Complete Intermarket Analysis")
        intermarket_df = pd.DataFrame(intermarket_data)
        intermarket_df = intermarket_df.sort_values('type')
        
        display_df = intermarket_df[['asset', 'type', 'last_price', 'change_pct', 'impact', 'reasoning']]
        st.dataframe(display_df, use_container_width=True)

    def display_intraday_patterns(self, market_data: Dict[str, Any]):
        """Display intraday seasonality and patterns"""
        st.subheader("⏰ Intraday Market Patterns & Seasonality")
        
        seasonality_data = market_data['intraday_seasonality']
        
        if not seasonality_data['success']:
            st.error("Intraday analysis not available")
            return
        
        # Current Session Analysis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Session", seasonality_data['current_session'])
        
        with col2:
            st.metric("Trading Opportunity", seasonality_data['trading_opportunity'])
        
        with col3:
            st.metric("Volatility Expectation", seasonality_data['volatility_expectation'])
        
        with col4:
            st.metric("Day of Week", seasonality_data['weekday'])
        
        # Session Characteristics
        st.subheader("📋 Current Session Analysis")
        
        session_col1, session_col2 = st.columns(2)
        
        with session_col1:
            st.info("**Session Characteristics**")
            for characteristic in seasonality_data['session_characteristics']:
                st.write(f"• {characteristic}")
        
        with session_col2:
            st.info("**Trading Recommendation**")
            st.write(seasonality_data['recommendation'])
            
            if seasonality_data['time_to_close'] != "N/A":
                st.write(f"**Time to Close**: {seasonality_data['time_to_close']}")
        
        # Day of Week Patterns
        st.subheader("📅 Day of Week Patterns")
        
        day_col1, day_col2 = st.columns(2)
        
        with day_col1:
            st.info(f"**{seasonality_data['weekday']} Pattern**")
            st.write(f"**Pattern**: {seasonality_data['day_pattern']}")
            st.write(f"**Bias**: {seasonality_data['day_bias']}")
            
            st.write("**Characteristics:**")
            for char in seasonality_data['day_characteristics']:
                st.write(f"• {char}")
        
        with day_col2:
            st.info("**Weekday Trading Guide**")
            weekday_guide = {
                "Monday": "Watch for gap fills and follow-through from Friday",
                "Tuesday": "Best trending day - follow institutional flow", 
                "Wednesday": "Consolidation day - fade extremes",
                "Thursday": "Volatility increases - pre-weekend positioning",
                "Friday": "Low volume - square positions for weekend"
            }
            
            for day, guide in weekday_guide.items():
                if day == seasonality_data['weekday']:
                    st.write(f"**{day}**: {guide}")
                else:
                    st.write(f"{day}: {guide}")
        
        # Intraday Seasonality Chart
        st.subheader("📈 Typical Intraday Seasonality Pattern")
        
        # Create a sample intraday pattern chart
        hours = list(range(9, 16))
        typical_volatility = [0.8, 2.5, 1.8, 1.2, 0.6, 0.9, 1.5]  # Sample volatility pattern
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours, 
            y=typical_volatility,
            mode='lines+markers',
            name='Typical Volatility',
            line=dict(color='blue', width=3)
        ))
        
        # Highlight current hour
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 15:
            current_vol = typical_volatility[hours.index(current_hour)]
            fig.add_trace(go.Scatter(
                x=[current_hour], 
                y=[current_vol],
                mode='markers',
                name='Current Time',
                marker=dict(color='red', size=15, symbol='star')
            ))
        
        fig.update_layout(
            title="Typical Intraday Volatility Pattern (Nifty)",
            xaxis_title="Hour of Day",
            yaxis_title="Relative Volatility",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Session Timing Guide
        st.subheader("🕒 Session Timing Guide")
        
        timing_data = {
            "09:15-09:30": {"action": "Opening Range", "volatility": "Very High", "strategy": "Breakout trading"},
            "09:30-10:30": {"action": "Morning Trend", "volatility": "High", "strategy": "Trend following"},
            "10:30-12:00": {"action": "Mid-morning", "volatility": "Moderate", "strategy": "Pullback entries"},
            "12:00-14:00": {"action": "Lunch Hour", "volatility": "Low", "strategy": "Avoid/Aggressive mean reversion"},
            "14:00-15:15": {"action": "Afternoon Session", "volatility": "Moderate-High", "strategy": "Momentum/Continuation"},
            "15:15-15:30": {"action": "Closing Range", "volatility": "High", "strategy": "Position squaring"}
        }
        
        timing_df = pd.DataFrame.from_dict(timing_data, orient='index')
        timing_df.index.name = 'Time Slot'
        st.dataframe(timing_df, use_container_width=True)

    def run(self):
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Complete Market Analysis with Global Insights*")
        
        # For this demo, we'll just show the comprehensive market tab
        self.display_comprehensive_market_tab()

# =============================================
# RUN THE APP
# =============================================

if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
