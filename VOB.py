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
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# TRADING SAFETY MANAGER
# =============================================

class TradingSafetyManager:
    """Comprehensive safety checks for trading signal reliability"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def should_trust_signals(self, df: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
        """Comprehensive signal reliability check"""
        detailed_report = {}
        
        # Basic checks
        basic_checks = {
            'market_hours': self.is_regular_market_hours(),
            'data_fresh': self.is_data_timestamp_recent(df, minutes=5) if df is not None else False,
            'sufficient_data': len(df) >= 50 if df is not None else False
        }
        
        detailed_report = basic_checks.copy()
        passed_checks = sum(basic_checks.values())
        total_checks = len(basic_checks)
        
        confidence = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        if confidence >= 70:
            return True, f"High reliability ({confidence:.1f}%)", detailed_report
        elif confidence >= 50:
            return True, f"Moderate reliability ({confidence:.1f}%)", detailed_report
        else:
            failed = [k for k, v in basic_checks.items() if not v]
            reason = f"Low reliability ({confidence:.1f}%): {', '.join(failed[:3])}"
            return False, reason, detailed_report

    def is_regular_market_hours(self) -> bool:
        """Check if current time is within regular market hours"""
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            
            market_open = datetime.strptime("09:15", "%H:%M").time()
            market_close = datetime.strptime("15:30", "%H:%M").time()
            
            is_weekday = now.weekday() < 5
            
            return (is_weekday and market_open <= current_time <= market_close)
        except:
            return False

    def is_data_timestamp_recent(self, df: pd.DataFrame, minutes: int = 5) -> bool:
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

# =============================================
# ENHANCED MARKET DATA FETCHER
# =============================================

class EnhancedMarketData:
    """Fetch market data from multiple sources"""

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

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

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """Fetch all enhanced market data"""
        result = {
            'timestamp': self.get_current_time_ist(),
            'india_vix': {},
            'summary': {}
        }

        # Fetch India VIX
        result['india_vix'] = self.fetch_india_vix()

        # Calculate summary
        result['summary'] = self._calculate_summary(result)

        return result

    def _calculate_summary(self, data: Dict) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {
            'total_data_points': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'overall_sentiment': 'NEUTRAL'
        }

        all_scores = []

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

        if all_scores:
            summary['avg_score'] = np.mean(all_scores)

            if summary['avg_score'] > 25:
                summary['overall_sentiment'] = 'BULLISH'
            elif summary['avg_score'] < -25:
                summary['overall_sentiment'] = 'BEARISH'
            else:
                summary['overall_sentiment'] = 'NEUTRAL'

        return summary

# =============================================
# BIAS ANALYSIS PRO
# =============================================

class BiasAnalysisPro:
    """Comprehensive Bias Analysis"""

    def __init__(self):
        self.config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'rsi_period': 14,
            'bias_strength': 60,
        }

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

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        """Analyze bias indicators"""
        try:
            df = self.fetch_data(symbol, period='7d', interval='5m')

            if df.empty or len(df) < 100:
                return {
                    'success': False,
                    'error': f'Insufficient data (fetched {len(df)} candles)',
                    'symbol': symbol
                }

            current_price = df['Close'].iloc[-1]
            bias_results = []

            # RSI
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

            # Calculate overall bias
            bullish_count = sum(1 for b in bias_results if 'BULLISH' in b['bias'])
            bearish_count = sum(1 for b in bias_results if 'BEARISH' in b['bias'])
            
            total_score = sum(b['score'] for b in bias_results)

            if bullish_count > bearish_count:
                overall_bias = "BULLISH"
                overall_confidence = 75
            elif bearish_count > bullish_count:
                overall_bias = "BEARISH"
                overall_confidence = 75
            else:
                overall_bias = "NEUTRAL"
                overall_confidence = 50

            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                'bias_results': bias_results,
                'overall_bias': overall_bias,
                'overall_score': total_score / len(bias_results) if bias_results else 0,
                'overall_confidence': overall_confidence,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': 0,
                'total_indicators': len(bias_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }

# =============================================
# MASTER DECISION ENGINE
# =============================================

class MasterDecisionEngine:
    """Master trading decision system"""

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def make_trading_decision(self,
                            price_data: pd.DataFrame,
                            bias_data: Dict[str, Any],
                            market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make final trading decision"""
        
        if price_data.empty or len(price_data) < 50:
            return self._create_empty_decision("Insufficient data")
        
        try:
            current_price = price_data['close'].iloc[-1]
            current_time = datetime.now(self.ist)
            
            decision = {
                'trade_decision': 'NO TRADE',
                'trade_direction': 'NEUTRAL',
                'confidence': 0,
                'trade_type': 'NONE',
                'position_size': 'NONE',
                'entry_zone': 'N/A',
                'targets': [],
                'stop_loss': 'N/A',
                'key_factors': [],
                'risk_level': 'HIGH',
                'timeframe': 'N/A',
                'simple_summary': 'No trade recommended',
                'timestamp': current_time,
                'current_price': current_price,
                'execution_approved': False
            }
            
            # Analyze bias if available
            if bias_data and bias_data.get('success'):
                overall_bias = bias_data.get('overall_bias', 'NEUTRAL')
                confidence = bias_data.get('overall_confidence', 0)
                
                if "BULLISH" in overall_bias and confidence >= 70:
                    decision.update({
                        'trade_decision': 'TRADE',
                        'trade_direction': 'LONG',
                        'confidence': confidence,
                        'trade_type': 'TREND_FOLLOWING',
                        'position_size': 'NORMAL',
                        'entry_zone': f"{current_price * 0.995:.0f}-{current_price:.0f}",
                        'targets': [current_price * 1.01, current_price * 1.02],
                        'stop_loss': current_price * 0.985,
                        'risk_level': 'LOW',
                        'timeframe': '1h-4h',
                        'simple_summary': 'BULLISH trend confirmed',
                        'execution_approved': True
                    })
                    decision['key_factors'].append(f"Bullish bias with {confidence}% confidence")
                    
                elif "BEARISH" in overall_bias and confidence >= 70:
                    decision.update({
                        'trade_decision': 'TRADE',
                        'trade_direction': 'SHORT',
                        'confidence': confidence,
                        'trade_type': 'TREND_FOLLOWING',
                        'position_size': 'NORMAL',
                        'entry_zone': f"{current_price:.0f}-{current_price * 1.005:.0f}",
                        'targets': [current_price * 0.99, current_price * 0.98],
                        'stop_loss': current_price * 1.015,
                        'risk_level': 'LOW',
                        'timeframe': '1h-4h',
                        'simple_summary': 'BEARISH trend confirmed',
                        'execution_approved': True
                    })
                    decision['key_factors'].append(f"Bearish bias with {confidence}% confidence")
            
            return decision
            
        except Exception as e:
            return self._create_error_decision(f"Decision engine error: {str(e)}")
    
    def _create_empty_decision(self, reason: str) -> Dict[str, Any]:
        """Create empty decision"""
        return {
            'trade_decision': 'NO TRADE',
            'trade_direction': 'NEUTRAL',
            'confidence': 0,
            'trade_type': 'NONE',
            'position_size': 'NONE',
            'entry_zone': 'N/A',
            'targets': [],
            'stop_loss': 'N/A',
            'key_factors': [f"Insufficient data: {reason}"],
            'risk_level': 'HIGH',
            'timeframe': 'N/A',
            'simple_summary': f'No decision - {reason}',
            'timestamp': datetime.now(self.ist),
            'execution_approved': False
        }
    
    def _create_error_decision(self, error_msg: str) -> Dict[str, Any]:
        """Create error decision"""
        decision = self._create_empty_decision(error_msg)
        decision['error'] = error_msg
        return decision
    
    def format_decision_for_display(self, decision: Dict[str, Any]) -> str:
        """Format decision for display"""
        
        if decision.get('trade_decision') == 'NO TRADE':
            return f"""
## 🚫 NO TRADE SIGNAL

**Reason:** {decision.get('simple_summary', 'Unknown')}
**Confidence:** {decision.get('confidence', 0)}%

**Key Factors:**
{chr(10).join(['• ' + factor for factor in decision.get('key_factors', [])])}
"""
        
        else:  # TRADE decision
            direction_emoji = "🟢" if decision['trade_direction'] == 'LONG' else "🔴"
            return f"""
## 🎯 {direction_emoji} TRADE SIGNAL: {decision['trade_direction']}

**Type:** {decision.get('trade_type', 'Unknown')}
**Confidence:** {decision.get('confidence', 0)}%
**Position Size:** {decision.get('position_size', 'Unknown')}

**💰 Entry Zone:** ₹{decision.get('entry_zone', 'N/A')}
**🎯 Target 1:** ₹{decision['targets'][0] if decision.get('targets') else 'N/A'}
**🎯 Target 2:** ₹{decision['targets'][1] if decision.get('targets') and len(decision['targets']) > 1 else 'N/A'}
**🛑 Stop Loss:** ₹{decision.get('stop_loss', 'N/A')}

**⏰ Timeframe:** {decision.get('timeframe', 'N/A')}
**📊 Risk Level:** {decision.get('risk_level', 'Unknown')}

**🔍 Key Factors:**
{chr(10).join(['• ' + factor for factor in decision.get('key_factors', [])])}

**✅ Execution Approved:** {decision.get('execution_approved', False)}
"""

# =============================================
# ENHANCED NIFTY APP
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        
        # Initialize components
        self.market_data_fetcher = EnhancedMarketData()
        self.safety_manager = TradingSafetyManager()
        self.bias_analyzer = BiasAnalysisPro()
        self.decision_engine = MasterDecisionEngine()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'comprehensive_bias_data' not in st.session_state:
            st.session_state.comprehensive_bias_data = None
        if 'last_comprehensive_bias_update' not in st.session_state:
            st.session_state.last_comprehensive_bias_update = None
        if 'enhanced_market_data' not in st.session_state:
            st.session_state.enhanced_market_data = None
        if 'last_market_data_update' not in st.session_state:
            st.session_state.last_market_data_update = None
        if 'master_decision' not in st.session_state:
            st.session_state.master_decision = None
        if 'last_decision_time' not in st.session_state:
            st.session_state.last_decision_time = None
        if 'decision_history' not in st.session_state:
            st.session_state.decision_history = []
    
    def fetch_intraday_data(self, interval: str = "5", days_back: int = 5) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Yahoo Finance"""
        try:
            ticker = yf.Ticker("^NSEI")
            df = ticker.history(period=f"{days_back}d", interval=f"{interval}m")
            
            if df.empty:
                return None
                
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def display_master_decision(self):
        """Display Master Decision Engine"""
        st.header("🧠 MASTER DECISION ENGINE")
        st.success("THE BRAIN IS NOW ACTIVE")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Intelligent trading decisions based on technical analysis")
        with col2:
            if st.button("🎯 Get Decision", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    self.generate_master_decision()
        
        st.divider()
        
        # Display current decision
        if st.session_state.master_decision:
            decision = st.session_state.master_decision
            
            # Decision Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trade_decision = decision.get('trade_decision', 'UNKNOWN')
                if trade_decision == 'TRADE':
                    st.success(f"**Decision:** {trade_decision}")
                else:
                    st.error(f"**Decision:** {trade_decision}")
            
            with col2:
                direction = decision.get('trade_direction', 'NEUTRAL')
                if direction == 'LONG':
                    st.success(f"**Direction:** {direction}")
                elif direction == 'SHORT':
                    st.error(f"**Direction:** {direction}")
                else:
                    st.info(f"**Direction:** {direction}")
            
            with col3:
                st.metric("Confidence", f"{decision.get('confidence', 0)}%")
            
            with col4:
                st.metric("Position Size", decision.get('position_size', 'NONE'))
            
            # Detailed decision display
            st.markdown(self.decision_engine.format_decision_for_display(decision))
        
        else:
            st.info("👆 Click 'Get Decision' to activate the Master Brain")

    def generate_master_decision(self):
        """Generate master trading decision"""
        try:
            # Fetch price data
            price_data = self.fetch_intraday_data(interval='5')
            bias_data = st.session_state.comprehensive_bias_data
            market_data = st.session_state.enhanced_market_data
            
            if price_data is not None and not price_data.empty:
                decision = self.decision_engine.make_trading_decision(
                    price_data=price_data,
                    bias_data=bias_data,
                    market_data=market_data
                )
                
                st.session_state.master_decision = decision
                st.session_state.last_decision_time = datetime.now(self.ist)
                
                # Store in history
                st.session_state.decision_history.append(decision)
                
                # Keep only last 20 decisions
                if len(st.session_state.decision_history) > 20:
                    st.session_state.decision_history.pop(0)
            else:
                st.error("Could not fetch price data")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

    def display_comprehensive_bias_analysis(self):
        """Display bias analysis"""
        st.header("🎯 Comprehensive Technical Bias Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Technical bias analysis with RSI and other indicators")
        with col2:
            if st.button("🔄 Update Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        bias_data = self.bias_analyzer.analyze_all_bias_indicators("^NSEI")
                        st.session_state.comprehensive_bias_data = bias_data
                        st.session_state.last_comprehensive_bias_update = datetime.now(self.ist)
                        if bias_data['success']:
                            st.success("Analysis completed!")
                        else:
                            st.error(f"Analysis failed: {bias_data['error']}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if not bias_data['success']:
                st.error(f"❌ Analysis failed: {bias_data['error']}")
                return
            
            # Display overall bias
            st.subheader("📊 Overall Market Bias")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
                st.metric("Overall Bias", f"{bias_color} {bias_data['overall_bias']}")
            with col2:
                st.metric("Score", f"{bias_data['overall_score']:.1f}")
            with col3:
                st.metric("Confidence", f"{bias_data['overall_confidence']:.1f}%")
            with col4:
                st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
            
            # Display indicators
            st.subheader("📈 Technical Indicators")
            bias_df = pd.DataFrame(bias_data['bias_results'])
            st.dataframe(bias_df[['indicator', 'value', 'bias', 'score']], use_container_width=True)
        
        else:
            st.info("👆 Click 'Update Analysis' to run analysis")

    def display_enhanced_market_data(self):
        """Display enhanced market data"""
        st.header("🌍 Enhanced Market Data")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Market analysis including India VIX")
        with col2:
            if st.button("🔄 Update Data", type="primary"):
                with st.spinner("Fetching..."):
                    try:
                        market_data = self.market_data_fetcher.fetch_all_enhanced_data()
                        st.session_state.enhanced_market_data = market_data
                        st.session_state.last_market_data_update = datetime.now(self.ist)
                        st.success("Data updated!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        if st.session_state.enhanced_market_data:
            market_data = st.session_state.enhanced_market_data
            
            # Display VIX
            if market_data['india_vix'].get('success'):
                vix = market_data['india_vix']
                st.subheader("🇮🇳 India VIX")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("VIX Value", f"{vix['value']:.2f}")
                with col2:
                    st.metric("Sentiment", vix['sentiment'])
                with col3:
                    st.metric("Bias", vix['bias'])
                with col4:
                    st.metric("Score", vix['score'])
            
            # Display summary
            st.subheader("📊 Market Summary")
            summary = market_data['summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Sentiment", summary['overall_sentiment'])
            with col2:
                st.metric("Avg Score", f"{summary['avg_score']:.1f}")
            with col3:
                st.metric("Data Points", summary['total_data_points'])
        
        else:
            st.info("👆 Click 'Update Data' to load market analysis")

    def run(self):
        """Main application"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Technical Analysis & Trading Decisions*")
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "🎯 Technical Bias", "🌍 Market Data", "🧠 Master Decision"
        ])
        
        with tab1:
            self.display_comprehensive_bias_analysis()
        
        with tab2:
            self.display_enhanced_market_data()
        
        with tab3:
            self.display_master_decision()

# =============================================
# RUN APPLICATION
# =============================================

if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
