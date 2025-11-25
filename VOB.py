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

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# CORE MODULES WITH COMPLETE OPTIONS ANALYSIS
# =============================================

class TradingSafetyManager:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def should_trust_signals(self, df: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
        basic_checks = {
            'market_hours': self.is_regular_market_hours(),
            'normal_volume': self.is_volume_normal(df),
            'data_fresh': self.is_data_timestamp_recent(df, minutes=2),
        }
        
        passed_checks = sum(basic_checks.values())
        total_checks = len(basic_checks)
        confidence = (passed_checks / total_checks) if total_checks > 0 else 0
        
        if confidence >= 0.8:
            return True, f"High reliability ({confidence:.1%})", basic_checks
        else:
            return False, f"Low reliability ({confidence:.1%})", basic_checks

    def is_regular_market_hours(self) -> bool:
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            market_open = datetime.strptime("09:15", "%H:%M").time()
            market_close = datetime.strptime("15:30", "%H:%M").time()
            is_weekday = now.weekday() < 5
            return is_weekday and market_open <= current_time <= market_close
        except:
            return False

    def is_volume_normal(self, df: pd.DataFrame) -> bool:
        try:
            if df is None or len(df) < 20:
                return False
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if avg_volume == 0:
                return False
            volume_ratio = current_volume / avg_volume
            return 0.3 <= volume_ratio <= 3.0
        except:
            return False

    def is_data_timestamp_recent(self, df: pd.DataFrame, minutes: int = 2) -> bool:
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

class NSEOptionsAnalyzer:
    """Complete NSE Options Analyzer with Real Data"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.NSE_INSTRUMENTS = {
            'NIFTY': {'lot_size': 50, 'atm_range': 100},
            'BANKNIFTY': {'lot_size': 25, 'atm_range': 200},
            'FINNIFTY': {'lot_size': 40, 'atm_range': 100},
        }
        
    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch real option chain data from NSE"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            # Get cookies first
            session.get("https://www.nseindia.com", timeout=10)
            
            if instrument in self.NSE_INSTRUMENTS:
                url = f"https://www.nseindia.com/api/option-chain-indices?symbol={instrument}"
            else:
                url = f"https://www.nseindia.com/api/option-chain-equities?symbol={instrument}"
                
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_greeks(self, option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks using Black-Scholes"""
        try:
            if T <= 0:
                return 0, 0, 0, 0, 0
                
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == 'CE':
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            else:  # PE
                delta = -norm.cdf(-d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
            
            return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
        except:
            return 0, 0, 0, 0, 0

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Comprehensive ATM bias analysis with real NSE data"""
        try:
            # Fetch option chain data
            chain_data = self.fetch_option_chain_data(instrument)
            if not chain_data['success']:
                return None
            
            data = chain_data['data']
            records = data['records']['data']
            underlying_value = data['records']['underlyingValue']
            expiry_dates = data['records']['expiryDates']
            
            if not records:
                return None
            
            # Use nearest expiry
            expiry = expiry_dates[0]
            
            # Filter records for current expiry
            filtered_records = []
            for item in records:
                if ('CE' in item and item['CE']['expiryDate'] == expiry) or \
                   ('PE' in item and item['PE']['expiryDate'] == expiry):
                    filtered_records.append(item)
            
            if not filtered_records:
                return None
            
            # Calculate totals
            total_ce_oi = sum(item['CE']['openInterest'] for item in filtered_records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in filtered_records if 'PE' in item)
            total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in filtered_records if 'CE' in item)
            total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in filtered_records if 'PE' in item)
            
            # Find ATM strike (closest to underlying)
            strikes = [item['strikePrice'] for item in filtered_records]
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_value))
            
            # Get ATM data
            atm_data = None
            for item in filtered_records:
                if item['strikePrice'] == atm_strike:
                    atm_data = item
                    break
            
            if not atm_data:
                return None
            
            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
            expiry_date = self.ist.localize(expiry_date.replace(hour=15, minute=30))
            T = max((expiry_date - today).total_seconds() / (365 * 24 * 3600), 0.001)
            r = 0.06  # Risk-free rate
            
            # Calculate Greeks for ATM
            atm_ce_iv = atm_data['CE']['impliedVolatility'] / 100 if 'CE' in atm_data else 0.2
            atm_pe_iv = atm_data['PE']['impliedVolatility'] / 100 if 'PE' in atm_data else 0.2
            
            ce_delta, ce_gamma, ce_vega, ce_theta, ce_rho = self.calculate_greeks(
                'CE', underlying_value, atm_strike, T, r, atm_ce_iv
            )
            pe_delta, pe_gamma, pe_vega, pe_theta, pe_rho = self.calculate_greeks(
                'PE', underlying_value, atm_strike, T, r, atm_pe_iv
            )
            
            # Calculate comprehensive metrics
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_change = total_pe_change / total_ce_change if total_ce_change != 0 else 0
            
            # Synthetic future calculation
            if 'CE' in atm_data and 'PE' in atm_data:
                synthetic_future = atm_strike + atm_data['CE']['lastPrice'] - atm_data['PE']['lastPrice']
                synthetic_diff = synthetic_future - underlying_value
                synthetic_bias = "Bullish" if synthetic_diff > 5 else "Bearish" if synthetic_diff < -5 else "Neutral"
            else:
                synthetic_bias = "Neutral"
                synthetic_diff = 0
            
            # Calculate bias score
            bias_score = 0
            
            # OI Bias
            if pcr_oi > 1.2: bias_score += 2
            elif pcr_oi < 0.8: bias_score -= 2
            
            # Change Bias
            if pcr_change > 1.2: bias_score += 1
            elif pcr_change < 0.8: bias_score -= 1
            
            # Synthetic Bias
            if synthetic_bias == "Bullish": bias_score += 1
            elif synthetic_bias == "Bearish": bias_score -= 1
            
            # Determine overall bias
            if bias_score >= 3:
                overall_bias = "Strong Bullish"
            elif bias_score >= 1:
                overall_bias = "Bullish"
            elif bias_score <= -3:
                overall_bias = "Strong Bearish"
            elif bias_score <= -1:
                overall_bias = "Bearish"
            else:
                overall_bias = "Neutral"
            
            # Prepare detailed ATM bias
            detailed_atm_bias = {}
            if atm_data:
                if 'CE' in atm_data:
                    detailed_atm_bias.update({
                        'CE_OI': atm_data['CE']['openInterest'],
                        'CE_Change': atm_data['CE']['changeinOpenInterest'],
                        'CE_Volume': atm_data['CE']['totalTradedVolume'],
                        'CE_IV': atm_data['CE']['impliedVolatility'],
                        'CE_LastPrice': atm_data['CE']['lastPrice'],
                        'CE_Delta': ce_delta,
                        'CE_Gamma': ce_gamma,
                        'CE_Vega': ce_vega
                    })
                
                if 'PE' in atm_data:
                    detailed_atm_bias.update({
                        'PE_OI': atm_data['PE']['openInterest'],
                        'PE_Change': atm_data['PE']['changeinOpenInterest'],
                        'PE_Volume': atm_data['PE']['totalTradedVolume'],
                        'PE_IV': atm_data['PE']['impliedVolatility'],
                        'PE_LastPrice': atm_data['PE']['lastPrice'],
                        'PE_Delta': pe_delta,
                        'PE_Gamma': pe_gamma,
                        'PE_Vega': pe_vega
                    })
                
                # Calculate bias metrics
                if 'CE' in atm_data and 'PE' in atm_data:
                    oi_bias = "Bullish" if detailed_atm_bias['PE_OI'] > detailed_atm_bias['CE_OI'] else "Bearish"
                    change_bias = "Bullish" if detailed_atm_bias['PE_Change'] > detailed_atm_bias['CE_Change'] else "Bearish"
                    volume_bias = "Bullish" if detailed_atm_bias['PE_Volume'] > detailed_atm_bias['CE_Volume'] else "Bearish"
                    iv_bias = "Bullish" if detailed_atm_bias['CE_IV'] > detailed_atm_bias['PE_IV'] else "Bearish"
                    
                    detailed_atm_bias.update({
                        'OI_Bias': oi_bias,
                        'Change_Bias': change_bias,
                        'Volume_Bias': volume_bias,
                        'IV_Bias': iv_bias,
                        'Synthetic_Bias': synthetic_bias
                    })
            
            # Prepare comprehensive metrics
            comprehensive_metrics = {
                'synthetic_bias': synthetic_bias,
                'synthetic_diff': synthetic_diff,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'total_ce_change': total_ce_change,
                'total_pe_change': total_pe_change,
                'expiry': expiry,
                'atm_strike': atm_strike
            }
            
            return {
                'instrument': instrument,
                'spot_price': underlying_value,
                'overall_bias': overall_bias,
                'bias_score': bias_score,
                'pcr_oi': pcr_oi,
                'pcr_change': pcr_change,
                'detailed_atm_bias': detailed_atm_bias,
                'comprehensive_metrics': comprehensive_metrics,
                'timestamp': datetime.now(self.ist)
            }
            
        except Exception as e:
            st.error(f"Error in options analysis for {instrument}: {str(e)}")
            return None

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive market bias across all instruments"""
        results = []
        for instrument in self.NSE_INSTRUMENTS.keys():
            try:
                bias_data = self.analyze_comprehensive_atm_bias(instrument)
                if bias_data:
                    results.append(bias_data)
            except Exception as e:
                st.warning(f"Could not fetch data for {instrument}: {str(e)}")
                continue
        return results

class EnhancedMarketData:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def fetch_india_vix(self) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]
                if vix_value > 25:
                    vix_sentiment, vix_bias, vix_score = "HIGH FEAR", "BEARISH", -75
                elif vix_value > 20:
                    vix_sentiment, vix_bias, vix_score = "ELEVATED FEAR", "BEARISH", -50
                elif vix_value > 15:
                    vix_sentiment, vix_bias, vix_score = "MODERATE", "NEUTRAL", 0
                elif vix_value > 12:
                    vix_sentiment, vix_bias, vix_score = "LOW VOLATILITY", "BULLISH", 40
                else:
                    vix_sentiment, vix_bias, vix_score = "COMPLACENCY", "NEUTRAL", 0

                return {
                    'success': True,
                    'value': vix_value,
                    'sentiment': vix_sentiment,
                    'bias': vix_bias,
                    'score': vix_score,
                }
        except:
            pass
        return {'success': False, 'error': 'India VIX data not available'}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        sectors_map = {
            '^CNXIT': 'NIFTY IT', '^CNXAUTO': 'NIFTY AUTO', '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL', '^CNXREALTY': 'NIFTY REALTY', '^CNXFMCG': 'NIFTY FMCG',
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
                    
                    if change_pct > 1.5: bias, score = "STRONG BULLISH", 75
                    elif change_pct > 0.5: bias, score = "BULLISH", 50
                    elif change_pct < -1.5: bias, score = "STRONG BEARISH", -75
                    elif change_pct < -0.5: bias, score = "BEARISH", -50
                    else: bias, score = "NEUTRAL", 0

                    sector_data.append({
                        'sector': name, 'last_price': last_price, 'change_pct': change_pct,
                        'bias': bias, 'score': score
                    })
            except:
                continue
        return sector_data

class BiasAnalysisPro:
    def __init__(self):
        self.config = {
            'rsi_period': 14, 'mfi_period': 10, 'dmi_period': 13, 'dmi_smoothing': 8
        }

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
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
        except:
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        try:
            df = self.fetch_data(symbol, period='7d', interval='5m')
            if df.empty or len(df) < 100:
                return {'success': False, 'error': 'Insufficient data'}

            current_price = df['Close'].iloc[-1]
            bias_results = []

            # RSI Analysis
            rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
            rsi_value = rsi.iloc[-1]
            rsi_bias = "BULLISH" if rsi_value > 50 else "BEARISH"
            bias_results.append({
                'indicator': 'RSI', 'value': f"{rsi_value:.2f}", 
                'bias': rsi_bias, 'score': 100 if rsi_bias == "BULLISH" else -100
            })

            # Volume Analysis (simplified)
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_bias = "BULLISH" if volume_ratio > 1.2 else "BEARISH" if volume_ratio < 0.8 else "NEUTRAL"
            bias_results.append({
                'indicator': 'Volume', 'value': f"{volume_ratio:.2f}x",
                'bias': volume_bias, 'score': 50 if volume_bias == "BULLISH" else -50 if volume_bias == "BEARISH" else 0
            })

            # Price Trend
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
            trend_bias = "BULLISH" if price_change > 0.5 else "BEARISH" if price_change < -0.5 else "NEUTRAL"
            bias_results.append({
                'indicator': 'Price Trend', 'value': f"{price_change:+.2f}%",
                'bias': trend_bias, 'score': 75 if trend_bias == "BULLISH" else -75 if trend_bias == "BEARISH" else 0
            })

            # Calculate overall bias
            bullish_count = len([b for b in bias_results if 'BULLISH' in b['bias']])
            bearish_count = len([b for b in bias_results if 'BEARISH' in b['bias']])
            total_indicators = len(bias_results)
            
            if bullish_count / total_indicators >= 0.6:
                overall_bias, overall_score = "BULLISH", 75
            elif bearish_count / total_indicators >= 0.6:
                overall_bias, overall_score = "BEARISH", -75
            else:
                overall_bias, overall_score = "NEUTRAL", 0

            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'bias_results': bias_results,
                'overall_bias': overall_bias,
                'overall_score': overall_score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'total_indicators': total_indicators
            }
        except Exception as e:
            return {'success': False, 'error': f"Bias analysis error: {str(e)}"}

# =============================================
# ENHANCED APP WITH COMPLETE OPTIONS TABULATION
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        
        # Initialize core components
        self.safety_manager = TradingSafetyManager()
        self.market_data_fetcher = EnhancedMarketData()
        self.bias_analyzer = BiasAnalysisPro()
        self.options_analyzer = NSEOptionsAnalyzer()
        
        # Initialize session state
        self.init_session_state()

    def init_session_state(self):
        if 'market_bias_data' not in st.session_state:
            st.session_state.market_bias_data = None
        if 'comprehensive_bias_data' not in st.session_state:
            st.session_state.comprehensive_bias_data = None
        if 'enhanced_market_data' not in st.session_state:
            st.session_state.enhanced_market_data = None

    def get_dhan_headers(self) -> Dict[str, str]:
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': "demo_token",
            'client-id': "demo_client"
        }

    def fetch_intraday_data(self, interval: str = "5") -> Optional[Dict[str, Any]]:
        try:
            payload = {
                "securityId": str(self.nifty_security_id),
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": str(interval),
            }
            
            response = requests.post(
                "https://api.dhan.co/v2/charts/intraday",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and 'open' in data and len(data['open']) > 0:
                    return data
            return None
        except:
            return None

    def process_data(self, api_data: Dict[str, Any]) -> pd.DataFrame:
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
        return df.set_index('datetime')

    def create_simple_chart(self, df: pd.DataFrame) -> Optional[go.Figure]:
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Nifty 50 Price', 'Volume'),
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
                name='Nifty 50'
            ),
            row=1, col=1
        )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig

    def display_price_analysis(self):
        st.header("📈 Price Analysis")
        
        # Fetch data
        api_data = self.fetch_intraday_data(interval='5')
        df = self.process_data(api_data) if api_data else pd.DataFrame()
        
        if not df.empty:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nifty Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("Volume", f"{latest['volume']:,}")
            with col3:
                is_trustworthy, reason, _ = self.safety_manager.should_trust_signals(df)
                status = "✅ Safe" if is_trustworthy else "❌ Unsafe"
                st.metric("Trading Safety", status, reason)
            
            # Display chart
            chart = self.create_simple_chart(df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.error("No data available")

    def display_options_analysis(self):
        st.header("📊 NSE Options Chain Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Real-time options chain data from NSE with comprehensive bias analysis")
        with col2:
            if st.button("🔄 Refresh Options Data", type="primary"):
                with st.spinner("Fetching live options data from NSE..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
        
        if st.session_state.market_bias_data:
            for instrument_data in st.session_state.market_bias_data:
                with st.expander(f"🎯 {instrument_data['instrument']} - Options Analysis", expanded=True):
                    # Basic Information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                    with col2:
                        bias_color = "🟢" if "Bullish" in instrument_data['overall_bias'] else "🔴" if "Bearish" in instrument_data['overall_bias'] else "🟡"
                        st.metric("Overall Bias", f"{bias_color} {instrument_data['overall_bias']}")
                    with col3:
                        st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
                    with col4:
                        pcr_color = "green" if instrument_data['pcr_oi'] > 1.0 else "red"
                        st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}", delta_color="off")
                    
                    st.divider()
                    
                    # Trading Recommendation
                    st.subheader("💡 Trading Recommendation")
                    bias_strength = abs(instrument_data['bias_score'])
                    
                    if "Strong Bullish" in instrument_data['overall_bias']:
                        st.success("""
                        **🎯 STRONG BULLISH SIGNAL**
                        - Consider LONG positions and CALL buying
                        - Look for dips to enter
                        - Support: Current price levels
                        """)
                    elif "Bullish" in instrument_data['overall_bias']:
                        st.info("""
                        **📈 BULLISH BIAS**
                        - Moderate LONG positions
                        - Wait for confirmations
                        - Use proper risk management
                        """)
                    elif "Strong Bearish" in instrument_data['overall_bias']:
                        st.error("""
                        **🎯 STRONG BEARISH SIGNAL**
                        - Consider SHORT positions and PUT buying
                        - Look for rallies to enter
                        - Resistance: Current price levels
                        """)
                    elif "Bearish" in instrument_data['overall_bias']:
                        st.warning("""
                        **📉 BEARISH BIAS**
                        - Moderate SHORT positions
                        - Wait for confirmations
                        - Use proper risk management
                        """)
                    else:
                        st.warning("""
                        **⚖️ NEUTRAL BIAS**
                        - Wait for clearer direction
                        - Consider range-bound strategies
                        - Monitor key levels
                        """)
        else:
            st.info("👆 Click 'Refresh Options Data' to load live options chain analysis")

    def display_technical_bias(self):
        st.header("🎯 Technical Bias Analysis")
        
        if st.button("🔄 Update Bias Analysis"):
            with st.spinner("Analyzing market bias..."):
                bias_data = self.bias_analyzer.analyze_all_bias_indicators("^NSEI")
                st.session_state.comprehensive_bias_data = bias_data
        
        if st.session_state.comprehensive_bias_data:
            bias_data = st.session_state.comprehensive_bias_data
            
            if bias_data['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    bias_color = "🟢" if bias_data['overall_bias'] == "BULLISH" else "🔴" if bias_data['overall_bias'] == "BEARISH" else "🟡"
                    st.metric("Overall Bias", f"{bias_color} {bias_data['overall_bias']}")
                with col2:
                    st.metric("Bias Score", f"{bias_data['overall_score']:.1f}")
                with col3:
                    st.metric("Current Price", f"₹{bias_data['current_price']:.2f}")
                with col4:
                    st.metric("Confidence", f"{(bias_data['bullish_count'] + bias_data['bearish_count']) / bias_data['total_indicators']:.0%}")
                
                # Display indicator results
                st.subheader("Technical Indicators")
                for indicator in bias_data['bias_results']:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{indicator['indicator']}**")
                    with col2:
                        st.write(indicator['value'])
                    with col3:
                        color = "green" if "BULLISH" in indicator['bias'] else "red" if "BEARISH" in indicator['bias'] else "orange"
                        st.write(f":{color}[{indicator['bias']}]")
            else:
                st.error(f"Analysis failed: {bias_data['error']}")
        else:
            st.info("Click 'Update Bias Analysis' to run technical analysis")

    def display_bias_tabulation(self):
        st.header("📋 Comprehensive Bias Tabulation")
        
        if not st.session_state.market_bias_data:
            st.info("No options data available. Please refresh options analysis first.")
            return
        
        for instrument_data in st.session_state.market_bias_data:
            with st.expander(f"📊 {instrument_data['instrument']} - Complete Bias Analysis", expanded=True):
                
                # ==================== BASIC INFORMATION ====================
                st.subheader("📈 Basic Information")
                basic_info_data = {
                    'Parameter': [
                        'Instrument', 'Spot Price', 'Overall Bias', 'Bias Score', 
                        'PCR OI', 'PCR Change', 'Timestamp'
                    ],
                    'Value': [
                        instrument_data['instrument'],
                        f"₹{instrument_data['spot_price']:.2f}",
                        instrument_data['overall_bias'],
                        f"{instrument_data['bias_score']:.2f}",
                        f"{instrument_data['pcr_oi']:.2f}",
                        f"{instrument_data['pcr_change']:.2f}",
                        instrument_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                basic_info_df = pd.DataFrame(basic_info_data)
                st.dataframe(basic_info_df, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # ==================== DETAILED ATM BIAS ====================
                if 'detailed_atm_bias' in instrument_data and instrument_data['detailed_atm_bias']:
                    st.subheader("🔍 Detailed ATM Bias Analysis")
                    detailed_bias = instrument_data['detailed_atm_bias']
                    
                    # Create comprehensive bias metrics table
                    bias_metrics_data = []
                    
                    # OI Analysis
                    if 'CE_OI' in detailed_bias and 'PE_OI' in detailed_bias:
                        oi_difference = detailed_bias['PE_OI'] - detailed_bias['CE_OI']
                        oi_ratio = detailed_bias['PE_OI'] / detailed_bias['CE_OI'] if detailed_bias['CE_OI'] > 0 else 0
                        bias_metrics_data.extend([
                            ['Call OI', f"{detailed_bias['CE_OI']:,.0f}", "-"],
                            ['Put OI', f"{detailed_bias['PE_OI']:,.0f}", "-"],
                            ['OI Difference', f"{oi_difference:+,.0f}", "Bullish" if oi_difference > 0 else "Bearish"],
                            ['OI Ratio', f"{oi_ratio:.2f}", "Bullish" if oi_ratio > 1 else "Bearish"]
                        ])
                    
                    # Change in OI Analysis
                    if 'CE_Change' in detailed_bias and 'PE_Change' in detailed_bias:
                        change_difference = detailed_bias['PE_Change'] - detailed_bias['CE_Change']
                        bias_metrics_data.extend([
                            ['Call OI Change', f"{detailed_bias['CE_Change']:+,.0f}", "-"],
                            ['Put OI Change', f"{detailed_bias['PE_Change']:+,.0f}", "-"],
                            ['OI Change Diff', f"{change_difference:+,.0f}", "Bullish" if change_difference > 0 else "Bearish"]
                        ])
                    
                    # Volume Analysis
                    if 'CE_Volume' in detailed_bias and 'PE_Volume' in detailed_bias:
                        volume_difference = detailed_bias['PE_Volume'] - detailed_bias['CE_Volume']
                        bias_metrics_data.extend([
                            ['Call Volume', f"{detailed_bias['CE_Volume']:,.0f}", "-"],
                            ['Put Volume', f"{detailed_bias['PE_Volume']:,.0f}", "-"],
                            ['Volume Diff', f"{volume_difference:+,.0f}", "Bullish" if volume_difference > 0 else "Bearish"]
                        ])
                    
                    # IV Analysis
                    if 'CE_IV' in detailed_bias and 'PE_IV' in detailed_bias:
                        iv_difference = detailed_bias['CE_IV'] - detailed_bias['PE_IV']
                        bias_metrics_data.extend([
                            ['Call IV', f"{detailed_bias['CE_IV']:.2f}%", "-"],
                            ['Put IV', f"{detailed_bias['PE_IV']:.2f}%", "-"],
                            ['IV Skew', f"{iv_difference:+.2f}%", "Bullish" if iv_difference > 0 else "Bearish"]
                        ])
                    
                    # Greeks Analysis
                    if 'CE_Delta' in detailed_bias:
                        bias_metrics_data.extend([
                            ['Call Delta', f"{detailed_bias['CE_Delta']:.4f}", "-"],
                            ['Put Delta', f"{detailed_bias['PE_Delta']:.4f}", "-"],
                            ['Call Gamma', f"{detailed_bias['CE_Gamma']:.4f}", "-"],
                            ['Put Gamma', f"{detailed_bias['PE_Gamma']:.4f}", "-"],
                            ['Call Vega', f"{detailed_bias['CE_Vega']:.4f}", "-"],
                            ['Put Vega', f"{detailed_bias['PE_Vega']:.4f}", "-"]
                        ])
                    
                    bias_metrics_df = pd.DataFrame(bias_metrics_data, columns=['Metric', 'Value', 'Bias'])
                    st.dataframe(bias_metrics_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                
                # ==================== COMPREHENSIVE METRICS ====================
                if 'comprehensive_metrics' in instrument_data and instrument_data['comprehensive_metrics']:
                    st.subheader("🎯 Advanced Option Metrics")
                    comp_metrics = instrument_data['comprehensive_metrics']
                    
                    comp_data = []
                    for key, value in comp_metrics.items():
                        if key not in ['timestamp']:
                            display_key = key.replace('_', ' ').title()
                            if isinstance(value, (int, float)):
                                if 'oi' in key.lower() or 'change' in key.lower():
                                    display_value = f"{value:,.0f}"
                                else:
                                    display_value = f"{value:.2f}"
                            else:
                                display_value = str(value)
                            comp_data.append([display_key, display_value])
                    
                    comp_df = pd.DataFrame(comp_data, columns=['Metric', 'Value'])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                
                # ==================== VISUAL ANALYSIS ====================
                st.subheader("📊 Visual Bias Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bias Score Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = instrument_data['bias_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{instrument_data['instrument']} Bias Score"},
                        gauge = {
                            'axis': {'range': [-5, 5]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-5, -2], 'color': "lightcoral"},
                                {'range': [-2, -1], 'color': "lightyellow"},
                                {'range': [-1, 1], 'color': "lightgray"},
                                {'range': [1, 2], 'color': "lightgreen"},
                                {'range': [2, 5], 'color': "limegreen"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': instrument_data['bias_score']}}
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # PCR Analysis Chart
                    pcr_data = {
                        'Type': ['Call OI', 'Put OI'],
                        'Value': [
                            instrument_data['comprehensive_metrics']['total_ce_oi'],
                            instrument_data['comprehensive_metrics']['total_pe_oi']
                        ]
                    }
                    pcr_df = pd.DataFrame(pcr_data)
                    
                    fig_bar = px.bar(
                        pcr_df, 
                        x='Type', 
                        y='Value', 
                        title="Call vs Put Open Interest",
                        color='Type',
                        color_discrete_map={'Call OI': 'red', 'Put OI': 'green'}
                    )
                    fig_bar.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # ==================== TRADING LEVELS ====================
                st.subheader("🎯 Key Trading Levels & Strategy")
                
                levels_col1, levels_col2 = st.columns(2)
                
                with levels_col1:
                    st.info("**Support & Resistance Levels**")
                    if 'detailed_atm_bias' in instrument_data:
                        detailed_bias = instrument_data['detailed_atm_bias']
                        if 'CE_LastPrice' in detailed_bias and 'PE_LastPrice' in detailed_bias:
                            atm_strike = instrument_data['comprehensive_metrics']['atm_strike']
                            spot_price = instrument_data['spot_price']
                            
                            st.write(f"**ATM Strike**: ₹{atm_strike:.2f}")
                            st.write(f"**Spot Price**: ₹{spot_price:.2f}")
                            st.write(f"**Call Price**: ₹{detailed_bias['CE_LastPrice']:.2f}")
                            st.write(f"**Put Price**: ₹{detailed_bias['PE_LastPrice']:.2f}")
                
                with levels_col2:
                    st.info("**Trading Strategy**")
                    bias = instrument_data['overall_bias']
                    bias_score = instrument_data['bias_score']
                    
                    if "Strong Bullish" in bias:
                        st.success("""
                        **Strategy**: Aggressive LONG
                        - Buy CE options
                        - Bull spreads
                        - Support: Current levels
                        """)
                    elif "Bullish" in bias:
                        st.info("""
                        **Strategy**: Moderate LONG  
                        - Wait for dips to buy
                        - Use defined risk strategies
                        """)
                    elif "Strong Bearish" in bias:
                        st.error("""
                        **Strategy**: Aggressive SHORT
                        - Buy PE options
                        - Bear spreads
                        - Resistance: Current levels
                        """)
                    elif "Bearish" in bias:
                        st.warning("""
                        **Strategy**: Moderate SHORT
                        - Wait for rallies to sell
                        - Use defined risk strategies
                        """)
                    else:
                        st.warning("""
                        **Strategy**: Wait & Watch
                        - Range-bound strategies
                        - Straddles/Strangles
                        - Wait for breakout
                        """)
                
                # ==================== RISK ANALYSIS ====================
                st.subheader("⚠️ Risk Analysis")
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    pcr_oi = instrument_data['pcr_oi']
                    if pcr_oi > 1.5 or pcr_oi < 0.5:
                        st.error("**PCR Risk**: HIGH")
                    elif pcr_oi > 1.2 or pcr_oi < 0.8:
                        st.warning("**PCR Risk**: MEDIUM")
                    else:
                        st.success("**PCR Risk**: LOW")
                
                with risk_col2:
                    bias_strength = abs(instrument_data['bias_score'])
                    if bias_strength >= 3:
                        st.success("**Bias Strength**: HIGH")
                    elif bias_strength >= 1:
                        st.info("**Bias Strength**: MEDIUM")
                    else:
                        st.warning("**Bias Strength**: LOW")
                
                with risk_col3:
                    if 'CE_IV' in detailed_bias and 'PE_IV' in detailed_bias:
                        avg_iv = (detailed_bias['CE_IV'] + detailed_bias['PE_IV']) / 2
                        if avg_iv > 25:
                            st.error("**IV Risk**: HIGH")
                        elif avg_iv > 18:
                            st.warning("**IV Risk**: MEDIUM")
                        else:
                            st.success("**IV Risk**: LOW")

    def display_market_data(self):
        st.header("🌍 Market Data")
        
        if st.button("🔄 Update Market Data"):
            with st.spinner("Fetching market data..."):
                vix_data = self.market_data_fetcher.fetch_india_vix()
                sector_data = self.market_data_fetcher.fetch_sector_indices()
                
                market_data = {
                    'india_vix': vix_data,
                    'sector_indices': sector_data
                }
                st.session_state.enhanced_market_data = market_data
        
        if st.session_state.enhanced_market_data:
            market_data = st.session_state.enhanced_market_data
            
            # VIX Data
            if market_data['india_vix']['success']:
                vix = market_data['india_vix']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("India VIX", f"{vix['value']:.2f}")
                with col2:
                    st.metric("Sentiment", vix['sentiment'])
                with col3:
                    st.metric("Bias", vix['bias'])
                with col4:
                    st.metric("Score", vix['score'])
            
            # Sector Data
            if market_data['sector_indices']:
                st.subheader("Sector Performance")
                sectors_df = pd.DataFrame(market_data['sector_indices'])
                st.dataframe(sectors_df, use_container_width=True)
        else:
            st.info("Click 'Update Market Data' to load market information")

    def run(self):
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Complete Options Chain Analysis & Bias Tabulation*")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price", "📊 Options", "🎯 Technical", "📋 Bias Tabulation", "🌍 Market"
        ])
        
        with tab1:
            self.display_price_analysis()
        
        with tab2:
            self.display_options_analysis()
        
        with tab3:
            self.display_technical_bias()
        
        with tab4:
            self.display_bias_tabulation()
        
        with tab5:
            self.display_market_data()

# =============================================
# RUN THE APP
# =============================================

if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
