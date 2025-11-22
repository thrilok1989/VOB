import import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
import pytz
import time
import warnings
from dhanhq import dhanhq
import yfinance as yf
from typing import Dict, List, Optional, Any

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Pro Nifty Dashboard (Dhan API)",
    page_icon="🚀",
    layout="wide"
)

# =============================================
# HELPER: YAHOO SESSION (For Global Markets)
# =============================================

def get_yfinance_session():
    """Create a session with browser headers to avoid Yahoo blocking"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def fetch_with_retry(ticker_obj, period, interval='1d', retries=3, delay=2):
    """Fetch data with a delay if rate limited/empty."""
    for attempt in range(retries):
        try:
            df = ticker_obj.history(period=period, interval=interval)
            if not df.empty:
                return df
            time.sleep(delay)
        except Exception:
            time.sleep(delay)
    return pd.DataFrame()

# =============================================
# 1. DHAN API ADAPTER (OFFICIAL V2)
# =============================================

class DhanAdapter:
    """
    Wrapper for DhanHQ API v2 endpoints:
    - /charts/intraday (Historical)
    - /optionchain (Greeks, OI)
    - /marketfeed/ltp (Live Price)
    """
    def __init__(self, client_id, access_token):
        self.dhan = dhanhq(client_id, access_token)
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Dhan Security IDs (Verified from Dhan Scrip Master)
        self.SECURITY_IDS = {
            'NIFTY': '13',
            'BANKNIFTY': '25',
            'FINNIFTY': '27'
        }

    def get_intraday_data(self, symbol: str, days: int = 5, interval_min: int = 15) -> pd.DataFrame:
        """Fetch Intraday Minute Charts"""
        try:
            security_id = self.SECURITY_IDS.get(symbol, '13')
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=days)
            
            # API Call: /v2/charts/intraday
            response = self.dhan.intraday_daily_minute_charts(
                security_id=security_id,
                exchange_segment='IDX_I',
                instrument_type='INDEX',
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d'),
                interval=str(interval_min)
            )
            
            if response.get('status') == 'success':
                data = response['data']
                if not data.get('start_Time'): return pd.DataFrame()
                    
                df = pd.DataFrame({
                    'Open': data['open'],
                    'High': data['high'],
                    'Low': data['low'],
                    'Close': data['close'],
                    'Volume': data['volume'], 
                    'timestamp': data['start_Time']
                })
                
                # Convert Timestamp
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s' if data['start_Time'][0] < 10000000000 else 'ms')
                df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(self.ist)
                df.set_index('datetime', inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Dhan Intraday Error: {e}")
            return pd.DataFrame()

    def get_option_chain(self, symbol: str, expiry_date: str = None) -> Dict:
        """Fetch Real-time Option Chain with Greeks"""
        try:
            security_id = self.SECURITY_IDS.get(symbol, '13')
            
            # 1. If expiry not provided, fetch list and pick earliest
            if not expiry_date:
                exp_list = self.dhan.option_chain_expiry_list(
                    security_id=security_id,
                    exchange_segment='IDX_I'
                )
                if exp_list.get('status') == 'success' and exp_list['data']:
                    # Filter for future dates
                    today = datetime.now().date()
                    valid_exopiries = [d for d in exp_list['data'] if datetime.strptime(d, "%Y-%m-%d").date() >= today]
                    if valid_exopiries:
                        expiry_date = valid_exopiries[0]
            
            if not expiry_date:
                return {'success': False, 'error': "Could not determine expiry"}

            # 2. Fetch Option Chain
            response = self.dhan.option_chain(
                security_id=security_id,
                exchange_segment='IDX_I',
                expiry_date=expiry_date
            )
            
            if response.get('status') == 'success':
                return {
                    'success': True,
                    'data': response['data'],
                    'expiry': expiry_date,
                    'spot': response['data'].get('last_price', 0)
                }
            return {'success': False, 'error': response.get('remarks', 'API Error')}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# =============================================
# 2. MARKET DATA FETCHER (Hybrid)
# =============================================

class EnhancedMarketData:
    """Fetches Global Data via Yahoo, Indian Data via Dhan"""
    def __init__(self):
        self.session = get_yfinance_session()

    def fetch_global_markets(self) -> pd.DataFrame:
        """Yahoo Finance for Global Indices"""
        tickers = {
            '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^N225': 'Nikkei 225', 
            '^FTSE': 'FTSE 100', '^GDAXI': 'DAX', 'DX-Y.NYB': 'DXY'
        }
        data = []
        for sym, name in tickers.items():
            try:
                t = yf.Ticker(sym, session=self.session)
                hist = fetch_with_retry(t, period='2d')
                if not hist.empty:
                    change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0])/hist['Close'].iloc[0])*100
                    data.append({'Market': name, 'Price': hist['Close'].iloc[-1], 'Change %': change})
            except: pass
        return pd.DataFrame(data)

# =============================================
# 3. ANALYSIS ENGINES
# =============================================

class BiasAnalyzer:
    """Technical Bias using Dhan Intraday Data"""
    def __init__(self, adapter: DhanAdapter):
        self.adapter = adapter

    def analyze(self, symbol: str):
        df = self.adapter.get_intraday_data(symbol, days=5, interval_min=15)
        if df.empty: return None

        # Indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        
        last = df.iloc[-1]
        
        # Scoring
        score = 0
        if last['RSI'] > 55: score += 1
        elif last['RSI'] < 45: score -= 1
        
        if last['EMA9'] > last['EMA21']: score += 1
        else: score -= 1
        
        if last['Close'] > last['Open']: score += 1 # Price Action
        
        bias = "NEUTRAL"
        if score >= 2: bias = "BULLISH"
        elif score <= -2: bias = "BEARISH"
        
        return {
            'bias': bias,
            'score': score,
            'rsi': last['RSI'],
            'price': last['Close'],
            'df': df
        }

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class OptionChainAnalyzer:
    """Analyzes Dhan Option Chain JSON"""
    
    def process(self, chain_response: Dict):
        if not chain_response.get('success'): return None
        
        raw_data = chain_response['data'].get('oc', {})
        spot = chain_response['data'].get('last_price', 0)
        
        # Convert nested JSON to DataFrame
        strikes_data = []
        for strike_price, details in raw_data.items():
            strike = float(strike_price)
            ce = details.get('ce', {})
            pe = details.get('pe', {})
            
            # Safe get for Greeks (handle None)
            ce_greeks = ce.get('greeks') or {}
            pe_greeks = pe.get('greeks') or {}
            
            strikes_data.append({
                'Strike': strike,
                'CE_OI': ce.get('oi', 0),
                'PE_OI': pe.get('oi', 0),
                'CE_LTP': ce.get('last_price', 0),
                'PE_LTP': pe.get('last_price', 0),
                'CE_IV': ce.get('implied_volatility', 0),
                'PE_IV': pe.get('implied_volatility', 0),
                'CE_Delta': ce_greeks.get('delta', 0),
                'PE_Delta': pe_greeks.get('delta', 0),
                'CE_Gamma': ce_greeks.get('gamma', 0),
                'PE_Gamma': pe_greeks.get('gamma', 0),
                'CE_Vega': ce_greeks.get('vega', 0),
                'PE_Vega': pe_greeks.get('vega', 0)
            })
            
        df = pd.DataFrame(strikes_data)
        
        # Analysis
        total_ce_oi = df['CE_OI'].sum()
        total_pe_oi = df['PE_OI'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Max Pain
        df['Total_Pain'] = df.apply(lambda row: self._calculate_pain(row, spot), axis=1)
        max_pain = df.loc[df['Total_Pain'].idxmin()]['Strike'] if not df.empty else spot
        
        return {
            'pcr': pcr,
            'max_pain': max_pain,
            'spot': spot,
            'expiry': chain_response['expiry'],
            'df': df,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi
        }

    def _calculate_pain(self, row, spot):
        strike = row['Strike']
        ce_pain = max(0, spot - strike) * row['CE_OI']
        pe_pain = max(0, strike - spot) * row['PE_OI']
        return ce_pain + pe_pain

# =============================================
# 4. VOLUME ORDER BLOCKS
# =============================================
class VolumeOrderBlocks:
    def detect(self, df: pd.DataFrame):
        if df.empty: return [], []
        
        # Logic: High Volume Candles + Price Rejection
        avg_vol = df['Volume'].rolling(20).mean()
        high_vol_candles = df[df['Volume'] > avg_vol * 1.5]
        
        bullish, bearish = [], []
        for idx, row in high_vol_candles.iterrows():
            body = abs(row['Close'] - row['Open'])
            range_len = row['High'] - row['Low']
            
            # Bullish: Green candle or Hammer
            if row['Close'] > row['Open']:
                bullish.append({'time': idx, 'price': row['Low'], 'vol': row['Volume']})
            # Bearish: Red candle or Shooting Star
            else:
                bearish.append({'time': idx, 'price': row['High'], 'vol': row['Volume']})
                
        return bullish[-3:], bearish[-3:] # Return last 3

# =============================================
# 5. MAIN APPLICATION
# =============================================

def main():
    st.title("🚀 Pro Nifty Dashboard (Powered by DhanHQ)")
    
    # 1. Credentials
    try:
        CLIENT_ID = st.secrets["dhan"]["client_id"]
        ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
        adapter = DhanAdapter(CLIENT_ID, ACCESS_TOKEN)
        market_data = EnhancedMarketData()
        bias_engine = BiasAnalyzer(adapter)
        chain_engine = OptionChainAnalyzer()
        vob_engine = VolumeOrderBlocks()
        
        st.sidebar.success("✅ Dhan API Connected")
    except Exception as e:
        st.error("❌ Dhan Credentials Missing in secrets.toml")
        st.stop()

    # 2. Controls
    symbol = st.sidebar.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    if st.sidebar.button("🔄 Refresh All"):
        st.rerun()

    # 3. Main Dashboard Layout
    tab1, tab2, tab3 = st.tabs(["📈 Technicals & Bias", "📊 Pro Option Chain", "🌍 Global Market"])

    # --- TAB 1: TECHNICALS ---
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        # A. Bias Analysis
        analysis = bias_engine.analyze(symbol)
        
        if analysis:
            with col2:
                st.subheader("Market Bias")
                bias_color = "green" if analysis['bias'] == "BULLISH" else "red" if analysis['bias'] == "BEARISH" else "gray"
                st.markdown(f"## :{bias_color}[{analysis['bias']}]")
                st.metric("Score", f"{analysis['score']}/3")
                st.metric("RSI (15m)", f"{analysis['rsi']:.2f}")
                st.metric("LTP", f"{analysis['price']:.2f}")
            
            with col1:
                # B. Chart with VOB
                df = analysis['df']
                bull, bear = vob_engine.detect(df)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    name=symbol
                ), row=1, col=1)
                
                # Volume
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
                
                # VOB Lines
                for b in bull:
                    fig.add_hline(y=b['price'], line_color="green", line_dash="dash", annotation_text="Bull Block", row=1, col=1)
                for b in bear:
                    fig.add_hline(y=b['price'], line_color="red", line_dash="dash", annotation_text="Bear Block", row=1, col=1)
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not fetch Technical Data from Dhan.")

    # --- TAB 2: OPTION CHAIN ---
    with tab2:
        st.subheader(f"{symbol} Option Chain (Official Dhan Data)")
        
        with st.spinner("Fetching Real-time Greeks & OI..."):
            chain_res = adapter.get_option_chain(symbol)
            oc_analysis = chain_engine.process(chain_res)
            
        if oc_analysis:
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spot Price", f"{oc_analysis['spot']}")
            m2.metric("Max Pain", f"{oc_analysis['max_pain']}")
            m3.metric("PCR (OI)", f"{oc_analysis['pcr']:.2f}")
            m4.metric("Expiry", oc_analysis['expiry'])
            
            # Data Table
            df_chain = oc_analysis['df']
            
            # Filter near ATM for display (Spot +/- 2%)
            spot = oc_analysis['spot']
            df_display = df_chain[
                (df_chain['Strike'] > spot * 0.98) & 
                (df_chain['Strike'] < spot * 1.02)
            ].sort_values('Strike')
            
            # Formatting for Display
            st.dataframe(
                df_display.style.background_gradient(subset=['CE_OI', 'PE_OI'], cmap="Blues"),
                use_container_width=True,
                column_config={
                    "Strike": st.column_config.NumberColumn(format="%.0f"),
                    "CE_OI": st.column_config.NumberColumn(format="%d"),
                    "PE_OI": st.column_config.NumberColumn(format="%d"),
                }
            )
            
            # PCR Gauge
            fig_pcr = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = oc_analysis['pcr'],
                title = {'text': "PCR Sentiment"},
                gauge = {
                    'axis': {'range': [0.5, 1.5]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0.5, 0.8], 'color': "red"},
                        {'range': [0.8, 1.2], 'color': "gray"},
                        {'range': [1.2, 1.5], 'color': "green"}
                    ]
                }
            ))
            fig_pcr.update_layout(height=300)
            st.plotly_chart(fig_pcr)
            
        else:
            st.warning("Option Chain data unavailable. Market might be closed or API limit reached.")

    # --- TAB 3: GLOBAL MARKETS ---
    with tab3:
        st.subheader("🌍 Global Indices (Yahoo Finance)")
        if st.button("Refresh Global Data"):
            df_global = market_data.fetch_global_markets()
            if not df_global.empty:
                st.dataframe(
                    df_global.style.applymap(
                        lambda x: 'color: green' if x > 0 else 'color: red', 
                        subset=['Change %']
                    ),
                    use_container_width=True
                )
            else:
                st.warning("Could not fetch global data.")

if __name__ == "__main__":
    main()
 as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
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
from dhanhq import dhanhq

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard (Dhan+)",
    page_icon="📈",
    layout="wide"
)

# =============================================
# HELPER: RETRY LOGIC & SESSION
# =============================================

def get_yfinance_session():
    """Create a session with browser headers to avoid Yahoo blocking"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def fetch_with_retry(ticker_obj, period, interval='1d', retries=3, delay=2):
    """Fetch data with a delay if rate limited/empty."""
    for attempt in range(retries):
        try:
            df = ticker_obj.history(period=period, interval=interval)
            if not df.empty:
                return df
            time.sleep(delay)
        except Exception:
            time.sleep(delay)
    return pd.DataFrame()

# =============================================
# 1. DHAN API ADAPTER
# =============================================

class DhanMarketAdapter:
    """Wrapper for official DhanHQ API calls"""
    def __init__(self, client_id, access_token):
        self.dhan = dhanhq(client_id, access_token)
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Dhan Security IDs (Map for common indices)
        self.SECURITY_IDS = {
            'NIFTY': '13',
            '^NSEI': '13',
            'BANKNIFTY': '25',
            '^NSEBANK': '25',
            'FINNIFTY': '27'
        }

    def get_intraday_data(self, symbol: str, days: int = 5, interval_min: int = 5) -> pd.DataFrame:
        """Fetch Intraday History for Technical Bias"""
        try:
            security_id = self.SECURITY_IDS.get(symbol, '13') # Default to Nifty
            
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=days)
            
            # API Call
            response = self.dhan.intraday_daily_minute_charts(
                security_id=security_id,
                exchange_segment='IDX_I',
                instrument_type='INDEX',
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d'),
                interval=str(interval_min)
            )
            
            if response.get('status') == 'success':
                data = response['data']
                if not data.get('start_Time'):
                    return pd.DataFrame()
                    
                df = pd.DataFrame({
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume'], # Dhan gives real volume/OI for indices
                    'timestamp': data['start_Time']
                })
                
                # Convert Dhan timestamp (usually epoch in nanoseconds or seconds)
                # Dhan typically returns a custom timestamp, handling standard conversion:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s' if data['start_Time'][0] < 10000000000 else 'ms')
                df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(self.ist)
                
                # Rename columns to Title Case for compatibility with existing code
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df.set_index('datetime', inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Dhan API Error: {e}")
            return pd.DataFrame()

# =============================================
# 2. ENHANCED MARKET DATA FETCHER (HYBRID)
# =============================================

class EnhancedMarketData:
    """
    Hybrid Fetcher: 
    - Dhan for India VIX/Sectors (if configured) or robust Yahoo Fallback
    - Yahoo for Global/Intermarket
    """

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.session = get_yfinance_session()

    def get_current_time_ist(self):
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX (Yahoo Fallback with Retry)"""
        try:
            ticker = yf.Ticker("^INDIAVIX", session=self.session)
            hist = fetch_with_retry(ticker, period="5d", interval="1d")

            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]
                if vix_value > 25: sentiment, bias, score = "HIGH FEAR", "BEARISH", -75
                elif vix_value > 20: sentiment, bias, score = "ELEVATED FEAR", "BEARISH", -50
                elif vix_value > 15: sentiment, bias, score = "MODERATE", "NEUTRAL", 0
                elif vix_value > 12: sentiment, bias, score = "LOW VOLATILITY", "BULLISH", 40
                else: sentiment, bias, score = "COMPLACENCY", "NEUTRAL", 0

                return {'success': True, 'value': vix_value, 'sentiment': sentiment, 'bias': bias, 'score': score, 'timestamp': self.get_current_time_ist(), 'source': 'Yahoo'}
        except Exception:
            pass
        return {'success': False, 'error': 'VIX data not available'}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices"""
        sectors_map = {
            '^CNXIT': 'NIFTY IT', '^CNXAUTO': 'NIFTY AUTO', '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL', '^NSEBANK': 'NIFTY BANK'
        }
        sector_data = []
        progress = st.progress(0)
        
        for idx, (symbol, name) in enumerate(sectors_map.items()):
            try:
                progress.progress((idx+1)/len(sectors_map), f"Fetching {name}...")
                ticker = yf.Ticker(symbol, session=self.session)
                hist = fetch_with_retry(ticker, period="2d", interval="1h") # 1h is safer than 1m

                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[0]
                    change_pct = ((last_price - prev_close) / prev_close) * 100
                    
                    if change_pct > 1.0: bias, score = "BULLISH", 50
                    elif change_pct < -1.0: bias, score = "BEARISH", -50
                    else: bias, score = "NEUTRAL", 0

                    sector_data.append({'sector': name, 'last_price': last_price, 'change_pct': change_pct, 'bias': bias, 'score': score})
            except:
                continue
        progress.empty()
        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        global_markets = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^N225': 'NIKKEI', '^GDAXI': 'DAX'}
        market_data = []
        for symbol, name in global_markets.items():
            try:
                ticker = yf.Ticker(symbol, session=self.session)
                hist = fetch_with_retry(ticker, period="2d", interval="1d")
                if not hist.empty:
                    pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0])/hist['Close'].iloc[0])*100
                    bias = "BULLISH" if pct > 0 else "BEARISH"
                    market_data.append({'market': name, 'last_price': hist['Close'].iloc[-1], 'change_pct': pct, 'bias': bias, 'score': 0})
            except: pass
        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        assets = {'DX-Y.NYB': 'DXY', 'CL=F': 'CRUDE', 'GC=F': 'GOLD'}
        data = []
        for symbol, name in assets.items():
            try:
                ticker = yf.Ticker(symbol, session=self.session)
                hist = fetch_with_retry(ticker, period="2d", interval="1d")
                if not hist.empty:
                    pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0])/hist['Close'].iloc[0])*100
                    data.append({'asset': name, 'last_price': hist['Close'].iloc[-1], 'change_pct': pct, 'bias': "NEUTRAL", 'score': 0})
            except: pass
        return data

    def analyze_sector_rotation(self) -> Dict:
        sectors = self.fetch_sector_indices()
        if not sectors: return {'success': False, 'sector_breadth': 50, 'leaders': [], 'laggards': []}
        sorted_s = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)
        return {'success': True, 'leaders': sorted_s[:2], 'laggards': sorted_s[-2:], 'sector_breadth': 50, 'sector_sentiment': 'Mixed', 'rotation_pattern': 'Analyzed'}

    def analyze_intraday_seasonality(self) -> Dict:
        now = self.get_current_time_ist()
        return {'success': True, 'current_time': now.strftime("%H:%M:%S"), 'session': "Live", 'session_bias': "Neutral", 'trading_recommendation': "Follow Trend"}

    def fetch_all_enhanced_data(self) -> Dict:
        return {
            'timestamp': self.get_current_time_ist(),
            'india_vix': self.fetch_india_vix(),
            'sector_indices': self.fetch_sector_indices(),
            'global_markets': self.fetch_global_markets(),
            'intermarket': self.fetch_intermarket_data(),
            'sector_rotation': self.analyze_sector_rotation(),
            'intraday_seasonality': self.analyze_intraday_seasonality(),
            'summary': {'overall_sentiment': 'NEUTRAL', 'avg_score': 0}
        }

# =============================================
# 3. BIAS ANALYSIS PRO (DHAN POWERED)
# =============================================

class BiasAnalysisPro:
    """Uses Dhan API for accurate Indian Index Data"""
    
    def __init__(self, dhan_adapter: Optional[DhanMarketAdapter]):
        self.dhan = dhan_adapter
        self.config = self._default_config()

    def _default_config(self) -> Dict:
        return {
            'rsi_period': 14, 'mfi_period': 10, 'dmi_period': 13,
            'bias_strength': 60, 'stocks': {}
        }

    def fetch_data(self, symbol: str, period: str = '5d', interval: str = '15m') -> pd.DataFrame:
        """Fetch data using Dhan for Nifty/BankNifty, YFinance for others"""
        
        # 1. Try Dhan API for Indian Indices
        if self.dhan and symbol in ['NIFTY', '^NSEI', 'BANKNIFTY', '^NSEBANK', 'FINNIFTY']:
            interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60}
            dhan_interval = interval_map.get(interval, 15)
            
            print(f"⚡ Fetching {symbol} from Dhan API...")
            df = self.dhan.get_intraday_data(symbol, days=5, interval_min=dhan_interval)
            
            if not df.empty:
                return df
                
        # 2. Fallback to Yahoo Finance
        print(f"⚠️ Falling back to YFinance for {symbol}")
        session = get_yfinance_session()
        ticker = yf.Ticker(symbol, session=session)
        df = fetch_with_retry(ticker, period=period, interval=interval)
        
        if df.empty: return pd.DataFrame()
        
        if 'Volume' not in df.columns: df['Volume'] = 0
        df['Volume'] = df['Volume'].fillna(0)
        
        # Synthetic Volume for Yahoo Indices (often 0)
        if df['Volume'].sum() == 0:
            df['Volume'] = ((df['High'] - df['Low']) / df['Open'] * 1000000).abs().astype(int)
            
        return df

    # --- Technical Indicators ---
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi_ratio = pos_flow / neg_flow.replace(0, 0.001)
        return (100 - (100 / (1 + mfi_ratio))).fillna(50)

    def calculate_dmi(self, df: pd.DataFrame, period=14):
        high, low = df['High'], df['Low']
        tr = pd.concat([high-low, abs(high-df['Close'].shift()), abs(low-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up, down = high.diff(), -low.diff()
        plus_di = 100 * (up.where((up > down) & (up > 0), 0).rolling(period).mean() / atr)
        minus_di = 100 * (down.where((down > up) & (down > 0), 0).rolling(period).mean() / atr)
        return plus_di, minus_di

    def calculate_volume_delta(self, df: pd.DataFrame):
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
        return up_vol - down_vol, up_vol > down_vol, down_vol > up_vol

    def calculate_ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def analyze_all_bias_indicators(self, symbol: str = "NIFTY") -> Dict:
        """Run complete analysis"""
        df = self.fetch_data(symbol, interval='15m') # Dhan prefers 15m for bias
        
        if df.empty or len(df) < 50:
            return {'success': False, 'error': f'Insufficient data for {symbol}'}

        results = []
        
        # 1. RSI
        rsi = self.calculate_rsi(df['Close'])
        rsi_val = rsi.iloc[-1]
        results.append({'indicator': 'RSI', 'value': f"{rsi_val:.1f}", 'bias': 'BULLISH' if rsi_val > 50 else 'BEARISH', 'score': 100 if rsi_val > 50 else -100, 'category': 'fast'})

        # 2. MFI
        mfi = self.calculate_mfi(df)
        mfi_val = mfi.iloc[-1]
        results.append({'indicator': 'MFI', 'value': f"{mfi_val:.1f}", 'bias': 'BULLISH' if mfi_val > 50 else 'BEARISH', 'score': 100 if mfi_val > 50 else -100, 'category': 'fast'})

        # 3. DMI
        p_di, m_di = self.calculate_dmi(df)
        results.append({'indicator': 'DMI', 'value': f"DI+ {p_di.iloc[-1]:.1f}", 'bias': 'BULLISH' if p_di.iloc[-1] > m_di.iloc[-1] else 'BEARISH', 'score': 100 if p_di.iloc[-1] > m_di.iloc[-1] else -100, 'category': 'fast'})

        # 4. EMA Cross
        ema5 = self.calculate_ema(df['Close'], 5)
        ema20 = self.calculate_ema(df['Close'], 20)
        results.append({'indicator': 'EMA Trend', 'value': '5 vs 20', 'bias': 'BULLISH' if ema5.iloc[-1] > ema20.iloc[-1] else 'BEARISH', 'score': 100 if ema5.iloc[-1] > ema20.iloc[-1] else -100, 'category': 'medium'})

        # 5. Volume Delta
        delta, v_bull, v_bear = self.calculate_volume_delta(df.tail(20))
        results.append({'indicator': 'Vol Delta', 'value': f"{delta}", 'bias': 'BULLISH' if v_bull else 'BEARISH', 'score': 100 if v_bull else -100, 'category': 'fast'})

        # Aggregate
        bull_count = sum(1 for r in results if r['bias'] == 'BULLISH')
        total_score = sum(r['score'] for r in results) / len(results)
        
        return {
            'success': True, 'symbol': symbol, 'current_price': df['Close'].iloc[-1],
            'bias_results': results, 'overall_bias': "BULLISH" if total_score > 0 else "BEARISH",
            'overall_score': total_score, 'overall_confidence': abs(total_score),
            'bullish_count': bull_count, 'bearish_count': len(results) - bull_count, 'neutral_count': 0, 'total_indicators': len(results)
        }

# =============================================
# 4. NSE OPTIONS ANALYZER (SCRAPER WITH HEADERS)
# =============================================

class NSEOptionsAnalyzer:
    """Uses requests with browser headers to scrape NSE (Robust Fallback)"""
    def __init__(self):
        self.refresh_interval = 2
        
    def fetch_option_chain_data(self, instrument: str) -> Dict:
        """Fetch from NSE website with robust headers"""
        try:
            symbol = instrument.upper().replace(' ', '%20')
            if 'NIFTY' in symbol: url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            else: url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            session = requests.Session()
            session.headers.update(headers)
            # Visit homepage first to set cookies
            session.get("https://www.nseindia.com", timeout=5)
            response = session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_nse_json(data, instrument)
            return {'success': False, 'error': f"Status {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _process_nse_json(self, data, instrument):
        """Process NSE JSON format"""
        try:
            records = data['records']['data']
            expiry = data['records']['expiryDates'][0]
            spot = data['records']['underlyingValue']
            
            ce_oi = sum(r['CE']['openInterest'] for r in records if 'CE' in r)
            pe_oi = sum(r['PE']['openInterest'] for r in records if 'PE' in r)
            
            return {
                'success': True, 'instrument': instrument, 'spot': spot, 'expiry': expiry,
                'total_ce_oi': ce_oi, 'total_pe_oi': pe_oi, 'records': records
            }
        except:
            return {'success': False, 'error': "Parse Error"}

    def get_overall_market_bias(self) -> List[Dict]:
        """Simple wrapper to get NIFTY analysis"""
        data = self.fetch_option_chain_data("NIFTY")
        if data['success']:
            pcr = data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0
            bias = "Bullish" if pcr > 1 else "Bearish"
            return [{
                'instrument': 'NIFTY', 'spot_price': data['spot'], 'overall_bias': bias,
                'bias_score': (pcr-1)*10, 'pcr_oi': pcr, 'pcr_change': 0,
                'atm_strike': round(data['spot']/50)*50
            }]
        return []

# =============================================
# 5. UTILS: VOLUME BLOCK & SPIKE
# =============================================

class VolumeOrderBlocks:
    """Identifies VOBs"""
    def __init__(self, sensitivity=5):
        self.sensitivity = sensitivity

    def detect_volume_order_blocks(self, df: pd.DataFrame):
        if df.empty: return [], []
        # Simplified Logic for Demo
        bullish, bearish = [], []
        
        # Look for candles with 2x avg volume
        avg_vol = df['Volume'].rolling(20).mean()
        spikes = df[df['Volume'] > avg_vol * 1.5]
        
        for idx, row in spikes.iterrows():
            if row['Close'] > row['Open']: # Bullish Spike
                bullish.append({'index': idx, 'upper': row['High'], 'lower': row['Low'], 'mid': (row['High']+row['Low'])/2, 'volume': row['Volume']})
            else:
                bearish.append({'index': idx, 'upper': row['High'], 'lower': row['Low'], 'mid': (row['High']+row['Low'])/2, 'volume': row['Volume']})
        return bullish[-5:], bearish[-5:]

class VolumeSpikeDetector:
    def __init__(self, threshold=2.5):
        self.threshold = threshold

    def detect_volume_spike(self, current_vol, timestamp):
        # Placeholder
        return False, 0.0

# =============================================
# 6. MAIN APP LOGIC
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        # 1. Setup Dhan Adapter
        try:
            self.dhan_client = st.secrets["dhan"]["client_id"]
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_adapter = DhanMarketAdapter(self.dhan_client, self.dhan_token)
            self.has_dhan = True
        except:
            st.warning("⚠️ Dhan credentials not found in `.streamlit/secrets.toml`. Using Yahoo fallback.")
            self.dhan_adapter = None
            self.has_dhan = False

        self.market_data = EnhancedMarketData()
        self.bias_analyzer = BiasAnalysisPro(self.dhan_adapter)
        self.options_analyzer = NSEOptionsAnalyzer()
        self.vob = VolumeOrderBlocks()

        if 'market_bias_data' not in st.session_state: st.session_state.market_bias_data = None
        if 'enhanced_market_data' not in st.session_state: st.session_state.enhanced_market_data = None

    def create_chart(self, df, bullish_blocks, bearish_blocks):
        if df.empty: return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
        
        for b in bullish_blocks:
            fig.add_hrect(y0=b['lower'], y1=b['upper'], fillcolor="green", opacity=0.1, row=1, col=1)
        for b in bearish_blocks:
            fig.add_hrect(y0=b['lower'], y1=b['upper'], fillcolor="red", opacity=0.1, row=1, col=1)
            
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        return fig

    def run(self):
        st.title("🚀 Enhanced Nifty Dashboard (Dhan Powered)")
        
        # Sidebar
        with st.sidebar:
            st.header("Settings")
            if self.has_dhan: st.success("✅ Dhan API Connected")
            else: st.error("❌ Dhan API Not Connected")
            
            if st.button("🔄 Refresh Data"):
                st.session_state.enhanced_market_data = None
                st.rerun()

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart & Bias", "📊 Options Chain", "🌍 Market Data", "📋 Signals"])

        # Tab 1: Chart & Bias
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Use Dhan for NIFTY chart
                df = self.bias_analyzer.fetch_data('NIFTY', interval='15m')
                if not df.empty:
                    bull, bear = self.vob.detect_volume_order_blocks(df)
                    st.plotly_chart(self.create_chart(df, bull, bear), use_container_width=True)
                else:
                    st.error("No Data. Check Dhan credentials.")
            
            with col2:
                if st.button("Analyze Technical Bias", type="primary"):
                    res = self.bias_analyzer.analyze_all_bias_indicators('NIFTY')
                    if res['success']:
                        st.metric("Overall Bias", res['overall_bias'], delta=f"{res['overall_score']:.1f}")
                        st.dataframe(pd.DataFrame(res['bias_results'])[['indicator', 'bias', 'score']])
                    else:
                        st.error(res['error'])

        # Tab 2: Options
        with tab2:
            st.subheader("NSE Option Chain Analysis")
            if st.button("Fetch Options Data"):
                with st.spinner("Scraping NSE..."):
                    data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = data
            
            if st.session_state.market_bias_data:
                for item in st.session_state.market_bias_data:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Instrument", item['instrument'])
                    c2.metric("Spot", item['spot_price'])
                    c3.metric("PCR", f"{item['pcr_oi']:.2f}", delta=item['overall_bias'])

        # Tab 3: Market Data
        with tab3:
            st.subheader("Global & Enhanced Market Data")
            if st.button("Load Market Data"):
                with st.spinner("Fetching (Yahoo with Retry)..."):
                    data = self.market_data.fetch_all_enhanced_data()
                    st.session_state.enhanced_market_data = data
            
            if st.session_state.enhanced_market_data:
                data = st.session_state.enhanced_market_data
                
                # India VIX
                vix = data['india_vix']
                if vix['success']:
                    st.metric("India VIX", f"{vix['value']:.2f}", f"{vix['bias']} ({vix['sentiment']})")
                
                # Global
                st.write("### Global Markets")
                gdf = pd.DataFrame(data['global_markets'])
                if not gdf.empty: st.dataframe(gdf, use_container_width=True)
                
                # Sectors
                st.write("### Sector Performance")
                sdf = pd.DataFrame(data['sector_indices'])
                if not sdf.empty: 
                    st.dataframe(sdf.sort_values('change_pct', ascending=False), use_container_width=True)

        # Tab 4: Signals (Placeholder)
        with tab4:
            st.info("Automated signals will appear here based on VOB and PCR analysis.")

if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
