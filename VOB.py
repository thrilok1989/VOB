import streamlit as st
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
