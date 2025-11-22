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
import numpy as np

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Pro Nifty Dashboard (Dhan API)",
    page_icon="🚀",
    layout="wide"
)

# =============================================
# HELPER: NETWORK UTILS
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
                # Dhan returns epoch seconds usually, sometimes ms. Auto-detect:
                is_ms = data['start_Time'][0] > 10000000000
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms' if is_ms else 's')
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
            
            # 1. If expiry not provided, fetch list and pick earliest active one
            if not expiry_date:
                exp_list = self.dhan.option_chain_expiry_list(
                    security_id=security_id,
                    exchange_segment='IDX_I'
                )
                if exp_list.get('status') == 'success' and exp_list['data']:
                    # Filter for future dates
                    today = datetime.now().date()
                    valid_expiries = []
                    for d in exp_list['data']:
                        try:
                            date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                            if date_obj >= today:
                                valid_expiries.append(d)
                        except: continue
                    
                    valid_expiries.sort()
                    if valid_expiries:
                        expiry_date = valid_expiries[0]
            
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
        # Fetch 15m data for reliable trend
        df = self.adapter.get_intraday_data(symbol, days=5, interval_min=15)
        if df.empty: return None

        # --- Calculate Indicators ---
        
        # 1. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. EMAs
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # 3. VWAP (Approximate for intraday)
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['Typical'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        last = df.iloc[-1]
        
        # --- Bias Scoring System ---
        score = 0
        reasons = []
        
        # RSI Check
        if last['RSI'] > 55: 
            score += 1
            reasons.append("RSI Bullish (>55)")
        elif last['RSI'] < 45: 
            score -= 1
            reasons.append("RSI Bearish (<45)")
        
        # Trend Check (EMA Crossover)
        if last['EMA9'] > last['EMA21']: 
            score += 1
            reasons.append("EMA Trend Bullish")
        else: 
            score -= 1
            reasons.append("EMA Trend Bearish")
        
        # Price vs VWAP
        if last['Close'] > last['VWAP']: 
            score += 1
            reasons.append("Price > VWAP")
        else: 
            score -= 1
            reasons.append("Price < VWAP")
            
        # Price Action (Close > Open)
        if last['Close'] > last['Open']: 
            score += 0.5
        else: 
            score -= 0.5
        
        # Final Verdict
        bias = "NEUTRAL"
        if score >= 2: bias = "BULLISH"
        elif score <= -2: bias = "BEARISH"
        
        return {
            'bias': bias,
            'score': score,
            'rsi': last['RSI'],
            'price': last['Close'],
            'ema_bull': last['EMA9'] > last['EMA21'],
            'df': df,
            'reasons': reasons
        }

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
                'CE_Gamma': ce_greeks.get('gamma', 0),
                'PE_Gamma': pe_greeks.get('gamma', 0),
            })
            
        df = pd.DataFrame(strikes_data)
        
        if df.empty: return None

        # Analysis
        total_ce_oi = df['CE_OI'].sum()
        total_pe_oi = df['PE_OI'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Max Pain Calculation
        df['Total_Pain'] = df.apply(lambda row: self._calculate_pain(row, spot), axis=1)
        max_pain = df.loc[df['Total_Pain'].idxmin()]['Strike'] if not df.empty else spot
        
        # Synthetic Future (ATM)
        atm_strike = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]['Strike'].values[0]
        atm_row = df[df['Strike'] == atm_strike].iloc[0]
        synthetic_fut = atm_strike + atm_row['CE_LTP'] - atm_row['PE_LTP']
        
        return {
            'pcr': pcr,
            'max_pain': max_pain,
            'spot': spot,
            'synthetic_fut': synthetic_fut,
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
        
        # Logic: Candles with 1.5x average volume imply institutional interest
        avg_vol = df['Volume'].rolling(20).mean()
        high_vol_candles = df[df['Volume'] > avg_vol * 1.5]
        
        bullish, bearish = [], []
        
        for idx, row in high_vol_candles.iterrows():
            # If Close > Open, it's a bullish high volume zone (Demand)
            if row['Close'] > row['Open']:
                bullish.append({'time': idx, 'price': row['Low'], 'vol': row['Volume']})
            # If Close < Open, it's a bearish high volume zone (Supply)
            else:
                bearish.append({'time': idx, 'price': row['High'], 'vol': row['Volume']})
                
        # Return only the most recent 3 blocks to avoid clutter
        return bullish[-3:], bearish[-3:] 

# =============================================
# 5. MAIN APPLICATION
# =============================================

def main():
    st.title("🚀 Pro Nifty Dashboard (Powered by DhanHQ)")
    
    # 1. Credentials Setup
    try:
        CLIENT_ID = st.secrets["dhan"]["client_id"]
        ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
        
        # Initialize Engines
        adapter = DhanAdapter(CLIENT_ID, ACCESS_TOKEN)
        market_data = EnhancedMarketData()
        bias_engine = BiasAnalyzer(adapter)
        chain_engine = OptionChainAnalyzer()
        vob_engine = VolumeOrderBlocks()
        
        st.sidebar.success("✅ Dhan API Connected")
    except Exception as e:
        st.error("❌ Dhan Credentials Missing in `.streamlit/secrets.toml`")
        st.info("Please add `[dhan]` section with `client_id` and `access_token`.")
        st.stop()

    # 2. Sidebar Controls
    st.sidebar.header("Controls")
    symbol = st.sidebar.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    if st.sidebar.button("🔄 Refresh All Data"):
        st.rerun()

    # 3. Main Dashboard Layout
    tab1, tab2, tab3 = st.tabs(["📈 Technicals & Bias", "📊 Pro Option Chain", "🌍 Global Market"])

    # --- TAB 1: TECHNICALS ---
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        # A. Bias Analysis
        with st.spinner(f"Analyzing {symbol} Trends..."):
            analysis = bias_engine.analyze(symbol)
        
        if analysis:
            with col2:
                st.subheader("Market Bias")
                bias_color = "green" if analysis['bias'] == "BULLISH" else "red" if analysis['bias'] == "BEARISH" else "gray"
                st.markdown(f"## :{bias_color}[{analysis['bias']}]")
                st.metric("Tech Score", f"{analysis['score']:.1f} / 3.0")
                
                st.markdown("---")
                st.metric("LTP", f"{analysis['price']:.2f}")
                st.metric("RSI (14)", f"{analysis['rsi']:.2f}")
                
                st.write("**Key Drivers:**")
                for reason in analysis['reasons']:
                    st.caption(f"• {reason}")

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
                
                # EMAs
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], line=dict(color='blue', width=1), name='EMA 9'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], line=dict(color='orange', width=1), name='EMA 21'), row=1, col=1)
                
                # Volume
                colors = ['green' if r['Close'] > r['Open'] else 'red' for i, r in df.iterrows()]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
                
                # VOB Lines
                for b in bull:
                    fig.add_hline(y=b['price'], line_color="green", line_dash="dash", row=1, col=1, annotation_text="Demand")
                for b in bear:
                    fig.add_hline(y=b['price'], line_color="red", line_dash="dash", row=1, col=1, annotation_text="Supply")
                
                fig.update_layout(height=650, xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                
        else:
            st.error("Could not fetch Technical Data from Dhan. Market might be closed or Token expired.")

    # --- TAB 2: OPTION CHAIN ---
    with tab2:
        st.subheader(f"{symbol} Option Chain Analysis (Official Dhan Data)")
        
        with st.spinner("Fetching Real-time Greeks & OI..."):
            chain_res = adapter.get_option_chain(symbol)
            oc_analysis = chain_engine.process(chain_res)
            
        if oc_analysis:
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spot Price", f"{oc_analysis['spot']}")
            m2.metric("Max Pain", f"{oc_analysis['max_pain']}")
            m3.metric("PCR (OI)", f"{oc_analysis['pcr']:.2f}")
            m4.metric("Synth Future", f"{oc_analysis['synthetic_fut']:.2f}")
            
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
            fig_pcr.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pcr, use_container_width=True)
            
            # Data Table
            df_chain = oc_analysis['df']
            
            # Filter near ATM for display (Spot +/- 2.5%)
            spot = oc_analysis['spot']
            df_display = df_chain[
                (df_chain['Strike'] > spot * 0.975) & 
                (df_chain['Strike'] < spot * 1.025)
            ].sort_values('Strike')
            
            st.write(f"**Option Chain Data ({oc_analysis['expiry']})**")
            st.dataframe(
                df_display.style.background_gradient(subset=['CE_OI', 'PE_OI'], cmap="Blues"),
                use_container_width=True,
                column_config={
                    "Strike": st.column_config.NumberColumn(format="%.0f"),
                    "CE_OI": st.column_config.NumberColumn(format="%d"),
                    "PE_OI": st.column_config.NumberColumn(format="%d"),
                    "CE_LTP": st.column_config.NumberColumn(format="%.2f"),
                    "PE_LTP": st.column_config.NumberColumn(format="%.2f"),
                }
            )
        else:
            st.warning("Option Chain data unavailable. Token might be expired or Market closed.")

    # --- TAB 3: GLOBAL MARKETS ---
    with tab3:
        st.subheader("🌍 Global Indices (Yahoo Finance)")
        if st.button("Refresh Global Data"):
            with st.spinner("Fetching Global Data..."):
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
                    st.warning("Could not fetch global data. Check internet connection.")
        else:
            st.info("Click Refresh to load Yahoo Finance data.")

if __name__ == "__main__":
    main()
