import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np
import math
from scipy.stats import norm
from typing import Dict, List, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Nifty Dashboard", page_icon="📈", layout="wide")

# =============================================
# ENHANCED MARKET DATA
# =============================================
class EnhancedMarketData:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def fetch_india_vix(self) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                vix = hist['Close'].iloc[-1]
                if vix > 25: sentiment, bias, score = "HIGH FEAR", "BEARISH", -75
                elif vix > 20: sentiment, bias, score = "ELEVATED", "BEARISH", -50
                elif vix > 15: sentiment, bias, score = "MODERATE", "NEUTRAL", 0
                else: sentiment, bias, score = "LOW", "BULLISH", 40
                return {'success': True, 'value': vix, 'sentiment': sentiment, 'bias': bias, 'score': score}
        except: pass
        return {'success': False}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        sectors = {'^CNXIT': 'IT', '^CNXBANK': 'BANK', '^CNXAUTO': 'AUTO', '^CNXPHARMA': 'PHARMA'}
        data = []
        for sym, name in sectors.items():
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    last, open_ = hist['Close'].iloc[-1], hist['Open'].iloc[0]
                    chg_pct = ((last - open_) / open_) * 100
                    bias = "BULLISH" if chg_pct > 0.5 else "BEARISH" if chg_pct < -0.5 else "NEUTRAL"
                    data.append({'sector': name, 'change_pct': chg_pct, 'bias': bias})
            except: pass
        return data

    def fetch_all(self) -> Dict[str, Any]:
        return {'vix': self.fetch_india_vix(), 'sectors': self.fetch_sector_indices()}

# =============================================
# NSE OPTIONS ANALYZER
# =============================================
class NSEOptionsAnalyzer:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.instruments = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']

    def calculate_greeks(self, opt_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple:
        try:
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            delta = norm.cdf(d1) if opt_type == 'CE' else -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
            vega = S*norm.pdf(d1)*math.sqrt(T)/100
            return round(delta, 4), round(gamma, 4), round(vega, 4), 0, 0
        except: return 0, 0, 0, 0, 0

    def fetch_option_chain(self, instrument: str) -> Dict[str, Any]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)
            
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={instrument}"
            response = session.get(url, timeout=10)
            data = response.json()
            
            records = data['records']['data']
            spot = data['records']['underlyingValue']
            expiry = data['records']['expiryDates'][0]
            
            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            
            return {
                'success': True, 'instrument': instrument, 'spot': spot, 'expiry': expiry,
                'total_ce_oi': total_ce_oi, 'total_pe_oi': total_pe_oi, 'records': records
            }
        except Exception as e:
            return {'success': False, 'instrument': instrument, 'error': str(e)}

    def analyze_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        try:
            data = self.fetch_option_chain(instrument)
            if not data['success']: return None
            
            spot = data['spot']
            records = data['records']
            expiry = data['expiry']
            
            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06
            
            # Process options
            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility']/100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)
                
                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility']/100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)
            
            if not calls or not puts: return None
            
            df = pd.merge(pd.DataFrame(calls), pd.DataFrame(puts), on='strikePrice', suffixes=('_CE', '_PE'))
            
            # Find ATM
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            atm_row = df[df['strikePrice'] == atm_strike].iloc[0]
            
            # Calculate bias score
            pcr_oi = data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 1
            score = 2 if pcr_oi > 1.1 else -2 if pcr_oi < 0.9 else 0
            overall = "BULLISH" if score >= 1 else "BEARISH" if score <= -1 else "NEUTRAL"
            
            return {
                'instrument': instrument, 'spot_price': spot, 'atm_strike': atm_strike,
                'overall_bias': overall, 'bias_score': score, 'pcr_oi': pcr_oi,
                'total_ce_oi': data['total_ce_oi'], 'total_pe_oi': data['total_pe_oi'],
                'atm_ce_price': atm_row['lastPrice_CE'], 'atm_pe_price': atm_row['lastPrice_PE'],
                'atm_ce_oi': atm_row['openInterest_CE'], 'atm_pe_oi': atm_row['openInterest_PE']
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_all_bias(self) -> List[Dict[str, Any]]:
        results = []
        for inst in self.instruments:
            bias = self.analyze_atm_bias(inst)
            if bias: results.append(bias)
        return results

# =============================================
# PRICE CHART COMPONENT
# =============================================
class PriceChart:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def fetch_price_data(self, symbol: str = "^NSEI", period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """Fetch price data for chart display"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            st.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def create_price_chart(self, df: pd.DataFrame, title: str = "NIFTY 50 Price Chart") -> go.Figure:
        """Create interactive price chart"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font=dict(size=20))
            return fig
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
        
        # Calculate and add moving averages
        if len(df) > 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA20'],
                line=dict(color='orange', width=2),
                name="MA20"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            template="plotly_dark",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig

# =============================================
# MAIN APP
# =============================================
class NiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.market_data = EnhancedMarketData()
        self.options_analyzer = NSEOptionsAnalyzer()
        self.price_chart = PriceChart()
        self.init_session()

    def init_session(self):
        if 'market_data' not in st.session_state: 
            st.session_state.market_data = None
        if 'options_data' not in st.session_state: 
            st.session_state.options_data = None
        if 'price_data' not in st.session_state: 
            st.session_state.price_data = None

    def run(self):
        # Custom CSS for better styling
        st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #ff7f0e;
                margin: 1rem 0;
            }
            .metric-card {
                background: rgba(255,255,255,0.1);
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #1f77b4;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">📈 Nifty Trading Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### *Real-time Market Data & Options Analysis*")
        
        # Display current time
        current_time = datetime.now(self.ist).strftime("%Y-%m-%d %H:%M:%S IST")
        st.write(f"**Last Updated:** {current_time}")
        
        # Tabs - Only Market Data, Options Bias, and Tabulation
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Charts", "🌍 Market Data", "🎯 Options Bias", "📋 Summary"])
        
        with tab1:
            self.display_live_charts()
        
        with tab2:
            self.display_market_data()
        
        with tab3:
            self.display_options_bias()
        
        with tab4:
            self.display_summary()

    def display_live_charts(self):
        st.header("📊 Live Price Charts")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Chart Settings")
            symbol = st.selectbox("Select Index", ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS"], index=0)
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo"], index=0)
            interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=0)
            
            if st.button("🔄 Update Chart", type="primary"):
                with st.spinner("Fetching chart data..."):
                    st.session_state.price_data = self.price_chart.fetch_price_data(symbol, period, interval)
        
        with col1:
            if st.session_state.price_data is None:
                st.session_state.price_data = self.price_chart.fetch_price_data()
            
            if not st.session_state.price_data.empty:
                chart_title = f"{symbol} Price Chart - {period} period"
                fig = self.price_chart.create_price_chart(st.session_state.price_data, chart_title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to fetch price data. Please try again.")

    def display_market_data(self):
        st.header("🌍 Enhanced Market Data")
        
        if st.button("🔄 Update Market Data", type="primary", key="market_update"):
            with st.spinner("Fetching latest market data..."):
                st.session_state.market_data = self.market_data.fetch_all()
                st.success("Market data updated successfully!")
        
        if st.session_state.market_data:
            data = st.session_state.market_data
            
            # VIX Section
            st.markdown('<div class="sub-header">🇮🇳 India VIX Analysis</div>', unsafe_allow_html=True)
            if data['vix']['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("VIX Value", f"{data['vix']['value']:.2f}")
                with col2:
                    st.metric("Market Sentiment", data['vix']['sentiment'])
                with col3:
                    st.metric("Volatility Bias", data['vix']['bias'])
                with col4:
                    st.metric("Fear Score", f"{data['vix']['score']}")
                
                # VIX Interpretation
                vix_value = data['vix']['value']
                if vix_value > 25:
                    st.warning("⚠️ High VIX indicates elevated market fear and potential volatility")
                elif vix_value > 20:
                    st.info("ℹ️ Elevated VIX suggests increased market uncertainty")
                else:
                    st.success("✅ Low VIX indicates stable market conditions")
            
            # Sector Performance
            st.markdown('<div class="sub-header">📈 Sector Performance</div>', unsafe_allow_html=True)
            if data['sectors']:
                sectors_df = pd.DataFrame(data['sectors']).sort_values('change_pct', ascending=False)
                
                # Create columns for sector display
                cols = st.columns(4)
                for idx, (_, sector) in enumerate(sectors_df.iterrows()):
                    with cols[idx % 4]:
                        color = "🟢" if sector['change_pct'] > 0 else "🔴" if sector['change_pct'] < 0 else "⚪"
                        bias_icon = "📈" if sector['bias'] == "BULLISH" else "📉" if sector['bias'] == "BEARISH" else "➡️"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{color} {sector['sector']}</h4>
                            <h3>{sector['change_pct']:+.2f}%</h3>
                            <p>{bias_icon} {sector['bias']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No sector data available. Click 'Update Market Data' to fetch latest information.")

    def display_options_bias(self):
        st.header("🎯 Options Chain Analysis")
        
        if st.button("🔄 Update Options Data", type="primary", key="options_update"):
            with st.spinner("Fetching options chain data..."):
                st.session_state.options_data = self.options_analyzer.get_all_bias()
                if st.session_state.options_data:
                    st.success(f"Options data loaded for {len(st.session_state.options_data)} instruments!")
                else:
                    st.error("Failed to fetch options data. Please try again.")
        
        if st.session_state.options_data:
            for data in st.session_state.options_data:
                # Determine color based on bias
                if data['overall_bias'] == "BULLISH":
                    border_color = "#00ff00"
                    icon = "🟢"
                elif data['overall_bias'] == "BEARISH":
                    border_color = "#ff0000"
                    icon = "🔴"
                else:
                    border_color = "#ffff00"
                    icon = "🟡"
                
                with st.expander(f"{icon} {data['instrument']} - {data['overall_bias']} | Spot: ₹{data['spot_price']:.2f}", expanded=True):
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{data['spot_price']:.2f}")
                    with col2:
                        st.metric("ATM Strike", f"₹{data['atm_strike']:.2f}")
                    with col3:
                        st.metric("Overall Bias", data['overall_bias'])
                    with col4:
                        st.metric("PCR OI", f"{data['pcr_oi']:.2f}")
                    with col5:
                        st.metric("Bias Score", f"{data['bias_score']:.2f}")
                    
                    # Detailed information
                    st.subheader("Options Chain Details")
                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                    
                    with detail_col1:
                        st.write("**Call Options (CE)**")
                        st.write(f"Total OI: {data['total_ce_oi']:,.0f}")
                        st.write(f"ATM Price: ₹{data['atm_ce_price']:.2f}")
                        st.write(f"ATM OI: {data['atm_ce_oi']:,.0f}")
                    
                    with detail_col2:
                        st.write("**Put Options (PE)**")
                        st.write(f"Total OI: {data['total_pe_oi']:,.0f}")
                        st.write(f"ATM Price: ₹{data['atm_pe_price']:.2f}")
                        st.write(f"ATM OI: {data['atm_pe_oi']:,.0f}")
                    
                    with detail_col3:
                        st.write("**Market Interpretation**")
                        if data['pcr_oi'] > 1.1:
                            st.success("Higher Put OI suggests bullish sentiment")
                        elif data['pcr_oi'] < 0.9:
                            st.error("Higher Call OI suggests bearish sentiment")
                        else:
                            st.info("Balanced OI suggests neutral sentiment")
                    
                    with detail_col4:
                        st.write("**Trading Bias**")
                        if data['bias_score'] > 0:
                            st.success("Bullish setup detected")
                        elif data['bias_score'] < 0:
                            st.error("Bearish setup detected")
                        else:
                            st.warning("Neutral setup - wait for confirmation")
        else:
            st.info("Click 'Update Options Data' to load options chain analysis for NIFTY, BANKNIFTY, and FINNIFTY.")

    def display_summary(self):
        st.header("📋 Comprehensive Market Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Quick Overview")
            
            # Market Data Summary
            if st.session_state.market_data:
                vix_data = st.session_state.market_data['vix']
                if vix_data['success']:
                    st.write(f"**VIX Level:** {vix_data['value']:.2f} ({vix_data['sentiment']})")
                    st.write(f"**Volatility Bias:** {vix_data['bias']}")
                
                if st.session_state.market_data['sectors']:
                    sectors = st.session_state.market_data['sectors']
                    bull_sectors = len([s for s in sectors if s['bias'] == 'BULLISH'])
                    bear_sectors = len([s for s in sectors if s['bias'] == 'BEARISH'])
                    st.write(f"**Sector Sentiment:** {bull_sectors} Bullish, {bear_sectors} Bearish")
            
            # Options Summary
            if st.session_state.options_data:
                st.write(f"**Options Instruments:** {len(st.session_state.options_data)} loaded")
                for opt in st.session_state.options_data:
                    st.write(f"- {opt['instrument']}: {opt['overall_bias']} (PCR: {opt['pcr_oi']:.2f})")
        
        with col2:
            st.subheader("🎯 Trading Insights")
            
            insights = []
            
            # Generate insights based on available data
            if st.session_state.market_data and st.session_state.market_data['vix']['success']:
                vix = st.session_state.market_data['vix']['value']
                if vix > 25:
                    insights.append("High VIX suggests caution - consider hedging strategies")
                elif vix < 15:
                    insights.append("Low VIX environment - suitable for premium selling")
            
            if st.session_state.options_data:
                for opt in st.session_state.options_data:
                    if opt['pcr_oi'] > 1.2:
                        insights.append(f"High PCR in {opt['instrument']} indicates bullish bias")
                    elif opt['pcr_oi'] < 0.8:
                        insights.append(f"Low PCR in {opt['instrument']} suggests bearish pressure")
            
            if not insights:
                insights.append("Update market and options data for detailed insights")
            
            for insight in insights:
                st.write(f"• {insight}")
        
        # Data Tables Section
        st.subheader("📈 Detailed Data Tables")
        
        # Market Data Table
        if st.session_state.market_data and st.session_state.market_data['sectors']:
            st.write("**Sector Performance Table**")
            sectors_df = pd.DataFrame(st.session_state.market_data['sectors'])
            st.dataframe(sectors_df, use_container_width=True, hide_index=True)
        
        # Options Data Table
        if st.session_state.options_data:
            st.write("**Options Analysis Table**")
            options_summary = []
            for data in st.session_state.options_data:
                options_summary.append({
                    'Instrument': data['instrument'],
                    'Spot Price': f"₹{data['spot_price']:.2f}",
                    'ATM Strike': f"₹{data['atm_strike']:.2f}",
                    'Bias': data['overall_bias'],
                    'Score': f"{data['bias_score']:.2f}",
                    'PCR OI': f"{data['pcr_oi']:.2f}",
                    'CE OI': f"{data['total_ce_oi']:,.0f}",
                    'PE OI': f"{data['total_pe_oi']:,.0f}"
                })
            st.dataframe(pd.DataFrame(options_summary), use_container_width=True, hide_index=True)

# Run the application
if __name__ == "__main__":
    app = NiftyApp()
    app.run()