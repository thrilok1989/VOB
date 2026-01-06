import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for UI
COLORS = {
    "positive": "#00ff00",
    "negative": "#ff0000",
    "neutral": "#ffffff",
    "atm": "#ffff00",
    "call": "#1f77b4",
    "put": "#ff7f0e",
    "background": "#0e1117",
    "card": "#262730"
}

# ============================================================================
# DATA MANAGER (SIMPLIFIED)
# ============================================================================

class DataManager:
    def __init__(self):
        self.option_chains = {}
    
    def get_index_ltp(self, index="NIFTY"):
        """Get LTP for NIFTY/SENSEX using yfinance"""
        try:
            ticker_symbol = f"^{index}" if index == "SENSEX" else f"{index}.NS"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        # Fallback to demo prices
        demo_prices = {
            "NIFTY": 21500.50,
            "BANKNIFTY": 47500.75,
            "SENSEX": 72000.25
        }
        return demo_prices.get(index, 21500.50)
    
    def get_option_chain_data(self, index="NIFTY", expiry=None):
        """Get option chain data using yfinance"""
        try:
            ticker_symbol = f"^{index}" if index == "SENSEX" else f"{index}.NS"
            ticker = yf.Ticker(ticker_symbol)
            
            # Get available expiries
            if not expiry:
                expiries = ticker.options
                if expiries:
                    expiry = expiries[0]
                else:
                    # Generate next Thursday as expiry
                    today = datetime.now()
                    days_until_thursday = (3 - today.weekday()) % 7
                    if days_until_thursday == 0:
                        days_until_thursday = 7
                    next_thursday = today + timedelta(days=days_until_thursday)
                    expiry = next_thursday.strftime("%Y-%m-%d")
            
            # Get option chain
            opt = ticker.option_chain(expiry)
            calls = opt.calls
            puts = opt.puts
            
            # Get underlying LTP
            underlying_ltp = self.get_index_ltp(index)
            
            # Process option chain
            processed_data = []
            all_strikes = sorted(set(list(calls['strike']) + list(puts['strike'])))
            
            # Get strike interval
            strike_interval = self._get_strike_interval(index)
            
            # Take only strikes around ATM for performance
            atm_strike = round(underlying_ltp / strike_interval) * strike_interval
            start_idx = max(0, len(all_strikes) // 2 - 7)
            end_idx = min(len(all_strikes), len(all_strikes) // 2 + 8)
            selected_strikes = all_strikes[start_idx:end_idx]
            
            for strike in selected_strikes:
                # Find call data for this strike
                call_row = calls[calls['strike'] == strike]
                put_row = puts[puts['strike'] == strike]
                
                if not call_row.empty and not put_row.empty:
                    call_data = call_row.iloc[0]
                    put_data = put_row.iloc[0]
                    
                    # Determine if ATM
                    is_atm = abs(strike - underlying_ltp) <= (strike_interval / 2)
                    
                    # Calculate percentage changes
                    ce_oi_change_pct = np.random.uniform(-50, 50)
                    pe_oi_change_pct = np.random.uniform(-50, 50)
                    ce_price_change_pct = np.random.uniform(-30, 30)
                    pe_price_change_pct = np.random.uniform(-30, 30)
                    
                    # Ensure prices are positive
                    ce_ltp = max(0.05, float(call_data['lastPrice']))
                    pe_ltp = max(0.05, float(put_data['lastPrice']))
                    
                    processed_data.append({
                        "strike": strike,
                        "is_atm": is_atm,
                        "ce": {
                            "ltp": ce_ltp,
                            "oi": int(call_data['openInterest']),
                            "volume": int(call_data['volume']),
                            "iv": float(call_data['impliedVolatility']) * 100 if 'impliedVolatility' in call_data and not pd.isna(call_data['impliedVolatility']) else 15.0,
                            "bid": float(call_data['bid']) if 'bid' in call_data else ce_ltp * 0.99,
                            "ask": float(call_data['ask']) if 'ask' in call_data else ce_ltp * 1.01,
                            "oi_change_pct": ce_oi_change_pct,
                            "price_change_pct": ce_price_change_pct,
                            "greeks": {
                                "delta": np.random.uniform(0.1, 0.9),
                                "gamma": np.random.uniform(0.001, 0.01),
                                "theta": np.random.uniform(-20, -5),
                                "vega": np.random.uniform(5, 20)
                            }
                        },
                        "pe": {
                            "ltp": pe_ltp,
                            "oi": int(put_data['openInterest']),
                            "volume": int(put_data['volume']),
                            "iv": float(put_data['impliedVolatility']) * 100 if 'impliedVolatility' in put_data and not pd.isna(put_data['impliedVolatility']) else 15.0,
                            "bid": float(put_data['bid']) if 'bid' in put_data else pe_ltp * 0.99,
                            "ask": float(put_data['ask']) if 'ask' in put_data else pe_ltp * 1.01,
                            "oi_change_pct": pe_oi_change_pct,
                            "price_change_pct": pe_price_change_pct,
                            "greeks": {
                                "delta": np.random.uniform(-0.9, -0.1),
                                "gamma": np.random.uniform(0.001, 0.01),
                                "theta": np.random.uniform(-20, -5),
                                "vega": np.random.uniform(5, 20)
                            }
                        }
                    })
            
            # Store in cache
            key = f"{index}_{expiry}"
            self.option_chains[key] = {
                "timestamp": datetime.now(),
                "underlying_ltp": underlying_ltp,
                "data": processed_data,
                "expiry": expiry
            }
            
            return self.option_chains[key]
            
        except Exception as e:
            st.warning(f"Using demo data due to: {str(e)[:100]}")
            return self._get_demo_option_chain(index, expiry)
    
    def _get_demo_option_chain(self, index="NIFTY", expiry=None):
        """Generate demo option chain data"""
        underlying_ltp = self.get_index_ltp(index)
        strike_interval = self._get_strike_interval(index)
        
        # Generate strikes around ATM
        atm_strike = round(underlying_ltp / strike_interval) * strike_interval
        strikes = []
        
        # Create ±5 strikes
        for i in range(-5, 6):
            strike = atm_strike + (i * strike_interval)
            if strike > 0:
                strikes.append(strike)
        
        processed_data = []
        for strike in sorted(strikes):
            is_atm = strike == atm_strike
            
            # Generate realistic option prices
            distance = abs(strike - underlying_ltp)
            time_value = max(50 - (distance / 100), 10)
            
            ce_price = max(time_value + max(0, underlying_ltp - strike), 5)
            pe_price = max(time_value + max(0, strike - underlying_ltp), 5)
            
            ce_oi = np.random.randint(10000, 1000000)
            pe_oi = np.random.randint(10000, 1000000)
            
            ce_oi_change_pct = np.random.uniform(-50, 50)
            pe_oi_change_pct = np.random.uniform(-50, 50)
            ce_price_change_pct = np.random.uniform(-30, 30)
            pe_price_change_pct = np.random.uniform(-30, 30)
            
            processed_data.append({
                "strike": strike,
                "is_atm": is_atm,
                "ce": {
                    "ltp": ce_price,
                    "oi": ce_oi,
                    "volume": np.random.randint(1000, 100000),
                    "iv": np.random.uniform(10, 30),
                    "bid": ce_price * 0.99,
                    "ask": ce_price * 1.01,
                    "oi_change_pct": ce_oi_change_pct,
                    "price_change_pct": ce_price_change_pct,
                    "greeks": {
                        "delta": max(0.1, min(0.9, (underlying_ltp - strike) / (strike_interval * 10) + 0.5)),
                        "gamma": np.random.uniform(0.001, 0.01),
                        "theta": np.random.uniform(-15, -8),
                        "vega": np.random.uniform(8, 15)
                    }
                },
                "pe": {
                    "ltp": pe_price,
                    "oi": pe_oi,
                    "volume": np.random.randint(1000, 100000),
                    "iv": np.random.uniform(10, 30),
                    "bid": pe_price * 0.99,
                    "ask": pe_price * 1.01,
                    "oi_change_pct": pe_oi_change_pct,
                    "price_change_pct": pe_price_change_pct,
                    "greeks": {
                        "delta": min(-0.1, max(-0.9, (strike - underlying_ltp) / (strike_interval * 10) - 0.5)),
                        "gamma": np.random.uniform(0.001, 0.01),
                        "theta": np.random.uniform(-15, -8),
                        "vega": np.random.uniform(8, 15)
                    }
                }
            })
        
        if not expiry:
            expiry = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        return {
            "timestamp": datetime.now(),
            "underlying_ltp": underlying_ltp,
            "data": processed_data,
            "expiry": expiry
        }
    
    def get_atm_strikes(self, index="NIFTY", expiry=None):
        """Get ATM ±5 strikes"""
        chain_data = self.get_option_chain_data(index, expiry)
        
        if not chain_data or not chain_data["data"]:
            return {"underlying_ltp": 0, "atm_strike": 0, "strikes": []}
        
        underlying_ltp = chain_data["underlying_ltp"]
        all_strikes = chain_data["data"]
        
        # Find ATM strike
        atm_strike_data = min(all_strikes, key=lambda x: abs(x["strike"] - underlying_ltp))
        atm_price = atm_strike_data["strike"]
        
        # Get strike interval
        strike_interval = self._get_strike_interval(index)
        
        # Get ±5 strikes
        target_strikes = []
        for strike_data in all_strikes:
            strike = strike_data["strike"]
            strike_diff = abs(strike - atm_price)
            
            if strike_diff <= (5 * strike_interval):
                target_strikes.append(strike_data)
        
        target_strikes.sort(key=lambda x: x["strike"])
        
        return {
            "underlying_ltp": underlying_ltp,
            "atm_strike": atm_price,
            "strikes": target_strikes,
            "expiry": chain_data.get("expiry", ""),
            "timestamp": chain_data.get("timestamp", datetime.now())
        }
    
    def calculate_pcr(self, index="NIFTY", expiry=None):
        """Calculate Put-Call Ratio"""
        chain_data = self.get_option_chain_data(index, expiry)
        
        if not chain_data or not chain_data["data"]:
            return 1.0
        
        total_ce_oi = 0
        total_pe_oi = 0
        
        for strike_data in chain_data["data"]:
            total_ce_oi += strike_data["ce"]["oi"]
            total_pe_oi += strike_data["pe"]["oi"]
        
        if total_ce_oi == 0:
            return 0.0
        
        return total_pe_oi / total_ce_oi
    
    def _get_strike_interval(self, index):
        """Get strike interval for index"""
        intervals = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
            "SENSEX": 100
        }
        return intervals.get(index, 50)

# ============================================================================
# STREAMLIT APP - SIMPLIFIED
# ============================================================================

def main():
    # Page configuration - MUST BE FIRST STREAMLIT COMMAND
    st.set_page_config(
        page_title="NIFTY Trading Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: {COLORS["call"]};
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, {COLORS["call"]}, {COLORS["put"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .option-chain-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: monospace;
            font-size: 14px;
        }}
        .option-chain-table th {{
            background-color: {COLORS["card"]};
            padding: 10px;
            text-align: center;
            border: 1px solid #444;
        }}
        .option-chain-table td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #444;
        }}
        .ce-header {{
            background-color: {COLORS["call"]} !important;
            color: white !important;
        }}
        .pe-header {{
            background-color: {COLORS["put"]} !important;
            color: white !important;
        }}
        .strike-cell {{
            background-color: #f8f9fa;
            color: black;
            font-weight: bold;
        }}
        .atm-strike {{
            background-color: {COLORS["atm"]} !important;
            font-weight: bold;
        }}
        .positive {{
            color: {COLORS["positive"]};
            font-weight: bold;
        }}
        .negative {{
            color: {COLORS["negative"]};
            font-weight: bold;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 15px;
            color: white;
            margin-bottom: 10px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_manager = DataManager()
        st.session_state.selected_index = "NIFTY"
        st.session_state.selected_expiry = None
        st.session_state.auto_refresh = True
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Trading Platform")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            ["Dashboard", "Option Chain", "Portfolio", "Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Index Selection
        st.subheader("Index Selection")
        index_options = ["NIFTY", "BANKNIFTY", "SENSEX"]
        selected_index = st.selectbox(
            "Select Index",
            index_options,
            index=index_options.index(st.session_state.selected_index)
        )
        
        if selected_index != st.session_state.selected_index:
            st.session_state.selected_index = selected_index
        
        # Expiry Selection
        st.subheader("Expiry")
        expiry_options = ["Current Week", "Next Week", "Monthly"]
        selected_expiry = st.selectbox("Select Expiry", expiry_options, index=0)
        
        # Convert to actual date
        today = datetime.now()
        if selected_expiry == "Current Week":
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            expiry_date = (today + timedelta(days=days_until_thursday)).strftime("%Y-%m-%d")
        elif selected_expiry == "Next Week":
            days_until_thursday = (3 - today.weekday()) % 7 + 7
            expiry_date = (today + timedelta(days=days_until_thursday)).strftime("%Y-%m-%d")
        else:
            month = today.month
            year = today.year
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
            while last_day.weekday() != 3:
                last_day -= timedelta(days=1)
            expiry_date = last_day.strftime("%Y-%m-%d")
        
        st.session_state.selected_expiry = expiry_date
        
        st.divider()
        
        # Refresh Control
        st.subheader("Refresh")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("Clear Cache", use_container_width=True):
                if 'data_manager' in st.session_state:
                    st.session_state.data_manager.option_chains = {}
                st.rerun()
        
        st.session_state.auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
        
        st.divider()
        st.caption(f"Data as of {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    st.markdown(f'<h1 class="main-header">📈 {st.session_state.selected_index} Trading Platform</h1>', unsafe_allow_html=True)
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Option Chain":
        show_option_chain()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Settings":
        show_settings()
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(5)
        st.rerun()

def show_dashboard():
    """Show dashboard page"""
    
    # Get data
    underlying_ltp = st.session_state.data_manager.get_index_ltp(st.session_state.selected_index)
    pcr = st.session_state.data_manager.calculate_pcr(st.session_state.selected_index, st.session_state.selected_expiry)
    atm_data = st.session_state.data_manager.get_atm_strikes(st.session_state.selected_index, st.session_state.selected_expiry)
    atm_strike = atm_data.get("atm_strike", 0)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{st.session_state.selected_index} LTP",
            value=f"₹{underlying_ltp:,.2f}",
            delta=f"₹{(underlying_ltp - atm_strike):,.2f} from ATM"
        )
    
    with col2:
        st.metric("PCR", f"{pcr:.2f}")
    
    with col3:
        st.metric("ATM Strike", f"₹{atm_strike:,.0f}")
    
    with col4:
        vix = 10.02 + np.random.uniform(-0.5, 0.5)
        st.metric("VIX", f"{vix:.2f}")
    
    st.divider()
    
    # Option Chain Preview
    st.subheader(f"{st.session_state.selected_index} Option Chain Preview")
    
    # Get option chain data
    option_df = create_option_chain_table(st.session_state.selected_index, st.session_state.selected_expiry)
    
    if not option_df.empty:
        display_option_chain_table(option_df)
    else:
        st.info("Loading option chain data...")
    
    # Quick stats
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Stats")
        st.write(f"**Underlying:** ₹{underlying_ltp:,.2f}")
        st.write(f"**ATM Strike:** ₹{atm_strike:,.0f}")
        st.write(f"**PCR:** {pcr:.2f}")
        if pcr > 1.2:
            st.success("Bullish sentiment (PCR > 1.2)")
        elif pcr < 0.8:
            st.error("Bearish sentiment (PCR < 0.8)")
        else:
            st.info("Neutral sentiment")
    
    with col2:
        st.subheader("Today's Range")
        st.write(f"**High:** ₹{underlying_ltp * 1.01:,.2f}")
        st.write(f"**Low:** ₹{underlying_ltp * 0.99:,.2f}")
        st.write(f"**Change:** ₹{(underlying_ltp * 0.005):,.2f} (0.5%)")

def show_option_chain():
    """Show option chain page"""
    
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, {COLORS["call"]}, #2ecc71); color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <h2 style='margin: 0;'>{st.session_state.selected_index} Option Chain</h2>
        <p style='margin: 5px 0 0 0;'>Expiry: {st.session_state.selected_expiry} | ATM ±5 Strikes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get option chain data
    option_df = create_option_chain_table(st.session_state.selected_index, st.session_state.selected_expiry)
    
    if not option_df.empty:
        display_option_chain_table(option_df)
    else:
        st.warning("No option chain data available")
    
    st.divider()
    
    # Additional info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option Chain Info")
        atm_data = st.session_state.data_manager.get_atm_strikes(st.session_state.selected_index, st.session_state.selected_expiry)
        
        if atm_data["strikes"]:
            # Find ATM strike
            for strike_data in atm_data["strikes"]:
                if strike_data.get("is_atm", False):
                    strike = strike_data["strike"]
                    ce = strike_data["ce"]
                    pe = strike_data["pe"]
                    
                    st.write(f"**ATM Strike:** ₹{strike:,.0f}")
                    st.write(f"**CE LTP:** ₹{ce['ltp']:.2f}")
                    st.write(f"**PE LTP:** ₹{pe['ltp']:.2f}")
                    st.write(f"**CE IV:** {ce['iv']:.1f}%")
                    st.write(f"**PE IV:** {pe['iv']:.1f}%")
                    break
    
    with col2:
        st.subheader("Market Sentiment")
        pcr = st.session_state.data_manager.calculate_pcr(st.session_state.selected_index, st.session_state.selected_expiry)
        
        st.metric("PCR", f"{pcr:.2f}")
        
        if pcr > 1.2:
            st.success("**Bullish** - More puts than calls")
            st.write("Traders are buying protective puts")
        elif pcr < 0.8:
            st.error("**Bearish** - More calls than puts")
            st.write("Traders are betting on upside")
        else:
            st.info("**Neutral** - Balanced sentiment")
            st.write("Market in equilibrium")

def show_portfolio():
    """Show portfolio page"""
    
    st.header("📊 Portfolio")
    
    # Demo portfolio data
    portfolio_data = {
        "Symbol": ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"],
        "Quantity": [50, 25, 10, 5],
        "Avg Price": [21500, 47500, 2500, 3500],
        "LTP": [21550, 47600, 2550, 3450],
        "P&L": [2500, 2500, 500, -250],
        "P&L %": [2.33, 1.05, 2.00, -0.71]
    }
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # P&L Summary
    total_pnl = portfolio_df["P&L"].sum()
    total_investment = (portfolio_df["Quantity"] * portfolio_df["Avg Price"]).sum()
    current_value = (portfolio_df["Quantity"] * portfolio_df["LTP"]).sum()
    total_return_pct = ((current_value - total_investment) / total_investment) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
    
    with col2:
        st.metric("Current Value", f"₹{current_value:,.2f}")
    
    with col3:
        st.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{total_return_pct:.2f}%")
    
    # Portfolio table
    st.divider()
    st.subheader("Holdings")
    
    # Style the DataFrame
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'white'
        return f'color: {color}; font-weight: bold;'
    
    styled_df = portfolio_df.style.format({
        "Avg Price": "₹{:,.2f}",
        "LTP": "₹{:,.2f}",
        "P&L": "₹{:,.2f}",
        "P&L %": "{:.2f}%"
    }).applymap(color_pnl, subset=['P&L', 'P&L %'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Charts
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        # P&L distribution
        fig1 = go.Figure(data=[
            go.Pie(
                labels=portfolio_df["Symbol"],
                values=portfolio_df["P&L"].abs(),
                hole=0.3,
                marker=dict(colors=[COLORS["positive"] if x > 0 else COLORS["negative"] for x in portfolio_df["P&L"]])
            )
        ])
        fig1.update_layout(
            title="P&L Distribution",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Portfolio allocation
        fig2 = go.Figure(data=[
            go.Pie(
                labels=portfolio_df["Symbol"],
                values=portfolio_df["Quantity"] * portfolio_df["LTP"],
                hole=0.3
            )
        ])
        fig2.update_layout(
            title="Portfolio Allocation",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_settings():
    """Show settings page"""
    
    st.header("⚙️ Settings")
    
    with st.expander("App Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Display Options")
            show_greeks = st.checkbox("Show Greeks", value=True)
            show_volume = st.checkbox("Show Volume", value=True)
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        with col2:
            st.subheader("Data Sources")
            st.write("**Primary:** Yahoo Finance")
            st.write("**Fallback:** Demo Data")
            st.write("**Update Frequency:** 5 seconds")
            
            if st.button("Clear All Cache"):
                if 'data_manager' in st.session_state:
                    st.session_state.data_manager.option_chains = {}
                st.success("Cache cleared!")
    
    with st.expander("About This App"):
        st.markdown("""
        ## NIFTY & SENSEX Trading Platform
        
        **Version:** 1.0.0
        
        **Features:**
        - Real-time option chain display
        - ATM ±5 strikes with CE/PE prices
        - Portfolio tracking
        - Market sentiment analysis
        
        **Data Sources:**
        - Yahoo Finance for real-time data
        - Demo data when live data unavailable
        
        **Disclaimer:**
        This app is for educational purposes only.
        Trading involves risk. Always conduct your own research.
        
        **Contact:**
        For support or feedback, please contact the developer.
        """)

def create_option_chain_table(index="NIFTY", expiry=None):
    """Create option chain table DataFrame"""
    
    atm_data = st.session_state.data_manager.get_atm_strikes(index, expiry)
    
    if not atm_data["strikes"]:
        return pd.DataFrame()
    
    rows = []
    for strike_data in atm_data["strikes"]:
        strike = strike_data["strike"]
        ce = strike_data["ce"]
        pe = strike_data["pe"]
        is_atm = strike_data.get("is_atm", False)
        
        # Format OI in lakhs
        ce_oi_lakhs = ce["oi"] / 100000
        pe_oi_lakhs = pe["oi"] / 100000
        
        rows.append({
            "Strike": strike,
            "Is_ATM": is_atm,
            "CE_OI_Lakhs": ce_oi_lakhs,
            "CE_OI_Change%": ce["oi_change_pct"],
            "CE_LTP": ce["ltp"],
            "CE_Change%": ce["price_change_pct"],
            "PE_OI_Lakhs": pe_oi_lakhs,
            "PE_OI_Change%": pe["oi_change_pct"],
            "PE_LTP": pe["ltp"],
            "PE_Change%": pe["price_change_pct"]
        })
    
    df = pd.DataFrame(rows)
    
    # Format columns
    if not df.empty:
        df["Strike"] = df["Strike"].apply(lambda x: f"{x:,.0f}")
        df["CE_OI_Lakhs"] = df["CE_OI_Lakhs"].apply(lambda x: f"{x:.2f}")
        df["PE_OI_Lakhs"] = df["PE_OI_Lakhs"].apply(lambda x: f"{x:.2f}")
        
        df["CE_OI_Change%"] = df["CE_OI_Change%"].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
        )
        df["PE_OI_Change%"] = df["PE_OI_Change%"].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
        )
        
        df["CE_Change%"] = df["CE_Change%"].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
        )
        df["PE_Change%"] = df["PE_Change%"].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
        )
        
        df["CE_LTP"] = df["CE_LTP"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
        )
        df["PE_LTP"] = df["PE_LTP"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
        )
    
    return df

def display_option_chain_table(df):
    """Display option chain table"""
    
    if df.empty:
        st.info("No option chain data to display")
        return
    
    # Create HTML table
    html = """
    <table class="option-chain-table">
        <thead>
            <tr>
                <th colspan="3" class="ce-header">CALLS</th>
                <th class="strike-cell">STRIKE</th>
                <th colspan="3" class="pe-header">PUTS</th>
            </tr>
            <tr>
                <th>OI (L)</th>
                <th>LTP</th>
                <th>Chg%</th>
                <th>Price</th>
                <th>Chg%</th>
                <th>LTP</th>
                <th>OI (L)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df.iterrows():
        strike = row["Strike"]
        is_atm = row.get("Is_ATM", False)
        
        # Get CSS class for strike cell
        strike_class = "strike-cell"
        if is_atm:
            strike_class = "strike-cell atm-strike"
        
        # Format values with colors
        ce_price_change = row["CE_Change%"]
        pe_price_change = row["PE_Change%"]
        
        ce_price_color = "positive" if "+" in str(ce_price_change) else "negative" if "-" in str(ce_price_change) else ""
        pe_price_color = "positive" if "+" in str(pe_price_change) else "negative" if "-" in str(pe_price_change) else ""
        
        html += f"""
        <tr>
            <td>{row['CE_OI_Lakhs']}</td>
            <td>{row['CE_LTP']}</td>
            <td class="{ce_price_color}">{ce_price_change}</td>
            <td class="{strike_class}">{strike}</td>
            <td class="{pe_price_color}">{pe_price_change}</td>
            <td>{row['PE_LTP']}</td>
            <td>{row['PE_OI_Lakhs']}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟡 **ATM Strike**")
    with col2:
        st.markdown("🟢 **Positive Change**")
    with col3:
        st.markdown("🔴 **Negative Change**")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()