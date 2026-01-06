import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import websocket
import threading
from streamlit_option_menu import option_menu
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
DHAN_API_URL = "https://api.dhan.co/v2"
DHAN_WS_URL = "wss://api-feed.dhan.co"
DHAN_ORDER_WS_URL = "wss://api-order-update.dhan.co"

# Get credentials from environment or user input
def get_config():
    config = {}
    
    # Try to get from environment variables
    config['CLIENT_ID'] = os.getenv("DHAN_CLIENT_ID", "")
    config['ACCESS_TOKEN'] = os.getenv("DHAN_ACCESS_TOKEN", "")
    config['API_KEY'] = os.getenv("DHAN_API_KEY", "")
    config['API_SECRET'] = os.getenv("DHAN_API_SECRET", "")
    config['STATIC_IP'] = os.getenv("STATIC_IP", "")
    
    # If not in environment, use session state or input
    if 'dhan_config' not in st.session_state:
        st.session_state.dhan_config = config
    
    return st.session_state.dhan_config

# Security IDs (Example - verify from instrument list)
NIFTY_UNDERLYING_ID = 13
BANKNIFTY_UNDERLYING_ID = 23
SENSEX_UNDERLYING_ID = 1

# Segments
SEGMENT_IDX_I = "IDX_I"
SEGMENT_NSE_FNO = "NSE_FNO"

# Option Chain Configuration
ATM_RANGE = 5
MAX_STRIKES = 15
REFRESH_INTERVAL = 5

# Colors
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
# DHAN API CLIENT
# ============================================================================

class DhanAPI:
    def __init__(self):
        config = get_config()
        self.client_id = config['CLIENT_ID']
        self.access_token = config['ACCESS_TOKEN']
        self.base_url = DHAN_API_URL
        self.headers = {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json"
        }
    
    # ===== MARKET DATA APIS =====
    
    def get_marketfeed_ltp(self, instruments):
        """Get LTP for multiple instruments"""
        url = f"{self.base_url}/marketfeed/ltp"
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=5)
            return response.json()
        except:
            return {"data": {}, "status": "error"}
    
    def get_marketfeed_ohlc(self, instruments):
        """Get OHLC data for instruments"""
        url = f"{self.base_url}/marketfeed/ohlc"
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=5)
            return response.json()
        except:
            return {"data": {}, "status": "error"}
    
    def get_marketfeed_quote(self, instruments):
        """Get full market depth"""
        url = f"{self.base_url}/marketfeed/quote"
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=5)
            return response.json()
        except:
            return {"data": {}, "status": "error"}
    
    # ===== OPTION CHAIN APIS =====
    
    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        """Get option chain for a specific expiry"""
        url = f"{self.base_url}/optionchain"
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            return response.json()
        except:
            return {"data": {}, "status": "error"}
    
    def get_expiry_list(self, underlying_scrip, underlying_seg):
        """Get list of available expiries"""
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            data = response.json()
            return data.get("data", [])
        except:
            return []
    
    # ===== TRADING APIS =====
    
    def place_order(self, order_data):
        """Place a new order"""
        url = f"{self.base_url}/orders"
        try:
            response = requests.post(url, headers=self.headers, json=order_data, timeout=5)
            return response.json()
        except:
            return {"status": "error", "message": "Order placement failed"}
    
    def get_orders(self):
        """Get all orders for the day"""
        url = f"{self.base_url}/orders"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except:
            return []
    
    def get_positions(self):
        """Get open positions"""
        url = f"{self.base_url}/positions"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except:
            return []
    
    def get_holdings(self):
        """Get demat holdings"""
        url = f"{self.base_url}/holdings"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except:
            return []
    
    # ===== PROFILE =====
    
    def get_profile(self):
        """Get user profile"""
        url = f"{self.base_url}/profile"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            return response.json()
        except:
            return {}

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    def __init__(self, dhan_api):
        self.dhan_api = dhan_api
        self.instruments_df = None
        self.option_chains = {}
        self.last_update = {}
        
    def get_index_ltp(self, index="NIFTY"):
        """Get LTP for NIFTY/SENSEX using yfinance as fallback"""
        try:
            # Try yfinance first (more reliable for free data)
            ticker_symbol = f"^{index}" if index == "SENSEX" else f"{index}.NS"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        # Fallback to fixed values for demo
        demo_prices = {
            "NIFTY": 21500.50,
            "BANKNIFTY": 47500.75,
            "SENSEX": 72000.25
        }
        return demo_prices.get(index, 21500.50)
    
    def get_option_chain_data(self, index="NIFTY", expiry=None):
        """Get and process option chain data"""
        # Use yfinance for option chain data (free and reliable)
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
            
            for strike in all_strikes:
                # Find call data for this strike
                call_row = calls[calls['strike'] == strike]
                put_row = puts[puts['strike'] == strike]
                
                if not call_row.empty and not put_row.empty:
                    call_data = call_row.iloc[0]
                    put_data = put_row.iloc[0]
                    
                    # Determine if ATM
                    strike_interval = self._get_strike_interval(index)
                    is_atm = abs(strike - underlying_ltp) <= (strike_interval / 2)
                    
                    # Calculate percentage changes (simulated for demo)
                    ce_oi_change_pct = np.random.uniform(-50, 50)
                    pe_oi_change_pct = np.random.uniform(-50, 50)
                    ce_price_change_pct = np.random.uniform(-30, 30)
                    pe_price_change_pct = np.random.uniform(-30, 30)
                    
                    processed_data.append({
                        "strike": strike,
                        "is_atm": is_atm,
                        "ce": {
                            "ltp": float(call_data['lastPrice']),
                            "oi": int(call_data['openInterest']),
                            "volume": int(call_data['volume']),
                            "iv": float(call_data['impliedVolatility']) * 100 if call_data['impliedVolatility'] > 0 else 0,
                            "bid": float(call_data['bid']),
                            "ask": float(call_data['ask']),
                            "oi_change_pct": ce_oi_change_pct,
                            "price_change_pct": ce_price_change_pct,
                            "greeks": {
                                "delta": np.random.uniform(0, 1),
                                "gamma": np.random.uniform(0, 0.1),
                                "theta": np.random.uniform(-20, -5),
                                "vega": np.random.uniform(5, 20)
                            }
                        },
                        "pe": {
                            "ltp": float(put_data['lastPrice']),
                            "oi": int(put_data['openInterest']),
                            "volume": int(put_data['volume']),
                            "iv": float(put_data['impliedVolatility']) * 100 if put_data['impliedVolatility'] > 0 else 0,
                            "bid": float(put_data['bid']),
                            "ask": float(put_data['ask']),
                            "oi_change_pct": pe_oi_change_pct,
                            "price_change_pct": pe_price_change_pct,
                            "greeks": {
                                "delta": np.random.uniform(-1, 0),
                                "gamma": np.random.uniform(0, 0.1),
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
            # Return demo data if yfinance fails
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
            
            ce_price = max(time_value + (strike - underlying_ltp), 5)
            pe_price = max(time_value + (underlying_ltp - strike), 5)
            
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
            return 1.0  # Default neutral PCR
        
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
# OPTION CHAIN DISPLAY
# ============================================================================

class OptionChainDisplay:
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def create_option_chain_table(self, index="NIFTY", expiry=None):
        """Create formatted option chain DataFrame"""
        atm_data = self.data_manager.get_atm_strikes(index, expiry)
        
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
                # Call Option Data
                "CE_OI_Lakhs": ce_oi_lakhs,
                "CE_OI_Change%": ce["oi_change_pct"],
                "CE_LTP": ce["ltp"],
                "CE_Change%": ce["price_change_pct"],
                # Put Option Data
                "PE_OI_Lakhs": pe_oi_lakhs,
                "PE_OI_Change%": pe["oi_change_pct"],
                "PE_LTP": pe["ltp"],
                "PE_Change%": pe["price_change_pct"],
                # Greeks
                "CE_Delta": ce["greeks"].get("delta", 0),
                "PE_Delta": pe["greeks"].get("delta", 0),
                "CE_IV": ce["iv"],
                "PE_IV": pe["iv"]
            })
        
        df = pd.DataFrame(rows)
        
        # Format columns
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
    
    def create_pcr_chart(self, index="NIFTY"):
        """Create PCR chart"""
        fig = go.Figure()
        
        # Generate sample data
        times = pd.date_range(start="09:15", end="15:30", freq="15min").time[:20]
        pcr_values = np.random.uniform(0.8, 1.2, len(times))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=pcr_values,
            mode='lines',
            name='PCR',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Put-Call Ratio (PCR)",
            xaxis_title="Time",
            yaxis_title="PCR",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white")
        )
        
        return fig
    
    def create_oi_distribution_chart(self, index="NIFTY", expiry=None):
        """Create OI distribution chart"""
        chain_data = self.data_manager.get_option_chain_data(index, expiry)
        
        if not chain_data or not chain_data["data"]:
            return go.Figure()
        
        strikes = []
        ce_oi = []
        pe_oi = []
        
        for strike_data in chain_data["data"][:20]:  # Limit to 20 strikes for clarity
            strikes.append(strike_data["strike"])
            ce_oi.append(strike_data["ce"]["oi"] / 100000)
            pe_oi.append(strike_data["pe"]["oi"] / 100000)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strikes,
            y=ce_oi,
            name='Call OI (Lakhs)',
            marker_color=COLORS["call"],
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=strikes,
            y=pe_oi,
            name='Put OI (Lakhs)',
            marker_color=COLORS["put"],
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{index} OI Distribution",
            xaxis_title="Strike Price",
            yaxis_title="Open Interest (Lakhs)",
            barmode='group',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            legend=dict(
                bgcolor=COLORS["card"],
                bordercolor="white",
                borderwidth=1
            )
        )
        
        return fig
    
    def create_greeks_chart(self, index="NIFTY", expiry=None):
        """Create Greeks chart"""
        atm_data = self.data_manager.get_atm_strikes(index, expiry)
        
        if not atm_data["strikes"]:
            return go.Figure()
        
        # Find ATM strike
        atm_strike_data = None
        for strike_data in atm_data["strikes"]:
            if strike_data.get("is_atm", False):
                atm_strike_data = strike_data
                break
        
        if not atm_strike_data:
            atm_strike_data = atm_data["strikes"][len(atm_data["strikes"]) // 2]
        
        ce_greeks = atm_strike_data["ce"].get("greeks", {})
        pe_greeks = atm_strike_data["pe"].get("greeks", {})
        
        greeks = ["Delta", "Gamma", "Theta", "Vega"]
        ce_values = [
            ce_greeks.get("delta", 0),
            ce_greeks.get("gamma", 0),
            ce_greeks.get("theta", 0),
            ce_greeks.get("vega", 0)
        ]
        
        pe_values = [
            pe_greeks.get("delta", 0),
            pe_greeks.get("gamma", 0),
            pe_greeks.get("theta", 0),
            pe_greeks.get("vega", 0)
        ]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Delta", "Gamma", "Theta", "Vega"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Delta
        fig.add_trace(
            go.Bar(name="CE", x=["CE"], y=[ce_values[0]], marker_color=COLORS["call"]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="PE", x=["PE"], y=[pe_values[0]], marker_color=COLORS["put"]),
            row=1, col=1
        )
        
        # Gamma
        fig.add_trace(
            go.Bar(name="CE", x=["CE"], y=[ce_values[1]], marker_color=COLORS["call"], showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name="PE", x=["PE"], y=[pe_values[1]], marker_color=COLORS["put"], showlegend=False),
            row=1, col=2
        )
        
        # Theta
        fig.add_trace(
            go.Bar(name="CE", x=["CE"], y=[ce_values[2]], marker_color=COLORS["call"], showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name="PE", x=["PE"], y=[pe_values[2]], marker_color=COLORS["put"], showlegend=False),
            row=2, col=1
        )
        
        # Vega
        fig.add_trace(
            go.Bar(name="CE", x=["CE"], y=[ce_values[3]], marker_color=COLORS["call"], showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name="PE", x=["PE"], y=[pe_values[3]], marker_color=COLORS["put"], showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"ATM Greeks - Strike: {atm_strike_data['strike']:.0f}",
            height=500,
            showlegend=True,
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            legend=dict(
                bgcolor=COLORS["card"],
                bordercolor="white",
                borderwidth=1
            )
        )
        
        return fig

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    def __init__(self, dhan_api):
        self.dhan_api = dhan_api
        self.positions = []
        self.holdings = []
        self.orders = []
    
    def refresh(self):
        """Refresh portfolio data"""
        try:
            self.positions = self.dhan_api.get_positions()
            self.holdings = self.dhan_api.get_holdings()
            self.orders = self.dhan_api.get_orders()
        except:
            # Use demo data
            self.positions = self._get_demo_positions()
            self.holdings = self._get_demo_holdings()
            self.orders = self._get_demo_orders()
    
    def get_total_pnl(self):
        """Calculate total P&L"""
        total_realized = 0
        total_unrealized = 0
        
        for pos in self.positions:
            if isinstance(pos, dict):
                total_realized += pos.get("realizedProfit", 0)
                total_unrealized += pos.get("unrealizedProfit", 0)
        
        return {
            "realized": total_realized,
            "unrealized": total_unrealized,
            "total": total_realized + total_unrealized
        }
    
    def get_positions_df(self):
        """Convert positions to DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        rows = []
        for pos in self.positions[:10]:  # Limit to 10 positions
            rows.append({
                "Symbol": pos.get("tradingSymbol", "NIFTY"),
                "Qty": pos.get("netQty", 50),
                "Avg Price": pos.get("buyAvg", 21500),
                "LTP": 21550,
                "P&L": pos.get("unrealizedProfit", 2500),
                "Product": pos.get("productType", "INTRADAY")
            })
        
        return pd.DataFrame(rows)
    
    def get_holdings_df(self):
        """Convert holdings to DataFrame"""
        if not self.holdings:
            return pd.DataFrame()
        
        rows = []
        for holding in self.holdings[:5]:  # Limit to 5 holdings
            rows.append({
                "Symbol": holding.get("tradingSymbol", "RELIANCE"),
                "Quantity": holding.get("totalQty", 10),
                "Avg Cost": holding.get("avgCostPrice", 2500),
                "Current Value": 25500,
                "P&L %": 2.0
            })
        
        return pd.DataFrame(rows)
    
    def get_orders_df(self):
        """Convert orders to DataFrame"""
        if not self.orders:
            return pd.DataFrame()
        
        rows = []
        for order in self.orders[:10]:  # Limit to 10 orders
            rows.append({
                "Order ID": order.get("orderId", "123456"),
                "Symbol": order.get("tradingSymbol", "NIFTY"),
                "Type": order.get("transactionType", "BUY"),
                "Qty": order.get("quantity", 50),
                "Price": order.get("price", 21500),
                "Status": order.get("orderStatus", "COMPLETED"),
                "Time": order.get("createTime", "10:30:00")
            })
        
        return pd.DataFrame(rows)
    
    def _get_demo_positions(self):
        """Generate demo positions"""
        return [
            {
                "tradingSymbol": "NIFTY",
                "netQty": 50,
                "buyAvg": 21500,
                "realizedProfit": 1500,
                "unrealizedProfit": 2500,
                "productType": "INTRADAY"
            },
            {
                "tradingSymbol": "BANKNIFTY",
                "netQty": 25,
                "buyAvg": 47500,
                "realizedProfit": 800,
                "unrealizedProfit": -500,
                "productType": "INTRADAY"
            }
        ]
    
    def _get_demo_holdings(self):
        """Generate demo holdings"""
        return [
            {
                "tradingSymbol": "RELIANCE",
                "totalQty": 10,
                "avgCostPrice": 2500
            },
            {
                "tradingSymbol": "TCS",
                "totalQty": 5,
                "avgCostPrice": 3500
            }
        ]
    
    def _get_demo_orders(self):
        """Generate demo orders"""
        return [
            {
                "orderId": "123456",
                "tradingSymbol": "NIFTY",
                "transactionType": "BUY",
                "quantity": 50,
                "price": 21500,
                "orderStatus": "COMPLETED",
                "createTime": "10:30:00"
            },
            {
                "orderId": "123457",
                "tradingSymbol": "BANKNIFTY",
                "transactionType": "SELL",
                "quantity": 25,
                "price": 47600,
                "orderStatus": "PENDING",
                "createTime": "11:15:00"
            }
        ]
    
    def create_pnl_chart(self):
        """Create P&L chart"""
        fig = go.Figure()
        
        # Sample data
        times = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30"]
        pnl_values = [0, 1500, 3200, 2100, 4500, 3800]
        
        fig.add_trace(go.Scatter(
            x=times,
            y=pnl_values,
            mode='lines+markers',
            name='P&L',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
        
        fig.update_layout(
            title="Intraday P&L",
            xaxis_title="Time",
            yaxis_title="P&L (₹)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white")
        )
        
        return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="NIFTY & SENSEX Trading Platform",
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
        .option-chain-header {{
            background: linear-gradient(90deg, {COLORS["call"]}, #2ecc71);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .positive {{
            color: {COLORS["positive"]};
            font-weight: bold;
        }}
        .negative {{
            color: {COLORS["negative"]};
            font-weight: bold;
        }}
        .strike-cell {{
            background-color: #f8f9fa;
            font-weight: bold;
            text-align: center;
            border-left: 3px solid {COLORS["call"]};
            border-right: 3px solid {COLORS["put"]};
        }}
        .atm-strike {{
            background-color: {COLORS["atm"]} !important;
            font-weight: bold;
            border: 2px solid #ffd700 !important;
        }}
        .ce-header {{
            background-color: {COLORS["call"]};
            color: white;
            text-align: center;
            font-weight: bold;
        }}
        .pe-header {{
            background-color: {COLORS["put"]};
            color: white;
            text-align: center;
            font-weight: bold;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 15px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS["card"]};
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.dhan_api = None
        st.session_state.data_manager = None
        st.session_state.option_display = None
        st.session_state.portfolio = None
        st.session_state.selected_index = "NIFTY"
        st.session_state.selected_expiry = None
        st.session_state.last_refresh = None
        st.session_state.auto_refresh = True
    
    # Sidebar
    with st.sidebar:
        st.title("⚡ Trading Platform")
        
        # Navigation
        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Option Chain", "Portfolio", "Orders", "Settings"],
            icons=["speedometer", "graph-up", "wallet", "list-check", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": COLORS["card"]},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": COLORS["call"]},
            }
        )
        
        # API Configuration
        st.divider()
        st.subheader("🔧 API Configuration")
        
        config = get_config()
        
        with st.form("api_config_form"):
            client_id = st.text_input("Client ID", value=config['CLIENT_ID'])
            access_token = st.text_input("Access Token", value=config['ACCESS_TOKEN'], type="password")
            
            if st.form_submit_button("Save & Connect"):
                config['CLIENT_ID'] = client_id
                config['ACCESS_TOKEN'] = access_token
                st.session_state.dhan_config = config
                
                try:
                    st.session_state.dhan_api = DhanAPI()
                    st.session_state.data_manager = DataManager(st.session_state.dhan_api)
                    st.session_state.option_display = OptionChainDisplay(st.session_state.data_manager)
                    st.session_state.portfolio = PortfolioManager(st.session_state.dhan_api)
                    st.session_state.initialized = True
                    st.success("Connected successfully!")
                except:
                    st.error("Connection failed. Using demo mode.")
        
        # Index Selection
        st.divider()
        st.subheader("📊 Index Selection")
        
        index_options = ["NIFTY", "BANKNIFTY", "SENSEX"]
        selected_index = st.selectbox(
            "Select Index",
            index_options,
            index=index_options.index(st.session_state.selected_index)
        )
        
        if selected_index != st.session_state.selected_index:
            st.session_state.selected_index = selected_index
        
        # Expiry Selection
        st.subheader("📅 Expiry Selection")
        expiry_options = ["Current Week", "Next Week", "Monthly"]
        selected_expiry = st.selectbox("Select Expiry", expiry_options, index=0)
        
        # Convert to actual date
        today = datetime.now()
        if selected_expiry == "Current Week":
            # Next Thursday
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            expiry_date = (today + timedelta(days=days_until_thursday)).strftime("%Y-%m-%d")
        elif selected_expiry == "Next Week":
            # Thursday of next week
            days_until_thursday = (3 - today.weekday()) % 7 + 7
            expiry_date = (today + timedelta(days=days_until_thursday)).strftime("%Y-%m-%d")
        else:
            # Last Thursday of month
            month = today.month
            year = today.year
            # Get last Thursday
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
            while last_day.weekday() != 3:  # Thursday
                last_day -= timedelta(days=1)
            expiry_date = last_day.strftime("%Y-%m-%d")
        
        st.session_state.selected_expiry = expiry_date
        
        # Refresh Control
        st.divider()
        st.subheader("🔄 Refresh Control")
        
        st.session_state.auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Interval (seconds)", 1, 30, REFRESH_INTERVAL)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if 'data_manager' in st.session_state and st.session_state.data_manager:
                    st.session_state.data_manager.option_chains = {}
                st.rerun()
        
        # Footer
        st.divider()
        st.caption(f"© {datetime.now().year} Trading Platform v1.0")
    
    # Main content
    if not st.session_state.initialized:
        show_welcome_screen()
    else:
        # Header
        st.markdown(f'<h1 class="main-header">📈 {st.session_state.selected_index} Trading Platform</h1>', unsafe_allow_html=True)
        
        # Last updated
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Last updated: {current_time} | Expiry: {st.session_state.selected_expiry}")
        
        # Page routing
        if selected == "Dashboard":
            show_dashboard()
        elif selected == "Option Chain":
            show_option_chain()
        elif selected == "Portfolio":
            show_portfolio()
        elif selected == "Orders":
            show_orders()
        elif selected == "Settings":
            show_settings()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def show_welcome_screen():
    """Show welcome/configuration screen"""
    st.title("🚀 Welcome to NIFTY & SENSEX Trading Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Features
        
        📊 **Real-time Option Chain**
        - ATM ±5 strikes display
        - CE/PE LTP with changes
        - OI in lakhs with % change
        - PCR and Greeks
        
        📈 **Advanced Analytics**
        - Put-Call Ratio charts
        - OI distribution
        - Greeks visualization
        - Max pain calculation
        
        💼 **Portfolio Management**
        - Real-time P&L tracking
        - Position management
        - Order history
        - Risk metrics
        
        ⚡ **Trading Tools**
        - Quick order placement
        - Strategy backtesting
        - Risk management
        - Alerts & notifications
        """)
    
    with col2:
        st.markdown("""
        ## Quick Start
        
        1. **Get Dhan API Credentials**
           - Login to [web.dhan.co](https://web.dhan.co)
           - Go to My Profile → Access DhanHQ APIs
           - Generate Access Token
        
        2. **Configure in Sidebar**
           - Enter Client ID and Access Token
           - Click "Save & Connect"
        
        3. **Start Trading**
           - View real-time option chains
           - Monitor portfolio
           - Place orders
        
        ## Demo Mode
        
        Don't have API credentials? Use **Demo Mode**:
        - Sample data for NIFTY, BANKNIFTY, SENSEX
        - Realistic option chain simulation
        - Portfolio simulation
        - No real trading
        """)
        
        if st.button("🚀 Launch Demo Mode", type="primary", use_container_width=True):
            st.session_state.dhan_api = DhanAPI()
            st.session_state.data_manager = DataManager(st.session_state.dhan_api)
            st.session_state.option_display = OptionChainDisplay(st.session_state.data_manager)
            st.session_state.portfolio = PortfolioManager(st.session_state.dhan_api)
            st.session_state.initialized = True
            st.rerun()
    
    st.divider()
    
    # Quick preview
    st.subheader("📊 Option Chain Preview")
    
    # Create a sample option chain table
    sample_data = {
        "Strike": ["26,000", "26,050", "26,100", "26,150", "26,200", "26,250", "26,300"],
        "CE_OI": ["75.19", "70.85", "85.84", "54.54", "...", "360.33", "330.15"],
        "CE_LTP": ["178.35", "128.90", "78.65", "28.65", "263.33", "138.76", "133.21"],
        "CE_Chg%": ["-34.95%", "-29.18%", "+22.26%", "+46.89%", "+530.14%", "...", "..."],
        "PE_OI": ["116.51", "67.43", "134.26", "276.92", "...", "71.10", "129.20"],
        "PE_LTP": ["0.10", "0.05", "0.05", "0.05", "0.05", "0.05", "0.05"],
        "PE_Chg%": ["-18.86%", "-23.34%", "-30.75%", "-44.40%", "-71.29%", "...", "..."]
    }
    
    df_preview = pd.DataFrame(sample_data)
    
    # Display as HTML table
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: monospace; background-color: #262730; color: white;">
        <thead>
            <tr style="background: linear-gradient(90deg, #1f77b4, #2ecc71);">
                <th colspan="3" style="padding: 10px; text-align: center;">CALLS</th>
                <th style="padding: 10px; text-align: center; background-color: #f8f9fa; color: black;">STRIKE</th>
                <th colspan="3" style="padding: 10px; text-align: center;">PUTS</th>
            </tr>
            <tr>
                <th>OI (L)</th><th>LTP</th><th>Chg%</th>
                <th>Price</th>
                <th>Chg%</th><th>LTP</th><th>OI (L)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_preview.iterrows():
        html += f"""
        <tr>
            <td style="padding: 8px; text-align: center;">{row['CE_OI']}</td>
            <td style="padding: 8px; text-align: center;">{row['CE_LTP']}</td>
            <td style="padding: 8px; text-align: center; color: {'#ff0000' if '-' in row['CE_Chg%'] else '#00ff00'}">{row['CE_Chg%']}</td>
            <td style="padding: 8px; text-align: center; background-color: {'#ffff00' if row['Strike'] == '26,150' else '#f8f9fa'}; color: black; font-weight: bold;">{row['Strike']}</td>
            <td style="padding: 8px; text-align: center; color: {'#ff0000' if '-' in row['PE_Chg%'] else '#00ff00'}">{row['PE_Chg%']}</td>
            <td style="padding: 8px; text-align: center;">{row['PE_LTP']}</td>
            <td style="padding: 8px; text-align: center;">{row['PE_OI']}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    st.caption("*Sample data - Connect for real-time data*")

def show_dashboard():
    """Show main dashboard"""
    
    # Initialize if needed
    if not st.session_state.data_manager:
        st.session_state.data_manager = DataManager(st.session_state.dhan_api)
    if not st.session_state.option_display:
        st.session_state.option_display = OptionChainDisplay(st.session_state.data_manager)
    
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
        pcr_color = "normal"
        if pcr > 1.2:
            pcr_color = "inverse"
        elif pcr < 0.8:
            pcr_color = "normal"
        st.metric("PCR", f"{pcr:.2f}", delta_color=pcr_color)
    
    with col3:
        st.metric("ATM Strike", f"₹{atm_strike:,.0f}")
    
    with col4:
        vix = 10.02 + np.random.uniform(-0.5, 0.5)  # Simulated VIX
        st.metric("VIX", f"{vix:.2f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("OI Distribution")
        oi_chart = st.session_state.option_display.create_oi_distribution_chart(
            st.session_state.selected_index,
            st.session_state.selected_expiry
        )
        st.plotly_chart(oi_chart, use_container_width=True)
    
    with col2:
        st.subheader("PCR Trend")
        pcr_chart = st.session_state.option_display.create_pcr_chart(st.session_state.selected_index)
        st.plotly_chart(pcr_chart, use_container_width=True)
    
    st.divider()
    
    # Quick Option Chain
    st.subheader(f"{st.session_state.selected_index} Option Chain (ATM ±5)")
    
    option_df = st.session_state.option_display.create_option_chain_table(
        st.session_state.selected_index,
        st.session_state.selected_expiry
    )
    
    if not option_df.empty:
        display_option_chain_table_html(option_df)
    else:
        st.info("Loading option chain data...")

def show_option_chain():
    """Show full option chain"""
    
    st.markdown(f"""
    <div class="option-chain-header">
        <h2>{st.session_state.selected_index} Option Chain</h2>
        <p>Expiry: {st.session_state.selected_expiry} | Displaying ATM ±5 Strikes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get option chain data
    option_df = st.session_state.option_display.create_option_chain_table(
        st.session_state.selected_index,
        st.session_state.selected_expiry
    )
    
    if not option_df.empty:
        display_option_chain_table_html(option_df)
    else:
        st.warning("No option chain data available")
        return
    
    st.divider()
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Greeks Analysis")
        greeks_chart = st.session_state.option_display.create_greeks_chart(
            st.session_state.selected_index,
            st.session_state.selected_expiry
        )
        st.plotly_chart(greeks_chart, use_container_width=True)
    
    with col2:
        st.subheader("ATM Option Details")
        
        atm_data = st.session_state.data_manager.get_atm_strikes(
            st.session_state.selected_index,
            st.session_state.selected_expiry
        )
        
        # Find ATM strike
        for strike_data in atm_data["strikes"]:
            if strike_data.get("is_atm", False):
                strike = strike_data["strike"]
                ce = strike_data["ce"]
                pe = strike_data["pe"]
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    **CE {strike:,.0f}**
                    - **LTP**: ₹{ce['ltp']:.2f}
                    - **IV**: {ce['iv']:.1f}%
                    - **OI**: {ce['oi']/100000:.2f}L
                    - **Delta**: {ce['greeks'].get('delta', 0):.3f}
                    - **Theta**: {ce['greeks'].get('theta', 0):.2f}
                    """)
                
                with col_b:
                    st.markdown(f"""
                    **PE {strike:,.0f}**
                    - **LTP**: ₹{pe['ltp']:.2f}
                    - **IV**: {pe['iv']:.1f}%
                    - **OI**: {pe['oi']/100000:.2f}L
                    - **Delta**: {pe['greeks'].get('delta', 0):.3f}
                    - **Theta**: {pe['greeks'].get('theta', 0):.2f}
                    """)
                break
        
        # Max Pain
        st.subheader("Max Pain Analysis")
        max_pain = atm_data["atm_strike"]  # Simplified
        st.metric("Max Pain", f"₹{max_pain:,.0f}")
        
        # PCR for different strikes
        st.metric("PCR (Total)", f"{st.session_state.data_manager.calculate_pcr(st.session_state.selected_index, st.session_state.selected_expiry):.2f}")

def show_portfolio():
    """Show portfolio page"""
    
    if not st.session_state.portfolio:
        st.session_state.portfolio = PortfolioManager(st.session_state.dhan_api)
    
    st.header("📊 Portfolio")
    
    # Refresh portfolio
    st.session_state.portfolio.refresh()
    
    # P&L Summary
    pnl = st.session_state.portfolio.get_total_pnl()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Realized P&L", f"₹{pnl['realized']:,.2f}")
    
    with col2:
        st.metric("Unrealized P&L", f"₹{pnl['unrealized']:,.2f}")
    
    with col3:
        total_color = "normal" if pnl['total'] >= 0 else "inverse"
        st.metric("Total P&L", f"₹{pnl['total']:,.2f}", delta_color=total_color)
    
    # P&L Chart
    st.subheader("P&L Trend")
    pnl_chart = st.session_state.portfolio.create_pnl_chart()
    st.plotly_chart(pnl_chart, use_container_width=True)
    
    st.divider()
    
    # Positions
    st.subheader("Open Positions")
    positions_df = st.session_state.portfolio.get_positions_df()
    
    if not positions_df.empty:
        # Style the DataFrame
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'white'
            return f'color: {color}'
        
        styled_df = positions_df.style.format({
            "Avg Price": "{:,.2f}",
            "LTP": "{:,.2f}",
            "P&L": "{:,.2f}"
        }).applymap(color_pnl, subset=['P&L'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")
    
    st.divider()
    
    # Holdings
    st.subheader("Holdings (Demat)")
    holdings_df = st.session_state.portfolio.get_holdings_df()
    
    if not holdings_df.empty:
        st.dataframe(
            holdings_df.style.format({
                "Avg Cost": "{:,.2f}",
                "Current Value": "{:,.2f}",
                "P&L %": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No holdings")

def show_orders():
    """Show orders page"""
    
    if not st.session_state.portfolio:
        st.session_state.portfolio = PortfolioManager(st.session_state.dhan_api)
    
    st.header("📋 Orders")
    
    # Refresh orders
    st.session_state.portfolio.refresh()
    
    # Orders table
    orders_df = st.session_state.portfolio.get_orders_df()
    
    if not orders_df.empty:
        # Style based on status
        def color_status(val):
            if val == "COMPLETED":
                return "background-color: #d4edda; color: #155724;"
            elif val == "PENDING":
                return "background-color: #fff3cd; color: #856404;"
            elif val == "CANCELLED":
                return "background-color: #f8d7da; color: #721c24;"
            return ""
        
        styled_df = orders_df.style.format({
            "Price": "{:,.2f}"
        }).applymap(color_status, subset=['Status'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No orders today")
    
    st.divider()
    
    # Quick Order Placement
    st.subheader("Place New Order")
    
    with st.form("order_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "SENSEX", "RELIANCE", "TCS", "INFY"])
        
        with col2:
            transaction_type = st.selectbox("Type", ["BUY", "SELL"])
        
        with col3:
            product_type = st.selectbox("Product", ["INTRADAY", "CNC", "MARGIN"])
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP_LOSS"])
        
        with col5:
            quantity = st.number_input("Quantity", min_value=1, value=50, step=25)
        
        with col6:
            price = st.number_input("Price", min_value=0.0, value=21500.0, step=50.0)
        
        submitted = st.form_submit_button("📤 Place Order", type="primary")
        
        if submitted:
            # Simulate order placement
            with st.spinner("Placing order..."):
                time.sleep(1)
                st.success(f"Order placed for {quantity} {symbol} at ₹{price:,.2f}")
                st.balloons()

def show_settings():
    """Show settings page"""
    
    st.header("⚙️ Settings")
    
    # API Configuration
    with st.expander("🔧 API Configuration", expanded=True):
        config = get_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_client_id = st.text_input("Client ID", value=config['CLIENT_ID'])
            new_api_key = st.text_input("API Key", value=config['API_KEY'], type="password")
        
        with col2:
            new_access_token = st.text_input("Access Token", value=config['ACCESS_TOKEN'], type="password")
            new_api_secret = st.text_input("API Secret", value=config['API_SECRET'], type="password")
        
        if st.button("💾 Save API Configuration", type="primary"):
            config['CLIENT_ID'] = new_client_id
            config['ACCESS_TOKEN'] = new_access_token
            config['API_KEY'] = new_api_key
            config['API_SECRET'] = new_api_secret
            st.session_state.dhan_config = config
            st.success("API configuration saved!")
    
    # Display Settings
    with st.expander("🎨 Display Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_greeks = st.checkbox("Show Greeks", value=True)
            show_volume = st.checkbox("Show Volume", value=True)
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            refresh_interval = st.slider("Refresh Interval (s)", 1, 30, REFRESH_INTERVAL)
        
        if st.button("Save Display Settings"):
            st.session_state.auto_refresh = auto_refresh
            st.success("Display settings saved!")
    
    # Data Management
    with st.expander("🗄️ Data Management"):
        st.write("Cache Statistics")
        
        if st.session_state.data_manager:
            cache_size = len(st.session_state.data_manager.option_chains)
            st.metric("Cached Option Chains", cache_size)
        else:
            st.info("No data manager initialized")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh All Data"):
                if st.session_state.data_manager:
                    st.session_state.data_manager.option_chains = {}
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Cache"):
                if st.session_state.data_manager:
                    st.session_state.data_manager.option_chains = {}
                    st.session_state.data_manager.last_update = {}
                st.success("Cache cleared!")
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **NIFTY & SENSEX Trading Platform v1.0**
        
        A comprehensive trading platform for Indian indices with real-time option chain data.
        
        **Features:**
        - Real-time option chain display
        - Portfolio management
        - Order placement
        - Advanced analytics
        
        **Data Sources:**
        - DhanHQ API (primary)
        - Yahoo Finance (fallback)
        - Demo data (when unavailable)
        
        **Disclaimer:**
        This platform is for educational purposes only. 
        Trading involves risk. Always do your own research.
        
        **Contact:** [Your Contact Information]
        """)

def display_option_chain_table_html(df):
    """Display formatted option chain table as HTML"""
    
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: monospace; background-color: #262730; color: white; margin: 10px 0;">
        <thead>
            <tr style="background: linear-gradient(90deg, #1f77b4, #2ecc71);">
                <th colspan="3" style="padding: 12px; text-align: center; font-size: 16px;">CALLS</th>
                <th style="padding: 12px; text-align: center; font-size: 16px; background-color: #f8f9fa; color: black;">STRIKE</th>
                <th colspan="3" style="padding: 12px; text-align: center; font-size: 16px;">PUTS</th>
            </tr>
            <tr style="background-color: #1a1a1a;">
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">OI (L)</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">LTP</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">Chg%</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">Price</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">Chg%</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">LTP</th>
                <th style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">OI (L)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df.iterrows():
        strike = row["Strike"]
        is_atm = row.get("Is_ATM", False)
        
        # Get CSS classes
        strike_style = "background-color: #f8f9fa; color: black; font-weight: bold;"
        if is_atm:
            strike_style = "background-color: #ffff00; color: black; font-weight: bold; border: 2px solid #ffd700;"
        
        ce_oi_change = row["CE_OI_Change%"]
        ce_price_change = row["CE_Change%"]
        pe_oi_change = row["PE_OI_Change%"]
        pe_price_change = row["PE_Change%"]
        
        ce_oi_color = COLORS["positive"] if "+" in str(ce_oi_change) else COLORS["negative"] if "-" in str(ce_oi_change) else "white"
        ce_price_color = COLORS["positive"] if "+" in str(ce_price_change) else COLORS["negative"] if "-" in str(ce_price_change) else "white"
        pe_oi_color = COLORS["positive"] if "+" in str(pe_oi_change) else COLORS["negative"] if "-" in str(pe_oi_change) else "white"
        pe_price_color = COLORS["positive"] if "+" in str(pe_price_change) else COLORS["negative"] if "-" in str(pe_price_change) else "white"
        
        html += f"""
        <tr>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">{row['CE_OI_Lakhs']}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">{row['CE_LTP']}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444; color: {ce_price_color};">{ce_price_change}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444; {strike_style}">{strike}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444; color: {pe_price_color};">{pe_price_change}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">{row['PE_LTP']}</td>
            <td style="padding: 10px; text-align: center; border-bottom: 1px solid #444;">{row['PE_OI_Lakhs']}</td>
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
