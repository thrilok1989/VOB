"""
Nifty Option Screener v7.0 - Simplified with Market Depth Analysis
All analysis preserved + Market Depth + Iceberg Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz
from math import log, sqrt
from scipy.stats import norm
import json

# -----------------------
#  CONFIG
# -----------------------
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

# Secrets (simplified)
try:
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except:
    st.error("❌ Missing Dhan credentials")
    st.stop()

DHAN_BASE_URL = "https://api.dhan.co"
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8

# -----------------------
#  UTILITY FUNCTIONS
# -----------------------
def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def strike_gap_from_series(series):
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 50
    mode = diffs.mode()
    return int(mode.iloc[0]) if not mode.empty else int(diffs.median())

# Black-Scholes Greeks
def bs_d1(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def bs_delta(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return 1.0 if (option_type=="call" and S>K) else (-1.0 if (option_type=="put" and S<K) else 0.0)
    d1 = bs_d1(S,K,r,sigma,tau)
    if option_type == "call":
        return norm.cdf(d1)
    return -norm.cdf(-d1)

def bs_gamma(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

# -----------------------
#  MARKET DEPTH ANALYSIS
# -----------------------
def get_market_depth():
    """Fetch market depth from Dhan API"""
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/depth"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        payload = {"IDX_I": [13]}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("IDX_I", {}).get("13", {}).get("depth", {})
    except Exception as e:
        st.warning(f"Depth fetch failed: {e}")
    return None

def analyze_market_depth(depth_data, spot_price):
    """Analyze market depth for iceberg orders and key levels"""
    if not depth_data:
        return None
    
    analysis = {
        'buy_side': {'levels': [], 'total_volume': 0, 'iceberg_count': 0},
        'sell_side': {'levels': [], 'total_volume': 0, 'iceberg_count': 0}
    }
    
    # Define iceberg threshold (large hidden orders)
    ICEBERG_THRESHOLD = 50000  # 50k shares
    
    # Analyze buy side (bids)
    if 'buy' in depth_data:
        for i, bid in enumerate(depth_data['buy'][:10]):  # Top 10 bids
            if isinstance(bid, list) and len(bid) >= 2:
                price, qty = bid[0], bid[1]
                analysis['buy_side']['levels'].append({
                    'price': price,
                    'quantity': qty,
                    'depth_level': i+1,
                    'distance_from_spot': spot_price - price,
                    'is_iceberg': qty > ICEBERG_THRESHOLD
                })
                analysis['buy_side']['total_volume'] += qty
                if qty > ICEBERG_THRESHOLD:
                    analysis['buy_side']['iceberg_count'] += 1
    
    # Analyze sell side (asks)
    if 'sell' in depth_data:
        for i, ask in enumerate(depth_data['sell'][:10]):  # Top 10 asks
            if isinstance(ask, list) and len(ask) >= 2:
                price, qty = ask[0], ask[1]
                analysis['sell_side']['levels'].append({
                    'price': price,
                    'quantity': qty,
                    'depth_level': i+1,
                    'distance_from_spot': price - spot_price,
                    'is_iceberg': qty > ICEBERG_THRESHOLD
                })
                analysis['sell_side']['total_volume'] += qty
                if qty > ICEBERG_THRESHOLD:
                    analysis['sell_side']['iceberg_count'] += 1
    
    # Calculate support/resistance from depth
    support_levels = []
    resistance_levels = []
    
    # Strong bids near spot are support
    for bid in analysis['buy_side']['levels'][:3]:
        if bid['distance_from_spot'] < 50 and bid['quantity'] > 10000:
            support_levels.append({
                'price': bid['price'],
                'strength': bid['quantity'] / 1000,
                'type': 'depth_bid',
                'iceberg': bid['is_iceberg']
            })
    
    # Strong asks near spot are resistance
    for ask in analysis['sell_side']['levels'][:3]:
        if ask['distance_from_spot'] < 50 and ask['quantity'] > 10000:
            resistance_levels.append({
                'price': ask['price'],
                'strength': ask['quantity'] / 1000,
                'type': 'depth_ask',
                'iceberg': ask['is_iceberg']
            })
    
    # Depth pressure analysis
    total_bid_volume = analysis['buy_side']['total_volume']
    total_ask_volume = analysis['sell_side']['total_volume']
    depth_pressure = (total_bid_volume - total_ask_volume) / max(total_bid_volume + total_ask_volume, 1)
    
    analysis['depth_summary'] = {
        'total_bid_volume': total_bid_volume,
        'total_ask_volume': total_ask_volume,
        'depth_pressure': depth_pressure,
        'iceberg_orders': analysis['buy_side']['iceberg_count'] + analysis['sell_side']['iceberg_count'],
        'support_levels': support_levels,
        'resistance_levels': resistance_levels
    }
    
    return analysis

# -----------------------
#  SELLER'S ANALYSIS FUNCTIONS
# -----------------------
def calculate_seller_max_pain(df):
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))
        
        ce_pain = ce_oi * max(0, ce_ltp) if strike < df["strikePrice"].mean() else 0
        pe_pain = pe_oi * max(0, pe_ltp) if strike > df["strikePrice"].mean() else 0
        
        pain = ce_pain + pe_pain
        pain_dict[strike] = pain
    
    if pain_dict:
        return min(pain_dict, key=pain_dict.get)
    return None

def calculate_seller_market_bias(merged_df, spot, atm_strike):
    polarity = 0.0
    
    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))
        
        if strike < atm_strike:
            if chg_ce > 0:
                polarity -= 2.0  # ITM CALL writing = very bearish
            elif chg_ce < 0:
                polarity += 1.5  # ITM CALL buying back = bullish
        
        elif strike > atm_strike:
            if chg_ce > 0:
                polarity -= 0.7  # OTM CALL writing = mildly bearish
            elif chg_ce < 0:
                polarity += 0.5  # OTM CALL buying back = mildly bullish
        
        if strike > atm_strike:
            if chg_pe > 0:
                polarity += 2.0  # ITM PUT writing = very bullish
            elif chg_pe < 0:
                polarity -= 1.5  # ITM PUT buying back = bearish
        
        elif strike < atm_strike:
            if chg_pe > 0:
                polarity += 0.7  # OTM PUT writing = mildly bullish
            elif chg_pe < 0:
                polarity -= 0.5  # OTM PUT buying back = mildly bearish
    
    # PCR adjustment
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:
            polarity += 1.0
        elif pcr < 0.5:
            polarity -= 1.0
    
    # Determine bias
    if polarity > 3.0:
        return {"bias": "STRONG BULLISH SELLERS 🚀", "polarity": polarity, "color": "#00ff88"}
    elif polarity > 1.0:
        return {"bias": "BULLISH SELLERS 📈", "polarity": polarity, "color": "#00cc66"}
    elif polarity < -3.0:
        return {"bias": "STRONG BEARISH SELLERS 🐻", "polarity": polarity, "color": "#ff4444"}
    elif polarity < -1.0:
        return {"bias": "BEARISH SELLERS 📉", "polarity": polarity, "color": "#ff6666"}
    else:
        return {"bias": "NEUTRAL SELLERS ⚖️", "polarity": polarity, "color": "#66b3ff"}

def analyze_oi_pcr_metrics(merged_df, spot, atm_strike):
    """Comprehensive OI and PCR analysis"""
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    
    pcr_total = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    # PCR Interpretation
    if pcr_total > 2.0:
        pcr_interpretation = "EXTREME PUT SELLING"
        pcr_sentiment = "STRONGLY BULLISH"
        pcr_color = "#00ff88"
    elif pcr_total > 1.5:
        pcr_interpretation = "HEAVY PUT SELLING"
        pcr_sentiment = "BULLISH"
        pcr_color = "#00cc66"
    elif pcr_total > 1.2:
        pcr_interpretation = "MODERATE PUT SELLING"
        pcr_sentiment = "MILD BULLISH"
        pcr_color = "#66ff66"
    elif pcr_total > 0.8:
        pcr_interpretation = "BALANCED"
        pcr_sentiment = "NEUTRAL"
        pcr_color = "#66b3ff"
    elif pcr_total > 0.5:
        pcr_interpretation = "MODERATE CALL SELLING"
        pcr_sentiment = "MILD BEARISH"
        pcr_color = "#ff9900"
    elif pcr_total > 0.3:
        pcr_interpretation = "HEAVY CALL SELLING"
        pcr_sentiment = "BEARISH"
        pcr_color = "#ff4444"
    else:
        pcr_interpretation = "EXTREME CALL SELLING"
        pcr_sentiment = "STRONGLY BEARISH"
        pcr_color = "#ff0000"
    
    return {
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_ce_chg": total_ce_chg,
        "total_pe_chg": total_pe_chg,
        "pcr_total": pcr_total,
        "pcr_interpretation": pcr_interpretation,
        "pcr_sentiment": pcr_sentiment,
        "pcr_color": pcr_color
    }

# -----------------------
#  DATA FETCHING
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_spot_price():
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        payload = {"IDX_I": [13]}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        if data.get("status") == "success":
            return float(data.get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price", 0))
    except:
        pass
    return 0.0

@st.cache_data(ttl=300)
def get_expiry_list():
    try:
        url = f"{DHAN_BASE_URL}/v2/optionchain/expirylist"
        payload = {"UnderlyingScrip":13,"UnderlyingSeg":"IDX_I"}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",[])
    except:
        pass
    return []

@st.cache_data(ttl=10)
def fetch_dhan_option_chain(expiry_date):
    try:
        url = f"{DHAN_BASE_URL}/v2/optionchain"
        payload = {"UnderlyingScrip":13,"UnderlyingSeg":"IDX_I","Expiry":expiry_date}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",{})
    except:
        pass
    return None

def parse_dhan_option_chain(chain_data):
    if not chain_data:
        return pd.DataFrame(), pd.DataFrame()
    
    ce_rows, pe_rows = [], []
    for strike_str, strike_data in chain_data.get("oc", {}).items():
        try:
            strike = int(float(strike_str))
        except:
            continue
        
        ce = strike_data.get("ce")
        pe = strike_data.get("pe")
        
        if ce:
            ce_rows.append({
                "strikePrice": strike,
                "OI_CE": safe_int(ce.get("oi",0)),
                "Chg_OI_CE": safe_int(ce.get("oi",0)) - safe_int(ce.get("previous_oi",0)),
                "Vol_CE": safe_int(ce.get("volume",0)),
                "LTP_CE": safe_float(ce.get("last_price",0.0)),
                "IV_CE": safe_float(ce.get("implied_volatility", np.nan))
            })
        
        if pe:
            pe_rows.append({
                "strikePrice": strike,
                "OI_PE": safe_int(pe.get("oi",0)),
                "Chg_OI_PE": safe_int(pe.get("oi",0)) - safe_int(pe.get("previous_oi",0)),
                "Vol_PE": safe_int(pe.get("volume",0)),
                "LTP_PE": safe_float(pe.get("last_price",0.0)),
                "IV_PE": safe_float(pe.get("implied_volatility", np.nan))
            })
    
    return pd.DataFrame(ce_rows), pd.DataFrame(pe_rows)

# -----------------------
#  MAIN APP
# -----------------------
st.set_page_config(page_title="Nifty Screener v7 - Simplified", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .seller-bullish { color: #00ff88 !important; }
    .seller-bearish { color: #ff4444 !important; }
    .seller-neutral { color: #66b3ff !important; }
    h1, h2, h3 { color: #ff66cc !important; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
        text-align: center;
    }
    
    .depth-buy { color: #00ff88; }
    .depth-sell { color: #ff4444; }
    .iceberg { color: #ff00ff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 NIFTY Option Screener v7.0")
st.markdown(f"**🕐 IST:** {get_ist_datetime_str()}")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Settings")
    st.markdown("**Seller Logic:**")
    st.markdown("- 📉 CALL Writing = BEARISH")
    st.markdown("- 📈 PUT Writing = BULLISH")
    st.markdown("- 🔄 Unwinding = Reversal signal")
    st.markdown("---")
    st.markdown("### 🎯 Market Depth")
    st.markdown("**Iceberg Detection:** >50k shares")
    st.markdown("**Analysis:** Support/Resistance from depth")

# Fetch data
spot = get_nifty_spot_price()
if spot == 0:
    st.error("Unable to fetch NIFTY spot")
    st.stop()

expiries = get_expiry_list()
if not expiries:
    st.error("Unable to fetch expiry list")
    st.stop()

expiry = st.selectbox("Select expiry", expiries, index=0)

# Fetch option chain
chain = fetch_dhan_option_chain(expiry)
if chain is None:
    st.error("Failed to fetch option chain")
    st.stop()

df_ce, df_pe = parse_dhan_option_chain(chain)
if df_ce.empty or df_pe.empty:
    st.error("Insufficient data")
    st.stop()

# Merge data
merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")
merged["strikePrice"] = merged["strikePrice"].astype(int)

# Filter ATM window
strike_gap = strike_gap_from_series(merged["strikePrice"])
atm_strike = min(merged["strikePrice"].tolist(), key=lambda x: abs(x - spot))
lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)
merged = merged[(merged["strikePrice"]>=lower) & (merged["strikePrice"]<=upper)]

# Calculate Greeks
tau = 7.0/365.0  # Simplified time decay
for i, row in merged.iterrows():
    strike = row["strikePrice"]
    iv_ce = safe_float(row.get("IV_CE", np.nan)) / 100.0 if not np.isnan(row.get("IV_CE")) else 0.25
    iv_pe = safe_float(row.get("IV_PE", np.nan)) / 100.0 if not np.isnan(row.get("IV_PE")) else 0.25
    
    try:
        gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, iv_ce, tau)
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, iv_pe, tau)
    except:
        gamma_ce = gamma_pe = 0.0
    
    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    merged.at[i,"GEX_CE"] = gamma_ce * notional * oi_ce
    merged.at[i,"GEX_PE"] = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_Net"] = merged.at[i,"GEX_CE"] + merged.at[i,"GEX_PE"]

# ====================
# MARKET DEPTH ANALYSIS
# ====================
st.markdown("---")
st.markdown("## 📊 MARKET DEPTH ANALYSIS (Iceberg Detection)")

depth_data = get_market_depth()
depth_analysis = analyze_market_depth(depth_data, spot) if depth_data else None

if depth_analysis:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bid Volume", f"{depth_analysis['depth_summary']['total_bid_volume']:,}")
        st.metric("Iceberg Orders (Bid)", depth_analysis['buy_side']['iceberg_count'])
    
    with col2:
        st.metric("Total Ask Volume", f"{depth_analysis['depth_summary']['total_ask_volume']:,}")
        st.metric("Iceberg Orders (Ask)", depth_analysis['sell_side']['iceberg_count'])
    
    with col3:
        pressure = depth_analysis['depth_summary']['depth_pressure']
        pressure_color = "#00ff88" if pressure > 0.1 else "#ff4444" if pressure < -0.1 else "#66b3ff"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color:#cccccc;">Depth Pressure</div>
            <div style="font-size: 1.5rem; color:{pressure_color}; font-weight:700;">
                {pressure:+.2f}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                {'Buying Pressure' if pressure > 0.1 else 'Selling Pressure' if pressure < -0.1 else 'Balanced'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display depth levels
    st.markdown("### 🎯 Depth Support/Resistance Levels")
    
    if depth_analysis['depth_summary']['support_levels']:
        st.markdown("#### 🛡️ Support from Depth (Strong Bids)")
        for support in depth_analysis['depth_summary']['support_levels'][:3]:
            iceberg_mark = "🚨 ICEBERG " if support['iceberg'] else ""
            st.markdown(f"- **₹{support['price']:.2f}** | Strength: {support['strength']:.1f}k {iceberg_mark}")
    
    if depth_analysis['depth_summary']['resistance_levels']:
        st.markdown("#### ⚡ Resistance from Depth (Strong Asks)")
        for resistance in depth_analysis['depth_summary']['resistance_levels'][:3]:
            iceberg_mark = "🚨 ICEBERG " if resistance['iceberg'] else ""
            st.markdown(f"- **₹{resistance['price']:.2f}** | Strength: {resistance['strength']:.1f}k {iceberg_mark}")
    
    # Top 5 depth levels display
    st.markdown("### 📈 Top 5 Depth Levels")
    
    depth_col1, depth_col2 = st.columns(2)
    
    with depth_col1:
        st.markdown("##### 🟢 Buy Side (Bids)")
        for i, bid in enumerate(depth_analysis['buy_side']['levels'][:5], 1):
            iceberg = "🧊 " if bid['is_iceberg'] else ""
            st.markdown(f"{i}. **₹{bid['price']:.2f}** | Qty: {bid['quantity']:,} {iceberg}")
    
    with depth_col2:
        st.markdown("##### 🔴 Sell Side (Asks)")
        for i, ask in enumerate(depth_analysis['sell_side']['levels'][:5], 1):
            iceberg = "🧊 " if ask['is_iceberg'] else ""
            st.markdown(f"{i}. **₹{ask['price']:.2f}** | Qty: {ask['quantity']:,} {iceberg}")
else:
    st.warning("Market depth data unavailable")

# ====================
# SELLER ANALYSIS
# ====================
st.markdown("---")
st.markdown("## 🎯 SELLER'S ANALYSIS")

# Calculate seller metrics
seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)
seller_max_pain = calculate_seller_max_pain(merged)
oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Spot Price", f"₹{spot:.2f}")
    st.metric("ATM Strike", f"₹{atm_strike}")

with col2:
    st.metric("Seller Bias", seller_bias_result["bias"])
    st.metric("Polarity", f"{seller_bias_result['polarity']:.2f}")

with col3:
    st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
    st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])

with col4:
    total_gex = merged["GEX_Net"].sum()
    st.metric("Total GEX", f"₹{int(total_gex):,}")
    if seller_max_pain:
        st.metric("Max Pain", f"₹{seller_max_pain:,}")

# OI Analysis
st.markdown("### 📊 OPEN INTEREST ANALYSIS")

oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)

with oi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#ff4444; font-weight:bold;">CALL OI</div>
        <div style="font-size: 1.8rem; color:#ff4444;">{oi_pcr_metrics['total_ce_oi']:,}</div>
        <div style="font-size: 0.9rem;">Δ: {oi_pcr_metrics['total_ce_chg']:+,}</div>
    </div>
    """, unsafe_allow_html=True)

with oi_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#00ff88; font-weight:bold;">PUT OI</div>
        <div style="font-size: 1.8rem; color:#00ff88;">{oi_pcr_metrics['total_pe_oi']:,}</div>
        <div style="font-size: 0.9rem;">Δ: {oi_pcr_metrics['total_pe_chg']:+,}</div>
    </div>
    """, unsafe_allow_html=True)

with oi_col3:
    pcr_color = oi_pcr_metrics['pcr_color']
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-weight:bold;">PUT-CALL RATIO</div>
        <div style="font-size: 2rem; color:{pcr_color}; font-weight:900;">
            {oi_pcr_metrics['pcr_total']:.2f}
        </div>
        <div style="font-size: 0.9rem; color:{pcr_color};">
            {oi_pcr_metrics['pcr_interpretation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with oi_col4:
    call_sellers = (merged["Chg_OI_CE"] > 0).sum()
    put_sellers = (merged["Chg_OI_PE"] > 0).sum()
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-weight:bold;">ACTIVE SELLERS</div>
        <div style="font-size: 1.5rem; color:#ff4444;">CALL: {call_sellers}</div>
        <div style="font-size: 1.5rem; color:#00ff88;">PUT: {put_sellers}</div>
    </div>
    """, unsafe_allow_html=True)

# Combined Support/Resistance
st.markdown("---")
st.markdown("## 🎯 COMBINED SUPPORT & RESISTANCE")

# Find nearest strikes for support/resistance
strikes = merged["strikePrice"].tolist()
supports = [s for s in strikes if s < spot]
resistances = [s for s in strikes if s > spot]

nearest_support = max(supports) if supports else None
nearest_resistance = min(resistances) if resistances else None

col_sup, col_res = st.columns(2)

with col_sup:
    st.markdown("### 🛡️ SUPPORT LEVELS")
    
    # Option-based support
    if nearest_support:
        sup_row = merged[merged["strikePrice"] == nearest_support].iloc[0]
        sup_put_oi = safe_int(sup_row.get("OI_PE", 0))
        st.markdown(f"**Option Support:** ₹{nearest_support:,}")
        st.markdown(f"PUT OI: {sup_put_oi:,}")
    
    # Depth-based support
    if depth_analysis and depth_analysis['depth_summary']['support_levels']:
        st.markdown("**Depth Support:**")
        for support in depth_analysis['depth_summary']['support_levels'][:2]:
            iceberg = "🧊 " if support['iceberg'] else ""
            st.markdown(f"- ₹{support['price']:.2f} ({support['strength']:.0f}k) {iceberg}")

with col_res:
    st.markdown("### ⚡ RESISTANCE LEVELS")
    
    # Option-based resistance
    if nearest_resistance:
        res_row = merged[merged["strikePrice"] == nearest_resistance].iloc[0]
        res_call_oi = safe_int(res_row.get("OI_CE", 0))
        st.markdown(f"**Option Resistance:** ₹{nearest_resistance:,}")
        st.markdown(f"CALL OI: {res_call_oi:,}")
    
    # Depth-based resistance
    if depth_analysis and depth_analysis['depth_summary']['resistance_levels']:
        st.markdown("**Depth Resistance:**")
        for resistance in depth_analysis['depth_summary']['resistance_levels'][:2]:
            iceberg = "🧊 " if resistance['iceberg'] else ""
            st.markdown(f"- ₹{resistance['price']:.2f} ({resistance['strength']:.0f}k) {iceberg}")

# ====================
# DETAILED DATA VIEW
# ====================
st.markdown("---")
st.markdown("## 📊 DETAILED OPTION CHAIN DATA")

# Show filtered data
display_cols = ["strikePrice", "OI_CE", "Chg_OI_CE", "OI_PE", "Chg_OI_PE", "IV_CE", "IV_PE", "GEX_Net"]

# Color formatting for display
def color_oi_change(val):
    if val > 0:
        return 'background-color: #2e1a1a; color: #ff6666'  # Red for OI increase
    elif val < 0:
        return 'background-color: #1a2e1a; color: #00ff88'  # Green for OI decrease
    return ''

def color_gex(val):
    if val > 0:
        return 'background-color: #1a2e1a; color: #00ff88'
    elif val < 0:
        return 'background-color: #2e1a1a; color: #ff6666'
    return ''

styled_df = merged[display_cols].style\
    .applymap(color_oi_change, subset=['Chg_OI_CE', 'Chg_OI_PE'])\
    .applymap(color_gex, subset=['GEX_Net'])\
    .format({
        'OI_CE': '{:,}',
        'Chg_OI_CE': '{:+,}',
        'OI_PE': '{:,}',
        'Chg_OI_PE': '{:+,}',
        'IV_CE': '{:.1f}%',
        'IV_PE': '{:.1f}%',
        'GEX_Net': '₹{:,.0f}'
    })

st.dataframe(styled_df, use_container_width=True, height=400)

# ====================
# TRADING INSIGHTS
# ====================
st.markdown("---")
st.markdown("## 💡 TRADING INSIGHTS")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 KEY OBSERVATIONS")
    
    # Seller bias insight
    bias_color = seller_bias_result['color']
    st.markdown(f"**Seller Bias:** <span style='color:{bias_color}'>{seller_bias_result['bias']}</span>", unsafe_allow_html=True)
    
    # PCR insight
    st.markdown(f"**PCR Signal:** {oi_pcr_metrics['pcr_interpretation']}")
    
    # Max Pain insight
    if seller_max_pain:
        direction = "ABOVE" if spot > seller_max_pain else "BELOW"
        st.markdown(f"**Max Pain:** Spot is {direction} max pain (₹{seller_max_pain:,})")
    
    # GEX insight
    total_gex = merged["GEX_Net"].sum()
    if total_gex > 1000000:
        st.success("**Positive GEX:** Market makers stabilizing (short gamma)")
    elif total_gex < -1000000:
        st.warning("**Negative GEX:** Market makers destabilizing (long gamma)")
    
    # Iceberg insight
    if depth_analysis and depth_analysis['depth_summary']['iceberg_orders'] > 0:
        st.warning(f"**Iceberg Orders:** {depth_analysis['depth_summary']['iceberg_orders']} large hidden orders detected")

with insight_col2:
    st.markdown("### 🛡️ RISK MANAGEMENT")
    
    if nearest_support and nearest_resistance:
        range_size = nearest_resistance - nearest_support
        spot_position = ((spot - nearest_support) / range_size * 100) if range_size > 0 else 50
        
        st.metric("Trading Range", f"₹{nearest_support:,} - ₹{nearest_resistance:,}")
        st.metric("Range Size", f"₹{range_size:,}")
        st.metric("Spot Position", f"{spot_position:.1f}%")
        
        # Trading suggestions
        if spot_position < 40:
            st.success("**Near Support:** Consider LONG with stop below support")
        elif spot_position > 60:
            st.warning("**Near Resistance:** Consider SHORT with stop above resistance")
        else:
            st.info("**Mid-Range:** Wait for breakout or trade range")
        
        # Stop loss suggestions
        st.markdown("**Stop Loss Ideas:**")
        if depth_analysis and depth_analysis['depth_summary']['support_levels']:
            best_support = depth_analysis['depth_summary']['support_levels'][0]['price'] if depth_analysis['depth_summary']['support_levels'] else nearest_support
            st.markdown(f"- LONG Stop: Below ₹{best_support:.2f}")
        
        if depth_analysis and depth_analysis['depth_summary']['resistance_levels']:
            best_resistance = depth_analysis['depth_summary']['resistance_levels'][0]['price'] if depth_analysis['depth_summary']['resistance_levels'] else nearest_resistance
            st.markdown(f"- SHORT Stop: Above ₹{best_resistance:.2f}")

# Final Summary
st.markdown("---")
st.markdown("### 📋 FINAL SUMMARY")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("**Market Structure:**")
    st.markdown(f"- Spot: ₹{spot:.2f}")
    st.markdown(f"- ATM: ₹{atm_strike}")
    st.markdown(f"- Bias: {seller_bias_result['bias']}")
    st.markdown(f"- PCR: {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']})")
    
    if seller_max_pain:
        st.markdown(f"- Max Pain: ₹{seller_max_pain:,}")

with summary_col2:
    st.markdown("**Key Levels:**")
    if nearest_support:
        st.markdown(f"- Support: ₹{nearest_support:,}")
    if nearest_resistance:
        st.markdown(f"- Resistance: ₹{nearest_resistance:,}")
    
    if depth_analysis:
        if depth_analysis['depth_summary']['support_levels']:
            st.markdown(f"- Depth Support: ₹{depth_analysis['depth_summary']['support_levels'][0]['price']:.2f}")
        if depth_analysis['depth_summary']['resistance_levels']:
            st.markdown(f"- Depth Resistance: ₹{depth_analysis['depth_summary']['resistance_levels'][0]['price']:.2f}")

st.markdown("---")
st.caption(f"🔄 Last Updated: {get_ist_datetime_str()} | NIFTY Option Screener v7.0")
