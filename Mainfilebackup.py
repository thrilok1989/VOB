# nifty_option_screener_v4_seller_perspective_complete.py
"""
Nifty Option Screener v4.0 — 100% SELLER'S PERSPECTIVE
EVERYTHING interpreted from Option Seller/Market Maker viewpoint
CALL building = BEARISH (sellers selling calls, expecting price to stay below)
PUT building = BULLISH (sellers selling puts, expecting price to stay above)
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
from supabase import create_client, Client

# -----------------------
#  IST TIMEZONE SETUP
# -----------------------
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_time_str():
    return get_ist_now().strftime("%H:%M:%S")

def get_ist_date_str():
    return get_ist_now().strftime("%Y-%m-%d")

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------
#  CONFIG
# -----------------------
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300

TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30 IST)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30 IST)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30 IST)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30 IST)"}
}

# -----------------------
#  SECRETS
# -----------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    SUPABASE_TABLE_PCR = st.secrets.get("SUPABASE_TABLE_PCR", "strike_pcr_snapshots")
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except Exception as e:
    st.error("❌ Missing credentials")
    st.stop()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"❌ Supabase failed: {e}")
    st.stop()

DHAN_BASE_URL = "https://api.dhan.co"
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# -----------------------
#  CUSTOM CSS - SELLER THEME
# -----------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* SELLER THEME COLORS */
    .seller-bullish { color: #00ff88 !important; font-weight: 700 !important; }
    .seller-bearish { color: #ff4444 !important; font-weight: 700 !important; }
    .seller-neutral { color: #66b3ff !important; font-weight: 700 !important; }
    
    .seller-bullish-bg { background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%); }
    .seller-bearish-bg { background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%); }
    .seller-neutral-bg { background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%); }
    
    h1, h2, h3 { color: #ff66cc !important; } /* Seller theme pink */
    
    .level-card {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff66cc;
        margin: 5px 0;
    }
    .level-card h4 { margin: 0; color: #ff66cc; font-size: 1.1rem; }
    .level-card p { margin: 5px 0; color: #fafafa; font-size: 1.3rem; font-weight: 700; }
    .level-card .sub-info { font-size: 0.9rem; color: #cccccc; margin-top: 5px; }
    
    .spot-card {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 10px 0;
        text-align: center;
    }
    .spot-card h3 { margin: 0; color: #ff9900; font-size: 1.3rem; }
    .spot-card .spot-price { font-size: 2.5rem; color: #ffcc00; font-weight: 700; margin: 10px 0; }
    .spot-card .distance { font-size: 1.1rem; color: #ffdd44; margin: 5px 0; }
    
    .nearest-level {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffcc;
        margin: 10px 0;
    }
    .nearest-level h4 { margin: 0; color: #00ffcc; font-size: 1.2rem; }
    .nearest-level .level-value { font-size: 1.8rem; color: #00ffcc; font-weight: 700; margin: 5px 0; }
    .nearest-level .level-distance { font-size: 1rem; color: #66ffdd; margin: 5px 0; }
    
    .seller-bias-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff66cc;
        margin: 15px 0;
        text-align: center;
    }
    .seller-bias-box h3 { margin: 0; color: #ff66cc; font-size: 1.4rem; }
    .seller-bias-box .bias-value { font-size: 2.2rem; font-weight: 900; margin: 10px 0; }
    
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .seller-support-building { background-color: #1a2e1a; border-left-color: #00ff88; color: #00ff88; }
    .seller-support-breaking { background-color: #2e1a1a; border-left-color: #ff4444; color: #ff6666; }
    .seller-resistance-building { background-color: #2e2a1a; border-left-color: #ffaa00; color: #ffcc44; }
    .seller-resistance-breaking { background-color: #1a1f2e; border-left-color: #00aaff; color: #00ccff; }
    .seller-bull-trap { background-color: #3e1a1a; border-left-color: #ff0000; color: #ff4444; font-weight: 700; }
    .seller-bear-trap { background-color: #1a3e1a; border-left-color: #00ff00; color: #00ff66; font-weight: 700; }
    
    .ist-time {
        background-color: #1a1f2e;
        color: #ff66cc;
        padding: 8px 15px;
        border-radius: 20px;
        border: 2px solid #ff66cc;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background-color: #ff66cc !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #ff99dd !important; }
    
    .greeks-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
    }
    
    .max-pain-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff9900;
        margin: 10px 0;
    }
    .max-pain-box h4 { margin: 0; color: #ff9900; }
    
    .seller-explanation {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff66cc;
        margin: 10px 0;
    }
    
    [data-testid="stMetricLabel"] { color: #cccccc !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #ff66cc !important; font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Nifty Screener v4 - Seller's Perspective", layout="wide")

def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh()

# -----------------------
#  UTILITY FUNCTIONS
# -----------------------
def safe_int(x):
    try:
        return int(x)
    except:
        return 0

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

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

def bs_vega(S,K,r,sigma,tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return S * norm.pdf(d1) * np.sqrt(tau)

def bs_theta(S,K,r,sigma,tau,option_type="call"):
    if sigma <=0 or tau<=0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    d2 = d1 - sigma*np.sqrt(tau)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau))
    if option_type=="call":
        term2 = r*K*np.exp(-r*tau)*norm.cdf(d2)
        return term1 - term2
    else:
        term2 = r*K*np.exp(-r*tau)*norm.cdf(-d2)
        return term1 + term2

# -----------------------
# 🔥 SELLER'S PERSPECTIVE FUNCTIONS
# -----------------------
def seller_strength_score(row, weights=SCORE_WEIGHTS):
    """
    SELLER's strength score: Higher when sellers are confident
    """
    chg_oi = abs(safe_float(row.get("Chg_OI_CE",0))) + abs(safe_float(row.get("Chg_OI_PE",0)))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    
    # SELLER LOGIC: Higher IV = More premium for sellers = Higher confidence
    # High OI = More positions to manage = Higher risk/reward
    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def seller_price_oi_divergence(chg_oi, vol, ltp_change, option_type="CE"):
    """
    SELLER'S interpretation of price-OI divergence
    """
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)
    
    if option_type == "CE":
        # CALL SELLER perspective
        if oi_up and vol_up and price_up:
            return "Sellers WRITING calls as price rises (Bearish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING calls on weakness (Strong bearish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back calls as price rises (Covering bearish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back calls on weakness (Reducing bearish exposure)"
    else:
        # PUT SELLER perspective
        if oi_up and vol_up and price_up:
            return "Sellers WRITING puts on strength (Bullish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING puts as price falls (Strong bullish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back puts on strength (Covering bullish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back puts as price falls (Reducing bullish exposure)"
    
    if oi_up and not vol_up:
        return "Sellers quietly WRITING options"
    if (not oi_up) and not vol_up:
        return "Sellers quietly UNWINDING"
    
    return "Sellers inactive"

def seller_itm_otm_interpretation(strike, atm, chg_oi_ce, chg_oi_pe):
    """
    SELLER'S interpretation of ITM/OTM activity
    """
    ce_interpretation = ""
    pe_interpretation = ""
    
    # CALL OPTIONS (SELLERS SELLING = BEARISH)
    if strike < atm:  # ITM CALLS
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING ITM CALLS = VERY BEARISH 🚨"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK ITM CALLS = BULLISH 📈"
        else:
            ce_interpretation = "No ITM CALL activity"
    
    elif strike > atm:  # OTM CALLS
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING OTM CALLS = MILD BEARISH 📉"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK OTM CALLS = MILD BULLISH 📊"
        else:
            ce_interpretation = "No OTM CALL activity"
    
    else:  # ATM
        ce_interpretation = "ATM CALL zone"
    
    # PUT OPTIONS (SELLERS SELLING = BULLISH)
    if strike > atm:  # ITM PUTS
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING ITM PUTS = VERY BULLISH 🚀"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK ITM PUTS = BEARISH 🐻"
        else:
            pe_interpretation = "No ITM PUT activity"
    
    elif strike < atm:  # OTM PUTS
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING OTM PUTS = MILD BULLISH 📈"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK OTM PUTS = MILD BEARISH 📉"
        else:
            pe_interpretation = "No OTM PUT activity"
    
    else:  # ATM
        pe_interpretation = "ATM PUT zone"
    
    return f"CALL Sellers: {ce_interpretation} | PUT Sellers: {pe_interpretation}"

def seller_gamma_pressure(row, atm, strike_gap):
    """
    SELLER'S gamma pressure: Negative = Sellers under pressure
    """
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    
    # SELLER LOGIC: Positive gamma = Sellers comfortable, Negative = Sellers under pressure
    chg_oi_sum = safe_float(row.get("Chg_OI_CE",0)) - safe_float(row.get("Chg_OI_PE",0))
    
    # Convert to seller perspective
    # CALL building (positive chg_oi_ce) = BEARISH pressure on sellers
    # PUT building (positive chg_oi_pe) = BULLISH comfort for sellers
    seller_pressure = -chg_oi_sum / dist  # Negative = sellers under pressure
    
    return seller_pressure

def seller_breakout_probability_index(merged_df, atm, strike_gap):
    """
    SELLER'S breakout probability: When sellers lose control
    """
    near_mask = merged_df["strikePrice"].between(atm-strike_gap, atm+strike_gap)
    
    # SELLER ATM activity
    atm_ce_build = merged_df.loc[near_mask, "Chg_OI_CE"].sum()
    atm_pe_build = merged_df.loc[near_mask, "Chg_OI_PE"].sum()
    
    # SELLER LOGIC: 
    # CALL building at ATM = Sellers bearish = Downside pressure
    # PUT building at ATM = Sellers bullish = Upside pressure
    seller_atm_bias = atm_pe_build - atm_ce_build  # Positive = bullish sellers
    atm_score = min(abs(seller_atm_bias)/50000.0, 1.0)
    
    # SELLER winding/unwinding balance
    winding_count = (merged_df[["CE_Winding","PE_Winding"]]=="Winding").sum().sum()
    unwinding_count = (merged_df[["CE_Winding","PE_Winding"]]=="Unwinding").sum().sum()
    
    # SELLER LOGIC: More winding = More selling = Higher conviction
    seller_conviction = winding_count/(winding_count+unwinding_count) if (winding_count+unwinding_count)>0 else 0.5
    
    # SELLER volume-OI activity
    vol_oi_scores = (merged_df[["Vol_CE","Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE","Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum()/100000.0, 1.0)
    
    # SELLER gamma pressure
    gamma = merged_df.apply(lambda r: seller_gamma_pressure(r, atm, strike_gap), axis=1).sum()
    gamma_score = min(abs(gamma)/10000.0, 1.0)
    
    # Combine with SELLER weights
    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"]*atm_score) + (w["winding_balance"]*seller_conviction) + (w["vol_oi_div"]*vol_oi_score) + (w["gamma_pressure"]*gamma_score)
    
    return int(np.clip(combined*100,0,100))

def center_of_mass_oi(df, oi_col):
    """Calculate center of mass for OI distribution"""
    if df.empty or oi_col not in df.columns:
        return 0
    total_oi = df[oi_col].sum()
    if total_oi == 0:
        return 0
    weighted_sum = (df["strikePrice"] * df[oi_col]).sum()
    return weighted_sum / total_oi

def calculate_seller_max_pain(df):
    """Calculate max pain strike from SELLER's perspective"""
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))
        
        # SELLER'S pain: Loss on written options
        # Sellers lose money when options are in-the-money
        ce_pain = ce_oi * max(0, ce_ltp) if strike < df["strikePrice"].mean() else 0
        pe_pain = pe_oi * max(0, pe_ltp) if strike > df["strikePrice"].mean() else 0
        
        pain = ce_pain + pe_pain
        pain_dict[strike] = pain
    
    if pain_dict:
        return min(pain_dict, key=pain_dict.get)
    return None

# -----------------------
# 🔥 SELLER-CENTRIC MARKET BIAS CALCULATION
# -----------------------
def calculate_seller_market_bias(merged_df, spot, atm_strike):
    """
    Market bias from 100% SELLER's perspective
    """
    polarity = 0.0
    
    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))  # ΔOI CE
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))  # ΔOI PE
        
        # ============================================
        # PURE SELLER LOGIC
        # ============================================
        
        # CALL OPTIONS: SELLERS WRITING = BEARISH
        if strike < atm_strike:  # ITM CALLS
            if chg_ce > 0:  # ITM CALL building = SELLERS WRITING
                polarity -= 2.0  # VERY BEARISH
            elif chg_ce < 0:  # ITM CALL unwinding = SELLERS BUYING BACK
                polarity += 1.5  # BULLISH
        
        elif strike > atm_strike:  # OTM CALLS
            if chg_ce > 0:  # OTM CALL building = SELLERS WRITING
                polarity -= 0.7  # MILD BEARISH
            elif chg_ce < 0:  # OTM CALL unwinding = SELLERS BUYING BACK
                polarity += 0.5  # MILD BULLISH
        
        # PUT OPTIONS: SELLERS WRITING = BULLISH
        if strike > atm_strike:  # ITM PUTS
            if chg_pe > 0:  # ITM PUT building = SELLERS WRITING
                polarity += 2.0  # VERY BULLISH
            elif chg_pe < 0:  # ITM PUT unwinding = SELLERS BUYING BACK
                polarity -= 1.5  # BEARISH
        
        elif strike < atm_strike:  # OTM PUTS
            if chg_pe > 0:  # OTM PUT building = SELLERS WRITING
                polarity += 0.7  # MILD BULLISH
            elif chg_pe < 0:  # OTM PUT unwinding = SELLERS BUYING BACK
                polarity -= 0.5  # MILD BEARISH
    
    # ============================================
    # SELLER CONFIDENCE METRICS
    # ============================================
    
    # 1. PCR from SELLER perspective
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:  # High PCR = More puts written = BULLISH SELLERS
            polarity += 1.0
        elif pcr < 0.5:  # Low PCR = More calls written = BEARISH SELLERS
            polarity -= 1.0
    
    # 2. IV Premium (SELLER pricing power)
    avg_iv_ce = merged_df["IV_CE"].mean()
    avg_iv_pe = merged_df["IV_PE"].mean()
    if avg_iv_ce > avg_iv_pe + 5:  # Higher CALL IV
        # Sellers charging more for calls = Bearish confidence
        polarity -= 0.3
    elif avg_iv_pe > avg_iv_ce + 5:  # Higher PUT IV
        # Sellers charging more for puts = Bullish confidence
        polarity += 0.3
    
    # 3. GEX (SELLER gamma risk)
    total_gex_ce = merged_df["GEX_CE"].sum()
    total_gex_pe = merged_df["GEX_PE"].sum()
    net_gex = total_gex_ce + total_gex_pe
    if net_gex < -1000000:  # Negative GEX = Sellers under pressure
        polarity -= 0.4
    elif net_gex > 1000000:  # Positive GEX = Sellers comfortable
        polarity += 0.4
    
    # 4. Max Pain alignment
    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        distance_to_spot = abs(spot - max_pain) / spot * 100
        if distance_to_spot < 1.0:  # Close to max pain
            polarity += 0.5  # Sellers in control
    
    # ============================================
    # FINAL SELLER BIAS
    # ============================================
    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes.",
            "action": "Bullish breakout likely. Sellers confident in upside."
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing. Moderate bullish sentiment.",
            "action": "Expect support to hold. Upside bias."
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes.",
            "action": "Bearish breakdown likely. Sellers confident in downside."
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing. Moderate bearish sentiment.",
            "action": "Expect resistance to hold. Downside bias."
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity. No clear directional bias.",
            "action": "Range-bound expected. Wait for clearer signals."
        }

# -----------------------
# 🔥 SPOT POSITION ANALYSIS - SELLER VIEW
# -----------------------
def analyze_spot_position_seller(spot, pcr_df, market_bias):
    """
    Analyze spot position from SELLER's perspective
    """
    # Sort by strike price
    sorted_df = pcr_df.sort_values("strikePrice").reset_index(drop=True)
    
    # All strikes sorted
    all_strikes = sorted_df["strikePrice"].tolist()
    
    # Find nearest support (highest strike below spot)
    supports_below = [s for s in all_strikes if s < spot]
    nearest_support = max(supports_below) if supports_below else None
    
    # Find nearest resistance (lowest strike above spot)
    resistances_above = [s for s in all_strikes if s > spot]
    nearest_resistance = min(resistances_above) if resistances_above else None
    
    # Get details for each level
    def get_level_details(strike, df):
        if strike is None:
            return None
        row = df[df["strikePrice"] == strike]
        if row.empty:
            return None
        
        # SELLER interpretation
        pcr = row.iloc[0]["PCR"]
        oi_ce = int(row.iloc[0]["OI_CE"])
        oi_pe = int(row.iloc[0]["OI_PE"])
        chg_oi_ce = int(row.iloc[0].get("Chg_OI_CE", 0))
        chg_oi_pe = int(row.iloc[0].get("Chg_OI_PE", 0))
        
        # SELLER strength
        if pcr > 1.5:
            seller_strength = "Strong PUT selling (Bullish sellers)"
        elif pcr > 1.0:
            seller_strength = "Moderate PUT selling"
        elif pcr < 0.5:
            seller_strength = "Strong CALL selling (Bearish sellers)"
        elif pcr < 1.0:
            seller_strength = "Moderate CALL selling"
        else:
            seller_strength = "Balanced selling"
        
        return {
            "strike": int(strike),
            "oi_ce": oi_ce,
            "oi_pe": oi_pe,
            "chg_oi_ce": chg_oi_ce,
            "chg_oi_pe": chg_oi_pe,
            "pcr": pcr,
            "seller_strength": seller_strength,
            "distance": abs(spot - strike),
            "distance_pct": abs(spot - strike) / spot * 100
        }
    
    nearest_sup = get_level_details(nearest_support, sorted_df)
    nearest_res = get_level_details(nearest_resistance, sorted_df)
    
    # SELLER range analysis
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        spot_position_pct = ((spot - nearest_sup["strike"]) / range_size * 100) if range_size > 0 else 50
        
        # SELLER range interpretation
        if spot_position_pct < 40:
            range_bias = "Near SELLER support (Bullish sellers defending)"
        elif spot_position_pct > 60:
            range_bias = "Near SELLER resistance (Bearish sellers defending)"
        else:
            range_bias = "Middle of SELLER range"
    else:
        range_size = 0
        spot_position_pct = 50
        range_bias = "Range undefined"
    
    return {
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "spot_in_range": (nearest_support, nearest_resistance),
        "range_size": range_size,
        "spot_position_pct": spot_position_pct,
        "range_bias": range_bias,
        "market_bias": market_bias
    }

# -----------------------
# PCR FUNCTIONS (Seller View)
# -----------------------
def compute_pcr_df(merged_df):
    df = merged_df.copy()
    df["OI_CE"] = pd.to_numeric(df.get("OI_CE", 0), errors="coerce").fillna(0).astype(int)
    df["OI_PE"] = pd.to_numeric(df.get("OI_PE", 0), errors="coerce").fillna(0).astype(int)
    
    def pcr_calc(row):
        ce = int(row["OI_CE"]) if row["OI_CE"] is not None else 0
        pe = int(row["OI_PE"]) if row["OI_PE"] is not None else 0
        if ce <= 0:
            if pe > 0:
                return float("inf")
            else:
                return np.nan
        return pe / ce
    
    df["PCR"] = df.apply(pcr_calc, axis=1)
    return df

def rank_support_resistance_seller(pcr_df):
    """
    Rank supports/resistances from SELLER's perspective
    """
    eps = 1e-6
    t = pcr_df.copy()
    
    # SELLER LOGIC: High PCR = Strong PUT selling = Strong support
    # Low PCR = Strong CALL selling = Strong resistance
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Support score: High PE OI + High PCR
    t["seller_support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    
    # Resistance score: High CE OI + Low PCR (inverse)
    t["seller_resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["seller_resistance_score"] = t["OI_CE"] + (t["seller_resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("seller_support_score", ascending=False).head(3)
    top_resists = t.sort_values("seller_resistance_score", ascending=False).head(3)
    
    return t, top_supports, top_resists

# -----------------------
# DHAN API
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
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            idx_data = data.get("data", {}).get("IDX_I", {})
            nifty_data = idx_data.get("13", {})
            ltp = nifty_data.get("last_price", 0.0)
            return float(ltp)
        return 0.0
    except Exception as e:
        st.warning(f"Dhan LTP failed: {e}")
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
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",[])
        return []
    except Exception as e:
        st.warning(f"Expiry list failed: {e}")
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
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",{})
        return None
    except Exception as e:
        st.warning(f"Option chain failed: {e}")
        return None

def parse_dhan_option_chain(chain_data):
    if not chain_data:
        return pd.DataFrame(), pd.DataFrame()
    oc = chain_data.get("oc",{})
    ce_rows, pe_rows = [], []
    for strike_str, strike_data in oc.items():
        try:
            strike = int(float(strike_str))
        except:
            continue
        ce = strike_data.get("ce")
        pe = strike_data.get("pe")
        if ce:
            ci = {
                "strikePrice": strike,
                "OI_CE": safe_int(ce.get("oi",0)),
                "Chg_OI_CE": safe_int(ce.get("oi",0)) - safe_int(ce.get("previous_oi",0)),
                "Vol_CE": safe_int(ce.get("volume",0)),
                "LTP_CE": safe_float(ce.get("last_price",0.0)),
                "IV_CE": safe_float(ce.get("implied_volatility", np.nan))
            }
            ce_rows.append(ci)
        if pe:
            pi = {
                "strikePrice": strike,
                "OI_PE": safe_int(pe.get("oi",0)),
                "Chg_OI_PE": safe_int(pe.get("oi",0)) - safe_int(pe.get("previous_oi",0)),
                "Vol_PE": safe_int(pe.get("volume",0)),
                "LTP_PE": safe_float(pe.get("last_price",0.0)),
                "IV_PE": safe_float(pe.get("implied_volatility", np.nan))
            }
            pe_rows.append(pi)
    return pd.DataFrame(ce_rows), pd.DataFrame(pe_rows)

# -----------------------
#  MAIN APP - SELLER'S PERSPECTIVE
# -----------------------
st.title("🎯 NIFTY Option Screener v4.0 — SELLER'S PERSPECTIVE")

current_ist = get_ist_datetime_str()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <span class='ist-time'>🕐 IST: {current_ist}</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class='seller-explanation'>
    <h3>🎯 SELLER'S LOGIC</h3>
    <p><strong>Options WRITING = Directional Bias:</strong></p>
    <ul>
    <li><span class='seller-bearish'>📉 CALL Writing</span> = BEARISH (expecting price to STAY BELOW)</li>
    <li><span class='seller-bullish'>📈 PUT Writing</span> = BULLISH (expecting price to STAY ABOVE)</li>
    <li><span class='seller-bullish'>🔄 CALL Buying Back</span> = BULLISH (covering bearish bets)</li>
    <li><span class='seller-bearish'>🔄 PUT Buying Back</span> = BEARISH (covering bullish bets)</li>
    </ul>
    <p><em>Market makers & institutions are primarily SELLERS, not buyers.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"**Current IST:** {get_ist_time_str()}")
    st.markdown(f"**Date:** {get_ist_date_str()}")
    
    save_interval = st.number_input("PCR Auto-save (sec)", value=SAVE_INTERVAL_SEC, min_value=60, step=60)
    
    if st.button("Clear Caches"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot")
        st.stop()
    
    expiries = get_expiry_list()
    if not expiries:
        st.error("Unable to fetch expiry list")
        st.stop()
    
    expiry = st.selectbox("Select expiry", expiries, index=0)

with col2:
    if spot > 0:
        st.metric("NIFTY Spot", f"₹{spot:.2f}")
        st.metric("Expiry", expiry)

# Fetch option chain
with st.spinner("Fetching option chain..."):
    chain = fetch_dhan_option_chain(expiry)
if chain is None:
    st.error("Failed to fetch option chain")
    st.stop()

df_ce, df_pe = parse_dhan_option_chain(chain)
if df_ce.empty or df_pe.empty:
    st.error("Insufficient CE/PE data")
    st.stop()

# Filter ATM window
strike_gap = strike_gap_from_series(df_ce["strikePrice"])
atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))
lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

df_ce = df_ce[(df_ce["strikePrice"]>=lower) & (df_ce["strikePrice"]<=upper)].reset_index(drop=True)
df_pe = df_pe[(df_pe["strikePrice"]>=lower) & (df_pe["strikePrice"]<=upper)].reset_index(drop=True)

merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# Compute tau
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
except Exception:
    tau = 7.0/365.0

# Session storage for prev LTP/IV
if "prev_ltps_seller" not in st.session_state:
    st.session_state["prev_ltps_seller"] = {}
if "prev_ivs_seller" not in st.session_state:
    st.session_state["prev_ivs_seller"] = {}

# Compute per-strike metrics with SELLER interpretation
for i, row in merged.iterrows():
    strike = int(row["strikePrice"])
    ltp_ce = safe_float(row.get("LTP_CE",0.0))
    ltp_pe = safe_float(row.get("LTP_PE",0.0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))

    key_ce = f"{expiry}_{strike}_CE"
    key_pe = f"{expiry}_{strike}_PE"
    prev_ce = st.session_state["prev_ltps_seller"].get(key_ce, None)
    prev_pe = st.session_state["prev_ltps_seller"].get(key_pe, None)
    prev_iv_ce = st.session_state["prev_ivs_seller"].get(key_ce, None)
    prev_iv_pe = st.session_state["prev_ivs_seller"].get(key_pe, None)

    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)
    ce_iv_delta = None if prev_iv_ce is None else (iv_ce - prev_iv_ce)
    pe_iv_delta = None if prev_iv_pe is None else (iv_pe - prev_iv_pe)

    st.session_state["prev_ltps_seller"][key_ce] = ltp_ce
    st.session_state["prev_ltps_seller"][key_pe] = ltp_pe
    st.session_state["prev_ivs_seller"][key_ce] = iv_ce
    st.session_state["prev_ivs_seller"][key_pe] = iv_pe

    chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))

    # SELLER winding/unwinding labels
    merged.at[i,"CE_Seller_Action"] = "WRITING" if chg_oi_ce>0 else ("BUYING BACK" if chg_oi_ce<0 else "HOLDING")
    merged.at[i,"PE_Seller_Action"] = "WRITING" if chg_oi_pe>0 else ("BUYING BACK" if chg_oi_pe<0 else "HOLDING")

    # SELLER divergence interpretation - FIXED FUNCTION NAME
    merged.at[i,"CE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_ce, safe_int(row.get("Vol_CE",0)), ce_price_delta, "CE")
    merged.at[i,"PE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_pe, safe_int(row.get("Vol_PE",0)), pe_price_delta, "PE")

    # SELLER ITM/OTM interpretation
    merged.at[i,"Seller_Interpretation"] = seller_itm_otm_interpretation(strike, atm_strike, chg_oi_ce, chg_oi_pe)

    # Greeks calculation
    sigma_ce = iv_ce/100.0 if not np.isnan(iv_ce) and iv_ce>0 else 0.25
    sigma_pe = iv_pe/100.0 if not np.isnan(iv_pe) and iv_pe>0 else 0.25

    try:
        delta_ce = bs_delta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
        gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
        vega_ce = bs_vega(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
        theta_ce = bs_theta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
    except Exception:
        delta_ce = gamma_ce = vega_ce = theta_ce = 0.0

    try:
        delta_pe = bs_delta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        vega_pe = bs_vega(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        theta_pe = bs_theta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
    except Exception:
        delta_pe = gamma_pe = vega_pe = theta_pe = 0.0

    merged.at[i,"Delta_CE"] = delta_ce
    merged.at[i,"Gamma_CE"] = gamma_ce
    merged.at[i,"Vega_CE"] = vega_ce
    merged.at[i,"Theta_CE"] = theta_ce
    merged.at[i,"Delta_PE"] = delta_pe
    merged.at[i,"Gamma_PE"] = gamma_pe
    merged.at[i,"Vega_PE"] = vega_pe
    merged.at[i,"Theta_PE"] = theta_pe

    # GEX calculation (SELLER exposure)
    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    gex_ce = gamma_ce * notional * oi_ce
    gex_pe = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_CE"] = gex_ce
    merged.at[i,"GEX_PE"] = gex_pe
    merged.at[i,"GEX_Net"] = gex_ce + gex_pe  # Both add to seller risk

    # SELLER strength score
    merged.at[i,"Seller_Strength_Score"] = seller_strength_score(row)

    # SELLER gamma pressure
    merged.at[i,"Seller_Gamma_Pressure"] = seller_gamma_pressure(row, atm_strike, strike_gap)

    merged.at[i,"CE_Price_Delta"] = ce_price_delta
    merged.at[i,"PE_Price_Delta"] = pe_price_delta
    merged.at[i,"CE_IV_Delta"] = ce_iv_delta
    merged.at[i,"PE_IV_Delta"] = pe_iv_delta

# Aggregations
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

# SELLER activity summary
ce_selling = (merged["Chg_OI_CE"] > 0).sum()
ce_buying_back = (merged["Chg_OI_CE"] < 0).sum()
pe_selling = (merged["Chg_OI_PE"] > 0).sum()
pe_buying_back = (merged["Chg_OI_PE"] < 0).sum()

# Greeks totals
total_gex_ce = merged["GEX_CE"].sum()
total_gex_pe = merged["GEX_PE"].sum()
total_gex_net = merged["GEX_Net"].sum()

# Calculate SELLER metrics
seller_max_pain = calculate_seller_max_pain(merged)
seller_breakout_index = seller_breakout_probability_index(merged, atm_strike, strike_gap)

# Calculate SELLER market bias
seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)

# Compute PCR
pcr_df = compute_pcr_df(merged)

# Get SELLER support/resistance rankings
ranked_current, seller_supports_df, seller_resists_df = rank_support_resistance_seller(pcr_df)

# Analyze spot position from SELLER perspective
spot_analysis = analyze_spot_position_seller(spot, ranked_current, seller_bias_result)

# ============================================
# 🎯 MAIN DASHBOARD - SELLER'S VIEW
# ============================================

# SELLER BIAS DISPLAY
st.markdown(f"""
<div class='seller-bias-box'>
    <h3>🎯 SELLER'S MARKET BIAS</h3>
    <div class='bias-value' style='color:{seller_bias_result["color"]}'>
        {seller_bias_result["bias"]}
    </div>
    <p>Polarity Score: {seller_bias_result["polarity"]:.2f}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='seller-explanation'>
    <h4>🧠 SELLER'S THINKING:</h4>
    <p><strong>{seller_bias_result["explanation"]}</strong></p>
    <p><strong>Action:</strong> {seller_bias_result["action"]}</p>
</div>
""", unsafe_allow_html=True)

# Core Metrics
st.markdown("## 📈 SELLER'S MARKET OVERVIEW")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot", f"₹{spot:.2f}")
    st.metric("ATM", f"₹{atm_strike}")
with col2:
    st.metric("CALL Sellers", f"{ce_selling} strikes")
    st.metric("PUT Sellers", f"{pe_selling} strikes")
with col3:
    st.metric("CALL Buying Back", f"{ce_buying_back} strikes")
    st.metric("PUT Buying Back", f"{pe_buying_back} strikes")
with col4:
    st.metric("Total GEX", f"₹{int(total_gex_net):,}")
    st.metric("Breakout Index", f"{seller_breakout_index}%")

# Max Pain Display
if seller_max_pain:
    distance_to_max_pain = abs(spot - seller_max_pain)
    st.markdown(f"""
    <div class='max-pain-box'>
        <h4>🎯 SELLER'S MAX PAIN (Preferred Level)</h4>
        <p style='font-size: 1.5rem; color: #ff9900; font-weight: bold; text-align: center;'>₹{seller_max_pain:,}</p>
        <p style='text-align: center; color: #cccccc;'>Distance from spot: ₹{distance_to_max_pain:.2f} ({distance_to_max_pain/spot*100:.2f}%)</p>
        <p style='text-align: center; color: #ffcc00;'>Sellers want price here to minimize losses</p>
    </div>
    """, unsafe_allow_html=True)

# SELLER Activity Summary
st.markdown("### 🔥 SELLER ACTIVITY HEATMAP")

seller_activity = pd.DataFrame([
    {"Activity": "CALL Writing (Bearish)", "Strikes": ce_selling, "Bias": "BEARISH", "Color": "#ff4444"},
    {"Activity": "CALL Buying Back (Bullish)", "Strikes": ce_buying_back, "Bias": "BULLISH", "Color": "#00ff88"},
    {"Activity": "PUT Writing (Bullish)", "Strikes": pe_selling, "Bias": "BULLISH", "Color": "#00ff88"},
    {"Activity": "PUT Buying Back (Bearish)", "Strikes": pe_buying_back, "Bias": "BEARISH", "Color": "#ff4444"}
])

st.dataframe(seller_activity, use_container_width=True)

st.markdown("---")

# ============================================
# 🎯 SPOT POSITION - SELLER'S VIEW
# ============================================

st.markdown("## 📍 SPOT POSITION (SELLER'S DEFENSE)")

nearest_sup = spot_analysis["nearest_support"]
nearest_res = spot_analysis["nearest_resistance"]

col_spot, col_range = st.columns([1, 1])

with col_spot:
    st.markdown(f"""
    <div class="spot-card">
        <h3>🎯 CURRENT SPOT</h3>
        <div class="spot-price">₹{spot:,.2f}</div>
        <div class="distance">ATM: ₹{atm_strike:,}</div>
        <div class="distance">Market Bias: <span style='color:{seller_bias_result["color"]}'>{seller_bias_result["bias"]}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    if nearest_sup and nearest_res:
        range_size = spot_analysis["range_size"]
        spot_position_pct = spot_analysis["spot_position_pct"]
        range_bias = spot_analysis["range_bias"]
        
        st.markdown(f"""
        <div class="spot-card">
            <h3>📊 SELLER'S DEFENSE RANGE</h3>
            <div class="distance">₹{nearest_sup['strike']:,} ← SPOT → ₹{nearest_res['strike']:,}</div>
            <div class="distance">Position: {spot_position_pct:.1f}% within range</div>
            <div class="distance">Range Width: ₹{range_size:,}</div>
            <div class="distance" style='color:#ffcc00;'>{range_bias}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# NEAREST LEVELS WITH SELLER INTERPRETATION
st.markdown("### 🎯 NEAREST SELLER DEFENSE LEVELS")

col_ns, col_nr = st.columns(2)

with col_ns:
    st.markdown("#### 🛡️ SELLER SUPPORT BELOW")
    
    if nearest_sup:
        sup = nearest_sup
        pcr_display = f"{sup['pcr']:.2f}" if not np.isinf(sup['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>💚 NEAREST SELLER SUPPORT</h4>
            <div class="level-value">₹{sup['strike']:,}</div>
            <div class="level-distance">⬇️ Distance: ₹{sup['distance']:.2f} ({sup['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                <strong>SELLER ACTIVITY:</strong> {sup['seller_strength']}<br>
                PUT OI: {sup['oi_pe']:,} | CALL OI: {sup['oi_ce']:,}<br>
                PCR: {pcr_display} | ΔCALL: {sup['chg_oi_ce']:+,} | ΔPUT: {sup['chg_oi_pe']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No seller support level below spot")

with col_nr:
    st.markdown("#### ⚡ SELLER RESISTANCE ABOVE")
    
    if nearest_res:
        res = nearest_res
        pcr_display = f"{res['pcr']:.2f}" if not np.isinf(res['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>🧡 NEAREST SELLER RESISTANCE</h4>
            <div class="level-value">₹{res['strike']:,}</div>
            <div class="level-distance">⬆️ Distance: ₹{res['distance']:.2f} ({res['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                <strong>SELLER ACTIVITY:</strong> {res['seller_strength']}<br>
                CALL OI: {res['oi_ce']:,} | PUT OI: {res['oi_pe']:,}<br>
                PCR: {pcr_display} | ΔCALL: {res['chg_oi_ce']:+,} | ΔPUT: {res['chg_oi_pe']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No seller resistance level above spot")

st.markdown("---")

# TOP SELLER DEFENSE LEVELS
st.markdown("### 🎯 TOP SELLER DEFENSE LEVELS (Strongest 3)")

col_s, col_r = st.columns(2)

with col_s:
    st.markdown("#### 🛡️ STRONGEST SELLER SUPPORTS")
    
    for i, (idx, row) in enumerate(seller_supports_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_pe = int(row["OI_PE"])
        oi_ce = int(row["OI_CE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        chg_oi_ce = int(row.get("Chg_OI_CE", 0))
        
        # SELLER interpretation
        if pcr > 1.5:
            seller_msg = "Heavy PUT writing - Strong bullish defense"
            color = "#00ff88"
        elif pcr > 1.0:
            seller_msg = "Moderate PUT writing - Bullish defense"
            color = "#00cc66"
        else:
            seller_msg = "Light PUT writing - Weak defense"
            color = "#cccccc"
        
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Seller Support #{i}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                <span style='color:{color}'><strong>{seller_msg}</strong></span><br>
                PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                PCR: {pcr_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_r:
    st.markdown("#### ⚡ STRONGEST SELLER RESISTANCES")
    
    for i, (idx, row) in enumerate(seller_resists_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_ce = int(row["OI_CE"])
        oi_pe = int(row["OI_PE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_ce = int(row.get("Chg_OI_CE", 0))
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        
        # SELLER interpretation
        if pcr < 0.5:
            seller_msg = "Heavy CALL writing - Strong bearish defense"
            color = "#ff4444"
        elif pcr < 1.0:
            seller_msg = "Moderate CALL writing - Bearish defense"
            color = "#ff6666"
        else:
            seller_msg = "Light CALL writing - Weak defense"
            color = "#cccccc"
        
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Seller Resistance #{i}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                <span style='color:{color}'><strong>{seller_msg}</strong></span><br>
                CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                PCR: {pcr_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# 📊 DETAILED DATA - SELLER VIEW
# ============================================

tab1, tab2, tab3 = st.tabs(["📊 Seller Activity", "🧮 Seller Greeks", "📈 Seller PCR"])

with tab1:
    st.markdown("### 📊 SELLER ACTIVITY BY STRIKE")
    
    seller_cols = [
        "strikePrice", 
        "OI_CE", "Chg_OI_CE", "CE_Seller_Action", "CE_Seller_Divergence",
        "OI_PE", "Chg_OI_PE", "PE_Seller_Action", "PE_Seller_Divergence",
        "Seller_Interpretation", "Seller_Strength_Score"
    ]
    
    # Ensure all columns exist
    for col in seller_cols:
        if col not in merged.columns:
            merged[col] = ""
    
    # Color code seller actions
    def color_seller_action(val):
        if "WRITING" in str(val):
            if "CALL" in str(val):
                return "background-color: #2e1a1a; color: #ff6666"
            else:
                return "background-color: #1a2e1a; color: #00ff88"
        elif "BUYING BACK" in str(val):
            if "CALL" in str(val):
                return "background-color: #1a2e1a; color: #00ff88"
            else:
                return "background-color: #2e1a1a; color: #ff6666"
        return ""
    
    seller_display = merged[seller_cols].copy()
    styled_df = seller_display.style.applymap(color_seller_action, subset=["CE_Seller_Action", "PE_Seller_Action"])
    st.dataframe(styled_df, use_container_width=True)

with tab2:
    st.markdown("### 🧮 SELLER GREEKS & GEX EXPOSURE")
    
    greeks_cols = [
        "strikePrice",
        "Delta_CE", "Gamma_CE", "Vega_CE", "Theta_CE", "GEX_CE",
        "Delta_PE", "Gamma_PE", "Vega_PE", "Theta_PE", "GEX_PE",
        "GEX_Net", "Seller_Gamma_Pressure"
    ]
    
    for col in greeks_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    
    # Format Greek values
    greeks_display = merged[greeks_cols].copy()
    
    # Color code GEX
    def color_gex(val):
        if val > 0:
            return "background-color: #1a2e1a; color: #00ff88"
        elif val < 0:
            return "background-color: #2e1a1a; color: #ff6666"
        return ""
    
    styled_greeks = greeks_display.style.applymap(color_gex, subset=["GEX_Net"])
    st.dataframe(styled_greeks, use_container_width=True)
    
    # GEX Interpretation
    st.markdown("#### 🎯 GEX INTERPRETATION (SELLER'S VIEW)")
    if total_gex_net > 0:
        st.success(f"**POSITIVE GEX (₹{int(total_gex_net):,}):** Sellers have POSITIVE gamma exposure. They're SHORT gamma and will BUY when price rises, SELL when price falls (stabilizing effect).")
    elif total_gex_net < 0:
        st.error(f"**NEGATIVE GEX (₹{int(total_gex_net):,}):** Sellers have NEGATIVE gamma exposure. They're LONG gamma and will SELL when price rises, BUY when price falls (destabilizing effect).")
    else:
        st.info("**NEUTRAL GEX:** Balanced seller gamma exposure.")

with tab3:
    st.markdown("### 📈 SELLER PCR ANALYSIS")
    
    pcr_display_cols = ["strikePrice", "OI_CE", "OI_PE", "PCR", "Chg_OI_CE", "Chg_OI_PE", "seller_support_score", "seller_resistance_score"]
    for col in pcr_display_cols:
        if col not in ranked_current.columns:
            ranked_current[col] = 0
    
    # Color PCR values
    def color_pcr(val):
        if val > 1.5:
            return "background-color: #1a2e1a; color: #00ff88"
        elif val > 1.0:
            return "background-color: #2e2a1a; color: #ffcc44"
        elif val > 0.5:
            return "background-color: #1a1f2e; color: #66b3ff"
        else:
            return "background-color: #2e1a1a; color: #ff4444"
    
    pcr_display = ranked_current[pcr_display_cols].copy()
    pcr_display["distance_from_spot"] = abs(pcr_display["strikePrice"] - spot)
    
    styled_pcr = pcr_display.style.applymap(color_pcr, subset=["PCR"])
    st.dataframe(styled_pcr.sort_values("distance_from_spot"), use_container_width=True)
    
    # PCR Interpretation
    avg_pcr = ranked_current["PCR"].replace([np.inf, -np.inf], np.nan).mean()
    if not np.isnan(avg_pcr):
        st.markdown(f"#### 🎯 AVERAGE PCR: {avg_pcr:.2f}")
        if avg_pcr > 1.5:
            st.success("**HIGH PCR (>1.5):** Heavy PUT selling relative to CALL selling. Sellers are BULLISH.")
        elif avg_pcr > 1.0:
            st.info("**MODERATE PCR (1.0-1.5):** More PUT selling than CALL selling. Sellers leaning BULLISH.")
        elif avg_pcr > 0.5:
            st.warning("**LOW PCR (0.5-1.0):** More CALL selling than PUT selling. Sellers leaning BEARISH.")
        else:
            st.error("**VERY LOW PCR (<0.5):** Heavy CALL selling relative to PUT selling. Sellers are BEARISH.")

# ============================================
# 🎯 TRADING INSIGHTS - SELLER PERSPECTIVE
# ============================================
st.markdown("---")
st.markdown("## 💡 SELLER'S TRADING INSIGHTS")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 KEY OBSERVATIONS")
    
    # Max Pain insight
    if seller_max_pain:
        max_pain_insight = ""
        if spot > seller_max_pain:
            max_pain_insight = f"Spot ABOVE max pain (₹{seller_max_pain:,}). Sellers losing on CALLs, gaining on PUTs."
        else:
            max_pain_insight = f"Spot BELOW max pain (₹{seller_max_pain:,}). Sellers gaining on CALLs, losing on PUTs."
        
        st.info(f"**Max Pain:** {max_pain_insight}")
    
    # GEX insight
    if total_gex_net > 0:
        st.success("**Gamma Exposure:** Sellers SHORT gamma. Expect reduced volatility and mean reversion.")
    elif total_gex_net < 0:
        st.warning("**Gamma Exposure:** Sellers LONG gamma. Expect increased volatility and momentum moves.")
    
    # PCR insight
    total_pcr = total_PE_OI / total_CE_OI if total_CE_OI > 0 else 0
    if total_pcr > 1.5:
        st.success(f"**Overall PCR ({total_pcr:.2f}):** Strong PUT selling dominance. Bullish seller conviction.")
    elif total_pcr < 0.7:
        st.error(f"**Overall PCR ({total_pcr:.2f}):** Strong CALL selling dominance. Bearish seller conviction.")

with insight_col2:
    st.markdown("### 🛡️ RISK MANAGEMENT")
    
    # Nearest levels insight
    if nearest_sup and nearest_res:
        risk_reward = (nearest_res["distance"] / nearest_sup["distance"]) if nearest_sup["distance"] > 0 else 0
        
        st.metric("Risk:Reward (Current Range)", f"1:{risk_reward:.2f}")
        
        # Stop loss suggestion
        if seller_bias_result["bias"].startswith("BULLISH"):
            stop_loss = f"Below seller support: ₹{nearest_sup['strike']:,}"
            target = f"Seller resistance: ₹{nearest_res['strike']:,}"
        elif seller_bias_result["bias"].startswith("BEARISH"):
            stop_loss = f"Above seller resistance: ₹{nearest_res['strike']:,}"
            target = f"Seller support: ₹{nearest_sup['strike']:,}"
        else:
            stop_loss = f"Range: ₹{nearest_sup['strike']:,} - ₹{nearest_res['strike']:,}"
            target = "Wait for breakout"
        
        st.info(f"**Stop Loss:** {stop_loss}")
        st.info(f"**Target:** {target}")

# Final Seller Summary
st.markdown("---")
st.markdown(f"""
<div class='seller-explanation'>
    <h3>🎯 FINAL SELLER ASSESSMENT</h3>
    <p><strong>Market Makers are telling us:</strong> {seller_bias_result["explanation"]}</p>
    <p><strong>Their game plan:</strong> {seller_bias_result["action"]}</p>
    <p><strong>Key defense levels:</strong> ₹{nearest_sup['strike'] if nearest_sup else 'N/A':,} (Support) | ₹{nearest_res['strike'] if nearest_res else 'N/A':,} (Resistance)</p>
    <p><strong>Preferred price level:</strong> ₹{seller_max_pain if seller_max_pain else 'N/A':,} (Max Pain)</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | ⏰ {get_ist_datetime_str()}")
st.caption("🎯 **NIFTY Option Screener v4.0 — 100% SELLER'S PERSPECTIVE** | Market Maker Logic Applied")