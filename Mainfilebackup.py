# nifty_option_screener_v4_complete_spot_analysis.py
"""
Nifty Option Screener v4.0 — COMPLETE with Spot Analysis
Combines all features: Greek calculations, GEX, PCR, Support/Resistance, Spot Position Analysis
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
#  CUSTOM CSS
# -----------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    [data-testid="stMetricLabel"] { color: #9ba4b5 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #00d4aa !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    h1, h2, h3 { color: #00d4aa !important; }
    
    .level-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00d4aa;
        margin: 5px 0;
    }
    .level-card h4 { margin: 0; color: #00d4aa; font-size: 1.1rem; }
    .level-card p { margin: 5px 0; color: #fafafa; font-size: 1.3rem; font-weight: 700; }
    .level-card .sub-info { font-size: 0.9rem; color: #9ba4b5; margin-top: 5px; }
    
    .spot-card {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff6600;
        margin: 10px 0;
        text-align: center;
    }
    .spot-card h3 { margin: 0; color: #ff6600; font-size: 1.3rem; }
    .spot-card .spot-price { font-size: 2.5rem; color: #ffaa00; font-weight: 700; margin: 10px 0; }
    .spot-card .distance { font-size: 1.1rem; color: #ffcc44; margin: 5px 0; }
    
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
    
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .support-building { background-color: #1a2e1a; border-left-color: #00ff88; color: #00ff88; }
    .support-breaking { background-color: #2e1a1a; border-left-color: #ff4444; color: #ff6666; }
    .resistance-building { background-color: #2e2a1a; border-left-color: #ffaa00; color: #ffcc44; }
    .resistance-breaking { background-color: #1a1f2e; border-left-color: #00aaff; color: #00ccff; }
    .bull-trap { background-color: #3e1a1a; border-left-color: #ff0000; color: #ff4444; font-weight: 700; }
    .bear-trap { background-color: #1a3e1a; border-left-color: #ffff00; color: #ffff66; font-weight: 700; }
    .no-trap { background-color: #1a2e2e; border-left-color: #00ffcc; color: #00ffcc; }
    
    .ist-time {
        background-color: #1a1f2e;
        color: #00d4aa;
        padding: 8px 15px;
        border-radius: 20px;
        border: 2px solid #00d4aa;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background-color: #00d4aa !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #00ffcc !important; }
    
    .greeks-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
    }
    .greeks-box h5 { margin: 0; color: #66b3ff; font-size: 0.9rem; }
    .greeks-box .greek-value { font-size: 1.2rem; color: #66b3ff; font-weight: 700; }
    
    .heatmap-positive { background-color: #1a2e1a; color: #00ff88; }
    .heatmap-negative { background-color: #2e1a1a; color: #ff4444; }
    .heatmap-neutral { background-color: #1a1f2e; color: #cccccc; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Nifty Screener v4 - Complete Spot Analysis", layout="wide")

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

def strike_strength_score(row, weights=SCORE_WEIGHTS):
    chg_oi = safe_float(row.get("Chg_OI_CE",0)) + safe_float(row.get("Chg_OI_PE",0))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def price_oi_divergence_label(chg_oi, vol, ltp_change):
    vol_up = vol>0
    oi_up = chg_oi>0
    price_up = (ltp_change is not None and ltp_change>0)
    if oi_up and vol_up and price_up:
        return "Fresh Build (Aggressive)"
    if oi_up and vol_up and not price_up:
        return "Seller Aggressive / Short Build"
    if not oi_up and vol_up and price_up:
        return "Long Covering (Buy squeeze)"
    if not oi_up and vol_up and not price_up:
        return "Put Covering / Sell w exit"
    if oi_up and not vol_up:
        return "Weak Build"
    if (not oi_up) and not vol_up:
        return "Weak Unwind"
    return "Neutral"

def interpret_itm_otm(strike, atm, chg_oi_ce, chg_oi_pe):
    if strike < atm:
        if chg_oi_ce < 0:
            ce = "Bullish (ITM CE Unwind)"
        elif chg_oi_ce > 0:
            ce = "Bearish (ITM CE Build)"
        else:
            ce = "NoSign (ITM CE)"
    elif strike > atm:
        if chg_oi_ce > 0:
            ce = "Resistance Forming (OTM CE Build)"
        elif chg_oi_ce < 0:
            ce = "Resistance Weakening (OTM CE Unwind) → Bullish"
        else:
            ce = "NoSign (OTM CE)"
    else:
        ce = "ATM CE zone"

    if strike > atm:
        if chg_oi_pe < 0:
            pe = "Bullish (ITM PE Unwind)"
        elif chg_oi_pe > 0:
            pe = "Bearish (ITM PE Build)"
        else:
            pe = "NoSign (ITM PE)"
    elif strike < atm:
        if chg_oi_pe > 0:
            pe = "Support Forming (OTM PE Build)"
        elif chg_oi_pe < 0:
            pe = "Support Weakening (OTM PE Unwind) → Bearish"
        else:
            pe = "NoSign (OTM PE)"
    else:
        pe = "ATM PE zone"

    return f"{ce} | {pe}"

def gamma_pressure_metric(row, atm, strike_gap):
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    chg_oi_sum = safe_float(row.get("Chg_OI_CE",0)) - safe_float(row.get("Chg_OI_PE",0))
    return chg_oi_sum / dist

def breakout_probability_index(merged_df, atm, strike_gap):
    near_mask = merged_df["strikePrice"].between(atm-strike_gap, atm+strike_gap)
    atm_chg_oi = merged_df.loc[near_mask, ["Chg_OI_CE","Chg_OI_PE"]].abs().sum().sum()
    atm_score = min(atm_chg_oi/50000.0, 1.0)
    winding_count = (merged_df[["CE_Winding","PE_Winding"]]=="Winding").sum().sum()
    unwinding_count = (merged_df[["CE_Winding","PE_Winding"]]=="Unwinding").sum().sum()
    winding_balance = winding_count/(winding_count+unwinding_count) if (winding_count+unwinding_count)>0 else 0.5
    vol_oi_scores = (merged_df[["Vol_CE","Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE","Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum()/100000.0, 1.0)
    gamma = merged_df.apply(lambda r: gamma_pressure_metric(r, atm, strike_gap), axis=1).abs().sum()
    gamma_score = min(gamma/10000.0, 1.0)
    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"]*atm_score) + (w["winding_balance"]*winding_balance) + (w["vol_oi_div"]*vol_oi_score) + (w["gamma_pressure"]*gamma_score)
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

# -----------------------
# 🔥 NEW: SPOT POSITION ANALYSIS
# -----------------------
def analyze_spot_position(spot, ranked_df):
    """
    Analyze spot position relative to all support/resistance levels
    Returns nearest support below, nearest resistance above, and next levels
    """
    # Sort by strike price
    sorted_df = ranked_df.sort_values("strikePrice").reset_index(drop=True)
    
    # Separate supports and resistances
    supports = sorted_df.sort_values("support_score", ascending=False).copy()
    resistances = sorted_df.sort_values("resistance_score", ascending=False).copy()
    
    # All strikes sorted
    all_strikes = sorted_df["strikePrice"].tolist()
    
    # Find nearest support (highest strike below spot)
    supports_below = [s for s in all_strikes if s < spot]
    nearest_support = max(supports_below) if supports_below else None
    
    # Find next support (second highest below spot)
    next_support = None
    if len(supports_below) >= 2:
        next_support = sorted(supports_below, reverse=True)[1]
    
    # Find nearest resistance (lowest strike above spot)
    resistances_above = [s for s in all_strikes if s > spot]
    nearest_resistance = min(resistances_above) if resistances_above else None
    
    # Find next resistance (second lowest above spot)
    next_resistance = None
    if len(resistances_above) >= 2:
        next_resistance = sorted(resistances_above)[1]
    
    # Get details for each level
    def get_level_details(strike, df):
        if strike is None:
            return None
        row = df[df["strikePrice"] == strike]
        if row.empty:
            return None
        return {
            "strike": int(strike),
            "oi_ce": int(row.iloc[0]["OI_CE"]),
            "oi_pe": int(row.iloc[0]["OI_PE"]),
            "chg_oi_ce": int(row.iloc[0].get("Chg_OI_CE", 0)),
            "chg_oi_pe": int(row.iloc[0].get("Chg_OI_PE", 0)),
            "pcr": row.iloc[0]["PCR"],
            "distance": abs(spot - strike),
            "distance_pct": abs(spot - strike) / spot * 100
        }
    
    return {
        "nearest_support": get_level_details(nearest_support, sorted_df),
        "next_support": get_level_details(next_support, sorted_df),
        "nearest_resistance": get_level_details(nearest_resistance, sorted_df),
        "next_resistance": get_level_details(next_resistance, sorted_df),
        "spot_in_range": (nearest_support, nearest_resistance)
    }

# -----------------------
# PCR FUNCTIONS
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

def rank_support_resistance_current(pcr_df):
    eps = 1e-6
    t = pcr_df.copy()
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    t["support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    t["resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["resistance_score"] = t["OI_CE"] + (t["resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("support_score", ascending=False).head(3)
    top_resists = t.sort_values("resistance_score", ascending=False).head(3)
    
    return t, top_supports, top_resists

def create_snapshot_tag():
    return get_ist_now().isoformat(timespec="seconds")

def save_pcr_snapshot_to_supabase(df_for_save, expiry, spot):
    if df_for_save is None or df_for_save.empty:
        return False, None, "no data"
    
    snapshot_tag = create_snapshot_tag()
    date_str = get_ist_date_str()
    time_str = get_ist_time_str()
    
    payload = []
    for _, r in df_for_save.iterrows():
        payload.append({
            "snapshot_tag": snapshot_tag,
            "date": date_str,
            "time": time_str,
            "expiry": expiry,
            "spot": float(spot or 0.0),
            "strike": int(r["strikePrice"]),
            "oi_ce": int(r.get("OI_CE",0)),
            "oi_pe": int(r.get("OI_PE",0)),
            "chg_oi_ce": int(r.get("Chg_OI_CE",0)) if "Chg_OI_CE" in r else 0,
            "chg_oi_pe": int(r.get("Chg_OI_PE",0)) if "Chg_OI_PE" in r else 0,
            "pcr": float(np.nan if pd.isna(r.get("PCR")) or np.isinf(r.get("PCR")) else r.get("PCR")),
            "ltp_ce": float(r.get("LTP_CE", 0.0)) if "LTP_CE" in r else 0.0,
            "ltp_pe": float(r.get("LTP_PE", 0.0)) if "LTP_PE" in r else 0.0
        })
    try:
        batch_size = 200
        for i in range(0, len(payload), batch_size):
            chunk = payload[i:i+batch_size]
            res = supabase.table(SUPABASE_TABLE_PCR).insert(chunk).execute()
            if res.status_code not in (200,201,204):
                return False, None, f"Insert failed {res.status_code}"
        return True, snapshot_tag, "saved"
    except Exception as e:
        return False, None, str(e)

def get_last_two_snapshot_tags():
    try:
        resp = supabase.table(SUPABASE_TABLE_PCR).select("snapshot_tag, created_at").order("created_at", desc=True).limit(2000).execute()
        if resp.status_code not in (200,201):
            return []
        rows = resp.data or []
        tags = []
        for r in rows:
            tag = r.get("snapshot_tag")
            if tag and tag not in tags:
                tags.append(tag)
            if len(tags) >= 2:
                break
        return tags
    except Exception as e:
        return []

def fetch_pcr_snapshot_by_tag(tag):
    try:
        resp = supabase.table(SUPABASE_TABLE_PCR).select("*").eq("snapshot_tag", tag).order("strike", {"ascending": True}).execute()
        if resp.status_code not in (200,201):
            return pd.DataFrame()
        rows = resp.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        mapping = {
            "strike":"strikePrice", "oi_ce":"OI_CE", "oi_pe":"OI_PE",
            "chg_oi_ce":"Chg_OI_CE","chg_oi_pe":"Chg_OI_PE",
            "pcr":"PCR","ltp_ce":"LTP_CE","ltp_pe":"LTP_PE"
        }
        for col in mapping.keys():
            if col not in df.columns:
                df[col] = 0
        df = df.rename(columns=mapping)
        for c in ["strikePrice","OI_CE","OI_PE","Chg_OI_CE","Chg_OI_PE","PCR","LTP_CE","LTP_PE"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df.sort_values("strikePrice").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()

def evaluate_trend(current_df, prev_df):
    if current_df is None or current_df.empty:
        return pd.DataFrame()
    current = current_df.set_index("strikePrice")
    prev = prev_df.set_index("strikePrice") if (prev_df is not None and not prev_df.empty) else pd.DataFrame()
    all_strikes = sorted(list(set(current.index).union(set(prev.index))))
    cur = current.reindex(all_strikes).fillna(0)
    prv = prev.reindex(all_strikes).fillna(0) if not prev.empty else pd.DataFrame(index=all_strikes).fillna(0)
    
    deltas = pd.DataFrame(index=all_strikes)
    deltas["OI_CE_now"] = cur["OI_CE"]
    deltas["OI_PE_now"] = cur["OI_PE"]
    deltas["PCR_now"] = cur["PCR"]
    deltas["OI_CE_prev"] = prv["OI_CE"] if "OI_CE" in prv.columns else 0
    deltas["OI_PE_prev"] = prv["OI_PE"] if "OI_PE" in prv.columns else 0
    deltas["PCR_prev"] = prv["PCR"] if "PCR" in prv.columns else 0
    deltas["ΔOI_CE"] = deltas["OI_CE_now"] - deltas["OI_CE_prev"]
    deltas["ΔOI_PE"] = deltas["OI_PE_now"] - deltas["OI_PE_prev"]
    deltas["ΔPCR"] = deltas["PCR_now"] - deltas["PCR_prev"]
    
    def label(row):
        oi_pe_up = row["ΔOI_PE"] > 0
        oi_pe_down = row["ΔOI_PE"] < 0
        oi_ce_up = row["ΔOI_CE"] > 0
        oi_ce_down = row["ΔOI_CE"] < 0
        pcr_up = row["ΔPCR"] > 0.05
        pcr_down = row["ΔPCR"] < -0.05
        
        if oi_pe_up and pcr_up:
            return "Support Building"
        if oi_pe_down and pcr_down:
            return "Support Breaking"
        if oi_ce_up and pcr_down:
            return "Resistance Building"
        if oi_ce_down and pcr_up:
            return "Resistance Breaking"
        if abs(row["ΔPCR"]) > 0.2:
            return "PCR Rapid Change"
        return "Neutral"
    
    deltas["Trend"] = deltas.apply(label, axis=1)
    deltas = deltas.reset_index().rename(columns={"index":"strikePrice"})
    return deltas

def rank_support_resistance(trend_df):
    """
    Rank supports and resistances using PCR + OI scores
    """
    eps = 1e-6
    t = trend_df.copy()
    t["PCR_now_clipped"] = t["PCR_now"].replace([np.inf, -np.inf], np.nan).fillna(0)
    t["support_score"] = t["OI_PE_now"] + (t["PCR_now_clipped"] * 100000.0)
    t["resistance_factor"] = t["PCR_now_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["resistance_score"] = t["OI_CE_now"] + (t["resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("support_score", ascending=False).head(3)["strikePrice"].astype(int).tolist()
    top_resists = t.sort_values("resistance_score", ascending=False).head(3)["strikePrice"].astype(int).tolist()
    return t, top_supports, top_resists

def detect_fake_breakout(spot, strong_support, strong_resist, trend_df):
    """Detect potential fake breakouts"""
    fake = None
    fake_hint = ""
    
    if strong_resist is not None and spot > strong_resist:
        row = trend_df[trend_df["strikePrice"]==strong_resist]
        if not row.empty and row.iloc[0]["ΔOI_CE"] > 0:
            fake = "Bull Trap"
            fake_hint = f"Price above resistance {strong_resist} but CE OI building → possible fake upside."
    
    if strong_support is not None and spot < strong_support:
        row = trend_df[trend_df["strikePrice"]==strong_support]
        if not row.empty and row.iloc[0]["ΔOI_PE"] > 0:
            fake = "Bear Trap"
            fake_hint = f"Price below support {strong_support} but PE OI building → possible fake downside."
    
    return fake, fake_hint

def generate_stop_loss_hint(spot, top_supports, top_resists, fake_type):
    """Generate stop-loss hint based on support/resistance levels"""
    if fake_type == "Bull Trap":
        return f"Keep SL near strong support {top_supports[0] if top_supports else 'N/A'} — trap likely reverse down."
    if fake_type == "Bear Trap":
        return f"Keep SL near strong resistance {top_resists[0] if top_resists else 'N/A'} — trap likely reverse up."
    if top_resists and spot > top_resists[0]:
        return f"Real upside: place SL near 2nd strongest support {top_supports[1] if len(top_supports)>1 else 'N/A'}"
    if top_supports and spot < top_supports[0]:
        return f"Real downside: place SL near 2nd strongest resistance {top_resists[1] if len(top_resists)>1 else 'N/A'}"
    return f"No clear breakout — SL inside range between {top_supports[1] if len(top_supports)>1 else 'N/A'} and {top_resists[1] if len(top_resists)>1 else 'N/A'}"

# -----------------------
#  SUPABASE DATABASE SETUP
# -----------------------
def create_tables_if_not_exist():
    """Check if tables exist"""
    try:
        res1 = supabase.table(SUPABASE_TABLE).select("id").limit(1).execute()
        res2 = supabase.table(SUPABASE_TABLE_PCR).select("id").limit(1).execute()
        
        if res1.status_code == 200 and res2.status_code == 200:
            return True
        return False
    except Exception as e:
        return False

def get_sql_for_tables():
    return """
-- Table 1: option_snapshots (for time-window snapshots)
CREATE TABLE IF NOT EXISTS option_snapshots (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    time_window VARCHAR(20) NOT NULL,
    underlying DECIMAL(10,2) NOT NULL,
    strike INTEGER NOT NULL,
    oi_ce BIGINT DEFAULT 0,
    chg_oi_ce BIGINT DEFAULT 0,
    vol_ce BIGINT DEFAULT 0,
    ltp_ce DECIMAL(10,2) DEFAULT 0.0,
    iv_ce DECIMAL(10,4) DEFAULT 0.0,
    oi_pe BIGINT DEFAULT 0,
    chg_oi_pe BIGINT DEFAULT 0,
    vol_pe BIGINT DEFAULT 0,
    ltp_pe DECIMAL(10,2) DEFAULT 0.0,
    iv_pe DECIMAL(10,4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table 2: strike_pcr_snapshots (for PCR batch snapshots)
CREATE TABLE IF NOT EXISTS strike_pcr_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_tag VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    time VARCHAR(20) NOT NULL,
    expiry VARCHAR(20) NOT NULL,
    spot DECIMAL(10,2) NOT NULL,
    strike INTEGER NOT NULL,
    oi_ce BIGINT DEFAULT 0,
    oi_pe BIGINT DEFAULT 0,
    chg_oi_ce BIGINT DEFAULT 0,
    chg_oi_pe BIGINT DEFAULT 0,
    pcr DECIMAL(10,4),
    ltp_ce DECIMAL(10,2) DEFAULT 0.0,
    ltp_pe DECIMAL(10,2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_option_snapshots_date_window ON option_snapshots(date, time_window);
CREATE INDEX IF NOT EXISTS idx_pcr_snapshot_tag ON strike_pcr_snapshots(snapshot_tag);
"""

# -----------------------
#  DHAN API
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
#  SUPABASE SNAPSHOT HELPERS
# -----------------------
def supabase_snapshot_exists(date_str, window):
    try:
        resp = supabase.table(SUPABASE_TABLE).select("id").eq("date", date_str).eq("time_window", window).limit(1).execute()
        if resp.status_code == 200:
            data = resp.data
            return len(data) > 0
        return False
    except Exception as e:
        return False

def save_snapshot_to_supabase(df, window, underlying):
    """Insert rows (one per strike) into supabase table with date & time_window."""
    if df is None or df.empty:
        return False, "no data"
    date_str = get_ist_date_str()
    payload = []
    for _, r in df.iterrows():
        payload.append({
            "date": date_str,
            "time_window": window,
            "underlying": float(underlying),
            "strike": int(r.get("strikePrice",0)),
            "oi_ce": int(safe_int(r.get("OI_CE",0))),
            "chg_oi_ce": int(safe_int(r.get("Chg_OI_CE",0))),
            "vol_ce": int(safe_int(r.get("Vol_CE",0))),
            "ltp_ce": float(safe_float(r.get("LTP_CE", np.nan)) or 0.0),
            "iv_ce": float(safe_float(r.get("IV_CE", np.nan) or 0.0)),
            "oi_pe": int(safe_int(r.get("OI_PE",0))),
            "chg_oi_pe": int(safe_int(r.get("Chg_OI_PE",0))),
            "vol_pe": int(safe_int(r.get("Vol_PE",0))),
            "ltp_pe": float(safe_float(r.get("LTP_PE", np.nan) or 0.0)),
            "iv_pe": float(safe_float(r.get("IV_PE", np.nan) or 0.0))
        })
    try:
        batch_size = 200
        for i in range(0, len(payload), batch_size):
            chunk = payload[i:i+batch_size]
            res = supabase.table(SUPABASE_TABLE).insert(chunk).execute()
            if res.status_code not in (200,201,204):
                return False, f"supabase insert returned {res.status_code}"
        return True, "saved"
    except Exception as e:
        return False, str(e)

# -----------------------
#  MAIN APP
# -----------------------
st.title("🎯 NIFTY Option Screener v4.0 — Complete Spot Analysis")

current_ist = get_ist_datetime_str()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <span class='ist-time'>🕐 IST: {current_ist}</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"### 🕐 Current IST")
    st.markdown(f"**{get_ist_time_str()}**")
    st.markdown(f"**{get_ist_date_str()}**")
    st.markdown("---")
    
    # Database setup
    st.subheader("Database Setup")
    if st.button("Check Database Connection"):
        if create_tables_if_not_exist():
            st.success("✅ Database connection successful!")
        else:
            st.warning("⚠️ Some tables may be missing")
    
    if st.button("Show SQL for Tables"):
        sql_commands = get_sql_for_tables()
        st.code(sql_commands, language="sql")
    
    st.markdown("---")
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
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# Compute per-strike metrics including Greeks
for i, row in merged.iterrows():
    strike = int(row["strikePrice"])
    ltp_ce = safe_float(row.get("LTP_CE",0.0))
    ltp_pe = safe_float(row.get("LTP_PE",0.0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))

    key_ce = f"{expiry}_{strike}_CE"
    key_pe = f"{expiry}_{strike}_PE"
    prev_ce = st.session_state["prev_ltps_v3"].get(key_ce, None)
    prev_pe = st.session_state["prev_ltps_v3"].get(key_pe, None)
    prev_iv_ce = st.session_state["prev_ivs_v3"].get(key_ce, None)
    prev_iv_pe = st.session_state["prev_ivs_v3"].get(key_pe, None)

    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)
    ce_iv_delta = None if prev_iv_ce is None else (iv_ce - prev_iv_ce)
    pe_iv_delta = None if prev_iv_pe is None else (iv_pe - prev_iv_pe)

    st.session_state["prev_ltps_v3"][key_ce] = ltp_ce
    st.session_state["prev_ltps_v3"][key_pe] = ltp_pe
    st.session_state["prev_ivs_v3"][key_ce] = iv_ce
    st.session_state["prev_ivs_v3"][key_pe] = iv_pe

    chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))

    merged.at[i,"CE_Winding"] = "Winding" if chg_oi_ce>0 else ("Unwinding" if chg_oi_ce<0 else "NoChange")
    merged.at[i,"PE_Winding"] = "Winding" if chg_oi_pe>0 else ("Unwinding" if chg_oi_pe<0 else "NoChange")

    merged.at[i,"CE_Divergence"] = price_oi_divergence_label(chg_oi_ce, safe_int(row.get("Vol_CE",0)), ce_price_delta)
    merged.at[i,"PE_Divergence"] = price_oi_divergence_label(chg_oi_pe, safe_int(row.get("Vol_PE",0)), pe_price_delta)

    merged.at[i,"Interpretation"] = interpret_itm_otm(strike, atm_strike, chg_oi_ce, chg_oi_pe)

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

    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    gex_ce = gamma_ce * notional * oi_ce
    gex_pe = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_CE"] = gex_ce
    merged.at[i,"GEX_PE"] = gex_pe
    merged.at[i,"GEX_Net"] = gex_ce - gex_pe

    merged.at[i,"Strength_Score"] = strike_strength_score(row)
    merged.at[i,"Gamma_Pressure"] = gamma_pressure_metric(row, atm_strike, strike_gap)

    merged.at[i,"CE_Price_Delta"] = ce_price_delta
    merged.at[i,"PE_Price_Delta"] = pe_price_delta
    merged.at[i,"CE_IV_Delta"] = ce_iv_delta
    merged.at[i,"PE_IV_Delta"] = pe_iv_delta

# Aggregations & summaries
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

itm_ce_mask = merged["strikePrice"] < atm_strike
otm_ce_mask = merged["strikePrice"] > atm_strike
itm_pe_mask = merged["strikePrice"] > atm_strike
otm_pe_mask = merged["strikePrice"] < atm_strike

ITM_CE_OI = merged.loc[itm_ce_mask, "OI_CE"].sum()
OTM_CE_OI = merged.loc[otm_ce_mask, "OI_CE"].sum()
ITM_PE_OI = merged.loc[itm_pe_mask, "OI_PE"].sum()
OTM_PE_OI = merged.loc[otm_pe_mask, "OI_PE"].sum()

ITM_CE_winding_count = (merged.loc[itm_ce_mask,"CE_Winding"]=="Winding").sum()
OTM_CE_winding_count = (merged.loc[otm_ce_mask,"CE_Winding"]=="Winding").sum()
ITM_PE_winding_count = (merged.loc[itm_pe_mask,"PE_Winding"]=="Winding").sum()
OTM_PE_winding_count = (merged.loc[otm_pe_mask,"PE_Winding"]=="Winding").sum()

itm_ce_winding_pct = (ITM_CE_winding_count / (merged.loc[itm_ce_mask].shape[0] or 1))*100
otm_ce_winding_pct = (OTM_CE_winding_count / (merged.loc[otm_ce_mask].shape[0] or 1))*100
itm_pe_winding_pct = (ITM_PE_winding_count / (merged.loc[itm_pe_mask].shape[0] or 1))*100
otm_pe_winding_pct = (OTM_PE_winding_count / (merged.loc[otm_pe_mask].shape[0] or 1))*100

merged["CE_Delta_Exposure"] = merged["Delta_CE"].fillna(0) * merged["OI_CE"].fillna(0) * LOT_SIZE
merged["PE_Delta_Exposure"] = merged["Delta_PE"].fillna(0) * merged["OI_PE"].fillna(0) * LOT_SIZE
net_delta_exposure = merged["CE_Delta_Exposure"].sum() + merged["PE_Delta_Exposure"].sum()

total_gex_ce = merged["GEX_CE"].sum()
total_gex_pe = merged["GEX_PE"].sum()
total_gex_net = merged["GEX_Net"].sum()

max_pain = None
try:
    max_pain = int(pd.Series({int(r["strikePrice"]): safe_float(r["LTP_CE"])*safe_int(r["OI_CE"]) + safe_float(r["LTP_PE"])*safe_int(r["OI_PE"]) for _, r in merged.iterrows()}).sort_values().index[0])
except Exception:
    max_pain = None

breakout_index = breakout_probability_index(merged, atm_strike, strike_gap)

ce_com = center_of_mass_oi(merged, "OI_CE")
pe_com = center_of_mass_oi(merged, "OI_PE")
atm_shift = []
if ce_com > atm_strike + strike_gap: atm_shift.append("CE build above ATM")
elif ce_com < atm_strike - strike_gap: atm_shift.append("CE build below ATM")
if pe_com > atm_strike + strike_gap: atm_shift.append("PE build above ATM")
elif pe_com < atm_strike - strike_gap: atm_shift.append("PE build below ATM")
atm_shift_str = " | ".join(atm_shift) if atm_shift else "Neutral"

# Market polarity
polarity = 0.0
for _, r in merged.iterrows():
    s = r["strikePrice"]
    chg_ce = safe_int(r.get("Chg_OI_CE",0))
    chg_pe = safe_int(r.get("Chg_OI_PE",0))
    if s < atm_strike:
        if chg_ce < 0: polarity += 1.0
        elif chg_ce > 0: polarity -= 1.0
    else:
        if chg_ce > 0: polarity -= 0.5
        elif chg_ce < 0: polarity += 0.5
    if s > atm_strike:
        if chg_pe < 0: polarity += 1.0
        elif chg_pe > 0: polarity -= 1.0
    else:
        if chg_pe > 0: polarity += 0.5
        elif chg_pe < 0: polarity -= 0.5

if polarity > 5: market_bias="Strong Bullish"
elif polarity > 1: market_bias="Bullish"
elif polarity < -5: market_bias="Strong Bearish"
elif polarity < -1: market_bias="Bearish"
else: market_bias="Neutral"

# Compute PCR
pcr_df = compute_pcr_df(merged)

# Get support/resistance rankings
ranked_current, supports_df, resists_df = rank_support_resistance_current(pcr_df)

# 🔥 ANALYZE SPOT POSITION
spot_analysis = analyze_spot_position(spot, ranked_current)

# Auto-save PCR
last_saved = st.session_state.get("last_pcr_auto_saved", 0)
if time.time() - last_saved > save_interval:
    ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
    if ok:
        st.session_state["last_pcr_auto_saved"] = time.time()

# Get trend analysis
tags = get_last_two_snapshot_tags()
trend_df = pd.DataFrame()

if len(tags) >= 2:
    cur_df = fetch_pcr_snapshot_by_tag(tags[0])
    prev_df = fetch_pcr_snapshot_by_tag(tags[1])
    
    if not cur_df.empty:
        trend_df = evaluate_trend(cur_df, prev_df)

# ============================================
# 🎯 MAIN DASHBOARD
# ============================================

st.markdown("## 📈 CORE MARKET METRICS")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot", f"₹{spot:.2f}")
    st.metric("ATM", f"₹{atm_strike}")
with col2:
    st.metric("Total CE OI", f"{int(total_CE_OI):,}")
    st.metric("Total PE OI", f"{int(total_PE_OI):,}")
with col3:
    st.metric("Total CE ΔOI", f"{int(total_CE_chg):,}")
    st.metric("Total PE ΔOI", f"{int(total_PE_chg):,}")
with col4:
    st.metric("Net Delta Exposure", f"{int(net_delta_exposure):,}")
    st.metric("Market Bias", market_bias)

st.markdown("### 🔎 ITM/OTM Pressure Summary (ATM ± 8)")
pressure_table = pd.DataFrame([
    {"Category":"ITM CE OI","Value":int(ITM_CE_OI),"Winding_Count":int(ITM_CE_winding_count),"Winding_%":f"{itm_ce_winding_pct:.1f}%"},
    {"Category":"OTM CE OI","Value":int(OTM_CE_OI),"Winding_Count":int(OTM_CE_winding_count),"Winding_%":f"{otm_ce_winding_pct:.1f}%"},
    {"Category":"ITM PE OI","Value":int(ITM_PE_OI),"Winding_Count":int(ITM_PE_winding_count),"Winding_%":f"{itm_pe_winding_pct:.1f}%"},
    {"Category":"OTM PE OI","Value":int(OTM_PE_OI),"Winding_Count":int(OTM_PE_winding_count),"Winding_%":f"{otm_pe_winding_pct:.1f}%"}
])
st.dataframe(pressure_table, use_container_width=True)

# Greeks Summary
st.markdown("### 🧮 GREEKS SUMMARY")
gex_col1, gex_col2, gex_col3, gex_col4 = st.columns(4)
with gex_col1:
    st.metric("Total GEX CE", f"₹{int(total_gex_ce):,}")
with gex_col2:
    st.metric("Total GEX PE", f"₹{int(total_gex_pe):,}")
with gex_col3:
    st.metric("Net GEX", f"₹{int(total_gex_net):,}")
with gex_col4:
    st.metric("Breakout Index", f"{breakout_index}%")

st.markdown("---")

# ============================================
# 🎯 SPOT POSITION & KEY LEVELS
# ============================================

st.markdown("## 🎯 SPOT POSITION & KEY LEVELS")

# Row 1: SPOT POSITION CARD
st.markdown("### 📍 NIFTY SPOT POSITION")

# Create spot position display
nearest_sup = spot_analysis["nearest_support"]
nearest_res = spot_analysis["nearest_resistance"]

col_spot, col_range = st.columns([1, 1])

with col_spot:
    st.markdown(f"""
    <div class="spot-card">
        <h3>🎯 CURRENT SPOT</h3>
        <div class="spot-price">₹{spot:,.2f}</div>
        <div class="distance">ATM Strike: ₹{atm_strike:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        spot_position_pct = ((spot - nearest_sup["strike"]) / range_size * 100) if range_size > 0 else 50
        
        st.markdown(f"""
        <div class="spot-card">
            <h3>📊 SPOT IN RANGE</h3>
            <div class="distance">₹{nearest_sup['strike']:,} ← SPOT → ₹{nearest_res['strike']:,}</div>
            <div class="distance">Position: {spot_position_pct:.1f}% within range</div>
            <div class="distance">Range Width: ₹{range_size:,}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Row 2: NEAREST LEVELS FROM SPOT
st.markdown("### 🎯 NEAREST SUPPORT & RESISTANCE FROM SPOT")

col_ns, col_nr = st.columns(2)

with col_ns:
    st.markdown("#### 🛡️ SUPPORT BELOW SPOT")
    
    if nearest_sup:
        sup = nearest_sup
        pcr_display = f"{sup['pcr']:.2f}" if not np.isinf(sup['pcr']) else "∞"
        
        # Get trend if available
        trend_label = ""
        if not trend_df.empty:
            trend_row = trend_df[trend_df["strikePrice"] == sup['strike']]
            if not trend_row.empty:
                trend = trend_row.iloc[0]["Trend"]
                if trend == "Support Building":
                    trend_label = " 🟢 BUILDING"
                elif trend == "Support Breaking":
                    trend_label = " 🔴 BREAKING"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>💚 NEAREST SUPPORT{trend_label}</h4>
            <div class="level-value">₹{sup['strike']:,}</div>
            <div class="level-distance">⬇️ Distance: ₹{sup['distance']:.2f} ({sup['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                PE OI: {sup['oi_pe']:,} | PCR: {pcr_display} | ΔOI: {sup['chg_oi_pe']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No support level below spot")
    
    # Next support
    next_sup = spot_analysis["next_support"]
    if next_sup:
        pcr_display = f"{next_sup['pcr']:.2f}" if not np.isinf(next_sup['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Next Support</h4>
            <p>₹{next_sup['strike']:,}</p>
            <div class="sub-info">
                ⬇️ Distance: ₹{next_sup['distance']:.2f} ({next_sup['distance_pct']:.2f}%) | PCR: {pcr_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_nr:
    st.markdown("#### ⚡ RESISTANCE ABOVE SPOT")
    
    if nearest_res:
        res = nearest_res
        pcr_display = f"{res['pcr']:.2f}" if not np.isinf(res['pcr']) else "∞"
        
        # Get trend if available
        trend_label = ""
        if not trend_df.empty:
            trend_row = trend_df[trend_df["strikePrice"] == res['strike']]
            if not trend_row.empty:
                trend = trend_row.iloc[0]["Trend"]
                if trend == "Resistance Building":
                    trend_label = " 🟡 BUILDING"
                elif trend == "Resistance Breaking":
                    trend_label = " 🔵 BREAKING"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>🧡 NEAREST RESISTANCE{trend_label}</h4>
            <div class="level-value">₹{res['strike']:,}</div>
            <div class="level-distance">⬆️ Distance: ₹{res['distance']:.2f} ({res['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                CE OI: {res['oi_ce']:,} | PCR: {pcr_display} | ΔOI: {res['chg_oi_ce']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No resistance level above spot")
    
    # Next resistance
    next_res = spot_analysis["next_resistance"]
    if next_res:
        pcr_display = f"{next_res['pcr']:.2f}" if not np.isinf(next_res['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Next Resistance</h4>
            <p>₹{next_res['strike']:,}</p>
            <div class="sub-info">
                ⬆️ Distance: ₹{next_res['distance']:.2f} ({next_res['distance_pct']:.2f}%) | PCR: {pcr_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Row 3: ALL SUPPORT & RESISTANCE LEVELS
st.markdown("### 🎯 ALL KEY LEVELS (Top 3 Each)")

col_s, col_r = st.columns(2)

with col_s:
    st.markdown("#### 🛡️ STRONGEST SUPPORTS")
    
    for i, (idx, row) in enumerate(supports_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_pe = int(row["OI_PE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        
        # Distance from spot
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        # Get trend
        trend_label = ""
        if not trend_df.empty:
            trend_row = trend_df[trend_df["strikePrice"] == strike]
            if not trend_row.empty:
                trend = trend_row.iloc[0]["Trend"]
                if trend == "Support Building":
                    trend_label = " 🟢 BUILDING"
                elif trend == "Support Breaking":
                    trend_label = " 🔴 BREAKING"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Support #{i}{trend_label}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%) | PE OI: {oi_pe:,} | PCR: {pcr_display} | ΔOI: {chg_oi_pe:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_r:
    st.markdown("#### ⚡ STRONGEST RESISTANCES")
    
    for i, (idx, row) in enumerate(resists_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_ce = int(row["OI_CE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_ce = int(row.get("Chg_OI_CE", 0))
        
        # Distance from spot
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        # Get trend
        trend_label = ""
        if not trend_df.empty:
            trend_row = trend_df[trend_df["strikePrice"] == strike]
            if not trend_row.empty:
                trend = trend_row.iloc[0]["Trend"]
                if trend == "Resistance Building":
                    trend_label = " 🟡 BUILDING"
                elif trend == "Resistance Breaking":
                    trend_label = " 🔵 BREAKING"
        
        st.markdown(f"""
        <div class="level-card">
            <h4>Resistance #{i}{trend_label}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%) | CE OI: {oi_ce:,} | PCR: {pcr_display} | ΔOI: {chg_oi_ce:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Row 4: Trading Insights
st.markdown("### 💡 TRADING INSIGHTS")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    if nearest_sup and nearest_res:
        mid_point = (nearest_sup["strike"] + nearest_res["strike"]) / 2
        if spot < mid_point:
            bias = "🔴 Bearish Bias"
            insight = f"Spot closer to support. Watch ₹{nearest_sup['strike']:,} for breakdown."
        else:
            bias = "🟢 Bullish Bias"
            insight = f"Spot closer to resistance. Watch ₹{nearest_res['strike']:,} for breakout."
        
        st.info(f"**{bias}**: {insight}")
    
    # Fake Breakout Detection
    if not trend_df.empty:
        fake_type, fake_hint = detect_fake_breakout(
            spot, 
            top_supports[0] if top_supports else None, 
            top_resists[0] if top_resists else None, 
            trend_df
        )
        if fake_type:
            st.warning(f"⚠️ **Fake Breakout Alert: {fake_type}**")
            st.info(f"{fake_hint}")

with col_insight2:
    if nearest_sup and nearest_res:
        risk_reward = (nearest_res["distance"] / nearest_sup["distance"]) if nearest_sup["distance"] > 0 else 0
        st.metric("Risk:Reward Ratio", f"1:{risk_reward:.2f}")
    
    # Max Pain
    if max_pain:
        st.metric("Max Pain", f"₹{max_pain:,}")

# Stop-loss Hint
if not trend_df.empty:
    stop_loss_hint = generate_stop_loss_hint(
        spot, 
        supports_df.head(3)["strikePrice"].tolist(), 
        resists_df.head(3)["strikePrice"].tolist(), 
        fake_type if 'fake_type' in locals() else None
    )
    st.markdown("### 🛡️ Stop-loss Hint")
    st.info(stop_loss_hint)

st.markdown("---")

# ============================================
# 📊 DETAILED DATA TABS
# ============================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 All Strikes", "🔥 PCR Analysis", "🧮 Greeks View", "💾 Snapshots"])

with tab1:
    st.markdown("### Complete Strike Table with Analysis")
    display_cols = [
        "strikePrice", "OI_CE", "Chg_OI_CE", "Vol_CE", "LTP_CE", 
        "CE_Price_Delta", "CE_IV_Delta", "CE_Winding", "CE_Divergence",
        "OI_PE", "Chg_OI_PE", "Vol_PE", "LTP_PE", 
        "PE_Price_Delta", "PE_IV_Delta", "PE_Winding", "PE_Divergence",
        "Interpretation", "Strength_Score"
    ]
    # Ensure all columns exist
    for col in display_cols:
        if col not in merged.columns:
            merged[col] = ""
    
    st.dataframe(merged[display_cols], use_container_width=True)

with tab2:
    st.markdown("### PCR & Score Analysis")
    analysis_df = ranked_current[["strikePrice", "OI_CE", "OI_PE", "PCR", "support_score", "resistance_score"]].copy()
    analysis_df["distance_from_spot"] = abs(analysis_df["strikePrice"] - spot)
    st.dataframe(analysis_df.sort_values("distance_from_spot"), use_container_width=True)
    
    # PCR Heatmap
    st.markdown("### 🔥 PCR Heatmap")
    pcr_heatmap = analysis_df[["strikePrice", "PCR"]].set_index("strikePrice")
    def color_pcr(val):
        try:
            if val > 1.5:
                return "background-color: #1a2e1a; color: #00ff88"
            elif val > 1.0:
                return "background-color: #2e2a1a; color: #ffcc44"
            elif val > 0.5:
                return "background-color: #1a1f2e; color: #66b3ff"
            else:
                return "background-color: #2e1a1a; color: #ff4444"
        except:
            return ""
    st.dataframe(pcr_heatmap.style.applymap(color_pcr), use_container_width=True)

with tab3:
    st.markdown("### 🧮 GREEKS & GEX DETAILS")
    greeks_cols = [
        "strikePrice", 
        "Delta_CE", "Gamma_CE", "Vega_CE", "Theta_CE", "GEX_CE",
        "Delta_PE", "Gamma_PE", "Vega_PE", "Theta_PE", "GEX_PE",
        "GEX_Net", "Gamma_Pressure"
    ]
    for col in greeks_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    
    # Format Greek values
    greeks_display = merged[greeks_cols].copy()
    for col in ["Delta_CE", "Delta_PE", "Gamma_CE", "Gamma_PE", "Vega_CE", "Vega_PE", "Theta_CE", "Theta_PE"]:
        if col in greeks_display.columns:
            greeks_display[col] = greeks_display[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(greeks_display, use_container_width=True)

with tab4:
    st.markdown("### Snapshot Manager")
    st.markdown(f"**Current IST:** {get_ist_datetime_str()}")
    
    # Manual save buttons
    col_man1, col_man2 = st.columns(2)
    with col_man1:
        if st.button("💾 Save PCR Snapshot"):
            ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
            if ok:
                st.success(f"Saved: {get_ist_time_str()} IST")
            else:
                st.error(f"Failed: {msg}")
    
    with col_man2:
        if st.button("📊 Save Time Snapshot (Current)"):
            ok, msg = save_snapshot_to_supabase(merged, "manual", spot)
            if ok:
                st.success(f"Saved time snapshot")
            else:
                st.error(f"Failed: {msg}")
    
    # Time window auto-save status
    st.markdown("#### Time Window Auto-save Status")
    today = get_ist_date_str()
    for w_key, w_info in TIME_WINDOWS.items():
        exists = supabase_snapshot_exists(today, w_key)
        status = "✅ Saved" if exists else "⏳ Waiting"
        st.write(f"{w_info['label']}: {status}")
    
    # Trend analysis display
    if not trend_df.empty:
        st.markdown("#### Trend Analysis")
        trend_display = trend_df.rename(columns={"OI_CE_now":"OI_CE","OI_PE_now":"OI_PE","PCR_now":"PCR"})
        active = trend_display[trend_display["Trend"] != "Neutral"]
        if not active.empty:
            st.dataframe(active[["strikePrice","OI_CE","OI_PE","ΔOI_CE","ΔOI_PE","ΔPCR","Trend"]], use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | 💾 Auto-save: {save_interval}s | ⏰ {get_ist_datetime_str()}")
st.caption("🎯 **Complete NIFTY Option Screener v4.0 with Spot Position Analysis** | All rights reserved")
