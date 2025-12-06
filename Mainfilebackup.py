# nifty_option_screener_v4_complete.py
"""
Nifty Option Screener v4.0 — DhanHQ + Supabase
Enhanced with PCR features, trend analysis, support/resistance ranking

Features:
 - ATM ± 8 strikes
 - Winding / Unwinding (CE/PE)
 - IV change analysis + Greeks (Black-Scholes approx)
 - GEX / Delta exposure
 - Max Pain, Breakout Index, ATM-shift
 - PCR per strike + Trend analysis
 - Support/Resistance ranking + Fake breakout detection
 - Stop-loss hint generation
 - Persistent snapshots stored to Supabase for windows:
    morning (09:15-10:30), mid (10:30-12:30), afternoon (14:00-15:30), evening (15:00-15:30)
 - Snapshot compare UI with trend labels
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from math import log, sqrt
from scipy.stats import norm
from supabase import create_client, Client

# -----------------------
#  CONFIG (tunable)
# -----------------------
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300  # auto-save interval for PCR snapshots
DEFAULT_CE_OI_FLOOR = 1  # avoid division by zero for PCR calc

# Time windows
TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30)"}
}

# -----------------------
#  SECRETS / CLIENTS
# -----------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    SUPABASE_TABLE_PCR = st.secrets.get("SUPABASE_TABLE_PCR", "strike_pcr_snapshots")
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except Exception as e:
    st.error("""
    ❌ Missing credentials in Streamlit secrets. 
    
    Required secrets in .streamlit/secrets.toml:
    SUPABASE_URL = "your-supabase-url"
    SUPABASE_ANON_KEY = "your-supabase-anon-key"
    SUPABASE_TABLE = "option_snapshots"
    SUPABASE_TABLE_PCR = "strike_pcr_snapshots"
    DHAN_CLIENT_ID = "your-dhan-client-id"
    DHAN_ACCESS_TOKEN = "your-dhan-access-token"
    """)
    st.stop()

# Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"❌ Supabase client creation failed: {e}")
    st.stop()

# Dhan base config
DHAN_BASE_URL = "https://api.dhan.co"
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# -----------------------
#  Custom CSS for better UI colors
# -----------------------
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #9ba4b5 !important;
        font-weight: 600;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-size: 1.8rem !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4aa !important;
    }
    
    /* Info boxes */
    .stAlert > div {
        background-color: #1a1f2e !important;
        border: 1px solid #00d4aa !important;
        color: #fafafa !important;
    }
    
    /* Success boxes */
    .stSuccess > div {
        background-color: #1a2e1a !important;
        border: 1px solid #00d4aa !important;
    }
    
    /* Warning boxes */
    .stWarning > div {
        background-color: #2e2a1a !important;
        border: 1px solid #ffa500 !important;
    }
    
    /* Error boxes */
    .stError > div {
        background-color: #2e1a1a !important;
        border: 1px solid #ff4b4b !important;
    }
    
    /* Dataframe */
    .dataframe {
        color: #fafafa !important;
        background-color: #1a1f2e !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00d4aa !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #00ffcc !important;
        border: 1px solid #00ffcc !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1f2e !important;
        color: #fafafa !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1f2e !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9ba4b5 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4aa !important;
        border-bottom-color: #00d4aa !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
#  Helpers: auto-refresh
# -----------------------
st.set_page_config(page_title="Nifty Option Screener v4 (Dhan + Supabase)", layout="wide")

def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh()

# -----------------------
#  Utility functions
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

# Black-Scholes helpers
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
            ce = "Resistance Weakening (OTM CE Unwind) -> Bullish"
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
            pe = "Support Weakening (OTM PE Unwind) -> Bearish"
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
# PCR Functions
# -----------------------
def compute_pcr_df(merged_df):
    """Compute PCR per strike safely and return DataFrame with PCR"""
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

def create_snapshot_tag():
    """Create unique batch id for PCR snapshots"""
    return datetime.utcnow().isoformat(timespec="seconds")

def save_pcr_snapshot_to_supabase(df_for_save, expiry, spot):
    """Save batch PCR snapshot to Supabase table. Returns (ok, snapshot_tag, message)."""
    if df_for_save is None or df_for_save.empty:
        return False, None, "no data"
    snapshot_tag = create_snapshot_tag()
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
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
                return False, None, f"Supabase insert failed status {res.status_code}"
        return True, snapshot_tag, "saved"
    except Exception as e:
        return False, None, str(e)

def get_last_two_snapshot_tags(expiry=None, date_filter=None):
    """Return last two distinct snapshot_tag values (most recent first)."""
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
        st.warning(f"Could not get snapshot tags: {e}")
        return []

def fetch_pcr_snapshot_by_tag(tag):
    """Return DataFrame for a given snapshot_tag"""
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
        st.warning(f"Fetch PCR snapshot by tag failed: {e}")
        return pd.DataFrame()

def evaluate_trend(current_df, prev_df):
    """Evaluate trend with labels"""
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
    """Rank supports and resistances using PCR + OI scores"""
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
#  Dhan API helpers - FIXED
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_spot_price():
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        payload = {"IDX_I": ["13"]}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID  # Fixed: lowercase 'client-id'
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            idx = data.get("data",{}).get("IDX_I",{})
            nifty = idx.get("13",{})
            return float(nifty.get("last_price", 0.0))
        return 0.0
    except Exception as e:
        st.warning(f"Dhan LTP fetch failed: {e}")
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
            "client-id": DHAN_CLIENT_ID  # Fixed: lowercase 'client-id'
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",[])
        return []
    except Exception as e:
        st.warning(f"Expiry list fetch failed: {e}")
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
            "client-id": DHAN_CLIENT_ID  # Fixed: lowercase 'client-id'
        }
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",{})
        else:
            st.warning(f"Option chain returned non-success: {data}")
            return None
    except Exception as e:
        st.warning(f"Option chain fetch failed: {e}")
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
#  Supabase snapshot helpers
# -----------------------
def supabase_snapshot_exists(date_str, window):
    try:
        resp = supabase.table(SUPABASE_TABLE).select("id").eq("date", date_str).eq("time_window", window).limit(1).execute()
        if resp.status_code == 200:
            data = resp.data
            return len(data) > 0
        return False
    except Exception as e:
        st.warning(f"Supabase check exists failed: {e}")
        return False

def save_snapshot_to_supabase(df, window, underlying):
    """Insert rows into supabase table"""
    if df is None or df.empty:
        return False, "no data"
    date_str = datetime.now().strftime("%Y-%m-%d")
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

def fetch_snapshot_from_supabase(date_str, window):
    """Return a pandas DataFrame for requested date/time_window"""
    try:
        resp = supabase.table(SUPABASE_TABLE).select("*").eq("date", date_str).eq("time_window", window).execute()
        if resp.status_code == 200:
            rows = resp.data
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            expected = ["strike","oi_ce","chg_oi_ce","vol_ce","ltp_ce","iv_ce","oi_pe","chg_oi_pe","vol_pe","ltp_pe","iv_pe","underlying","time_window"]
            for col in expected:
                if col not in df.columns:
                    df[col] = np.nan
            df = df.rename(columns={
                "strike":"strikePrice",
                "time_window":"window",
                "oi_ce":"OI_CE","chg_oi_ce":"Chg_OI_CE","vol_ce":"Vol_CE","ltp_ce":"LTP_CE","iv_ce":"IV_CE",
                "oi_pe":"OI_PE","chg_oi_pe":"Chg_OI_PE","vol_pe":"Vol_PE","ltp_pe":"LTP_PE","iv_pe":"IV_PE"
            })
            for c in ["strikePrice","OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","IV_CE","OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","IV_PE"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("strikePrice").reset_index(drop=True)
        else:
            st.warning(f"Supabase fetch returned status {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Supabase fetch failed: {e}")
        return pd.DataFrame()

def compare_snapshots_df(old_df, new_df):
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        return None
    old = old_df.set_index("strikePrice")
    new = new_df.set_index("strikePrice")
    all_strikes = sorted(list(set(old.index).union(set(new.index))))
    old = old.reindex(all_strikes).fillna(0)
    new = new.reindex(all_strikes).fillna(0)
    cols = ["OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","IV_CE","OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","IV_PE"]
    diff = (new[cols] - old[cols]).reset_index()
    diff.columns = ["strikePrice"] + [f"Δ{c}" for c in cols]
    diff["abs_change"] = diff[[f"ΔChg_OI_CE", f"ΔChg_OI_PE"]].abs().sum(axis=1)
    return diff.sort_values("abs_change", ascending=False).reset_index(drop=True)

def snapshot_summary(df):
    if df is None or df.empty:
        return {"CE_OI":0,"PE_OI":0,"CE_ΔOI":0,"PE_ΔOI":0,"CE_Vol":0,"PE_Vol":0,"Avg_IV_CE":0,"Avg_IV_PE":0}
    return {
        "CE_OI": int(df["OI_CE"].sum()),
        "PE_OI": int(df["OI_PE"].sum()),
        "CE_ΔOI": int(df["Chg_OI_CE"].sum()),
        "PE_ΔOI": int(df["Chg_OI_PE"].sum()),
        "CE_Vol": int(df["Vol_CE"].sum()),
        "PE_Vol": int(df["Vol_PE"].sum()),
        "Avg_IV_CE": float(df["IV_CE"].dropna().mean()) if "IV_CE" in df.columns else 0,
        "Avg_IV_PE": float(df["IV_PE"].dropna().mean()) if "IV_PE" in df.columns else 0
    }

def bias_score_from_summary(summary):
    score = 0.0
    score += -summary["CE_ΔOI"] / 10000.0
    score += summary["PE_ΔOI"] / 10000.0
    score += (summary["PE_Vol"] - summary["CE_Vol"]) / 100000.0
    return score

# -----------------------
#  Database Setup Functions
# -----------------------
def create_tables_if_not_exist():
    """Check if tables exist"""
    try:
        res1 = supabase.table(SUPABASE_TABLE).select("id").limit(1).execute()
        res2 = supabase.table(SUPABASE_TABLE_PCR).select("id").limit(1).execute()
        
        if res1.status_code == 200 and res2.status_code == 200:
            st.sidebar.success("✅ Both Supabase tables exist")
            return True
        else:
            st.sidebar.warning("Some tables missing. Use SQL below to create them.")
            return False
    except Exception as e:
        st.sidebar.warning(f"Table check failed: {e}")
        return False

def get_sql_for_tables():
    """Return SQL commands to create required tables"""
    return """
-- SUPABASE TABLES SETUP FOR NIFTY OPTION SCREENER
-- Run these SQL commands in Supabase SQL Editor

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

CREATE INDEX IF NOT EXISTS idx_option_snapshots_date_window ON option_snapshots(date, time_window);

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

CREATE INDEX IF NOT EXISTS idx_pcr_snapshot_tag ON strike_pcr_snapshots(snapshot_tag);
CREATE INDEX IF NOT EXISTS idx_pcr_created_at ON strike_pcr_snapshots(created_at DESC);

-- Enable RLS
ALTER TABLE option_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE strike_pcr_snapshots ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Allow public read access" ON option_snapshots FOR SELECT USING (true);
CREATE POLICY "Allow public read access" ON strike_pcr_snapshots FOR SELECT USING (true);
CREATE POLICY "Allow insert access" ON option_snapshots FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert access" ON strike_pcr_snapshots FOR INSERT WITH CHECK (true);
"""

# -----------------------
#  MAIN APP FLOW
# -----------------------
st.title("📊 NIFTY Option Screener v4.0 — DhanHQ + Supabase")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup & Configuration")
    
    st.subheader("Supabase Database Setup")
    if st.button("Check Database Connection"):
        if create_tables_if_not_exist():
            st.success("✅ Database connection successful!")
        else:
            st.warning("⚠️ Some tables may be missing")
    
    if st.button("Show SQL for Table Creation"):
        sql_commands = get_sql_for_tables()
        st.code(sql_commands, language="sql")
        st.download_button(
            label="Download SQL Script",
            data=sql_commands,
            file_name="supabase_setup.sql",
            mime="text/sql"
        )
    
    st.markdown("---")
    st.subheader("PCR Settings")
    save_interval = st.number_input("PCR Auto-save interval (seconds)", 
                                    value=SAVE_INTERVAL_SEC, 
                                    min_value=60, 
                                    step=60)
    
    st.markdown("---")
    st.subheader("App Controls")
    if st.button("Clear Caches"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("Reset Session State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
st.markdown("""
<div style='background-color: #1a1f2e; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #00d4aa;'>
<h3 style='color: #00d4aa;'>📋 About This App</h3>
<p style='color: #fafafa;'>This Nifty Option Screener v4.0 combines real-time option chain data from DhanHQ with persistent storage in Supabase. 
It provides advanced analytics including PCR analysis, trend detection, support/resistance levels, and fake breakout alerts.</p>
<p style='color: #9ba4b5;'><strong>Features:</strong> ATM analysis, Greeks calculation, GEX tracking, Max Pain, PCR trends, and more.</p>
</div>
""", unsafe_allow_html=True)

# Fetch spot and expiry
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot. Check Dhan credentials.")
        st.stop()
    
    expiries = get_expiry_list()
    if not expiries:
        st.error("Unable to fetch expiry list")
        st.stop()
    
    expiry = st.selectbox("Select expiry", expiries, index=0)

with col2:
    if spot > 0:
        st.metric("NIFTY Spot", f"{spot:.2f}")
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

# Determine strike gap and ATM
strike_gap = strike_gap_from_series(df_ce["strikePrice"])
atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))
lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

df_ce = df_ce[(df_ce["strikePrice"]>=lower) & (df_ce["strikePrice"]<=upper)].reset_index(drop=True)
df_pe = df_pe[(df_pe["strikePrice"]>=lower) & (df_pe["strikePrice"]<=upper)].reset_index(drop=True)

merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# Session storage
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# Compute tau
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
except Exception:
    tau = 7.0/365.0

# Compute per-strike metrics
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

# Aggregations
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

# UI: Core Metrics
st.markdown("## 📈 Core Market Metrics")
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.metric("Spot", f"{spot:.2f}")
    st.metric("ATM", f"{atm_strike}")
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

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Strike Details", "🔥 OI Heatmap", "🧠 Max Pain & PCR"])

with tab1:
    st.markdown("### 🧾 Strike Table (ATM ± 8)")
    display_cols = ["strikePrice","OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","CE_Price_Delta","CE_IV_Delta","CE_Winding","CE_Divergence","Delta_CE","Gamma_CE","GEX_CE",
                    "OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","PE_Price_Delta","PE_IV_Delta","PE_Winding","PE_Divergence","Delta_PE","Gamma_PE","GEX_PE",
                    "Strength_Score","Gamma_Pressure","Interpretation"]
    for c in display_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    display_df = merged[display_cols].copy()
    st.dataframe(display_df, use_container_width=True)

with tab2:
    st.markdown("### 🔥 Heatmap ΔOI")
    chg_oi_heatmap = merged[["strikePrice","Chg_OI_CE","Chg_OI_PE"]].set_index("strikePrice")
    def color_chg(val):
        try:
            if val>0: return "background-color:#d4f4dd"
            if val<0: return "background-color:#f8d7da"
            return ""
        except:
            return ""
    st.dataframe(chg_oi_heatmap.style.applymap(color_chg), use_container_width=True)

with tab3:
    st.markdown("### 🧠 Max Pain & ATM Shift")
    col1,col2 = st.columns([1,2])
    with col1:
        st.metric("Approx Max Pain", f"{max_pain if max_pain else 'N/A'}")
        st.metric("Breakout Index", f"{breakout_index}%")
    with col2:
        st.info(f"ATM Shift: {atm_shift_str}")
        st.info(f"PCR Trend Analysis available below")

# PCR Analysis Section
st.markdown("---")
st.header("📊 PCR Analysis & Trend Detection")

pcr_df = compute_pcr_df(merged)
pcr_display_cols = ["strikePrice", "OI_CE", "OI_PE", "PCR", "Chg_OI_CE", "Chg_OI_PE", "LTP_CE", "LTP_PE"]
st.markdown("### Current Strike PCR")
st.dataframe(pcr_df[pcr_display_cols].sort_values("strikePrice").reset_index(drop=True), use_container_width=True)

# PCR Snapshot Management
st.markdown("### PCR Snapshots (Batch-based)")

colA, colB = st.columns([1,1])
with colA:
    if st.button("💾 Save PCR Snapshot (Manual)"):
        ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
        if ok:
            st.success(f"Saved PCR snapshot tag: {tag}")
            st.session_state["last_pcr_saved_tag"] = tag
        else:
            st.error(f"Save failed: {msg}")

# Auto-save logic
last_auto_saved = st.session_state.get("last_pcr_auto_saved", 0)
now_ts = time.time()
if now_ts - last_auto_saved > save_interval:
    ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
    if ok:
        st.success(f"Auto-saved PCR snapshot: {tag}")
        st.session_state["last_pcr_saved_tag"] = tag
        st.session_state["last_pcr_auto_saved"] = now_ts
    else:
        st.warning(f"Auto-save failed: {msg}")
        st.session_state["last_pcr_auto_saved"] = now_ts

# Trend Analysis
tags = get_last_two_snapshot_tags()
current_tag = tags[0] if tags else None
previous_tag = tags[1] if len(tags) > 1 else None

if current_tag:
    cur_df = fetch_pcr_snapshot_by_tag(current_tag)
else:
    cur_df = pd.DataFrame()

prev_df = fetch_pcr_snapshot_by_tag(previous_tag) if previous_tag else pd.DataFrame()

if not cur_df.empty:
    trend_df = evaluate_trend(cur_df, prev_df)
    
    if not trend_df.empty:
        trend_display = trend_df.copy()
        trend_display = trend_display.rename(columns={
            "OI_CE_now": "OI_CE", "OI_PE_now": "OI_PE", "PCR_now": "PCR"
        })
        
        st.markdown("### 📈 Trend Analysis (PCR + OI Deltas)")
        show_cols = ["strikePrice", "OI_CE", "OI_PE", "ΔOI_CE", "ΔOI_PE", "PCR", "PCR_prev", "ΔPCR", "Trend"]
        for c in show_cols:
            if c not in trend_display.columns:
                trend_display[c] = np.nan
        
        st.dataframe(trend_display[show_cols].sort_values("strikePrice").reset_index(drop=True), use_container_width=True)
        
        # Rank Supports/Resistances
        ranked_df, top_supports, top_resists = rank_support_resistance(trend_display)
        
        st.markdown("### 🎯 Top Support/Resistance Levels")
        col_s, col_r = st.columns(2)
        with col_s:
            st.subheader("Supports")
            for i, sup in enumerate(top_supports[:3], 1):
                st.metric(f"Support #{i}", f"{sup}")
        with col_r:
            st.subheader("Resistances")
            for i, res in enumerate(top_resists[:3], 1):
                st.metric(f"Resistance #{i}", f"{res}")
        
        # Fake Breakout Detection
        fake_type, fake_hint = detect_fake_breakout(spot, 
                                                    top_supports[0] if top_supports else None, 
                                                    top_resists[0] if top_resists else None, 
                                                    trend_display)
        
        if fake_type:
            st.warning(f"⚠️ **Fake Breakout Alert: {fake_type}**")
            st.info(f"{fake_hint}")
        else:
            st.success("✅ No immediate fake breakout detected")
        
        # Stop-loss Hint
        stop_loss_hint = generate_stop_loss_hint(spot, top_supports, top_resists, fake_type)
        st.markdown("### 🛡️ Stop-loss Hint")
        st.info(stop_loss_hint)
    else:
        st.info("No trend data available. Save more PCR snapshots to enable trend analysis.")
else:
    st.info("No PCR snapshots found. Use the 'Save PCR Snapshot' button to start tracking trends.")

# PCR Snapshot Comparison
st.markdown("---")
st.header("🔍 Compare PCR Snapshots")

try:
    resp = supabase.table(SUPABASE_TABLE_PCR).select("snapshot_tag, created_at").order("created_at", desc=True).limit(2000).execute()
    tags_list = []
    if resp.status_code in (200,201):
        for r in (resp.data or []):
            tag = r.get("snapshot_tag")
            if tag and tag not in tags_list:
                tags_list.append(tag)
    tags_list = tags_list[:50]
except Exception:
    tags_list = []

if tags_list:
    col_left, col_right = st.columns(2)
    with col_left:
        left_tag = st.selectbox("Left snapshot tag", options=tags_list, index=0 if tags_list else None)
    with col_right:
        right_tag = st.selectbox("Right snapshot tag", options=tags_list, index=1 if len(tags_list)>1 else 0)
    
    if st.button("Compare PCR Snapshots"):
        if not left_tag or not right_tag or left_tag == right_tag:
            st.warning("Choose two different snapshot tags.")
        else:
            left_df = fetch_pcr_snapshot_by_tag(left_tag)
            right_df = fetch_pcr_snapshot_by_tag(right_tag)
            comp_df = evaluate_trend(right_df, left_df)
            
            if not comp_df.empty:
                st.markdown(f"### Comparison: {left_tag} → {right_tag}")
                comp_display = comp_df[["strikePrice", "OI_CE_now", "OI_PE_now", "ΔOI_CE", "ΔOI_PE", "PCR_now", "PCR_prev", "ΔPCR", "Trend"]]
                st.dataframe(comp_display, use_container_width=True)
                
                total_ce_delta = comp_df["ΔOI_CE"].sum()
                total_pe_delta = comp_df["ΔOI_PE"].sum()
                avg_pcr_delta = comp_df["ΔPCR"].mean()
                
                st.markdown("**Summary:**")
                st.write(f"- Total CE OI Δ: {int(total_ce_delta):,}")
                st.write(f"- Total PE OI Δ: {int(total_pe_delta):,}")
                st.write(f"- Average PCR Δ: {avg_pcr_delta:.3f}")
            else:
                st.warning("Comparison returned no data.")
else:
    st.info("No PCR snapshots available for comparison.")

# Time Window Snapshots
st.markdown("---")
st.header("📦 Time Window Snapshots")

def now_in_window(w_key):
    now = datetime.now()
    s_h,s_m = TIME_WINDOWS[w_key]["start"]
    e_h,e_m = TIME_WINDOWS[w_key]["end"]
    start = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    return start <= now <= end

today = datetime.now().strftime("%Y-%m-%d")
for w in TIME_WINDOWS.keys():
    try:
        if now_in_window(w) and not supabase_snapshot_exists(today, w):
            ok, msg = save_snapshot_to_supabase(merged, w, spot)
            if ok:
                st.info(f"Auto-saved snapshot for {TIME_WINDOWS[w]['label']} ({today})")
            else:
                st.warning(f"Auto-save {w} failed: {msg}")
    except Exception as e:
        st.warning(f"Auto-save check error for {w}: {e}")

# Manual capture buttons
st.markdown("### Manual Snapshot Capture")
c1,c2,c3,c4 = st.columns(4)
with c1:
    if st.button("📥 Morning Snapshot"):
        ok,msg = save_snapshot_to_supabase(merged, "morning", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c2:
    if st.button("📥 Mid Snapshot"):
        ok,msg = save_snapshot_to_supabase(merged, "mid", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c3:
    if st.button("📥 Afternoon Snapshot"):
        ok,msg = save_snapshot_to_supabase(merged, "afternoon", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c4:
    if st.button("📥 Evening Snapshot"):
        ok,msg = save_snapshot_to_supabase(merged, "evening", spot)
        st.success("Saved." if ok else f"Failed: {msg}")

# Snapshot compare UI
st.markdown("### 🔎 Compare Time Window Snapshots")
saved_files = []
for d_offset in range(0,7):
    dt = (datetime.now() - timedelta(days=d_offset)).strftime("%Y-%m-%d")
    for w in TIME_WINDOWS.keys():
        df_tmp = fetch_snapshot_from_supabase(dt, w)
        if df_tmp is not None and not df_tmp.empty:
            saved_files.append(f"{dt}__{w}")

if not saved_files:
    st.info("No saved time-window snapshots found.")

if saved_files:
    colL, colR = st.columns(2)
    with colL:
        left_choice = st.selectbox("Left snapshot (date__window)", options=saved_files, index=0 if saved_files else None)
    with colR:
        right_choice = st.selectbox("Right snapshot (date__window)", options=saved_files, index=1 if len(saved_files)>1 else 0)

    def load_time_window_choice(choice_str):
        if not choice_str:
            return None
        date_str, w = choice_str.split("__")
        return fetch_snapshot_from_supabase(date_str, w)

    if left_choice and right_choice:
        left_df = load_time_window_choice(left_choice)
        right_df = load_time_window_choice(right_choice)
        diff_df = compare_snapshots_df(left_df, right_df)
        st.subheader(f"Snapshot Diff: {left_choice} → {right_choice}")
        if diff_df is not None:
            st.dataframe(diff_df.drop(columns=["abs_change"]).head(100), use_container_width=True)
            s_left = snapshot_summary(left_df)
            s_right = snapshot_summary(right_df)
            b_left = bias_score_from_summary(s_left)
            b_right = bias_score_from_summary(s_right)
            st.markdown("**Summary:**")
            st.write(f"- Left ({left_choice}): CE_ΔOI={s_left['CE_ΔOI']}, PE_ΔOI={s_left['PE_ΔOI']}")
            st.write(f"- Right ({right_choice}): CE_ΔOI={s_right['CE_ΔOI']}, PE_ΔOI={s_right['PE_ΔOI']}")
            st.write(f"- Bias Score Δ = {(b_right - b_left):.3f}")

# Quick comparison
if st.button("Quick: Morning → Afternoon (today)"):
    left = fetch_snapshot_from_supabase(today, "morning")
    right = fetch_snapshot_from_supabase(today, "afternoon")
    if left is None or left.empty or right is None or right.empty:
        st.warning("Missing snapshots for Morning or Afternoon today.")
    else:
        d = compare_snapshots_df(left, right)
        st.dataframe(d.drop(columns=["abs_change"]).head(200), use_container_width=True)
        st.write("Bias Δ:", bias_score_from_summary(snapshot_summary(right)) - bias_score_from_summary(snapshot_summary(left)))

# Footer
st.markdown("---")
st.markdown("""
<div style='background-color: #1a1f2e; padding: 20px; border-radius: 10px; border: 1px solid #00d4aa;'>
<h3 style='color: #00d4aa;'>📚 Credits & Information</h3>
<p style='color: #fafafa;'><strong>App:</strong> Nifty Option Screener v4.0</p>
<p style='color: #9ba4b5;'><strong>Data Source:</strong> DhanHQ API</p>
<p style='color: #9ba4b5;'><strong>Database:</strong> Supabase PostgreSQL</p>
<p style='color: #9ba4b5;'><strong>Disclaimer:</strong> This tool is for educational purposes only. Trading involves risk.</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"""
**App Status:** Auto-refresh every {AUTO_REFRESH_SEC} seconds | PCR auto-save: {save_interval} seconds
**Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
