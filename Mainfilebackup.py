# nifty_option_screener_v4_complete_fixed.py
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
 - Persistent snapshots stored to Supabase
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
SAVE_INTERVAL_SEC = 300

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

# -----------------------
#  Custom CSS
# -----------------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stMetricLabel"] {
        color: #9ba4b5 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-size: 1.8rem !important;
    }
    h1, h2, h3 {
        color: #00d4aa !important;
    }
    .stAlert > div {
        background-color: #1a1f2e !important;
        border: 1px solid #00d4aa !important;
        color: #fafafa !important;
    }
    .stSuccess > div {
        background-color: #1a2e1a !important;
        border: 1px solid #00d4aa !important;
    }
    .stWarning > div {
        background-color: #2e2a1a !important;
        border: 1px solid #ffa500 !important;
    }
    .stError > div {
        background-color: #2e1a1a !important;
        border: 1px solid #ff4b4b !important;
    }
    .stButton > button {
        background-color: #00d4aa !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #00ffcc !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
#  CONFIG
# -----------------------
st.set_page_config(page_title="Nifty Option Screener v4", layout="wide")

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
        return "Fresh Build"
    if oi_up and vol_up and not price_up:
        return "Seller Aggressive"
    if not oi_up and vol_up and price_up:
        return "Long Covering"
    if not oi_up and vol_up and not price_up:
        return "Put Covering"
    if oi_up and not vol_up:
        return "Weak Build"
    if (not oi_up) and not vol_up:
        return "Weak Unwind"
    return "Neutral"

def interpret_itm_otm(strike, atm, chg_oi_ce, chg_oi_pe):
    if strike < atm:
        ce = "Bullish (ITM CE Unwind)" if chg_oi_ce < 0 else "Bearish (ITM CE Build)" if chg_oi_ce > 0 else "NoSign"
    elif strike > atm:
        ce = "Resistance Forming" if chg_oi_ce > 0 else "Resistance Weakening" if chg_oi_ce < 0 else "NoSign"
    else:
        ce = "ATM CE"

    if strike > atm:
        pe = "Bullish (ITM PE Unwind)" if chg_oi_pe < 0 else "Bearish (ITM PE Build)" if chg_oi_pe > 0 else "NoSign"
    elif strike < atm:
        pe = "Support Forming" if chg_oi_pe > 0 else "Support Weakening" if chg_oi_pe < 0 else "NoSign"
    else:
        pe = "ATM PE"

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
    return datetime.utcnow().isoformat(timespec="seconds")

def save_pcr_snapshot_to_supabase(df_for_save, expiry, spot):
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
        pcr_up = row["ΔPCR"] > 0.05
        pcr_down = row["ΔPCR"] < -0.05
        
        if oi_pe_up and pcr_up:
            return "Support Building"
        if oi_pe_down and pcr_down:
            return "Support Breaking"
        if oi_ce_up and pcr_down:
            return "Resistance Building"
        if abs(row["ΔPCR"]) > 0.2:
            return "PCR Rapid Change"
        return "Neutral"
    
    deltas["Trend"] = deltas.apply(label, axis=1)
    deltas = deltas.reset_index().rename(columns={"index":"strikePrice"})
    return deltas

def rank_support_resistance(trend_df):
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
    if fake_type == "Bull Trap":
        return f"Keep SL near support {top_supports[0] if top_supports else 'N/A'}"
    if fake_type == "Bear Trap":
        return f"Keep SL near resistance {top_resists[0] if top_resists else 'N/A'}"
    if top_resists and spot > top_resists[0]:
        return f"Real upside: SL near {top_supports[1] if len(top_supports)>1 else 'N/A'}"
    if top_supports and spot < top_supports[0]:
        return f"Real downside: SL near {top_resists[1] if len(top_resists)>1 else 'N/A'}"
    return f"No clear breakout"

# -----------------------
#  Dhan API - FIXED
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_spot_price():
    """Fetch NIFTY spot from Dhan Market Quote API"""
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        
        # CRITICAL: Payload must be {"IDX_I": [array of security IDs as integers]}
        payload = {
            "IDX_I": [13]  # NIFTY 50 security ID is 13 (integer, not string)
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,  # lowercase with hyphen
            "client-id": DHAN_CLIENT_ID  # lowercase with hyphen
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "success":
            # Navigate: data.IDX_I.13.last_price
            idx_data = data.get("data", {}).get("IDX_I", {})
            nifty_data = idx_data.get("13", {})
            ltp = nifty_data.get("last_price", 0.0)
            return float(ltp)
        
        return 0.0
        
    except requests.exceptions.HTTPError as e:
        st.warning(f"Dhan LTP HTTP Error: {e} - Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
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
            "client-id": DHAN_CLIENT_ID
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
            "client-id": DHAN_CLIENT_ID
        }
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status")=="success":
            return data.get("data",{})
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
#  Supabase helpers
# -----------------------
def supabase_snapshot_exists(date_str, window):
    try:
        resp = supabase.table(SUPABASE_TABLE).select("id").eq("date", date_str).eq("time_window", window).limit(1).execute()
        if resp.status_code == 200:
            return len(resp.data) > 0
        return False
    except:
        return False

def save_snapshot_to_supabase(df, window, underlying):
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
                return False, f"Insert failed {res.status_code}"
        return True, "saved"
    except Exception as e:
        return False, str(e)

def fetch_snapshot_from_supabase(date_str, window):
    try:
        resp = supabase.table(SUPABASE_TABLE).select("*").eq("date", date_str).eq("time_window", window).execute()
        if resp.status_code == 200:
            rows = resp.data
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
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
        return pd.DataFrame()
    except:
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
        return {"CE_OI":0,"PE_OI":0,"CE_ΔOI":0,"PE_ΔOI":0}
    return {
        "CE_OI": int(df["OI_CE"].sum()),
        "PE_OI": int(df["OI_PE"].sum()),
        "CE_ΔOI": int(df["Chg_OI_CE"].sum()),
        "PE_ΔOI": int(df["Chg_OI_PE"].sum())
    }

def bias_score_from_summary(summary):
    score = 0.0
    score += -summary["CE_ΔOI"] / 10000.0
    score += summary["PE_ΔOI"] / 10000.0
    return score

def get_sql_for_tables():
    return """
-- SUPABASE SETUP
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

ALTER TABLE option_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE strike_pcr_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read" ON option_snapshots FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON strike_pcr_snapshots FOR SELECT USING (true);
CREATE POLICY "Allow insert" ON option_snapshots FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert" ON strike_pcr_snapshots FOR INSERT WITH CHECK (true);
"""

# -----------------------
#  MAIN APP
# -----------------------
st.title("📊 NIFTY Option Screener v4.0 — DhanHQ + Supabase")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("Show SQL Setup"):
        st.code(get_sql_for_tables(), language="sql")
    
    st.markdown("---")
    save_interval = st.number_input("PCR Auto-save (sec)", value=SAVE_INTERVAL_SEC, min_value=60, step=60)
    
    if st.button("Clear Caches"):
        st.cache_data.clear()
        st.rerun()

st.markdown("""
<div style='background-color: #1a1f2e; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #00d4aa;'>
<h3 style='color: #00d4aa;'>📋 About</h3>
<p style='color: #fafafa;'>Real-time NIFTY option chain analysis with PCR trends, support/resistance detection, and fake breakout alerts.</p>
</div>
""", unsafe_allow_html=True)

# Fetch data
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot. Check credentials or token validity.")
        st.info("Note: Dhan access tokens expire after 24 hours. Generate a new token from web.dhan.co if needed.")
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

# Session state
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# Compute tau
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    tau = max((expiry_dt - datetime.now()).total_seconds() / (365.25*24*3600), 1/365.25)
except:
    tau = 7.0/365.0

# Compute metrics
for i, row in merged.iterrows():
    strike = int(row["strikePrice"])
    
    ltp_ce = safe_float(row.get("LTP_CE",0.0))
    ltp_pe = safe_float(row.get("LTP_PE",0.0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))

    key_ce = f"{expiry}_{strike}_CE"
    key_pe = f"{expiry}_{strike}_PE"
    prev_ce = st.session_state["prev_ltps_v3"].get(key_ce)
    prev_pe = st.session_state["prev_ltps_v3"].get(key_pe)
    prev_iv_ce = st.session_state["prev_ivs_v3"].get(key_ce)
    prev_iv_pe = st.session_state["prev_ivs_v3"].get(key_pe)

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
    except:
        delta_ce = gamma_ce = vega_ce = theta_ce = 0.0

    try:
        delta_pe = bs_delta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        vega_pe = bs_vega(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        theta_pe = bs_theta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
    except:
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

merged["CE_Delta_Exposure"] = merged["Delta_CE"].fillna(0) * merged["OI_CE"].fillna(0) * LOT_SIZE
merged["PE_Delta_Exposure"] = merged["Delta_PE"].fillna(0) * merged["OI_PE"].fillna(0) * LOT_SIZE
net_delta_exposure = merged["CE_Delta_Exposure"].sum() + merged["PE_Delta_Exposure"].sum()

total_gex_ce = merged["GEX_CE"].sum()
total_gex_pe = merged["GEX_PE"].sum()
breakout_index = breakout_probability_index(merged, atm_strike, strike_gap)

# Market bias
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

market_bias = "Strong Bullish" if polarity > 5 else "Bullish" if polarity > 1 else "Strong Bearish" if polarity < -5 else "Bearish" if polarity < -1 else "Neutral"

# UI Display
st.markdown("## 📈 Core Metrics")
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.metric("Spot", f"₹{spot:.2f}")
    st.metric("ATM", f"₹{atm_strike}")
with col2:
    st.metric("CE OI", f"{int(total_CE_OI):,}")
    st.metric("PE OI", f"{int(total_PE_OI):,}")
with col3:
    st.metric("CE ΔOI", f"{int(total_CE_chg):,}")
    st.metric("PE ΔOI", f"{int(total_PE_chg):,}")
with col4:
    st.metric("Delta Exp", f"{int(net_delta_exposure):,}")
    st.metric("Bias", market_bias)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Strike Details", "🔥 Heatmap", "📈 PCR Analysis"])

with tab1:
    st.markdown("### Strike Table")
    display_cols = ["strikePrice","OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","CE_Winding","Delta_CE","Gamma_CE","GEX_CE",
                    "OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","PE_Winding","Delta_PE","Gamma_PE","GEX_PE"]
    for c in display_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    st.dataframe(merged[display_cols], use_container_width=True)

with tab2:
    st.markdown("### OI Change Heatmap")
    heatmap = merged[["strikePrice","Chg_OI_CE","Chg_OI_PE"]].set_index("strikePrice")
    def color_chg(val):
        if val>0: return "background-color:#d4f4dd"
        if val<0: return "background-color:#f8d7da"
        return ""
    st.dataframe(heatmap.style.applymap(color_chg), use_container_width=True)

with tab3:
    st.markdown("### PCR Analysis")
    pcr_df = compute_pcr_df(merged)
    st.dataframe(pcr_df[["strikePrice","OI_CE","OI_PE","PCR","Chg_OI_CE","Chg_OI_PE"]], use_container_width=True)
    
    if st.button("💾 Save PCR Snapshot"):
        ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
        if ok:
            st.success(f"Saved: {tag}")
        else:
            st.error(f"Failed: {msg}")
    
    # Auto-save logic
    last_saved = st.session_state.get("last_pcr_auto_saved", 0)
    if time.time() - last_saved > save_interval:
        ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
        if ok:
            st.info(f"Auto-saved: {tag}")
            st.session_state["last_pcr_auto_saved"] = time.time()
    
    # Trend analysis
    tags = get_last_two_snapshot_tags()
    if len(tags) >= 2:
        cur_df = fetch_pcr_snapshot_by_tag(tags[0])
        prev_df = fetch_pcr_snapshot_by_tag(tags[1])
        
        if not cur_df.empty:
            trend_df = evaluate_trend(cur_df, prev_df)
            
            if not trend_df.empty:
                st.markdown("### Trend Analysis")
                trend_display = trend_df.rename(columns={"OI_CE_now":"OI_CE","OI_PE_now":"OI_PE","PCR_now":"PCR"})
                st.dataframe(trend_display[["strikePrice","OI_CE","OI_PE","ΔOI_CE","ΔOI_PE","PCR","ΔPCR","Trend"]], use_container_width=True)
                
                # Support/Resistance
                ranked, supports, resists = rank_support_resistance(trend_display)
                
                col_s, col_r = st.columns(2)
                with col_s:
                    st.subheader("Top Supports")
                    for i, sup in enumerate(supports[:3], 1):
                        st.metric(f"#{i}", f"₹{sup}")
                with col_r:
                    st.subheader("Top Resistances")
                    for i, res in enumerate(resists[:3], 1):
                        st.metric(f"#{i}", f"₹{res}")
                
                # Fake breakout
                fake, hint = detect_fake_breakout(spot, supports[0] if supports else None, resists[0] if resists else None, trend_display)
                if fake:
                    st.warning(f"⚠️ {fake}: {hint}")
                else:
                    st.success("✅ No fake breakout detected")
                
                # Stop-loss hint
                sl_hint = generate_stop_loss_hint(spot, supports, resists, fake)
                st.info(f"🛡️ SL Hint: {sl_hint}")

# Footer
st.markdown("---")
st.caption(f"Auto-refresh: {AUTO_REFRESH_SEC}s | Last update: {datetime.now().strftime('%H:%M:%S')}")
