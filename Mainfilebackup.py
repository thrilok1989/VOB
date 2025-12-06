# nifty_option_screener_v4_immediate_levels.py
"""
Nifty Option Screener v4.0 — IMMEDIATE SUPPORT/RESISTANCE
Shows levels INSTANTLY from current PCR, trends from snapshots
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
    st.error("❌ Missing credentials in secrets")
    st.stop()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"❌ Supabase failed: {e}")
    st.stop()

DHAN_BASE_URL = "https://api.dhan.co"

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
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Nifty Option Screener v4 - Immediate Levels", layout="wide")

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

def now_in_window(w_key):
    now = get_ist_now()
    s_h, s_m = TIME_WINDOWS[w_key]["start"]
    e_h, e_m = TIME_WINDOWS[w_key]["end"]
    start = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    return start <= now <= end

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
    """
    IMMEDIATE support/resistance from CURRENT PCR
    No snapshots required!
    """
    eps = 1e-6
    t = pcr_df.copy()
    
    # Handle infinity PCR values
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Support score = High PE OI + High PCR
    t["support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    
    # Resistance score = High CE OI + Low PCR (inverse)
    t["resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["resistance_score"] = t["OI_CE"] + (t["resistance_factor"] * 100000.0)
    
    # Get top 3
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
        return f"Keep SL near support {top_supports[0] if len(top_supports)>0 else 'N/A'}"
    if fake_type == "Bear Trap":
        return f"Keep SL near resistance {top_resists[0] if len(top_resists)>0 else 'N/A'}"
    if len(top_resists) > 0 and spot > top_resists[0]:
        return f"Real upside: SL near {top_supports[1] if len(top_supports)>1 else top_supports[0] if len(top_supports)>0 else 'N/A'}"
    if len(top_supports) > 0 and spot < top_supports[0]:
        return f"Real downside: SL near {top_resists[1] if len(top_resists)>1 else top_resists[0] if len(top_resists)>0 else 'N/A'}"
    return f"No clear breakout"

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
#  MAIN APP
# -----------------------
st.title("🎯 NIFTY Option Screener v4.0 — Immediate Levels (IST)")

current_ist = get_ist_datetime_str()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <span class='ist-time'>🕐 IST: {current_ist}</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"### 🕐 Current IST")
    st.markdown(f"**{get_ist_time_str()}**")
    st.markdown(f"**{get_ist_date_str()}**")
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

# Session state
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# Compute tau
try:
    expiry_dt_ist = IST.localize(datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30))
    now_ist = get_ist_now()
    tau = max((expiry_dt_ist - now_ist).total_seconds() / (365.25*24*3600), 1/365.25)
except:
    tau = 7.0/365.0

# Compute metrics (shortened version - same as before)
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

    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)

    st.session_state["prev_ltps_v3"][key_ce] = ltp_ce
    st.session_state["prev_ltps_v3"][key_pe] = ltp_pe

    chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))

    merged.at[i,"CE_Winding"] = "Winding" if chg_oi_ce>0 else ("Unwinding" if chg_oi_ce<0 else "NoChange")
    merged.at[i,"PE_Winding"] = "Winding" if chg_oi_pe>0 else ("Unwinding" if chg_oi_pe<0 else "NoChange")
    merged.at[i,"CE_Divergence"] = price_oi_divergence_label(chg_oi_ce, safe_int(row.get("Vol_CE",0)), ce_price_delta)
    merged.at[i,"PE_Divergence"] = price_oi_divergence_label(chg_oi_pe, safe_int(row.get("Vol_PE",0)), pe_price_delta)

# Aggregations
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

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

# ========================================
# 🔥 CRITICAL: COMPUTE PCR & IMMEDIATE LEVELS
# ========================================
pcr_df = compute_pcr_df(merged)

# 🔥 GET IMMEDIATE SUPPORT/RESISTANCE (No snapshots needed!)
ranked_current, supports_df, resists_df = rank_support_resistance_current(pcr_df)

# Extract strike prices
immediate_supports = supports_df["strikePrice"].astype(int).tolist()
immediate_resists = resists_df["strikePrice"].astype(int).tolist()

# Auto-save PCR
last_saved = st.session_state.get("last_pcr_auto_saved", 0)
if time.time() - last_saved > save_interval:
    ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
    if ok:
        st.session_state["last_pcr_auto_saved"] = time.time()

# Get trend analysis (from snapshots if available)
tags = get_last_two_snapshot_tags()
trend_df = pd.DataFrame()
fake_type = None
fake_hint = ""
sl_hint = ""

if len(tags) >= 2:
    cur_df = fetch_pcr_snapshot_by_tag(tags[0])
    prev_df = fetch_pcr_snapshot_by_tag(tags[1])
    
    if not cur_df.empty:
        trend_df = evaluate_trend(cur_df, prev_df)
        if not trend_df.empty:
            fake_type, fake_hint = detect_fake_breakout(
                spot, 
                immediate_supports[0] if immediate_supports else None, 
                immediate_resists[0] if immediate_resists else None, 
                trend_df
            )
            sl_hint = generate_stop_loss_hint(spot, immediate_supports, immediate_resists, fake_type)

# ============================================
# 🎯 MAIN DASHBOARD
# ============================================

st.markdown("## 🎯 INSTITUTIONAL-GRADE ANALYSIS DASHBOARD")

# Row 1: Core Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("NIFTY Spot", f"₹{spot:.2f}")
with col2:
    st.metric("ATM Strike", f"₹{atm_strike}")
with col3:
    st.metric("Market Bias", market_bias)
with col4:
    breakout_index = breakout_probability_index(merged, atm_strike, strike_gap)
    st.metric("Breakout Index", f"{breakout_index}%")

st.markdown("---")

# Row 2: TRAP DETECTION
st.markdown("### 🚨 TRAP & BREAKOUT DETECTION")

if fake_type:
    if fake_type == "Bull Trap":
        st.markdown(f"""
        <div class="alert-box bull-trap">
            <h3>⚠️ BULL TRAP DETECTED!</h3>
            <p>{fake_hint}</p>
            <p><strong>Action:</strong> {sl_hint}</p>
            <p style='font-size:0.9rem; margin-top:10px;'>⏰ {get_ist_time_str()} IST</p>
        </div>
        """, unsafe_allow_html=True)
    elif fake_type == "Bear Trap":
        st.markdown(f"""
        <div class="alert-box bear-trap">
            <h3>⚠️ BEAR TRAP DETECTED!</h3>
            <p>{fake_hint}</p>
            <p><strong>Action:</strong> {sl_hint}</p>
            <p style='font-size:0.9rem; margin-top:10px;'>⏰ {get_ist_time_str()} IST</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-box no-trap">
        <h3>✅ NO TRAP DETECTED</h3>
        <p>Market moving genuinely. SL Hint: {sl_hint if sl_hint else "Monitor key levels"}</p>
        <p style='font-size:0.9rem; margin-top:10px;'>⏰ {get_ist_time_str()} IST</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Row 3: 🔥 IMMEDIATE SUPPORT & RESISTANCE
st.markdown("### 🎯 KEY SUPPORT & RESISTANCE LEVELS (LIVE PCR)")

col_s, col_r = st.columns(2)

with col_s:
    st.markdown("#### 🛡️ STRONGEST SUPPORTS")
    
    for i, (idx, row) in enumerate(supports_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_pe = int(row["OI_PE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        
        # Get trend if available
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
                PE OI: {oi_pe:,} | PCR: {pcr_display} | ΔOI: {chg_oi_pe:+,}
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
        
        # Get trend if available
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
                CE OI: {oi_ce:,} | PCR: {pcr_display} | ΔOI: {chg_oi_ce:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Row 4: Trend Signals (if snapshots available)
st.markdown("### 📊 TREND SIGNALS (Snapshot-based)")

if not trend_df.empty:
    trend_counts = trend_df["Trend"].value_counts()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        support_build = trend_counts.get("Support Building", 0)
        st.markdown(f"""
        <div class="alert-box support-building">
            <h4>Support Building</h4>
            <p style="font-size:2rem; margin:0;">{support_build}</p>
            <p style="margin:0; font-size:0.8rem;">strikes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        support_break = trend_counts.get("Support Breaking", 0)
        st.markdown(f"""
        <div class="alert-box support-breaking">
            <h4>Support Breaking</h4>
            <p style="font-size:2rem; margin:0;">{support_break}</p>
            <p style="margin:0; font-size:0.8rem;">strikes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        resist_build = trend_counts.get("Resistance Building", 0)
        st.markdown(f"""
        <div class="alert-box resistance-building">
            <h4>Resistance Building</h4>
            <p style="font-size:2rem; margin:0;">{resist_build}</p>
            <p style="margin:0; font-size:0.8rem;">strikes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        resist_break = trend_counts.get("Resistance Breaking", 0)
        st.markdown(f"""
        <div class="alert-box resistance-breaking">
            <h4>Resistance Breaking</h4>
            <p style="font-size:2rem; margin:0;">{resist_break}</p>
            <p style="margin:0; font-size:0.8rem;">strikes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        pcr_rapid = trend_counts.get("PCR Rapid Change", 0)
        st.markdown(f"""
        <div class="alert-box pcr-rapid" style="background-color: #2e1a2e; border-left-color: #ff00ff; color: #ff66ff;">
            <h4>PCR Rapid</h4>
            <p style="font-size:2rem; margin:0;">{pcr_rapid}</p>
            <p style="margin:0; font-size:0.8rem;">GAMMA SHOCK</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("⏳ Collecting snapshots for trend analysis... Auto-saving every 5 minutes")

st.markdown("---")

# OI Metrics
st.markdown("### 📈 OPEN INTEREST METRICS")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total CE OI", f"{int(total_CE_OI):,}")
with col2:
    st.metric("Total PE OI", f"{int(total_PE_OI):,}")
with col3:
    st.metric("CE ΔOI", f"{int(total_CE_chg):,}")
with col4:
    st.metric("PE ΔOI", f"{int(total_PE_chg):,}")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Strike Details", "🔥 PCR Table", "🧠 Snapshot Manager"])

with tab1:
    st.markdown("### Strike-wise Data")
    display_cols = ["strikePrice","OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","CE_Winding",
                    "OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","PE_Winding"]
    st.dataframe(merged[display_cols], use_container_width=True)

with tab2:
    st.markdown("### PCR Analysis (Current)")
    st.dataframe(pcr_df[["strikePrice","OI_CE","OI_PE","Chg_OI_CE","Chg_OI_PE","PCR"]], use_container_width=True)

with tab3:
    st.markdown("### PCR Snapshot Management")
    st.markdown(f"**Current IST:** {get_ist_datetime_str()}")
    
    if st.button("💾 Save PCR Snapshot Manually"):
        ok, tag, msg = save_pcr_snapshot_to_supabase(pcr_df, expiry, spot)
        if ok:
            st.success(f"Saved at {get_ist_time_str()} IST")
        else:
            st.error(f"Failed: {msg}")
    
    if not trend_df.empty:
        st.markdown("#### Trend Details")
        trend_display = trend_df.rename(columns={"OI_CE_now":"OI_CE","OI_PE_now":"OI_PE","PCR_now":"PCR"})
        active = trend_display[trend_display["Trend"] != "Neutral"].head(10)
        if not active.empty:
            st.dataframe(active[["strikePrice","OI_CE","OI_PE","ΔOI_CE","ΔOI_PE","PCR","ΔPCR","Trend"]], use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | 💾 Auto-save: {save_interval}s | ⏰ {get_ist_datetime_str()}")
st.caption("🕐 All timestamps in IST | 🎯 Support/Resistance from LIVE PCR")
