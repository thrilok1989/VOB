# nifty_option_screener_v3_dhan_supabase.py
"""
Nifty Option Screener v3.0 — DhanHQ + Supabase snapshot engine
Features:
 - ATM ± 8 strikes
 - Winding / Unwinding (CE/PE)
 - IV change analysis + Greeks (Black-Scholes approx)
 - GEX / Delta exposure
 - Max Pain, Breakout Index, ATM-shift
 - Persistent snapshots stored to Supabase for windows:
    morning (09:15-10:30), mid (10:30-12:30), afternoon (14:00-15:30), evening (15:00-15:30)
 - Snapshot compare UI
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

# Time windows (your requested Afternoon 14:00-15:30)
TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30)"}
}

# -----------------------
#  SECRETS / CLIENTS
# -----------------------
# Expect user to set these in Streamlit secrets as described in the README at top
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except Exception as e:
    st.error("❌ Missing credentials in Streamlit secrets. See script header for required secrets.")
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
#  Helpers: auto-refresh
# -----------------------
st.set_page_config(page_title="Nifty Option Screener v3 (Dhan + Supabase)", layout="wide")
def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.experimental_rerun()
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

# -----------------------
#  Dhan API helpers
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_spot_price():
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        payload = {"IDX_I": ["13"]}
        headers = {
            "Accept":"application/json",
            "Content-Type":"application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
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
        headers = {"Accept":"application/json","Content-Type":"application/json","access-token":DHAN_ACCESS_TOKEN,"client-id":DHAN_CLIENT_ID}
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
        headers = {"Accept":"application/json","Content-Type":"application/json","access-token":DHAN_ACCESS_TOKEN,"client-id":DHAN_CLIENT_ID}
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
        resp = supabase.table(SUPABASE_TABLE).select("id").eq("date", date_str).eq("window", window).limit(1).execute()
        if resp.status_code == 200:
            data = resp.data
            return len(data) > 0
        return False
    except Exception as e:
        st.warning(f"Supabase check exists failed: {e}")
        return False

def save_snapshot_to_supabase(df, window, underlying):
    """Insert rows (one per strike) into supabase table with date & window."""
    if df is None or df.empty:
        return False, "no data"
    date_str = datetime.now().strftime("%Y-%m-%d")
    payload = []
    # limit to relevant cols
    for _, r in df.iterrows():
        payload.append({
            "date": date_str,
            "window": window,
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
        # Insert in batches to avoid huge single payload if many strikes
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
    """Return a pandas DataFrame for requested date/window or None"""
    try:
        resp = supabase.table(SUPABASE_TABLE).select("*").eq("date", date_str).eq("window", window).execute()
        if resp.status_code == 200:
            rows = resp.data
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            # normalize column names
            expected = ["strike","oi_ce","chg_oi_ce","vol_ce","ltp_ce","iv_ce","oi_pe","chg_oi_pe","vol_pe","ltp_pe","iv_pe","underlying"]
            for col in expected:
                if col not in df.columns:
                    df[col] = np.nan
            df = df.rename(columns={
                "strike":"strikePrice",
                "oi_ce":"OI_CE","chg_oi_ce":"Chg_OI_CE","vol_ce":"Vol_CE","ltp_ce":"LTP_CE","iv_ce":"IV_CE",
                "oi_pe":"OI_PE","chg_oi_pe":"Chg_OI_PE","vol_pe":"Vol_PE","ltp_pe":"LTP_PE","iv_pe":"IV_PE"
            })
            # cast numeric
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
    # add absolute change for sorting
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
#  MAIN APP FLOW
# -----------------------
st.title("📊 NIFTY Option Screener v3.0 — DhanHQ + Supabase Snapshots")

# Sidebar: token check
with st.sidebar:
    st.header("Credentials")
    st.write("Supabase table:", SUPABASE_TABLE)
    if st.button("Check Supabase Connection"):
        try:
            res = supabase.table(SUPABASE_TABLE).select("id").limit(1).execute()
            st.success("Supabase connection OK")
        except Exception as e:
            st.error(f"Supabase check failed: {e}")
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.experimental_rerun()

# Fetch spot and expiry list
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

# Filter window
df_ce = df_ce[(df_ce["strikePrice"]>=lower) & (df_ce["strikePrice"]<=upper)].reset_index(drop=True)
df_pe = df_pe[(df_pe["strikePrice"]>=lower) & (df_pe["strikePrice"]<=upper)].reset_index(drop=True)

merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# session storage for prev LTP/IV
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# compute tau using expiry date if available (expiry format YYYY-MM-DD assumed)
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
except Exception:
    tau = 7.0/365.0

# compute per-strike metrics
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

# -----------------------
#  UI: display core tables and metrics
# -----------------------
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

st.markdown("### 🧾 Strike Table (ATM ± 8)")
display_cols = ["strikePrice","OI_CE","Chg_OI_CE","Vol_CE","LTP_CE","CE_Price_Delta","CE_IV_Delta","CE_Winding","CE_Divergence","Delta_CE","Gamma_CE","GEX_CE",
                "OI_PE","Chg_OI_PE","Vol_PE","LTP_PE","PE_Price_Delta","PE_IV_Delta","PE_Winding","PE_Divergence","Delta_PE","Gamma_PE","GEX_PE",
                "Strength_Score","Gamma_Pressure","Interpretation"]
for c in display_cols:
    if c not in merged.columns:
        merged[c] = np.nan
display_df = merged[display_cols].copy()
st.dataframe(display_df, use_container_width=True)

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

st.markdown("### 🧠 Max Pain & ATM Shift")
col1,col2 = st.columns([1,2])
with col1:
    st.metric("Approx Max Pain", f"{max_pain if max_pain else 'N/A'}")
with col2:
    st.info(f"ATM Shift: {atm_shift_str}")

# -----------------------
#  SNAPSHOT: Supabase storage (auto + manual) + compare UI
# -----------------------
st.markdown("---")
st.header("📦 Snapshots (Supabase) — Morning / Mid / Afternoon / Evening")

def now_in_window(w_key):
    now = datetime.now()
    s_h,s_m = TIME_WINDOWS[w_key]["start"]
    e_h,e_m = TIME_WINDOWS[w_key]["end"]
    start = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    return start <= now <= end

today = datetime.now().strftime("%Y-%m-%d")
# Auto-save if in a window and no snapshot saved for today
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
c1,c2,c3,c4 = st.columns(4)
with c1:
    if st.button("📥 Save Morning Snapshot (manual)"):
        ok,msg = save_snapshot_to_supabase(merged, "morning", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c2:
    if st.button("📥 Save Mid Snapshot (manual)"):
        ok,msg = save_snapshot_to_supabase(merged, "mid", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c3:
    if st.button("📥 Save Afternoon Snapshot (manual)"):
        ok,msg = save_snapshot_to_supabase(merged, "afternoon", spot)
        st.success("Saved." if ok else f"Failed: {msg}")
with c4:
    if st.button("📥 Save Evening Snapshot (manual)"):
        ok,msg = save_snapshot_to_supabase(merged, "evening", spot)
        st.success("Saved." if ok else f"Failed: {msg}")

# Snapshot compare UI
st.markdown("### 🔎 Compare saved snapshots")
# fetch list of dates & windows from Supabase (simple approach: fetch distinct date/window)
try:
    resp = supabase.rpc("rpc_get_snapshot_dates", {}).execute() if False else None
except Exception:
    resp = None

# fallback: simple list by querying recent days for each window
saved_files = []
for d_offset in range(0,7):  # last 7 days
    dt = (datetime.now() - timedelta(days=d_offset)).strftime("%Y-%m-%d")
    for w in TIME_WINDOWS.keys():
        df_tmp = fetch_snapshot_from_supabase(dt, w)
        if df_tmp is not None and not df_tmp.empty:
            saved_files.append(f"{dt}__{w}")

if not saved_files:
    st.info("No saved snapshots found (today or recent days). Use manual capture or keep app running during windows for auto-capture.")

colL, colR = st.columns(2)
with colL:
    left_choice = st.selectbox("Left snapshot (date__window)", options=saved_files, index=0 if saved_files else None)
with colR:
    right_choice = st.selectbox("Right snapshot (date__window)", options=saved_files, index=1 if len(saved_files)>1 else 0)

def load_choice(choice_str):
    if not choice_str:
        return None
    date_str, w = choice_str.split("__")
    return fetch_snapshot_from_supabase(date_str, w)

if left_choice and right_choice:
    left_df = load_choice(left_choice)
    right_df = load_choice(right_choice)
    diff_df = compare_snapshots_df(left_df, right_df)
    st.subheader(f"Snapshot Diff: {left_choice} → {right_choice}")
    if diff_df is not None:
        st.dataframe(diff_df.drop(columns=["abs_change"]).head(100), use_container_width=True)
        s_left = snapshot_summary(left_df)
        s_right = snapshot_summary(right_df)
        b_left = bias_score_from_summary(s_left)
        b_right = bias_score_from_summary(s_right)
        st.markdown("**Summary:**")
        st.write(f"- Left ({left_choice}): CE_ΔOI={s_left['CE_ΔOI']}, PE_ΔOI={s_left['PE_ΔOI']}, CE_Vol={s_left['CE_Vol']}, PE_Vol={s_left['PE_Vol']}, AvgIV_CE={s_left['Avg_IV_CE']:.2f}")
        st.write(f"- Right ({right_choice}): CE_ΔOI={s_right['CE_ΔOI']}, PE_ΔOI={s_right['PE_ΔOI']}, CE_Vol={s_right['CE_Vol']}, PE_Vol={s_right['PE_Vol']}, AvgIV_CE={s_right['Avg_IV_CE']:.2f}")
        st.write(f"- Bias Score Left -> {b_left:.3f} | Right -> {b_right:.3f} | ΔBias = {(b_right - b_left):.3f}")
    else:
        st.warning("No valid diff available.")

# Quick comparison button: Morning->Afternoon
if st.button("Quick: Morning → Afternoon (today)"):
    left = fetch_snapshot_from_supabase(today, "morning")
    right = fetch_snapshot_from_supabase(today, "afternoon")
    if left is None or left.empty or right is None or right.empty:
        st.warning("Missing snapshots for Morning or Afternoon today.")
    else:
        d = compare_snapshots_df(left, right)
        st.dataframe(d.drop(columns=["abs_change"]).head(200), use_container_width=True)
        st.write("Bias Δ:", bias_score_from_summary(snapshot_summary(right)) - bias_score_from_summary(snapshot_summary(left)))

st.markdown("---")
st.caption("Notes: Supabase table must exist. Use secrets to store keys. For heavier loads, consider adding RLS policies and server-side inserts.")