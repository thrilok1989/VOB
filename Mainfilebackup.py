# nifty_option_screener_v3.py
"""
Nifty Option Screener v3.0 — Full Option Chain Engine
Features:
 - ATM ± 8 strikes (by strike gap)
 - Winding/Unwinding (8-type logic using OI + price delta)
 - IV change analysis
 - Greeks (Delta, Gamma, Vega, Theta) via Black-Scholes (European approx)
 - Gamma Exposure (GEX) heuristic
 - Delta imbalance (CE vs PE)
 - Max Pain (approx)
 - Volume-OI-Price divergence tags
 - Strike strength score
 - Breakout Probability Index (heuristic)
 - ATM-shift detection (center-of-mass)
 - Auto Trade Suggestion (heuristic)
 - Developer tuning controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from math import log, sqrt, exp
from scipy.stats import norm

# -----------------------
#  USER TUNABLE CONSTANTS
# -----------------------
AUTO_REFRESH_SEC = 60            # auto-refresh frequency
LOT_SIZE = 50                    # default NIFTY lot size (change if needed)
RISK_FREE_RATE = 0.06            # annual risk-free rate (approx)
PRICE_DELTA_MIN = 0.005          # minimal LTP delta (absolute) to consider (in index points)
OI_DELTA_MIN = 1                 # minimal change in OI to consider
ATM_STRIKE_WINDOW = 8            # we use ATM ± 8 strikes (gaps)
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}

# -----------------------
#  UTIL FUNCTIONS
# -----------------------
st.set_page_config(page_title="Nifty Option Screener v3.0", layout="wide")
st.title("📊 NIFTY Option Screener v3.0 — Full Option Chain + Greeks + GEX")

def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh()

@st.cache_data(ttl=180)
def fetch_nse_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=headers, timeout=5)  # seed session
        r = s.get(url, headers=headers, timeout=7)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NSE data: {e}")
        return None

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

# Black-Scholes for European options (approx for index options)
def bs_d1(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def bs_d2(S, K, r, sigma, tau):
    return bs_d1(S, K, r, sigma, tau) - sigma * np.sqrt(tau)

def bs_delta(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0 or sigma <= 0:
        # intrinsic fallback
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, sigma, tau)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

def bs_gamma(S, K, r, sigma, tau):
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, sigma, tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

def bs_vega(S, K, r, sigma, tau):
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, sigma, tau)
    return S * norm.pdf(d1) * np.sqrt(tau)

def bs_theta(S, K, r, sigma, tau, option_type="call"):
    # approximate theta per year
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, sigma, tau)
    d2 = d1 - sigma * np.sqrt(tau)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau))
    if option_type == "call":
        term2 = r * K * np.exp(-r * tau) * norm.cdf(d2)
        return term1 - term2
    else:
        term2 = r * K * np.exp(-r * tau) * norm.cdf(-d2)
        return term1 + term2

# Score for each strike
def strike_strength_score(row, weights=SCORE_WEIGHTS):
    chg_oi = safe_float(row.get("Chg_OI_CE", 0)) + safe_float(row.get("Chg_OI_PE", 0))
    vol = safe_float(row.get("Vol_CE", 0)) + safe_float(row.get("Vol_PE", 0))
    oi = safe_float(row.get("OI_CE", 0)) + safe_float(row.get("OI_PE", 0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    score = weights["chg_oi"] * chg_oi + weights["volume"] * vol + weights["oi"] * oi + weights["iv"] * iv
    return score

# Max Pain approx (using premium * OI)
def max_pain_from_expiry(records):
    pain = {}
    for item in records:
        strike = item.get("strikePrice")
        ce = item.get("CE")
        pe = item.get("PE")
        ce_loss = 0
        pe_loss = 0
        if ce:
            ce_loss = safe_float(ce.get("lastPrice", 0)) * safe_int(ce.get("openInterest", 0))
        if pe:
            pe_loss = safe_float(pe.get("lastPrice", 0)) * safe_int(pe.get("openInterest", 0))
        pain.setdefault(strike, 0)
        pain[strike] += ce_loss + pe_loss
    if not pain:
        return None
    pain_series = pd.Series(pain).sort_values()
    return int(pain_series.index[0])

# Price-OI-Vol divergence tag
def price_oi_divergence_label(chg_oi, vol, ltp_change):
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)
    if oi_up and vol_up and price_up:
        return "Fresh Build (Aggressive)"
    if oi_up and vol_up and not price_up:
        return "Seller Aggressive / Short Build"
    if not oi_up and vol_up and price_up:
        return "Long Covering (Buy squeeze)"
    if not oi_up and vol_up and not price_up:
        return "Put Covering / Sell with exit"
    if oi_up and not vol_up:
        return "Weak Build"
    if (not oi_up) and not vol_up:
        return "Weak Unwind"
    return "Neutral"

# Interpret ITM/OTM
def interpret_itm_otm(strike, atm, chg_oi_ce, chg_oi_pe):
    if strike < atm:
        # ITM Call
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

# Gamma pressure heuristic
def gamma_pressure_metric(row, atm, strike_gap):
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    chg_oi_sum = safe_float(row.get("Chg_OI_CE", 0)) - safe_float(row.get("Chg_OI_PE", 0))
    return chg_oi_sum / dist

# Breakout index heuristic
def breakout_probability_index(merged_df, atm, strike_gap):
    near_mask = merged_df["strikePrice"].between(atm - strike_gap, atm + strike_gap)
    atm_chg_oi = merged_df.loc[near_mask, ["Chg_OI_CE", "Chg_OI_PE"]].abs().sum().sum()
    atm_score = min(atm_chg_oi / 50000.0, 1.0)

    winding_count = (merged_df[["CE_Winding", "PE_Winding"]] == "Winding").sum().sum()
    unwinding_count = (merged_df[["CE_Winding", "PE_Winding"]] == "Unwinding").sum().sum()
    winding_balance = winding_count / (winding_count + unwinding_count) if (winding_count + unwinding_count) > 0 else 0.5

    vol_oi_scores = (merged_df[["Vol_CE", "Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE", "Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum() / 100000.0, 1.0)

    gamma = merged_df.apply(lambda r: gamma_pressure_metric(r, atm, strike_gap), axis=1).abs().sum()
    gamma_score = min(gamma / 10000.0, 1.0)

    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"] * atm_score) + (w["winding_balance"] * winding_balance) + (w["vol_oi_div"] * vol_oi_score) + (w["gamma_pressure"] * gamma_score)
    return int(np.clip(combined * 100, 0, 100))

# -----------------------
#  MAIN APP FLOW
# -----------------------
data = fetch_nse_option_chain()
if data is None:
    st.stop()

records = data.get("records", {})
raw = records.get("data", [])
expiries = records.get("expiryDates", [])
spot = safe_float(records.get("underlyingValue", 0.0))

expiry = st.selectbox("Select expiry", expiries, index=0 if expiries else None)

# Build CE/PE lists for chosen expiry
expiry_rows = [r for r in raw if r.get("expiryDate") == expiry]
if not expiry_rows:
    st.warning("No data for selected expiry")
    st.stop()

ce_rows = []
pe_rows = []
for it in expiry_rows:
    strike = safe_int(it.get("strikePrice", 0))
    ce = it.get("CE")
    pe = it.get("PE")
    if ce:
        ce_rows.append({
            "strikePrice": strike,
            "OI_CE": safe_int(ce.get("openInterest")),
            "Chg_OI_CE": safe_int(ce.get("changeinOpenInterest")),
            "Vol_CE": safe_int(ce.get("totalTradedVolume")),
            "LTP_CE": safe_float(ce.get("lastPrice")),
            "IV_CE": safe_float(ce.get("impliedVolatility"))
        })
    if pe:
        pe_rows.append({
            "strikePrice": strike,
            "OI_PE": safe_int(pe.get("openInterest")),
            "Chg_OI_PE": safe_int(pe.get("changeinOpenInterest")),
            "Vol_PE": safe_int(pe.get("totalTradedVolume")),
            "LTP_PE": safe_float(pe.get("lastPrice")),
            "IV_PE": safe_float(pe.get("impliedVolatility"))
        })

df_ce = pd.DataFrame(ce_rows)
df_pe = pd.DataFrame(pe_rows)
if df_ce.empty or df_pe.empty:
    st.warning("Insufficient CE/PE data")
    st.stop()

strike_gap = strike_gap_from_series(df_ce["strikePrice"])
atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))
lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

df_ce = df_ce[(df_ce["strikePrice"] >= lower) & (df_ce["strikePrice"] <= upper)].reset_index(drop=True)
df_pe = df_pe[(df_pe["strikePrice"] >= lower) & (df_pe["strikePrice"] <= upper)].reset_index(drop=True)

merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# session storage for previous LTPs and IVs
if "prev_ltps_v3" not in st.session_state:
    st.session_state["prev_ltps_v3"] = {}
if "prev_ivs_v3" not in st.session_state:
    st.session_state["prev_ivs_v3"] = {}

# compute Greeks, deltas, labels, gex etc.
for i, row in merged.iterrows():
    strike = int(row["strikePrice"])
    # ltp and iv
    ltp_ce = safe_float(row.get("LTP_CE", 0.0))
    ltp_pe = safe_float(row.get("LTP_PE", 0.0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))

    # previous values (for change detection)
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

    # OI changes
    chg_oi_ce = safe_int(row.get("Chg_OI_CE", 0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE", 0))

    # winding/unwinding
    merged.at[i, "CE_Winding"] = "Winding" if chg_oi_ce > 0 else ("Unwinding" if chg_oi_ce < 0 else "NoChange")
    merged.at[i, "PE_Winding"] = "Winding" if chg_oi_pe > 0 else ("Unwinding" if chg_oi_pe < 0 else "NoChange")

    # divergence tags
    merged.at[i, "CE_Divergence"] = price_oi_divergence_label(chg_oi_ce, safe_int(row.get("Vol_CE", 0)), ce_price_delta)
    merged.at[i, "PE_Divergence"] = price_oi_divergence_label(chg_oi_pe, safe_int(row.get("Vol_PE", 0)), pe_price_delta)

    # strike interpretation
    merged.at[i, "Interpretation"] = interpret_itm_otm(strike, atm_strike, chg_oi_ce, chg_oi_pe)

    # time to expiry in years
    # we don't have exact time-of-day; for intraday decisions, using days-to-expiry/365 is ok
    # assume remaining days from expiry list? We only have expiry string; approximate tau = (expiry_date - today)/365
    # For simplicity we'll approximate tau as 7 days for weekly and 30 days for monthly if unknown.
    # Better: user can supply exact expiry date parsing. We'll approximate using 7 days for safety.
    tau = 7.0 / 365.0

    # use IV if available; if not, fallback to small value to avoid divide by zero
    sigma_ce = iv_ce / 100.0 if not np.isnan(iv_ce) and iv_ce > 0 else 0.25
    sigma_pe = iv_pe / 100.0 if not np.isnan(iv_pe) and iv_pe > 0 else 0.25

    # Greeks
    # For CE and PE we compute delta (call positive 0..1, put negative -1..0), gamma, vega, theta (yearly)
    try:
        delta_ce = bs_delta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
        gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
        vega_ce = bs_vega(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
        theta_ce = bs_theta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
    except Exception:
        delta_ce = 0.0; gamma_ce = 0.0; vega_ce = 0.0; theta_ce = 0.0

    try:
        delta_pe = bs_delta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        vega_pe = bs_vega(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
        theta_pe = bs_theta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
    except Exception:
        delta_pe = 0.0; gamma_pe = 0.0; vega_pe = 0.0; theta_pe = 0.0

    merged.at[i, "Delta_CE"] = delta_ce
    merged.at[i, "Gamma_CE"] = gamma_ce
    merged.at[i, "Vega_CE"] = vega_ce
    merged.at[i, "Theta_CE"] = theta_ce

    merged.at[i, "Delta_PE"] = delta_pe
    merged.at[i, "Gamma_PE"] = gamma_pe
    merged.at[i, "Vega_PE"] = vega_pe
    merged.at[i, "Theta_PE"] = theta_pe

    # GEX contributions (approx)
    # One contract notional ~ LOT_SIZE * Spot
    oi_ce = safe_int(row.get("OI_CE", 0))
    oi_pe = safe_int(row.get("OI_PE", 0))
    notional_per_contract = LOT_SIZE * spot
    # dollar gamma contribution approx = gamma * notional_per_contract * number_of_contracts
    gex_ce = gamma_ce * notional_per_contract * oi_ce
    gex_pe = gamma_pe * notional_per_contract * oi_pe
    merged.at[i, "GEX_CE"] = gex_ce
    merged.at[i, "GEX_PE"] = gex_pe
    merged.at[i, "GEX_Net"] = gex_ce - gex_pe

    # strength score
    merged.at[i, "Strength_Score"] = strike_strength_score(row)

    # gamma pressure heuristic
    merged.at[i, "Gamma_Pressure"] = gamma_pressure_metric(row, atm_strike, strike_gap)

    # price deltas & iv deltas
    merged.at[i, "CE_Price_Delta"] = ce_price_delta
    merged.at[i, "PE_Price_Delta"] = pe_price_delta
    merged.at[i, "CE_IV_Delta"] = ce_iv_delta
    merged.at[i, "PE_IV_Delta"] = pe_iv_delta

# Aggregations
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

# ITM/OTM masks
itm_ce_mask = merged["strikePrice"] < atm_strike
otm_ce_mask = merged["strikePrice"] > atm_strike
itm_pe_mask = merged["strikePrice"] > atm_strike
otm_pe_mask = merged["strikePrice"] < atm_strike

ITM_CE_OI = merged.loc[itm_ce_mask, "OI_CE"].sum()
OTM_CE_OI = merged.loc[otm_ce_mask, "OI_CE"].sum()
ITM_PE_OI = merged.loc[itm_pe_mask, "OI_PE"].sum()
OTM_PE_OI = merged.loc[otm_pe_mask, "OI_PE"].sum()

ITM_CE_winding_count = (merged.loc[itm_ce_mask, "CE_Winding"] == "Winding").sum()
OTM_CE_winding_count = (merged.loc[otm_ce_mask, "CE_Winding"] == "Winding").sum()
ITM_PE_winding_count = (merged.loc[itm_pe_mask, "PE_Winding"] == "Winding").sum()
OTM_PE_winding_count = (merged.loc[otm_pe_mask, "PE_Winding"] == "Winding").sum()

itm_ce_winding_pct = (ITM_CE_winding_count / (merged.loc[itm_ce_mask].shape[0] or 1)) * 100
otm_ce_winding_pct = (OTM_CE_winding_count / (merged.loc[otm_ce_mask].shape[0] or 1)) * 100
itm_pe_winding_pct = (ITM_PE_winding_count / (merged.loc[itm_pe_mask].shape[0] or 1)) * 100
otm_pe_winding_pct = (OTM_PE_winding_count / (merged.loc[otm_pe_mask].shape[0] or 1)) * 100

# net delta exposure (approx) = sum(delta * OI * lot_size)
merged["CE_Delta_Exposure"] = merged["Delta_CE"].fillna(0) * merged["OI_CE"].fillna(0) * LOT_SIZE
merged["PE_Delta_Exposure"] = merged["Delta_PE"].fillna(0) * merged["OI_PE"].fillna(0) * LOT_SIZE
net_delta_exposure = merged["CE_Delta_Exposure"].sum() + merged["PE_Delta_Exposure"].sum()  # PE delta is negative already

# GEX totals
total_gex_ce = merged["GEX_CE"].sum()
total_gex_pe = merged["GEX_PE"].sum()
total_gex_net = merged["GEX_Net"].sum()

# Max pain
max_pain = max_pain_from_expiry(expiry_rows)

# Breakout index
breakout_index = breakout_probability_index(merged, atm_strike, strike_gap)

# ATM shift detection (center-of-mass)
def center_of_mass_oi(merged_df, col):
    s = merged_df[["strikePrice", col]].dropna()
    if s[col].sum() == 0:
        return atm_strike
    return int((s["strikePrice"] * s[col]).sum() / s[col].sum())

ce_com = center_of_mass_oi(merged, "OI_CE")
pe_com = center_of_mass_oi(merged, "OI_PE")

atm_shift = []
if ce_com > atm_strike + strike_gap:
    atm_shift.append("CE build above ATM")
elif ce_com < atm_strike - strike_gap:
    atm_shift.append("CE build below ATM")
if pe_com > atm_strike + strike_gap:
    atm_shift.append("PE build above ATM")
elif pe_com < atm_strike - strike_gap:
    atm_shift.append("PE build below ATM")
atm_shift_str = " | ".join(atm_shift) if atm_shift else "Neutral"

# polarity simple measure from wind/unwind counts and ITM logic
polarity = 0.0
for _, r in merged.iterrows():
    s = r["strikePrice"]
    chg_ce = safe_int(r.get("Chg_OI_CE", 0))
    chg_pe = safe_int(r.get("Chg_OI_PE", 0))
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

if polarity > 5:
    market_bias = "Strong Bullish"
elif polarity > 1:
    market_bias = "Bullish"
elif polarity < -5:
    market_bias = "Strong Bearish"
elif polarity < -1:
    market_bias = "Bearish"
else:
    market_bias = "Neutral"

# -----------------------
#  UI: show metrics & tables
# -----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Underlying (Spot)", f"{spot:.2f}")
    st.metric("ATM Strike", atm_strike)
with col2:
    st.metric("Total CE OI (window)", int(total_CE_OI))
    st.metric("Total PE OI (window)", int(total_PE_OI))
with col3:
    st.metric("Total CE ΔOI", int(total_CE_chg))
    st.metric("Total PE ΔOI", int(total_PE_chg))
with col4:
    st.metric("Net Delta Exposure (est)", int(net_delta_exposure))
    st.metric("Market Bias", market_bias)

st.markdown("### 🔎 ITM / OTM Pressure Summary (ATM ± 8 strikes)")
pressure_table = pd.DataFrame([
    {"Category": "ITM CE OI", "Value": int(ITM_CE_OI), "Winding_Count": int(ITM_CE_winding_count), "Winding_%": round(itm_ce_winding_pct, 1)},
    {"Category": "OTM CE OI", "Value": int(OTM_CE_OI), "Winding_Count": int(OTM_CE_winding_count), "Winding_%": round(otm_ce_winding_pct, 1)},
    {"Category": "ITM PE OI", "Value": int(ITM_PE_OI), "Winding_Count": int(ITM_PE_winding_count), "Winding_%": round(itm_pe_winding_pct, 1)},
    {"Category": "OTM PE OI", "Value": int(OTM_PE_OI), "Winding_Count": int(OTM_PE_winding_count), "Winding_%": round(otm_pe_winding_pct, 1)}
])
st.dataframe(pressure_table, use_container_width=True)

st.markdown("### 🔁 Winding vs Unwinding Summary")
winding_unwind_summary = pd.DataFrame([
    {"Metric": "CE Winding Count", "Value": int((merged["CE_Winding"] == "Winding").sum())},
    {"Metric": "CE Unwinding Count", "Value": int((merged["CE_Winding"] == "Unwinding").sum())},
    {"Metric": "PE Winding Count", "Value": int((merged["PE_Winding"] == "Winding").sum())},
    {"Metric": "PE Unwinding Count", "Value": int((merged["PE_Winding"] == "Unwinding").sum())}
])
st.dataframe(winding_unwind_summary, use_container_width=True)

st.markdown("### 🧾 Strike-level Table (ATM ± 8 strikes)")
display_cols = [
    "strikePrice",
    "OI_CE", "Chg_OI_CE", "Vol_CE", "LTP_CE", "CE_Price_Delta", "CE_IV_Delta", "CE_Winding", "CE_Divergence", "Delta_CE", "Gamma_CE", "GEX_CE",
    "OI_PE", "Chg_OI_PE", "Vol_PE", "LTP_PE", "PE_Price_Delta", "PE_IV_Delta", "PE_Winding", "PE_Divergence", "Delta_PE", "Gamma_PE", "GEX_PE",
    "Strength_Score", "Gamma_Pressure", "Polarity", "Interpretation"
]
for c in display_cols:
    if c not in merged.columns:
        merged[c] = np.nan
st.dataframe(merged[display_cols].reset_index(drop=True), use_container_width=True)

st.markdown("### 🔥 Change-in-OI Heatmap (visual hint)")
chg_oi_heatmap = merged[["strikePrice", "Chg_OI_CE", "Chg_OI_PE"]].set_index("strikePrice")
def color_chg(val):
    try:
        if val > 0:
            return "background-color: #d4f4dd"
        elif val < 0:
            return "background-color: #f8d7da"
        else:
            return ""
    except:
        return ""
st.dataframe(chg_oi_heatmap.style.applymap(color_chg), use_container_width=True)

st.markdown("### 🧠 Max Pain (approx) & ATM Shift")
col1, col2 = st.columns([1,2])
with col1:
    st.metric("Approx. Max Pain", max_pain)
with col2:
    st.info(f"ATM Shift info: {atm_shift_str}")

st.markdown("### ⚙️ Greeks / Exposure Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total GEX CE (approx)", int(total_gex_ce))
    st.metric("Total GEX PE (approx)", int(total_gex_pe))
with col2:
    st.metric("Net GEX (CE - PE)", int(total_gex_net))
    st.metric("Breakout Index", f"{breakout_index}%")
with col3:
    st.metric("Net Delta Exposure (est)", int(net_delta_exposure))
    st.metric("Lot Size used", LOT_SIZE)

st.markdown("### 🤖 Auto Trade Suggestion (heuristic & explainable)")
suggestions = []
# Bull rule
if market_bias in ["Strong Bullish", "Bullish"] and breakout_index >= 60 and (ITM_CE_winding_count <= ITM_PE_winding_count):
    suggestions.append(("BULL", "Consider BUY CALLs / Bull Spreads — conditions favour upside"))
# Bear rule
if market_bias in ["Strong Bearish", "Bearish"] and breakout_index >= 60 and (ITM_PE_winding_count <= ITM_CE_winding_count):
    suggestions.append(("BEAR", "Consider BUY PUTs / Bear Spreads — conditions favour downside"))
# Gamma spike risk
if merged["Gamma_Pressure"].abs().max() > 5000 and breakout_index >= 50:
    suggestions.append(("SPIKE RISK", "High gamma pressure near ATM — expect sharp moves / higher intraday vol"))
# IV spike check
avg_iv_delta = merged[["CE_IV_Delta", "PE_IV_Delta"]].abs().stack().dropna().mean() if not merged[["CE_IV_Delta", "PE_IV_Delta"]].empty else 0
if avg_iv_delta > 0.5:
    suggestions.append(("IV MOVE", "IV changing quickly — premiums unstable, prefer smaller sizing"))

if not suggestions:
    suggestions = [("NEUTRAL","No decisive winding/unwinding consensus — consider selling premium or waiting")]

for tag, text in suggestions:
    if tag == "BULL":
        st.success(f"🟩 {tag}: {text}")
    elif tag == "BEAR":
        st.error(f"🟥 {tag}: {text}")
    else:
        st.warning(f"⚠️ {tag}: {text}")

with st.expander("⚙️ Developer / Tuning Controls & Notes"):
    st.write("Tuning constants (change in script):")
    st.write(f" - AUTO_REFRESH_SEC = {AUTO_REFRESH_SEC}")
    st.write(f" - LOT_SIZE = {LOT_SIZE}")
    st.write(f" - RISK_FREE_RATE = {RISK_FREE_RATE}")
    st.write(f" - PRICE_DELTA_MIN = {PRICE_DELTA_MIN}")
    st.write(f" - OI_DELTA_MIN = {OI_DELTA_MIN}")
    st.write(f" - ATM_STRIKE_WINDOW = {ATM_STRIKE_WINDOW}")
    st.write(f" - Strike gap detected = {strike_gap}")
    st.write(" - Score weights:", SCORE_WEIGHTS)
    st.write(" - Breakout index weights:", BREAKOUT_INDEX_WEIGHTS)
    st.write("Session prev LTP keys sample (first 40):")
    st.write({k: v for k, v in list(st.session_state["prev_ltps_v3"].items())[:40]})
st.caption("Notes: Approximations used (Black-Scholes Greeks, approximate GEX using lot size). Replace fetch_nse_option_chain with broker API for production reliability.")

# End script
