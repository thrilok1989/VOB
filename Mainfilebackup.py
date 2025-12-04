# nifty_option_screener_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from functools import lru_cache

st.set_page_config(page_title="Nifty Option Screener v2.0", layout="wide")

# --------------------------
# ---------- SETTINGS -------
# --------------------------
AUTO_REFRESH_SEC = 60                # auto-refresh frequency
PRICE_DELTA_MIN = 0.01              # minimal price delta to consider (absolute)
OI_DELTA_MIN = 1                    # minimal OI delta to consider
ATM_STRIKE_WINDOW = 8               # ATM ± 8 strikes (confirmed)
SCORE_WEIGHTS = {
    "chg_oi": 2.0,
    "volume": 0.5,
    "oi": 0.2,
    "iv": 0.3
}
BREAKOUT_INDEX_WEIGHTS = {
    "atm_oi_shift": 0.4,
    "winding_balance": 0.3,
    "vol_oi_div": 0.2,
    "gamma_pressure": 0.1
}

# --------------------------
# ---- Helper Functions ----
# --------------------------
def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh()

@st.cache_data(ttl=180)
def fetch_nse_option_chain():
    """Fetch NSE option chain JSON for NIFTY. Defensive for NSE blocking."""
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    s = requests.Session()
    try:
        # seed session
        s.get("https://www.nseindia.com", headers=headers, timeout=5)
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

def classify_strike_type(strike, atm):
    if strike == atm:
        return "ATM"
    elif strike < atm:
        # lower strike -> ITM for Calls, OTM for Puts
        return "ITM_CALL"  # shorthand
    else:
        return "OTM_CALL"

def get_winding_label(chg_oi):
    if chg_oi > 0:
        return "Winding"
    elif chg_oi < 0:
        return "Unwinding"
    else:
        return "NoChange"

def price_oi_divergence_label(chg_oi, vol, ltp_change):
    """Mapping for Volume-OI-Price divergence"""
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)
    if oi_up and vol_up and price_up:
        return "Fresh Build (Aggressive)"
    if oi_up and vol_up and not price_up:
        return "Short Buildup / Seller Aggression"
    if not oi_up and vol_up and price_up:
        return "Long Covering (Buy-side squeeze)"
    if not oi_up and vol_up and not price_up:
        return "Put Covering / Price fall with exit"
    if oi_up and not vol_up:
        return "Weak Build"
    if (not oi_up) and not vol_up:
        return "Weak Unwind"
    return "Neutral"

def interpret_itm_otm(strike, atm, chg_oi_ce, chg_oi_pe):
    """
    Returns interpretation string given OI changes on both sides.
    Use the rules described in the conversation.
    """
    int_ce = ""
    int_pe = ""
    # CALL side
    if strike < atm:
        # ITM CALL
        if chg_oi_ce < 0:
            int_ce = "Bullish (ITM CE Unwind)"
        elif chg_oi_ce > 0:
            int_ce = "Bearish (ITM CE Winding)"
        else:
            int_ce = "NoSign (ITM CE)"
    elif strike > atm:
        # OTM CALL
        if chg_oi_ce > 0:
            int_ce = "Resistance Forming (OTM CE Winding)"
        elif chg_oi_ce < 0:
            int_ce = "Resistance Weakening (OTM CE Unwind) → Bullish"
        else:
            int_ce = "NoSign (OTM CE)"
    else:
        int_ce = "ATM CE Zone"

    # PUT side
    if strike > atm:
        # ITM PUT
        if chg_oi_pe < 0:
            int_pe = "Bullish (ITM PE Unwind)"
        elif chg_oi_pe > 0:
            int_pe = "Bearish (ITM PE Winding)"
        else:
            int_pe = "NoSign (ITM PE)"
    elif strike < atm:
        # OTM PUT
        if chg_oi_pe > 0:
            int_pe = "Support Forming (OTM PE Winding)"
        elif chg_oi_pe < 0:
            int_pe = "Support Weakening (OTM PE Unwind) → Bearish"
        else:
            int_pe = "NoSign (OTM PE)"
    else:
        int_pe = "ATM PE Zone"

    return f"{int_ce} | {int_pe}"

def strike_strength_score(row, weights=SCORE_WEIGHTS):
    """Custom strike strength combining several factors."""
    chg_oi = safe_float(row.get("Chg_OI_CE", 0)) + safe_float(row.get("Chg_OI_PE", 0))
    vol = safe_float(row.get("Vol_CE", 0)) + safe_float(row.get("Vol_PE", 0))
    oi = safe_float(row.get("OI_CE", 0)) + safe_float(row.get("OI_PE", 0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if not np.isnan(iv_ce) or not np.isnan(iv_pe) else 0

    score = (weights["chg_oi"] * chg_oi) + (weights["volume"] * vol) + (weights["oi"] * oi) + (weights["iv"] * iv)
    # normalize a bit
    return score

def max_pain_from_expiry(records):
    """Compute max pain strike for given expiry records (records: list of 'data' items)."""
    # build list of strikes and total loss (writer pain) = sum of option premiums * number of contracts...
    # But NSE option chain doesn't provide total premium easily. We'll approximate with lastPrice * openInterest.
    pain = {}
    for item in records:
        strike = item.get("strikePrice")
        ce = item.get("CE")
        pe = item.get("PE")
        # approximate loss for writers at each strike level by premium * OI
        ce_loss = 0
        pe_loss = 0
        if ce:
            ce_loss = safe_float(ce.get("lastPrice", 0)) * safe_int(ce.get("openInterest", 0))
        if pe:
            pe_loss = safe_float(pe.get("lastPrice", 0)) * safe_int(pe.get("openInterest", 0))
        # total pain for each possible underlying settlement (we need to compute across strike space),
        # the simple approach: for each candidate settlement (strike), compute sum losses:
        # We'll instead compute a simplified metric: per strike, sum of premiums*OI as pain near that strike.
        pain.setdefault(strike, 0)
        pain[strike] += ce_loss + pe_loss
    # lower pain value means max pain; but since we approximated, we return the strike with minimum aggregated pain
    if not pain:
        return None
    pain_df = pd.Series(pain).sort_values()
    return int(pain_df.index[0])

def gamma_pressure_metric(row, atm):
    """
    Heuristic: change in OI weighted by inverse distance from ATM
    Larger positive => pressure for move towards that side
    """
    strike = row["strikePrice"]
    dist = abs(strike - atm) / 50.0  # normalize by strike gap (approx 50)
    dist = max(dist, 1e-6)
    chg_oi_sum = safe_float(row.get("Chg_OI_CE", 0)) - safe_float(row.get("Chg_OI_PE", 0))
    # weight by closeness: closer strikes have bigger gamma effect
    return chg_oi_sum / (dist)

def breakout_probability_index(merged_df, atm):
    """Heuristic 0-100 breakout probability based on several signals."""
    # 1) ATM OI shift: magnitude of OI built immediately around ATM (±1 strike)
    near_mask = merged_df["strikePrice"].between(atm - 50, atm + 50)
    atm_chg_oi = merged_df.loc[near_mask, ["Chg_OI_CE", "Chg_OI_PE"]].abs().sum().sum()
    atm_score = min(atm_chg_oi / 50000.0, 1.0)  # scale

    # 2) winding balance: count of winding vs unwinding in ATM±8 range
    winding_count = (merged_df[["CE_Winding", "PE_Winding"]] == "Winding").sum().sum()
    unwinding_count = (merged_df[["CE_Winding", "PE_Winding"]] == "Unwinding").sum().sum()
    if winding_count + unwinding_count == 0:
        winding_balance = 0.5
    else:
        winding_balance = winding_count / (winding_count + unwinding_count)

    # 3) vol-oi divergence score (presence of high vol + chg_oi)
    vol_oi_scores = (merged_df[["Vol_CE", "Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE", "Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum() / 100000.0, 1.0)

    # 4) gamma pressure
    gamma = merged_df.apply(lambda r: gamma_pressure_metric(r, atm), axis=1).abs().sum()
    gamma_score = min(gamma / 10000.0, 1.0)

    # Combine with weights
    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"] * atm_score) + (w["winding_balance"] * winding_balance) + (w["vol_oi_div"] * vol_oi_score) + (w["gamma_pressure"] * gamma_score)
    return int(np.clip(combined * 100, 0, 100))

# --------------------------
# ---- MAIN APP LOGIC -----
# --------------------------
st.title("📊 NIFTY Option Screener v2.0 — Full Option-Chain Engine")
st.markdown("Auto-refresh every 60s • ATM ± 8 strikes • Full Winding/Unwinding • Max Pain • Breakout Index")

data = fetch_nse_option_chain()
if data is None:
    st.stop()

records = data.get("records", {})
raw = records.get("data", [])
expiries = records.get("expiryDates", [])
spot = safe_float(records.get("underlyingValue", 0.0))

expiry = st.selectbox("Select expiry", expiries, index=0 if expiries else None)

# Filter raw for selected expiry
expiry_rows = [r for r in raw if r.get("expiryDate") == expiry]
if not expiry_rows:
    st.warning("No data for selected expiry")
    st.stop()

# Build CE and PE lists including LTP, IV etc.
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

# Determine strike gap and ATM
strike_gap = strike_gap_from_series(df_ce["strikePrice"])
atm_strike = min(df_ce["strikePrice"].tolist(), key=lambda x: abs(x - spot))

lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)

# Filter to ATM ± 8 gaps
df_ce = df_ce[(df_ce["strikePrice"] >= lower) & (df_ce["strikePrice"] <= upper)].reset_index(drop=True)
df_pe = df_pe[(df_pe["strikePrice"] >= lower) & (df_pe["strikePrice"] <= upper)].reset_index(drop=True)

# Merge
merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice").reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# session store for prev LTP to compute price deltas (for winding classification)
if "prev_ltps" not in st.session_state:
    st.session_state["prev_ltps"] = {}

labels = []
polarity_sum = 0

for i, r in merged.iterrows():
    strike = int(r["strikePrice"])

    # keys for session
    key_ce = f"{expiry}_{strike}_CE"
    key_pe = f"{expiry}_{strike}_PE"

    ltp_ce = safe_float(r.get("LTP_CE", 0.0))
    ltp_pe = safe_float(r.get("LTP_PE", 0.0))
    prev_ce = st.session_state["prev_ltps"].get(key_ce, None)
    prev_pe = st.session_state["prev_ltps"].get(key_pe, None)

    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)

    # update prev store
    st.session_state["prev_ltps"][key_ce] = ltp_ce
    st.session_state["prev_ltps"][key_pe] = ltp_pe

    # OI changes
    chg_oi_ce = safe_int(r.get("Chg_OI_CE", 0))
    chg_oi_pe = safe_int(r.get("Chg_OI_PE", 0))

    # Winding labels
    ce_wind = get_winding_label(chg_oi_ce)
    pe_wind = get_winding_label(chg_oi_pe)
    merged.at[i, "CE_Winding"] = ce_wind
    merged.at[i, "PE_Winding"] = pe_wind

    # Price-OI divergence labels
    ce_div = price_oi_divergence_label(chg_oi_ce, safe_int(r.get("Vol_CE", 0)), ce_price_delta)
    pe_div = price_oi_divergence_label(chg_oi_pe, safe_int(r.get("Vol_PE", 0)), pe_price_delta)
    merged.at[i, "CE_Divergence"] = ce_div
    merged.at[i, "PE_Divergence"] = pe_div

    # Strike type interpretation
    merged.at[i, "Interpretation"] = interpret_itm_otm(strike, atm_strike, chg_oi_ce, chg_oi_pe)

    # Strength score
    merged.at[i, "Strength_Score"] = strike_strength_score(r)

    # gamma pressure
    merged.at[i, "Gamma_Pressure"] = gamma_pressure_metric(r, atm_strike)

    # polarity for market bias (simple)
    # +1 if CE short covering or CE unwind near ITM (bullish), -1 for opposite
    pol = 0
    # CE logic
    if strike < atm_strike:
        # ITM CE
        if chg_oi_ce < 0:
            pol += 1
        elif chg_oi_ce > 0:
            pol -= 1
    else:
        # OTM CE
        if chg_oi_ce > 0:
            pol -= 0.5
        elif chg_oi_ce < 0:
            pol += 0.5
    # PE logic
    if strike > atm_strike:
        # ITM PE
        if chg_oi_pe < 0:
            pol += 1
        elif chg_oi_pe > 0:
            pol -= 1
    else:
        # OTM PE
        if chg_oi_pe > 0:
            pol += 0.5
        elif chg_oi_pe < 0:
            pol -= 0.5

    polarity_sum += pol
    merged.at[i, "Polarity"] = pol
    merged.at[i, "CE_Price_Delta"] = ce_price_delta
    merged.at[i, "PE_Price_Delta"] = pe_price_delta

# Aggregate summaries
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

# ITM vs OTM aggregates
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

# Winding percentages
itm_ce_winding_pct = (ITM_CE_winding_count / (merged.loc[itm_ce_mask].shape[0] or 1)) * 100
otm_ce_winding_pct = (OTM_CE_winding_count / (merged.loc[otm_ce_mask].shape[0] or 1)) * 100
itm_pe_winding_pct = (ITM_PE_winding_count / (merged.loc[itm_pe_mask].shape[0] or 1)) * 100
otm_pe_winding_pct = (OTM_PE_winding_count / (merged.loc[otm_pe_mask].shape[0] or 1)) * 100

# Heatmap-ready table for change in OI
chg_oi_heatmap = merged[["strikePrice", "Chg_OI_CE", "Chg_OI_PE"]].set_index("strikePrice")

# Max pain (approx)
max_pain = max_pain_from_expiry(expiry_rows)

# Breakout probability
breakout_index = breakout_probability_index(merged, atm_strike)

# ATM-shift detection: is OI building above/below ATM such that distribution shifts?
# compute center of mass of OI on CE side and PE side
def center_of_mass_oi(side_col):
    s = merged[["strikePrice", side_col]].dropna()
    if s[side_col].sum() == 0:
        return atm_strike
    return int((s["strikePrice"] * s[side_col]).sum() / s[side_col].sum())

ce_com = center_of_mass_oi("OI_CE")
pe_com = center_of_mass_oi("OI_PE")

atm_shift = "Neutral"
if ce_com > atm_strike + strike_gap:
    atm_shift = "Upward (CE build above ATM)"
elif ce_com < atm_strike - strike_gap:
    atm_shift = "Downward (CE build below ATM)"
if pe_com > atm_strike + strike_gap:
    atm_shift += " | PE build above ATM"
elif pe_com < atm_strike - strike_gap:
    atm_shift += " | PE build below ATM"

# Market bias simplified
if polarity_sum > 3:
    market_bias = "Strong Bullish"
elif polarity_sum > 1:
    market_bias = "Bullish"
elif polarity_sum < -3:
    market_bias = "Strong Bearish"
elif polarity_sum < -1:
    market_bias = "Bearish"
else:
    market_bias = "Neutral"

# Display top-line metrics
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
    st.metric("Breakout Index", f"{breakout_index}%")
    st.metric("Market Bias", market_bias)

st.markdown("### 🔎 ITM / OTM Pressure Summary (ATM ± 8 strikes)")
pressure_table = pd.DataFrame([
    {
        "Category": "ITM CE OI",
        "Value": int(ITM_CE_OI),
        "Winding_Count": int(ITM_CE_winding_count),
        "Winding_%": round(itm_ce_winding_pct, 1)
    },
    {
        "Category": "OTM CE OI",
        "Value": int(OTM_CE_OI),
        "Winding_Count": int(OTM_CE_winding_count),
        "Winding_%": round(otm_ce_winding_pct, 1)
    },
    {
        "Category": "ITM PE OI",
        "Value": int(ITM_PE_OI),
        "Winding_Count": int(ITM_PE_winding_count),
        "Winding_%": round(itm_pe_winding_pct, 1)
    },
    {
        "Category": "OTM PE OI",
        "Value": int(OTM_PE_OI),
        "Winding_Count": int(OTM_PE_winding_count),
        "Winding_%": round(otm_pe_winding_pct, 1)
    }
])
st.dataframe(pressure_table, use_container_width=True)

# Winding vs Unwinding summary counts
winding_unwind_summary = pd.DataFrame([{
    "Metric": "CE Winding Count",
    "Value": int((merged["CE_Winding"] == "Winding").sum())
}, {
    "Metric": "CE Unwinding Count",
    "Value": int((merged["CE_Winding"] == "Unwinding").sum())
}, {
    "Metric": "PE Winding Count",
    "Value": int((merged["PE_Winding"] == "Winding").sum())
}, {
    "Metric": "PE Unwinding Count",
    "Value": int((merged["PE_Winding"] == "Unwinding").sum())
}])
st.markdown("### 🔁 Winding vs Unwinding Summary")
st.dataframe(winding_unwind_summary, use_container_width=True)

st.markdown("### 🧾 Strike-level Table (ATM ± 8 strikes)")
# Select display columns
display_cols = [
    "strikePrice", "OI_CE", "Chg_OI_CE", "Vol_CE", "LTP_CE", "CE_Price_Delta", "CE_Winding", "CE_Divergence",
    "OI_PE", "Chg_OI_PE", "Vol_PE", "LTP_PE", "PE_Price_Delta", "PE_Winding", "PE_Divergence",
    "Strength_Score", "Gamma_Pressure", "Polarity", "Interpretation"
]
for c in display_cols:
    if c not in merged.columns:
        merged[c] = np.nan

st.dataframe(merged[display_cols].reset_index(drop=True), use_container_width=True)

st.markdown("### 🔥 Change-in-OI Heatmap (visual hint)")
# We'll color positive/negative for quick visuals using pandas styling exported to HTML via st.write
heatmap_df = chg_oi_heatmap.copy()
# simple color map: positive green, negative red
def color_chg(val):
    try:
        if val > 0:
            return "background-color: #d4f4dd"  # light green
        elif val < 0:
            return "background-color: #f8d7da"  # light red
        else:
            return ""
    except:
        return ""

st.dataframe(heatmap_df.style.applymap(color_chg), use_container_width=True)

st.markdown("### 🧠 Max Pain (approximate) & ATM Shift")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Approx. Max Pain", max_pain)
with col2:
    st.info(f"ATM Shift info: {atm_shift}")

# Auto trade suggestion based on combined signals (careful: heuristic)
st.markdown("### 🤖 Auto Trade Suggestion (heuristic & winding-aware)")
suggestions = []
# rules
# strong bullish if: market_bias bullish, breakout_index high, ITM CE unwind dominant, ITM PE unwind present
if market_bias in ["Strong Bullish", "Bullish"] and breakout_index >= 60 and (ITM_CE_winding_count < ITM_PE_winding_count or ITM_CE_winding_count == 0):
    suggestions.append(("BULL", "Consider BUY CALLs or Bull Spreads — conditions favour upside."))
# strong bearish
if market_bias in ["Strong Bearish", "Bearish"] and breakout_index >= 60 and (ITM_PE_winding_count < ITM_CE_winding_count or ITM_PE_winding_count == 0):
    suggestions.append(("BEAR", "Consider BUY PUTs or Bear Spreads — conditions favour downside."))

# gamma squeeze detection
if merged["Gamma_Pressure"].abs().max() > 5000 and breakout_index >= 50:
    suggestions.append(("SPIKE RISK", "High gamma pressure near ATM — expect sharp moves."))

if not suggestions:
    suggestions = [("NEUTRAL", "No decisive winding/unwinding consensus — consider waiting or selling premium.")]

for tag, text in suggestions:
    if tag == "BULL":
        st.success(f"🟩 {tag}: {text}")
    elif tag == "BEAR":
        st.error(f"🟥 {tag}: {text}")
    else:
        st.warning(f"⚠️ {tag}: {text}")

# Developer / tuning controls
with st.expander("⚙️ Developer / Tuning Controls & Notes"):
    st.write("Tuning constants in this file:")
    st.write(f" - ATM_STRIKE_WINDOW = {ATM_STRIKE_WINDOW} (strikes each side)")
    st.write(f" - Strike gap detected = {strike_gap}")
    st.write(" - PRICE_DELTA_MIN, OI_DELTA_MIN can be adjusted for noise filtering")
    st.write(" - Score weights:", SCORE_WEIGHTS)
    st.write(" - Breakout index weights:", BREAKOUT_INDEX_WEIGHTS)
    st.write("Session stored previous LTP samples (first 40 keys):")
    st.write({k: v for k, v in list(st.session_state['prev_ltps'].items())[:40]})

st.markdown("---")
st.caption("Notes: This is a heuristic-driven tool. Use risk management. Consider hooking to broker APIs (Dhan/Upstox) for reliable LTP/IV updates and to avoid NSE blocking.")

# End of script
