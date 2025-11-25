import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, time as dtime

st.set_page_config(page_title="Expiry Spike Detector", layout="wide")

# ---------------------------
# Helper: Read secrets
# ---------------------------
# DHAN V2 uses Client ID and Access Token
CLIENT_ID = st.secrets.get("dhanauth", {}).get("CLIENT_ID")
ACCESS_TOKEN = st.secrets.get("dhanauth", {}).get("ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("TELEGRAM_CHAT_ID")

# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Expiry Spike Detector — Settings")

# Mapping common Indices to Dhan Scrip Codes (Essential for V2 API)
INDEX_MAP = {
    "NIFTY": {"code": "13", "segment": "NSE_IDX"},
    "BANKNIFTY": {"code": "25", "segment": "NSE_IDX"},
    "FINNIFTY": {"code": "27", "segment": "NSE_IDX"}
}

symbol_selection = st.sidebar.selectbox("Symbol", list(INDEX_MAP.keys()), index=0)
expiry = st.sidebar.text_input("Expiry Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15)
run_live = st.sidebar.checkbox("Auto-refresh live detection", value=False)
show_raw = st.sidebar.checkbox("Show raw option chain", value=False)
threshold_alert = st.sidebar.slider("Alert threshold (score >= )", 10, 100, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("**Streamlit Secrets needed:**")
st.sidebar.markdown("`dhanauth.CLIENT_ID`, `dhanauth.ACCESS_TOKEN` and telegram credentials.")

# ---------------------------
# Helper functions
# ---------------------------

def send_telegram_message(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        r = requests.post(url, json=payload, timeout=5)
        return r.ok
    except Exception:
        return False


def fetch_option_chain_dhan(symbol_key: str, expiry_date: str):
    """
    Fetch option chain using Dhan API V2.
    """
    if not ACCESS_TOKEN or not CLIENT_ID:
        st.warning("Dhan credentials missing in Streamlit secrets.")
        return None

    # Get the Scrip Code and Segment from the mapper
    scrip_details = INDEX_MAP.get(symbol_key)
    if not scrip_details:
        st.error("Unknown Symbol. Please add it to INDEX_MAP in the script.")
        return None

    url = "https://api.dhan.co/v2/option-chain"
    
    headers = {
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Dhan V2 Payload Structure
    payload = {
        "UnderlyingScripCode": scrip_details["code"],
        "UnderlyingSeg": scrip_details["segment"],
        "Expiry": expiry_date
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data
    except Exception as e:
        st.error(f"Failed to fetch option chain: {e}")
        return None


def normalize_option_chain(raw):
    """
    Normalize Dhan V2 response into DataFrame.
    Dhan V2 structure: {'data': {'OC': { 'strike_price': { 'ce': {...}, 'pe': {...} } }}}
    """
    items = []
    try:
        # Access the nested data structure
        if 'data' not in raw:
            return None
            
        # Dhan raw data is often dictionary keyed by strike, or list
        chain_data = raw['data']
        
        # If it comes as a dict with specific structure (sometimes OC key is present)
        if 'OC' in chain_data:
            chain_data = chain_data['OC']
            
        # Iterate through strikes
        for strike_price, details in chain_data.items():
            ce = details.get('ce', {})
            pe = details.get('pe', {})
            
            # Note: Dhan API does not natively return Greeks (Gamma, IV, Delta) in standard feed.
            # We default them to 0.0 to keep the logic running based on Price/OI.
            
            items.append({
                'strike': float(strike_price),
                
                # Call Data
                'ce_ltp': float(ce.get('last_price', 0.0)),
                'ce_iv': float(ce.get('implied_volatility', 0.0)), # Likely 0 unless premium feed
                'ce_gamma': float(ce.get('gamma', 0.0)),           # Likely 0
                'ce_oi': float(ce.get('oi', 0.0)),
                'ce_oi_change': float(ce.get('oi_change', 0.0)),   # Check strictly if API gives 'oi_change' or requires calc
                'ce_volume': float(ce.get('volume', 0.0)),
                
                # Put Data
                'pe_ltp': float(pe.get('last_price', 0.0)),
                'pe_iv': float(pe.get('implied_volatility', 0.0)), # Likely 0
                'pe_gamma': float(pe.get('gamma', 0.0)),           # Likely 0
                'pe_oi': float(pe.get('oi', 0.0)),
                'pe_oi_change': float(pe.get('oi_change', 0.0)),
                'pe_volume': float(pe.get('volume', 0.0))
            })
            
        df = pd.DataFrame(items).sort_values('strike').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Normalization Error: {e}")
        return None


# ---------------------------
# Gamma + Expiry Spike Logic
# ---------------------------

def gamma_sequence_expiry(option_df: pd.DataFrame, spot_price: float):
    # find nearest strike index (atm)
    strikes = option_df['strike'].values
    if len(strikes) == 0: return None
    
    atm_idx = int(np.argmin(np.abs(strikes - spot_price)))
    atm = option_df.iloc[atm_idx]

    # safe extraction
    ce_gamma = float(atm.get('ce_gamma', 0) or 0)
    pe_gamma = float(atm.get('pe_gamma', 0) or 0)
    ce_oi_chg = float(atm.get('ce_oi_change', 0) or 0)
    pe_oi_chg = float(atm.get('pe_oi_change', 0) or 0)
    ce_iv = float(atm.get('ce_iv', 0) or 0)
    pe_iv = float(atm.get('pe_iv', 0) or 0)
    ce_ltp = float(atm.get('ce_ltp', 0) or 0)
    pe_ltp = float(atm.get('pe_ltp', 0) or 0)

    # 1. gamma pressure (scaled)
    # NOTE: Since Dhan doesn't return Gamma, this will be 0 unless you calculate it externally.
    gamma_pressure = (abs(ce_gamma) + abs(pe_gamma)) * 10000

    # 2. hedge imbalance - absolute difference in OI change across ATM and near strikes
    hedge_imbalance = abs(ce_oi_chg - pe_oi_chg)

    # 3. gamma flip: both sides show OI decrease (short unwinding) -> explosive
    gamma_flip = (ce_oi_chg < 0 and pe_oi_chg < 0)

    # 4. iv crush detection: average IV dropping low
    avg_iv = (ce_iv + pe_iv) / 2 if (ce_iv or pe_iv) else 0
    iv_crush = avg_iv < 15 and avg_iv > 0

    # 5. straddle compression: ATM straddle very cheap relative to spot
    straddle_price = ce_ltp + pe_ltp
    compression = (straddle_price / spot_price) < 0.005 if spot_price > 0 else False

    # 6. gamma spread across strikes (std dev of total gammas)
    total_gamma = (option_df['ce_gamma'].fillna(0).astype(float).values + option_df['pe_gamma'].fillna(0).astype(float).values)
    gamma_spread = np.nanstd(total_gamma)

    # 7. expected move proxy
    denominator = (option_df['ce_oi'].fillna(0).abs().sum() + option_df['pe_oi'].fillna(0).abs().sum())
    expected_move = gamma_pressure * (1 + (hedge_imbalance / max(1, denominator)))

    # Build score
    score = 0
    if gamma_pressure > 60: score += 20
    if gamma_pressure > 120: score += 20
    if hedge_imbalance > 50000: score += 20 # Increased threshold for Index OI
    if iv_crush: score += 15
    if compression: score += 20
    if gamma_flip: score += 25
    # scale with gamma spread
    if gamma_spread > 0.003: score += 10

    score = min(score, 100)

    direction = 'NEUTRAL'
    # directional inference: if pe side showing more oi build (positive change), dealers shorting downside -> price likely UP
    # Fallback to OI Logic if Gamma is 0
    if ce_gamma == 0 and pe_gamma == 0:
        # Pure OI Logic
        if pe_oi_chg > ce_oi_chg:
             direction = 'UP (OI Based)'
        elif ce_oi_chg > pe_oi_chg:
             direction = 'DOWN (OI Based)'
    else:
        ce_side = (option_df['ce_gamma'].fillna(0).astype(float).values * option_df['ce_oi_change'].fillna(0).astype(float).values).sum()
        pe_side = (option_df['pe_gamma'].fillna(0).astype(float).values * option_df['pe_oi_change'].fillna(0).astype(float).values).sum()
        if pe_side > ce_side * 1.1:
            direction = 'UP'
        elif ce_side > pe_side * 1.1:
            direction = 'DOWN'

    return {
        'gamma_pressure': round(float(gamma_pressure), 2),
        'hedge_imbalance': round(float(hedge_imbalance), 2),
        'iv_crush': bool(iv_crush),
        'straddle_compression': bool(compression),
        'gamma_flip': bool(gamma_flip),
        'gamma_spread': round(float(gamma_spread), 6),
        'expiry_spike_score': int(score),
        'expected_move_points': round(float(expected_move), 2),
        'direction': direction,
        'atm_strike': int(atm['strike'])
    }


# ---------------------------
# Main app flow
# ---------------------------

st.title("📣 Expiry Day Spike Detector — Dhan API v2")
st.markdown("Detect expiry-day sudden moves. **Note:** Dhan API does not return Greeks (Gamma/IV) natively. Signals rely heavily on Price/OI/Compression.")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Expiry Spike Scan")
    st.write(f"Symbol: **{symbol_selection}** | Expiry: **{expiry or 'Not set'}**")

with col2:
    st.header("Controls")
    if st.button("Fetch Now"):
        st.session_state['fetch_now'] = True

# Live loop control
if 'fetch_now' not in st.session_state:
    st.session_state['fetch_now'] = False

# simple auto refresh logic
if run_live:
    st.info(f"Auto-refresh enabled — every {refresh_secs}s")
    try:
        st_autorefresh = st.experimental_memo(lambda: None)  # placeholder to silence lint
    except Exception:
        pass

# Display area
result_area = st.empty()
raw_area = st.expander("Raw option chain (collapsed)")

# Minimal spot fetch: attempt to get spot from option chain or external
def fetch_spot_from_chain(df, raw_data):
    # Try fetching from raw data 'last_price' field if available at root
    if raw_data and 'last_price' in raw_data:
        return float(raw_data['last_price'])
    
    # Fallback: find mid of ATM strikes
    if df is None or df.empty:
        return None
    strikes = df['strike'].values
    return float((strikes.min() + strikes.max()) / 2)


# Single-run fetch+compute function
def run_detection_once():
    raw = fetch_option_chain_dhan(symbol_selection, expiry)
    if raw is None:
        return None, None
    df = normalize_option_chain(raw)
    if df is None or df.empty:
        st.error("Could not normalize option chain from Dhan response. Check structure or API.")
        return None, raw

    # best-effort spot
    spot = fetch_spot_from_chain(df, raw)
    if spot is None:
        # fallback: ask user
        st.warning("Could not infer spot from option chain. Provide spot price in sidebar.")
        return None, df

    gamma = gamma_sequence_expiry(df, spot)
    return gamma, df


# Run once or in loop
should_run = st.session_state.get('fetch_now', False) or not expiry == '' and not run_live

if should_run:
    gamma, df = run_detection_once()
    if df is not None and show_raw:
        raw_area.dataframe(df)
    if gamma is None:
        st.warning("No data — check API or expiry/symbol settings.")
    else:
        score = gamma['expiry_spike_score']
        colA, colB, colC = st.columns(3)
        colA.metric("Expiry Spike Score", f"{score}%")
        colB.metric("Direction", gamma['direction'])
        colC.metric("Expected Move (pts)", gamma['expected_move_points'])
        
        st.markdown(f"**ATM Strike:** {gamma['atm_strike']}")

        st.markdown("**Details**")
        st.write({
            'Gamma Pressure': gamma['gamma_pressure'],
            'Hedge Imbalance': gamma['hedge_imbalance'],
            'Gamma Spread': gamma['gamma_spread'],
            'IV Crush': gamma['iv_crush'],
            'Straddle Compression': gamma['straddle_compression'],
            'Gamma Flip': gamma['gamma_flip']
        })

        # Alert
        if score >= threshold_alert:
            st.warning(f"🚨 High expiry spike probability: {score}% — Direction: {gamma['direction']}")
            # send telegram optionally
            send_telegram_message(f"Expiry Spike Alert for {symbol_selection} ({expiry}) — Score {score}% — Dir: {gamma['direction']} — Expected Move: {gamma['expected_move_points']} pts")

# If auto-run enabled, simple loop with sleep (limited iterations to avoid runaway)
if run_live and expiry:
    max_cycles = 1000
    cycle = 0
    container = st.empty()
    while cycle < max_cycles:
        gamma, df = run_detection_once()
        if df is not None and show_raw:
            raw_area.dataframe(df)
        if gamma:
            container.empty()
            with container.container():
                colA, colB, colC = st.columns(3)
                colA.metric("Expiry Spike Score", f"{gamma['expiry_spike_score']}%")
                colB.metric("Direction", gamma['direction'])
                colC.metric("Expected Move", gamma['expected_move_points'])
                
                st.write(pd.DataFrame([{
                    'Direction': gamma['direction'],
                    'Expected Move': gamma['expected_move_points'],
                    'Gamma Pressure': gamma['gamma_pressure'],
                    'Hedge Imbalance': gamma['hedge_imbalance'],
                    'Gamma Flip': gamma['gamma_flip']
                }]))
            
            if gamma['expiry_spike_score'] >= threshold_alert:
                st.toast(f"🚨 High Spike Prob: {gamma['expiry_spike_score']}%")
                send_telegram_message(f"Alert {symbol_selection}: Score {gamma['expiry_spike_score']}% — {gamma['direction']}")
        
        time.sleep(refresh_secs)
        cycle += 1
        st.rerun()

# Footer notes
st.markdown("---")
st.caption("App uses Dhan API v2 option chain structure.")
