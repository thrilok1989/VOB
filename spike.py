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
DHAN_API_KEY = st.secrets.get("dhanauth", {}).get("DHAN_API_KEY")
DHAN_API_SECRET = st.secrets.get("dhanauth", {}).get("DHAN_API_SECRET")
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("TELEGRAM_CHAT_ID")

# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Expiry Spike Detector — Settings")
symbol = st.sidebar.text_input("Symbol (e.g. NIFTY50 index ticker)", value="NIFTY")
expiry = st.sidebar.text_input("Expiry Date (YYYY-MM-DD)", value="")
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15)
run_live = st.sidebar.checkbox("Auto-refresh live detection", value=False)
show_raw = st.sidebar.checkbox("Show raw option chain", value=False)
threshold_alert = st.sidebar.slider("Alert threshold (score >= )", 10, 100, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("**Streamlit Secrets needed:**")
st.sidebar.markdown("`dhanauth.DHAN_API_KEY`, optional `dhanauth.DHAN_API_SECRET` and telegram credentials (optional)")

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


def fetch_option_chain_dhan(symbol: str, expiry_date: str):
    """
    Fetch option chain using Dhan API.
    NOTE: Replace endpoint and auth if your Dhan API differs. This is a placeholder structure.

    Dhan API typically requires API key headers and a specific endpoint.
    """
    if not DHAN_API_KEY:
        st.warning("Dhan API key missing in Streamlit secrets. Add dhanauth.DHAN_API_KEY")
        return None

    # Example placeholder endpoint - adjust to actual Dhan endpoints
    url = f"https://api.dhan.co/v1/option_chain/{symbol}?expiry={expiry_date}"
    headers = {"Authorization": f"Bearer {DHAN_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        # expected: data['option_chain'] or similar structure
        return data
    except Exception as e:
        st.error(f"Failed to fetch option chain: {e}")
        return None


# Utility to normalize option chain into DataFrame
def normalize_option_chain(raw):
    # Raw structure varies. Try common shapes, otherwise return None
    # Expected fields per strike: strike, ce: {ltp, iv, gamma, oi_change, oi, volume}, pe: {...}
    items = []
    try:
        chain = raw.get('option_chain') or raw.get('data') or raw
        for entry in chain:
            strike = entry.get('strike')
            ce = entry.get('CE') or entry.get('ce') or entry.get('call') or {}
            pe = entry.get('PE') or entry.get('pe') or entry.get('put') or {}
            items.append({
                'strike': strike,
                'ce_ltp': ce.get('last_price') or ce.get('ltp') or np.nan,
                'ce_iv': ce.get('iv') or np.nan,
                'ce_gamma': ce.get('gamma') or np.nan,
                'ce_oi': ce.get('oi') or np.nan,
                'ce_oi_change': ce.get('oi_change') or ce.get('change_in_oi') or 0,
                'ce_volume': ce.get('volume') or 0,
                'pe_ltp': pe.get('last_price') or pe.get('ltp') or np.nan,
                'pe_iv': pe.get('iv') or np.nan,
                'pe_gamma': pe.get('gamma') or np.nan,
                'pe_oi': pe.get('oi') or np.nan,
                'pe_oi_change': pe.get('oi_change') or pe.get('change_in_oi') or 0,
                'pe_volume': pe.get('volume') or 0
            })
        df = pd.DataFrame(items).sort_values('strike').reset_index(drop=True)
        return df
    except Exception:
        return None


# ---------------------------
# Gamma + Expiry Spike Logic
# ---------------------------

def gamma_sequence_expiry(option_df: pd.DataFrame, spot_price: float):
    # find nearest strike index (atm)
    strikes = option_df['strike'].values
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
    expected_move = gamma_pressure * (1 + (hedge_imbalance / max(1, (option_df['ce_oi'].fillna(0).abs().sum() + option_df['pe_oi'].fillna(0).abs().sum()))))

    # Build score
    score = 0
    if gamma_pressure > 60: score += 20
    if gamma_pressure > 120: score += 20
    if hedge_imbalance > 5000: score += 20
    if iv_crush: score += 15
    if compression: score += 20
    if gamma_flip: score += 25
    # scale with gamma spread
    if gamma_spread > 0.003: score += 10

    score = min(score, 100)

    direction = 'NEUTRAL'
    # directional inference: if pe side showing more oi build (positive change), dealers shorting downside -> price likely UP
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
        'atm_strike': int(option_df['strike'].iloc[ int(np.argmin(np.abs(option_df['strike'] - float(option_df['strike'].mean()))) ) ])
    }


# ---------------------------
# Main app flow
# ---------------------------

st.title("📣 Expiry Day Spike Detector — Dhan API")
st.markdown("Detect expiry-day sudden moves using Gamma Sequence + Option Chain signals. Designed for NIFTY/BANKNIFTY expiry days.")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Expiry Spike Scan")
    st.write(f"Symbol: **{symbol}** | Expiry: **{expiry or 'Not set'}**")

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

def fetch_spot_from_chain(df):
    # best effort: find mid of ATM strikes or use underlying field if present
    if df is None or df.empty:
        return None
    # attempt to guess spot from chain if present
    strikes = df['strike'].values
    return float((strikes.min() + strikes.max()) / 2)


# Single-run fetch+compute function

def run_detection_once():
    raw = fetch_option_chain_dhan(symbol, expiry)
    if raw is None:
        return None, None
    df = normalize_option_chain(raw)
    if df is None or df.empty:
        st.error("Could not normalize option chain from Dhan response. Check structure or API.")
        return None, raw

    # best-effort spot
    spot = fetch_spot_from_chain(df)
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
            send_telegram_message(f"Expiry Spike Alert for {symbol} ({expiry}) — Score {score}% — Dir: {gamma['direction']} — Expected Move: {gamma['expected_move_points']} pts")

# If auto-run enabled, simple loop with sleep (limited iterations to avoid runaway)
if run_live and expiry:
    max_cycles = 1000
    cycle = 0
    while cycle < max_cycles:
        gamma, df = run_detection_once()
        if df is not None and show_raw:
            raw_area.dataframe(df)
        if gamma:
            result_area.metric("Expiry Spike Score", f"{gamma['expiry_spike_score']}%")
            result_area.write(pd.DataFrame([{
                'Direction': gamma['direction'],
                'Expected Move': gamma['expected_move_points'],
                'Gamma Pressure': gamma['gamma_pressure'],
                'Hedge Imbalance': gamma['hedge_imbalance'],
                'Gamma Flip': gamma['gamma_flip']
            }]))
            if gamma['expiry_spike_score'] >= threshold_alert:
                st.warning(f"🚨 High expiry spike probability: {gamma['expiry_spike_score']}% — Dir: {gamma['direction']}")
                send_telegram_message(f"Expiry Spike Alert for {symbol} ({expiry}) — Score {gamma['expiry_spike_score']}% — Dir: {gamma['direction']} — Expected Move: {gamma['expected_move_points']} pts")
        time.sleep(refresh_secs)
        cycle += 1
        st.experimental_rerun()

# Footer notes
st.markdown("---")
st.caption("App uses Dhan API option chain structure — you may need to edit fetch_option_chain_dhan() to match exact Dhan response schema. Adjust thresholds to your taste.")

