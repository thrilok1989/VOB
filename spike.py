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
CLIENT_ID = st.secrets.get("dhanauth", {}).get("CLIENT_ID")
ACCESS_TOKEN = st.secrets.get("dhanauth", {}).get("ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("TELEGRAM_CHAT_ID")

# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Expiry Spike Detector — Settings")

# Correct Mapping for Dhan Option Chain API (Use IDX_I for Indices)
INDEX_MAP = {
    "NIFTY": {"code": 13, "segment": "IDX_I"},      # 13 is Nifty 50
    "BANKNIFTY": {"code": 25, "segment": "IDX_I"},  # 25 is Bank Nifty
    "FINNIFTY": {"code": 27, "segment": "IDX_I"}    # 27 is Fin Nifty
}

symbol_selection = st.sidebar.selectbox("Symbol", list(INDEX_MAP.keys()), index=0)
expiry = st.sidebar.text_input("Expiry Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15)
run_live = st.sidebar.checkbox("Auto-refresh live detection", value=False)
show_raw = st.sidebar.checkbox("Show raw option chain", value=False)
threshold_alert = st.sidebar.slider("Alert threshold (score >= )", 10, 100, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("**Streamlit Secrets needed:**")
st.sidebar.markdown("`dhanauth.CLIENT_ID`, `dhanauth.ACCESS_TOKEN`")

# ---------------------------
# Helper functions
# ---------------------------

def send_telegram_message(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass

def fetch_option_chain_dhan(symbol_key: str, expiry_date: str):
    """
    Fetch option chain using the CORRECT Dhan API V2 Endpoint.
    """
    if not ACCESS_TOKEN or not CLIENT_ID:
        st.warning("Dhan credentials missing in Streamlit secrets.")
        return None

    scrip_details = INDEX_MAP.get(symbol_key)
    
    # CORRECT ENDPOINT: No hyphen between 'option' and 'chain'
    url = "https://api.dhan.co/v2/optionchain"
    
    headers = {
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # CORRECT PAYLOAD KEYS: UnderlyingScrip (not Code) and UnderlyingSeg
    payload = {
        "UnderlyingScrip": int(scrip_details["code"]),
        "UnderlyingSeg": scrip_details["segment"],
        "Expiry": expiry_date
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        if r.status_code != 200:
            st.error(f"API Error {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

def normalize_option_chain(raw):
    """
    Normalize Dhan V2 response into DataFrame.
    Structure: {'data': {'oc': { 'strike_price': { 'ce': {...}, 'pe': {...} } }}}
    """
    items = []
    try:
        if not raw or 'data' not in raw:
            return None
            
        data_block = raw['data']
        # Dhan V2 often returns the chain under the key 'oc' (lowercase)
        chain_data = data_block.get('oc', {})
        
        for strike_price, details in chain_data.items():
            ce = details.get('ce', {})
            pe = details.get('pe', {})
            
            # Extract Greeks if available (Dhan V2 sometimes includes them in 'greeks' dict)
            ce_greeks = ce.get('greeks', {})
            pe_greeks = pe.get('greeks', {})

            items.append({
                'strike': float(strike_price),
                
                # Call Data
                'ce_ltp': float(ce.get('last_price', 0.0)),
                'ce_iv': float(ce.get('implied_volatility', 0.0)),
                'ce_gamma': float(ce_greeks.get('gamma', 0.0)),
                'ce_oi': float(ce.get('oi', 0.0)),
                'ce_oi_change': float(ce.get('oi_change', 0.0)), # Check if 'previous_oi' exists to calc change if needed
                'ce_volume': float(ce.get('volume', 0.0)),
                
                # Put Data
                'pe_ltp': float(pe.get('last_price', 0.0)),
                'pe_iv': float(pe.get('implied_volatility', 0.0)),
                'pe_gamma': float(pe_greeks.get('gamma', 0.0)),
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
    strikes = option_df['strike'].values
    if len(strikes) == 0: return None
    
    # Find ATM
    atm_idx = int(np.argmin(np.abs(strikes - spot_price)))
    atm = option_df.iloc[atm_idx]

    ce_gamma = float(atm.get('ce_gamma', 0))
    pe_gamma = float(atm.get('pe_gamma', 0))
    ce_oi_chg = float(atm.get('ce_oi_change', 0))
    pe_oi_chg = float(atm.get('pe_oi_change', 0))
    
    # Logic Factors
    gamma_pressure = (abs(ce_gamma) + abs(pe_gamma)) * 10000
    hedge_imbalance = abs(ce_oi_chg - pe_oi_chg)
    
    # Gamma Flip: Dealers shorting (OI drop) on both sides -> Explosive
    gamma_flip = (ce_oi_chg < 0 and pe_oi_chg < 0)
    
    # Straddle Compression (Is ATM cheap?)
    straddle_price = atm['ce_ltp'] + atm['pe_ltp']
    compression = (straddle_price / spot_price) < 0.005 if spot_price > 0 else False

    score = 0
    if gamma_pressure > 60: score += 20
    if hedge_imbalance > 50000: score += 20
    if compression: score += 25
    if gamma_flip: score += 30

    # Direction Inference
    direction = 'NEUTRAL'
    if ce_oi_chg < pe_oi_chg: # Puts being written/longs unwound -> Bullish
        direction = 'UP'
    elif ce_oi_chg > pe_oi_chg:
        direction = 'DOWN'

    return {
        'gamma_pressure': round(gamma_pressure, 2),
        'hedge_imbalance': round(hedge_imbalance, 2),
        'straddle_compression': compression,
        'gamma_flip': gamma_flip,
        'expiry_spike_score': min(int(score), 100),
        'direction': direction,
        'atm_strike': int(atm['strike'])
    }

# ---------------------------
# Main Execution
# ---------------------------

st.title("📣 Expiry Day Spike Detector — Dhan API v2")

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"Symbol: **{symbol_selection}** | Expiry: **{expiry}**")

with col2:
    if st.button("Fetch Now"):
        st.session_state['fetch_now'] = True

if 'fetch_now' not in st.session_state:
    st.session_state['fetch_now'] = False

# Auto-refresh logic
if run_live:
    try:
        # Use st.cache_resource or manual sleep for loop
        st_autorefresh = st.empty() 
    except: pass

def run_analysis():
    raw = fetch_option_chain_dhan(symbol_selection, expiry)
    if not raw: return None
    
    # Extract Spot Price from 'data' -> 'last_price'
    spot_price = raw.get('data', {}).get('last_price', 0)
    
    df = normalize_option_chain(raw)
    if df is None or df.empty:
        st.error("Empty Data parsed.")
        return None
        
    if spot_price == 0:
        # Fallback spot
        spot_price = (df['strike'].min() + df['strike'].max()) / 2
        
    gamma = gamma_sequence_expiry(df, spot_price)
    return gamma, df

# Trigger
if st.session_state['fetch_now'] or run_live:
    res = run_analysis()
    if res:
        gamma, df = res
        
        # Display Metrics
        colA, colB, colC = st.columns(3)
        colA.metric("Spike Score", f"{gamma['expiry_spike_score']}%")
        colB.metric("Direction", gamma['direction'])
        colC.metric("ATM Strike", gamma['atm_strike'])
        
        st.json(gamma)
        
        if show_raw:
            st.dataframe(df)

        if run_live:
            time.sleep(refresh_secs)
            st.rerun()
