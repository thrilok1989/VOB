import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, time as dtime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

st.set_page_config(page_title="Expiry Spike Detector", layout="wide")

# ---------------------------
# Helper: Read secrets
# ---------------------------
CLIENT_ID = st.secrets.get("dhanauth", {}).get("CLIENT_ID")
ACCESS_TOKEN = st.secrets.get("dhanauth", {}).get("ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = st.secrets.get("telegram", {}).get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = st.secrets.get("telegram", {}).get("TELEGRAM_CHAT_ID")

# ---------------------------
# IST TIMEZONE SETUP
# ---------------------------
ist = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(ist)

def get_ist_time_string():
    """Get current IST time as string"""
    return get_ist_time().strftime("%Y-%m-%d %H:%M:%S")

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
expiry = st.sidebar.text_input("Expiry Date (YYYY-MM-DD)", value=get_ist_time().strftime("%Y-%m-%d"))
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=30, max_value=300, value=60)
run_live = st.sidebar.checkbox("🚀 AUTO-RUN & AUTO-REFRESH", value=True)
show_raw = st.sidebar.checkbox("Show raw option chain", value=False)
show_charts = st.sidebar.checkbox("Show analysis charts", value=True)
threshold_alert = st.sidebar.slider("Alert threshold (score ≥ )", 10, 100, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("**Streamlit Secrets needed:**")
st.sidebar.markdown("`dhanauth.CLIENT_ID`, `dhanauth.ACCESS_TOKEN`")

# ---------------------------
# Initialize Session State
# ---------------------------
if 'last_alert_score' not in st.session_state:
    st.session_state.last_alert_score = 0
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'fetch_now' not in st.session_state:
    st.session_state.fetch_now = True  # Auto-start
if 'last_api_response' not in st.session_state:
    st.session_state.last_api_response = None
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = None
if 'consecutive_signals' not in st.session_state:
    st.session_state.consecutive_signals = 0

# ---------------------------
# GAMMA SEQUENCE ENGINE - FIXED
# ---------------------------

def gamma_sequence_engine(option_chain):
    """
    Gamma Sequence - Volatility Shock Detection Model
    Detects when market is about to explode with sudden moves
    """
    # FIX: Proper DataFrame empty check
    if option_chain is None or (hasattr(option_chain, 'empty') and option_chain.empty):
        return None
    
    try:
        # Convert to list if DataFrame
        if isinstance(option_chain, pd.DataFrame):
            chain_list = option_chain.to_dict('records')
        else:
            chain_list = option_chain
            
        # FIX: Check if we have valid data
        if not chain_list or len(chain_list) == 0:
            return None
            
        strikes = np.array([x.get("strike", 0) for x in chain_list])
        avg_strike = np.mean(strikes)
        atm_index = np.argmin(abs(strikes - avg_strike))
        
        # FIX: Check if index is valid
        if atm_index >= len(chain_list):
            return None
            
        atm = chain_list[atm_index]
        
        # Extract gamma values with safe defaults
        ce_gammas = np.array([x.get("ce_gamma", 0) for x in chain_list])
        pe_gammas = np.array([x.get("pe_gamma", 0) for x in chain_list])
        
        # 1. Gamma Pressure (Market Maker Hedging Intensity)
        gamma_pressure = (ce_gammas[atm_index] + pe_gammas[atm_index]) * 10000
        
        # 2. Gamma Spread (Volatility Expansion/Compression)
        gamma_spread = np.std(ce_gammas + pe_gammas)
        
        # 3. Directional Gamma (Dealer Hedging Direction)
        ce_side = np.sum([x.get("ce_gamma", 0) * x.get("ce_oi_change", 0) for x in chain_list])
        pe_side = np.sum([x.get("pe_gamma", 0) * x.get("pe_oi_change", 0) for x in chain_list])
        
        if ce_side > pe_side:
            direction = "down"  # Dealers hedging up → price pushed down
        elif pe_side > ce_side:
            direction = "up"
        else:
            direction = "neutral"
        
        # 4. Expected Move (Points)
        expected_move = (gamma_spread * 100)
        
        # 5. Gamma Flip (Explosive Move Signal)
        total_gamma_flow = abs(ce_side) + abs(pe_side)
        gamma_flip = total_gamma_flow > 0 and abs(ce_side - pe_side) / total_gamma_flow < 0.1
        
        # 6. Volatility Warning
        if gamma_spread > 0.003:
            volatility_warning = "high"
        elif gamma_spread > 0.0015:
            volatility_warning = "medium"
        else:
            volatility_warning = "low"
        
        # Calculate Gamma Sequence Score (0-100)
        gamma_score = min(gamma_pressure * 2, 40)  # Base pressure
        gamma_score += 20 if volatility_warning == "high" else 10 if volatility_warning == "medium" else 0
        gamma_score += 25 if gamma_flip else 0
        gamma_score += 15 if direction != "neutral" else 0
        
        return {
            "gamma_pressure": round(min(gamma_pressure, 100), 2),
            "gamma_score": min(int(gamma_score), 100),
            "expected_move": round(expected_move, 2),
            "volatility_warning": volatility_warning,
            "direction": direction,
            "gamma_flip": gamma_flip,
            "gamma_spread": round(gamma_spread, 4),
            "ce_gamma_flow": round(ce_side, 2),
            "pe_gamma_flow": round(pe_side, 2)
        }
        
    except Exception as e:
        st.error(f"Gamma Sequence Error: {e}")
        return None

# ---------------------------
# REVERSAL PROBABILITY ENGINE - FIXED
# ---------------------------

def reversal_probability_engine(option_chain, spot_price):
    """
    Reversal Probability Zone Detection
    Identifies potential trend reversal points
    """
    # FIX: Proper DataFrame empty check
    if option_chain is None or (hasattr(option_chain, 'empty') and option_chain.empty):
        return None
        
    try:
        # Find key strikes
        strikes = option_chain['strike'].values
        atm_idx = int(np.argmin(np.abs(strikes - spot_price)))
        
        # FIX: Check if index is valid
        if atm_idx >= len(option_chain):
            return None
        
        # Analyze OI concentrations
        max_ce_oi = option_chain['ce_oi'].max()
        max_pe_oi = option_chain['pe_oi'].max()
        
        ce_oi_strike = option_chain.loc[option_chain['ce_oi'] == max_ce_oi, 'strike'].iloc[0]
        pe_oi_strike = option_chain.loc[option_chain['pe_oi'] == max_pe_oi, 'strike'].iloc[0]
        
        # Calculate reversal probability
        call_wall_strength = max_ce_oi / option_chain['ce_oi'].sum() if option_chain['ce_oi'].sum() > 0 else 0
        put_wall_strength = max_pe_oi / option_chain['pe_oi'].sum() if option_chain['pe_oi'].sum() > 0 else 0
        
        reversal_score = int((call_wall_strength + put_wall_strength) * 50)
        bias = "BULLISH" if pe_oi_strike > ce_oi_strike else "BEARISH" if ce_oi_strike > pe_oi_strike else "NEUTRAL"
        
        return {
            "reversal_score": reversal_score,
            "bias": bias,
            "call_wall": int(ce_oi_strike),
            "put_wall": int(pe_oi_strike),
            "call_wall_strength": round(call_wall_strength * 100, 1),
            "put_wall_strength": round(put_wall_strength * 100, 1)
        }
        
    except Exception as e:
        st.error(f"Reversal Probability Error: {e}")
        return None

# ---------------------------
# MASTER BIAS ENGINE
# ---------------------------

def master_bias_engine(gamma_data, reversal_data, expiry_data):
    """
    Combined Master Bias Engine
    Integrates Gamma Sequence + Reversal Probability + Expiry Spike Detection
    """
    technical_bias = 0
    volatility_bias = 0
    reversal_bias = 0
    directional_bias = 0
    
    # Gamma Sequence Contributions
    if gamma_data:
        if gamma_data["direction"] == "up":
            technical_bias += 2
            directional_bias += 1
        elif gamma_data["direction"] == "down":
            technical_bias -= 2
            directional_bias -= 1
            
        if gamma_data["volatility_warning"] == "high":
            volatility_bias += 2
            technical_bias += 3
        elif gamma_data["volatility_warning"] == "medium":
            volatility_bias += 1
            technical_bias += 1
            
        if gamma_data["gamma_flip"]:
            reversal_bias += 3
            technical_bias += 4
            
        technical_bias += gamma_data["gamma_score"] * 0.3
    
    # Reversal Probability Contributions
    if reversal_data:
        if reversal_data["bias"] == "BULLISH":
            technical_bias += 2
            directional_bias += 1
        elif reversal_data["bias"] == "BEARISH":
            technical_bias -= 2
            directional_bias -= 1
            
        technical_bias += reversal_data["reversal_score"] * 0.2
    
    # Expiry Spike Contributions
    if expiry_data:
        technical_bias += expiry_data["expiry_spike_score"] * 0.4
        
        if expiry_data["direction"] == "UP":
            directional_bias += 1
        elif expiry_data["direction"] == "DOWN":
            directional_bias -= 1
    
    # Final Bias Calculation
    total_bias = min(max(technical_bias, -10), 10)
    
    # Determine overall direction
    if total_bias > 3:
        overall_direction = "STRONG BULLISH"
        direction_emoji = "📈🔥"
    elif total_bias > 1:
        overall_direction = "BULLISH"
        direction_emoji = "📈"
    elif total_bias < -3:
        overall_direction = "STRONG BEARISH"
        direction_emoji = "📉🔥"
    elif total_bias < -1:
        overall_direction = "BEARISH"
        direction_emoji = "📉"
    else:
        overall_direction = "NEUTRAL"
        direction_emoji = "➡️"
    
    return {
        "master_bias_score": round(total_bias, 2),
        "overall_direction": overall_direction,
        "direction_emoji": direction_emoji,
        "technical_bias": round(technical_bias, 2),
        "volatility_bias": volatility_bias,
        "reversal_bias": reversal_bias,
        "directional_bias": directional_bias,
        "components": {
            "gamma_sequence": gamma_data,
            "reversal_probability": reversal_data,
            "expiry_spike": expiry_data
        }
    }

# ---------------------------
# SMART ALERT SYSTEM
# ---------------------------

def send_telegram_message(msg: str):
    """Send alert via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def is_market_hours():
    """Check if current time is within IST market hours"""
    now_ist = get_ist_time().time()
    market_open = dtime(9, 15)  # 9:15 AM IST
    market_close = dtime(15, 30)  # 3:30 PM IST
    return market_open <= now_ist <= market_close

def is_clear_signal(master_bias, threshold):
    """Determine if we have a clear trading signal"""
    if not master_bias:
        return False
    
    score = abs(master_bias['master_bias_score'])
    direction = master_bias['overall_direction']
    
    # Clear signal conditions
    conditions = [
        score >= threshold,
        "NEUTRAL" not in direction,
        master_bias['components']['gamma_sequence'] and master_bias['components']['gamma_sequence'].get('gamma_score', 0) >= 60,
        master_bias['components']['expiry_spike'] and master_bias['components']['expiry_spike'].get('expiry_spike_score', 0) >= 50
    ]
    
    return all(conditions)

def check_and_send_alert(master_bias, threshold, symbol):
    """Smart alert system with rate limiting and confirmation"""
    if not master_bias:
        return False
    
    current_time = get_ist_time()
    
    # Rate limiting: Don't send alerts more than once every 5 minutes
    if st.session_state.last_alert_time:
        time_diff = (current_time - st.session_state.last_alert_time).total_seconds() / 60
        if time_diff < 5:
            return False
    
    # Check for clear signal
    if is_clear_signal(master_bias, threshold):
        st.session_state.consecutive_signals += 1
    else:
        st.session_state.consecutive_signals = 0
        return False
    
    # Require 2 consecutive clear signals to avoid false alarms
    if st.session_state.consecutive_signals >= 2:
        score = master_bias['master_bias_score']
        direction = master_bias['overall_direction']
        emoji = master_bias['direction_emoji']
        
        gamma = master_bias['components']['gamma_sequence'] or {}
        reversal = master_bias['components']['reversal_probability'] or {}
        expiry = master_bias['components']['expiry_spike'] or {}
        
        message = f"""🚨 **CLEAR TRADING SIGNAL DETECTED** 🚨

📊 **Symbol:** {symbol}
🎯 **Master Bias Score:** {score}
📈 **Direction:** {emoji} {direction}
⏰ **Time:** {current_time.strftime('%H:%M:%S IST')}

🔥 **Gamma Sequence**
   - Score: {gamma.get('gamma_score', 0)}/100
   - Pressure: {gamma.get('gamma_pressure', 0)}
   - Volatility: {gamma.get('volatility_warning', 'N/A').upper()}
   - Flip: {'✅ YES' if gamma.get('gamma_flip') else '❌ NO'}

🔄 **Reversal Probability**
   - Score: {reversal.get('reversal_score', 0)}/100
   - Bias: {reversal.get('bias', 'N/A')}
   - Call Wall: {reversal.get('call_wall', 'N/A')}
   - Put Wall: {reversal.get('put_wall', 'N/A')}

⚡ **Expiry Spike**
   - Score: {expiry.get('expiry_spike_score', 0)}/100
   - Pressure: {expiry.get('gamma_pressure', 0)}
   - Direction: {expiry.get('direction', 'N/A')}

💡 **Expected Move:** {gamma.get('expected_move', 0)} points

**Action:** Consider {direction.split()[-1]} positions with proper risk management."""

        if send_telegram_message(message):
            st.session_state.last_alert_time = current_time
            st.session_state.last_alert_score = score
            return True
    
    return False

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def fetch_option_chain_dhan(symbol_key: str, expiry_date: str):
    """Fetch option chain using Dhan API V2"""
    if not ACCESS_TOKEN or not CLIENT_ID:
        st.error("❌ Dhan credentials missing in Streamlit secrets.")
        return None

    scrip_details = INDEX_MAP.get(symbol_key)
    if not scrip_details:
        st.error("❌ Invalid symbol selection")
        return None
    
    url = "https://api.dhan.co/v2/optionchain"
    
    headers = {
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "UnderlyingScrip": int(scrip_details["code"]),
        "UnderlyingSeg": scrip_details["segment"],
        "Expiry": expiry_date
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"❌ API Error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        return None

def normalize_option_chain(raw):
    """Normalize Dhan V2 response into DataFrame"""
    items = []
    try:
        if not raw or 'data' not in raw:
            st.error("❌ No data in API response")
            return None, 0
            
        data_block = raw['data']
        chain_data = data_block.get('oc', {})
        
        if not chain_data:
            st.warning("⚠️ No option chain data found in response")
            return None, 0
        
        # Extract spot price
        spot_price = (data_block.get('last_price') or 
                     data_block.get('spot_price') or 
                     data_block.get('underlyingValue') or 
                     0)
        
        for strike_price, details in chain_data.items():
            if not isinstance(details, dict):
                continue
                
            ce = details.get('ce', {})
            pe = details.get('pe', {})
            
            ce_greeks = ce.get('greeks', {}) if isinstance(ce, dict) else {}
            pe_greeks = pe.get('greeks', {}) if isinstance(pe, dict) else {}

            try:
                item = {
                    'strike': float(strike_price),
                    'ce_ltp': float(ce.get('last_price', 0.0)) if ce else 0.0,
                    'ce_iv': float(ce.get('implied_volatility', 0.0)) if ce else 0.0,
                    'ce_gamma': float(ce_greeks.get('gamma', 0.0)),
                    'ce_oi': float(ce.get('oi', 0.0)) if ce else 0.0,
                    'ce_oi_change': float(ce.get('oi_change', 0.0)) if ce else 0.0,
                    'ce_volume': float(ce.get('volume', 0.0)) if ce else 0.0,
                    'pe_ltp': float(pe.get('last_price', 0.0)) if pe else 0.0,
                    'pe_iv': float(pe.get('implied_volatility', 0.0)) if pe else 0.0,
                    'pe_gamma': float(pe_greeks.get('gamma', 0.0)),
                    'pe_oi': float(pe.get('oi', 0.0)) if pe else 0.0,
                    'pe_oi_change': float(pe.get('oi_change', 0.0)) if pe else 0.0,
                    'pe_volume': float(pe.get('volume', 0.0)) if pe else 0.0
                }
                items.append(item)
            except (ValueError, TypeError):
                continue
                
        if not items:
            st.error("❌ No valid option data parsed")
            return None, spot_price
            
        df = pd.DataFrame(items).sort_values('strike').reset_index(drop=True)
        return df, spot_price
        
    except Exception as e:
        st.error(f"❌ Normalization Error: {e}")
        return None, 0

def gamma_sequence_expiry(option_df: pd.DataFrame, spot_price: float):
    """Original expiry spike detection"""
    # FIX: Proper DataFrame empty check
    if option_df is None or (hasattr(option_df, 'empty') and option_df.empty):
        return None
        
    try:
        strikes = option_df['strike'].values
        atm_idx = int(np.argmin(np.abs(strikes - spot_price)))
        
        # FIX: Check if index is valid
        if atm_idx >= len(option_df):
            return None
            
        atm = option_df.iloc[atm_idx]

        ce_gamma = float(atm.get('ce_gamma', 0))
        pe_gamma = float(atm.get('pe_gamma', 0))
        ce_oi_chg = float(atm.get('ce_oi_change', 0))
        pe_oi_chg = float(atm.get('pe_oi_change', 0))
        
        # Logic Factors
        gamma_pressure = (abs(ce_gamma) + abs(pe_gamma)) * 10000
        hedge_imbalance = abs(ce_oi_chg - pe_oi_chg)
        
        gamma_flip = (ce_oi_chg < 0 and pe_oi_chg < 0)
        
        straddle_price = atm['ce_ltp'] + atm['pe_ltp']
        compression = (straddle_price / spot_price) < 0.005 if spot_price > 0 else False

        score = 0
        if gamma_pressure > 60: score += 20
        if hedge_imbalance > 50000: score += 20
        if compression: score += 25
        if gamma_flip: score += 30

        direction = 'NEUTRAL'
        if ce_oi_chg < pe_oi_chg:
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
            'direction_strength': round(abs(ce_oi_chg - pe_oi_chg) / max(1, (abs(ce_oi_chg) + abs(pe_oi_chg))) * 100, 1),
            'atm_strike': int(atm['strike']),
            'straddle_price': round(straddle_price, 2),
            'timestamp': get_ist_time()
        }
    except Exception as e:
        st.error(f"Expiry Spike Error: {e}")
        return None

# ---------------------------
# TABULATED DISPLAY FUNCTIONS
# ---------------------------

def create_master_bias_table(master_bias):
    """Create comprehensive master bias table"""
    if not master_bias:
        return None
        
    components = master_bias['components']
    
    data = {
        "Engine": [
            "🎯 MASTER BIAS ENGINE", 
            "🔥 GAMMA SEQUENCE", 
            "🔄 REVERSAL PROBABILITY", 
            "⚡ EXPIRY SPIKE"
        ],
        "Score": [
            f"{master_bias['master_bias_score']} | {master_bias['direction_emoji']}",
            f"{components['gamma_sequence'].get('gamma_score', 0) if components['gamma_sequence'] else 'N/A'}/100",
            f"{components['reversal_probability'].get('reversal_score', 0) if components['reversal_probability'] else 'N/A'}/100", 
            f"{components['expiry_spike'].get('expiry_spike_score', 0) if components['expiry_spike'] else 'N/A'}/100"
        ],
        "Direction": [
            master_bias['overall_direction'],
            components['gamma_sequence'].get('direction', 'N/A').upper() if components['gamma_sequence'] else 'N/A',
            components['reversal_probability'].get('bias', 'N/A') if components['reversal_probability'] else 'N/A',
            components['expiry_spike'].get('direction', 'N/A') if components['expiry_spike'] else 'N/A'
        ],
        "Key Metric": [
            f"Volatility: {components['gamma_sequence'].get('volatility_warning', 'N/A').upper() if components['gamma_sequence'] else 'N/A'}",
            f"Flip: {components['gamma_sequence'].get('gamma_flip', False) if components['gamma_sequence'] else 'N/A'}",
            f"Call Wall: {components['reversal_probability'].get('call_wall', 'N/A') if components['reversal_probability'] else 'N/A'}",
            f"Pressure: {components['expiry_spike'].get('gamma_pressure', 0) if components['expiry_spike'] else 'N/A'}"
        ]
    }
    
    df = pd.DataFrame(data)
    return df

# ---------------------------
# MAIN EXECUTION - AUTO RUN
# ---------------------------

st.title("🚀 Auto-Run Master Bias Engine — 1 Minute Refresh")
st.markdown("---")

# Market status with IST
market_status = "🟢 MARKET OPEN" if is_market_hours() else "🔴 MARKET CLOSED"
st.sidebar.markdown(f"**{market_status}**")

# Display current IST time
current_ist_time = get_ist_time_string()
st.sidebar.markdown(f"**Last Update:** {current_ist_time} IST")

# Auto-run status
if run_live:
    st.sidebar.success("🔄 AUTO-RUN ACTIVE - Refreshing every minute")
else:
    st.sidebar.warning("⏸️ AUTO-RUN PAUSED")

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"**Symbol:** {symbol_selection} | **Expiry:** {expiry} | **Alert Threshold:** {threshold_alert}%")

with col2:
    if st.button("🔄 Manual Refresh"):
        st.session_state.fetch_now = True

# Main analysis function
def run_comprehensive_analysis():
    """Run all analysis engines"""
    with st.spinner("🔄 Fetching live data from Dhan API..."):
        raw = fetch_option_chain_dhan(symbol_selection, expiry)
    
    if not raw: 
        return None
    
    st.session_state.last_api_response = raw
    
    df, spot_price = normalize_option_chain(raw)
    if df is None:
        return None
        
    # Run all engines
    gamma_data = gamma_sequence_engine(df)
    reversal_data = reversal_probability_engine(df, spot_price) 
    expiry_data = gamma_sequence_expiry(df, spot_price)
    
    # Master bias engine
    master_bias = master_bias_engine(gamma_data, reversal_data, expiry_data)
    
    return master_bias, df, spot_price

# Always run analysis when auto-run is enabled
if run_live or st.session_state.fetch_now:
    
    # Market hours check
    if run_live and not is_market_hours():
        st.warning("⏸️ Market is closed. Auto-refresh paused until market hours (9:15 AM - 3:30 PM IST).")
        run_live = False
    
    if run_live or st.session_state.fetch_now:
        result = run_comprehensive_analysis()
        
        if result:
            master_bias, df, spot_price = result
            
            # Store in history
            st.session_state.analysis_history.append({
                'timestamp': get_ist_time(),
                'master_score': master_bias['master_bias_score'],
                'direction': master_bias['overall_direction'],
                'gamma_score': master_bias['components']['gamma_sequence'].get('gamma_score', 0) if master_bias['components']['gamma_sequence'] else 0
            })
            st.session_state.analysis_history = st.session_state.analysis_history[-50:]
            
            # Display MASTER BIAS
            st.subheader("🎯 LIVE MASTER BIAS DASHBOARD")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Master Score with color coding
            master_score = master_bias['master_bias_score']
            score_color = "red" if abs(master_score) >= 7 else "orange" if abs(master_score) >= 4 else "green"
            
            col1.metric(
                "Master Bias Score", 
                f"{master_score}",
                delta=f"🚨 {master_bias['overall_direction']}" if abs(master_score) >= 7 else f"⚠️ {master_bias['overall_direction']}" if abs(master_score) >= 4 else f"✅ {master_bias['overall_direction']}"
            )
            
            col2.metric("Direction", f"{master_bias['direction_emoji']} {master_bias['overall_direction']}")
            col3.metric("Gamma Score", f"{master_bias['components']['gamma_sequence'].get('gamma_score', 0) if master_bias['components']['gamma_sequence'] else 0}/100")
            col4.metric("Spot Price", f"{spot_price:.2f}")
            
            # Alert Status
            if check_and_send_alert(master_bias, threshold_alert, symbol_selection):
                st.success("📢 CLEAR SIGNAL DETECTED - Telegram Alert Sent!")
            else:
                if st.session_state.consecutive_signals > 0:
                    st.info(f"🔍 Signal Building: {st.session_state.consecutive_signals}/2 confirmations")
                else:
                    st.info("🔍 Monitoring for clear signals...")
            
            # Tabulated Results
            st.subheader("📊 ENGINE BREAKDOWN")
            master_table = create_master_bias_table(master_bias)
            if master_table is not None:
                st.dataframe(master_table, use_container_width=True, hide_index=True)
            
            # Raw data
            if show_raw and df is not None:
                with st.expander("📋 RAW OPTION CHAIN DATA"):
                    st.dataframe(df.style.format({
                        'strike': '{:.0f}',
                        'ce_ltp': '{:.2f}',
                        'pe_ltp': '{:.2f}',
                        'ce_oi': '{:.0f}',
                        'pe_oi': '{:.0f}',
                        'ce_oi_change': '{:.0f}',
                        'pe_oi_change': '{:.0f}'
                    }), use_container_width=True)

            # Auto-refresh handling
            if run_live:
                refresh_container = st.empty()
                for i in range(refresh_secs, 0, -1):
                    with refresh_container:
                        st.info(f"🔄 Next auto-refresh in {i} seconds...")
                    time.sleep(1)
                st.rerun()
                
        else:
            st.error("❌ Failed to fetch or analyze data")
            if run_live:
                time.sleep(refresh_secs)
                st.rerun()

else:
    # Initial state
    st.info("👆 Enable 'AUTO-RUN & AUTO-REFRESH' to start continuous monitoring")

# Footer
st.markdown("---")
st.markdown("""
**🚀 Auto-Run Features:**
- 📊 Continuous monitoring every 1 minute
- 📡 Smart Telegram alerts for clear signals  
- 🔍 2-step signal confirmation to avoid false alarms
- ⏰ IST Market hours detection (9:15 AM - 3:30 PM)
- 🎯 Multi-engine bias analysis
- 🕐 All times in IST (India Standard Time)
""")