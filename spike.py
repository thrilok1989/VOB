import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, time as dtime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
show_charts = st.sidebar.checkbox("Show analysis charts", value=True)
threshold_alert = st.sidebar.slider("Alert threshold (score >= )", 10, 100, 70)

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
    st.session_state.fetch_now = False
if 'last_api_response' not in st.session_state:
    st.session_state.last_api_response = None

# ---------------------------
# Helper functions
# ---------------------------

def send_telegram_message(msg: str):
    """Send alert via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.sidebar.warning("Telegram credentials not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.sidebar.error(f"Telegram send failed: {e}")
        return False

def is_market_hours():
    """Check if current time is within market hours"""
    now = datetime.now().time()
    return (dtime(9, 15) <= now <= dtime(15, 30))

def fetch_option_chain_dhan(symbol_key: str, expiry_date: str):
    """
    Fetch option chain using the CORRECT Dhan API V2 Endpoint.
    """
    if not ACCESS_TOKEN or not CLIENT_ID:
        st.error("❌ Dhan credentials missing in Streamlit secrets.")
        return None

    scrip_details = INDEX_MAP.get(symbol_key)
    if not scrip_details:
        st.error("❌ Invalid symbol selection")
        return None
    
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
        with st.spinner("🔍 Fetching option chain from Dhan API..."):
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            
        if r.status_code == 200:
            st.sidebar.success("✅ API call successful")
            return r.json()
        else:
            st.error(f"❌ API Error {r.status_code}: {r.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏰ API request timeout")
        return None
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        return None

def normalize_option_chain(raw):
    """
    Normalize Dhan V2 response into DataFrame with enhanced error handling.
    """
    items = []
    try:
        if not raw or 'data' not in raw:
            st.error("❌ No data in API response")
            return None, 0
            
        data_block = raw['data']
        
        # Debug: Show API structure
        if show_raw:
            with st.expander("🔧 API Response Structure Debug"):
                st.write("Top-level keys:", list(raw.keys()))
                st.write("Data keys:", list(data_block.keys()))
        
        # Dhan V2 often returns the chain under the key 'oc' (lowercase)
        chain_data = data_block.get('oc', {})
        
        if not chain_data:
            st.warning("⚠️ No option chain data found in response")
            return None, 0
        
        # Extract spot price with multiple fallbacks
        spot_price = (data_block.get('last_price') or 
                     data_block.get('spot_price') or 
                     data_block.get('underlyingValue') or 
                     0)
        
        for strike_price, details in chain_data.items():
            if not isinstance(details, dict):
                continue
                
            ce = details.get('ce', {})
            pe = details.get('pe', {})
            
            # Extract Greeks if available
            ce_greeks = ce.get('greeks', {}) if isinstance(ce, dict) else {}
            pe_greeks = pe.get('greeks', {}) if isinstance(pe, dict) else {}

            try:
                item = {
                    'strike': float(strike_price),
                    
                    # Call Data
                    'ce_ltp': float(ce.get('last_price', 0.0)) if ce else 0.0,
                    'ce_iv': float(ce.get('implied_volatility', 0.0)) if ce else 0.0,
                    'ce_gamma': float(ce_greeks.get('gamma', 0.0)),
                    'ce_oi': float(ce.get('oi', 0.0)) if ce else 0.0,
                    'ce_oi_change': float(ce.get('oi_change', 0.0)) if ce else 0.0,
                    'ce_volume': float(ce.get('volume', 0.0)) if ce else 0.0,
                    
                    # Put Data
                    'pe_ltp': float(pe.get('last_price', 0.0)) if pe else 0.0,
                    'pe_iv': float(pe.get('implied_volatility', 0.0)) if pe else 0.0,
                    'pe_gamma': float(pe_greeks.get('gamma', 0.0)),
                    'pe_oi': float(pe.get('oi', 0.0)) if pe else 0.0,
                    'pe_oi_change': float(pe.get('oi_change', 0.0)) if pe else 0.0,
                    'pe_volume': float(pe.get('volume', 0.0)) if pe else 0.0
                }
                items.append(item)
            except (ValueError, TypeError) as e:
                continue  # Skip invalid entries
                
        if not items:
            st.error("❌ No valid option data parsed")
            return None, spot_price
            
        df = pd.DataFrame(items).sort_values('strike').reset_index(drop=True)
        return df, spot_price
        
    except Exception as e:
        st.error(f"❌ Normalization Error: {e}")
        return None, 0

# ---------------------------
# Gamma + Expiry Spike Logic
# ---------------------------

def gamma_sequence_expiry(option_df: pd.DataFrame, spot_price: float):
    """Enhanced gamma analysis with better scoring"""
    if option_df is None or option_df.empty:
        return None
        
    strikes = option_df['strike'].values
    
    # Find ATM strike
    atm_idx = int(np.argmin(np.abs(strikes - spot_price)))
    if atm_idx >= len(option_df):
        return None
        
    atm = option_df.iloc[atm_idx]

    ce_gamma = float(atm.get('ce_gamma', 0))
    pe_gamma = float(atm.get('pe_gamma', 0))
    ce_oi_chg = float(atm.get('ce_oi_change', 0))
    pe_oi_chg = float(atm.get('pe_oi_change', 0))
    
    # Logic Factors with dynamic scoring
    gamma_pressure = (abs(ce_gamma) + abs(pe_gamma)) * 10000
    hedge_imbalance = abs(ce_oi_chg - pe_oi_chg)
    
    # Gamma Flip: Dealers shorting (OI drop) on both sides -> Explosive
    gamma_flip = (ce_oi_chg < 0 and pe_oi_chg < 0)
    
    # Straddle Compression (Is ATM cheap?)
    straddle_price = atm['ce_ltp'] + atm['pe_ltp']
    compression = (straddle_price / spot_price) < 0.005 if spot_price > 0 else False

    # Dynamic scoring
    score = 0
    if gamma_pressure > 40: score += min(30, (gamma_pressure - 40) / 2)
    if hedge_imbalance > 25000: score += min(25, hedge_imbalance / 10000)
    if compression: score += 25
    if gamma_flip: score += 30

    # Direction Inference with confidence
    direction = 'NEUTRAL'
    direction_strength = abs(ce_oi_chg - pe_oi_chg) / max(1, (abs(ce_oi_chg) + abs(pe_oi_chg)))
    
    if ce_oi_chg < pe_oi_chg and direction_strength > 0.1:  # Puts being written -> Bullish
        direction = 'UP'
    elif ce_oi_chg > pe_oi_chg and direction_strength > 0.1:  # Calls being written -> Bearish
        direction = 'DOWN'

    return {
        'gamma_pressure': round(gamma_pressure, 2),
        'hedge_imbalance': round(hedge_imbalance, 2),
        'straddle_compression': compression,
        'gamma_flip': gamma_flip,
        'expiry_spike_score': min(int(score), 100),
        'direction': direction,
        'direction_strength': round(direction_strength * 100, 1),
        'atm_strike': int(atm['strike']),
        'straddle_price': round(straddle_price, 2),
        'timestamp': datetime.now()
    }

def check_and_alert(gamma_data, threshold, symbol):
    """Enhanced alert system with rate limiting"""
    if not gamma_data:
        return
        
    score = gamma_data['expiry_spike_score']
    direction = gamma_data['direction']
    
    # Only alert if crossing threshold upward (avoid repeated alerts)
    if (score >= threshold and st.session_state.last_alert_score < threshold):
        message = f"""🚨 EXPIRY SPIKE ALERT 🚨
Symbol: {symbol}
Score: {score}% 
Direction: {direction}
ATM Strike: {gamma_data['atm_strike']}
Time: {datetime.now().strftime('%H:%M:%S')}

Gamma Pressure: {gamma_data['gamma_pressure']}
Hedge Imbalance: {gamma_data['hedge_imbalance']:,.0f}
Gamma Flip: {gamma_data['gamma_flip']}
Compression: {gamma_data['straddle_compression']}"""
        
        if send_telegram_message(message):
            st.sidebar.success("📢 Alert sent to Telegram!")
        else:
            st.sidebar.warning("⚠️ Alert creation failed")
            
    st.session_state.last_alert_score = score

def create_analysis_charts(gamma_data, df, spot_price):
    """Create visualization charts for analysis"""
    if df is None or gamma_data is None:
        return None
        
    # Chart 1: Gamma Distribution
    fig_gamma = make_subplots(
        subplot_titles=['Gamma Distribution Around ATM', 'OI Change Pattern'],
        rows=1, cols=2
    )
    
    # Filter strikes around ATM
    atm_strike = gamma_data['atm_strike']
    strike_range = df[(df['strike'] >= atm_strike - 500) & (df['strike'] <= atm_strike + 500)]
    
    # Gamma plot
    fig_gamma.add_trace(
        go.Scatter(x=strike_range['strike'], y=strike_range['ce_gamma'], 
                  name='CE Gamma', line=dict(color='green')),
        row=1, col=1
    )
    fig_gamma.add_trace(
        go.Scatter(x=strike_range['strike'], y=strike_range['pe_gamma'], 
                  name='PE Gamma', line=dict(color='red')),
        row=1, col=1
    )
    fig_gamma.add_vline(x=atm_strike, line_dash="dash", line_color="yellow")
    
    # OI Change plot
    fig_gamma.add_trace(
        go.Bar(x=strike_range['strike'], y=strike_range['ce_oi_change'], 
               name='CE OI Change', marker_color='lightgreen'),
        row=1, col=2
    )
    fig_gamma.add_trace(
        go.Bar(x=strike_range['strike'], y=strike_range['pe_oi_change'], 
               name='PE OI Change', marker_color='lightcoral'),
        row=1, col=2
    )
    fig_gamma.add_vline(x=atm_strike, line_dash="dash", line_color="yellow", row=1, col=2)
    
    fig_gamma.update_layout(height=400, showlegend=True)
    
    return fig_gamma

def create_score_trend_chart():
    """Create trend chart of spike scores over time"""
    if len(st.session_state.analysis_history) < 2:
        return None
        
    history_df = pd.DataFrame(st.session_state.analysis_history)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'], 
        y=history_df['score'],
        mode='lines+markers',
        name='Spike Score',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(y=threshold_alert, line_dash="dash", line_color="red", 
                  annotation_text=f"Alert Threshold ({threshold_alert}%)")
    
    fig.update_layout(
        title="Spike Score Trend Over Time",
        xaxis_title="Time",
        yaxis_title="Spike Score (%)",
        height=300
    )
    
    return fig

# ---------------------------
# Main Execution
# ---------------------------

st.title("📣 Expiry Day Spike Detector — Dhan API v2")
st.markdown("---")

# Market status
market_status = "🟢 MARKET OPEN" if is_market_hours() else "🔴 MARKET CLOSED"
st.sidebar.markdown(f"**{market_status}**")

col1, col2 = st.columns([2, 1])
with col1:
    st.write(f"**Symbol:** {symbol_selection} | **Expiry:** {expiry} | **Alert Threshold:** {threshold_alert}%")

with col2:
    if st.button("🔄 Fetch Now", type="primary"):
        st.session_state.fetch_now = True

# Main analysis function
def run_analysis():
    """Main analysis pipeline"""
    raw = fetch_option_chain_dhan(symbol_selection, expiry)
    if not raw: 
        return None
    
    st.session_state.last_api_response = raw
    
    df, spot_price = normalize_option_chain(raw)
    if df is None:
        return None
        
    gamma = gamma_sequence_expiry(df, spot_price)
    return gamma, df, spot_price

# Trigger analysis
if st.session_state.fetch_now or run_live:
    
    # Market hours check for auto-refresh
    if run_live and not is_market_hours():
        st.warning("⏸️ Market is closed. Auto-refresh paused.")
        run_live = False
    
    result = run_analysis()
    
    if result:
        gamma, df, spot_price = result
        
        # Store in history
        st.session_state.analysis_history.append({
            'timestamp': datetime.now(),
            'score': gamma['expiry_spike_score'],
            'direction': gamma['direction'],
            'gamma_pressure': gamma['gamma_pressure']
        })
        # Keep only last 50 entries
        st.session_state.analysis_history = st.session_state.analysis_history[-50:]
        
        # Display main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Score with color coding
        score_color = "red" if gamma['expiry_spike_score'] >= threshold_alert else "orange" if gamma['expiry_spike_score'] >= 50 else "green"
        col1.metric(
            "Spike Score", 
            f"{gamma['expiry_spike_score']}%",
            delta="🚨 HIGH" if gamma['expiry_spike_score'] >= threshold_alert else "⚠️ MEDIUM" if gamma['expiry_spike_score'] >= 50 else "✅ LOW"
        )
        
        # Direction with strength
        direction_icon = "📈" if gamma['direction'] == 'UP' else "📉" if gamma['direction'] == 'DOWN' else "➡️"
        col2.metric("Direction", f"{direction_icon} {gamma['direction']}")
        
        col3.metric("ATM Strike", gamma['atm_strike'])
        col4.metric("Spot Price", f"{spot_price:.2f}")
        
        # Visual progress bar
        st.progress(gamma['expiry_spike_score'] / 100, text=f"Spike Probability: {gamma['expiry_spike_score']}%")
        
        # Check and send alerts
        check_and_alert(gamma, threshold_alert, symbol_selection)
        
        # Detailed metrics expander
        with st.expander("📊 Detailed Analysis Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Gamma Pressure", f"{gamma['gamma_pressure']:.1f}")
                st.metric("Straddle Compression", "✅ YES" if gamma['straddle_compression'] else "❌ NO")
                
            with col2:
                st.metric("Hedge Imbalance", f"{gamma['hedge_imbalance']:,.0f}")
                st.metric("Gamma Flip", "✅ YES" if gamma['gamma_flip'] else "❌ NO")
            
            st.json(gamma)
        
        # Charts
        if show_charts:
            st.subheader("📈 Analysis Charts")
            
            fig_gamma = create_analysis_charts(gamma, df, spot_price)
            if fig_gamma:
                st.plotly_chart(fig_gamma, use_container_width=True)
            
            fig_trend = create_score_trend_chart()
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Raw data
        if show_raw and df is not None:
            with st.expander("📋 Raw Option Chain Data"):
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
            with st.empty():
                for i in range(refresh_secs, 0, -1):
                    st.write(f"🔄 Next refresh in {i} seconds...")
                    time.sleep(1)
            st.rerun()
            
    else:
        st.error("❌ Failed to fetch or analyze data")

else:
    # Initial state
    st.info("👆 Click 'Fetch Now' or enable 'Auto-refresh' to start analysis")
    
    # Show sample of what the tool does
    st.markdown("""
    ### What this tool detects:
    - **Gamma Pressure**: Market maker hedging activity
    - **Hedge Imbalance**: Uneven positioning between calls and puts  
    - **Straddle Compression**: Cheap ATM options suggesting impending move
    - **Gamma Flip**: Dealers shorting both sides (explosive potential)
    
    ### Alert Thresholds:
    - 🟢 < 50%: Low spike probability
    - 🟡 50-69%: Medium spike probability  
    - 🔴 ≥ 70%: High spike probability
    """)

# Footer
st.markdown("---")
st.markdown("*Built for expiry day trading • Data via Dhan API • Monitor gamma dynamics for potential spikes*")