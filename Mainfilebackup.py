"""
Nifty Option Screener v7.0 — 100% SELLER'S PERSPECTIVE
Comprehensive Options Analysis from Market Maker/Institutional Seller Viewpoint

FEATURES INCLUDED:
1. 🎯 Seller's Perspective Analysis
2. 🚀 Moment Detector (Momentum, Orderbook, Gamma, OI Acceleration)
3. 🧠 AI-Powered Analysis (Perplexity)
4. 📅 Expiry Spike Detector
5. 📊 Enhanced OI/PCR Analytics
6. 📱 Telegram Signal Generation
7. 💾 Supabase Data Storage

EVERYTHING interpreted from Option Seller/Market Maker viewpoint:
• CALL building = BEARISH (sellers selling calls, expecting price to stay below)
• PUT building = BULLISH (sellers selling puts, expecting price to stay above)
"""

# ============================================
# IMPORTS & CONFIGURATION
# ============================================
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
import os
import json

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
#  CONFIGURATION
# -----------------------
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300

# Moment detector weights
MOMENT_WEIGHTS = {
    "momentum_burst": 0.40,
    "orderbook_pressure": 0.20,
    "gamma_cluster": 0.25,
    "oi_acceleration": 0.15
}

TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30 IST)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30 IST)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30 IST)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30 IST)"}
}

# -----------------------
#  SECRETS & API KEYS
# -----------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    SUPABASE_TABLE_PCR = st.secrets.get("SUPABASE_TABLE_PCR", "strike_pcr_snapshots")
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
    ENABLE_AI_ANALYSIS = st.secrets.get("ENABLE_AI_ANALYSIS", "false").lower() == "true"
except Exception as e:
    st.error("❌ Missing credentials")
    st.stop()

# Initialize Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"❌ Supabase failed: {e}")
    supabase = None

DHAN_BASE_URL = "https://api.dhan.co"
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# ============================================
# 🧠 AI ANALYSIS CLASS (PERPLEXITY)
# ============================================
class TradingAI:
    """AI-powered trading analysis using Perplexity"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.enabled = bool(self.api_key) and ENABLE_AI_ANALYSIS
        
        if self.enabled:
            try:
                from perplexity import Perplexity
                if not os.environ.get("PERPLEXITY_API_KEY"):
                    os.environ["PERPLEXITY_API_KEY"] = self.api_key
                self.client = Perplexity()
                self.model = "sonar-pro"
            except Exception as e:
                st.warning(f"⚠️ AI Analysis Disabled: {e}")
                self.enabled = False
    
    def is_enabled(self):
        return self.enabled
    
    def generate_analysis(self, market_data, signal_data, moment_metrics, expiry_spike_data):
        """Generate AI analysis of current market conditions"""
        if not self.enabled:
            return None
        
        try:
            analysis_prompt = f"""
            You are an expert options trader analyzing Nifty options.
            
            Current Setup:
            Spot: ₹{market_data['spot']:,.2f}
            Seller Bias: {market_data['seller_bias']}
            Max Pain: ₹{market_data['max_pain']:,}
            PCR: {market_data['total_pcr']:.2f}
            Signal: {signal_data['position_type']} ({signal_data['signal_strength']})
            Confidence: {signal_data['confidence']:.0f}%
            Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
            
            Provide actionable insights including:
            1. Probability assessment
            2. Key risk factors
            3. Stop loss/target adjustments
            4. Expiry spike mitigation
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert options trader."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"AI Analysis Error: {e}")
            return None

trading_ai = TradingAI(PERPLEXITY_API_KEY)

# ============================================
# 🔥 UPDATED SELLER-FOCUSED EXPIRY SPIKE DETECTOR
# ============================================
def detect_expiry_spikes_seller_perspective(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str):
    """
    Detect potential expiry day spikes from 100% SELLER'S PERSPECTIVE
    """
    
    if days_to_expiry > 5:
        return {
            "active": False,
            "probability": 0,
            "message": "Expiry >5 days away, spike detection not active",
            "type": None,
            "key_levels": [],
            "score": 0
        }
    
    spike_score = 0
    spike_factors = []
    spike_type = None
    key_levels = []
    
    # SELLER'S MAX PAIN CALCULATION
    seller_max_pain = calculate_seller_max_pain(merged_df)
    
    # ============================================
    # 🔥 SELLER'S PRIMARY CONCERN: MAX PAIN GRAVITY
    # ============================================
    if seller_max_pain:
        max_pain_distance = abs(spot - seller_max_pain) / spot * 100
        
        if max_pain_distance > 2.0:
            spike_score += 25
            if spot > seller_max_pain:
                spike_factors.append(f"Spot ABOVE Max Pain (₹{seller_max_pain:,}) → CALL sellers under pressure")
                spike_type = "CALL SELLER SQUEEZE (UPWARD)"
            else:
                spike_factors.append(f"Spot BELOW Max Pain (₹{seller_max_pain:,}) → PUT sellers under pressure")
                spike_type = "PUT SELLER SQUEEZE (DOWNWARD)"
            key_levels.append(f"Max Pain: ₹{seller_max_pain:,}")
        
        elif max_pain_distance > 1.0:
            spike_score += 15
            spike_factors.append(f"Spot moderately far from Max Pain ({max_pain_distance:.1f}%)")
    
    # ============================================
    # 🔥 SELLER'S POSITION: WHO IS WRITING OPTIONS?
    # ============================================
    total_call_writing = merged_df[merged_df["Chg_OI_CE"] > 0]["Chg_OI_CE"].sum()
    total_put_writing = merged_df[merged_df["Chg_OI_PE"] > 0]["Chg_OI_PE"].sum()
    
    # Massive CALL writing (Bearish sellers)
    if total_call_writing > 2000000:
        spike_score += 20
        spike_factors.append(f"Massive CALL writing ({total_call_writing:,} contracts) = BEARISH SELLERS")
        if spot > seller_max_pain:
            spike_factors.append("CALL writers in danger if price rises → Forced buying spike")
    
    # Massive PUT writing (Bullish sellers)
    if total_put_writing > 2000000:
        spike_score += 20
        spike_factors.append(f"Massive PUT writing ({total_put_writing:,} contracts) = BULLISH SELLERS")
        if spot < seller_max_pain:
            spike_factors.append("PUT writers in danger if price falls → Forced selling spike")
    
    # ============================================
    # 🔥 SELLER'S GAMMA EXPOSURE
    # ============================================
    total_gex_net = merged_df["GEX_Net"].sum()
    
    if days_to_expiry <= 1:
        spike_score += 15
        spike_factors.append("GAMMA FLIP ZONE (Expiry day)")
        
        if total_gex_net < -1000000:
            spike_score += 10
            spike_factors.append("SELLERS NEGATIVE GAMMA → Destabilizing (explosive moves)")
            if spike_type is None:
                spike_type = "NEGATIVE GAMMA SPIKE"
        elif total_gex_net > 1000000:
            spike_score += 5
            spike_factors.append("SELLERS POSITIVE GAMMA → Stabilizing (mean reversion)")
    
    # ============================================
    # 🔥 SELLER'S UNWINDING ACTIVITY
    # ============================================
    ce_unwind_count = (merged_df["Chg_OI_CE"] < 0).sum()
    pe_unwind_count = (merged_df["Chg_OI_PE"] < 0).sum()
    total_unwind_strikes = ce_unwind_count + pe_unwind_count
    
    if total_unwind_strikes > 15:
        spike_score += 10
        spike_factors.append(f"Massive unwinding ({total_unwind_strikes} strikes)")
        
        if ce_unwind_count > pe_unwind_count:
            spike_factors.append("CALL sellers buying back → BULLISH signal")
        else:
            spike_factors.append("PUT sellers buying back → BEARISH signal")
    
    # ============================================
    # 🔥 SELLER'S ATM CONCENTRATION RISK
    # ============================================
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_from_series(merged_df["strikePrice"]))]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    total_oi_near_atm = atm_ce_oi + atm_pe_oi
    total_oi_all = merged_df["OI_CE"].sum() + merged_df["OI_PE"].sum()
    
    if total_oi_all > 0:
        atm_concentration = total_oi_near_atm / total_oi_all
        if atm_concentration > 0.5:
            spike_score += 15
            spike_factors.append(f"High ATM OI ({atm_concentration:.1%}) → Gamma squeeze risk")
    
    # ============================================
    # 🔥 SELLER'S PCR EXTREMES
    # ============================================
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        
        if pcr > 1.8:
            spike_score += 10
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) = Heavy PUT WRITING")
            if spike_type is None:
                spike_type = "PUT SELLER RISK SPIKE"
        elif pcr < 0.5:
            spike_score += 10
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) = Heavy CALL WRITING")
            if spike_type is None:
                spike_type = "CALL SELLER RISK SPIKE"
    
    # ============================================
    # 🔥 FINAL SPIKE PROBABILITY & TYPE
    # ============================================
    probability = min(100, int(spike_score * 1.5))
    
    if probability >= 70:
        intensity = "HIGH SELLER SQUEEZE RISK"
        color = "#ff0000"
    elif probability >= 50:
        intensity = "MODERATE SELLER HEDGING PRESSURE"
        color = "#ff9900"
    elif probability >= 30:
        intensity = "LOW SELLER VOLATILITY"
        color = "#ffff00"
    else:
        intensity = "SELLER POSITIONS SAFE"
        color = "#00ff00"
    
    if spike_type is None:
        spike_type = "SELLER HEDGING SPIKE"
    
    return {
        "active": days_to_expiry <= 5,
        "probability": probability,
        "score": spike_score,
        "intensity": intensity,
        "type": spike_type,
        "color": color,
        "factors": spike_factors,
        "key_levels": key_levels,
        "days_to_expiry": days_to_expiry,
        "expiry_date": expiry_date_str,
        "seller_max_pain": seller_max_pain,
        "call_writing": int(total_call_writing),
        "put_writing": int(total_put_writing),
        "seller_gamma": int(total_gex_net),
        "message": f"Expiry in {days_to_expiry:.1f} days"
    }

# ============================================
# 📊 ENHANCED OI & PCR ANALYZER
# ============================================
def analyze_oi_pcr_metrics(merged_df, spot, atm_strike):
    """Comprehensive OI and PCR analysis"""
    
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    total_oi = total_ce_oi + total_pe_oi
    
    pcr_total = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    # PCR Interpretation
    if pcr_total > 2.0:
        pcr_interpretation = "EXTREME PUT SELLING"
        pcr_sentiment = "STRONGLY BULLISH"
        pcr_color = "#00ff88"
    elif pcr_total > 1.5:
        pcr_interpretation = "HEAVY PUT SELLING"
        pcr_sentiment = "BULLISH"
        pcr_color = "#00cc66"
    elif pcr_total > 1.2:
        pcr_interpretation = "MODERATE PUT SELLING"
        pcr_sentiment = "MILD BULLISH"
        pcr_color = "#66ff66"
    elif pcr_total > 0.8:
        pcr_interpretation = "BALANCED"
        pcr_sentiment = "NEUTRAL"
        pcr_color = "#66b3ff"
    elif pcr_total > 0.5:
        pcr_interpretation = "MODERATE CALL SELLING"
        pcr_sentiment = "MILD BEARISH"
        pcr_color = "#ff9900"
    elif pcr_total > 0.3:
        pcr_interpretation = "HEAVY CALL SELLING"
        pcr_sentiment = "BEARISH"
        pcr_color = "#ff4444"
    else:
        pcr_interpretation = "EXTREME CALL SELLING"
        pcr_sentiment = "STRONGLY BEARISH"
        pcr_color = "#ff0000"
    
    # Max OI Strikes
    max_ce_oi_row = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_row = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None
    
    max_ce_strike = int(max_ce_oi_row["strikePrice"]) if max_ce_oi_row is not None else 0
    max_ce_oi_val = int(max_ce_oi_row["OI_CE"]) if max_ce_oi_row is not None else 0
    max_pe_strike = int(max_pe_oi_row["strikePrice"]) if max_pe_oi_row is not None else 0
    max_pe_oi_val = int(max_pe_oi_row["OI_PE"]) if max_pe_oi_row is not None else 0
    
    return {
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_oi": total_oi,
        "total_ce_chg": total_ce_chg,
        "total_pe_chg": total_pe_chg,
        "pcr_total": pcr_total,
        "pcr_interpretation": pcr_interpretation,
        "pcr_sentiment": pcr_sentiment,
        "pcr_color": pcr_color,
        "max_ce_strike": max_ce_strike,
        "max_ce_oi": max_ce_oi_val,
        "max_pe_strike": max_pe_strike,
        "max_pe_oi": max_pe_oi_val
    }

# ============================================
# 📱 TELEGRAM FUNCTIONS
# ============================================
def send_telegram_message(bot_token, chat_id, message):
    """Send message to Telegram"""
    try:
        if not bot_token or not chat_id:
            return False, "Telegram credentials not configured"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "Signal sent to Telegram channel!"
        else:
            return False, f"Failed to send: {response.status_code}"
    except Exception as e:
        return False, f"Telegram error: {str(e)}"

def generate_telegram_signal(entry_signal, spot, seller_bias_result, seller_max_pain, 
                           nearest_sup, nearest_res, moment_metrics, expiry_spike_data):
    """Generate Telegram signal"""
    
    if entry_signal["position_type"] == "NEUTRAL":
        return None
    
    position_type = entry_signal["position_type"]
    signal_strength = entry_signal["signal_strength"]
    confidence = entry_signal["confidence"]
    optimal_entry_price = entry_signal["optimal_entry_price"]
    stop_loss = entry_signal["stop_loss"]
    target = entry_signal["target"]
    
    signal_emoji = "🚀" if position_type == "LONG" else "🐻"
    current_time = get_ist_datetime_str()
    
    # Format stop loss and target
    stop_loss_str = f"₹{stop_loss:,.0f}" if stop_loss else "N/A"
    target_str = f"₹{target:,.0f}" if target else "N/A"
    
    # Expiry spike info
    expiry_info = ""
    if expiry_spike_data.get("active", False) and expiry_spike_data.get("probability", 0) > 50:
        spike_emoji = "🚨" if expiry_spike_data['probability'] > 70 else "⚠️"
        expiry_info = f"\n{spike_emoji} *Expiry Spike Risk*: {expiry_spike_data['probability']}% - {expiry_spike_data['type']}"
    
    message = f"""
🎯 *NIFTY OPTION TRADE SETUP*

*Position*: {signal_emoji} {position_type} ({signal_strength})
*Entry Price*: ₹{optimal_entry_price:,.0f}
*Current Spot*: ₹{spot:,.0f}

*Risk Management*:
🛑 Stop Loss: {stop_loss_str}
🎯 Target: {target_str}

*Key Levels*:
🛡️ Support: ₹{nearest_sup['strike']:,}
⚡ Resistance: ₹{nearest_res['strike']:,}
🎯 Max Pain: ₹{seller_max_pain:,}

*Seller Bias*: {seller_bias_result['bias']}
*Confidence*: {confidence:.0f}%

*Expiry Context*:
📅 Days to Expiry: {expiry_spike_data.get('days_to_expiry', 0):.1f}
{expiry_info if expiry_info else "📊 Expiry spike risk: Low"}

⏰ {current_time} IST

#NiftyOptions #OptionSelling #TradingSignal
"""
    return message

# ============================================
# 🚀 MOMENT DETECTOR FUNCTIONS
# ============================================
def _init_history():
    """Initialize session state for moment history"""
    if "moment_history" not in st.session_state:
        st.session_state["moment_history"] = []
    if "prev_ltps" not in st.session_state:
        st.session_state["prev_ltps"] = {}
    if "prev_ivs" not in st.session_state:
        st.session_state["prev_ivs"] = {}

def _snapshot_from_state(ts, spot, atm_strike, merged: pd.DataFrame):
    """Create snapshot for moment analysis"""
    total_vol = float(merged["Vol_CE"].sum() + merged["Vol_PE"].sum())
    total_iv = float(merged[["IV_CE", "IV_PE"]].mean().mean()) if not merged.empty else 0.0
    total_abs_doi = float(merged["Chg_OI_CE"].abs().sum() + merged["Chg_OI_PE"].abs().sum())
    
    per = {}
    for _, r in merged[["strikePrice", "OI_CE", "OI_PE"]].iterrows():
        per[int(r["strikePrice"])] = {"oi_ce": int(r["OI_CE"]), "oi_pe": int(r["OI_PE"])}
    
    return {
        "ts": ts,
        "spot": float(spot),
        "atm": int(atm_strike),
        "totals": {"vol": total_vol, "iv": total_iv, "abs_doi": total_abs_doi},
        "per_strike": per
    }

def compute_momentum_burst(history):
    """Compute momentum burst score"""
    if len(history) < 2:
        return {"available": False, "score": 0, "note": "Need at least 2 refresh points."}
    
    s_prev, s_now = history[-2], history[-1]
    dt = max((s_now["ts"] - s_prev["ts"]).total_seconds(), 1.0)
    
    dvol = (s_now["totals"]["vol"] - s_prev["totals"]["vol"]) / dt
    div = (s_now["totals"]["iv"] - s_prev["totals"]["iv"]) / dt
    ddoi = (s_now["totals"]["abs_doi"] - s_prev["totals"]["abs_doi"]) / dt
    
    burst_raw = abs(dvol) * abs(div) * abs(ddoi)
    score = int(100 * min(burst_raw / max(burst_raw, 1) * 2.5, 1.0))
    
    return {"available": True, "score": score, 
            "note": "Momentum burst detected" if score > 60 else "No strong burst"}

def compute_gamma_cluster(merged: pd.DataFrame, atm_strike: int, window: int = 2):
    """Compute gamma cluster score"""
    if merged.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    want = [atm_strike + i for i in range(-window, window + 1)]
    subset = merged[merged["strikePrice"].isin(want)]
    if subset.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    cluster = float((subset["Gamma_CE"].abs().fillna(0) + subset["Gamma_PE"].abs().fillna(0)).sum())
    score = int(100 * min(cluster / max(cluster, 1) * 2.0, 1.0))
    return {"available": True, "score": score, "cluster": cluster}

# ============================================
# 🎯 SELLER'S PERSPECTIVE CORE FUNCTIONS
# ============================================
def seller_strength_score(row, weights=SCORE_WEIGHTS):
    """Calculate seller strength score"""
    chg_oi = abs(safe_float(row.get("Chg_OI_CE",0))) + abs(safe_float(row.get("Chg_OI_PE",0)))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    
    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def seller_price_oi_divergence(chg_oi, vol, ltp_change, option_type="CE"):
    """Interpret seller divergence"""
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)
    
    if option_type == "CE":
        if oi_up and vol_up and price_up:
            return "Sellers WRITING calls as price rises (Bearish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING calls on weakness (Strong bearish)"
    else:
        if oi_up and vol_up and price_up:
            return "Sellers WRITING puts on strength (Bullish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING puts as price falls (Strong bullish)"
    
    return "Sellers inactive"

def calculate_seller_max_pain(df):
    """Calculate seller's max pain"""
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))
        
        ce_pain = ce_oi * max(0, ce_ltp) if strike < df["strikePrice"].mean() else 0
        pe_pain = pe_oi * max(0, pe_ltp) if strike > df["strikePrice"].mean() else 0
        
        pain = ce_pain + pe_pain
        pain_dict[strike] = pain
    
    if pain_dict:
        return min(pain_dict, key=pain_dict.get)
    return None

def calculate_seller_market_bias(merged_df, spot, atm_strike):
    """Calculate overall seller market bias"""
    polarity = 0.0
    
    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))
        
        if strike < atm_strike:
            if chg_ce > 0:
                polarity -= 2.0
            elif chg_ce < 0:
                polarity += 1.5
        
        elif strike > atm_strike:
            if chg_ce > 0:
                polarity -= 0.7
            elif chg_ce < 0:
                polarity += 0.5
        
        if strike > atm_strike:
            if chg_pe > 0:
                polarity += 2.0
            elif chg_pe < 0:
                polarity -= 1.5
        
        elif strike < atm_strike:
            if chg_pe > 0:
                polarity += 0.7
            elif chg_pe < 0:
                polarity -= 0.5
    
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:
            polarity += 1.0
        elif pcr < 0.5:
            polarity -= 1.0
    
    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively WRITING PUTS (bullish conviction)",
            "action": "Bullish breakout likely"
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing",
            "action": "Expect support to hold"
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively WRITING CALLS (bearish conviction)",
            "action": "Bearish breakdown likely"
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing",
            "action": "Expect resistance to hold"
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity",
            "action": "Range-bound expected"
        }

# ============================================
# 📈 ENTRY SIGNAL CALCULATION
# ============================================
def calculate_entry_signal_extended(
    spot, merged_df, atm_strike, seller_bias_result, seller_max_pain,
    nearest_sup, nearest_res, moment_metrics
):
    """Calculate optimal entry signal"""
    
    signal_score = 0
    signal_reasons = []
    optimal_entry_price = spot
    position_type = "NEUTRAL"
    
    seller_bias = seller_bias_result["bias"]
    
    if "STRONG BULLISH" in seller_bias or "BULLISH" in seller_bias:
        signal_score += 40
        position_type = "LONG"
        signal_reasons.append(f"Seller bias: {seller_bias}")
    elif "STRONG BEARISH" in seller_bias or "BEARISH" in seller_bias:
        signal_score += 40
        position_type = "SHORT"
        signal_reasons.append(f"Seller bias: {seller_bias}")
    else:
        signal_score += 10
        position_type = "NEUTRAL"
    
    if seller_max_pain:
        distance_to_max_pain = abs(spot - seller_max_pain)
        distance_pct = (distance_to_max_pain / spot) * 100
        
        if distance_pct < 0.5:
            signal_score += 15
            signal_reasons.append(f"Spot close to Max Pain (₹{seller_max_pain:,})")
            optimal_entry_price = seller_max_pain
    
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        if range_size > 0:
            position_in_range = ((spot - nearest_sup["strike"]) / range_size) * 100
            
            if position_type == "LONG" and position_in_range < 40:
                signal_score += 20
                signal_reasons.append(f"Near support (₹{nearest_sup['strike']:,})")
                optimal_entry_price = nearest_sup["strike"] + (range_size * 0.1)
            elif position_type == "SHORT" and position_in_range > 60:
                signal_score += 20
                signal_reasons.append(f"Near resistance (₹{nearest_res['strike']:,})")
                optimal_entry_price = nearest_res["strike"] - (range_size * 0.1)
    
    confidence = min(max(signal_score, 0), 100)
    
    if confidence >= 80:
        signal_strength = "STRONG"
        signal_color = "#00ff88" if position_type == "LONG" else "#ff4444"
    elif confidence >= 60:
        signal_strength = "MODERATE"
        signal_color = "#00cc66" if position_type == "LONG" else "#ff6666"
    elif confidence >= 40:
        signal_strength = "WEAK"
        signal_color = "#66b3ff"
    else:
        signal_strength = "NO SIGNAL"
        signal_color = "#cccccc"
        position_type = "NEUTRAL"
    
    return {
        "position_type": position_type,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "optimal_entry_price": optimal_entry_price,
        "current_spot": spot,
        "signal_color": signal_color,
        "reasons": signal_reasons
    }

# ============================================
# 🎯 UTILITY FUNCTIONS
# ============================================
def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def strike_gap_from_series(series):
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 50
    mode = diffs.mode()
    return int(mode.iloc[0]) if not mode.empty else int(diffs.median())

# Black-Scholes Greeks
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

# ============================================
# 📡 DHAN API FUNCTIONS
# ============================================
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

# ============================================
# 🎨 STREAMLIT UI & DISPLAY
# ============================================
st.set_page_config(
    page_title="Nifty Screener v7 - Seller's Perspective",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(r"""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* Seller Theme Colors */
    .seller-bullish { color: #00ff88 !important; font-weight: 700 !important; }
    .seller-bearish { color: #ff4444 !important; font-weight: 700 !important; }
    .seller-neutral { color: #66b3ff !important; font-weight: 700 !important; }
    
    h1, h2, h3 { color: #ff66cc !important; }
    
    .card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid;
        margin: 10px 0;
    }
    
    .signal-card {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid #ff9900;
        text-align: center;
    }
    
    .metric-card {
        background: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 5px 0;
    }
    
    .expiry-high-risk { border-color: #ff0000 !important; animation: pulse 2s infinite; }
    .expiry-medium-risk { border-color: #ff9900 !important; }
    .expiry-low-risk { border-color: #00ff00 !important; }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    [data-testid="stMetricLabel"] { color: #cccccc !important; }
    [data-testid="stMetricValue"] { color: #ff66cc !important; font-size: 1.6rem !important; }
    
    .stButton > button {
        background-color: #ff66cc !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and Time
st.title("🎯 NIFTY Option Screener v7.0 — 100% SELLER'S PERSPECTIVE")
current_ist = get_ist_datetime_str()
st.markdown(f"<div style='text-align: center; margin-bottom: 20px;'><span style='background-color: #1a1f2e; color: #ff66cc; padding: 8px 15px; border-radius: 20px; border: 2px solid #ff66cc; font-weight: 700;'>🕐 IST: {current_ist}</span></div>", unsafe_allow_html=True)

# ============================================
# 📊 SIDEBAR CONTROLS
# ============================================
with st.sidebar:
    st.markdown("### 🎯 CONTROLS")
    
    # Expiry selection
    expiries = get_expiry_list()
    if expiries:
        expiry = st.selectbox("Select Expiry", expiries, index=0)
    else:
        st.error("No expiries available")
        st.stop()
    
    st.markdown("---")
    
    # AI Settings
    st.markdown("### 🧠 AI ANALYSIS")
    enable_ai = st.checkbox("Enable AI Analysis", value=trading_ai.is_enabled())
    
    if enable_ai and not trading_ai.is_enabled():
        st.warning("Add PERPLEXITY_API_KEY to secrets")
    
    st.markdown("---")
    
    # Telegram Settings
    st.markdown("### 📱 TELEGRAM")
    auto_send = st.checkbox("Auto-send signals", value=False)
    show_preview = st.checkbox("Show signal preview", value=True)
    
    st.markdown("---")
    
    # System
    st.markdown("### ⚙️ SYSTEM")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown(f"**Refresh:** {AUTO_REFRESH_SEC}s")
    st.markdown(f"**Expiry:** {expiry}")

# ============================================
# 📈 MAIN DATA FETCHING & PROCESSING
# ============================================
# Fetch data
spot = get_nifty_spot_price()
if spot == 0:
    st.error("Unable to fetch NIFTY spot")
    st.stop()

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

# Calculate days to expiry
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
    days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
except Exception:
    tau = 7.0/365.0
    days_to_expiry = 7.0

# Initialize session state
_init_history()
if "prev_ltps_seller" not in st.session_state:
    st.session_state["prev_ltps_seller"] = {}
if "prev_ivs_seller" not in st.session_state:
    st.session_state["prev_ivs_seller"] = {}

# ============================================
# 🎯 COMPUTE ALL METRICS
# ============================================
# Compute per-strike metrics
for i, row in merged.iterrows():
    strike = int(row["strikePrice"])
    ltp_ce = safe_float(row.get("LTP_CE",0.0))
    ltp_pe = safe_float(row.get("LTP_PE",0.0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    
    key_ce = f"{expiry}_{strike}_CE"
    key_pe = f"{expiry}_{strike}_PE"
    prev_ce = st.session_state["prev_ltps_seller"].get(key_ce, None)
    prev_pe = st.session_state["prev_ltps_seller"].get(key_pe, None)
    
    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)
    
    st.session_state["prev_ltps_seller"][key_ce] = ltp_ce
    st.session_state["prev_ltps_seller"][key_pe] = ltp_pe
    
    chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))
    
    # Seller actions
    merged.at[i,"CE_Seller_Action"] = "WRITING" if chg_oi_ce>0 else ("BUYING BACK" if chg_oi_ce<0 else "HOLDING")
    merged.at[i,"PE_Seller_Action"] = "WRITING" if chg_oi_pe>0 else ("BUYING BACK" if chg_oi_pe<0 else "HOLDING")
    
    # Greeks calculation
    sigma_ce = iv_ce/100.0 if not np.isnan(iv_ce) and iv_ce>0 else 0.25
    sigma_pe = iv_pe/100.0 if not np.isnan(iv_pe) and iv_pe>0 else 0.25
    
    try:
        delta_ce = bs_delta(spot, strike, RISK_FREE_RATE, sigma_ce, tau, option_type="call")
        gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_ce, tau)
    except Exception:
        delta_ce = gamma_ce = 0.0
    
    try:
        delta_pe = bs_delta(spot, strike, RISK_FREE_RATE, sigma_pe, tau, option_type="put")
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, sigma_pe, tau)
    except Exception:
        delta_pe = gamma_pe = 0.0
    
    merged.at[i,"Delta_CE"] = delta_ce
    merged.at[i,"Gamma_CE"] = gamma_ce
    merged.at[i,"Delta_PE"] = delta_pe
    merged.at[i,"Gamma_PE"] = gamma_pe
    
    # GEX calculation
    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    gex_ce = gamma_ce * notional * oi_ce
    gex_pe = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_CE"] = gex_ce
    merged.at[i,"GEX_PE"] = gex_pe
    merged.at[i,"GEX_Net"] = gex_ce + gex_pe
    
    merged.at[i,"Seller_Strength_Score"] = seller_strength_score(row)

# Store moment snapshot
st.session_state["moment_history"].append(
    _snapshot_from_state(get_ist_now(), spot, atm_strike, merged)
)
st.session_state["moment_history"] = st.session_state["moment_history"][-10:]

# ============================================
# 📊 COMPUTE ANALYTICS
# ============================================
# Core metrics
total_gex_net = merged["GEX_Net"].sum()
seller_max_pain = calculate_seller_max_pain(merged)
seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)

# OI/PCR metrics
oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)

# Moment metrics
moment_metrics = {
    "momentum_burst": compute_momentum_burst(st.session_state["moment_history"]),
    "gamma_cluster": compute_gamma_cluster(merged, atm_strike, window=2)
}

# Expiry spike detection
expiry_spike_data = detect_expiry_spikes_seller_perspective(
    merged, spot, atm_strike, days_to_expiry, expiry
)

# Entry signal
entry_signal = calculate_entry_signal_extended(
    spot=spot,
    merged_df=merged,
    atm_strike=atm_strike,
    seller_bias_result=seller_bias_result,
    seller_max_pain=seller_max_pain,
    nearest_sup={"strike": atm_strike - strike_gap},
    nearest_res={"strike": atm_strike + strike_gap},
    moment_metrics=moment_metrics
)

# ============================================
# 🎯 MAIN DASHBOARD LAYOUT
# ============================================

# Row 1: Key Metrics
st.markdown("## 📊 MARKET OVERVIEW")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("NIFTY Spot", f"₹{spot:.2f}")
    st.metric("ATM Strike", f"₹{atm_strike}")
with col2:
    st.metric("Seller Bias", seller_bias_result["bias"])
    st.metric("Polarity", f"{seller_bias_result['polarity']:.2f}")
with col3:
    st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
    st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])
with col4:
    st.metric("Total GEX", f"₹{int(total_gex_net):,}")
    st.metric("Days to Expiry", f"{days_to_expiry:.1f}")

# Row 2: Seller's Max Pain
if seller_max_pain:
    distance = abs(spot - seller_max_pain)
    st.markdown(f"""
    <div class='card' style='border-color:#ff9900;'>
        <h3>🎯 SELLER'S MAX PAIN</h3>
        <div style='font-size: 2rem; color: #ff9900; font-weight: bold; text-align: center;'>₹{seller_max_pain:,}</div>
        <div style='text-align: center; color: #cccccc;'>Distance: ₹{distance:.2f} ({distance/spot*100:.2f}%)</div>
        <div style='text-align: center; color: #ffcc00;'>Sellers want price here to minimize losses</div>
    </div>
    """, unsafe_allow_html=True)

# Row 3: Trading Signal
st.markdown("## 🎯 TRADING SIGNAL")

if entry_signal["position_type"] != "NEUTRAL":
    signal_color = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
    st.markdown(f"""
    <div class='signal-card'>
        <div style='font-size: 2.5rem; font-weight: 900; color:{signal_color};'>
            {entry_signal["position_type"]} {entry_signal["signal_strength"]}
        </div>
        <div style='font-size: 1.5rem; color: #ffcc00; margin: 10px 0;'>
            Entry: ₹{entry_signal["optimal_entry_price"]:,.2f}
        </div>
        <div style='font-size: 1.2rem; color: #cccccc;'>
            Confidence: {entry_signal["confidence"]:.0f}% | Spot: ₹{spot:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='signal-card'>
        <div style='font-size: 2.5rem; font-weight: 900; color:#cccccc;'>
            ⚠️ NO CLEAR SIGNAL
        </div>
        <div style='font-size: 1.2rem; color: #ffcc00; margin: 10px 0;'>
            Wait for better setup
        </div>
        <div style='font-size: 1rem; color: #cccccc;'>
            Confidence: {entry_signal["confidence"]:.0f}% | Spot: ₹{spot:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Row 4: Moment Detector & Expiry Spike
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("### 🚀 MOMENT DETECTOR")
    mb = moment_metrics["momentum_burst"]
    gc = moment_metrics["gamma_cluster"]
    
    col_m1a, col_m1b = st.columns(2)
    with col_m1a:
        if mb["available"]:
            color = "#ff00ff" if mb["score"] > 70 else ("#ff9900" if mb["score"] > 40 else "#66b3ff")
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{color};'>
                <div style='font-size: 0.9rem; color:#cccccc;'>Momentum Burst</div>
                <div style='font-size: 1.5rem; color:{color}; font-weight:700;'>{mb["score"]}/100</div>
                <div style='font-size: 0.8rem; color:#aaaaaa;'>{mb["note"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_m1b:
        if gc["available"]:
            color = "#ff00ff" if gc["score"] > 70 else ("#ff9900" if gc["score"] > 40 else "#66b3ff")
            st.markdown(f"""
            <div class='metric-card' style='border-left-color:{color};'>
                <div style='font-size: 0.9rem; color:#cccccc;'>Gamma Cluster</div>
                <div style='font-size: 1.5rem; color:{color}; font-weight:700;'>{gc["score"]}/100</div>
                <div style='font-size: 0.8rem; color:#aaaaaa;'>ATM concentration</div>
            </div>
            """, unsafe_allow_html=True)

with col_m2:
    st.markdown("### 📅 EXPIRY SPIKE DETECTOR")
    
    if expiry_spike_data["active"]:
        risk_class = "expiry-high-risk" if expiry_spike_data["probability"] >= 70 else ("expiry-medium-risk" if expiry_spike_data["probability"] >= 50 else "expiry-low-risk")
        
        st.markdown(f"""
        <div class='card {risk_class}'>
            <h3 style='color:{expiry_spike_data["color"]};'>{expiry_spike_data["intensity"]}</h3>
            <div style='font-size: 2rem; color:{expiry_spike_data["color"]}; font-weight:900; text-align:center;'>
                {expiry_spike_data["probability"]}%
            </div>
            <div style='font-size: 1.2rem; color:#ffffff; text-align:center;'>
                {expiry_spike_data["type"]}
            </div>
            <div style='font-size: 0.9rem; color:#cccccc; text-align:center;'>
                Days to expiry: {days_to_expiry:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Seller-specific metrics
        col_sell1, col_sell2 = st.columns(2)
        with col_sell1:
            st.metric("CALL Writing", f"{expiry_spike_data.get('call_writing', 0):,}")
        with col_sell2:
            st.metric("PUT Writing", f"{expiry_spike_data.get('put_writing', 0):,}")
    else:
        st.info("📅 Expiry spike detector inactive (expiry >5 days away)")

# Row 5: OI/PCR Analytics
st.markdown("## 📊 OI & PCR ANALYTICS")

col_o1, col_o2, col_o3, col_o4 = st.columns(4)

with col_o1:
    st.metric("Total CALL OI", f"{oi_pcr_metrics['total_ce_oi']:,}")
    st.metric("Δ CALL OI", f"{oi_pcr_metrics['total_ce_chg']:+,}")

with col_o2:
    st.metric("Total PUT OI", f"{oi_pcr_metrics['total_pe_oi']:,}")
    st.metric("Δ PUT OI", f"{oi_pcr_metrics['total_pe_chg']:+,}")

with col_o3:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color:{oi_pcr_metrics["pcr_color"]};'>
        <div style='font-size: 0.9rem; color:#cccccc;'>PCR (TOTAL)</div>
        <div style='font-size: 1.5rem; color:{oi_pcr_metrics["pcr_color"]}; font-weight:700;'>
            {oi_pcr_metrics['pcr_total']:.2f}
        </div>
        <div style='font-size: 0.8rem; color:{oi_pcr_metrics["pcr_color"]};'>
            {oi_pcr_metrics['pcr_interpretation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_o4:
    st.metric("Max CALL OI Strike", f"₹{oi_pcr_metrics['max_ce_strike']:,}")
    st.caption(f"OI: {oi_pcr_metrics['max_ce_oi']:,}")
    st.metric("Max PUT OI Strike", f"₹{oi_pcr_metrics['max_pe_strike']:,}")
    st.caption(f"OI: {oi_pcr_metrics['max_pe_oi']:,}")

# Row 6: Detailed Tabs
st.markdown("## 📈 DETAILED ANALYSIS")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Seller Activity", 
    "🧮 Seller Greeks", 
    "🚀 Moment Details", 
    "📅 Expiry Analysis",
    "🧠 AI Analysis"
])

with tab1:
    st.markdown("### 📊 SELLER ACTIVITY BY STRIKE")
    seller_cols = [
        "strikePrice", 
        "OI_CE", "Chg_OI_CE", "CE_Seller_Action",
        "OI_PE", "Chg_OI_PE", "PE_Seller_Action",
        "Seller_Strength_Score"
    ]
    
    for col in seller_cols:
        if col not in merged.columns:
            merged[col] = ""
    
    st.dataframe(merged[seller_cols], use_container_width=True)

with tab2:
    st.markdown("### 🧮 SELLER GREEKS & GEX")
    greeks_cols = [
        "strikePrice",
        "Delta_CE", "Gamma_CE", "GEX_CE",
        "Delta_PE", "Gamma_PE", "GEX_PE",
        "GEX_Net"
    ]
    
    for col in greeks_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    
    st.dataframe(merged[greeks_cols], use_container_width=True)
    
    # GEX Interpretation
    if total_gex_net > 0:
        st.success(f"**POSITIVE GEX (₹{int(total_gex_net):,}):** Sellers SHORT gamma → stabilizing effect")
    elif total_gex_net < 0:
        st.error(f"**NEGATIVE GEX (₹{int(total_gex_net):,}):** Sellers LONG gamma → explosive moves")

with tab3:
    st.markdown("### 🚀 MOMENT DETECTOR DETAILS")
    
    if moment_metrics["momentum_burst"]["available"]:
        st.metric("Momentum Burst Score", f"{moment_metrics['momentum_burst']['score']}/100")
        st.info(f"**Note:** {moment_metrics['momentum_burst']['note']}")
    
    if moment_metrics["gamma_cluster"]["available"]:
        st.metric("Gamma Cluster Score", f"{moment_metrics['gamma_cluster']['score']}/100")
        st.metric("Raw Cluster Value", f"{moment_metrics['gamma_cluster']['cluster']:.2f}")
        
        if moment_metrics["gamma_cluster"]["score"] > 70:
            st.success("**HIGH GAMMA CLUSTER:** Expect sharp moves around ATM")
        elif moment_metrics["gamma_cluster"]["score"] > 40:
            st.info("**MODERATE GAMMA CLUSTER:** Some gamma concentration")
        else:
            st.warning("**LOW GAMMA CLUSTER:** Smoother moves expected")

with tab4:
    st.markdown("### 📅 EXPIRY SPIKE ANALYSIS")
    
    if expiry_spike_data["active"]:
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.metric("Spike Probability", f"{expiry_spike_data['probability']}%")
            st.metric("Spike Score", f"{expiry_spike_data['score']}/100")
            st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
        
        with col_exp2:
            st.metric("Spike Type", expiry_spike_data['type'])
            st.metric("Intensity", expiry_spike_data['intensity'])
            st.metric("Seller Gamma", f"₹{expiry_spike_data.get('seller_gamma', 0):,}")
        
        st.markdown("#### ⚠️ SPIKE TRIGGERS")
        for factor in expiry_spike_data["factors"]:
            st.markdown(f"• {factor}")
        
        if expiry_spike_data["key_levels"]:
            st.markdown("#### 🎯 KEY LEVELS")
            for level in expiry_spike_data["key_levels"]:
                st.markdown(f"• {level}")
    else:
        st.info("Expiry spike detector inactive")

with tab5:
    st.markdown("### 🧠 AI-POWERED ANALYSIS")
    
    if trading_ai.is_enabled() and enable_ai:
        if st.button("🤖 Generate AI Analysis"):
            with st.spinner("AI is analyzing market conditions..."):
                market_data = {
                    'spot': spot,
                    'seller_bias': seller_bias_result['bias'],
                    'max_pain': seller_max_pain if seller_max_pain else 0,
                    'total_pcr': oi_pcr_metrics['pcr_total']
                }
                
                ai_analysis = trading_ai.generate_analysis(
                    market_data, entry_signal, moment_metrics, expiry_spike_data
                )
                
                if ai_analysis:
                    st.success("✅ AI Analysis Generated!")
                    st.markdown(f"""
                    <div style="
                        background-color: #1a1f2e;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid #aa00ff;
                        margin: 10px 0;
                        white-space: pre-wrap;
                    ">
                    {ai_analysis}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to generate AI analysis")
    else:
        st.info("AI Analysis requires Perplexity API key")

# ============================================
# 📱 TELEGRAM SIGNAL SECTION
# ============================================
st.markdown("---")
st.markdown("## 📱 TELEGRAM SIGNAL")

# Generate signal
telegram_signal = None
if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
    telegram_signal = generate_telegram_signal(
        entry_signal, spot, seller_bias_result, seller_max_pain,
        {"strike": atm_strike - strike_gap}, {"strike": atm_strike + strike_gap},
        moment_metrics, expiry_spike_data
    )

if telegram_signal:
    st.success("🎯 **TRADE SIGNAL GENERATED**")
    
    if show_preview:
        st.markdown("### 📋 Signal Preview:")
        st.code(telegram_signal, language=None)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.success("✅ Signal copied to clipboard!")
    
    with col_t2:
        if st.button("📱 Send to Telegram", use_container_width=True):
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Telegram credentials not configured")
    
    # Auto-send
    if auto_send and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
        if success:
            st.info(f"📤 Auto-sent: {message}")
else:
    st.info("📭 No active trade signal to send")

# ============================================
# 🎯 FINAL SUMMARY
# ============================================
st.markdown("---")
st.markdown("## 💡 FINAL ASSESSMENT")

st.markdown(f"""
<div class='card' style='border-color:#ff66cc;'>
    <h3>🎯 SELLER'S PERSPECTIVE SUMMARY</h3>
    
    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;'>
        <div>
            <div style='font-size: 0.9rem; color:#cccccc;'>Market Bias</div>
            <div style='font-size: 1.2rem; color:{seller_bias_result["color"]}; font-weight:700;'>
                {seller_bias_result["bias"]}
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>{seller_bias_result["explanation"]}</div>
        </div>
        
        <div>
            <div style='font-size: 0.9rem; color:#cccccc;'>PCR Sentiment</div>
            <div style='font-size: 1.2rem; color:{oi_pcr_metrics["pcr_color"]}; font-weight:700;'>
                {oi_pcr_metrics['pcr_sentiment']}
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,}</div>
        </div>
        
        <div>
            <div style='font-size: 0.9rem; color:#cccccc;'>Trading Signal</div>
            <div style='font-size: 1.2rem; color:{entry_signal["signal_color"]}; font-weight:700;'>
                {entry_signal["position_type"]} ({entry_signal["signal_strength"]})
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>Confidence: {entry_signal["confidence"]:.0f}%</div>
        </div>
        
        <div>
            <div style='font-size: 0.9rem; color:#cccccc;'>Expiry Risk</div>
            <div style='font-size: 1.2rem; color:{expiry_spike_data["color"] if expiry_spike_data["active"] else "#66b3ff"}; font-weight:700;'>
                {expiry_spike_data["intensity"] if expiry_spike_data["active"] else "LOW"}
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>Days: {days_to_expiry:.1f}</div>
        </div>
    </div>
    
    <div style='margin-top: 20px; padding-top: 20px; border-top: 1px solid #444;'>
        <div style='font-size: 0.9rem; color:#ffcc00;'>
            <strong>Action Plan:</strong> {seller_bias_result["action"]}
        </div>
        {f"<div style='font-size: 0.8rem; color:#00ffcc; margin-top: 10px;'><strong>Signal Entry:</strong> ₹{entry_signal['optimal_entry_price']:,.2f}</div>" if entry_signal['position_type'] != 'NEUTRAL' else ""}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# 📝 FOOTER
# ============================================
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | ⏰ Last update: {get_ist_datetime_str()}")
st.caption("🎯 **NIFTY Option Screener v7.0 — 100% SELLER'S PERSPECTIVE** | All features integrated")

# Auto-refresh
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

if time.time() - st.session_state["last_refresh"] > AUTO_REFRESH_SEC:
    st.session_state["last_refresh"] = time.time()
    st.rerun()
