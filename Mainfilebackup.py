"""
Nifty Option Screener v7.0 — 100% SELLER'S PERSPECTIVE
Complete Organized Version with All Features

FEATURES:
1. 🎯 Seller's Perspective Analysis
2. 🚀 Moment Detector (Momentum, Orderbook, Gamma, OI Acceleration)
3. 📅 Expiry Spike Detector (Seller-Focused)
4. 📊 Enhanced OI/PCR Analytics
5. 🧠 AI-Powered Analysis (Perplexity)
6. 📱 Telegram Signal Generation
7. 💾 Supabase Data Storage

EVERYTHING from Option Seller/Market Maker viewpoint
"""

# ============================================
# IMPORTS
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

# ============================================
# CONFIGURATION
# ============================================
# Timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_time_str():
    return get_ist_now().strftime("%H:%M:%S")

def get_ist_date_str():
    return get_ist_now().strftime("%Y-%m-%d")

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

# Constants
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

# ============================================
# SECRETS & API KEYS
# ============================================
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

# Initialize AI
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
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
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
    
    # ATM Concentration
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 3
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    atm_total_oi = atm_ce_oi + atm_pe_oi
    atm_concentration_pct = (atm_total_oi / total_oi * 100) if total_oi > 0 else 0
    
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
        "max_pe_oi": max_pe_oi_val,
        "atm_concentration_pct": atm_concentration_pct,
        "atm_ce_oi": atm_ce_oi,
        "atm_pe_oi": atm_pe_oi
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
    
    signal_emoji = "🚀" if position_type == "LONG" else "🐻"
    current_time = get_ist_datetime_str()
    
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

def _norm01(x, lo, hi):
    """Normalize value to 0-1 range"""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

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
    score = int(100 * _norm01(burst_raw, 0.0, max(1.0, burst_raw * 2.5)))
    
    return {"available": True, "score": score, 
            "note": "Momentum burst (energy) is rising" if score > 60 else "No strong energy burst detected"}

def compute_gamma_cluster(merged: pd.DataFrame, atm_strike: int, window: int = 2):
    """Compute gamma cluster score"""
    if merged.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    want = [atm_strike + i for i in range(-window, window + 1)]
    subset = merged[merged["strikePrice"].isin(want)]
    if subset.empty:
        return {"available": False, "score": 0, "cluster": 0.0}
    
    cluster = float((subset["Gamma_CE"].abs().fillna(0) + subset["Gamma_PE"].abs().fillna(0)).sum())
    score = int(100 * _norm01(cluster, 0.0, max(1.0, cluster * 2.0)))
    return {"available": True, "score": score, "cluster": cluster}

def compute_oi_velocity_acceleration(history, atm_strike, window_strikes=3):
    """Compute OI velocity and acceleration"""
    if len(history) < 3:
        return {"available": False, "score": 0, "note": "Need 3+ refresh points for OI acceleration."}
    
    s0, s1, s2 = history[-3], history[-2], history[-1]
    dt1 = max((s1["ts"] - s0["ts"]).total_seconds(), 1.0)
    dt2 = max((s2["ts"] - s1["ts"]).total_seconds(), 1.0)
    
    def cluster_strikes(atm):
        return [atm + i for i in range(-window_strikes, window_strikes + 1) if (atm + i) in s2["per_strike"]]
    
    strikes = cluster_strikes(atm_strike)
    if not strikes:
        return {"available": False, "score": 0, "note": "No ATM cluster strikes found."}
    
    vel = []
    acc = []
    for k in strikes:
        o0 = s0["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        o1 = s1["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        o2 = s2["per_strike"].get(k, {"oi_ce": 0, "oi_pe": 0})
        
        t0 = o0["oi_ce"] + o0["oi_pe"]
        t1 = o1["oi_ce"] + o1["oi_pe"]
        t2 = o2["oi_ce"] + o2["oi_pe"]
        
        v1 = (t1 - t0) / dt1
        v2 = (t2 - t1) / dt2
        a = (v2 - v1) / dt2
        
        vel.append(abs(v2))
        acc.append(abs(a))
    
    vel_score = _norm01(np.median(vel), 0, max(1.0, np.percentile(vel, 90)))
    acc_score = _norm01(np.median(acc), 0, max(1.0, np.percentile(acc, 90)))
    
    score = int(100 * (0.6 * vel_score + 0.4 * acc_score))
    return {"available": True, "score": score, 
            "note": "OI speed-up detected in ATM cluster" if score > 60 else "OI changes are slow/steady"}

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
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back calls as price rises (Covering bearish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back calls on weakness (Reducing bearish exposure)"
    else:
        if oi_up and vol_up and price_up:
            return "Sellers WRITING puts on strength (Bullish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING puts as price falls (Strong bullish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back puts on strength (Covering bullish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back puts as price falls (Reducing bullish exposure)"
    
    if oi_up and not vol_up:
        return "Sellers quietly WRITING options"
    if (not oi_up) and not vol_up:
        return "Sellers quietly UNWINDING"
    
    return "Sellers inactive"

def seller_itm_otm_interpretation(strike, atm, chg_oi_ce, chg_oi_pe):
    """Interpret ITM/OTM seller activity"""
    ce_interpretation = ""
    pe_interpretation = ""
    
    if strike < atm:
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING ITM CALLS = VERY BEARISH 🚨"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK ITM CALLS = BULLISH 📈"
        else:
            ce_interpretation = "No ITM CALL activity"
    
    elif strike > atm:
        if chg_oi_ce > 0:
            ce_interpretation = "SELLERS WRITING OTM CALLS = MILD BEARISH 📉"
        elif chg_oi_ce < 0:
            ce_interpretation = "SELLERS BUYING BACK OTM CALLS = MILD BULLISH 📊"
        else:
            ce_interpretation = "No OTM CALL activity"
    
    else:
        ce_interpretation = "ATM CALL zone"
    
    if strike > atm:
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING ITM PUTS = VERY BULLISH 🚀"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK ITM PUTS = BEARISH 🐻"
        else:
            pe_interpretation = "No ITM PUT activity"
    
    elif strike < atm:
        if chg_oi_pe > 0:
            pe_interpretation = "SELLERS WRITING OTM PUTS = MILD BULLISH 📈"
        elif chg_oi_pe < 0:
            pe_interpretation = "SELLERS BUYING BACK OTM PUTS = MILD BEARISH 📉"
        else:
            pe_interpretation = "No OTM PUT activity"
    
    else:
        pe_interpretation = "ATM PUT zone"
    
    return f"CALL Sellers: {ce_interpretation} | PUT Sellers: {pe_interpretation}"

def seller_gamma_pressure(row, atm, strike_gap):
    """Calculate seller gamma pressure"""
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    
    chg_oi_sum = safe_float(row.get("Chg_OI_CE",0)) - safe_float(row.get("Chg_OI_PE",0))
    seller_pressure = -chg_oi_sum / dist
    
    return seller_pressure

def seller_breakout_probability_index(merged_df, atm, strike_gap):
    """Calculate seller breakout probability"""
    near_mask = merged_df["strikePrice"].between(atm-strike_gap, atm+strike_gap)
    
    atm_ce_build = merged_df.loc[near_mask, "Chg_OI_CE"].sum()
    atm_pe_build = merged_df.loc[near_mask, "Chg_OI_PE"].sum()
    seller_atm_bias = atm_pe_build - atm_ce_build
    atm_score = min(abs(seller_atm_bias)/50000.0, 1.0)
    
    ce_writing_count = (merged_df["CE_Seller_Action"] == "WRITING").sum()
    pe_writing_count = (merged_df["PE_Seller_Action"] == "WRITING").sum()
    ce_buying_back_count = (merged_df["CE_Seller_Action"] == "BUYING BACK").sum()
    pe_buying_back_count = (merged_df["PE_Seller_Action"] == "BUYING BACK").sum()
    
    total_actions = ce_writing_count + pe_writing_count + ce_buying_back_count + pe_buying_back_count
    if total_actions > 0:
        seller_conviction = (ce_writing_count + pe_writing_count) / total_actions
    else:
        seller_conviction = 0.5
    
    vol_oi_scores = (merged_df[["Vol_CE","Vol_PE"]].sum(axis=1) * merged_df[["Chg_OI_CE","Chg_OI_PE"]].abs().sum(axis=1)).fillna(0)
    vol_oi_score = min(vol_oi_scores.sum()/100000.0, 1.0)
    
    gamma = merged_df.apply(lambda r: seller_gamma_pressure(r, atm, strike_gap), axis=1).sum()
    gamma_score = min(abs(gamma)/10000.0, 1.0)
    
    w = BREAKOUT_INDEX_WEIGHTS
    combined = (w["atm_oi_shift"]*atm_score) + (w["winding_balance"]*seller_conviction) + (w["vol_oi_div"]*vol_oi_score) + (w["gamma_pressure"]*gamma_score)
    
    return int(np.clip(combined*100,0,100))

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
    
    avg_iv_ce = merged_df["IV_CE"].mean()
    avg_iv_pe = merged_df["IV_PE"].mean()
    if avg_iv_ce > avg_iv_pe + 5:
        polarity -= 0.3
    elif avg_iv_pe > avg_iv_ce + 5:
        polarity += 0.3
    
    total_gex_ce = merged_df["GEX_CE"].sum()
    total_gex_pe = merged_df["GEX_PE"].sum()
    net_gex = total_gex_ce + total_gex_pe
    if net_gex < -1000000:
        polarity -= 0.4
    elif net_gex > 1000000:
        polarity += 0.4
    
    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        distance_to_spot = abs(spot - max_pain) / spot * 100
        if distance_to_spot < 1.0:
            polarity += 0.5
    
    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes.",
            "action": "Bullish breakout likely. Sellers confident in upside."
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing. Moderate bullish sentiment.",
            "action": "Expect support to hold. Upside bias."
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes.",
            "action": "Bearish breakdown likely. Sellers confident in downside."
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing. Moderate bearish sentiment.",
            "action": "Expect resistance to hold. Downside bias."
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity. No clear directional bias.",
            "action": "Range-bound expected. Wait for clearer signals."
        }

def analyze_spot_position_seller(spot, pcr_df, market_bias):
    """Analyze spot position from seller perspective"""
    sorted_df = pcr_df.sort_values("strikePrice").reset_index(drop=True)
    all_strikes = sorted_df["strikePrice"].tolist()
    
    supports_below = [s for s in all_strikes if s < spot]
    nearest_support = max(supports_below) if supports_below else None
    
    resistances_above = [s for s in all_strikes if s > spot]
    nearest_resistance = min(resistances_above) if resistances_above else None
    
    def get_level_details(strike, df):
        if strike is None:
            return None
        row = df[df["strikePrice"] == strike]
        if row.empty:
            return None
        
        pcr = row.iloc[0]["PCR"]
        oi_ce = int(row.iloc[0]["OI_CE"])
        oi_pe = int(row.iloc[0]["OI_PE"])
        chg_oi_ce = int(row.iloc[0].get("Chg_OI_CE", 0))
        chg_oi_pe = int(row.iloc[0].get("Chg_OI_PE", 0))
        
        if pcr > 1.5:
            seller_strength = "Strong PUT selling (Bullish sellers)"
        elif pcr > 1.0:
            seller_strength = "Moderate PUT selling"
        elif pcr < 0.5:
            seller_strength = "Strong CALL selling (Bearish sellers)"
        elif pcr < 1.0:
            seller_strength = "Moderate CALL selling"
        else:
            seller_strength = "Balanced selling"
        
        return {
            "strike": int(strike),
            "oi_ce": oi_ce,
            "oi_pe": oi_pe,
            "chg_oi_ce": chg_oi_ce,
            "chg_oi_pe": chg_oi_pe,
            "pcr": pcr,
            "seller_strength": seller_strength,
            "distance": abs(spot - strike),
            "distance_pct": abs(spot - strike) / spot * 100
        }
    
    nearest_sup = get_level_details(nearest_support, sorted_df)
    nearest_res = get_level_details(nearest_resistance, sorted_df)
    
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        spot_position_pct = ((spot - nearest_sup["strike"]) / range_size * 100) if range_size > 0 else 50
        
        if spot_position_pct < 40:
            range_bias = "Near SELLER support (Bullish sellers defending)"
        elif spot_position_pct > 60:
            range_bias = "Near SELLER resistance (Bearish sellers defending)"
        else:
            range_bias = "Middle of SELLER range"
    else:
        range_size = 0
        spot_position_pct = 50
        range_bias = "Range undefined"
    
    return {
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "spot_in_range": (nearest_support, nearest_resistance),
        "range_size": range_size,
        "spot_position_pct": spot_position_pct,
        "range_bias": range_bias,
        "market_bias": market_bias
    }

def compute_pcr_df(merged_df):
    """Compute PCR dataframe"""
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

def rank_support_resistance_seller(pcr_df):
    """Rank support and resistance from seller perspective"""
    eps = 1e-6
    t = pcr_df.copy()
    
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    t["seller_support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    
    t["seller_resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["seller_resistance_score"] = t["OI_CE"] + (t["seller_resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("seller_support_score", ascending=False).head(3)
    top_resists = t.sort_values("seller_resistance_score", ascending=False).head(3)
    
    return t, top_supports, top_resists

# ============================================
# 📈 ENTRY SIGNAL CALCULATION
# ============================================
def calculate_realistic_stop_loss_target(position_type, entry_price, nearest_sup, nearest_res, strike_gap, max_risk_pct=1.5):
    """Calculate realistic stop loss and target"""
    stop_loss = None
    target = None
    
    if not nearest_sup or not nearest_res:
        return stop_loss, target
    
    max_risk_points = entry_price * (max_risk_pct / 100)
    
    if position_type == "LONG":
        stop_loss_support = nearest_sup["strike"] - (strike_gap * 1.5)
        stop_loss_pct = entry_price - max_risk_points
        stop_loss = max(stop_loss_support, stop_loss_pct)
        
        risk_amount = entry_price - stop_loss
        target_rr = entry_price + (risk_amount * 2)
        target_resistance = nearest_res["strike"] - strike_gap
        target = min(target_rr, target_resistance)
        
        if target <= entry_price:
            target = entry_price + risk_amount
    
    elif position_type == "SHORT":
        stop_loss_resistance = nearest_res["strike"] + (strike_gap * 1.5)
        stop_loss_pct = entry_price + max_risk_points
        stop_loss = min(stop_loss_resistance, stop_loss_pct)
        
        risk_amount = stop_loss - entry_price
        target_rr = entry_price - (risk_amount * 2)
        target_support = nearest_sup["strike"] + strike_gap
        target = max(target_rr, target_support)
        
        if target >= entry_price:
            target = entry_price - risk_amount
    
    if stop_loss:
        stop_loss = round(stop_loss / 50) * 50
    if target:
        target = round(target / 50) * 50
    
    return stop_loss, target

def calculate_entry_signal_extended(
    spot, merged_df, atm_strike, seller_bias_result, seller_max_pain,
    seller_supports_df, seller_resists_df, nearest_sup, nearest_res,
    seller_breakout_index, moment_metrics
):
    """Calculate optimal entry signal with Moment Detector integration"""
    
    signal_score = 0
    signal_reasons = []
    optimal_entry_price = spot
    position_type = "NEUTRAL"
    confidence = 0
    
    # 1. SELLER BIAS ANALYSIS (40% weight)
    seller_bias = seller_bias_result["bias"]
    seller_polarity = seller_bias_result["polarity"]
    
    if "STRONG BULLISH" in seller_bias or "BULLISH" in seller_bias:
        signal_score += 40
        position_type = "LONG"
        signal_reasons.append(f"Seller bias: {seller_bias} (Polarity: {seller_polarity:.1f})")
    elif "STRONG BEARISH" in seller_bias or "BEARISH" in seller_bias:
        signal_score += 40
        position_type = "SHORT"
        signal_reasons.append(f"Seller bias: {seller_bias} (Polarity: {seller_polarity:.1f})")
    else:
        signal_score += 10
        position_type = "NEUTRAL"
        signal_reasons.append("Seller bias: Neutral - Wait for clearer signal")
    
    # 2. MAX PAIN ALIGNMENT (15% weight)
    if seller_max_pain:
        distance_to_max_pain = abs(spot - seller_max_pain)
        distance_pct = (distance_to_max_pain / spot) * 100
        
        if distance_pct < 0.5:
            signal_score += 15
            signal_reasons.append(f"Spot VERY close to Max Pain (₹{seller_max_pain:,}, {distance_pct:.2f}%)")
            optimal_entry_price = seller_max_pain
        elif distance_pct < 1.0:
            signal_score += 10
            signal_reasons.append(f"Spot close to Max Pain (₹{seller_max_pain:,}, {distance_pct:.2f}%)")
            if position_type == "LONG" and spot < seller_max_pain:
                optimal_entry_price = min(spot + (seller_max_pain - spot) * 0.5, seller_max_pain)
            elif position_type == "SHORT" and spot > seller_max_pain:
                optimal_entry_price = max(spot - (spot - seller_max_pain) * 0.5, seller_max_pain)
    
    # 3. SUPPORT/RESISTANCE ALIGNMENT (20% weight)
    if nearest_sup and nearest_res:
        range_size = nearest_res["strike"] - nearest_sup["strike"]
        if range_size > 0:
            position_in_range = ((spot - nearest_sup["strike"]) / range_size) * 100
            
            if position_type == "LONG":
                if position_in_range < 40:
                    signal_score += 20
                    signal_reasons.append(f"Ideal LONG entry: Near support (₹{nearest_sup['strike']:,})")
                    optimal_entry_price = nearest_sup["strike"] + (range_size * 0.1)
                elif position_in_range < 60:
                    signal_score += 10
                    signal_reasons.append("OK LONG entry: Middle of range")
                else:
                    signal_score += 5
                    
            elif position_type == "SHORT":
                if position_in_range > 60:
                    signal_score += 20
                    signal_reasons.append(f"Ideal SHORT entry: Near resistance (₹{nearest_res['strike']:,})")
                    optimal_entry_price = nearest_res["strike"] - (range_size * 0.1)
                elif position_in_range > 40:
                    signal_score += 10
                    signal_reasons.append("OK SHORT entry: Middle of range")
                else:
                    signal_score += 5
    
    # 4. BREAKOUT INDEX (15% weight)
    if seller_breakout_index > 80:
        signal_score += 15
        signal_reasons.append(f"High Breakout Index ({seller_breakout_index}%): Strong momentum expected")
    elif seller_breakout_index > 60:
        signal_score += 10
        signal_reasons.append(f"Moderate Breakout Index ({seller_breakout_index}%): Some momentum expected")
    
    # 5. PCR ANALYSIS (10% weight)
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        total_pcr = total_pe_oi / total_ce_oi
        if position_type == "LONG" and total_pcr > 1.5:
            signal_score += 10
            signal_reasons.append(f"Strong PCR ({total_pcr:.2f}): Heavy PUT selling confirms bullish bias")
        elif position_type == "SHORT" and total_pcr < 0.7:
            signal_score += 10
            signal_reasons.append(f"Strong PCR ({total_pcr:.2f}): Heavy CALL selling confirms bearish bias")
    
    # 6. GEX ANALYSIS (Adjustment factor)
    total_gex_net = merged_df["GEX_Net"].sum()
    if total_gex_net > 1000000:
        if position_type == "LONG":
            signal_score += 5
            signal_reasons.append("Positive GEX: Supports LONG position (stabilizing)")
    elif total_gex_net < -1000000:
        if position_type == "SHORT":
            signal_score += 5
            signal_reasons.append("Negative GEX: Supports SHORT position (destabilizing)")
    
    # 7. MOMENT DETECTOR FEATURES (30% total weight)
    mb = moment_metrics.get("momentum_burst", {})
    if mb.get("available", False):
        mb_score = mb.get("score", 0)
        signal_score += int(12 * (mb_score / 100.0))
        signal_reasons.append(f"Momentum burst: {mb_score}/100 - {mb.get('note', '')}")
    
    gc = moment_metrics.get("gamma_cluster", {})
    if gc.get("available", False):
        gc_score = gc.get("score", 0)
        signal_score += int(6 * (gc_score / 100.0))
        signal_reasons.append(f"Gamma cluster: {gc_score}/100 (ATM concentration)")
    
    oi_accel = moment_metrics.get("oi_accel", {})
    if oi_accel.get("available", False):
        oi_score = oi_accel.get("score", 0)
        signal_score += int(4 * (oi_score / 100.0))
        signal_reasons.append(f"OI acceleration: {oi_score}/100 ({oi_accel.get('note', '')})")
    
    # FINAL SIGNAL CALCULATION
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
        optimal_entry_price = spot
    
    # Calculate stop loss and target
    stop_loss = None
    target = None
    
    if nearest_sup and nearest_res and position_type != "NEUTRAL":
        strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
        stop_loss, target = calculate_realistic_stop_loss_target(
            position_type, optimal_entry_price, nearest_sup, nearest_res, strike_gap_val
        )
    
    return {
        "position_type": position_type,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "optimal_entry_price": optimal_entry_price,
        "current_spot": spot,
        "signal_color": signal_color,
        "reasons": signal_reasons,
        "stop_loss": stop_loss,
        "target": target,
        "max_pain": seller_max_pain,
        "nearest_support": nearest_sup["strike"] if nearest_sup else None,
        "nearest_resistance": nearest_res["strike"] if nearest_res else None,
        "moment_metrics": moment_metrics
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
    
    .seller-bullish-bg { background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%); }
    .seller-bearish-bg { background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%); }
    .seller-neutral-bg { background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%); }
    
    /* Moment Detector Colors */
    .moment-high { color: #ff00ff !important; font-weight: 800 !important; }
    .moment-medium { color: #ff9900 !important; font-weight: 700 !important; }
    .moment-low { color: #66b3ff !important; font-weight: 600 !important; }
    
    /* OI/PCR Colors */
    .pcr-extreme-bullish { color: #00ff88 !important; }
    .pcr-bullish { color: #00cc66 !important; }
    .pcr-mild-bullish { color: #66ff66 !important; }
    .pcr-neutral { color: #66b3ff !important; }
    .pcr-mild-bearish { color: #ff9900 !important; }
    .pcr-bearish { color: #ff4444 !important; }
    .pcr-extreme-bearish { color: #ff0000 !important; }
    
    h1, h2, h3 { color: #ff66cc !important; }
    
    .level-card {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff66cc;
        margin: 5px 0;
    }
    .level-card h4 { margin: 0; color: #ff66cc; font-size: 1.1rem; }
    .level-card p { margin: 5px 0; color: #fafafa; font-size: 1.3rem; font-weight: 700; }
    .level-card .sub-info { font-size: 0.9rem; color: #cccccc; margin-top: 5px; }
    
    .spot-card {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 10px 0;
        text-align: center;
    }
    .spot-card h3 { margin: 0; color: #ff9900; font-size: 1.3rem; }
    .spot-card .spot-price { font-size: 2.5rem; color: #ffcc00; font-weight: 700; margin: 10px 0; }
    .spot-card .distance { font-size: 1.1rem; color: #ffdd44; margin: 5px 0; }
    
    .nearest-level {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffcc;
        margin: 10px 0;
    }
    .nearest-level h4 { margin: 0; color: #00ffcc; font-size: 1.2rem; }
    .nearest-level .level-value { font-size: 1.8rem; color: #00ffcc; font-weight: 700; margin: 5px 0; }
    .nearest-level .level-distance { font-size: 1rem; color: #66ffdd; margin: 5px 0; }
    
    .seller-bias-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff66cc;
        margin: 15px 0;
        text-align: center;
    }
    .seller-bias-box h3 { margin: 0; color: #ff66cc; font-size: 1.4rem; }
    .seller-bias-box .bias-value { font-size: 2.2rem; font-weight: 900; margin: 10px 0; }
    
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .seller-support-building { background-color: #1a2e1a; border-left-color: #00ff88; color: #00ff88; }
    .seller-support-breaking { background-color: #2e1a1a; border-left-color: #ff4444; color: #ff6666; }
    .seller-resistance-building { background-color: #2e2a1a; border-left-color: #ffaa00; color: #ffcc44; }
    .seller-resistance-breaking { background-color: #1a1f2e; border-left-color: #00aaff; color: #00ccff; }
    .seller-bull-trap { background-color: #3e1a1a; border-left-color: #ff0000; color: #ff4444; font-weight: 700; }
    .seller-bear-trap { background-color: #1a3e1a; border-left-color: #00ff00; color: #00ff66; font-weight: 700; }
    
    .ist-time {
        background-color: #1a1f2e;
        color: #ff66cc;
        padding: 8px 15px;
        border-radius: 20px;
        border: 2px solid #ff66cc;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background-color: #ff66cc !important;
        color: #0e1117 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #ff99dd !important; }
    
    .greeks-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
    }
    
    .max-pain-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a2e2e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff9900;
        margin: 10px 0;
    }
    .max-pain-box h4 { margin: 0; color: #ff9900; }
    
    .seller-explanation {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff66cc;
        margin: 10px 0;
    }
    
    .entry-signal-box {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 15px 0;
        text-align: center;
    }
    .entry-signal-box h3 { margin: 0; color: #ff9900; font-size: 1.4rem; }
    .entry-signal-box .signal-value { font-size: 2.5rem; font-weight: 900; margin: 15px 0; }
    .entry-signal-box .signal-explanation { font-size: 1.1rem; color: #ffdd44; margin: 10px 0; }
    .entry-signal-box .entry-price { font-size: 1.8rem; color: #ffcc00; font-weight: 700; margin: 10px 0; }
    
    /* MOMENT DETECTOR BOXES */
    .moment-box {
        background: linear-gradient(135deg, #1a1f3e 0%, #2a2f4e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffff;
        margin: 10px 0;
        text-align: center;
    }
    .moment-box h4 { margin: 0; color: #00ffff; font-size: 1.1rem; }
    .moment-box .moment-value { font-size: 1.8rem; font-weight: 900; margin: 10px 0; }
    
    /* TELEGRAM SIGNAL BOX */
    .telegram-box {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #0088cc;
        margin: 15px 0;
    }
    .telegram-box h3 { margin: 0; color: #00aaff; font-size: 1.4rem; }
    
    /* AI ANALYSIS BOX */
    .ai-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #1a1f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #aa00ff;
        margin: 15px 0;
    }
    .ai-box h3 { margin: 0; color: #aa00ff; font-size: 1.4rem; }
    
    /* EXPIRY SPIKE DETECTOR STYLES */
    .expiry-high-risk {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%) !important;
        border: 3px solid #ff0000 !important;
        animation: pulse 2s infinite;
    }
    
    .expiry-medium-risk {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%) !important;
        border: 3px solid #ff9900 !important;
    }
    
    .expiry-low-risk {
        background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%) !important;
        border: 3px solid #00ff00 !important;
    }
    
    /* OI/PCR BOXES */
    .oi-pcr-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #66b3ff;
        margin: 15px 0;
    }
    
    .oi-pcr-metric {
        background: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #66b3ff;
        margin: 10px 0;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    [data-testid="stMetricLabel"] { color: #cccccc !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #ff66cc !important; font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 🎯 MAIN APP - COMPLETE ORGANIZED VERSION
# ============================================
st.title("🎯 NIFTY Option Screener v7.0 — 100% SELLER'S PERSPECTIVE")

current_ist = get_ist_datetime_str()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <span class='ist-time'>🕐 IST: {current_ist}</span>
</div>
""", unsafe_allow_html=True)

# ============================================
# 📊 SIDEBAR CONTROLS
# ============================================
with st.sidebar:
    st.markdown("""
    <div class='seller-explanation'>
    <h3>🎯 SELLER'S LOGIC</h3>
    <p><strong>Options WRITING = Directional Bias:</strong></p>
    <ul>
    <li><span class='seller-bearish'>📉 CALL Writing</span> = BEARISH (expecting price to STAY BELOW)</li>
    <li><span class='seller-bullish'>📈 PUT Writing</span> = BULLISH (expecting price to STAY ABOVE)</li>
    <li><span class='seller-bullish'>🔄 CALL Buying Back</span> = BULLISH (covering bearish bets)</li>
    <li><span class='seller-bearish'>🔄 PUT Buying Back</span> = BEARISH (covering bullish bets)</li>
    </ul>
    <p><em>Market makers & institutions are primarily SELLERS, not buyers.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 MOMENT DETECTOR FEATURES")
    st.markdown("""
    1. **Momentum Burst**: Volume × IV × ΔOI changes
    2. **Orderbook Pressure**: Buy/Sell depth imbalance
    3. **Gamma Cluster**: ATM gamma concentration
    4. **OI Acceleration**: Speed of OI changes
    """)
    
    st.markdown("---")
    st.markdown("### 📊 ENHANCED OI/PCR ANALYTICS")
    st.markdown("""
    **New Metrics:**
    1. Total OI Analysis (CALL/PUT)
    2. PCR Interpretation & Sentiment
    3. OI Concentration & Skew
    4. ITM/OTM OI Distribution
    5. Max OI Strikes
    6. Historical PCR Context
    """)
    
    st.markdown("---")
    st.markdown("### 📅 EXPIRY SPIKE DETECTOR")
    st.markdown("""
    **Activation:** ≤5 days to expiry
    
    **Detection Factors:**
    1. ATM OI Concentration
    2. Max Pain Distance
    3. PCR Extremes
    4. Massive OI Walls
    5. Gamma Flip Risk
    6. Unwinding Activity
    """)
    
    st.markdown("---")
    st.markdown("### 📱 TELEGRAM SIGNALS")
    st.markdown("""
    **Signal Conditions:**
    - Position ≠ NEUTRAL
    - Confidence ≥ 40%
    - New signal detected
    """)
    
    st.markdown("---")
    st.markdown("### 🧠 AI ANALYSIS")
    if trading_ai.is_enabled():
        st.success("✅ AI Analysis ENABLED")
        st.metric("AI Model", "Perplexity Sonar-Pro")
    else:
        st.warning("⚠️ AI Analysis DISABLED")
        st.info("Add PERPLEXITY_API_KEY to secrets to enable")
    
    st.markdown("---")
    st.markdown("### ⚙️ CONTROLS")
    
    # Expiry selection
    expiries = get_expiry_list()
    if expiries:
        expiry = st.selectbox("Select Expiry", expiries, index=0)
    else:
        st.error("No expiries available")
        st.stop()
    
    # AI Settings
    enable_ai = st.checkbox("Enable AI Analysis", value=trading_ai.is_enabled())
    
    # Telegram Settings
    auto_send = st.checkbox("Auto-send Telegram signals", value=False)
    show_preview = st.checkbox("Show signal preview", value=True)
    
    # System
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Current IST:** {get_ist_time_str()}")
    st.markdown(f"**Date:** {get_ist_date_str()}")
    st.markdown(f"**Expiry:** {expiry}")

# ============================================
# 📈 MAIN DATA FETCHING & PROCESSING
# ============================================
# Fetch data
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot")
        st.stop()

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

# Compute tau and days to expiry
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
    days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
except Exception:
    tau = 7.0/365.0
    days_to_expiry = 7.0

# Session storage for prev LTP/IV
if "prev_ltps_seller" not in st.session_state:
    st.session_state["prev_ltps_seller"] = {}
if "prev_ivs_seller" not in st.session_state:
    st.session_state["prev_ivs_seller"] = {}

# Initialize moment history
_init_history()

# ============================================
# 🎯 COMPUTE PER-STRIKE METRICS
# ============================================
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
    prev_iv_ce = st.session_state["prev_ivs_seller"].get(key_ce, None)
    prev_iv_pe = st.session_state["prev_ivs_seller"].get(key_pe, None)

    ce_price_delta = None if prev_ce is None else (ltp_ce - prev_ce)
    pe_price_delta = None if prev_pe is None else (ltp_pe - prev_pe)
    ce_iv_delta = None if prev_iv_ce is None else (iv_ce - prev_iv_ce)
    pe_iv_delta = None if prev_iv_pe is None else (iv_pe - prev_iv_pe)

    st.session_state["prev_ltps_seller"][key_ce] = ltp_ce
    st.session_state["prev_ltps_seller"][key_pe] = ltp_pe
    st.session_state["prev_ivs_seller"][key_ce] = iv_ce
    st.session_state["prev_ivs_seller"][key_pe] = iv_pe

    chg_oi_ce = safe_int(row.get("Chg_OI_CE",0))
    chg_oi_pe = safe_int(row.get("Chg_OI_PE",0))

    # SELLER winding/unwinding labels
    merged.at[i,"CE_Seller_Action"] = "WRITING" if chg_oi_ce>0 else ("BUYING BACK" if chg_oi_ce<0 else "HOLDING")
    merged.at[i,"PE_Seller_Action"] = "WRITING" if chg_oi_pe>0 else ("BUYING BACK" if chg_oi_pe<0 else "HOLDING")

    # SELLER divergence interpretation
    merged.at[i,"CE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_ce, safe_int(row.get("Vol_CE",0)), ce_price_delta, "CE")
    merged.at[i,"PE_Seller_Divergence"] = seller_price_oi_divergence(chg_oi_pe, safe_int(row.get("Vol_PE",0)), pe_price_delta, "PE")

    # SELLER ITM/OTM interpretation
    merged.at[i,"Seller_Interpretation"] = seller_itm_otm_interpretation(strike, atm_strike, chg_oi_ce, chg_oi_pe)

    # Greeks calculation
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

    # GEX calculation (SELLER exposure)
    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    gex_ce = gamma_ce * notional * oi_ce
    gex_pe = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_CE"] = gex_ce
    merged.at[i,"GEX_PE"] = gex_pe
    merged.at[i,"GEX_Net"] = gex_ce + gex_pe

    # SELLER strength score
    merged.at[i,"Seller_Strength_Score"] = seller_strength_score(row)

    # SELLER gamma pressure
    merged.at[i,"Seller_Gamma_Pressure"] = seller_gamma_pressure(row, atm_strike, strike_gap)

    merged.at[i,"CE_Price_Delta"] = ce_price_delta
    merged.at[i,"PE_Price_Delta"] = pe_price_delta
    merged.at[i,"CE_IV_Delta"] = ce_iv_delta
    merged.at[i,"PE_IV_Delta"] = pe_iv_delta

# ============================================
# 📊 COMPUTE AGGREGATE METRICS
# ============================================
# Aggregations
total_CE_OI = merged["OI_CE"].sum()
total_PE_OI = merged["OI_PE"].sum()
total_CE_chg = merged["Chg_OI_CE"].sum()
total_PE_chg = merged["Chg_OI_PE"].sum()

# SELLER activity summary
ce_selling = (merged["Chg_OI_CE"] > 0).sum()
ce_buying_back = (merged["Chg_OI_CE"] < 0).sum()
pe_selling = (merged["Chg_OI_PE"] > 0).sum()
pe_buying_back = (merged["Chg_OI_PE"] < 0).sum()

# Greeks totals
total_gex_ce = merged["GEX_CE"].sum()
total_gex_pe = merged["GEX_PE"].sum()
total_gex_net = merged["GEX_Net"].sum()

# Store moment snapshot
st.session_state["moment_history"].append(
    _snapshot_from_state(get_ist_now(), spot, atm_strike, merged)
)
st.session_state["moment_history"] = st.session_state["moment_history"][-10:]

# ============================================
# 🎯 COMPUTE ALL ANALYTICS
# ============================================
# Calculate SELLER metrics
seller_max_pain = calculate_seller_max_pain(merged)
seller_breakout_index = seller_breakout_probability_index(merged, atm_strike, strike_gap)

# Calculate SELLER market bias
seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)

# Compute PCR
pcr_df = compute_pcr_df(merged)

# Get SELLER support/resistance rankings
ranked_current, seller_supports_df, seller_resists_df = rank_support_resistance_seller(pcr_df)

# Analyze spot position from SELLER perspective
spot_analysis = analyze_spot_position_seller(spot, ranked_current, seller_bias_result)
nearest_sup = spot_analysis["nearest_support"]
nearest_res = spot_analysis["nearest_resistance"]

# OI/PCR metrics
oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)

# Moment metrics
moment_metrics = {
    "momentum_burst": compute_momentum_burst(st.session_state["moment_history"]),
    "gamma_cluster": compute_gamma_cluster(merged, atm_strike, window=2),
    "oi_accel": compute_oi_velocity_acceleration(st.session_state["moment_history"], atm_strike, window_strikes=2)
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
    seller_supports_df=seller_supports_df,
    seller_resists_df=seller_resists_df,
    nearest_sup=nearest_sup,
    nearest_res=nearest_res,
    seller_breakout_index=seller_breakout_index,
    moment_metrics=moment_metrics
)

# ============================================
# 🎯 MAIN DASHBOARD DISPLAY
# ============================================

# Row 1: Key Metrics
st.markdown("## 📊 MARKET OVERVIEW")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot", f"₹{spot:.2f}")
    st.metric("ATM", f"₹{atm_strike}")
with col2:
    st.metric("CALL Sellers", f"{ce_selling} strikes")
    st.metric("PUT Sellers", f"{pe_selling} strikes")
with col3:
    st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
    st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])
with col4:
    st.metric("Total GEX", f"₹{int(total_gex_net):,}")
    st.metric("Breakout Index", f"{seller_breakout_index}%")

# Row 2: Seller's Max Pain
if seller_max_pain:
    distance_to_max_pain = abs(spot - seller_max_pain)
    st.markdown(f"""
    <div class='max-pain-box'>
        <h4>🎯 SELLER'S MAX PAIN (Preferred Level)</h4>
        <p style='font-size: 1.5rem; color: #ff9900; font-weight: bold; text-align: center;'>₹{seller_max_pain:,}</p>
        <p style='text-align: center; color: #cccccc;'>Distance from spot: ₹{distance_to_max_pain:.2f} ({distance_to_max_pain/spot*100:.2f}%)</p>
        <p style='text-align: center; color: #ffcc00;'>Sellers want price here to minimize losses</p>
    </div>
    """, unsafe_allow_html=True)

# Row 3: Trading Signal
st.markdown("## 🎯 TRADING SIGNAL")

if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
    signal_color = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
    st.markdown(f"""
    <div class='entry-signal-box'>
        <h3>🎯 ACTIVE TRADING SIGNAL</h3>
        <div class='signal-value' style='color:{signal_color}'>
            {entry_signal["position_type"]} {entry_signal["signal_strength"]}
        </div>
        <div class='entry-price'>Entry: ₹{entry_signal["optimal_entry_price"]:,.2f}</div>
        <div class='signal-explanation'>
            Confidence: {entry_signal["confidence"]:.0f}% | Spot: ₹{spot:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='entry-signal-box'>
        <h3>⚠️ NO CLEAR SIGNAL</h3>
        <div class='signal-value' style='color:#cccccc'>
            WAIT FOR SETUP
        </div>
        <div class='signal-explanation'>
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
            <div class='moment-box'>
                <h4>💥 MOMENTUM BURST</h4>
                <div class='moment-value' style='color:{color}'>{mb["score"]}/100</div>
                <div style='font-size: 0.8rem; color:#cccccc;'>{mb["note"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_m1b:
        if gc["available"]:
            color = "#ff00ff" if gc["score"] > 70 else ("#ff9900" if gc["score"] > 40 else "#66b3ff")
            st.markdown(f"""
            <div class='moment-box'>
                <h4>🌀 GAMMA CLUSTER</h4>
                <div class='moment-value' style='color:{color}'>{gc["score"]}/100</div>
                <div style='font-size: 0.8rem; color:#cccccc;'>ATM ±2 concentration</div>
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
            st.caption("Bearish sellers exposed")
        with col_sell2:
            st.metric("PUT Writing", f"{expiry_spike_data.get('put_writing', 0):,}")
            st.caption("Bullish sellers exposed")
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
    <div class='oi-pcr-metric'>
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
        "OI_CE", "Chg_OI_CE", "CE_Seller_Action", "CE_Seller_Divergence",
        "OI_PE", "Chg_OI_PE", "PE_Seller_Action", "PE_Seller_Divergence",
        "Seller_Interpretation", "Seller_Strength_Score"
    ]
    
    # Ensure all columns exist
    for col in seller_cols:
        if col not in merged.columns:
            merged[col] = ""
    
    st.dataframe(merged[seller_cols], use_container_width=True)

with tab2:
    st.markdown("### 🧮 SELLER GREEKS & GEX EXPOSURE")
    greeks_cols = [
        "strikePrice",
        "Delta_CE", "Gamma_CE", "Vega_CE", "Theta_CE", "GEX_CE",
        "Delta_PE", "Gamma_PE", "Vega_PE", "Theta_PE", "GEX_PE",
        "GEX_Net", "Seller_Gamma_Pressure"
    ]
    
    for col in greeks_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    
    st.dataframe(merged[greeks_cols], use_container_width=True)
    
    # GEX Interpretation
    st.markdown("#### 🎯 GEX INTERPRETATION (SELLER'S VIEW)")
    if total_gex_net > 0:
        st.success(f"**POSITIVE GEX (₹{int(total_gex_net):,}):** Sellers have POSITIVE gamma exposure. They're SHORT gamma and will BUY when price rises, SELL when price falls (stabilizing effect).")
    elif total_gex_net < 0:
        st.error(f"**NEGATIVE GEX (₹{int(total_gex_net):,}):** Sellers have NEGATIVE gamma exposure. They're LONG gamma and will SELL when price rises, BUY when price falls (destabilizing effect).")
    else:
        st.info("**NEUTRAL GEX:** Balanced seller gamma exposure.")

with tab3:
    st.markdown("### 🚀 MOMENT DETECTOR DETAILS")
    
    # Momentum Burst Details
    st.markdown("#### 💥 MOMENTUM BURST ANALYSIS")
    mb = moment_metrics["momentum_burst"]
    if mb["available"]:
        col_mb1, col_mb2 = st.columns(2)
        with col_mb1:
            st.metric("Score", f"{mb['score']}/100")
            if mb["score"] > 70:
                st.success("**STRONG MOMENTUM:** High energy for directional move")
            elif mb["score"] > 40:
                st.info("**MODERATE MOMENTUM:** Some energy building")
            else:
                st.warning("**LOW MOMENTUM:** Market is calm")
        with col_mb2:
            st.info(f"**Note:** {mb['note']}")
    else:
        st.warning("Momentum burst data unavailable. Need more refresh points.")
    
    st.markdown("---")
    
    # Gamma Cluster Details
    st.markdown("#### 🌀 GAMMA CLUSTER ANALYSIS")
    gc = moment_metrics["gamma_cluster"]
    if gc["available"]:
        col_gc1, col_gc2 = st.columns(2)
        with col_gc1:
            st.metric("Cluster Score", f"{gc['score']}/100")
            st.metric("Raw Cluster Value", f"{gc['cluster']:.2f}")
        with col_gc2:
            if gc["score"] > 70:
                st.success("**HIGH GAMMA CLUSTER:** Strong concentration around ATM - expect sharp moves")
            elif gc["score"] > 40:
                st.info("**MODERATE GAMMA CLUSTER:** Some gamma concentration")
            else:
                st.warning("**LOW GAMMA CLUSTER:** Gamma spread out - smoother moves expected")

with tab4:
    st.markdown("### 📅 EXPIRY SPIKE ANALYSIS")
    
    if expiry_spike_data["active"]:
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.metric("Spike Probability", f"{expiry_spike_data['probability']}%")
            st.metric("Spike Score", f"{expiry_spike_data['score']}/100")
        
        with col_exp2:
            st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
            st.metric("Spike Type", expiry_spike_data['type'])
        
        with col_exp3:
            intensity = expiry_spike_data['intensity']
            intensity_color = expiry_spike_data['color']
            
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.2);
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid {intensity_color};
                margin: 10px 0;
            ">
                <div style="font-size: 0.9rem; color:#cccccc;">Spike Intensity</div>
                <div style="font-size: 1.2rem; color:{intensity_color}; font-weight:700;">{intensity}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Spike Triggers
        st.markdown("#### ⚠️ SPIKE TRIGGERS DETECTED")
        if expiry_spike_data.get("factors"):
            for factor in expiry_spike_data["factors"]:
                st.markdown(f"• {factor}")
        else:
            st.info("No spike triggers detected")
        
        # Key Levels
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
                    <div class='ai-box'>
                        <h3>🤖 AI MARKET ANALYSIS</h3>
                        <div style="white-space: pre-wrap; line-height: 1.6; margin-top: 15px;">
                        {ai_analysis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to generate AI analysis")
        else:
            st.info("Click 'Generate AI Analysis' to get AI insights")
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
        nearest_sup, nearest_res, moment_metrics, expiry_spike_data
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
<div class='seller-bias-box'>
    <h3>🎯 SELLER'S MARKET BIAS</h3>
    <div class='bias-value' style='color:{seller_bias_result["color"]}'>
        {seller_bias_result["bias"]}
    </div>
    <p>Polarity Score: {seller_bias_result["polarity"]:.2f}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='seller-explanation'>
    <h4>🧠 SELLER'S THINKING:</h4>
    <p><strong>{seller_bias_result["explanation"]}</strong></p>
    <p><strong>Action:</strong> {seller_bias_result["action"]}</p>
</div>
""", unsafe_allow_html=True)

# Final metrics summary
st.markdown("### 📊 SUMMARY METRICS")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Spot Position", f"{spot_analysis['spot_position_pct']:.1f}%")
    st.caption(spot_analysis['range_bias'])
with col_s2:
    st.metric("Range Size", f"₹{spot_analysis['range_size']:,}")
    st.caption("Support ↔ Resistance")
with col_s3:
    st.metric("Seller Conviction", f"{seller_breakout_index}%")
    st.caption("Breakout Probability")
with col_s4:
    st.metric("Signal Confidence", f"{entry_signal['confidence']:.0f}%")
    st.caption(f"{entry_signal['position_type']} {entry_signal['signal_strength']}")

# ============================================
# 📝 FOOTER
# ============================================
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | ⏰ Last update: {get_ist_datetime_str()}")
st.caption("🎯 **NIFTY Option Screener v7.0 — 100% SELLER'S PERSPECTIVE** | All features integrated")

# Requirements note
st.markdown("""
<small>
**Requirements:** 
`streamlit pandas numpy requests pytz scipy supabase perplexity-client python-dotenv` | 
**AI:** Perplexity API key required | 
**Data:** Dhan API required
</small>
""", unsafe_allow_html=True)

# Auto-refresh
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

if time.time() - st.session_state["last_refresh"] > AUTO_REFRESH_SEC:
    st.session_state["last_refresh"] = time.time()
    st.rerun()
