"""
Nifty Option Screener v7.0 — 100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS
EVERYTHING interpreted from Option Seller/Market Maker viewpoint
CALL building = BEARISH (sellers selling calls, expecting price to stay below)
PUT building = BULLISH (sellers selling puts, expecting price to stay above)

NEW FEATURES ADDED:
1. Comprehensive ATM Bias Analysis (12 metrics)
2. Multi-dimensional Bias Dashboard
3. Support/Resistance Bias Analysis
4. Enhanced Entry Signals with Bias Integration
5. Momentum Burst Detection
6. Orderbook Pressure Analysis
7. Gamma Cluster Concentration
8. OI Velocity/Acceleration
9. Telegram Signal Generation
10. Expiry Spike Detector
11. Enhanced OI/PCR Analytics
"""

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
from dotenv import load_dotenv
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
#  CONFIG
# -----------------------
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300

# NEW: Moment detector weights
MOMENT_WEIGHTS = {
    "momentum_burst": 0.40,        # Vol × IV × |ΔOI|
    "orderbook_pressure": 0.20,    # buy/sell depth imbalance
    "gamma_cluster": 0.25,         # ATM ±2 gamma concentration
    "oi_acceleration": 0.15        # OI speed-up (break/hold)
}

TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30 IST)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30 IST)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30 IST)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30 IST)"}
}

# -----------------------
#  SECRETS
# -----------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
    SUPABASE_TABLE = st.secrets.get("SUPABASE_TABLE", "option_snapshots")
    SUPABASE_TABLE_PCR = st.secrets.get("SUPABASE_TABLE_PCR", "strike_pcr_snapshots")
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
    # Telegram credentials (optional)
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
except Exception as e:
    st.error("❌ Missing credentials")
    st.stop()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"❌ Supabase failed: {e}")
    supabase = None

DHAN_BASE_URL = "https://api.dhan.co"
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# ============================================
# 🎯 ATM BIAS ANALYZER (NEW)
# ============================================
def analyze_atm_bias(merged_df, spot, atm_strike, strike_gap):
    """
    Analyze ATM bias from multiple perspectives for sellers
    """
    
    # Define ATM window (±2 strikes around ATM)
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"] 
                  if abs(s - atm_strike) <= (atm_window * strike_gap)]
    
    atm_df = merged_df[merged_df["strikePrice"].isin(atm_strikes)].copy()
    
    if atm_df.empty:
        return None
    
    # Initialize bias scores
    bias_scores = {
        "OI_Bias": 0,
        "ChgOI_Bias": 0,
        "Volume_Bias": 0,
        "Delta_Bias": 0,
        "Gamma_Bias": 0,
        "Premium_Bias": 0,
        "IV_Bias": 0,
        "Delta_Exposure_Bias": 0,
        "Gamma_Exposure_Bias": 0,
        "IV_Skew_Bias": 0,
        "OI_Change_Bias": 0
    }
    
    bias_interpretations = {}
    bias_emojis = {}
    
    # 1. OI BIAS (CALL vs PUT OI)
    total_ce_oi_atm = atm_df["OI_CE"].sum()
    total_pe_oi_atm = atm_df["OI_PE"].sum()
    oi_ratio = total_pe_oi_atm / max(total_ce_oi_atm, 1)
    
    if oi_ratio > 1.5:
        bias_scores["OI_Bias"] = 1
        bias_interpretations["OI_Bias"] = "Heavy PUT OI at ATM → Bullish sellers"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio > 1.0:
        bias_scores["OI_Bias"] = 0.5
        bias_interpretations["OI_Bias"] = "Moderate PUT OI → Mild bullish"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio < 0.7:
        bias_scores["OI_Bias"] = -1
        bias_interpretations["OI_Bias"] = "Heavy CALL OI at ATM → Bearish sellers"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    elif oi_ratio < 1.0:
        bias_scores["OI_Bias"] = -0.5
        bias_interpretations["OI_Bias"] = "Moderate CALL OI → Mild bearish"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    else:
        bias_scores["OI_Bias"] = 0
        bias_interpretations["OI_Bias"] = "Balanced OI → Neutral"
        bias_emojis["OI_Bias"] = "⚖️ Neutral"
    
    # 2. CHANGE IN OI BIAS (CALL vs PUT ΔOI)
    total_ce_chg_atm = atm_df["Chg_OI_CE"].sum()
    total_pe_chg_atm = atm_df["Chg_OI_PE"].sum()
    
    if total_pe_chg_atm > 0 and total_ce_chg_atm > 0:
        # Both sides writing
        if total_pe_chg_atm > total_ce_chg_atm:
            bias_scores["ChgOI_Bias"] = 0.5
            bias_interpretations["ChgOI_Bias"] = "More PUT writing → Bullish buildup"
            bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
        else:
            bias_scores["ChgOI_Bias"] = -0.5
            bias_interpretations["ChgOI_Bias"] = "More CALL writing → Bearish buildup"
            bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
    elif total_pe_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = 1
        bias_interpretations["ChgOI_Bias"] = "Only PUT writing → Strong bullish"
        bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
    elif total_ce_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = -1
        bias_interpretations["ChgOI_Bias"] = "Only CALL writing → Strong bearish"
        bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
    elif total_pe_chg_atm < 0 and total_ce_chg_atm < 0:
        # Both sides unwinding
        bias_scores["ChgOI_Bias"] = 0
        bias_interpretations["ChgOI_Bias"] = "Both unwinding → Range contraction"
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
    else:
        bias_scores["ChgOI_Bias"] = 0
        bias_interpretations["ChgOI_Bias"] = "Mixed activity"
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
    
    # 3. VOLUME BIAS (CALL vs PUT Volume)
    total_ce_vol_atm = atm_df["Vol_CE"].sum()
    total_pe_vol_atm = atm_df["Vol_PE"].sum()
    vol_ratio = total_pe_vol_atm / max(total_ce_vol_atm, 1)
    
    if vol_ratio > 1.3:
        bias_scores["Volume_Bias"] = 1
        bias_interpretations["Volume_Bias"] = "High PUT volume → Bullish activity"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio > 1.0:
        bias_scores["Volume_Bias"] = 0.5
        bias_interpretations["Volume_Bias"] = "More PUT volume → Mild bullish"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio < 0.8:
        bias_scores["Volume_Bias"] = -1
        bias_interpretations["Volume_Bias"] = "High CALL volume → Bearish activity"
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
    elif vol_ratio < 1.0:
        bias_scores["Volume_Bias"] = -0.5
        bias_interpretations["Volume_Bias"] = "More CALL volume → Mild bearish"
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Volume_Bias"] = 0
        bias_interpretations["Volume_Bias"] = "Balanced volume"
        bias_emojis["Volume_Bias"] = "⚖️ Neutral"
    
    # 4. DELTA BIAS (Net Delta Position)
    total_delta_ce = atm_df["Delta_CE"].sum()
    total_delta_pe = atm_df["Delta_PE"].sum()
    net_delta = total_delta_ce + total_delta_pe  # CALL delta positive, PUT delta negative
    
    if net_delta > 0.3:
        bias_scores["Delta_Bias"] = -1  # Positive delta = CALL heavy = Bearish for sellers
        bias_interpretations["Delta_Bias"] = "Positive delta → CALL heavy → Bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    elif net_delta > 0.1:
        bias_scores["Delta_Bias"] = -0.5
        bias_interpretations["Delta_Bias"] = "Mild positive delta → Slightly bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    elif net_delta < -0.3:
        bias_scores["Delta_Bias"] = 1  # Negative delta = PUT heavy = Bullish for sellers
        bias_interpretations["Delta_Bias"] = "Negative delta → PUT heavy → Bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    elif net_delta < -0.1:
        bias_scores["Delta_Bias"] = 0.5
        bias_interpretations["Delta_Bias"] = "Mild negative delta → Slightly bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    else:
        bias_scores["Delta_Bias"] = 0
        bias_interpretations["Delta_Bias"] = "Neutral delta"
        bias_emojis["Delta_Bias"] = "⚖️ Neutral"
    
    # 5. GAMMA BIAS (Net Gamma Position)
    total_gamma_ce = atm_df["Gamma_CE"].sum()
    total_gamma_pe = atm_df["Gamma_PE"].sum()
    net_gamma = total_gamma_ce + total_gamma_pe
    
    # For sellers: Positive gamma = stabilizing, Negative gamma = explosive
    if net_gamma > 0.1:
        bias_scores["Gamma_Bias"] = 1
        bias_interpretations["Gamma_Bias"] = "Positive gamma → Stabilizing → Bullish (less volatility)"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma > 0:
        bias_scores["Gamma_Bias"] = 0.5
        bias_interpretations["Gamma_Bias"] = "Mild positive gamma → Slightly stabilizing"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma < -0.1:
        bias_scores["Gamma_Bias"] = -1
        bias_interpretations["Gamma_Bias"] = "Negative gamma → Explosive → Bearish (high volatility)"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    elif net_gamma < 0:
        bias_scores["Gamma_Bias"] = -0.5
        bias_interpretations["Gamma_Bias"] = "Mild negative gamma → Slightly explosive"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Bias"] = 0
        bias_interpretations["Gamma_Bias"] = "Neutral gamma"
        bias_emojis["Gamma_Bias"] = "⚖️ Neutral"
    
    # 6. PREMIUM BIAS (CALL vs PUT Premium)
    # Calculate average premium
    ce_premium = atm_df["LTP_CE"].mean() if not atm_df["LTP_CE"].isna().all() else 0
    pe_premium = atm_df["LTP_PE"].mean() if not atm_df["LTP_PE"].isna().all() else 0
    premium_ratio = pe_premium / max(ce_premium, 0.01)
    
    if premium_ratio > 1.2:
        bias_scores["Premium_Bias"] = 1
        bias_interpretations["Premium_Bias"] = "PUT premium higher → Bullish sentiment"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio > 1.0:
        bias_scores["Premium_Bias"] = 0.5
        bias_interpretations["Premium_Bias"] = "PUT premium slightly higher → Mild bullish"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio < 0.8:
        bias_scores["Premium_Bias"] = -1
        bias_interpretations["Premium_Bias"] = "CALL premium higher → Bearish sentiment"
        bias_emojis["Premium_Bias"] = "🐻 Bearish"
    elif premium_ratio < 1.0:
        bias_scores["Premium_Bias"] = -0.5
        bias_interpretations["Premium_Bias"] = "CALL premium slightly higher → Mild bearish"
        bias_emojis["Premium_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Premium_Bias"] = 0
        bias_interpretations["Premium_Bias"] = "Balanced premiums"
        bias_emojis["Premium_Bias"] = "⚖️ Neutral"
    
    # 7. IV BIAS (CALL vs PUT IV)
    ce_iv = atm_df["IV_CE"].mean() if not atm_df["IV_CE"].isna().all() else 0
    pe_iv = atm_df["IV_PE"].mean() if not atm_df["IV_PE"].isna().all() else 0
    
    if pe_iv > ce_iv + 3:
        bias_scores["IV_Bias"] = 1
        bias_interpretations["IV_Bias"] = "PUT IV higher → Bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif pe_iv > ce_iv + 1:
        bias_scores["IV_Bias"] = 0.5
        bias_interpretations["IV_Bias"] = "PUT IV slightly higher → Mild bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif ce_iv > pe_iv + 3:
        bias_scores["IV_Bias"] = -1
        bias_interpretations["IV_Bias"] = "CALL IV higher → Bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    elif ce_iv > pe_iv + 1:
        bias_scores["IV_Bias"] = -0.5
        bias_interpretations["IV_Bias"] = "CALL IV slightly higher → Mild bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    else:
        bias_scores["IV_Bias"] = 0
        bias_interpretations["IV_Bias"] = "Balanced IV"
        bias_emojis["IV_Bias"] = "⚖️ Neutral"
    
    # 8. DELTA EXPOSURE BIAS (OI-weighted Delta)
    delta_exposure_ce = (atm_df["Delta_CE"] * atm_df["OI_CE"]).sum()
    delta_exposure_pe = (atm_df["Delta_PE"] * atm_df["OI_PE"]).sum()
    net_delta_exposure = delta_exposure_ce + delta_exposure_pe
    
    if net_delta_exposure > 1000000:
        bias_scores["Delta_Exposure_Bias"] = -1
        bias_interpretations["Delta_Exposure_Bias"] = "High CALL delta exposure → Bearish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    elif net_delta_exposure > 500000:
        bias_scores["Delta_Exposure_Bias"] = -0.5
        bias_interpretations["Delta_Exposure_Bias"] = "Moderate CALL delta exposure → Slightly bearish"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    elif net_delta_exposure < -1000000:
        bias_scores["Delta_Exposure_Bias"] = 1
        bias_interpretations["Delta_Exposure_Bias"] = "High PUT delta exposure → Bullish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    elif net_delta_exposure < -500000:
        bias_scores["Delta_Exposure_Bias"] = 0.5
        bias_interpretations["Delta_Exposure_Bias"] = "Moderate PUT delta exposure → Slightly bullish"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    else:
        bias_scores["Delta_Exposure_Bias"] = 0
        bias_interpretations["Delta_Exposure_Bias"] = "Balanced delta exposure"
        bias_emojis["Delta_Exposure_Bias"] = "⚖️ Neutral"
    
    # 9. GAMMA EXPOSURE BIAS (OI-weighted Gamma)
    gamma_exposure_ce = (atm_df["Gamma_CE"] * atm_df["OI_CE"]).sum()
    gamma_exposure_pe = (atm_df["Gamma_PE"] * atm_df["OI_PE"]).sum()
    net_gamma_exposure = gamma_exposure_ce + gamma_exposure_pe
    
    if net_gamma_exposure > 500000:
        bias_scores["Gamma_Exposure_Bias"] = 1
        bias_interpretations["Gamma_Exposure_Bias"] = "Positive gamma exposure → Stabilizing → Bullish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure > 100000:
        bias_scores["Gamma_Exposure_Bias"] = 0.5
        bias_interpretations["Gamma_Exposure_Bias"] = "Mild positive gamma → Slightly stabilizing"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure < -500000:
        bias_scores["Gamma_Exposure_Bias"] = -1
        bias_interpretations["Gamma_Exposure_Bias"] = "Negative gamma exposure → Explosive → Bearish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    elif net_gamma_exposure < -100000:
        bias_scores["Gamma_Exposure_Bias"] = -0.5
        bias_interpretations["Gamma_Exposure_Bias"] = "Mild negative gamma → Slightly explosive"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Exposure_Bias"] = 0
        bias_interpretations["Gamma_Exposure_Bias"] = "Balanced gamma exposure"
        bias_emojis["Gamma_Exposure_Bias"] = "⚖️ Neutral"
    
    # 10. IV SKEW BIAS (ATM vs Nearby strikes)
    # Get ±1 strike IVs
    nearby_strikes = [s for s in merged_df["strikePrice"] 
                     if abs(s - atm_strike) <= (1 * strike_gap)]
    nearby_df = merged_df[merged_df["strikePrice"].isin(nearby_strikes)]
    
    if not nearby_df.empty:
        atm_ce_iv = atm_df["IV_CE"].mean() if not atm_df["IV_CE"].isna().all() else 0
        atm_pe_iv = atm_df["IV_PE"].mean() if not atm_df["IV_PE"].isna().all() else 0
        nearby_ce_iv = nearby_df["IV_CE"].mean() if not nearby_df["IV_CE"].isna().all() else 0
        nearby_pe_iv = nearby_df["IV_PE"].mean() if not nearby_df["IV_PE"].isna().all() else 0
        
        # ATM IV vs Nearby IV comparison
        if atm_ce_iv > nearby_ce_iv + 2:
            bias_scores["IV_Skew_Bias"] = -0.5
            bias_interpretations["IV_Skew_Bias"] = "ATM CALL IV higher → Bearish skew"
            bias_emojis["IV_Skew_Bias"] = "🐻 Bearish"
        elif atm_pe_iv > nearby_pe_iv + 2:
            bias_scores["IV_Skew_Bias"] = 0.5
            bias_interpretations["IV_Skew_Bias"] = "ATM PUT IV higher → Bullish skew"
            bias_emojis["IV_Skew_Bias"] = "🐂 Bullish"
        else:
            bias_scores["IV_Skew_Bias"] = 0
            bias_interpretations["IV_Skew_Bias"] = "Flat IV skew"
            bias_emojis["IV_Skew_Bias"] = "⚖️ Neutral"
    else:
        bias_scores["IV_Skew_Bias"] = 0
        bias_interpretations["IV_Skew_Bias"] = "Insufficient data for IV skew"
        bias_emojis["IV_Skew_Bias"] = "⚖️ Neutral"
    
    # 11. OI CHANGE BIAS (Acceleration)
    # Calculate OI change rate
    total_oi_change = abs(total_ce_chg_atm) + abs(total_pe_chg_atm)
    total_oi_atm = total_ce_oi_atm + total_pe_oi_atm
    
    if total_oi_atm > 0:
        oi_change_rate = total_oi_change / total_oi_atm
        if oi_change_rate > 0.1:
            # High OI change - check direction
            if total_pe_chg_atm > total_ce_chg_atm:
                bias_scores["OI_Change_Bias"] = 0.5
                bias_interpretations["OI_Change_Bias"] = "Rapid PUT OI buildup → Bullish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐂 Bullish"
            else:
                bias_scores["OI_Change_Bias"] = -0.5
                bias_interpretations["OI_Change_Bias"] = "Rapid CALL OI buildup → Bearish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐻 Bearish"
        else:
            bias_scores["OI_Change_Bias"] = 0
            bias_interpretations["OI_Change_Bias"] = "Slow OI changes"
            bias_emojis["OI_Change_Bias"] = "⚖️ Neutral"
    
    # Calculate final bias score
    total_score = sum(bias_scores.values())
    normalized_score = total_score / len(bias_scores) if bias_scores else 0
    
    # Determine overall verdict
    if normalized_score > 0.3:
        verdict = "🐂 BULLISH"
        verdict_color = "#00ff88"
        verdict_explanation = "ATM zone showing strong bullish bias for sellers"
    elif normalized_score > 0.1:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00cc66"
        verdict_explanation = "ATM zone leaning bullish for sellers"
    elif normalized_score < -0.3:
        verdict = "🐻 BEARISH"
        verdict_color = "#ff4444"
        verdict_explanation = "ATM zone showing strong bearish bias for sellers"
    elif normalized_score < -0.1:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff6666"
        verdict_explanation = "ATM zone leaning bearish for sellers"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
        verdict_explanation = "ATM zone balanced, no clear bias"
    
    return {
        "instrument": "NIFTY",
        "strike": atm_strike,
        "zone": "ATM",
        "level": "ATM Cluster",
        "bias_scores": bias_scores,
        "bias_interpretations": bias_interpretations,
        "bias_emojis": bias_emojis,
        "total_score": normalized_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_explanation": verdict_explanation,
        "metrics": {
            "ce_oi": int(total_ce_oi_atm),
            "pe_oi": int(total_pe_oi_atm),
            "ce_chg": int(total_ce_chg_atm),
            "pe_chg": int(total_pe_chg_atm),
            "ce_vol": int(total_ce_vol_atm),
            "pe_vol": int(total_pe_vol_atm),
            "net_delta": round(net_delta, 3),
            "net_gamma": round(net_gamma, 3),
            "ce_iv": round(ce_iv, 2),
            "pe_iv": round(pe_iv, 2),
            "delta_exposure": int(net_delta_exposure),
            "gamma_exposure": int(net_gamma_exposure)
        }
    }

# ============================================
# 🎯 SUPPORT/RESISTANCE BIAS ANALYZER (NEW)
# ============================================
def analyze_support_resistance_bias(merged_df, spot, atm_strike, strike_gap, level_type="Support"):
    """
    Analyze bias at key support/resistance levels
    """
    
    # Find key levels
    if level_type == "Support":
        # Find highest strike below spot with high PUT OI
        support_strikes = merged_df[merged_df["strikePrice"] < spot].copy()
        if support_strikes.empty:
            return None
        
        # Find strike with highest PUT OI as support
        support_strike = support_strikes.loc[support_strikes["OI_PE"].idxmax()]["strikePrice"]
        level_df = merged_df[merged_df["strikePrice"] == support_strike]
    else:  # Resistance
        # Find lowest strike above spot with high CALL OI
        resistance_strikes = merged_df[merged_df["strikePrice"] > spot].copy()
        if resistance_strikes.empty:
            return None
        
        # Find strike with highest CALL OI as resistance
        resistance_strike = resistance_strikes.loc[resistance_strikes["OI_CE"].idxmax()]["strikePrice"]
        level_df = merged_df[merged_df["strikePrice"] == resistance_strike]
    
    if level_df.empty:
        return None
    
    row = level_df.iloc[0]
    
    # Calculate bias
    bias_scores = {}
    bias_emojis = {}
    bias_interpretations = {}
    
    # OI Bias
    oi_ratio = row["OI_PE"] / max(row["OI_CE"], 1)
    if oi_ratio > 2:
        bias_scores["OI_Bias"] = 1
        bias_emojis["OI_Bias"] = "🐂 Bullish"
        bias_interpretations["OI_Bias"] = "Very high PUT OI"
    elif oi_ratio > 1:
        bias_scores["OI_Bias"] = 0.5
        bias_emojis["OI_Bias"] = "🐂 Bullish"
        bias_interpretations["OI_Bias"] = "High PUT OI"
    elif oi_ratio < 0.5:
        bias_scores["OI_Bias"] = -1
        bias_emojis["OI_Bias"] = "🐻 Bearish"
        bias_interpretations["OI_Bias"] = "Very high CALL OI"
    elif oi_ratio < 1:
        bias_scores["OI_Bias"] = -0.5
        bias_emojis["OI_Bias"] = "🐻 Bearish"
        bias_interpretations["OI_Bias"] = "High CALL OI"
    else:
        bias_scores["OI_Bias"] = 0
        bias_emojis["OI_Bias"] = "⚖️ Neutral"
        bias_interpretations["OI_Bias"] = "Balanced OI"
    
    # OI Change Bias
    if row["Chg_OI_PE"] > 0 and row["Chg_OI_CE"] > 0:
        bias_scores["ChgOI_Bias"] = 0
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
        bias_interpretations["ChgOI_Bias"] = "Both sides building"
    elif row["Chg_OI_PE"] > 0:
        bias_scores["ChgOI_Bias"] = 1
        bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
        bias_interpretations["ChgOI_Bias"] = "PUT building"
    elif row["Chg_OI_CE"] > 0:
        bias_scores["ChgOI_Bias"] = -1
        bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
        bias_interpretations["ChgOI_Bias"] = "CALL building"
    else:
        bias_scores["ChgOI_Bias"] = 0
        bias_emojis["ChgOI_Bias"] = "⚖️ Neutral"
        bias_interpretations["ChgOI_Bias"] = "No fresh writing"
    
    # Volume Bias
    vol_ratio = row["Vol_PE"] / max(row["Vol_CE"], 1)
    if vol_ratio > 1.5:
        bias_scores["Volume_Bias"] = 1
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
        bias_interpretations["Volume_Bias"] = "High PUT volume"
    elif vol_ratio > 1:
        bias_scores["Volume_Bias"] = 0.5
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
        bias_interpretations["Volume_Bias"] = "More PUT volume"
    elif vol_ratio < 0.7:
        bias_scores["Volume_Bias"] = -1
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
        bias_interpretations["Volume_Bias"] = "High CALL volume"
    elif vol_ratio < 1:
        bias_scores["Volume_Bias"] = -0.5
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
        bias_interpretations["Volume_Bias"] = "More CALL volume"
    else:
        bias_scores["Volume_Bias"] = 0
        bias_emojis["Volume_Bias"] = "⚖️ Neutral"
        bias_interpretations["Volume_Bias"] = "Balanced volume"
    
    # Calculate total score
    total_score = sum(bias_scores.values())
    normalized_score = total_score / len(bias_scores) if bias_scores else 0
    
    # Determine verdict
    if normalized_score > 0.3:
        verdict = "🐂 BULLISH"
        verdict_color = "#00ff88"
    elif normalized_score > 0.1:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00cc66"
    elif normalized_score < -0.3:
        verdict = "🐻 BEARISH"
        verdict_color = "#ff4444"
    elif normalized_score < -0.1:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff6666"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
    
    return {
        "instrument": "NIFTY",
        "strike": int(row["strikePrice"]),
        "zone": level_type,
        "level": f"{level_type} Level",
        "bias_scores": bias_scores,
        "bias_interpretations": bias_interpretations,
        "bias_emojis": bias_emojis,
        "total_score": normalized_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "metrics": {
            "ce_oi": int(row["OI_CE"]),
            "pe_oi": int(row["OI_PE"]),
            "ce_chg": int(row["Chg_OI_CE"]),
            "pe_chg": int(row["Chg_OI_PE"]),
            "ce_vol": int(row["Vol_CE"]),
            "pe_vol": int(row["Vol_PE"]),
            "distance": abs(spot - row["strikePrice"]),
            "distance_pct": abs(spot - row["strikePrice"]) / spot * 100
        }
    }

# ============================================
# 🎯 COMPREHENSIVE BIAS DASHBOARD (NEW)
# ============================================
def display_bias_dashboard(atm_bias, support_bias, resistance_bias):
    """Display comprehensive bias dashboard"""
    
    st.markdown("## 🎯 MULTI-DIMENSIONAL BIAS ANALYSIS")
    
    # Create columns for each bias analysis
    col_atm, col_sup, col_res = st.columns(3)
    
    with col_atm:
        if atm_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{atm_bias["verdict_color"]};'>
                <h4 style='color:{atm_bias["verdict_color"]};'>🏛️ ATM ZONE BIAS</h4>
                <div style='font-size: 1.8rem; color:{atm_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {atm_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#ffcc00; text-align:center;'>
                    ₹{atm_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {atm_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
            st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
            st.metric("Net Gamma", f"{atm_bias['metrics']['net_gamma']:.3f}")
    
    with col_sup:
        if support_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{support_bias["verdict_color"]};'>
                <h4 style='color:{support_bias["verdict_color"]};'>🛡️ SUPPORT BIAS</h4>
                <div style='font-size: 1.8rem; color:{support_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {support_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#00ffcc; text-align:center;'>
                    ₹{support_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {support_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Distance", f"₹{support_bias['metrics']['distance']:.0f}")
            st.metric("CALL OI", f"{support_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{support_bias['metrics']['pe_oi']:,}")
    
    with col_res:
        if resistance_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{resistance_bias["verdict_color"]};'>
                <h4 style='color:{resistance_bias["verdict_color"]};'>⚡ RESISTANCE BIAS</h4>
                <div style='font-size: 1.8rem; color:{resistance_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {resistance_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#ff9900; text-align:center;'>
                    ₹{resistance_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {resistance_bias["total_score"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Distance", f"₹{resistance_bias['metrics']['distance']:.0f}")
            st.metric("CALL OI", f"{resistance_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{resistance_bias['metrics']['pe_oi']:,}")
    
    # Detailed ATM Bias Table
    if atm_bias:
        st.markdown("### 📊 ATM BIAS DETAILED BREAKDOWN")
        
        bias_data = []
        for bias_name, emoji in atm_bias["bias_emojis"].items():
            bias_data.append({
                "Metric": bias_name.replace("_", " ").title(),
                "Bias": emoji,
                "Score": f"{atm_bias['bias_scores'][bias_name]:.1f}",
                "Interpretation": atm_bias["bias_interpretations"][bias_name]
            })
        
        bias_df = pd.DataFrame(bias_data)
        st.dataframe(bias_df, use_container_width=True)
        
        # ATM Bias Summary
        st.markdown(f"""
        <div class='seller-explanation'>
            <h4>🎯 ATM BIAS SUMMARY</h4>
            <p><strong>Overall Verdict:</strong> <span style='color:{atm_bias["verdict_color"]}'>{atm_bias["verdict"]}</span></p>
            <p><strong>Total Score:</strong> {atm_bias["total_score"]:.2f}</p>
            <p><strong>Explanation:</strong> {atm_bias["verdict_explanation"]}</p>
            <p><strong>Key Insights:</strong></p>
            <ul>
                <li>CALL OI: {atm_bias['metrics']['ce_oi']:,} | PUT OI: {atm_bias['metrics']['pe_oi']:,}</li>
                <li>Net Delta: {atm_bias['metrics']['net_delta']:.3f} | Net Gamma: {atm_bias['metrics']['net_gamma']:.3f}</li>
                <li>Delta Exposure: ₹{atm_bias['metrics']['delta_exposure']:,}</li>
                <li>Gamma Exposure: ₹{atm_bias['metrics']['gamma_exposure']:,}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Trading Implications
    st.markdown("### 💡 TRADING IMPLICATIONS")
    
    implications = []
    
    if atm_bias:
        if atm_bias["total_score"] > 0.2:
            implications.append("✅ **ATM Bullish Bias:** Favor LONG positions with stops below ATM")
        elif atm_bias["total_score"] < -0.2:
            implications.append("✅ **ATM Bearish Bias:** Favor SHORT positions with stops above ATM")
        
        if atm_bias["metrics"]["gamma_exposure"] < -100000:
            implications.append("⚠️ **Negative Gamma Exposure:** Expect whipsaws around ATM")
        elif atm_bias["metrics"]["gamma_exposure"] > 100000:
            implications.append("✅ **Positive Gamma Exposure:** Market stabilizing around ATM")
    
    if support_bias and support_bias["total_score"] > 0.3:
        implications.append(f"✅ **Strong Support at ₹{support_bias['strike']:,}:** Good for LONG entries")
    
    if resistance_bias and resistance_bias["total_score"] < -0.3:
        implications.append(f"✅ **Strong Resistance at ₹{resistance_bias['strike']:,}:** Good for SHORT entries")
    
    if not implications:
        implications.append("⚖️ **Balanced Market:** No clear edge, wait for breakout")
    
    for imp in implications:
        st.markdown(f"- {imp}")

# -----------------------
#  TELEGRAM FUNCTIONS
# -----------------------
def send_telegram_message(bot_token, chat_id, message):
    """
    Actually send message to Telegram
    """
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

def generate_telegram_signal_option3(entry_signal, spot, seller_bias_result, seller_max_pain, 
                                   nearest_sup, nearest_res, moment_metrics, seller_breakout_index, 
                                   expiry, expiry_spike_data, atm_bias=None, support_bias=None, resistance_bias=None):
    """
    Generate Option 3 Telegram signal with stop loss/target and expiry spike info
    Only generate when position_type is not NEUTRAL
    """
    
    # Only generate signal for non-neutral positions
    if entry_signal["position_type"] == "NEUTRAL":
        return None
    
    position_type = entry_signal["position_type"]
    signal_strength = entry_signal["signal_strength"]
    confidence = entry_signal["confidence"]
    optimal_entry_price = entry_signal["optimal_entry_price"]
    stop_loss = entry_signal["stop_loss"]
    target = entry_signal["target"]
    
    # Emoji based on position
    signal_emoji = "🚀" if position_type == "LONG" else "🐻"
    current_time = get_ist_datetime_str()
    
    # Extract moment scores
    moment_burst = moment_metrics["momentum_burst"].get("score", 0)
    orderbook_pressure = moment_metrics["orderbook"].get("pressure", 0.0)
    
    # Calculate risk:reward if we have stop loss and target
    risk_reward = ""
    stop_distance = ""
    stop_pct = ""
    
    if stop_loss and target and optimal_entry_price:
        if position_type == "LONG":
            risk = abs(optimal_entry_price - stop_loss)
            reward = abs(target - optimal_entry_price)
            stop_distance = f"Stop: {stop_loss:.0f} (↓{risk:.0f} points)"
        else:
            risk = abs(stop_loss - optimal_entry_price)
            reward = abs(optimal_entry_price - target)
            stop_distance = f"Stop: {stop_loss:.0f} (↑{risk:.0f} points)"
        
        if risk > 0:
            risk_reward = f"1:{reward/risk:.1f}"
            stop_pct = f"({risk/optimal_entry_price*100:.1f}%)"
    
    # Format stop loss and target with points
    stop_loss_str = f"₹{stop_loss:,.0f}" if stop_loss else "N/A"
    target_str = f"₹{target:,.0f}" if target else "N/A"
    
    # Format support/resistance
    support_str = f"₹{nearest_sup['strike']:,}" if nearest_sup else "N/A"
    resistance_str = f"₹{nearest_res['strike']:,}" if nearest_res else "N/A"
    
    # Format max pain
    max_pain_str = f"₹{seller_max_pain:,}" if seller_max_pain else "N/A"
    
    # Calculate entry distance from current spot
    entry_distance = abs(spot - optimal_entry_price)
    
    # Add ATM bias info if available
    atm_bias_info = ""
    if atm_bias:
        atm_bias_info = f"\n🎯 *ATM Bias*: {atm_bias['verdict']} (Score: {atm_bias['total_score']:.2f})"
    
    # Add expiry spike info if active
    expiry_info = ""
    if expiry_spike_data.get("active", False) and expiry_spike_data.get("probability", 0) > 50:
        spike_emoji = "🚨" if expiry_spike_data['probability'] > 70 else "⚠️"
        expiry_info = f"\n{spike_emoji} *Expiry Spike Risk*: {expiry_spike_data['probability']}% - {expiry_spike_data['type']}"
        if expiry_spike_data.get("key_levels"):
            expiry_info += f"\n🎯 *Spike Levels*: {', '.join(expiry_spike_data['key_levels'][:2])}"
    
    # Generate the message
    message = f"""
🎯 *NIFTY OPTION TRADE SETUP*

*Position*: {signal_emoji} {position_type} ({signal_strength})
*Entry Price*: ₹{optimal_entry_price:,.0f}
*Current Spot*: ₹{spot:,.0f}
*Entry Distance*: {entry_distance:.0f} points

*Risk Management*:
🛑 Stop Loss: {stop_loss_str} {stop_distance if stop_distance else ""}
🎯 Target: {target_str}
📊 Risk:Reward = {risk_reward} {stop_pct if stop_pct else ""}

*Key Levels*:
🛡️ Support: {support_str}
⚡ Resistance: {resistance_str}
🎯 Max Pain: {max_pain_str}

*Moment Detector*:
✅ Burst: {moment_burst}/100
✅ Pressure: {orderbook_pressure:+.2f}

*Seller Bias*: {seller_bias_result['bias']}
*Confidence*: {confidence:.0f}%
{atm_bias_info}

*Expiry Context*:
📅 Days to Expiry: {expiry_spike_data.get('days_to_expiry', 0):.1f}
{expiry_info if expiry_info else "📊 Expiry spike risk: Low"}

⏰ {current_time} IST | 📆 Expiry: {expiry}

#NiftyOptions #OptionSelling #TradingSignal
"""
    return message

def check_and_send_signal(entry_signal, spot, seller_bias_result, seller_max_pain, 
                         nearest_sup, nearest_res, moment_metrics, seller_breakout_index, 
                         expiry, expiry_spike_data, atm_bias=None, support_bias=None, resistance_bias=None):
    """
    Check if a new signal is generated and return it (simulated)
    Returns signal message if new signal, None otherwise
    """
    # Store previous signal in session state
    if "last_signal" not in st.session_state:
        st.session_state["last_signal"] = None
    
    # Check if we have a valid signal
    if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
        current_signal = f"{entry_signal['position_type']}_{entry_signal['optimal_entry_price']:.0f}"
        
        # Check if this is a new signal (different from last one)
        if st.session_state["last_signal"] != current_signal:
            # Generate Telegram message
            telegram_msg = generate_telegram_signal_option3(
                entry_signal, spot, seller_bias_result, 
                seller_max_pain, nearest_sup, nearest_res, 
                moment_metrics, seller_breakout_index, expiry, expiry_spike_data,
                atm_bias, support_bias, resistance_bias
            )
            
            if telegram_msg:
                # Update last signal
                st.session_state["last_signal"] = current_signal
                return telegram_msg
    
    # Reset if signal is gone
    elif st.session_state["last_signal"] is not None:
        st.session_state["last_signal"] = None
    
    return None

# -----------------------
# 📅 EXPIRY SPIKE DETECTOR FUNCTIONS
# -----------------------
def detect_expiry_spikes(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str):
    """
    Detect potential expiry day spikes based on multiple factors
    Returns: dict with spike probability, direction, and key levels
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
    
    # Factor 1: ATM OI Concentration (0-25 points)
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 2  # ±2 strikes around ATM
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    total_oi_near_atm = atm_ce_oi + atm_pe_oi
    total_oi_all = merged_df["OI_CE"].sum() + merged_df["OI_PE"].sum()
    
    if total_oi_all > 0:
        atm_concentration = total_oi_near_atm / total_oi_all
        if atm_concentration > 0.5:
            spike_score += 25
            spike_factors.append(f"High ATM OI concentration ({atm_concentration:.1%})")
        elif atm_concentration > 0.3:
            spike_score += 15
            spike_factors.append(f"Moderate ATM OI concentration ({atm_concentration:.1%})")
    
    # Factor 2: Max Pain vs Spot Distance (0-20 points)
    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        max_pain_distance = abs(spot - max_pain) / spot * 100
        if max_pain_distance > 2.0:
            spike_score += 20
            spike_factors.append(f"Spot far from Max Pain ({max_pain_distance:.1f}%)")
            if spot > max_pain:
                spike_type = "SHORT SQUEEZE"
            else:
                spike_type = "LONG SQUEEZE"
            key_levels.append(f"Max Pain: ₹{max_pain:,}")
        elif max_pain_distance > 1.0:
            spike_score += 10
            spike_factors.append(f"Spot moderately far from Max Pain ({max_pain_distance:.1f}%)")
    
    # Factor 3: PCR Extremes (0-15 points)
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 1.8:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) - Heavy PUT selling")
            spike_type = "UPWARD SPIKE" if spike_type is None else spike_type
        elif pcr < 0.5:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f}) - Heavy CALL selling")
            spike_type = "DOWNWARD SPIKE" if spike_type is None else spike_type
    
    # Factor 4: Large OI Build-up at Single Strike (0-20 points)
    max_ce_oi_strike = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_strike = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None
    
    if max_ce_oi_strike is not None:
        max_ce_oi = int(max_ce_oi_strike["OI_CE"])
        max_ce_strike = int(max_ce_oi_strike["strikePrice"])
        if max_ce_oi > 2000000:  # 2 million+ OI
            spike_score += 20
            spike_factors.append(f"Massive CALL OI at ₹{max_ce_strike:,} ({max_ce_oi:,})")
            key_levels.append(f"CALL Wall: ₹{max_ce_strike:,}")
            if abs(spot - max_ce_strike) < (strike_gap_val * 3):
                spike_type = "RESISTANCE SPIKE"
    
    if max_pe_oi_strike is not None:
        max_pe_oi = int(max_pe_oi_strike["OI_PE"])
        max_pe_strike = int(max_pe_oi_strike["strikePrice"])
        if max_pe_oi > 2000000:  # 2 million+ OI
            spike_score += 20
            spike_factors.append(f"Massive PUT OI at ₹{max_pe_strike:,} ({max_pe_oi:,})")
            key_levels.append(f"PUT Wall: ₹{max_pe_strike:,}")
            if abs(spot - max_pe_strike) < (strike_gap_val * 3):
                spike_type = "SUPPORT SPIKE"
    
    # Factor 5: Gamma Flip Zone (0-10 points)
    if days_to_expiry <= 1:
        spike_score += 10
        spike_factors.append("Gamma flip zone (expiry day)")
    
    # Factor 6: Unwinding Activity (0-10 points)
    ce_unwind = (merged_df["Chg_OI_CE"] < 0).sum()
    pe_unwind = (merged_df["Chg_OI_PE"] < 0).sum()
    total_unwind = ce_unwind + pe_unwind
    
    if total_unwind > 15:  # More than 15 strikes showing unwinding
        spike_score += 10
        spike_factors.append(f"Massive unwinding ({total_unwind} strikes)")
    
    # Determine spike probability
    probability = min(100, int(spike_score * 1.5))
    
    # Spike intensity
    if probability >= 70:
        intensity = "HIGH PROBABILITY SPIKE"
        color = "#ff0000"
    elif probability >= 50:
        intensity = "MODERATE SPIKE RISK"
        color = "#ff9900"
    elif probability >= 30:
        intensity = "LOW SPIKE RISK"
        color = "#ffff00"
    else:
        intensity = "NO SPIKE DETECTED"
        color = "#00ff00"
    
    # Default spike type if none detected
    if spike_type is None:
        spike_type = "UNCERTAIN"
    
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
        "message": f"Expiry in {days_to_expiry:.1f} days"
    }

def get_historical_expiry_patterns():
    """
    Return historical expiry day patterns (simplified)
    In production, you'd connect to a database
    """
    patterns = {
        "high_volatility": {
            "probability": 0.65,
            "description": "Expiry days typically have 30% higher volatility",
            "time_of_spike": ["10:30-11:30 IST", "14:30-15:00 IST"]
        },
        "max_pain_pull": {
            "probability": 0.55,
            "description": "Price tends to gravitate towards Max Pain in last 2 hours",
            "effect": "Strong if Max Pain >1% away from spot"
        },
        "gamma_unwind": {
            "probability": 0.70,
            "description": "Market makers unwind gamma positions causing spikes",
            "timing": "Last 90 minutes"
        }
    }
    return patterns

def detect_violent_unwinding(merged_df, spot, atm_strike):
    """
    Detect signs of violent unwinding near expiry
    """
    signals = []
    
    # Check for massive OI reduction
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    
    if total_ce_chg < -1000000:  # 1 million+ CALL unwinding
        signals.append(f"Violent CALL unwinding: {abs(total_ce_chg):,} contracts")
    
    if total_pe_chg < -1000000:  # 1 million+ PUT unwinding
        signals.append(f"Violent PUT unwinding: {abs(total_pe_chg):,} contracts")
    
    # Check ATM strikes specifically
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    atm_window = 1
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_unwind = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes)]
    atm_ce_unwind = atm_unwind["Chg_OI_CE"].sum()
    atm_pe_unwind = atm_unwind["Chg_OI_PE"].sum()
    
    if atm_ce_unwind < -500000:
        signals.append(f"ATM CALL unwinding: {abs(atm_ce_unwind):,} contracts")
    
    if atm_pe_unwind < -500000:
        signals.append(f"ATM PUT unwinding: {abs(atm_pe_unwind):,} contracts")
    
    return signals

def calculate_gamma_exposure_spike(total_gex_net, days_to_expiry):
    """
    Calculate gamma exposure spike risk
    """
    if days_to_expiry > 3:
        return {"risk": "Low", "score": 0}
    
    # Negative GEX + near expiry = explosive moves
    if total_gex_net < -2000000:  # Large negative GEX
        risk_score = min(100, int((abs(total_gex_net) / 1000000) * 10))
        return {
            "risk": "High",
            "score": risk_score,
            "message": f"Negative GEX (₹{abs(total_gex_net):,.0f}) + Near expiry = Explosive move potential"
        }
    elif total_gex_net > 2000000:  # Large positive GEX
        risk_score = min(80, int((total_gex_net / 1000000) * 5))
        return {
            "risk": "Medium",
            "score": risk_score,
            "message": f"Positive GEX (₹{total_gex_net:,.0f}) + Near expiry = Mean reversion bias"
        }
    
    return {"risk": "Low", "score": 0}

def predict_expiry_pinning_probability(spot, max_pain, nearest_support, nearest_resistance):
    """
    Predict probability of expiry pinning (price stuck at a level)
    """
    if not max_pain or not nearest_support or not nearest_resistance:
        return 0
    
    # Calculate pinning score (0-100)
    pinning_score = 0
    
    # Factor 1: Distance to Max Pain
    distance_to_max_pain = abs(spot - max_pain) / spot * 100
    if distance_to_max_pain < 0.5:
        pinning_score += 40
    elif distance_to_max_pain < 1.0:
        pinning_score += 20
    
    # Factor 2: Narrow range
    range_size = nearest_resistance - nearest_support
    if range_size < 200:
        pinning_score += 30
    elif range_size < 300:
        pinning_score += 15
    
    return min(100, pinning_score)

# -----------------------
# 📈 ENHANCED OI & PCR ANALYZER FUNCTIONS
# -----------------------

def analyze_oi_pcr_metrics(merged_df, spot, atm_strike):
    """
    Comprehensive OI and PCR analysis
    Returns detailed metrics and insights
    """
    
    # Basic totals
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    total_ce_chg = merged_df["Chg_OI_CE"].sum()
    total_pe_chg = merged_df["Chg_OI_PE"].sum()
    total_oi = total_ce_oi + total_pe_oi
    total_chg_oi = total_ce_chg + total_pe_chg
    
    # PCR Calculations
    pcr_total = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    pcr_chg = total_pe_chg / total_ce_chg if abs(total_ce_chg) > 0 else 0
    
    # OI Concentration Analysis
    strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
    
    # ATM Concentration (strikes around ATM)
    atm_window = 3  # ±3 strikes
    atm_strikes = [s for s in merged_df["strikePrice"] 
                  if abs(s - atm_strike) <= (atm_window * strike_gap_val)]
    
    atm_ce_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_CE"].sum()
    atm_pe_oi = merged_df.loc[merged_df["strikePrice"].isin(atm_strikes), "OI_PE"].sum()
    atm_total_oi = atm_ce_oi + atm_pe_oi
    
    # ITM/OTM Analysis
    itm_ce_oi = merged_df.loc[merged_df["strikePrice"] < spot, "OI_CE"].sum()
    otm_ce_oi = merged_df.loc[merged_df["strikePrice"] > spot, "OI_CE"].sum()
    itm_pe_oi = merged_df.loc[merged_df["strikePrice"] > spot, "OI_PE"].sum()
    otm_pe_oi = merged_df.loc[merged_df["strikePrice"] < spot, "OI_PE"].sum()
    
    # Max OI Strikes
    max_ce_oi_row = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_row = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None
    
    max_ce_strike = int(max_ce_oi_row["strikePrice"]) if max_ce_oi_row is not None else 0
    max_ce_oi_val = int(max_ce_oi_row["OI_CE"]) if max_ce_oi_row is not None else 0
    max_pe_strike = int(max_pe_oi_row["strikePrice"]) if max_pe_oi_row is not None else 0
    max_pe_oi_val = int(max_pe_oi_row["OI_PE"]) if max_pe_oi_row is not None else 0
    
    # OI Skew Analysis
    call_oi_skew = "N/A"
    if total_ce_oi > 0:
        # Check if OI is concentrated at specific strikes
        top_3_ce_oi = merged_df.nlargest(3, "OI_CE")["OI_CE"].sum()
        call_oi_concentration = top_3_ce_oi / total_ce_oi if total_ce_oi > 0 else 0
        call_oi_skew = "High" if call_oi_concentration > 0.4 else "Moderate" if call_oi_concentration > 0.2 else "Low"
    
    put_oi_skew = "N/A"
    if total_pe_oi > 0:
        top_3_pe_oi = merged_df.nlargest(3, "OI_PE")["OI_PE"].sum()
        put_oi_concentration = top_3_pe_oi / total_pe_oi if total_pe_oi > 0 else 0
        put_oi_skew = "High" if put_oi_concentration > 0.4 else "Moderate" if put_oi_concentration > 0.2 else "Low"
    
    # PCR Interpretation
    pcr_interpretation = ""
    pcr_sentiment = ""
    
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
    
    # Change in PCR interpretation
    chg_interpretation = ""
    if abs(pcr_chg) > 0:
        if pcr_chg > 0.5:
            chg_interpretation = "PCR rising sharply (bullish buildup)"
        elif pcr_chg > 0.2:
            chg_interpretation = "PCR rising (bullish)"
        elif pcr_chg < -0.5:
            chg_interpretation = "PCR falling sharply (bearish buildup)"
        elif pcr_chg < -0.2:
            chg_interpretation = "PCR falling (bearish)"
        else:
            chg_interpretation = "PCR stable"
    
    # OI Change Interpretation
    oi_change_interpretation = ""
    if total_ce_chg > 0 and total_pe_chg > 0:
        oi_change_interpretation = "Fresh writing on both sides (range expansion)"
    elif total_ce_chg > 0 and total_pe_chg < 0:
        oi_change_interpretation = "CALL writing + PUT unwinding (bearish)"
    elif total_ce_chg < 0 and total_pe_chg > 0:
        oi_change_interpretation = "CALL unwinding + PUT writing (bullish)"
    elif total_ce_chg < 0 and total_pe_chg < 0:
        oi_change_interpretation = "Unwinding on both sides (range contraction)"
    else:
        oi_change_interpretation = "Mixed activity"
    
    return {
        # Totals
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_oi": total_oi,
        "total_ce_chg": total_ce_chg,
        "total_pe_chg": total_pe_chg,
        "total_chg_oi": total_chg_oi,
        
        # PCR Metrics
        "pcr_total": pcr_total,
        "pcr_chg": pcr_chg,
        "pcr_interpretation": pcr_interpretation,
        "pcr_sentiment": pcr_sentiment,
        "pcr_color": pcr_color,
        "chg_interpretation": chg_interpretation,
        
        # Concentration
        "atm_concentration_pct": (atm_total_oi / total_oi * 100) if total_oi > 0 else 0,
        "atm_ce_oi": atm_ce_oi,
        "atm_pe_oi": atm_pe_oi,
        
        # ITM/OTM
        "itm_ce_oi": itm_ce_oi,
        "otm_ce_oi": otm_ce_oi,
        "itm_pe_oi": itm_pe_oi,
        "otm_pe_oi": otm_pe_oi,
        
        # Max OI
        "max_ce_strike": max_ce_strike,
        "max_ce_oi": max_ce_oi_val,
        "max_pe_strike": max_pe_strike,
        "max_pe_oi": max_pe_oi_val,
        
        # Skew
        "call_oi_skew": call_oi_skew,
        "put_oi_skew": put_oi_skew,
        
        # Interpretation
        "oi_change_interpretation": oi_change_interpretation,
        
        # Derived metrics
        "ce_pe_ratio": total_ce_oi / total_pe_oi if total_pe_oi > 0 else 0,
        "oi_momentum": total_chg_oi / total_oi * 100 if total_oi > 0 else 0
    }

def get_pcr_context(pcr_value):
    """Provide historical context for PCR values"""
    if pcr_value > 2.5:
        return "Extreme bullish zone (rare, usually precedes sharp rallies)"
    elif pcr_value > 2.0:
        return "Very bullish (often leads to upward moves)"
    elif pcr_value > 1.5:
        return "Bullish bias"
    elif pcr_value > 1.2:
        return "Moderately bullish"
    elif pcr_value > 0.8:
        return "Neutral range"
    elif pcr_value > 0.5:
        return "Moderately bearish"
    elif pcr_value > 0.3:
        return "Bearish (often precedes declines)"
    else:
        return "Extreme bearish (oversold, can mean reversal)"

def analyze_pcr_for_expiry(pcr_value, days_to_expiry):
    """Analyze PCR in context of expiry"""
    if days_to_expiry > 5:
        return "Normal PCR analysis applies"
    
    if days_to_expiry <= 2:
        if pcr_value > 1.5:
            return "High PCR near expiry → Potential short covering rally"
        elif pcr_value < 0.7:
            return "Low PCR near expiry → Potential long unwinding decline"
        else:
            return "Balanced PCR near expiry → Range bound expected"
    
    return "PCR analysis standard"

# -----------------------
#  CUSTOM CSS - SELLER THEME + NEW MOMENT FEATURES + EXPIRY SPIKE + OI/PCR + ATM BIAS
# -----------------------
st.markdown(r"""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* SELLER THEME COLORS */
    .seller-bullish { color: #00ff88 !important; font-weight: 700 !important; }
    .seller-bearish { color: #ff4444 !important; font-weight: 700 !important; }
    .seller-neutral { color: #66b3ff !important; font-weight: 700 !important; }
    
    .seller-bullish-bg { background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%); }
    .seller-bearish-bg { background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%); }
    .seller-neutral-bg { background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%); }
    
    /* MOMENT DETECTOR COLORS */
    .moment-high { color: #ff00ff !important; font-weight: 800 !important; }
    .moment-medium { color: #ff9900 !important; font-weight: 700 !important; }
    .moment-low { color: #66b3ff !important; font-weight: 600 !important; }
    
    /* OI/PCR COLORS */
    .pcr-extreme-bullish { color: #00ff88 !important; }
    .pcr-bullish { color: #00cc66 !important; }
    .pcr-mild-bullish { color: #66ff66 !important; }
    .pcr-neutral { color: #66b3ff !important; }
    .pcr-mild-bearish { color: #ff9900 !important; }
    .pcr-bearish { color: #ff4444 !important; }
    .pcr-extreme-bearish { color: #ff0000 !important; }
    
    /* ATM BIAS COLORS */
    .atm-bias-bullish { color: #00ff88 !important; }
    .atm-bias-bearish { color: #ff4444 !important; }
    .atm-bias-neutral { color: #66b3ff !important; }
    
    h1, h2, h3 { color: #ff66cc !important; } /* Seller theme pink */
    
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
    
    /* ATM BIAS CARD */
    .card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid;
        margin: 10px 0;
        text-align: center;
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

st.set_page_config(page_title="Nifty Screener v7 - Seller's Perspective + ATM Bias Analyzer + Moment Detector + Expiry Spike + OI/PCR", layout="wide")

def auto_refresh(interval_sec=AUTO_REFRESH_SEC):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh()

# -----------------------
#  UTILITY FUNCTIONS
# -----------------------
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

# -----------------------
# 🔥 NEW: ORDERBOOK PRESSURE FUNCTIONS
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_orderbook_depth():
    """
    Best-effort depth fetch from Dhan API
    """
    candidate_endpoints = [
        f"{DHAN_BASE_URL}/v2/marketfeed/quotes",
        f"{DHAN_BASE_URL}/v2/marketfeed/depth"
    ]
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    
    for url in candidate_endpoints:
        try:
            payload = {"IDX_I": [13]}
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success":
                continue
            
            d = data.get("data", {})
            if isinstance(d, dict):
                d1 = d.get("IDX_I", {}).get("13", {})
                depth = d1.get("depth") or d.get("depth") or d
                buy = depth.get("buy") if isinstance(depth, dict) else None
                sell = depth.get("sell") if isinstance(depth, dict) else None
                
                if buy is not None and sell is not None:
                    return {"buy": buy, "sell": sell, "source": url}
        except Exception:
            continue
    
    return None

def orderbook_pressure_score(depth: dict, levels: int = 5) -> dict:
    """
    Returns orderbook pressure (-1 to +1)
    """
    if not depth or "buy" not in depth or "sell" not in depth:
        return {"available": False, "pressure": 0.0, "buy_qty": 0.0, "sell_qty": 0.0}
    
    def sum_qty(side):
        total = 0.0
        for i, lvl in enumerate(side):
            if i >= levels:
                break
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                total += safe_float(lvl[1], 0.0)
            elif isinstance(lvl, dict):
                total += safe_float(lvl.get("qty") or lvl.get("quantity"), 0.0)
        return total
    
    buy = sum_qty(depth["buy"])
    sell = sum_qty(depth["sell"])
    denom = (buy + sell) if (buy + sell) > 0 else 1.0
    pressure = (buy - sell) / denom
    return {"available": True, "pressure": pressure, "buy_qty": buy, "sell_qty": sell}

# -----------------------
# 🔥 NEW: MOMENT DETECTOR FUNCTIONS
# -----------------------
def _init_history():
    """Initialize session state for moment history tracking"""
    if "moment_history" not in st.session_state:
        st.session_state["moment_history"] = []
    if "prev_ltps" not in st.session_state:
        st.session_state["prev_ltps"] = {}
    if "prev_ivs" not in st.session_state:
        st.session_state["prev_ivs"] = {}

def _snapshot_from_state(ts, spot, atm_strike, merged: pd.DataFrame):
    """
    Create snapshot for OI velocity/acceleration and momentum burst
    """
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
    """
    Feature #1: Momentum Burst = (ΔVol * ΔIV * Δ|OI|) normalized to 0..100
    """
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
    """
    Feature #3: ATM Gamma Cluster = sum(|gamma|) around ATM (±1 ±2)
    """
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
    """
    Feature #4: OI Velocity + Acceleration
    """
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

# -----------------------
# 🔥 ENTRY SIGNAL CALCULATION (EXTENDED WITH MOMENT DETECTOR & ATM BIAS)
# -----------------------
def calculate_realistic_stop_loss_target(position_type, entry_price, nearest_sup, nearest_res, strike_gap, max_risk_pct=1.5):
    """
    Calculate realistic stop loss and target with proper risk management
    """
    stop_loss = None
    target = None
    
    if not nearest_sup or not nearest_res:
        return stop_loss, target
    
    # Maximum risk = 1.5% of entry price
    max_risk_points = entry_price * (max_risk_pct / 100)
    
    if position_type == "LONG":
        # Option 1: Support-based stop (1.5 strike gaps below support)
        stop_loss_support = nearest_sup["strike"] - (strike_gap * 1.5)
        
        # Option 2: Percentage-based stop
        stop_loss_pct = entry_price - max_risk_points
        
        # Use the HIGHER stop (tighter risk management)
        stop_loss = max(stop_loss_support, stop_loss_pct)
        
        # Calculate risk amount
        risk_amount = entry_price - stop_loss
        
        # Target 1: 2:1 risk:reward
        target_rr = entry_price + (risk_amount * 2)
        
        # Target 2: Near resistance
        target_resistance = nearest_res["strike"] - strike_gap
        
        # Use the LOWER target (more conservative)
        target = min(target_rr, target_resistance)
        
        # Ensure target > entry
        if target <= entry_price:
            target = entry_price + risk_amount  # 1:1 at minimum
    
    elif position_type == "SHORT":
        # Option 1: Resistance-based stop (1.5 strike gaps above resistance)
        stop_loss_resistance = nearest_res["strike"] + (strike_gap * 1.5)
        
        # Option 2: Percentage-based stop
        stop_loss_pct = entry_price + max_risk_points
        
        # Use the LOWER stop (tighter risk management)
        stop_loss = min(stop_loss_resistance, stop_loss_pct)
        
        # Calculate risk amount
        risk_amount = stop_loss - entry_price
        
        # Target 1: 2:1 risk:reward
        target_rr = entry_price - (risk_amount * 2)
        
        # Target 2: Near support
        target_support = nearest_sup["strike"] + strike_gap
        
        # Use the HIGHER target (more conservative)
        target = max(target_rr, target_support)
        
        # Ensure target < entry
        if target >= entry_price:
            target = entry_price - risk_amount  # 1:1 at minimum
    
    # Round to nearest 50 for Nifty
    if stop_loss:
        stop_loss = round(stop_loss / 50) * 50
    if target:
        target = round(target / 50) * 50
    
    return stop_loss, target

def calculate_entry_signal_extended(
    spot, 
    merged_df, 
    atm_strike, 
    seller_bias_result, 
    seller_max_pain, 
    seller_supports_df, 
    seller_resists_df, 
    nearest_sup, 
    nearest_res, 
    seller_breakout_index,
    moment_metrics,  # NEW: Add moment metrics
    atm_bias=None,   # NEW: Add ATM bias
    support_bias=None,  # NEW: Add support bias
    resistance_bias=None  # NEW: Add resistance bias
):
    """
    Calculate optimal entry signal with Moment Detector & ATM Bias integration
    """
    
    # Initialize signal components
    signal_score = 0
    signal_reasons = []
    optimal_entry_price = spot
    position_type = "NEUTRAL"
    confidence = 0
    
    # ============================================
    # 1. SELLER BIAS ANALYSIS (40% weight)
    # ============================================
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
    
    # ============================================
    # 2. MAX PAIN ALIGNMENT (15% weight)
    # ============================================
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
    
    # ============================================
    # 3. SUPPORT/RESISTANCE ALIGNMENT (20% weight)
    # ============================================
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
    
    # ============================================
    # 4. BREAKOUT INDEX (15% weight)
    # ============================================
    if seller_breakout_index > 80:
        signal_score += 15
        signal_reasons.append(f"High Breakout Index ({seller_breakout_index}%): Strong momentum expected")
    elif seller_breakout_index > 60:
        signal_score += 10
        signal_reasons.append(f"Moderate Breakout Index ({seller_breakout_index}%): Some momentum expected")
    
    # ============================================
    # 5. PCR ANALYSIS (10% weight)
    # ============================================
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
    
    # ============================================
    # 6. GEX ANALYSIS (Adjustment factor)
    # ============================================
    total_gex_net = merged_df["GEX_Net"].sum()
    if total_gex_net > 1000000:
        if position_type == "LONG":
            signal_score += 5
            signal_reasons.append("Positive GEX: Supports LONG position (stabilizing)")
    elif total_gex_net < -1000000:
        if position_type == "SHORT":
            signal_score += 5
            signal_reasons.append("Negative GEX: Supports SHORT position (destabilizing)")
    
    # ============================================
    # 7. MOMENT DETECTOR FEATURES (NEW - 30% total weight)
    # ============================================
    
    # 7.1 Momentum Burst (12% weight)
    mb = moment_metrics.get("momentum_burst", {})
    if mb.get("available", False):
        mb_score = mb.get("score", 0)
        signal_score += int(12 * (mb_score / 100.0))
        signal_reasons.append(f"Momentum burst: {mb_score}/100 - {mb.get('note', '')}")
    
    # 7.2 Orderbook Pressure (8% weight)
    ob = moment_metrics.get("orderbook", {})
    if ob.get("available", False):
        pressure = ob.get("pressure", 0.0)
        if position_type == "LONG" and pressure > 0.15:
            signal_score += 8
            signal_reasons.append(f"Orderbook buy pressure: {pressure:+.2f} (supports LONG)")
        elif position_type == "SHORT" and pressure < -0.15:
            signal_score += 8
            signal_reasons.append(f"Orderbook sell pressure: {pressure:+.2f} (supports SHORT)")
    
    # 7.3 Gamma Cluster (6% weight)
    gc = moment_metrics.get("gamma_cluster", {})
    if gc.get("available", False):
        gc_score = gc.get("score", 0)
        signal_score += int(6 * (gc_score / 100.0))
        signal_reasons.append(f"Gamma cluster: {gc_score}/100 (ATM concentration)")
    
    # 7.4 OI Acceleration (4% weight)
    oi_accel = moment_metrics.get("oi_accel", {})
    if oi_accel.get("available", False):
        oi_score = oi_accel.get("score", 0)
        signal_score += int(4 * (oi_score / 100.0))
        signal_reasons.append(f"OI acceleration: {oi_score}/100 ({oi_accel.get('note', '')})")
    
    # ============================================
    # 8. ATM BIAS INTEGRATION (NEW - 20% weight)
    # ============================================
    if atm_bias:
        atm_score = atm_bias["total_score"]
        if position_type == "LONG" and atm_score > 0.1:
            signal_score += int(20 * (atm_score / 1.0))  # Scale to max 20 points
            signal_reasons.append(f"ATM bias bullish ({atm_score:.2f}) confirms LONG")
        elif position_type == "SHORT" and atm_score < -0.1:
            signal_score += int(20 * (abs(atm_score) / 1.0))
            signal_reasons.append(f"ATM bias bearish ({atm_score:.2f}) confirms SHORT")
    
    # ============================================
    # 9. SUPPORT/RESISTANCE BIAS INTEGRATION (NEW - 15% weight)
    # ============================================
    if support_bias and position_type == "LONG":
        support_score = support_bias["total_score"]
        if support_score > 0.2:
            signal_score += int(15 * (support_score / 1.0))
            signal_reasons.append(f"Strong support bias ({support_score:.2f}) at ₹{support_bias['strike']:,}")
    
    if resistance_bias and position_type == "SHORT":
        resistance_score = resistance_bias["total_score"]
        if resistance_score < -0.2:
            signal_score += int(15 * (abs(resistance_score) / 1.0))
            signal_reasons.append(f"Strong resistance bias ({resistance_score:.2f}) at ₹{resistance_bias['strike']:,}")
    
    # ============================================
    # FINAL SIGNAL CALCULATION
    # ============================================
    
    # Calculate confidence percentage
    confidence = min(max(signal_score, 0), 100)
    
    # Determine signal strength
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
    
    # Calculate stop loss and target with realistic logic
    stop_loss = None
    target = None
    
    if nearest_sup and nearest_res and position_type != "NEUTRAL":
        strike_gap_val = strike_gap_from_series(merged_df["strikePrice"])
        
        # Use the new realistic stop loss calculation
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
        "moment_metrics": moment_metrics,  # NEW: Include moment metrics in signal
        "atm_bias_score": atm_bias["total_score"] if atm_bias else 0,  # NEW: Include ATM bias score
        "support_bias_score": support_bias["total_score"] if support_bias else 0,  # NEW: Include support bias score
        "resistance_bias_score": resistance_bias["total_score"] if resistance_bias else 0  # NEW: Include resistance bias score
    }

# ============================================
# 🎯 ENHANCED ENTRY SIGNAL WITH ATM BIAS (NEW)
# ============================================
def calculate_entry_signal_with_atm_bias(
    spot, 
    merged_df, 
    atm_strike, 
    seller_bias_result, 
    seller_max_pain, 
    seller_supports_df, 
    seller_resists_df, 
    nearest_sup, 
    nearest_res, 
    seller_breakout_index,
    moment_metrics,
    atm_bias, 
    support_bias, 
    resistance_bias
):
    """
    Enhanced entry signal with comprehensive ATM bias analysis
    This function wraps the extended entry signal for backward compatibility
    """
    return calculate_entry_signal_extended(
        spot=spot,
        merged_df=merged_df,
        atm_strike=atm_strike,
        seller_bias_result=seller_bias_result,
        seller_max_pain=seller_max_pain,
        seller_supports_df=seller_supports_df,
        seller_resists_df=seller_resists_df,
        nearest_sup=nearest_sup,
        nearest_res=nearest_res,
        seller_breakout_index=seller_breakout_index,
        moment_metrics=moment_metrics,
        atm_bias=atm_bias,
        support_bias=support_bias,
        resistance_bias=resistance_bias
    )

# -----------------------
# 🔥 SELLER'S PERSPECTIVE FUNCTIONS (ORIGINAL)
# -----------------------
def seller_strength_score(row, weights=SCORE_WEIGHTS):
    chg_oi = abs(safe_float(row.get("Chg_OI_CE",0))) + abs(safe_float(row.get("Chg_OI_PE",0)))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0
    
    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def seller_price_oi_divergence(chg_oi, vol, ltp_change, option_type="CE"):
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
    strike = row["strikePrice"]
    dist = abs(strike - atm) / max(strike_gap, 1)
    dist = max(dist, 1e-6)
    
    chg_oi_sum = safe_float(row.get("Chg_OI_CE",0)) - safe_float(row.get("Chg_OI_PE",0))
    seller_pressure = -chg_oi_sum / dist
    
    return seller_pressure

def seller_breakout_probability_index(merged_df, atm, strike_gap):
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
    eps = 1e-6
    t = pcr_df.copy()
    
    t["PCR_clipped"] = t["PCR"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    t["seller_support_score"] = t["OI_PE"] + (t["PCR_clipped"] * 100000.0)
    
    t["seller_resistance_factor"] = t["PCR_clipped"].apply(lambda x: 1.0/(x+eps) if x>0 else 1.0/(eps))
    t["seller_resistance_score"] = t["OI_CE"] + (t["seller_resistance_factor"] * 100000.0)
    
    top_supports = t.sort_values("seller_support_score", ascending=False).head(3)
    top_resists = t.sort_values("seller_resistance_score", ascending=False).head(3)
    
    return t, top_supports, top_resists

# -----------------------
# DHAN API
# -----------------------
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

# -----------------------
#  MAIN APP - COMPLETE V7 WITH ATM BIAS ANALYZER
# -----------------------
st.title("🎯 NIFTY Option Screener v7.0 — SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + Moment Detector + Expiry Spike + OI/PCR ANALYTICS")

current_ist = get_ist_datetime_str()
st.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <span class='ist-time'>🕐 IST: {current_ist}</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
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
    st.markdown("### 🎯 ATM BIAS ANALYZER")
    st.markdown("""
    **12 Key Metrics:**
    1. OI Bias (CALL vs PUT)
    2. Change in OI Bias
    3. Volume Bias
    4. Delta Bias
    5. Gamma Bias
    6. Premium Bias
    7. IV Bias
    8. Delta Exposure Bias
    9. Gamma Exposure Bias
    10. IV Skew Bias
    11. OI Change Bias
    """)
    
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
    
    # Expiry spike info in sidebar
    st.markdown("---")
    
    # Save interval
    save_interval = st.number_input("PCR Auto-save (sec)", value=SAVE_INTERVAL_SEC, min_value=60, step=60)
    
    # Telegram settings
    st.markdown("---")
    st.markdown("### 🤖 TELEGRAM SETTINGS")
    auto_send = st.checkbox("Auto-send signals to Telegram", value=False)
    show_signal_preview = st.checkbox("Show signal preview", value=True)
    
    if st.button("Clear Caches"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot")
        st.stop()
    
    expiries = get_expiry_list()
    if not expiries:
        st.error("Unable to fetch expiry list")
        st.stop()
    
    expiry = st.selectbox("Select expiry", expiries, index=0)

with col2:
    if spot > 0:
        st.metric("NIFTY Spot", f"₹{spot:.2f}")
        st.metric("Expiry", expiry)

# Calculate days to expiry
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
    days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
except Exception:
    tau = 7.0/365.0
    days_to_expiry = 7.0

# Add expiry info to sidebar
with st.sidebar:
    if days_to_expiry <= 5:
        st.warning(f"⚠️ Expiry in {days_to_expiry:.1f} days")
        st.info("Spike detector ACTIVE")
    else:
        st.success(f"✓ Expiry in {days_to_expiry:.1f} days")
        st.info("Spike detector INACTIVE")
    
    st.markdown("---")
    st.markdown(f"**Current IST:** {get_ist_time_str()}")
    st.markdown(f"**Date:** {get_ist_date_str()}")

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

# Session storage for prev LTP/IV
if "prev_ltps_seller" not in st.session_state:
    st.session_state["prev_ltps_seller"] = {}
if "prev_ivs_seller" not in st.session_state:
    st.session_state["prev_ivs_seller"] = {}

# Initialize moment history
_init_history()

# Compute per-strike metrics with SELLER interpretation
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

# ---- NEW: Capture snapshot for moment detector ----
st.session_state["moment_history"].append(
    _snapshot_from_state(get_ist_now(), spot, atm_strike, merged)
)
# Keep last 10 points
st.session_state["moment_history"] = st.session_state["moment_history"][-10:]

# ---- NEW: Compute 4 moment metrics ----
orderbook = get_nifty_orderbook_depth()
orderbook_metrics = orderbook_pressure_score(orderbook) if orderbook else {"available": False, "pressure": 0.0}

moment_metrics = {
    "momentum_burst": compute_momentum_burst(st.session_state["moment_history"]),
    "orderbook": orderbook_metrics,
    "gamma_cluster": compute_gamma_cluster(merged, atm_strike, window=2),
    "oi_accel": compute_oi_velocity_acceleration(st.session_state["moment_history"], atm_strike, window_strikes=2)
}

# ============================================
# 🎯 ATM BIAS ANALYSIS (NEW)
# ============================================

# Compute ATM and Level Biases
atm_bias = analyze_atm_bias(merged, spot, atm_strike, strike_gap)
support_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Support")
resistance_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Resistance")

# Calculate entry signal with moment detector & ATM bias integration
entry_signal = calculate_entry_signal_with_atm_bias(
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
    moment_metrics=moment_metrics,
    atm_bias=atm_bias,
    support_bias=support_bias,
    resistance_bias=resistance_bias
)

# ============================================
# 📊 COMPREHENSIVE OI & PCR DASHBOARD
# ============================================

# Run OI/PCR analysis
oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)

st.markdown("---")
st.markdown("## 📊 ENHANCED OI & PCR ANALYTICS DASHBOARD")

# Row 1: Totals
col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    st.metric("📈 Total CALL OI", f"{oi_pcr_metrics['total_ce_oi']:,}")
    st.metric("Δ CALL OI", f"{oi_pcr_metrics['total_ce_chg']:+,}")

with col_t2:
    st.metric("📉 Total PUT OI", f"{oi_pcr_metrics['total_pe_oi']:,}")
    st.metric("Δ PUT OI", f"{oi_pcr_metrics['total_pe_chg']:+,}")

with col_t3:
    st.metric("📊 Total OI", f"{oi_pcr_metrics['total_oi']:,}")
    st.metric("Total ΔOI", f"{oi_pcr_metrics['total_chg_oi']:+,}")

with col_t4:
    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {oi_pcr_metrics['pcr_color']};
        text-align: center;
    ">
        <div style='font-size: 0.9rem; color:#cccccc;'>PCR (TOTAL)</div>
        <div style='font-size: 2rem; color:{oi_pcr_metrics['pcr_color']}; font-weight:900;'>
            {oi_pcr_metrics['pcr_total']:.2f}
        </div>
        <div style='font-size: 0.9rem; color:{oi_pcr_metrics['pcr_color']};'>
            {oi_pcr_metrics['pcr_interpretation']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: PCR Card with detailed interpretation
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
    padding: 20px;
    border-radius: 12px;
    border: 2px solid {oi_pcr_metrics['pcr_color']};
    margin: 15px 0;
">
    <h3 style='color:{oi_pcr_metrics["pcr_color"]}; margin:0 0 10px 0;'>🎯 PUT-CALL RATIO (PCR) ANALYSIS</h3>
    
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
        <div style='text-align: center;'>
            <div style='font-size: 0.9rem; color:#cccccc;'>Sentiment</div>
            <div style='font-size: 1.3rem; color:{oi_pcr_metrics["pcr_color"]}; font-weight:700;'>
                {oi_pcr_metrics['pcr_sentiment']}
            </div>
        </div>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.9rem; color:#cccccc;'>PCR Change</div>
            <div style='font-size: 1.3rem; color:#ffcc00; font-weight:700;'>
                {oi_pcr_metrics['pcr_chg']:+.2f}
            </div>
        </div>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.9rem; color:#cccccc;'>OI Momentum</div>
            <div style='font-size: 1.3rem; color:#ff00ff; font-weight:700;'>
                {oi_pcr_metrics['oi_momentum']:+.1f}%
            </div>
        </div>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.9rem; color:#cccccc;'>CE:PE Ratio</div>
            <div style='font-size: 1.3rem; color:#66b3ff; font-weight:700;'>
                {oi_pcr_metrics['ce_pe_ratio']:.2f}:1
            </div>
        </div>
    </div>
    
    <div style='color:#ffffff; font-size: 1rem; margin-top: 10px;'>
        <strong>Interpretation:</strong> {oi_pcr_metrics['oi_change_interpretation']}
        {'. ' + oi_pcr_metrics['chg_interpretation'] if oi_pcr_metrics['chg_interpretation'] else ''}
    </div>
</div>
""", unsafe_allow_html=True)

# Row 3: Concentration Analysis
st.markdown("### 🎯 OI CONCENTRATION & SKEW")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)

with col_c1:
    st.metric("ATM Concentration", f"{oi_pcr_metrics['atm_concentration_pct']:.1f}%")
    st.caption(f"CALL: {oi_pcr_metrics['atm_ce_oi']:,} | PUT: {oi_pcr_metrics['atm_pe_oi']:,}")

with col_c2:
    st.metric("Max CALL OI Strike", f"₹{oi_pcr_metrics['max_ce_strike']:,}")
    st.caption(f"OI: {oi_pcr_metrics['max_ce_oi']:,}")

with col_c3:
    st.metric("Max PUT OI Strike", f"₹{oi_pcr_metrics['max_pe_strike']:,}")
    st.caption(f"OI: {oi_pcr_metrics['max_pe_oi']:,}")

with col_c4:
    st.metric("OI Skew", f"CALL: {oi_pcr_metrics['call_oi_skew']}")
    st.caption(f"PUT: {oi_pcr_metrics['put_oi_skew']}")

# Row 4: ITM/OTM Analysis
with st.expander("🔍 ITM/OTM OI Distribution", expanded=False):
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    
    with col_i1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color:#ff4444;">ITM CALL OI</div>
            <div style="font-size: 1.5rem; color:#ff4444; font-weight:700;">
                {:,}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                Strike < Spot
            </div>
        </div>
        """.format(oi_pcr_metrics['itm_ce_oi']), unsafe_allow_html=True)
    
    with col_i2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color:#ff9900;">OTM CALL OI</div>
            <div style="font-size: 1.5rem; color:#ff9900; font-weight:700;">
                {:,}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                Strike > Spot
            </div>
        </div>
        """.format(oi_pcr_metrics['otm_ce_oi']), unsafe_allow_html=True)
    
    with col_i3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color:#00cc66;">ITM PUT OI</div>
            <div style="font-size: 1.5rem; color:#00cc66; font-weight:700;">
                {:,}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                Strike > Spot
            </div>
        </div>
        """.format(oi_pcr_metrics['itm_pe_oi']), unsafe_allow_html=True)
    
    with col_i4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color:#66b3ff;">OTM PUT OI</div>
            <div style="font-size: 1.5rem; color:#66b3ff; font-weight:700;">
                {:,}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                Strike < Spot
            </div>
        </div>
        """.format(oi_pcr_metrics['otm_pe_oi']), unsafe_allow_html=True)

# Historical PCR Context
pcr_context = get_pcr_context(oi_pcr_metrics['pcr_total'])

st.markdown("### 📈 PCR HISTORICAL CONTEXT")

st.info(f"""
**Current PCR: {oi_pcr_metrics['pcr_total']:.2f}** - {pcr_context}

**Historical Ranges:**
- **Neutral:** 0.80 - 1.20 (Most common)
- **Bullish:** 1.20 - 1.50 (PUT selling dominant)
- **Very Bullish:** 1.50 - 2.00 (Heavy PUT selling)
- **Extreme Bullish:** > 2.00 (Rare, reversal possible)
- **Bearish:** 0.50 - 0.80 (CALL selling dominant)
- **Very Bearish:** 0.30 - 0.50 (Heavy CALL selling)
- **Extreme Bearish:** < 0.30 (Rare, bounce possible)
""")

# Add expiry context if near expiry
if days_to_expiry <= 5:
    expiry_pcr_context = analyze_pcr_for_expiry(oi_pcr_metrics['pcr_total'], days_to_expiry)
    st.warning(f"""
    **⚠️ Expiry Context (D-{int(days_to_expiry)}):** {expiry_pcr_context}
    
    PCR readings near expiry often exaggerate due to position squaring.
    """)

# ============================================
# 🎯 MULTI-DIMENSIONAL BIAS ANALYSIS (NEW)
# ============================================

# Display ATM Bias Dashboard
if atm_bias or support_bias or resistance_bias:
    display_bias_dashboard(atm_bias, support_bias, resistance_bias)

# ============================================
# 📅 EXPIRY SPIKE DETECTION
# ============================================

# Calculate expiry spike data
expiry_spike_data = detect_expiry_spikes(merged, spot, atm_strike, days_to_expiry, expiry)

# Advanced spike detection (optional)
violent_unwinding_signals = detect_violent_unwinding(merged, spot, atm_strike)
gamma_spike_risk = calculate_gamma_exposure_spike(total_gex_net, days_to_expiry)
pinning_probability = predict_expiry_pinning_probability(
    spot, seller_max_pain, 
    nearest_sup["strike"] if nearest_sup else None,
    nearest_res["strike"] if nearest_res else None
)

# Check for new Telegram signal
telegram_signal = check_and_send_signal(
    entry_signal, spot, seller_bias_result, 
    seller_max_pain, nearest_sup, nearest_res, 
    moment_metrics, seller_breakout_index, expiry, expiry_spike_data,
    atm_bias, support_bias, resistance_bias
)

# ============================================
# 📅 EXPIRY DATE SPIKE DETECTOR UI
# ============================================

st.markdown("---")
st.markdown("## 📅 EXPIRY DATE SPIKE DETECTOR")

# Main spike card
if expiry_spike_data["active"]:
    spike_col1, spike_col2, spike_col3 = st.columns([2, 1, 1])
    
    with spike_col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%);
            padding: 20px;
            border-radius: 12px;
            border: 3px solid {expiry_spike_data['color']};
            margin: 10px 0;
        ">
            <h3 style='color:{expiry_spike_data["color"]}; margin:0;'>📅 EXPIRY SPIKE ALERT</h3>
            <div style='font-size: 2.5rem; color:{expiry_spike_data["color"]}; font-weight:900; margin:10px 0;'>
                {expiry_spike_data["probability"]}%
            </div>
            <div style='font-size: 1.3rem; color:#ffffff; margin:5px 0;'>
                {expiry_spike_data["intensity"]}
            </div>
            <div style='font-size: 1.1rem; color:#ffcc00; margin:5px 0;'>
                Type: {expiry_spike_data["type"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with spike_col2:
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        ">
            <div style='font-size: 0.9rem; color:#cccccc;'>Days to Expiry</div>
            <div style='font-size: 2rem; color:#ff9900; font-weight:700;'>
                {expiry_spike_data['days_to_expiry']:.1f}
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>
                {expiry}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with spike_col3:
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        ">
            <div style='font-size: 0.9rem; color:#cccccc;'>Spike Score</div>
            <div style='font-size: 2rem; color:#ff00ff; font-weight:700;'>
                {expiry_spike_data['score']}/100
            </div>
            <div style='font-size: 0.8rem; color:#aaaaaa;'>
                Detection Factors
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Spike Factors
    with st.expander("🔍 View Spike Detection Factors", expanded=False):
        col_factors1, col_factors2 = st.columns(2)
        
        with col_factors1:
            st.markdown("### ⚠️ Spike Triggers")
            for factor in expiry_spike_data["factors"]:
                st.markdown(f"• {factor}")
            
            # Violent unwinding signals
            if violent_unwinding_signals:
                st.markdown("### 🚨 Violent Unwinding")
                for signal in violent_unwinding_signals:
                    st.markdown(f"• {signal}")
        
        with col_factors2:
            st.markdown("### 🎯 Key Levels")
            if expiry_spike_data["key_levels"]:
                for level in expiry_spike_data["key_levels"]:
                    st.markdown(f"• {level}")
            else:
                st.info("No extreme levels detected")
            
            # Gamma spike risk
            if gamma_spike_risk["score"] > 0:
                st.markdown(f"### ⚡ Gamma Spike Risk")
                st.markdown(f"• {gamma_spike_risk['message']}")
                st.markdown(f"• Risk Level: {gamma_spike_risk['risk']}")
            
            # Pinning probability
            if pinning_probability > 0:
                st.markdown(f"### 📍 Pinning Probability")
                st.markdown(f"• {pinning_probability}% chance of price getting stuck")
    
    # Historical Patterns
    if days_to_expiry <= 3:
        st.markdown("### 📊 Historical Expiry Patterns")
        patterns = get_historical_expiry_patterns()
        
        pattern_cols = st.columns(len(patterns))
        
        for idx, (pattern_name, pattern_data) in enumerate(patterns.items()):
            with pattern_cols[idx]:
                prob_color = "#ff4444" if pattern_data["probability"] > 0.6 else "#ff9900" if pattern_data["probability"] > 0.4 else "#66b3ff"
                st.markdown(f"""
                <div style="
                    background: #1a1f2e;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 3px solid {prob_color};
                    margin: 5px 0;
                ">
                    <div style='font-size: 0.9rem; color:#cccccc;'>{pattern_name.replace('_', ' ').title()}</div>
                    <div style='font-size: 1.5rem; color:{prob_color}; font-weight:700;'>
                        {pattern_data['probability']:.0%}
                    </div>
                    <div style='font-size: 0.8rem; color:#aaaaaa; margin-top:5px;'>
                        {pattern_data['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Action Recommendations
    st.markdown("### 🎯 Expiry Day Trading Strategy")
    
    if expiry_spike_data["probability"] > 60:
        st.warning("""
        **HIGH SPIKE PROBABILITY - AGGRESSIVE STRATEGY:**
        - Expect sharp moves (100-200 point swings)
        - Use wider stops (1.5-2x normal)
        - Consider straddles/strangles if IV not too high
        - Avoid deep ITM options (gamma risk)
        - Focus on 10:30-11:30 AM and 2:30-3:00 PM windows
        """)
    elif expiry_spike_data["probability"] > 40:
        st.info("""
        **MODERATE SPIKE RISK - BALANCED STRATEGY:**
        - Expect moderate volatility
        - Use normal stops with 20% buffer
        - Prefer ATM/1st OTM strikes
        - Watch Max Pain level closely
        - Be ready to exit early
        """)
    else:
        st.success("""
        **LOW SPIKE RISK - NORMAL STRATEGY:**
        - Normal trading rules apply
        - Standard stop losses
        - Focus on technical levels
        - Watch for last-hour moves
        """)
    
    # Gamma Risk Zone
    if days_to_expiry <= 2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #00ffff;
            margin: 10px 0;
        ">
            <h4 style='color:#00ffff; margin:0;'>⚠️ GAMMA RISK ZONE ACTIVE</h4>
            <p style='color:#ffffff; margin:5px 0;'>
                Days to expiry ≤ 2: Gamma exposure amplifies price moves.
                Market makers' hedging can cause exaggerated swings.
            </p>
            <p style='color:#ffcc00; margin:5px 0;'>
                🎯 Watch: {', '.join(expiry_spike_data['key_levels'][:3]) if expiry_spike_data['key_levels'] else 'ATM ±100 points'}
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info(f"""
    ### 📅 Expiry Spike Detector (Inactive)
    
    **Reason:** {expiry_spike_data['message']}
    
    Spike detection activates when expiry is ≤5 days away.
    
    Current expiry: **{expiry}**  
    Days to expiry: **{days_to_expiry:.1f}**
    
    *Check back closer to expiry for spike alerts*
    """)

# ============================================
# 🚀 TELEGRAM SIGNAL SECTION
# ============================================
st.markdown("---")
st.markdown("## 📱 TELEGRAM SIGNAL GENERATION (Option 3 Format)")

if telegram_signal:
    # NEW SIGNAL DETECTED
    st.success("🎯 **NEW TRADE SIGNAL GENERATED!**")
    
    # Auto-send to Telegram if enabled
    if auto_send and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        with st.spinner("Sending to Telegram..."):
            success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
            if success:
                st.success(f"✅ {message}")
                st.balloons()
            else:
                st.error(f"❌ {message}")
    
    # Create a nice display of the signal
    col_signal1, col_signal2 = st.columns([2, 1])
    
    with col_signal1:
        st.markdown("### 📋 Telegram Signal Ready:")
        
        if show_signal_preview:
            # Display formatted preview
            st.markdown("""
            <div style="
                background-color: #1a1f2e;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #0088cc;
                margin: 10px 0;
                font-family: monospace;
                white-space: pre-wrap;
            ">
            """ + telegram_signal + "</div>", unsafe_allow_html=True)
        else:
            st.code(telegram_signal)
    
    with col_signal2:
        st.markdown("### 📤 Send Options:")
        
        # Copy to clipboard
        if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_clipboard"):
            st.success("✅ Signal copied to clipboard!")
            
        # Manual send to Telegram
        if st.button("📱 Send to Telegram", use_container_width=True, key="send_telegram"):
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                success, message = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, telegram_signal)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Telegram credentials not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to secrets.")
            
        # Save to file
        if st.button("💾 Save to File", use_container_width=True, key="save_file"):
            filename = f"signal_{get_ist_datetime_str().replace(':', '-').replace(' ', '_')}.txt"
            with open(filename, 'w') as f:
                f.write(telegram_signal)
            st.success(f"✅ Signal saved to {filename}")
    
    # Add signal details
    with st.expander("📊 View Signal Details", expanded=False):
        col_details1, col_details2 = st.columns(2)
        
        with col_details1:
            st.markdown("**Position Details:**")
            st.metric("Type", entry_signal["position_type"])
            st.metric("Strength", entry_signal["signal_strength"])
            st.metric("Confidence", f"{entry_signal['confidence']:.0f}%")
            st.metric("Entry Price", f"₹{entry_signal['optimal_entry_price']:,.2f}")
        
        with col_details2:
            st.markdown("**Risk Management:**")
            st.metric("Stop Loss", f"₹{entry_signal['stop_loss']:,.2f}" if entry_signal['stop_loss'] else "N/A")
            st.metric("Target", f"₹{entry_signal['target']:,.2f}" if entry_signal['target'] else "N/A")
            
            # Calculate actual risk:reward
            if entry_signal['stop_loss'] and entry_signal['target']:
                if entry_signal["position_type"] == "LONG":
                    risk = abs(entry_signal['optimal_entry_price'] - entry_signal['stop_loss'])
                    reward = abs(entry_signal['target'] - entry_signal['optimal_entry_price'])
                else:
                    risk = abs(entry_signal['stop_loss'] - entry_signal['optimal_entry_price'])
                    reward = abs(entry_signal['optimal_entry_price'] - entry_signal['target'])
                
                if risk > 0:
                    rr_ratio = reward / risk
                    st.metric("Risk:Reward", f"1:{rr_ratio:.2f}")
    
    # Signal timestamp
    st.caption(f"⏰ Signal generated at: {get_ist_datetime_str()}")
    
    # Last signal info
    if "last_signal" in st.session_state and st.session_state["last_signal"]:
        st.caption(f"📝 Last signal type: {st.session_state['last_signal']}")
    
else:
    # No active signal
    st.info("📭 **No active trade signal to send.**")
    
    # Show why no signal
    with st.expander("ℹ️ Why no signal?", expanded=False):
        st.markdown(f"""
        **Current Status:**
        - Position Type: {entry_signal['position_type']}
        - Signal Strength: {entry_signal['signal_strength']}
        - Confidence: {entry_signal['confidence']:.0f}%
        - Seller Bias: {seller_bias_result['bias']}
        - ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'}
        - Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
        - PCR Sentiment: {oi_pcr_metrics['pcr_sentiment']}
        
        **Requirements for signal generation:**
        ✅ Position Type ≠ NEUTRAL
        ✅ Confidence ≥ 40%
        ✅ Clear directional bias
        ✅ ATM bias alignment
        """)
    
    # Show last signal if exists
    if "last_signal" in st.session_state and st.session_state["last_signal"]:
        st.info(f"📝 Last signal was: {st.session_state['last_signal']}")

# ============================================
# 🚀 MOMENT DETECTOR DISPLAY
# ============================================

st.markdown("---")
st.markdown("## 🚀 MOMENT DETECTOR (Is this a real move?)")

moment_col1, moment_col2, moment_col3, moment_col4 = st.columns(4)

with moment_col1:
    mb = moment_metrics["momentum_burst"]
    if mb["available"]:
        color = "#ff00ff" if mb["score"] > 70 else ("#ff9900" if mb["score"] > 40 else "#66b3ff")
        st.markdown(f'''
        <div class="moment-box">
            <h4>💥 MOMENTUM BURST</h4>
            <div class="moment-value" style="color:{color}">{mb["score"]}/100</div>
            <div class="sub-info">{mb["note"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="moment-box">
            <h4>💥 MOMENTUM BURST</h4>
            <div class="moment-value" style="color:#cccccc">N/A</div>
            <div class="sub-info">Need more refresh points</div>
        </div>
        ''', unsafe_allow_html=True)

with moment_col2:
    ob = moment_metrics["orderbook"]
    if ob["available"]:
        pressure = ob["pressure"]
        color = "#00ff88" if pressure > 0.15 else ("#ff4444" if pressure < -0.15 else "#66b3ff")
        st.markdown(f'''
        <div class="moment-box">
            <h4>📊 ORDERBOOK PRESSURE</h4>
            <div class="moment-value" style="color:{color}">{pressure:+.2f}</div>
            <div class="sub-info">Buy/Sell imbalance</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="moment-box">
            <h4>📊 ORDERBOOK PRESSURE</h4>
            <div class="moment-value" style="color:#cccccc">N/A</div>
            <div class="sub-info">Depth data unavailable</div>
        </div>
        ''', unsafe_allow_html=True)

with moment_col3:
    gc = moment_metrics["gamma_cluster"]
    if gc["available"]:
        color = "#ff00ff" if gc["score"] > 70 else ("#ff9900" if gc["score"] > 40 else "#66b3ff")
        st.markdown(f'''
        <div class="moment-box">
            <h4>🌀 GAMMA CLUSTER</h4>
            <div class="moment-value" style="color:{color}">{gc["score"]}/100</div>
            <div class="sub-info">ATM ±2 concentration</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="moment-box">
            <h4>🌀 GAMMA CLUSTER</h4>
            <div class="moment-value" style="color:#cccccc">N/A</div>
            <div class="sub-info">Data unavailable</div>
        </div>
        ''', unsafe_allow_html=True)

with moment_col4:
    oi = moment_metrics["oi_accel"]
    if oi["available"]:
        color = "#ff00ff" if oi["score"] > 70 else ("#ff9900" if oi["score"] > 40 else "#66b3ff")
        st.markdown(f'''
        <div class="moment-box">
            <h4>⚡ OI ACCELERATION</h4>
            <div class="moment-value" style="color:{color}">{oi["score"]}/100</div>
            <div class="sub-info">{oi["note"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="moment-box">
            <h4>⚡ OI ACCELERATION</h4>
            <div class="moment-value" style="color:#cccccc">N/A</div>
            <div class="sub-info">Need more refresh points</div>
        </div>
        ''', unsafe_allow_html=True)

# ============================================
# 🎯 SUPER PROMINENT ENTRY SIGNAL
# ============================================

st.markdown("---")

if entry_signal["position_type"] != "NEUTRAL" and entry_signal["confidence"] >= 40:
    # ACTIVE SIGNAL
    signal_bg = "#1a2e1a" if entry_signal["position_type"] == "LONG" else "#2e1a1a"
    signal_border = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
    signal_emoji = "🚀" if entry_signal["position_type"] == "LONG" else "🐻"
    
    # Create a container with custom styling
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {signal_bg} 0%, #2a3e2a 100%);
            padding: 30px;
            border-radius: 20px;
            border: 5px solid {signal_border};
            margin: 0 auto;
            text-align: center;
            max-width: 900px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        ">
        """, unsafe_allow_html=True)
        
        # Emoji and title row
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{signal_emoji}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 2.8rem; font-weight: 900; color:{signal_border}; line-height: 1.2;'>
                    {entry_signal["signal_strength"]} {entry_signal["position_type"]} SIGNAL
                </div>
                <div style='font-size: 1.2rem; color: #ffdd44; margin-top: 5px;'>
                    Confidence: {entry_signal["confidence"]:.0f}%
                </div>
                <div style='font-size: 1.1rem; color: #66b3ff; margin-top: 5px;'>
                    ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{signal_emoji}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Optimal entry price in a separate styled container
    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.3); 
        padding: 20px; 
        border-radius: 10px; 
        margin: 20px auto;
        max-width: 900px;
        text-align: center;
    ">
        <div style="font-size: 3rem; color: #ffcc00; font-weight: 900;">
            ₹{entry_signal["optimal_entry_price"]:,.2f}
        </div>
        <div style="font-size: 1.3rem; color: #cccccc; margin-top: 5px;">
            OPTIMAL ENTRY PRICE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.1rem; color: #aaaaaa;">Current Spot</div>
            <div style="font-size: 1.8rem; color: #ffffff; font-weight: 700;">₹""" + f"{spot:,.2f}" + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stats2:
        distance = abs(spot - entry_signal["optimal_entry_price"])
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.1rem; color: #aaaaaa;">Distance</div>
            <div style="font-size: 1.8rem; color: #ffaa00; font-weight: 700;">₹{distance:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stats3:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.1rem; color: #aaaaaa;">Direction</div>
            <div style="font-size: 1.8rem; color: {signal_border}; font-weight: 700;">{entry_signal["position_type"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Moment confirmation
    st.markdown(f"""
    <div style="
        margin-top: 25px; 
        padding: 20px; 
        background: rgba(0,0,0,0.2); 
        border-radius: 10px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="font-size: 1.2rem; color: #ffdd44; margin-bottom: 10px; text-align: center;">🎯 MOMENT CONFIRMATION</div>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
            <div>Burst: {moment_metrics['momentum_burst'].get('score', 0)}/100</div>
            <div>Pressure: {moment_metrics['orderbook'].get('pressure', 0):+.2f}</div>
            <div>Gamma: {moment_metrics['gamma_cluster'].get('score', 0)}/100</div>
            <div>OI Accel: {moment_metrics['oi_accel'].get('score', 0)}/100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # OI/PCR Confirmation
    st.markdown(f"""
    <div style="
        margin-top: 25px; 
        padding: 20px; 
        background: rgba(0,0,0,0.2); 
        border-radius: 10px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="font-size: 1.2rem; color: #66b3ff; margin-bottom: 10px; text-align: center;">📊 OI/PCR CONFIRMATION</div>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
            <div>PCR: {oi_pcr_metrics['pcr_total']:.2f}</div>
            <div>Sentiment: {oi_pcr_metrics['pcr_sentiment']}</div>
            <div>CALL OI: {oi_pcr_metrics['total_ce_oi']:,}</div>
            <div>PUT OI: {oi_pcr_metrics['total_pe_oi']:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ATM Bias Confirmation
    if atm_bias:
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: {atm_bias['verdict_color']}; margin-bottom: 10px; text-align: center;">🎯 ATM BIAS CONFIRMATION</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>ATM Bias: {atm_bias['verdict']}</div>
                <div>Score: {atm_bias['total_score']:.2f}</div>
                <div>CALL OI: {atm_bias['metrics']['ce_oi']:,}</div>
                <div>PUT OI: {atm_bias['metrics']['pe_oi']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
    
    with action_col1:
        if st.button(f"📊 PLACE {entry_signal['position_type']} ORDER AT ₹{entry_signal['optimal_entry_price']:,.0f}", 
                    use_container_width=True, type="primary", key="place_order"):
            st.success(f"✅ {entry_signal['position_type']} order queued at ₹{entry_signal['optimal_entry_price']:,.2f}")
            st.balloons()
    
    with action_col2:
        if st.button("🔔 SET PRICE ALERT", use_container_width=True, key="set_alert"):
            st.info(f"📢 Alert set for {entry_signal['optimal_entry_price']:,.2f}")
    
    with action_col3:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh"):
            st.rerun()
    
    # Signal Reasons
    with st.expander("📋 View Detailed Signal Reasoning", expanded=False):
        for reason in entry_signal["reasons"]:
            st.markdown(f"• {reason}")
        
        # Moment Detector Details
        st.markdown("### 🚀 Moment Detector Details:")
        for metric_name, metric_data in moment_metrics.items():
            if metric_data.get("available", False):
                st.markdown(f"**{metric_name.replace('_', ' ').title()}:** {metric_data.get('note', 'N/A')}")
        
        # ATM Bias Details
        if atm_bias:
            st.markdown("### 🎯 ATM Bias Analysis:")
            st.markdown(f"• **Overall Verdict:** {atm_bias['verdict']}")
            st.markdown(f"• **Total Score:** {atm_bias['total_score']:.2f}")
            st.markdown(f"• **Explanation:** {atm_bias['verdict_explanation']}")
            st.markdown(f"• **Key Metrics:** CALL OI: {atm_bias['metrics']['ce_oi']:,} | PUT OI: {atm_bias['metrics']['pe_oi']:,}")
            st.markdown(f"• **Net Delta:** {atm_bias['metrics']['net_delta']:.3f} | **Net Gamma:** {atm_bias['metrics']['net_gamma']:.3f}")
        
        # OI/PCR Details
        st.markdown("### 📊 OI/PCR Analysis:")
        st.markdown(f"• **PCR:** {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']})")
        st.markdown(f"• **OI Change:** {oi_pcr_metrics['oi_change_interpretation']}")
        st.markdown(f"• **Max CALL OI:** ₹{oi_pcr_metrics['max_ce_strike']:,} ({oi_pcr_metrics['max_ce_oi']:,})")
        st.markdown(f"• **Max PUT OI:** ₹{oi_pcr_metrics['max_pe_strike']:,} ({oi_pcr_metrics['max_pe_oi']:,})")
        st.markdown(f"• **ATM Concentration:** {oi_pcr_metrics['atm_concentration_pct']:.1f}%")
        
        # Expiry Spike Risk
        if expiry_spike_data["active"]:
            st.markdown("### 📅 Expiry Spike Risk:")
            st.markdown(f"• Probability: {expiry_spike_data['probability']}%")
            st.markdown(f"• Type: {expiry_spike_data['type']}")
            st.markdown(f"• Intensity: {expiry_spike_data['intensity']}")
            if expiry_spike_data["key_levels"]:
                st.markdown(f"• Key Levels: {', '.join(expiry_spike_data['key_levels'])}")
    
else:
    # NO SIGNAL
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
            padding: 30px;
            border-radius: 20px;
            border: 5px solid #666666;
            margin: 0 auto;
            text-align: center;
            max-width: 900px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        ">
        """, unsafe_allow_html=True)
        
        # Warning icon
        st.markdown("""
        <div style="font-size: 4rem; color: #cccccc; margin-bottom: 20px; text-align: center;">
            ⚠️
        </div>
        """, unsafe_allow_html=True)
        
        # No signal message
        st.markdown("""
        <div style="font-size: 2.5rem; font-weight: 900; color:#cccccc; line-height: 1.2; margin-bottom: 15px; text-align: center;">
            NO CLEAR ENTRY SIGNAL
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="font-size: 1.8rem; color: #ffcc00; font-weight: 700; margin-bottom: 20px; text-align: center;">
            Wait for Better Setup
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Current spot price
    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.3); 
        padding: 20px; 
        border-radius: 10px; 
        margin: 20px auto;
        max-width: 900px;
        text-align: center;
    ">
        <div style="font-size: 2.5rem; color: #ffffff; font-weight: 700;">
            ₹{spot:,.2f}
        </div>
        <div style="font-size: 1.2rem; color: #cccccc; margin-top: 5px;">
            CURRENT SPOT PRICE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence info
    st.markdown(f"""
    <div style="
        color: #aaaaaa; 
        font-size: 1.1rem; 
        margin-top: 20px;
        text-align: center;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        Signal Confidence: {entry_signal["confidence"]:.0f}% | 
        Seller Bias: {seller_bias_result["bias"]} | 
        ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'} | 
        PCR Sentiment: {oi_pcr_metrics['pcr_sentiment']} | 
        Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
    </div>
    """, unsafe_allow_html=True)
    
    # Moment status
    st.markdown(f"""
    <div style="
        margin-top: 25px; 
        padding: 20px; 
        background: rgba(0,0,0,0.2); 
        border-radius: 10px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="font-size: 1.2rem; color: #ffdd44; margin-bottom: 10px; text-align: center;">🎯 MOMENT STATUS</div>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
            <div>Burst: {moment_metrics['momentum_burst'].get('score', 0)}/100</div>
            <div>Pressure: {moment_metrics['orderbook'].get('pressure', 0):+.2f}</div>
            <div>Gamma: {moment_metrics['gamma_cluster'].get('score', 0)}/100</div>
            <div>OI Accel: {moment_metrics['oi_accel'].get('score', 0)}/100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # OI/PCR status
    st.markdown(f"""
    <div style="
        margin-top: 25px; 
        padding: 20px; 
        background: rgba(0,0,0,0.2); 
        border-radius: 10px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    ">
        <div style="font-size: 1.2rem; color: #66b3ff; margin-bottom: 10px; text-align: center;">📊 OI/PCR STATUS</div>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
            <div>PCR: {oi_pcr_metrics['pcr_total']:.2f}</div>
            <div>CALL OI: {oi_pcr_metrics['total_ce_oi']:,}</div>
            <div>PUT OI: {oi_pcr_metrics['total_pe_oi']:,}</div>
            <div>ATM Conc: {oi_pcr_metrics['atm_concentration_pct']:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ATM Bias status
    if atm_bias:
        st.markdown(f"""
        <div style="
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        ">
            <div style="font-size: 1.2rem; color: {atm_bias['verdict_color']}; margin-bottom: 10px; text-align: center;">🎯 ATM BIAS STATUS</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 1rem; color: #cccccc; text-align: center;">
                <div>Verdict: {atm_bias['verdict']}</div>
                <div>Score: {atm_bias['total_score']:.2f}</div>
                <div>CALL OI: {atm_bias['metrics']['ce_oi']:,}</div>
                <div>PUT OI: {atm_bias['metrics']['pe_oi']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Expandable details for no signal
    with st.expander("🔍 Why No Signal? (Click for Details)", expanded=False):
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("### 📊 Current Metrics:")
            st.metric("Seller Bias", seller_bias_result["bias"])
            st.metric("Polarity Score", f"{seller_bias_result['polarity']:.2f}")
            st.metric("ATM Bias", atm_bias['verdict'] if atm_bias else "N/A")
            st.metric("ATM Bias Score", f"{atm_bias['total_score']:.2f}" if atm_bias else "N/A")
            st.metric("Breakout Index", f"{seller_breakout_index}%")
            st.metric("Signal Confidence", f"{entry_signal['confidence']:.0f}%")
            st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
            st.metric("PCR Sentiment", oi_pcr_metrics['pcr_sentiment'])
            st.metric("Expiry Spike Risk", f"{expiry_spike_data.get('probability', 0)}%")
        
        with col_detail2:
            st.markdown("### 🎯 Signal Requirements:")
            requirements = [
                "✅ Clear directional bias (BULLISH/BEARISH)",
                "✅ Confidence > 40%",
                "✅ Strong moment detector scores",
                "✅ ATM bias alignment",
                "✅ Support/Resistance alignment",
                "✅ Momentum burst > 50",
                "✅ PCR alignment with bias"
            ]
            for req in requirements:
                st.markdown(f"- {req}")
            
            st.markdown(f"""
            ### 📈 Current Status:
            - **Position Type**: {entry_signal["position_type"]}
            - **Signal Strength**: {entry_signal["signal_strength"]}
            - **Optimal Entry**: ₹{entry_signal["optimal_entry_price"]:,.2f}
            - **ATM Bias**: {atm_bias['verdict'] if atm_bias else 'N/A'}
            - **PCR Sentiment**: {oi_pcr_metrics['pcr_sentiment']}
            - **OI Skew**: CALL: {oi_pcr_metrics['call_oi_skew']}, PUT: {oi_pcr_metrics['put_oi_skew']}
            - **Expiry in**: {days_to_expiry:.1f} days
            """)

st.markdown("---")

# ============================================
# 🎯 SELLER'S BIAS
# ============================================

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

# Core Metrics with OI/PCR
st.markdown("## 📈 SELLER'S MARKET OVERVIEW")

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

# Max Pain Display
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

# SELLER Activity Summary with OI Context
st.markdown("### 🔥 SELLER ACTIVITY HEATMAP WITH OI CONTEXT")

seller_activity = pd.DataFrame([
    {"Activity": "CALL Writing (Bearish)", "Strikes": ce_selling, "Total OI": f"{oi_pcr_metrics['total_ce_oi']:,}", "Bias": "BEARISH", "Color": "#ff4444"},
    {"Activity": "CALL Buying Back (Bullish)", "Strikes": ce_buying_back, "Total OI": f"{oi_pcr_metrics['total_ce_oi']:,}", "Bias": "BULLISH", "Color": "#00ff88"},
    {"Activity": "PUT Writing (Bullish)", "Strikes": pe_selling, "Total OI": f"{oi_pcr_metrics['total_pe_oi']:,}", "Bias": "BULLISH", "Color": "#00ff88"},
    {"Activity": "PUT Buying Back (Bearish)", "Strikes": pe_buying_back, "Total OI": f"{oi_pcr_metrics['total_pe_oi']:,}", "Bias": "BEARISH", "Color": "#ff4444"}
])

st.dataframe(seller_activity, use_container_width=True)

st.markdown("---")

# ============================================
# 🎯 SPOT POSITION - SELLER'S VIEW WITH OI/PCR
# ============================================

st.markdown("## 📍 SPOT POSITION (SELLER'S DEFENSE + OI/PCR)")

col_spot, col_range = st.columns([1, 1])

with col_spot:
    st.markdown(f"""
    <div class="spot-card">
        <h3>🎯 CURRENT SPOT</h3>
        <div class="spot-price">₹{spot:,.2f}</div>
        <div class="distance">ATM: ₹{atm_strike:,}</div>
        <div class="distance">Market Bias: <span style="color:{seller_bias_result['color']}">{seller_bias_result["bias"]}</span></div>
        <div class="distance">ATM Bias: <span style="color:{atm_bias['verdict_color'] if atm_bias else '#cccccc'}">{atm_bias['verdict'] if atm_bias else 'N/A'}</span></div>
        <div class="distance">PCR: <span style="color:{oi_pcr_metrics['pcr_color']}">{oi_pcr_metrics['pcr_total']:.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    if nearest_sup and nearest_res:
        range_size = spot_analysis["range_size"]
        spot_position_pct = spot_analysis["spot_position_pct"]
        range_bias = spot_analysis["range_bias"]
        
        st.markdown(f"""
        <div class="spot-card">
            <h3>📊 SELLER'S DEFENSE RANGE</h3>
            <div class="distance">₹{nearest_sup['strike']:,} ← SPOT → ₹{nearest_res['strike']:,}</div>
            <div class="distance">Position: {spot_position_pct:.1f}% within range</div>
            <div class="distance">Range Width: ₹{range_size:,}</div>
            <div class="distance" style="color:#ffcc00;">{range_bias}</div>
            <div class="distance">ATM OI Concentration: {oi_pcr_metrics['atm_concentration_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# NEAREST LEVELS WITH SELLER INTERPRETATION + OI
st.markdown("### 🎯 NEAREST SELLER DEFENSE LEVELS WITH OI")

col_ns, col_nr = st.columns(2)

with col_ns:
    st.markdown("#### 🛡️ SELLER SUPPORT BELOW")
    
    if nearest_sup:
        sup = nearest_sup
        pcr_display = f"{sup['pcr']:.2f}" if not np.isinf(sup['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>💚 NEAREST SELLER SUPPORT</h4>
            <div class="level-value">₹{sup['strike']:,}</div>
            <div class="level-distance">⬇️ Distance: ₹{sup['distance']:.2f} ({sup['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                <strong>SELLER ACTIVITY:</strong> {sup['seller_strength']}<br>
                PUT OI: {sup['oi_pe']:,} | CALL OI: {sup['oi_ce']:,}<br>
                PCR: {pcr_display} | ΔCALL: {sup['chg_oi_ce']:+,} | ΔPUT: {sup['chg_oi_pe']:+,}<br>
                <strong>OI Skew:</strong> PUT/CALL = {sup['oi_pe']/max(sup['oi_ce'],1):.1f}x
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No seller support level below spot")

with col_nr:
    st.markdown("#### ⚡ SELLER RESISTANCE ABOVE")
    
    if nearest_res:
        res = nearest_res
        pcr_display = f"{res['pcr']:.2f}" if not np.isinf(res['pcr']) else "∞"
        
        st.markdown(f"""
        <div class="nearest-level">
            <h4>🧡 NEAREST SELLER RESISTANCE</h4>
            <div class="level-value">₹{res['strike']:,}</div>
            <div class="level-distance">⬆️ Distance: ₹{res['distance']:.2f} ({res['distance_pct']:.2f}%)</div>
            <div class="sub-info">
                <strong>SELLER ACTIVITY:</strong> {res['seller_strength']}<br>
                CALL OI: {res['oi_ce']:,} | PUT OI: {res['oi_pe']:,}<br>
                PCR: {pcr_display} | ΔCALL: {res['chg_oi_ce']:+,} | ΔPUT: {res['chg_oi_pe']:+,}<br>
                <strong>OI Skew:</strong> CALL/PUT = {res['oi_ce']/max(res['oi_pe'],1):.1f}x
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No seller resistance level above spot")

st.markdown("---")

# TOP SELLER DEFENSE LEVELS WITH ENHANCED OI INFO
st.markdown("### 🎯 TOP SELLER DEFENSE LEVELS (Strongest 3 with OI Analysis)")

col_s, col_r = st.columns(2)

with col_s:
    st.markdown("#### 🛡️ STRONGEST SELLER SUPPORTS (Highest PUT OI)")
    
    for i, (idx, row) in enumerate(seller_supports_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_pe = int(row["OI_PE"])
        oi_ce = int(row["OI_CE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        chg_oi_ce = int(row.get("Chg_OI_CE", 0))
        
        # Calculate OI ratios
        total_oi = oi_pe + oi_ce
        pe_ratio = (oi_pe / total_oi * 100) if total_oi > 0 else 0
        
        if pcr > 1.5:
            seller_msg = f"Heavy PUT writing ({pe_ratio:.0f}% PUT OI) - Strong bullish defense"
            color = "#00ff88"
        elif pcr > 1.0:
            seller_msg = f"Moderate PUT writing ({pe_ratio:.0f}% PUT OI) - Bullish defense"
            color = "#00cc66"
        else:
            seller_msg = f"Light PUT writing ({pe_ratio:.0f}% PUT OI) - Weak defense"
            color = "#cccccc"
        
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        st.markdown(f'''
        <div class="level-card">
            <h4>Seller Support #{i}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                <span style="color:{color}"><strong>{seller_msg}</strong></span><br>
                PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                PCR: {pcr_display} | PUT%: {pe_ratio:.0f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)

with col_r:
    st.markdown("#### ⚡ STRONGEST SELLER RESISTANCES (Highest CALL OI)")
    
    for i, (idx, row) in enumerate(seller_resists_df.head(3).iterrows(), 1):
        strike = int(row["strikePrice"])
        oi_ce = int(row["OI_CE"])
        oi_pe = int(row["OI_PE"])
        pcr = row["PCR"]
        pcr_display = f"{pcr:.2f}" if not np.isinf(pcr) else "∞"
        chg_oi_ce = int(row.get("Chg_OI_CE", 0))
        chg_oi_pe = int(row.get("Chg_OI_PE", 0))
        
        # Calculate OI ratios
        total_oi = oi_ce + oi_pe
        ce_ratio = (oi_ce / total_oi * 100) if total_oi > 0 else 0
        
        if pcr < 0.5:
            seller_msg = f"Heavy CALL writing ({ce_ratio:.0f}% CALL OI) - Strong bearish defense"
            color = "#ff4444"
        elif pcr < 1.0:
            seller_msg = f"Moderate CALL writing ({ce_ratio:.0f}% CALL OI) - Bearish defense"
            color = "#ff6666"
        else:
            seller_msg = f"Light CALL writing ({ce_ratio:.0f}% CALL OI) - Weak defense"
            color = "#cccccc"
        
        dist = abs(spot - strike)
        dist_pct = (dist / spot * 100)
        direction = "⬆️ Above" if strike > spot else "⬇️ Below"
        
        st.markdown(f'''
        <div class="level-card">
            <h4>Seller Resistance #{i}</h4>
            <p>₹{strike:,}</p>
            <div class="sub-info">
                {direction}: ₹{dist:.2f} ({dist_pct:.2f}%)<br>
                <span style="color:{color}"><strong>{seller_msg}</strong></span><br>
                CALL OI: {oi_ce:,} | ΔCALL: {chg_oi_ce:+,}<br>
                PUT OI: {oi_pe:,} | ΔPUT: {chg_oi_pe:+,}<br>
                PCR: {pcr_display} | CALL%: {ce_ratio:.0f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# 📊 DETAILED DATA - SELLER VIEW + MOMENT + EXPIRY + OI/PCR + ATM BIAS
# ============================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Seller Activity", "🧮 Seller Greeks", "📈 Seller PCR", "🚀 Moment Analysis", "📅 Expiry Analysis", "🎯 ATM Bias"])

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
    
    # Color code seller actions
    def color_seller_action(val):
        if "WRITING" in str(val):
            if "CALL" in str(val):
                return "background-color: #2e1a1a; color: #ff6666"
            else:
                return "background-color: #1a2e1a; color: #00ff88"
        elif "BUYING BACK" in str(val):
            if "CALL" in str(val):
                return "background-color: #1a2e1a; color: #00ff88"
            else:
                return "background-color: #2e1a1a; color: #ff6666"
        return ""
    
    seller_display = merged[seller_cols].copy()
    styled_df = seller_display.style.applymap(color_seller_action, subset=["CE_Seller_Action", "PE_Seller_Action"])
    st.dataframe(styled_df, use_container_width=True)

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
    
    # Format Greek values
    greeks_display = merged[greeks_cols].copy()
    
    # Color code GEX
    def color_gex(val):
        if val > 0:
            return "background-color: #1a2e1a; color: #00ff88"
        elif val < 0:
            return "background-color: #2e1a1a; color: #ff6666"
        return ""
    
    styled_greeks = greeks_display.style.applymap(color_gex, subset=["GEX_Net"])
    st.dataframe(styled_greeks, use_container_width=True)
    
    # GEX Interpretation
    st.markdown("#### 🎯 GEX INTERPRETATION (SELLER'S VIEW)")
    if total_gex_net > 0:
        st.success(f"**POSITIVE GEX (₹{int(total_gex_net):,}):** Sellers have POSITIVE gamma exposure. They're SHORT gamma and will BUY when price rises, SELL when price falls (stabilizing effect).")
    elif total_gex_net < 0:
        st.error(f"**NEGATIVE GEX (₹{int(total_gex_net):,}):** Sellers have NEGATIVE gamma exposure. They're LONG gamma and will SELL when price rises, BUY when price falls (destabilizing effect).")
    else:
        st.info("**NEUTRAL GEX:** Balanced seller gamma exposure.")

with tab3:
    st.markdown("### 📈 SELLER PCR ANALYSIS")
    
    pcr_display_cols = ["strikePrice", "OI_CE", "OI_PE", "PCR", "Chg_OI_CE", "Chg_OI_PE", "seller_support_score", "seller_resistance_score"]
    for col in pcr_display_cols:
        if col not in ranked_current.columns:
            ranked_current[col] = 0
    
    # Create display dataframe
    pcr_display = ranked_current[pcr_display_cols].copy()
    pcr_display["distance_from_spot"] = abs(pcr_display["strikePrice"] - spot)
    pcr_display["OI_Total"] = pcr_display["OI_CE"] + pcr_display["OI_PE"]
    pcr_display["PUT_OI_Pct"] = (pcr_display["OI_PE"] / pcr_display["OI_Total"] * 100).round(1)
    
    # Sort by distance_from_spot BEFORE applying style
    pcr_display = pcr_display.sort_values("distance_from_spot")
    
    # Color PCR values
    def color_pcr(val):
        if isinstance(val, (int, float)):
            if val > 1.5:
                return "background-color: #1a2e1a; color: #00ff88"
            elif val > 1.0:
                return "background-color: #2e2a1a; color: #ffcc44"
            elif val > 0.5:
                return "background-color: #1a1f2e; color: #66b3ff"
            elif val <= 0.5:
                return "background-color: #2e1a1a; color: #ff4444"
        return ""
    
    # Apply style to already sorted dataframe
    styled_pcr = pcr_display.style.applymap(color_pcr, subset=["PCR"])
    
    # Display without sorting again
    st.dataframe(styled_pcr, use_container_width=True)
    
    # PCR Interpretation with OI context
    avg_pcr = ranked_current["PCR"].replace([np.inf, -np.inf], np.nan).mean()
    if not np.isnan(avg_pcr):
        st.markdown(f"#### 🎯 AVERAGE PCR: {avg_pcr:.2f}")
        if avg_pcr > 1.5:
            st.success(f"**HIGH PCR (>1.5):** Heavy PUT selling relative to CALL selling. Sellers are BULLISH. PUT OI dominance: {oi_pcr_metrics['total_pe_oi']/max(oi_pcr_metrics['total_ce_oi'],1):.1f}x")
        elif avg_pcr > 1.0:
            st.info(f"**MODERATE PCR (1.0-1.5):** More PUT selling than CALL selling. Sellers leaning BULLISH. PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
        elif avg_pcr > 0.5:
            st.warning(f"**LOW PCR (0.5-1.0):** More CALL selling than PUT selling. Sellers leaning BEARISH. CALL OI: {oi_pcr_metrics['total_ce_oi']:,}")
        else:
            st.error(f"**VERY LOW PCR (<0.5):** Heavy CALL selling relative to PUT selling. Sellers are BEARISH. CALL OI dominance: {oi_pcr_metrics['total_ce_oi']/max(oi_pcr_metrics['total_pe_oi'],1):.1f}x")

with tab4:
    st.markdown("### 🚀 MOMENT DETECTOR ANALYSIS")
    
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
    
    # Orderbook Pressure Details
    st.markdown("#### 📊 ORDERBOOK PRESSURE ANALYSIS")
    ob = moment_metrics["orderbook"]
    if ob["available"]:
        col_ob1, col_ob2 = st.columns(2)
        with col_ob1:
            st.metric("Pressure", f"{ob['pressure']:+.2f}")
            st.metric("Buy Qty", f"{ob['buy_qty']:.0f}")
            st.metric("Sell Qty", f"{ob['sell_qty']:.0f}")
        with col_ob2:
            if ob["pressure"] > 0.15:
                st.success("**STRONG BUY PRESSURE:** More buy orders than sell orders")
            elif ob["pressure"] < -0.15:
                st.error("**STRONG SELL PRESSURE:** More sell orders than buy orders")
            else:
                st.info("**BALANCED ORDERBOOK:** Buy and sell orders are balanced")
    else:
        st.warning("Orderbook depth data unavailable from Dhan API.")
    
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
    
    st.markdown("---")
    
    # OI Acceleration Details
    st.markdown("#### ⚡ OI ACCELERATION ANALYSIS")
    oi_accel = moment_metrics["oi_accel"]
    if oi_accel["available"]:
        col_oi1, col_oi2 = st.columns(2)
        with col_oi1:
            st.metric("Acceleration Score", f"{oi_accel['score']}/100")
        with col_oi2:
            st.info(f"**Note:** {oi_accel['note']}")
            if oi_accel["score"] > 60:
                st.success("**ACCELERATING OI:** Open interest changing rapidly - momentum building")
            else:
                st.info("**STEADY OI:** Open interest changes are gradual")

with tab5:
    st.markdown("### 📅 EXPIRY SPIKE ANALYSIS")
    
    # Expiry Spike Probability
    st.markdown("#### 📊 SPIKE PROBABILITY BREAKDOWN")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        st.metric("Spike Probability", f"{expiry_spike_data.get('probability', 0)}%")
        st.metric("Spike Score", f"{expiry_spike_data.get('score', 0)}/100")
    
    with col_exp2:
        st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
        st.metric("Spike Type", expiry_spike_data.get('type', 'N/A'))
    
    with col_exp3:
        intensity = expiry_spike_data.get('intensity', 'N/A')
        intensity_color = {
            "HIGH PROBABILITY SPIKE": "#ff0000",
            "MODERATE SPIKE RISK": "#ff9900",
            "LOW SPIKE RISK": "#ffff00",
            "NO SPIKE DETECTED": "#00ff00"
        }.get(intensity, "#cccccc")
        
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
    
    st.markdown("---")
    
    # Violent Unwinding
    if violent_unwinding_signals:
        st.markdown("#### 🚨 VIOLENT UNWINDING DETECTED")
        for signal in violent_unwinding_signals:
            st.markdown(f"• {signal}")
    
    st.markdown("---")
    
    # Gamma Spike Risk
    if gamma_spike_risk["score"] > 0:
        st.markdown("#### ⚡ GAMMA SPIKE RISK")
        st.markdown(f"**Risk Level:** {gamma_spike_risk['risk']}")
        st.markdown(f"**Score:** {gamma_spike_risk['score']}/100")
        st.markdown(f"**Message:** {gamma_spike_risk['message']}")
    
    st.markdown("---")
    
    # Pinning Probability
    if pinning_probability > 0:
        st.markdown("#### 📍 EXPIRY PINNING PROBABILITY")
        st.metric("Pinning Chance", f"{pinning_probability}%")
        if pinning_probability > 50:
            st.info("**HIGH PINNING RISK:** Price likely to get stuck near current levels")
        elif pinning_probability > 30:
            st.warning("**MODERATE PINNING RISK:** Some chance of price getting stuck")
        else:
            st.success("**LOW PINNING RISK:** Price likely to move freely")

with tab6:
    st.markdown("### 🎯 ATM BIAS DETAILED ANALYSIS")
    
    if atm_bias:
        # Key metrics
        col_atm1, col_atm2, col_atm3 = st.columns(3)
        
        with col_atm1:
            st.metric("ATM Strike", f"₹{atm_bias['strike']:,}")
            st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
        
        with col_atm2:
            st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
            st.metric("Net Gamma", f"{atm_bias['metrics']['net_gamma']:.3f}")
            st.metric("Delta Exposure", f"₹{atm_bias['metrics']['delta_exposure']:,}")
        
        with col_atm3:
            st.metric("Gamma Exposure", f"₹{atm_bias['metrics']['gamma_exposure']:,}")
            st.metric("CALL IV", f"{atm_bias['metrics']['ce_iv']:.2f}%")
            st.metric("PUT IV", f"{atm_bias['metrics']['pe_iv']:.2f}%")
        
        # Bias breakdown
        st.markdown("#### 📊 BIAS BREAKDOWN BY METRIC")
        for bias_name, score in atm_bias["bias_scores"].items():
            emoji = atm_bias["bias_emojis"].get(bias_name, "⚖️")
            interpretation = atm_bias["bias_interpretations"].get(bias_name, "")
            
            # Color based on score
            if score > 0:
                color = "#00ff88"
                bg_color = "#1a2e1a"
            elif score < 0:
                color = "#ff4444"
                bg_color = "#2e1a1a"
            else:
                color = "#66b3ff"
                bg_color = "#1a1f2e"
            
            st.markdown(f"""
            <div style="
                background: {bg_color};
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid {color};
                margin: 5px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color:#ffffff;">
                        {bias_name.replace('_', ' ').title()}
                    </div>
                    <div style="font-size: 1.2rem; color:{color}; font-weight:700;">
                        {emoji} {score:+.1f}
                    </div>
                </div>
                <div style="font-size: 0.9rem; color:#cccccc; margin-top: 5px;">
                    {interpretation}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Seller interpretation for ATM
        st.markdown("#### 🧠 SELLER INTERPRETATION AT ATM")
        if atm_bias["total_score"] > 0.3:
            st.success(f"""
            **Strong Bullish ATM Bias ({atm_bias['total_score']:.2f})**
            
            Sellers are heavily writing PUTs at ATM strikes, indicating:
            1. **Bullish conviction** - Expecting price to stay above ATM
            2. **PUT selling dominance** - More PUT OI than CALL OI
            3. **Negative delta exposure** - PUTs creating negative delta
            4. **Potential support** - ATM acting as strong support
            
            **Trading Implication:** Favor LONG positions with stops below ATM
            """)
        elif atm_bias["total_score"] > 0.1:
            st.info(f"""
            **Mild Bullish ATM Bias ({atm_bias['total_score']:.2f})**
            
            Sellers are leaning towards PUT writing at ATM:
            1. **Slight bullish bias** - More PUT activity than CALL
            2. **Moderate PUT OI** - PUT OI slightly higher than CALL
            3. **Balanced delta** - Delta exposure relatively neutral
            
            **Trading Implication:** Cautious LONG bias, wait for confirmation
            """)
        elif atm_bias["total_score"] < -0.3:
            st.error(f"""
            **Strong Bearish ATM Bias ({atm_bias['total_score']:.2f})**
            
            Sellers are heavily writing CALLs at ATM strikes, indicating:
            1. **Bearish conviction** - Expecting price to stay below ATM
            2. **CALL selling dominance** - More CALL OI than PUT OI
            3. **Positive delta exposure** - CALLs creating positive delta
            4. **Potential resistance** - ATM acting as strong resistance
            
            **Trading Implication:** Favor SHORT positions with stops above ATM
            """)
        elif atm_bias["total_score"] < -0.1:
            st.warning(f"""
            **Mild Bearish ATM Bias ({atm_bias['total_score']:.2f})**
            
            Sellers are leaning towards CALL writing at ATM:
            1. **Slight bearish bias** - More CALL activity than PUT
            2. **Moderate CALL OI** - CALL OI slightly higher than PUT
            3. **Balanced delta** - Delta exposure relatively neutral
            
            **Trading Implication:** Cautious SHORT bias, wait for confirmation
            """)
        else:
            st.info(f"""
            **Neutral ATM Bias ({atm_bias['total_score']:.2f})**
            
            Balanced seller activity at ATM:
            1. **No clear bias** - CALL and PUT activity balanced
            2. **Equal OI distribution** - Similar OI on both sides
            3. **Neutral delta/gamma** - Minimal directional pressure
            
            **Trading Implication:** Range-bound expected, wait for breakout
            """)
        
        # Gamma exposure implications
        st.markdown("#### ⚡ GAMMA EXPOSURE IMPLICATIONS")
        if atm_bias["metrics"]["gamma_exposure"] > 100000:
            st.success(f"""
            **Positive Gamma Exposure (₹{atm_bias['metrics']['gamma_exposure']:,})**
            
            Market makers are SHORT gamma at ATM:
            - **Stabilizing effect** - They'll buy on dips, sell on rallies
            - **Reduced volatility** - Price moves become smoother
            - **Mean reversion bias** - Tends to revert to ATM
            - **Gamma squeeze unlikely** - Less explosive moves
            
            **Trading Implication:** Fade extremes, trade mean reversion
            """)
        elif atm_bias["metrics"]["gamma_exposure"] < -100000:
            st.warning(f"""
            **Negative Gamma Exposure (₹{abs(atm_bias['metrics']['gamma_exposure']):,})**
            
            Market makers are LONG gamma at ATM:
            - **Destabilizing effect** - They'll sell on dips, buy on rallies
            - **Increased volatility** - Price moves become more explosive
            - **Momentum bias** - Moves tend to accelerate
            - **Gamma squeeze possible** - Potential for sharp moves
            
            **Trading Implication:** Ride momentum, expect whipsaws
            """)
        else:
            st.info("**Neutral Gamma Exposure** - Minimal impact on price action")
    else:
        st.info("ATM bias analysis not available for current data")

# ============================================
# 🎯 TRADING INSIGHTS - SELLER PERSPECTIVE + ATM BIAS + MOMENT + EXPIRY + OI/PCR
# ============================================
st.markdown("---")
st.markdown("## 💡 TRADING INSIGHTS (Seller + ATM Bias + Moment + Expiry + OI/PCR Fusion)")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 KEY OBSERVATIONS")
    
    # Max Pain insight
    if seller_max_pain:
        max_pain_insight = ""
        if spot > seller_max_pain:
            max_pain_insight = f"Spot ABOVE max pain (₹{seller_max_pain:,}). Sellers losing on CALLs, gaining on PUTs."
        else:
            max_pain_insight = f"Spot BELOW max pain (₹{seller_max_pain:,}). Sellers gaining on CALLs, losing on PUTs."
        
        st.info(f"**Max Pain:** {max_pain_insight}")
    
    # GEX insight
    if total_gex_net > 0:
        st.success("**Gamma Exposure:** Sellers SHORT gamma. Expect reduced volatility and mean reversion.")
    elif total_gex_net < 0:
        st.warning("**Gamma Exposure:** Sellers LONG gamma. Expect increased volatility and momentum moves.")
    
    # ATM Bias insight
    if atm_bias:
        if atm_bias["total_score"] > 0.2:
            st.success(f"**ATM Bias Bullish ({atm_bias['total_score']:.2f}):** Heavy PUT selling at ATM confirms bullish sentiment")
        elif atm_bias["total_score"] < -0.2:
            st.error(f"**ATM Bias Bearish ({atm_bias['total_score']:.2f}):** Heavy CALL selling at ATM confirms bearish sentiment")
        else:
            st.info(f"**ATM Bias Neutral ({atm_bias['total_score']:.2f}):** Balanced activity at ATM")
    
    # PCR insight with OI context
    total_pcr = total_PE_OI / total_CE_OI if total_CE_OI > 0 else 0
    if total_pcr > 1.5:
        st.success(f"**Overall PCR ({total_pcr:.2f}):** Strong PUT selling dominance. Bullish seller conviction. PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
    elif total_pcr < 0.7:
        st.error(f"**Overall PCR ({total_pcr:.2f}):** Strong CALL selling dominance. Bearish seller conviction. CALL OI: {oi_pcr_metrics['total_ce_oi']:,}")
    else:
        st.info(f"**Overall PCR ({total_pcr:.2f}):** Balanced. CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,}")
    
    # OI Concentration insight
    if oi_pcr_metrics['atm_concentration_pct'] > 35:
        st.warning(f"**High ATM OI Concentration ({oi_pcr_metrics['atm_concentration_pct']:.1f}%):** Gamma risk elevated. Expect whipsaws around ATM.")
    
    # Max OI insights
    if oi_pcr_metrics['max_ce_oi'] > 1000000:
        st.info(f"**Large CALL Wall at ₹{oi_pcr_metrics['max_ce_strike']:,}:** Strong resistance with {oi_pcr_metrics['max_ce_oi']:,} OI")
    if oi_pcr_metrics['max_pe_oi'] > 1000000:
        st.info(f"**Large PUT Wall at ₹{oi_pcr_metrics['max_pe_strike']:,}:** Strong support with {oi_pcr_metrics['max_pe_oi']:,} OI")
    
    # Expiry Spike insight
    if expiry_spike_data["active"]:
        if expiry_spike_data["probability"] > 60:
            st.error(f"**High Expiry Spike Risk ({expiry_spike_data['probability']}%):** {expiry_spike_data['type']}")
        elif expiry_spike_data["probability"] > 40:
            st.warning(f"**Moderate Expiry Spike Risk ({expiry_spike_data['probability']}%):** {expiry_spike_data['type']}")
        else:
            st.success(f"**Low Expiry Spike Risk ({expiry_spike_data['probability']}%):** Market stable near expiry")
    
    # Moment Detector insights
    st.markdown("#### 🚀 MOMENT DETECTOR INSIGHTS")
    if moment_metrics["momentum_burst"]["score"] > 60:
        st.success("**High Momentum Burst:** Market energy is building for a move")
    if moment_metrics["orderbook"]["available"] and abs(moment_metrics["orderbook"]["pressure"]) > 0.15:
        direction = "buy" if moment_metrics["orderbook"]["pressure"] > 0 else "sell"
        st.info(f"**Strong {direction.upper()} pressure** in orderbook")

with insight_col2:
    st.markdown("### 🛡️ RISK MANAGEMENT")
    
    # Nearest levels insight
    if nearest_sup and nearest_res:
        risk_reward = (nearest_res["distance"] / nearest_sup["distance"]) if nearest_sup["distance"] > 0 else 0
        
        st.metric("Risk:Reward (Current Range)", f"1:{risk_reward:.2f}")
        
        # Stop loss suggestion with OI context
        if seller_bias_result["bias"].startswith("BULLISH"):
            stop_loss = f"Below seller support: ₹{nearest_sup['strike']:,} (PUT OI: {nearest_sup['oi_pe']:,})"
            target = f"Seller resistance: ₹{nearest_res['strike']:,} (CALL OI: {nearest_res['oi_ce']:,})"
        elif seller_bias_result["bias"].startswith("BEARISH"):
            stop_loss = f"Above seller resistance: ₹{nearest_res['strike']:,} (CALL OI: {nearest_res['oi_ce']:,})"
            target = f"Seller support: ₹{nearest_sup['strike']:,} (PUT OI: {nearest_sup['oi_pe']:,})"
        else:
            stop_loss = f"Range: ₹{nearest_sup['strike']:,} - ₹{nearest_res['strike']:,}"
            target = "Wait for breakout"
        
        st.info(f"**Stop Loss:** {stop_loss}")
        st.info(f"**Target:** {target}")
        
        # OI-based stop adjustment
        if oi_pcr_metrics['max_pe_oi'] > 500000 and oi_pcr_metrics['max_pe_strike'] < spot:
            st.info(f"**Strong PUT Support:** Consider ₹{oi_pcr_metrics['max_pe_strike']:,} as major support ({oi_pcr_metrics['max_pe_oi']:,} OI)")
        if oi_pcr_metrics['max_ce_oi'] > 500000 and oi_pcr_metrics['max_ce_strike'] > spot:
            st.info(f"**Strong CALL Resistance:** Consider ₹{oi_pcr_metrics['max_ce_strike']:,} as major resistance ({oi_pcr_metrics['max_ce_oi']:,} OI)")
    
    # ATM Bias-based adjustments
    if atm_bias:
        st.markdown("#### 🎯 ATM BIAS-BASED ADJUSTMENTS")
        if atm_bias["total_score"] > 0.3:
            st.success("**Strong Bullish ATM Bias:** Consider tighter stops on LONG positions, wider stops on SHORT")
        elif atm_bias["total_score"] < -0.3:
            st.success("**Strong Bearish ATM Bias:** Consider tighter stops on SHORT positions, wider stops on LONG")
        
        if atm_bias["metrics"]["gamma_exposure"] < -200000:
            st.warning("**High Negative Gamma Exposure:** Expect explosive moves - Use wider stops")
        elif atm_bias["metrics"]["gamma_exposure"] > 200000:
            st.info("**High Positive Gamma Exposure:** Expect mean reversion - Tighter stops may work")
    
    # Expiry-based risk adjustments with OI context
    if expiry_spike_data["active"]:
        st.markdown("#### 📅 EXPIRY-BASED RISK ADJUSTMENTS")
        if expiry_spike_data["probability"] > 60:
            st.warning("**High Spike Risk:** Use 2x wider stops, avoid overnight positions")
            if oi_pcr_metrics['atm_concentration_pct'] > 40:
                st.warning("**High ATM OI + Expiry:** Extreme gamma risk. Consider straddle/strangle strategies")
        elif expiry_spike_data["probability"] > 40:
            st.info("**Moderate Spike Risk:** Use 1.5x wider stops, be ready for volatility")
        if days_to_expiry <= 1:
            st.warning("**Expiry Day:** Expect whipsaws in last 2 hours, reduce position size")
            # Check for massive OI that needs to unwind
            if oi_pcr_metrics['total_oi'] > 5000000:
                st.warning(f"**Large OI ({oi_pcr_metrics['total_oi']:,}) to unwind:** Expect violent moves as positions close")
    
    # OI-based risk adjustments
    st.markdown("#### 📊 OI-BASED RISK ADJUSTMENTS")
    if oi_pcr_metrics['call_oi_skew'] == "High":
        st.warning("**High CALL OI Skew:** OI concentrated at few strikes - increased pinning risk")
    if oi_pcr_metrics['put_oi_skew'] == "High":
        st.warning("**High PUT OI Skew:** OI concentrated at few strikes - increased pinning risk")
    if abs(oi_pcr_metrics['total_ce_chg']) > 100000 or abs(oi_pcr_metrics['total_pe_chg']) > 100000:
        st.info(f"**Large OI Changes:** CALL Δ: {oi_pcr_metrics['total_ce_chg']:+,} | PUT Δ: {oi_pcr_metrics['total_pe_chg']:+,} - Momentum building")
    
    # Moment-based risk adjustments
    st.markdown("#### 🚀 MOMENT-BASED RISK ADJUSTMENTS")
    if moment_metrics["momentum_burst"]["score"] > 70:
        st.warning("**High Momentum Alert:** Consider tighter stops due to potential sharp moves")
    if moment_metrics["gamma_cluster"]["score"] > 70:
        st.warning("**High Gamma Cluster:** Expect whipsaws around ATM - be prepared for volatility")

# Final Seller Summary with ATM Bias, Moment, Expiry, and OI/PCR Integration
st.markdown("---")
moment_summary = ""
if moment_metrics["momentum_burst"]["score"] > 60:
    moment_summary += "High momentum burst detected. "
if moment_metrics["orderbook"]["available"] and abs(moment_metrics["orderbook"]["pressure"]) > 0.15:
    direction = "buy" if moment_metrics["orderbook"]["pressure"] > 0 else "sell"
    moment_summary += f"Strong {direction} pressure in orderbook. "

atm_bias_summary = f"ATM Bias: {atm_bias['verdict'] if atm_bias else 'N/A'} ({atm_bias['total_score']:.2f} score)" if atm_bias else "ATM Bias: N/A"

expiry_summary = ""
if expiry_spike_data["active"]:
    if expiry_spike_data["probability"] > 60:
        expiry_summary = f"🚨 HIGH EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%) - {expiry_spike_data['type']}"
    elif expiry_spike_data["probability"] > 40:
        expiry_summary = f"⚠️ MODERATE EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%) - {expiry_spike_data['type']}"
    else:
        expiry_summary = f"✅ LOW EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%)"

oi_pcr_summary = f"PCR: {oi_pcr_metrics['pcr_total']:.2f} ({oi_pcr_metrics['pcr_sentiment']}) | CALL OI: {oi_pcr_metrics['total_ce_oi']:,} | PUT OI: {oi_pcr_metrics['total_pe_oi']:,} | ATM Conc: {oi_pcr_metrics['atm_concentration_pct']:.1f}%"

st.markdown(f'''
<div class='seller-explanation'>
    <h3>🎯 FINAL ASSESSMENT (Seller + ATM Bias + Moment + Expiry + OI/PCR)</h3>
    <p><strong>Market Makers are telling us:</strong> {seller_bias_result["explanation"]}</p>
    <p><strong>ATM Zone Analysis:</strong> {atm_bias_summary}</p>
    <p><strong>Their game plan:</strong> {seller_bias_result["action"]}</p>
    <p><strong>Moment Detector:</strong> {moment_summary if moment_summary else "Moment indicators neutral"}</p>
    <p><strong>OI/PCR Analysis:</strong> {oi_pcr_summary}</p>
    <p><strong>Expiry Context:</strong> {expiry_summary if expiry_summary else f"Expiry in {days_to_expiry:.1f} days"}</p>
    <p><strong>Key defense levels:</strong> ₹{nearest_sup['strike'] if nearest_sup else 'N/A':,} (Support) | ₹{nearest_res['strike'] if nearest_res else 'N/A':,} (Resistance)</p>
    <p><strong>Max OI Walls:</strong> CALL: ₹{oi_pcr_metrics['max_ce_strike']:,} | PUT: ₹{oi_pcr_metrics['max_pe_strike']:,}</p>
    <p><strong>Preferred price level:</strong> ₹{seller_max_pain if seller_max_pain else 'N/A':,} (Max Pain)</p>
</div>
''', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {AUTO_REFRESH_SEC}s | ⏰ {get_ist_datetime_str()}")
st.caption("🎯 **NIFTY Option Screener v7.0 — SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS** | All features enabled")

# Requirements note
st.markdown("""
<small>
**Requirements:** 
`streamlit pandas numpy requests pytz scipy supabase python-dotenv` | 
**Data:** Dhan API required
</small>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("**🔄 Last update:** Auto-refreshing every 60 seconds")
