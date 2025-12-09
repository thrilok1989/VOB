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
12. ATM ±2 Strike Bias Analysis
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
# 🎯 ATM BIAS ANALYZER (NEW) - WITH ±2 STRIKES
# ============================================
def analyze_atm_bias(merged_df, spot, atm_strike, strike_gap):
    """
    Analyze ATM bias from multiple perspectives for sellers
    Includes ±2 strikes around ATM
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
        "OI_Change_Bias": 0,
        "Price_Level_Bias": 0,
        "Vega_Bias": 0,
        "Theta_Bias": 0
    }
    
    bias_interpretations = {}
    bias_emojis = {}
    
    # Calculate totals for ATM ±2 zone
    total_ce_oi_atm = atm_df["OI_CE"].sum()
    total_pe_oi_atm = atm_df["OI_PE"].sum()
    total_ce_chg_atm = atm_df["Chg_OI_CE"].sum()
    total_pe_chg_atm = atm_df["Chg_OI_PE"].sum()
    total_ce_vol_atm = atm_df["Vol_CE"].sum()
    total_pe_vol_atm = atm_df["Vol_PE"].sum()
    
    # 1. OI BIAS (CALL vs PUT OI)
    oi_ratio = total_pe_oi_atm / max(total_ce_oi_atm, 1)
    
    if oi_ratio > 1.5:
        bias_scores["OI_Bias"] = 1
        bias_interpretations["OI_Bias"] = f"Heavy PUT OI at ATM±2 ({oi_ratio:.2f}:1) → Bullish sellers"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio > 1.0:
        bias_scores["OI_Bias"] = 0.5
        bias_interpretations["OI_Bias"] = f"Moderate PUT OI ({oi_ratio:.2f}:1) → Mild bullish"
        bias_emojis["OI_Bias"] = "🐂 Bullish"
    elif oi_ratio < 0.7:
        bias_scores["OI_Bias"] = -1
        bias_interpretations["OI_Bias"] = f"Heavy CALL OI at ATM±2 ({oi_ratio:.2f}:1) → Bearish sellers"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    elif oi_ratio < 1.0:
        bias_scores["OI_Bias"] = -0.5
        bias_interpretations["OI_Bias"] = f"Moderate CALL OI ({oi_ratio:.2f}:1) → Mild bearish"
        bias_emojis["OI_Bias"] = "🐻 Bearish"
    else:
        bias_scores["OI_Bias"] = 0
        bias_interpretations["OI_Bias"] = "Balanced OI → Neutral"
        bias_emojis["OI_Bias"] = "⚖️ Neutral"
    
    # 2. CHANGE IN OI BIAS (CALL vs PUT ΔOI)
    if total_pe_chg_atm > 0 and total_ce_chg_atm > 0:
        # Both sides writing
        if total_pe_chg_atm > total_ce_chg_atm:
            bias_scores["ChgOI_Bias"] = 0.5
            bias_interpretations["ChgOI_Bias"] = f"More PUT writing ({total_pe_chg_atm:,} vs {total_ce_chg_atm:,}) → Bullish buildup"
            bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
        else:
            bias_scores["ChgOI_Bias"] = -0.5
            bias_interpretations["ChgOI_Bias"] = f"More CALL writing ({total_ce_chg_atm:,} vs {total_pe_chg_atm:,}) → Bearish buildup"
            bias_emojis["ChgOI_Bias"] = "🐻 Bearish"
    elif total_pe_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = 1
        bias_interpretations["ChgOI_Bias"] = f"Only PUT writing ({total_pe_chg_atm:,}) → Strong bullish"
        bias_emojis["ChgOI_Bias"] = "🐂 Bullish"
    elif total_ce_chg_atm > 0:
        bias_scores["ChgOI_Bias"] = -1
        bias_interpretations["ChgOI_Bias"] = f"Only CALL writing ({total_ce_chg_atm:,}) → Strong bearish"
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
    vol_ratio = total_pe_vol_atm / max(total_ce_vol_atm, 1)
    
    if vol_ratio > 1.3:
        bias_scores["Volume_Bias"] = 1
        bias_interpretations["Volume_Bias"] = f"High PUT volume ({vol_ratio:.2f}:1) → Bullish activity"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio > 1.0:
        bias_scores["Volume_Bias"] = 0.5
        bias_interpretations["Volume_Bias"] = f"More PUT volume ({vol_ratio:.2f}:1) → Mild bullish"
        bias_emojis["Volume_Bias"] = "🐂 Bullish"
    elif vol_ratio < 0.8:
        bias_scores["Volume_Bias"] = -1
        bias_interpretations["Volume_Bias"] = f"High CALL volume ({vol_ratio:.2f}:1) → Bearish activity"
        bias_emojis["Volume_Bias"] = "🐻 Bearish"
    elif vol_ratio < 1.0:
        bias_scores["Volume_Bias"] = -0.5
        bias_interpretations["Volume_Bias"] = f"More CALL volume ({vol_ratio:.2f}:1) → Mild bearish"
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
        bias_interpretations["Delta_Bias"] = f"Positive delta ({net_delta:.3f}) → CALL heavy → Bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    elif net_delta > 0.1:
        bias_scores["Delta_Bias"] = -0.5
        bias_interpretations["Delta_Bias"] = f"Mild positive delta ({net_delta:.3f}) → Slightly bearish"
        bias_emojis["Delta_Bias"] = "🐻 Bearish"
    elif net_delta < -0.3:
        bias_scores["Delta_Bias"] = 1  # Negative delta = PUT heavy = Bullish for sellers
        bias_interpretations["Delta_Bias"] = f"Negative delta ({net_delta:.3f}) → PUT heavy → Bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    elif net_delta < -0.1:
        bias_scores["Delta_Bias"] = 0.5
        bias_interpretations["Delta_Bias"] = f"Mild negative delta ({net_delta:.3f}) → Slightly bullish"
        bias_emojis["Delta_Bias"] = "🐂 Bullish"
    else:
        bias_scores["Delta_Bias"] = 0
        bias_interpretations["Delta_Bias"] = f"Neutral delta ({net_delta:.3f})"
        bias_emojis["Delta_Bias"] = "⚖️ Neutral"
    
    # 5. GAMMA BIAS (Net Gamma Position)
    total_gamma_ce = atm_df["Gamma_CE"].sum()
    total_gamma_pe = atm_df["Gamma_PE"].sum()
    net_gamma = total_gamma_ce + total_gamma_pe
    
    # For sellers: Positive gamma = stabilizing, Negative gamma = explosive
    if net_gamma > 0.1:
        bias_scores["Gamma_Bias"] = 1
        bias_interpretations["Gamma_Bias"] = f"Positive gamma ({net_gamma:.3f}) → Stabilizing → Bullish (less volatility)"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma > 0:
        bias_scores["Gamma_Bias"] = 0.5
        bias_interpretations["Gamma_Bias"] = f"Mild positive gamma ({net_gamma:.3f}) → Slightly stabilizing"
        bias_emojis["Gamma_Bias"] = "🐂 Bullish"
    elif net_gamma < -0.1:
        bias_scores["Gamma_Bias"] = -1
        bias_interpretations["Gamma_Bias"] = f"Negative gamma ({net_gamma:.3f}) → Explosive → Bearish (high volatility)"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    elif net_gamma < 0:
        bias_scores["Gamma_Bias"] = -0.5
        bias_interpretations["Gamma_Bias"] = f"Mild negative gamma ({net_gamma:.3f}) → Slightly explosive"
        bias_emojis["Gamma_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Bias"] = 0
        bias_interpretations["Gamma_Bias"] = f"Neutral gamma ({net_gamma:.3f})"
        bias_emojis["Gamma_Bias"] = "⚖️ Neutral"
    
    # 6. PREMIUM BIAS (CALL vs PUT Premium)
    ce_premium = atm_df["LTP_CE"].mean() if not atm_df["LTP_CE"].isna().all() else 0
    pe_premium = atm_df["LTP_PE"].mean() if not atm_df["LTP_PE"].isna().all() else 0
    premium_ratio = pe_premium / max(ce_premium, 0.01)
    
    if premium_ratio > 1.2:
        bias_scores["Premium_Bias"] = 1
        bias_interpretations["Premium_Bias"] = f"PUT premium higher ({premium_ratio:.2f}:1) → Bullish sentiment"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio > 1.0:
        bias_scores["Premium_Bias"] = 0.5
        bias_interpretations["Premium_Bias"] = f"PUT premium slightly higher ({premium_ratio:.2f}:1) → Mild bullish"
        bias_emojis["Premium_Bias"] = "🐂 Bullish"
    elif premium_ratio < 0.8:
        bias_scores["Premium_Bias"] = -1
        bias_interpretations["Premium_Bias"] = f"CALL premium higher ({premium_ratio:.2f}:1) → Bearish sentiment"
        bias_emojis["Premium_Bias"] = "🐻 Bearish"
    elif premium_ratio < 1.0:
        bias_scores["Premium_Bias"] = -0.5
        bias_interpretations["Premium_Bias"] = f"CALL premium slightly higher ({premium_ratio:.2f}:1) → Mild bearish"
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
        bias_interpretations["IV_Bias"] = f"PUT IV higher ({pe_iv:.1f}% vs {ce_iv:.1f}%) → Bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif pe_iv > ce_iv + 1:
        bias_scores["IV_Bias"] = 0.5
        bias_interpretations["IV_Bias"] = f"PUT IV slightly higher ({pe_iv:.1f}% vs {ce_iv:.1f}%) → Mild bullish fear"
        bias_emojis["IV_Bias"] = "🐂 Bullish"
    elif ce_iv > pe_iv + 3:
        bias_scores["IV_Bias"] = -1
        bias_interpretations["IV_Bias"] = f"CALL IV higher ({ce_iv:.1f}% vs {pe_iv:.1f}%) → Bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    elif ce_iv > pe_iv + 1:
        bias_scores["IV_Bias"] = -0.5
        bias_interpretations["IV_Bias"] = f"CALL IV slightly higher ({ce_iv:.1f}% vs {pe_iv:.1f}%) → Mild bearish fear"
        bias_emojis["IV_Bias"] = "🐻 Bearish"
    else:
        bias_scores["IV_Bias"] = 0
        bias_interpretations["IV_Bias"] = f"Balanced IV (CALL: {ce_iv:.1f}%, PUT: {pe_iv:.1f}%)"
        bias_emojis["IV_Bias"] = "⚖️ Neutral"
    
    # 8. DELTA EXPOSURE BIAS (OI-weighted Delta)
    delta_exposure_ce = (atm_df["Delta_CE"] * atm_df["OI_CE"]).sum()
    delta_exposure_pe = (atm_df["Delta_PE"] * atm_df["OI_PE"]).sum()
    net_delta_exposure = delta_exposure_ce + delta_exposure_pe
    
    if net_delta_exposure > 1000000:
        bias_scores["Delta_Exposure_Bias"] = -1
        bias_interpretations["Delta_Exposure_Bias"] = f"High CALL delta exposure (₹{net_delta_exposure:,}) → Bearish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    elif net_delta_exposure > 500000:
        bias_scores["Delta_Exposure_Bias"] = -0.5
        bias_interpretations["Delta_Exposure_Bias"] = f"Moderate CALL delta exposure (₹{net_delta_exposure:,}) → Slightly bearish"
        bias_emojis["Delta_Exposure_Bias"] = "🐻 Bearish"
    elif net_delta_exposure < -1000000:
        bias_scores["Delta_Exposure_Bias"] = 1
        bias_interpretations["Delta_Exposure_Bias"] = f"High PUT delta exposure (₹{abs(net_delta_exposure):,}) → Bullish pressure"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    elif net_delta_exposure < -500000:
        bias_scores["Delta_Exposure_Bias"] = 0.5
        bias_interpretations["Delta_Exposure_Bias"] = f"Moderate PUT delta exposure (₹{abs(net_delta_exposure):,}) → Slightly bullish"
        bias_emojis["Delta_Exposure_Bias"] = "🐂 Bullish"
    else:
        bias_scores["Delta_Exposure_Bias"] = 0
        bias_interpretations["Delta_Exposure_Bias"] = f"Balanced delta exposure (₹{net_delta_exposure:,})"
        bias_emojis["Delta_Exposure_Bias"] = "⚖️ Neutral"
    
    # 9. GAMMA EXPOSURE BIAS (OI-weighted Gamma)
    gamma_exposure_ce = (atm_df["Gamma_CE"] * atm_df["OI_CE"]).sum()
    gamma_exposure_pe = (atm_df["Gamma_PE"] * atm_df["OI_PE"]).sum()
    net_gamma_exposure = gamma_exposure_ce + gamma_exposure_pe
    
    if net_gamma_exposure > 500000:
        bias_scores["Gamma_Exposure_Bias"] = 1
        bias_interpretations["Gamma_Exposure_Bias"] = f"Positive gamma exposure (₹{net_gamma_exposure:,}) → Stabilizing → Bullish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure > 100000:
        bias_scores["Gamma_Exposure_Bias"] = 0.5
        bias_interpretations["Gamma_Exposure_Bias"] = f"Mild positive gamma (₹{net_gamma_exposure:,}) → Slightly stabilizing"
        bias_emojis["Gamma_Exposure_Bias"] = "🐂 Bullish"
    elif net_gamma_exposure < -500000:
        bias_scores["Gamma_Exposure_Bias"] = -1
        bias_interpretations["Gamma_Exposure_Bias"] = f"Negative gamma exposure (₹{abs(net_gamma_exposure):,}) → Explosive → Bearish"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    elif net_gamma_exposure < -100000:
        bias_scores["Gamma_Exposure_Bias"] = -0.5
        bias_interpretations["Gamma_Exposure_Bias"] = f"Mild negative gamma (₹{abs(net_gamma_exposure):,}) → Slightly explosive"
        bias_emojis["Gamma_Exposure_Bias"] = "🐻 Bearish"
    else:
        bias_scores["Gamma_Exposure_Bias"] = 0
        bias_interpretations["Gamma_Exposure_Bias"] = f"Balanced gamma exposure (₹{net_gamma_exposure:,})"
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
            bias_interpretations["IV_Skew_Bias"] = f"ATM CALL IV higher ({atm_ce_iv:.1f}% vs {nearby_ce_iv:.1f}%) → Bearish skew"
            bias_emojis["IV_Skew_Bias"] = "🐻 Bearish"
        elif atm_pe_iv > nearby_pe_iv + 2:
            bias_scores["IV_Skew_Bias"] = 0.5
            bias_interpretations["IV_Skew_Bias"] = f"ATM PUT IV higher ({atm_pe_iv:.1f}% vs {nearby_pe_iv:.1f}%) → Bullish skew"
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
                bias_interpretations["OI_Change_Bias"] = f"Rapid PUT OI buildup ({oi_change_rate:.1%}) → Bullish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐂 Bullish"
            else:
                bias_scores["OI_Change_Bias"] = -0.5
                bias_interpretations["OI_Change_Bias"] = f"Rapid CALL OI buildup ({oi_change_rate:.1%}) → Bearish acceleration"
                bias_emojis["OI_Change_Bias"] = "🐻 Bearish"
        else:
            bias_scores["OI_Change_Bias"] = 0
            bias_interpretations["OI_Change_Bias"] = "Slow OI changes"
            bias_emojis["OI_Change_Bias"] = "⚖️ Neutral"
    
    # 12. PRICE LEVEL BIAS (Where is spot relative to ATM zone?)
    avg_atm_strike = atm_df["strikePrice"].mean()
    spot_vs_atm_pct = ((spot - avg_atm_strike) / avg_atm_strike) * 100
    
    if spot_vs_atm_pct > 0.5:
        bias_scores["Price_Level_Bias"] = -0.5
        bias_interpretations["Price_Level_Bias"] = f"Spot {spot_vs_atm_pct:.2f}% above ATM zone → Bearish tilt"
        bias_emojis["Price_Level_Bias"] = "🐻 Bearish"
    elif spot_vs_atm_pct < -0.5:
        bias_scores["Price_Level_Bias"] = 0.5
        bias_interpretations["Price_Level_Bias"] = f"Spot {abs(spot_vs_atm_pct):.2f}% below ATM zone → Bullish tilt"
        bias_emojis["Price_Level_Bias"] = "🐂 Bullish"
    else:
        bias_scores["Price_Level_Bias"] = 0
        bias_interpretations["Price_Level_Bias"] = f"Spot near ATM zone ({spot_vs_atm_pct:.2f}%)"
        bias_emojis["Price_Level_Bias"] = "⚖️ Neutral"
    
    # 13. VEGA BIAS (Volatility exposure)
    total_vega_ce = atm_df["Vega_CE"].sum()
    total_vega_pe = atm_df["Vega_PE"].sum()
    net_vega = total_vega_ce + total_vega_pe
    
    if net_vega > 10000:
        bias_scores["Vega_Bias"] = -0.5
        bias_interpretations["Vega_Bias"] = f"High vega exposure ({net_vega:.0f}) → Sensitive to IV changes"
        bias_emojis["Vega_Bias"] = "📈 Volatile"
    elif net_vega < -10000:
        bias_scores["Vega_Bias"] = 0.5
        bias_interpretations["Vega_Bias"] = f"Negative vega ({net_vega:.0f}) → IV increase hurts"
        bias_emojis["Vega_Bias"] = "📉 IV Sensitive"
    else:
        bias_scores["Vega_Bias"] = 0
        bias_interpretations["Vega_Bias"] = "Moderate vega exposure"
        bias_emojis["Vega_Bias"] = "⚖️ Neutral"
    
    # 14. THETA BIAS (Time decay)
    total_theta_ce = atm_df["Theta_CE"].sum()
    total_theta_pe = atm_df["Theta_PE"].sum()
    net_theta = total_theta_ce + total_theta_pe
    
    if net_theta > 10000:
        bias_scores["Theta_Bias"] = 1
        bias_interpretations["Theta_Bias"] = f"Positive theta ({net_theta:.0f}) → Time decay benefits sellers"
        bias_emojis["Theta_Bias"] = "⏳ Decay"
    elif net_theta < -10000:
        bias_scores["Theta_Bias"] = -1
        bias_interpretations["Theta_Bias"] = f"Negative theta ({net_theta:.0f}) → Time decay hurts"
        bias_emojis["Theta_Bias"] = "⌛ Hurts"
    else:
        bias_scores["Theta_Bias"] = 0
        bias_interpretations["Theta_Bias"] = "Moderate theta"
        bias_emojis["Theta_Bias"] = "⚖️ Neutral"
    
    # Calculate final bias score
    total_score = sum(bias_scores.values())
    normalized_score = total_score / len(bias_scores) if bias_scores else 0
    
    # Determine overall verdict
    if normalized_score > 0.3:
        verdict = "🐂 STRONG BULLISH"
        verdict_color = "#00ff88"
        verdict_explanation = "ATM±2 zone showing strong bullish bias for sellers"
    elif normalized_score > 0.1:
        verdict = "🐂 Mild Bullish"
        verdict_color = "#00cc66"
        verdict_explanation = "ATM±2 zone leaning bullish for sellers"
    elif normalized_score < -0.3:
        verdict = "🐻 STRONG BEARISH"
        verdict_color = "#ff4444"
        verdict_explanation = "ATM±2 zone showing strong bearish bias for sellers"
    elif normalized_score < -0.1:
        verdict = "🐻 Mild Bearish"
        verdict_color = "#ff6666"
        verdict_explanation = "ATM±2 zone leaning bearish for sellers"
    else:
        verdict = "⚖️ NEUTRAL"
        verdict_color = "#66b3ff"
        verdict_explanation = "ATM±2 zone balanced, no clear bias"
    
    # Get strikes included in ATM±2
    atm_strikes_list = sorted(atm_strikes)
    
    return {
        "instrument": "NIFTY",
        "strike": atm_strike,
        "strikes_included": atm_strikes_list,
        "zone": "ATM±2",
        "level": "ATM Cluster ±2 Strikes",
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
            "net_vega": round(net_vega, 0),
            "net_theta": round(net_theta, 0),
            "ce_iv": round(ce_iv, 2),
            "pe_iv": round(pe_iv, 2),
            "delta_exposure": int(net_delta_exposure),
            "gamma_exposure": int(net_gamma_exposure),
            "oi_ratio": round(oi_ratio, 2),
            "vol_ratio": round(vol_ratio, 2),
            "premium_ratio": round(premium_ratio, 2),
            "strike_range": f"{min(atm_strikes_list):,} to {max(atm_strikes_list):,}",
            "strike_count": len(atm_strikes_list)
        }
    }

# ============================================
# 🎯 STRIKE-BY-STRIKE BIAS ANALYSIS (ATM±2)
# ============================================
def analyze_strike_by_strike_bias(merged_df, spot, atm_strike, strike_gap):
    """
    Analyze bias for each individual strike in ATM±2 zone
    Returns detailed analysis per strike
    """
    
    # Define ATM window (±2 strikes around ATM)
    atm_window = 2
    atm_strikes = [s for s in merged_df["strikePrice"] 
                  if abs(s - atm_strike) <= (atm_window * strike_gap)]
    
    atm_df = merged_df[merged_df["strikePrice"].isin(atm_strikes)].copy().sort_values("strikePrice")
    
    if atm_df.empty:
        return None
    
    strike_analyses = []
    
    for _, row in atm_df.iterrows():
        strike = int(row["strikePrice"])
        
        # Calculate individual strike bias
        strike_bias = 0
        strike_factors = []
        
        # OI Ratio at this strike
        oi_ce = int(row["OI_CE"])
        oi_pe = int(row["OI_PE"])
        oi_ratio = oi_pe / max(oi_ce, 1)
        
        if oi_ratio > 1.5:
            strike_bias += 1
            strike_factors.append(f"High PUT OI ({oi_ratio:.1f}:1)")
        elif oi_ratio > 1.0:
            strike_bias += 0.5
            strike_factors.append(f"More PUT OI ({oi_ratio:.1f}:1)")
        elif oi_ratio < 0.7:
            strike_bias -= 1
            strike_factors.append(f"High CALL OI ({oi_ratio:.1f}:1)")
        elif oi_ratio < 1.0:
            strike_bias -= 0.5
            strike_factors.append(f"More CALL OI ({oi_ratio:.1f}:1)")
        
        # OI Change
        chg_oi_ce = int(row["Chg_OI_CE"])
        chg_oi_pe = int(row["Chg_OI_PE"])
        
        if chg_oi_pe > 0 and chg_oi_ce <= 0:
            strike_bias += 1
            strike_factors.append(f"PUT writing ({chg_oi_pe:+,})")
        elif chg_oi_pe > 0 and chg_oi_ce > 0 and chg_oi_pe > chg_oi_ce:
            strike_bias += 0.5
            strike_factors.append(f"More PUT writing")
        elif chg_oi_ce > 0 and chg_oi_pe <= 0:
            strike_bias -= 1
            strike_factors.append(f"CALL writing ({chg_oi_ce:+,})")
        elif chg_oi_ce > 0 and chg_oi_pe > 0 and chg_oi_ce > chg_oi_pe:
            strike_bias -= 0.5
            strike_factors.append(f"More CALL writing")
        
        # Premium comparison
        ltp_ce = float(row["LTP_CE"])
        ltp_pe = float(row["LTP_PE"])
        if ltp_ce > 0 and ltp_pe > 0:
            premium_ratio = ltp_pe / ltp_ce
            if premium_ratio > 1.2:
                strike_bias += 0.5
                strike_factors.append(f"PUT premium high")
            elif premium_ratio < 0.8:
                strike_bias -= 0.5
                strike_factors.append(f"CALL premium high")
        
        # IV comparison
        iv_ce = float(row["IV_CE"]) if not pd.isna(row["IV_CE"]) else 0
        iv_pe = float(row["IV_PE"]) if not pd.isna(row["IV_PE"]) else 0
        if iv_ce > 0 and iv_pe > 0:
            if iv_pe > iv_ce + 3:
                strike_bias += 0.5
                strike_factors.append(f"PUT IV higher")
            elif iv_ce > iv_pe + 3:
                strike_bias -= 0.5
                strike_factors.append(f"CALL IV higher")
        
        # Determine verdict for this strike
        if strike_bias > 0.5:
            verdict = "🐂 Bullish"
            color = "#00ff88"
        elif strike_bias > 0:
            verdict = "🐂 Mild Bullish"
            color = "#00cc66"
        elif strike_bias < -0.5:
            verdict = "🐻 Bearish"
            color = "#ff4444"
        elif strike_bias < 0:
            verdict = "🐻 Mild Bearish"
            color = "#ff6666"
        else:
            verdict = "⚖️ Neutral"
            color = "#66b3ff"
        
        # Distance from spot
        distance_from_spot = strike - spot
        distance_pct = (distance_from_spot / spot) * 100
        
        # Position relative to spot
        if strike < spot:
            position = "ITM PUT/OTM CALL"
        elif strike > spot:
            position = "ITM CALL/OTM PUT"
        else:
            position = "ATM"
        
        strike_analyses.append({
            "strike": strike,
            "distance_from_spot": distance_from_spot,
            "distance_pct": distance_pct,
            "position": position,
            "oi_ce": oi_ce,
            "oi_pe": oi_pe,
            "oi_ratio": oi_ratio,
            "chg_oi_ce": chg_oi_ce,
            "chg_oi_pe": chg_oi_pe,
            "ltp_ce": ltp_ce,
            "ltp_pe": ltp_pe,
            "iv_ce": iv_ce,
            "iv_pe": iv_pe,
            "delta_ce": float(row["Delta_CE"]),
            "delta_pe": float(row["Delta_PE"]),
            "gamma_ce": float(row["Gamma_CE"]),
            "gamma_pe": float(row["Gamma_PE"]),
            "bias_score": strike_bias,
            "verdict": verdict,
            "color": color,
            "factors": strike_factors
        })
    
    return strike_analyses

# ============================================
# 🎯 COMPREHENSIVE BIAS DASHBOARD (ENHANCED)
# ============================================
def display_bias_dashboard(atm_bias, support_bias, resistance_bias, strike_analyses=None):
    """Display comprehensive bias dashboard with strike-by-strike analysis"""
    
    st.markdown("## 🎯 MULTI-DIMENSIONAL BIAS ANALYSIS")
    
    # Create columns for each bias analysis
    col_atm, col_sup, col_res = st.columns(3)
    
    with col_atm:
        if atm_bias:
            st.markdown(f"""
            <div class='card' style='border-color:{atm_bias["verdict_color"]};'>
                <h4 style='color:{atm_bias["verdict_color"]};'>🏛️ ATM±2 ZONE BIAS</h4>
                <div style='font-size: 1.8rem; color:{atm_bias["verdict_color"]}; font-weight:900; text-align:center;'>
                    {atm_bias["verdict"]}
                </div>
                <div style='font-size: 1.2rem; color:#ffcc00; text-align:center;'>
                    ₹{atm_bias["strike"]:,}
                </div>
                <div style='font-size: 0.9rem; color:#cccccc; text-align:center; margin-top:10px;'>
                    Score: {atm_bias["total_score"]:.2f}
                </div>
                <div style='font-size: 0.8rem; color:#aaaaaa; text-align:center;'>
                    Strikes: {atm_bias['metrics']['strike_range']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
            st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
            st.metric("OI Ratio", f"{atm_bias['metrics']['oi_ratio']:.2f}")
            st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
    
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
    
    # STRIKE-BY-STRIKE ANALYSIS (NEW)
    if strike_analyses:
        st.markdown("### 🔍 STRIKE-BY-STRIKE ANALYSIS (ATM±2)")
        
        # Create a DataFrame for display
        strike_data = []
        for analysis in strike_analyses:
            strike_data.append({
                "Strike": f"₹{analysis['strike']:,}",
                "Distance": f"{analysis['distance_from_spot']:+.0f}",
                "Position": analysis['position'],
                "CALL OI": f"{analysis['oi_ce']:,}",
                "PUT OI": f"{analysis['oi_pe']:,}",
                "ΔCALL": f"{analysis['chg_oi_ce']:+,}",
                "ΔPUT": f"{analysis['chg_oi_pe']:+,}",
                "OI Ratio": f"{analysis['oi_ratio']:.2f}",
                "Bias": analysis['verdict'],
                "Score": f"{analysis['bias_score']:.2f}"
            })
        
        strike_df = pd.DataFrame(strike_data)
        
        # Color the bias column
        def color_bias(val):
            if "Bullish" in val:
                return "background-color: #1a2e1a; color: #00ff88"
            elif "Bearish" in val:
                return "background-color: #2e1a1a; color: #ff4444"
            else:
                return "background-color: #1a1f2e; color: #66b3ff"
        
        styled_df = strike_df.style.applymap(color_bias, subset=["Bias"])
        st.dataframe(styled_df, use_container_width=True, height=300)
        
        # Add heatmap visualization
        st.markdown("#### 🎯 BIAS HEATMAP ACROSS STRIKES")
        heatmap_cols = st.columns(len(strike_analyses))
        
        for idx, analysis in enumerate(strike_analyses):
            with heatmap_cols[idx]:
                # Calculate heat intensity
                intensity = min(100, max(0, 50 + (analysis['bias_score'] * 25)))
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(to bottom, rgba(0,0,0,0) {100-intensity}%, {analysis['color']} {intensity}%);
                    padding: 10px;
                    border-radius: 8px;
                    border: 2px solid {analysis['color']};
                    text-align: center;
                    height: 120px;
                    margin: 5px;
                ">
                    <div style='font-size: 1.1rem; font-weight:700; color:{analysis['color']};'>
                        ₹{analysis['strike']:,}
                    </div>
                    <div style='font-size: 0.9rem; color:#ffffff;'>
                        {analysis['verdict']}
                    </div>
                    <div style='font-size: 0.8rem; color:#cccccc;'>
                        OI: {analysis['oi_ce']:,}|{analysis['oi_pe']:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed ATM Bias Table
    if atm_bias:
        st.markdown("### 📊 ATM±2 BIAS DETAILED BREAKDOWN")
        
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
            <h4>🎯 ATM±2 BIAS SUMMARY</h4>
            <p><strong>Overall Verdict:</strong> <span style='color:{atm_bias["verdict_color"]}'>{atm_bias["verdict"]}</span></p>
            <p><strong>Total Score:</strong> {atm_bias["total_score"]:.2f}</p>
            <p><strong>Explanation:</strong> {atm_bias["verdict_explanation"]}</p>
            <p><strong>Strikes Analyzed:</strong> {', '.join([f'₹{s:,}' for s in atm_bias['strikes_included']])}</p>
            <p><strong>Key Insights:</strong></p>
            <ul>
                <li>CALL OI: {atm_bias['metrics']['ce_oi']:,} | PUT OI: {atm_bias['metrics']['pe_oi']:,}</li>
                <li>OI Ratio: {atm_bias['metrics']['oi_ratio']:.2f} | Vol Ratio: {atm_bias['metrics']['vol_ratio']:.2f}</li>
                <li>Net Delta: {atm_bias['metrics']['net_delta']:.3f} | Net Gamma: {atm_bias['metrics']['net_gamma']:.3f}</li>
                <li>Delta Exposure: ₹{atm_bias['metrics']['delta_exposure']:,}</li>
                <li>Gamma Exposure: ₹{atm_bias['metrics']['gamma_exposure']:,}</li>
                <li>CALL IV: {atm_bias['metrics']['ce_iv']:.1f}% | PUT IV: {atm_bias['metrics']['pe_iv']:.1f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Trading Implications
    st.markdown("### 💡 TRADING IMPLICATIONS")
    
    implications = []
    
    if atm_bias:
        if atm_bias["total_score"] > 0.2:
            implications.append("✅ **ATM±2 Bullish Bias:** Favor LONG positions with stops below ATM zone")
        elif atm_bias["total_score"] < -0.2:
            implications.append("✅ **ATM±2 Bearish Bias:** Favor SHORT positions with stops above ATM zone")
        
        if atm_bias["metrics"]["gamma_exposure"] < -100000:
            implications.append("⚠️ **Negative Gamma Exposure:** Expect whipsaws around ATM zone")
        elif atm_bias["metrics"]["gamma_exposure"] > 100000:
            implications.append("✅ **Positive Gamma Exposure:** Market stabilizing around ATM zone")
        
        # Strike-by-strike implications
        if strike_analyses:
            bullish_strikes = [s for s in strike_analyses if "Bullish" in s['verdict']]
            bearish_strikes = [s for s in strike_analyses if "Bearish" in s['verdict']]
            
            if len(bullish_strikes) > len(bearish_strikes):
                implications.append(f"✅ **{len(bullish_strikes)}/{len(strike_analyses)} strikes bullish:** Strong zone-wide bias")
            elif len(bearish_strikes) > len(bullish_strikes):
                implications.append(f"⚠️ **{len(bearish_strikes)}/{len(strike_analyses)} strikes bearish:** Strong zone-wide bias")
    
    if support_bias and support_bias["total_score"] > 0.3:
        implications.append(f"✅ **Strong Support at ₹{support_bias['strike']:,}:** Good for LONG entries")
    
    if resistance_bias and resistance_bias["total_score"] < -0.3:
        implications.append(f"✅ **Strong Resistance at ₹{resistance_bias['strike']:,}:** Good for SHORT entries")
    
    if not implications:
        implications.append("⚖️ **Balanced Market:** No clear edge, wait for breakout")
    
    for imp in implications:
        st.markdown(f"- {imp}")

# ============================================
# 🎯 SUPPORT/RESISTANCE BIAS ANALYZER
# ============================================
def analyze_support_resistance_bias(merged_df, spot, atm_strike, strike_gap, level_type="Support"):
    """Analyze bias at key support/resistance levels"""
    
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
# 🎯 ENTRY SIGNAL WITH ATM BIAS INTEGRATION
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
    resistance_bias,
    strike_analyses=None
):
    """
    Enhanced entry signal with comprehensive ATM bias analysis
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
    # 2. ATM BIAS INTEGRATION (25% weight)
    # ============================================
    if atm_bias:
        atm_score = atm_bias["total_score"]
        if position_type == "LONG" and atm_score > 0.1:
            signal_score += int(25 * (atm_score / 1.0))
            signal_reasons.append(f"ATM±2 bias bullish ({atm_score:.2f}) confirms LONG")
        elif position_type == "SHORT" and atm_score < -0.1:
            signal_score += int(25 * (abs(atm_score) / 1.0))
            signal_reasons.append(f"ATM±2 bias bearish ({atm_score:.2f}) confirms SHORT")
        else:
            signal_score -= 10
            signal_reasons.append(f"ATM±2 bias ({atm_score:.2f}) conflicts with seller bias")
    
    # ============================================
    # 3. STRIKE-BY-STRIKE CONFIRMATION (15% weight)
    # ============================================
    if strike_analyses:
        bullish_count = len([s for s in strike_analyses if "Bullish" in s['verdict']])
        bearish_count = len([s for s in strike_analyses if "Bearish" in s['verdict']])
        total_strikes = len(strike_analyses)
        
        if position_type == "LONG" and bullish_count > bearish_count:
            confirmation_ratio = bullish_count / total_strikes
            signal_score += int(15 * confirmation_ratio)
            signal_reasons.append(f"{bullish_count}/{total_strikes} ATM±2 strikes bullish ({confirmation_ratio:.0%})")
        elif position_type == "SHORT" and bearish_count > bullish_count:
            confirmation_ratio = bearish_count / total_strikes
            signal_score += int(15 * confirmation_ratio)
            signal_reasons.append(f"{bearish_count}/{total_strikes} ATM±2 strikes bearish ({confirmation_ratio:.0%})")
    
    # ============================================
    # 4. SUPPORT/RESISTANCE BIAS INTEGRATION (10% weight)
    # ============================================
    if support_bias and position_type == "LONG":
        support_score = support_bias["total_score"]
        if support_score > 0.2:
            signal_score += int(10 * (support_score / 1.0))
            signal_reasons.append(f"Strong support bias ({support_score:.2f}) at ₹{support_bias['strike']:,}")
    
    if resistance_bias and position_type == "SHORT":
        resistance_score = resistance_bias["total_score"]
        if resistance_score < -0.2:
            signal_score += int(10 * (abs(resistance_score) / 1.0))
            signal_reasons.append(f"Strong resistance bias ({resistance_score:.2f}) at ₹{resistance_bias['strike']:,}")
    
    # ============================================
    # 5. MAX PAIN ALIGNMENT (5% weight)
    # ============================================
    if seller_max_pain:
        distance_to_max_pain = abs(spot - seller_max_pain)
        distance_pct = (distance_to_max_pain / spot) * 100
        
        if distance_pct < 0.5:
            signal_score += 5
            signal_reasons.append(f"Spot VERY close to Max Pain (₹{seller_max_pain:,}, {distance_pct:.2f}%)")
            optimal_entry_price = seller_max_pain
    
    # ============================================
    # 6. MOMENT DETECTOR FEATURES (5% weight)
    # ============================================
    mb = moment_metrics.get("momentum_burst", {})
    if mb.get("available", False):
        mb_score = mb.get("score", 0)
        signal_score += int(5 * (mb_score / 100.0))
        signal_reasons.append(f"Momentum burst: {mb_score}/100")
    
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
    
    return {
        "position_type": position_type,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "optimal_entry_price": optimal_entry_price,
        "current_spot": spot,
        "signal_color": signal_color,
        "reasons": signal_reasons,
        "atm_bias_score": atm_bias["total_score"] if atm_bias else 0,
        "strike_confirmation": f"{bullish_count}/{total_strikes}" if strike_analyses else "N/A"
    }

# ============================================
# 🎯 DISPLAY ATM±2 ANALYSIS SECTION
# ============================================
def display_atm_analysis(atm_bias, strike_analyses):
    """Display detailed ATM±2 analysis section"""
    
    st.markdown("## 🎯 ATM±2 STRIKE ANALYSIS")
    
    if not atm_bias:
        st.info("ATM bias analysis not available")
        return
    
    # Summary card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ATM Strike", f"₹{atm_bias['strike']:,}")
        st.caption(f"Zone: {atm_bias['metrics']['strike_range']}")
    
    with col2:
        st.metric("Total Score", f"{atm_bias['total_score']:.2f}")
        st.caption(f"Verdict: {atm_bias['verdict']}")
    
    with col3:
        st.metric("CALL OI", f"{atm_bias['metrics']['ce_oi']:,}")
        st.caption(f"Δ: {atm_bias['metrics']['ce_chg']:+,}")
    
    with col4:
        st.metric("PUT OI", f"{atm_bias['metrics']['pe_oi']:,}")
        st.caption(f"Δ: {atm_bias['metrics']['pe_chg']:+,}")
    
    # Detailed metrics
    with st.expander("📊 DETAILED ATM±2 METRICS", expanded=True):
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.metric("OI Ratio", f"{atm_bias['metrics']['oi_ratio']:.2f}")
            st.metric("Volume Ratio", f"{atm_bias['metrics']['vol_ratio']:.2f}")
            st.metric("Premium Ratio", f"{atm_bias['metrics']['premium_ratio']:.2f}")
            st.metric("Net Delta", f"{atm_bias['metrics']['net_delta']:.3f}")
        
        with col_metrics2:
            st.metric("Net Gamma", f"{atm_bias['metrics']['net_gamma']:.3f}")
            st.metric("CALL IV", f"{atm_bias['metrics']['ce_iv']:.1f}%")
            st.metric("PUT IV", f"{atm_bias['metrics']['pe_iv']:.1f}%")
            st.metric("Strikes Analyzed", atm_bias['metrics']['strike_count'])
    
    # Strike-by-strike details
    if strike_analyses:
        st.markdown("### 🔍 INDIVIDUAL STRIKE ANALYSIS")
        
        for analysis in strike_analyses:
            with st.expander(f"Strike ₹{analysis['strike']:,} - {analysis['verdict']} ({analysis['distance_from_spot']:+.0f} from spot)", expanded=False):
                col_strike1, col_strike2, col_strike3 = st.columns(3)
                
                with col_strike1:
                    st.metric("CALL OI", f"{analysis['oi_ce']:,}")
                    st.metric("Δ CALL", f"{analysis['chg_oi_ce']:+,}")
                    st.metric("CALL IV", f"{analysis['iv_ce']:.1f}%")
                
                with col_strike2:
                    st.metric("PUT OI", f"{analysis['oi_pe']:,}")
                    st.metric("Δ PUT", f"{analysis['chg_oi_pe']:+,}")
                    st.metric("PUT IV", f"{analysis['iv_pe']:.1f}%")
                
                with col_strike3:
                    st.metric("OI Ratio", f"{analysis['oi_ratio']:.2f}")
                    st.metric("Bias Score", f"{analysis['bias_score']:.2f}")
                    st.metric("Position", analysis['position'])
                
                # Key factors
                if analysis['factors']:
                    st.markdown("**Key Factors:**")
                    for factor in analysis['factors']:
                        st.markdown(f"- {factor}")

# ============================================
# 🎯 MAIN APPLICATION DISPLAY
# ============================================
def main():
    """Main application function"""
    
    # Set page config
    st.set_page_config(
        page_title="Nifty Screener v7 - Seller's Perspective + ATM Bias Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #fafafa; }
        .stButton > button { background-color: #ff66cc; color: white; }
        .stButton > button:hover { background-color: #ff99dd; }
        h1, h2, h3 { color: #ff66cc; }
        .card { 
            background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
            padding: 20px; border-radius: 12px; border: 3px solid;
            margin: 10px 0; text-align: center;
        }
        .seller-explanation {
            background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
            padding: 15px; border-radius: 10px; border-left: 4px solid #ff66cc;
            margin: 10px 0;
        }
        .moment-box {
            background: linear-gradient(135deg, #1a1f3e 0%, #2a2f4e 100%);
            padding: 15px; border-radius: 10px; border: 2px solid #00ffff;
            margin: 10px 0; text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎯 NIFTY Option Screener v7.0 — SELLER'S PERSPECTIVE + ATM±2 BIAS ANALYZER")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ SETTINGS")
        
        # Expiry selection
        expiries = get_expiry_list()
        if expiries:
            expiry = st.selectbox("Select Expiry", expiries, index=0)
        else:
            st.error("No expiries available")
            return
        
        # ATM window selection
        atm_window = st.slider("ATM Window (± strikes)", 1, 5, 2)
        
        # Display options
        st.markdown("### 📊 DISPLAY OPTIONS")
        show_strike_details = st.checkbox("Show Strike-by-Strike Details", True)
        show_bias_breakdown = st.checkbox("Show Bias Breakdown", True)
        
        # Refresh control
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        spot = get_nifty_spot_price()
        if spot == 0:
            st.error("Failed to fetch spot price")
            return
        
        chain = fetch_dhan_option_chain(expiry)
        if not chain:
            st.error("Failed to fetch option chain")
            return
    
    # Parse data
    df_ce, df_pe = parse_dhan_option_chain(chain)
    if df_ce.empty or df_pe.empty:
        st.error("Insufficient data")
        return
    
    # Merge data
    merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")
    
    # Calculate ATM
    strike_gap = strike_gap_from_series(merged["strikePrice"])
    atm_strike = min(merged["strikePrice"].tolist(), key=lambda x: abs(x - spot))
    
    # Calculate biases
    atm_bias = analyze_atm_bias(merged, spot, atm_strike, strike_gap)
    strike_analyses = analyze_strike_by_strike_bias(merged, spot, atm_strike, strike_gap)
    support_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Support")
    resistance_bias = analyze_support_resistance_bias(merged, spot, atm_strike, strike_gap, "Resistance")
    
    # Calculate other metrics (simplified for example)
    seller_bias_result = {"bias": "NEUTRAL", "polarity": 0}
    moment_metrics = {"momentum_burst": {"available": False}}
    
    # Calculate entry signal
    entry_signal = calculate_entry_signal_with_atm_bias(
        spot=spot,
        merged_df=merged,
        atm_strike=atm_strike,
        seller_bias_result=seller_bias_result,
        seller_max_pain=None,
        seller_supports_df=None,
        seller_resists_df=None,
        nearest_sup=None,
        nearest_res=None,
        seller_breakout_index=50,
        moment_metrics=moment_metrics,
        atm_bias=atm_bias,
        support_bias=support_bias,
        resistance_bias=resistance_bias,
        strike_analyses=strike_analyses
    )
    
    # ============================================
    # MAIN DISPLAY
    # ============================================
    
    # Current time
    current_ist = get_ist_datetime_str()
    st.markdown(f"**🕐 Last Updated:** {current_ist} IST | **Spot:** ₹{spot:,.2f} | **ATM:** ₹{atm_strike:,}")
    st.markdown("---")
    
    # 1. BIAS DASHBOARD
    display_bias_dashboard(atm_bias, support_bias, resistance_bias, strike_analyses)
    st.markdown("---")
    
    # 2. ENTRY SIGNAL
    st.markdown("## 🚀 ENTRY SIGNAL")
    
    if entry_signal["position_type"] != "NEUTRAL":
        signal_color = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%);
            padding: 25px;
            border-radius: 15px;
            border: 4px solid {signal_color};
            text-align: center;
            margin: 20px 0;
        ">
            <div style="font-size: 2.5rem; color:{signal_color}; font-weight:900;">
                {entry_signal["signal_strength"]} {entry_signal["position_type"]}
            </div>
            <div style="font-size: 1.8rem; color:#ffcc00; font-weight:700; margin: 10px 0;">
                Entry: ₹{entry_signal["optimal_entry_price"]:,.2f}
            </div>
            <div style="font-size: 1.2rem; color:#66b3ff;">
                Confidence: {entry_signal["confidence"]:.0f}%
            </div>
            <div style="font-size: 1rem; color:#cccccc; margin-top: 10px;">
                ATM Bias Score: {entry_signal["atm_bias_score"]:.2f} | 
                Strike Confirmation: {entry_signal["strike_confirmation"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal reasons
        with st.expander("📋 Signal Reasons"):
            for reason in entry_signal["reasons"]:
                st.markdown(f"• {reason}")
    else:
        st.info("⚖️ **NO CLEAR ENTRY SIGNAL** - Market conditions are neutral")
    
    st.markdown("---")
    
    # 3. DETAILED ATM ANALYSIS
    display_atm_analysis(atm_bias, strike_analyses if show_strike_details else None)
    st.markdown("---")
    
    # 4. RAW DATA (Optional)
    with st.expander("📁 VIEW RAW DATA"):
        st.dataframe(merged, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **NIFTY Option Screener v7.0** | ATM±2 Bias Analyzer | All data is for informational purposes only")

# ============================================
# UTILITY FUNCTIONS (from your original code)
# ============================================
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

# Run the app
if __name__ == "__main__":
    main()
