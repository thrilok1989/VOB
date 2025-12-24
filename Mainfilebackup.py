"""
Nifty Option Screener v7.0 - Complete Simplified Version
Seller's Perspective + Market Depth + Iceberg Detection + OI/PCR Analysis
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
import json

# -----------------------
#  CONFIGURATION
# -----------------------
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

def get_ist_time_str():
    return get_ist_now().strftime("%H:%M:%S")

# Auto-refresh
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8

# Load secrets
try:
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except:
    st.error("❌ Missing Dhan credentials. Add them to Streamlit secrets.")
    st.stop()

DHAN_BASE_URL = "https://api.dhan.co"

# -----------------------
#  CUSTOM CSS
# -----------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* Colors */
    .bullish { color: #00ff88 !important; font-weight: 700 !important; }
    .bearish { color: #ff4444 !important; font-weight: 700 !important; }
    .neutral { color: #66b3ff !important; font-weight: 700 !important; }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        margin: 5px 0;
        text-align: center;
    }
    
    .signal-card {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 10px 0;
        text-align: center;
    }
    
    .depth-card {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffcc;
        margin: 10px 0;
    }
    
    .iceberg-alert {
        animation: pulse 2s infinite;
        border-color: #ff00ff !important;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 255, 0); }
    }
    
    h1, h2, h3 { color: #ff66cc !important; }
    
    [data-testid="stMetricLabel"] { color: #cccccc !important; }
    [data-testid="stMetricValue"] { color: #ff66cc !important; font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

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

# -----------------------
#  MARKET DEPTH ANALYSIS
# -----------------------
@st.cache_data(ttl=5)
def get_market_depth():
    """Fetch NIFTY market depth from Dhan API"""
    try:
        url = f"{DHAN_BASE_URL}/v2/marketfeed/depth"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
            "client-id": DHAN_CLIENT_ID
        }
        payload = {"IDX_I": [13]}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("IDX_I", {}).get("13", {}).get("depth", {})
    except Exception as e:
        st.warning(f"Depth fetch failed: {e}")
    return None

def analyze_market_depth(depth_data, spot_price):
    """
    Comprehensive market depth analysis with iceberg detection
    Returns: dict with buy/sell analysis, iceberg orders, support/resistance levels
    """
    if not depth_data:
        return None
    
    analysis = {
        'buy_side': {'levels': [], 'total_volume': 0, 'iceberg_count': 0, 'weighted_avg_price': 0},
        'sell_side': {'levels': [], 'total_volume': 0, 'iceberg_count': 0, 'weighted_avg_price': 0},
        'summary': {},
        'iceberg_orders': [],
        'support_levels': [],
        'resistance_levels': []
    }
    
    # Iceberg detection thresholds
    ICEBERG_THRESHOLD = 50000  # 50k shares considered large order
    STRONG_LEVEL_THRESHOLD = 20000  # 20k shares for strong level
    
    # Analyze buy side (bids)
    if 'buy' in depth_data:
        total_bid_value = 0
        total_bid_volume = 0
        
        for i, bid in enumerate(depth_data['buy'][:15]):  # Top 15 bids
            if isinstance(bid, list) and len(bid) >= 2:
                price = safe_float(bid[0], 0)
                qty = safe_int(bid[1], 0)
                
                bid_info = {
                    'price': price,
                    'quantity': qty,
                    'depth_level': i + 1,
                    'distance_from_spot': spot_price - price,
                    'distance_pct': ((spot_price - price) / spot_price * 100) if spot_price > 0 else 0,
                    'is_iceberg': qty > ICEBERG_THRESHOLD,
                    'is_strong': qty > STRONG_LEVEL_THRESHOLD
                }
                
                analysis['buy_side']['levels'].append(bid_info)
                analysis['buy_side']['total_volume'] += qty
                total_bid_value += price * qty
                total_bid_volume += qty
                
                if qty > ICEBERG_THRESHOLD:
                    analysis['buy_side']['iceberg_count'] += 1
                    analysis['iceberg_orders'].append({
                        'side': 'BUY',
                        'price': price,
                        'quantity': qty,
                        'depth_level': i + 1,
                        'size_category': 'LARGE' if qty > 100000 else 'MEDIUM'
                    })
        
        # Calculate weighted average price for bids
        if total_bid_volume > 0:
            analysis['buy_side']['weighted_avg_price'] = total_bid_value / total_bid_volume
    
    # Analyze sell side (asks)
    if 'sell' in depth_data:
        total_ask_value = 0
        total_ask_volume = 0
        
        for i, ask in enumerate(depth_data['sell'][:15]):  # Top 15 asks
            if isinstance(ask, list) and len(ask) >= 2:
                price = safe_float(ask[0], 0)
                qty = safe_int(ask[1], 0)
                
                ask_info = {
                    'price': price,
                    'quantity': qty,
                    'depth_level': i + 1,
                    'distance_from_spot': price - spot_price,
                    'distance_pct': ((price - spot_price) / spot_price * 100) if spot_price > 0 else 0,
                    'is_iceberg': qty > ICEBERG_THRESHOLD,
                    'is_strong': qty > STRONG_LEVEL_THRESHOLD
                }
                
                analysis['sell_side']['levels'].append(ask_info)
                analysis['sell_side']['total_volume'] += qty
                total_ask_value += price * qty
                total_ask_volume += qty
                
                if qty > ICEBERG_THRESHOLD:
                    analysis['sell_side']['iceberg_count'] += 1
                    analysis['iceberg_orders'].append({
                        'side': 'SELL',
                        'price': price,
                        'quantity': qty,
                        'depth_level': i + 1,
                        'size_category': 'LARGE' if qty > 100000 else 'MEDIUM'
                    })
        
        # Calculate weighted average price for asks
        if total_ask_volume > 0:
            analysis['sell_side']['weighted_avg_price'] = total_ask_value / total_ask_volume
    
    # Calculate depth pressure (buy/sell imbalance)
    total_volume = analysis['buy_side']['total_volume'] + analysis['sell_side']['total_volume']
    if total_volume > 0:
        depth_pressure = (analysis['buy_side']['total_volume'] - analysis['sell_side']['total_volume']) / total_volume
    else:
        depth_pressure = 0
    
    # Identify support levels from strong bids near spot
    for bid in analysis['buy_side']['levels']:
        if bid['is_strong'] and bid['distance_from_spot'] < 50:  # Within 50 points
            support_score = bid['quantity'] / 1000  # Normalize to thousands
            analysis['support_levels'].append({
                'price': bid['price'],
                'strength': support_score,
                'quantity': bid['quantity'],
                'depth_level': bid['depth_level'],
                'type': 'depth_bid',
                'iceberg': bid['is_iceberg'],
                'distance_from_spot': bid['distance_from_spot']
            })
    
    # Identify resistance levels from strong asks near spot
    for ask in analysis['sell_side']['levels']:
        if ask['is_strong'] and ask['distance_from_spot'] < 50:  # Within 50 points
            resistance_score = ask['quantity'] / 1000  # Normalize to thousands
            analysis['resistance_levels'].append({
                'price': ask['price'],
                'strength': resistance_score,
                'quantity': ask['quantity'],
                'depth_level': ask['depth_level'],
                'type': 'depth_ask',
                'iceberg': ask['is_iceberg'],
                'distance_from_spot': ask['distance_from_spot']
            })
    
    # Sort levels by strength
    analysis['support_levels'].sort(key=lambda x: x['strength'], reverse=True)
    analysis['resistance_levels'].sort(key=lambda x: x['strength'], reverse=True)
    
    # Calculate market depth summary
    analysis['summary'] = {
        'total_bid_volume': analysis['buy_side']['total_volume'],
        'total_ask_volume': analysis['sell_side']['total_volume'],
        'depth_pressure': depth_pressure,
        'total_iceberg_orders': len(analysis['iceberg_orders']),
        'bid_iceberg_count': analysis['buy_side']['iceberg_count'],
        'ask_iceberg_count': analysis['sell_side']['iceberg_count'],
        'avg_bid_price': analysis['buy_side']['weighted_avg_price'],
        'avg_ask_price': analysis['sell_side']['weighted_avg_price'],
        'bid_ask_spread': analysis['sell_side']['weighted_avg_price'] - analysis['buy_side']['weighted_avg_price'],
        'strong_supports': len([s for s in analysis['support_levels'] if s['strength'] > 20]),
        'strong_resistances': len([r for r in analysis['resistance_levels'] if r['strength'] > 20])
    }
    
    return analysis

# -----------------------
#  SELLER'S ANALYSIS FUNCTIONS
# -----------------------
def calculate_seller_max_pain(df):
    """Calculate Max Pain level where sellers have minimum loss"""
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))
        
        # Pain calculation
        ce_pain = ce_oi * max(0, ce_ltp)
        pe_pain = pe_oi * max(0, pe_ltp)
        pain = ce_pain + pe_pain
        
        pain_dict[strike] = pain
    
    if pain_dict:
        return min(pain_dict, key=pain_dict.get)
    return None

def calculate_seller_market_bias(merged_df, spot, atm_strike):
    """
    Calculate seller's market bias based on OI changes
    Returns: dict with bias, polarity, and color
    """
    polarity = 0.0
    
    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))
        
        # ITM CALL writing (very bearish)
        if strike < atm_strike and chg_ce > 0:
            polarity -= 2.0
        
        # ITM CALL buying back (bullish)
        elif strike < atm_strike and chg_ce < 0:
            polarity += 1.5
        
        # OTM CALL writing (mildly bearish)
        elif strike > atm_strike and chg_ce > 0:
            polarity -= 0.7
        
        # OTM CALL buying back (mildly bullish)
        elif strike > atm_strike and chg_ce < 0:
            polarity += 0.5
        
        # ITM PUT writing (very bullish)
        if strike > atm_strike and chg_pe > 0:
            polarity += 2.0
        
        # ITM PUT buying back (bearish)
        elif strike > atm_strike and chg_pe < 0:
            polarity -= 1.5
        
        # OTM PUT writing (mildly bullish)
        elif strike < atm_strike and chg_pe > 0:
            polarity += 0.7
        
        # OTM PUT buying back (mildly bearish)
        elif strike < atm_strike and chg_pe < 0:
            polarity -= 0.5
    
    # PCR adjustment
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:
            polarity += 1.0  # Extreme put selling
        elif pcr < 0.5:
            polarity -= 1.0  # Extreme call selling
    
    # IV skew adjustment
    avg_iv_ce = merged_df["IV_CE"].mean()
    avg_iv_pe = merged_df["IV_PE"].mean()
    if avg_iv_ce > avg_iv_pe + 5:
        polarity -= 0.3  # Higher call IV = bearish
    elif avg_iv_pe > avg_iv_ce + 5:
        polarity += 0.3  # Higher put IV = bullish
    
    # Determine bias
    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively writing PUTS (bullish conviction)",
            "action": "Expect upside movement, support likely to hold"
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing",
            "action": "Moderate bullish bias, watch support levels"
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively writing CALLS (bearish conviction)",
            "action": "Expect downside movement, resistance likely to hold"
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing",
            "action": "Moderate bearish bias, watch resistance levels"
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity",
            "action": "Range-bound expected, wait for clearer signals"
        }

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
    
    # PCR Interpretation
    pcr_interpretation = ""
    pcr_sentiment = ""
    pcr_color = "#66b3ff"
    
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
    
    # Max OI Strikes
    max_ce_oi_row = merged_df.loc[merged_df["OI_CE"].idxmax()] if not merged_df.empty else None
    max_pe_oi_row = merged_df.loc[merged_df["OI_PE"].idxmax()] if not merged_df.empty else None
    
    max_ce_strike = int(max_ce_oi_row["strikePrice"]) if max_ce_oi_row is not None else 0
    max_ce_oi_val = int(max_ce_oi_row["OI_CE"]) if max_ce_oi_row is not None else 0
    max_pe_strike = int(max_pe_oi_row["strikePrice"]) if max_pe_oi_row is not None else 0
    max_pe_oi_val = int(max_pe_oi_row["OI_PE"]) if max_pe_oi_row is not None else 0
    
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
        
        # Max OI
        "max_ce_strike": max_ce_strike,
        "max_ce_oi": max_ce_oi_val,
        "max_pe_strike": max_pe_strike,
        "max_pe_oi": max_pe_oi_val,
        
        # Interpretation
        "oi_change_interpretation": oi_change_interpretation,
        
        # Derived metrics
        "ce_pe_ratio": total_ce_oi / total_pe_oi if total_pe_oi > 0 else 0,
        "oi_momentum": total_chg_oi / total_oi * 100 if total_oi > 0 else 0
    }

def detect_expiry_spikes(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str):
    """
    Detect potential expiry day spikes
    Returns spike probability and key levels
    """
    if days_to_expiry > 5:
        return {
            "active": False,
            "probability": 0,
            "message": "Expiry >5 days away",
            "type": None
        }
    
    spike_score = 0
    spike_factors = []
    spike_type = None
    
    # Factor 1: Max Pain vs Spot Distance
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
    
    # Factor 2: PCR Extremes
    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 1.8:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f})")
            spike_type = "UPWARD SPIKE"
        elif pcr < 0.5:
            spike_score += 15
            spike_factors.append(f"Extreme PCR ({pcr:.2f})")
            spike_type = "DOWNWARD SPIKE"
    
    # Factor 3: Large OI Build-up
    max_ce_oi = merged_df["OI_CE"].max() if not merged_df.empty else 0
    max_pe_oi = merged_df["OI_PE"].max() if not merged_df.empty else 0
    
    if max_ce_oi > 2000000:
        spike_score += 20
        spike_factors.append(f"Massive CALL OI ({max_ce_oi:,})")
    
    if max_pe_oi > 2000000:
        spike_score += 20
        spike_factors.append(f"Massive PUT OI ({max_pe_oi:,})")
    
    # Determine spike probability
    probability = min(100, int(spike_score * 1.5))
    
    if probability >= 70:
        intensity = "HIGH PROBABILITY SPIKE"
        color = "#ff0000"
    elif probability >= 50:
        intensity = "MODERATE SPIKE RISK"
        color = "#ff9900"
    else:
        intensity = "LOW SPIKE RISK"
        color = "#00ff00"
    
    return {
        "active": days_to_expiry <= 5,
        "probability": probability,
        "score": spike_score,
        "intensity": intensity,
        "type": spike_type,
        "color": color,
        "factors": spike_factors,
        "days_to_expiry": days_to_expiry
    }

def calculate_entry_signal(spot, merged_df, atm_strike, seller_bias, seller_max_pain, 
                          support_levels, resistance_levels, depth_analysis=None):
    """
    Calculate optimal entry signal
    """
    signal_score = 0
    signal_reasons = []
    position_type = "NEUTRAL"
    
    # 1. Seller Bias Analysis
    if "BULLISH" in seller_bias["bias"]:
        signal_score += 40
        position_type = "LONG"
        signal_reasons.append(f"Seller bias: {seller_bias['bias']}")
    elif "BEARISH" in seller_bias["bias"]:
        signal_score += 40
        position_type = "SHORT"
        signal_reasons.append(f"Seller bias: {seller_bias['bias']}")
    
    # 2. PCR Analysis
    oi_metrics = analyze_oi_pcr_metrics(merged_df, spot, atm_strike)
    if position_type == "LONG" and oi_metrics["pcr_total"] > 1.2:
        signal_score += 20
        signal_reasons.append(f"PCR supports LONG ({oi_metrics['pcr_total']:.2f})")
    elif position_type == "SHORT" and oi_metrics["pcr_total"] < 0.8:
        signal_score += 20
        signal_reasons.append(f"PCR supports SHORT ({oi_metrics['pcr_total']:.2f})")
    
    # 3. Market Depth Analysis
    if depth_analysis:
        pressure = depth_analysis["summary"]["depth_pressure"]
        if position_type == "LONG" and pressure > 0.1:
            signal_score += 15
            signal_reasons.append(f"Depth pressure bullish ({pressure:+.2f})")
        elif position_type == "SHORT" and pressure < -0.1:
            signal_score += 15
            signal_reasons.append(f"Depth pressure bearish ({pressure:+.2f})")
    
    # 4. Max Pain Alignment
    if seller_max_pain:
        distance = abs(spot - seller_max_pain) / spot * 100
        if distance < 1.0:
            signal_score += 15
            signal_reasons.append(f"Near Max Pain (₹{seller_max_pain:,})")
    
    # 5. Support/Resistance Alignment
    if support_levels and resistance_levels:
        if position_type == "LONG" and spot < resistance_levels[0]["price"]:
            signal_score += 10
            signal_reasons.append(f"Below resistance (₹{resistance_levels[0]['price']:.2f})")
        elif position_type == "SHORT" and spot > support_levels[0]["price"]:
            signal_score += 10
            signal_reasons.append(f"Above support (₹{support_levels[0]['price']:.2f})")
    
    # Calculate confidence
    confidence = min(100, signal_score)
    
    # Determine signal strength
    if confidence >= 70:
        signal_strength = "STRONG"
        signal_color = "#00ff88" if position_type == "LONG" else "#ff4444"
    elif confidence >= 50:
        signal_strength = "MODERATE"
        signal_color = "#00cc66" if position_type == "LONG" else "#ff6666"
    elif confidence >= 30:
        signal_strength = "WEAK"
        signal_color = "#66b3ff"
    else:
        signal_strength = "NO SIGNAL"
        signal_color = "#cccccc"
        position_type = "NEUTRAL"
    
    # Calculate optimal entry price
    if position_type == "LONG" and support_levels:
        optimal_entry = support_levels[0]["price"]
    elif position_type == "SHORT" and resistance_levels:
        optimal_entry = resistance_levels[0]["price"]
    else:
        optimal_entry = spot
    
    # Calculate stop loss and target
    stop_loss = None
    target = None
    
    if position_type == "LONG" and support_levels and resistance_levels:
        stop_loss = support_levels[0]["price"] * 0.995  # 0.5% below support
        target = resistance_levels[0]["price"] * 1.01   # 1% above resistance
    elif position_type == "SHORT" and support_levels and resistance_levels:
        stop_loss = resistance_levels[0]["price"] * 1.005  # 0.5% above resistance
        target = support_levels[0]["price"] * 0.99        # 1% below support
    
    return {
        "position_type": position_type,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "optimal_entry_price": optimal_entry,
        "signal_color": signal_color,
        "reasons": signal_reasons,
        "stop_loss": stop_loss,
        "target": target
    }

# -----------------------
#  DATA FETCHING FUNCTIONS
# -----------------------
@st.cache_data(ttl=5)
def get_nifty_spot_price():
    """Fetch NIFTY spot price from Dhan API"""
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
            return float(nifty_data.get("last_price", 0.0))
    except Exception as e:
        st.warning(f"Dhan LTP failed: {e}")
    return 0.0

@st.cache_data(ttl=300)
def get_expiry_list():
    """Fetch option expiry list from Dhan API"""
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
    except Exception as e:
        st.warning(f"Expiry list failed: {e}")
    return []

@st.cache_data(ttl=10)
def fetch_dhan_option_chain(expiry_date):
    """Fetch option chain data for specific expiry"""
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
    except Exception as e:
        st.warning(f"Option chain failed: {e}")
    return None

def parse_dhan_option_chain(chain_data):
    """Parse Dhan option chain data into DataFrames"""
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
            ce_rows.append({
                "strikePrice": strike,
                "OI_CE": safe_int(ce.get("oi",0)),
                "Chg_OI_CE": safe_int(ce.get("oi",0)) - safe_int(ce.get("previous_oi",0)),
                "Vol_CE": safe_int(ce.get("volume",0)),
                "LTP_CE": safe_float(ce.get("last_price",0.0)),
                "IV_CE": safe_float(ce.get("implied_volatility", np.nan))
            })
        
        if pe:
            pe_rows.append({
                "strikePrice": strike,
                "OI_PE": safe_int(pe.get("oi",0)),
                "Chg_OI_PE": safe_int(pe.get("oi",0)) - safe_int(pe.get("previous_oi",0)),
                "Vol_PE": safe_int(pe.get("volume",0)),
                "LTP_PE": safe_float(pe.get("last_price",0.0)),
                "IV_PE": safe_float(pe.get("implied_volatility", np.nan))
            })
    
    return pd.DataFrame(ce_rows), pd.DataFrame(pe_rows)

# -----------------------
#  MAIN APPLICATION
# -----------------------
st.set_page_config(
    page_title="Nifty Screener v7 - Complete",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 NIFTY Option Screener v7.0")
st.markdown(f"**🕐 Current IST:** {get_ist_datetime_str()}")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("#### 🎯 Seller Logic")
    st.markdown("""
    - **CALL Writing** = BEARISH (expect price below)
    - **PUT Writing** = BULLISH (expect price above)
    - **Unwinding** = Reversal signal
    """)
    
    st.markdown("#### 📊 Market Depth")
    st.markdown("""
    **Iceberg Detection:**
    - >50k shares = Large hidden order
    - Indicates institutional activity
    """)
    
    st.markdown("#### 📈 Key Metrics")
    st.markdown("""
    - **PCR:** Put-Call Ratio sentiment
    - **Max Pain:** Where sellers minimize loss
    - **GEX:** Gamma Exposure effect
    - **Depth Pressure:** Buy/Sell imbalance
    """)
    
    st.markdown("---")
    
    # Refresh controls
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=True)
    if st.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown(f"**Last update:** {get_ist_time_str()}")
    
    # Info
    st.markdown("---")
    st.caption("**Data Source:** Dhan API")
    st.caption("**Version:** 7.0 Complete")

# Fetch spot price
col1, col2 = st.columns([1, 2])
with col1:
    with st.spinner("Fetching NIFTY spot..."):
        spot = get_nifty_spot_price()
    
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot price")
        st.stop()
    
    expiries = get_expiry_list()
    if not expiries:
        st.error("Unable to fetch expiry list")
        st.stop()
    
    expiry = st.selectbox("Select expiry", expiries, index=0)

with col2:
    st.markdown(f"""
    <div class="signal-card">
        <h3>📊 NIFTY CURRENT</h3>
        <div style="font-size: 2.5rem; color: #ffcc00; font-weight: 700;">
            ₹{spot:,.2f}
        </div>
        <div style="font-size: 1.1rem; color: #cccccc;">
            Expiry: {expiry}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Fetch option chain
with st.spinner("Fetching option chain..."):
    chain = fetch_dhan_option_chain(expiry)
if chain is None:
    st.error("Failed to fetch option chain")
    st.stop()

df_ce, df_pe = parse_dhan_option_chain(chain)
if df_ce.empty or df_pe.empty:
    st.error("Insufficient option chain data")
    st.stop()

# Merge and filter data
merged = pd.merge(df_ce, df_pe, on="strikePrice", how="outer")\
    .sort_values("strikePrice")\
    .reset_index(drop=True)
merged["strikePrice"] = merged["strikePrice"].astype(int)

# Filter ATM window
strike_gap = strike_gap_from_series(merged["strikePrice"])
atm_strike = min(merged["strikePrice"].tolist(), key=lambda x: abs(x - spot))
lower = atm_strike - (ATM_STRIKE_WINDOW * strike_gap)
upper = atm_strike + (ATM_STRIKE_WINDOW * strike_gap)
merged = merged[(merged["strikePrice"]>=lower) & (merged["strikePrice"]<=upper)]

# Calculate time to expiry
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    now = datetime.now()
    tau = max((expiry_dt - now).total_seconds() / (365.25*24*3600), 1/365.25)
    days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
except:
    tau = 7.0/365.0
    days_to_expiry = 7.0

# Calculate Greeks and GEX
for i, row in merged.iterrows():
    strike = row["strikePrice"]
    iv_ce = safe_float(row.get("IV_CE", np.nan)) / 100.0 if not np.isnan(row.get("IV_CE")) else 0.25
    iv_pe = safe_float(row.get("IV_PE", np.nan)) / 100.0 if not np.isnan(row.get("IV_PE")) else 0.25
    
    # Calculate gamma
    try:
        gamma_ce = bs_gamma(spot, strike, RISK_FREE_RATE, iv_ce, tau)
        gamma_pe = bs_gamma(spot, strike, RISK_FREE_RATE, iv_pe, tau)
    except:
        gamma_ce = gamma_pe = 0.0
    
    # Calculate GEX
    oi_ce = safe_int(row.get("OI_CE",0))
    oi_pe = safe_int(row.get("OI_PE",0))
    notional = LOT_SIZE * spot
    
    merged.at[i,"Gamma_CE"] = gamma_ce
    merged.at[i,"Gamma_PE"] = gamma_pe
    merged.at[i,"GEX_CE"] = gamma_ce * notional * oi_ce
    merged.at[i,"GEX_PE"] = gamma_pe * notional * oi_pe
    merged.at[i,"GEX_Net"] = merged.at[i,"GEX_CE"] + merged.at[i,"GEX_PE"]

# ====================
# MARKET DEPTH ANALYSIS
# ====================
st.markdown("---")
st.markdown("## 📊 MARKET DEPTH ANALYSIS (Iceberg Detection)")

# Fetch market depth
depth_data = get_market_depth()
depth_analysis = analyze_market_depth(depth_data, spot) if depth_data else None

if depth_analysis:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pressure = depth_analysis["summary"]["depth_pressure"]
        pressure_color = "#00ff88" if pressure > 0.1 else "#ff4444" if pressure < -0.1 else "#66b3ff"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color:#cccccc;">Depth Pressure</div>
            <div style="font-size: 1.5rem; color:{pressure_color}; font-weight:700;">
                {pressure:+.2f}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                {'Buying' if pressure > 0.1 else 'Selling' if pressure < -0.1 else 'Balanced'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        iceberg_count = depth_analysis["summary"]["total_iceberg_orders"]
        iceberg_color = "#ff00ff" if iceberg_count > 0 else "#cccccc"
        st.markdown(f"""
        <div class="metric-card {'iceberg-alert' if iceberg_count > 0 else ''}">
            <div style="font-size: 0.9rem; color:#cccccc;">Iceberg Orders</div>
            <div style="font-size: 1.5rem; color:{iceberg_color}; font-weight:700;">
                {iceberg_count}
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                Hidden large orders
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Total Bid Volume", f"{depth_analysis['summary']['total_bid_volume']:,}")
    
    with col4:
        st.metric("Total Ask Volume", f"{depth_analysis['summary']['total_ask_volume']:,}")
    
    # Depth Levels Display
    st.markdown("### 📈 Top 5 Depth Levels")
    
    depth_col1, depth_col2 = st.columns(2)
    
    with depth_col1:
        st.markdown("##### 🟢 Buy Side (Bids)")
        if depth_analysis['buy_side']['levels']:
            for i, bid in enumerate(depth_analysis['buy_side']['levels'][:5], 1):
                iceberg = "🧊 " if bid['is_iceberg'] else ""
                distance_color = "#00ff88" if bid['distance_from_spot'] < 20 else "#ff9900" if bid['distance_from_spot'] < 50 else "#cccccc"
                st.markdown(f"""
                {i}. **₹{bid['price']:.2f}** | Qty: {bid['quantity']:,} {iceberg}
                &nbsp;&nbsp;&nbsp;&nbsp;*Distance: <span style='color:{distance_color}'>{bid['distance_from_spot']:.1f} points</span>*
                """, unsafe_allow_html=True)
        else:
            st.info("No bid data available")
    
    with depth_col2:
        st.markdown("##### 🔴 Sell Side (Asks)")
        if depth_analysis['sell_side']['levels']:
            for i, ask in enumerate(depth_analysis['sell_side']['levels'][:5], 1):
                iceberg = "🧊 " if ask['is_iceberg'] else ""
                distance_color = "#ff4444" if ask['distance_from_spot'] < 20 else "#ff9900" if ask['distance_from_spot'] < 50 else "#cccccc"
                st.markdown(f"""
                {i}. **₹{ask['price']:.2f}** | Qty: {ask['quantity']:,} {iceberg}
                &nbsp;&nbsp;&nbsp;&nbsp;*Distance: <span style='color:{distance_color}'>{ask['distance_from_spot']:.1f} points</span>*
                """, unsafe_allow_html=True)
        else:
            st.info("No ask data available")
    
    # Support/Resistance from Depth
    st.markdown("### 🎯 Depth-Based Support & Resistance")
    
    if depth_analysis['support_levels'] or depth_analysis['resistance_levels']:
        support_col, resistance_col = st.columns(2)
        
        with support_col:
            st.markdown("##### 🛡️ Support Levels (Strong Bids)")
            if depth_analysis['support_levels']:
                for support in depth_analysis['support_levels'][:3]:
                    iceberg = "🚨 ICEBERG " if support['iceberg'] else ""
                    st.markdown(f"""
                    - **₹{support['price']:.2f}**
                      Strength: {support['strength']:.1f}k
                      Qty: {support['quantity']:,}
                      {iceberg}
                    """)
            else:
                st.info("No strong support levels detected")
        
        with resistance_col:
            st.markdown("##### ⚡ Resistance Levels (Strong Asks)")
            if depth_analysis['resistance_levels']:
                for resistance in depth_analysis['resistance_levels'][:3]:
                    iceberg = "🚨 ICEBERG " if resistance['iceberg'] else ""
                    st.markdown(f"""
                    - **₹{resistance['price']:.2f}**
                      Strength: {resistance['strength']:.1f}k
                      Qty: {resistance['quantity']:,}
                      {iceberg}
                    """)
            else:
                st.info("No strong resistance levels detected")
    
    # Iceberg Orders Detail
    if depth_analysis['iceberg_orders']:
        st.markdown("### 🚨 Iceberg Orders Detected")
        iceberg_df = pd.DataFrame(depth_analysis['iceberg_orders'])
        st.dataframe(iceberg_df, use_container_width=True)
else:
    st.warning("⚠️ Market depth data unavailable")

# ====================
# SELLER'S ANALYSIS
# ====================
st.markdown("---")
st.markdown("## 🎯 SELLER'S ANALYSIS & OPTION DATA")

# Calculate seller metrics
seller_bias_result = calculate_seller_market_bias(merged, spot, atm_strike)
seller_max_pain = calculate_seller_max_pain(merged)
oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)
expiry_spike_data = detect_expiry_spikes(merged, spot, atm_strike, days_to_expiry, expiry)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Spot Price", f"₹{spot:.2f}")
    st.metric("ATM Strike", f"₹{atm_strike}")

with col2:
    bias_color = seller_bias_result['color']
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color:#cccccc;">Seller Bias</div>
        <div style="font-size: 1.3rem; color:{bias_color}; font-weight:700;">
            {seller_bias_result['bias']}
        </div>
        <div style="font-size: 0.8rem; color:#aaaaaa;">
            Polarity: {seller_bias_result['polarity']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pcr_color = oi_pcr_metrics['pcr_color']
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color:#cccccc;">PCR Analysis</div>
        <div style="font-size: 1.8rem; color:{pcr_color}; font-weight:700;">
            {oi_pcr_metrics['pcr_total']:.2f}
        </div>
        <div style="font-size: 0.8rem; color:{pcr_color};">
            {oi_pcr_metrics['pcr_sentiment']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_gex = merged["GEX_Net"].sum()
    gex_color = "#00ff88" if total_gex > 0 else "#ff4444" if total_gex < 0 else "#66b3ff"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color:#cccccc;">Gamma Exposure</div>
        <div style="font-size: 1.5rem; color:{gex_color}; font-weight:700;">
            ₹{int(total_gex):,}
        </div>
        <div style="font-size: 0.8rem; color:#aaaaaa;">
            {'Positive' if total_gex > 0 else 'Negative' if total_gex < 0 else 'Neutral'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# OI Analysis Expanded
st.markdown("### 📊 OPEN INTEREST ANALYSIS")

oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)

with oi_col1:
    st.metric("Total CALL OI", f"{oi_pcr_metrics['total_ce_oi']:,}")
    st.metric("Δ CALL OI", f"{oi_pcr_metrics['total_ce_chg']:+,}")

with oi_col2:
    st.metric("Total PUT OI", f"{oi_pcr_metrics['total_pe_oi']:,}")
    st.metric("Δ PUT OI", f"{oi_pcr_metrics['total_pe_chg']:+,}")

with oi_col3:
    call_sellers = (merged["Chg_OI_CE"] > 0).sum()
    put_sellers = (merged["Chg_OI_PE"] > 0).sum()
    st.metric("CALL Writing Strikes", call_sellers)
    st.metric("PUT Writing Strikes", put_sellers)

with oi_col4:
    if seller_max_pain:
        distance = abs(spot - seller_max_pain)
        st.metric("Max Pain", f"₹{seller_max_pain:,}")
        st.metric("Distance from Spot", f"₹{distance:.2f}")

# Expiry Spike Analysis
if expiry_spike_data["active"]:
    st.markdown("### 📅 EXPIRY SPIKE ANALYSIS")
    
    spike_col1, spike_col2, spike_col3 = st.columns(3)
    
    with spike_col1:
        spike_color = expiry_spike_data["color"]
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color:#cccccc;">Spike Probability</div>
            <div style="font-size: 1.8rem; color:{spike_color}; font-weight:700;">
                {expiry_spike_data['probability']}%
            </div>
            <div style="font-size: 0.8rem; color:#aaaaaa;">
                {expiry_spike_data['intensity']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with spike_col2:
        st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
        if expiry_spike_data["type"]:
            st.metric("Spike Type", expiry_spike_data["type"])
    
    with spike_col3:
        if expiry_spike_data["factors"]:
            with st.expander("View Spike Factors"):
                for factor in expiry_spike_data["factors"]:
                    st.markdown(f"- {factor}")

# ====================
# TRADE SIGNAL GENERATION
# ====================
st.markdown("---")
st.markdown("## 🎯 TRADE SIGNAL GENERATION")

# Combine depth and option-based support/resistance
option_supports = []
option_resistances = []

# Find option-based support (high PUT OI below spot)
for _, row in merged.iterrows():
    strike = row["strikePrice"]
    put_oi = safe_int(row.get("OI_PE", 0))
    
    if strike < spot and put_oi > 50000:  # Significant PUT OI
        option_supports.append({
            'price': strike,
            'strength': put_oi / 1000,
            'type': 'option_put_oi',
            'quantity': put_oi
        })

# Find option-based resistance (high CALL OI above spot)
for _, row in merged.iterrows():
    strike = row["strikePrice"]
    call_oi = safe_int(row.get("OI_CE", 0))
    
    if strike > spot and call_oi > 50000:  # Significant CALL OI
        option_resistances.append({
            'price': strike,
            'strength': call_oi / 1000,
            'type': 'option_call_oi',
            'quantity': call_oi
        })

# Sort by strength
option_supports.sort(key=lambda x: x['strength'], reverse=True)
option_resistances.sort(key=lambda x: x['strength'], reverse=True)

# Combine with depth levels
all_supports = option_supports.copy()
all_resistances = option_resistances.copy()

if depth_analysis:
    all_supports.extend(depth_analysis['support_levels'])
    all_resistances.extend(depth_analysis['resistance_levels'])

# Sort combined levels
all_supports.sort(key=lambda x: x['strength'], reverse=True)
all_resistances.sort(key=lambda x: x['strength'], reverse=True)

# Generate trade signal
entry_signal = calculate_entry_signal(
    spot=spot,
    merged_df=merged,
    atm_strike=atm_strike,
    seller_bias=seller_bias_result,
    seller_max_pain=seller_max_pain,
    support_levels=all_supports,
    resistance_levels=all_resistances,
    depth_analysis=depth_analysis
)

# Display Signal
st.markdown("### 📈 TRADE SIGNAL")

if entry_signal["position_type"] != "NEUTRAL":
    signal_color = entry_signal["signal_color"]
    signal_emoji = "🚀" if entry_signal["position_type"] == "LONG" else "🐻"
    
    st.markdown(f"""
    <div class="signal-card" style="border-color: {signal_color};">
        <h3>{signal_emoji} {entry_signal['signal_strength']} {entry_signal['position_type']} SIGNAL</h3>
        <div style="font-size: 2.2rem; color: {signal_color}; font-weight: 900;">
            Confidence: {entry_signal['confidence']:.0f}%
        </div>
        <div style="font-size: 1.5rem; color: #ffcc00; font-weight: 700; margin: 10px 0;">
            Entry: ₹{entry_signal['optimal_entry_price']:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Signal Details
    col_sig1, col_sig2 = st.columns(2)
    
    with col_sig1:
        st.markdown("**Signal Reasons:**")
        for reason in entry_signal["reasons"]:
            st.markdown(f"- {reason}")
    
    with col_sig2:
        st.markdown("**Risk Management:**")
        if entry_signal["stop_loss"]:
            st.metric("Stop Loss", f"₹{entry_signal['stop_loss']:,.2f}")
        if entry_signal["target"]:
            st.metric("Target", f"₹{entry_signal['target']:,.2f}")
        
        # Risk:Reward
        if entry_signal["stop_loss"] and entry_signal["target"]:
            if entry_signal["position_type"] == "LONG":
                risk = abs(entry_signal["optimal_entry_price"] - entry_signal["stop_loss"])
                reward = abs(entry_signal["target"] - entry_signal["optimal_entry_price"])
            else:
                risk = abs(entry_signal["stop_loss"] - entry_signal["optimal_entry_price"])
                reward = abs(entry_signal["optimal_entry_price"] - entry_signal["target"])
            
            if risk > 0:
                rr_ratio = reward / risk
                st.metric("Risk:Reward", f"1:{rr_ratio:.2f}")
else:
    st.markdown("""
    <div class="signal-card">
        <h3>⚖️ NO CLEAR SIGNAL</h3>
        <div style="font-size: 1.8rem; color: #cccccc; font-weight: 700;">
            Wait for better setup
        </div>
        <div style="font-size: 1.1rem; color: #aaaaaa; margin-top: 10px;">
            Confidence: {:.0f}% | Market conditions neutral
        </div>
    </div>
    """.format(entry_signal['confidence']), unsafe_allow_html=True)

# ====================
# COMBINED SUPPORT/RESISTANCE
# ====================
st.markdown("---")
st.markdown("## 🎯 COMBINED SUPPORT & RESISTANCE LEVELS")

support_col, resistance_col = st.columns(2)

with support_col:
    st.markdown("### 🛡️ SUPPORT LEVELS")
    
    if all_supports:
        for i, support in enumerate(all_supports[:5], 1):
            source = "🧊 Depth" if support.get('iceberg') else "📊 Options" if support['type'].startswith('option') else "📈 Depth"
            st.markdown(f"""
            {i}. **₹{support['price']:.2f}**
               Strength: {support['strength']:.1f}k
               Source: {source}
            """)
    else:
        st.info("No support levels identified")

with resistance_col:
    st.markdown("### ⚡ RESISTANCE LEVELS")
    
    if all_resistances:
        for i, resistance in enumerate(all_resistances[:5], 1):
            source = "🧊 Depth" if resistance.get('iceberg') else "📊 Options" if resistance['type'].startswith('option') else "📈 Depth"
            st.markdown(f"""
            {i}. **₹{resistance['price']:.2f}**
               Strength: {resistance['strength']:.1f}k
               Source: {source}
            """)
    else:
        st.info("No resistance levels identified")

# ====================
# DETAILED OPTION CHAIN
# ====================
st.markdown("---")
st.markdown("## 📊 DETAILED OPTION CHAIN DATA")

# Display option chain data
display_cols = [
    "strikePrice", 
    "OI_CE", "Chg_OI_CE", 
    "OI_PE", "Chg_OI_PE",
    "IV_CE", "IV_PE",
    "GEX_Net"
]

# Format the display
def color_oi_change(val):
    if val > 0:
        return 'background-color: #2e1a1a; color: #ff6666'
    elif val < 0:
        return 'background-color: #1a2e1a; color: #00ff88'
    return ''

def color_gex(val):
    if val > 0:
        return 'background-color: #1a2e1a; color: #00ff88'
    elif val < 0:
        return 'background-color: #2e1a1a; color: #ff6666'
    return ''

styled_df = merged[display_cols].style\
    .applymap(color_oi_change, subset=['Chg_OI_CE', 'Chg_OI_PE'])\
    .applymap(color_gex, subset=['GEX_Net'])\
    .format({
        'strikePrice': '{:,}',
        'OI_CE': '{:,}',
        'Chg_OI_CE': '{:+,}',
        'OI_PE': '{:,}',
        'Chg_OI_PE': '{:+,}',
        'IV_CE': '{:.1f}%',
        'IV_PE': '{:.1f}%',
        'GEX_Net': '₹{:,.0f}'
    })

st.dataframe(styled_df, use_container_width=True, height=400)

# ====================
# TRADING INSIGHTS
# ====================
st.markdown("---")
st.markdown("## 💡 TRADING INSIGHTS & ANALYSIS")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 MARKET ANALYSIS")
    
    # Seller Bias Analysis
    st.markdown(f"**Seller Activity:** {seller_bias_result['explanation']}")
    st.markdown(f"**Expected Action:** {seller_bias_result['action']}")
    
    # PCR Analysis
    st.markdown(f"**PCR Interpretation:** {oi_pcr_metrics['pcr_interpretation']}")
    st.markdown(f"**OI Change Pattern:** {oi_pcr_metrics['oi_change_interpretation']}")
    
    # GEX Analysis
    total_gex = merged["GEX_Net"].sum()
    if total_gex > 1000000:
        st.success("**Gamma Exposure:** Positive GEX (stabilizing effect). Market makers are short gamma.")
    elif total_gex < -1000000:
        st.warning("**Gamma Exposure:** Negative GEX (destabilizing effect). Market makers are long gamma.")
    
    # Iceberg Insights
    if depth_analysis and depth_analysis['summary']['total_iceberg_orders'] > 0:
        st.warning(f"**Iceberg Alert:** {depth_analysis['summary']['total_iceberg_orders']} large hidden orders detected. Institutional activity present.")
    
    # Expiry Insights
    if expiry_spike_data["active"] and expiry_spike_data["probability"] > 50:
        st.warning(f"**Expiry Risk:** High spike probability ({expiry_spike_data['probability']}%). Trade cautiously.")

with insight_col2:
    st.markdown("### 🛡️ RISK MANAGEMENT")
    
    # Range Analysis
    if all_supports and all_resistances:
        best_support = all_supports[0]['price']
        best_resistance = all_resistances[0]['price']
        range_size = best_resistance - best_support
        
        st.metric("Trading Range", f"₹{best_support:.2f} - ₹{best_resistance:.2f}")
        st.metric("Range Size", f"₹{range_size:.2f}")
        
        # Position in range
        if range_size > 0:
            position_pct = ((spot - best_support) / range_size) * 100
            position_color = "#00ff88" if position_pct < 40 else "#ff4444" if position_pct > 60 else "#ff9900"
            st.markdown(f"**Spot Position:** <span style='color:{position_color}'>{position_pct:.1f}% within range</span>", unsafe_allow_html=True)
        
        # Stop Loss Suggestions
        st.markdown("**Stop Loss Ideas:**")
        if entry_signal["position_type"] == "LONG":
            stop_levels = []
            if all_supports:
                stop_levels.append(f"Below support: ₹{best_support:.2f}")
            if depth_analysis and depth_analysis['support_levels']:
                depth_stop = depth_analysis['support_levels'][0]['price']
                stop_levels.append(f"Below depth support: ₹{depth_stop:.2f}")
            
            for stop in stop_levels[:2]:
                st.markdown(f"- {stop}")
        
        elif entry_signal["position_type"] == "SHORT":
            stop_levels = []
            if all_resistances:
                stop_levels.append(f"Above resistance: ₹{best_resistance:.2f}")
            if depth_analysis and depth_analysis['resistance_levels']:
                depth_stop = depth_analysis['resistance_levels'][0]['price']
                stop_levels.append(f"Above depth resistance: ₹{depth_stop:.2f}")
            
            for stop in stop_levels[:2]:
                st.markdown(f"- {stop}")
    
    # Trading Volume Insights
    total_volume = merged["Vol_CE"].sum() + merged["Vol_PE"].sum()
    if total_volume > 1000000:
        st.info(f"**High Trading Volume:** {total_volume:,} contracts. Active market participation.")

# ====================
# FINAL SUMMARY
# ====================
st.markdown("---")
st.markdown("## 📋 FINAL MARKET SUMMARY")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("#### 📊 Market Structure")
    st.markdown(f"- **Spot:** ₹{spot:.2f}")
    st.markdown(f"- **ATM:** ₹{atm_strike}")
    st.markdown(f"- **Bias:** {seller_bias_result['bias']}")
    st.markdown(f"- **PCR:** {oi_pcr_metrics['pcr_total']:.2f}")
    st.markdown(f"- **GEX:** ₹{int(merged['GEX_Net'].sum()):,}")

with summary_col2:
    st.markdown("#### 🎯 Key Levels")
    if all_supports:
        st.markdown(f"- **Support:** ₹{all_supports[0]['price']:.2f}")
    if all_resistances:
        st.markdown(f"- **Resistance:** ₹{all_resistances[0]['price']:.2f}")
    if seller_max_pain:
        st.markdown(f"- **Max Pain:** ₹{seller_max_pain:,}")
    
    # Depth summary
    if depth_analysis:
        st.markdown(f"- **Depth Pressure:** {depth_analysis['summary']['depth_pressure']:+.2f}")

with summary_col3:
    st.markdown("#### ⚠️ Risk Factors")
    
    risk_factors = []
    
    # Expiry risk
    if expiry_spike_data["active"] and expiry_spike_data["probability"] > 50:
        risk_factors.append(f"Expiry spike risk: {expiry_spike_data['probability']}%")
    
    # Iceberg risk
    if depth_analysis and depth_analysis['summary']['total_iceberg_orders'] > 0:
        risk_factors.append(f"Iceberg orders: {depth_analysis['summary']['total_iceberg_orders']}")
    
    # GEX risk
    total_gex = merged["GEX_Net"].sum()
    if abs(total_gex) > 2000000:
        risk_factors.append(f"Extreme GEX: ₹{int(total_gex):,}")
    
    # PCR extremes
    if oi_pcr_metrics["pcr_total"] > 2.0 or oi_pcr_metrics["pcr_total"] < 0.5:
        risk_factors.append(f"Extreme PCR: {oi_pcr_metrics['pcr_total']:.2f}")
    
    if risk_factors:
        for factor in risk_factors[:3]:
            st.markdown(f"- {factor}")
    else:
        st.markdown("- No major risk factors detected")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown(f"**🔄 Auto-refresh:** {AUTO_REFRESH_SEC}s | **⏰ Last Update:** {get_ist_datetime_str()}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(AUTO_REFRESH_SEC)
    st.rerun()

# Requirements note
st.caption("""
**Requirements:** `streamlit pandas numpy requests pytz scipy` | **Data Source:** Dhan API | 
**Version:** NIFTY Option Screener v7.0 Complete
""")
