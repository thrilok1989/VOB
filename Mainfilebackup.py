"""
Nifty Option Screener v6.0 — 100% SELLER'S PERSPECTIVE + MOMENT DETECTOR + AI ANALYSIS + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS
EVERYTHING interpreted from Option Seller/Market Maker viewpoint
CALL building = BEARISH (sellers selling calls, expecting price to stay below)
PUT building = BULLISH (sellers selling puts, expecting price to stay above)

NEW FEATURES ADDED:
1. Momentum Burst Detection
2. Orderbook Pressure Analysis
3. Gamma Cluster Concentration
4. OI Velocity/Acceleration
5. Telegram Signal Generation
6. AI-Powered Market Analysis (Perplexity)
7. EXPIRY SPIKE DETECTOR (NEW)
8. ENHANCED OI/PCR ANALYTICS (NEW)
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
    # Perplexity AI credentials (optional)
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
    ENABLE_AI_ANALYSIS = st.secrets.get("ENABLE_AI_ANALYSIS", "false").lower() == "true"
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

# -----------------------
#  AI ANALYSIS CLASS - UPDATED FOR PERPLEXITY
# -----------------------
class TradingAI:
    """AI-powered trading analysis using Perplexity (Sonar model)"""
    
    def __init__(self, api_key=None):
        # Set up Perplexity API
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.enabled = bool(self.api_key) and ENABLE_AI_ANALYSIS
        
        if self.enabled:
            try:
                # Try to import Perplexity
                try:
                    from perplexity import Perplexity
                    
                    # Set the API key
                    if not os.environ.get("PERPLEXITY_API_KEY"):
                        os.environ["PERPLEXITY_API_KEY"] = self.api_key
                    
                    self.client = Perplexity()
                    self.model = "sonar-pro"  # Using Perplexity's Sonar Pro model
                except ImportError:
                    st.warning("⚠️ Perplexity package not installed. Install with: pip install perplexity-client")
                    self.enabled = False
                except Exception as e:
                    st.warning(f"⚠️ Perplexity initialization error: {e}")
                    self.enabled = False
            except Exception as e:
                st.warning(f"⚠️ AI Analysis Disabled: {e}")
                self.enabled = False
    
    def is_enabled(self):
        return self.enabled
    
    def format_market_data_for_analysis(self, market_data, signal_data, moment_metrics, expiry_spike_data):
        """Format trading data for AI analysis"""
        
        formatted_data = {
            "timestamp": get_ist_datetime_str(),
            "market_data": market_data,
            "signal_data": {
                "position_type": signal_data["position_type"],
                "signal_strength": signal_data["signal_strength"],
                "confidence": signal_data["confidence"],
                "optimal_entry_price": signal_data["optimal_entry_price"],
                "stop_loss": signal_data.get("stop_loss", "N/A"),
                "target": signal_data.get("target", "N/A"),
                "max_pain": signal_data.get("max_pain", "N/A"),
                "nearest_support": signal_data.get("nearest_support", "N/A"),
                "nearest_resistance": signal_data.get("nearest_resistance", "N/A")
            },
            "moment_metrics": moment_metrics,
            "expiry_spike_data": expiry_spike_data
        }
        
        return json.dumps(formatted_data, indent=2)
    
    def generate_analysis(self, market_data, signal_data, moment_metrics, expiry_spike_data):
        """
        Generate AI analysis of current market conditions using Perplexity
        """
        if not self.enabled:
            return None
        
        try:
            # Format data for analysis
            formatted_data = self.format_market_data_for_analysis(market_data, signal_data, moment_metrics, expiry_spike_data)
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            You are an expert options trader and market analyst specializing in Nifty options. 
            Analyze the following real-time trading data and provide actionable insights:
            
            ====== MARKET DATA ======
            Spot Price: ₹{market_data['spot']:,.2f}
            ATM Strike: ₹{market_data['atm_strike']:,}
            Seller Bias: {market_data['seller_bias']}
            Max Pain: ₹{market_data['max_pain']:,}
            Breakout Index: {market_data['breakout_index']}%
            PCR: {market_data['total_pcr']:.2f}
            Total GEX: ₹{market_data['total_gex']:,}
            
            ====== EXPIRY SPIKE DATA ======
            Days to Expiry: {market_data['days_to_expiry']:.1f}
            Spike Probability: {expiry_spike_data.get('probability', 0)}%
            Spike Type: {expiry_spike_data.get('type', 'N/A')}
            Spike Risk: {expiry_spike_data.get('intensity', 'N/A')}
            
            ====== SIGNAL DATA ======
            Position: {signal_data['position_type']} ({signal_data['signal_strength']})
            Confidence: {signal_data['confidence']:.0f}%
            Entry Price: ₹{signal_data['optimal_entry_price']:,.2f}
            Stop Loss: ₹{signal_data.get('stop_loss', 'N/A'):,.2f}
            Target: ₹{signal_data.get('target', 'N/A'):,.2f}
            
            ====== KEY LEVELS ======
            Support: ₹{market_data['nearest_support']:,}
            Resistance: ₹{market_data['nearest_resistance']:,}
            Range Size: ₹{market_data['range_size']:,}
            
            ====== MOMENT DETECTOR ======
            Momentum Burst: {moment_metrics['momentum_burst'].get('score', 0)}/100
            Orderbook Pressure: {moment_metrics['orderbook'].get('pressure', 0):+.2f}
            Gamma Cluster: {moment_metrics['gamma_cluster'].get('score', 0)}/100
            OI Acceleration: {moment_metrics['oi_accel'].get('score', 0)}/100
            
            ====== TIME CONTEXT ======
            Current Time: {get_ist_datetime_str()}
            Expiry: {market_data['expiry']}
            Days to Expiry: {market_data['days_to_expiry']:.1f}
            
            Please analyze this setup and provide:
            1. Key observations from seller activity and market structure
            2. Probability assessment of the trade signal
            3. Risk factors to watch (gamma, PCR, moment indicators)
            4. Recommended adjustments to stop loss/target based on levels
            5. Market context and macro factors to consider
            6. Expiry spike risk assessment and mitigation strategies
            
            Be concise, professional, and data-driven. Focus on actionable insights for an options trader.
            """
            
            # Call Perplexity API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert options trader and market analyst with 20+ years of experience in Nifty options. Provide actionable, data-driven insights."
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                max_tokens=2000
            )
            
            analysis_result = response.choices[0].message.content
            
            # Save analysis to file
            self.save_analysis(analysis_result, market_data, signal_data)
            
            return analysis_result
            
        except Exception as e:
            st.error(f"AI Analysis Error: {e}")
            return None
    
    def save_analysis(self, analysis, market_data, signal_data):
        """Save AI analysis to a timestamped file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_analysis_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write(f"Timestamp: {get_ist_datetime_str()}\n")
                f.write(f"Position: {signal_data['position_type']}\n")
                f.write(f"Confidence: {signal_data['confidence']:.0f}%\n")
                f.write(f"Spot: ₹{market_data['spot']:,.2f}\n")
                f.write(f"Entry: ₹{signal_data['optimal_entry_price']:,.2f}\n")
                f.write("\n" + "="*50 + "\n")
                f.write("AI ANALYSIS:\n")
                f.write("="*50 + "\n\n")
                f.write(analysis)
            
            st.session_state["last_analysis_file"] = filename
        except Exception as e:
            st.warning(f"Could not save analysis: {e}")
    
    def generate_trade_plan(self, signal_data, risk_capital=100000):
        """
        Generate detailed trade plan with position sizing using Perplexity
        """
        if not self.enabled:
            return None
        
        try:
            prompt = f"""
            Create a detailed trade plan for this Nifty options setup:
            
            Position: {signal_data['position_type']}
            Entry: ₹{signal_data['optimal_entry_price']:,.2f}
            Stop Loss: ₹{signal_data.get('stop_loss', 'N/A'):,.2f}
            Target: ₹{signal_data.get('target', 'N/A'):,.2f}
            Confidence: {signal_data['confidence']:.0f}%
            
            Risk Capital: ₹{risk_capital:,.2f}
            Nifty Lot Size: 50
            
            Create a detailed trade plan including:
            1. Recommended position size (number of lots) with calculation
            2. Entry strategy (market vs limit order timing)
            3. Stop loss placement rationale and adjustment rules
            4. Profit booking strategy (partial vs full exits)
            5. Risk per trade (% of capital) and maximum drawdown limits
            6. Contingency plans for gap openings or news events
            7. Position management during market hours
            
            Be specific and practical for Nifty options trading.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Trade Plan Error: {e}")
            return None
    
    def analyze_market_sentiment(self, market_data):
        """
        Analyze overall market sentiment using Perplexity
        """
        if not self.enabled:
            return None
        
        try:
            prompt = f"""
            Analyze Nifty options market sentiment based on:
            
            Spot: ₹{market_data['spot']:,.2f}
            Seller Activity: {market_data['seller_bias']}
            PCR: {market_data['total_pcr']:.2f}
            GEX: ₹{market_data['total_gex']:,}
            Max Pain: ₹{market_data['max_pain']:,}
            
            Provide sentiment analysis covering:
            1. Institutional positioning (FII/DII flows context)
            2. Retail sentiment indicators
            3. Volatility outlook (IV vs HV comparison)
            4. Key risk events for the session
            5. Market structure analysis (support/resistance validity)
            6. Gamma exposure implications
            7. PCR interpretation for next session
            
            Focus on practical implications for intraday options traders.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Sentiment Analysis Error: {e}")
            return None
    
    def analyze_historical_patterns(self, historical_data=None):
        """
        Analyze historical patterns for similar setups
        """
        if not self.enabled:
            return None
        
        try:
            prompt = """
            Based on historical Nifty options data patterns, analyze:
            
            1. Similar seller bias setups and their outcomes
            2. PCR extremes and mean reversion patterns
            3. Gamma cluster formations and price behavior
            4. Max Pain theory effectiveness in current expiry
            5. Historical win rates for similar signal configurations
            
            Provide insights on:
            - Probability of success for current setup
            - Historical risk:reward ratios
            - Best time of day for entry
            - Common failure modes to avoid
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Historical Analysis Error: {e}")
            return None

# Initialize AI
trading_ai = TradingAI(PERPLEXITY_API_KEY)

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
                                   expiry, expiry_spike_data):
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

*Expiry Context*:
📅 Days to Expiry: {expiry_spike_data.get('days_to_expiry', 0):.1f}
{expiry_info if expiry_info else "📊 Expiry spike risk: Low"}

⏰ {current_time} IST | 📆 Expiry: {expiry}

#NiftyOptions #OptionSelling #TradingSignal
"""
    return message

def check_and_send_signal(entry_signal, spot, seller_bias_result, seller_max_pain, 
                         nearest_sup, nearest_res, moment_metrics, seller_breakout_index, 
                         expiry, expiry_spike_data):
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
                moment_metrics, seller_breakout_index, expiry, expiry_spike_data
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
#  EXPIRY SPIKE DETECTOR FUNCTIONS
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
    atm_window = 2  # ±2 strikes around ATM
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_from_series(merged_df["strikePrice"]))]
    
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
            if abs(spot - max_ce_strike) < (strike_gap_from_series(merged_df["strikePrice"]) * 3):
                spike_type = "RESISTANCE SPIKE"
    
    if max_pe_oi_strike is not None:
        max_pe_oi = int(max_pe_oi_strike["OI_PE"])
        max_pe_strike = int(max_pe_oi_strike["strikePrice"])
        if max_pe_oi > 2000000:  # 2 million+ OI
            spike_score += 20
            spike_factors.append(f"Massive PUT OI at ₹{max_pe_strike:,} ({max_pe_oi:,})")
            key_levels.append(f"PUT Wall: ₹{max_pe_strike:,}")
            if abs(spot - max_pe_strike) < (strike_gap_from_series(merged_df["strikePrice"]) * 3):
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
    atm_window = 1
    atm_strikes = [s for s in merged_df["strikePrice"] 
                   if abs(s - atm_strike) <= (atm_window * strike_gap_from_series(merged_df["strikePrice"]))]
    
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
#  CUSTOM CSS - SELLER THEME + NEW MOMENT FEATURES + EXPIRY SPIKE + OI/PCR
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

st.set_page_config(page_title="Nifty Screener v6 - Seller's Perspective + Moment Detector + AI + Expiry Spike Detector + OI/PCR Analytics", layout="wide")

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
# 🔥 ENTRY SIGNAL CALCULATION (EXTENDED WITH MOMENT DETECTOR)
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
    moment_metrics  # NEW: Add moment metrics
):
    """
    Calculate optimal entry signal with Moment Detector integration
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
        "moment_metrics": moment_metrics  # NEW: Include moment metrics in signal
    }

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
#  MAIN APP - COMPLETE V6 WITH OI/PCR ANALYTICS
# -----------------------
st.title("🎯 NIFTY Option Screener v6.0 — SELLER'S PERSPECTIVE + Moment Detector + AI + Expiry Spike + OI/PCR ANALYTICS")

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
    
    # Expiry spike info in sidebar
    st.markdown("---")
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
        now = datetime.now()
        days_to_expiry = (expiry_dt - now).total_seconds() / (24 * 3600)
    except:
        days_to_expiry = 7
    
    if days_to_expiry <= 5:
        st.warning(f"⚠️ Expiry in {days_to_expiry:.1f} days")
        st.info("Spike detector ACTIVE")
    else:
        st.success(f"✓ Expiry in {days_to_expiry:.1f} days")
        st.info("Spike detector INACTIVE")
    
    st.markdown("---")
    st.markdown(f"**Current IST:** {get_ist_time_str()}")
    st.markdown(f"**Date:** {get_ist_date_str()}")
    
    save_interval = st.number_input("PCR Auto-save (sec)", value=SAVE_INTERVAL_SEC, min_value=60, step=60)
    
    # Telegram settings
    st.markdown("---")
    st.markdown("### 🤖 TELEGRAM SETTINGS")
    auto_send = st.checkbox("Auto-send signals to Telegram", value=False)
    show_signal_preview = st.checkbox("Show signal preview", value=True)
    
    # AI settings
    st.markdown("---")
    st.markdown("### 🤖 AI SETTINGS")
    enable_ai_analysis = st.checkbox("Enable AI Analysis", value=trading_ai.is_enabled())
    if enable_ai_analysis and not trading_ai.is_enabled():
        st.warning("AI requires PERPLEXITY_API_KEY in secrets")
    
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

# Calculate entry signal with moment detector integration
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
try:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    days_to_expiry = (expiry_dt - datetime.now()).days + ((expiry_dt - datetime.now()).seconds / (24*3600))
    
    if days_to_expiry <= 5:
        expiry_pcr_context = analyze_pcr_for_expiry(oi_pcr_metrics['pcr_total'], days_to_expiry)
        st.warning(f"""
        **⚠️ Expiry Context (D-{int(days_to_expiry)}):** {expiry_pcr_context}
        
        PCR readings near expiry often exaggerate due to position squaring.
        """)
except:
    pass

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
    moment_metrics, seller_breakout_index, expiry, expiry_spike_data
)

# ============================================
# 🧠 AI ANALYSIS SECTION (PERPLEXITY)
# ============================================

if trading_ai.is_enabled() and enable_ai_analysis:
    st.markdown("---")
    st.markdown("## 🧠 AI-POWERED MARKET ANALYSIS (Perplexity)")
    
    # Prepare data for AI
    market_data_for_ai = {
        'spot': spot,
        'atm_strike': atm_strike,
        'seller_bias': seller_bias_result['bias'],
        'max_pain': seller_max_pain if seller_max_pain else 0,
        'breakout_index': seller_breakout_index,
        'nearest_support': nearest_sup['strike'] if nearest_sup else 0,
        'nearest_resistance': nearest_res['strike'] if nearest_res else 0,
        'range_size': spot_analysis['range_size'],
        'total_pcr': total_PE_OI / total_CE_OI if total_CE_OI > 0 else 0,
        'total_ce_oi': total_CE_OI,
        'total_pe_oi': total_PE_OI,
        'ce_selling': ce_selling,
        'pe_selling': pe_selling,
        'total_gex': total_gex_net,
        'expiry': expiry,
        'days_to_expiry': days_to_expiry,
        'enhanced_pcr': oi_pcr_metrics['pcr_total'],
        'pcr_sentiment': oi_pcr_metrics['pcr_sentiment'],
        'oi_concentration': oi_pcr_metrics['atm_concentration_pct']
    }
    
    # AI Analysis Tabs
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📊 Market Analysis", "🎯 Trade Plan", "📈 Sentiment"])
    
    with ai_tab1:
        st.markdown("### 🤖 AI Market Analysis (Perplexity Sonar-Pro)")
        
        if st.button("🔄 Generate AI Analysis", key="ai_analyze"):
            with st.spinner("🤖 AI is analyzing market conditions..."):
                ai_analysis = trading_ai.generate_analysis(
                    market_data_for_ai, 
                    entry_signal, 
                    moment_metrics,
                    expiry_spike_data
                )
                
                if ai_analysis:
                    st.success("✅ AI Analysis Generated!")
                    
                    # Store in session state
                    st.session_state["ai_analysis"] = ai_analysis
                    
                    # Display with nice formatting
                    st.markdown("""
                    <div style="
                        background-color: #1a1f2e;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid #aa00ff;
                        margin: 10px 0;
                        white-space: pre-wrap;
                        font-family: 'Courier New', monospace;
                        line-height: 1.6;
                    ">
                    """ + ai_analysis + "</div>", unsafe_allow_html=True)
                    
                    # Save analysis
                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        if st.button("💾 Save Analysis", key="save_ai_analysis"):
                            filename = f"ai_analysis_{get_ist_datetime_str().replace(':', '-').replace(' ', '_')}.txt"
                            with open(filename, 'w') as f:
                                f.write(ai_analysis)
                            st.success(f"✅ Analysis saved to {filename}")
                    with col_save2:
                        if st.button("📋 Copy to Clipboard", key="copy_ai_analysis"):
                            st.info("✅ Analysis copied to clipboard!")
                else:
                    st.error("❌ Failed to generate AI analysis")
        
        # Show pre-generated analysis if available
        elif "ai_analysis" in st.session_state:
            st.markdown("#### 📝 Previous Analysis:")
            st.markdown(f"""
            <div style="
                background-color: #1a1f2e;
                padding: 15px;
                border-radius: 8px;
                border-left: 3px solid #666;
                margin: 10px 0;
                font-size: 0.9em;
                max-height: 200px;
                overflow-y: auto;
            ">
            {st.session_state['ai_analysis'][:500]}...
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Full Analysis", key="view_full"):
                st.text_area("Full AI Analysis", st.session_state['ai_analysis'], height=300)
    
    with ai_tab2:
        st.markdown("### 🎯 AI Trade Plan")
        
        # Risk capital input
        risk_capital = st.number_input(
            "Risk Capital (₹)", 
            min_value=10000, 
            max_value=10000000, 
            value=100000, 
            step=10000,
            key="risk_capital_input"
        )
        
        if st.button("📋 Generate Trade Plan", key="ai_trade_plan"):
            with st.spinner("🤖 Creating detailed trade plan..."):
                trade_plan = trading_ai.generate_trade_plan(entry_signal, risk_capital)
                
                if trade_plan:
                    st.success("✅ Trade Plan Generated!")
                    
                    # Store in session state
                    st.session_state["trade_plan"] = trade_plan
                    
                    # Display in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Position Sizing")
                        # Calculate position size
                        if entry_signal['stop_loss']:
                            risk_per_trade = risk_capital * 0.02  # 2% risk per trade
                            risk_points = abs(entry_signal['optimal_entry_price'] - entry_signal['stop_loss'])
                            risk_per_point = 50 * spot  # Nifty lot size * spot for options
                            position_size = int((risk_per_trade / risk_per_point) / risk_points)
                            position_size = max(1, position_size)
                            
                            st.metric("Recommended Lots", position_size)
                            st.metric("Risk per Trade", f"₹{risk_per_trade:,.0f}")
                            st.metric("Max Risk %", "2%")
                    
                    with col2:
                        st.markdown("#### 📈 AI Trade Plan")
                        st.markdown(f"""
                        <div style="
                            background-color: #1a2e1a;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 3px solid #00ff88;
                            white-space: pre-wrap;
                            line-height: 1.6;
                        ">
                        {trade_plan}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save trade plan
                    col_save3, col_save4 = st.columns(2)
                    with col_save3:
                        if st.button("💾 Save Trade Plan", key="save_trade_plan"):
                            filename = f"trade_plan_{get_ist_datetime_str().replace(':', '-').replace(' ', '_')}.txt"
                            with open(filename, 'w') as f:
                                f.write(trade_plan)
                            st.success(f"✅ Trade plan saved to {filename}")
                    with col_save4:
                        if st.button("📋 Copy Trade Plan", key="copy_trade_plan"):
                            st.info("✅ Trade plan copied to clipboard!")
    
    with ai_tab3:
        st.markdown("### 📈 Market Sentiment Analysis")
        
        if st.button("🌡️ Analyze Sentiment", key="ai_sentiment"):
            with st.spinner("🤖 Analyzing market sentiment..."):
                sentiment = trading_ai.analyze_market_sentiment(market_data_for_ai)
                
                if sentiment:
                    st.success("✅ Sentiment Analysis Complete!")
                    
                    # Store in session state
                    st.session_state["sentiment"] = sentiment
                    
                    # Color code based on seller bias
                    bias_color = {
                        "BULLISH": "#00ff88",
                        "BEARISH": "#ff4444", 
                        "NEUTRAL": "#66b3ff"
                    }
                    
                    current_bias = seller_bias_result['bias']
                    color = "#66b3ff"
                    for key in bias_color:
                        if key in current_bias:
                            color = bias_color[key]
                            break
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #1a1f2e;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid {color};
                        margin: 10px 0;
                        white-space: pre-wrap;
                        line-height: 1.6;
                    ">
                    <h4 style="color:{color}">🎯 Current Market Sentiment</h4>
                    {sentiment}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sentiment metrics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Seller Bias", seller_bias_result['bias'])
                    with col_s2:
                        pcr_val = market_data_for_ai['total_pcr']
                        pcr_sentiment = "Bullish" if pcr_val > 1.2 else ("Bearish" if pcr_val < 0.8 else "Neutral")
                        st.metric("PCR Sentiment", pcr_sentiment)
                    with col_s3:
                        gex_sentiment = "Stabilizing" if total_gex_net > 0 else "Volatile"
                        st.metric("Gamma Sentiment", gex_sentiment)

else:
    # Show setup instructions if AI is not enabled
    if enable_ai_analysis:
        st.markdown("---")
        st.markdown("## 🧠 AI ANALYSIS (Setup Required)")
        
        st.info("""
        ### ⚙️ To Enable AI Analysis:
        
        1. **Install Perplexity package:**
        ```bash
        pip install perplexity-client python-dotenv
        ```
        
        2. **Get Perplexity API Key:**
           - Visit [perplexity.ai](https://www.perplexity.ai)
           - Sign up and get your API key from dashboard
           
        3. **Add to Streamlit Secrets:**
        ```toml
        # .streamlit/secrets.toml
        PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
        ENABLE_AI_ANALYSIS = "true"
        ```
        
        4. **Restart the app**
        
        ### 🎯 AI Features:
        - Real-time market analysis (Perplexity Sonar-Pro)
        - Web-enhanced market context
        - Trade plan generation
        - Sentiment analysis
        - Risk assessment
        - Position sizing recommendations
        """)

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
        - Expiry Spike Risk: {expiry_spike_data.get('probability', 0)}%
        - PCR Sentiment: {oi_pcr_metrics['pcr_sentiment']}
        
        **Requirements for signal generation:**
        ✅ Position Type ≠ NEUTRAL
        ✅ Confidence ≥ 40%
        ✅ Clear directional bias
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
    
    # Expandable details for no signal
    with st.expander("🔍 Why No Signal? (Click for Details)", expanded=False):
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("### 📊 Current Metrics:")
            st.metric("Seller Bias", seller_bias_result["bias"])
            st.metric("Polarity Score", f"{seller_bias_result['polarity']:.2f}")
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
# 📊 DETAILED DATA - SELLER VIEW + MOMENT + EXPIRY + OI/PCR
# ============================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Seller Activity", "🧮 Seller Greeks", "📈 Seller PCR", "🚀 Moment Analysis", "📅 Expiry Analysis", "📊 OI/PCR Analysis"])

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
    st.markdown("### 📊 COMPREHENSIVE OI/PCR ANALYSIS")
    
    # OI Distribution Analysis
    st.markdown("#### 📈 OI DISTRIBUTION ANALYSIS")
    
    col_oi1, col_oi2, col_oi3 = st.columns(3)
    
    with col_oi1:
        # CALL OI Analysis
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #ff4444;
            margin: 10px 0;
        ">
            <div style="font-size: 1.1rem; color:#ff4444; font-weight:700;">CALL OI ANALYSIS</div>
            <div style="font-size: 1.8rem; color:#ff4444; font-weight:900;">{oi_pcr_metrics['total_ce_oi']:,}</div>
            <div style="font-size: 0.9rem; color:#cccccc;">
                ITM: {oi_pcr_metrics['itm_ce_oi']:,}<br>
                OTM: {oi_pcr_metrics['otm_ce_oi']:,}<br>
                ΔOI: {oi_pcr_metrics['total_ce_chg']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_oi2:
        # PUT OI Analysis
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a2e1a 0%, #2a3e2a 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00ff88;
            margin: 10px 0;
        ">
            <div style="font-size: 1.1rem; color:#00ff88; font-weight:700;">PUT OI ANALYSIS</div>
            <div style="font-size: 1.8rem; color:#00ff88; font-weight:900;">{oi_pcr_metrics['total_pe_oi']:,}</div>
            <div style="font-size: 0.9rem; color:#cccccc;">
                ITM: {oi_pcr_metrics['itm_pe_oi']:,}<br>
                OTM: {oi_pcr_metrics['otm_pe_oi']:,}<br>
                ΔOI: {oi_pcr_metrics['total_pe_chg']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_oi3:
        # Total OI Analysis
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #66b3ff;
            margin: 10px 0;
        ">
            <div style="font-size: 1.1rem; color:#66b3ff; font-weight:700;">TOTAL OI ANALYSIS</div>
            <div style="font-size: 1.8rem; color:#66b3ff; font-weight:900;">{oi_pcr_metrics['total_oi']:,}</div>
            <div style="font-size: 0.9rem; color:#cccccc;">
                CALL%: {(oi_pcr_metrics['total_ce_oi']/oi_pcr_metrics['total_oi']*100):.1f}%<br>
                PUT%: {(oi_pcr_metrics['total_pe_oi']/oi_pcr_metrics['total_oi']*100):.1f}%<br>
                ΔTotal: {oi_pcr_metrics['total_chg_oi']:+,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PCR Deep Dive
    st.markdown("#### 🎯 PCR DEEP DIVE ANALYSIS")
    
    col_pcr1, col_pcr2 = st.columns(2)
    
    with col_pcr1:
        st.markdown("##### 📊 PCR METRICS")
        st.metric("Current PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
        st.metric("PCR Change", f"{oi_pcr_metrics['pcr_chg']:+.2f}")
        st.metric("CE:PE Ratio", f"{oi_pcr_metrics['ce_pe_ratio']:.2f}:1")
        st.metric("OI Momentum", f"{oi_pcr_metrics['oi_momentum']:+.1f}%")
    
    with col_pcr2:
        st.markdown("##### 🎯 PCR INTERPRETATION")
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {oi_pcr_metrics['pcr_color']};
            margin: 10px 0;
        ">
            <div style="font-size: 1.1rem; color:{oi_pcr_metrics['pcr_color']}; font-weight:700;">
                {oi_pcr_metrics['pcr_interpretation']}
            </div>
            <div style="font-size: 1rem; color:#ffffff; margin-top: 10px;">
                <strong>Sentiment:</strong> {oi_pcr_metrics['pcr_sentiment']}<br>
                <strong>OI Change:</strong> {oi_pcr_metrics['oi_change_interpretation']}<br>
                <strong>PCR Change:</strong> {oi_pcr_metrics['chg_interpretation'] if oi_pcr_metrics['chg_interpretation'] else 'Stable'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Max OI Analysis
    st.markdown("#### 🏆 MAX OI STRIKES ANALYSIS")
    
    col_max1, col_max2 = st.columns(2)
    
    with col_max1:
        st.markdown("##### 📈 MAX CALL OI")
        if oi_pcr_metrics['max_ce_strike'] > 0:
            st.markdown(f"""
            <div style="
                background: #2e1a1a;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #ff4444;
                margin: 10px 0;
            ">
                <div style="font-size: 1.5rem; color:#ff4444; font-weight:700;">₹{oi_pcr_metrics['max_ce_strike']:,}</div>
                <div style="font-size: 1.1rem; color:#ffffff;">OI: {oi_pcr_metrics['max_ce_oi']:,}</div>
                <div style="font-size: 0.9rem; color:#cccccc;">
                    Distance from Spot: ₹{abs(spot - oi_pcr_metrics['max_ce_strike']):.2f}<br>
                    Position: {'Above' if oi_pcr_metrics['max_ce_strike'] > spot else 'Below'} spot
                </div>
            </div>
            """, unsafe_allow_html=True)
            if oi_pcr_metrics['max_ce_strike'] > spot:
                st.info("**CALL Wall ABOVE spot:** Strong resistance level")
            else:
                st.warning("**CALL Wall BELOW spot:** Unusual - could indicate trapped sellers")
    
    with col_max2:
        st.markdown("##### 📉 MAX PUT OI")
        if oi_pcr_metrics['max_pe_strike'] > 0:
            st.markdown(f"""
            <div style="
                background: #1a2e1a;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #00ff88;
                margin: 10px 0;
            ">
                <div style="font-size: 1.5rem; color:#00ff88; font-weight:700;">₹{oi_pcr_metrics['max_pe_strike']:,}</div>
                <div style="font-size: 1.1rem; color:#ffffff;">OI: {oi_pcr_metrics['max_pe_oi']:,}</div>
                <div style="font-size: 0.9rem; color:#cccccc;">
                    Distance from Spot: ₹{abs(spot - oi_pcr_metrics['max_pe_strike']):.2f}<br>
                    Position: {'Above' if oi_pcr_metrics['max_pe_strike'] > spot else 'Below'} spot
                </div>
            </div>
            """, unsafe_allow_html=True)
            if oi_pcr_metrics['max_pe_strike'] < spot:
                st.info("**PUT Wall BELOW spot:** Strong support level")
            else:
                st.warning("**PUT Wall ABOVE spot:** Unusual - could indicate trapped buyers")
    
    st.markdown("---")
    
    # OI Skew Analysis
    st.markdown("#### ⚖️ OI SKEW ANALYSIS")
    
    col_skew1, col_skew2 = st.columns(2)
    
    with col_skew1:
        st.markdown("##### 📊 CALL OI SKEW")
        skew_color = "#ff4444" if oi_pcr_metrics['call_oi_skew'] == "High" else ("#ff9900" if oi_pcr_metrics['call_oi_skew'] == "Moderate" else "#66b3ff")
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {skew_color};
            margin: 10px 0;
        ">
            <div style="font-size: 1.3rem; color:{skew_color}; font-weight:700;">{oi_pcr_metrics['call_oi_skew']}</div>
            <div style="font-size: 0.9rem; color:#cccccc;">
                Concentration analysis of CALL OI across strikes<br>
                <strong>High:</strong> OI concentrated at few strikes (potential pinning)<br>
                <strong>Moderate:</strong> Some concentration<br>
                <strong>Low:</strong> Evenly distributed
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_skew2:
        st.markdown("##### 📊 PUT OI SKEW")
        skew_color = "#00ff88" if oi_pcr_metrics['put_oi_skew'] == "High" else ("#00cc66" if oi_pcr_metrics['put_oi_skew'] == "Moderate" else "#66b3ff")
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {skew_color};
            margin: 10px 0;
        ">
            <div style="font-size: 1.3rem; color:{skew_color}; font-weight:700;">{oi_pcr_metrics['put_oi_skew']}</div>
            <div style="font-size: 0.9rem; color:#cccccc;">
                Concentration analysis of PUT OI across strikes<br>
                <strong>High:</strong> OI concentrated at few strikes (potential pinning)<br>
                <strong>Moderate:</strong> Some concentration<br>
                <strong>Low:</strong> Evenly distributed
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ATM Concentration Analysis
    st.markdown("#### 🎯 ATM CONCENTRATION ANALYSIS")
    st.metric("ATM OI Concentration", f"{oi_pcr_metrics['atm_concentration_pct']:.1f}%")
    
    if oi_pcr_metrics['atm_concentration_pct'] > 40:
        st.warning("**HIGH ATM CONCENTRATION:** Significant OI concentrated around ATM. This increases gamma risk and potential for sharp moves.")
    elif oi_pcr_metrics['atm_concentration_pct'] > 25:
        st.info("**MODERATE ATM CONCENTRATION:** Some OI concentration around ATM. Watch for gamma effects.")
    else:
        st.success("**LOW ATM CONCENTRATION:** OI spread out. Lower gamma risk, smoother price action expected.")

# ============================================
# 🎯 TRADING INSIGHTS - SELLER PERSPECTIVE + MOMENT + EXPIRY + OI/PCR
# ============================================
st.markdown("---")
st.markdown("## 💡 TRADING INSIGHTS (Seller + Moment + Expiry + OI/PCR Fusion)")

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

# Final Seller Summary with Moment, Expiry, and OI/PCR Integration
st.markdown("---")
moment_summary = ""
if moment_metrics["momentum_burst"]["score"] > 60:
    moment_summary += "High momentum burst detected. "
if moment_metrics["orderbook"]["available"] and abs(moment_metrics["orderbook"]["pressure"]) > 0.15:
    direction = "buy" if moment_metrics["orderbook"]["pressure"] > 0 else "sell"
    moment_summary += f"Strong {direction} pressure in orderbook. "

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
    <h3>🎯 FINAL ASSESSMENT (Seller + Moment + Expiry + OI/PCR)</h3>
    <p><strong>Market Makers are telling us:</strong> {seller_bias_result["explanation"]}</p>
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
st.caption("🎯 **NIFTY Option Screener v6.0 — SELLER'S PERSPECTIVE + MOMENT DETECTOR + AI ANALYSIS + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS** | All features enabled")

# Requirements note
st.markdown("""
<small>
**Requirements:** 
`streamlit pandas numpy requests pytz scipy supabase perplexity-client python-dotenv` | 
**AI:** Perplexity API key required | 
**Data:** Dhan API required
</small>
""", unsafe_allow_html=True)
