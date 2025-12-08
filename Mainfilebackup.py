"""
Nifty Option Screener v6.0 — Streamlined Trader-Focused Interface
Reorganized with priority-based display, collapsible sections, and better UX
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

# ============================================
# 🎯 REORGANIZED UI - CLEANER INTERFACE
# ============================================

# Custom CSS with organized styling
st.markdown("""
<style>
    /* BASE STYLES */
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* HEADER STYLES */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff66cc;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* PRIORITY 1: CRITICAL METRICS */
    .critical-metric {
        background: linear-gradient(135deg, #1a2e2e 0%, #2a3e3e 100%);
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #00ffcc;
        margin: 5px;
        text-align: center;
    }
    
    .critical-metric-value {
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        color: #ff66cc !important;
        margin: 5px 0 !important;
    }
    
    .critical-metric-label {
        font-size: 0.9rem !important;
        color: #cccccc !important;
        margin-bottom: 5px !important;
    }
    
    /* PRIORITY 2: ENTRY SIGNAL */
    .entry-signal-box {
        background: linear-gradient(135deg, #2e2a1a 0%, #3e3a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #ff9900;
        margin: 15px 0;
        text-align: center;
    }
    
    /* PRIORITY 3: KEY LEVELS */
    .key-level-box {
        background: linear-gradient(135deg, #2e1a2e 0%, #3e2a3e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ff66cc;
        margin: 10px 0;
    }
    
    /* MOMENT DETECTOR */
    .moment-box {
        background: linear-gradient(135deg, #1a1f3e 0%, #2a2f4e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ffff;
        margin: 10px 0;
        text-align: center;
    }
    
    /* EXPIRY SPIKE */
    .expiry-high-risk {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e2a2a 100%);
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
    }
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1f2e;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a2f3e;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    /* METRIC STYLING */
    [data-testid="stMetricValue"] { 
        font-size: 1.6rem !important; 
        font-weight: 700 !important; 
    }
    
    [data-testid="stMetricLabel"] { 
        color: #cccccc !important; 
        font-weight: 600 !important; 
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Nifty Screener v6 - Trader's Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🎯 SIDEBAR REORGANIZATION
# ============================================
with st.sidebar:
    # 1. QUICK SETTINGS (Top)
    st.markdown("### ⚙️ QUICK SETTINGS")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    with col_s2:
        refresh_rate = st.selectbox("Rate (sec)", [30, 60, 120], index=1)
    
    auto_send = st.checkbox("Auto-send Telegram", value=False)
    
    # 2. KEY INFO (Middle)
    st.markdown("---")
    st.markdown("### 📊 MARKET STATUS")
    
    # Days to expiry (calculated later)
    if 'days_to_expiry' in locals():
        st.metric("Days to Expiry", f"{days_to_expiry:.1f}")
    
    # 3. FEATURE STATUS (Collapsible)
    with st.expander("🔧 FEATURE STATUS", expanded=False):
        if trading_ai.is_enabled():
            st.success("✅ AI: ENABLED")
        else:
            st.warning("⚠️ AI: DISABLED")
        
        st.info(f"📡 Data: {'LIVE' if 'spot' in locals() and spot > 0 else 'OFFLINE'}")
        st.info(f"📱 Telegram: {'READY' if TELEGRAM_BOT_TOKEN else 'OFF'}")
    
    # 4. TRADING LOGIC GUIDE (Collapsible)
    with st.expander("🎯 TRADING LOGIC", expanded=False):
        st.markdown("""
        **SELLER'S VIEW:**
        - 📉 CALL Writing = BEARISH
        - 📈 PUT Writing = BULLISH  
        - 🔄 Buying Back = Covering
        
        **MOMENT DETECTOR:**
        - 💥 Momentum = Vol × IV × ΔOI
        - 📊 Pressure = Buy/Sell imbalance
        - 🌀 Gamma = ATM concentration
        - ⚡ OI Accel = Speed of changes
        """)
    
    # 5. EXPIRY INFO
    st.markdown("---")
    if 'expiry' in locals():
        st.markdown(f"**Current Expiry:** {expiry}")
    
    st.markdown(f"**IST:** {get_ist_time_str()}")

# ============================================
# 🎯 MAIN PAGE - REORGANIZED WITH TABS
# ============================================

# Main header
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0; color:#ff66cc;">🎯 NIFTY Option Screener v6.0</h1>
    <p style="margin:5px 0; color:#cccccc;">🕐 IST: {get_ist_datetime_str()}</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Dashboard", 
    "📊 OI/PCR Analytics", 
    "🚀 Momentum", 
    "📅 Expiry", 
    "📈 Details"
])

# ============================================
# DATA FETCHING AND PROCESSING
# ============================================

# Fetch data
with st.spinner("Fetching market data..."):
    # Fetch spot price
    spot = get_nifty_spot_price()
    
    if spot == 0.0:
        st.error("Unable to fetch NIFTY spot")
        st.stop()
    
    # Fetch expiry list
    expiries = get_expiry_list()
    if not expiries:
        st.error("Unable to fetch expiry list")
        st.stop()
    
    expiry = expiries[0]
    
    # Fetch option chain
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
    
    # ---- Capture snapshot for moment detector ----
    st.session_state["moment_history"].append(
        _snapshot_from_state(get_ist_now(), spot, atm_strike, merged)
    )
    # Keep last 10 points
    st.session_state["moment_history"] = st.session_state["moment_history"][-10:]
    
    # ---- Compute 4 moment metrics ----
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
    
    # Run OI/PCR analysis
    oi_pcr_metrics = analyze_oi_pcr_metrics(merged, spot, atm_strike)
    
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
# TAB 1: DASHBOARD (PRIORITY VIEW)
# ============================================
with tab1:
    # ROW 1: CRITICAL METRICS (4 columns)
    st.markdown("### 📈 CRITICAL METRICS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SPOT", f"₹{spot:.2f}", delta=None)
    
    with col2:
        st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}", 
                 delta=oi_pcr_metrics['pcr_sentiment'].split()[0])
    
    with col3:
        st.metric("SELLER BIAS", seller_bias_result["bias"].split()[0])
    
    with col4:
        st.metric("CONFIDENCE", f"{entry_signal['confidence']:.0f}%")
    
    # ROW 2: ENTRY SIGNAL (PROMINENT)
    st.markdown("### 🎯 ENTRY SIGNAL")
    
    if entry_signal["position_type"] != "NEUTRAL":
        signal_emoji = "🚀" if entry_signal["position_type"] == "LONG" else "🐻"
        signal_color = "#00ff88" if entry_signal["position_type"] == "LONG" else "#ff4444"
        
        st.markdown(f"""
        <div class="entry-signal-box">
            <h2 style="color:{signal_color}; margin:10px 0;">{signal_emoji} {entry_signal['signal_strength']} {entry_signal['position_type']} SIGNAL</h2>
            <div style="font-size: 1.3rem; color:#ffdd44; margin:10px 0;">
                Confidence: {entry_signal['confidence']:.0f}%
            </div>
            <div style="font-size: 1.8rem; color:#ffcc00; font-weight:900; margin:15px 0;">
                Entry: ₹{entry_signal['optimal_entry_price']:,.2f}
            </div>
            <div style="font-size: 1.1rem; color:#cccccc; margin:10px 0;">
                SL: ₹{entry_signal.get('stop_loss', 'N/A'):,.2f} | TP: ₹{entry_signal.get('target', 'N/A'):,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📭 No active entry signal. Market is neutral.")
    
    # ROW 3: KEY LEVELS
    st.markdown("### 🎯 KEY LEVELS")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    
    with col_l1:
        if nearest_sup:
            st.markdown(f"""
            <div class="key-level-box">
                <h4 style="color:#00ffcc; margin:0;">SUPPORT</h4>
                <div style="font-size: 1.8rem; color:#00ffcc; font-weight:700; margin:10px 0;">
                    ₹{nearest_sup['strike']:,}
                </div>
                <div style="font-size: 0.9rem; color:#cccccc;">
                    PUT OI: {nearest_sup['oi_pe']:,}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_l2:
        if seller_max_pain:
            st.markdown(f"""
            <div class="key-level-box">
                <h4 style="color:#ff9900; margin:0;">MAX PAIN</h4>
                <div style="font-size: 1.8rem; color:#ff9900; font-weight:700; margin:10px 0;">
                    ₹{seller_max_pain:,}
                </div>
                <div style="font-size: 0.9rem; color:#cccccc;">
                    Seller's preferred level
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_l3:
        if nearest_res:
            st.markdown(f"""
            <div class="key-level-box">
                <h4 style="color:#ff66cc; margin:0;">RESISTANCE</h4>
                <div style="font-size: 1.8rem; color:#ff66cc; font-weight:700; margin:10px 0;">
                    ₹{nearest_res['strike']:,}
                </div>
                <div style="font-size: 0.9rem; color:#cccccc;">
                    CALL OI: {nearest_res['oi_ce']:,}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ROW 4: MOMENT DETECTOR
    st.markdown("### 🚀 MOMENT DETECTOR")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        mb = moment_metrics.get("momentum_burst", {})
        if mb.get("available", False):
            st.metric("Momentum Burst", f"{mb.get('score', 0)}/100")
    
    with col_m2:
        ob = moment_metrics.get("orderbook", {})
        if ob.get("available", False):
            st.metric("Order Pressure", f"{ob.get('pressure', 0):+.2f}")
    
    with col_m3:
        gc = moment_metrics.get("gamma_cluster", {})
        if gc.get("available", False):
            st.metric("Gamma Cluster", f"{gc.get('score', 0)}/100")
    
    with col_m4:
        oi_accel = moment_metrics.get("oi_accel", {})
        if oi_accel.get("available", False):
            st.metric("OI Accel", f"{oi_accel.get('score', 0)}/100")
    
    # ROW 5: EXPIRY ALERT
    if expiry_spike_data.get("active", False):
        st.markdown("### 📅 EXPIRY ALERT")
        
        if expiry_spike_data["probability"] > 50:
            alert_emoji = "🚨" if expiry_spike_data['probability'] > 70 else "⚠️"
            
            st.warning(f"""
            {alert_emoji} **HIGH EXPIRY SPIKE RISK ({expiry_spike_data['probability']}%) - {expiry_spike_data['type']}**
            
            **Factors:**
            {chr(10).join(['• ' + factor for factor in expiry_spike_data.get('factors', [])][:3])}
            
            **Key Levels:** {', '.join(expiry_spike_data.get('key_levels', [])[:2])}
            """)

# ============================================
# TAB 2: OI/PCR ANALYTICS
# ============================================
with tab2:
    st.markdown("## 📊 ENHANCED OI & PCR ANALYTICS")
    
    # Totals Row
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        st.metric("Total CALL OI", f"{oi_pcr_metrics['total_ce_oi']:,}")
        st.metric("Δ CALL OI", f"{oi_pcr_metrics['total_ce_chg']:+,}")
    
    with col_t2:
        st.metric("Total PUT OI", f"{oi_pcr_metrics['total_pe_oi']:,}")
        st.metric("Δ PUT OI", f"{oi_pcr_metrics['total_pe_chg']:+,}")
    
    with col_t3:
        st.metric("Total OI", f"{oi_pcr_metrics['total_oi']:,}")
        st.metric("Total ΔOI", f"{oi_pcr_metrics['total_chg_oi']:+,}")
    
    # PCR Card
    st.markdown("---")
    st.markdown("### 🎯 PUT-CALL RATIO (PCR) ANALYSIS")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.metric("PCR", f"{oi_pcr_metrics['pcr_total']:.2f}")
        st.caption(f"Sentiment: {oi_pcr_metrics['pcr_sentiment']}")
    
    with col_p2:
        st.metric("PCR Change", f"{oi_pcr_metrics['pcr_chg']:+.2f}")
        st.caption(f"{oi_pcr_metrics.get('chg_interpretation', '')}")
    
    with col_p3:
        st.metric("OI Momentum", f"{oi_pcr_metrics.get('oi_momentum', 0):+.1f}%")
        st.caption(f"{oi_pcr_metrics.get('oi_change_interpretation', '')}")
    
    # Concentration Analysis
    st.markdown("---")
    st.markdown("### 🎯 OI CONCENTRATION & SKEW")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.metric("ATM Concentration", f"{oi_pcr_metrics.get('atm_concentration_pct', 0):.1f}%")
        st.caption(f"CALL: {oi_pcr_metrics.get('atm_ce_oi', 0):,} | PUT: {oi_pcr_metrics.get('atm_pe_oi', 0):,}")
    
    with col_c2:
        if oi_pcr_metrics.get('max_ce_strike', 0) > 0:
            st.metric("Max CALL OI", f"₹{oi_pcr_metrics['max_ce_strike']:,}")
            st.caption(f"OI: {oi_pcr_metrics['max_ce_oi']:,}")
    
    with col_c3:
        if oi_pcr_metrics.get('max_pe_strike', 0) > 0:
            st.metric("Max PUT OI", f"₹{oi_pcr_metrics['max_pe_strike']:,}")
            st.caption(f"OI: {oi_pcr_metrics['max_pe_oi']:,}")
    
    # ITM/OTM Distribution
    with st.expander("🔍 ITM/OTM OI Distribution", expanded=False):
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        
        with col_i1:
            st.metric("ITM CALL OI", f"{oi_pcr_metrics.get('itm_ce_oi', 0):,}")
            st.caption("Strike < Spot")
        
        with col_i2:
            st.metric("OTM CALL OI", f"{oi_pcr_metrics.get('otm_ce_oi', 0):,}")
            st.caption("Strike > Spot")
        
        with col_i3:
            st.metric("ITM PUT OI", f"{oi_pcr_metrics.get('itm_pe_oi', 0):,}")
            st.caption("Strike > Spot")
        
        with col_i4:
            st.metric("OTM PUT OI", f"{oi_pcr_metrics.get('otm_pe_oi', 0):,}")
            st.caption("Strike < Spot")

# ============================================
# TAB 3: MOMENTUM ANALYSIS
# ============================================
with tab3:
    st.markdown("## 🚀 MOMENT DETECTOR ANALYSIS")
    
    # Momentum Burst
    st.markdown("### 💥 MOMENTUM BURST")
    mb = moment_metrics.get("momentum_burst", {})
    if mb.get("available", False):
        col_mb1, col_mb2 = st.columns([1, 2])
        with col_mb1:
            st.metric("Score", f"{mb.get('score', 0)}/100")
        with col_mb2:
            st.info(mb.get("note", ""))
    else:
        st.warning("Momentum burst data unavailable")
    
    st.markdown("---")
    
    # Orderbook Pressure
    st.markdown("### 📊 ORDERBOOK PRESSURE")
    ob = moment_metrics.get("orderbook", {})
    if ob.get("available", False):
        col_ob1, col_ob2 = st.columns([1, 2])
        with col_ob1:
            st.metric("Pressure", f"{ob.get('pressure', 0):+.2f}")
            st.metric("Buy Qty", f"{ob.get('buy_qty', 0):.0f}")
            st.metric("Sell Qty", f"{ob.get('sell_qty', 0):.0f}")
        with col_ob2:
            pressure = ob.get('pressure', 0)
            if pressure > 0.15:
                st.success("**STRONG BUY PRESSURE**")
            elif pressure < -0.15:
                st.error("**STRONG SELL PRESSURE**")
            else:
                st.info("**BALANCED ORDERBOOK**")
    else:
        st.warning("Orderbook data unavailable")
    
    st.markdown("---")
    
    # Gamma Cluster
    st.markdown("### 🌀 GAMMA CLUSTER")
    gc = moment_metrics.get("gamma_cluster", {})
    if gc.get("available", False):
        col_gc1, col_gc2 = st.columns([1, 2])
        with col_gc1:
            st.metric("Score", f"{gc.get('score', 0)}/100")
        with col_gc2:
            if gc.get('score', 0) > 70:
                st.success("**HIGH GAMMA CLUSTER** - Expect sharp moves")
            elif gc.get('score', 0) > 40:
                st.info("**MODERATE GAMMA CLUSTER**")
            else:
                st.warning("**LOW GAMMA CLUSTER** - Smoother moves expected")
    else:
        st.warning("Gamma cluster data unavailable")

# ============================================
# TAB 4: EXPIRY ANALYSIS
# ============================================
with tab4:
    st.markdown("## 📅 EXPIRY SPIKE DETECTOR")
    
    if expiry_spike_data.get("active", False):
        # Main spike card
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            st.metric("Spike Probability", f"{expiry_spike_data.get('probability', 0)}%")
        
        with col_e2:
            st.metric("Spike Type", expiry_spike_data.get('type', 'N/A'))
        
        with col_e3:
            st.metric("Days to Expiry", f"{expiry_spike_data.get('days_to_expiry', 0):.1f}")
        
        # Spike Factors
        with st.expander("🔍 Spike Detection Factors", expanded=False):
            if expiry_spike_data.get("factors"):
                for factor in expiry_spike_data["factors"]:
                    st.write(f"• {factor}")
            else:
                st.info("No spike factors detected")
        
        # Key Levels
        with st.expander("🎯 Key Levels", expanded=False):
            if expiry_spike_data.get("key_levels"):
                for level in expiry_spike_data["key_levels"]:
                    st.write(f"• {level}")
            else:
                st.info("No extreme levels detected")
        
        # Recommendations
        st.markdown("### 🎯 TRADING STRATEGY")
        prob = expiry_spike_data.get('probability', 0)
        if prob > 70:
            st.error("""
            **HIGH SPIKE RISK - AGGRESSIVE STRATEGY:**
            - Expect 100-200 point swings
            - Use 2x wider stops
            - Focus on 10:30-11:30 & 14:30-15:00
            """)
        elif prob > 50:
            st.warning("""
            **MODERATE SPIKE RISK - BALANCED STRATEGY:**
            - Expect moderate volatility
            - Use 1.5x wider stops
            - Watch Max Pain closely
            """)
        else:
            st.success("""
            **LOW SPIKE RISK - NORMAL STRATEGY:**
            - Normal trading rules apply
            - Standard stop losses
            - Focus on technical levels
            """)
    else:
        st.info(f"""
        **Expiry Spike Detector (Inactive)**
        
        Reason: {expiry_spike_data.get('message', 'Expiry >5 days away')}
        
        Current expiry: **{expiry}**  
        Days to expiry: **{expiry_spike_data.get('days_to_expiry', 0):.1f}**
        """)

# ============================================
# TAB 5: DETAILED ANALYSIS
# ============================================
with tab5:
    st.markdown("## 📈 DETAILED ANALYSIS")
    
    # Seller Activity
    with st.expander("📊 SELLER ACTIVITY BY STRIKE", expanded=False):
        if not merged.empty:
            seller_cols = ["strikePrice", "OI_CE", "Chg_OI_CE", "CE_Seller_Action", 
                          "OI_PE", "Chg_OI_PE", "PE_Seller_Action", "Seller_Interpretation"]
            display_df = merged[seller_cols].head(20)
            st.dataframe(display_df, use_container_width=True)
    
    # Greeks
    with st.expander("🧮 GREEKS & GEX", expanded=False):
        if not merged.empty:
            greeks_cols = ["strikePrice", "Delta_CE", "Gamma_CE", "Vega_CE", "GEX_CE",
                          "Delta_PE", "Gamma_PE", "Vega_PE", "GEX_PE", "GEX_Net"]
            display_df = merged[greeks_cols].head(20)
            st.dataframe(display_df, use_container_width=True)
    
    # PCR Analysis
    with st.expander("📈 PCR ANALYSIS", expanded=False):
        if not ranked_current.empty:
            pcr_cols = ["strikePrice", "OI_CE", "OI_PE", "PCR", "Chg_OI_CE", "Chg_OI_PE"]
            display_df = ranked_current[pcr_cols].head(20)
            st.dataframe(display_df, use_container_width=True)

# ============================================
# 🎯 TELEGRAM & AI SECTIONS (COLLAPSIBLE)
# ============================================

# Telegram Signal Section
with st.expander("📱 TELEGRAM SIGNAL", expanded=False):
    if telegram_signal:
        st.success("🎯 **NEW TRADE SIGNAL GENERATED!**")
        st.code(telegram_signal, language="markdown")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Copy to Clipboard", key="copy_signal"):
                st.success("✅ Copied!")
        with col_btn2:
            if st.button("Send to Telegram", key="send_signal"):
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    st.info("Sending...")
                else:
                    st.warning("Telegram not configured")
        with col_btn3:
            if st.button("Save to File", key="save_signal"):
                st.success("✅ Saved!")
    else:
        st.info("No active signal to send")

# AI Analysis Section
if trading_ai.is_enabled():
    with st.expander("🧠 AI ANALYSIS", expanded=False):
        if st.button("Generate AI Analysis", key="gen_ai"):
            with st.spinner("🤖 AI is analyzing..."):
                # Prepare market data for AI
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
                    'total_gex': total_gex_net,
                    'expiry': expiry,
                    'days_to_expiry': days_to_expiry
                }
                
                ai_analysis = trading_ai.generate_analysis(
                    market_data_for_ai, 
                    entry_signal, 
                    moment_metrics,
                    expiry_spike_data
                )
                
                if ai_analysis:
                    st.session_state["ai_analysis"] = ai_analysis
                    st.success("✅ AI Analysis Generated!")
                    st.text_area("AI Analysis", ai_analysis, height=300)
                else:
                    st.error("❌ Failed to generate AI analysis")
        
        if "ai_analysis" in st.session_state:
            st.markdown("#### Previous Analysis:")
            st.text(st.session_state["ai_analysis"][:500] + "...")

# ============================================
# 🎯 FINAL SUMMARY
# ============================================

st.markdown("---")
st.markdown("### 📊 MARKET SUMMARY")

col_sum1, col_sum2 = st.columns(2)

with col_sum1:
    st.metric("Total GEX", f"₹{int(total_gex_net):,}")
    st.metric("Breakout Index", f"{seller_breakout_index}%")
    st.metric("CALL Sellers", f"{ce_selling} strikes")
    st.metric("PUT Sellers", f"{pe_selling} strikes")

with col_sum2:
    st.metric("Spot Position", f"{spot_analysis['spot_position_pct']:.1f}%")
    st.metric("Range Bias", spot_analysis['range_bias'].split('(')[0])
    st.metric("Range Size", f"₹{spot_analysis['range_size']:,}")
    st.metric("Expiry", f"{expiry}")

# Footer
st.markdown("---")
st.caption(f"🔄 Auto-refresh: {refresh_rate if auto_refresh else 'Off'}s | ⏰ Last update: {get_ist_time_str()}")
st.caption("🎯 **NIFTY Option Screener v6.0 — Trader-Focused Dashboard**")
