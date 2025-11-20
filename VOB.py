import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
import json
import time
import numpy as np
from collections import deque
import warnings
import math
from scipy.stats import norm
import io
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# TRADING SIGNAL MANAGER WITH COOLDOWN
# =============================================

class TradingSignalManager:
    """Manage trading signals with cooldown periods"""
    
    def __init__(self, cooldown_minutes=15):
        self.cooldown_minutes = cooldown_minutes
        self.last_signal_time = {}
        self.sent_signals = set()
        
    def can_send_signal(self, signal_type, instrument):
        """Check if signal can be sent based on cooldown"""
        key = f"{signal_type}_{instrument}"
        current_time = datetime.now()
        
        if key in self.last_signal_time:
            last_sent = self.last_signal_time[key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return False, self.cooldown_minutes - int(time_diff)
        
        self.last_signal_time[key] = current_time
        return True, 0
    
    def generate_trading_recommendation(self, instrument_data):
        """Generate trading recommendation based on comprehensive analysis"""
        try:
            overall_bias = instrument_data['overall_bias']
            bias_score = instrument_data['bias_score']
            spot_price = instrument_data['spot_price']
            comp_metrics = instrument_data.get('comprehensive_metrics', {})
            detailed_bias = instrument_data.get('detailed_atm_bias', {})
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(instrument_data, comp_metrics)
            
            # Generate signal based on bias strength and confidence
            if "Strong Bullish" in overall_bias and confidence >= 75 and bias_score >= 3:
                signal_type = "STRONG_BUY"
                direction = "BULLISH"
                strength = "HIGH"
            elif "Bullish" in overall_bias and confidence >= 60 and bias_score >= 2:
                signal_type = "BUY"
                direction = "BULLISH" 
                strength = "MEDIUM"
            elif "Strong Bearish" in overall_bias and confidence >= 75 and bias_score <= -3:
                signal_type = "STRONG_SELL"
                direction = "BEARISH"
                strength = "HIGH"
            elif "Bearish" in overall_bias and confidence >= 60 and bias_score <= -2:
                signal_type = "SELL"
                direction = "BEARISH"
                strength = "MEDIUM"
            else:
                return None
            
            # Get key levels
            call_resistance = comp_metrics.get('call_resistance', spot_price + 100)
            put_support = comp_metrics.get('put_support', spot_price - 100)
            max_pain = comp_metrics.get('max_pain_strike', spot_price)
            
            # Generate entry/exit levels
            if direction == "BULLISH":
                entry_zone = f"{put_support:.0f}-{spot_price:.0f}"
                targets = [
                    spot_price + (call_resistance - spot_price) * 0.5,
                    call_resistance
                ]
                stop_loss = put_support - 20
            else:  # BEARISH
                entry_zone = f"{spot_price:.0f}-{call_resistance:.0f}"
                targets = [
                    spot_price - (spot_price - put_support) * 0.5,
                    put_support
                ]
                stop_loss = call_resistance + 20
            
            recommendation = {
                'instrument': instrument_data['instrument'],
                'signal_type': signal_type,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'spot_price': spot_price,
                'bias_score': bias_score,
                'entry_zone': entry_zone,
                'targets': [f"{t:.0f}" for t in targets],
                'stop_loss': f"{stop_loss:.0f}",
                'call_resistance': f"{call_resistance:.0f}",
                'put_support': f"{put_support:.0f}",
                'max_pain': f"{max_pain:.0f}",
                'pcr_oi': instrument_data['pcr_oi'],
                'key_metrics': {
                    'synthetic_bias': comp_metrics.get('synthetic_bias', 'N/A'),
                    'atm_buildup': comp_metrics.get('atm_buildup', 'N/A'),
                    'vega_bias': comp_metrics.get('atm_vega_bias', 'N/A')
                }
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None
    
    def calculate_confidence_score(self, instrument_data, comp_metrics):
        """Calculate confidence score for trading signal"""
        confidence = 50  # Base confidence
        
        # PCR Confidence
        pcr_oi = instrument_data['pcr_oi']
        if pcr_oi > 1.3 or pcr_oi < 0.7:
            confidence += 15
        elif pcr_oi > 1.1 or pcr_oi < 0.9:
            confidence += 10
        
        # Bias Score Confidence
        bias_score = abs(instrument_data['bias_score'])
        if bias_score >= 3:
            confidence += 20
        elif bias_score >= 2:
            confidence += 15
        elif bias_score >= 1:
            confidence += 10
        
        # Synthetic Bias Confidence
        synthetic_bias = comp_metrics.get('synthetic_bias', 'Neutral')
        if 'Bullish' in synthetic_bias or 'Bearish' in synthetic_bias:
            confidence += 10
        
        # Max Pain Confidence
        dist_mp = abs(comp_metrics.get('distance_from_max_pain', 0))
        if dist_mp > 100:
            confidence += 10
        elif dist_mp > 50:
            confidence += 5
        
        # Multiple confirmation factors
        confirming_factors = 0
        if comp_metrics.get('synthetic_bias', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
        if comp_metrics.get('total_vega_bias', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
        if comp_metrics.get('atm_buildup', 'Neutral') == instrument_data['overall_bias']:
            confirming_factors += 1
            
        confidence += confirming_factors * 5
        
        return min(confidence, 95)  # Cap at 95%
    
    def format_signal_message(self, recommendation):
        """Format trading signal for Telegram notification"""
        emoji = "🟢" if recommendation['direction'] == "BULLISH" else "🔴"
        strength_emoji = "🔥" if recommendation['strength'] == "HIGH" else "⚡"
        
        message = f"""
{strength_emoji} {emoji} *TRADING SIGNAL ALERT* {emoji} {strength_emoji}

🎯 *{recommendation['instrument']} - {recommendation['signal_type']}*
⏰ Time: {recommendation['timestamp'].strftime('%H:%M:%S')} IST
📊 Confidence: {recommendation['confidence']}%

💰 Current Price: ₹{recommendation['spot_price']:.2f}
📈 Bias Score: {recommendation['bias_score']:.2f}
🔢 PCR OI: {recommendation['pcr_oi']:.2f}

🎯 *TRADING PLAN:*
• Entry Zone: ₹{recommendation['entry_zone']}
• Target 1: ₹{recommendation['targets'][0]}
• Target 2: ₹{recommendation['targets'][1]}
• Stop Loss: ₹{recommendation['stop_loss']}

📊 *KEY LEVELS:*
• Call Resistance: ₹{recommendation['call_resistance']}
• Put Support: ₹{recommendation['put_support']}
• Max Pain: ₹{recommendation['max_pain']}

🔍 *CONFIRMING METRICS:*
• Synthetic Bias: {recommendation['key_metrics']['synthetic_bias']}
• ATM Buildup: {recommendation['key_metrics']['atm_buildup']}
• Vega Bias: {recommendation['key_metrics']['vega_bias']}

⏳ *Next signal in {self.cooldown_minutes} minutes*

⚠️ *Risk Disclaimer: Trade at your own risk. Use proper position sizing and risk management.*
"""
        return message

# =============================================
# MODIFIED NSE OPTIONS ANALYZER WITH AUTO-REFRESH
# =============================================

class NSEOptionsAnalyzer:
    """Integrated NSE Options Analyzer with auto-refresh capability"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
                'FINNIFTY': {'lot_size': 40, 'atm_range': 200, 'zone_size': 100},
            },
            'stocks': {
                'RELIANCE': {'lot_size': 250, 'atm_range': 100, 'zone_size': 50},
                'TCS': {'lot_size': 150, 'atm_range': 100, 'zone_size': 50},
            }
        }
        self.last_refresh_time = {}
        self.refresh_interval = 60  # 1 minute default refresh
        
    def set_refresh_interval(self, minutes):
        """Set auto-refresh interval"""
        self.refresh_interval = minutes
    
    def should_refresh_data(self, instrument):
        """Check if data should be refreshed based on last refresh time"""
        current_time = datetime.now(self.ist)
        
        if instrument not in self.last_refresh_time:
            self.last_refresh_time[instrument] = current_time
            return True
        
        last_refresh = self.last_refresh_time[instrument]
        time_diff = (current_time - last_refresh).total_seconds() / 60
        
        if time_diff >= self.refresh_interval:
            self.last_refresh_time[instrument] = current_time
            return True
        
        return False

    # ... (keep all your existing NSEOptionsAnalyzer methods exactly as they are) ...
    # Include all the existing methods: calculate_greeks, fetch_option_chain_data, 
    # delta_volume_bias, final_verdict, determine_level, calculate_max_pain,
    # calculate_synthetic_future_bias, calculate_atm_buildup_pattern, 
    # calculate_atm_vega_bias, find_call_resistance_put_support, 
    # calculate_total_vega_bias, detect_unusual_activity, 
    # calculate_overall_buildup_pattern, analyze_comprehensive_atm_bias,
    # calculate_detailed_atm_bias, get_overall_market_bias

    def get_overall_market_bias(self, force_refresh=False):
        """Get comprehensive market bias across all instruments with auto-refresh"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                with st.spinner(f"Fetching {instrument} options data..."):
                    bias_data = self.analyze_comprehensive_atm_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
            else:
                # Return cached data if available and not forcing refresh
                if hasattr(self, 'cached_bias_data') and instrument in self.cached_bias_data:
                    results.append(self.cached_bias_data[instrument])
        
        # Cache the results
        self.cached_bias_data = {data['instrument']: data for data in results}
        
        return results

# =============================================
# MODIFIED ENHANCED NIFTY APP WITH TRADING SIGNALS
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        
        # Initialize all indicators
        self.vob_indicator = VolumeOrderBlocks(sensitivity=5)
        self.volume_spike_detector = VolumeSpikeDetector(lookback_period=20, spike_threshold=2.5)
        self.alert_manager = AlertManager(cooldown_minutes=10)
        self.options_analyzer = NSEOptionsAnalyzer()
        self.trading_signal_manager = TradingSignalManager(cooldown_minutes=15)
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'sent_volume_block_alerts' not in st.session_state:
            st.session_state.sent_volume_block_alerts = set()
        if 'sent_volume_spike_alerts' not in st.session_state:
            st.session_state.sent_volume_spike_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        if 'volume_history' not in st.session_state:
            st.session_state.volume_history = []
        if 'market_bias_data' not in st.session_state:
            st.session_state.market_bias_data = None
        if 'last_bias_update' not in st.session_state:
            st.session_state.last_bias_update = None
        if 'last_signal_check' not in st.session_state:
            st.session_state.last_signal_check = None
        if 'sent_trading_signals' not in st.session_state:
            st.session_state.sent_trading_signals = {}
    
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_client_id = st.secrets["dhan"]["client_id"]
            self.supabase_url = st.secrets["supabase"]["url"]
            self.supabase_key = st.secrets["supabase"]["anon_key"]
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except KeyError as e:
            st.error(f"Missing secret: {e}")
            st.stop()
    
    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            if self.supabase_url and self.supabase_key:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
                self.supabase.table('nifty_data').select("id").limit(1).execute()
            else:
                self.supabase = None
        except Exception as e:
            st.warning(f"Supabase connection error: {str(e)}")
            self.supabase = None

    def check_trading_signals(self):
        """Check for trading signals and send notifications"""
        if not st.session_state.market_bias_data:
            return
        
        current_time = datetime.now(self.ist)
        
        # Check if we should check for signals (every 2 minutes)
        if (st.session_state.last_signal_check and 
            (current_time - st.session_state.last_signal_check).total_seconds() < 120):
            return
        
        st.session_state.last_signal_check = current_time
        
        signals_sent = []
        
        for instrument_data in st.session_state.market_bias_data:
            # Generate trading recommendation
            recommendation = self.trading_signal_manager.generate_trading_recommendation(instrument_data)
            
            if recommendation:
                instrument = recommendation['instrument']
                signal_type = recommendation['signal_type']
                
                # Check cooldown
                can_send, minutes_remaining = self.trading_signal_manager.can_send_signal(signal_type, instrument)
                
                if can_send:
                    # Format and send message
                    message = self.trading_signal_manager.format_signal_message(recommendation)
                    
                    if self.send_telegram_message(message):
                        signals_sent.append(f"{instrument} {signal_type}")
                        st.success(f"Trading signal sent: {instrument} {signal_type}")
                        
                        # Store in session state
                        signal_key = f"{instrument}_{signal_type}_{current_time.strftime('%Y%m%d_%H%M')}"
                        st.session_state.sent_trading_signals[signal_key] = recommendation
                else:
                    st.info(f"Cooldown active for {instrument}: {minutes_remaining} min remaining")
        
        if signals_sent:
            st.rerun()
    
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
            return False

    def display_trading_signals_panel(self):
        """Display panel for trading signals and settings"""
        st.sidebar.header("🎯 Trading Signals")
        
        # Signal settings
        signal_cooldown = st.sidebar.slider(
            "Signal Cooldown (min)", 
            min_value=5, 
            max_value=60, 
            value=15,
            help="Minimum time between trading signals for same instrument"
        )
        
        self.trading_signal_manager.cooldown_minutes = signal_cooldown
        
        options_refresh = st.sidebar.slider(
            "Options Data Refresh (min)",
            min_value=1,
            max_value=10,
            value=2,
            help="How often to refresh options chain data"
        )
        
        self.options_analyzer.set_refresh_interval(options_refresh)
        
        enable_trading_signals = st.sidebar.checkbox(
            "Enable Trading Signals",
            value=True,
            help="Send automated trading recommendations based on options analysis"
        )
        
        # Display recent signals
        if st.session_state.sent_trading_signals:
            st.sidebar.subheader("Recent Signals")
            recent_signals = list(st.session_state.sent_trading_signals.values())[-5:]  # Last 5 signals
            
            for signal in reversed(recent_signals):
                emoji = "🟢" if signal['direction'] == "BULLISH" else "🔴"
                with st.sidebar.expander(f"{emoji} {signal['instrument']} {signal['signal_type']}", expanded=False):
                    st.write(f"Time: {signal['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"Confidence: {signal['confidence']}%")
                    st.write(f"Entry: ₹{signal['entry_zone']}")
                    st.write(f"Targets: ₹{signal['targets'][0]}, ₹{signal['targets'][1]}")
                    st.write(f"SL: ₹{signal['stop_loss']}")
        
        return enable_trading_signals

    def run(self):
        """Main application with trading signals"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Volume Analysis, Comprehensive Options Chain & Trading Signals*")
        
        # Sidebar with trading signals panel
        with st.sidebar:
            st.header("🔧 API Status")
            if st.button("Test API Connection"):
                self.test_api_connection()
            
            # Trading signals settings
            enable_trading_signals = self.display_trading_signals_panel()
            
            st.header("📊 Chart Settings")
            timeframe = st.selectbox("Timeframe", ['1', '3', '5', '15'], index=1)
            
            st.subheader("Volume Order Blocks")
            vob_sensitivity = st.slider("Sensitivity", 3, 10, 5)
            alert_threshold = st.slider("Alert Threshold (points)", 1, 10, 5)
            
            st.subheader("Volume Spike Detection")
            spike_threshold = st.slider("Spike Threshold (x avg)", 2.0, 5.0, 2.5)
            
            st.subheader("Alert Cooldown")
            cooldown_minutes = st.slider("Cooldown (minutes)", 1, 30, 10)
            
            st.subheader("Alerts")
            volume_block_alerts = st.checkbox("Volume Block Alerts", value=True)
            volume_spike_alerts = st.checkbox("Volume Spike Alerts", value=True)
            telegram_enabled = st.checkbox("Enable Telegram", value=bool(self.telegram_bot_token))
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Update cooldown periods
        self.alert_manager.cooldown_minutes = cooldown_minutes
        
        # Main content - Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Options Analysis", "🎯 Trading Signals"])
        
        with tab1:
            # Price Analysis Tab (keep existing code)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            self.vob_indicator = VolumeOrderBlocks(sensitivity=vob_sensitivity)
            self.volume_spike_detector.spike_threshold = spike_threshold
            
            df = pd.DataFrame()
            with st.spinner("Fetching market data..."):
                api_data = self.fetch_intraday_data(interval=timeframe)
                if api_data:
                    df = self.process_data(api_data)
            
            if not df.empty:
                # ... (keep existing price analysis code) ...
                pass
        
        with tab2:
            # Options Analysis Tab with auto-refresh
            st.header("📊 NSE Options Chain Analysis - Auto Refresh")
            
            # Auto-refresh toggle
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f"Options data auto-refreshes every {self.options_analyzer.refresh_interval} minutes")
            with col2:
                if st.button("🔄 Force Refresh", type="primary"):
                    with st.spinner("Force refreshing options data..."):
                        bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                        st.session_state.market_bias_data = bias_data
                        st.session_state.last_bias_update = datetime.now(self.ist)
                        st.success("Options data refreshed!")
            with col3:
                if st.session_state.last_bias_update:
                    st.write(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')}")
            
            st.divider()
            
            # Display options analysis
            self.display_comprehensive_options_analysis()
            
            # Auto-refresh logic
            current_time = datetime.now(self.ist)
            if (st.session_state.last_bias_update is None or 
                (current_time - st.session_state.last_bias_update).total_seconds() > self.options_analyzer.refresh_interval * 60):
                
                with st.spinner("Auto-refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = current_time
                    st.rerun()
        
        with tab3:
            # Trading Signals Tab
            st.header("🎯 Automated Trading Signals")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Trading signals generated from comprehensive options chain analysis")
            with col2:
                if st.button("Check Signals Now", type="primary"):
                    self.check_trading_signals()
            
            st.divider()
            
            # Display current market conditions
            if st.session_state.market_bias_data:
                st.subheader("Current Market Conditions")
                
                for instrument_data in st.session_state.market_bias_data:
                    with st.expander(f"{instrument_data['instrument']} - Signal Readiness", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Bias", instrument_data['overall_bias'])
                        with col2:
                            st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
                        with col3:
                            st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                        with col4:
                            confidence = self.trading_signal_manager.calculate_confidence_score(
                                instrument_data, 
                                instrument_data.get('comprehensive_metrics', {})
                            )
                            st.metric("Signal Confidence", f"{confidence}%")
                        
                        # Generate and display potential signal
                        recommendation = self.trading_signal_manager.generate_trading_recommendation(instrument_data)
                        if recommendation:
                            st.success(f"✅ **{recommendation['signal_type']} Signal Ready**")
                            st.write(f"Strength: {recommendation['strength']} | Confidence: {recommendation['confidence']}%")
                            
                            if enable_trading_signals and telegram_enabled:
                                can_send, minutes_remaining = self.trading_signal_manager.can_send_signal(
                                    recommendation['signal_type'], 
                                    recommendation['instrument']
                                )
                                if can_send:
                                    st.button(
                                        f"Send {recommendation['instrument']} {recommendation['signal_type']} Signal",
                                        on_click=lambda: self.send_trading_signal(recommendation),
                                        key=f"send_{recommendation['instrument']}"
                                    )
                                else:
                                    st.warning(f"Cooldown active: {minutes_remaining} minutes remaining")
                        else:
                            st.info("📊 Monitoring market conditions...")
            
            # Display signal history
            if st.session_state.sent_trading_signals:
                st.divider()
                st.subheader("Signal History")
                
                signals_df = pd.DataFrame(list(st.session_state.sent_trading_signals.values()))
                if not signals_df.empty:
                    # Format the dataframe for display
                    display_df = signals_df[['instrument', 'signal_type', 'direction', 'confidence', 'timestamp']].copy()
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                    display_df = display_df.sort_values('timestamp', ascending=False)
                    
                    st.dataframe(display_df, use_container_width=True)
        
        # Check for trading signals automatically
        if enable_trading_signals and telegram_enabled:
            self.check_trading_signals()
        
        # Cleanup and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()
    
    def send_trading_signal(self, recommendation):
        """Send individual trading signal"""
        message = self.trading_signal_manager.format_signal_message(recommendation)
        if self.send_telegram_message(message):
            st.success(f"Trading signal sent for {recommendation['instrument']}!")
            
            # Store in session state
            current_time = datetime.now(self.ist)
            signal_key = f"{recommendation['instrument']}_{recommendation['signal_type']}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            st.session_state.sent_trading_signals[signal_key] = recommendation
            
            # Update last signal time
            self.trading_signal_manager.last_signal_time[
                f"{recommendation['signal_type']}_{recommendation['instrument']}"
            ] = current_time

# =============================================
# KEEP ALL EXISTING SUPPORTING CLASSES
# =============================================

# Include all your existing classes exactly as they are:
# VolumeSpikeDetector, VolumeOrderBlocks, AlertManager

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()