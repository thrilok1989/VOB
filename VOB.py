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
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

class HTFSupportResistance:
    """Higher Time Frame Support/Resistance Levels by BigBeluga"""
    
    def __init__(self):
        self.levels = {
            '5min': {'pivot_high': None, 'pivot_low': None, 'color': '#00ff88'},
            '10min': {'pivot_high': None, 'pivot_low': None, 'color': '#298ada'},
            '15min': {'pivot_high': None, 'pivot_low': None, 'color': '#9c27b0'},
            '1D': {'pivot_high': None, 'pivot_low': None, 'color': '#ff9800'},
            '1W': {'pivot_high': None, 'pivot_low': None, 'color': '#ff5722'}
        }
        self.sent_alerts = set()
        
    def calculate_pivot_levels(self, df, length=5):
        """Calculate pivot highs and lows"""
        if len(df) < length * 2:
            return None, None
            
        pivot_highs = []
        pivot_lows = []
        
        for i in range(length, len(df) - length):
            # Check for pivot high
            is_pivot_high = True
            for j in range(1, length + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                pivot_highs.append({'index': df.index[i], 'price': df['high'].iloc[i]})
            
            # Check for pivot low
            is_pivot_low = True
            for j in range(1, length + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                pivot_lows.append({'index': df.index[i], 'price': df['low'].iloc[i]})
        
        # Get the most recent pivot levels
        latest_high = pivot_highs[-1]['price'] if pivot_highs else None
        latest_low = pivot_lows[-1]['price'] if pivot_lows else None
        
        return latest_high, latest_low
    
    def update_htf_levels(self, df_5min, df_15min, df_1D, df_1W):
        """Update all HTF levels from different timeframes"""
        try:
            # 5-minute levels
            if len(df_5min) >= 10:
                ph_5min, pl_5min = self.calculate_pivot_levels(df_5min, 4)
                if ph_5min:
                    self.levels['5min']['pivot_high'] = ph_5min
                if pl_5min:
                    self.levels['5min']['pivot_low'] = pl_5min
            
            # 15-minute levels (using 15min data for 10min approximation)
            if len(df_15min) >= 10:
                ph_15min, pl_15min = self.calculate_pivot_levels(df_15min, 5)
                if ph_15min:
                    self.levels['10min']['pivot_high'] = ph_15min
                    self.levels['15min']['pivot_high'] = ph_15min
                if pl_15min:
                    self.levels['10min']['pivot_low'] = pl_15min
                    self.levels['15min']['pivot_low'] = pl_15min
            
            # Daily levels
            if len(df_1D) >= 10:
                ph_1D, pl_1D = self.calculate_pivot_levels(df_1D, 5)
                if ph_1D:
                    self.levels['1D']['pivot_high'] = ph_1D
                if pl_1D:
                    self.levels['1D']['pivot_low'] = pl_1D
            
            # Weekly levels (approximated from daily data)
            if len(df_1D) >= 35:  # 7 weeks of data
                ph_1W, pl_1W = self.calculate_pivot_levels(df_1D, 5)
                if ph_1W:
                    self.levels['1W']['pivot_high'] = ph_1W
                if pl_1W:
                    self.levels['1W']['pivot_low'] = pl_1W
                    
        except Exception as e:
            st.warning(f"HTF levels calculation error: {e}")
    
    def check_price_near_htf_levels(self, current_price, threshold=5):
        """Check if price is near any HTF support/resistance level"""
        nearby_levels = []
        
        for tf, level_data in self.levels.items():
            for level_type in ['pivot_high', 'pivot_low']:
                level_price = level_data[level_type]
                if level_price is not None:
                    distance = abs(current_price - level_price)
                    if distance <= threshold:
                        level_name = "Resistance" if level_type == "pivot_high" else "Support"
                        nearby_levels.append({
                            'timeframe': tf,
                            'type': level_name,
                            'price': level_price,
                            'distance': distance
                        })
        
        return nearby_levels

class VolumeFootprint:
    """Real-Time HTF Volume Footprint by BigBeluga"""
    
    def __init__(self, bins=10, timeframe='1D'):
        self.bins = bins
        self.timeframe = timeframe
        self.profile_high = None
        self.profile_low = None
        self.poc_price = None
        self.volume_profile = {}
        self.sent_alerts = set()
    
    def calculate_volume_profile(self, df):
        """Calculate volume profile and Point of Control (POC)"""
        if len(df) < 10:
            return None, None, None
        
        # Use recent data for profile
        recent_data = df.tail(50)  # Last 50 candles
        
        profile_high = recent_data['high'].max()
        profile_low = recent_data['low'].min()
        
        if profile_high == profile_low:
            return None, None, None
        
        # Calculate bin size
        bin_size = (profile_high - profile_low) / self.bins
        
        # Initialize volume bins
        volume_bins = {i: 0 for i in range(self.bins)}
        
        # Distribute volume into bins
        for idx, row in recent_data.iterrows():
            price_range = (row['high'] - row['low'])
            if price_range == 0:
                continue
                
            volume_per_point = row['volume'] / price_range
            
            for bin_idx in range(self.bins):
                bin_low = profile_low + (bin_idx * bin_size)
                bin_high = bin_low + bin_size
                
                # Calculate overlap between candle and bin
                overlap_low = max(row['low'], bin_low)
                overlap_high = min(row['high'], bin_high)
                overlap_range = max(0, overlap_high - overlap_low)
                
                if overlap_range > 0:
                    volume_bins[bin_idx] += volume_per_point * overlap_range
        
        # Find POC (bin with highest volume)
        if volume_bins:
            poc_bin = max(volume_bins, key=volume_bins.get)
            poc_price = profile_low + (poc_bin * bin_size) + (bin_size / 2)
            
            self.profile_high = profile_high
            self.profile_low = profile_low
            self.poc_price = poc_price
            self.volume_profile = volume_bins
            
            return profile_high, profile_low, poc_price
        
        return None, None, None
    
    def check_price_near_poc(self, current_price, threshold=5):
        """Check if price is near Point of Control"""
        if self.poc_price is None:
            return False, 0
        
        distance = abs(current_price - self.poc_price)
        return distance <= threshold, distance

class VolumeSpikeDetector:
    """Detect sudden volume spikes in real-time"""
    
    def __init__(self, lookback_period=20, spike_threshold=2.5):
        self.lookback_period = lookback_period
        self.spike_threshold = spike_threshold
        self.volume_history = deque(maxlen=lookback_period)
        self.sent_alerts = set()
        
    def detect_volume_spike(self, current_volume, timestamp):
        """Detect if current volume is a spike compared to historical average"""
        if len(self.volume_history) < 5:
            self.volume_history.append(current_volume)
            return False, 0
        
        volume_array = np.array(list(self.volume_history))
        avg_volume = np.mean(volume_array)
        std_volume = np.std(volume_array)
        
        self.volume_history.append(current_volume)
        
        if avg_volume == 0:
            return False, 0
        
        volume_ratio = current_volume / avg_volume
        is_spike = (volume_ratio > self.spike_threshold) and (current_volume > avg_volume + 2 * std_volume)
        
        return is_spike, volume_ratio

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator by BigBeluga"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_lines_count = 500
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.sent_alerts = set()
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df, period=200):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def detect_volume_order_blocks(self, df):
        """Detect Volume Order Blocks based on the Pine Script logic"""
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.calculate_ema(df['close'], self.length1)
        ema2 = self.calculate_ema(df['close'], self.length2)
        
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        atr = self.calculate_atr(df)
        atr1 = atr * 2 / 3
        
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(len(df)):
            if cross_up.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                lowest_idx = lookback_data['low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'low']
                
                vol = lookback_data['volume'].sum()
                
                open_price = lookback_data.loc[lowest_idx, 'open']
                close_price = lookback_data.loc[lowest_idx, 'close']
                src = min(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (src - lowest_price) < atr1.iloc[i] * 0.5:
                    src = lowest_price + atr1.iloc[i] * 0.5
                
                mid = (src + lowest_price) / 2
                
                bullish_blocks.append({
                    'index': lowest_idx,
                    'upper': src,
                    'lower': lowest_price,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bullish'
                })
                
            elif cross_down.iloc[i]:
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                highest_idx = lookback_data['high'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'high']
                
                vol = lookback_data['volume'].sum()
                
                open_price = lookback_data.loc[highest_idx, 'open']
                close_price = lookback_data.loc[highest_idx, 'close']
                src = max(open_price, close_price)
                
                if pd.notna(atr.iloc[i]) and (highest_price - src) < atr1.iloc[i] * 0.5:
                    src = highest_price - atr1.iloc[i] * 0.5
                
                mid = (src + highest_price) / 2
                
                bearish_blocks.append({
                    'index': highest_idx,
                    'upper': highest_price,
                    'lower': src,
                    'mid': mid,
                    'volume': vol,
                    'type': 'bearish'
                })
        
        bullish_blocks = self.filter_overlapping_blocks(bullish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        bearish_blocks = self.filter_overlapping_blocks(bearish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        
        return bullish_blocks, bearish_blocks
    
    def filter_overlapping_blocks(self, blocks, atr_value):
        if not blocks:
            return []
        
        filtered_blocks = []
        for block in blocks:
            overlap = False
            for existing_block in filtered_blocks:
                if abs(block['mid'] - existing_block['mid']) < atr_value:
                    overlap = True
                    break
            if not overlap:
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def check_price_near_blocks(self, current_price, blocks, threshold=5):
        nearby_blocks = []
        for block in blocks:
            distance_to_upper = abs(current_price - block['upper'])
            distance_to_lower = abs(current_price - block['lower'])
            distance_to_mid = abs(current_price - block['mid'])
            
            if (distance_to_upper <= threshold or 
                distance_to_lower <= threshold or 
                distance_to_mid <= threshold):
                nearby_blocks.append(block)
        
        return nearby_blocks

class AlertManager:
    """Manage cooldown periods for all alerts"""
    
    def __init__(self, cooldown_minutes=10):
        self.cooldown_minutes = cooldown_minutes
        self.alert_timestamps = {}
        
    def can_send_alert(self, alert_type, alert_id):
        """Check if alert can be sent (cooldown period passed)"""
        key = f"{alert_type}_{alert_id}"
        current_time = datetime.now()
        
        if key in self.alert_timestamps:
            last_sent = self.alert_timestamps[key]
            time_diff = (current_time - last_sent).total_seconds() / 60
            if time_diff < self.cooldown_minutes:
                return False
        
        self.alert_timestamps[key] = current_time
        return True
    
    def cleanup_old_alerts(self, max_age_hours=24):
        """Clean up old alert timestamps"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.alert_timestamps.items():
            time_diff = (current_time - timestamp).total_seconds() / 3600
            if time_diff > max_age_hours:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.alert_timestamps[key]

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        
        # Initialize all indicators
        self.vob_indicator = VolumeOrderBlocks(sensitivity=5)
        self.volume_spike_detector = VolumeSpikeDetector(lookback_period=20, spike_threshold=2.5)
        self.htf_sr = HTFSupportResistance()
        self.volume_footprint = VolumeFootprint(bins=10, timeframe='1D')
        self.alert_manager = AlertManager(cooldown_minutes=10)
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'sent_rsi_alerts' not in st.session_state:
            st.session_state.sent_rsi_alerts = set()
        if 'sent_rsi_oi_alerts' not in st.session_state:
            st.session_state.sent_rsi_oi_alerts = set()
        if 'sent_volume_block_alerts' not in st.session_state:
            st.session_state.sent_volume_block_alerts = set()
        if 'sent_volume_spike_alerts' not in st.session_state:
            st.session_state.sent_volume_spike_alerts = set()
        if 'sent_htf_level_alerts' not in st.session_state:
            st.session_state.sent_htf_level_alerts = set()
        if 'sent_poc_alerts' not in st.session_state:
            st.session_state.sent_poc_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        if 'volume_history' not in st.session_state:
            st.session_state.volume_history = []
    
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
    
    def get_dhan_headers(self):
        """Get headers for DhanHQ API calls"""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.dhan_token,
            'client-id': self.dhan_client_id
        }
    
    def test_api_connection(self):
        """Test DhanHQ API connection"""
        st.info("🔍 Testing API connection...")
        test_payload = {"IDX_I": [self.nifty_security_id]}
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/marketfeed/ltp",
                headers=self.get_dhan_headers(),
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ API Connection Successful!")
                return True
            else:
                st.error(f"❌ API Connection Failed: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ API Test Failed: {str(e)}")
            return False

    def fetch_intraday_data(self, interval="5", days_back=5):
        """Fetch intraday data from DhanHQ API"""
        try:
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=min(days_back, 90))
            
            from_date = start_date.strftime("%Y-%m-%d 09:15:00")
            to_date = end_date.strftime("%Y-%m-%d 15:30:00")
            
            payload = {
                "securityId": str(self.nifty_security_id),
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": str(interval),
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                "https://api.dhan.co/v2/charts/intraday",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data or 'open' not in data or len(data['open']) == 0:
                    st.warning("⚠️ API returned empty data")
                    return None
                st.success(f"✅ Data fetched: {len(data['open'])} candles")
                return data
            else:
                st.error(f"❌ API Error {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"❌ Data fetch error: {str(e)}")
            return None

    def process_data(self, api_data):
        """Process API data into DataFrame"""
        if not api_data or 'open' not in api_data:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': api_data['timestamp'],
            'open': api_data['open'],
            'high': api_data['high'],
            'low': api_data['low'],
            'close': api_data['close'],
            'volume': api_data['volume']
        })
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(self.ist)
        df = df.set_index('datetime')
        
        return df

    def send_telegram_message(self, message):
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
            return False

    def check_volume_block_alerts(self, current_price, bullish_blocks, bearish_blocks, threshold=5):
        """Check if price is near volume order blocks and send alerts"""
        if not bullish_blocks and not bearish_blocks:
            return False
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        # Check bullish blocks
        nearby_bullish = self.vob_indicator.check_price_near_blocks(current_price, bullish_blocks, threshold)
        for block in nearby_bullish:
            alert_id = f"vol_block_bullish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_block", alert_id):
                message = f"""🚨 PRICE NEAR BULLISH VOLUME ORDER BLOCK!

📊 Nifty 50 Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

🎯 Volume Order Block:
• Type: BULLISH 
• Upper: ₹{block['upper']:.2f}
• Lower: ₹{block['lower']:.2f}
• Mid: ₹{block['mid']:.2f}
• Volume: {block['volume']:,}

📈 Distance to Block: {abs(current_price - block['mid']):.2f} points

💡 Trading Suggestion:
Consider LONG positions with stop below support

⏳ Next alert in 10 minutes

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Bullish Volume Block alert sent!")
                    alert_sent = True
        
        # Check bearish blocks
        nearby_bearish = self.vob_indicator.check_price_near_blocks(current_price, bearish_blocks, threshold)
        for block in nearby_bearish:
            alert_id = f"vol_block_bearish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_block", alert_id):
                message = f"""🚨 PRICE NEAR BEARISH VOLUME ORDER BLOCK!

📊 Nifty 50 Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

🎯 Volume Order Block:
• Type: BEARISH 
• Upper: ₹{block['upper']:.2f}
• Lower: ₹{block['lower']:.2f}
• Mid: ₹{block['mid']:.2f}
• Volume: {block['volume']:,}

📉 Distance to Block: {abs(current_price - block['mid']):.2f} points

💡 Trading Suggestion:
Consider SHORT positions with stop above resistance

⏳ Next alert in 10 minutes

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Bearish Volume Block alert sent!")
                    alert_sent = True
        
        return alert_sent

    def check_volume_spike_alerts(self, df):
        """Check for sudden volume spikes and send alerts"""
        if df.empty or len(df) < 2:
            return False
        
        current_candle = df.iloc[-1]
        previous_candle = df.iloc[-2]
        
        current_volume = current_candle['volume']
        current_time = current_candle.name
        current_price = current_candle['close']
        price_change = current_candle['close'] - previous_candle['close']
        price_change_pct = (price_change / previous_candle['close']) * 100
        
        # Detect volume spike
        is_spike, volume_ratio = self.volume_spike_detector.detect_volume_spike(current_volume, current_time)
        
        if is_spike:
            alert_id = f"volume_spike_{current_time.strftime('%Y%m%d_%H%M')}"
            
            if self.alert_manager.can_send_alert("volume_spike", alert_id):
                spike_type = "BUYING" if price_change > 0 else "SELLING"
                emoji = "🟢" if price_change > 0 else "🔴"
                
                message = f"""📈 SUDDEN VOLUME SPIKE DETECTED!

{emoji} Nifty 50 Volume Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

📊 Volume Analysis:
• Current Volume: {current_volume:,}
• Volume Ratio: {volume_ratio:.1f}x average
• Price Change: ₹{price_change:+.2f} ({price_change_pct:+.2f}%)

🎯 Spike Type: {spike_type} PRESSURE

💡 Market Interpretation:
High volume with {spike_type.lower()} pressure indicates 
strong institutional activity

⏳ Next alert in 10 minutes

⚡ Immediate Action:
Watch for breakout/breakdown confirmation!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Volume Spike alert sent! ({volume_ratio:.1f}x average)")
                    return True
        
        return False

    def check_htf_level_alerts(self, current_price, threshold=5):
        """Check if price is near HTF support/resistance levels"""
        nearby_levels = self.htf_sr.check_price_near_htf_levels(current_price, threshold)
        
        if not nearby_levels:
            return False
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        for level in nearby_levels:
            alert_id = f"htf_{level['timeframe']}_{level['type']}_{level['price']}"
            
            if self.alert_manager.can_send_alert("htf_level", alert_id):
                message = f"""🎯 PRICE NEAR HTF {level['type'].upper()}!

📊 Nifty 50 HTF Level Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

🎯 {level['timeframe']} {level['type']}:
• Level Price: ₹{level['price']:.2f}
• Distance: {level['distance']:.2f} points

💡 Trading Suggestion:
{level['type'].upper()} levels often act as strong 
reversal or breakout points

⏳ Next alert in 10 minutes

⚠️ Watch for price reaction at this level!"""
                
                if self.send_telegram_message(message):
                    st.success(f"HTF {level['type']} alert sent for {level['timeframe']}!")
                    alert_sent = True
        
        return alert_sent

    def check_poc_alerts(self, current_price, threshold=5):
        """Check if price is near Volume Footprint POC"""
        is_near_poc, distance = self.volume_footprint.check_price_near_poc(current_price, threshold)
        
        if is_near_poc:
            current_time = datetime.now(self.ist)
            alert_id = f"poc_{self.volume_footprint.poc_price}"
            
            if self.alert_manager.can_send_alert("poc", alert_id):
                message = f"""📊 PRICE NEAR VOLUME POC (Point of Control)!

🎯 Nifty 50 Volume Profile Alert
⏰ Time: {current_time.strftime('%H:%M:%S')} IST
💰 Current Price: ₹{current_price:.2f}

📈 Volume Footprint Analysis:
• POC Price: ₹{self.volume_footprint.poc_price:.2f}
• Distance: {distance:.2f} points
• Profile High: ₹{self.volume_footprint.profile_high:.2f}
• Profile Low: ₹{self.volume_footprint.profile_low:.2f}

💡 Market Interpretation:
POC represents the price with highest trading activity
Often acts as strong support/resistance

⏳ Next alert in 10 minutes

⚡ Watch for institutional activity at this level!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Volume POC alert sent! (Distance: {distance:.2f} points)")
                    return True
        
        return False

    def create_comprehensive_chart(self, df, bullish_blocks, bearish_blocks, interval):
        """Create comprehensive chart with all indicators"""
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Nifty 50 Comprehensive Analysis - {interval} Min', 'Volume with Indicators'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add HTF Support/Resistance Levels
        for tf, level_data in self.htf_sr.levels.items():
            color = level_data['color']
            
            # Pivot High (Resistance)
            if level_data['pivot_high'] is not None:
                fig.add_hline(
                    y=level_data['pivot_high'],
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{tf} Resistance",
                    annotation_position="top right",
                    row=1, col=1
                )
            
            # Pivot Low (Support)
            if level_data['pivot_low'] is not None:
                fig.add_hline(
                    y=level_data['pivot_low'],
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{tf} Support",
                    annotation_position="bottom right",
                    row=1, col=1
                )
        
        # Add Volume Footprint POC
        if self.volume_footprint.poc_price is not None:
            fig.add_hline(
                y=self.volume_footprint.poc_price,
                line_dash="dot",
                line_color='#298ada',
                line_width=3,
                annotation_text="POC",
                annotation_position="left",
                row=1, col=1
            )
        
        # Add Volume Order Blocks
        colors = {'bullish': '#26ba9f', 'bearish': '#6626ba'}
        
        for block in bullish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(38, 186, 159, 0.1)',
                line=dict(color=colors['bullish'], width=1),
                row=1, col=1
            )
        
        for block in bearish_blocks:
            fig.add_shape(
                type="rect",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['lower'],
                fillcolor='rgba(102, 38, 186, 0.1)',
                line=dict(color=colors['bearish'], width=1),
                row=1, col=1
            )
        
        # Volume bars with spike detection
        bar_colors = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if i < len(df) - 1:
                bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
            else:
                current_volume = row['volume']
                if len(df) > 5:
                    avg_volume = df['volume'].iloc[-6:-1].mean()
                    if current_volume > avg_volume * 2.5:
                        bar_colors.append('#ffeb3b')
                    else:
                        bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
                else:
                    bar_colors.append('#00ff88' if row['close'] >= row['open'] else '#ff4444')
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=bar_colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', side="right")
        
        return fig

    def run(self):
        """Main application"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*HTF Support/Resistance, Volume Footprint, Order Blocks & Real-time Alerts*")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 API Status")
            if st.button("Test API Connection"):
                self.test_api_connection()
            
            st.header("📊 Settings")
            
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
            htf_level_alerts = st.checkbox("HTF Level Alerts", value=True)
            poc_alerts = st.checkbox("Volume POC Alerts", value=True)
            telegram_enabled = st.checkbox("Enable Telegram", value=bool(self.telegram_bot_token))
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Update cooldown period
        self.alert_manager.cooldown_minutes = cooldown_minutes
        
        # Main content
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        # Update detectors
        self.vob_indicator = VolumeOrderBlocks(sensitivity=vob_sensitivity)
        self.volume_spike_detector.spike_threshold = spike_threshold
        
        # Fetch data for different timeframes
        df_5min = pd.DataFrame()
        df_15min = pd.DataFrame()
        df_1D = pd.DataFrame()
        
        with st.spinner("Fetching market data..."):
            # Fetch 5min data for current chart
            api_data_5min = self.fetch_intraday_data(interval=timeframe)
            if api_data_5min:
                df_5min = self.process_data(api_data_5min)
            
            # Fetch 15min data for HTF levels
            api_data_15min = self.fetch_intraday_data(interval="15")
            if api_data_15min:
                df_15min = self.process_data(api_data_15min)
            
            # Fetch 1D data (approximated from intraday)
            api_data_1D = self.fetch_intraday_data(interval="60", days_back=30)
            if api_data_1D:
                df_1D = self.process_data(api_data_1D)
        
        if not df_5min.empty:
            latest = df_5min.iloc[-1]
            current_price = latest['close']
            current_volume = latest['volume']
            
            # Update all indicators
            bullish_blocks, bearish_blocks = self.vob_indicator.detect_volume_order_blocks(df_5min)
            self.htf_sr.update_htf_levels(df_5min, df_15min, df_1D, df_1D)  # Using same for weekly
            self.volume_footprint.calculate_volume_profile(df_1D)
            
            # Display metrics
            with col1:
                st.metric("Nifty Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("HTF Levels", f"{sum(1 for tf in self.htf_sr.levels.values() if tf['pivot_high'] or tf['pivot_low'])}")
            with col3:
                poc_status = "✅" if self.volume_footprint.poc_price else "❌"
                st.metric("Volume POC", poc_status)
            with col4:
                st.metric("Volume Blocks", f"{len(bullish_blocks) + len(bearish_blocks)}")
            with col5:
                if len(df_5min) > 5:
                    avg_vol = df_5min['volume'].iloc[-6:-1].mean()
                    vol_ratio = current_volume / avg_vol if avg_vol > 0 else 0
                    st.metric("Volume Ratio", f"{vol_ratio:.1f}x")
            with col6:
                alert_status = "✅ Active" if telegram_enabled else "❌ Inactive"
                st.metric("Alerts", alert_status)
            
            # Create and display chart
            chart = self.create_comprehensive_chart(df_5min, bullish_blocks, bearish_blocks, timeframe)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Check for alerts
            alerts_sent = []
            if telegram_enabled:
                if volume_block_alerts:
                    block_alert = self.check_volume_block_alerts(current_price, bullish_blocks, bearish_blocks, alert_threshold)
                    if block_alert:
                        alerts_sent.append("Volume Block")
                
                if volume_spike_alerts:
                    spike_alert = self.check_volume_spike_alerts(df_5min)
                    if spike_alert:
                        alerts_sent.append("Volume Spike")
                
                if htf_level_alerts:
                    htf_alert = self.check_htf_level_alerts(current_price, alert_threshold)
                    if htf_alert:
                        alerts_sent.append("HTF Level")
                
                if poc_alerts:
                    poc_alert = self.check_poc_alerts(current_price, alert_threshold)
                    if poc_alert:
                        alerts_sent.append("Volume POC")
            
            if alerts_sent:
                st.success(f"📱 Alerts sent: {', '.join(alerts_sent)} (Cooldown: {cooldown_minutes}min)")
            
            # Display HTF Levels Summary
            st.subheader("🎯 Higher Time Frame Levels")
            htf_cols = st.columns(5)
            
            for i, (tf, level_data) in enumerate(self.htf_sr.levels.items()):
                with htf_cols[i]:
                    st.write(f"**{tf}**")
                    if level_data['pivot_high']:
                        distance_high = abs(current_price - level_data['pivot_high'])
                        st.write(f"Resistance: ₹{level_data['pivot_high']:.2f}")
                        st.write(f"Distance: {distance_high:.2f}")
                    if level_data['pivot_low']:
                        distance_low = abs(current_price - level_data['pivot_low'])
                        st.write(f"Support: ₹{level_data['pivot_low']:.2f}")
                        st.write(f"Distance: {distance_low:.2f}")
            
            # Display Volume Footprint Info
            if self.volume_footprint.poc_price:
                st.subheader("📊 Volume Footprint Profile")
                footprint_cols = st.columns(4)
                
                with footprint_cols[0]:
                    st.metric("POC Price", f"₹{self.volume_footprint.poc_price:.2f}")
                with footprint_cols[1]:
                    st.metric("Profile High", f"₹{self.volume_footprint.profile_high:.2f}")
                with footprint_cols[2]:
                    st.metric("Profile Low", f"₹{self.volume_footprint.profile_low:.2f}")
                with footprint_cols[3]:
                    distance_poc = abs(current_price - self.volume_footprint.poc_price)
                    st.metric("Distance to POC", f"{distance_poc:.2f}")
        
        else:
            st.error("No data available. Please check your API credentials and try again.")
        
        # Cleanup old alerts and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
