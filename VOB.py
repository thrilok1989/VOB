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
# NSE OPTIONS ANALYZER INTEGRATION
# =============================================

class NSEOptionsAnalyzer:
    """Integrated NSE Options Analyzer with enhanced features"""
    
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
        
    def calculate_greeks(self, option_type, S, K, T, r, sigma):
        """Calculate option Greeks"""
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == 'CE':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
                
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
            
            return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
        except:
            return 0, 0, 0, 0, 0

    def fetch_option_chain_data(self, instrument):
        """Fetch option chain data from NSE"""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}" if instrument in self.NSE_INSTRUMENTS['indices'] else \
                  f"https://www.nseindia.com/api/option-chain-equities?symbol={url_instrument}"

            response = session.get(url, timeout=10)
            data = response.json()

            records = data['records']['data']
            expiry = data['records']['expiryDates'][0]
            underlying = data['records']['underlyingValue']

            # Calculate totals
            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item)
            total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item)

            return {
                'success': True,
                'instrument': instrument,
                'spot': underlying,
                'expiry': expiry,
                'total_ce_oi': total_ce_oi,
                'total_pe_oi': total_pe_oi,
                'total_ce_change': total_ce_change,
                'total_pe_change': total_pe_change,
                'records': records
            }
        except Exception as e:
            return {
                'success': False,
                'instrument': instrument,
                'error': str(e)
            }

    def analyze_atm_bias(self, instrument):
        """Analyze ATM bias for an instrument"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                return None

            records = data['records']
            spot = data['spot']
            expiry = data['expiry']

            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06

            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)

                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)

            if not calls or not puts:
                return None

            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

            # Find ATM strike
            atm_range = self.NSE_INSTRUMENTS['indices'].get(instrument, {}).get('atm_range', 200)
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            df_atm = df[abs(df['strikePrice'] - atm_strike) <= atm_range]

            if df_atm.empty:
                return None

            # Calculate ATM bias metrics
            total_score = 0
            weights = {
                "oi_bias": 2, "chg_oi_bias": 2, "volume_bias": 1, 
                "iv_bias": 1, "premium_bias": 1, "delta_bias": 1
            }

            for _, row in df_atm.iterrows():
                score = 0
                
                # OI Bias
                oi_bias = "Bullish" if row['openInterest_PE'] > row['openInterest_CE'] else "Bearish"
                score += weights["oi_bias"] if oi_bias == "Bullish" else -weights["oi_bias"]
                
                # Change in OI Bias
                chg_oi_bias = "Bullish" if row['changeinOpenInterest_PE'] > row['changeinOpenInterest_CE'] else "Bearish"
                score += weights["chg_oi_bias"] if chg_oi_bias == "Bullish" else -weights["chg_oi_bias"]
                
                # Volume Bias
                volume_bias = "Bullish" if row['totalTradedVolume_PE'] > row['totalTradedVolume_CE'] else "Bearish"
                score += weights["volume_bias"] if volume_bias == "Bullish" else -weights["volume_bias"]
                
                # IV Bias
                iv_bias = "Bullish" if row['impliedVolatility_PE'] > row['impliedVolatility_CE'] else "Bearish"
                score += weights["iv_bias"] if iv_bias == "Bullish" else -weights["iv_bias"]
                
                # Premium Bias
                premium_bias = "Bullish" if row['lastPrice_PE'] > row['lastPrice_CE'] else "Bearish"
                score += weights["premium_bias"] if premium_bias == "Bullish" else -weights["premium_bias"]
                
                # Delta Bias
                delta_bias = "Bullish" if abs(row['Delta_PE']) > abs(row['Delta_CE']) else "Bearish"
                score += weights["delta_bias"] if delta_bias == "Bullish" else -weights["delta_bias"]

                total_score += score

            # Normalize score
            avg_score = total_score / len(df_atm) if len(df_atm) > 0 else 0
            
            # Determine overall bias
            if avg_score >= 2:
                overall_bias = "Strong Bullish"
            elif avg_score >= 0.5:
                overall_bias = "Bullish"
            elif avg_score <= -2:
                overall_bias = "Strong Bearish"
            elif avg_score <= -0.5:
                overall_bias = "Bearish"
            else:
                overall_bias = "Neutral"

            return {
                'instrument': instrument,
                'spot_price': spot,
                'atm_strike': atm_strike,
                'overall_bias': overall_bias,
                'bias_score': avg_score,
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'pcr_change': abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change']
            }

        except Exception as e:
            return None

    def get_overall_market_bias(self):
        """Get overall market bias across all instruments"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            bias_data = self.analyze_atm_bias(instrument)
            if bias_data:
                results.append(bias_data)
        
        return results

# =============================================
# EXISTING DASHBOARD CLASSES (UPDATED)
# =============================================

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
        self.alert_manager = AlertManager(cooldown_minutes=10)
        self.options_analyzer = NSEOptionsAnalyzer()
        
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
        """Check if price is near volume order blocks and send alerts with ATM bias"""
        if not bullish_blocks and not bearish_blocks:
            return False
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        # Get current market bias
        market_bias = self.get_current_market_bias()
        
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

📊 OVERALL MARKET BIAS:
{market_bias}

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

📊 OVERALL MARKET BIAS:
{market_bias}

💡 Trading Suggestion:
Consider SHORT positions with stop above resistance

⏳ Next alert in 10 minutes

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.success(f"Bearish Volume Block alert sent!")
                    alert_sent = True
        
        return alert_sent

    def check_volume_spike_alerts(self, df):
        """Check for sudden volume spikes and send alerts with ATM bias"""
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
                # Get current market bias
                market_bias = self.get_current_market_bias()
                
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

📊 OVERALL MARKET BIAS:
{market_bias}

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

    def get_current_market_bias(self):
        """Get current market bias from options analyzer"""
        try:
            # Update market bias if needed
            current_time = datetime.now(self.ist)
            if (st.session_state.last_bias_update is None or 
                (current_time - st.session_state.last_bias_update).total_seconds() > 300):  # 5 minutes
                
                with st.spinner("Updating market bias..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = current_time
            
            bias_data = st.session_state.market_bias_data
            if not bias_data:
                return "Market bias data not available"
            
            # Format market bias message
            bias_message = "📊 OPTIONS MARKET BIAS ANALYSIS:\n\n"
            
            for instrument_data in bias_data:
                bias_message += f"• {instrument_data['instrument']}:\n"
                bias_message += f"  Spot: ₹{instrument_data['spot_price']:.2f}\n"
                bias_message += f"  Bias: {instrument_data['overall_bias']} (Score: {instrument_data['bias_score']:.2f})\n"
                bias_message += f"  PCR OI: {instrument_data['pcr_oi']:.2f} | PCR Δ: {instrument_data['pcr_change']:.2f}\n"
                bias_message += f"  CE OI: {instrument_data['total_ce_oi']:,} | PE OI: {instrument_data['total_pe_oi']:,}\n\n"
            
            return bias_message
            
        except Exception as e:
            return f"Market bias analysis temporarily unavailable"

    def display_options_analysis(self):
        """Display NSE Options Analysis"""
        st.header("📊 NSE Options Chain Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Real-time options market bias analysis across major indices")
        with col2:
            if st.button("🔄 Update Options Data", type="primary"):
                with st.spinner("Fetching latest options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    st.success("Options data updated!")
        
        st.divider()
        
        # Display current market bias
        if st.session_state.market_bias_data:
            bias_data = st.session_state.market_bias_data
            
            st.subheader("🎯 Current Market Bias")
            
            # Create metrics for each instrument
            cols = st.columns(len(bias_data))
            for idx, instrument_data in enumerate(bias_data):
                with cols[idx]:
                    bias_color = "🟢" if "Bullish" in instrument_data['overall_bias'] else "🔴" if "Bearish" in instrument_data['overall_bias'] else "🟡"
                    st.metric(
                        f"{instrument_data['instrument']}",
                        f"{bias_color} {instrument_data['overall_bias']}",
                        f"Score: {instrument_data['bias_score']:.2f}"
                    )
            
            st.divider()
            
            # Detailed analysis
            st.subheader("📈 Detailed Options Analysis")
            
            for instrument_data in bias_data:
                with st.expander(f"📊 {instrument_data['instrument']} Detailed Analysis"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                    with col2:
                        st.metric("ATM Strike", f"₹{instrument_data['atm_strike']:.2f}")
                    with col3:
                        st.metric("PCR (OI)", f"{instrument_data['pcr_oi']:.2f}")
                    with col4:
                        st.metric("PCR (Δ OI)", f"{instrument_data['pcr_change']:.2f}")
                    
                    # OI Analysis
                    st.markdown("#### 📊 Open Interest Analysis")
                    oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)
                    
                    with oi_col1:
                        st.metric("Total CE OI", f"{instrument_data['total_ce_oi']:,}")
                    with oi_col2:
                        st.metric("Total PE OI", f"{instrument_data['total_pe_oi']:,}")
                    with oi_col3:
                        st.metric("CE Δ OI", f"{instrument_data['total_ce_change']:,}")
                    with oi_col4:
                        st.metric("PE Δ OI", f"{instrument_data['total_pe_change']:,}")
                    
                    # Trading recommendation based on bias
                    st.markdown("#### 💡 Trading Recommendation")
                    if "Bullish" in instrument_data['overall_bias']:
                        st.success(f"""
                        **{instrument_data['instrument']} shows BULLISH bias**
                        
                        ✅ Consider LONG/CALL positions
                        ✅ Look for support levels to enter
                        ✅ Target resistance levels for exits
                        ⚠️ Use proper stop losses
                        """)
                    elif "Bearish" in instrument_data['overall_bias']:
                        st.error(f"""
                        **{instrument_data['instrument']} shows BEARISH bias**
                        
                        ✅ Consider SHORT/PUT positions  
                        ✅ Look for resistance levels to enter
                        ✅ Target support levels for exits
                        ⚠️ Use proper stop losses
                        """)
                    else:
                        st.warning(f"""
                        **{instrument_data['instrument']} shows NEUTRAL bias**
                        
                        🔄 Wait for clear directional bias
                        🔄 Consider range-bound strategies
                        🔄 Reduce position sizes
                        ⚠️ Monitor key levels closely
                        """)
        else:
            st.info("👆 Click 'Update Options Data' to load options chain analysis")
            
            st.markdown("""
            ### About Options Chain Analysis
            
            This section provides:
            
            - **Real-time market bias** across major indices
            - **PCR (Put-Call Ratio)** analysis
            - **Open Interest** buildup patterns
            - **Trading recommendations** based on options data
            
            **Instruments analyzed:**
            - NIFTY, BANKNIFTY, FINNIFTY
            
            **How to use:**
            1. Click 'Update Options Data' for latest analysis
            2. Review market bias for each instrument
            3. Check PCR ratios and OI patterns
            4. Use insights for trading decisions
            """)

    def create_comprehensive_chart(self, df, bullish_blocks, bearish_blocks, interval):
        """Create comprehensive chart with Volume Order Blocks"""
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Nifty 50 Analysis - {interval} Min', 'Volume with Spike Detection'),
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
        st.markdown("*Volume Analysis, Options Chain & Real-time Alerts*")
        
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
            telegram_enabled = st.checkbox("Enable Telegram", value=bool(self.telegram_bot_token))
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Update cooldown period
        self.alert_manager.cooldown_minutes = cooldown_minutes
        
        # Main content - Tabs
        tab1, tab2 = st.tabs(["📈 Price Analysis", "📊 Options Analysis"])
        
        with tab1:
            # Price Analysis Tab
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            # Update detectors
            self.vob_indicator = VolumeOrderBlocks(sensitivity=vob_sensitivity)
            self.volume_spike_detector.spike_threshold = spike_threshold
            
            # Fetch data
            df = pd.DataFrame()
            with st.spinner("Fetching market data..."):
                api_data = self.fetch_intraday_data(interval=timeframe)
                if api_data:
                    df = self.process_data(api_data)
            
            if not df.empty:
                latest = df.iloc[-1]
                current_price = latest['close']
                current_volume = latest['volume']
                
                # Detect Volume Order Blocks
                bullish_blocks, bearish_blocks = self.vob_indicator.detect_volume_order_blocks(df)
                
                # Calculate volume statistics
                if len(df) > 5:
                    avg_vol = df['volume'].iloc[-6:-1].mean()
                    volume_ratio = current_volume / avg_vol if avg_vol > 0 else 0
                else:
                    volume_ratio = 0
                
                # Display metrics
                with col1:
                    st.metric("Nifty Price", f"₹{current_price:.2f}")
                with col2:
                    st.metric("Current Volume", f"{current_volume:,}")
                with col3:
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
                with col4:
                    st.metric("Bullish Blocks", len(bullish_blocks))
                with col5:
                    st.metric("Bearish Blocks", len(bearish_blocks))
                with col6:
                    if volume_spike_alerts and telegram_enabled:
                        st.metric("Alerts Status", "✅ Active")
                    else:
                        st.metric("Alerts Status", "❌ Inactive")
                
                # Create and display chart
                chart = self.create_comprehensive_chart(df, bullish_blocks, bearish_blocks, timeframe)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Check for alerts
                alerts_sent = []
                if telegram_enabled:
                    if volume_block_alerts:
                        block_alert = self.check_volume_block_alerts(
                            current_price, bullish_blocks, bearish_blocks, alert_threshold
                        )
                        if block_alert:
                            alerts_sent.append("Volume Block")
                    
                    if volume_spike_alerts:
                        spike_alert = self.check_volume_spike_alerts(df)
                        if spike_alert:
                            alerts_sent.append("Volume Spike")
                
                if alerts_sent:
                    st.success(f"📱 Alerts sent: {', '.join(alerts_sent)} (Cooldown: {cooldown_minutes}min)")
                
                # Real-time volume monitoring
                st.subheader("🔍 Live Volume Monitoring")
                vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
                
                with vol_col1:
                    if len(df) > 5:
                        avg_vol = df['volume'].iloc[-6:-1].mean()
                        st.metric("20-period Avg Volume", f"{avg_vol:,.0f}")
                
                with vol_col2:
                    st.metric("Current Volume", f"{current_volume:,.0f}")
                
                with vol_col3:
                    if volume_ratio > spike_threshold:
                        st.error(f"Volume Spike: {volume_ratio:.1f}x")
                    elif volume_ratio > 1.5:
                        st.warning(f"High Volume: {volume_ratio:.1f}x")
                    else:
                        st.success(f"Normal: {volume_ratio:.1f}x")
                
                with vol_col4:
                    if len(df) > 1:
                        price_change = latest['close'] - df.iloc[-2]['close']
                        st.metric("Price Change", f"₹{price_change:+.2f}")
            
            else:
                st.error("No data available. Please check your API credentials and try again.")
        
        with tab2:
            # Options Analysis Tab
            self.display_options_analysis()
        
        # Cleanup old alerts and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
