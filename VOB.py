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
        return atr * 3  # 3x ATR as in original script
    
    def detect_volume_order_blocks(self, df):
        """Detect Volume Order Blocks based on the Pine Script logic"""
        if len(df) < self.length2:
            return [], []
        
        # Calculate EMAs
        ema1 = self.calculate_ema(df['close'], self.length1)
        ema2 = self.calculate_ema(df['close'], self.length2)
        
        # Calculate crossovers
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        atr1 = atr * 2 / 3  # 2x ATR as in original script
        
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(len(df)):
            if cross_up.iloc[i]:
                # Bullish block detection
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                lowest_idx = lookback_data['low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'low']
                
                # Calculate volume for the block
                vol = lookback_data['volume'].sum()
                
                # Determine base price
                open_price = lookback_data.loc[lowest_idx, 'open']
                close_price = lookback_data.loc[lowest_idx, 'close']
                src = min(open_price, close_price)
                
                # Adjust base price if needed (ATR condition)
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
                # Bearish block detection
                lookback_data = df.iloc[max(0, i - self.length2):i+1]
                if len(lookback_data) == 0:
                    continue
                    
                highest_idx = lookback_data['high'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'high']
                
                # Calculate volume for the block
                vol = lookback_data['volume'].sum()
                
                # Determine base price
                open_price = lookback_data.loc[highest_idx, 'open']
                close_price = lookback_data.loc[highest_idx, 'close']
                src = max(open_price, close_price)
                
                # Adjust base price if needed (ATR condition)
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
        
        # Filter overlapping blocks
        bullish_blocks = self.filter_overlapping_blocks(bullish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        bearish_blocks = self.filter_overlapping_blocks(bearish_blocks, atr.iloc[-1] if len(atr) > 0 else 0)
        
        return bullish_blocks, bearish_blocks
    
    def filter_overlapping_blocks(self, blocks, atr_value):
        """Filter out overlapping blocks based on ATR distance"""
        if not blocks:
            return []
        
        filtered_blocks = []
        
        for block in blocks:
            # Check if this block overlaps with any existing block
            overlap = False
            for existing_block in filtered_blocks:
                if abs(block['mid'] - existing_block['mid']) < atr_value:
                    overlap = True
                    break
            
            if not overlap:
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def check_price_near_blocks(self, current_price, blocks, threshold=5):
        """Check if current price is within threshold points of any volume order block"""
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

class EnhancedNiftyApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"
        self.vob_indicator = VolumeOrderBlocks(sensitivity=5)
        
        # Initialize session state
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'sent_rsi_alerts' not in st.session_state:
            st.session_state.sent_rsi_alerts = set()
        if 'sent_rsi_oi_alerts' not in st.session_state:
            st.session_state.sent_rsi_oi_alerts = set()
        if 'sent_volume_block_alerts' not in st.session_state:
            st.session_state.sent_volume_block_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        
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
            return
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        # Check bullish blocks
        nearby_bullish = self.vob_indicator.check_price_near_blocks(current_price, bullish_blocks, threshold)
        for block in nearby_bullish:
            alert_id = f"vol_block_bullish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if alert_id not in st.session_state.sent_volume_block_alerts:
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

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.session_state.sent_volume_block_alerts.add(alert_id)
                    st.success(f"Bullish Volume Block alert sent!")
                    alert_sent = True
        
        # Check bearish blocks
        nearby_bearish = self.vob_indicator.check_price_near_blocks(current_price, bearish_blocks, threshold)
        for block in nearby_bearish:
            alert_id = f"vol_block_bearish_{block['index'].strftime('%Y%m%d_%H%M')}"
            
            if alert_id not in st.session_state.sent_volume_block_alerts:
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

⚠️ Trade at your own risk!"""
                
                if self.send_telegram_message(message):
                    st.session_state.sent_volume_block_alerts.add(alert_id)
                    st.success(f"Bearish Volume Block alert sent!")
                    alert_sent = True
        
        # Clean up old alerts
        if len(st.session_state.sent_volume_block_alerts) > 50:
            alerts_list = list(st.session_state.sent_volume_block_alerts)
            st.session_state.sent_volume_block_alerts = set(alerts_list[-25:])
        
        return alert_sent

    def create_volume_blocks_chart(self, df, bullish_blocks, bearish_blocks, interval):
        """Create chart with Volume Order Blocks"""
        if df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'Nifty 50 with Volume Order Blocks - {interval} Min', 'Volume'),
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
        
        # Bullish blocks (Support zones)
        for block in bullish_blocks:
            # Upper line (base)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['upper'],
                line=dict(color=colors['bullish'], width=2),
                row=1, col=1
            )
            
            # Lower line (support)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['lower'],
                x1=df.index[-1], y1=block['lower'],
                line=dict(color=colors['bullish'], width=2),
                row=1, col=1
            )
            
            # Mid line (dashed)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['mid'],
                x1=df.index[-1], y1=block['mid'],
                line=dict(color=colors['bullish'], width=1, dash='dash'),
                row=1, col=1
            )
            
            # Fill between upper and lower
            fig.add_trace(
                go.Scatter(
                    x=[block['index'], block['index'], df.index[-1], df.index[-1]],
                    y=[block['upper'], block['lower'], block['lower'], block['upper']],
                    fill="toself",
                    fillcolor='rgba(38, 186, 159, 0.1)',
                    line=dict(width=0),
                    showlegend=False,
                    name=f"Bullish Block {block['index'].strftime('%H:%M')}"
                ),
                row=1, col=1
            )
            
            # Volume label
            volume_pct = (block['volume'] / sum(b['volume'] for b in bullish_blocks)) * 100 if bullish_blocks else 0
            fig.add_annotation(
                x=df.index[-1],
                y=block['mid'],
                text=f"⮜ {volume_pct:.1f}% ({block['volume']:,})",
                showarrow=False,
                xanchor='left',
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
                row=1, col=1
            )
        
        # Bearish blocks (Resistance zones)
        for block in bearish_blocks:
            # Upper line (resistance)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['upper'],
                x1=df.index[-1], y1=block['upper'],
                line=dict(color=colors['bearish'], width=2),
                row=1, col=1
            )
            
            # Lower line (base)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['lower'],
                x1=df.index[-1], y1=block['lower'],
                line=dict(color=colors['bearish'], width=2),
                row=1, col=1
            )
            
            # Mid line (dashed)
            fig.add_shape(
                type="line",
                x0=block['index'], y0=block['mid'],
                x1=df.index[-1], y1=block['mid'],
                line=dict(color=colors['bearish'], width=1, dash='dash'),
                row=1, col=1
            )
            
            # Fill between upper and lower
            fig.add_trace(
                go.Scatter(
                    x=[block['index'], block['index'], df.index[-1], df.index[-1]],
                    y=[block['upper'], block['lower'], block['lower'], block['upper']],
                    fill="toself",
                    fillcolor='rgba(102, 38, 186, 0.1)',
                    line=dict(width=0),
                    showlegend=False,
                    name=f"Bearish Block {block['index'].strftime('%H:%M')}"
                ),
                row=1, col=1
            )
            
            # Volume label
            volume_pct = (block['volume'] / sum(b['volume'] for b in bearish_blocks)) * 100 if bearish_blocks else 0
            fig.add_annotation(
                x=df.index[-1],
                y=block['mid'],
                text=f"⮜ {volume_pct:.1f}% ({block['volume']:,})",
                showarrow=False,
                xanchor='left',
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
                row=1, col=1
            )
        
        # Volume bars
        bar_colors = ['#00ff88' if close >= open else '#ff4444' 
                     for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=bar_colors,
                opacity=0.6
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
        st.title("📈 Enhanced Nifty Trading Dashboard")
        st.markdown("*With Volume Order Blocks, VOB Zones & OI Analysis*")
        
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
            show_mid_lines = st.checkbox("Show Mid Lines", value=True)
            
            st.subheader("Alerts")
            volume_block_alerts = st.checkbox("Volume Block Alerts", value=True)
            telegram_enabled = st.checkbox("Enable Telegram", value=bool(self.telegram_bot_token))
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Main content
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Update indicator sensitivity
        self.vob_indicator = VolumeOrderBlocks(sensitivity=vob_sensitivity)
        
        # Fetch data
        df = pd.DataFrame()
        with st.spinner("Fetching market data..."):
            api_data = self.fetch_intraday_data(interval=timeframe)
            if api_data:
                df = self.process_data(api_data)
        
        if not df.empty:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Detect Volume Order Blocks
            bullish_blocks, bearish_blocks = self.vob_indicator.detect_volume_order_blocks(df)
            
            # Display metrics
            with col1:
                st.metric("Nifty Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("Bullish Blocks", len(bullish_blocks))
            with col3:
                st.metric("Bearish Blocks", len(bearish_blocks))
            with col4:
                total_volume = sum(b['volume'] for b in bullish_blocks) + sum(b['volume'] for b in bearish_blocks)
                st.metric("Total Block Volume", f"{total_volume:,}")
            with col5:
                if volume_block_alerts and telegram_enabled:
                    st.metric("Block Alerts", "✅ Active")
                else:
                    st.metric("Alerts", "❌ Inactive")
            
            # Create and display chart
            chart = self.create_volume_blocks_chart(df, bullish_blocks, bearish_blocks, timeframe)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Check for alerts
            if volume_block_alerts and telegram_enabled and not df.empty:
                alert_sent = self.check_volume_block_alerts(
                    current_price, bullish_blocks, bearish_blocks, alert_threshold
                )
                if alert_sent:
                    st.info("📱 Volume Order Block alert sent to Telegram!")
            
            # Display blocks information
            st.subheader("🎯 Volume Order Blocks Summary")
            
            if bullish_blocks or bearish_blocks:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🟢 Bullish Blocks (Support)**")
                    for i, block in enumerate(bullish_blocks[-5:]):  # Show last 5
                        st.write(f"""
                        **Block {i+1}** - {block['index'].strftime('%H:%M')}
                        - Upper: ₹{block['upper']:.2f}
                        - Lower: ₹{block['lower']:.2f}
                        - Mid: ₹{block['mid']:.2f}
                        - Volume: {block['volume']:,}
                        - Distance: {abs(current_price - block['mid']):.2f} points
                        """)
                
                with col2:
                    st.write("**🔴 Bearish Blocks (Resistance)**")
                    for i, block in enumerate(bearish_blocks[-5:]):  # Show last 5
                        st.write(f"""
                        **Block {i+1}** - {block['index'].strftime('%H:%M')}
                        - Upper: ₹{block['upper']:.2f}
                        - Lower: ₹{block['lower']:.2f}
                        - Mid: ₹{block['mid']:.2f}
                        - Volume: {block['volume']:,}
                        - Distance: {abs(current_price - block['mid']):.2f} points
                        """)
            else:
                st.info("No Volume Order Blocks detected yet. The indicator needs more data to identify blocks.")
            
            # Data table
            with st.expander("📈 View Raw Data"):
                st.dataframe(df.tail(20))
        
        else:
            st.error("No data available. Please check your API credentials and try again.")
        
        # Auto refresh
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()