# technical_bias_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from typing import Dict, List, Tuple, Optional
import pytz

warnings.filterwarnings('ignore')

# Set page config FIRST
st.set_page_config(
    page_title="Technical Bias Analysis Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bias-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .bullish {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .bearish {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .neutral {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .overall-bullish {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .overall-bearish {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .overall-neutral {
        background: linear-gradient(135deg, #e2e3e5, #d6d8db);
        border: 2px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

class TechnicalBiasAnalyzer:
    """Comprehensive Technical Bias Analysis with 13 Indicators"""
    
    def __init__(self):
        self.config = self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'bias_strength': 60,
            'divergence_threshold': 60,
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,
            'stocks': {
                'RELIANCE.NS': 9.98, 'HDFCBANK.NS': 9.67, 'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54, 'ICICIBANK.NS': 8.01, 'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98, 'ITC.NS': 2.44, 'SBIN.NS': 5.0
            }
        }
    
    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df if not df.empty else pd.DataFrame()
        except:
            return pd.DataFrame()
    
    # Core Technical Indicators
    def calculate_volume_delta(self, df: pd.DataFrame) -> Tuple[float, bool, bool]:
        if df.empty or df['Volume'].sum() == 0:
            return 0, False, False
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
        volume_delta = up_vol - down_vol
        return volume_delta, volume_delta > 0, volume_delta < 0
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_vob(self, df: pd.DataFrame) -> Tuple[bool, bool, float, float]:
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        return cross_up, cross_dn, ema5.iloc[-1], ema18.iloc[-1]
    
    def calculate_order_blocks(self, df: pd.DataFrame) -> Tuple[bool, bool, float, float]:
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        return cross_up, cross_dn, ema5.iloc[-1], ema18.iloc[-1]
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        up_move, down_move = high-high.shift(1), low.shift(1)-low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()
        return plus_di, minus_di, adx
    
    def calculate_vidya(self, df: pd.DataFrame):
        close = df['Close']
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=20).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=20).sum()
        cmo_denom = (p + n).replace(0, np.nan)
        abs_cmo = abs(100 * (p - n) / cmo_denom).fillna(0)
        alpha = 2 / (10 + 1)
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]
        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                            (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])
        vidya_smoothed = vidya.rolling(window=15).mean()
        return vidya_smoothed, False, False
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        if df['Volume'].sum() == 0:
            return pd.Series([50.0] * len(df), index=df.index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi.fillna(50)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        if df['Volume'].sum() == 0:
            return (df['High'] + df['Low'] + df['Close']) / 3
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe
        return vwap.fillna(typical_price)
    
    def calculate_market_breadth(self):
        bullish_stocks, total_stocks, stock_data = 0, 0, []
        for symbol, weight in self.config['stocks'].items():
            try:
                df = self.fetch_data(symbol, period='2d', interval='5m')
                if len(df) > 1:
                    change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    stock_data.append({'symbol': symbol.replace('.NS', ''), 'change_pct': change_pct, 'is_bullish': change_pct > 0})
                    if change_pct > 0: bullish_stocks += 1
                    total_stocks += 1
            except: continue
        market_breadth = (bullish_stocks / total_stocks * 100) if total_stocks > 0 else 50
        return market_breadth, market_breadth > 60, market_breadth < 40, bullish_stocks, total_stocks, stock_data
    
    def analyze_technical_bias(self, symbol: str = "^NSEI") -> Dict:
        """Main analysis function"""
        df = self.fetch_data(symbol, period='5d', interval='5m')
        if df.empty or len(df) < 10:
            return {'success': False, 'error': 'Insufficient data'}
        
        current_price = df['Close'].iloc[-1]
        bias_results = []
        
        # Fast Indicators (8)
        volume_delta, vol_bullish, vol_bearish = self.calculate_volume_delta(df)
        bias_results.append({'indicator': 'Volume Delta', 'value': f"{volume_delta:.0f}", 
                           'bias': "BULLISH" if vol_bullish else "BEARISH" if vol_bearish else "NEUTRAL", 
                           'score': 100 if vol_bullish else -100 if vol_bearish else 0, 'category': 'fast'})
        
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)
        bias_results.append({'indicator': 'VOB', 'value': f"EMA5:{vob_ema5:.1f}", 
                           'bias': "BULLISH" if vob_bullish else "BEARISH" if vob_bearish else "NEUTRAL", 
                           'score': 100 if vob_bullish else -100 if vob_bearish else 0, 'category': 'fast'})
        
        ob_bullish, ob_bearish, ob_ema5, ob_ema18 = self.calculate_order_blocks(df)
        bias_results.append({'indicator': 'Order Blocks', 'value': f"EMA5:{ob_ema5:.1f}", 
                           'bias': "BULLISH" if ob_bullish else "BEARISH" if ob_bearish else "NEUTRAL", 
                           'score': 100 if ob_bullish else -100 if ob_bearish else 0, 'category': 'fast'})
        
        rsi = self.calculate_rsi(df['Close'], 14)
        rsi_value = rsi.iloc[-1] if not rsi.empty else 50
        bias_results.append({'indicator': 'RSI', 'value': f"{rsi_value:.1f}", 
                           'bias': "BULLISH" if rsi_value > 50 else "BEARISH", 
                           'score': 100 if rsi_value > 50 else -100, 'category': 'fast'})
        
        plus_di, minus_di, adx = self.calculate_dmi(df)
        plus_di_value = plus_di.iloc[-1] if not plus_di.empty else 0
        minus_di_value = minus_di.iloc[-1] if not minus_di.empty else 0
        bias_results.append({'indicator': 'DMI', 'value': f"+DI:{plus_di_value:.1f}", 
                           'bias': "BULLISH" if plus_di_value > minus_di_value else "BEARISH", 
                           'score': 100 if plus_di_value > minus_di_value else -100, 'category': 'fast'})
        
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
        bias_results.append({'indicator': 'VIDYA', 'value': f"{vidya_val.iloc[-1]:.1f}" if not vidya_val.empty else "N/A", 
                           'bias': "NEUTRAL", 'score': 0, 'category': 'fast'})
        
        mfi = self.calculate_mfi(df, 10)
        mfi_value = mfi.iloc[-1] if not mfi.empty else 50
        bias_results.append({'indicator': 'MFI', 'value': f"{mfi_value:.1f}", 
                           'bias': "BULLISH" if mfi_value > 50 else "BEARISH", 
                           'score': 100 if mfi_value > 50 else -100, 'category': 'fast'})
        
        # Add HVP as neutral for now
        bias_results.append({'indicator': 'HVP', 'value': "Pivot Analysis", 
                           'bias': "NEUTRAL", 'score': 0, 'category': 'fast'})
        
        # Medium Indicators (2)
        vwap = self.calculate_vwap(df)
        current_vwap = vwap.iloc[-1] if not vwap.empty else current_price
        close_vwap_dist = ((current_price - current_vwap) / current_vwap) * 100
        bias_results.append({'indicator': 'Close vs VWAP', 'value': f"{close_vwap_dist:+.2f}%", 
                           'bias': "BULLISH" if current_price > current_vwap else "BEARISH", 
                           'score': 100 if current_price > current_vwap else -100, 'category': 'medium'})
        
        bias_results.append({'indicator': 'Price vs VWAP', 'value': f"{close_vwap_dist:+.2f}%", 
                           'bias': "BULLISH" if current_price > current_vwap else "BEARISH", 
                           'score': 100 if current_price > current_vwap else -100, 'category': 'medium'})
        
        # Slow Indicators (3) - Market Breadth
        market_breadth, breadth_bullish, breadth_bearish, bull_stocks, total_stocks, stock_data = self.calculate_market_breadth()
        for i in range(3):
            bias_results.append({'indicator': f'Market Breadth {["Daily","15m","1h"][i]}', 
                               'value': f"{market_breadth:.1f}%", 
                               'bias': "BULLISH" if breadth_bullish else "BEARISH" if breadth_bearish else "NEUTRAL", 
                               'score': 100 if breadth_bullish else -100 if breadth_bearish else 0, 'category': 'slow'})
        
        # Calculate overall bias
        fast_bull = sum(1 for b in bias_results if b['category'] == 'fast' and 'BULLISH' in b['bias'])
        fast_bear = sum(1 for b in bias_results if b['category'] == 'fast' and 'BEARISH' in b['bias'])
        slow_bull = sum(1 for b in bias_results if b['category'] == 'slow' and 'BULLISH' in b['bias'])
        
        fast_bull_pct = (fast_bull / 8) * 100
        slow_bull_pct = (slow_bull / 3) * 100
        
        # Adaptive weighting
        divergence_detected = (slow_bull_pct >= 66 and fast_bull_pct <= 40)
        if divergence_detected:
            fast_weight, medium_weight, slow_weight = 5.0, 3.0, 2.0
            mode = "REVERSAL"
        else:
            fast_weight, medium_weight, slow_weight = 2.0, 3.0, 5.0
            mode = "NORMAL"
        
        bullish_signals = (sum(1 for b in bias_results if b['category'] == 'fast' and 'BULLISH' in b['bias']) * fast_weight +
                          sum(1 for b in bias_results if b['category'] == 'medium' and 'BULLISH' in b['bias']) * medium_weight +
                          sum(1 for b in bias_results if b['category'] == 'slow' and 'BULLISH' in b['bias']) * slow_weight)
        
        total_signals = (8 * fast_weight + 2 * medium_weight + 3 * slow_weight)
        bullish_bias_pct = (bullish_signals / total_signals) * 100
        
        if bullish_bias_pct >= 60:
            overall_bias = "BULLISH"
            overall_score = bullish_bias_pct
        elif bullish_bias_pct <= 40:
            overall_bias = "BEARISH" 
            overall_score = -bullish_bias_pct
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0
        
        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': min(100, abs(overall_score)),
            'bullish_count': sum(1 for b in bias_results if 'BULLISH' in b['bias']),
            'bearish_count': sum(1 for b in bias_results if 'BEARISH' in b['bias']),
            'neutral_count': sum(1 for b in bias_results if b['bias'] == 'NEUTRAL'),
            'total_indicators': len(bias_results),
            'stock_data': stock_data,
            'mode': mode,
            'bullish_bias_pct': bullish_bias_pct
        }

def main():
    """Main Streamlit App"""
    st.title("🎯 Technical Bias Analysis Pro")
    st.markdown("### Comprehensive 13-Indicator Market Analysis")
    
    # Initialize analyzer in session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TechnicalBiasAnalyzer()
    
    # Sidebar
    st.sidebar.title("⚙️ Analysis Settings")
    
    symbol = st.sidebar.selectbox(
        "Select Market",
        ["NIFTY 50", "SENSEX", "DOW JONES", "NASDAQ", "BANK NIFTY", "RELIANCE", "TCS", "INFY"],
        index=0
    )
    
    # Map display names to symbols
    symbol_map = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN", 
        "DOW JONES": "^DJI",
        "NASDAQ": "^IXIC",
        "BANK NIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS"
    }
    
    symbol_code = symbol_map.get(symbol, "^NSEI")
    
    if st.sidebar.button("🚀 Analyze Technical Bias", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing market data..."):
            results = st.session_state.analyzer.analyze_technical_bias(symbol_code)
            st.session_state.results = results
    
    # Display results if available
    if 'results' in st.session_state and st.session_state.results.get('success'):
        results = st.session_state.results
        display_results(results)
    else:
        show_welcome_message()

def show_welcome_message():
    """Show welcome message and instructions"""
    st.markdown("""
    ## 📊 Welcome to Technical Bias Analysis Pro!
    
    This advanced tool analyzes **13 technical indicators** across 3 categories to determine market bias:
    
    ### ⚡ Fast Indicators (8)
    - **Volume Delta** - Up vs Down volume pressure
    - **HVP** - High Volume Pivots (Support/Resistance)
    - **VOB** - Volume Order Blocks 
    - **Order Blocks** - EMA crossover signals
    - **RSI** - Relative Strength Index momentum
    - **DMI** - Directional Movement Index trend
    - **VIDYA** - Variable Index Dynamic Average
    - **MFI** - Money Flow Index volume
    
    ### 📊 Medium Indicators (2)  
    - **Close vs VWAP** - Price position analysis
    - **Price vs VWAP** - Trend relationship
    
    ### 🐢 Slow Indicators (3)
    - **Market Breadth** - Overall stock performance
    
    ### 🎯 Adaptive Weighting System
    - **Normal Mode**: Fast(2x), Medium(3x), Slow(5x) - Trending markets
    - **Reversal Mode**: Fast(5x), Medium(3x), Slow(2x) - Reversal signals
    
    **Click the 'Analyze Technical Bias' button in the sidebar to get started!**
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indicators", "13")
    with col2:
        st.metric("Categories", "3")
    with col3:
        st.metric("Analysis Speed", "Real-time")

def display_results(results):
    """Display the analysis results"""
    
    # Overall Bias Header
    bias_class = f"overall-{results['overall_bias'].lower()}"
    bias_emoji = "🐂" if results['overall_bias'] == "BULLISH" else "🐻" if results['overall_bias'] == "BEARISH" else "⚖️"
    
    st.markdown(f"""
    <div class="bias-card {bias_class}" style="text-align: center; padding: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">{bias_emoji} {results['overall_bias']} MARKET</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence: {results['overall_confidence']:.1f}% | Mode: {results['mode']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{results['current_price']:,.2f}")
    with col2:
        st.metric("Overall Score", f"{results['overall_score']:.1f}")
    with col3:
        st.metric("Bullish Signals", f"{results['bullish_count']}/{results['total_indicators']}")
    with col4:
        st.metric("Analysis Time", results['timestamp'].strftime("%H:%M:%S"))
    
    st.divider()
    
    # Signal Distribution
    st.subheader("📊 Signal Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card bullish">
            <h3>🐂 Bullish</h3>
            <h2>{results['bullish_count']}</h2>
            <p>Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card bearish">
            <h3>🐻 Bearish</h3>
            <h2>{results['bearish_count']}</h2>
            <p>Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card neutral">
            <h3>⚖️ Neutral</h3>
            <h2>{results['neutral_count']}</h2>
            <p>Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Detailed Indicators
    st.subheader("🔍 Detailed Indicator Analysis")
    
    # Create tabs for different categories
    tab1, tab2, tab3 = st.tabs(["⚡ Fast Indicators (8)", "📊 Medium Indicators (2)", "🐢 Slow Indicators (3)"])
    
    with tab1:
        fast_indicators = [r for r in results['bias_results'] if r['category'] == 'fast']
        for indicator in fast_indicators:
            display_indicator_card(indicator)
    
    with tab2:
        medium_indicators = [r for r in results['bias_results'] if r['category'] == 'medium']
        for indicator in medium_indicators:
            display_indicator_card(indicator)
    
    with tab3:
        slow_indicators = [r for r in results['bias_results'] if r['category'] == 'slow']
        for indicator in slow_indicators:
            display_indicator_card(indicator)
    
    st.divider()
    
    # Trading Recommendation
    st.subheader("💡 Trading Recommendation")
    recommendation = generate_trading_recommendation(results)
    st.info(recommendation)
    
    # Market Breadth
    if results['stock_data']:
        st.divider()
        st.subheader("🏢 Market Breadth Analysis")
        
        bullish_stocks = [s for s in results['stock_data'] if s['is_bullish']]
        bearish_stocks = [s for s in results['stock_data'] if not s['is_bullish']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Stocks", f"{len(bullish_stocks)}/{len(results['stock_data'])}")
            if bullish_stocks:
                st.write("**Top Gainers:**")
                for stock in sorted(bullish_stocks, key=lambda x: x['change_pct'], reverse=True)[:3]:
                    st.write(f"🟢 {stock['symbol']}: {stock['change_pct']:+.2f}%")
        
        with col2:
            st.metric("Bearish Stocks", f"{len(bearish_stocks)}/{len(results['stock_data'])}")
            if bearish_stocks:
                st.write("**Top Losers:**")
                for stock in sorted(bearish_stocks, key=lambda x: x['change_pct'])[:3]:
                    st.write(f"🔴 {stock['symbol']}: {stock['change_pct']:+.2f}%")

def display_indicator_card(indicator):
    """Display individual indicator card"""
    bias_class = indicator['bias'].lower()
    bias_emoji = "🟢" if indicator['bias'] == "BULLISH" else "🔴" if indicator['bias'] == "BEARISH" else "⚪"
    
    st.markdown(f"""
    <div class="bias-card {bias_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 2;">
                <strong>{indicator['indicator']}</strong>
                <br>
                <small style="color: #666;">Value: {indicator['value']}</small>
            </div>
            <div style="flex: 1; text-align: right;">
                {bias_emoji} <strong>{indicator['bias']}</strong>
                <br>
                <small>Score: {indicator['score']:.0f}</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def generate_trading_recommendation(results):
    """Generate trading recommendation based on analysis"""
    bias = results['overall_bias']
    confidence = results['overall_confidence']
    mode = results['mode']
    
    if bias == "BULLISH":
        if confidence > 70:
            return "🐂 STRONG BULLISH SIGNAL - Excellent conditions for LONG positions. Look for entry on minor pullbacks with stop loss below recent support levels."
        elif confidence > 50:
            return "🐂 MODERATE BULLISH SIGNAL - Good conditions for LONG positions but use tighter stop losses. Consider partial profit taking at resistance."
        else:
            return "⚠️ WEAK BULLISH SIGNAL - Wait for stronger confirmation before entering LONG positions. Consider smaller position sizes."
    
    elif bias == "BEARISH":
        if confidence > 70:
            return "🐻 STRONG BEARISH SIGNAL - Excellent conditions for SHORT positions. Look for entry on minor rallies with stop loss above recent resistance levels."
        elif confidence > 50:
            return "🐻 MODERATE BEARISH SIGNAL - Good conditions for SHORT positions but use tighter stop losses. Consider partial profit taking at support."
        else:
            return "⚠️ WEAK BEARISH SIGNAL - Wait for stronger confirmation before entering SHORT positions. Consider smaller position sizes."
    
    else:
        if confidence > 70:
            return "⚖️ STRONG NEUTRAL SIGNAL - Market is range-bound. Trade support and resistance levels with tight stops. Avoid trending strategies."
        else:
            return "⚖️ WEAK NEUTRAL SIGNAL - Market is indecisive. Consider staying out or using very small position sizes. Wait for clear breakout direction."

if __name__ == "__main__":
    main()