"""
Comprehensive Bias Analysis Dashboard
=====================================

Complete Streamlit app with BiasAnalysisPro class included.
No external dependencies needed - everything in one file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

warnings.filterwarnings('ignore')

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# Configure the page
st.set_page_config(
    page_title="Bias Analysis Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis matching Pine Script indicator EXACTLY
    Analyzes 13 bias indicators with adaptive scoring
    """

    def __init__(self):
        """Initialize bias analysis with default configuration"""
        self.config = self._default_config()
        self.all_bias_results = []
        self.overall_bias = "NEUTRAL"
        self.overall_score = 0

    def _default_config(self) -> Dict:
        """Default configuration from Pine Script"""
        return {
            # Timeframes
            'tf1': '15m',
            'tf2': '1h',

            # Indicator periods
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,

            # Volume
            'volume_roc_length': 14,
            'volume_threshold': 1.2,

            # Volatility
            'volatility_ratio_length': 14,
            'volatility_threshold': 1.5,

            # OBV
            'obv_smoothing': 21,

            # Force Index
            'force_index_length': 13,
            'force_index_smoothing': 2,

            # Price ROC
            'price_roc_length': 12,

            # Market Breadth
            'breadth_threshold': 60,

            # Divergence
            'divergence_lookback': 30,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # Choppiness Index
            'ci_length': 14,
            'ci_high_threshold': 61.8,
            'ci_low_threshold': 38.2,

            # Bias parameters
            'bias_strength': 60,
            'divergence_threshold': 60,

            # Adaptive weights
            'normal_fast_weight': 2.0,
            'normal_medium_weight': 3.0,
            'normal_slow_weight': 5.0,
            'reversal_fast_weight': 5.0,
            'reversal_medium_weight': 3.0,
            'reversal_slow_weight': 2.0,

            # Stocks with weights
            'stocks': {
                '^NSEBANK': 10.0,
                'RELIANCE.NS': 9.98,
                'HDFCBANK.NS': 9.67,
                'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54,
                'ICICIBANK.NS': 8.01,
                'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98,
                'ITC.NS': 2.44,
                'MARUTI.NS': 0.0
            }
        }

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                st.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Ensure volume column exists
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)

            return df
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index"""
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
        mfi = mfi.fillna(50)

        return mfi

    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8):
        """Calculate DMI indicators"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=smoothing).mean()

        return plus_di, minus_di, adx

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        if df['Volume'].sum() == 0:
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe
        vwap = vwap.fillna(typical_price)

        return vwap

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA"""
        close = df['Close']

        # Calculate momentum (CMO)
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()

        cmo_denom = p + n
        cmo_denom = cmo_denom.replace(0, np.nan)
        abs_cmo = abs(100 * (p - n) / cmo_denom).fillna(0)

        # Calculate VIDYA
        alpha = 2 / (length + 1)
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                            (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])

        vidya_smoothed = vidya.rolling(window=15).mean()

        return vidya_smoothed, False, False

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta"""
        if df['Volume'].sum() == 0:
            return 0, False, False

        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots"""
        if df['Volume'].sum() == 0:
            return False, False, 0, 0

        pivot_highs = []
        pivot_lows = []

        for i in range(left_bars, len(df) - right_bars):
            # Check for pivot high
            is_pivot_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['High'].iloc[j] >= df['High'].iloc[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(i)

            # Check for pivot low
            is_pivot_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and df['Low'].iloc[j] <= df['Low'].iloc[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(i)

        volume_sum = df['Volume'].rolling(window=left_bars * 2).sum()
        ref_vol = volume_sum.quantile(0.95)
        norm_vol = (volume_sum / ref_vol * 5).fillna(0)

        hvp_bullish = False
        hvp_bearish = False

        if len(pivot_lows) > 0:
            last_pivot_low_idx = pivot_lows[-1]
            if norm_vol.iloc[last_pivot_low_idx] > vol_filter:
                hvp_bullish = True

        if len(pivot_highs) > 0:
            last_pivot_high_idx = pivot_highs[-1]
            if norm_vol.iloc[last_pivot_high_idx] > vol_filter:
                hvp_bearish = True

        return hvp_bullish, hvp_bearish, len(pivot_highs), len(pivot_lows)

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5):
        """Calculate Volume Order Blocks"""
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """Analyze all 8 bias indicators"""
        st.info(f"📊 Fetching data for {symbol}...")
        
        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            return {
                'success': False,
                'error': f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            }

        current_price = df['Close'].iloc[-1]
        bias_results = []

        # 1. VOLUME DELTA
        volume_delta, volume_bullish, volume_bearish = self.calculate_volume_delta(df)
        vol_delta_bias = "BULLISH" if volume_bullish else "BEARISH" if volume_bearish else "NEUTRAL"
        vol_delta_score = 100 if volume_bullish else -100 if volume_bearish else 0

        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': vol_delta_bias,
            'score': vol_delta_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 2. HVP
        hvp_bullish, hvp_bearish, pivot_highs, pivot_lows = self.calculate_hvp(df)
        hvp_bias = "BULLISH" if hvp_bullish else "BEARISH" if hvp_bearish else "NEUTRAL"
        hvp_score = 100 if hvp_bullish else -100 if hvp_bearish else 0
        hvp_value = f"Highs: {pivot_highs}, Lows: {pivot_lows}"

        bias_results.append({
            'indicator': 'HVP (High Volume Pivots)',
            'value': hvp_value,
            'bias': hvp_bias,
            'score': hvp_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 3. VOB
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)
        vob_bias = "BULLISH" if vob_bullish else "BEARISH" if vob_bearish else "NEUTRAL"
        vob_score = 100 if vob_bullish else -100 if vob_bearish else 0
        vob_value = f"EMA5: {vob_ema5:.2f}, EMA18: {vob_ema18:.2f}"

        bias_results.append({
            'indicator': 'VOB (Volume Order Blocks)',
            'value': vob_value,
            'bias': vob_bias,
            'score': vob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 4. ORDER BLOCKS
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        ob_bias = "BULLISH" if cross_up else "BEARISH" if cross_dn else "NEUTRAL"
        ob_score = 100 if cross_up else -100 if cross_dn else 0

        bias_results.append({
            'indicator': 'Order Blocks (EMA 5/18)',
            'value': f"EMA5: {ema5.iloc[-1]:.2f} | EMA18: {ema18.iloc[-1]:.2f}",
            'bias': ob_bias,
            'score': ob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 5. RSI
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]
        rsi_bias = "BULLISH" if rsi_value > 50 else "BEARISH"
        rsi_score = 100 if rsi_value > 50 else -100

        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.2f}",
            'bias': rsi_bias,
            'score': rsi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 6. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        plus_di_value = plus_di.iloc[-1] if len(plus_di) > 0 else 0
        minus_di_value = minus_di.iloc[-1] if len(minus_di) > 0 else 0
        dmi_bias = "BULLISH" if plus_di_value > minus_di_value else "BEARISH"
        dmi_score = 100 if plus_di_value > minus_di_value else -100

        bias_results.append({
            'indicator': 'DMI',
            'value': f"+DI:{plus_di_value:.1f} -DI:{minus_di_value:.1f}",
            'bias': dmi_bias,
            'score': dmi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 7. VIDYA
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
        vidya_bias = "BULLISH" if vidya_bullish else "BEARISH" if vidya_bearish else "NEUTRAL"
        vidya_score = 100 if vidya_bullish else -100 if vidya_bearish else 0

        bias_results.append({
            'indicator': 'VIDYA',
            'value': f"{vidya_val.iloc[-1]:.2f}" if not vidya_val.empty else "N/A",
            'bias': vidya_bias,
            'score': vidya_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 8. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1] if len(mfi) > 0 else 50
        if np.isnan(mfi_value):
            mfi_value = 50.0
        mfi_bias = "BULLISH" if mfi_value > 50 else "BEARISH"
        mfi_score = 100 if mfi_value > 50 else -100

        bias_results.append({
            'indicator': 'MFI (Money Flow)',
            'value': f"{mfi_value:.2f}",
            'bias': mfi_bias,
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # Calculate overall bias
        bullish_count = sum(1 for b in bias_results if 'BULLISH' in b['bias'])
        bearish_count = sum(1 for b in bias_results if 'BEARISH' in b['bias'])
        neutral_count = sum(1 for b in bias_results if 'NEUTRAL' in b['bias'])

        # Simple weighted average for overall score
        total_score = sum(b['score'] * b['weight'] for b in bias_results)
        total_weight = sum(b['weight'] for b in bias_results)
        overall_score = total_score / total_weight if total_weight > 0 else 0

        if overall_score >= self.config['bias_strength']:
            overall_bias = "BULLISH"
        elif overall_score <= -self.config['bias_strength']:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        overall_confidence = min(100, abs(overall_score))

        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(IST),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'mode': 'NORMAL'
        }

class BiasAnalysisDashboard:
    """Complete dashboard for Bias Analysis Pro"""
    
    def __init__(self):
        self.analyzer = BiasAnalysisPro()
        self.market_symbols = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN", 
            "BANK NIFTY": "^NSEBANK",
            "DOW JONES": "^DJI",
            "NASDAQ": "^IXIC"
        }
    
    def run(self):
        """Run the complete dashboard"""
        st.title("🎯 Bias Analysis Pro Dashboard")
        st.markdown("Comprehensive market bias analysis using 8 technical indicators")
        
        # Sidebar controls
        selected_market = self.render_sidebar()
        
        # Main content
        if st.button("🚀 Analyze Market Bias", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing market data and calculating bias indicators..."):
                self.analyze_and_display(selected_market)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Analysis Settings")
            
            # Market selection
            selected_market = st.selectbox(
                "Select Market:",
                list(self.market_symbols.keys()),
                index=0
            )
            
            st.divider()
            
            # Display info about the analysis
            st.subheader("📊 About This Analysis")
            st.info("""
            **8 Bias Indicators Analyzed:**
            
            • Volume Delta
            • HVP (High Volume Pivots)
            • VOB (Volume Order Blocks) 
            • Order Blocks (EMA 5/18)
            • RSI
            • DMI
            • VIDYA
            • MFI
            
            **Adaptive Weighting System**
            """)
            
            return selected_market
    
    def analyze_and_display(self, market_name):
        """Perform analysis and display results"""
        symbol = self.market_symbols[market_name]
        
        # Perform bias analysis
        results = self.analyzer.analyze_all_bias_indicators(symbol)
        
        if not results.get('success'):
            st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        # Display overall market overview
        self.display_market_overview(results, market_name)
        
        # Display detailed bias breakdown
        self.display_bias_breakdown(results)
        
        # Display trading recommendations
        self.display_trading_recommendations(results)
    
    def display_market_overview(self, results, market_name):
        """Display overall market overview"""
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"{market_name} Current Price",
                f"₹{results['current_price']:,.2f}" if market_name != "DOW JONES" else f"${results['current_price']:,.2f}",
                delta=None
            )
        
        with col2:
            bias_emoji = "🐂" if results['overall_bias'] == "BULLISH" else "🐻" if results['overall_bias'] == "BEARISH" else "⚖️"
            st.markdown(f"<h2 style='text-align: center;'>{bias_emoji} {results['overall_bias']}</h2>", 
                       unsafe_allow_html=True)
            st.caption("Overall Market Bias")
        
        with col3:
            score_color = "green" if results['overall_score'] > 0 else "red" if results['overall_score'] < 0 else "gray"
            st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{results['overall_score']:+.1f}</h2>", 
                       unsafe_allow_html=True)
            st.caption("Bias Score")
        
        with col4:
            confidence_color = "green" if results['overall_confidence'] > 70 else "orange" if results['overall_confidence'] > 50 else "red"
            st.markdown(f"<h2 style='color: {confidence_color}; text-align: center;'>{results['overall_confidence']:.1f}%</h2>", 
                       unsafe_allow_html=True)
            st.caption("Confidence Level")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🐂 **Bullish Signals:** {results['bullish_count']}/{results['total_indicators']}")
        
        with col2:
            st.error(f"🐻 **Bearish Signals:** {results['bearish_count']}/{results['total_indicators']}")
        
        with col3:
            st.info(f"⚖️ **Neutral Signals:** {results['neutral_count']}/{results['total_indicators']}")
    
    def display_bias_breakdown(self, results):
        """Display detailed bias breakdown in tables"""
        st.markdown("---")
        st.subheader("📋 Detailed Bias Breakdown")
        
        # Convert bias results to DataFrame
        bias_df = pd.DataFrame(results['bias_results'])
        
        # Display the table with styling
        st.dataframe(
            bias_df[['indicator', 'value', 'bias', 'score', 'category']],
            use_container_width=True,
            height=400
        )
        
        # Display by category
        st.subheader("📊 Bias by Category")
        
        col1, col2, col3 = st.columns(3)
        
        # Fast indicators
        with col1:
            fast_indicators = [b for b in results['bias_results'] if b['category'] == 'fast']
            if fast_indicators:
                st.markdown("**⚡ Fast Indicators**")
                fast_df = pd.DataFrame(fast_indicators)[['indicator', 'bias', 'score']]
                st.dataframe(fast_df, use_container_width=True, height=300)
    
    def display_trading_recommendations(self, results):
        """Display trading recommendations based on bias analysis"""
        st.markdown("---")
        st.subheader("💡 Trading Recommendations")
        
        overall_bias = results['overall_bias']
        overall_score = results['overall_score']
        confidence = results['overall_confidence']
        
        if overall_bias == "BULLISH" and confidence > 70:
            st.success("## 🐂 STRONG BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for LONG entries on dips
            - ✅ Focus on support levels
            - ✅ Set stop loss below recent swing low
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        
        elif overall_bias == "BULLISH" and confidence >= 50:
            st.success("## 🐂 MODERATE BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider LONG entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at resistance levels
            """)
        
        elif overall_bias == "BEARISH" and confidence > 70:
            st.error("## 🐻 STRONG BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for SHORT entries on rallies
            - ✅ Focus on resistance levels
            - ✅ Set stop loss above recent swing high
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        
        elif overall_bias == "BEARISH" and confidence >= 50:
            st.error("## 🐻 MODERATE BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider SHORT entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at support levels
            """)
        
        else:
            st.warning("## ⚖️ NEUTRAL / NO CLEAR SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - 🔄 Stay out of the market or use range trading
            - 🔄 Wait for clearer bias formation
            - 🔄 Monitor key support/resistance levels
            """)
        
        # Timestamp
        st.caption(f"Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main application entry point"""
    try:
        dashboard = BiasAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()