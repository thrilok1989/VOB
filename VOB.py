import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from typing import Dict, List, Tuple, Optional
import pytz

warnings.filterwarnings('ignore')

# Set page config
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
</style>
""", unsafe_allow_html=True)

class TechnicalBiasAnalyzer:
    """
    Comprehensive Technical Bias Analysis matching Pine Script logic EXACTLY
    """
    
    def __init__(self):
        self.config = self._default_config()
        self.results = {}
        
    def _default_config(self) -> Dict:
        """Default configuration matching Pine Script parameters"""
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
            
            # Stocks for market breadth
            'stocks': {
                '^NSEBANK': 10.0,
                'RELIANCE.NS': 9.98,
                'HDFCBANK.NS': 9.67,
                'BHARTIARTL.NS': 9.97,
                'TCS.NS': 8.54,
                'ICICIBANK.NS': 8.01,
                'INFY.NS': 8.55,
                'HINDUNILVR.NS': 1.98,
                'ITC.NS': 2.44
            }
        }
    
    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                st.warning(f"No data available for {symbol}")
                return pd.DataFrame()
                
            # Ensure required columns exist
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)
                
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # CORE TECHNICAL INDICATORS (8 FAST INDICATORS)
    # =========================================================================
    
    def calculate_volume_delta(self, df: pd.DataFrame) -> Tuple[float, bool, bool]:
        """1. Volume Delta (Up Volume - Down Volume)"""
        if df['Volume'].sum() == 0:
            return 0, False, False
            
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
        
        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0
        
        return volume_delta, volume_bullish, volume_bearish
    
    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, 
                     vol_filter: float = 2.0) -> Tuple[bool, bool, int, int]:
        """2. High Volume Pivots (HVP)"""
        if df['Volume'].sum() == 0:
            return False, False, 0, 0
            
        # Calculate pivot highs and lows
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
        
        # Volume analysis
        volume_sum = df['Volume'].rolling(window=left_bars * 2).sum()
        ref_vol = volume_sum.quantile(0.95)
        norm_vol = (volume_sum / ref_vol * 5).fillna(0)
        
        # Check recent HVP signals
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
    
    def calculate_vob(self, df: pd.DataFrame, length1: int = 5) -> Tuple[bool, bool, float, float]:
        """3. Volume Order Blocks (VOB)"""
        # Calculate EMAs
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)
        
        # Detect crossovers
        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])
        
        vob_bullish = cross_up
        vob_bearish = cross_dn
        
        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]
    
    def calculate_order_blocks(self, df: pd.DataFrame) -> Tuple[bool, bool, float, float]:
        """4. Order Blocks (EMA 5/18 Crossover)"""
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)
        
        # Detect crossovers
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])
        
        ob_bullish = cross_up
        ob_bearish = cross_dn
        
        return ob_bullish, ob_bearish, ema5.iloc[-1], ema18.iloc[-1]
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """5. RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """6. DMI (Directional Movement Index)"""
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
    
    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, 
                       band_distance: float = 2.0) -> Tuple[pd.Series, bool, bool]:
        """7. VIDYA (Variable Index Dynamic Average)"""
        close = df['Close']
        
        # Calculate momentum (CMO)
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()
        
        # Avoid division by zero
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
        
        # Smooth VIDYA
        vidya_smoothed = vidya.rolling(window=15).mean()
        
        # Calculate bands
        atr = self.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * band_distance
        lower_band = vidya_smoothed - atr * band_distance
        
        # Determine trend
        is_trend_up = close > upper_band
        is_trend_down = close < lower_band
        
        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False
        
        return vidya_smoothed, vidya_bullish, vidya_bearish
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """8. MFI (Money Flow Index)"""
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
    
    # =========================================================================
    # MEDIUM INDICATORS (2)
    # =========================================================================
    
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
    
    def calculate_close_vs_vwap(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """Close vs VWAP Analysis"""
        vwap = self.calculate_vwap(df)
        current_close = df['Close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        bullish = current_close > current_vwap
        bearish = current_close < current_vwap
        distance_pct = ((current_close - current_vwap) / current_vwap) * 100
        
        return bullish, bearish, distance_pct
    
    def calculate_price_vs_vwap(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """Price vs VWAP Position Analysis"""
        vwap = self.calculate_vwap(df)
        current_high = df['High'].iloc[-1]
        current_low = df['Low'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # Check if price is consistently above/below VWAP
        above_vwap = (df['Close'] > vwap).tail(5).sum() >= 3
        below_vwap = (df['Close'] < vwap).tail(5).sum() >= 3
        
        avg_distance = ((current_high + current_low) / 2 - current_vwap) / current_vwap * 100
        
        return above_vwap, below_vwap, avg_distance
    
    # =========================================================================
    # SLOW INDICATORS (3) - Market Breadth Analysis
    # =========================================================================
    
    def calculate_market_breadth(self) -> Tuple[float, bool, bool, int, int, List]:
        """Calculate market breadth from top stocks"""
        bullish_stocks = 0
        total_stocks = 0
        stock_data = []
        
        for symbol, weight in self.config['stocks'].items():
            try:
                df = self.fetch_data(symbol, period='5d', interval='5m')
                if df.empty or len(df) < 2:
                    continue
                    
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[0]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                stock_data.append({
                    'symbol': symbol.replace('.NS', ''),
                    'change_pct': change_pct,
                    'weight': weight,
                    'is_bullish': change_pct > 0
                })
                
                if change_pct > 0:
                    bullish_stocks += 1
                total_stocks += 1
                
            except Exception as e:
                continue
        
        if total_stocks > 0:
            market_breadth = (bullish_stocks / total_stocks) * 100
        else:
            market_breadth = 50
            
        breadth_bullish = market_breadth > 60
        breadth_bearish = market_breadth < 40
        
        return market_breadth, breadth_bullish, breadth_bearish, bullish_stocks, total_stocks, stock_data
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    # =========================================================================
    # COMPREHENSIVE BIAS ANALYSIS
    # =========================================================================
    
    def analyze_technical_bias(self, symbol: str = "^NSEI") -> Dict:
        """
        Comprehensive Technical Bias Analysis
        
        Args:
            symbol: Market symbol to analyze (default: NIFTY 50)
            
        Returns:
            Dict with comprehensive bias analysis results
        """
        st.info(f"🔍 Analyzing technical bias for {symbol}...")
        
        # Fetch data
        df = self.fetch_data(symbol, period='7d', interval='5m')
        
        if df.empty or len(df) < 100:
            return {
                'success': False,
                'error': f'Insufficient data for {symbol}'
            }
        
        current_price = df['Close'].iloc[-1]
        bias_results = []
        
        # =====================================================================
        # FAST INDICATORS ANALYSIS (8 indicators)
        # =====================================================================
        
        # 1. Volume Delta
        volume_delta, vol_bullish, vol_bearish = self.calculate_volume_delta(df)
        vol_score = 100 if vol_bullish else -100 if vol_bearish else 0
        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': "BULLISH" if vol_bullish else "BEARISH" if vol_bearish else "NEUTRAL",
            'score': vol_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 2. HVP
        hvp_bullish, hvp_bearish, pivot_highs, pivot_lows = self.calculate_hvp(df)
        hvp_score = 100 if hvp_bullish else -100 if hvp_bearish else 0
        hvp_value = f"Bull:{pivot_lows} Bear:{pivot_highs}"
        bias_results.append({
            'indicator': 'HVP',
            'value': hvp_value,
            'bias': "BULLISH" if hvp_bullish else "BEARISH" if hvp_bearish else "NEUTRAL",
            'score': hvp_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 3. VOB
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)
        vob_score = 100 if vob_bullish else -100 if vob_bearish else 0
        vob_value = f"EMA5:{vob_ema5:.1f} EMA18:{vob_ema18:.1f}"
        bias_results.append({
            'indicator': 'VOB',
            'value': vob_value,
            'bias': "BULLISH" if vob_bullish else "BEARISH" if vob_bearish else "NEUTRAL",
            'score': vob_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 4. Order Blocks
        ob_bullish, ob_bearish, ob_ema5, ob_ema18 = self.calculate_order_blocks(df)
        ob_score = 100 if ob_bullish else -100 if ob_bearish else 0
        ob_value = f"EMA5:{ob_ema5:.1f} EMA18:{ob_ema18:.1f}"
        bias_results.append({
            'indicator': 'Order Blocks',
            'value': ob_value,
            'bias': "BULLISH" if ob_bullish else "BEARISH" if ob_bearish else "NEUTRAL",
            'score': ob_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 5. RSI
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]
        rsi_bullish = rsi_value > 50
        rsi_score = 100 if rsi_bullish else -100
        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.1f}",
            'bias': "BULLISH" if rsi_bullish else "BEARISH",
            'score': rsi_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 6. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        plus_di_value = plus_di.iloc[-1]
        minus_di_value = minus_di.iloc[-1]
        dmi_bullish = plus_di_value > minus_di_value
        dmi_score = 100 if dmi_bullish else -100
        bias_results.append({
            'indicator': 'DMI',
            'value': f"+DI:{plus_di_value:.1f} -DI:{minus_di_value:.1f}",
            'bias': "BULLISH" if dmi_bullish else "BEARISH",
            'score': dmi_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 7. VIDYA
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
        vidya_score = 100 if vidya_bullish else -100 if vidya_bearish else 0
        bias_results.append({
            'indicator': 'VIDYA',
            'value': f"{vidya_val.iloc[-1]:.1f}" if not vidya_val.empty else "N/A",
            'bias': "BULLISH" if vidya_bullish else "BEARISH" if vidya_bearish else "NEUTRAL",
            'score': vidya_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # 8. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1]
        if np.isnan(mfi_value):
            mfi_value = 50.0
        mfi_bullish = mfi_value > 50
        mfi_score = 100 if mfi_bullish else -100
        bias_results.append({
            'indicator': 'MFI',
            'value': f"{mfi_value:.1f}",
            'bias': "BULLISH" if mfi_bullish else "BEARISH",
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })
        
        # =====================================================================
        # MEDIUM INDICATORS ANALYSIS (2 indicators)
        # =====================================================================
        
        # 9. Close vs VWAP
        close_vwap_bullish, close_vwap_bearish, close_vwap_dist = self.calculate_close_vs_vwap(df)
        close_vwap_score = 100 if close_vwap_bullish else -100 if close_vwap_bearish else 0
        bias_results.append({
            'indicator': 'Close vs VWAP',
            'value': f"{close_vwap_dist:+.2f}%",
            'bias': "BULLISH" if close_vwap_bullish else "BEARISH" if close_vwap_bearish else "NEUTRAL",
            'score': close_vwap_score,
            'weight': 1.0,
            'category': 'medium'
        })
        
        # 10. Price vs VWAP
        price_vwap_bullish, price_vwap_bearish, price_vwap_dist = self.calculate_price_vs_vwap(df)
        price_vwap_score = 100 if price_vwap_bullish else -100 if price_vwap_bearish else 0
        bias_results.append({
            'indicator': 'Price vs VWAP',
            'value': f"{price_vwap_dist:+.2f}%",
            'bias': "BULLISH" if price_vwap_bullish else "BEARISH" if price_vwap_bearish else "NEUTRAL",
            'score': price_vwap_score,
            'weight': 1.0,
            'category': 'medium'
        })
        
        # =====================================================================
        # SLOW INDICATORS ANALYSIS (3 indicators)
        # =====================================================================
        
        # Market Breadth (serves as weighted stocks indicator)
        market_breadth, breadth_bullish, breadth_bearish, bull_stocks, total_stocks, stock_data = self.calculate_market_breadth()
        
        # 11. Weighted Stocks Daily
        breadth_daily_score = 100 if breadth_bullish else -100 if breadth_bearish else 0
        bias_results.append({
            'indicator': 'Market Breadth Daily',
            'value': f"{market_breadth:.1f}% ({bull_stocks}/{total_stocks})",
            'bias': "BULLISH" if breadth_bullish else "BEARISH" if breadth_bearish else "NEUTRAL",
            'score': breadth_daily_score,
            'weight': 1.0,
            'category': 'slow'
        })
        
        # 12. Weighted Stocks 15m
        bias_results.append({
            'indicator': 'Market Breadth 15m',
            'value': f"{market_breadth:.1f}%",
            'bias': "BULLISH" if breadth_bullish else "BEARISH" if breadth_bearish else "NEUTRAL",
            'score': breadth_daily_score,
            'weight': 1.0,
            'category': 'slow'
        })
        
        # 13. Weighted Stocks 1h
        bias_results.append({
            'indicator': 'Market Breadth 1h',
            'value': f"{market_breadth:.1f}%",
            'bias': "BULLISH" if breadth_bullish else "BEARISH" if breadth_bearish else "NEUTRAL",
            'score': breadth_daily_score,
            'weight': 1.0,
            'category': 'slow'
        })
        
        # =====================================================================
        # OVERALL BIAS CALCULATION (Matching Pine Script Logic)
        # =====================================================================
        
        # Count signals by category
        fast_bull = sum(1 for b in bias_results if b['category'] == 'fast' and 'BULLISH' in b['bias'])
        fast_bear = sum(1 for b in bias_results if b['category'] == 'fast' and 'BEARISH' in b['bias'])
        fast_total = sum(1 for b in bias_results if b['category'] == 'fast')
        
        medium_bull = sum(1 for b in bias_results if b['category'] == 'medium' and 'BULLISH' in b['bias'])
        medium_bear = sum(1 for b in bias_results if b['category'] == 'medium' and 'BEARISH' in b['bias'])
        medium_total = sum(1 for b in bias_results if b['category'] == 'medium')
        
        slow_bull = sum(1 for b in bias_results if b['category'] == 'slow' and 'BULLISH' in b['bias'])
        slow_bear = sum(1 for b in bias_results if b['category'] == 'slow' and 'BEARISH' in b['bias'])
        slow_total = sum(1 for b in bias_results if b['category'] == 'slow')
        
        # Calculate percentages
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0
        
        slow_bull_pct = (slow_bull / slow_total) * 100 if slow_total > 0 else 0
        slow_bear_pct = (slow_bear / slow_total) * 100 if slow_total > 0 else 0
        
        # Adaptive weighting
        divergence_threshold = self.config['divergence_threshold']
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence
        
        if divergence_detected:
            fast_weight = self.config['reversal_fast_weight']
            medium_weight = self.config['reversal_medium_weight']
            slow_weight = self.config['reversal_slow_weight']
            mode = "REVERSAL"
        else:
            fast_weight = self.config['normal_fast_weight']
            medium_weight = self.config['normal_medium_weight']
            slow_weight = self.config['normal_slow_weight']
            mode = "NORMAL"
        
        # Calculate weighted scores
        bullish_signals = (fast_bull * fast_weight) + (medium_bull * medium_weight) + (slow_bull * slow_weight)
        bearish_signals = (fast_bear * fast_weight) + (medium_bear * medium_weight) + (slow_bear * slow_weight)
        total_signals = (fast_total * fast_weight) + (medium_total * medium_weight) + (slow_total * slow_weight)
        
        bullish_bias_pct = (bullish_signals / total_signals) * 100 if total_signals > 0 else 0
        bearish_bias_pct = (bearish_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Determine overall bias
        bias_strength = self.config['bias_strength']
        
        if bullish_bias_pct >= bias_strength:
            overall_bias = "BULLISH"
            overall_score = bullish_bias_pct
            overall_confidence = min(100, bullish_bias_pct)
        elif bearish_bias_pct >= bias_strength:
            overall_bias = "BEARISH"
            overall_score = -bearish_bias_pct
            overall_confidence = min(100, bearish_bias_pct)
        else:
            overall_bias = "NEUTRAL"
            overall_score = 0
            overall_confidence = 100 - max(bullish_bias_pct, bearish_bias_pct)
        
        # Count totals
        bullish_count = sum(1 for b in bias_results if 'BULLISH' in b['bias'])
        bearish_count = sum(1 for b in bias_results if 'BEARISH' in b['bias'])
        neutral_count = sum(1 for b in bias_results if b['bias'] == 'NEUTRAL')
        
        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'bias_results': bias_results,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'stock_data': stock_data,
            'mode': mode,
            'fast_bull_pct': fast_bull_pct,
            'fast_bear_pct': fast_bear_pct,
            'slow_bull_pct': slow_bull_pct,
            'slow_bear_pct': slow_bear_pct,
            'bullish_bias_pct': bullish_bias_pct,
            'bearish_bias_pct': bearish_bias_pct
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🎯 Technical Bias Analysis Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive 13-Indicator Market Analysis")
    
    # Sidebar
    st.sidebar.title("Settings")
    symbol = st.sidebar.selectbox(
        "Select Market",
        ["^NSEI (NIFTY 50)", "^BSESN (SENSEX)", "^DJI (DOW JONES)", "AAPL", "TSLA", "MSFT"],
        index=0
    )
    
    # Extract symbol code
    symbol_code = symbol.split()[0] if ' ' in symbol else symbol
    
    if st.sidebar.button("🚀 Analyze Technical Bias", type="primary"):
        with st.spinner("Analyzing market data and calculating indicators..."):
            analyzer = TechnicalBiasAnalyzer()
            results = analyzer.analyze_technical_bias(symbol_code)
            
            if results['success']:
                display_results(results)
            else:
                st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")

    # Display instructions when no analysis has been run
    if 'results' not in locals():
        st.info("""
        ### 📊 About Technical Bias Analysis Pro
        
        This tool analyzes **13 technical indicators** across 3 categories to determine market bias:
        
        #### ⚡ Fast Indicators (8)
        - **Volume Delta** - Up vs Down volume
        - **HVP** - High Volume Pivots  
        - **VOB** - Volume Order Blocks
        - **Order Blocks** - EMA 5/18 crossover
        - **RSI** - Relative Strength Index
        - **DMI** - Directional Movement Index
        - **VIDYA** - Variable Index Dynamic Average
        - **MFI** - Money Flow Index
        
        #### 📊 Medium Indicators (2)
        - **Close vs VWAP** - Price position relative to VWAP
        - **Price vs VWAP** - Overall price relationship with VWAP
        
        #### 🐢 Slow Indicators (3)  
        - **Market Breadth** - Weighted stock performance analysis
        
        #### 🎯 Adaptive Scoring
        - **Normal Mode**: Fast(2x), Medium(3x), Slow(5x) weights
        - **Reversal Mode**: Fast(5x), Medium(3x), Slow(2x) weights
        - Automatically detects divergence for mode switching
        
        **Click the 'Analyze Technical Bias' button to get started!**
        """)

def display_results(results):
    """Display the analysis results in Streamlit"""
    
    # Header Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{results['current_price']:,.2f}")
    
    with col2:
        bias_emoji = "🐂" if results['overall_bias'] == "BULLISH" else "🐻" if results['overall_bias'] == "BEARISH" else "⚖️"
        st.metric("Overall Bias", f"{bias_emoji} {results['overall_bias']}")
    
    with col3:
        score_color = "green" if results['overall_score'] > 0 else "red" if results['overall_score'] < 0 else "gray"
        st.metric("Overall Score", f"{results['overall_score']:.1f}")
    
    with col4:
        confidence_color = "green" if results['overall_confidence'] > 70 else "orange" if results['overall_confidence'] > 50 else "red"
        st.metric("Confidence", f"{results['overall_confidence']:.1f}%")
    
    st.divider()
    
    # Signal Distribution
    st.subheader("📊 Signal Distribution")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🐂 Bullish", f"{results['bullish_count']}/{results['total_indicators']}")
    
    with col2:
        st.metric("🐻 Bearish", f"{results['bearish_count']}/{results['total_indicators']}")
    
    with col3:
        st.metric("⚖️ Neutral", f"{results['neutral_count']}/{results['total_indicators']}")
    
    with col4:
        st.metric("🔧 Mode", results['mode'])
    
    st.divider()
    
    # Detailed Indicator Analysis
    st.subheader("📋 Detailed Indicator Analysis")
    
    # Create tabs for different categories
    tab1, tab2, tab3 = st.tabs(["⚡ Fast Indicators (8)", "📊 Medium Indicators (2)", "🐢 Slow Indicators (3)"])
    
    with tab1:
        display_indicator_table([r for r in results['bias_results'] if r['category'] == 'fast'])
    
    with tab2:
        display_indicator_table([r for r in results['bias_results'] if r['category'] == 'medium'])
    
    with tab3:
        display_indicator_table([r for r in results['bias_results'] if r['category'] == 'slow'])
    
    st.divider()
    
    # Category Summary
    st.subheader("📈 Category Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fast Indicators Bullish", f"{results['fast_bull_pct']:.1f}%")
        st.metric("Fast Indicators Bearish", f"{results['fast_bear_pct']:.1f}%")
    
    with col2:
        st.metric("Slow Indicators Bullish", f"{results['slow_bull_pct']:.1f}%")
        st.metric("Slow Indicators Bearish", f"{results['slow_bear_pct']:.1f}%")
    
    with col3:
        st.metric("Overall Weighted Bullish", f"{results['bullish_bias_pct']:.1f}%")
        st.metric("Overall Weighted Bearish", f"{results['bearish_bias_pct']:.1f}%")
    
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
    
    # Timestamp
    st.caption(f"Last updated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}")

def display_indicator_table(indicators):
    """Display indicators in a formatted table"""
    for indicator in indicators:
        bias_class = "bullish" if "BULLISH" in indicator['bias'] else "bearish" if "BEARISH" in indicator['bias'] else "neutral"
        bias_emoji = "🟢" if "BULLISH" in indicator['bias'] else "🔴" if "BEARISH" in indicator['bias'] else "⚪"
        
        st.markdown(f"""
        <div class="bias-card {bias_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{indicator['indicator']}</strong><br>
                    <small>Value: {indicator['value']}</small>
                </div>
                <div style="text-align: right;">
                    {bias_emoji} <strong>{indicator['bias']}</strong><br>
                    <small>Score: {indicator['score']:.0f}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_trading_recommendation(results):
    """Generate trading recommendation based on bias analysis"""
    overall_bias = results['overall_bias']
    confidence = results['overall_confidence']
    mode = results['mode']
    
    recommendations = {
        'BULLISH': {
            'high': "🐂 STRONG BULLISH SIGNAL - Look for LONG entries on dips with stop loss below key support levels",
            'medium': "🐂 MODERATE BULLISH SIGNAL - Consider LONG entries with caution, use tighter stop losses",
            'low': "⚠️ WEAK BULLISH SIGNAL - Wait for confirmation before entering LONG positions"
        },
        'BEARISH': {
            'high': "🐻 STRONG BEARISH SIGNAL - Look for SHORT entries on rallies with stop loss above key resistance", 
            'medium': "🐻 MODERATE BEARISH SIGNAL - Consider SHORT entries with caution, use tighter stop losses",
            'low': "⚠️ WEAK BEARISH SIGNAL - Wait for confirmation before entering SHORT positions"
        },
        'NEUTRAL': {
            'high': "⚖️ STRONG NEUTRAL SIGNAL - Range-bound market expected, trade support/resistance levels",
            'medium': "⚖️ MODERATE NEUTRAL SIGNAL - Wait for breakout/breakdown confirmation before taking positions",
            'low': "⚖️ WEAK NEUTRAL SIGNAL - Market indecisive, consider staying out or using very small position sizes"
        }
    }
    
    # Determine confidence level
    if confidence >= 70:
        conf_level = 'high'
    elif confidence >= 50:
        conf_level = 'medium'
    else:
        conf_level = 'low'
    
    base_recommendation = recommendations[overall_bias][conf_level]
    
    # Add mode-specific note
    if mode == "REVERSAL":
        base_recommendation += " | 🔄 REVERSAL MODE DETECTED - High risk/reward potential, be prepared for trend changes"
    
    return base_recommendation

if __name__ == "__main__":
    main()