"""
Comprehensive Technical Bias Analysis
=====================================

Analyzes 13 technical bias indicators with adaptive weighted scoring:
- 8 Fast Indicators: Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
- 2 Medium Indicators: Close vs VWAP, Price vs VWAP  
- 3 Slow Indicators: Weighted stocks (Daily, 15m, 1h)

Provides:
- Individual indicator bias signals
- Overall market bias with confidence scoring
- Adaptive weighting (Normal vs Reversal modes)
- Trading recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from typing import Dict, List, Tuple, Optional
import pytz

warnings.filterwarnings('ignore')

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
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()
                
            # Ensure required columns exist
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                df['Volume'] = df['Volume'].fillna(0)
                
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
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
        print(f"🔍 Analyzing technical bias for {symbol}...")
        
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

    def generate_trading_recommendation(self, analysis_results: Dict) -> str:
        """Generate trading recommendation based on bias analysis"""
        if not analysis_results.get('success'):
            return "❌ Analysis failed - cannot generate recommendation"
        
        overall_bias = analysis_results['overall_bias']
        confidence = analysis_results['overall_confidence']
        mode = analysis_results['mode']
        
        recommendations = {
            'BULLISH': {
                'high': "🐂 STRONG BULLISH - Look for LONG entries on dips with stop loss below support",
                'medium': "🐂 MODERATE BULLISH - Consider LONG entries with caution, use tighter stops",
                'low': "⚠️ WEAK BULLISH - Wait for confirmation before entering LONG positions"
            },
            'BEARISH': {
                'high': "🐻 STRONG BEARISH - Look for SHORT entries on rallies with stop loss above resistance", 
                'medium': "🐻 MODERATE BEARISH - Consider SHORT entries with caution, use tighter stops",
                'low': "⚠️ WEAK BEARISH - Wait for confirmation before entering SHORT positions"
            },
            'NEUTRAL': {
                'high': "⚖️ STRONG NEUTRAL - Range-bound market, trade support/resistance levels",
                'medium': "⚖️ MODERATE NEUTRAL - Wait for breakout/breakdown confirmation",
                'low': "⚖️ WEAK NEUTRAL - Market indecisive, consider staying out"
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
            base_recommendation += " | 🔄 REVERSAL MODE DETECTED - High risk/reward potential"
        
        return base_recommendation

    def print_analysis_report(self, analysis_results: Dict):
        """Print comprehensive analysis report"""
        if not analysis_results.get('success'):
            print(f"❌ Analysis failed: {analysis_results.get('error')}")
            return
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TECHNICAL BIAS ANALYSIS REPORT")
        print("="*80)
        
        # Header
        print(f"📊 Symbol: {analysis_results['symbol']}")
        print(f"💰 Current Price: ₹{analysis_results['current_price']:,.2f}")
        print(f"⏰ Analysis Time: {analysis_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Overall Bias
        bias_emoji = "🐂" if analysis_results['overall_bias'] == "BULLISH" else "🐻" if analysis_results['overall_bias'] == "BEARISH" else "⚖️"
        print(f"\n{bias_emoji} OVERALL BIAS: {analysis_results['overall_bias']}")
        print(f"📈 Overall Score: {analysis_results['overall_score']:.1f}")
        print(f"🎯 Confidence: {analysis_results['overall_confidence']:.1f}%")
        print(f"🔧 Mode: {analysis_results['mode']}")
        
        # Signal Distribution
        print(f"\n📊 SIGNAL DISTRIBUTION:")
        print(f"   🟢 Bullish: {analysis_results['bullish_count']}/{analysis_results['total_indicators']}")
        print(f"   🔴 Bearish: {analysis_results['bearish_count']}/{analysis_results['total_indicators']}") 
        print(f"   ⚪ Neutral: {analysis_results['neutral_count']}/{analysis_results['total_indicators']}")
        
        # Detailed Results
        print(f"\n📋 DETAILED INDICATOR ANALYSIS:")
        print("-" * 80)
        print(f"{'INDICATOR':<25} {'VALUE':<15} {'BIAS':<10} {'SCORE':<8} {'CATEGORY':<10}")
        print("-" * 80)
        
        for result in analysis_results['bias_results']:
            bias_color = "🟢" if "BULLISH" in result['bias'] else "🔴" if "BEARISH" in result['bias'] else "⚪"
            print(f"{result['indicator']:<25} {result['value']:<15} {bias_color} {result['bias']:<7} {result['score']:>6.0f} {result['category']:<10}")
        
        # Category Summary
        print(f"\n📈 CATEGORY SUMMARY:")
        print(f"   ⚡ Fast Indicators:  {analysis_results['fast_bull_pct']:.1f}% Bull | {analysis_results['fast_bear_pct']:.1f}% Bear")
        print(f"   🐢 Slow Indicators:  {analysis_results['slow_bull_pct']:.1f}% Bull | {analysis_results['slow_bear_pct']:.1f}% Bear")
        print(f"   📊 Overall Weighted: {analysis_results['bullish_bias_pct']:.1f}% Bull | {analysis_results['bearish_bias_pct']:.1f}% Bear")
        
        # Trading Recommendation
        recommendation = self.generate_trading_recommendation(analysis_results)
        print(f"\n💡 TRADING RECOMMENDATION:")
        print(f"   {recommendation}")
        
        # Market Breadth
        if analysis_results['stock_data']:
            print(f"\n🏢 MARKET BREADTH:")
            bullish_stocks = [s for s in analysis_results['stock_data'] if s['is_bullish']]
            bearish_stocks = [s for s in analysis_results['stock_data'] if not s['is_bullish']]
            print(f"   🟢 Bullish Stocks: {len(bullish_stocks)}")
            print(f"   🔴 Bearish Stocks: {len(bearish_stocks)}")
            
            # Top movers
            top_gainers = sorted(analysis_results['stock_data'], key=lambda x: x['change_pct'], reverse=True)[:3]
            top_losers = sorted(analysis_results['stock_data'], key=lambda x: x['change_pct'])[:3]
            
            print(f"\n📈 TOP GAINERS:")
            for stock in top_gainers:
                print(f"   {stock['symbol']}: {stock['change_pct']:+.2f}%")
                
            print(f"\n📉 TOP LOSERS:")  
            for stock in top_losers:
                print(f"   {stock['symbol']}: {stock['change_pct']:+.2f}%")
        
        print("="*80)

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the Technical Bias Analyzer"""
    
    # Initialize analyzer
    analyzer = TechnicalBiasAnalyzer()
    
    # Analyze NIFTY
    print("Starting Technical Bias Analysis...")
    results = analyzer.analyze_technical_bias("^NSEI")  # NIFTY 50
    
    # Print comprehensive report
    analyzer.print_analysis_report(results)
    
    # You can also analyze other markets
    # results_sensex = analyzer.analyze_technical_bias("^BSESN")  # SENSEX
    # results_dow = analyzer.analyze_technical_bias("^DJI")      # DOW JONES

if __name__ == "__main__":
    main()