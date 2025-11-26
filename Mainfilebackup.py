"""
app.py — Complete single-file Streamlit app wrapping the BiasAnalysisPro engine (Option A)
- The BiasAnalysisPro class and all indicator logic are included exactly as provided.
- Streamlit UI added: tabs for Raw Data, Indicators, Bias Results, Market Breadth.
- Run with: streamlit run app.py
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

# Import Dhan API for Indian indices volume data
try:
    from dhan_data_fetcher import DhanDataFetcher
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    print("Warning: Dhan API not available. Volume data may be missing for Indian indices.")


class BiasAnalysisPro:
    """
    Comprehensive Bias Analysis matching Pine Script indicator EXACTLY
    Analyzes 13 bias indicators:
    - Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
    - Medium (2): Close vs VWAP, Price vs VWAP
    - Slow (3): Weighted stocks (Daily, TF1, TF2)
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
                '^NSEBANK': 10.0,  # BANKNIFTY Index
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

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Dhan API (for Indian indices) or Yahoo Finance (for others)
        Note: Yahoo Finance limits intraday data - use 7d max for 5m interval
        """
        # Check if this is an Indian index that needs Dhan API
        indian_indices = {'^NSEI': 'NIFTY', '^BSESN': 'SENSEX', '^NSEBANK': 'BANKNIFTY'}

        if symbol in indian_indices and DHAN_AVAILABLE:
            try:
                # Use Dhan API for Indian indices to get proper volume data
                dhan_instrument = indian_indices[symbol]
                fetcher = DhanDataFetcher()

                # Convert interval to Dhan API format (1, 5, 15, 25, 60)
                interval_map = {'1m': '1', '5m': '5', '15m': '15', '1h': '60'}
                dhan_interval = interval_map.get(interval, '5')

                # Calculate date range for historical data (7 days) - Use IST timezone
                now_ist = datetime.now(IST)
                to_date = now_ist.strftime('%Y-%m-%d %H:%M:%S')
                from_date = (now_ist - timedelta(days=7)).replace(hour=9, minute=15, second=0).strftime('%Y-%m-%d %H:%M:%S')

                # Fetch intraday data with 7 days historical range
                result = fetcher.fetch_intraday_data(dhan_instrument, interval=dhan_interval, from_date=from_date, to_date=to_date)

                if result.get('success') and result.get('data') is not None:
                    df = result['data']

                    # Ensure column names match yfinance format (capitalized)
                    df.columns = [col.capitalize() for col in df.columns]

                    # Set timestamp as index
                    if 'Timestamp' in df.columns:
                        df.set_index('Timestamp', inplace=True)

                    # Ensure volume column exists and has valid data
                    if 'Volume' not in df.columns:
                        df['Volume'] = 0
                    else:
                        # Replace NaN volumes with 0
                        df['Volume'] = df['Volume'].fillna(0)

                    if not df.empty:
                        print(f"✅ Fetched {len(df)} candles for {symbol} from Dhan API with volume data (from {from_date} to {to_date})")
                        return df
                    else:
                        print(f"⚠️  Warning: Empty data from Dhan API for {symbol}, falling back to yfinance")
                else:
                    print(f"Warning: Dhan API failed for {symbol}: {result.get('error')}, falling back to yfinance")
            except Exception as e:
                print(f"Error fetching from Dhan API for {symbol}: {e}, falling back to yfinance")

        # Fallback to Yahoo Finance for non-Indian indices or if Dhan fails
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data for {symbol}")
                return pd.DataFrame()

            # Ensure volume column exists (even if it's zeros for indices)
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            else:
                # Replace NaN volumes with 0
                df['Volume'] = df['Volume'].fillna(0)

            # Warn if volume is all zeros (common for Yahoo Finance indices)
            if df['Volume'].sum() == 0 and symbol in indian_indices:
                print(f"⚠️  Warning: Volume data is zero for {symbol} from Yahoo Finance")

            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral MFI (50) if no volume data
            return pd.Series([50.0] * len(df), index=df.index)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        # Avoid division by zero
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))

        # Fill NaN with neutral value (50)
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
        """Calculate VWAP with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return typical price as fallback if no volume data
            return (df['High'] + df['Low'] + df['Close']) / 3

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()

        # Avoid division by zero
        cumulative_volume_safe = cumulative_volume.replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume_safe

        # Fill NaN with typical price
        vwap = vwap.fillna(typical_price)

        return vwap

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20, band_distance: float = 2.0):
        """Calculate VIDYA (Variable Index Dynamic Average) matching Pine Script"""
        close = df['Close']

        # Calculate momentum (CMO - Chande Momentum Oscillator)
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

        # Determine trend based on band crossovers
        is_trend_up = close > upper_band
        is_trend_down = close < lower_band

        # Get current state
        vidya_bullish = is_trend_up.iloc[-1] if len(is_trend_up) > 0 else False
        vidya_bearish = is_trend_down.iloc[-1] if len(is_trend_down) > 0 else False

        return vidya_smoothed, vidya_bullish, vidya_bearish

    def calculate_volume_delta(self, df: pd.DataFrame):
        """Calculate Volume Delta (up_vol - down_vol) matching Pine Script"""
        if df['Volume'].sum() == 0:
            return 0, False, False

        # Calculate up and down volume
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()

        volume_delta = up_vol - down_vol
        volume_bullish = volume_delta > 0
        volume_bearish = volume_delta < 0

        return volume_delta, volume_bullish, volume_bearish

    def calculate_hvp(self, df: pd.DataFrame, left_bars: int = 15, right_bars: int = 15, vol_filter: float = 2.0):
        """Calculate High Volume Pivots matching Pine Script
        Returns: (hvp_bullish, hvp_bearish, pivot_high_count, pivot_low_count)
        """
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

        # Calculate volume sum and reference
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

    def calculate_vob(self, df: pd.DataFrame, length1: int = 5):
        """Calculate Volume Order Blocks matching Pine Script
        Returns: (vob_bullish, vob_bearish, ema1_value, ema2_value)
        """
        # Calculate EMAs
        length2 = length1 + 13
        ema1 = self.calculate_ema(df['Close'], length1)
        ema2 = self.calculate_ema(df['Close'], length2)

        # Detect crossovers
        cross_up = (ema1.iloc[-2] <= ema2.iloc[-2]) and (ema1.iloc[-1] > ema2.iloc[-1])
        cross_dn = (ema1.iloc[-2] >= ema2.iloc[-2]) and (ema1.iloc[-1] < ema2.iloc[-1])

        # In real implementation, we would check if price touched OB zones
        # For simplicity, using crossover signals
        vob_bullish = cross_up
        vob_bearish = cross_dn

        return vob_bullish, vob_bearish, ema1.iloc[-1], ema2.iloc[-1]

    # =========================================================================
    # ENHANCED INDICATORS (KEPT FOR COMPATIBILITY)
    # =========================================================================

    def calculate_volatility_ratio(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volatility Ratio"""
        atr = self.calculate_atr(df, length)
        stdev = df['Close'].rolling(window=length).std()
        volatility_ratio = (stdev / atr) * 100

        high_volatility = volatility_ratio.iloc[-1] > self.config['volatility_threshold']
        low_volatility = volatility_ratio.iloc[-1] < (self.config['volatility_threshold'] * 0.5)

        return volatility_ratio, high_volatility, low_volatility

    def calculate_volume_roc(self, df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, bool, bool]:
        """Calculate Volume Rate of Change with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral volume ROC if no volume data
            neutral_roc = pd.Series([0.0] * len(df), index=df.index)
            return neutral_roc, False, False

        # Avoid division by zero
        volume_shifted = df['Volume'].shift(length).replace(0, np.nan)
        volume_roc = ((df['Volume'] - df['Volume'].shift(length)) / volume_shifted) * 100

        # Fill NaN with 0
        volume_roc = volume_roc.fillna(0)

        # Check for strong/weak volume (handle NaN gracefully)
        last_value = volume_roc.iloc[-1] if not np.isnan(volume_roc.iloc[-1]) else 0
        strong_volume = last_value > self.config['volume_threshold']
        weak_volume = last_value < -self.config['volume_threshold']

        return volume_roc, strong_volume, weak_volume

    def calculate_obv(self, df: pd.DataFrame, smoothing: int = 21):
        """Calculate On Balance Volume with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral OBV if no volume data
            neutral_obv = pd.Series([0.0] * len(df), index=df.index)
            neutral_obv_ma = pd.Series([0.0] * len(df), index=df.index)
            return neutral_obv, neutral_obv_ma, False, False

        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_ma = obv.rolling(window=smoothing).mean()

        # Handle potential NaN or missing values
        obv = obv.fillna(0)
        obv_ma = obv_ma.fillna(0)

        # Safe comparison with fallback
        try:
            obv_rising = obv.iloc[-1] > obv.iloc[-2] if len(obv) >= 2 else False
            obv_falling = obv.iloc[-1] < obv.iloc[-2] if len(obv) >= 2 else False
            obv_bullish = obv.iloc[-1] > obv_ma.iloc[-1] and obv_rising
            obv_bearish = obv.iloc[-1] < obv_ma.iloc[-1] and obv_falling
        except:
            obv_bullish = False
            obv_bearish = False

        return obv, obv_ma, obv_bullish, obv_bearish

    def calculate_force_index(self, df: pd.DataFrame, length: int = 13, smoothing: int = 2):
        """Calculate Force Index with NaN/zero handling"""
        # Check if volume data is available
        if df['Volume'].sum() == 0:
            # Return neutral force index if no volume data
            neutral_force = pd.Series([0.0] * len(df), index=df.index)
            return neutral_force, False, False

        force_index = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        force_index = force_index.fillna(0)

        force_index_ma = force_index.ewm(span=length, adjust=False).mean()
        force_index_smoothed = force_index_ma.ewm(span=smoothing, adjust=False).mean()

        # Handle potential NaN
        force_index_smoothed = force_index_smoothed.fillna(0)

        # Safe comparison with fallback
        try:
            force_rising = force_index_smoothed.iloc[-1] > force_index_smoothed.iloc[-2] if len(force_index_smoothed) >= 2 else False
            force_falling = force_index_smoothed.iloc[-1] < force_index_smoothed.iloc[-2] if len(force_index_smoothed) >= 2 else False
            force_bullish = force_index_smoothed.iloc[-1] > 0 and force_rising
            force_bearish = force_index_smoothed.iloc[-1] < 0 and force_falling
        except:
            force_bullish = False
            force_bearish = False

        return force_index_smoothed, force_bullish, force_bearish

    def calculate_price_roc(self, df: pd.DataFrame, length: int = 12):
        """Calculate Price Rate of Change"""
        price_roc = ((df['Close'] - df['Close'].shift(length)) / df['Close'].shift(length)) * 100

        price_momentum_bullish = price_roc.iloc[-1] > 0
        price_momentum_bearish = price_roc.iloc[-1] < 0

        return price_roc, price_momentum_bullish, price_momentum_bearish

    def calculate_choppiness_index(self, df: pd.DataFrame, period: int = 14):
        """Calculate Choppiness Index"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        sum_true_range = true_range.rolling(window=period).sum()
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()

        ci = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(period)

        market_chopping = ci.iloc[-1] > self.config['ci_high_threshold']
        market_trending = ci.iloc[-1] < self.config['ci_low_threshold']

        return ci, market_chopping, market_trending

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 30):
        """Detect RSI/MACD Divergences"""
        rsi = self.calculate_rsi(df['Close'], 14)

        # MACD
        macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

        close_series = df['Close'].tail(lookback)
        rsi_series = rsi.tail(lookback)
        macd_series = macd_line.tail(lookback)

        # Bullish divergence
        lowest_close_idx = close_series.idxmin()
        lowest_rsi_idx = rsi_series.idxmin()
        bullish_rsi_divergence = (lowest_close_idx == close_series.index[-1] and
                                  rsi_series.iloc[-1] > rsi_series.loc[lowest_rsi_idx] and
                                  rsi_series.iloc[-1] < self.config['rsi_oversold'])

        # Bearish divergence
        highest_close_idx = close_series.idxmax()
        highest_rsi_idx = rsi_series.idxmax()
        bearish_rsi_divergence = (highest_close_idx == close_series.index[-1] and
                                  rsi_series.iloc[-1] < rsi_series.loc[highest_rsi_idx] and
                                  rsi_series.iloc[-1] > self.config['rsi_overbought'])

        return bullish_rsi_divergence, bearish_rsi_divergence

    # =========================================================================
    # MARKET BREADTH & STOCKS ANALYSIS
    # =========================================================================

    def _fetch_stock_data(self, symbol: str, weight: float):
        """Helper function to fetch single stock data for parallel processing"""
        try:
            # Use 5d period with 5m interval (Yahoo Finance limitation for intraday data)
            df = self.fetch_data(symbol, period='5d', interval='5m')
            if df.empty or len(df) < 2:
                return None

            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[0]
            change_pct = ((current_price - prev_price) / prev_price) * 100

            return {
                'symbol': symbol.replace('.NS', ''),
                'change_pct': change_pct,
                'weight': weight,
                'is_bullish': change_pct > 0
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None

    def calculate_market_breadth(self):
        """Calculate market breadth from top stocks (optimized with parallel processing)"""
        bullish_stocks = 0
        total_stocks = 0
        stock_data = []

        # Optimize: Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_stock = {
                executor.submit(self._fetch_stock_data, symbol, weight): (symbol, weight)
                for symbol, weight in self.config['stocks'].items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_stock):
                result = future.result()
                if result:
                    stock_data.append({
                        'symbol': result['symbol'],
                        'change_pct': result['change_pct'],
                        'weight': result['weight']
                    })
                    if result['is_bullish']:
                        bullish_stocks += 1
                    total_stocks += 1

        if total_stocks > 0:
            market_breadth = (bullish_stocks / total_stocks) * 100
        else:
            market_breadth = 50

        breadth_bullish = market_breadth > self.config['breadth_threshold']
        breadth_bearish = market_breadth < (100 - self.config['breadth_threshold'])

        return market_breadth, breadth_bullish, breadth_bearish, bullish_stocks, total_stocks, stock_data

    # =========================================================================
    # COMPREHENSIVE BIAS ANALYSIS
    # =========================================================================

    def analyze_all_bias_indicators(self, symbol: str = "^NSEI") -> Dict:
        """
        Analyze all 8 bias indicators:
        Fast (8): Volume Delta, HVP, VOB, Order Blocks, RSI, DMI, VIDYA, MFI
        """

        print(f"Fetching data for {symbol}...")
        # Use 7d period with 5m interval (Yahoo Finance limitation for intraday data)
        df = self.fetch_data(symbol, period='7d', interval='5m')

        if df.empty or len(df) < 100:
            error_msg = f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }

        current_price = df['Close'].iloc[-1]

        # Initialize bias results list
        bias_results = []
        stock_data = []  # Empty since we removed Weighted Stocks indicators

        # =====================================================================
        # FAST INDICATORS (8 total)
        # =====================================================================

        # 1. VOLUME DELTA
        volume_delta, volume_bullish, volume_bearish = self.calculate_volume_delta(df)

        if volume_bullish:
            vol_delta_bias = "BULLISH"
            vol_delta_score = 100
        elif volume_bearish:
            vol_delta_bias = "BEARISH"
            vol_delta_score = -100
        else:
            vol_delta_bias = "NEUTRAL"
            vol_delta_score = 0

        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{volume_delta:.0f}",
            'bias': vol_delta_bias,
            'score': vol_delta_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 2. HVP (High Volume Pivots)
        hvp_bullish, hvp_bearish, pivot_highs, pivot_lows = self.calculate_hvp(df)

        if hvp_bullish:
            hvp_bias = "BULLISH"
            hvp_score = 100
            hvp_value = f"Bull Signal (Lows: {pivot_lows}, Highs: {pivot_highs})"
        elif hvp_bearish:
            hvp_bias = "BEARISH"
            hvp_score = -100
            hvp_value = f"Bear Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"
        else:
            hvp_bias = "NEUTRAL"
            hvp_score = 0
            hvp_value = f"No Signal (Highs: {pivot_highs}, Lows: {pivot_lows})"

        bias_results.append({
            'indicator': 'HVP (High Volume Pivots)',
            'value': hvp_value,
            'bias': hvp_bias,
            'score': hvp_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 3. VOB (Volume Order Blocks)
        vob_bullish, vob_bearish, vob_ema5, vob_ema18 = self.calculate_vob(df)

        if vob_bullish:
            vob_bias = "BULLISH"
            vob_score = 100
            vob_value = f"Bull Cross (EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f})"
        elif vob_bearish:
            vob_bias = "BEARISH"
            vob_score = -100
            vob_value = f"Bear Cross (EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f})"
        else:
            vob_bias = "NEUTRAL"
            vob_score = 0
            # Determine if EMA5 is above or below EMA18
            if vob_ema5 > vob_ema18:
                vob_value = f"EMA5: {vob_ema5:.2f} > EMA18: {vob_ema18:.2f} (No Cross)"
            else:
                vob_value = f"EMA5: {vob_ema5:.2f} < EMA18: {vob_ema18:.2f} (No Cross)"

        bias_results.append({
            'indicator': 'VOB (Volume Order Blocks)',
            'value': vob_value,
            'bias': vob_bias,
            'score': vob_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # 4. ORDER BLOCKS (EMA Crossover)
        ema5 = self.calculate_ema(df['Close'], 5)
        ema18 = self.calculate_ema(df['Close'], 18)

        # Detect crossovers
        cross_up = (ema5.iloc[-2] <= ema18.iloc[-2]) and (ema5.iloc[-1] > ema18.iloc[-1])
        cross_dn = (ema5.iloc[-2] >= ema18.iloc[-2]) and (ema5.iloc[-1] < ema18.iloc[-1])

        if cross_up:
            ob_bias = "BULLISH"
            ob_score = 100
        elif cross_dn:
            ob_bias = "BEARISH"
            ob_score = -100
        else:
            ob_bias = "NEUTRAL"
            ob_score = 0

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

        if rsi_value > 50:
            rsi_bias = "BULLISH"
            rsi_score = 100
        else:
            rsi_bias = "BEARISH"
            rsi_score = -100

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
        plus_di_value = plus_di.iloc[-1]
        minus_di_value = minus_di.iloc[-1]
        adx_value = adx.iloc[-1]

        if plus_di_value > minus_di_value:
            dmi_bias = "BULLISH"
            dmi_score = 100
        else:
            dmi_bias = "BEARISH"
            dmi_score = -100

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

        if vidya_bullish:
            vidya_bias = "BULLISH"
            vidya_score = 100
        elif vidya_bearish:
            vidya_bias = "BEARISH"
            vidya_score = -100
        else:
            vidya_bias = "NEUTRAL"
            vidya_score = 0

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
        mfi_value = mfi.iloc[-1]

        if np.isnan(mfi_value):
            mfi_value = 50.0  # Neutral default

        if mfi_value > 50:
            mfi_bias = "BULLISH"
            mfi_score = 100
        else:
            mfi_bias = "BEARISH"
            mfi_score = -100

        bias_results.append({
            'indicator': 'MFI (Money Flow)',
            'value': f"{mfi_value:.2f}",
            'bias': mfi_bias,
            'score': mfi_score,
            'weight': 1.0,
            'category': 'fast'
        })

        # =====================================================================
        # CALCULATE OVERALL BIAS (MATCHING PINE SCRIPT LOGIC)
        # =====================================================================
        fast_bull = 0
        fast_bear = 0
        fast_total = 0

        medium_bull = 0
        medium_bear = 0
        medium_total = 0

        slow_bull = 0
        slow_bear = 0
        slow_total = 0

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for bias in bias_results:
            if 'BULLISH' in bias['bias']:
                bullish_count += 1
                if bias['category'] == 'fast':
                    fast_bull += 1
                elif bias['category'] == 'medium':
                    medium_bull += 1
                elif bias['category'] == 'slow':
                    slow_bull += 1
            elif 'BEARISH' in bias['bias']:
                bearish_count += 1
                if bias['category'] == 'fast':
                    fast_bear += 1
                elif bias['category'] == 'medium':
                    medium_bear += 1
                elif bias['category'] == 'slow':
                    slow_bear += 1
            else:
                neutral_count += 1

            if bias['category'] == 'fast':
                fast_total += 1
            elif bias['category'] == 'medium':
                medium_total += 1
            elif bias['category'] == 'slow':
                slow_total += 1

        # Calculate percentages
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        medium_bull_pct = (medium_bull / medium_total) * 100 if medium_total > 0 else 0
        medium_bear_pct = (medium_bear / medium_total) * 100 if medium_total > 0 else 0

        slow_bull_pct = (slow_bull / slow_total) * 100 if slow_total > 0 else 0
        slow_bear_pct = (slow_bear / slow_total) * 100 if slow_total > 0 else 0

        # Adaptive weighting (matching Pine Script)
        # Check for divergence
        divergence_threshold = self.config['divergence_threshold']
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        # Determine mode
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
            'stock_data': stock_data,
            'mode': mode,
            'fast_bull_pct': fast_bull_pct,
            'fast_bear_pct': fast_bear_pct,
            'slow_bull_pct': slow_bull_pct,
            'slow_bear_pct': slow_bear_pct,
            'bullish_bias_pct': bullish_bias_pct,
            'bearish_bias_pct': bearish_bias_pct
        }


# -----------------------------
# STREAMLIT APP UI (ENTRY)
# -----------------------------
st.set_page_config(page_title="Bias Analysis Pro", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Bias Analysis Pro — Complete Single-file App")
st.markdown(
    "This Streamlit app wraps the **BiasAnalysisPro** engine (Pine → Python) and shows raw data, "
    "indicator outputs, and the final adaptive bias. The core engine is preserved exactly as provided."
)

# Sidebar inputs
st.sidebar.header("Data & Symbol")
symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

# Analysis object
analysis = BiasAnalysisPro()

# Tabs
tabs = st.tabs(["Raw Data", "Indicators", "Bias Summary", "Market Breadth", "Settings"])

# Shared state storage
if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'last_symbol' not in st.session_state:
    st.session_state['last_symbol'] = None
if 'fetch_time' not in st.session_state:
    st.session_state['fetch_time'] = None

# Fetch & Analyze button
with st.sidebar:
    if st.button("Fetch Data & Run Analysis"):
        st.session_state['last_symbol'] = symbol_input
        with st.spinner("Fetching data (this may take a few seconds)..."):
            df_fetched = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
            st.session_state['last_df'] = df_fetched
            st.session_state['fetch_time'] = datetime.now(IST)

        if df_fetched is None or df_fetched.empty:
            st.error("No data fetched. Check symbol or network.")
        else:
            with st.spinner("Running full bias analysis..."):
                result = analysis.analyze_all_bias_indicators(symbol_input)
                st.session_state['last_result'] = result

# RAW DATA TAB
with tabs[0]:
    st.subheader("Raw Market Data")
    if st.session_state['last_df'] is None:
        st.info("No data loaded yet. Click **Fetch Data & Run Analysis** in the sidebar.")
    else:
        df = st.session_state['last_df']
        # Show last 200 rows in a scrollable container
        st.write(f"Showing last {min(200, len(df))} rows for {st.session_state.get('last_symbol', symbol_input)}")
        st.dataframe(df.tail(200), use_container_width=True)
        st.write("---")
        st.write("Data fetch time (IST):", st.session_state.get('fetch_time'))

# INDICATORS TAB
with tabs[1]:
    st.subheader("Indicator Outputs (selected)")
    if st.session_state['last_df'] is None:
        st.info("No data loaded yet. Click **Fetch Data & Run Analysis** in the sidebar.")
    else:
        df = st.session_state['last_df']

        # Build indicators table (selected ones)
        ind_rows = []

        try:
            rsi_series = analysis.calculate_rsi(df['Close'], analysis.config['rsi_period'])
            ind_rows.append(("RSI", round(float(rsi_series.iloc[-1]), 2)))
        except Exception as e:
            ind_rows.append(("RSI", f"ERR: {e}"))

        try:
            mfi_series = analysis.calculate_mfi(df, analysis.config['mfi_period'])
            ind_rows.append(("MFI", round(float(mfi_series.iloc[-1]), 2)))
        except Exception as e:
            ind_rows.append(("MFI", f"ERR: {e}"))

        try:
            plus_di, minus_di, adx = analysis.calculate_dmi(df, analysis.config['dmi_period'], analysis.config['dmi_smoothing'])
            ind_rows.append(("+DI", round(float(plus_di.iloc[-1]), 2)))
            ind_rows.append(("-DI", round(float(minus_di.iloc[-1]), 2)))
            ind_rows.append(("ADX", round(float(adx.iloc[-1]), 2)))
        except Exception as e:
            ind_rows.append(("DMI", f"ERR: {e}"))

        try:
            vwap_series = analysis.calculate_vwap(df)
            ind_rows.append(("VWAP", round(float(vwap_series.iloc[-1]), 2)))
        except Exception as e:
            ind_rows.append(("VWAP", f"ERR: {e}"))

        try:
            vidya_val, vidya_bull, vidya_bear = analysis.calculate_vidya(df)
            vidya_display = round(float(vidya_val.iloc[-1]), 2) if not vidya_val.empty else "N/A"
            ind_rows.append(("VIDYA", vidya_display))
            ind_rows.append(("VIDYA Bullish", bool(vidya_bull)))
            ind_rows.append(("VIDYA Bearish", bool(vidya_bear)))
        except Exception as e:
            ind_rows.append(("VIDYA", f"ERR: {e}"))

        try:
            vol_delta, vol_bull, vol_bear = analysis.calculate_volume_delta(df)
            ind_rows.append(("Volume Delta", int(vol_delta) if isinstance(vol_delta, (int, float)) else vol_delta))
            ind_rows.append(("Volume Bullish", bool(vol_bull)))
            ind_rows.append(("Volume Bearish", bool(vol_bear)))
        except Exception as e:
            ind_rows.append(("Volume Delta", f"ERR: {e}"))

        try:
            hvp_bull, hvp_bear, pivh, pivl = analysis.calculate_hvp(df)
            ind_rows.append(("HVP Bullish", bool(hvp_bull)))
            ind_rows.append(("HVP Bearish", bool(hvp_bear)))
            ind_rows.append(("HVP Pivot Highs", int(pivh)))
            ind_rows.append(("HVP Pivot Lows", int(pivl)))
        except Exception as e:
            ind_rows.append(("HVP", f"ERR: {e}"))

        try:
            vob_bull, vob_bear, ema1, ema2 = analysis.calculate_vob(df)
            ind_rows.append(("VOB Bullish", bool(vob_bull)))
            ind_rows.append(("VOB Bearish", bool(vob_bear)))
            ind_rows.append(("EMA5", round(float(ema1), 2)))
            ind_rows.append(("EMA18", round(float(ema2), 2)))
        except Exception as e:
            ind_rows.append(("VOB", f"ERR: {e}"))

        ind_df = pd.DataFrame(ind_rows, columns=["Indicator", "Value"])
        st.dataframe(ind_df, use_container_width=True)

# BIAS SUMMARY TAB
with tabs[2]:
    st.subheader("Overall Bias Summary")
    if st.session_state['last_result'] is None:
        st.info("No analysis run yet. Click **Fetch Data & Run Analysis** in the sidebar.")
    else:
        res = st.session_state['last_result']
        if not res.get('success', False):
            st.error(f"Analysis failed: {res.get('error')}")
        else:
            st.markdown(f"**Symbol:** `{res['symbol']}`")
            st.markdown(f"**Timestamp (IST):** {res['timestamp']}")
            st.metric("Current Price", f"{res['current_price']:.2f}")
            st.metric("Overall Bias", res['overall_bias'], delta=f"Confidence: {res['overall_confidence']:.1f}%")
            st.write("Mode:", res.get('mode', 'N/A'))

            # Show bias results table
            bias_table = pd.DataFrame(res['bias_results'])
            # Reorder columns for nicer view
            cols_order = ['indicator', 'value', 'bias', 'score', 'weight', 'category']
            bias_table = bias_table[cols_order]
            bias_table.columns = [c.capitalize() for c in bias_table.columns]
            st.subheader("Indicator-level Biases")
            st.dataframe(bias_table, use_container_width=True)

            # Summary stats
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Bullish Indicators", int(res['bullish_count']))
            col2.metric("Bearish Indicators", int(res['bearish_count']))
            col3.metric("Neutral Indicators", int(res['neutral_count']))

            st.write("Weighted Bias Percentages:")
            st.write(f"- Fast Bull %: {res.get('fast_bull_pct', 0):.1f}%")
            st.write(f"- Fast Bear %: {res.get('fast_bear_pct', 0):.1f}%")
            st.write(f"- Slow Bull %: {res.get('slow_bull_pct', 0):.1f}%")
            st.write(f"- Slow Bear %: {res.get('slow_bear_pct', 0):.1f}%")
            st.write(f"- Bullish Bias % (weighted): {res.get('bullish_bias_pct', 0):.1f}%")
            st.write(f"- Bearish Bias % (weighted): {res.get('bearish_bias_pct', 0):.1f}%")

# MARKET BREADTH TAB
with tabs[3]:
    st.subheader("Market Breadth — Top Stocks (configured list)")
    if st.button("Run Market Breadth Scan"):
        with st.spinner("Calculating market breadth..."):
            mb_result = analysis.calculate_market_breadth()
        mb_pct, breadth_bullish, breadth_bearish, bullish_stocks, total_stocks, stock_data = mb_result
        st.metric("Market Breadth (%)", f"{mb_pct:.1f}%")
        st.write(f"Bullish stocks: {bullish_stocks} / {total_stocks}")
        if len(stock_data) > 0:
            stock_df = pd.DataFrame(stock_data)
            st.dataframe(stock_df.sort_values('change_pct', ascending=False).reset_index(drop=True))
        else:
            st.info("No stock data available (possible fetch issues).")

# SETTINGS TAB
with tabs[4]:
    st.subheader("Settings & Notes")
    st.markdown(
        "- The BiasAnalysisPro engine and indicator calculations are left **unchanged** from the provided script.\n"
        "- If you want the file split into `bias_module.py` + `app.py` (cleaner), I can provide that.\n"
        "- Dhan API: If you have `dhan_data_fetcher` installed and configured, the app will use it for Indian indices volume data.\n"
        "- Yahoo Finance intraday limitations: 5m intraday historical is limited to ~7 days. Use appropriate `period`/`interval` combos.\n"
    )

# Footer
st.markdown("---")
st.caption("BiasAnalysisPro — converted to a single-file Streamlit app (Option A).")
