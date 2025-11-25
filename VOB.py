#!/usr/bin/env python3
"""
Technical Bias Analysis Script
Analyzes market bias using multiple technical indicators
Provides weighted scoring and clear trading signals
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalBiasAnalyzer:
    """Comprehensive technical bias analysis with weighted scoring"""
    
    def __init__(self):
        self.config = {
            # Indicator periods
            'rsi_period': 14,
            'mfi_period': 10,
            'dmi_period': 13,
            'dmi_smoothing': 8,
            'atr_period': 14,
            
            # VWAP settings
            'vwap_std_dev': 2.0,
            
            # Bias thresholds
            'bias_strength': 60,
            
            # Adaptive weights (Normal mode)
            'fast_weight': 2.0,    # Fast indicators (Volume, RSI, DMI)
            'medium_weight': 3.0,  # Medium indicators (VWAP)
            'slow_weight': 5.0,    # Slow indicators (Trend)
        }
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"❌ No data for {symbol}")
                return pd.DataFrame()
            
            # Ensure volume column exists
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Money Flow Index"""
        if df['Volume'].sum() == 0:
            return pd.Series([50] * len(df), index=df.index)
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = (money_flow.where(typical_price > typical_price.shift(1), 0)
                         .rolling(window=period).sum())
        negative_flow = (money_flow.where(typical_price < typical_price.shift(1), 0)
                         .rolling(window=period).sum())
        
        mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi.fillna(50)
    
    def calculate_dmi(self, df: pd.DataFrame, period: int = 13, smoothing: int = 8) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Directional Movement Index (+DI, -DI, ADX)"""
        high, low, close = df['High'], df['Low'], df['Close']
        
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
        
        return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        if df['Volume'].sum() == 0:
            return (df['High'] + df['Low'] + df['Close']) / 3
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cum_vol = df['Volume'].cumsum().replace(0, np.nan)
        vwap = (typical_price * df['Volume']).cumsum() / cum_vol
        return vwap.fillna(typical_price)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high, low, close = df['High'], df['Low'], df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_delta(self, df: pd.DataFrame) -> Tuple[float, bool, bool]:
        """Calculate volume delta (up volume - down volume)"""
        if df['Volume'].sum() == 0:
            return 0, False, False
        
        up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
        down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
        
        volume_delta = up_vol - down_vol
        return volume_delta, volume_delta > 0, volume_delta < 0
    
    def calculate_vidya(self, df: pd.DataFrame, length: int = 10, momentum: int = 20) -> Tuple[pd.Series, bool, bool]:
        """Calculate Variable Index Dynamic Average"""
        close = df['Close']
        
        # Chande Momentum Oscillator
        m = close.diff()
        p = m.where(m >= 0, 0.0).rolling(window=momentum).sum()
        n = (-m.where(m < 0, 0.0)).rolling(window=momentum).sum()
        
        abs_cmo = abs(100 * (p - n) / (p + n).replace(0, np.nan)).fillna(0)
        
        # VIDYA calculation
        alpha = 2 / (length + 1)
        vidya = close.copy()
        
        for i in range(1, len(close)):
            vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] +
                           (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])
        
        # Smooth VIDYA
        vidya_smoothed = vidya.rolling(window=15).mean()
        
        # Bands
        atr = self.calculate_atr(df, 200)
        upper_band = vidya_smoothed + atr * 2.0
        lower_band = vidya_smoothed - atr * 2.0
        
        # Trend detection
        is_bullish = close.iloc[-1] > upper_band.iloc[-1] if len(upper_band) > 0 else False
        is_bearish = close.iloc[-1] < lower_band.iloc[-1] if len(lower_band) > 0 else False
        
        return vidya_smoothed, is_bullish, is_bearish
    
    # =========================================================================
    # BIAS ANALYSIS
    # =========================================================================
    
    def analyze_bias(self, symbol: str = "^NSEI") -> Dict:
        """Perform comprehensive bias analysis"""
        
        print(f"\n📊 Analyzing bias for {symbol}...")
        
        # Fetch data
        df = self.fetch_data(symbol, period='7d', interval='5m')
        
        if df.empty or len(df) < 100:
            return {
                'success': False,
                'error': f'Insufficient data (fetched {len(df)} candles, need at least 100)'
            }
        
        bias_results = []
        current_price = df['Close'].iloc[-1]
        
        # 1. VOLUME DELTA
        vol_delta, vol_bullish, vol_bearish = self.calculate_volume_delta(df)
        
        bias_results.append({
            'indicator': 'Volume Delta',
            'value': f"{vol_delta:,.0f}",
            'bias': 'BULLISH' if vol_bullish else 'BEARISH' if vol_bearish else 'NEUTRAL',
            'score': 100 if vol_bullish else -100 if vol_bearish else 0,
            'category': 'fast'
        })
        
        # 2. RSI
        rsi = self.calculate_rsi(df['Close'], self.config['rsi_period'])
        rsi_value = rsi.iloc[-1]
        
        bias_results.append({
            'indicator': 'RSI',
            'value': f"{rsi_value:.1f}",
            'bias': 'BULLISH' if rsi_value > 50 else 'BEARISH',
            'score': 100 if rsi_value > 50 else -100,
            'category': 'fast'
        })
        
        # 3. DMI
        plus_di, minus_di, adx = self.calculate_dmi(df, self.config['dmi_period'], self.config['dmi_smoothing'])
        
        bias_results.append({
            'indicator': 'DMI',
            'value': f"+{plus_di.iloc[-1]:.1f}/-{minus_di.iloc[-1]:.1f}",
            'bias': 'BULLISH' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'BEARISH',
            'score': 100 if plus_di.iloc[-1] > minus_di.iloc[-1] else -100,
            'category': 'fast'
        })
        
        # 4. MFI
        mfi = self.calculate_mfi(df, self.config['mfi_period'])
        mfi_value = mfi.iloc[-1]
        
        bias_results.append({
            'indicator': 'Money Flow',
            'value': f"{mfi_value:.1f}",
            'bias': 'BULLISH' if mfi_value > 50 else 'BEARISH',
            'score': 100 if mfi_value > 50 else -100,
            'category': 'fast'
        })
        
        # 5. VIDYA
        vidya_val, vidya_bullish, vidya_bearish = self.calculate_vidya(df)
        
        bias_results.append({
            'indicator': 'VIDYA',
            'value': f"{vidya_val.iloc[-1]:.2f}" if not vidya_val.empty else "N/A",
            'bias': 'BULLISH' if vidya_bullish else 'BEARISH' if vidya_bearish else 'NEUTRAL',
            'score': 100 if vidya_bullish else -100 if vidya_bearish else 0,
            'category': 'slow'
        })
        
        # 6. VWAP Analysis
        vwap = self.calculate_vwap(df)
        current_vwap = vwap.iloc[-1]
        close_above_vwap = current_price > current_vwap
        
        bias_results.append({
            'indicator': 'Price vs VWAP',
            'value': f"Price: ₹{current_price:.2f} | VWAP: ₹{current_vwap:.2f}",
            'bias': 'BULLISH' if close_above_vwap else 'BEARISH',
            'score': 100 if close_above_vwap else -100,
            'category': 'medium'
        })
        
        # Calculate overall bias
        bullish_count = sum(1 for b in bias_results if b['bias'] == 'BULLISH')
        bearish_count = sum(1 for b in bias_results if b['bias'] == 'BEARISH')
        neutral_count = len(bias_results) - bullish_count - bearish_count
        
        # Weighted scoring
        weighted_score = 0
        total_weight = 0
        
        for bias in bias_results:
            weight = self.config['fast_weight'] if bias['category'] == 'fast' else \
                    self.config['medium_weight'] if bias['category'] == 'medium' else self.config['slow_weight']
            
            weighted_score += bias['score'] * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight
        
        # Determine bias
        bias_strength = self.config['bias_strength']
        if overall_score >= bias_strength:
            overall_bias = 'BULLISH'
        elif overall_score <= -bias_strength:
            overall_bias = 'BEARISH'
        else:
            overall_bias = 'NEUTRAL'
        
        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'overall_bias': overall_bias,
            'overall_score': overall_score,
            'bias_results': bias_results,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_indicators': len(bias_results),
            'timestamp': datetime.now()
        }
    
    def print_bias_report(self, results: Dict):
        """Print formatted bias analysis report"""
        
        if not results.get('success'):
            print(f"\n❌ Error: {results.get('error')}")
            return
        
        print(f"\n{'='*60}")
        print(f" TECHNICAL BIAS ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"Symbol: {results['symbol']}")
        print(f"Current Price: ₹{results['current_price']:,.2f}")
        print(f"Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n{'='*60}")
        print(f" OVERALL BIAS: {results['overall_bias']}")
        print(f" Score: {results['overall_score']:.2f}/100")
        print(f"{'='*60}")
        
        print(f"\n📊 Signal Distribution:")
        print(f"  🐂 Bullish: {results['bullish_count']}/{results['total_indicators']}")
        print(f"  🐻 Bearish: {results['bearish_count']}/{results['total_indicators']}")
        print(f"  ⚖️ Neutral: {results['neutral_count']}/{results['total_indicators']}")
        
        print(f"\n{'='*60}")
        print(f" INDICATOR BREAKDOWN")
        print(f"{'='*60}")
        
        for bias in results['bias_results']:
            status_emoji = "🟢" if bias['bias'] == 'BULLISH' else "🔴" if bias['bias'] == 'BEARISH' else "⚪"
            print(f"\n{status_emoji} {bias['indicator']}")
            print(f"   Value: {bias['value']}")
            print(f"   Bias: {bias['bias']} ({bias['score']:+.0f})")
            print(f"   Category: {bias['category'].title()}")
        
        print(f"\n{'='*60}")
        print(f" TRADING RECOMMENDATION")
        print(f"{'='*60}")
        
        if results['overall_bias'] == 'BULLISH' and results['overall_score'] >= 70:
            print("✅ STRONG BULLISH SIGNAL")
            print("   → Consider LONG positions on dips")
            print("   → Set stop loss below recent swing low")
            print("   → Target: 1:2 Risk-Reward or higher")
        elif results['overall_bias'] == 'BULLISH':
            print("✅ MODERATE BULLISH SIGNAL")
            print("   → Consider LONG with caution")
            print("   → Use tighter stop losses")
            print("   → Take partial profits at resistance")
        elif results['overall_bias'] == 'BEARISH' and results['overall_score'] <= -70:
            print("🔴 STRONG BEARISH SIGNAL")
            print("   → Consider SHORT positions on rallies")
            print("   → Set stop loss above recent swing high")
            print("   → Target: 1:2 Risk-Reward or higher")
        elif results['overall_bias'] == 'BEARISH':
            print("🔴 MODERATE BEARISH SIGNAL")
            print("   → Consider SHORT with caution")
            print("   → Use tighter stop losses")
            print("   → Take partial profits at support")
        else:
            print("⚖️ NEUTRAL / NO CLEAR SIGNAL")
            print("   → Stay out or use range trading")
            print("   → Wait for clearer bias formation")
            print("   → Reduce position sizes if trading")
        
        print(f"\n{'='*60}\n")


def main():
    """Main execution function"""
    
    print("=" * 60)
    print(" TECHNICAL BIAS ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TechnicalBiasAnalyzer()
    
    # Analysis parameters
    symbols = {
        "1": ("^NSEI", "NIFTY 50"),
        "2": ("^BSESN", "SENSEX"),
        "3": ("^NSEBANK", "BANKNIFTY")
    }
    
    print("Select Market:")
    for key, (symbol, name) in symbols.items():
        print(f"  {key}. {name} ({symbol})")
    
    choice = input("\nEnter choice (1-3) or custom symbol: ").strip()
    
    if choice in symbols:
        symbol, name = symbols[choice]
    else:
        symbol = choice
        name = choice
    
    # Run analysis
    results = analyzer.analyze_bias(symbol)
    analyzer.print_bias_report(results)
    
    # Ask for refresh
    while True:
        refresh = input("Refresh analysis? (y/n): ").strip().lower()
        if refresh == 'y':
            results = analyzer.analyze_bias(symbol)
            analyzer.print_bias_report(results)
        else:
            print("\n✅ Analysis complete. Goodbye!")
            break


if __name__ == "__main__":
    main()
