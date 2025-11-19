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
# NSE OPTIONS ANALYZER INTEGRATION WITH FULL ATM BIAS
# =============================================

class NSEOptionsAnalyzer:
    """Integrated NSE Options Analyzer with complete ATM bias analysis"""
    
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

    def delta_volume_bias(self, price, volume, chg_oi):
        """Calculate delta volume bias"""
        if price > 0 and volume > 0 and chg_oi > 0:
            return "Bullish"
        elif price < 0 and volume > 0 and chg_oi > 0:
            return "Bearish"
        elif price > 0 and volume > 0 and chg_oi < 0:
            return "Bullish"
        elif price < 0 and volume > 0 and chg_oi < 0:
            return "Bearish"
        else:
            return "Neutral"

    def final_verdict(self, score):
        """Determine final verdict based on score"""
        if score >= 4:
            return "Strong Bullish"
        elif score >= 2:
            return "Bullish"
        elif score <= -4:
            return "Strong Bearish"
        elif score <= -2:
            return "Bearish"
        else:
            return "Neutral"

    def determine_level(self, row):
        """Determine support/resistance level based on OI"""
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']

        # Strong Support condition
        if pe_oi > 1.12 * ce_oi:
            return "Support"
        # Strong Resistance condition
        elif ce_oi > 1.12 * pe_oi:
            return "Resistance"
        # Neutral if none dominant
        else:
            return "Neutral"

    def calculate_max_pain(self, df_full_chain):
        """Calculate Max Pain strike"""
        try:
            strikes = df_full_chain['strikePrice'].unique()
            pain_values = []

            for strike in strikes:
                call_pain = 0
                put_pain = 0

                # Calculate pain for all strikes
                for _, row in df_full_chain.iterrows():
                    row_strike = row['strikePrice']

                    # Call pain: If strike price > current strike, calls are ITM
                    if row_strike < strike:
                        call_pain += (strike - row_strike) * row.get('openInterest_CE', 0)

                    # Put pain: If strike price < current strike, puts are ITM
                    if row_strike > strike:
                        put_pain += (row_strike - strike) * row.get('openInterest_PE', 0)

                total_pain = call_pain + put_pain
                pain_values.append({'strike': strike, 'pain': total_pain})

            # Max pain is the strike with minimum total pain
            max_pain_data = min(pain_values, key=lambda x: x['pain'])
            return max_pain_data['strike']
        except:
            return None

    def calculate_synthetic_future_bias(self, atm_ce_price, atm_pe_price, atm_strike, spot_price):
        """Calculate Synthetic Future Bias at ATM"""
        try:
            synthetic_future = atm_strike + atm_ce_price - atm_pe_price
            difference = synthetic_future - spot_price

            if difference > 5:  # Threshold can be adjusted
                return "Bullish", synthetic_future, difference
            elif difference < -5:
                return "Bearish", synthetic_future, difference
            else:
                return "Neutral", synthetic_future, difference
        except:
            return "Neutral", 0, 0

    def calculate_atm_buildup_pattern(self, atm_ce_oi, atm_pe_oi, atm_ce_change, atm_pe_change):
        """Determine ATM buildup pattern based on OI changes"""
        try:
            # Classify based on OI changes
            if atm_ce_change > 0 and atm_pe_change > 0:
                if atm_ce_change > atm_pe_change:
                    return "Long Buildup (Bearish)"
                else:
                    return "Short Buildup (Bullish)"
            elif atm_ce_change < 0 and atm_pe_change < 0:
                if abs(atm_ce_change) > abs(atm_pe_change):
                    return "Short Covering (Bullish)"
                else:
                    return "Long Unwinding (Bearish)"
            elif atm_ce_change > 0 and atm_pe_change < 0:
                return "Call Writing (Bearish)"
            elif atm_ce_change < 0 and atm_pe_change > 0:
                return "Put Writing (Bullish)"
            else:
                return "Neutral"
        except:
            return "Neutral"

    def calculate_atm_vega_bias(self, atm_ce_vega, atm_pe_vega, atm_ce_oi, atm_pe_oi):
        """Calculate ATM Vega exposure bias"""
        try:
            ce_vega_exposure = atm_ce_vega * atm_ce_oi
            pe_vega_exposure = atm_pe_vega * atm_pe_oi

            total_vega_exposure = ce_vega_exposure + pe_vega_exposure

            if pe_vega_exposure > ce_vega_exposure * 1.1:
                return "Bullish (High Put Vega)", total_vega_exposure
            elif ce_vega_exposure > pe_vega_exposure * 1.1:
                return "Bearish (High Call Vega)", total_vega_exposure
            else:
                return "Neutral", total_vega_exposure
        except:
            return "Neutral", 0

    def find_call_resistance_put_support(self, df_full_chain, spot_price):
        """Find key resistance (from Call OI) and support (from Put OI) strikes"""
        try:
            # Find strikes above spot with highest Call OI (Resistance)
            above_spot = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            if not above_spot.empty:
                call_resistance = above_spot.nlargest(1, 'openInterest_CE')['strikePrice'].values[0]
            else:
                call_resistance = None

            # Find strikes below spot with highest Put OI (Support)
            below_spot = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            if not below_spot.empty:
                put_support = below_spot.nlargest(1, 'openInterest_PE')['strikePrice'].values[0]
            else:
                put_support = None

            return call_resistance, put_support
        except:
            return None, None

    def calculate_total_vega_bias(self, df_full_chain):
        """Calculate total Vega bias across all strikes"""
        try:
            total_ce_vega = (df_full_chain['Vega_CE'] * df_full_chain['openInterest_CE']).sum()
            total_pe_vega = (df_full_chain['Vega_PE'] * df_full_chain['openInterest_PE']).sum()

            total_vega = total_ce_vega + total_pe_vega

            if total_pe_vega > total_ce_vega * 1.1:
                return "Bullish (Put Heavy)", total_vega, total_ce_vega, total_pe_vega
            elif total_ce_vega > total_pe_vega * 1.1:
                return "Bearish (Call Heavy)", total_vega, total_ce_vega, total_pe_vega
            else:
                return "Neutral", total_vega, total_ce_vega, total_pe_vega
        except:
            return "Neutral", 0, 0, 0

    def detect_unusual_activity(self, df_full_chain, spot_price):
        """Detect strikes with unusual activity (high volume relative to OI)"""
        try:
            unusual_strikes = []

            for _, row in df_full_chain.iterrows():
                strike = row['strikePrice']

                # Check Call side
                ce_oi = row.get('openInterest_CE', 0)
                ce_volume = row.get('totalTradedVolume_CE', 0)
                if ce_oi > 0 and ce_volume / ce_oi > 0.5:  # Volume > 50% of OI
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'CE',
                        'volume_oi_ratio': ce_volume / ce_oi if ce_oi > 0 else 0,
                        'volume': ce_volume,
                        'oi': ce_oi
                    })

                # Check Put side
                pe_oi = row.get('openInterest_PE', 0)
                pe_volume = row.get('totalTradedVolume_PE', 0)
                if pe_oi > 0 and pe_volume / pe_oi > 0.5:
                    unusual_strikes.append({
                        'strike': strike,
                        'type': 'PE',
                        'volume_oi_ratio': pe_volume / pe_oi if pe_oi > 0 else 0,
                        'volume': pe_volume,
                        'oi': pe_oi
                    })

            # Sort by volume/OI ratio and return top 5
            unusual_strikes.sort(key=lambda x: x['volume_oi_ratio'], reverse=True)
            return unusual_strikes[:5]
        except:
            return []

    def calculate_overall_buildup_pattern(self, df_full_chain, spot_price):
        """Calculate overall buildup pattern across ITM, ATM, and OTM strikes"""
        try:
            # Separate into ITM, ATM, OTM
            itm_calls = df_full_chain[df_full_chain['strikePrice'] < spot_price].copy()
            otm_calls = df_full_chain[df_full_chain['strikePrice'] > spot_price].copy()
            atm_strikes = df_full_chain[abs(df_full_chain['strikePrice'] - spot_price) <= 50].copy()

            # Calculate OI changes for each zone
            itm_ce_change = itm_calls['changeinOpenInterest_CE'].sum() if not itm_calls.empty else 0
            itm_pe_change = itm_calls['changeinOpenInterest_PE'].sum() if not itm_calls.empty else 0

            otm_ce_change = otm_calls['changeinOpenInterest_CE'].sum() if not otm_calls.empty else 0
            otm_pe_change = otm_calls['changeinOpenInterest_PE'].sum() if not otm_calls.empty else 0

            atm_ce_change = atm_strikes['changeinOpenInterest_CE'].sum() if not atm_strikes.empty else 0
            atm_pe_change = atm_strikes['changeinOpenInterest_PE'].sum() if not atm_strikes.empty else 0

            # Determine pattern
            patterns = []

            if itm_pe_change > 0 and otm_ce_change > 0:
                patterns.append("Protective Strategy (Bullish)")
            elif itm_ce_change > 0 and otm_pe_change > 0:
                patterns.append("Protective Strategy (Bearish)")

            if atm_ce_change > atm_pe_change and abs(atm_ce_change) > 1000:
                patterns.append("Strong Call Writing (Bearish)")
            elif atm_pe_change > atm_ce_change and abs(atm_pe_change) > 1000:
                patterns.append("Strong Put Writing (Bullish)")

            if otm_ce_change > itm_ce_change and otm_ce_change > 1000:
                patterns.append("OTM Call Buying (Bullish)")
            elif otm_pe_change > itm_pe_change and otm_pe_change > 1000:
                patterns.append("OTM Put Buying (Bearish)")

            return " | ".join(patterns) if patterns else "Balanced/Neutral"

        except:
            return "Neutral"

    def analyze_comprehensive_atm_bias(self, instrument):
        """Comprehensive ATM bias analysis with all metrics"""
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

            # Get ATM row data
            atm_df = df[df['strikePrice'] == atm_strike]
            if not atm_df.empty:
                atm_ce_price = atm_df['lastPrice_CE'].values[0]
                atm_pe_price = atm_df['lastPrice_PE'].values[0]
                atm_ce_oi = atm_df['openInterest_CE'].values[0]
                atm_pe_oi = atm_df['openInterest_PE'].values[0]
                atm_ce_change = atm_df['changeinOpenInterest_CE'].values[0]
                atm_pe_change = atm_df['changeinOpenInterest_PE'].values[0]
                atm_ce_vega = atm_df['Vega_CE'].values[0]
                atm_pe_vega = atm_df['Vega_PE'].values[0]
            else:
                return None

            # Calculate all comprehensive metrics
            synthetic_bias, synthetic_future, synthetic_diff = self.calculate_synthetic_future_bias(
                atm_ce_price, atm_pe_price, atm_strike, spot
            )
            
            atm_buildup = self.calculate_atm_buildup_pattern(
                atm_ce_oi, atm_pe_oi, atm_ce_change, atm_pe_change
            )
            
            atm_vega_bias, atm_vega_exposure = self.calculate_atm_vega_bias(
                atm_ce_vega, atm_pe_vega, atm_ce_oi, atm_pe_oi
            )
            
            max_pain_strike = self.calculate_max_pain(df)
            distance_from_max_pain = spot - max_pain_strike if max_pain_strike else 0
            
            call_resistance, put_support = self.find_call_resistance_put_support(df, spot)
            
            total_vega_bias, total_vega, total_ce_vega_exp, total_pe_vega_exp = self.calculate_total_vega_bias(df)
            
            unusual_activity = self.detect_unusual_activity(df, spot)
            
            overall_buildup = self.calculate_overall_buildup_pattern(df, spot)

            # Calculate comprehensive bias score
            weights = {
                "oi_bias": 2, "chg_oi_bias": 2, "volume_bias": 1, 
                "iv_bias": 1, "premium_bias": 1, "delta_bias": 1,
                "synthetic_bias": 2, "vega_bias": 1, "max_pain_bias": 1
            }

            total_score = 0
            
            # OI Bias
            oi_bias = "Bullish" if data['total_pe_oi'] > data['total_ce_oi'] else "Bearish"
            total_score += weights["oi_bias"] if oi_bias == "Bullish" else -weights["oi_bias"]
            
            # Change in OI Bias
            chg_oi_bias = "Bullish" if data['total_pe_change'] > data['total_ce_change'] else "Bearish"
            total_score += weights["chg_oi_bias"] if chg_oi_bias == "Bullish" else -weights["chg_oi_bias"]
            
            # Synthetic Bias
            total_score += weights["synthetic_bias"] if synthetic_bias == "Bullish" else -weights["synthetic_bias"] if synthetic_bias == "Bearish" else 0
            
            # Vega Bias
            vega_bias_score = 1 if "Bullish" in atm_vega_bias else -1 if "Bearish" in atm_vega_bias else 0
            total_score += weights["vega_bias"] * vega_bias_score
            
            # Max Pain Bias (if spot above max pain, bullish)
            max_pain_bias = "Bullish" if distance_from_max_pain > 0 else "Bearish" if distance_from_max_pain < 0 else "Neutral"
            total_score += weights["max_pain_bias"] if max_pain_bias == "Bullish" else -weights["max_pain_bias"] if max_pain_bias == "Bearish" else 0

            overall_bias = self.final_verdict(total_score)

            return {
                'instrument': instrument,
                'spot_price': spot,
                'atm_strike': atm_strike,
                'overall_bias': overall_bias,
                'bias_score': total_score,
                'pcr_oi': data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 0,
                'pcr_change': abs(data['total_pe_change']) / abs(data['total_ce_change']) if data['total_ce_change'] != 0 else 0,
                'total_ce_oi': data['total_ce_oi'],
                'total_pe_oi': data['total_pe_oi'],
                'total_ce_change': data['total_ce_change'],
                'total_pe_change': data['total_pe_change'],
                'comprehensive_metrics': {
                    'synthetic_bias': synthetic_bias,
                    'synthetic_future': synthetic_future,
                    'synthetic_diff': synthetic_diff,
                    'atm_buildup': atm_buildup,
                    'atm_vega_bias': atm_vega_bias,
                    'atm_vega_exposure': atm_vega_exposure,
                    'max_pain_strike': max_pain_strike,
                    'distance_from_max_pain': distance_from_max_pain,
                    'call_resistance': call_resistance,
                    'put_support': put_support,
                    'total_vega_bias': total_vega_bias,
                    'total_vega': total_vega,
                    'unusual_activity_count': len(unusual_activity),
                    'overall_buildup': overall_buildup
                }
            }

        except Exception as e:
            return None

    def get_overall_market_bias(self):
        """Get comprehensive market bias across all instruments"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            bias_data = self.analyze_comprehensive_atm_bias(instrument)
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

    def format_market_bias_for_alerts(self):
        """Format market bias data for Telegram alerts"""
        try:
            bias_data = st.session_state.market_bias_data
            if not bias_data:
                return "Market bias data not available"
            
            message = "📊 COMPREHENSIVE OPTIONS MARKET BIAS:\n\n"
            
            for instrument_data in bias_data:
                message += f"🎯 {instrument_data['instrument']}:\n"
                message += f"   • Spot: ₹{instrument_data['spot_price']:.2f}\n"
                message += f"   • Overall Bias: {instrument_data['overall_bias']} (Score: {instrument_data['bias_score']:.2f})\n"
                message += f"   • PCR OI: {instrument_data['pcr_oi']:.2f} | PCR Δ: {instrument_data['pcr_change']:.2f}\n"
                
                # Add comprehensive metrics
                comp_metrics = instrument_data.get('comprehensive_metrics', {})
                if comp_metrics:
                    message += f"   • Synthetic Bias: {comp_metrics.get('synthetic_bias', 'N/A')}\n"
                    message += f"   • ATM Buildup: {comp_metrics.get('atm_buildup', 'N/A')}\n"
                    message += f"   • Vega Bias: {comp_metrics.get('atm_vega_bias', 'N/A')}\n"
                    message += f"   • Max Pain: {comp_metrics.get('max_pain_strike', 'N/A')} (Dist: {comp_metrics.get('distance_from_max_pain', 0):+.1f})\n"
                    message += f"   • Call Res: {comp_metrics.get('call_resistance', 'N/A')} | Put Sup: {comp_metrics.get('put_support', 'N/A')}\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            return f"Market bias analysis temporarily unavailable"

    def check_volume_block_alerts(self, current_price, bullish_blocks, bearish_blocks, threshold=5):
        """Check if price is near volume order blocks and send alerts with comprehensive ATM bias"""
        if not bullish_blocks and not bearish_blocks:
            return False
        
        current_time = datetime.now(self.ist)
        alert_sent = False
        
        # Get comprehensive market bias
        market_bias = self.format_market_bias_for_alerts()
        
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
        """Check for sudden volume spikes and send alerts with comprehensive ATM bias"""
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
                # Get comprehensive market bias
                market_bias = self.format_market_bias_for_alerts()
                
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

    def display_comprehensive_options_analysis(self):
        """Display comprehensive NSE Options Analysis with all ATM bias metrics"""
        st.header("📊 NSE Options Chain Analysis - Comprehensive ATM Bias")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Real-time options market bias analysis with comprehensive ATM metrics")
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
            
            st.subheader("🎯 Current Market Bias Summary")
            
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
            
            # Detailed comprehensive analysis for each instrument
            for instrument_data in bias_data:
                with st.expander(f"📊 {instrument_data['instrument']} - Comprehensive ATM Analysis", expanded=True):
                    comp_metrics = instrument_data.get('comprehensive_metrics', {})
                    
                    # Key Metrics Row
                    st.markdown("#### 🎯 Key ATM Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                        st.metric("ATM Strike", f"₹{instrument_data['atm_strike']:.2f}")
                    
                    with col2:
                        st.metric("PCR (OI)", f"{instrument_data['pcr_oi']:.2f}")
                        st.metric("PCR (Δ OI)", f"{instrument_data['pcr_change']:.2f}")
                    
                    with col3:
                        st.metric("Synthetic Bias", comp_metrics.get('synthetic_bias', 'N/A'))
                        st.metric("ATM Buildup", comp_metrics.get('atm_buildup', 'N/A'))
                    
                    with col4:
                        st.metric("Vega Bias", comp_metrics.get('atm_vega_bias', 'N/A'))
                        st.metric("Max Pain", f"₹{comp_metrics.get('max_pain_strike', 'N/A')}")
                    
                    # Advanced Metrics Row
                    st.markdown("#### 📈 Advanced Options Metrics")
                    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                    
                    with adv_col1:
                        dist_mp = comp_metrics.get('distance_from_max_pain', 0)
                        mp_color = "🟢" if dist_mp > 0 else "🔴" if dist_mp < 0 else "🟡"
                        st.metric("Dist from Max Pain", f"{mp_color} {dist_mp:+.1f}")
                    
                    with adv_col2:
                        call_res = comp_metrics.get('call_resistance', 'N/A')
                        st.metric("Call Resistance", f"₹{call_res}" if call_res != 'N/A' else 'N/A')
                    
                    with adv_col3:
                        put_sup = comp_metrics.get('put_support', 'N/A')
                        st.metric("Put Support", f"₹{put_sup}" if put_sup != 'N/A' else 'N/A')
                    
                    with adv_col4:
                        total_vega = comp_metrics.get('total_vega', 0)
                        st.metric("Total Vega", f"{total_vega:,.0f}")
                    
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
                    
                    # Trading recommendation based on comprehensive analysis
                    st.markdown("#### 💡 Comprehensive Trading Recommendation")
                    
                    overall_bias = instrument_data['overall_bias']
                    bias_score = instrument_data['bias_score']
                    pcr_oi = instrument_data['pcr_oi']
                    
                    if "Strong Bullish" in overall_bias or (bias_score >= 2 and pcr_oi > 1.2):
                        st.success(f"""
                        **🎯 STRONG BULLISH SIGNAL - {instrument_data['instrument']}**
                        
                        ✅ **High Confidence LONG/CALL positions**
                        ✅ Enter near support levels: ₹{comp_metrics.get('put_support', 'N/A')}
                        ✅ Target resistance: ₹{comp_metrics.get('call_resistance', 'N/A')}
                        ⚠️ Stop loss below key support
                        
                        **Supporting Factors:**
                        • Positive PCR ratios
                        • Bullish synthetic future
                        • Put-heavy vega exposure
                        • Favorable max pain position
                        """)
                    elif "Bullish" in overall_bias:
                        st.info(f"""
                        **📈 BULLISH BIAS - {instrument_data['instrument']}**
                        
                        ✅ **Consider LONG/CALL positions**
                        ✅ Wait for pullback to enter
                        ✅ Moderate position sizing
                        ⚠️ Use tight stop losses
                        
                        **Key Levels:**
                        • Support: ₹{comp_metrics.get('put_support', 'N/A')}
                        • Resistance: ₹{comp_metrics.get('call_resistance', 'N/A')}
                        """)
                    elif "Strong Bearish" in overall_bias or (bias_score <= -2 and pcr_oi < 0.8):
                        st.error(f"""
                        **🎯 STRONG BEARISH SIGNAL - {instrument_data['instrument']}**
                        
                        ✅ **High Confidence SHORT/PUT positions**
                        ✅ Enter near resistance levels: ₹{comp_metrics.get('call_resistance', 'N/A')}
                        ✅ Target support: ₹{comp_metrics.get('put_support', 'N/A')}
                        ⚠️ Stop loss above key resistance
                        
                        **Supporting Factors:**
                        • Negative PCR ratios
                        • Bearish synthetic future  
                        • Call-heavy vega exposure
                        • Unfavorable max pain position
                        """)
                    elif "Bearish" in overall_bias:
                        st.warning(f"""
                        **📉 BEARISH BIAS - {instrument_data['instrument']}**
                        
                        ✅ **Consider SHORT/PUT positions**
                        ✅ Wait for rally to enter
                        ✅ Moderate position sizing
                        ⚠️ Use tight stop losses
                        
                        **Key Levels:**
                        • Resistance: ₹{comp_metrics.get('call_resistance', 'N/A')}
                        • Support: ₹{comp_metrics.get('put_support', 'N/A')}
                        """)
                    else:
                        st.warning(f"""
                        **⚖️ NEUTRAL/UNCLEAR BIAS - {instrument_data['instrument']}**
                        
                        🔄 **Wait for clear directional bias**
                        🔄 Consider range-bound strategies
                        🔄 Reduce position sizes
                        🔄 Monitor key levels closely
                        
                        **Key Observation Points:**
                        • PCR near 1.0 (balanced)
                        • Mixed signals across metrics
                        • Wait for breakout confirmation
                        """)
                    
                    # Unusual Activity
                    unusual_count = comp_metrics.get('unusual_activity_count', 0)
                    if unusual_count > 0:
                        st.markdown(f"#### ⚠️ Unusual Activity Detected: {unusual_count} strikes")
                    
                    # Overall Buildup Pattern
                    overall_buildup = comp_metrics.get('overall_buildup', 'N/A')
                    st.markdown(f"#### 🔍 Overall Buildup Pattern: **{overall_buildup}**")
        
        else:
            st.info("👆 Click 'Update Options Data' to load comprehensive options analysis")
            
            st.markdown("""
            ### About Comprehensive Options Analysis
            
            This section provides institutional-grade options market analysis:
            
            **Key Features:**
            - 🎯 **Comprehensive ATM Bias Analysis**
            - 📊 **PCR (Put-Call Ratio)** with OI and changes
            - 🔍 **Synthetic Future Pricing**
            - 📈 **Vega Exposure Analysis**
            - 🎯 **Max Pain Theory**
            - 🛡️ **Support/Resistance from OI**
            - ⚠️ **Unusual Activity Detection**
            
            **Instruments Analyzed:**
            - NIFTY, BANKNIFTY, FINNIFTY
            
            **How to Use:**
            1. Click 'Update Options Data' for latest analysis
            2. Review comprehensive metrics for each instrument
            3. Check trading recommendations
            4. Use insights for informed trading decisions
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
        st.markdown("*Volume Analysis, Comprehensive Options Chain & Real-time Alerts*")
        
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
                    if (volume_spike_alerts or volume_block_alerts) and telegram_enabled:
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
            self.display_comprehensive_options_analysis()
        
        # Cleanup old alerts and auto refresh
        self.alert_manager.cleanup_old_alerts()
        time.sleep(30)
        st.rerun()

# Run the app
if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
