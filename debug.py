import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
import numpy as np
import math
from scipy.stats import norm
import warnings
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Nifty Options Analysis",
    page_icon="📈",
    layout="wide"
)

# =============================================
# NSE OPTIONS ANALYZER WITH AUTO-REFRESH
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
            }
        }
        self.last_refresh_time = {}
        self.refresh_interval = 2  # 2 minutes default refresh
        self.cached_bias_data = {}
        
    def set_refresh_interval(self, minutes: int):
        """Set auto-refresh interval"""
        self.refresh_interval = minutes
    
    def should_refresh_data(self, instrument: str) -> bool:
        """Check if data should be refreshed based on last refresh time"""
        current_time = datetime.now(self.ist)
        
        if instrument not in self.last_refresh_time:
            self.last_refresh_time[instrument] = current_time
            return True
        
        last_refresh = self.last_refresh_time[instrument]
        time_diff = (current_time - last_refresh).total_seconds() / 60
        
        if time_diff >= self.refresh_interval:
            self.last_refresh_time[instrument] = current_time
            return True
        
        return False
        
    def calculate_greeks(self, option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
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

    def fetch_option_chain_data(self, instrument: str) -> Dict[str, Any]:
        """Fetch option chain data from NSE"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=10)

            url_instrument = instrument.replace(' ', '%20')
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={url_instrument}"

            response = session.get(url, timeout=15)
            response.raise_for_status()
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

    def delta_volume_bias(self, price: float, volume: float, chg_oi: float) -> str:
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

    def final_verdict(self, score: float) -> str:
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

    def determine_level(self, row: pd.Series) -> str:
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

    def calculate_max_pain(self, df_full_chain: pd.DataFrame) -> Optional[float]:
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

    def calculate_synthetic_future_bias(self, atm_ce_price: float, atm_pe_price: float, atm_strike: float, spot_price: float) -> Tuple[str, float, float]:
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

    def calculate_atm_buildup_pattern(self, atm_ce_oi: float, atm_pe_oi: float, atm_ce_change: float, atm_pe_change: float) -> str:
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

    def calculate_atm_vega_bias(self, atm_ce_vega: float, atm_pe_vega: float, atm_ce_oi: float, atm_pe_oi: float) -> Tuple[str, float]:
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

    def find_call_resistance_put_support(self, df_full_chain: pd.DataFrame, spot_price: float) -> Tuple[Optional[float], Optional[float]]:
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

    def calculate_total_vega_bias(self, df_full_chain: pd.DataFrame) -> Tuple[str, float, float, float]:
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

    def detect_unusual_activity(self, df_full_chain: pd.DataFrame, spot_price: float) -> List[Dict[str, Any]]:
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

    def calculate_overall_buildup_pattern(self, df_full_chain: pd.DataFrame, spot_price: float) -> str:
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

    def analyze_comprehensive_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Comprehensive ATM bias analysis with all metrics"""
        try:
            data = self.fetch_option_chain_data(instrument)
            if not data['success']:
                st.warning(f"Failed to fetch data for {instrument}: {data.get('error', 'Unknown error')}")
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
                st.warning(f"No valid option data found for {instrument}")
                return None

            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            
            # Merge calls and puts on strike price
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'), how='outer').sort_values('strikePrice')
            df = df.fillna(0)  # Fill NaN values with 0 for missing options

            # Find ATM strike
            atm_range = self.NSE_INSTRUMENTS['indices'].get(instrument, {}).get('atm_range', 200)
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            df_atm = df[abs(df['strikePrice'] - atm_strike) <= atm_range]

            if df_atm.empty:
                st.warning(f"No ATM data found for {instrument}")
                return None

            # Get ATM row data
            atm_df = df[df['strikePrice'] == atm_strike]
            if not atm_df.empty:
                atm_row = atm_df.iloc[0]
                atm_ce_price = atm_row.get('lastPrice_CE', 0)
                atm_pe_price = atm_row.get('lastPrice_PE', 0)
                atm_ce_oi = atm_row.get('openInterest_CE', 0)
                atm_pe_oi = atm_row.get('openInterest_PE', 0)
                atm_ce_change = atm_row.get('changeinOpenInterest_CE', 0)
                atm_pe_change = atm_row.get('changeinOpenInterest_PE', 0)
                atm_ce_vega = atm_row.get('Vega_CE', 0)
                atm_pe_vega = atm_row.get('Vega_PE', 0)
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

            # Calculate detailed ATM bias breakdown
            detailed_atm_bias = self.calculate_detailed_atm_bias(df_atm, atm_strike, spot)

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
                'detailed_atm_bias': detailed_atm_bias,
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
            st.error(f"Error in ATM bias analysis for {instrument}: {e}")
            return None

    def calculate_detailed_atm_bias(self, df_atm: pd.DataFrame, atm_strike: float, spot_price: float) -> Dict[str, Any]:
        """Calculate detailed ATM bias breakdown for all metrics"""
        try:
            detailed_bias = {}
            
            for _, row in df_atm.iterrows():
                if row['strikePrice'] == atm_strike:
                    # Calculate per-strike delta and gamma exposure
                    ce_delta_exp = row.get('Delta_CE', 0) * row.get('openInterest_CE', 0)
                    pe_delta_exp = row.get('Delta_PE', 0) * row.get('openInterest_PE', 0)
                    ce_gamma_exp = row.get('Gamma_CE', 0) * row.get('openInterest_CE', 0)
                    pe_gamma_exp = row.get('Gamma_PE', 0) * row.get('openInterest_PE', 0)

                    net_delta_exp = ce_delta_exp + pe_delta_exp
                    net_gamma_exp = ce_gamma_exp + pe_gamma_exp
                    strike_iv_skew = row.get('impliedVolatility_PE', 0) - row.get('impliedVolatility_CE', 0)

                    delta_exp_bias = "Bullish" if net_delta_exp > 0 else "Bearish" if net_delta_exp < 0 else "Neutral"
                    gamma_exp_bias = "Bullish" if net_gamma_exp > 0 else "Bearish" if net_gamma_exp < 0 else "Neutral"
                    iv_skew_bias = "Bullish" if strike_iv_skew > 0 else "Bearish" if strike_iv_skew < 0 else "Neutral"

                    detailed_bias = {
                        "Strike": row['strikePrice'],
                        "Zone": 'ATM',
                        "Level": self.determine_level(row),
                        "OI_Bias": "Bullish" if row.get('openInterest_CE', 0) < row.get('openInterest_PE', 0) else "Bearish",
                        "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
                        "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
                        "Delta_Bias": "Bullish" if abs(row.get('Delta_PE', 0)) > abs(row.get('Delta_CE', 0)) else "Bearish",
                        "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) < row.get('Gamma_PE', 0) else "Bearish",
                        "Premium_Bias": "Bullish" if row.get('lastPrice_CE', 0) < row.get('lastPrice_PE', 0) else "Bearish",
                        "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
                        "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
                        "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "Bearish",
                        "DVP_Bias": self.delta_volume_bias(
                            row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                            row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                            row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
                        ),
                        "Delta_Exposure_Bias": delta_exp_bias,
                        "Gamma_Exposure_Bias": gamma_exp_bias,
                        "IV_Skew_Bias": iv_skew_bias,
                        # Raw values for display
                        "CE_OI": row.get('openInterest_CE', 0),
                        "PE_OI": row.get('openInterest_PE', 0),
                        "CE_Change": row.get('changeinOpenInterest_CE', 0),
                        "PE_Change": row.get('changeinOpenInterest_PE', 0),
                        "CE_Volume": row.get('totalTradedVolume_CE', 0),
                        "PE_Volume": row.get('totalTradedVolume_PE', 0),
                        "CE_Price": row.get('lastPrice_CE', 0),
                        "PE_Price": row.get('lastPrice_PE', 0),
                        "CE_IV": row.get('impliedVolatility_CE', 0),
                        "PE_IV": row.get('impliedVolatility_PE', 0),
                        "Delta_CE": row.get('Delta_CE', 0),
                        "Delta_PE": row.get('Delta_PE', 0),
                        "Gamma_CE": row.get('Gamma_CE', 0),
                        "Gamma_PE": row.get('Gamma_PE', 0)
                    }
                    break
            
            return detailed_bias
            
        except Exception as e:
            st.error(f"Error in detailed ATM bias: {e}")
            return {}

    def get_overall_market_bias(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive market bias across all instruments with auto-refresh"""
        instruments = list(self.NSE_INSTRUMENTS['indices'].keys())
        results = []
        
        for instrument in instruments:
            if force_refresh or self.should_refresh_data(instrument):
                try:
                    bias_data = self.analyze_comprehensive_atm_bias(instrument)
                    if bias_data:
                        results.append(bias_data)
                        # Update cache
                        self.cached_bias_data[instrument] = bias_data
                except Exception as e:
                    st.error(f"Error fetching {instrument}: {e}")
                    # Use cached data if available
                    if instrument in self.cached_bias_data:
                        results.append(self.cached_bias_data[instrument])
            else:
                # Return cached data if available and not forcing refresh
                if instrument in self.cached_bias_data:
                    results.append(self.cached_bias_data[instrument])
        
        return results

# =============================================
# PRICE DATA FETCHER USING NSE API (SAME AS OLD APP)
# =============================================

class PriceDataFetcher:
    """Fetch real price data for charts using NSE API"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def fetch_nifty_intraday_data(self, symbol: str = "NIFTY 50") -> pd.DataFrame:
        """Fetch Nifty 50 intraday data from NSE API"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            # Get Nifty index data
            url = "https://www.nseindia.com/api/chart-databyindex?index=NIFTY%2050"
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process the data into DataFrame
                timestamps = data['grapthData']
                
                df_data = []
                for ts_data in timestamps:
                    timestamp = datetime.fromtimestamp(ts_data[0] / 1000, tz=self.ist)
                    price = ts_data[1]
                    df_data.append({'timestamp': timestamp, 'close': price})
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                
                # Add OHLC data (for intraday, we approximate OHLC from close prices)
                if len(df) > 0:
                    df['open'] = df['close'].shift(1)
                    df['high'] = df[['open', 'close']].max(axis=1)
                    df['low'] = df[['open', 'close']].min(axis=1)
                    df['volume'] = 0  # NSE doesn't provide volume for indices in this endpoint
                    
                    # Handle first row
                    df.iloc[0, df.columns.get_loc('open')] = df.iloc[0]['close']
                    df.iloc[0, df.columns.get_loc('high')] = df.iloc[0]['close']
                    df.iloc[0, df.columns.get_loc('low')] = df.iloc[0]['close']
                
                return df
            else:
                st.warning(f"NSE API returned status code: {response.status_code}")
                return self.generate_sample_data()
                
        except Exception as e:
            st.warning(f"Error fetching NSE data: {e}. Using sample data.")
            return self.generate_sample_data()
    
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample price data when API fails"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), 
                             end=datetime.now(), 
                             freq='5min', 
                             tz=self.ist)
        
        # Generate realistic Nifty-like prices
        np.random.seed(42)
        base_price = 22000
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * (1 + np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100000, 500000, len(dates))
        })
        df.set_index('timestamp', inplace=True)
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df

# =============================================
# SIMPLIFIED NIFTY APP WITH PROPER CHARTS (SAME AS OLD APP)
# =============================================

class SimplifiedNiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.options_analyzer = NSEOptionsAnalyzer()
        self.price_fetcher = PriceDataFetcher()
        
        # Initialize session state
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'market_bias_data' not in st.session_state:
            st.session_state.market_bias_data = None
        if 'last_bias_update' not in st.session_state:
            st.session_state.last_bias_update = None
        if 'price_data' not in st.session_state:
            st.session_state.price_data = None
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '5min'

    def fetch_and_process_price_data(self):
        """Fetch and process price data for charts"""
        try:
            df = self.price_fetcher.fetch_nifty_intraday_data()
            
            if not df.empty:
                st.session_state.price_data = df
                return True
            return False
        except Exception as e:
            st.error(f"Error processing price data: {e}")
            return False

    def create_comprehensive_chart(self, df: pd.DataFrame, interval: str) -> go.Figure:
        """Create comprehensive candlestick chart with volume - EXACTLY LIKE OLD APP"""
        if df.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No data available", height=600)
            return fig
        
        # Create subplots - EXACTLY like old app
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'NIFTY 50 - {interval} Chart', 'Volume'),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Candlestick chart - EXACT colors and styling from old app
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='NIFTY 50',
                increasing_line_color='#00ff88',  # Same green as old app
                decreasing_line_color='#ff4444',  # Same red as old app
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add moving averages - EXACTLY like old app
        if len(df) > 20:
            df['MA20'] = df['close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA20'],
                    line=dict(color='orange', width=1.5),
                    name='MA 20'
                ),
                row=1, col=1
            )
        
        if len(df) > 50:
            df['MA50'] = df['close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA50'],
                    line=dict(color='blue', width=1.5),
                    name='MA 50'
                ),
                row=1, col=1
            )
        
        # Volume bars with color coding - EXACTLY like old app
        colors = ['#00ff88' if close >= open_ else '#ff4444' 
                 for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout - EXACTLY like old app
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0),
            title=f"NIFTY 50 Price Action - {interval}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis2_title="Time",
            yaxis2_title="Volume",
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update axes - EXACTLY like old app
        fig.update_xaxes(
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.3)',
            showspikes=True,
            spikethickness=1,
            spikecolor="white"
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.3)',
            showspikes=True,
            spikethickness=1,
            spikecolor="white"
        )
        
        return fig

    def display_comprehensive_options_analysis(self):
        """Display comprehensive NSE Options Analysis with detailed ATM bias tabulation"""
        st.header("📊 NSE Options Chain Analysis - Auto Refresh")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"Options data auto-refreshes every {self.options_analyzer.refresh_interval} minutes")
        with col2:
            if st.button("🔄 Force Refresh", type="primary"):
                with st.spinner("Force refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    if bias_data:
                        st.success("Options data refreshed!")
                    else:
                        st.warning("No data received. Check internet connection.")
        with col3:
            if st.session_state.last_bias_update:
                st.write(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')}")
        
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
            
            # Detailed analysis for each instrument
            for instrument_data in bias_data:
                comp_metrics = instrument_data.get('comprehensive_metrics', {})
                detailed_bias = instrument_data.get('detailed_atm_bias', {})
                
                with st.expander(f"🎯 {instrument_data['instrument']} - Detailed ATM Bias Analysis", expanded=True):
                    
                    # Basic Information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Spot Price", f"₹{instrument_data['spot_price']:.2f}")
                    with col2:
                        st.metric("ATM Strike", f"₹{instrument_data['atm_strike']:.2f}")
                    with col3:
                        st.metric("PCR OI", f"{instrument_data['pcr_oi']:.2f}")
                    with col4:
                        st.metric("PCR Δ OI", f"{instrument_data['pcr_change']:.2f}")
                    
                    st.divider()
                    
                    # Trading Recommendation
                    st.subheader("💡 Trading Recommendation")
                    
                    overall_bias = instrument_data['overall_bias']
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if "Strong Bullish" in overall_bias:
                            st.success(f"""
                            **🎯 HIGH CONFIDENCE BULLISH SIGNAL**
                            
                            **Recommended Action:** Aggressive LONG/CALL positions
                            **Entry Zone:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 50):.0f} - ₹{instrument_data['spot_price']:.0f}
                            **Target 1:** ₹{instrument_data['spot_price'] + (comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100) - instrument_data['spot_price']) * 0.5:.0f}
                            **Target 2:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100):.0f}
                            **Stop Loss:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100) - 20:.0f}
                            """)
                        elif "Bullish" in overall_bias:
                            st.info(f"""
                            **📈 BULLISH BIAS**
                            
                            **Recommended Action:** Consider LONG/CALL positions
                            **Entry Zone:** Wait for pullback to support
                            **Target:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 80):.0f}
                            **Stop Loss:** Below key support
                            """)
                        elif "Strong Bearish" in overall_bias:
                            st.error(f"""
                            **🎯 HIGH CONFIDENCE BEARISH SIGNAL**
                            
                            **Recommended Action:** Aggressive SHORT/PUT positions
                            **Entry Zone:** ₹{instrument_data['spot_price']:.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 50):.0f}
                            **Target 1:** ₹{instrument_data['spot_price'] - (instrument_data['spot_price'] - comp_metrics.get('put_support', instrument_data['spot_price'] - 100)) * 0.5:.0f}
                            **Target 2:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 100):.0f}
                            **Stop Loss:** ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 100) + 20:.0f}
                            """)
                        elif "Bearish" in overall_bias:
                            st.warning(f"""
                            **📉 BEARISH BIAS**
                            
                            **Recommended Action:** Consider SHORT/PUT positions
                            **Entry Zone:** Wait for rally to resistance
                            **Target:** ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 80):.0f}
                            **Stop Loss:** Above key resistance
                            """)
                        else:
                            st.warning(f"""
                            **⚖️ NEUTRAL/UNCLEAR BIAS**
                            
                            **Recommended Action:** Wait for clear directional bias
                            **Strategy:** Consider range-bound strategies
                            **Key Levels:** Monitor ₹{comp_metrics.get('put_support', instrument_data['spot_price'] - 50):.0f} - ₹{comp_metrics.get('call_resistance', instrument_data['spot_price'] + 50):.0f}
                            """)
                    
                    with col2:
                        st.metric("Overall Bias", overall_bias)
                        st.metric("Bias Score", f"{instrument_data['bias_score']:.2f}")
        
        else:
            st.info("👆 Options data will auto-refresh. Click 'Force Refresh' to load immediately.")

    def display_option_chain_bias_tabulation(self):
        """Display all option chain bias data in comprehensive tabulation"""
        st.header("📋 Comprehensive Option Chain Bias Data")
        
        if not st.session_state.market_bias_data:
            st.info("No option chain data available. Please refresh options analysis first.")
            return
        
        for instrument_data in st.session_state.market_bias_data:
            with st.expander(f"🎯 {instrument_data['instrument']} - Complete Bias Analysis", expanded=True):
                
                # Basic Information Table
                st.subheader("📊 Basic Information")
                basic_info = pd.DataFrame({
                    'Metric': [
                        'Instrument', 'Spot Price', 'ATM Strike', 'Overall Bias', 
                        'Bias Score', 'PCR OI', 'PCR Change OI'
                    ],
                    'Value': [
                        instrument_data['instrument'],
                        f"₹{instrument_data['spot_price']:.2f}",
                        f"₹{instrument_data['atm_strike']:.2f}",
                        instrument_data['overall_bias'],
                        f"{instrument_data['bias_score']:.2f}",
                        f"{instrument_data['pcr_oi']:.2f}",
                        f"{instrument_data['pcr_change']:.2f}"
                    ]
                })
                st.dataframe(basic_info, use_container_width=True, hide_index=True)
                
                # Detailed ATM Bias Table
                if 'detailed_atm_bias' in instrument_data and instrument_data['detailed_atm_bias']:
                    st.subheader("🔍 Detailed ATM Bias Analysis")
                    detailed_bias = instrument_data['detailed_atm_bias']
                    
                    # Create comprehensive table for detailed bias
                    bias_metrics = []
                    bias_values = []
                    bias_signals = []
                    
                    for key, value in detailed_bias.items():
                        if key not in ['Strike', 'Zone', 'CE_OI', 'PE_OI', 'CE_Change', 'PE_Change', 
                                     'CE_Volume', 'PE_Volume', 'CE_Price', 'PE_Price', 'CE_IV', 'PE_IV',
                                     'Delta_CE', 'Delta_PE', 'Gamma_CE', 'Gamma_PE']:
                            bias_metrics.append(key.replace('_', ' ').title())
                            bias_values.append(str(value))
                            
                            # Determine signal strength
                            if 'Bullish' in str(value):
                                bias_signals.append('🟢 Bullish')
                            elif 'Bearish' in str(value):
                                bias_signals.append('🔴 Bearish')
                            else:
                                bias_signals.append('🟡 Neutral')
                    
                    detailed_df = pd.DataFrame({
                        'Metric': bias_metrics,
                        'Value': bias_values,
                        'Signal': bias_signals
                    })
                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Raw values table
                    st.subheader("📈 Raw Option Data")
                    raw_data = []
                    if 'CE_OI' in detailed_bias:
                        raw_data.append(['Call OI', f"{detailed_bias['CE_OI']:,.0f}"])
                        raw_data.append(['Put OI', f"{detailed_bias['PE_OI']:,.0f}"])
                        raw_data.append(['Call OI Change', f"{detailed_bias['CE_Change']:,.0f}"])
                        raw_data.append(['Put OI Change', f"{detailed_bias['PE_Change']:,.0f}"])
                        raw_data.append(['Call Volume', f"{detailed_bias['CE_Volume']:,.0f}"])
                        raw_data.append(['Put Volume', f"{detailed_bias['PE_Volume']:,.0f}"])
                        raw_data.append(['Call Price', f"₹{detailed_bias['CE_Price']:.2f}"])
                        raw_data.append(['Put Price', f"₹{detailed_bias['PE_Price']:.2f}"])
                        raw_data.append(['Call IV', f"{detailed_bias['CE_IV']:.2f}%"])
                        raw_data.append(['Put IV', f"{detailed_bias['PE_IV']:.2f}%"])
                        raw_data.append(['Call Delta', f"{detailed_bias['Delta_CE']:.4f}"])
                        raw_data.append(['Put Delta', f"{detailed_bias['Delta_PE']:.4f}"])
                        raw_data.append(['Call Gamma', f"{detailed_bias['Gamma_CE']:.4f}"])
                        raw_data.append(['Put Gamma', f"{detailed_bias['Gamma_PE']:.4f}"])
                    
                    raw_df = pd.DataFrame(raw_data, columns=['Parameter', 'Value'])
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)
                
                # Comprehensive Metrics Table
                if 'comprehensive_metrics' in instrument_data and instrument_data['comprehensive_metrics']:
                    st.subheader("🎯 Advanced Option Metrics")
                    comp_metrics = instrument_data['comprehensive_metrics']
                    
                    comp_data = []
                    for key, value in comp_metrics.items():
                        if key not in ['total_vega', 'total_ce_vega_exp', 'total_pe_vega_exp']:
                            comp_data.append([
                                key.replace('_', ' ').title(),
                                str(value) if not isinstance(value, (int, float)) else f"{value:.2f}"
                            ])
                    
                    comp_df = pd.DataFrame(comp_data, columns=['Metric', 'Value'])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    def run(self):
        """Main application with proper price action charts"""
        st.title("📈 Nifty Options Analysis Dashboard")
        st.markdown("*Options Chain Analysis & ATM Bias Tabulation*")
        
        # Sidebar settings
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Chart settings
            st.subheader("📈 Chart Settings")
            timeframe = st.selectbox(
                "Timeframe",
                ['5min', '15min', '30min', '1hour'],
                index=0
            )
            
            if st.button("🔄 Refresh Chart", type="secondary"):
                with st.spinner("Fetching latest price data..."):
                    if self.fetch_and_process_price_data():
                        st.success("Chart data refreshed!")
                    else:
                        st.error("Failed to fetch chart data")
            
            # Options settings
            st.subheader("📊 Options Settings")
            refresh_interval = st.slider(
                "Refresh Interval (minutes)",
                min_value=1,
                max_value=10,
                value=2
            )
            self.options_analyzer.set_refresh_interval(refresh_interval)
            
            if st.button("🔄 Refresh All Data", type="primary"):
                with st.spinner("Refreshing all data..."):
                    # Refresh price data
                    self.fetch_and_process_price_data()
                    
                    # Refresh options data
                    bias_data = self.options_analyzer.get_overall_market_bias(force_refresh=True)
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = datetime.now(self.ist)
                    
                    if bias_data:
                        st.success("All data refreshed!")
                    else:
                        st.warning("Options data not available. Check internet connection.")
        
        # Main content - Tabs
        tab1, tab2, tab3 = st.tabs([
            "📈 Price Action", "📊 Options Analysis", "📋 Bias Tabulation"
        ])
        
        with tab1:
            st.header("📈 NIFTY 50 Price Action")
            
            # Fetch price data if not available
            if st.session_state.price_data is None:
                with st.spinner("Loading price data..."):
                    self.fetch_and_process_price_data()
            
            # Display chart
            if st.session_state.price_data is not None:
                df = st.session_state.price_data
                
                # Display current price metrics
                if not df.empty:
                    latest = df.iloc[-1]
                    prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                    price_change = latest['close'] - prev_close
                    price_change_pct = (price_change / prev_close) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Current Price",
                            f"₹{latest['close']:.2f}",
                            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                        )
                    with col2:
                        st.metric("Open", f"₹{latest['open']:.2f}")
                    with col3:
                        st.metric("High", f"₹{latest['high']:.2f}")
                    with col4:
                        st.metric("Low", f"₹{latest['low']:.2f}")
                
                # Create and display chart
                chart = self.create_comprehensive_chart(df, timeframe)
                st.plotly_chart(chart, use_container_width=True)
                
                # Chart info
                st.info(f"""
                **Chart Features:**
                - **Candlestick Chart**: Real-time NIFTY 50 price action
                - **Moving Averages**: 20-period and 50-period moving averages
                - **Volume Bars**: Color-coded volume (green for up, red for down)
                - **Timeframe**: {timeframe} intervals
                - **Auto-refresh**: Manual refresh available in sidebar
                """)
            else:
                st.error("Unable to load price data. Please check your internet connection.")
        
        with tab2:
            self.display_comprehensive_options_analysis()
            
            # Auto-refresh logic for options data
            current_time = datetime.now(self.ist)
            if (st.session_state.last_bias_update is None or 
                (current_time - st.session_state.last_bias_update).total_seconds() > self.options_analyzer.refresh_interval * 60):
                
                with st.spinner("Auto-refreshing options data..."):
                    bias_data = self.options_analyzer.get_overall_market_bias()
                    st.session_state.market_bias_data = bias_data
                    st.session_state.last_bias_update = current_time
                    if bias_data:
                        st.rerun()
        
        with tab3:
            self.display_option_chain_bias_tabulation()

# Run the app
if __name__ == "__main__":
    app = SimplifiedNiftyApp()
    app.run()
