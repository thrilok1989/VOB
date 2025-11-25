import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np
import math
from scipy.stats import norm
from typing import Dict, List, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Nifty Dashboard", page_icon="📈", layout="wide")

# =============================================
# ENHANCED MARKET DATA
# =============================================
class EnhancedMarketData:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def fetch_india_vix(self) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                vix = hist['Close'].iloc[-1]
                if vix > 25: sentiment, bias, score = "HIGH FEAR", "BEARISH", -75
                elif vix > 20: sentiment, bias, score = "ELEVATED", "BEARISH", -50
                elif vix > 15: sentiment, bias, score = "MODERATE", "NEUTRAL", 0
                else: sentiment, bias, score = "LOW", "BULLISH", 40
                return {'success': True, 'value': vix, 'sentiment': sentiment, 'bias': bias, 'score': score}
        except: pass
        return {'success': False}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        sectors = {'^CNXIT': 'IT', '^CNXBANK': 'BANK', '^CNXAUTO': 'AUTO', '^CNXPHARMA': 'PHARMA'}
        data = []
        for sym, name in sectors.items():
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    last, open_ = hist['Close'].iloc[-1], hist['Open'].iloc[0]
                    chg_pct = ((last - open_) / open_) * 100
                    bias = "BULLISH" if chg_pct > 0.5 else "BEARISH" if chg_pct < -0.5 else "NEUTRAL"
                    data.append({'sector': name, 'change_pct': chg_pct, 'bias': bias})
            except: pass
        return data

    def fetch_all(self) -> Dict[str, Any]:
        return {'vix': self.fetch_india_vix(), 'sectors': self.fetch_sector_indices()}

# =============================================
# BIAS ANALYSIS
# =============================================
class BiasAnalyzer:
    def __init__(self):
        self.config = {'rsi_period': 14, 'mfi_period': 10, 'dmi_period': 13}

    def fetch_data(self, symbol: str, period: str = '7d', interval: str = '5m') -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if 'Volume' not in df.columns: df['Volume'] = 0
            return df.fillna(0)
        except: return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_mfi(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        if df['Volume'].sum() == 0: return pd.Series([50.0] * len(df), index=df.index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
        return 100 - (100 / (1 + pos / neg.replace(0, np.nan))).fillna(50)

    def analyze(self, symbol: str = "^NSEI") -> Dict[str, Any]:
        try:
            df = self.fetch_data(symbol)
            if df.empty or len(df) < 100:
                return {'success': False, 'error': 'Insufficient data'}

            results = []
            
            # RSI
            rsi = self.calculate_rsi(df['Close'])
            rsi_val = rsi.iloc[-1]
            results.append({
                'indicator': 'RSI', 'value': f"{rsi_val:.2f}",
                'bias': "BULLISH" if rsi_val > 50 else "BEARISH", 
                'score': 100 if rsi_val > 50 else -100
            })
            
            # MFI
            mfi = self.calculate_mfi(df)
            mfi_val = mfi.iloc[-1]
            results.append({
                'indicator': 'MFI', 'value': f"{mfi_val:.2f}",
                'bias': "BULLISH" if mfi_val > 50 else "BEARISH",
                'score': 100 if mfi_val > 50 else -100
            })

            # Volume Delta
            up_vol = ((df['Close'] > df['Open']).astype(int) * df['Volume']).sum()
            down_vol = ((df['Close'] < df['Open']).astype(int) * df['Volume']).sum()
            vol_delta = up_vol - down_vol
            results.append({
                'indicator': 'Volume Delta', 'value': f"{vol_delta:.0f}",
                'bias': "BULLISH" if vol_delta > 0 else "BEARISH",
                'score': 100 if vol_delta > 0 else -100
            })

            bull_count = sum(1 for r in results if r['bias'] == 'BULLISH')
            bear_count = len(results) - bull_count
            total_score = sum(r['score'] for r in results) / len(results)
            
            if total_score >= 50: overall = "BULLISH"
            elif total_score <= -50: overall = "BEARISH"
            else: overall = "NEUTRAL"

            return {
                'success': True, 'symbol': symbol, 'current_price': df['Close'].iloc[-1],
                'bias_results': results, 'overall_bias': overall, 'overall_score': total_score,
                'bullish_count': bull_count, 'bearish_count': bear_count
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# =============================================
# NSE OPTIONS ANALYZER
# =============================================
class NSEOptionsAnalyzer:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.instruments = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']

    def calculate_greeks(self, opt_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Tuple:
        try:
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            delta = norm.cdf(d1) if opt_type == 'CE' else -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
            vega = S*norm.pdf(d1)*math.sqrt(T)/100
            return round(delta, 4), round(gamma, 4), round(vega, 4), 0, 0
        except: return 0, 0, 0, 0, 0

    def fetch_option_chain(self, instrument: str) -> Dict[str, Any]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            session = requests.Session()
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)
            
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={instrument}"
            response = session.get(url, timeout=10)
            data = response.json()
            
            records = data['records']['data']
            spot = data['records']['underlyingValue']
            expiry = data['records']['expiryDates'][0]
            
            total_ce_oi = sum(item['CE']['openInterest'] for item in records if 'CE' in item)
            total_pe_oi = sum(item['PE']['openInterest'] for item in records if 'PE' in item)
            
            return {
                'success': True, 'instrument': instrument, 'spot': spot, 'expiry': expiry,
                'total_ce_oi': total_ce_oi, 'total_pe_oi': total_pe_oi, 'records': records
            }
        except Exception as e:
            return {'success': False, 'instrument': instrument, 'error': str(e)}

    def analyze_atm_bias(self, instrument: str) -> Optional[Dict[str, Any]]:
        try:
            data = self.fetch_option_chain(instrument)
            if not data['success']: return None
            
            spot = data['spot']
            records = data['records']
            expiry = data['expiry']
            
            # Calculate time to expiry
            today = datetime.now(self.ist)
            expiry_date = self.ist.localize(datetime.strptime(expiry, "%d-%b-%Y"))
            T = max((expiry_date - today).days, 1) / 365
            r = 0.06
            
            # Process options
            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    if ce['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('CE', spot, ce['strikePrice'], T, r, ce['impliedVolatility']/100)
                        ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    calls.append(ce)
                
                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    if pe['impliedVolatility'] > 0:
                        greeks = self.calculate_greeks('PE', spot, pe['strikePrice'], T, r, pe['impliedVolatility']/100)
                        pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                    puts.append(pe)
            
            if not calls or not puts: return None
            
            df = pd.merge(pd.DataFrame(calls), pd.DataFrame(puts), on='strikePrice', suffixes=('_CE', '_PE'))
            
            # Find ATM
            atm_strike = min(df['strikePrice'], key=lambda x: abs(x - spot))
            atm_row = df[df['strikePrice'] == atm_strike].iloc[0]
            
            # Calculate bias score
            pcr_oi = data['total_pe_oi'] / data['total_ce_oi'] if data['total_ce_oi'] > 0 else 1
            score = 2 if pcr_oi > 1.1 else -2 if pcr_oi < 0.9 else 0
            overall = "BULLISH" if score >= 1 else "BEARISH" if score <= -1 else "NEUTRAL"
            
            return {
                'instrument': instrument, 'spot_price': spot, 'atm_strike': atm_strike,
                'overall_bias': overall, 'bias_score': score, 'pcr_oi': pcr_oi,
                'total_ce_oi': data['total_ce_oi'], 'total_pe_oi': data['total_pe_oi'],
                'atm_ce_price': atm_row['lastPrice_CE'], 'atm_pe_price': atm_row['lastPrice_PE'],
                'atm_ce_oi': atm_row['openInterest_CE'], 'atm_pe_oi': atm_row['openInterest_PE']
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_all_bias(self) -> List[Dict[str, Any]]:
        results = []
        for inst in self.instruments:
            bias = self.analyze_atm_bias(inst)
            if bias: results.append(bias)
        return results

# =============================================
# MAIN APP
# =============================================
class NiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.market_data = EnhancedMarketData()
        self.bias_analyzer = BiasAnalyzer()
        self.options_analyzer = NSEOptionsAnalyzer()
        self.init_session()

    def init_session(self):
        if 'market_data' not in st.session_state: st.session_state.market_data = None
        if 'bias_data' not in st.session_state: st.session_state.bias_data = None
        if 'options_data' not in st.session_state: st.session_state.options_data = None

    def run(self):
        st.title("📈 Nifty Trading Dashboard")
        st.markdown("*Complete Bias Analysis & Market Data*")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🌍 Market Data", "🎯 Technical Bias", "📊 Options Bias", "📋 Tabulation"])
        
        with tab1:
            self.display_market_data()
        
        with tab2:
            self.display_technical_bias()
        
        with tab3:
            self.display_options_bias()
        
        with tab4:
            self.display_tabulation()

    def display_market_data(self):
        st.header("🌍 Enhanced Market Data")
        
        if st.button("🔄 Update Market Data", type="primary"):
            with st.spinner("Fetching..."):
                st.session_state.market_data = self.market_data.fetch_all()
                st.success("Updated!")
        
        if st.session_state.market_data:
            data = st.session_state.market_data
            
            # VIX
            st.subheader("🇮🇳 India VIX")
            if data['vix']['success']:
                col1, col2, col3 = st.columns(3)
                col1.metric("VIX", f"{data['vix']['value']:.2f}")
                col2.metric("Sentiment", data['vix']['sentiment'])
                col3.metric("Bias", data['vix']['bias'])
            
            # Sectors
            st.subheader("📈 Sector Performance")
            if data['sectors']:
                df = pd.DataFrame(data['sectors']).sort_values('change_pct', ascending=False)
                for _, sector in df.iterrows():
                    col1, col2, col3 = st.columns([2,1,1])
                    col1.write(f"**{sector['sector']}**")
                    col2.metric("Change", f"{sector['change_pct']:+.2f}%")
                    col3.write(sector['bias'])

    def display_technical_bias(self):
        st.header("🎯 Technical Bias Analysis")
        
        if st.button("🔄 Update Technical Bias", type="primary"):
            with st.spinner("Analyzing..."):
                st.session_state.bias_data = self.bias_analyzer.analyze("^NSEI")
                st.success("Done!" if st.session_state.bias_data['success'] else "Failed!")
        
        if st.session_state.bias_data and st.session_state.bias_data['success']:
            data = st.session_state.bias_data
            
            # Overall
            st.subheader("📊 Overall Bias")
            col1, col2, col3 = st.columns(3)
            col1.metric("Bias", data['overall_bias'])
            col2.metric("Score", f"{data['overall_score']:.1f}")
            col3.metric("Price", f"₹{data['current_price']:.2f}")
            
            # Indicators
            st.subheader("📈 Indicators")
            df = pd.DataFrame(data['bias_results'])
            st.dataframe(df, use_container_width=True)

    def display_options_bias(self):
        st.header("📊 Options Chain Bias")
        
        if st.button("🔄 Update Options Data", type="primary"):
            with st.spinner("Fetching..."):
                st.session_state.options_data = self.options_analyzer.get_all_bias()
                st.success(f"Loaded {len(st.session_state.options_data)} instruments!")
        
        if st.session_state.options_data:
            for data in st.session_state.options_data:
                with st.expander(f"🎯 {data['instrument']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Spot", f"₹{data['spot_price']:.2f}")
                    col2.metric("ATM", f"₹{data['atm_strike']:.2f}")
                    col3.metric("Bias", data['overall_bias'])
                    col4.metric("PCR", f"{data['pcr_oi']:.2f}")
                    
                    st.write(f"**Score:** {data['bias_score']:.2f}")
                    st.write(f"**CE OI:** {data['total_ce_oi']:,.0f} | **PE OI:** {data['total_pe_oi']:,.0f}")

    def display_tabulation(self):
        st.header("📋 Complete Bias Tabulation")
        
        # Technical Bias
        if st.session_state.bias_data and st.session_state.bias_data['success']:
            st.subheader("🎯 Technical Bias Summary")
            data = st.session_state.bias_data
            summary = pd.DataFrame([{
                'Metric': 'Overall Bias', 'Value': data['overall_bias']
            }, {
                'Metric': 'Score', 'Value': f"{data['overall_score']:.2f}"
            }, {
                'Metric': 'Bullish', 'Value': data['bullish_count']
            }, {
                'Metric': 'Bearish', 'Value': data['bearish_count']
            }])
            st.dataframe(summary, use_container_width=True, hide_index=True)
            
            st.write("**Detailed Indicators:**")
            df = pd.DataFrame(data['bias_results'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Options Bias
        if st.session_state.options_data:
            st.subheader("📊 Options Bias Summary")
            opt_data = []
            for data in st.session_state.options_data:
                opt_data.append({
                    'Instrument': data['instrument'],
                    'Spot': f"₹{data['spot_price']:.2f}",
                    'ATM': f"₹{data['atm_strike']:.2f}",
                    'Bias': data['overall_bias'],
                    'Score': f"{data['bias_score']:.2f}",
                    'PCR OI': f"{data['pcr_oi']:.2f}",
                    'CE OI': f"{data['total_ce_oi']:,.0f}",
                    'PE OI': f"{data['total_pe_oi']:,.0f}"
                })
            st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)

# Run
if __name__ == "__main__":
    app = NiftyApp()
    app.run()
