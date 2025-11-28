import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
import math
from scipy.stats import norm
import plotly.express as px
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib
import xgboost as xgb
from supabase import create_client, Client
import psycopg2
from psycopg2.extras import Json
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# =============================================
# SUPABASE CONFIGURATION
# =============================================

class SupabaseManager:
    """Supabase client manager for single source of truth"""
    
    def __init__(self):
        try:
            # Try to get from secrets, fallback to environment variables
            supabase_url = st.secrets.get("SUPABASE", {}).get("URL", "")
            supabase_key = st.secrets.get("SUPABASE", {}).get("KEY", "")
            
            if supabase_url and supabase_key:
                self.client: Client = create_client(supabase_url, supabase_key)
                self.connected = True
                logger.info("✅ Supabase connected successfully")
            else:
                self.connected = False
                logger.warning("❌ Supabase credentials not found")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            self.connected = False
    
    def insert_market_features(self, features_data: Dict) -> bool:
        """Insert market features row"""
        if not self.connected:
            return False
            
        try:
            response = self.client.table("market_features").insert(features_data).execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error inserting market features: {e}")
            return False
    
    def get_latest_features(self, symbol: str, timeframe: str, limit: int = 100):
        """Get latest market features"""
        if not self.connected:
            return None
            
        try:
            response = self.client.table("market_features")\
                .select("*")\
                .eq("symbol", symbol)\
                .eq("timeframe", timeframe)\
                .order("ts", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching features: {e}")
            return None
    
    def update_ml_confidence(self, row_id: int, ml_confidence: float, model_version: str):
        """Update ML confidence for a row"""
        if not self.connected:
            return False
            
        try:
            response = self.client.table("market_features")\
                .update({
                    "ml_confidence": ml_confidence,
                    "model_version": model_version
                })\
                .eq("id", row_id)\
                .execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error updating ML confidence: {e}")
            return False

# Initialize Supabase
supabase = SupabaseManager()

# =============================================
# GROQ LLM INTEGRATION
# =============================================

class GroqLLM:
    """Groq LLM for market analysis and explanations"""
    
    def __init__(self):
        self.api_key = st.secrets.get("GROQ", {}).get("API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.available = bool(self.api_key)
    
    def analyze_market_situation(self, market_data: Dict) -> Dict:
        """Analyze market situation using LLM"""
        if not self.available:
            return self._get_fallback_analysis(market_data)
        
        prompt = self._craft_decision_prompt(market_data)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a professional intraday market analyst for NIFTY. 
                        You will only use the provided structured data to produce a concise actionable message. 
                        Output must be valid JSON (single object) with keys: 
                        - overall_trend
                        - bias (Bullish/Bearish/Neutral)
                        - buy_levels (list)
                        - sell_levels (list) 
                        - stop_loss
                        - targets (list)
                        - decision (BUY/SELL/WAIT)
                        - confidence (0-100)
                        - short_reason (1-2 sentences)
                        - warnings (list)
                        - expiry_comment (if expiry within 0-3 days)"""
                    },
                    {
                        "role": "user",
                        "content": json.dumps(market_data, indent=2)
                    }
                ],
                "temperature": 0.15,
                "max_tokens": 400
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            llm_output = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            return json.loads(llm_output)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._get_fallback_analysis(market_data)
    
    def _craft_decision_prompt(self, row: Dict) -> Dict:
        """Craft LLM prompt from market data"""
        return {
            "symbol": row.get('symbol', 'NIFTY'),
            "timeframe": row.get('timeframe', '15m'),
            "ts": row.get('ts', datetime.now(IST).isoformat()),
            "price": row.get('price', 0),
            "unified_bias": row.get('unified_bias', 0),
            "ml_confidence": row.get('ml_confidence', 0),
            "technical_score": row.get('technical_score', 0),
            "option_score": row.get('option_score', 0),
            "institutional_score": row.get('institutional_score', 0),
            "htf_trend": row.get('htf_trend', {}),
            "vob_signal": row.get('vob_signal', 'neutral'),
            "gamma_sequence": row.get('gamma_sequence', {}),
            "oi_trend": row.get('oi_trend', {}),
            "expiry": {
                "date": str(row.get('expiry_date', '')),
                "days_to_expiry": row.get('days_to_expiry', 999),
                "expiry_spike": row.get('expiry_spike', False)
            }
        }
    
    def _get_fallback_analysis(self, market_data: Dict) -> Dict:
        """Fallback analysis when LLM is unavailable"""
        unified_bias = market_data.get('unified_bias', 0)
        ml_confidence = market_data.get('ml_confidence', 0)
        
        if unified_bias >= 70 and ml_confidence >= 70:
            decision = "BUY"
            bias = "Bullish"
        elif unified_bias <= 30 and ml_confidence >= 70:
            decision = "SELL" 
            bias = "Bearish"
        else:
            decision = "WAIT"
            bias = "Neutral"
            
        return {
            "overall_trend": bias,
            "bias": bias,
            "buy_levels": [],
            "sell_levels": [],
            "stop_loss": 0,
            "targets": [],
            "decision": decision,
            "confidence": ml_confidence,
            "short_reason": "Automated analysis based on bias and ML confidence",
            "warnings": ["LLM service unavailable - using fallback logic"],
            "expiry_comment": ""
        }

# Initialize LLM
llm_analyzer = GroqLLM()

# =============================================
# ADVANCED FEATURE ENGINEERING
# =============================================

class AdvancedFeatureEngine:
    """Advanced feature engineering with all new indicators"""
    
    def __init__(self):
        self.feature_list = []
        
    def calculate_put_call_pressure(self, option_chain: Dict) -> Dict:
        """Calculate Put-Call Pressure Curve"""
        strikes = option_chain.get('strikes', [])
        if not strikes:
            return {}
            
        pressure_data = {}
        for strike_data in strikes:
            strike = strike_data['strike']
            ce_oi = strike_data.get('CE_OI', 0)
            pe_oi = strike_data.get('PE_OI', 0)
            ce_volume = strike_data.get('CE_volume', 0)
            pe_volume = strike_data.get('PE_volume', 0)
            
            # Pressure calculation
            oi_pressure = (ce_oi - pe_oi) / max(ce_oi + pe_oi, 1)
            volume_pressure = (ce_volume - pe_volume) / max(ce_volume + pe_volume, 1)
            total_pressure = (oi_pressure + volume_pressure) / 2
            
            pressure_data[strike] = {
                'oi_pressure': oi_pressure,
                'volume_pressure': volume_pressure,
                'total_pressure': total_pressure,
                'dominance': 'CALL' if total_pressure > 0.1 else 'PUT' if total_pressure < -0.1 else 'NEUTRAL'
            }
            
        return pressure_data
    
    def calculate_iv_dynamics(self, option_chain: Dict, historical_iv: List) -> Dict:
        """Calculate IV Crush/Expansion patterns"""
        strikes = option_chain.get('strikes', [])
        current_iv = {}
        
        for strike_data in strikes:
            strike = strike_data['strike']
            ce_iv = strike_data.get('CE_IV', 0)
            pe_iv = strike_data.get('PE_IV', 0)
            current_iv[strike] = (ce_iv + pe_iv) / 2
            
        # IV expansion/crush detection
        if historical_iv:
            prev_iv = historical_iv[-1] if isinstance(historical_iv[-1], dict) else historical_iv[-1]
            iv_expansion = {}
            
            for strike, iv in current_iv.items():
                prev_strike_iv = prev_iv.get(strike, iv)
                iv_change = (iv - prev_strike_iv) / max(prev_strike_iv, 0.01)
                iv_expansion[strike] = {
                    'current_iv': iv,
                    'iv_change': iv_change,
                    'expansion': iv_change > 0.05,  # 5% threshold
                    'crush': iv_change < -0.05
                }
                
            return iv_expansion
        return {}
    
    def calculate_delta_neutral_zones(self, option_chain: Dict) -> Dict:
        """Calculate Delta Neutral and Max Gamma zones"""
        strikes = option_chain.get('strikes', [])
        gamma_data = {}
        
        for strike_data in strikes:
            strike = strike_data['strike']
            # Simplified gamma calculation (in real implementation, use Black-Scholes)
            ce_oi = strike_data.get('CE_OI', 0)
            pe_oi = strike_data.get('PE_OI', 0)
            total_oi = ce_oi + pe_oi
            
            # Gamma approximation
            gamma = min(ce_oi, pe_oi) / max(total_oi, 1)
            gamma_data[strike] = gamma
            
        # Find max gamma strike
        if gamma_data:
            max_gamma_strike = max(gamma_data.items(), key=lambda x: x[1])
            neutral_zone = {
                'max_gamma_strike': max_gamma_strike[0],
                'max_gamma_value': max_gamma_strike[1],
                'neutral_range': [max_gamma_strike[0] - 50, max_gamma_strike[0] + 50]  # ±50 points
            }
            return neutral_zone
            
        return {}
    
    def detect_trap_patterns(self, price_data: pd.DataFrame, option_chain: Dict, vix: float) -> Dict:
        """Detect bull/bear traps"""
        if len(price_data) < 2:
            return {}
            
        current_price = price_data['Close'].iloc[-1]
        prev_price = price_data['Close'].iloc[-2]
        price_change = (current_price - prev_price) / prev_price
        
        # Get ATM strikes
        atm_strike = self._find_atm_strike(option_chain, current_price)
        atm_data = next((s for s in option_chain.get('strikes', []) if s['strike'] == atm_strike), {})
        
        ce_oi_change = atm_data.get('CE_OI_change', 0)
        pe_oi_change = atm_data.get('PE_OI_change', 0)
        vix_change = vix - getattr(self, 'prev_vix', vix)
        self.prev_vix = vix
        
        # Trap detection rules
        bull_trap = (ce_oi_change > 0.03 and price_change > 0 and vix_change > 0)
        bear_trap = (pe_oi_change > 0.03 and price_change < 0 and vix_change > 0)
        
        return {
            'bull_trap_detected': bull_trap,
            'bear_trap_detected': bear_trap,
            'confidence': max(abs(ce_oi_change), abs(pe_oi_change)) * 100,
            'triggers': {
                'ce_oi_change': ce_oi_change,
                'pe_oi_change': pe_oi_change,
                'price_change': price_change,
                'vix_change': vix_change
            }
        }
    
    def analyze_opening_behavior(self, opening_data: Dict) -> Dict:
        """Analyze first 5-minute market opening behavior"""
        return {
            'opening_strength': opening_data.get('volume_ratio', 1.0),
            'oi_buildup': opening_data.get('oi_change', 0),
            'direction_confirmation': opening_data.get('price_change', 0),
            'bias_strength': min(100, abs(opening_data.get('price_change', 0)) * 1000)
        }
    
    def calculate_time_based_behavior(self, current_time: datetime) -> Dict:
        """Time-based market behavior matrix"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Market phases
        if hour == 9 and minute <= 45:
            phase = "OPENING_FAKEOUTS"
            confidence_modifier = 0.7
        elif (hour == 9 and minute > 45) or (hour == 10 and minute <= 45):
            phase = "REAL_TREND"
            confidence_modifier = 1.0
        elif (hour >= 11 and hour < 12) or (hour == 12 and minute <= 30):
            phase = "CONSOLIDATION" 
            confidence_modifier = 0.8
        elif hour >= 14 and hour < 15:
            phase = "TREND_CONTINUATION"
            confidence_modifier = 0.9
        elif hour >= 15:
            phase = "SHORT_COVERING"
            confidence_modifier = 0.6
        else:
            phase = "NORMAL"
            confidence_modifier = 0.85
            
        return {
            'market_phase': phase,
            'confidence_modifier': confidence_modifier,
            'preferred_action': 'WAIT' if phase == "OPENING_FAKEOUTS" else 'TREND'
        }
    
    def detect_price_structure(self, price_data: pd.DataFrame) -> Dict:
        """Detect price structure algebra patterns"""
        if len(price_data) < 10:
            return {}
            
        highs = price_data['High']
        lows = price_data['Low']
        
        # Internal/External Break of Structure
        recent_high = highs.iloc[-5:].max()
        recent_low = lows.iloc[-5:].min()
        prev_high = highs.iloc[-10:-5].max()
        prev_low = lows.iloc[-10:-5].min()
        
        ibos = (highs.iloc[-1] > prev_high) or (lows.iloc[-1] < prev_low)
        ebos = (highs.iloc[-1] > recent_high) or (lows.iloc[-1] < recent_low)
        
        # Change of Character (CHoCH)
        higher_highs = highs.iloc[-1] > highs.iloc[-2] > highs.iloc[-3]
        lower_lows = lows.iloc[-1] < lows.iloc[-2] < lows.iloc[-3]
        
        return {
            'internal_bos': ibos,
            'external_bos': ebos,
            'higher_highs': higher_highs,
            'lower_lows': lower_lows,
            'market_structure': 'BULLISH' if higher_highs else 'BEARISH' if lower_lows else 'RANGING',
            'breakout_imminent': ibos or ebos
        }
    
    def calculate_volume_profile(self, price_data: pd.DataFrame) -> Dict:
        """Calculate volume profile and POC"""
        if len(price_data) < 20:
            return {}
            
        # Simple volume profile calculation
        price_bins = np.linspace(price_data['Low'].min(), price_data['High'].max(), 20)
        volume_profile = {}
        
        for i in range(len(price_bins) - 1):
            low_bound = price_bins[i]
            high_bound = price_bins[i + 1]
            
            mask = (price_data['Low'] >= low_bound) & (price_data['High'] <= high_bound)
            bin_volume = price_data.loc[mask, 'Volume'].sum()
            volume_profile[f"{low_bound:.1f}-{high_bound:.1f}"] = bin_volume
            
        # Point of Control (POC)
        if volume_profile:
            poc_level = max(volume_profile.items(), key=lambda x: x[1])
            return {
                'volume_profile': volume_profile,
                'poc': poc_level[0],
                'poc_volume': poc_level[1],
                'value_area': self._calculate_value_area(volume_profile)
            }
            
        return {}
    
    def _calculate_value_area(self, volume_profile: Dict) -> Dict:
        """Calculate value area from volume profile"""
        total_volume = sum(volume_profile.values())
        if total_volume == 0:
            return {}
            
        # Find 70% value area
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area_levels = []
        
        for level, volume in sorted_levels:
            cumulative_volume += volume
            value_area_levels.append(level)
            if cumulative_volume / total_volume >= 0.7:
                break
                
        return {
            'levels': value_area_levels,
            'total_volume_covered': cumulative_volume / total_volume
        }
    
    def _find_atm_strike(self, option_chain: Dict, current_price: float) -> int:
        """Find ATM strike"""
        strikes = [s['strike'] for s in option_chain.get('strikes', [])]
        if not strikes:
            return round(current_price / 50) * 50  # Approximate NIFTY strike
            
        return min(strikes, key=lambda x: abs(x - current_price))
    
    def detect_liquidity_sweeps(self, price_data: pd.DataFrame) -> Dict:
        """Detect liquidity sweeps at key levels"""
        if len(price_data) < 20:
            return {}
            
        current_high = price_data['High'].iloc[-1]
        current_low = price_data['Low'].iloc[-1]
        
        # Key levels: previous day high/low, swing points
        prev_day_high = price_data['High'].iloc[-20:].max()
        prev_day_low = price_data['Low'].iloc[-20:].min()
        
        high_sweep = current_high >= prev_day_high * 0.999  # Within 0.1%
        low_sweep = current_low <= prev_day_low * 1.001
        
        return {
            'liquidity_sweep_high': high_sweep,
            'liquidity_sweep_low': low_sweep,
            'sweep_confidence': max(
                0.8 if high_sweep else 0,
                0.8 if low_sweep else 0
            ),
            'reversal_probability': 0.7 if (high_sweep or low_sweep) else 0.3
        }
    
    def calculate_momentum_curvature(self, price_data: pd.DataFrame) -> Dict:
        """Calculate momentum curvature for reversal detection"""
        if len(price_data) < 10:
            return {}
            
        returns = price_data['Close'].pct_change().dropna()
        if len(returns) < 5:
            return {}
            
        # Simple momentum curvature (second derivative approximation)
        mom1 = returns.iloc[-1]
        mom2 = returns.iloc[-2] 
        mom3 = returns.iloc[-3]
        
        curvature = (mom1 - 2*mom2 + mom3)
        curvature_trend = "ACCELERATING" if curvature > 0.001 else "DECELERATING" if curvature < -0.001 else "STABLE"
        
        return {
            'momentum_curvature': curvature,
            'curvature_trend': curvature_trend,
            'reversal_imminent': abs(curvature) > 0.002,
            'continuation_likely': curvature > 0.001
        }

# =============================================
# ENHANCED ML PREDICTION ENGINE
# =============================================

class EnhancedMLPredictionEngine:
    """Enhanced ML engine with XGBoost and advanced features"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_version = "v2.0"
        self.is_trained = False
        
    def prepare_advanced_features(self, technical_data, options_data, volume_data, 
                                market_intel, advanced_features) -> Dict:
        """Prepare comprehensive feature vector with all new indicators"""
        
        features = {}
        
        # Technical Features (existing)
        if technical_data:
            features.update({
                'rsi': technical_data.get('rsi', 50),
                'macd_signal': 1 if technical_data.get('macd_signal') == 'bullish' else -1,
                'ema_trend': self._encode_trend(technical_data.get('ema_trend', 'neutral')),
                'atr_normalized': technical_data.get('atr', 0) / max(1, technical_data.get('price', 1)),
                'sr_distance': technical_data.get('sr_distance', 0),
                'htf_trend_score': technical_data.get('htf_trend_score', 0)
            })
        
        # Volume/Order Block Features (existing)
        if volume_data:
            features.update({
                'volume_spike': volume_data.get('volume_spike_ratio', 1.0),
                'bullish_blocks': volume_data.get('bullish_blocks_count', 0),
                'bearish_blocks': volume_data.get('bearish_blocks_count', 0),
                'structure_break': 1 if volume_data.get('structure_break', False) else 0,
                'liquidity_grab': 1 if volume_data.get('liquidity_grab', False) else 0
            })
        
        # Options Chain Features (existing)
        if options_data:
            features.update({
                'oi_ratio': options_data.get('oi_ce', 1) / max(1, options_data.get('oi_pe', 1)),
                'iv_skew': options_data.get('iv_ce', 0) - options_data.get('iv_pe', 0),
                'max_pain_shift': options_data.get('max_pain_shift', 0),
                'pcr_value': options_data.get('pcr', 1.0),
                'atm_bias_score': options_data.get('atm_bias_score', 0),
                'oi_unwinding': options_data.get('oi_unwinding_ratio', 0)
            })
        
        # Advanced Features (NEW)
        if advanced_features:
            # Put-Call Pressure
            pcp = advanced_features.get('put_call_pressure', {})
            if pcp:
                avg_pressure = np.mean([v.get('total_pressure', 0) for v in pcp.values()])
                features['pcp_avg_pressure'] = avg_pressure
            
            # IV Dynamics
            iv_dynamics = advanced_features.get('iv_dynamics', {})
            if iv_dynamics:
                iv_expansion_count = sum(1 for v in iv_dynamics.values() if v.get('expansion', False))
                features['iv_expansion_ratio'] = iv_expansion_count / max(len(iv_dynamics), 1)
            
            # Delta Neutral Zones
            delta_neutral = advanced_features.get('delta_neutral', {})
            if delta_neutral:
                features['in_neutral_zone'] = 1 if delta_neutral.get('in_zone', False) else 0
            
            # Trap Detection
            traps = advanced_features.get('trap_detection', {})
            if traps:
                features['trap_confidence'] = traps.get('confidence', 0)
                features['bull_trap'] = 1 if traps.get('bull_trap_detected', False) else 0
                features['bear_trap'] = 1 if traps.get('bear_trap_detected', False) else 0
            
            # Opening Behavior
            opening = advanced_features.get('opening_behavior', {})
            if opening:
                features['opening_strength'] = opening.get('opening_strength', 1.0)
                features['opening_bias'] = opening.get('bias_strength', 0)
            
            # Time-based Behavior
            time_behavior = advanced_features.get('time_behavior', {})
            if time_behavior:
                features['confidence_modifier'] = time_behavior.get('confidence_modifier', 1.0)
            
            # Price Structure
            price_structure = advanced_features.get('price_structure', {})
            if price_structure:
                features['breakout_imminent'] = 1 if price_structure.get('breakout_imminent', False) else 0
                features['market_structure'] = self._encode_structure(price_structure.get('market_structure', 'RANGING'))
            
            # Volume Profile
            volume_profile = advanced_features.get('volume_profile', {})
            if volume_profile:
                features['poc_volume_ratio'] = volume_profile.get('poc_volume', 0) / max(volume_profile.get('total_volume', 1), 1)
            
            # Liquidity Sweeps
            liquidity = advanced_features.get('liquidity_sweeps', {})
            if liquidity:
                features['liquidity_sweep_confidence'] = liquidity.get('sweep_confidence', 0)
                features['reversal_probability'] = liquidity.get('reversal_probability', 0.5)
            
            # Momentum Curvature
            momentum = advanced_features.get('momentum_curvature', {})
            if momentum:
                features['momentum_curvature'] = momentum.get('momentum_curvature', 0)
                features['reversal_imminent'] = 1 if momentum.get('reversal_imminent', False) else 0
        
        # Market Intel Features (existing)
        if market_intel:
            features.update({
                'india_vix': market_intel.get('india_vix', 0),
                'sgx_nifty_change': market_intel.get('sgx_nifty_change', 0),
                'global_futures_trend': self._encode_trend(market_intel.get('global_trend', 'neutral')),
                'fii_dii_trend': market_intel.get('fii_dii_trend', 0),
                'sector_rotation': market_intel.get('sector_rotation_strength', 0)
            })
        
        # Derived Features (existing)
        features.update({
            'bias_alignment': self._calculate_bias_alignment(technical_data, options_data, market_intel),
            'momentum_strength': self._calculate_momentum_strength(technical_data, volume_data, advanced_features),
            'institutional_pressure': self._calculate_institutional_pressure(options_data, market_intel, advanced_features)
        })
        
        return features
    
    def _encode_trend(self, trend):
        """Encode trend as numerical value"""
        trend_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
        return trend_map.get(trend, 0)
    
    def _encode_structure(self, structure):
        """Encode market structure as numerical value"""
        structure_map = {'BULLISH': 1, 'BEARISH': -1, 'RANGING': 0}
        return structure_map.get(structure, 0)
    
    def _calculate_bias_alignment(self, technical_data, options_data, market_intel):
        """Calculate how aligned different biases are"""
        biases = []
        
        if technical_data:
            biases.append(technical_data.get('technical_bias_score', 0))
        if options_data:
            biases.append(options_data.get('options_bias_score', 0))
        if market_intel:
            biases.append(market_intel.get('institutional_bias_score', 0))
        
        if not biases:
            return 0
            
        alignment = sum(biases) / (max(1, sum(abs(b) for b in biases)))
        return alignment
    
    def _calculate_momentum_strength(self, technical_data, volume_data, advanced_features):
        """Calculate combined momentum strength with advanced features"""
        momentum = 0
        
        if technical_data:
            momentum += technical_data.get('rsi_strength', 0) * 0.25
            momentum += technical_data.get('trend_strength', 0) * 0.35
        
        if volume_data:
            momentum += volume_data.get('volume_momentum', 0) * 0.20
        
        if advanced_features:
            momentum_curve = advanced_features.get('momentum_curvature', {})
            if momentum_curve.get('continuation_likely', False):
                momentum += 0.20
        
        return min(100, max(0, momentum * 100))
    
    def _calculate_institutional_pressure(self, options_data, market_intel, advanced_features):
        """Calculate institutional pressure with trap detection"""
        pressure = 0
        
        if options_data:
            pressure += options_data.get('oi_pressure', 0) * 0.4
            pressure += options_data.get('iv_pressure', 0) * 0.3
        
        if market_intel:
            pressure += market_intel.get('institutional_bias', 0) * 0.2
        
        if advanced_features:
            traps = advanced_features.get('trap_detection', {})
            # Reduce pressure if traps detected
            if traps.get('bull_trap_detected', False) or traps.get('bear_trap_detected', False):
                pressure *= 0.5
        
        return pressure
    
    def train_xgboost_model(self, features: List[Dict], labels: List) -> bool:
        """Train XGBoost model with advanced features"""
        try:
            if not features or not labels:
                logger.error("No features or labels for training")
                return False
            
            # Convert to DataFrame
            X = pd.DataFrame(features)
            y = pd.Series(labels)
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Calculate metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"✅ XGBoost Model Trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return False
    
    def predict_confidence(self, feature_vector: Dict) -> float:
        """Predict ML confidence score (0-100)"""
        if not self.is_trained or self.model is None:
            return 50.0
        
        try:
            # Create feature array in correct order
            X = np.array([[feature_vector.get(col, 0) for col in self.feature_columns]])
            
            # Predict probability
            proba = self.model.predict_proba(X)[0]
            confidence = max(proba) * 100  # Highest class probability
            
            return min(100, max(0, confidence))
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 50.0
    
    def save_model(self):
        """Save trained model and feature columns"""
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_version': self.model_version,
                'trained_at': datetime.now(IST)
            }
            joblib.dump(model_data, f"xgboost_model_{self.model_version}.pkl")
            logger.info("✅ XGBoost model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model"""
        try:
            model_data = joblib.load(f"xgboost_model_{self.model_version}.pkl")
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            logger.info("✅ XGBoost model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

# =============================================
# ENHANCED MASTER TRIGGER ENGINE
# =============================================

class EnhancedMasterTriggerEngine:
    """Enhanced master trigger with all new features"""
    
    def __init__(self):
        self.ml_engine = EnhancedMLPredictionEngine()
        self.llm_analyzer = GroqLLM()
        self.feature_engine = AdvancedFeatureEngine()
        self.telegram_notifier = telegram_notifier
        self.last_trigger_time = {}
        self.trigger_cooldown = 300  # 5 minutes
        
    def should_trigger_analysis(self, symbol: str, unified_bias: float, ml_confidence: float, 
                              advanced_features: Dict) -> bool:
        """Enhanced trigger conditions with advanced features"""
        current_time = datetime.now(IST)
        
        # Cooldown check
        if symbol in self.last_trigger_time:
            time_diff = (current_time - self.last_trigger_time[symbol]).total_seconds()
            if time_diff < self.trigger_cooldown:
                return False
        
        # Base conditions
        base_trigger = unified_bias >= 80 and ml_confidence >= 85
        
        if not base_trigger:
            return False
            
        # Advanced condition checks
        trap_detection = advanced_features.get('trap_detection', {})
        if trap_detection.get('bull_trap_detected', False) or trap_detection.get('bear_trap_detected', False):
            logger.info("Trap detected - suppressing trigger")
            return False
            
        # Time-based filtering
        time_behavior = advanced_features.get('time_behavior', {})
        if time_behavior.get('market_phase') == "OPENING_FAKEOUTS":
            logger.info("Opening fakeout phase - suppressing trigger")
            return False
            
        # Liquidity sweep check
        liquidity = advanced_features.get('liquidity_sweeps', {})
        if liquidity.get('reversal_probability', 0) > 0.6:
            logger.info("High reversal probability - suppressing trigger")
            return False
            
        return True
    
    async def execute_enhanced_trigger(self, symbol: str, market_data: Dict, 
                                     advanced_features: Dict) -> Dict:
        """Execute enhanced master trigger with all features"""
        try:
            logger.info("🚀 ENHANCED MASTER TRIGGER ACTIVATED!")
            
            # Prepare features for ML
            feature_vector = self.ml_engine.prepare_advanced_features(
                market_data.get('technical_data', {}),
                market_data.get('options_data', {}),
                market_data.get('volume_data', {}),
                market_data.get('market_intel', {}),
                advanced_features
            )
            
            # Get ML confidence
            ml_confidence = self.ml_engine.predict_confidence(feature_vector)
            
            # Prepare LLM analysis
            llm_payload = self._prepare_llm_payload(symbol, market_data, advanced_features, ml_confidence)
            llm_analysis = self.llm_analyzer.analyze_market_situation(llm_payload)
            
            # Generate prediction
            prediction = self._generate_enhanced_prediction(market_data, llm_analysis, advanced_features)
            
            # Send comprehensive alert
            await self._send_enhanced_alert(symbol, market_data, llm_analysis, prediction, ml_confidence)
            
            # Log to Supabase
            self._log_trigger_to_supabase(symbol, market_data, llm_analysis, prediction, ml_confidence)
            
            return {
                'triggered': True,
                'timestamp': datetime.now(IST),
                'symbol': symbol,
                'unified_bias': market_data.get('unified_bias', 0),
                'ml_confidence': ml_confidence,
                'llm_analysis': llm_analysis,
                'prediction': prediction,
                'advanced_features': advanced_features
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced trigger: {e}")
            return {'triggered': False, 'error': str(e)}
    
    def _prepare_llm_payload(self, symbol: str, market_data: Dict, 
                           advanced_features: Dict, ml_confidence: float) -> Dict:
        """Prepare comprehensive LLM payload"""
        
        return {
            "symbol": symbol,
            "timeframe": market_data.get('timeframe', '15m'),
            "ts": datetime.now(IST).isoformat(),
            "price": market_data.get('current_price', 0),
            "unified_bias": market_data.get('unified_bias', 0),
            "ml_confidence": ml_confidence,
            "technical_score": market_data.get('technical_score', 0),
            "option_score": market_data.get('option_score', 0),
            "institutional_score": market_data.get('institutional_score', 0),
            "htf_trend": advanced_features.get('htf_trend', {}),
            "vob_signal": advanced_features.get('vob_signal', 'neutral'),
            "gamma_sequence": advanced_features.get('gamma_sequence', {}),
            "oi_trend": advanced_features.get('oi_trend', {}),
            "put_call_pressure": advanced_features.get('put_call_pressure', {}),
            "iv_dynamics": advanced_features.get('iv_dynamics', {}),
            "delta_neutral": advanced_features.get('delta_neutral', {}),
            "trap_detection": advanced_features.get('trap_detection', {}),
            "price_structure": advanced_features.get('price_structure', {}),
            "volume_profile": advanced_features.get('volume_profile', {}),
            "liquidity_sweeps": advanced_features.get('liquidity_sweeps', {}),
            "momentum_curvature": advanced_features.get('momentum_curvature', {}),
            "time_behavior": advanced_features.get('time_behavior', {}),
            "expiry": {
                "date": market_data.get('expiry_date', ''),
                "days_to_expiry": market_data.get('days_to_expiry', 999),
                "expiry_spike": market_data.get('expiry_spike', False)
            }
        }
    
    def _generate_enhanced_prediction(self, market_data: Dict, llm_analysis: Dict, 
                                    advanced_features: Dict) -> Dict:
        """Generate enhanced prediction with all features"""
        
        # Base prediction from LLM
        prediction = {
            'timeframe': '1-4 hours',
            'direction': llm_analysis.get('bias', 'Neutral'),
            'confidence': llm_analysis.get('confidence', 0),
            'targets': llm_analysis.get('targets', []),
            'stoploss': llm_analysis.get('stop_loss', 0),
            'decision': llm_analysis.get('decision', 'WAIT'),
            'probability': llm_analysis.get('confidence', 0),
            'validity_period': '4 hours'
        }
        
        # Enhance with advanced features
        time_behavior = advanced_features.get('time_behavior', {})
        if time_behavior.get('market_phase') == "EXPIRY_DAY":
            prediction['targets'] = [f"{t} (reduced)" for t in prediction['targets']]
            prediction['warnings'] = prediction.get('warnings', []) + ["Expiry day - reduced targets"]
        
        # Adjust based on trap detection
        traps = advanced_features.get('trap_detection', {})
        if traps.get('bull_trap_detected', False) and prediction['direction'] == 'Bullish':
            prediction['decision'] = 'WAIT'
            prediction['warnings'] = prediction.get('warnings', []) + ["Bull trap detected"]
        elif traps.get('bear_trap_detected', False) and prediction['direction'] == 'Bearish':
            prediction['decision'] = 'WAIT'
            prediction['warnings'] = prediction.get('warnings', []) + ["Bear trap detected"]
        
        return prediction
    
    async def _send_enhanced_alert(self, symbol: str, market_data: Dict, llm_analysis: Dict,
                                 prediction: Dict, ml_confidence: float):
        """Send enhanced Telegram alert"""
        
        if not self.telegram_notifier.is_configured():
            return
        
        message = f"""
🚀 **ENHANCED MASTER TRIGGER - {symbol}**

📊 **Signal Strength:**
• Unified Bias: `{market_data.get('unified_bias', 0):.1f}`
• ML Confidence: `{ml_confidence:.1f}`
• LLM Confidence: `{llm_analysis.get('confidence', 0):.1f}`

🎯 **AI Prediction:**
• Direction: `{prediction['direction']}`
• Decision: `{prediction['decision']}`
• Timeframe: `{prediction['timeframe']}`
• Probability: `{prediction['probability']:.1f}%`

📈 **Targets:** {', '.join(prediction.get('targets', []))}
🛡️ **Stoploss:** {prediction.get('stoploss', 'N/A')}

💡 **Reasoning:** {llm_analysis.get('short_reason', 'Advanced pattern detection')}

⚠️ **Warnings:**
{chr(10).join(['• ' + warning for warning in llm_analysis.get('warnings', [])])}

🕒 **Market Phase:** {market_data.get('market_phase', 'Normal')}
⏰ **Timestamp:** {datetime.now(IST).strftime('%H:%M:%S')}
        """
        
        self.telegram_notifier.send_message(message, "ENHANCED_TRIGGER")
    
    def _log_trigger_to_supabase(self, symbol: str, market_data: Dict, llm_analysis: Dict,
                               prediction: Dict, ml_confidence: float):
        """Log trigger to Supabase"""
        if not supabase.connected:
            return
            
        alert_data = {
            'symbol': symbol,
            'timeframe': market_data.get('timeframe', '15m'),
            'ts': datetime.now(IST).isoformat(),
            'unified_bias': market_data.get('unified_bias', 0),
            'ml_confidence': ml_confidence,
            'decision': prediction.get('decision', 'WAIT'),
            'payload': {
                'llm_analysis': llm_analysis,
                'prediction': prediction,
                'market_data': market_data
            },
            'sent': True
        }
        
        supabase.client.table("alerts").insert(alert_data).execute()

# =============================================
# TELEGRAM NOTIFICATION SYSTEM
# =============================================

class TelegramNotifier:
    """Telegram notification system for bias alerts"""
    
    def __init__(self):
        # Get credentials from Streamlit secrets
        self.bot_token = st.secrets.get("TELEGRAM", {}).get("BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM", {}).get("CHAT_ID", "")
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes cooldown between same type alerts
        
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)
        
    def send_message(self, message: str, alert_type: str = "INFO") -> bool:
        """Send message to Telegram"""
        try:
            if not self.is_configured():
                print("Telegram credentials not configured in secrets")
                return False
            
            # Check cooldown
            current_time = time.time()
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                    print(f"Alert {alert_type} in cooldown")
                    return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.last_alert_time[alert_type] = current_time
                print(f"✅ Telegram alert sent: {alert_type}")
                return True
            else:
                print(f"❌ Telegram send failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    def send_bias_alert(self, technical_bias: str, options_bias: str, atm_bias: str, overall_bias: str, score: float):
        """Send comprehensive bias alert"""
        # Check if all three components are aligned
        components = [technical_bias, options_bias, atm_bias]
        bullish_count = sum(1 for bias in components if "BULL" in bias.upper())
        bearish_count = sum(1 for bias in components if "BEAR" in bias.upper())
        
        if bullish_count >= 2 or bearish_count >= 2:
            emoji = "🚀" if bullish_count >= 2 else "🔻"
            alert_type = "STRONG_BULL" if bullish_count >= 2 else "STRONG_BEAR"
            
            message = f"""
{emoji} <b>STRONG BIAS ALERT - NIFTY 50</b> {emoji}

📊 <b>Component Analysis:</b>
• Technical Analysis: <b>{technical_bias}</b>
• Options Chain: <b>{options_bias}</b>  
• ATM Detailed: <b>{atm_bias}</b>

🎯 <b>Overall Bias:</b> <code>{overall_bias}</code>
⭐ <b>Confidence Score:</b> <code>{score:.1f}/100</code>

⏰ <b>Time:</b> {datetime.now(IST).strftime('%H:%M:%S')}
            
💡 <b>Market Insight:</b>
{'Bullish momentum detected across multiple timeframes' if bullish_count >= 2 else 'Bearish pressure building across indicators'}
"""
            
            return self.send_message(message, alert_type)
        
        return False

# Initialize Telegram Notifier
telegram_notifier = TelegramNotifier()

# =============================================
# ORIGINAL BIAS ANALYSIS PRO (KEPT INTACT)
# =============================================

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
        # CALCULATE OVERALL BIAS (MATCHING PINE SCRIPT LOGIC) - FIXED
        # =====================================================================
        fast_bull = 0
        fast_bear = 0
        fast_total = 0

        medium_bull = 0
        medium_bear = 0
        medium_total = 0

        # FIX 1: Disable slow category completely
        slow_bull = 0
        slow_bear = 0
        slow_total = 0  # Set to zero to avoid division by zero

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
                # Skip slow category
            elif 'BEARISH' in bias['bias']:
                bearish_count += 1
                if bias['category'] == 'fast':
                    fast_bear += 1
                elif bias['category'] == 'medium':
                    medium_bear += 1
                # Skip slow category
            else:
                neutral_count += 1

            if bias['category'] == 'fast':
                fast_total += 1
            elif bias['category'] == 'medium':
                medium_total += 1
            # Skip slow category counting

        # Calculate percentages - FIXED for slow category
        fast_bull_pct = (fast_bull / fast_total) * 100 if fast_total > 0 else 0
        fast_bear_pct = (fast_bear / fast_total) * 100 if fast_total > 0 else 0

        medium_bull_pct = (medium_bull / medium_total) * 100 if medium_total > 0 else 0
        medium_bear_pct = (medium_bear / medium_total) * 100 if medium_total > 0 else 0

        # FIX 1: Set slow percentages to 0 since we disabled slow indicators
        slow_bull_pct = 0
        slow_bear_pct = 0

        # Adaptive weighting (matching Pine Script)
        # Check for divergence - FIXED for slow category
        divergence_threshold = self.config['divergence_threshold']
        # Since slow_bull_pct is 0, divergence won't trigger incorrectly
        bullish_divergence = slow_bull_pct >= 66 and fast_bear_pct >= divergence_threshold
        bearish_divergence = slow_bear_pct >= 66 and fast_bull_pct >= divergence_threshold
        divergence_detected = bullish_divergence or bearish_divergence

        # Determine mode - FIXED: Always use normal mode since slow indicators disabled
        if divergence_detected and slow_total > 0:  # Only if we had slow indicators
            fast_weight = self.config['reversal_fast_weight']
            medium_weight = self.config['reversal_medium_weight']
            slow_weight = self.config['reversal_slow_weight']
            mode = "REVERSAL"
        else:
            # Use normal weights, ignore slow weight
            fast_weight = self.config['normal_fast_weight']
            medium_weight = self.config['normal_medium_weight']
            slow_weight = 0  # FIX: Set slow weight to 0 since no slow indicators
            mode = "NORMAL"

        # Calculate weighted scores - FIXED: Exclude slow category
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

# =============================================
# VOLUME ORDER BLOCKS
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator"""
    
    def __init__(self, sensitivity=5):
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.max_lines_count = 500
        self.bullish_blocks = deque(maxlen=15)
        self.bearish_blocks = deque(maxlen=15)
        self.sent_alerts = set()
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df: pd.DataFrame, period=200) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def detect_volume_order_blocks(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect Volume Order Blocks based on the Pine Script logic"""
        if len(df) < self.length2:
            return [], []
        
        ema1 = self.calculate_ema(df['Close'], self.length1)
        ema2 = self.calculate_ema(df['Close'], self.length2)
        
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
                    
                lowest_idx = lookback_data['Low'].idxmin()
                lowest_price = lookback_data.loc[lowest_idx, 'Low']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[lowest_idx, 'Open']
                close_price = lookback_data.loc[lowest_idx, 'Close']
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
                    
                highest_idx = lookback_data['High'].idxmax()
                highest_price = lookback_data.loc[highest_idx, 'High']
                
                vol = lookback_data['Volume'].sum()
                
                open_price = lookback_data.loc[highest_idx, 'Open']
                close_price = lookback_data.loc[highest_idx, 'Close']
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
    
    def filter_overlapping_blocks(self, blocks: List[Dict[str, Any]], atr_value: float) -> List[Dict[str, Any]]:
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

def plot_vob(df: pd.DataFrame, bullish_blocks: List[Dict], bearish_blocks: List[Dict]) -> go.Figure:
    """Plot Volume Order Blocks on candlestick chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=('Price with Volume Order Blocks', 'Volume'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                 row=1, col=1)
    
    # Add bullish blocks
    for block in bullish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="green", 
                     annotation_text=f"Bull Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="green", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    
    # Add bearish blocks
    for block in bearish_blocks:
        fig.add_hline(y=block['upper'], line_dash="dash", line_color="red",
                     annotation_text=f"Bear Block", row=1, col=1)
        fig.add_hline(y=block['lower'], line_dash="dash", line_color="red", row=1, col=1)
        # Fill between lines
        fig.add_shape(type="rect", x0=block['index'], x1=df.index[-1],
                     y0=block['lower'], y1=block['upper'],
                     fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    
    # Volume
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
                 row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, 
                     template='plotly_dark',
                     height=600,
                     showlegend=True)
    
    return fig

# =============================================
# SIMPLIFIED NSE OPTIONS ANALYZER
# =============================================

class SimpleOptionsAnalyzer:
    """Simplified options analyzer for demo purposes"""
    
    def __init__(self):
        self.NSE_INSTRUMENTS = {
            'indices': {
                'NIFTY': {'lot_size': 50, 'atm_range': 200, 'zone_size': 100},
                'BANKNIFTY': {'lot_size': 25, 'atm_range': 400, 'zone_size': 200},
            }
        }
    
    def get_sample_options_data(self, instrument="NIFTY"):
        """Generate sample options data for demo"""
        import random
        
        return {
            'instrument': instrument,
            'spot_price': 22150 + random.randint(-100, 100),
            'atm_strike': 22100,
            'overall_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
            'bias_score': random.uniform(-5, 5),
            'pcr_oi': random.uniform(0.8, 1.2),
            'pcr_change': random.uniform(0.9, 1.1),
            'total_ce_oi': random.randint(5000000, 8000000),
            'total_pe_oi': random.randint(5000000, 8000000),
            'total_ce_change': random.randint(-100000, 100000),
            'total_pe_change': random.randint(-100000, 100000),
            'detailed_atm_bias': {
                'OI_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'ChgOI_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'Volume_Bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
            },
            'comprehensive_metrics': {
                'synthetic_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'atm_buildup': random.choice(['Long Buildup', 'Short Buildup', 'Neutral']),
                'max_pain_strike': 22100 + random.randint(-100, 100),
                'total_vega_bias': random.choice(['Bullish', 'Bearish', 'Neutral']),
                'call_resistance': 22200 + random.randint(0, 100),
                'put_support': 22000 - random.randint(0, 100)
            }
        }

# =============================================
# ENHANCED STREAMLIT APP UI
# =============================================

def add_enhanced_ml_analysis_tab():
    """Add enhanced ML Analysis tab"""
    
    st.header("🤖 ENHANCED ML Prediction Engine")
    st.markdown("### Advanced ML with XGBoost, Groq LLM & Master Trigger")
    
    # Initialize session state
    if 'enhanced_ml_engine' not in st.session_state:
        st.session_state.enhanced_ml_engine = EnhancedMLPredictionEngine()
        st.session_state.enhanced_ml_engine.load_model()
    
    if 'enhanced_master_engine' not in st.session_state:
        st.session_state.enhanced_master_engine = EnhancedMasterTriggerEngine()
    
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = AdvancedFeatureEngine()
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 XGBoost Predictions", 
        "📊 Advanced Features",
        "🤖 LLM Analysis",
        "🚀 Enhanced Trigger"
    ])
    
    with tab1:
        st.subheader("XGBoost ML Predictions")
        
        if st.session_state.get('analysis_complete'):
            feature_data = prepare_enhanced_feature_data()
            
            if feature_data:
                ml_confidence = st.session_state.enhanced_ml_engine.predict_confidence(feature_data)
                unified_bias = st.session_state.overall_nifty_score
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("XGBoost Confidence", f"{ml_confidence:.1f}%")
                
                with col2:
                    st.metric("Unified Bias", f"{unified_bias:.1f}")
                
                with col3:
                    st.metric("Model Version", st.session_state.enhanced_ml_engine.model_version)
                
                with col4:
                    bias_alignment = feature_data.get('bias_alignment', 0)
                    st.metric("Bias Alignment", f"{bias_alignment:.1%}")
                
                # XGBoost Training Section
                st.subheader("🎯 XGBoost Model Training")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Train XGBoost Model", use_container_width=True):
                        with st.spinner("Training XGBoost model..."):
                            # This would fetch training data from Supabase in real implementation
                            st.info("Training would use Supabase historical data")
                
                with col2:
                    if st.button("📊 Model Metrics", use_container_width=True):
                        st.info("Model metrics would show here from Supabase")
        
        else:
            st.info("Run complete analysis first to generate ML predictions")
    
    with tab2:
        st.subheader("📊 Advanced Feature Analysis")
        
        if st.session_state.get('analysis_complete'):
            advanced_features = st.session_state.get('advanced_features', {})
            
            # Display advanced features
            st.subheader("Real-time Advanced Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trap Detection
                traps = advanced_features.get('trap_detection', {})
                if traps:
                    st.metric("Bull Trap", "DETECTED" if traps.get('bull_trap_detected') else "CLEAR")
                    st.metric("Bear Trap", "DETECTED" if traps.get('bear_trap_detected') else "CLEAR")
                    st.metric("Trap Confidence", f"{traps.get('confidence', 0):.1f}%")
                
                # IV Dynamics
                iv_dynamics = advanced_features.get('iv_dynamics', {})
                if iv_dynamics:
                    expansion_count = sum(1 for v in iv_dynamics.values() if v.get('expansion', False))
                    crush_count = sum(1 for v in iv_dynamics.values() if v.get('crush', False))
                    st.metric("IV Expansion", expansion_count)
                    st.metric("IV Crush", crush_count)
            
            with col2:
                # Price Structure
                price_structure = advanced_features.get('price_structure', {})
                if price_structure:
                    st.metric("Market Structure", price_structure.get('market_structure', 'UNKNOWN'))
                    st.metric("Breakout Imminent", "✅ YES" if price_structure.get('breakout_imminent') else "❌ NO")
                    st.metric("Internal BOS", "✅ YES" if price_structure.get('internal_bos') else "❌ NO")
                
                # Time Behavior
                time_behavior = advanced_features.get('time_behavior', {})
                if time_behavior:
                    st.metric("Market Phase", time_behavior.get('market_phase', 'UNKNOWN'))
                    st.metric("Confidence Modifier", f"{time_behavior.get('confidence_modifier', 1.0):.2f}")
            
            # Volume Profile
            volume_profile = advanced_features.get('volume_profile', {})
            if volume_profile:
                st.subheader("Volume Profile")
                st.write(f"POC: {volume_profile.get('poc', 'N/A')}")
                st.write(f"POC Volume: {volume_profile.get('poc_volume', 0):,}")
        
        else:
            st.info("Run complete analysis to see advanced features")
    
    with tab3:
        st.subheader("🤖 Groq LLM Market Analysis")
        
        if st.session_state.get('analysis_complete'):
            market_data = prepare_llm_market_data()
            
            if st.button("🔄 Run LLM Analysis", type="primary"):
                with st.spinner("LLM analyzing market situation..."):
                    llm_analysis = llm_analyzer.analyze_market_situation(market_data)
                    
                    st.subheader("LLM Analysis Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Decision", llm_analysis.get('decision', 'WAIT'))
                        st.metric("Confidence", f"{llm_analysis.get('confidence', 0):.1f}%")
                        st.metric("Bias", llm_analysis.get('bias', 'Neutral'))
                    
                    with col2:
                        st.metric("Stop Loss", llm_analysis.get('stop_loss', 'N/A'))
                        st.write("Targets:", llm_analysis.get('targets', []))
                    
                    st.write("**Reasoning:**", llm_analysis.get('short_reason', ''))
                    
                    warnings = llm_analysis.get('warnings', [])
                    if warnings:
                        st.write("**Warnings:**")
                        for warning in warnings:
                            st.write(f"- {warning}")
        
        else:
            st.info("Run complete analysis first for LLM analysis")
    
    with tab4:
        st.subheader("🚀 Enhanced Master Trigger")
        
        if st.session_state.get('analysis_complete'):
            market_data = prepare_llm_market_data()
            advanced_features = st.session_state.get('advanced_features', {})
            unified_bias = st.session_state.overall_nifty_score
            ml_confidence = st.session_state.enhanced_ml_engine.predict_confidence(
                prepare_enhanced_feature_data()
            )
            
            trigger_ready = st.session_state.enhanced_master_engine.should_trigger_analysis(
                'NIFTY', unified_bias, ml_confidence, advanced_features
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if trigger_ready:
                    st.success("🎯 ENHANCED TRIGGER READY!")
                    st.write("Advanced conditions met:")
                    st.write(f"- Unified Bias: {unified_bias:.1f} >= 80")
                    st.write(f"- ML Confidence: {ml_confidence:.1f} >= 85")
                    st.write("- No traps detected")
                    st.write("- Favorable market phase")
                    
                    if st.button("🔥 EXECUTE ENHANCED TRIGGER", type="primary"):
                        with st.spinner("Executing enhanced analysis..."):
                            # This would be async in production
                            st.success("Enhanced trigger would execute here")
                            st.info("Check Telegram for alerts")
                else:
                    st.warning("🕐 Enhanced trigger conditions not met")
                    st.write("Required conditions:")
                    st.write("- Unified Bias >= 80")
                    st.write("- ML Confidence >= 85") 
                    st.write("- No trap patterns")
                    st.write("- Favorable time phase")
                    st.write(f"Current: Bias={unified_bias:.1f}, ML={ml_confidence:.1f}")
            
            with col2:
                st.subheader("Trigger Analytics")
                
                # Show advanced feature status
                st.write("**Feature Status:**")
                traps = advanced_features.get('trap_detection', {})
                st.write(f"- Traps: {'CLEAR' if not (traps.get('bull_trap_detected') or traps.get('bear_trap_detected')) else 'DETECTED'}")
                
                time_behavior = advanced_features.get('time_behavior', {})
                st.write(f"- Market Phase: {time_behavior.get('market_phase', 'UNKNOWN')}")
                
                liquidity = advanced_features.get('liquidity_sweeps', {})
                st.write(f"- Liquidity Sweeps: {'CLEAR' if not (liquidity.get('liquidity_sweep_high') or liquidity.get('liquidity_sweep_low')) else 'DETECTED'}")
        
        else:
            st.info("Run complete analysis to activate enhanced trigger")

def prepare_enhanced_feature_data():
    """Prepare enhanced feature data from all analysis components"""
    feature_data = {}
    
    # Technical data
    if st.session_state.get('last_result') and st.session_state['last_result'].get('success'):
        tech_data = st.session_state['last_result']
        feature_data['technical_bias_score'] = tech_data.get('overall_score', 0)
        feature_data['rsi'] = 50
        feature_data['ema_trend'] = 'bullish' if tech_data.get('overall_bias') == 'BULLISH' else 'bearish'
    
    # Options data
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY':
                feature_data['options_bias_score'] = instrument_data.get('bias_score', 0)
                feature_data['pcr_value'] = instrument_data.get('pcr_oi', 1.0)
                feature_data['atm_bias_score'] = instrument_data.get('bias_score', 0)
                break
    
    # Advanced features
    advanced_features = st.session_state.get('advanced_features', {})
    if advanced_features:
        # Add advanced features to feature data
        traps = advanced_features.get('trap_detection', {})
        if traps:
            feature_data['trap_confidence'] = traps.get('confidence', 0)
            feature_data['bull_trap'] = 1 if traps.get('bull_trap_detected') else 0
            feature_data['bear_trap'] = 1 if traps.get('bear_trap_detected') else 0
        
        time_behavior = advanced_features.get('time_behavior', {})
        if time_behavior:
            feature_data['confidence_modifier'] = time_behavior.get('confidence_modifier', 1.0)
        
        price_structure = advanced_features.get('price_structure', {})
        if price_structure:
            feature_data['breakout_imminent'] = 1 if price_structure.get('breakout_imminent') else 0
    
    # Unified bias
    feature_data['unified_bias'] = st.session_state.overall_nifty_score
    feature_data['sentiment_score'] = 50
    
    return feature_data

def prepare_llm_market_data():
    """Prepare market data for LLM analysis"""
    market_data = {
        'symbol': 'NIFTY',
        'timeframe': '15m',
        'current_price': 0,
        'unified_bias': st.session_state.overall_nifty_score,
        'technical_score': 0,
        'option_score': 0,
        'institutional_score': 0,
        'expiry_date': '',
        'days_to_expiry': 999,
        'expiry_spike': False
    }
    
    # Add price data
    if st.session_state.get('last_df') is not None:
        df = st.session_state['last_df']
        if not df.empty:
            market_data['current_price'] = df['Close'].iloc[-1]
    
    # Add option scores
    if st.session_state.market_bias_data:
        for instrument_data in st.session_state.market_bias_data:
            if instrument_data['instrument'] == 'NIFTY':
                market_data['option_score'] = instrument_data.get('bias_score', 0)
                break
    
    return market_data

def show_enhanced_overview():
    """Show enhanced overview dashboard"""
    st.header("🎯 Enhanced Market Overview")
    
    if not st.session_state.get('analysis_complete'):
        st.info("Run enhanced analysis to see overview")
        return
    
    # Enhanced metrics grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unified Bias", 
                 f"{st.session_state.overall_nifty_score:.1f}",
                 st.session_state.overall_nifty_bias)
    
    with col2:
        ml_confidence = st.session_state.enhanced_ml_engine.predict_confidence(
            prepare_enhanced_feature_data()
        )
        st.metric("XGBoost Confidence", f"{ml_confidence:.1f}%")
    
    with col3:
        # Show advanced feature status
        advanced_features = st.session_state.get('advanced_features', {})
        traps = advanced_features.get('trap_detection', {})
        trap_status = "CLEAR" if not (traps.get('bull_trap_detected') or traps.get('bear_trap_detected')) else "ALERT"
        st.metric("Trap Status", trap_status)
    
    with col4:
        time_behavior = advanced_features.get('time_behavior', {})
        market_phase = time_behavior.get('market_phase', 'UNKNOWN')
        st.metric("Market Phase", market_phase)
    
    # LLM Quick Analysis
    st.subheader("🤖 Quick LLM Insight")
    
    if st.button("Get Quick Analysis", type="secondary"):
        market_data = prepare_llm_market_data()
        llm_analysis = llm_analyzer.analyze_market_situation(market_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Decision:** {llm_analysis.get('decision', 'WAIT')}")
            st.write(f"**Confidence:** {llm_analysis.get('confidence', 0):.1f}%")
            st.write(f"**Bias:** {llm_analysis.get('bias', 'Neutral')}")
        
        with col2:
            st.write(f"**Stop Loss:** {llm_analysis.get('stop_loss', 'N/A')}")
            st.write(f"**Targets:** {llm_analysis.get('targets', [])}")
        
        st.write(f"**Reason:** {llm_analysis.get('short_reason', '')}")

def show_advanced_features_dashboard():
    """Show advanced features dashboard"""
    st.header("📊 Advanced Features Dashboard")
    
    if not st.session_state.get('analysis_complete'):
        st.info("Run enhanced analysis to see advanced features")
        return
    
    advanced_features = st.session_state.get('advanced_features', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trap Detection")
        traps = advanced_features.get('trap_detection', {})
        if traps:
            st.metric("Bull Trap", "✅ DETECTED" if traps.get('bull_trap_detected') else "❌ CLEAR")
            st.metric("Bear Trap", "✅ DETECTED" if traps.get('bear_trap_detected') else "❌ CLEAR")
            st.metric("Confidence", f"{traps.get('confidence', 0):.1f}%")
        else:
            st.info("No trap data available")
    
    with col2:
        st.subheader("Price Structure")
        price_structure = advanced_features.get('price_structure', {})
        if price_structure:
            st.metric("Market Structure", price_structure.get('market_structure', 'UNKNOWN'))
            st.metric("Breakout Imminent", "✅ YES" if price_structure.get('breakout_imminent') else "❌ NO")
            st.metric("Internal BOS", "✅ YES" if price_structure.get('internal_bos') else "❌ NO")
        else:
            st.info("No structure data available")
    
    st.subheader("Volume Profile")
    volume_profile = advanced_features.get('volume_profile', {})
    if volume_profile:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("POC", volume_profile.get('poc', 'N/A'))
        with col2:
            st.metric("POC Volume", f"{volume_profile.get('poc_volume', 0):,}")
        with col3:
            value_area = volume_profile.get('value_area', {})
            st.metric("Value Area Coverage", f"{value_area.get('total_volume_covered', 0):.1%}")
    else:
        st.info("No volume profile data available")

def show_realtime_analytics():
    """Show real-time analytics dashboard"""
    st.header("⚡ Real-time Analytics")
    
    # This would connect to WebSocket or streaming data in production
    st.info("Real-time analytics dashboard would show here")
    st.write("Features would include:")
    st.write("- Live option chain updates")
    st.write("- Real-time gamma exposure")
    st.write("- Instant trap detection")
    st.write("- Continuous ML inference")

def show_master_triggers_dashboard():
    """Show master triggers dashboard"""
    st.header("🚀 Master Triggers Dashboard")
    
    st.info("Master triggers monitoring dashboard would show here")
    st.write("Monitoring:")
    st.write("- Trigger conditions met")
    st.write("- LLM analysis results") 
    st.write("- Telegram alert status")
    st.write("- Performance metrics")

def run_enhanced_analysis():
    """Run enhanced analysis with all new features"""
    try:
        # Run existing analysis
        if not run_complete_analysis():
            return False
        
        # Enhanced feature engineering
        if st.session_state['last_df'] is not None:
            df = st.session_state['last_df']
            feature_engine = st.session_state.feature_engine
            
            # Calculate all advanced features
            advanced_features = {}
            
            # Sample option chain data (in real app, this would come from API)
            sample_option_chain = {
                'strikes': [
                    {'strike': 22000, 'CE_OI': 100000, 'PE_OI': 80000, 'CE_IV': 15.5, 'PE_IV': 16.0},
                    {'strike': 22100, 'CE_OI': 120000, 'PE_OI': 90000, 'CE_IV': 16.0, 'PE_IV': 16.5},
                    {'strike': 22200, 'CE_OI': 150000, 'PE_OI': 110000, 'CE_IV': 16.5, 'PE_IV': 17.0},
                ]
            }
            
            # Calculate advanced features
            advanced_features['put_call_pressure'] = feature_engine.calculate_put_call_pressure(sample_option_chain)
            advanced_features['iv_dynamics'] = feature_engine.calculate_iv_dynamics(sample_option_chain, [])
            advanced_features['delta_neutral'] = feature_engine.calculate_delta_neutral_zones(sample_option_chain)
            advanced_features['trap_detection'] = feature_engine.detect_trap_patterns(df, sample_option_chain, 15.0)
            advanced_features['time_behavior'] = feature_engine.calculate_time_based_behavior(datetime.now(IST))
            advanced_features['price_structure'] = feature_engine.detect_price_structure(df)
            advanced_features['volume_profile'] = feature_engine.calculate_volume_profile(df)
            advanced_features['liquidity_sweeps'] = feature_engine.detect_liquidity_sweeps(df)
            advanced_features['momentum_curvature'] = feature_engine.calculate_momentum_curvature(df)
            
            st.session_state.advanced_features = advanced_features
            
            # Store in Supabase (commented for demo)
            if supabase.connected:
                features_data = {
                    'symbol': 'NIFTY',
                    'timeframe': '5m',
                    'ts': datetime.now(IST).isoformat(),
                    'ohlc': {
                        'o': df['Open'].iloc[-1] if not df.empty else 0,
                        'h': df['High'].iloc[-1] if not df.empty else 0,
                        'l': df['Low'].iloc[-1] if not df.empty else 0,
                        'c': df['Close'].iloc[-1] if not df.empty else 0,
                        'volume': df['Volume'].iloc[-1] if not df.empty else 0
                    },
                    'features': prepare_enhanced_feature_data(),
                    'unified_bias': st.session_state.overall_nifty_score,
                    'technical_score': st.session_state['last_result'].get('overall_score', 0) if st.session_state.get('last_result') else 0,
                    'option_score': 0,  # Would be calculated from options data
                    'institutional_score': 0,  # Would be calculated
                    'volume_confirmation': True,
                    'gamma_sequence': advanced_features.get('gamma_sequence', {}),
                    'option_chain': sample_option_chain,
                    'atm_strikes': {'atm': 22100, 'strikes': [22000, 22100, 22200]},
                    'oi_trend': advanced_features.get('oi_trend', {}),
                    'ml_confidence': st.session_state.enhanced_ml_engine.predict_confidence(prepare_enhanced_feature_data()),
                    'model_version': st.session_state.enhanced_ml_engine.model_version
                }
                
                # supabase.insert_market_features(features_data)  # Uncomment to enable
            
        logger.info("✅ Enhanced analysis complete with all features")
        return True
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        return False

def run_complete_analysis():
    """Run complete analysis for all tabs"""
    try:
        # Technical Analysis
        analysis = BiasAnalysisPro()
        df_fetched = analysis.fetch_data("^NSEI", period='7d', interval='5m')
        
        if df_fetched is None or df_fetched.empty:
            st.error("No data fetched. Check symbol or network.")
            return False
            
        st.session_state['last_df'] = df_fetched
        st.session_state['fetch_time'] = datetime.now(IST)

        # Run bias analysis
        result = analysis.analyze_all_bias_indicators("^NSEI")
        st.session_state['last_result'] = result

        # Run Volume Order Blocks analysis
        vob_indicator = VolumeOrderBlocks(sensitivity=5)
        if st.session_state['last_df'] is not None:
            df = st.session_state['last_df']
            bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
            st.session_state.vob_blocks = {
                'bullish': bullish_blocks,
                'bearish': bearish_blocks
            }

        # Run options analysis
        options_analyzer = SimpleOptionsAnalyzer()
        enhanced_bias_data = []
        instruments = ['NIFTY', 'BANKNIFTY']
        
        for instrument in instruments:
            bias_data = options_analyzer.get_sample_options_data(instrument)
            enhanced_bias_data.append(bias_data)
        
        st.session_state.market_bias_data = enhanced_bias_data
        st.session_state.last_bias_update = datetime.now(IST)
        
        # Calculate overall bias
        st.session_state.overall_nifty_bias = "BULLISH" if result.get('overall_bias') == 'BULLISH' else "BEARISH" if result.get('overall_bias') == 'BEARISH' else "NEUTRAL"
        st.session_state.overall_nifty_score = result.get('overall_score', 0)
        
        st.session_state.analysis_complete = True
        return True
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        return False

# =============================================
# MAIN ENHANCED APP
# =============================================

def main():
    st.set_page_config(
        page_title="Bias Analysis Pro - ENHANCED", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Bias Analysis Pro — ENHANCED EDITION")
    st.markdown("""
    **Advanced market analysis with:**
    - 🤖 XGBoost ML Predictions
    - 🧠 Groq LLM Reasoning
    - 📊 Advanced Feature Engineering
    - 🚀 Enhanced Master Triggers
    - ⚡ 30-Second Refresh Cycle
    """)
    
    # Enhanced Sidebar
    st.sidebar.header("⚡ Enhanced Controls")
    
    # Auto-refresh with 30-second option
    st.sidebar.header("Auto-Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Enable 30s Auto-Refresh", value=False)
    refresh_interval = st.sidebar.selectbox("Refresh Interval", 
                                          options=['30s', '1m', '2m', '5m'], 
                                          index=0)
    
    # Advanced features toggle
    st.sidebar.header("Advanced Features")
    enable_advanced = st.sidebar.checkbox("Enable Advanced Features", value=True)
    enable_llm = st.sidebar.checkbox("Enable LLM Analysis", value=True)
    enable_xgboost = st.sidebar.checkbox("Enable XGBoost ML", value=True)
    
    # Initialize enhanced components
    if 'enhanced_ml_engine' not in st.session_state:
        st.session_state.enhanced_ml_engine = EnhancedMLPredictionEngine()
        st.session_state.enhanced_ml_engine.load_model()
    
    if 'enhanced_master_engine' not in st.session_state:
        st.session_state.enhanced_master_engine = EnhancedMasterTriggerEngine()
    
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = AdvancedFeatureEngine()
    
    # Enhanced analysis button
    if st.sidebar.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True):
        with st.spinner("Running enhanced analysis with all features..."):
            if run_enhanced_analysis():
                st.sidebar.success("Enhanced analysis complete!")
                st.rerun()
    
    # Display enhanced metrics
    st.sidebar.markdown("---")
    st.sidebar.header("Enhanced Metrics")
    
    if st.session_state.get('analysis_complete'):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("XGBoost Conf", f"{st.session_state.get('ml_confidence', 0):.1f}%")
        with col2:
            st.metric("LLM Ready", "✅" if enable_llm else "❌")
    
    # Enhanced main tabs
    tabs = st.tabs([
        "🎯 Enhanced Overview", 
        "🤖 ML & AI Engine", 
        "📊 Advanced Features",
        "⚡ Real-time Analytics",
        "🚀 Master Triggers"
    ])
    
    with tabs[0]:
        show_enhanced_overview()
    
    with tabs[1]:
        add_enhanced_ml_analysis_tab()
    
    with tabs[2]:
        show_advanced_features_dashboard()
    
    with tabs[3]:
        show_realtime_analytics()
    
    with tabs[4]:
        show_master_triggers_dashboard()

if __name__ == "__main__":
    main()