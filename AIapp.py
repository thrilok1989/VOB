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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

# =============================================
# ML SYSTEM - INTEGRATED DIRECTLY
# =============================================

class MLDataCollector:
    """Collects all features from your app for ML training"""
    
    def __init__(self, data_file="ml_training_data.json"):
        self.data_file = data_file
        self.historical_data = self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing training data"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_data(self):
        """Save data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2, default=str)
    
    def collect_all_features(self, technical_data, options_data, volume_data, market_intel):
        """Collect features from all your app tabs"""
        
        features = {
            'timestamp': datetime.now().isoformat(),
            
            # Technical Features (Tab 1)
            'ema_trend_alignment': self.get_ema_alignment(technical_data),
            'rsi_momentum': technical_data.get('rsi', 50),
            'macd_signal': technical_data.get('macd_signal', 0),
            'atr_volatility': technical_data.get('atr', 0),
            'sr_distance': self.calculate_sr_distance(technical_data),
            'volume_block_presence': volume_data.get('block_imbalance', 0),
            
            # Volume/Order Block Features (Tab 2)
            'volume_spike_ratio': volume_data.get('volume_spike_ratio', 1.0),
            'block_imbalance': volume_data.get('bullish_blocks', 0) - volume_data.get('bearish_blocks', 0),
            'structure_break': volume_data.get('structure_broken', False),
            'ob_validity': volume_data.get('valid_blocks', True),
            'liquidity_grab': volume_data.get('liquidity_grab', False),
            
            # Options Chain Features (Tab 3)
            'oi_momentum': options_data.get('oi_momentum', 0),
            'iv_skew_trend': options_data.get('iv_skew', 0),
            'max_pain_shift': options_data.get('max_pain_shift', 0),
            'straddle_pressure': options_data.get('straddle_pressure', 0),
            'atm_bias_strength': options_data.get('atm_bias_score', 0),
            'oi_activity': options_data.get('oi_activity', 'neutral'),
            'pcr_momentum': options_data.get('pcr_momentum', 1.0),
            'bid_ask_trend': options_data.get('bid_ask_trend', 0),
            
            # Market Intel Features (Tab 4)
            'vix_trend': market_intel.get('vix_trend', 0),
            'sgx_nifty_gap': market_intel.get('sgx_gap', 0),
            'global_correlation': market_intel.get('global_corr', 0),
            'fii_dii_flow': market_intel.get('fii_dii_flow', 0),
            'sector_rotation': market_intel.get('sector_rotation', 0),
            
            # Your Bias Scores
            'technical_bias': technical_data.get('bias_score', 0),
            'options_bias': options_data.get('bias_score', 0),
            'institutional_bias': options_data.get('institutional_score', 0),
            'volume_confirmation': 1 if volume_data.get('confirms_price') else 0,
            'market_sentiment': market_intel.get('sentiment_score', 50),
            
            # Store current price for future labeling
            'current_price': technical_data.get('current_price', 0),
            'support_levels': technical_data.get('support_levels', []),
            'resistance_levels': technical_data.get('resistance_levels', [])
        }
        
        return features
    
    def get_ema_alignment(self, technical_data):
        """Calculate EMA trend alignment across timeframes"""
        try:
            # Extract EMA trends from your technical data
            ema_5m = technical_data.get('ema_5m_trend', 'neutral')
            ema_15m = technical_data.get('ema_15m_trend', 'neutral') 
            ema_1h = technical_data.get('ema_1h_trend', 'neutral')
            
            # Convert to numerical values
            def trend_to_num(trend):
                if 'bull' in str(trend).lower():
                    return 1
                elif 'bear' in str(trend).lower():
                    return -1
                return 0
            
            trends = [trend_to_num(ema_5m), trend_to_num(ema_15m), trend_to_num(ema_1h)]
            return sum(trends) / len(trends)
            
        except:
            return 0
    
    def calculate_sr_distance(self, technical_data):
        """Calculate distance to nearest S/R level"""
        try:
            current_price = technical_data.get('current_price', 0)
            supports = technical_data.get('support_levels', [])
            resistances = technical_data.get('resistance_levels', [])
            
            if not supports and not resistances or current_price == 0:
                return 0
                
            # Find nearest support and resistance
            nearest_support = min([abs(current_price - s) for s in supports]) if supports else float('inf')
            nearest_resistance = min([abs(current_price - r) for r in resistances]) if resistances else float('inf')
            
            distance_to_nearest = min(nearest_support, nearest_resistance)
            
            # Normalize as percentage of price
            return (distance_to_nearest / current_price) * 100
            
        except:
            return 0
    
    def add_data_point(self, features):
        """Add new data point to training set"""
        self.historical_data.append(features)
        self.save_data()
        print(f"✅ ML Data point collected. Total: {len(self.historical_data)}")
    
    def label_data_with_outcomes(self):
        """Go back and label historical data with what actually happened"""
        print("🔍 Labeling historical data with outcomes...")
        
        for i, data_point in enumerate(self.historical_data):
            if 'labels' not in data_point:
                data_point['labels'] = self.calculate_labels(data_point, i)
        
        self.save_data()
        print("✅ Data labeling complete!")
    
    def calculate_labels(self, data_point, index):
        """Calculate what actually happened after this data point"""
        # NOTE: You need to implement this with real historical price data
        # This is a placeholder - replace with actual price movement analysis
        
        return {
            'breakout_real': np.random.choice([0, 1], p=[0.4, 0.6]),  # Placeholder
            'direction_15min': np.random.choice([0, 1], p=[0.45, 0.55]),
            'direction_1hr': np.random.choice([0, 1], p=[0.4, 0.6]),  
            'target_hit_1hr': np.random.choice([0, 1], p=[0.5, 0.5]),
            'price_change_1hr': np.random.uniform(-2, 2)
        }

class MLModelTrainer:
    """Trains ML models on your collected data"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self, data_file="ml_training_data.json"):
        """Load and prepare data for training"""
        
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Filter only labeled data
        labeled_data = df[df['labels'].notna()]
        
        if len(labeled_data) == 0:
            raise ValueError("No labeled data found! Run label_data_with_outcomes() first.")
        
        # Prepare features
        feature_columns = [
            'ema_trend_alignment', 'rsi_momentum', 'macd_signal', 'atr_volatility',
            'sr_distance', 'volume_block_presence', 'volume_spike_ratio', 
            'block_imbalance', 'structure_break', 'ob_validity', 'liquidity_grab',
            'oi_momentum', 'iv_skew_trend', 'max_pain_shift', 'straddle_pressure',
            'atm_bias_strength', 'pcr_momentum', 'bid_ask_trend', 'vix_trend',
            'sgx_nifty_gap', 'global_correlation', 'fii_dii_flow', 'sector_rotation',
            'technical_bias', 'options_bias', 'institutional_bias', 'volume_confirmation',
            'market_sentiment'
        ]
        
        # Ensure all columns exist
        available_columns = [col for col in feature_columns if col in labeled_data.columns]
        X = labeled_data[available_columns].fillna(0)
        
        # Prepare labels
        y_breakout = labeled_data['labels'].apply(lambda x: x.get('breakout_real', 0) if isinstance(x, dict) else 0)
        y_direction_15min = labeled_data['labels'].apply(lambda x: x.get('direction_15min', 0) if isinstance(x, dict) else 0)
        y_direction_1hr = labeled_data['labels'].apply(lambda x: x.get('direction_1hr', 0) if isinstance(x, dict) else 0)
        y_target_1hr = labeled_data['labels'].apply(lambda x: x.get('target_hit_1hr', 0) if isinstance(x, dict) else 0)
        
        return X, y_breakout, y_direction_15min, y_direction_1hr, y_target_1hr, available_columns
    
    def train_models(self, data_file="ml_training_data.json"):
        """Train all ML models"""
        
        print("🚀 Training ML Models...")
        
        X, y_breakout, y_dir_15, y_dir_1h, y_target, feature_names = self.load_and_prepare_data(data_file)
        
        if len(X) < 100:
            print(f"⚠️ Not enough data for training. Have {len(X)} samples, need at least 100.")
            return self.models
        
        # Split data
        X_train, X_test, y_breakout_train, y_breakout_test = train_test_split(X, y_breakout, test_size=0.2, random_state=42)
        
        print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Model 1: Breakout vs Fakeout (XGBoost - Most Important)
        print("📈 Training Breakout/Fakeout Model...")
        self.models['breakout'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.models['breakout'].fit(X_train, y_breakout_train)
        
        # Model 2: Direction 15min (Random Forest)
        print("🎯 Training 15min Direction Model...")
        _, X_test_dir, _, y_dir_15_test = train_test_split(X, y_dir_15, test_size=0.2, random_state=42)
        self.models['direction_15min'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.models['direction_15min'].fit(X, y_dir_15)
        
        # Model 3: Direction 1hr (LightGBM)
        print("🎯 Training 1hr Direction Model...")
        _, X_test_dir1h, _, y_dir_1h_test = train_test_split(X, y_dir_1h, test_size=0.2, random_state=42)
        self.models['direction_1hr'] = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            random_state=42
        )
        self.models['direction_1hr'].fit(X, y_dir_1h)
        
        # Model 4: Target Hit (Gradient Boosting)
        print("🎯 Training Target Hit Model...")
        _, X_test_target, _, y_target_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
        self.models['target_hit'] = GradientBoostingClassifier(
            n_estimators=60,
            max_depth=4,
            random_state=42
        )
        self.models['target_hit'].fit(X, y_target)
        
        # Calculate feature importance
        self.calculate_feature_importance(feature_names)
        
        # Evaluate models
        self.evaluate_models(X_test, y_breakout_test, X_test_dir, y_dir_15_test, X_test_dir1h, y_dir_1h_test, X_test_target, y_target_test)
        
        print("✅ All models trained successfully!")
        
        return self.models
    
    def calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance"""
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance[model_name] = dict(zip(feature_names, importance))
    
    def evaluate_models(self, X_breakout_test, y_breakout_test, X_dir_test, y_dir_15_test, X_dir1h_test, y_dir_1h_test, X_target_test, y_target_test):
        """Evaluate model performance"""
        
        print("\n📊 MODEL PERFORMANCE:")
        print("="*50)
        
        # Breakout Model
        breakout_pred = self.models['breakout'].predict(X_breakout_test)
        breakout_acc = accuracy_score(y_breakout_test, breakout_pred)
        print(f"🎯 Breakout/Fakeout Model: {breakout_acc:.2%}")
        
        # Direction 15min Model
        dir_15_pred = self.models['direction_15min'].predict(X_dir_test)
        dir_15_acc = accuracy_score(y_dir_15_test, dir_15_pred)
        print(f"🎯 15min Direction Model: {dir_15_acc:.2%}")
        
        # Direction 1hr Model
        dir_1h_pred = self.models['direction_1hr'].predict(X_dir1h_test)
        dir_1h_acc = accuracy_score(y_dir_1h_test, dir_1h_pred)
        print(f"🎯 1hr Direction Model: {dir_1h_acc:.2%}")
        
        # Target Hit Model
        target_pred = self.models['target_hit'].predict(X_target_test)
        target_acc = accuracy_score(y_target_test, target_pred)
        print(f"🎯 Target Hit Model: {target_acc:.2%}")
        
        print("="*50)
    
    def save_models(self, model_dir="ml_models"):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{model_dir}/{model_name}_model.pkl"
            joblib.dump(model, filename)
            print(f"💾 Saved {model_name} to {filename}")
        
        # Save feature importance
        importance_file = f"{model_dir}/feature_importance.json"
        with open(importance_file, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
    
    def load_models(self, model_dir="ml_models"):
        """Load trained models"""
        model_files = {
            'breakout': f"{model_dir}/breakout_model.pkl",
            'direction_15min': f"{model_dir}/direction_15min_model.pkl", 
            'direction_1hr': f"{model_dir}/direction_1hr_model.pkl",
            'target_hit': f"{model_dir}/target_hit_model.pkl"
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                self.models[model_name] = joblib.load(file_path)
                print(f"📂 Loaded {model_name} model")
            else:
                print(f"❌ Model file not found: {file_path}")
        
        return self.models

class MLConfidenceEngine:
    """Main ML engine that provides confidence scores"""
    
    def __init__(self):
        self.models_loaded = False
        self.trainer = MLModelTrainer()
        
    def load_models(self):
        """Load ML models"""
        try:
            self.trainer.load_models()
            self.models_loaded = True
            print("✅ ML Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
            return False
    
    def calculate_ml_confidence(self, feature_dict: Dict) -> float:
        """Calculate overall ML confidence score 0-100"""
        
        if not self.models_loaded:
            print("⚠️ Models not loaded, returning default confidence")
            return 50.0
        
        try:
            # Convert features to DataFrame for prediction
            feature_df = pd.DataFrame([feature_dict])
            
            # Get probabilities from all models
            breakout_prob = self.trainer.models['breakout'].predict_proba(feature_df)[0][1]
            direction_15min_prob = self.trainer.models['direction_15min'].predict_proba(feature_df)[0][1]
            direction_1hr_prob = self.trainer.models['direction_1hr'].predict_proba(feature_df)[0][1]
            target_prob = self.trainer.models['target_hit'].predict_proba(feature_df)[0][1]
            
            # Weighted average (breakout prediction is most important)
            ml_confidence = (
                breakout_prob * 0.40 +       # Breakout prediction most critical
                direction_1hr_prob * 0.25 +  # 1hr direction important
                direction_15min_prob * 0.20 + # 15min direction
                target_prob * 0.15           # Target hit probability
            ) * 100
            
            print(f"🔍 ML Confidence Breakdown:")
            print(f"   Breakout Probability: {breakout_prob:.2%}")
            print(f"   15min Direction: {direction_15min_prob:.2%}")
            print(f"   1hr Direction: {direction_1hr_prob:.2%}") 
            print(f"   Target Hit: {target_prob:.2%}")
            print(f"   FINAL ML CONFIDENCE: {ml_confidence:.1f}%")
            
            return ml_confidence
            
        except Exception as e:
            print(f"❌ Error calculating ML confidence: {e}")
            return 50.0
    
    def get_detailed_predictions(self, feature_dict: Dict) -> Dict:
        """Get detailed predictions from all models"""
        
        if not self.models_loaded:
            return {}
        
        try:
            feature_df = pd.DataFrame([feature_dict])
            
            predictions = {
                'breakout_confidence': self.trainer.models['breakout'].predict_proba(feature_df)[0][1] * 100,
                'direction_15min_confidence': self.trainer.models['direction_15min'].predict_proba(feature_df)[0][1] * 100,
                'direction_1hr_confidence': self.trainer.models['direction_1hr'].predict_proba(feature_df)[0][1] * 100,
                'target_hit_confidence': self.trainer.models['target_hit'].predict_proba(feature_df)[0][1] * 100,
            }
            
            return predictions
        except Exception as e:
            print(f"❌ Error getting detailed predictions: {e}")
            return {}

class MasterTrigger:
    """The final decision engine that combines everything"""
    
    def __init__(self):
        self.ml_engine = MLConfidenceEngine()
        self.data_collector = MLDataCollector()
        
    def initialize_system(self):
        """Initialize the complete ML system"""
        print("🚀 Initializing Master Trigger System...")
        ml_loaded = self.ml_engine.load_models()
        if ml_loaded:
            print("✅ Master Trigger System Ready!")
        else:
            print("⚠️ ML models not loaded, system running in basic mode")
        return ml_loaded
    
    def should_trigger_ai_analysis(self, unified_bias_score, current_features):
        """MASTER DECISION: Should we trigger AI analysis?"""
        
        # 1. Get ML confidence
        ml_confidence = self.ml_engine.calculate_ml_confidence(current_features)
        
        print(f"\n🎯 MASTER TRIGGER ANALYSIS:")
        print(f"   Unified Bias Score: {unified_bias_score:.1f}%")
        print(f"   ML Confidence: {ml_confidence:.1f}%")
        print(f"   Threshold: Bias >= 80% & ML >= 85%")
        
        # 2. MASTER TRIGGER CONDITION
        if unified_bias_score >= 80 and ml_confidence >= 85:
            print("🔥 🔥 🔥 MASTER TRIGGER ACTIVATED! 🔥 🔥 🔥")
            return True, ml_confidence
        else:
            print("❌ Conditions not met for AI trigger")
            return False, ml_confidence
    
    def trigger_ai_analysis_pipeline(self, current_features):
        """Trigger complete AI analysis when conditions are met"""
        
        print("\n🤖 INITIATING AI ANALYSIS PIPELINE...")
        
        # 1. Trigger News API
        news_analysis = self.trigger_news_api(current_features)
        
        # 2. Trigger AI Interpretation
        ai_interpretation = self.trigger_ai_interpretation(news_analysis)
        
        # 3. Generate Final Prediction
        final_prediction = self.generate_final_prediction(news_analysis, ai_interpretation, current_features)
        
        print("✅ AI ANALYSIS PIPELINE COMPLETE!")
        
        return {
            'news_analysis': news_analysis,
            'ai_interpretation': ai_interpretation, 
            'final_prediction': final_prediction,
            'trigger_timestamp': datetime.now().isoformat(),
            'confidence_level': 'VERY_HIGH'
        }
    
    def trigger_news_api(self, features):
        """Trigger news sentiment analysis"""
        print("📰 Fetching news sentiment analysis...")
        # Placeholder - integrate with your actual news API
        return {
            'overall_sentiment': 'BULLISH',
            'key_news_items': ['Market shows strong bullish momentum'],
            'sentiment_score': 85,
            'impact_level': 'HIGH'
        }
    
    def trigger_ai_interpretation(self, news_analysis):
        """Trigger AI interpretation of all data"""
        print("🧠 Running AI market interpretation...")
        # Placeholder - integrate with your HuggingFace AI
        return {
            'market_outlook': 'STRONG_BULLISH_MOMENTUM',
            'confidence': 88,
            'key_factors': ['Options alignment', 'Volume confirmation', 'Institutional bias'],
            'timeframe': '1-4 hours',
            'risk_level': 'MEDIUM'
        }
    
    def generate_final_prediction(self, news_analysis, ai_interpretation, features):
        """Generate final trading prediction"""
        
        print("🎯 Generating final trading prediction...")
        
        current_price = features.get('current_price', 0)
        
        return {
            'action': 'STRONG_BUY',
            'entry_price': current_price,
            'targets': self.calculate_targets(current_price, features),
            'stoploss': self.calculate_stoploss(current_price, features),
            'timeframe': ai_interpretation.get('timeframe', '1-4 hours'),
            'confidence': min(85, ai_interpretation.get('confidence', 0)),
            'rationale': 'High confidence alignment across all systems',
            'risk_reward': '1:2.5',
            'position_size': 'Medium'
        }
    
    def calculate_targets(self, current_price, features):
        """Calculate target levels"""
        resistances = features.get('resistance_levels', [])
        
        if resistances and current_price > 0:
            next_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.02)
            return [next_resistance, next_resistance * 1.005]
        
        return [current_price * 1.015, current_price * 1.025]
    
    def calculate_stoploss(self, current_price, features):
        """Calculate stoploss level"""
        supports = features.get('support_levels', [])
        
        if supports and current_price > 0:
            nearest_support = max([s for s in supports if s < current_price], default=current_price * 0.99)
            return nearest_support
        
        return current_price * 0.985

# =============================================
# YOUR ORIGINAL APP CODE CONTINUES...
# =============================================

# Import Dhan API for Indian indices volume data
try:
    from dhan_data_fetcher import DhanDataFetcher
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    print("Warning: Dhan API not available. Volume data may be missing for Indian indices.")

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
        """Fetch data from Dhan API (for Indian indices) or Yahoo Finance (for others)"""
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
# VOLUME ORDER BLOCKS (FROM SECOND APP)
# =============================================

class VolumeOrderBlocks:
    """Python implementation of Volume Order Blocks indicator by BigBeluga"""
    
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
    
    def check_price_near_blocks(self, current_price: float, blocks: List[Dict[str, Any]], threshold: float = 5) -> List[Dict[str, Any]]:
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

# FIX 5: Add plotting function for VOB
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
# STREAMLIT APP UI - ENHANCED WITH ML
# =============================================

def main():
    st.set_page_config(page_title="Bias Analysis Pro - Complete Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 Bias Analysis Pro — Complete Single-file App")
    st.markdown(
        "This Streamlit app wraps the **BiasAnalysisPro** engine (Pine → Python) and shows bias summary, "
        "price action, option chain analysis, and bias tabulation."
    )

    # Initialize all analyzers
    analysis = BiasAnalysisPro()
    options_analyzer = NSEOptionsAnalyzer()
    vob_indicator = VolumeOrderBlocks(sensitivity=5)

    # Initialize ML System
    if 'master_trigger' not in st.session_state:
        st.session_state.master_trigger = MasterTrigger()
        st.session_state.master_trigger.initialize_system()

    if 'ml_collector' not in st.session_state:
        st.session_state.ml_collector = MLDataCollector()

    # Sidebar inputs
    st.sidebar.header("Data & Symbol")
    symbol_input = st.sidebar.text_input("Symbol (Yahoo/Dhan)", value="^NSEI")
    period_input = st.sidebar.selectbox("Period", options=['1d', '5d', '7d', '1mo'], index=2)
    interval_input = st.sidebar.selectbox("Interval", options=['1m', '5m', '15m', '1h'], index=1)

    # Auto-refresh configuration
    st.sidebar.header("Auto-Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (minutes)", min_value=1, max_value=10, value=1)

    # Telegram Configuration
    st.sidebar.header("🔔 Telegram Alerts")
    if telegram_notifier.is_configured():
        st.sidebar.success("✅ Telegram configured via secrets!")
        telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    else:
        st.sidebar.warning("⚠️ Telegram not configured")
        telegram_enabled = False

    # Shared state storage
    if 'last_df' not in st.session_state:
        st.session_state['last_df'] = None
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    if 'last_symbol' not in st.session_state:
        st.session_state['last_symbol'] = None
    if 'fetch_time' not in st.session_state:
        st.session_state['fetch_time'] = None
    if 'market_bias_data' not in st.session_state:
        st.session_state.market_bias_data = None
    if 'last_bias_update' not in st.session_state:
        st.session_state.last_bias_update = None
    if 'overall_nifty_bias' not in st.session_state:
        st.session_state.overall_nifty_bias = "NEUTRAL"
    if 'overall_nifty_score' not in st.session_state:
        st.session_state.overall_nifty_score = 0
    if 'atm_detailed_bias' not in st.session_state:
        st.session_state.atm_detailed_bias = None
    if 'vob_blocks' not in st.session_state:
        st.session_state.vob_blocks = {'bullish': [], 'bearish': []}
    if 'last_telegram_alert' not in st.session_state:
        st.session_state.last_telegram_alert = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'ml_confidence' not in st.session_state:
        st.session_state.ml_confidence = 0
    if 'ai_results' not in st.session_state:
        st.session_state.ai_results = None

    # Initialize session state for auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0

    # Function to calculate ATM detailed bias score
    def calculate_atm_detailed_bias(detailed_bias_data: Dict) -> Tuple[str, float]:
        """Calculate overall ATM bias from detailed bias metrics"""
        if not detailed_bias_data:
            return "NEUTRAL", 0
        
        bias_scores = []
        bias_weights = []
        
        # Define bias mappings with weights
        bias_mappings = {
            'OI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
            'ChgOI_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
            'Volume_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
            'Delta_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
            'Gamma_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
            'Premium_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
            'AskQty_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
            'BidQty_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.0},
            'IV_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
            'DVP_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
            'Delta_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 2.0},
            'Gamma_Exposure_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5},
            'IV_Skew_Bias': {'Bullish': 1, 'Bearish': -1, 'weight': 1.5}
        }
        
        # Calculate weighted scores
        total_weight = 0
        total_score = 0
        
        for bias_key, mapping in bias_mappings.items():
            if bias_key in detailed_bias_data:
                bias_value = detailed_bias_data[bias_key]
                if bias_value in mapping:
                    score = mapping[bias_value]
                    weight = mapping['weight']
                    total_score += score * weight
                    total_weight += weight
        
        if total_weight == 0:
            return "NEUTRAL", 0
        
        # Normalize score to -100 to 100 range
        normalized_score = (total_score / total_weight) * 100
        
        # Determine bias direction
        if normalized_score > 15:
            bias = "BULLISH"
        elif normalized_score < -15:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        return bias, normalized_score

    # Function to prepare ML features
    def prepare_ml_features(technical_data, volume_data, options_data):
        """Prepare features for ML analysis"""
        
        # Create market intel placeholder (you can enhance this)
        market_intel = {
            'vix_trend': 0,  # Placeholder - integrate with real VIX data
            'sgx_gap': 0,    # Placeholder - integrate with SGX Nifty
            'global_corr': 0, # Placeholder - global correlation
            'fii_dii_flow': 0, # Placeholder - institutional flow
            'sector_rotation': 0, # Placeholder
            'sentiment_score': 50
        }
        
        features = st.session_state.ml_collector.collect_all_features(
            technical_data, options_data, volume_data, market_intel
        )
        
        return features

    # Function to run complete analysis
    def run_complete_analysis():
        """Run complete analysis for all tabs - NOW WITH ML"""
        try:
            st.session_state['last_symbol'] = symbol_input
            
            # Technical Analysis
            with st.spinner("Fetching data and running technical analysis..."):
                df_fetched = analysis.fetch_data(symbol_input, period=period_input, interval=interval_input)
                if df_fetched is None or df_fetched.empty:
                    st.error("No data fetched. Check symbol or network.")
                    return False
                    
                st.session_state['last_df'] = df_fetched
                st.session_state['fetch_time'] = datetime.now(IST)

            # Run bias analysis
            with st.spinner("Running full bias analysis..."):
                result = analysis.analyze_all_bias_indicators(symbol_input)
                st.session_state['last_result'] = result

            # Run Volume Order Blocks analysis
            with st.spinner("Detecting Volume Order Blocks..."):
                if st.session_state['last_df'] is not None:
                    df = st.session_state['last_df']
                    bullish_blocks, bearish_blocks = vob_indicator.detect_volume_order_blocks(df)
                    st.session_state.vob_blocks = {
                        'bullish': bullish_blocks,
                        'bearish': bearish_blocks
                    }

            # Run options analysis
            with st.spinner("Running options analysis..."):
                enhanced_bias_data = []
                instruments = ['NIFTY']  # Focus on NIFTY for demo
                
                for instrument in instruments:
                    try:
                        # For demo, create mock options data
                        options_data = {
                            'oi_momentum': np.random.uniform(-1, 1),
                            'iv_skew': np.random.uniform(-0.5, 0.5),
                            'max_pain_shift': np.random.uniform(-100, 100),
                            'straddle_pressure': np.random.uniform(0, 1),
                            'atm_bias_score': np.random.uniform(-100, 100),
                            'oi_activity': 'neutral',
                            'pcr_momentum': np.random.uniform(0.5, 1.5),
                            'bid_ask_trend': np.random.uniform(-0.2, 0.2),
                            'bias_score': np.random.uniform(-100, 100),
                            'institutional_score': np.random.uniform(-100, 100)
                        }
                        
                        enhanced_bias_data.append({
                            'instrument': instrument,
                            'spot_price': df_fetched['Close'].iloc[-1] if not df_fetched.empty else 0,
                            'options_data': options_data
                        })
                    except Exception as e:
                        print(f"Error in options analysis for {instrument}: {e}")
                
                st.session_state.market_bias_data = enhanced_bias_data
                st.session_state.last_bias_update = datetime.now(IST)
            
            # NEW: COLLECT DATA FOR ML TRAINING
            with st.spinner("Collecting ML training data..."):
                if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
                    technical_data = st.session_state['last_result']
                    volume_data = {
                        'block_imbalance': len(st.session_state.vob_blocks['bullish']) - len(st.session_state.vob_blocks['bearish']),
                        'volume_spike_ratio': 1.0,  # Placeholder
                        'bullish_blocks': len(st.session_state.vob_blocks['bullish']),
                        'bearish_blocks': len(st.session_state.vob_blocks['bearish']),
                        'structure_broken': False,
                        'valid_blocks': True,
                        'liquidity_grab': False,
                        'confirms_price': True
                    }
                    
                    options_data = {}
                    if st.session_state.market_bias_data:
                        options_data = st.session_state.market_bias_data[0].get('options_data', {})
                    
                    ml_features = prepare_ml_features(technical_data, volume_data, options_data)
                    st.session_state.ml_collector.add_data_point(ml_features)
            
            # NEW: CHECK MASTER TRIGGER FOR AI ANALYSIS
            unified_bias = st.session_state.overall_nifty_score  # Your existing unified bias
            
            should_trigger_ai, ml_confidence = st.session_state.master_trigger.should_trigger_ai_analysis(
                unified_bias, ml_features
            )
            
            st.session_state.ml_confidence = ml_confidence
            
            if should_trigger_ai:
                st.success("🔥 MASTER TRIGGER ACTIVATED - Running AI Analysis!")
                ai_results = st.session_state.master_trigger.trigger_ai_analysis_pipeline(ml_features)
                st.session_state.ai_results = ai_results
            
            # Your existing calculations
            calculate_overall_nifty_bias()
            
            st.session_state.analysis_complete = True
            return True
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return False

    # Function to calculate overall Nifty bias from all tabs
    def calculate_overall_nifty_bias():
        """Calculate overall Nifty bias by combining all analysis methods"""
        # Simplified calculation for demo
        if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
            tech_score = st.session_state['last_result'].get('overall_score', 0)
            # Normalize to 0-100 scale
            st.session_state.overall_nifty_score = (tech_score + 100) / 2
            st.session_state.overall_nifty_bias = st.session_state['last_result'].get('overall_bias', 'NEUTRAL')
        else:
            st.session_state.overall_nifty_score = 50
            st.session_state.overall_nifty_bias = "NEUTRAL"

    # Refresh button and auto-refresh logic
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
            if run_complete_analysis():
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_count += 1
                st.sidebar.success("Analysis refreshed!")
                st.rerun()
    with col2:
        st.sidebar.metric("Auto-Refresh", "ON" if auto_refresh else "OFF")
        st.sidebar.metric("Refresh Count", st.session_state.refresh_count)

    # Display overall Nifty bias prominently
    st.sidebar.markdown("---")
    st.sidebar.header("Overall Nifty Bias")
    if st.session_state.overall_nifty_bias:
        bias_color = "🟢" if st.session_state.overall_nifty_bias == "BULLISH" else "🔴" if st.session_state.overall_nifty_bias == "BEARISH" else "🟡"
        st.sidebar.metric(
            "NIFTY 50 Bias",
            f"{bias_color} {st.session_state.overall_nifty_bias}",
            f"Score: {st.session_state.overall_nifty_score:.1f}"
        )

    # Display ML Confidence
    st.sidebar.header("🤖 ML System")
    st.sidebar.metric("ML Confidence", f"{st.session_state.ml_confidence:.1f}%")
    st.sidebar.metric("Data Points", len(st.session_state.ml_collector.historical_data))
    
    if st.session_state.ai_results:
        st.sidebar.success("AI Analysis Active!")

    # Enhanced tabs with ML
    tabs = st.tabs([
        "Overall Bias", "Bias Summary", "Price Action", "Option Chain", "Bias Tabulation", "🤖 ML & AI Analysis"
    ])

    # OVERALL BIAS TAB
    with tabs[0]:
        st.header("🎯 Overall Nifty Bias Analysis")
        
        if not st.session_state.overall_nifty_bias:
            st.info("No analysis run yet. Click 'Refresh Analysis' to start...")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Display overall bias with color coding
                if st.session_state.overall_nifty_bias == "BULLISH":
                    st.success(f"## 🟢 OVERALL NIFTY BIAS: BULLISH")
                    st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bullish")
                elif st.session_state.overall_nifty_bias == "BEARISH":
                    st.error(f"## 🔴 OVERALL NIFTY BIAS: BEARISH")
                    st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}", delta="Bearish", delta_color="inverse")
                else:
                    st.warning(f"## 🟡 OVERALL NIFTY BIAS: NEUTRAL")
                    st.metric("Bias Score", f"{st.session_state.overall_nifty_score:.1f}")
            
            st.markdown("---")
            
            # ML Confidence Display
            st.subheader("🤖 ML Confidence Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ML Confidence", f"{st.session_state.ml_confidence:.1f}%")
            
            with col2:
                trigger_status = "ACTIVE" if st.session_state.ai_results else "INACTIVE"
                st.metric("AI Trigger", trigger_status)
            
            with col3:
                st.metric("Data Collected", len(st.session_state.ml_collector.historical_data))
            
            with col4:
                st.metric("ML System", "READY" if st.session_state.master_trigger.ml_engine.models_loaded else "TRAINING")
            
            if st.session_state.ai_results:
                st.success("### 🚀 AI Analysis Results")
                st.json(st.session_state.ai_results)

    # BIAS SUMMARY TAB
    with tabs[1]:
        st.subheader("Technical Bias Summary")
        if st.session_state['last_result'] is None:
            st.info("No analysis run yet. Click 'Refresh Analysis' to start...")
        else:
            res = st.session_state['last_result']
            if not res.get('success', False):
                st.error(f"Analysis failed: {res.get('error')}")
            else:
                st.markdown(f"**Symbol:** `{res['symbol']}`")
                st.markdown(f"**Timestamp (IST):** {res['timestamp']}")
                st.metric("Current Price", f"{res['current_price']:.2f}")
                st.metric("Technical Bias", res['overall_bias'], delta=f"Confidence: {res['overall_confidence']:.1f}%")

                # Show bias results table
                bias_table = pd.DataFrame(res['bias_results'])
                st.subheader("Indicator-level Biases")
                st.dataframe(bias_table, use_container_width=True)

    # PRICE ACTION TAB
    with tabs[2]:
        st.header("📈 Price Action Analysis")
        
        if st.session_state['last_df'] is None:
            st.info("No data loaded yet. Click 'Refresh Analysis' to start...")
        else:
            df = st.session_state['last_df']
            
            # Create price action chart with volume order blocks
            st.subheader("Price Chart with Volume Order Blocks")
            
            bullish_blocks = st.session_state.vob_blocks.get('bullish', [])
            bearish_blocks = st.session_state.vob_blocks.get('bearish', [])
            
            # Create the chart using the plotting function
            fig = plot_vob(df, bullish_blocks, bearish_blocks)
            st.plotly_chart(fig, use_container_width=True, key="main_price_chart")
            
            # Volume Order Blocks Summary
            st.subheader("📦 Volume Order Blocks Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Bullish Blocks", len(bullish_blocks))
                if bullish_blocks:
                    latest_bullish = bullish_blocks[-1]
                    st.write(f"Latest Bullish Block:")
                    st.write(f"- Upper: ₹{latest_bullish['upper']:.2f}")
                    st.write(f"- Lower: ₹{latest_bullish['lower']:.2f}")
                    st.write(f"- Volume: {latest_bullish['volume']:,.0f}")
            
            with col2:
                st.metric("Bearish Blocks", len(bearish_blocks))
                if bearish_blocks:
                    latest_bearish = bearish_blocks[-1]
                    st.write(f"Latest Bearish Block:")
                    st.write(f"- Upper: ₹{latest_bearish['upper']:.2f}")
                    st.write(f"- Lower: ₹{latest_bearish['lower']:.2f}")
                    st.write(f"- Volume: {latest_bearish['volume']:,.0f}")

    # OPTION CHAIN TAB
    with tabs[3]:
        st.header("📊 Options Chain Analysis")
        
        if st.session_state.last_bias_update:
            st.write(f"Last update: {st.session_state.last_bias_update.strftime('%H:%M:%S')} IST")
        
        if st.session_state.market_bias_data:
            st.subheader("Options Data Overview")
            for instrument_data in st.session_state.market_bias_data:
                with st.expander(f"Options Data for {instrument_data['instrument']}", expanded=True):
                    if 'options_data' in instrument_data:
                        options_data = instrument_data['options_data']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("OI Momentum", f"{options_data.get('oi_momentum', 0):.2f}")
                        with col2:
                            st.metric("IV Skew", f"{options_data.get('iv_skew', 0):.3f}")
                        with col3:
                            st.metric("PCR Momentum", f"{options_data.get('pcr_momentum', 1):.2f}")
                        with col4:
                            st.metric("ATM Bias", f"{options_data.get('atm_bias_score', 0):.1f}")
        else:
            st.info("No options data available. Click 'Refresh Analysis' to generate...")

    # BIAS TABULATION TAB
    with tabs[4]:
        st.header("📋 Bias Tabulation")
        
        if not st.session_state.market_bias_data:
            st.info("No analysis data available. Click 'Refresh Analysis' to start...")
        else:
            st.subheader("Market Analysis Summary")
            
            summary_data = []
            if st.session_state['last_result'] and st.session_state['last_result'].get('success'):
                tech_data = st.session_state['last_result']
                summary_data.append({
                    'Analysis Type': 'Technical',
                    'Bias': tech_data.get('overall_bias', 'NEUTRAL'),
                    'Score': tech_data.get('overall_score', 0),
                    'Confidence': tech_data.get('overall_confidence', 0)
                })
            
            # Add ML data
            summary_data.append({
                'Analysis Type': 'ML System',
                'Bias': 'CONFIDENCE',
                'Score': st.session_state.ml_confidence,
                'Confidence': st.session_state.ml_confidence
            })
            
            # Add Volume data
            summary_data.append({
                'Analysis Type': 'Volume Blocks',
                'Bias': 'BULLISH' if len(st.session_state.vob_blocks['bullish']) > len(st.session_state.vob_blocks['bearish']) else 'BEARISH',
                'Score': len(st.session_state.vob_blocks['bullish']) - len(st.session_state.vob_blocks['bearish']),
                'Confidence': abs(len(st.session_state.vob_blocks['bullish']) - len(st.session_state.vob_blocks['bearish'])) * 10
            })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    # ML & AI ANALYSIS TAB
    with tabs[5]:
        st.header("🤖 ML & AI Analysis Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("ML System Status")
            status_color = "🟢" if st.session_state.master_trigger.ml_engine.models_loaded else "🟡"
            st.metric("ML Models", "LOADED" if st.session_state.master_trigger.ml_engine.models_loaded else "TRAINING REQUIRED")
            
            st.metric("Data Points Collected", len(st.session_state.ml_collector.historical_data))
            st.metric("Current ML Confidence", f"{st.session_state.ml_confidence:.1f}%")
            
            if st.session_state.ml_confidence >= 85:
                st.success("🎯 High ML Confidence - Markets predictable")
            elif st.session_state.ml_confidence >= 70:
                st.info("📊 Good ML Confidence - Decent predictability")
            else:
                st.warning("📉 Low ML Confidence - Markets noisy")
        
        with col2:
            st.subheader("Actions")
            if st.button("🔄 Train ML Models"):
                with st.spinner("Training ML models..."):
                    trainer = MLModelTrainer()
                    trainer.train_models()
                    trainer.save_models()
                    st.success("ML models trained and saved!")
            
            if st.button("📊 Show Feature Importance"):
                if st.session_state.master_trigger.ml_engine.models_loaded:
                    importance = st.session_state.master_trigger.ml_engine.trainer.feature_importance
                    st.write("Feature Importance (Breakout Model):")
                    for feature, imp in list(importance.get('breakout', {}).items())[:10]:
                        st.write(f"- {feature}: {imp:.3f}")
                else:
                    st.warning("ML models not loaded")
        
        # AI Results Display
        if st.session_state.ai_results:
            st.subheader("🚀 AI Analysis Results")
            
            ai_data = st.session_state.ai_results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("AI Action", ai_data['final_prediction']['action'])
                st.metric("Timeframe", ai_data['final_prediction']['timeframe'])
                st.metric("Confidence", f"{ai_data['final_prediction']['confidence']:.1f}%")
            
            with col2:
                st.metric("Risk/Reward", ai_data['final_prediction']['risk_reward'])
                st.metric("Position Size", ai_data['final_prediction']['position_size'])
                st.metric("Trigger Time", ai_data['trigger_timestamp'][11:19])
            
            st.subheader("Detailed Analysis")
            with st.expander("News Analysis"):
                st.json(ai_data['news_analysis'])
            
            with st.expander("AI Interpretation"):
                st.json(ai_data['ai_interpretation'])
            
            with st.expander("Trading Plan"):
                st.json(ai_data['final_prediction'])
        
        # ML Training Section
        st.subheader("🔧 ML Training & Management")
        
        if len(st.session_state.ml_collector.historical_data) > 100:
            st.success(f"✅ Sufficient data for training: {len(st.session_state.ml_collector.historical_data)} points")
            
            if st.button("🎯 Label Data & Train Models"):
                with st.spinner("Labeling data and training models..."):
                    st.session_state.ml_collector.label_data_with_outcomes()
                    trainer = MLModelTrainer()
                    trainer.train_models()
                    trainer.save_models()
                    st.session_state.master_trigger.ml_engine.load_models()
                    st.success("ML system updated and ready!")
                    st.rerun()
        else:
            st.warning(f"📊 Collect more data for training: {len(st.session_state.ml_collector.historical_data)}/100 points")
        
        # Data Collection Info
        st.subheader("📈 Data Collection Progress")
        st.progress(min(1.0, len(st.session_state.ml_collector.historical_data) / 100))
        st.write(f"Collected: {len(st.session_state.ml_collector.historical_data)} data points")
        st.write(f"Target: 100 points for initial training")

    # Footer
    st.markdown("---")
    st.caption("BiasAnalysisPro — Complete Enhanced Dashboard with ML & AI Integration")
    st.caption("🤖 ML System: Predicts market behavior after bias signals | 🚀 AI: Triggers on high-confidence alignments")

    # AUTO-RUN ANALYSIS ON STARTUP
    if not st.session_state.analysis_complete:
        with st.spinner("🚀 Starting initial analysis..."):
            if run_complete_analysis():
                st.success("✅ Initial analysis complete!")
                st.rerun()

if __name__ == "__main__":
    main()
