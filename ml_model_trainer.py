import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb
import joblib
from typing import Tuple, Dict, Any

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
        
        X = labeled_data[feature_columns]
        
        # Prepare labels
        y_breakout = labeled_data['labels'].apply(lambda x: x.get('breakout_real', 0))
        y_direction_15min = labeled_data['labels'].apply(lambda x: x.get('direction_15min', 0))
        y_direction_1hr = labeled_data['labels'].apply(lambda x: x.get('direction_1hr', 0))
        y_target_1hr = labeled_data['labels'].apply(lambda x: x.get('target_hit_1hr', 0))
        
        return X, y_breakout, y_direction_15min, y_direction_1hr, y_target_1hr, feature_columns
    
    def train_models(self, data_file="ml_training_data.json"):
        """Train all ML models"""
        
        print("🚀 Training ML Models...")
        
        X, y_breakout, y_dir_15, y_dir_1h, y_target, feature_names = self.load_and_prepare_data(data_file)
        
        # Split data
        X_train, X_test, y_breakout_train, y_breakout_test = train_test_split(X, y_breakout, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
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
        X_train_dir, X_test_dir, y_dir_15_train, y_dir_15_test = train_test_split(X, y_dir_15, test_size=0.2, random_state=42)
        self.models['direction_15min'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.models['direction_15min'].fit(X_train_dir, y_dir_15_train)
        
        # Model 3: Direction 1hr (LightGBM)
        print("🎯 Training 1hr Direction Model...")
        X_train_dir1h, X_test_dir1h, y_dir_1h_train, y_dir_1h_test = train_test_split(X, y_dir_1h, test_size=0.2, random_state=42)
        self.models['direction_1hr'] = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            random_state=42
        )
        self.models['direction_1hr'].fit(X_train_dir1h, y_dir_1h_train)
        
        # Model 4: Target Hit (Gradient Boosting)
        print("🎯 Training Target Hit Model...")
        X_train_target, X_test_target, y_target_train, y_target_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
        self.models['target_hit'] = GradientBoostingClassifier(
            n_estimators=60,
            max_depth=4,
            random_state=42
        )
        self.models['target_hit'].fit(X_train_target, y_target_train)
        
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
