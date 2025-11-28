# ml_confidence_engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any

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
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
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
        
        feature_df = pd.DataFrame([feature_dict])
        
        predictions = {
            'breakout_confidence': self.trainer.models['breakout'].predict_proba(feature_df)[0][1] * 100,
            'direction_15min_confidence': self.trainer.models['direction_15min'].predict_proba(feature_df)[0][1] * 100,
            'direction_1hr_confidence': self.trainer.models['direction_1hr'].predict_proba(feature_df)[0][1] * 100,
            'target_hit_confidence': self.trainer.models['target_hit'].predict_proba(feature_df)[0][1] * 100,
        }
        
        return predictions
