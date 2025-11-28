import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any
import joblib

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
        # Your existing EMA data from multiple timeframes
        ema_5m = technical_data.get('ema_5m_trend', 'neutral')
        ema_15m = technical_data.get('ema_15m_trend', 'neutral') 
        ema_1h = technical_data.get('ema_1h_trend', 'neutral')
        
        trends = [ema_5m, ema_15m, ema_1h]
        bullish_count = sum(1 for t in trends if 'bull' in str(t).lower())
        bearish_count = sum(1 for t in trends if 'bear' in str(t).lower())
        
        return (bullish_count - bearish_count) / len(trends)
    
    def calculate_sr_distance(self, technical_data):
        """Calculate distance to nearest S/R level"""
        current_price = technical_data.get('current_price', 0)
        supports = technical_data.get('support_levels', [])
        resistances = technical_data.get('resistance_levels', [])
        
        if not supports and not resistances:
            return 0
            
        # Find nearest support and resistance
        nearest_support = min([abs(current_price - s) for s in supports]) if supports else float('inf')
        nearest_resistance = min([abs(current_price - r) for r in resistances]) if resistances else float('inf')
        
        distance_to_nearest = min(nearest_support, nearest_resistance)
        
        # Normalize as percentage of price
        return (distance_to_nearest / current_price) * 100 if current_price > 0 else 0
    
    def add_data_point(self, technical_data, options_data, volume_data, market_intel):
        """Add new data point to training set"""
        features = self.collect_all_features(technical_data, options_data, volume_data, market_intel)
        self.historical_data.append(features)
        self.save_data()
        print(f"✅ Data point collected. Total: {len(self.historical_data)}")
    
    def label_data_with_outcomes(self):
        """Go back and label historical data with what actually happened"""
        print("🔍 Labeling historical data with outcomes...")
        
        for i, data_point in enumerate(self.historical_data):
            if 'labels' not in data_point:
                # You'll need to implement this based on your historical price data
                # This is where you check what happened after each data point
                data_point['labels'] = self.calculate_labels(data_point, i)
        
        self.save_data()
        print("✅ Data labeling complete!")
    
    def calculate_labels(self, data_point, index):
        """Calculate what actually happened after this data point"""
        # This is a CRITICAL function - you need historical price data
        # For now, returning dummy labels - IMPLEMENT WITH REAL DATA
        
        return {
            'breakout_real': np.random.choice([0, 1]),  # 1 if real breakout, 0 if fakeout
            'direction_15min': np.random.choice([0, 1]), # 1 if bullish next 15min
            'direction_1hr': np.random.choice([0, 1]),   # 1 if bullish next 1hr  
            'target_hit_1hr': np.random.choice([0, 1]),  # 1 if target hit within 1hr
            'price_change_1hr': 0.5  # Actual price change percentage
        }
