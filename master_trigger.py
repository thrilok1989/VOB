class MasterTrigger:
    """The final decision engine that combines everything"""
    
    def __init__(self):
        self.ml_engine = MLConfidenceEngine()
        self.data_collector = MLDataCollector()
        
    def initialize_system(self):
        """Initialize the complete ML system"""
        print("🚀 Initializing Master Trigger System...")
        self.ml_engine.load_models()
        print("✅ Master Trigger System Ready!")
    
    def should_trigger_ai_analysis(self, technical_data, options_data, volume_data, market_intel, unified_bias_score):
        """MASTER DECISION: Should we trigger AI analysis?"""
        
        # 1. Collect current features for ML
        current_features = self.data_collector.collect_all_features(
            technical_data, options_data, volume_data, market_intel
        )
        
        # 2. Get ML confidence
        ml_confidence = self.ml_engine.calculate_ml_confidence(current_features)
        
        print(f"\n🎯 MASTER TRIGGER ANALYSIS:")
        print(f"   Unified Bias Score: {unified_bias_score:.1f}%")
        print(f"   ML Confidence: {ml_confidence:.1f}%")
        print(f"   Threshold: Bias >= 80% & ML >= 85%")
        
        # 3. MASTER TRIGGER CONDITION
        if unified_bias_score >= 80 and ml_confidence >= 85:
            print("🔥 🔥 🔥 MASTER TRIGGER ACTIVATED! 🔥 🔥 🔥")
            return True, ml_confidence, current_features
        else:
            print("❌ Conditions not met for AI trigger")
            return False, ml_confidence, current_features
    
    def trigger_ai_analysis_pipeline(self, technical_data, options_data, volume_data, market_intel, current_features):
        """Trigger complete AI analysis when conditions are met"""
        
        print("\n🤖 INITIATING AI ANALYSIS PIPELINE...")
        
        # 1. Trigger News API
        news_analysis = self.trigger_news_api(current_features)
        
        # 2. Trigger AI Interpretation
        ai_interpretation = self.trigger_ai_interpretation(
            technical_data, options_data, volume_data, market_intel, news_analysis
        )
        
        # 3. Generate Final Prediction
        final_prediction = self.generate_final_prediction(
            technical_data, options_data, volume_data, market_intel,
            news_analysis, ai_interpretation, current_features
        )
        
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
        # Integrate with your news API
        print("📰 Fetching news sentiment analysis...")
        return {
            'overall_sentiment': 'BULLISH',
            'key_news_items': ['Market shows strong bullish momentum'],
            'sentiment_score': 85,
            'impact_level': 'HIGH'
        }
    
    def trigger_ai_interpretation(self, technical_data, options_data, volume_data, market_intel, news_analysis):
        """Trigger AI interpretation of all data"""
        # Integrate with your HuggingFace AI
        print("🧠 Running AI market interpretation...")
        return {
            'market_outlook': 'STRONG_BULLISH_MOMENTUM',
            'confidence': 88,
            'key_factors': ['Options alignment', 'Volume confirmation', 'Institutional bias'],
            'timeframe': '1-4 hours',
            'risk_level': 'MEDIUM'
        }
    
    def generate_final_prediction(self, technical_data, options_data, volume_data, market_intel, news_analysis, ai_interpretation, features):
        """Generate final trading prediction"""
        
        print("🎯 Generating final trading prediction...")
        
        return {
            'action': 'STRONG_BUY',
            'entry_price': technical_data.get('current_price', 0),
            'targets': self.calculate_targets(technical_data),
            'stoploss': self.calculate_stoploss(technical_data),
            'timeframe': ai_interpretation.get('timeframe', '1-4 hours'),
            'confidence': min(85, ai_interpretation.get('confidence', 0)),  # Cap at ML confidence
            'rationale': 'High confidence alignment across all systems',
            'risk_reward': '1:2.5',
            'position_size': 'Medium'
        }
    
    def calculate_targets(self, technical_data):
        """Calculate target levels"""
        current_price = technical_data.get('current_price', 0)
        resistances = technical_data.get('resistance_levels', [])
        
        if resistances:
            next_resistance = min([r for r in resistances if r > current_price], default=current_price * 1.02)
            return [next_resistance, next_resistance * 1.005]
        
        return [current_price * 1.015, current_price * 1.025]
    
    def calculate_stoploss(self, technical_data):
        """Calculate stoploss level"""
        current_price = technical_data.get('current_price', 0)
        supports = technical_data.get('support_levels', [])
        
        if supports:
            nearest_support = max([s for s in supports if s < current_price], default=current_price * 0.99)
            return nearest_support
        
        return current_price * 0.985
