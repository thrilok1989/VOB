"""
Comprehensive Bias Analysis Dashboard
=====================================

Complete Streamlit app that uses BiasAnalysisPro to fetch data
and display all 13 bias indicators with beautiful tables and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from bias_analysis import BiasAnalysisPro  # Your existing class

# Configure the page
st.set_page_config(
    page_title="Bias Analysis Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bias-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .bullish {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .bearish {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .neutral {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class BiasAnalysisDashboard:
    """Complete dashboard for Bias Analysis Pro"""
    
    def __init__(self):
        self.analyzer = BiasAnalysisPro()
        self.market_symbols = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN", 
            "BANK NIFTY": "^NSEBANK",
            "DOW JONES": "^DJI",
            "NASDAQ": "^IXIC"
        }
    
    def run(self):
        """Run the complete dashboard"""
        st.markdown('<h1 class="main-header">🎯 Bias Analysis Pro Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar controls
        selected_market = self.render_sidebar()
        
        # Main content
        if st.button("🚀 Analyze Market Bias", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing market data and calculating bias indicators..."):
                self.analyze_and_display(selected_market)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Analysis Settings")
            
            # Market selection
            selected_market = st.selectbox(
                "Select Market:",
                list(self.market_symbols.keys()),
                index=0
            )
            
            st.divider()
            
            # Display info about the analysis
            st.subheader("📊 About This Analysis")
            st.info("""
            **13 Bias Indicators Analyzed:**
            
            **Fast (8):**
            • Volume Delta
            • HVP (High Volume Pivots)
            • VOB (Volume Order Blocks) 
            • Order Blocks (EMA 5/18)
            • RSI
            • DMI
            • VIDYA
            • MFI
            
            **Adaptive Weighting:**
            • Normal Mode: Fast(2x), Medium(3x), Slow(5x)
            • Reversal Mode: Fast(5x), Medium(3x), Slow(2x)
            """)
            
            return selected_market
    
    def analyze_and_display(self, market_name):
        """Perform analysis and display results"""
        symbol = self.market_symbols[market_name]
        
        # Perform bias analysis
        results = self.analyzer.analyze_all_bias_indicators(symbol)
        
        if not results.get('success'):
            st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        # Display overall market overview
        self.display_market_overview(results, market_name)
        
        # Display detailed bias breakdown
        self.display_bias_breakdown(results)
        
        # Display visualizations
        self.display_visualizations(results)
        
        # Display trading recommendations
        self.display_trading_recommendations(results)
    
    def display_market_overview(self, results, market_name):
        """Display overall market overview"""
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"{market_name} Current Price",
                f"₹{results['current_price']:,.2f}" if market_name != "DOW JONES" else f"${results['current_price']:,.2f}",
                delta=None
            )
        
        with col2:
            bias_emoji = "🐂" if results['overall_bias'] == "BULLISH" else "🐻" if results['overall_bias'] == "BEARISH" else "⚖️"
            bias_color = "green" if results['overall_bias'] == "BULLISH" else "red" if results['overall_bias'] == "BEARISH" else "gray"
            st.markdown(f"<h2 style='color: {bias_color}; text-align: center;'>{bias_emoji} {results['overall_bias']}</h2>", 
                       unsafe_allow_html=True)
            st.caption("Overall Market Bias")
        
        with col3:
            score_color = "green" if results['overall_score'] > 0 else "red" if results['overall_score'] < 0 else "gray"
            st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{results['overall_score']:+.1f}</h2>", 
                       unsafe_allow_html=True)
            st.caption("Bias Score")
        
        with col4:
            confidence_color = "green" if results['overall_confidence'] > 70 else "orange" if results['overall_confidence'] > 50 else "red"
            st.markdown(f"<h2 style='color: {confidence_color}; text-align: center;'>{results['overall_confidence']:.1f}%</h2>", 
                       unsafe_allow_html=True)
            st.caption("Confidence Level")
        
        # Mode and statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mode_emoji = "🔄" if results['mode'] == "REVERSAL" else "📊"
            st.info(f"{mode_emoji} **Mode:** {results['mode']}")
        
        with col2:
            st.success(f"🐂 **Bullish Signals:** {results['bullish_count']}/{results['total_indicators']}")
        
        with col3:
            st.error(f"🐻 **Bearish Signals:** {results['bearish_count']}/{results['total_indicators']}")
    
    def display_bias_breakdown(self, results):
        """Display detailed bias breakdown in tables"""
        st.markdown("---")
        st.subheader("📋 Detailed Bias Breakdown")
        
        # Convert bias results to DataFrame
        bias_df = pd.DataFrame(results['bias_results'])
        
        # Create styled DataFrame
        styled_df = bias_df[['indicator', 'value', 'bias', 'score', 'category']].copy()
        
        # Apply styling
        def color_bias(val):
            if 'BULLISH' in str(val):
                return 'color: #28a745; font-weight: bold;'
            elif 'BEARISH' in str(val):
                return 'color: #dc3545; font-weight: bold;'
            else:
                return 'color: #6c757d; font-weight: bold;'
        
        def color_score(val):
            try:
                score = float(val)
                if score > 50:
                    return 'color: #28a745; font-weight: bold;'
                elif score > 0:
                    return 'color: #20c997; font-weight: bold;'
                elif score < -50:
                    return 'color: #dc3545; font-weight: bold;'
                elif score < 0:
                    return 'color: #fd7e14; font-weight: bold;'
                else:
                    return 'color: #6c757d; font-weight: bold;'
            except:
                return ''
        
        # Apply styling
        styled_df = styled_df.style.map(color_bias, subset=['bias']) \
                                  .map(color_score, subset=['score'])
        
        # Display the table
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Display by category
        st.subheader("📊 Bias by Category")
        
        col1, col2, col3 = st.columns(3)
        
        # Fast indicators
        with col1:
            fast_indicators = [b for b in results['bias_results'] if b['category'] == 'fast']
            if fast_indicators:
                st.markdown("**⚡ Fast Indicators**")
                fast_df = pd.DataFrame(fast_indicators)[['indicator', 'bias', 'score']]
                st.dataframe(fast_df, use_container_width=True, height=300)
        
        # Medium indicators  
        with col2:
            medium_indicators = [b for b in results['bias_results'] if b['category'] == 'medium']
            if medium_indicators:
                st.markdown("**📊 Medium Indicators**")
                medium_df = pd.DataFrame(medium_indicators)[['indicator', 'bias', 'score']]
                st.dataframe(medium_df, use_container_width=True, height=300)
            else:
                st.info("No medium indicators configured")
        
        # Slow indicators
        with col3:
            slow_indicators = [b for b in results['bias_results'] if b['category'] == 'slow']
            if slow_indicators:
                st.markdown("**🐢 Slow Indicators**")
                slow_df = pd.DataFrame(slow_indicators)[['indicator', 'bias', 'score']]
                st.dataframe(slow_df, use_container_width=True, height=300)
            else:
                st.info("No slow indicators configured")
    
    def display_visualizations(self, results):
        """Display visualizations of bias data"""
        st.markdown("---")
        st.subheader("📈 Bias Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bias distribution pie chart
            labels = ['Bullish', 'Bearish', 'Neutral']
            values = [results['bullish_count'], results['bearish_count'], results['neutral_count']]
            colors = ['#28a745', '#dc3545', '#6c757d']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                marker_colors=colors,
                hole=0.3
            )])
            fig_pie.update_layout(
                title="Bias Signal Distribution",
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Score bar chart
            indicators = [b['indicator'] for b in results['bias_results']]
            scores = [b['score'] for b in results['bias_results']]
            colors = ['#28a745' if s > 0 else '#dc3545' if s < 0 else '#6c757d' for s in scores]
            
            fig_bar = go.Figure(data=[go.Bar(
                x=indicators,
                y=scores,
                marker_color=colors,
                text=[f"{s:+.0f}" for s in scores],
                textposition='auto'
            )])
            fig_bar.update_layout(
                title="Indicator Scores",
                xaxis_tickangle=-45,
                yaxis_title="Score",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Weighted contribution chart
        st.subheader("🎯 Weighted Contribution Analysis")
        
        weighted_data = []
        for bias in results['bias_results']:
            weighted_score = bias['score'] * bias['weight']
            weighted_data.append({
                'Indicator': bias['indicator'],
                'Category': bias['category'],
                'Weighted_Score': weighted_score,
                'Raw_Score': bias['score'],
                'Weight': bias['weight']
            })
        
        weighted_df = pd.DataFrame(weighted_data)
        weighted_df = weighted_df.sort_values('Weighted_Score', ascending=True)
        
        fig_weighted = px.bar(
            weighted_df,
            y='Indicator',
            x='Weighted_Score',
            color='Category',
            orientation='h',
            title="Weighted Contribution to Overall Bias",
            color_discrete_map={
                'fast': '#1f77b4',
                'medium': '#ff7f0e', 
                'slow': '#2ca02c'
            }
        )
        st.plotly_chart(fig_weighted, use_container_width=True)
    
    def display_trading_recommendations(self, results):
        """Display trading recommendations based on bias analysis"""
        st.markdown("---")
        st.subheader("💡 Trading Recommendations")
        
        overall_bias = results['overall_bias']
        overall_score = results['overall_score']
        confidence = results['overall_confidence']
        mode = results['mode']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if overall_bias == "BULLISH" and confidence > 70:
                st.success("## 🐂 STRONG BULLISH SIGNAL")
                st.info("""
                **Recommended Strategy:**
                - ✅ Look for LONG entries on dips
                - ✅ Focus on support levels and VOB support touches
                - ✅ Set stop loss below recent swing low
                - ✅ Target: Risk-Reward ratio 1:2 or higher
                - ✅ Consider buying calls or bullish spreads
                """)
            
            elif overall_bias == "BULLISH" and confidence >= 50:
                st.success("## 🐂 MODERATE BULLISH SIGNAL")
                st.info("""
                **Recommended Strategy:**
                - ⚠️ Consider LONG entries with caution
                - ⚠️ Use tighter stop losses
                - ⚠️ Take partial profits at resistance levels
                - ⚠️ Monitor for trend confirmation
                - ⚠️ Consider defined-risk bullish strategies
                """)
            
            elif overall_bias == "BEARISH" and confidence > 70:
                st.error("## 🐻 STRONG BEARISH SIGNAL")
                st.info("""
                **Recommended Strategy:**
                - ✅ Look for SHORT entries on rallies
                - ✅ Focus on resistance levels and VOB resistance touches
                - ✅ Set stop loss above recent swing high
                - ✅ Target: Risk-Reward ratio 1:2 or higher
                - ✅ Consider buying puts or bearish spreads
                """)
            
            elif overall_bias == "BEARISH" and confidence >= 50:
                st.error("## 🐻 MODERATE BEARISH SIGNAL")
                st.info("""
                **Recommended Strategy:**
                - ⚠️ Consider SHORT entries with caution
                - ⚠️ Use tighter stop losses
                - ⚠️ Take partial profits at support levels
                - ⚠️ Monitor for trend reversal signs
                - ⚠️ Consider defined-risk bearish strategies
                """)
            
            else:
                st.warning("## ⚖️ NEUTRAL / NO CLEAR SIGNAL")
                st.info("""
                **Recommended Strategy:**
                - 🔄 Stay out of the market or use range trading
                - 🔄 Wait for clearer bias formation
                - 🔄 Monitor key support/resistance levels
                - 🔄 Reduce position sizes if trading
                - 🔄 Consider non-directional strategies
                """)
        
        with col2:
            # Key metrics for quick reference
            st.metric("Overall Bias", overall_bias)
            st.metric("Bias Score", f"{overall_score:+.1f}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.metric("Analysis Mode", mode)
            
            # Divergence warning
            if mode == "REVERSAL":
                st.warning("⚠️ **REVERSAL MODE ACTIVE**")
                st.caption("Market showing divergence patterns - higher probability of trend reversal")
            
            # Timestamp
            st.caption(f"Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main application entry point"""
    try:
        dashboard = BiasAnalysisDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()