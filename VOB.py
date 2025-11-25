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
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px

warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Nifty Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# =============================================
# ENHANCED MARKET DATA FETCHER WITH CHARTS
# =============================================

class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources with chart generation
    """

    def __init__(self):
        """Initialize enhanced market data fetcher"""
        self.ist = pytz.timezone('Asia/Kolkata')
        self.dhan_fetcher = None

    def get_current_time_ist(self):
        """Get current time in IST"""
        return datetime.now(self.ist)

    def fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch India VIX from Yahoo Finance"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]

                # VIX Interpretation
                if vix_value > 25:
                    vix_sentiment = "HIGH FEAR"
                    vix_bias = "BEARISH"
                    vix_score = -75
                elif vix_value > 20:
                    vix_sentiment = "ELEVATED FEAR"
                    vix_bias = "BEARISH"
                    vix_score = -50
                elif vix_value > 15:
                    vix_sentiment = "MODERATE"
                    vix_bias = "NEUTRAL"
                    vix_score = 0
                elif vix_value > 12:
                    vix_sentiment = "LOW VOLATILITY"
                    vix_bias = "BULLISH"
                    vix_score = 40
                else:
                    vix_sentiment = "COMPLACENCY"
                    vix_bias = "NEUTRAL"
                    vix_score = 0

                return {
                    'success': True,
                    'source': 'Yahoo Finance',
                    'value': vix_value,
                    'sentiment': vix_sentiment,
                    'bias': vix_bias,
                    'score': vix_score,
                    'timestamp': self.get_current_time_ist()
                }
        except Exception as e:
            print(f"Error fetching VIX: {e}")

        return {'success': False, 'error': 'India VIX data not available'}

    def create_vix_trend_chart(self) -> Optional[go.Figure]:
        """Create VIX trend chart"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="5d", interval="1h")

            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines+markers',
                    name='India VIX',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4)
                ))

                # Add horizontal lines for VIX levels
                fig.add_hline(y=25, line_dash="dash", line_color="red", 
                            annotation_text="High Fear (25)")
                fig.add_hline(y=15, line_dash="dash", line_color="yellow", 
                            annotation_text="Moderate (15)")
                fig.add_hline(y=12, line_dash="dash", line_color="green", 
                            annotation_text="Low Vol (12)")

                fig.update_layout(
                    title="India VIX Trend (5 Days)",
                    xaxis_title="Time",
                    yaxis_title="VIX Value",
                    height=400,
                    template="plotly_dark",
                    hovermode='x unified'
                )
                return fig
        except Exception as e:
            print(f"Error creating VIX chart: {e}")
        return None

    def create_sector_performance_chart(self, sectors: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create sector performance bar chart"""
        if not sectors:
            return None

        df = pd.DataFrame(sectors)
        df = df.sort_values('change_pct', ascending=True)

        colors = ['#26ba9f' if x > 0 else '#ba2626' for x in df['change_pct']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df['sector'],
            x=df['change_pct'],
            orientation='h',
            marker=dict(color=colors),
            text=df['change_pct'].apply(lambda x: f"{x:.2f}%"),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Sector Performance Today (%)",
            xaxis_title="Change %",
            yaxis_title="Sector",
            height=500,
            template="plotly_dark",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        )
        return fig

    def create_global_markets_heatmap(self, globalmarkets: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create global markets heatmap"""
        if not globalmarkets:
            return None

        df = pd.DataFrame(globalmarkets)

        fig = go.Figure(data=go.Heatmap(
            z=[df['change_pct'].values],
            x=df['market'],
            y=['Change %'],
            colorscale='RdYlGn',
            zmid=0,
            text=df['change_pct'].apply(lambda x: f"{x:.2f}%").values.reshape(1, -1),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='<b>%{x}</b><br>Change: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title="Global Markets Performance",
            height=200,
            template="plotly_dark"
        )
        return fig

    def create_intermarket_chart(self, intermarket: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create intermarket analysis chart"""
        if not intermarket:
            return None

        df = pd.DataFrame(intermarket)

        fig = go.Figure()
        colors = ['#26ba9f' if x > 0 else '#ba2626' for x in df['change_pct']]

        fig.add_trace(go.Bar(
            x=df['asset'],
            y=df['change_pct'],
            marker=dict(color=colors),
            text=df['change_pct'].apply(lambda x: f"{x:+.2f}%"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Intermarket Performance",
            xaxis_title="Asset",
            yaxis_title="Change %",
            height=350,
            template="plotly_dark",
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        )
        return fig

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """Fetch sector indices from Yahoo Finance"""
        sectors_map = {
            '^CNXIT': 'NIFTY IT',
            '^CNXAUTO': 'NIFTY AUTO',
            '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL',
            '^CNXREALTY': 'NIFTY REALTY',
            '^CNXFMCG': 'NIFTY FMCG',
            '^CNXBANK': 'NIFTY BANK'
        }

        sector_data = []

        for symbol, name in sectors_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[0]
                    high_price = hist['High'].max()
                    low_price = hist['Low'].min()

                    change_pct = ((last_price - open_price) / open_price) * 100

                    # Determine bias
                    if change_pct > 1.5:
                        bias = "STRONG BULLISH"
                        score = 75
                    elif change_pct > 0.5:
                        bias = "BULLISH"
                        score = 50
                    elif change_pct < -1.5:
                        bias = "STRONG BEARISH"
                        score = -75
                    elif change_pct < -0.5:
                        bias = "BEARISH"
                        score = -50
                    else:
                        bias = "NEUTRAL"
                        score = 0

                    sector_data.append({
                        'sector': name,
                        'last_price': last_price,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return sector_data

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """Fetch global market indices"""
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG'
        }

        market_data = []

        for symbol, name in global_markets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_close - prev_close) / prev_close) * 100

                    bias = "BULLISH" if change_pct > 0.5 else "BEARISH" if change_pct < -0.5 else "NEUTRAL"
                    score = 50 if change_pct > 0.5 else -50 if change_pct < -0.5 else 0

                    market_data.append({
                        'market': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """Fetch intermarket data"""
        intermarket_assets = {
            'GC=F': 'GOLD',
            'CL=F': 'CRUDE OIL',
            'INR=X': 'USD/INR'
        }

        intermarket_data = []

        for symbol, name in intermarket_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_close - prev_close) / prev_close) * 100

                    bias = "BULLISH" if change_pct > 0 else "BEARISH" if change_pct < 0 else "NEUTRAL"
                    score = 40 if change_pct > 0 else -40 if change_pct < 0 else 0

                    intermarket_data.append({
                        'asset': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return intermarket_data

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """Fetch all enhanced market data with improved error handling"""
        print("Fetching enhanced market data...")

        result = {
            "timestamp": self.get_current_time_ist(),
            "india_vix": {},
            "sector_indices": [],
            "global_markets": [],
            "intermarket": [],
            "summary": {},
            "errors": []
        }

        # 1. Fetch India VIX with error tracking
        print("- Fetching India VIX...")
        try:
            result['india_vix'] = self.fetch_india_vix()
        except Exception as e:
            result['errors'].append(f"India VIX: {str(e)}")
            result['india_vix'] = {'success': False, 'error': str(e)}

        # 2. Fetch Sector Indices
        print("- Fetching sector indices...")
        try:
            result['sector_indices'] = self.fetch_sector_indices()
        except Exception as e:
            result['errors'].append(f"Sectors: {str(e)}")

        # 3. Fetch Global Markets
        print("- Fetching global markets...")
        try:
            result['global_markets'] = self.fetch_global_markets()
        except Exception as e:
            result['errors'].append(f"Global Markets: {str(e)}")

        # 4. Fetch Intermarket Data
        print("- Fetching intermarket data...")
        try:
            result['intermarket'] = self.fetch_intermarket_data()
        except Exception as e:
            result['errors'].append(f"Intermarket: {str(e)}")

        # 5. Calculate summary
        result['summary'] = self._calculate_summary(result)

        # Display errors if any
        if result['errors']:
            print(f"⚠️ Encountered {len(result['errors'])} errors during data fetch")
            for error in result['errors']:
                print(f"  - {error}")
        else:
            print("✅ Enhanced market data fetch completed successfully!")

        return result

    def _calculate_summary(self, data: Dict) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {
            'total_data_points': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'overall_sentiment': 'NEUTRAL'
        }

        all_scores = []

        # Count India VIX
        if data['india_vix'].get('success'):
            summary['total_data_points'] += 1
            all_scores.append(data['india_vix']['score'])
            bias = data['india_vix']['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count sectors
        for sector in data['sector_indices']:
            summary['total_data_points'] += 1
            all_scores.append(sector['score'])
            bias = sector['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count global markets
        for market in data['global_markets']:
            summary['total_data_points'] += 1
            all_scores.append(market['score'])
            bias = market['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Calculate average score
        if all_scores:
            summary['avg_score'] = np.mean(all_scores)

            # Determine overall sentiment
            if summary['avg_score'] > 25:
                summary['overall_sentiment'] = 'BULLISH'
            elif summary['avg_score'] < -25:
                summary['overall_sentiment'] = 'BEARISH'
            else:
                summary['overall_sentiment'] = 'NEUTRAL'

        return summary


# =============================================
# ENHANCED NIFTY APP WITH MARKET DATA DISPLAY
# =============================================

class EnhancedNiftyApp:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.market_data_fetcher = EnhancedMarketData()
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state"""
        if 'enhanced_market_data' not in st.session_state:
            st.session_state.enhanced_market_data = None
        if 'last_market_data_update' not in st.session_state:
            st.session_state.last_market_data_update = None

    def display_enhanced_market_data(self):
        """Display comprehensive enhanced market data with charts"""
        st.header("📊 Enhanced Market Data Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.info("Comprehensive market analysis from multiple sources including India VIX, global markets, and sector rotation")

        with col2:
            if st.button("🔄 Update Market Data", type="primary"):
                with st.spinner("Fetching comprehensive market data..."):
                    try:
                        market_data = self.market_data_fetcher.fetch_all_enhanced_data()
                        st.session_state.enhanced_market_data = market_data
                        st.session_state.last_market_data_update = datetime.now(self.ist)

                        if market_data.get('errors'):
                            st.warning(f"⚠️ Data fetched with {len(market_data['errors'])} warnings")
                            with st.expander("View Errors"):
                                for error in market_data['errors']:
                                    st.text(f"• {error}")
                        else:
                            st.success("✅ Market data updated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error fetching market data: {str(e)}")
                        st.info("Try refreshing in a few seconds or check your internet connection")

        st.divider()

        if st.session_state.last_market_data_update:
            st.write(f"Last update: {st.session_state.last_market_data_update.strftime('%H:%M:%S')} IST")

        if st.session_state.enhanced_market_data:
            market_data = st.session_state.enhanced_market_data

            # Overall Summary
            st.subheader("📊 Market Summary")
            summary = market_data['summary']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Sentiment", summary['overall_sentiment'])
            with col2:
                st.metric("Average Score", f"{summary['avg_score']:.1f}")
            with col3:
                st.metric("Bullish Signals", summary['bullish_count'])
            with col4:
                st.metric("Total Data Points", summary['total_data_points'])

            st.divider()

            # Create tabs for different market data categories
            tab1, tab2, tab3, tab4 = st.tabs([
                "🇮🇳 India VIX", "📈 Sector Analysis", "🌍 Global Markets", "🔄 Intermarket"
            ])

            with tab1:
                self.display_india_vix_data(market_data['india_vix'])

            with tab2:
                self.display_sector_data(market_data['sector_indices'])

            with tab3:
                self.display_global_markets(market_data['global_markets'])

            with tab4:
                self.display_intermarket_data(market_data['intermarket'])

        else:
            st.info("👆 Click 'Update Market Data' to load comprehensive market analysis")

    def display_india_vix_data(self, vix_data: Dict[str, Any]):
        """Display India VIX data with chart"""
        if not vix_data.get('success'):
            st.error("India VIX data not available")
            return

        st.subheader("📊 India VIX - Fear Index")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VIX Value", f"{vix_data['value']:.2f}")
        with col2:
            st.metric("Sentiment", vix_data['sentiment'])
        with col3:
            st.metric("Bias", vix_data['bias'])
        with col4:
            st.metric("Score", vix_data['score'])

        # Add VIX trend chart
        chart = self.market_data_fetcher.create_vix_trend_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("VIX trend chart could not be generated")

        # Interpretation guide
        with st.expander("📖 VIX Interpretation Guide"):
            st.markdown("""
            **VIX Levels:**
            - **Below 12**: Complacency - Very low volatility
            - **12-15**: Low Volatility - Bullish environment
            - **15-20**: Moderate Volatility - Neutral zone
            - **20-25**: Elevated Fear - Caution advised
            - **Above 25**: High Fear - Market stress
            """)

    def display_sector_data(self, sectors: List[Dict[str, Any]]):
        """Display sector indices data with performance chart"""
        st.subheader("📈 Nifty Sector Performance")

        if not sectors:
            st.info("No sector data available")
            return

        # Add sector performance chart
        chart = self.market_data_fetcher.create_sector_performance_chart(sectors)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        st.markdown("---")

        # Display as metrics grid
        sector_df = pd.DataFrame(sectors)
        sector_df = sector_df.sort_values('change_pct', ascending=False)

        st.markdown("#### Sector Metrics")
        cols = st.columns(4)
        for idx, sector in enumerate(sector_df.head(8).itertuples()):
            with cols[idx % 4]:
                color = "🟢" if sector.change_pct > 0 else "🔴"
                st.metric(
                    f"{color} {sector.sector}",
                    f"₹{sector.last_price:.0f}",
                    f"{sector.change_pct:+.2f}%"
                )

        # Detailed table
        with st.expander("📊 Detailed Sector Data"):
            display_df = sector_df[['sector', 'last_price', 'change_pct', 'bias', 'score']].copy()
            display_df.columns = ['Sector', 'Price', 'Change %', 'Bias', 'Score']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    def display_global_markets(self, globalmarkets: List[Dict[str, Any]]):
        """Display global markets with heatmap"""
        st.subheader("🌍 Global Market Performance")

        if not globalmarkets:
            st.info("No global market data available")
            return

        # Add heatmap
        chart = self.market_data_fetcher.create_global_markets_heatmap(globalmarkets)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        st.markdown("---")

        # Display as metrics
        cols = st.columns(len(globalmarkets))
        for idx, market in enumerate(globalmarkets):
            with cols[idx]:
                color = "🟢" if market['change_pct'] > 0 else "🔴"
                st.metric(
                    f"{color} {market['market']}",
                    f"{market['last_price']:.2f}",
                    f"{market['change_pct']:+.2f}%"
                )

        # Detailed table
        with st.expander("📊 Detailed Global Markets Data"):
            df = pd.DataFrame(globalmarkets)
            display_df = df[['market', 'last_price', 'prev_close', 'change_pct', 'bias']].copy()
            display_df.columns = ['Market', 'Current Price', 'Previous Close', 'Change %', 'Bias']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    def display_intermarket_data(self, intermarket: List[Dict[str, Any]]):
        """Display intermarket data with chart"""
        st.subheader("🔄 Intermarket Analysis")

        if not intermarket:
            st.info("No intermarket data available")
            return

        # Add intermarket chart
        chart = self.market_data_fetcher.create_intermarket_chart(intermarket)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        st.markdown("---")

        # Display as metrics
        cols = st.columns(3)
        for idx, asset in enumerate(intermarket):
            with cols[idx % 3]:
                color = "🟢" if asset['change_pct'] > 0 else "🔴"
                st.metric(
                    f"{color} {asset['asset']}",
                    f"₹{asset['last_price']:.2f}",
                    f"{asset['change_pct']:+.2f}%"
                )

        # Intermarket correlation insights
        with st.expander("📖 Intermarket Correlation Guide"):
            st.markdown("""
            **Key Relationships:**
            - **Gold ↑ + INR ↓**: Risk-off sentiment (bearish for equities)
            - **Crude Oil ↑**: Inflationary pressure (bearish for India)
            - **USD/INR ↑**: Rupee weakness (impacts FII flows)
            """)

        # Detailed table
        with st.expander("📊 Detailed Intermarket Data"):
            df = pd.DataFrame(intermarket)
            display_df = df[['asset', 'last_price', 'prev_close', 'change_pct', 'bias']].copy()
            display_df.columns = ['Asset', 'Current Price', 'Previous Close', 'Change %', 'Bias']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    def run(self):
        """Main application"""
        st.title("📈 Advanced Nifty Trading Dashboard")
        st.markdown("*Enhanced Market Data Analysis with Charts*")

        # Main content
        self.display_enhanced_market_data()


# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    app = EnhancedNiftyApp()
    app.run()
