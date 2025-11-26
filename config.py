import streamlit as st

# ═══════════════════════════════════════════════════════════════════════
# DHAN API CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════

def get_dhan_credentials():
    """Load DhanHQ credentials from secrets"""
    try:
        return {
            'client_id': st.secrets["DHAN"]["CLIENT_ID"],
            'access_token': st.secrets["DHAN"]["ACCESS_TOKEN"]
        }
    except Exception as e:
        st.error(f"⚠️ DhanHQ credentials missing: {e}")
        return None

def get_telegram_credentials():
    """Load Telegram credentials from secrets"""
    try:
        return {
            'bot_token': st.secrets["TELEGRAM"]["BOT_TOKEN"],
            'chat_id': st.secrets["TELEGRAM"]["CHAT_ID"],
            'enabled': True
        }
    except:
        return {'enabled': False}

# ═══════════════════════════════════════════════════════════════════════
# TRADING SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# Lot sizes
LOT_SIZES = {
    "NIFTY": 75,
    "SENSEX": 30
}

# Strike intervals
STRIKE_INTERVALS = {
    "NIFTY": 50,
    "SENSEX": 100
}

# SENSEX correlation factor
SENSEX_NIFTY_RATIO = 3.3

# HTF Pivot settings
HTF_TIMEFRAMES = [3, 5, 10, 15]  # minutes

# Advanced HTF Pivot settings (BigBeluga Pine Script conversion)
HTF_PIVOTS_ADVANCED = {
    'enabled': True,
    'timeframes': [
        {'length': 4, 'interval_min': 15, 'label': '15m', 'color': '#00FF00', 'style': 'solid', 'enabled': True},
        {'length': 5, 'interval_min': 60, 'label': '1H', 'color': '#0000FF', 'style': 'solid', 'enabled': True},
        {'length': 5, 'interval_min': 240, 'label': '4H', 'color': '#800080', 'style': 'solid', 'enabled': True},
        {'length': 5, 'interval_min': 1440, 'label': 'D', 'color': '#FFA500', 'style': 'dashed', 'enabled': True}
    ],
    'offset': 15,  # Label offset
    'text_size': 'small',  # Text size
    'shadow_width': 5,  # Shadow line width
    'shadow_transparency': 85  # Shadow transparency (0-100)
}

# Reversal Probability settings (LuxAlgo conversion)
REVERSAL_PROBABILITY = {
    'enabled': True,
    'swing_length': 20,  # Pivot detection length
    'max_reversals': 1000,  # Maximum historical reversals to analyze
    'normalize_data': False,  # Normalize price deltas as percentages
    'percentiles': [25, 50, 75, 90],  # Probability percentiles to display
    'no_overlapping': True,  # Prevent overlapping zones
    'show_marks': True,  # Show pivot marks on chart
    'colors': {
        'bullish': '#089981',
        'bearish': '#F23645'
    }
}

# Trading parameters
STOP_LOSS_OFFSET = 10  # Points
SIGNALS_REQUIRED = 3
VOB_TOUCH_TOLERANCE = 2  # Points (for entry trigger)
HTF_PIVOT_TOUCH_TOLERANCE = 5  # Points (for HTF pivot entry trigger)

# Strike selection
STRIKE_SELECTION = {
    "NORMAL_DAY": "ATM",
    "EXPIRY_DAY": "ITM_PLUS_ONE"
}

# ═══════════════════════════════════════════════════════════════════════
# VOB DETECTION SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# BigBeluga VOB parameters (Simple version)
VOB_VOLUME_LENGTH = 20
VOB_VOLUME_MULTIPLIER = 1.5

# Advanced VOB parameters (BigBeluga Pine Script conversion)
VOB_ADVANCED = {
    'enabled': True,
    'sensitivity': 5,  # Detection sensitivity (EMA period base)
    'mid_line': True,  # Show midpoint of order blocks
    'trend_shadow': True,  # Show trend shadow/fill
    'bullish_color': '#26ba9f',  # Teal for bullish OBs
    'bearish_color': '#6626ba',  # Purple for bearish OBs
    'max_levels': 15  # Maximum OB levels to track
}

# High Volume Points settings (BigBeluga Pine Script conversion)
HIGH_VOLUME_POINTS = {
    'enabled': True,
    'left_bars': 15,  # Bars to left of pivot
    'right_bars': 15,  # Bars to right of pivot
    'volume_filter': 2.0,  # Minimum normalized volume (0-6 scale)
    'circle_diameter': 1.0,  # Circle size scaling factor
    'show_levels': True,  # Show horizontal levels from HV points
    'upper_color': '#fda05e',  # Orange for pivot highs
    'lower_color': '#2fd68e',  # Green for pivot lows
    'volume_lookback': 300,  # Bars to analyze for volume percentile
    'percentile_rank': 95  # Percentile for reference volume
}

# ═══════════════════════════════════════════════════════════════════════
# OI ANALYSIS SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# ATM strike range for OI analysis
ATM_STRIKE_RANGE = 10  # ATM ±10 strikes

# OI refresh interval
OI_REFRESH_INTERVAL = 60  # seconds (1 minute)

# ═══════════════════════════════════════════════════════════════════════
# CHART SETTINGS
# ═══════════════════════════════════════════════════════════════════════

CHART_CANDLES_DISPLAY = 100  # Number of 1-min candles to show
CHART_HEIGHT = 600  # pixels
CHART_THEME = 'plotly_dark'

# Chart colors
CHART_COLORS = {
    'candle_up': '#089981',
    'candle_down': '#f23645',
    'pivot_3min': '#3B82F6',   # Blue
    'pivot_5min': '#8B5CF6',   # Purple
    'pivot_10min': '#EC4899',  # Pink
    'pivot_15min': '#F59E0B',  # Orange
    'vob_support': '#10B981',  # Green
    'vob_resistance': '#EF4444', # Red
    'current_price': '#FFFFFF'  # White
}

# ═══════════════════════════════════════════════════════════════════════
# UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════

AUTO_REFRESH_INTERVAL = 60  # seconds
DEMO_MODE = False  # Set True for testing

APP_TITLE = "🎯 HTF Pivot Trading System"
APP_SUBTITLE = "Multi-Timeframe Pivot Analysis | BigBeluga VOB | Live Chart"

# Color scheme
COLORS = {
    'bullish': '#089981',
    'bearish': '#f23645',
    'neutral': '#787B86',
    'success': '#00E676',
    'warning': '#FFB74D',
    'danger': '#FF3D00'
}

# ═══════════════════════════════════════════════════════════════════════
# ALERT SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# Alert threshold distances (in points)
ALERT_THRESHOLDS = {
    'CRITICAL': 2,   # Alert when ≤2 points away
    'WARNING': 5,    # Alert when ≤5 points away
    'INFO': 10       # Alert when ≤10 points away
}

# VOB Proximity Alert settings
VOB_PROXIMITY_ALERT = {
    'enabled': True,
    'notify_on_approach': True,   # Send alert when approaching VOB
    'notify_on_touch': True,      # Send alert when touching VOB
    'cooldown_seconds': 300       # Wait 5 minutes between alerts for same level
}

# High Volume Points Alert settings
HVP_ALERT = {
    'enabled': True,
    'notify_on_formation': True,  # Alert when new HVP forms
    'notify_on_proximity': True,  # Alert when price near HVP
    'proximity_threshold': 5,     # Points distance to trigger proximity alert
    'cooldown_seconds': 300       # Wait 5 minutes between alerts for same level
}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCED INDICATORS SETTINGS (from Pine Script)
# ═══════════════════════════════════════════════════════════════════════

# Volatility Ratio
VOLATILITY_LENGTH = 14
VOLATILITY_THRESHOLD = 1.5

# Volume ROC
VOLUME_ROC_LENGTH = 14
VOLUME_THRESHOLD = 1.2

# OBV Smoothing
OBV_SMOOTHING = 21

# Force Index
FORCE_INDEX_LENGTH = 13
FORCE_INDEX_SMOOTHING = 2

# Price ROC
PRICE_ROC_LENGTH = 12

# Divergence Detection
DIVERGENCE_LOOKBACK = 30
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Choppiness Index
CHOPPINESS_LENGTH = 14
CHOPPINESS_HIGH_THRESHOLD = 61.8
CHOPPINESS_LOW_THRESHOLD = 38.2

# Market Breadth
BREADTH_THRESHOLD = 60  # % of stocks bullish to consider market bullish

# Range Detection
RANGE_PCT_THRESHOLD = 2.0  # % range to consider range-bound
RANGE_MIN_BARS = 20  # minimum bars in range
EMA_SPREAD_THRESHOLD = 0.5  # % EMA spread for range detection

# Adaptive Bias Calculation
DIVERGENCE_THRESHOLD = 60  # % threshold for divergence detection
NORMAL_FAST_WEIGHT = 2.0  # Fast indicators weight in normal mode
NORMAL_MEDIUM_WEIGHT = 3.0  # Medium indicators weight in normal mode
NORMAL_SLOW_WEIGHT = 5.0  # Slow indicators weight in normal mode
REVERSAL_FAST_WEIGHT = 5.0  # Fast indicators weight in reversal mode
REVERSAL_MEDIUM_WEIGHT = 3.0  # Medium indicators weight in reversal mode
REVERSAL_SLOW_WEIGHT = 2.0  # Slow indicators weight in reversal mode

# Bias Strength Threshold
BIAS_STRENGTH = 60  # Minimum % to show bias (can be adjusted for range)

# ═══════════════════════════════════════════════════════════════════════
# DHAN API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

DHAN_BASE_URL = "https://api.dhan.co/v2"

DHAN_ENDPOINTS = {
    'intraday': f"{DHAN_BASE_URL}/charts/intraday",
    'quote': f"{DHAN_BASE_URL}/marketfeed/quote",
    'ohlc': f"{DHAN_BASE_URL}/marketfeed/ohlc",
    'orders': f"{DHAN_BASE_URL}/orders",
    'positions': f"{DHAN_BASE_URL}/positions",
    'option_chain': f"{DHAN_BASE_URL}/optionchain"
}

# Exchange segments
EXCHANGE_SEGMENTS = {
    "NIFTY": "NSE_FNO",
    "SENSEX": "BSE_FNO"
}

# Security IDs (underlying)
SECURITY_IDS = {
    "NIFTY": "13",      # NIFTY Index
    "SENSEX": "51",     # SENSEX Index (verify this)
    "NIFTY_IDX": "13"   # For index price fetch
}

# Instrument types
INSTRUMENT_TYPES = {
    "NIFTY": "OPTIDX",
    "SENSEX": "OPTIDX"
}
