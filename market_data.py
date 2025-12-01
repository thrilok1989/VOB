"""
Market Data Module - Using Dhan API
====================================

Fetches market data from Dhan API with proper rate limiting.
Replaces the old NSE-based data fetching.
"""

from datetime import datetime
import pytz
from dhan_data_fetcher import get_nifty_data, get_sensex_data, DhanDataFetcher
from market_hours_scheduler import scheduler, is_within_trading_hours

IST = pytz.timezone("Asia/Kolkata")


def is_market_open():
    """
    Check if NSE market is open for regular trading
    Uses centralized market hours scheduler (IST: 9:15 AM - 3:30 PM)
    """
    return scheduler.is_market_open()


def fetch_nifty_data():
    """
    Fetch live NIFTY data from Dhan API

    Returns:
        Dict with NIFTY data including spot price, ATM strike, expiry dates, etc.
    """
    try:
        data = get_nifty_data()
        return data
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def fetch_sensex_data():
    """
    Fetch live SENSEX data from Dhan API

    Returns:
        Dict with SENSEX data including spot price, ATM strike, etc.
    """
    try:
        data = get_sensex_data()
        return data
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def fetch_all_market_data():
    """
    Fetch all market data (NIFTY, SENSEX) in one sequential call

    This uses the rate-limited sequential fetching from DhanDataFetcher

    Returns:
        Dict with all market data
    """
    try:
        fetcher = DhanDataFetcher()
        all_data = fetcher.fetch_all_data_sequential()
        return all_data
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def check_vob_touch(current_price, vob_level, tolerance=5):
    """Check if price touched VOB level"""
    return abs(current_price - vob_level) <= tolerance


def get_market_status():
    """
    Get current market status with detailed information
    Uses centralized market hours scheduler
    """
    status = scheduler.get_market_status()
    now = scheduler.get_current_time_ist()

    # Determine if market is open based on scheduler
    if status['is_market_open']:
        return {
            'open': True,
            'message': '🟢 Market Open (Regular)',
            'time': now.strftime('%H:%M:%S IST'),
            'session': status['session']
        }
    elif status['session'] == 'pre_market':
        return {
            'open': True,
            'message': '🟡 Pre-Market Session',
            'time': now.strftime('%H:%M:%S IST'),
            'session': status['session']
        }
    elif status['session'] == 'post_market':
        return {
            'open': True,
            'message': '🟡 Post-Market Session',
            'time': now.strftime('%H:%M:%S IST'),
            'session': status['session']
        }
    elif status['is_weekend']:
        return {
            'open': False,
            'message': '🔴 Market Closed (Weekend)',
            'session': status['session']
        }
    elif status['is_holiday']:
        return {
            'open': False,
            'message': '🔴 Market Closed (Holiday)',
            'session': status['session']
        }
    else:
        # Outside trading hours
        next_open = scheduler.get_next_market_open()
        return {
            'open': False,
            'message': f'🔴 Market Closed (Outside Trading Hours)',
            'next_open': next_open.strftime('%Y-%m-%d %H:%M IST'),
            'session': status['session']
        }
