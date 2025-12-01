"""
Data Cache Manager with Background Loading
===========================================

This module provides:
- Thread-safe caching for all data sources
- Background data loading and auto-refresh
- Pre-loading of all tab data on startup
- Optimized refresh cycles to prevent API rate limiting
- Smart cache invalidation and updates

Cache Strategy (Optimized for Rate Limiting):
- NIFTY/SENSEX data: 60-second TTL, background refresh every 45 seconds (config-driven)
- Bias Analysis: 60-second TTL, background refresh every 300 seconds (5 minutes)
- Option Chain: 60-second TTL, background refresh
- Advanced Charts: 60-second TTL, background refresh

Note: Refresh intervals increased to prevent HTTP 429 (rate limit) errors
Previous 10-second interval caused overlapping cycles and exceeded API limits
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
import streamlit as st
import pandas as pd
from functools import wraps
from market_hours_scheduler import scheduler, is_within_trading_hours
import config


class DataCacheManager:
    """
    Thread-safe data cache manager with background loading and auto-refresh
    """

    def __init__(self):
        """Initialize cache manager"""
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_locks = {}
        self._background_threads = {}
        self._stop_threads = threading.Event()
        self._main_lock = threading.Lock()

        # Cache TTLs (in seconds)
        self.ttl_config = {
            'nifty_data': 60,
            'sensex_data': 60,
            'bias_analysis': 60,
            'option_chain': 60,
            'advanced_chart': 60,
        }

        # Background refresh intervals (in seconds)
        # Note: These are dynamically adjusted based on market session
        self.refresh_intervals = {
            'market_data': 10,      # NIFTY/SENSEX (adjusted by market session)
            'analysis_data': 60,    # All analysis (adjusted by market session)
        }

        # Market hours awareness
        self.market_hours_enabled = getattr(config, 'MARKET_HOURS_ENABLED', True)

    def _get_lock(self, cache_key: str) -> threading.Lock:
        """Get or create a lock for a cache key"""
        with self._main_lock:
            if cache_key not in self._cache_locks:
                self._cache_locks[cache_key] = threading.Lock()
            return self._cache_locks[cache_key]

    def get(self, cache_key: str, default=None) -> Any:
        """
        Get value from cache

        Args:
            cache_key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key in self._cache:
                # Check if cache is still valid
                if cache_key in self.ttl_config:
                    ttl = self.ttl_config[cache_key]
                    timestamp = self._cache_timestamps.get(cache_key, 0)
                    if time.time() - timestamp < ttl:
                        return self._cache[cache_key]
                else:
                    # No TTL configured, return cached value
                    return self._cache[cache_key]

            return default

    def set(self, cache_key: str, value: Any):
        """
        Set value in cache

        Args:
            cache_key: Cache key
            value: Value to cache
        """
        lock = self._get_lock(cache_key)
        with lock:
            self._cache[cache_key] = value
            self._cache_timestamps[cache_key] = time.time()

    def invalidate(self, cache_key: str):
        """
        Invalidate cache entry

        Args:
            cache_key: Cache key to invalidate
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
            if cache_key in self._cache_timestamps:
                del self._cache_timestamps[cache_key]

    def is_valid(self, cache_key: str) -> bool:
        """
        Check if cache entry is valid

        Args:
            cache_key: Cache key

        Returns:
            True if cache is valid, False otherwise
        """
        lock = self._get_lock(cache_key)
        with lock:
            if cache_key not in self._cache:
                return False

            if cache_key in self.ttl_config:
                ttl = self.ttl_config[cache_key]
                timestamp = self._cache_timestamps.get(cache_key, 0)
                return time.time() - timestamp < ttl

            return True

    def get_or_load(self, cache_key: str, loader_func: Callable, *args, **kwargs) -> Any:
        """
        Get from cache or load using loader function

        Args:
            cache_key: Cache key
            loader_func: Function to load data if not cached
            *args: Arguments for loader function
            **kwargs: Keyword arguments for loader function

        Returns:
            Cached or loaded value
        """
        # Try to get from cache first
        cached_value = self.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Load data
        try:
            value = loader_func(*args, **kwargs)
            self.set(cache_key, value)
            return value
        except Exception as e:
            # Return cached value even if expired, better than nothing
            lock = self._get_lock(cache_key)
            with lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
            raise e

    def start_background_refresh(self, cache_key: str, loader_func: Callable,
                                 interval: int = 60, *args, **kwargs):
        """
        Start background refresh for a cache key

        Args:
            cache_key: Cache key
            loader_func: Function to load data
            interval: Refresh interval in seconds
            *args: Arguments for loader function
            **kwargs: Keyword arguments for loader function
        """
        if cache_key in self._background_threads:
            return  # Already running

        def refresh_loop():
            """Background refresh loop with market hours awareness"""
            while not self._stop_threads.is_set():
                try:
                    # Check if market hours validation is enabled
                    if self.market_hours_enabled:
                        # Only fetch data during trading hours
                        if is_within_trading_hours():
                            # Load data
                            value = loader_func(*args, **kwargs)
                            self.set(cache_key, value)

                            # Use session-based refresh interval
                            session = scheduler.get_market_session()
                            actual_interval = scheduler.get_refresh_interval(session)
                        else:
                            # Market closed - use minimal refresh interval
                            actual_interval = config.REFRESH_INTERVALS.get('closed', 300)
                            # Optionally update cache with "market closed" status
                            # (only if needed by the application)
                    else:
                        # Market hours checking disabled - always refresh
                        value = loader_func(*args, **kwargs)
                        self.set(cache_key, value)
                        actual_interval = interval

                except Exception as e:
                    print(f"Background refresh error for {cache_key}: {e}")
                    actual_interval = interval  # Use default on error

                # Wait for interval or stop event
                self._stop_threads.wait(actual_interval)

        # Start thread
        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()
        self._background_threads[cache_key] = thread

    def stop_all_background_refresh(self):
        """Stop all background refresh threads"""
        self._stop_threads.set()

        # Wait for all threads to finish
        for thread in self._background_threads.values():
            thread.join(timeout=5)

        self._background_threads.clear()
        self._stop_threads.clear()

    def clear_all(self):
        """Clear all cache entries"""
        with self._main_lock:
            self._cache.clear()
            self._cache_timestamps.clear()


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> DataCacheManager:
    """
    Get global cache manager instance

    Returns:
        DataCacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DataCacheManager()
    return _cache_manager


def cache_with_ttl(cache_key: str, ttl: int = 60):
    """
    Decorator to cache function results with TTL

    Args:
        cache_key: Cache key
        ttl: Time to live in seconds

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()

            # Try to get from cache
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Load data
            value = func(*args, **kwargs)
            cache_manager.set(cache_key, value)
            return value

        return wrapper
    return decorator


def preload_all_data():
    """
    Pre-load all data for all tabs in background

    This function should be called on app startup to pre-load:
    - NIFTY/SENSEX data
    - Bias Analysis data
    - Option Chain data (for main instruments)
    - Advanced Chart data (for main symbols)
    """
    cache_manager = get_cache_manager()

    # Import here to avoid circular imports
    from market_data import fetch_nifty_data, fetch_sensex_data
    from bias_analysis import BiasAnalysisPro

    def load_market_data():
        """Load market data in background"""
        try:
            # Load NIFTY data
            nifty_data = fetch_nifty_data()
            cache_manager.set('nifty_data', nifty_data)

            # Load SENSEX data
            sensex_data = fetch_sensex_data()
            cache_manager.set('sensex_data', sensex_data)
        except Exception as e:
            print(f"Error loading market data: {e}")

    def load_bias_analysis_data():
        """Load bias analysis data in background"""
        try:
            if 'bias_analyzer' in st.session_state:
                analyzer = st.session_state.bias_analyzer
            else:
                analyzer = BiasAnalysisPro()

            # Default to NIFTY analysis
            results = analyzer.analyze_all_bias_indicators("^NSEI")
            cache_manager.set('bias_analysis', results)
        except Exception as e:
            print(f"Error loading bias analysis data: {e}")

    # Start background threads for continuous refresh

    # Market data: refresh using config interval to prevent rate limiting
    # Uses regular session interval from config (45 seconds) to avoid overlapping cycles
    cache_manager.start_background_refresh(
        'market_data_refresh',
        load_market_data,
        interval=config.REFRESH_INTERVALS['regular']
    )

    # Bias analysis: refresh every 300 seconds (5 minutes) to reduce API load
    # Previous 60-second interval was contributing to rate limiting
    cache_manager.start_background_refresh(
        'bias_analysis_refresh',
        load_bias_analysis_data,
        interval=300
    )

    # Initial load (immediate)
    initial_load_thread = threading.Thread(target=load_market_data, daemon=True)
    initial_load_thread.start()


def get_cached_nifty_data():
    """
    Get cached NIFTY data (non-blocking)

    Returns cached data or triggers background load if not available.
    Never blocks - returns None if data not ready yet.

    Returns:
        NIFTY data dict or None if not yet loaded
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get('nifty_data')

    if cached_data is not None:
        return cached_data

    # If not cached, trigger background load (non-blocking)
    # Don't block the UI - let background thread handle it
    def load_in_background():
        try:
            from market_data import fetch_nifty_data
            data = fetch_nifty_data()
            cache_manager.set('nifty_data', data)
        except Exception as e:
            print(f"Error loading NIFTY data in background: {e}")

    # Start background thread
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()

    # Return None immediately (non-blocking)
    return None


def get_cached_sensex_data():
    """
    Get cached SENSEX data (non-blocking)

    Returns cached data or triggers background load if not available.
    Never blocks - returns None if data not ready yet.

    Returns:
        SENSEX data dict or None if not yet loaded
    """
    cache_manager = get_cache_manager()
    cached_data = cache_manager.get('sensex_data')

    if cached_data is not None:
        return cached_data

    # If not cached, trigger background load (non-blocking)
    # Don't block the UI - let background thread handle it
    def load_in_background():
        try:
            from market_data import fetch_sensex_data
            data = fetch_sensex_data()
            cache_manager.set('sensex_data', data)
        except Exception as e:
            print(f"Error loading SENSEX data in background: {e}")

    # Start background thread
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()

    # Return None immediately (non-blocking)
    return None


def get_cached_bias_analysis_results():
    """
    Get cached Bias Analysis results

    Returns:
        Bias analysis results dict or None
    """
    cache_manager = get_cache_manager()
    return cache_manager.get('bias_analysis')


def invalidate_all_caches():
    """Invalidate all caches (useful for manual refresh)"""
    cache_manager = get_cache_manager()
    cache_manager.clear_all()
