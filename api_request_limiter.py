"""
Global API Request Limiter
===========================

Centralized rate limiting for all API requests to prevent HTTP 429 errors.
This module provides:
- Global request queue (thread-safe across all instances)
- Per-endpoint rate limiting
- Exponential backoff for 429 responses
- Circuit breaker pattern for repeated failures
- Request tracking and metrics

Usage:
    from api_request_limiter import global_rate_limiter

    # Before making an API request
    global_rate_limiter.wait_for_slot('quote')
    response = requests.get(url)

    # Handle 429 responses
    if response.status_code == 429:
        global_rate_limiter.handle_rate_limit_error('quote')
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Thread-safe global rate limiter for API requests

    Implements:
    - Per-endpoint rate limiting
    - Request queue with minimum intervals
    - Exponential backoff for 429 errors
    - Circuit breaker for repeated failures
    """

    def __init__(self):
        """Initialize the global rate limiter"""
        self._lock = threading.RLock()

        # Rate limits (seconds between requests per endpoint type)
        self.rate_limits = {
            'quote': 1.0,        # 1 request/second (OHLC, LTP)
            'data': 0.2,         # 5 requests/second (Historical data)
            'option_chain': 3.0, # 1 request/3 seconds (Option chain)
            'default': 1.0       # Default for unknown endpoints
        }

        # Last request timestamp per endpoint
        self._last_request_time: Dict[str, float] = {}

        # Request count per endpoint (for metrics)
        self._request_count: Dict[str, int] = {}

        # Exponential backoff state
        self._backoff_until: Dict[str, float] = {}
        self._backoff_count: Dict[str, int] = {}
        self._max_backoff_time = 60.0  # Maximum backoff: 60 seconds

        # Circuit breaker state
        self._circuit_broken: Dict[str, bool] = {}
        self._circuit_break_until: Dict[str, float] = {}
        self._failure_count: Dict[str, int] = {}
        self._max_failures = 5  # Break circuit after 5 consecutive failures
        self._circuit_break_duration = 300.0  # 5 minutes

        # Global request queue (all endpoints)
        self._global_queue = deque(maxlen=1000)
        self._min_global_interval = 0.1  # Minimum 100ms between ANY requests
        self._last_global_request = 0.0

    def wait_for_slot(self, api_type: str) -> bool:
        """
        Wait for an available request slot (thread-safe)

        Args:
            api_type: Type of API endpoint ('quote', 'data', 'option_chain')

        Returns:
            True if slot acquired, False if circuit breaker is active
        """
        with self._lock:
            # Check circuit breaker
            if self._is_circuit_broken(api_type):
                logger.warning(f"Circuit breaker active for {api_type}. "
                             f"Waiting until {datetime.fromtimestamp(self._circuit_break_until[api_type])}")
                return False

            # Check exponential backoff
            if api_type in self._backoff_until:
                wait_until = self._backoff_until[api_type]
                if time.time() < wait_until:
                    wait_time = wait_until - time.time()
                    logger.info(f"Exponential backoff active for {api_type}. "
                              f"Waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                else:
                    # Backoff period expired, reset
                    del self._backoff_until[api_type]
                    self._backoff_count[api_type] = 0

            # Wait for global rate limit
            self._wait_global_interval()

            # Wait for endpoint-specific rate limit
            self._wait_endpoint_interval(api_type)

            # Update timestamps
            current_time = time.time()
            self._last_request_time[api_type] = current_time
            self._last_global_request = current_time

            # Update metrics
            self._request_count[api_type] = self._request_count.get(api_type, 0) + 1
            self._global_queue.append((api_type, current_time))

            return True

    def _wait_global_interval(self):
        """Wait for minimum global interval between ANY requests"""
        if self._last_global_request > 0:
            elapsed = time.time() - self._last_global_request
            if elapsed < self._min_global_interval:
                wait_time = self._min_global_interval - elapsed
                time.sleep(wait_time)

    def _wait_endpoint_interval(self, api_type: str):
        """Wait for endpoint-specific rate limit"""
        min_interval = self.rate_limits.get(api_type, self.rate_limits['default'])

        if api_type in self._last_request_time:
            elapsed = time.time() - self._last_request_time[api_type]
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                time.sleep(wait_time)

    def handle_rate_limit_error(self, api_type: str):
        """
        Handle HTTP 429 (Rate Limit Exceeded) error with exponential backoff

        Args:
            api_type: Type of API endpoint that was rate limited
        """
        with self._lock:
            # Increment backoff count
            count = self._backoff_count.get(api_type, 0) + 1
            self._backoff_count[api_type] = count

            # Calculate exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
            backoff_time = min(2 ** (count - 1), self._max_backoff_time)
            self._backoff_until[api_type] = time.time() + backoff_time

            logger.warning(f"Rate limit hit for {api_type}. "
                         f"Backing off for {backoff_time:.2f} seconds "
                         f"(attempt {count})")

            # Track failures for circuit breaker
            self._record_failure(api_type)

    def handle_success(self, api_type: str):
        """
        Record successful request (resets failure count)

        Args:
            api_type: Type of API endpoint
        """
        with self._lock:
            # Reset failure count on success
            if api_type in self._failure_count:
                self._failure_count[api_type] = 0

            # Close circuit breaker if it was open
            if api_type in self._circuit_broken:
                del self._circuit_broken[api_type]
                logger.info(f"Circuit breaker closed for {api_type}")

    def _record_failure(self, api_type: str):
        """Record API failure and potentially trigger circuit breaker"""
        failures = self._failure_count.get(api_type, 0) + 1
        self._failure_count[api_type] = failures

        if failures >= self._max_failures:
            # Trip circuit breaker
            self._circuit_broken[api_type] = True
            self._circuit_break_until[api_type] = time.time() + self._circuit_break_duration
            logger.error(f"Circuit breaker OPEN for {api_type}. "
                       f"Too many failures ({failures}). "
                       f"Will retry after {self._circuit_break_duration} seconds")

    def _is_circuit_broken(self, api_type: str) -> bool:
        """Check if circuit breaker is active for endpoint"""
        if api_type not in self._circuit_broken:
            return False

        # Check if circuit break period has expired
        if time.time() >= self._circuit_break_until[api_type]:
            # Try to close circuit
            del self._circuit_broken[api_type]
            del self._circuit_break_until[api_type]
            self._failure_count[api_type] = 0
            logger.info(f"Circuit breaker closed for {api_type} (timeout expired)")
            return False

        return True

    def get_metrics(self) -> Dict:
        """
        Get rate limiter metrics

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            return {
                'request_counts': dict(self._request_count),
                'backoff_active': {
                    k: (v - time.time())
                    for k, v in self._backoff_until.items()
                    if v > time.time()
                },
                'circuit_breakers': {
                    k: {
                        'active': v,
                        'until': datetime.fromtimestamp(self._circuit_break_until.get(k, 0)).isoformat()
                    }
                    for k, v in self._circuit_broken.items()
                },
                'recent_requests': len(self._global_queue),
                'failure_counts': dict(self._failure_count)
            }

    def reset(self):
        """Reset all rate limiting state (useful for testing)"""
        with self._lock:
            self._last_request_time.clear()
            self._request_count.clear()
            self._backoff_until.clear()
            self._backoff_count.clear()
            self._circuit_broken.clear()
            self._circuit_break_until.clear()
            self._failure_count.clear()
            self._global_queue.clear()
            self._last_global_request = 0.0
            logger.info("Rate limiter state reset")


# Global singleton instance
global_rate_limiter = GlobalRateLimiter()


# Convenience decorator for automatic rate limiting
def rate_limited(api_type: str):
    """
    Decorator to automatically apply rate limiting to functions

    Usage:
        @rate_limited('quote')
        def fetch_data():
            response = requests.get(url)
            return response
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Wait for slot
            if not global_rate_limiter.wait_for_slot(api_type):
                raise Exception(f"Circuit breaker active for {api_type}")

            # Execute function
            try:
                result = func(*args, **kwargs)

                # Check for 429 in response
                if hasattr(result, 'status_code'):
                    if result.status_code == 429:
                        global_rate_limiter.handle_rate_limit_error(api_type)
                        raise Exception(f"Rate limit exceeded for {api_type}")
                    elif result.status_code == 200:
                        global_rate_limiter.handle_success(api_type)

                return result
            except Exception as e:
                global_rate_limiter._record_failure(api_type)
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    print("Global Rate Limiter Test")
    print("=" * 50)

    # Simulate requests
    for i in range(5):
        print(f"\nRequest {i+1}:")
        if global_rate_limiter.wait_for_slot('quote'):
            print(f"  ✓ Slot acquired for 'quote'")

        time.sleep(0.5)

    # Show metrics
    print("\nMetrics:")
    print(global_rate_limiter.get_metrics())
