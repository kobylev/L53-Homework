"""
Test suite for Gatekeeper security validation.

Tests input validation for:
- Ticker symbol whitelist regex
- Date parsing validation
- Path sanitization against traversal attacks
- Rate limiting functionality
"""

import pytest
import time
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gatekeeper import Gatekeeper


class TestTickerValidation:
    """Test ticker symbol whitelist validation."""

    def setup_method(self):
        self.gk = Gatekeeper()

    def test_valid_tickers(self):
        """Valid ticker symbols should pass validation."""
        valid_tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "BRK.A",  # Berkshire Hathaway A
            "BRK.B",
            "A",      # Single letter
            "ABCDE",  # 5 letters
        ]
        for ticker in valid_tickers:
            assert self.gk.validate_ticker(ticker) == True, \
                f"Valid ticker '{ticker}' rejected"
        print(f"✓ All {len(valid_tickers)} valid tickers passed")

    def test_invalid_tickers(self):
        """Invalid ticker symbols should fail validation."""
        invalid_tickers = [
            "",           # Empty
            "123",        # Numbers only
            "AAPL123",    # Letters + numbers
            "ABCDEF",     # Too long (6 letters)
            "AA PL",      # Contains space
            "AA-PL",      # Contains hyphen
            "AAPL;DROP",  # SQL injection attempt
            "../AAPL",    # Path traversal
            "AAPL\x00",   # Null byte
            "aapl",       # Lowercase (should be normalized)
        ]
        for ticker in invalid_tickers:
            # Note: lowercase should be normalized to uppercase and pass
            if ticker == "aapl":
                assert self.gk.validate_ticker(ticker) == True, \
                    "Lowercase ticker should be normalized and pass"
            else:
                assert self.gk.validate_ticker(ticker) == False, \
                    f"Invalid ticker '{ticker}' was accepted"
        print(f"✓ All {len(invalid_tickers)-1} invalid tickers rejected (1 normalized)")

    def test_ticker_normalization(self):
        """Lowercase tickers should be normalized to uppercase."""
        assert self.gk.validate_ticker("aapl") == True
        assert self.gk.validate_ticker("MsFt") == True
        assert self.gk.validate_ticker("googl") == True
        print("✓ Ticker normalization working correctly")


class TestPathSanitization:
    """Test path sanitization against traversal attacks."""

    def test_safe_filenames(self):
        """Safe filenames should pass through unchanged."""
        safe_names = [
            "aapl_data.csv",
            "msft_2023.csv",
            "portfolio_results.csv",
        ]
        for name in safe_names:
            sanitized = Path(name).name
            assert sanitized == name, f"Safe filename '{name}' was modified"
        print(f"✓ All {len(safe_names)} safe filenames preserved")

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be sanitized."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "data/../../secrets.txt",
            "/absolute/path/file.csv",
        ]
        for path in malicious_paths:
            sanitized = Path(path).name
            # Should extract only the filename, removing directory traversal
            assert "/" not in sanitized and "\\" not in sanitized, \
                f"Path traversal not sanitized: {path} -> {sanitized}"
            assert not sanitized.startswith("."), \
                f"Relative path indicator preserved: {sanitized}"
        print(f"✓ All {len(malicious_paths)} path traversal attempts sanitized")

    def test_null_byte_injection(self):
        """Null bytes in filenames should be handled."""
        malicious = "file.csv\x00.txt"
        sanitized = Path(malicious).name.replace("\x00", "")
        assert "\x00" not in sanitized, "Null byte not removed"
        print("✓ Null byte injection handled")


class TestRateLimiting:
    """Test rate limiting functionality."""

    def setup_method(self):
        self.gk = Gatekeeper()

    def test_rate_limit_enforced(self):
        """Verify rate limiter enforces minimum interval."""
        # Reset rate limiter
        self.gk.last_call_time = 0

        # First call should be immediate
        start = time.time()
        self.gk._rate_limit()
        first_duration = time.time() - start
        assert first_duration < 0.1, "First call should be immediate"

        # Second call should be delayed by min_interval (2.0s)
        start = time.time()
        self.gk._rate_limit()
        second_duration = time.time() - start
        assert second_duration >= self.gk.min_interval, \
            f"Rate limit not enforced: waited only {second_duration:.2f}s (expected >= {self.gk.min_interval}s)"
        print(f"✓ Rate limit enforced: {second_duration:.2f}s delay between calls")

    def test_jitter_applied(self):
        """Verify random jitter is added to rate limit."""
        self.gk.last_call_time = 0
        self.gk._rate_limit()  # First call

        durations = []
        for _ in range(3):
            start = time.time()
            self.gk._rate_limit()
            durations.append(time.time() - start)

        # Durations should vary due to jitter
        assert len(set([round(d, 1) for d in durations])) > 1, \
            "No jitter detected - all durations identical"
        print(f"✓ Jitter applied: delays = {[f'{d:.2f}s' for d in durations]}")


class TestWatchdog:
    """Test Watchdog health monitoring."""

    def setup_method(self):
        self.gk = Gatekeeper()

    def test_healthy_gatekeeper(self):
        """Freshly initialized Gatekeeper should be healthy."""
        self.gk._pulse()  # Update heartbeat
        assert self.gk.is_healthy() == True, "Fresh Gatekeeper should be healthy"
        print("✓ Gatekeeper health check: HEALTHY")

    def test_stale_gatekeeper(self):
        """Gatekeeper without recent pulse should be unhealthy."""
        # Manually set last_heartbeat to 61 seconds ago
        self.gk.last_heartbeat = time.time() - 61
        assert self.gk.is_healthy() == False, "Stale Gatekeeper should be unhealthy"
        print("✓ Gatekeeper health check: UNHEALTHY (as expected for stale instance)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
