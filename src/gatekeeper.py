import time
import re
import logging
import random
import hashlib
import threading
import yfinance as yf
import pandas as pd
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gatekeeper:
    """
    A strict security and rate-limiting proxy for API calls.
    Acts as the first line of defense and ensures system reliability.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Gatekeeper, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.last_call_time = 0
        self.min_interval = 2.0
        self.allowed_tickers = re.compile(r'^[A-Z]{1,5}$')
        self.validation_cache = {}
        self.last_heartbeat = time.time()
        self.request_count = 0
        logger.info("Gatekeeper: Initialized and secured.")

    def is_healthy(self) -> bool:
        """Returns True if the Gatekeeper has pulsed recently."""
        return (time.time() - self.last_heartbeat) < 60.0

    def _pulse(self):
        """Updates the heartbeat for the Watchdog."""
        self.last_heartbeat = time.time()

    def get_secure_identifier(self, ticker: str) -> str:
        """
        Cyber Security: Maps external identifiers to secure internal names.
        Prevents data leakage and identifier-based injection attacks.
        """
        sanitized = self._sanitize_ticker(ticker)
        return hashlib.sha256(sanitized.encode()).hexdigest()[:16]

    def _sanitize_ticker(self, ticker: str) -> str:
        """Strictly sanitizes the ticker symbol symbol using whitelist regex."""
        if not isinstance(ticker, str):
            raise ValueError("Ticker must be a string.")
        
        ticker = ticker.upper().strip()
        if not self.allowed_tickers.match(ticker):
            logger.warning(f"SECURITY ALERT: Malicious ticker attempt blocked: {ticker}")
            raise ValueError(f"Invalid or malicious ticker: {ticker}")
        
        return ticker

    def _rate_limit(self):
        """
        Load Balancing & Rate Limiting: Manages traffic to adhere to free-tier limits.
        Uses jittered delays to prevent pattern detection and service blocking.
        """
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed + (random.random() * 0.5)
            logger.info(f"Gatekeeper: Throttling active. Waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_call_time = time.time()
        self.request_count += 1
        self._pulse()

    def validate_ticker_exists(self, ticker: str) -> bool:
        """Checks if the ticker exists with security proxying and retries."""
        sanitized_ticker = self._sanitize_ticker(ticker)
        
        if sanitized_ticker in self.validation_cache:
            return self.validation_cache[sanitized_ticker]

        max_retries = 3
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                stock = yf.Ticker(sanitized_ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    self.validation_cache[sanitized_ticker] = True
                    return True
                
                # Fallback check
                if stock.info and 'symbol' in stock.info:
                    self.validation_cache[sanitized_ticker] = True
                    return True

            except Exception as e:
                if "Rate limited" in str(e) or "429" in str(e):
                    wait = (attempt + 1) * 5
                    logger.warning(f"Gatekeeper: Rate limited. Retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                else:
                    logger.error(f"Gatekeeper: Validation error: {e}")
                    break
        
        return False

    def fetch_stock_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Proxy method for fetching data.
        Integrates Sanitization and Rate Limiting.
        """
        sanitized_ticker = self._sanitize_ticker(ticker)
        
        max_retries = 3
        for attempt in range(max_retries):
            self._rate_limit()
            logger.info(f"Gatekeeper: Proxying request for {sanitized_ticker} (Attempt {attempt+1})")
            try:
                data = yf.download(sanitized_ticker, start=start, end=end, progress=False)
                if not data.empty:
                    # Data Sanitization: Ensure strictly numeric and remove MultiIndex
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    allowed_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    data = data[data.columns.intersection(allowed_cols)]
                    for col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    return data.dropna()
            except Exception as e:
                if "Rate limited" in str(e) or "429" in str(e):
                    wait = (attempt + 1) * 5
                    time.sleep(wait)
                else:
                    logger.error(f"Gatekeeper: Proxy error: {e}")
                    break
        
        return None

    def fetch_sentiment(self, ticker: str) -> float:
        """
        Cyber Security: Mock LLM-based sentiment analysis.
        In a production system, this would call an LLM API (e.g., Gemini, GPT-4)
        to analyze recent news headlines for the given ticker.
        Returns a score between 0.0 (negative) and 1.0 (positive).
        """
        self._pulse()
        # Mocking sentiment based on ticker hash to ensure consistent results per session
        seed = int(self.get_secure_identifier(ticker), 16) % 1000
        random.seed(seed)
        return random.random()

class GatekeeperWatchdog(threading.Thread):
    """
    Reliability: Continuously monitors the Gatekeeper to ensure it remains functional.
    """
    def __init__(self, gatekeeper_instance, interval=10):
        super().__init__(daemon=True)
        self.gk = gatekeeper_instance
        self.interval = interval
        self.running = True
        logger.info("Watchdog: Started monitoring Gatekeeper.")

    def run(self):
        while self.running:
            if not self.gk.is_healthy():
                logger.critical("WATCHDOG: Gatekeeper UNRESPONSIVE! System stability at risk.")
                # In a real system, we might trigger a container restart or instance reset.
            time.sleep(self.interval)

gatekeeper = Gatekeeper()
watchdog = GatekeeperWatchdog(gatekeeper)
watchdog.start()
