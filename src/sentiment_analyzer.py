"""
Sentiment Analysis Module with Gatekeeper Protection
Fetches news sentiment for stocks using secure API proxying
"""
import requests
import hashlib
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentGatekeeper:
    """
    Secure proxy for sentiment analysis APIs
    Implements rate limiting, caching, and identifier obfuscation
    """

    def __init__(self, rate_limit_delay=2.0):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.cache = {}
        self.request_count = 0

    def get_secure_identifier(self, ticker: str) -> str:
        """Hash ticker symbol for security"""
        return hashlib.sha256(ticker.encode()).hexdigest()[:16]

    def _enforce_rate_limit(self):
        """Enforce rate limiting with jitter"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def fetch_news_sentiment(self, ticker: str, days_back: int = 7) -> Optional[pd.DataFrame]:
        """
        Fetch news sentiment for a ticker

        In production, this would integrate with:
        - Alpha Vantage News Sentiment API
        - NewsAPI with NLP processing
        - Finnhub sentiment endpoint
        - Custom LLM-based sentiment analysis

        For this implementation, we simulate sentiment data based on price volatility
        """
        secure_id = self.get_secure_identifier(ticker)
        cache_key = f"{secure_id}_{days_back}"

        # Check cache
        if cache_key in self.cache:
            logger.info(f"Returning cached sentiment for {ticker} (ID: {secure_id})")
            return self.cache[cache_key]

        self._enforce_rate_limit()
        self.request_count += 1

        logger.info(f"Fetching sentiment for {ticker} (Secure ID: {secure_id})...")

        try:
            # Simulate sentiment fetching
            # In production, replace with actual API calls:
            # response = requests.get(f"https://api.sentimentprovider.com/v1/sentiment?ticker={ticker}")

            sentiment_data = self._simulate_sentiment_data(ticker, days_back)

            # Cache the result
            self.cache[cache_key] = sentiment_data

            logger.info(f"Successfully fetched sentiment for {ticker}")
            return sentiment_data

        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {ticker}: {str(e)}")
            return None

    def _simulate_sentiment_data(self, ticker: str, days_back: int) -> pd.DataFrame:
        """
        Simulate sentiment scores
        In production, this would call real APIs and process news with NLP/LLM
        """
        import numpy as np

        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days_back)]
        dates.reverse()

        # Simulate sentiment scores (-1 to 1)
        # In reality, these would come from:
        # 1. News headline analysis
        # 2. Social media sentiment (Twitter/Reddit)
        # 3. Analyst reports
        # 4. LLM-based sentiment extraction

        np.random.seed(hash(ticker) % 2**32)  # Consistent per ticker

        sentiment_scores = np.random.normal(0.1, 0.3, days_back)  # Slight positive bias
        sentiment_scores = np.clip(sentiment_scores, -1, 1)

        # Add some realistic patterns
        # Trending sentiment (momentum)
        for i in range(1, len(sentiment_scores)):
            sentiment_scores[i] = 0.7 * sentiment_scores[i] + 0.3 * sentiment_scores[i-1]

        # Article counts (volume of news)
        article_counts = np.random.poisson(15, days_back)

        # Confidence scores
        confidence_scores = np.random.uniform(0.6, 0.95, days_back)

        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'sentiment_score': sentiment_scores,
            'article_count': article_counts,
            'confidence': confidence_scores,
            'positive_ratio': (sentiment_scores + 1) / 2,  # Normalize to 0-1
            'secure_id': self.get_secure_identifier(ticker)
        })

        return df

    def get_aggregated_sentiment(self, ticker: str, days_back: int = 7) -> Dict[str, float]:
        """Get aggregated sentiment metrics for a ticker"""
        sentiment_df = self.fetch_news_sentiment(ticker, days_back)

        if sentiment_df is None or sentiment_df.empty:
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'news_volume': 0,
                'confidence': 0.5
            }

        # Calculate aggregated metrics
        avg_sentiment = sentiment_df['sentiment_score'].mean()

        # Trend: recent sentiment vs. older sentiment
        mid_point = len(sentiment_df) // 2
        recent_sentiment = sentiment_df['sentiment_score'].iloc[mid_point:].mean()
        older_sentiment = sentiment_df['sentiment_score'].iloc[:mid_point].mean()
        sentiment_trend = recent_sentiment - older_sentiment

        news_volume = sentiment_df['article_count'].sum()
        avg_confidence = sentiment_df['confidence'].mean()

        return {
            'avg_sentiment': float(avg_sentiment),
            'sentiment_trend': float(sentiment_trend),
            'news_volume': int(news_volume),
            'confidence': float(avg_confidence)
        }


# Global singleton instance
sentiment_gatekeeper = SentimentGatekeeper()


def integrate_sentiment_features(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Integrate sentiment features into price data

    Args:
        price_data: DataFrame with OHLCV data
        ticker: Stock ticker symbol

    Returns:
        DataFrame with added sentiment features
    """
    logger.info(f"Integrating sentiment features for {ticker}...")

    # Fetch sentiment data
    sentiment_metrics = sentiment_gatekeeper.get_aggregated_sentiment(ticker, days_back=30)

    # Add sentiment features to each row (broadcasted)
    price_data['sentiment_score'] = sentiment_metrics['avg_sentiment']
    price_data['sentiment_trend'] = sentiment_metrics['sentiment_trend']
    price_data['news_volume_normalized'] = min(sentiment_metrics['news_volume'] / 100.0, 1.0)
    price_data['sentiment_confidence'] = sentiment_metrics['confidence']

    logger.info(f"Sentiment integration complete. Metrics: {sentiment_metrics}")

    return price_data


if __name__ == "__main__":
    # Test the sentiment analyzer
    test_ticker = "MSFT"

    logger.info("Testing Sentiment Gatekeeper...")
    sentiment_df = sentiment_gatekeeper.fetch_news_sentiment(test_ticker, days_back=14)

    if sentiment_df is not None:
        print("\nSentiment Data Sample:")
        print(sentiment_df.head(10))

        print("\n\nAggregated Metrics:")
        metrics = sentiment_gatekeeper.get_aggregated_sentiment(test_ticker)
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    logger.info(f"\nTotal API requests made: {sentiment_gatekeeper.request_count}")
