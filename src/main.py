import argparse
import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train
from src.evaluate import evaluate
from src.gatekeeper import gatekeeper
from src.config import ASSETS_DIR, TICKER

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="FinTech RL Trading Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "full"], default="full",
                        help="Mode to run: train, evaluate, or full")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL, TSLA)")
    
    args = parser.parse_args()
    
    ticker = args.ticker
    secure_id = gatekeeper.get_secure_identifier(ticker or TICKER)
    cache_file = os.path.join(ASSETS_DIR, f"{secure_id}_data.csv")

    if ticker:
        logger.info(f"Checking ticker: {ticker} (Secure ID: {secure_id})")
        # If cache exists, we skip validation to save API calls
        if not os.path.exists(cache_file):
            if not gatekeeper.validate_ticker_exists(ticker):
                logger.error(f"Ticker '{ticker}' is invalid or not found. Exiting.")
                sys.exit(1)
        logger.info(f"Ticker '{ticker}' ready.")

    if args.mode in ["train", "full"]:
        logger.info("Starting Training Phase...")
        train(ticker=ticker)
        
    if args.mode in ["evaluate", "full"]:
        logger.info("Starting Evaluation Phase...")
        evaluate(ticker=ticker)

if __name__ == "__main__":
    main()
