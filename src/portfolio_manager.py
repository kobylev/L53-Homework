"""
Multi-Ticker Portfolio Management System
Supports simultaneous trading across multiple stocks with diversification
"""
import torch
import numpy as np
import pandas as pd
import logging
import os
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DuelingDQN
from src.datasets import TradingDataset, get_train_test_split
from src.config import DEVICE, WINDOW_SIZE, INITIAL_BALANCE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTickerPortfolio:
    """
    Portfolio manager for trading multiple tickers simultaneously
    Implements diversification and risk management strategies
    """

    def __init__(self,
                 tickers: List[str],
                 initial_balance: float = INITIAL_BALANCE,
                 max_position_size: float = 0.3,  # Max 30% per ticker
                 transaction_fee: float = 0.001):
        """
        Args:
            tickers: List of stock ticker symbols
            initial_balance: Starting capital
            max_position_size: Maximum portfolio allocation per ticker (0-1)
            transaction_fee: Transaction cost as decimal (0.001 = 0.1%)
        """
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_fee = transaction_fee

        # Portfolio state
        self.cash = initial_balance
        self.holdings = {ticker: 0 for ticker in tickers}  # shares held
        self.datasets = {}
        self.models = {}
        self.current_prices = {}

        logger.info(f"Initialized portfolio with {len(tickers)} tickers: {tickers}")
        logger.info(f"Max position size: {max_position_size*100}% per ticker")

    def load_datasets(self):
        """Load historical data for all tickers"""
        logger.info("Loading datasets for all tickers...")

        for ticker in self.tickers:
            try:
                dataset = TradingDataset(ticker=ticker)
                self.datasets[ticker] = dataset
                logger.info(f"  ✓ Loaded {ticker}: {len(dataset.data)} data points")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {ticker}: {str(e)}")

        logger.info(f"Successfully loaded {len(self.datasets)}/{len(self.tickers)} datasets")

    def load_models(self, model_dir: str = "assets"):
        """Load trained models for each ticker"""
        logger.info("Loading trained models...")

        for ticker in self.tickers:
            model_path = os.path.join(model_dir, f"{ticker}_model.pth")

            if os.path.exists(model_path):
                model = DuelingDQN().to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                self.models[ticker] = model
                logger.info(f"  ✓ Loaded model for {ticker}")
            else:
                # Use default model if ticker-specific model doesn't exist
                default_model_path = os.path.join(model_dir, "trading_model.pth")
                if os.path.exists(default_model_path):
                    model = DuelingDQN().to(DEVICE)
                    model.load_state_dict(torch.load(default_model_path, map_location=DEVICE))
                    model.eval()
                    self.models[ticker] = model
                    logger.warning(f"  ! Using default model for {ticker}")

        logger.info(f"Loaded {len(self.models)}/{len(self.tickers)} models")

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + holdings)"""
        holdings_value = sum(
            self.holdings[ticker] * self.current_prices.get(ticker, 0)
            for ticker in self.tickers
        )
        return self.cash + holdings_value

    def get_position_size(self, ticker: str) -> float:
        """Get current position size as fraction of portfolio"""
        if ticker not in self.current_prices:
            return 0.0

        position_value = self.holdings[ticker] * self.current_prices[ticker]
        total_value = self.get_portfolio_value()

        return position_value / total_value if total_value > 0 else 0.0

    def execute_trade(self, ticker: str, action: int, current_price: float) -> Dict:
        """
        Execute a trade for a specific ticker

        Args:
            ticker: Stock ticker
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Current stock price

        Returns:
            Dictionary with trade details
        """
        self.current_prices[ticker] = current_price
        portfolio_value_before = self.get_portfolio_value()

        trade_details = {
            'ticker': ticker,
            'action': action,
            'price': current_price,
            'shares_traded': 0,
            'cash_before': self.cash,
            'cash_after': self.cash,
            'holdings_before': self.holdings[ticker],
            'holdings_after': self.holdings[ticker],
            'portfolio_value_before': portfolio_value_before
        }

        # Check position size limit
        current_position = self.get_position_size(ticker)

        if action == 1:  # Buy
            # Calculate maximum allowed purchase
            max_position_value = self.max_position_size * portfolio_value_before
            current_position_value = self.holdings[ticker] * current_price
            available_allocation = max_position_value - current_position_value

            if available_allocation > 0 and self.cash > current_price:
                # Buy as much as allowed within constraints
                max_shares_by_cash = self.cash // (current_price * (1 + self.transaction_fee))
                max_shares_by_allocation = int(available_allocation // (current_price * (1 + self.transaction_fee)))

                shares_to_buy = min(max_shares_by_cash, max_shares_by_allocation)

                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    self.cash -= cost
                    self.holdings[ticker] += shares_to_buy
                    trade_details['shares_traded'] = shares_to_buy

        elif action == 2:  # Sell
            if self.holdings[ticker] > 0:
                shares_to_sell = self.holdings[ticker]
                proceeds = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.cash += proceeds
                self.holdings[ticker] = 0
                trade_details['shares_traded'] = -shares_to_sell

        # Update trade details
        trade_details['cash_after'] = self.cash
        trade_details['holdings_after'] = self.holdings[ticker]
        trade_details['portfolio_value_after'] = self.get_portfolio_value()

        return trade_details

    def backtest(self, test_ratio: float = 0.2) -> pd.DataFrame:
        """
        Backtest the portfolio strategy

        Args:
            test_ratio: Fraction of data to use for testing

        Returns:
            DataFrame with backtest results
        """
        logger.info("Starting multi-ticker backtest...")

        # Prepare test environments
        test_data = {}
        test_prices = {}

        for ticker in self.tickers:
            if ticker not in self.datasets or ticker not in self.models:
                logger.warning(f"Skipping {ticker} - missing data or model")
                continue

            dataset = self.datasets[ticker]
            _, ticker_test_data, _ = get_train_test_split(dataset, split_ratio=1-test_ratio)

            # Get original prices
            original_prices = dataset.data['Close'].values
            split_idx = int(len(dataset.data) * (1-test_ratio))
            ticker_test_prices = original_prices[split_idx:]

            test_data[ticker] = ticker_test_data
            test_prices[ticker] = ticker_test_prices

        # Find minimum test length across all tickers
        min_test_len = min(len(test_data[t]) for t in test_data.keys())

        # Backtest
        portfolio_history = []
        step_indices = {ticker: 0 for ticker in test_data.keys()}

        for step in range(min_test_len):
            step_trades = []

            # Get actions for all tickers
            for ticker in test_data.keys():
                state = test_data[ticker][step].to(DEVICE).unsqueeze(0)
                current_price = test_prices[ticker][step + WINDOW_SIZE - 1]

                # Get model prediction
                with torch.no_grad():
                    q_values = self.models[ticker](state)
                    action = q_values.argmax(dim=1).item()

                # Execute trade
                trade = self.execute_trade(ticker, action, current_price)
                step_trades.append(trade)

            # Record portfolio state
            portfolio_snapshot = {
                'step': step,
                'total_value': self.get_portfolio_value(),
                'cash': self.cash,
                **{f'{ticker}_holdings': self.holdings[ticker] for ticker in self.tickers},
                **{f'{ticker}_position_pct': self.get_position_size(ticker)*100 for ticker in self.tickers}
            }
            portfolio_history.append(portfolio_snapshot)

            if step % 50 == 0:
                logger.info(f"Step {step}/{min_test_len} | Portfolio: ${self.get_portfolio_value():.2f}")

        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_history)

        # Calculate performance metrics
        final_value = results_df['total_value'].iloc[-1]
        roi = ((final_value - self.initial_balance) / self.initial_balance) * 100
        max_value = results_df['total_value'].max()
        max_drawdown = ((max_value - results_df['total_value'].min()) / max_value) * 100

        logger.info("\n" + "="*70)
        logger.info("MULTI-TICKER PORTFOLIO BACKTEST RESULTS")
        logger.info("="*70)
        logger.info(f"Tickers Traded: {', '.join(self.tickers)}")
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        logger.info(f"Total Return: ${final_value - self.initial_balance:.2f}")
        logger.info(f"ROI: {roi:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Final Cash: ${self.cash:.2f}")

        for ticker in self.tickers:
            if ticker in self.holdings:
                logger.info(f"  {ticker}: {self.holdings[ticker]} shares (${self.holdings[ticker] * self.current_prices[ticker]:.2f})")

        logger.info("="*70 + "\n")

        # Save results
        results_path = os.path.join("assets", "logs", "portfolio_backtest.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Backtest results saved to {results_path}")

        return results_df


if __name__ == "__main__":
    # Example usage
    tickers = ["MSFT", "AAPL", "GOOGL"]

    portfolio = MultiTickerPortfolio(
        tickers=tickers,
        initial_balance=50000,
        max_position_size=0.35
    )

    # Load data and models
    portfolio.load_datasets()
    portfolio.load_models()

    # Run backtest
    if len(portfolio.datasets) > 0 and len(portfolio.models) > 0:
        results = portfolio.backtest(test_ratio=0.2)
    else:
        logger.error("Cannot run backtest - missing data or models")
