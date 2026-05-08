import pandas as pd
import numpy as np
import torch
import os
import sys
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gatekeeper import gatekeeper
from src.config import WINDOW_SIZE, TICKER, START_DATE, END_DATE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingDataset:
    def __init__(self, ticker=TICKER, start=START_DATE, end=END_DATE, window_size=WINDOW_SIZE):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.window_size = window_size
        self.data = self._load_and_process()
        
    def _load_and_process(self):
        import os
        secure_id = gatekeeper.get_secure_identifier(self.ticker)
        cache_path = os.path.join("assets", f"{secure_id}_data.csv")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading secured data for {self.ticker} (ID: {secure_id})...")
            # Use low_memory=False to avoid DtypeWarning on large files
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            
            # Cleaning: If yfinance MultiIndex garbage is present (Ticker/Price rows)
            if 'Ticker' in df.index or 'Price' in df.index:
                df = df.drop(['Ticker', 'Price'], errors='ignore')
            
            # Force numeric conversion for all columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
        else:
            df = gatekeeper.fetch_stock_data(self.ticker, self.start, self.end)
            if df is None:
                raise ValueError("Failed to fetch data.")
            
            # Flatten MultiIndex if it exists before saving
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.to_csv(cache_path)
            logger.info(f"Saved {self.ticker} to local cache.")
        
        # Calculate Technical Indicators
        # yfinance returns MultiIndex columns sometimes, flatten them if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        
        # Drop rows with NaN from indicators
        df = df.dropna()
        
        # Select features: Close, Volume, RSI, MACD
        features = df[['Close', 'Volume', 'RSI', 'MACD']].copy()
        
        # Normalize features (Min-Max scaling for stability in RL)
        self.min_vals = features.min()
        self.max_vals = features.max()
        normalized_df = (features - self.min_vals) / (self.max_vals - self.min_vals)
        
        return normalized_df

    def get_windows(self):
        data_arr = self.data.values
        windows = []
        for i in range(len(data_arr) - self.window_size):
            window = data_arr[i : i + self.window_size]
            windows.append(window)
        
        # Convert to Tensor [Samples, Features, Window] for 1D CNN
        windows = np.array(windows) # [Samples, Window, Features]
        windows = np.transpose(windows, (0, 2, 1)) # [Samples, Features, Window]
        return torch.tensor(windows, dtype=torch.float32)

def get_train_test_split(dataset, split_ratio=0.8):
    windows = dataset.get_windows()
    split_idx = int(len(windows) * split_ratio)
    train_data = windows[:split_idx]
    test_data = windows[split_idx:]
    return train_data, test_data

class TradingEnv:
    """A simple trading environment for the RL agent."""
    def __init__(self, data_tensor, original_prices, initial_balance=10000.0, fee=0.001):
        self.data = data_tensor # [Samples, Features, Window]
        self.prices = original_prices # Corresponding Close prices
        self.initial_balance = initial_balance
        self.fee = fee
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.total_reward = 0
        self.done = False
        return self.data[self.current_step]

    def step(self, action):
        # Actions: 0: Hold, 1: Buy, 2: Sell
        current_price = self.prices[self.current_step + WINDOW_SIZE - 1]
        
        reward = 0
        if action == 1: # Buy
            if self.balance > current_price:
                shares_to_buy = self.balance // (current_price * (1 + self.fee))
                if shares_to_buy > 0:
                    self.shares += shares_to_buy
                    self.balance -= shares_to_buy * current_price * (1 + self.fee)
        elif action == 2: # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price * (1 - self.fee)
                self.shares = 0
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate Portfolio Value
        portfolio_value = self.balance + self.shares * current_price
        
        # Reward is the change in portfolio value
        # Simple reward: total portfolio value at the end or step-wise gain
        next_price = self.prices[self.current_step + WINDOW_SIZE - 1]
        next_portfolio_value = self.balance + self.shares * next_price
        reward = (next_portfolio_value - portfolio_value) / portfolio_value
        
        self.total_reward += reward
        return self.data[self.current_step], reward, self.done, portfolio_value

class PaperTradingEnv(TradingEnv):
    """
    Live Trading Integration: Mock Paper Trading Mode.
    Simulates real-time execution by using the most recent market data available
    and applying realistic latency/slippage models.
    """
    def __init__(self, ticker, initial_balance=10000.0, fee=0.001):
        # Fetch the most recent 60 days to ensure we have a full 30-day window + buffer
        dataset = TradingDataset(ticker=ticker)
        self.full_dataset = dataset
        data_tensor = dataset.get_windows()
        prices = dataset.data['Close'].values * (dataset.max_vals['Close'] - dataset.min_vals['Close']) + dataset.min_vals['Close']
        
        super().__init__(data_tensor, prices, initial_balance, fee)
        logger.info(f"PaperTrading: Initialized for {ticker} with ${initial_balance:.2f} balance.")

    def step(self, action):
        # Add mock slippage (0.01% - 0.05%)
        slippage = random.uniform(0.0001, 0.0005)
        self.fee += slippage
        
        obs, reward, done, portfolio_value = super().step(action)
        
        # Reset fee for next step
        self.fee -= slippage
        
        # Log active paper trade
        action_name = ["HOLD", "BUY", "SELL"][action]
        logger.info(f"PAPER TRADE: Action={action_name} | Portfolio: ${portfolio_value:.2f} | Step: {self.current_step}")
        
        return obs, reward, done, portfolio_value

class MultiTickerDataset:
    """
    Portfolio Management: Handles multiple tickers simultaneously.
    Provides concatenated data windows for the agent to make portfolio-wide decisions.
    """
    def __init__(self, tickers, start=START_DATE, end=END_DATE, window_size=WINDOW_SIZE):
        self.tickers = tickers
        self.datasets = {ticker: TradingDataset(ticker, start, end, window_size) for ticker in tickers}
        self.window_size = window_size

    def get_combined_windows(self):
        # Align datasets on common dates
        common_index = None
        for ds in self.datasets.values():
            if common_index is None:
                common_index = ds.data.index
            else:
                common_index = common_index.intersection(ds.data.index)
        
        combined_windows = []
        for ticker in self.tickers:
            ds = self.datasets[ticker]
            aligned_data = ds.data.loc[common_index].values
            windows = []
            for i in range(len(aligned_data) - self.window_size):
                windows.append(aligned_data[i : i + self.window_size])
            combined_windows.append(np.array(windows))
        
        # Shape: [Tickers, Samples, Window, Features]
        combined_windows = np.array(combined_windows)
        # Transpose to: [Samples, Tickers, Features, Window]
        combined_windows = np.transpose(combined_windows, (1, 0, 3, 2))
        return torch.tensor(combined_windows, dtype=torch.float32), common_index

class MultiTickerEnv:
    """
    Environment for managing a portfolio of multiple assets.
    The agent receives state information for all assets and chooses an action for each.
    """
    def __init__(self, tickers, data_tensor, prices_df, initial_balance=10000.0, fee=0.001):
        self.tickers = tickers
        self.data = data_tensor # [Samples, Tickers, Features, Window]
        self.prices = prices_df # Aligned prices for all tickers
        self.initial_balance = initial_balance
        self.fee = fee
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        self.done = False
        return self.data[self.current_step]

    def step(self, actions):
        # actions: list of actions, one for each ticker
        total_portfolio_value = self.balance
        current_prices = {ticker: self.prices.iloc[self.current_step + WINDOW_SIZE - 1][ticker] for ticker in self.tickers}
        
        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            price = current_prices[ticker]
            
            if action == 1: # Buy
                if self.balance > price:
                    shares_to_buy = self.balance // (len(self.tickers) * price * (1 + self.fee))
                    if shares_to_buy > 0:
                        self.shares[ticker] += shares_to_buy
                        self.balance -= shares_to_buy * price * (1 + self.fee)
            elif action == 2: # Sell
                if self.shares[ticker] > 0:
                    self.balance += self.shares[ticker] * price * (1 - self.fee)
                    self.shares[ticker] = 0
            
            total_portfolio_value += self.shares[ticker] * price
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
            
        return self.data[self.current_step], 0, self.done, total_portfolio_value
