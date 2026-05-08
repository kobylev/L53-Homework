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
