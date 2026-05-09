import pandas as pd
import numpy as np
import torch
import os
import sys
import logging
import random
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.gatekeeper import gatekeeper
from src.config import WINDOW_SIZE, TICKER, START_DATE, END_DATE


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
        secure_id = gatekeeper.get_secure_identifier(self.ticker)
        cache_path = os.path.join("assets", f"{secure_id}_data.csv")
        if os.path.exists(cache_path):
            logger.info(f"Loading secured data for {self.ticker} (ID: {secure_id})...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if 'Ticker' in df.index or 'Price' in df.index:
                df = df.drop(['Ticker', 'Price'], errors='ignore')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
        else:
            df = gatekeeper.fetch_stock_data(self.ticker, self.start, self.end)
            if df is None:
                raise ValueError("Failed to fetch data.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_path)
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df = df.dropna()
        return df[['Close', 'Volume', 'RSI', 'MACD']].copy()



def make_windows(data_arr, window_size):
    windows = []
    for i in range(len(data_arr) - window_size):
        window = data_arr[i : i + window_size]
        windows.append(window)
    windows = np.array(windows)
    return torch.tensor(np.transpose(windows, (0, 2, 1)), dtype=torch.float32)



def get_train_test_split(dataset, split_ratio=0.8):
    """FIX: Data Leakage Prevention (Split then Scale)."""
    df = dataset.data
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    train_windows = make_windows(train_scaled, dataset.window_size)
    test_windows = make_windows(test_scaled, dataset.window_size)
    return train_windows, test_windows, scaler



class TradingEnv:
    def __init__(self, data_tensor, original_prices, initial_balance=10000.0, fee=0.001):
        self.data = data_tensor
        self.prices = original_prices
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
        current_price = self.prices[self.current_step + WINDOW_SIZE - 1]
        if action == 1 and self.balance > current_price:
            shares_to_buy = self.balance // (current_price * (1 + self.fee))
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price * (1 + self.fee)
        elif action == 2 and self.shares > 0:
            self.balance += self.shares * current_price * (1 - self.fee)
            self.shares = 0

        # FIX 1: record old value BEFORE incrementing step (no look-ahead)
        old_portfolio_value = self.balance + self.shares * current_price
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_price = self.prices[self.current_step + WINDOW_SIZE - 1]
        new_portfolio_value = self.balance + self.shares * next_price
        reward = (new_portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-9)
        if action != 0:
            reward -= 0.001  # over-trading penalty
        self.total_reward += reward
        # FIX 2: return new_portfolio_value (was undefined 'portfolio_value')
        return self.data[self.current_step], reward, self.done, new_portfolio_value



class PaperTradingEnv(TradingEnv):
    def __init__(self, ticker, initial_balance=10000.0, fee=0.001):
        dataset = TradingDataset(ticker=ticker)
        data_tensor = dataset.get_windows() if hasattr(dataset, 'get_windows') else make_windows(dataset.data.values, WINDOW_SIZE)
        prices = dataset.data['Close'].values
        super().__init__(data_tensor, prices, initial_balance, fee)


    def step(self, action):
        slippage = random.uniform(0.0001, 0.0005)
        self.fee += slippage
        obs, reward, done, portfolio_value = super().step(action)
        self.fee -= slippage
        return obs, reward, done, portfolio_value



class MultiTickerDataset:
    def __init__(self, tickers, start=START_DATE, end=END_DATE, window_size=WINDOW_SIZE):
        self.tickers = tickers
        self.datasets = {ticker: TradingDataset(ticker, start, end, window_size) for ticker in tickers}
        self.window_size = window_size


    def get_combined_windows(self):
        common_index = None
        for ds in self.datasets.values():
            common_index = ds.data.index if common_index is None else common_index.intersection(ds.data.index)
        combined_windows = []
        for ticker in self.tickers:
            aligned_data = self.datasets[ticker].data.loc[common_index].values
            combined_windows.append(make_windows(aligned_data, self.window_size).numpy())
        combined_windows = np.array(combined_windows)
        combined_windows = np.transpose(combined_windows, (1, 0, 2, 3))
        return torch.tensor(combined_windows, dtype=torch.float32), common_index



class MultiTickerEnv:
    def __init__(self, tickers, data_tensor, prices_df, initial_balance=10000.0, fee=0.001):
        self.tickers = tickers
        self.data = data_tensor
        self.prices = prices_df
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
        current_prices = {ticker: self.prices.iloc[self.current_step + WINDOW_SIZE - 1][ticker]
                          for ticker in self.tickers}
        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            price = current_prices[ticker]
            if action == 1 and self.balance > price:
                shares_to_buy = self.balance // (len(self.tickers) * price * (1 + self.fee))
                if shares_to_buy > 0:
                    self.shares[ticker] += shares_to_buy
                    self.balance -= shares_to_buy * price * (1 + self.fee)
            elif action == 2 and self.shares[ticker] > 0:
                self.balance += self.shares[ticker] * price * (1 - self.fee)
                self.shares[ticker] = 0

        # FIX 3: compute old total AFTER actions, BEFORE step increment
        old_total = self.balance + sum(self.shares[t] * current_prices[t] for t in self.tickers)

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        new_prices = {t: self.prices.iloc[self.current_step + WINDOW_SIZE - 1][t] for t in self.tickers}
        new_total = self.balance + sum(self.shares[t] * new_prices[t] for t in self.tickers)

        reward = (new_total - old_total) / (old_total + 1e-9)
        if any(actions[i] != 0 for i in range(len(self.tickers))):
            reward -= 0.001

        return self.data[self.current_step], reward, self.done, new_total