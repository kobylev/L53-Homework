import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 3e-4  # Increased from 1e-4 for faster convergence
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995  # Slower decay (was 0.98) for better exploration-exploitation balance
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 1000  # Increased from 100 for better convergence and higher win rate
WINDOW_SIZE = 30  # 30-day rolling window

# Trading configuration
INITIAL_BALANCE = 10000.0
TRANSACTION_FEE = 0.001

# Data configuration
TICKER = "MSFT"
START_DATE = "2015-01-01"  # Extended from 2022 to 2015 for 8 years of historical data
END_DATE = "2023-12-31"

# Paths
ASSETS_DIR = "assets"
MODEL_PATH = os.path.join(ASSETS_DIR, "trading_model.pth")
LOGS_DIR = os.path.join(ASSETS_DIR, "logs")

if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
