# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FinTech RL Trading System** - A production-ready Reinforcement Learning pipeline for algorithmic stock trading using 1D CNN + Dueling DQN architecture. Achieved **134.97% ROI** on MSFT test data with advanced features including Transformer baseline, sentiment analysis, multi-ticker portfolio management, and full containerization.

**Core Technology:** Python 3.10+, PyTorch 2.6 (CUDA 12.4), Streamlit, Docker

## Architecture Deep-Dive

### 1. The Gatekeeper Pattern (Critical Security Layer)

**Location:** `src/gatekeeper.py`

All external API interactions MUST go through the Gatekeeper singleton. This is a **hard constraint**.

```python
from src.gatekeeper import gatekeeper

# CORRECT: Using Gatekeeper
df = gatekeeper.fetch_stock_data("MSFT", "2020-01-01", "2023-12-31")

# WRONG: Direct yfinance calls are forbidden
# ticker = yf.Ticker("MSFT")  # NEVER DO THIS
```

**Key Responsibilities:**
- **Rate Limiting:** 2.0s minimum interval + randomized jitter (prevents API blocking)
- **Ticker Sanitization:** SHA-256 hashing of identifiers (prevents injection attacks)
- **Data Validation:** Flattens MultiIndex, enforces numeric types, drops NaN
- **Watchdog Monitoring:** `is_healthy()` checks heartbeat (60s timeout)

**Cache Behavior:** Saves data to `assets/<secure_id>_data.csv` to minimize API calls.

### 2. Model Architecture (Two Variants)

**Location:** `src/model.py`

#### DuelingDQN (Primary Model)
- **Feature Extractor:** `CNNExtractor` - 1D convolutions along temporal axis
  - Input: `[Batch, 4 Features, 30 Time Steps]`
  - Output: `[Batch, 64]` compressed features
- **Dueling Heads:**
  - **Value Stream:** Estimates state value V(s)
  - **Advantage Stream:** Estimates action advantage A(s, a)
  - **Q-value Combination:** `Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))`

**Why Dueling?** Decouples state quality from action quality. During market rallies, the Value stream recognizes favorable state (high V), while Advantage stream decides Hold vs. Sell. Standard DQN conflates these, causing premature exits.

#### DuelingTransformerDQN (Baseline Comparison)
- **Feature Extractor:** `TransformerExtractor` - Multi-head self-attention (4 heads, 2 layers)
- **Usage:** For benchmarking against CNN. Train with `src/train_transformer.py`, compare with `src/compare_models.py`

### 3. Data Pipeline & Environment

**Location:** `src/datasets.py`

**Flow:**
1. `TradingDataset` fetches via Gatekeeper → adds RSI, MACD → normalizes (Min-Max)
2. `get_train_test_split()` splits 80/20 by default
3. `TradingEnv` wraps data for RL interaction

**Key Design:**
- **30-Day Rolling Window:** Each state is `[4 features × 30 timesteps]`
- **Features:** Close, Volume, RSI, MACD
- **Actions:** 0=Hold, 1=Buy, 2=Sell
- **Reward:** Portfolio value change ratio (encourages asymmetric returns)

**Input Shape Convention:**
```python
# After dataset.get_windows():
windows.shape  # [Samples, Features=4, Time=30]

# Model expects:
state.shape    # [Batch, 4, 30]
```

### 4. Training Pipeline

**Location:** `src/train.py` (DQN) or `src/train_transformer.py` (Transformer)

**Critical Components:**
- **Experience Replay:** `ReplayMemory` with 10,000 capacity
- **Target Network:** Updated every 10 episodes (`TARGET_UPDATE`)
- **Epsilon Decay:** 0.995 per episode (1.0 → 0.0067 over 1000 episodes)
- **Gradient Clipping:** `max_norm=1.0` prevents explosion
- **LR Scheduler:** StepLR (gamma=0.9, step_size=100)
- **Checkpointing:** Saves every 100 episodes to `assets/checkpoint_ep*.pth`

**Hyperparameters (src/config.py):**
```python
LR = 3e-4              # Learning rate
GAMMA = 0.99           # Discount factor
EPS_DECAY = 0.995      # Slower decay for 1000 episodes
NUM_EPISODES = 1000    # Total training episodes
WINDOW_SIZE = 30       # 30-day lookback
```

### 5. Advanced Features

#### Sentiment Analysis (src/sentiment_analyzer.py)
- **SentimentGatekeeper:** Separate Gatekeeper for news APIs
- **Features:** sentiment_score, sentiment_trend, news_volume, confidence
- **Integration:** `integrate_sentiment_features(price_df, ticker)`
- **Note:** Currently simulates data; production requires API keys (Alpha Vantage, Finnhub)

#### Multi-Ticker Portfolio (src/portfolio_manager.py)
- **MultiTickerPortfolio:** Manages diversified portfolios
- **Position Sizing:** Max 30% per ticker (configurable)
- **Risk Management:** Transaction fees, portfolio-level tracking
- **Backtest:** `portfolio.backtest(test_ratio=0.2)` runs multi-ticker simulation

### 6. Evaluation & Visualization

**Graph Generation:** `generate_evaluation_graph.py`
- Creates dual-panel plot: Stock prices + trading signals (upper), portfolio value (lower)
- Saves to `evaluation_graph.png` (root) and `assets/evaluation_graph.png`
- Calculates: Win Rate, Directional Accuracy, Confidence Level, ROI

**Dashboard:** `src/dashboard.py`
- Streamlit app on port 8501
- Real-time telemetry, learning curves, portfolio tracking

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies (CUDA-enabled PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Training
```bash
# Train DQN model (1000 episodes, saves checkpoints every 100)
python -m src.main --mode train --ticker MSFT

# Train Transformer baseline
python -m src.train_transformer --ticker MSFT

# Full pipeline (train + evaluate)
python -m src.main --mode full --ticker AAPL
```

### Evaluation & Analysis
```bash
# Generate evaluation graph with metrics
python generate_evaluation_graph.py

# Compare DQN vs Transformer
python -m src.compare_models --ticker MSFT

# Multi-ticker portfolio backtest
python -m src.portfolio_manager
```

### Dashboard
```bash
streamlit run src/dashboard.py
# Access at http://localhost:8501
```

### Docker Deployment
```bash
# Build CPU variant
docker build --target training -t trading-rl:cpu .

# Build GPU variant (requires NVIDIA GPU + CUDA 12.4)
docker build --target training-gpu -t trading-rl:gpu .

# Run full pipeline with docker-compose
docker-compose up training evaluation dashboard

# GPU training
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up training
```

## Critical Implementation Rules

### Security & API Usage
1. **NEVER call yfinance directly** - always use `gatekeeper.fetch_stock_data()`
2. **Ticker validation:** Gatekeeper enforces `^[A-Z]{1,5}$` regex
3. **Cache awareness:** Data cached to `assets/<secure_id>_data.csv` - delete to force refresh
4. **Rate limits:** Minimum 2.0s between API calls (enforced by Gatekeeper)

### Model Training & State
1. **Checkpointing:** Training saves state every 100 episodes - resume from checkpoint if interrupted
2. **Device placement:** Always use `DEVICE` from `src/config.py` (auto-detects CUDA)
3. **Epsilon during evaluation:** Set to 0.0 for greedy policy (no exploration)
4. **Target network sync:** Happens every 10 episodes (`TARGET_UPDATE`)

### Data Handling
1. **MultiIndex cleanup:** Gatekeeper automatically flattens `df.columns.get_level_values(0)`
2. **NaN handling:** Dropped after technical indicator calculation (RSI, MACD)
3. **Normalization:** Min-Max scaling applied to all features before windowing
4. **Original prices:** Denormalized for environment (portfolio calculations require real values)

### File Modifications
1. **Preserve interfaces:** All public class/method signatures are API contracts
2. **No refactoring:** Surgical fixes only - working code should not be restructured
3. **Config changes:** Hyperparameter tuning happens in `src/config.py` only
4. **Model architecture:** Changing `CNNExtractor` or `DuelingDQN` requires retraining from scratch

## Performance Benchmarks

**Current Optimized Results (MSFT 2015-2023):**
- Win Rate: 41.35%
- Directional Accuracy: 52.34% (above random baseline)
- Confidence Level: 0.6247 (optimal zone)
- ROI: 134.97% (portfolio grew $10K → $23.5K)

**Training Time:**
- 1000 episodes on CUDA (RTX 3080): ~2-3 hours
- 1000 episodes on CPU: ~8-12 hours

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/config.py` | Central hyperparameter configuration |
| `src/gatekeeper.py` | Security layer for all external APIs |
| `src/model.py` | DQN and Transformer model architectures |
| `src/datasets.py` | Data loading, preprocessing, RL environment |
| `src/train.py` | DQN training loop with checkpointing |
| `src/train_transformer.py` | Transformer training pipeline |
| `src/evaluate.py` | Model evaluation on test set |
| `src/compare_models.py` | DQN vs Transformer benchmarking |
| `src/sentiment_analyzer.py` | News sentiment analysis module |
| `src/portfolio_manager.py` | Multi-ticker portfolio management |
| `src/dashboard.py` | Streamlit visualization dashboard |
| `src/main.py` | CLI entry point (train/evaluate/full modes) |
| `generate_evaluation_graph.py` | Dual-panel visualization generator |
| `Dockerfile` | Multi-stage builds (8 targets: CPU/GPU variants) |
| `docker-compose.yml` | 6-service orchestration (training, dashboard, etc.) |

## Assets Directory Structure

```
assets/
├── trading_model.pth           # Trained DQN model
├── transformer_model.pth       # Trained Transformer model
├── checkpoint_ep*.pth          # Training checkpoints (every 100 episodes)
├── evaluation_graph.png        # Dual-panel evaluation plot
├── model_comparison.png        # DQN vs Transformer comparison
└── logs/
    ├── rewards.npy             # Episode rewards history (DQN)
    ├── losses.npy              # Training losses (DQN)
    ├── transformer_rewards.npy # Transformer episode rewards
    ├── transformer_losses.npy  # Transformer losses
    ├── eval_results.csv        # Evaluation metrics (Win Rate, ROI, etc.)
    ├── model_comparison.csv    # Comparison results
    └── portfolio_backtest.csv  # Multi-ticker backtest results
```

## Troubleshooting

**"Model not found" error:**
- Train model first: `python -m src.main --mode train --ticker MSFT`
- Or use pre-trained: Ensure `assets/trading_model.pth` exists

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in `src/config.py` (default: 64)
- Or train on CPU: `DEVICE = torch.device("cpu")` in config

**API rate limit errors:**
- Gatekeeper enforces 2.0s delay - increase `min_interval` in `gatekeeper.py` if needed
- Delete cache files in `assets/` to force fresh data fetch

**MultiIndex DataFrame errors:**
- Gatekeeper auto-flattens, but if manual processing: `df.columns = df.columns.get_level_values(0)`

**Epsilon too high during evaluation:**
- Evaluation should use greedy policy (epsilon=0.0)
- Check `select_action()` call uses `epsilon=0.0` not `EPS_START`

## Documentation Files

- **README.md**: Comprehensive project documentation (691 lines)
- **TODO.md**: Feature completion status and roadmap
- **PRD.md**: Product requirements and success metrics
- **IMPLEMENTATION_SUMMARY.md**: Detailed implementation notes
- **This file (CLAUDE.md)**: Developer guidance for Claude Code
