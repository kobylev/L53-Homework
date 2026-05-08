# Product Requirements Document (PRD) - RL Trading System

## 1. Executive Summary
**Project Name:** FinTech RL Trading Pipeline  
**Vision:** To build a robust, secure, and high-performance Reinforcement Learning system for algorithmic stock trading using 1D CNNs and Dueling DQN.  
**Objective:** Automate trading decisions (Buy, Sell, Hold) based on technical indicators while ensuring security and scalability through a dedicated Gatekeeper module.

## 2. Technical Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch (with CUDA 12 support)
- **Data Source:** yfinance API
- **Indicators:** RSI, MACD (via `ta` library)
- **Visualization:** Streamlit, Plotly
- **Security:** Gatekeeper Pattern (Rate-limiting & Sanitization)

## 3. Core Features
### 3.1 Data Pipeline & Security
- **Ticker Support:** US Stock symbols.
- **Gatekeeper:** Strict input sanitization and rate-limiting to prevent API abuse and injection attacks.
    - **Load Balancing:** Throttling and management of outgoing requests to respect free-tier limits.
    - **Cyber Security:** Automatic renaming of external identifiers to secure, internal hashes (SHA-256) to prevent injection and data leakage.
    - **Data Sanitization:** Intercepts and validates all incoming API dataframes, ensuring strict numeric types and schema compliance.
- **Reliability:** Background **Watchdog** process to monitor the health and responsiveness of the Gatekeeper module.
- **Preprocessing:** 30-day rolling windows with Min-Max normalization.

### 3.2 Model Architecture
- **Base:** 1D CNN for temporal feature extraction across individual channels.
- **RL Head:** Dueling DQN to decouple State Value (V) from Action Advantage (A).
- **Optimization:** MSE Loss with Adam optimizer and Experience Replay.

### 3.3 Dashboard & Telemetry
- **Historical Analysis:** Candlestick-style charts with trading signal overlays.
- **System Monitoring:** Real-time CPU/GPU/Memory telemetry.
- **Performance:** Win Rate (%), Confidence Level, and Learning Curves.

## 4. Non-Functional Requirements
- **Performance:** GPU acceleration must be utilized if available.
- **Maintainability:** Modular structure adhering to `AI_EXPERT_COURSE` standards.
- **Security:** No raw API calls allowed outside the Gatekeeper.

## 5. Success Metrics
- **Win Rate:** > 50% (Long-term target).
- **Convergence:** Reward curve should show a positive trend over 1000+ episodes.
- **Stability:** Final Portfolio Value > Initial Balance on historical test sets.

## 6. Implementation Status & Recent Updates

### Phase 1: Core Infrastructure ✅ (Completed)
- Initial project structure with modular design
- Gatekeeper security module with rate-limiting and sanitization
- 1D CNN + Dueling DQN implementation
- GPU acceleration support (CUDA 12)
- Streamlit dashboard with real-time telemetry

### Phase 2: Optimization & Tuning ✅ (Completed - May 2026)
The following optimizations were implemented to improve model performance:

**Hyperparameter Tuning:**
- Learning Rate: 1e-4 → 3e-4 (improved convergence)
- Epsilon Decay: 0.98 → 0.995 (better exploration-exploitation balance)
- Training Episodes: 100 → 1000 (10x increase for convergence)
- Historical Data: 2 years → 8 years (2015-2023)

**Training Enhancements:**
- Gradient clipping (max_norm=1.0) for stability
- Learning rate scheduler (StepLR: gamma=0.9, step_size=100)
- Periodic checkpointing every 100 episodes
- Improved logging with learning rate tracking

**Expected Impact:**
- Win Rate: 35.96% → >50%
- Directional Accuracy: 48.21% → >55%
- Portfolio ROI: 7.58% → >15%

### Phase 3: Future Enhancements 🔴 (Planned)
- **Transformer Baseline:** Direct performance comparison with attention-based models
- **Sentiment Analysis:** NLP-based news sentiment integration via LLM
- **Portfolio Management:** Multi-ticker support for diversification
- **Live Trading:** Paper trading mode with Alpaca/Interactive Brokers API
- **Dockerization:** Container-based deployment for production
