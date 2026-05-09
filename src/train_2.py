import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DuelingDQN, select_action
from src.datasets import TradingDataset, TradingEnv, get_train_test_split
from src.config import (
    DEVICE, LR, GAMMA, EPS_START, EPS_END, EPS_DECAY,
    BATCH_SIZE, REPLAY_MEMORY_SIZE, TARGET_UPDATE,
    NUM_EPISODES, WINDOW_SIZE, INITIAL_BALANCE,
    TICKER, START_DATE, END_DATE, ASSETS_DIR, LOGS_DIR, MODEL_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log"))
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.stack(next_states).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  Evaluation Metrics
# ─────────────────────────────────────────────
def compute_metrics(portfolio_values, actions, prices, initial_balance=INITIAL_BALANCE, risk_free=0.0):
    """Compute Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, ROI."""
    pv = np.array(portfolio_values)
    returns = np.diff(pv) / (pv[:-1] + 1e-9)

    roi = (pv[-1] - initial_balance) / initial_balance * 100

    # Win Rate: steps where return > 0
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0.0

    # Sharpe Ratio (annualised, 252 trading days)
    excess = returns - risk_free / 252
    sharpe = (np.mean(excess) / (np.std(excess) + 1e-9)) * np.sqrt(252)

    # Sortino Ratio (penalise only downside)
    downside = returns[returns < 0]
    sortino_std = np.std(downside) if len(downside) > 0 else 1e-9
    sortino = (np.mean(excess) / (sortino_std + 1e-9)) * np.sqrt(252)

    # Maximum Drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / (peak + 1e-9)
    max_drawdown = drawdown.min() * 100

    # Calmar Ratio
    annualised_roi = roi * (252 / len(pv))
    calmar = annualised_roi / (abs(max_drawdown) + 1e-9)

    # Action distribution
    acts = np.array(actions)
    n = len(acts) + 1e-9
    hold_pct  = (acts == 0).sum() / n * 100
    buy_pct   = (acts == 1).sum() / n * 100
    sell_pct  = (acts == 2).sum() / n * 100

    return {
        "ROI_%":           round(roi, 4),
        "Win_Rate_%":      round(win_rate, 4),
        "Sharpe_Ratio":    round(sharpe, 4),
        "Sortino_Ratio":   round(sortino, 4),
        "Max_Drawdown_%":  round(max_drawdown, 4),
        "Calmar_Ratio":    round(calmar, 4),
        "Final_Portfolio": round(float(pv[-1]), 2),
        "Hold_%":          round(hold_pct, 2),
        "Buy_%":           round(buy_pct, 2),
        "Sell_%":          round(sell_pct, 2),
    }


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────
def train(ticker=TICKER, start=START_DATE, end=END_DATE, episodes=NUM_EPISODES, eval_only=False):
    logger.info(f"=== L53 Dueling DQN Training | {ticker} | {start} → {end} | Device: {DEVICE} ===")

    # ── Data ──────────────────────────────────
    dataset = TradingDataset(ticker=ticker, start=start, end=end)
    train_windows, test_windows, scaler = get_train_test_split(dataset, split_ratio=0.8)
    prices = dataset.data['Close'].values

    split_idx = int(len(prices) * 0.8)
    train_prices = prices[:split_idx + WINDOW_SIZE]
    test_prices  = prices[split_idx:]

    train_env = TradingEnv(train_windows, train_prices)
    test_env  = TradingEnv(test_windows,  test_prices)
    logger.info(f"Train steps: {len(train_windows)} | Test steps: {len(test_windows)}")

    # ── Model ─────────────────────────────────
    model = DuelingDQN(n_actions=3, use_target_network=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    buffer = ReplayBuffer()

    if eval_only and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"Loaded model from {MODEL_PATH} — skipping training.")
        _evaluate(model, test_env, test_prices, ticker)
        return

    # ── Epsilon schedule ──────────────────────
    epsilon = EPS_START
    episode_rewards = []
    best_reward = -np.inf

    for episode in range(1, episodes + 1):
        state = train_env.reset()
        ep_reward = 0.0
        loss_val = 0.0
        steps = 0

        while not train_env.done:
            state_t = state.to(DEVICE)
            action = select_action(model, state_t, epsilon)

            next_state, reward, done, _ = train_env.step(action)
            buffer.push(state_t.cpu(), action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps += 1

            # ── Learn ─────────────────────────
            if len(buffer) >= BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(BATCH_SIZE)

                with torch.no_grad():
                    # Double DQN: online selects action, target evaluates
                    next_actions = model(next_states_b).argmax(dim=1)
                    next_q = model.target_net(next_states_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_b + GAMMA * next_q * (1 - dones_b)

                current_q = model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.soft_update_target(tau=0.005)
                loss_val = loss.item()

        # ── Epsilon decay ─────────────────────
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(ep_reward)

        # ── Hard target sync ──────────────────
        if episode % TARGET_UPDATE == 0:
            model.sync_target()

        # ── Logging ───────────────────────────
        if episode % 50 == 0 or episode == 1:
            avg_r = np.mean(episode_rewards[-50:])
            logger.info(
                f"Ep {episode:>4}/{episodes} | "
                f"ε={epsilon:.3f} | "
                f"Avg Reward (50): {avg_r:>8.4f} | "
                f"Loss: {loss_val:.5f} | "
                f"Buffer: {len(buffer)}"
            )

        # ── Save best ─────────────────────────
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(model.state_dict(), MODEL_PATH)

    logger.info(f"Training complete. Best episode reward: {best_reward:.4f}")
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save reward curve
    np.save(os.path.join(LOGS_DIR, "rewards.npy"), np.array(episode_rewards))

    # ── Evaluate on test set ───────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _evaluate(model, test_env, test_prices, ticker)


# ─────────────────────────────────────────────
#  Evaluation Pass
# ─────────────────────────────────────────────
def _evaluate(model, env, prices, ticker):
    model.eval()
    state = env.reset()
    portfolio_values = [INITIAL_BALANCE]
    actions_taken = []
    confidences = []
    price_list = []
    date_list = []

    step_idx = 0
    while not env.done:
        state_t = state.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = model(state_t)
            probs = torch.softmax(q_values, dim=1)
            action = q_values.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()

        next_state, reward, done, portfolio_val = env.step(action)
        portfolio_values.append(float(portfolio_val))
        actions_taken.append(action)
        confidences.append(round(confidence, 4))

        price_idx = step_idx + WINDOW_SIZE
        price_list.append(float(prices[price_idx]) if price_idx < len(prices) else float(prices[-1]))
        date_list.append(step_idx)
        step_idx += 1
        state = next_state

    metrics = compute_metrics(portfolio_values, actions_taken, price_list)

    # ── Print metrics ─────────────────────────
    logger.info("=" * 55)
    logger.info(f"  EVALUATION RESULTS — {ticker}")
    logger.info("=" * 55)
    for k, v in metrics.items():
        logger.info(f"  {k:<22}: {v}")
    logger.info("=" * 55)

    # ── Save eval_results.csv ─────────────────
    eval_path = os.path.join(LOGS_DIR, "eval_results.csv")
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Price", "Action", "PortfolioValue", "Confidence"])
        writer.writeheader()
        for i in range(len(actions_taken)):
            writer.writerow({
                "Date":           date_list[i],
                "Price":          price_list[i],
                "Action":         actions_taken[i],
                "PortfolioValue": portfolio_values[i + 1],
                "Confidence":     confidences[i],
            })
    logger.info(f"Saved eval results → {eval_path}")

    # ── Save metrics.json ─────────────────────
    metrics_path = os.path.join(LOGS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics      → {metrics_path}")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L53 Dueling DQN Trader — Training & Evaluation")
    parser.add_argument("--ticker",   type=str,   default=TICKER,       help="Stock ticker symbol")
    parser.add_argument("--start",    type=str,   default=START_DATE,   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      type=str,   default=END_DATE,     help="End date YYYY-MM-DD")
    parser.add_argument("--episodes", type=int,   default=NUM_EPISODES, help="Number of training episodes")
    parser.add_argument("--eval",     action="store_true",              help="Skip training, run eval only")
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        episodes=args.episodes,
        eval_only=args.eval,
    )
