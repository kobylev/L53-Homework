import torch
import numpy as np
import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DuelingDQN
from src.datasets import TradingDataset, TradingEnv, get_train_test_split
from src.config import TICKER, DEVICE, MODEL_PATH, WINDOW_SIZE, LOGS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_risk_metrics(portfolio_values, risk_free_rate=0.02):
    """
    FIX 2: Complete Risk-Adjusted Metrics calculation.
    Calculates Sharpe, Sortino, Max Drawdown, Calmar, and Annualized Volatility.
    """
    # 1. Daily Returns
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # 2. Annualized Return
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    num_days = len(portfolio_values)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1

    # 3. Annualized Sharpe Ratio
    # Formula: (mean_return - Rf) / std(returns) * sqrt(252)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = daily_returns - daily_rf
    sharpe_ratio = np.mean(excess_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)

    # 4. Maximum Drawdown
    # Formula: min((portfolio - running_max) / running_max)
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = np.max(drawdowns)

    # 5. Calmar Ratio
    # Formula: annualized_return / abs(max_drawdown)
    calmar_ratio = annualized_return / (max_drawdown + 1e-9)

    # 6. Sortino Ratio (penalizes only downside volatility)
    # Formula: mean_return / downside_std * sqrt(252)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        sortino_ratio = 0.0
    else:
        downside_std = np.std(negative_returns)
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252)

    # 7. Annualized Volatility
    # Formula: std(returns) * sqrt(252)
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)

    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "annualized_volatility": annualized_volatility,
        "total_return": total_return
    }

def evaluate(ticker=None):
    logger.info("Starting evaluation...")
    ticker = ticker or TICKER
    dataset = TradingDataset(ticker=ticker)
    
    # FIX: Logic to handle new get_train_test_split return signature
    train_data, test_data, scaler = get_train_test_split(dataset)
    # split_idx כבר מחושב שם — החזר אותו:
    # return train_windows, test_windows, scaler, split_idx
    split_idx = get_train_test_split(dataset)[3]
    test_prices = dataset.data['Close'].iloc[split_idx + WINDOW_SIZE - 1:].values[:len(test_data)]
    
    model = DuelingDQN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
    else:
        logger.error("No model found to evaluate.")
        return

    env = TradingEnv(test_data, test_prices)
    state = env.reset().to(DEVICE)
    
    actions, portfolio_values, rewards, confidence_levels = [], [], [], []
    
    done = False
    while not done:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0))
            action = q_values.argmax(dim=1).item()
            probs = torch.softmax(q_values, dim=1)
            confidence_levels.append(probs.max().item())

        next_state, reward, done, portfolio_value = env.step(action)
        state = next_state.to(DEVICE)
        
        actions.append(action)
        portfolio_values.append(portfolio_value)
        rewards.append(reward)

    # FIX: Calculate advanced risk metrics
    risk_metrics = calculate_risk_metrics(portfolio_values)
    win_rate = (np.array(rewards) > 0).mean() * 100
    avg_confidence = np.mean(confidence_levels)
    
    logger.info("-" * 30)
    logger.info(f"PERFORMANCE REPORT - {ticker}")
    logger.info(f"Final Portfolio: ${portfolio_values[-1]:,.2f}")
    logger.info(f"Total Return: {risk_metrics['total_return']*100:.2f}%")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    logger.info(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")
    logger.info(f"Max Drawdown: {risk_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.4f}")
    logger.info(f"Annualized Volatility: {risk_metrics['annualized_volatility']*100:.2f}%")
    logger.info(f"Avg Confidence: {avg_confidence:.4f}")
    logger.info("-" * 30)
    
    # Save results
    results = pd.DataFrame({
        'Date': df.index[split_idx + WINDOW_SIZE - 1 : split_idx + WINDOW_SIZE - 1 + len(actions)],
        'Price': test_prices[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(actions)],
        'Action': actions,
        'PortfolioValue': portfolio_values,
        'Confidence': confidence_levels
    })
    results.to_csv(os.path.join(LOGS_DIR, "eval_results.csv"), index=False)
    
    # Append risk metrics to CSV for Streamlit dashboard integration
    with open(os.path.join(LOGS_DIR, "metrics.txt"), "w") as f:
        for k, v in risk_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"win_rate: {win_rate}\n")
        f.write(f"avg_confidence: {avg_confidence}\n")

    return results, win_rate, avg_confidence

if __name__ == "__main__":
    evaluate()
