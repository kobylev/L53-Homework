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
from src.config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(ticker=None):
    logger.info("Starting evaluation...")
    ticker = ticker or TICKER
    dataset = TradingDataset(ticker=ticker)
    _, test_data = get_train_test_split(dataset)
    
    # Need full prices for the test period
    full_prices = dataset.data['Close'].values * (dataset.max_vals['Close'] - dataset.min_vals['Close']) + dataset.min_vals['Close']
    split_idx = int(len(dataset.get_windows()) * 0.8)
    test_prices = full_prices[split_idx : split_idx + len(test_data) + WINDOW_SIZE]
    
    model = DuelingDQN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        logger.error("No model found to evaluate.")
        return

    env = TradingEnv(test_data, test_prices)
    state = env.reset().to(DEVICE)
    
    actions = []
    portfolio_values = []
    rewards = []
    confidence_levels = []
    
    done = False
    while not done:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0))
            action = q_values.argmax(dim=1).item()
            
            # Confidence Level = Softmax probability of chosen action
            probs = torch.softmax(q_values, dim=1)
            confidence = probs.max().item()
            confidence_levels.append(confidence)

        next_state, reward, done, portfolio_value = env.step(action)
        state = next_state.to(DEVICE)
        
        actions.append(action)
        portfolio_values.append(portfolio_value)
        rewards.append(reward)

    # Calculate Win Rate (% of steps with positive reward)
    win_rate = (np.array(rewards) > 0).mean() * 100
    
    # Directional Accuracy calculation
    # Define directional target: 1 if price went up, 0 if down/flat
    actual_deltas = np.diff(test_prices[WINDOW_SIZE-1:])
    actual_direction = (actual_deltas > 0).astype(int)
    
    # Model's 'implied' direction: Buy (1) implies Up, Sell (2) implies Down, Hold (0) is neutral
    # To calculate 'Accuracy', we compare if Buy preceded an Up move or Sell preceded a Down move
    predicted_actions = np.array(actions[:-1]) # exclude last action as we don't have next price
    correct_directional_predictions = 0
    total_active_predictions = 0
    
    for i in range(len(predicted_actions)):
        if predicted_actions[i] == 1: # Predicted Up
            total_active_predictions += 1
            if actual_direction[i] == 1:
                correct_directional_predictions += 1
        elif predicted_actions[i] == 2: # Predicted Down
            total_active_predictions += 1
            if actual_direction[i] == 0:
                correct_directional_predictions += 1
                
    directional_accuracy = (correct_directional_predictions / total_active_predictions * 100) if total_active_predictions > 0 else 0
    avg_confidence = np.mean(confidence_levels)
    
    logger.info(f"Evaluation Complete | Win Rate: {win_rate:.2f}% | Directional Accuracy: {directional_accuracy:.2f}% | Avg Confidence: {avg_confidence:.4f} | Final Portfolio: ${portfolio_values[-1]:.2f}")
    
    # Save results for Dashboard
    results = pd.DataFrame({
        'Date': dataset.data.index[split_idx + WINDOW_SIZE - 1 : split_idx + WINDOW_SIZE - 1 + len(actions)],
        'Price': test_prices[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(actions)],
        'Action': actions,
        'PortfolioValue': portfolio_values,
        'Confidence': confidence_levels
    })
    results.to_csv(os.path.join(LOGS_DIR, "eval_results.csv"), index=False)
    
    # Save accuracy for README/Analysis
    with open(os.path.join(LOGS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"Final Portfolio: ${portfolio_values[-1]:.2f}\n")
    
    # Generate Plots
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['PortfolioValue'], label='Portfolio Value')
    plt.title(f"Portfolio Value Over Time ({TICKER})")
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, "portfolio_value.png"))
    plt.close()

    return results, win_rate, avg_confidence

if __name__ == "__main__":
    evaluate()
