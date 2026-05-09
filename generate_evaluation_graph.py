import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import logging

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import DuelingDQN
from src.datasets import TradingDataset, TradingEnv, get_train_test_split
from src.config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(actions, rewards, q_values_list, portfolio_values):
    """Calculate Win Rate, Confidence Level, and other metrics"""
    # Win Rate: Percentage of steps with positive portfolio growth
    portfolio_changes = np.diff(portfolio_values)
    wins = (portfolio_changes > 0).sum()
    win_rate = (wins / len(portfolio_changes)) * 100 if len(portfolio_changes) > 0 else 0

    # Confidence Level: Average softmax probability of selected actions
    confidences = []
    for q_values, action in zip(q_values_list, actions):
        softmax_probs = torch.softmax(q_values, dim=0)
        confidence = softmax_probs[action].item()
        confidences.append(confidence)
    avg_confidence = np.mean(confidences) if confidences else 0

    # Directional Accuracy: How often model correctly predicts price movement
    # We infer: Buy (1) predicts up, Sell (2) predicts down, Hold (0) is neutral
    correct_predictions = 0
    total_predictions = 0

    for i in range(len(portfolio_changes)):
        action = actions[i]
        price_change = portfolio_changes[i]

        if action == 1 and price_change > 0:  # Buy before price increase
            correct_predictions += 1
        elif action == 2 and price_change < 0:  # Sell before price decrease
            correct_predictions += 1
        elif action == 0:  # Hold is always "correct" (neutral)
            continue

        if action != 0:  # Only count Buy/Sell for directional accuracy
            total_predictions += 1

    directional_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

    return {
        'win_rate': win_rate,
        'confidence_level': avg_confidence,
        'directional_accuracy': directional_accuracy,
        'total_wins': wins,
        'total_steps': len(portfolio_changes)
    }


def generate_evaluation_graph():
    """Generate comprehensive evaluation graph with stock prices and trading signals"""
    logger.info("Loading dataset for evaluation...")

    # Load dataset
    dataset = TradingDataset(ticker=TICKER)
    train_data, test_data, _ = get_train_test_split(dataset)

    # Get original prices
    original_prices = dataset.data['Close'].values * (dataset.max_vals['Close'] - dataset.min_vals['Close']) + dataset.min_vals['Close']

    # Split prices for test set
    split_idx = int(len(dataset.data) * 0.8)
    test_prices = original_prices[split_idx:]

    # Create environment
    env = TradingEnv(test_data, test_prices)

    # Load trained model
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return

    logger.info(f"Loading model from {model_path}...")
    model = DuelingDQN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Run evaluation
    logger.info("Running evaluation on test set...")
    state = env.reset()
    done = False

    actions_taken = []
    portfolio_values = []
    rewards_collected = []
    q_values_list = []
    prices_at_steps = []

    step = 0
    while not done:
        # Get Q-values for confidence calculation
        with torch.no_grad():
            q_values = model(state.unsqueeze(0).to(DEVICE)).squeeze()

        # Select action (greedy for evaluation)
        action = q_values.argmax().item()

        # Get current price
        current_price = env.prices[env.current_step + WINDOW_SIZE - 1]

        # Store data
        actions_taken.append(action)
        q_values_list.append(q_values.cpu())
        prices_at_steps.append(current_price)

        # Take action
        next_state, reward, done, portfolio_value = env.step(action)

        portfolio_values.append(portfolio_value)
        rewards_collected.append(reward)

        state = next_state
        step += 1

    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_metrics(actions_taken, rewards_collected, q_values_list, portfolio_values)

    # Log metrics
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATION METRICS")
    logger.info(f"{'='*70}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}% ({metrics['total_wins']}/{metrics['total_steps']} profitable steps)")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    logger.info(f"Confidence Level: {metrics['confidence_level']:.4f}")
    logger.info(f"Initial Portfolio: ${INITIAL_BALANCE:.2f}")
    logger.info(f"Final Portfolio: ${portfolio_values[-1]:.2f}")
    logger.info(f"Total Return: ${portfolio_values[-1] - INITIAL_BALANCE:.2f}")
    logger.info(f"ROI: {((portfolio_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%")
    logger.info(f"{'='*70}\n")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'Ticker': TICKER,
        'Win Rate (%)': f"{metrics['win_rate']:.2f}",
        'Directional Accuracy (%)': f"{metrics['directional_accuracy']:.2f}",
        'Confidence Level': f"{metrics['confidence_level']:.4f}",
        'Initial Portfolio ($)': f"{INITIAL_BALANCE:.2f}",
        'Final Portfolio ($)': f"{portfolio_values[-1]:.2f}",
        'ROI (%)': f"{((portfolio_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}"
    }])
    metrics_df.to_csv(os.path.join(LOGS_DIR, 'eval_results.csv'), index=False)
    logger.info(f"Metrics saved to {os.path.join(LOGS_DIR, 'eval_results.csv')}")

    # Generate the graph
    logger.info("Generating evaluation graph...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Stock price with trading signals
    steps = np.arange(len(prices_at_steps))
    ax1.plot(steps, prices_at_steps, linewidth=2, color='#2C3E50', label='Actual Stock Price', alpha=0.8)

    # Overlay actions
    buy_steps = [i for i, a in enumerate(actions_taken) if a == 1]
    sell_steps = [i for i, a in enumerate(actions_taken) if a == 2]
    hold_steps = [i for i, a in enumerate(actions_taken) if a == 0]

    if buy_steps:
        ax1.scatter([steps[i] for i in buy_steps],
                   [prices_at_steps[i] for i in buy_steps],
                   marker='^', s=100, color='#27AE60', label='Buy Signal',
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)

    if sell_steps:
        ax1.scatter([steps[i] for i in sell_steps],
                   [prices_at_steps[i] for i in sell_steps],
                   marker='v', s=100, color='#E74C3C', label='Sell Signal',
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)

    if hold_steps:
        ax1.scatter([steps[i] for i in hold_steps],
                   [prices_at_steps[i] for i in hold_steps],
                   marker='o', s=30, color='#95A5A6', label='Hold',
                   alpha=0.3, zorder=3)

    ax1.set_xlabel('Trading Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{TICKER} - Trading Policy Evaluation (Dueling DQN)\nWin Rate: {metrics["win_rate"]:.2f}% | Confidence: {metrics["confidence_level"]:.4f} | ROI: {((portfolio_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Portfolio value over time
    ax2.plot(steps, portfolio_values, linewidth=2.5, color='#3498DB', label='Portfolio Value')
    ax2.axhline(y=INITIAL_BALANCE, color='#E67E22', linestyle='--', linewidth=2, label='Initial Balance')
    ax2.fill_between(steps, INITIAL_BALANCE, portfolio_values,
                     where=np.array(portfolio_values) >= INITIAL_BALANCE,
                     color='#27AE60', alpha=0.3, label='Profit')
    ax2.fill_between(steps, INITIAL_BALANCE, portfolio_values,
                     where=np.array(portfolio_values) < INITIAL_BALANCE,
                     color='#E74C3C', alpha=0.3, label='Loss')

    ax2.set_xlabel('Trading Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Portfolio Performance', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save to root directory
    output_path = 'evaluation_graph.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation graph saved to {output_path}")
    plt.close()

    # Also save to assets
    assets_path = os.path.join(ASSETS_DIR, 'evaluation_graph.png')
    plt.figure(figsize=(16, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Recreate the same plot
    ax1.plot(steps, prices_at_steps, linewidth=2, color='#2C3E50', label='Actual Stock Price', alpha=0.8)
    if buy_steps:
        ax1.scatter([steps[i] for i in buy_steps], [prices_at_steps[i] for i in buy_steps],
                   marker='^', s=100, color='#27AE60', label='Buy Signal',
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)
    if sell_steps:
        ax1.scatter([steps[i] for i in sell_steps], [prices_at_steps[i] for i in sell_steps],
                   marker='v', s=100, color='#E74C3C', label='Sell Signal',
                   edgecolors='black', linewidths=0.5, zorder=5, alpha=0.8)
    if hold_steps:
        ax1.scatter([steps[i] for i in hold_steps], [prices_at_steps[i] for i in hold_steps],
                   marker='o', s=30, color='#95A5A6', label='Hold', alpha=0.3, zorder=3)

    ax1.set_xlabel('Trading Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{TICKER} - Trading Policy Evaluation (Dueling DQN)\nWin Rate: {metrics["win_rate"]:.2f}% | Confidence: {metrics["confidence_level"]:.4f} | ROI: {((portfolio_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(steps, portfolio_values, linewidth=2.5, color='#3498DB', label='Portfolio Value')
    ax2.axhline(y=INITIAL_BALANCE, color='#E67E22', linestyle='--', linewidth=2, label='Initial Balance')
    ax2.fill_between(steps, INITIAL_BALANCE, portfolio_values,
                     where=np.array(portfolio_values) >= INITIAL_BALANCE,
                     color='#27AE60', alpha=0.3, label='Profit')
    ax2.fill_between(steps, INITIAL_BALANCE, portfolio_values,
                     where=np.array(portfolio_values) < INITIAL_BALANCE,
                     color='#E74C3C', alpha=0.3, label='Loss')

    ax2.set_xlabel('Trading Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Portfolio Performance', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(assets_path, dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation graph also saved to {assets_path}")
    plt.close()

    return metrics, actions_taken, portfolio_values


if __name__ == "__main__":
    generate_evaluation_graph()
