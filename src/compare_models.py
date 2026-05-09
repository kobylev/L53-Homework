import torch
import numpy as np
import pandas as pd
import logging
import os
import sys
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DuelingDQN, DuelingTransformerDQN, select_action
from src.datasets import TradingDataset, TradingEnv, get_train_test_split
from src.config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, env, model_name="Model"):
    """Evaluate a trained model on the environment"""
    state = env.reset()
    done = False
    total_reward = 0
    actions_taken = []
    portfolio_values = []

    while not done:
        action = select_action(model, state.to(DEVICE), epsilon=0.0)  # Greedy evaluation
        actions_taken.append(action)
        next_state, reward, done, portfolio_value = env.step(action)
        portfolio_values.append(portfolio_value)
        state = next_state
        total_reward += reward

    # Calculate metrics
    final_portfolio = portfolio_values[-1]
    initial_portfolio = env.initial_balance
    roi = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100

    actions_array = np.array(actions_taken)
    buy_ratio = (actions_array == 1).sum() / len(actions_array) * 100
    sell_ratio = (actions_array == 2).sum() / len(actions_array) * 100
    hold_ratio = (actions_array == 0).sum() / len(actions_array) * 100

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Evaluation Results:")
    logger.info(f"{'='*60}")
    logger.info(f"Total Reward: {total_reward:.4f}")
    logger.info(f"Initial Portfolio: ${initial_portfolio:.2f}")
    logger.info(f"Final Portfolio: ${final_portfolio:.2f}")
    logger.info(f"ROI: {roi:.2f}%")
    logger.info(f"Buy Actions: {buy_ratio:.2f}%")
    logger.info(f"Sell Actions: {sell_ratio:.2f}%")
    logger.info(f"Hold Actions: {hold_ratio:.2f}%")
    logger.info(f"{'='*60}\n")

    return {
        'model_name': model_name,
        'total_reward': total_reward,
        'initial_portfolio': initial_portfolio,
        'final_portfolio': final_portfolio,
        'roi': roi,
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'hold_ratio': hold_ratio,
        'portfolio_values': portfolio_values,
        'actions': actions_taken
    }


def compare_models(ticker=None):
    """Compare DQN and Transformer models"""
    logger.info("Loading dataset for model comparison...")
    ticker = ticker or TICKER
    dataset = TradingDataset(ticker=ticker)
    _, test_data, _ = get_train_test_split(dataset)
    original_prices = dataset.data['Close'].values

    # Adjust slice for test data
    split_idx = int(len(dataset.data) * 0.8)
    test_prices = original_prices[split_idx:]

    env = TradingEnv(test_data, test_prices)

    results = []

    # Evaluate DQN model
    dqn_model_path = os.path.join(ASSETS_DIR, "trading_model.pth")
    if os.path.exists(dqn_model_path):
        logger.info("Loading DQN model...")
        dqn_model = DuelingDQN().to(DEVICE)
        dqn_model.load_state_dict(torch.load(dqn_model_path, map_location=DEVICE))
        dqn_model.eval()
        dqn_results = evaluate_model(dqn_model, env, "Dueling DQN (1D CNN)")
        results.append(dqn_results)
    else:
        logger.warning(f"DQN model not found at {dqn_model_path}")

    # Reset environment for fair comparison
    env.reset()

    # Evaluate Transformer model
    transformer_model_path = os.path.join(ASSETS_DIR, "transformer_model.pth")
    if os.path.exists(transformer_model_path):
        logger.info("Loading Transformer model...")
        transformer_model = DuelingTransformerDQN().to(DEVICE)
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=DEVICE))
        transformer_model.eval()
        transformer_results = evaluate_model(transformer_model, env, "Dueling Transformer DQN")
        results.append(transformer_results)
    else:
        logger.warning(f"Transformer model not found at {transformer_model_path}")

    if len(results) < 2:
        logger.error("Need both models trained for comparison. Run train.py and train_transformer.py first.")
        return

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Total Reward': f"{r['total_reward']:.4f}",
            'ROI (%)': f"{r['roi']:.2f}",
            'Final Portfolio ($)': f"{r['final_portfolio']:.2f}",
            'Buy (%)': f"{r['buy_ratio']:.2f}",
            'Sell (%)': f"{r['sell_ratio']:.2f}",
            'Hold (%)': f"{r['hold_ratio']:.2f}"
        }
        for r in results
    ])

    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    logger.info("="*80 + "\n")

    # Save comparison results
    comparison_path = os.path.join(LOGS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Comparison saved to {comparison_path}")

    # Plot portfolio value comparison
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r['portfolio_values'], label=r['model_name'], linewidth=2)

    plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', label='Initial Balance')
    plt.xlabel('Trading Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Comparison: DQN vs Transformer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_plot_path = os.path.join(ASSETS_DIR, "model_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300)
    logger.info(f"Comparison plot saved to {comparison_plot_path}")
    plt.close()

    # Determine winner
    winner = max(results, key=lambda x: x['roi'])
    logger.info(f"\n🏆 WINNER: {winner['model_name']} with {winner['roi']:.2f}% ROI\n")

    return results, comparison_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare DQN and Transformer trading models")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol")
    args = parser.parse_args()

    compare_models(ticker=args.ticker)
