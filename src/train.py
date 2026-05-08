import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import logging
import os
import sys
from collections import deque

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DuelingDQN, select_action
from src.datasets import TradingDataset, TradingEnv, get_train_test_split
from src.config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train(ticker=None):
    # Load data
    logger.info("Initializing dataset...")
    ticker = ticker or TICKER
    dataset = TradingDataset(ticker=ticker)
    train_data, _ = get_train_test_split(dataset)
    original_prices = dataset.data['Close'].values * (dataset.max_vals['Close'] - dataset.min_vals['Close']) + dataset.min_vals['Close']
    
    env = TradingEnv(train_data, original_prices[:len(train_data)+WINDOW_SIZE])
    
    # Initialize models
    policy_net = DuelingDQN().to(DEVICE)
    target_net = DuelingDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    # Learning rate scheduler for gradual decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    
    epsilon = EPS_START
    
    episode_rewards = []
    losses = []
    
    logger.info(f"Starting training for {NUM_EPISODES} episodes...")
    for episode in range(NUM_EPISODES):
        state = env.reset().to(DEVICE)
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(policy_net, state, epsilon)
            next_state, reward, done, portfolio_value = env.step(action)
            next_state = next_state.to(DEVICE)
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Perform optimization step
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.stack(states).to(DEVICE)
                actions = torch.tensor(actions).unsqueeze(1).to(DEVICE)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                next_states = torch.stack(next_states).to(DEVICE)
                dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
                
                # Q(s, a)
                current_q_values = policy_net(states).gather(1, actions)
                
                # max Q(s', a') from target net
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
                
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent explosion during long training
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())
        
        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Learning rate scheduling
        scheduler.step()

        # Periodic checkpointing every 100 episodes
        if (episode + 1) % 100 == 0:
            checkpoint_path = os.path.join(ASSETS_DIR, f"checkpoint_ep{episode+1}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'rewards': episode_rewards
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at episode {episode+1}")

        episode_rewards.append(total_reward)
        logger.info(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.4f} | Epsilon: {epsilon:.3f} | Portfolio: ${portfolio_value:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model
    torch.save(policy_net.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Save training metrics
    np.save(os.path.join(LOGS_DIR, "rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(LOGS_DIR, "losses.npy"), np.array(losses))
    
    return policy_net, episode_rewards, losses

if __name__ == "__main__":
    train()
