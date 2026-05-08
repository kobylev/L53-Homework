import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class CNNExtractor(nn.Module):
    def __init__(self, in_channels=4):
        super(CNNExtractor, self).__init__()
        # 1D CNN along the temporal axis (30 days)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [Batch, 64, 1]
        return x.view(x.size(0), -1) # [Batch, 64]

class DuelingDQN(nn.Module):
    def __init__(self, n_actions=3, use_target_network=True):
        super(DuelingDQN, self).__init__()
        self.feature_extractor = CNNExtractor()

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Target network (frozen copy for stable Q-value bootstrapping)
        if use_target_network:
            self.target_net = deepcopy(self)
            self.target_net.eval()  # Set to eval mode
            for param in self.target_net.parameters():
                param.requires_grad = False
        else:
            self.target_net = None

    def forward(self, x):
        features = self.feature_extractor(x)
        advantage = self.advantage(features)
        value = self.value(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def sync_target(self):
        """
        Hard update: Copy online network weights to target network.
        Use this periodically (e.g., every N episodes) for traditional DQN.
        """
        if self.target_net is not None:
            self.target_net.load_state_dict(self.state_dict())

    def soft_update_target(self, tau=0.005):
        """
        Soft update (Polyak averaging): Gradually update target network.
        target_param = tau * online_param + (1 - tau) * target_param

        Args:
            tau: Interpolation coefficient (0 < tau << 1, typically 0.001-0.01)

        Use this after every optimizer step for smoother Q-value targets.
        """
        if self.target_net is not None:
            for target_param, online_param in zip(self.target_net.parameters(), self.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    # TRAINING LOOP INTEGRATION GUIDE:
    # ================================
    # 1. Initialize model with target network:
    #    model = DuelingDQN(n_actions=3, use_target_network=True)
    #
    # 2. During training, compute target Q-values using target_net:
    #    with torch.no_grad():
    #        next_q_values = model.target_net(next_states).max(1)[0]
    #        target_q = rewards + GAMMA * next_q_values * (1 - dones)
    #
    # 3. Compute online Q-values and loss:
    #    current_q = model(states).gather(1, actions.unsqueeze(1))
    #    loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
    #
    # 4. After optimizer step, update target network:
    #    Option A (soft update - recommended, use every step):
    #        model.soft_update_target(tau=0.005)
    #
    #    Option B (hard update - use every N episodes, e.g., N=10):
    #        if episode % TARGET_UPDATE == 0:
    #            model.sync_target()

class TransformerExtractor(nn.Module):
    """Transformer-based feature extractor for temporal patterns"""
    def __init__(self, in_channels=4, d_model=64, nhead=4, num_layers=2, seq_len=30):
        super(TransformerExtractor, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [Batch, Channels, Time] -> [Batch, Time, Channels]
        x = x.permute(0, 2, 1)

        # Project to d_model
        x = self.input_projection(x)  # [Batch, Time, d_model]

        # Add positional encoding
        x = x + self.pos_encoder.unsqueeze(0)

        # Apply transformer
        x = self.transformer(x)  # [Batch, Time, d_model]

        # Pool across time dimension
        x = x.permute(0, 2, 1)  # [Batch, d_model, Time]
        x = self.pool(x)  # [Batch, d_model, 1]

        return x.view(x.size(0), -1)  # [Batch, d_model]


class TransformerDQN(nn.Module):
    """Transformer-based DQN for trading decisions"""
    def __init__(self, n_actions=3):
        super(TransformerDQN, self).__init__()
        self.feature_extractor = TransformerExtractor()

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values


class DuelingTransformerDQN(nn.Module):
    """Dueling architecture with Transformer backbone for improved stability"""
    def __init__(self, n_actions=3):
        super(DuelingTransformerDQN, self).__init__()
        self.feature_extractor = TransformerExtractor()

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        advantage = self.advantage(features)
        value = self.value(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

def select_action(model, state, epsilon, n_actions=3):
    if torch.rand(1).item() < epsilon:
        return torch.randint(0, n_actions, (1,)).item()
    else:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0))
            return q_values.argmax(dim=1).item()
