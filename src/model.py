import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, n_actions=3):
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
        
    def forward(self, x):
        features = self.feature_extractor(x)
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

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
