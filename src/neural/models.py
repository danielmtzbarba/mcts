# muzero_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                2, 32, kernel_size=3, padding=1
            ),  # input: 2 planes (player 1, player 2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, obs):
        """
        obs: tensor of shape (batch_size, 2, 6, 7)
        Returns: tensor (batch_size, 64, 6, 7)
        """
        return self.conv_layers(obs)


class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Action will be one-hot encoded (size 7)
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 7 + 7, 512), nn.ReLU(), nn.Linear(512, 64 * 6 * 7)
        )

    def forward(self, hidden_state, action):
        """
        hidden_state: tensor (batch_size, 64, 6, 7)
        action: tensor (batch_size, 7) one-hot action vector
        """
        batch_size = hidden_state.size(0)
        hidden_flat = hidden_state.view(batch_size, -1)
        x = torch.cat([hidden_flat, action], dim=1)
        next_hidden = self.fc(x)
        return next_hidden.view(batch_size, 64, 6, 7)


class PredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = nn.Linear(64 * 6 * 7, 7)  # 7 columns to choose from
        self.value_head = nn.Linear(
            64 * 6 * 7, 1
        )  # single value output (win probability)

    def forward(self, hidden_state):
        """
        hidden_state: tensor (batch_size, 64, 6, 7)
        Returns:
            policy_logits: (batch_size, 7)
            value: (batch_size, 1)
        """
        batch_size = hidden_state.size(0)
        hidden_flat = hidden_state.view(batch_size, -1)

        policy_logits = self.policy_head(hidden_flat)
        value = self.value_head(hidden_flat)

        return policy_logits, value
