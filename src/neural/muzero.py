import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroAgent(nn.Module):
    def __init__(self, hidden_dim=128, action_space_size=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size

        # --- Representation network ---
        self.representation_net = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # --- Dynamics network ---
        self.dynamics_net= nn.Sequential(
            nn.Linear(hidden_dim + action_space_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden_dim, 1)

        # --- Prediction network ---
        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def representation(self, observation):
        """
        observation: (batch, 6, H, W)
        returns: (batch, hidden_dim)
        """
        hidden = self.representation_net(observation)           # (batch, hidden_dim, H, W)
        hidden = hidden.mean(dim=[2, 3])                         # Global average pooling -> (batch, hidden_dim)
        return hidden

    def dynamics(self, hidden_state, action_one_hot):
        """
        hidden_state: (batch, hidden_dim)
        action_one_hot: (batch, action_space_size)
        returns: next_hidden_state (batch, hidden_dim), reward (batch, 1)
        """
        x = torch.cat([hidden_state, action_one_hot], dim=1)     # (batch, hidden_dim + action_space_size)
        next_hidden = self.dynamics_net(x)                        # (batch, hidden_dim)
        reward = self.reward_head(next_hidden)                   # (batch, 1)
        return next_hidden, reward

    def prediction(self, hidden_state):
        """
        hidden_state: (batch, hidden_dim)
        returns: policy_logits (batch, action_space_size), value (batch, 1)
        """
        policy_logits = self.policy_head(hidden_state)           # (batch, action_space_size)
        value = self.value_head(hidden_state)                    # (batch, 1)
        return policy_logits, value

    def initial_inference(self, observation):
        """
        The first step: from observation to initial hidden state, policy and value
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action_one_hot):
        """
        After the initial step: from hidden state and action to next hidden, policy, value, reward
        """
        next_hidden_state, reward = self.dynamics(hidden_state, action_one_hot)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, policy_logits, value, reward
