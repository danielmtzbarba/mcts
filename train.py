# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import numpy as np
from tqdm import tqdm

from src.neural.muzero import MuZeroAgent   # (combines representation, dynamics, prediction)
from src.mcts.mucts_nav import MuZeroMCTS
from CarlaBEV.envs import CarlaBEV          # Your gymnasium environment
from src.utils import rgb_to_semantic_mask  # transforms RGB to semantic mask

from self_play import run_self_play 


class ReplayBuffer:
    def __init__(self, max_size):
        self.capacity = max_size
        self.buffer = []

    def add_episode(self, episode):
        self.buffer.append(episode)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size, num_unroll_steps):
        batch = random.sample(self.buffer, batch_size)

        observations = []
        actions = []
        rewards = []
        policies = []

        for episode in batch:
            # Pick a random starting point
            if len(episode) <= num_unroll_steps:
                idx = 0
            else:
                idx = random.randint(0, len(episode) - num_unroll_steps - 1)

            obs, action, reward, policy  = zip(*episode[idx : idx + num_unroll_steps + 1])

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            policies.append(policy)

        return observations, actions, rewards, policies

    def __len__(self):
        return len(self.buffer)






# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_self_play_episodes = 1000
num_simulations = 50
num_unroll_steps = 10
batch_size = 64
buffer_size = 50000
learning_rate = 1e-3
discount = 0.997
train_after_steps = 1000
train_every_n_steps = 10

action_space_size = 5

# --- Initialize ---
env = CarlaBEV(size=128, discrete=True)
network = MuZeroAgent(128, action_space_size).to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(max_size=buffer_size)
mcts = MuZeroMCTS(
    network=network,
    action_space_size=action_space_size,
    num_simulations=num_simulations
)

# --- Self-Play + Training Loop ---
def train():
    global_step = 0

    for episode in range(num_self_play_episodes):
        print(f"Starting Episode {episode}")

        obs, _ = env.reset()
        obs = rgb_to_semantic_mask(obs)
        done = False
        episode_data = []

        # Self-Play
        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, channels, height, width)
            legal_actions = list(range(action_space_size))  # assume all actions are legal
            current_player = 1  # single-agent for now
            root = mcts.run(obs_tensor)

            # Get improved policy from visit counts
            visit_counts = np.array([
                root.children[a].visit_count if a in root.children else 0 
                for a in range(action_space_size)
            ])
            policy = visit_counts / np.sum(visit_counts)

            action = np.random.choice(action_space_size, p=policy)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_processed = rgb_to_semantic_mask(next_obs)

            # Store
            episode_data.append((obs, action, reward, policy))
            obs = next_obs_processed
            global_step += 1

        # Push episode to buffer
        replay_buffer.add_episode(episode_data)
        
        if len(replay_buffer) > batch_size:
            # --- Train ---
            if global_step > train_after_steps and episode % train_every_n_steps == 0:
                print("Training...")
                for _ in range(5):  # multiple gradient steps per episode
                    batch = replay_buffer.sample_batch(batch_size, num_unroll_steps)

                    loss = 0

                    # Each item is (observation, action, reward, policy)
                    for obs_batch, action_batch, reward_batch, policy_batch in batch:
                        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).unsqueeze(0).to(device)

                        # Initial Inference
                        hidden_state = network.representation(obs_batch)
                        pred_policy_logits, pred_value = network.prediction(hidden_state)

                        # Policy Loss
                        target_policy = torch.tensor(policy_batch, dtype=torch.float32).unsqueeze(0).to(device)
                        policy_loss = nn.functional.cross_entropy(pred_policy_logits, target_policy)

                        # Value Loss
                        target_value = torch.tensor([reward_batch], dtype=torch.float32).to(device)
                        value_loss = nn.functional.mse_loss(pred_value.squeeze(), target_value.squeeze())

                        total_loss = policy_loss + value_loss
                        loss += total_loss

                    loss = loss / len(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Episode {episode}, Loss: {loss.item():.4f}")

            # --- Save Model ---
            if episode % 50 == 0:
                torch.save(network.state_dict(), f"muzero_checkpoint_{episode}.pth")
                print(f"Model checkpoint saved at episode {episode}")

    print("Training completed!")

if __name__ == "__main__":
    train()

