# train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from tqdm import tqdm
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from src.neural.muzero import MuZeroAgent
from src.mcts.mucts_nav import MuZeroMCTS
from CarlaBEV.envs import CarlaBEV  # Your gymnasium environment

from src.mcts.self_play import self_play

log_dir = os.path.join("runs", "muzero_nav")
writer = SummaryWriter(log_dir=log_dir)


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity

    def add(self, episode):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train_network(
    model, optimizer, batch, num_unroll_steps, global_step=0, writer=None
):
    model.train()
    losses = []
    for episode in batch:
        obs_list, action_list, reward_list, policy_list = zip(*episode)
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32).cuda()
        hidden_state, pred_policy, pred_value = model.initial_inference(obs_tensor)
        loss = F.mse_loss(
            pred_value.squeeze(), torch.tensor(reward_list, dtype=torch.float32).cuda()
        )
        losses.append(loss)

    total_loss = sum(losses)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Console logging
    print(f"[Train] Step {global_step}: Loss = {total_loss.item():.4f}")

    # TensorBoard logging
    if writer:
        writer.add_scalar("Loss/total", total_loss.item(), global_step)


# --- Self-Play + Training Loop ---
def train(num_episodes=1000):
    # --- Initialize ---
    env = CarlaBEV(size=128, discrete=True)
    network = MuZeroAgent(128, action_space_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=action_space_size,
        num_simulations=num_simulations,
    )
    global_step, ep = 1, 0
    # Training loop
    for it in range(1, num_episodes + 1):
        print(f"\n--- Training Iteration {it}/{num_episodes} ---")
        # Self-play
        for ep in range(num_self_play_episodes):
            episode_data, total_reward = self_play(env, mcts)
            replay_buffer.add(episode_data)
            # Console logging

            print(
                f"[Self-Play] Episode {global_step}: Total Reward = {total_reward:.2f}"
            )

            # TensorBoard logging
            if writer:
                writer.add_scalar("Reward/episode", total_reward, global_step)

            global_step += 1

        # Train if enough data
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            train_network(
                network,
                optimizer,
                batch,
                num_unroll_steps,
                global_step=global_step,
                writer=writer,
            )

        # Save checkpoint periodically
        if (it) % 100 == 0:
            torch.save(network.state_dict(), f"muzero_nav_checkpoint_ep{it}.pth")
            print(f"Checkpoint saved at episode {it}")

    print("Training completed!")


# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_self_play_episodes = 20
num_simulations = 50
num_unroll_steps = 20
batch_size = 4
buffer_size = 10000
learning_rate = 1e-3

action_space_size = 5

if __name__ == "__main__":
    train()
