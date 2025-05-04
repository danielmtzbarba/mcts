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
from src.games.carlabev import make_env 

from src.mcts.self_play import self_play

from torch.amp import GradScaler, autocast

log_dir = os.path.join("runs", "muzero_nav")
writer = SummaryWriter(log_dir=log_dir)

scaler = GradScaler()  # Initialize the gradient scaler

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
        obs_tensor = torch.stack([obs.cpu() for obs in obs_list]).squeeze(1).cuda()
        reward_tensor = torch.stack([torch.tensor(rew, dtype=torch.float32) for rew in reward_list]).squeeze(1).cuda()
        
        with autocast(device_type="cuda"):
            hidden_state, pred_policy, pred_value = model.initial_inference(obs_tensor)
            loss = F.mse_loss(pred_value.squeeze(), reward_tensor)
            losses.append(loss)

    total_loss = sum(losses)
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Console logging
    print(f"[Train] Step {global_step}: Loss = {total_loss.item():.4f}")

    # TensorBoard logging
    if writer:
        writer.add_scalar("Loss/total", total_loss.item(), global_step)


# --- Self-Play + Training Loop ---
def train(num_episodes=1000):
    # --- Initialize ---
    env = make_env(seed=0, capture_video=True, run_name="muzero_nav", size=128)
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

num_self_play_episodes = 32
num_simulations = 100
num_unroll_steps = 3
batch_size = 16
buffer_size = 500
learning_rate = 1e-3

action_space_size = 5

if __name__ == "__main__":
    obs_shape = (4, 6, 128, 128)  # Example: 4 stacked frames, 6 channels, 128x128 resolution
    obs_size = np.prod(obs_shape) * 4  # 4 bytes per float32
    action_size = 1 * 4  # 4 bytes per int32
    reward_size = 1 * 4  # 4 bytes per float32
    policy_size = action_space_size * 4  # 4 bytes per float32

    episode_memory = (obs_size + action_size + reward_size + policy_size) * num_unroll_steps
    replay_buffer_memory = episode_memory * buffer_size / (1024 ** 2)  # Convert to MB
    print(f"Replay buffer memory: {replay_buffer_memory:.2f} MB")

    batch_memory = episode_memory * batch_size / (1024 ** 2)  # Convert to MB
    print(f"Batch memory: {batch_memory:.2f} MB")
    # Start training
    train()
