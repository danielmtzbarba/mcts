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
    
    def average_reward(self, window=10):
        mean_rewards = 0
        last_episodes = self.buffer[-window:]
        for episode in last_episodes:
            mean_rewards += sum([reward for _, _, reward, _ in episode]) / len(episode) 
        return mean_rewards / len(last_episodes) 

    def __len__(self):
        return len(self.buffer)


def prepare_tensors(batch):
    obs_list, action_list, reward_list, policy_list = zip(*batch)
    obs_tensor = torch.stack([obs.cpu() for obs in obs_list]).squeeze(1).cuda()
    reward_tensor = torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in reward_list]).cuda()
    policy_tensor = torch.stack([torch.tensor(policy, dtype=torch.float32) for policy in policy_list]).cuda()
    action_tensor = torch.stack([torch.tensor(action, dtype=torch.long) for action in action_list]).cuda()

    # Ensure action_tensor has a batch dimension
    if action_tensor.dim() == 1:  # Single batch element
        action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension

    return obs_tensor, action_tensor, reward_tensor, policy_tensor

def train_network(
    model, optimizer, batch, num_unroll_steps=5, global_step=0, writer=None
):
    model.train()
    losses = []
    value_losses = []
    policy_losses = []
    reward_losses = []

    for episode in batch:
        obs_tensor, action_tensor, reward_tensor, policy_tensor = prepare_tensors(episode)
        root_obs = obs_tensor[0].unsqueeze(0)          
        # Unroll the episode
        hidden_state, pred_policy, pred_value = model.initial_inference(root_obs)
        total_loss = 0.0
        for step in range(num_unroll_steps):
            try:
                action = action_tensor[:, step] 
            except IndexError:
                break  # Break if action_tensor is shorter than expected
            
            with autocast(device_type="cuda"):
        
                hidden_state, policy_logits, pred_value, pred_reward = model.recurrent_inference(
                    hidden_state, action)
                # Calculate losses 
                reward_loss = F.mse_loss(pred_reward, reward_tensor[step].unsqueeze(0))
                value_loss = F.mse_loss(pred_value, reward_tensor[step].unsqueeze(0))
                pred_policy = F.softmax(policy_logits, dim=1) 
                policy_loss = -torch.sum(policy_tensor * torch.log(pred_policy + 1e-8), dim=1).mean()
                
                # Combine losses
                step_loss = (0.01 * policy_loss) + value_loss + reward_loss
                total_loss += step_loss.item()

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Append individual losses for logging
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        reward_losses.append(reward_loss.item())
        losses.append(total_loss)

    # Console logging
    print(f"[Train] Step {global_step}: Total Loss = {total_loss:.4f}")

    # TensorBoard logging
    if writer:
        writer.add_scalar("Loss/total", np.mean(losses), global_step)
        writer.add_scalar("Loss/value", np.mean(value_losses), global_step)
        writer.add_scalar("Loss/policy", np.mean(policy_losses), global_step)
        writer.add_scalar("Loss/reward", np.mean(reward_losses), global_step)


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
        c_puct=4.0,
    )
    global_step, ep = 1, 0
    # Training loop
    for it in range(1, num_episodes + 1):
        print(f"\n--- Training Iteration {it}/{num_episodes} ---")
        # Self-play
        for ep in range(num_self_play_episodes):
            episode_data, info = self_play(env, mcts)
            replay_buffer.add(episode_data)

            # Console logging
            print(
                f"[Self-Play] Episode {global_step}: Return = {info["ep"]["return"][0]:.2f}"
            )
                
            # TensorBoard logging
            if writer:
                writer.add_scalar("Stats/episode_return", info["termination"]["return"][0], global_step)
                writer.add_scalar("Stats/episode_length", info["termination"]["length"][0], global_step)
                writer.add_scalar("Stats/distance2target", info["env"]["dist2target_t"][0], global_step) 

                if global_step % 10 == 0:  # Log moving average every 10 iterations
                    writer.add_scalar("Stats/moving_avg_rwd", replay_buffer.average_reward(), global_step) 
                    

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
        if (it) % 50 == 0:
            torch.save(network.state_dict(), f"out/models/muzero_nav/{run_name}.pth")
            print(f"Checkpoint saved at episode {it}")

    print("Training completed!")


# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()  # Initialize the gradient scaler
#
run_name = "muzero_nav_unroll_5"
log_dir = os.path.join("runs", run_name)
writer = SummaryWriter(log_dir=log_dir)
# Hyperparameters
num_self_play_episodes = 25
num_simulations = 100
num_unroll_steps = 10
batch_size = 16 
buffer_size = 200
learning_rate = 1e-4
action_space_size = 5

if __name__ == "__main__":
    train(num_episodes=250)
