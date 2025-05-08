import torch
from random import random


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
    reward_tensor = torch.stack(
        [torch.tensor(reward, dtype=torch.float32) for reward in reward_list]
    ).cuda()
    policy_tensor = torch.stack(
        [torch.tensor(policy, dtype=torch.float32) for policy in policy_list]
    ).cuda()
    action_tensor = torch.stack(
        [torch.tensor(action, dtype=torch.long) for action in action_list]
    ).cuda()

    # Ensure action_tensor has a batch dimension
    if action_tensor.dim() == 1:  # Single batch element
        action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension

    return obs_tensor, action_tensor, reward_tensor, policy_tensor
