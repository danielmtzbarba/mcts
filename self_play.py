# self_play.py

import torch
import numpy as np

from src.mcts.mucts import MuZeroMCTS

def run_self_play(env, mcts, network, num_episodes):
    """
    Run multiple self-play episodes using MCTS.
    """
    all_episodes = []

    for episode_idx in range(num_episodes):
        obs = env.reset()
        done = False
        episode = []

        while not done:
            obs_tensor = torch.from_numpy(obs).float()  # Assuming obs = (C, H, W)
            root = mcts.run(obs_tensor)

            # Get improved policy from visit counts
            visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(mcts.action_space_size)])
            policy = visit_counts / np.sum(visit_counts)

            # Sample action proportional to visit counts
            action = np.random.choice(mcts.action_space_size, p=policy)

            # Environment step
            next_obs, reward, done, info = env.step(action)

            # Store (obs, policy, reward)
            episode.append((obs, policy, reward))

            obs = next_obs

        all_episodes.append(episode)

    return all_episodes
