import torch
import numpy as np

from src.utils import rgb_to_semantic_mask


def self_play(env, mcts):
    obs, _ = env.reset()
    obs = rgb_to_semantic_mask(obs)
    done = False
    episode = []
    total_reward = 0

    while not done:
        observation = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).cuda()
        root = mcts.run(observation)
        visit_counts = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(mcts.action_space_size)
            ]
        )
        policy = visit_counts / visit_counts.sum()
        action = np.random.choice(mcts.action_space_size, p=policy)
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        episode.append((obs, action, reward, policy))
        obs = rgb_to_semantic_mask(next_obs)

    return episode, total_reward
