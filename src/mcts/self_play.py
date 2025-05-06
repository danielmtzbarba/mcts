import torch
import numpy as np

def self_play(env, mcts):
    next_obs, _ = env.reset()
    b, f, c, h, w = next_obs.shape
    done = False
    episode = []
    total_reward = 0
    while not done:
        obs = torch.tensor(next_obs, dtype=torch.float32).cuda()
        obs = obs.view((b, -1, h, w))  # Combine f and c into a single channel dimension
        
        root = mcts.run(obs)
        visit_counts = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(mcts.action_space_size)
            ]
        )
        policy = visit_counts / visit_counts.sum()
        action = np.random.choice(mcts.action_space_size, p=policy)
        next_obs, reward, done, _, info = env.step([action])
        total_reward += reward
        obs = obs.cpu()
       # action = torch.tensor(action, dtype=torch.long)
        episode.append((obs, action, reward, policy))
    return episode, info 
