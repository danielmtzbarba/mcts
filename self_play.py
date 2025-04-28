# self_play.py

import torch
import numpy as np

from src.mcts.mucts import MuZeroMCTS


def self_play(env, network, num_simulations):
    memory = []

    obs = env.reset()
    done = False

    while not done:
        # 1. Preprocess observation
        obs_tensor = preprocess_obs(obs).unsqueeze(0)  # (batch_size=1)

        # 2. Initial inference (representation)
        hidden_state, policy_logits, value = network.initial_inference(obs_tensor)

        # 3. Run MCTS
        action, search_stats = run_mcts(hidden_state, network, num_simulations)

        # 4. Environment step
        next_obs, reward, done, info = env.step(action)

        # 5. Save data
        memory.append(
            {
                "observation": obs,
                "action": action,
                "reward": reward,
                "policy": search_stats["policy"],  # Final MCTS visit distribution
                "value": value.item(),
            }
        )

        obs = next_obs

    return memory
