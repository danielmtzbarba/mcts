# self_play.py

import torch
import numpy as np

from src.mcts.mucts import MuZeroMCTS
from src.games.connect_four_ai import ConnectFour  # Your game environment


class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.policies = []
        self.rewards = []
        self.players = []

    def store_search_statistics(self, root):
        visit_counts = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(7)
            ]
        )
        policy = visit_counts / np.sum(visit_counts)
        self.policies.append(policy)

    def store(self, observation, action, reward, player):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.players.append(player)


def play_game(representation_net, dynamics_net, prediction_net, num_simulations=50):
    """
    Play a single self-play game and return its history
    """
    game = ConnectFour()
    history = GameHistory()

    mcts = MuZeroMCTS(
        representation_net=representation_net,
        dynamics_net=dynamics_net,
        prediction_net=prediction_net,
        action_space_size=7,
        num_simulations=num_simulations,
    )

    obs = torch.tensor(game.get_observation(), dtype=torch.float32)  # (2, 6, 7)

    while not game.is_terminal():
        legal_actions = game.legal_actions()

        root = mcts.run(obs, legal_actions, game.current_player)

        # Choose action based on visit counts (softmax temperature)
        visit_counts = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(7)
            ]
        )
        action = np.argmax(
            visit_counts
        )  # Greedy during self-play, can add randomness later

        history.store(
            obs, action, reward=0, player=game.current_player
        )  # reward will be updated later
        history.store_search_statistics(root)

        game.step(action)

        obs = torch.tensor(game.get_observation(), dtype=torch.float32)

    # Update rewards at the end
    winner = game.get_winner()
    for idx, player in enumerate(history.players):
        if winner == 0:
            reward = 0  # Draw
        elif player == winner:
            reward = 1
        else:
            reward = -1
        history.rewards[idx] = reward

    return history
