import torch
import numpy as np


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, representation_net, prediction_net, num_simulations=50):
        self.representation_net = representation_net
        self.prediction_net = prediction_net
        self.num_simulations = num_simulations

    def run(self, game):
        obs = torch.tensor(game.get_observation(), dtype=torch.float32).unsqueeze(0)
        hidden = self.representation_net(obs)
        policy_logits, _ = self.prediction_net(hidden)
        policy = torch.softmax(policy_logits, dim=1).squeeze().detach().numpy()

        root = Node(0)
        legal_actions = game.legal_actions()

        for action in legal_actions:
            root.children[action] = Node(prior=policy[action])

        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone()

            search_path = [node]

            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                scratch_game.step(action)
                search_path.append(node)

            # Expansion
            if not scratch_game.is_terminal():
                obs = torch.tensor(
                    scratch_game.get_observation(), dtype=torch.float32
                ).unsqueeze(0)
                hidden = self.representation_net(obs)
                policy_logits, value = self.prediction_net(hidden)
                policy = torch.softmax(policy_logits, dim=1).squeeze().detach().numpy()

                legal_actions = scratch_game.legal_actions()
                for action in legal_actions:
                    node.children[action] = Node(prior=policy[action])

            else:
                value = torch.tensor([scratch_game.get_winner()], dtype=torch.float32)

            # Backpropagate
            self.backpropagate(search_path, value.item())

        return root

    def select_child(self, node):
        C_PUCT = 1.5  # Exploration constant

        total_visits = sum(child.visit_count for child in node.children.values())

        def ucb_score(parent, child):
            prior_score = (
                C_PUCT
                * child.prior
                * (np.sqrt(parent.visit_count + 1e-8) / (1 + child.visit_count))
            )
            value_score = child.value()
            return value_score + prior_score

        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # alternate players
