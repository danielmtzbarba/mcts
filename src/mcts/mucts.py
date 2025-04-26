# muzero_mcts.py

import torch
import torch.nn.functional as F
import numpy as np


class MCTSNode:
    def __init__(self, hidden_state, prior, player):
        self.hidden_state = hidden_state  # tensor
        self.prior = prior  # float (prior probability from policy net)
        self.player = player  # which player made this move

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MuZeroMCTS:
    def __init__(
        self,
        representation_net,
        dynamics_net,
        prediction_net,
        action_space_size,
        num_simulations=50,
        c_puct=1.0,
    ):
        self.representation_net = representation_net
        self.dynamics_net = dynamics_net
        self.prediction_net = prediction_net

        self.action_space_size = action_space_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, observation, legal_actions, current_player):
        """
        Main function: Runs MCTS starting from an observation
        """
        with torch.no_grad():
            hidden_state = self.representation_net(
                observation.unsqueeze(0)
            )  # (1, channels, 6, 7)
            policy_logits, value = self.prediction_net(hidden_state)

        policy = F.softmax(policy_logits, dim=1)[0]  # (7,)

        # Mask illegal actions
        mask = torch.zeros_like(policy)
        mask[legal_actions] = 1
        policy = policy * mask
        policy /= policy.sum()  # Renormalize

        root = MCTSNode(hidden_state, prior=0.0, player=current_player)

        # Expand root
        for action in legal_actions:
            root.children[action] = MCTSNode(
                hidden_state, prior=policy[action].item(), player=3 - current_player
            )  # Switch player

        # Search for best move
        for _ in range(self.num_simulations):
            self.search(root)

        return root

    def search(self, node):
        """
        Simulate one path in the tree
        """
        path = [node]
        current_node = node

        # Selection
        while current_node.children:
            action, current_node = self.select_action(current_node)
            path.append(current_node)

        # Expansion
        if current_node.visit_count == 0:
            # Predict policy and value from current hidden
            with torch.no_grad():
                policy_logits, value = self.prediction_net(current_node.hidden_state)

            policy = F.softmax(policy_logits, dim=1)[0]

            # Assume all actions are legal here (later you could predict legal moves too)
            for action in range(self.action_space_size):
                current_node.children[action] = MCTSNode(
                    hidden_state=current_node.hidden_state,  # same hidden for now, will simulate next
                    prior=policy[action].item(),
                    player=3 - current_node.player,
                )

            leaf_value = value.item()
        else:
            leaf_value = 0  # If already expanded, no rollout

        # Backpropagation
        self.backup(path, leaf_value)

    def select_action(self, node):
        """
        Pick the child with highest UCB score
        """
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child, total_visits)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        # Simulate dynamics
        action_onehot = (
            F.one_hot(torch.tensor(best_action), num_classes=self.action_space_size)
            .float()
            .unsqueeze(0)
        )  # (1, 7)
        with torch.no_grad():
            next_hidden = self.dynamics_net(node.hidden_state, action_onehot)

        best_child.hidden_state = next_hidden  # Update child's hidden state

        return best_action, best_child

    def ucb_score(self, parent, child, total_visits):
        """
        Standard UCB formula used in AlphaZero / MuZero
        """
        pb_c = np.log((total_visits + 19652) / 19652) + self.c_puct
        pb_c *= np.sqrt(total_visits) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value

        return prior_score + value_score

    def backup(self, path, value):
        """
        Backpropagate value up the path
        """
        for node in reversed(path):
            node.value_sum += value if node.player == path[-1].player else -value
            node.visit_count += 1
