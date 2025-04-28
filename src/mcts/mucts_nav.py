import numpy as np
import torch
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, hidden_state, prior):
        self.hidden_state = hidden_state  # Tensor representing state
        self.prior = prior  # Prior probability from policy network
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

    def run(self, observation):
        """
        Main function: Runs MCTS starting from an observation
        """

        with torch.no_grad():
            # Encode observation into hidden state
            hidden_state = self.representation_net(
                observation.unsqueeze(0)
            )  # (1, C, H, W)
            policy_logits, _ = self.prediction_net(hidden_state)

        policy = F.softmax(policy_logits, dim=1)[0]  # (action_space_size,)

        # Create root node
        root = MCTSNode(hidden_state, prior=1.0)  # Root has dummy prior (ignored)

        # Expand root with initial policy
        for action in range(self.action_space_size):
            root.children[action] = MCTSNode(
                hidden_state=hidden_state, prior=policy[action].item()
            )

        # Run simulations
        for _ in range(self.num_simulations):
            self.search(root)

        return root

    def search(self, node):
        """
        Simulate one path through the tree
        """
        path = [node]
        current_node = node

        # Selection: Traverse the tree
        while current_node.children:
            action, current_node = self.select_action(current_node)
            path.append(current_node)

        # Expansion: Predict new policy and value
        with torch.no_grad():
            policy_logits, value = self.prediction_net(current_node.hidden_state)

        policy = F.softmax(policy_logits, dim=1)[0]

        # Expand node
        for action in range(self.action_space_size):
            current_node.children[action] = MCTSNode(
                hidden_state=current_node.hidden_state,  # Will update after dynamics
                prior=policy[action].item(),
            )

        leaf_value = value.item()

        # Backpropagation
        self.backup(path, leaf_value)

    def select_action(self, node):
        """
        Pick the child with highest UCB score and simulate dynamics
        """
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            ucb = self.ucb_score(node, child, total_visits)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        # After picking the best child, simulate dynamics to next hidden state
        action_onehot = (
            F.one_hot(torch.tensor(best_action), num_classes=self.action_space_size)
            .float()
            .unsqueeze(0)
        )  # (1, action_space_size)

        with torch.no_grad():
            next_hidden = self.dynamics_net(node.hidden_state, action_onehot)

        best_child.hidden_state = next_hidden

        return best_action, best_child

    def ucb_score(self, parent, child, total_visits):
        """
        Standard MuZero UCB formula
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
            node.value_sum += value
            node.visit_count += 1
