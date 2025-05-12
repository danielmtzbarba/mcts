import numpy as np
import torch
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, hidden_state, prior, c_puct=1.0):
        self.hidden_state = hidden_state  # Tensor representing state
        self.prior = prior  # Prior probability from policy network
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode
        self.c_puct = c_puct

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MuZeroMCTS:
    def __init__(self, network, action_space_size, num_simulations=50):
        self.muzero = network
        self.dynamics_net = network.dynamics_net
        self.prediction_net = network.policy_head

        self.action_space_size = action_space_size
        self.num_simulations = num_simulations

    def run(self, observation, isTraining):
        """
        Main function: Runs MCTS starting from an observation
        """
        with torch.no_grad():
            hidden_state, policy_logits, _ = self.muzero.initial_inference(observation)

        policy = F.softmax(policy_logits, dim=1)[0]  # (action_space_size,)

        # Apply Dirichlet noise to encourage exploration (only at root during training)
        if isTraining:
            epsilon = 0.25  # Mixing factor
            alpha = 0.03  # Dirichlet concentration
            dirichlet_noise = np.random.dirichlet([alpha] * self.action_space_size)
            noise_tensor = torch.tensor(
                dirichlet_noise, dtype=policy.dtype, device=policy.device
            )
            policy = (1 - epsilon) * policy + epsilon * noise_tensor

        # Create root node
        root = MCTSNode(hidden_state[0], prior=1.0)

        # Expand root with (noisy) policy
        for action in range(self.action_space_size):
            with torch.no_grad():
                next_hidden, _, _, _ = self.muzero.recurrent_inference(
                    hidden_state, torch.tensor([action]).to(hidden_state.device)
                )
                root.children[action] = MCTSNode(
                    hidden_state=next_hidden, prior=policy[action].item()
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
            next_hidden, policy_logits, value, reward = self.muzero.recurrent_inference(
                current_node.hidden_state,
                torch.tensor([action]).to(current_node.hidden_state.device),
            )

        policy = F.softmax(policy_logits, dim=1)[0]
        current_node.children[action] = MCTSNode(
            hidden_state=next_hidden,
            prior=policy[action].item(),
        )

        # Expand node
        for action in range(self.action_space_size):
            current_node.children[action] = MCTSNode(
                hidden_state=current_node.hidden_state[0],  # Will update after dynamics
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
        best_action = (
            torch.tensor(best_action, dtype=torch.long).unsqueeze(0).to("cuda:0")
        )  # (1,)
        if node.hidden_state.dim() == 1:
            node.hidden_state = node.hidden_state.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            next_hidden, _, _, _ = self.muzero.recurrent_inference(
                node.hidden_state, best_action
            )

        best_child.hidden_state = next_hidden

        return best_action.item(), best_child

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
        discount = 0.99  # Example discount factor
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value *= discount
