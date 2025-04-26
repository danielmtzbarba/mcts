import math
import numpy as np
import random


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.legal_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.41):
        """Choose the best child using UCB1 formula."""
        choices_weights = [
            (child.total_reward / child.visits)
            + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        legal_actions = self.state.get_legal_actions()
        tried_actions = [child.action for child in self.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        if not untried_actions:
            raise Exception("No untried actions left — should not call expand() here")
        #
        action = random.choice(untried_actions)
        #
        child_state = self.state.copy().perform_action(action)
        child_node = Node(child_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        """Backpropagate reward to this node and its ancestors."""
        self.visits += 1
        self.total_reward += reward


class MCTS:
    def __init__(self, time_limit=None, iteration_limit=1000):
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit

    def search(self, initial_state, debug=True):
        root = Node(initial_state)

        for _ in range(self.iteration_limit):
            node = root

            # Selection
            while not node.state.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            reward = self.rollout(node.state, depth=5)

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        if debug:
            self._debug(root)

        # Return the action corresponding to the best child
        best_child = root.best_child(c_param=0)  # Exploitation only

        return best_child.action

    def evaluate_state(self, state) -> float:
        """
        Very simple heuristic: center column control + win condition
        """
        if state.winner == state.current_player:
            return 1.0
        elif state.winner == 3 - state.current_player:
            return 0.0
        elif state.winner is None and state.is_terminal():
            return 0.5

        score = 0
        # Example: reward center column
        center_col = np.array(state.board)[:, 3]
        score += list(center_col).count(state.current_player) * 0.1

        return 0.5 + score  # adjust toward win

    def rollout(self, state, depth) -> float:
        # Only play 1-2 random moves, then evaluate
        current = state.copy()
        rollout_depth = depth

        for _ in range(rollout_depth):
            if current.is_terminal():
                break
            legal = current.get_legal_actions()
            action = random.choice(legal)
            current.perform_action(action)

        return self.evaluate_state(current)

    def _debug(self, root):
        print("MCTS Action Stats:")
        for child in root.children:
            win_rate = child.total_reward / child.visits
            print(
                f" - Move {child.action}: visits={child.visits}, win_rate={win_rate:.2f}"
            )

        best_child = root.best_child(c_param=0)  # Exploitation only
        print(" - Best Action:", best_child.action)
