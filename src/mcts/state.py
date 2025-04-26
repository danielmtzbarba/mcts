from abc import ABC, abstractmethod


class GameState(ABC):
    """Abstract base class representing the state of a game or decision process."""

    @abstractmethod
    def get_legal_actions(self):
        """Return a list of all possible actions from this state."""
        pass

    @abstractmethod
    def perform_action(self, action):
        """Return the new state after applying the given action."""
        pass

    @abstractmethod
    def is_terminal(self):
        """Return True if the state is a terminal state."""
        pass

    @abstractmethod
    def get_reward(self):
        """Return the reward for this state (used for backpropagation)."""
        pass

    @abstractmethod
    def copy(self):
        """Return a deep copy of the game state."""
        pass
