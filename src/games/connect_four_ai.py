# connect_four_gui.py

import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
from src.neural.models import RepresentationNetwork, PredictionNetwork
from src.mcts.mcts_play import MCTS


class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 0 empty, 1 player1, 2 player2
        self.current_player = 1
        self.done = False
        self.winner = 0

    def clone(self):
        """Create a deep copy of the game (important for MCTS simulations)"""
        cloned = ConnectFour()
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.done = self.done
        cloned.winner = self.winner
        return cloned

    def legal_actions(self):
        """Returns list of legal columns to play"""
        return [c for c in range(7) if self.board[0][c] == 0]

    def step(self, action):
        """Play a move. Update board and switch player."""
        if self.done:
            raise Exception("Game is over!")

        for row in reversed(range(6)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break

        self.check_winner()
        if not self.done:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2

    def check_winner(self):
        """Check if current player won"""
        for c in range(7 - 3):
            for r in range(6):
                if (
                    self.board[r, c]
                    == self.board[r, c + 1]
                    == self.board[r, c + 2]
                    == self.board[r, c + 3]
                    != 0
                ):
                    self.winner = self.board[r, c]
                    self.done = True
                    return

        for c in range(7):
            for r in range(6 - 3):
                if (
                    self.board[r, c]
                    == self.board[r + 1, c]
                    == self.board[r + 2, c]
                    == self.board[r + 3, c]
                    != 0
                ):
                    self.winner = self.board[r, c]
                    self.done = True
                    return

        for c in range(7 - 3):
            for r in range(6 - 3):
                if (
                    self.board[r, c]
                    == self.board[r + 1, c + 1]
                    == self.board[r + 2, c + 2]
                    == self.board[r + 3, c + 3]
                    != 0
                ):
                    self.winner = self.board[r, c]
                    self.done = True
                    return

        for c in range(7 - 3):
            for r in range(3, 6):
                if (
                    self.board[r, c]
                    == self.board[r - 1, c + 1]
                    == self.board[r - 2, c + 2]
                    == self.board[r - 3, c + 3]
                    != 0
                ):
                    self.winner = self.board[r, c]
                    self.done = True
                    return

        if np.all(self.board != 0):
            self.done = True
            self.winner = 0  # Draw

    def is_terminal(self):
        return self.done

    def get_observation(self):
        """
        Return board as 2 channels:
        [player1 pieces, player2 pieces]
        shape = (2, 6, 7)
        """
        obs = np.zeros((2, 6, 7), dtype=np.float32)
        obs[0] = self.board == 1
        obs[1] = self.board == 2
        return obs

    def get_winner(self):
        return self.winner

    def render(self):
        """Optional: for debugging"""
        print(self.board[::-1])


class ConnectFourGUI:
    def __init__(self, master, model_paths=None):
        self.master = master
        master.title("Connect Four vs MuZero AI")

        self.canvas = tk.Canvas(master, width=700, height=600, bg="blue")
        self.canvas.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.pack(side=tk.RIGHT)

        self.canvas.bind("<Button-1>", self.handle_click)

        self.game = ConnectFour()

        self.mcts = None
        # AI loading
        self.representation_net = None
        self.prediction_net = None
        if model_paths:
            self.load_model(model_paths)

        self.draw_board()

    def load_model(self, model_paths):
        rep_path, pred_path = model_paths

        self.representation_net = RepresentationNetwork()
        self.prediction_net = PredictionNetwork()

        self.representation_net.load_state_dict(
            torch.load(rep_path, map_location=torch.device("cpu"), weights_only=True)
        )
        self.prediction_net.load_state_dict(
            torch.load(pred_path, map_location=torch.device("cpu"), weights_only=True)
        )

        self.representation_net.eval()
        self.prediction_net.eval()

        self.mcts = MCTS(
            self.representation_net, self.prediction_net, num_simulations=50
        )

    def reset_game(self):
        self.game = ConnectFour()
        self.draw_board()

    def handle_click(self, event):
        if self.game.is_terminal():
            return

        col = event.x // (700 // 7)

        if col not in self.game.legal_actions():
            return

        self.game.step(col)
        self.draw_board()

        if self.game.is_terminal():
            self.show_winner()
            return

        self.master.after(500, self.ai_move)

    def ai_move(self):
        if self.mcts is None:
            return  # No AI loaded

        root = self.mcts.run(self.game)
        self.debug_mcts(root)
        #
        action_visits = [
            (action, child.visit_count) for action, child in root.children.items()
        ]
        best_action = max(action_visits, key=lambda x: x[1])[0]

        self.game.step(best_action)
        self.draw_board()

        if self.game.is_terminal():
            self.show_winner()

    def draw_board(self):
        self.canvas.delete("all")

        for c in range(7):
            for r in range(6):
                x_start = c * (700 // 7)
                y_start = r * (600 // 6)
                x_end = x_start + (700 // 7)
                y_end = y_start + (600 // 6)

                color = "white"
                if self.game.board[r][c] == 1:
                    color = "red"
                elif self.game.board[r][c] == 2:
                    color = "yellow"

                self.canvas.create_oval(
                    x_start + 5, y_start + 5, x_end - 5, y_end - 5, fill=color
                )

    def show_winner(self):
        winner = self.game.get_winner()

        if winner > 0:
            messagebox.showinfo("Game Over", f"Player {winner} wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")

    def debug_mcts(self, root):
        print("\n=== MCTS Analysis ===")
        for action, child in root.children.items():
            action_col = action  # (0-6 for Connect Four)
            win_rate = (child.value() + 1) / 2  # Normalize from [-1, 1] to [0, 1]
            print(
                f"Column {action_col}: {child.visit_count} visits, Win rate ~ {win_rate:.2%}"
            )
        print("======================\n")
