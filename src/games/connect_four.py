import copy
import tkinter as tk
from tkinter import messagebox

from src.mcts import GameState, MCTS

CELL_SIZE = 80
PADDING = 10
AI_THINKING_TIME = 500  # milliseconds to wait before AI moves (for visual smoothness)

ROWS = 6
COLS = 7
CONNECT = 4


class ConnectFourGame(GameState):
    def __init__(self, board=None, current_player=2):
        self.board = board if board is not None else [[0] * COLS for _ in range(ROWS)]
        self.current_player = current_player
        self.last_move = None
        self.winner = None

    def get_legal_actions(self):
        return [col for col in range(COLS) if self.board[0][col] == 0]

    def perform_action(self, action):
        for row in reversed(range(ROWS)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                self.last_move = (row, action)
                self.check_winner(row, action)
                self.current_player = 3 - self.current_player
                return self  # <-- Add this
        raise ValueError(f"Invalid action: column {action} is full")

    def is_terminal(self):
        return self.winner is not None or all(
            self.board[0][col] != 0 for col in range(COLS)
        )

    def get_reward(self):
        if self.winner == 1:
            return 1 if self.current_player == 2 else 0
        elif self.winner == 2:
            return 1 if self.current_player == 1 else 0
        else:
            return 0.5  # Draw

    def copy(self):
        return ConnectFourGame(copy.deepcopy(self.board), self.current_player)

    def check_winner(self, row, col):
        """Check if the current move causes a win."""
        player = self.board[row][col]

        def count_dir(dr, dc):
            r, c = row + dr, col + dc
            count = 0
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            return count

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            total = 1 + count_dir(dr, dc) + count_dir(-dr, -dc)
            if total >= CONNECT:
                self.winner = player
                return


class ConnectFourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four - Play vs AI")
        self.canvas = tk.Canvas(
            root, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE, bg="blue"
        )
        self.canvas.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_game)
        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.reset_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.quit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.handle_click)

        self.mcts = MCTS(iteration_limit=2000)
        self.game = ConnectFourGame()
        self.draw_board()

        # Let AI make the first move if AI is Player 1
        if self.game.current_player == 1:
            self.root.after(AI_THINKING_TIME, self.ai_move)

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0 = c * CELL_SIZE + PADDING
                y0 = r * CELL_SIZE + PADDING
                x1 = x0 + CELL_SIZE - 2 * PADDING
                y1 = y0 + CELL_SIZE - 2 * PADDING
                color = "white"
                if self.game.board[r][c] == 1:
                    color = "red"
                elif self.game.board[r][c] == 2:
                    color = "yellow"
                self.canvas.create_oval(x0, y0, x1, y1, fill=color)

    def handle_click(self, event):
        col = event.x // CELL_SIZE
        if self.game.is_terminal() or self.game.current_player != 2:
            return

        col = event.x // CELL_SIZE

        if col in self.game.get_legal_actions():
            self.game.perform_action(col)
            self.draw_board()

            if self.game.is_terminal():
                self.show_winner()
            else:
                self.root.after(AI_THINKING_TIME, self.ai_move)

    def ai_move(self):
        if self.game.current_player != 1 or self.game.is_terminal():
            return

        ai_action = self.mcts.search(self.game)
        self.game.perform_action(ai_action)
        self.draw_board()

        if self.game.is_terminal():
            self.show_winner()

    def show_winner(self):
        if self.game.winner:
            messagebox.showinfo("Game Over", f"Player {self.game.winner} wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")

    def reset_game(self):
        self.game = ConnectFourGame()
        self.draw_board()
