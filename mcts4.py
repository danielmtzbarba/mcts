import tkinter as tk
from src.games.connect_four import ConnectFourGUI


if __name__ == "__main__":
    root = tk.Tk()
    gui = ConnectFourGUI(root)
    root.mainloop()
