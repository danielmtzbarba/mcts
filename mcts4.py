import tkinter as tk
from src.games.connect_four_ai import ConnectFourGUI

model_paths = "out/models/representation_1000.pth", "out/models/prediction_1000.pth"

if __name__ == "__main__":
    root = tk.Tk()
    gui = ConnectFourGUI(root, model_paths)
    root.mainloop()
