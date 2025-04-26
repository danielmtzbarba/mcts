import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

# Discrete action space
accel_vals = [0.0, 0.33, 0.66, 1.0]
steer_vals = [-1.0, -0.33, 0.33, 1.0]

# Grid world
GRID_SIZE = 20
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Add obstacle
grid[10, 10] = -1  # red (obstacle)
grid[18, 18] = 2  # green (goal)

# Vehicle state
state = {"x": 1.0, "y": 1.0, "heading": 0.0, "velocity": 0.0}


# Simple dynamics model
def step(state, action):
    accel, steer = action
    new_state = state.copy()
    new_state["heading"] += steer * 0.1
    new_state["velocity"] = min(new_state["velocity"] + accel * 0.1, 1.0)
    new_state["x"] += new_state["velocity"] * math.cos(new_state["heading"])
    new_state["y"] += new_state["velocity"] * math.sin(new_state["heading"])
    return new_state


# Render state
def render(grid, state, step_num):
    img = grid.copy()
    x = int(round(state["x"]))
    y = int(round(state["y"]))
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        img[y, x] = 1  # vehicle = blue

    cmap = colors.ListedColormap(["white", "blue", "green", "red"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(img, cmap=cmap, norm=norm)
    plt.title(f"Step {step_num}")
    plt.grid(False)
    plt.pause(0.5)
    plt.clf()


# Run a simple action sequence
plt.figure(figsize=(6, 6))

for i in range(12):
    action = (accel_vals[min(i // 3, 3)], steer_vals[i % 4])
    state = step(state, action)
    render(grid, state, i)

plt.close()
