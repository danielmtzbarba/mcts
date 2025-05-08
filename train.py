# train.py

from types import SimpleNamespace

import os
import yaml

#
import torch
import torch.optim as optim
from torch.amp import GradScaler

from src.neural.muzero import MuZeroAgent
from src.mcts.mucts_nav import MuZeroMCTS
from src.mcts.self_play import self_play, evaluate
from src.games.carlabev import make_env

from src.drl.replay_buffer import ReplayBuffer
from src.drl.learn import train_network

from src.logger import DRLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(run_name, config, num_episodes=50, eval_episodes=10):
    logger = DRLogger(run_name)

    # --- Initialize ---
    env = make_env(seed=0, capture_video=True, run_name=run_name, size=128)
    replay_buffer = ReplayBuffer(capacity=config.buffer_size)
    #
    network = MuZeroAgent(hidden_dim=128, action_space_size=5).to(device)
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=5,
        num_simulations=config.num_simulations,
        c_puct=2.0,
    )
    #
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    # Training loop
    for it in range(1, num_episodes + 1):
        print(f"\n--- Training Iteration {it}/{num_episodes} ---")

        # Self-play
        self_play(env, mcts, replay_buffer, config.num_self_play_episodes, logger)

        # Train if enough data
        if len(replay_buffer) >= config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            train_network(
                network, optimizer, scaler, batch, config.num_unroll_steps, logger
            )

        # Evaluation
        if (it) % 5 == 0:
            evaluate(env, mcts, eval_episodes, logger)

        # Save checkpoint periodically
        if (it) % 50 == 0:
            torch.save(network.state_dict(), f"out/models/muzero_nav/{run_name}.pth")
            print(f"Checkpoint saved at episode {it}")

    print("Training completed!")
    return


def save_run_config_yaml(run_name, config_dict):
    """
    Saves a dictionary of hyperparameters to runs/<run_name>/params.yaml

    Args:
        run_name (str): Name of the experiment/run.
        config_dict (dict): Dictionary of hyperparameters.
    """
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"✅ Hyperparameters saved to {config_path}")


# Hyperparameters
config = {
    "num_self_play_episodes": 25,
    "num_simulations": 100,
    "num_unroll_steps": 10,
    "batch_size": 16,
    "buffer_size": 500,
    "learning_rate": 1e-4,
    "c_puct": 2.0,
}
#

if __name__ == "__main__":
    run_name = f"muzeronav_u{config['num_unroll_steps']}_cp{config['c_puct']}_sim{config['num_simulations']}_bs{config['batch_size']}_buf{config['buffer_size']}_lr{config['learning_rate']}"
    config = SimpleNamespace(**config)
    save_run_config_yaml(run_name, config)
    train(run_name, config, num_episodes=50, eval_episodes=10)
