# hyperparameters.py
import optuna

from types import SimpleNamespace

#
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


def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_self_play_episodes = trial.suggest_categorical(
        "num_self_play_episodes", [25, 50, 100]
    )
    num_simulations = trial.suggest_categorical("num_simulations", [25, 50, 100])
    num_unroll_steps = trial.suggest_categorical("num_unroll_steps", [5, 10])
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    buffer_size = trial.suggest_categorical("buffer_size", [200, 500, 1000])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-3])
    c_puct = trial.suggest_categorical("c_puct", [1.0, 2.0, 4.0])

    # Hyperparameters
    config = {
        "hidden_dim": hidden_dim,
        "num_self_play_episodes": num_self_play_episodes,
        "num_simulations": num_simulations,
        "num_unroll_steps": num_unroll_steps,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_rate": learning_rate,
        "c_puct": c_puct,
    }

    run_name = f"muzeronav-dim{config['hidden_dim']}_us{config['num_unroll_steps']}_cp{config['c_puct']}_sim{config['num_simulations']}_bs{config['batch_size']}_buf{config['buffer_size']}_lr{config['learning_rate']}"
    config = SimpleNamespace(**config)
    save_run_config_yaml(run_name, config)

    # --- Initialize ---
    logger = DRLogger(run_name)
    env = make_env(seed=0, capture_video=False, run_name=run_name, size=128)
    eval_env = make_env(seed=1, capture_video=True, run_name=run_name, size=128)
    replay_buffer = ReplayBuffer(capacity=config.buffer_size)
    #
    network = MuZeroAgent(hidden_dim=config.hidden_dim, action_space_size=5).to(device)
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=5,
        num_simulations=config.num_simulations,
    )
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    decay = 0.9846

    num_episodes = 10
    eval_episodes = 3
    # Training loop
    for it in range(1, num_episodes + 1):
        print(f"\n--- Training Iteration {it}/{num_episodes} ---")
        #current_cpuct = max(config.c_puct * (decay**logger.num_ep), 1.0)
        mcts.c_puct = config.c_puct
        # Self-play
        self_play(env, mcts, replay_buffer, config.num_self_play_episodes, logger)

        # Train if enough data
        if len(replay_buffer) >= config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            train_network(
                network, optimizer, scaler, batch, config.num_unroll_steps, logger
            )

        mcts.c_puct = 2.0
        avg_ret = evaluate(eval_env, mcts, eval_episodes, logger)

        # Save checkpoint periodically
        if (it) % 5 == 0:
            torch.save(network.state_dict(), f"out/models/muzero_nav/{run_name}.pth")
            print(f"Checkpoint saved at episode {it}")

    logger.logger.info("-----")
    logger.logger.info(run_name)
    logger.logger.info(f"Average Evaluation Return = {avg_ret:.4f}")
    logger.logger.info("-----")
    logger.logger.info("\n")

    return avg_ret


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


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:")
    print(study.best_trial.params)
