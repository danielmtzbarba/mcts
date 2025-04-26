# train.py

import torch
import torch.nn.functional as F
import random

from self_play import play_game
from src.neural.models import RepresentationNetwork, DynamicsNetwork, PredictionNetwork


class ReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = []
        self.capacity = capacity

    def add(self, game_history):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(game_history)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def train(
    representation_net,
    dynamics_net,
    prediction_net,
    optimizer,
    replay_buffer,
    batch_size=16,
):
    """
    Train on a batch of data
    """
    games = replay_buffer.sample(batch_size)

    total_policy_loss = 0
    total_value_loss = 0

    for game in games:
        for obs, target_policy, target_value in zip(
            game.observations, game.policies, game.rewards
        ):
            obs = obs.unsqueeze(0)  # Add batch dimension
            target_policy = torch.tensor(target_policy, dtype=torch.float32).unsqueeze(
                0
            )  # (1, 7)
            target_value = torch.tensor([target_value], dtype=torch.float32)  # (1,)

            hidden = representation_net(obs)  # (1, 64, 6, 7)

            pred_policy_logits, pred_value = prediction_net(hidden)

            value_loss = F.mse_loss(pred_value.squeeze(), target_value.squeeze())
            log_probs = F.log_softmax(pred_policy_logits, dim=1)
            policy_loss = F.kl_div(log_probs, target_policy, reduction="batchmean")

            total_value_loss += value_loss
            total_policy_loss += policy_loss

    loss = total_value_loss + total_policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    # Create networks
    representation_net = RepresentationNetwork()
    dynamics_net = DynamicsNetwork()
    prediction_net = PredictionNetwork()

    params = (
        list(representation_net.parameters())
        + list(dynamics_net.parameters())
        + list(prediction_net.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-3)

    replay_buffer = ReplayBuffer()

    for iteration in range(1, 1001):
        print(f"=== Iteration {iteration} ===")

        # Self-play
        history = play_game(
            representation_net, dynamics_net, prediction_net, num_simulations=50
        )
        replay_buffer.add(history)

        # Only train if enough games
        if len(replay_buffer.buffer) >= 10:
            loss = train(
                representation_net,
                dynamics_net,
                prediction_net,
                optimizer,
                replay_buffer,
                batch_size=8,
            )
            print(f"Train loss: {loss:.4f}")

        # Optional: Save models
        if iteration % 100 == 0:
            torch.save(
                representation_net.state_dict(), f"representation_{iteration}.pth"
            )
            torch.save(dynamics_net.state_dict(), f"dynamics_{iteration}.pth")
            torch.save(prediction_net.state_dict(), f"prediction_{iteration}.pth")


if __name__ == "__main__":
    main()
