import torch
import torch.nn.functional as F
from src.neural.models import RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from src.utils import plot_semantic_masks, plot_semantic_img, rgb_to_semantic_mask
from CarlaBEV.envs import CarlaBEV
from src.games.carlabev import make_env 


def test_loop(env):
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    total_reward = 0
    for _ in range(3000):
        # this is where you would insert your policy
        action = env.action_space.sample()
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
            total_reward = 0

    env.close()


if __name__ == "__main__":
    obs = torch.randn(2, 2, 6, 7)  # batch_size=2
    action = F.one_hot(torch.randint(0, 7, (2,)), num_classes=7).float()

    rep_net = RepresentationNetwork()
    dyn_net = DynamicsNetwork()
    pred_net = PredictionNetwork()

    hidden = rep_net(obs)
    next_hidden = dyn_net(hidden, action)
    policy_logits, value = pred_net(hidden)

    print(f"Hidden shape: {hidden.shape}")
    print(f"Next hidden shape: {next_hidden.shape}")
    print(f"Policy logits: {policy_logits.shape}, Value: {value.shape}")
    env = make_env(0, True, "test", 128)
    test_loop(env)
    
