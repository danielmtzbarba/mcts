import torch
import torch.nn.functional as F
from src.neural.models import RepresentationNetwork, DynamicsNetwork, PredictionNetwork

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
