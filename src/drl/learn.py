import numpy as np
import torch
import torch.nn.functional as F

from torch.amp import autocast
from src.drl.replay_buffer import prepare_tensors


def train_network(model, optimizer, scaler, replay_buffer, batch_size=8 , num_unroll_steps=5, logger=None):
    model.train()
    losses = []
    value_losses = []
    policy_losses = []
    reward_losses = []

    batch, indices, weights = replay_buffer.sample(batch_size, beta=0.4)

    for i, episode in enumerate(batch):
        obs_tensor, action_tensor, reward_tensor, policy_tensor = prepare_tensors(
            episode
        )
        # Compute predicted values and TD errors
        hiddens = model.representation(obs_tensor.cuda())
        predicted_values = model.value_head(hiddens.cuda())
        observed_returns = np.array([sum([step[2] for step in episode])])
        td_errors = np.abs(observed_returns - predicted_values.detach().cpu().numpy())

        # Update priorities in the replay buffer
        replay_buffer.update_priorities(indices[i:i+1], td_errors)

        root_obs = obs_tensor[0].unsqueeze(0)
        # Unroll the episode
        hidden_state, pred_policy, pred_value = model.initial_inference(root_obs)
        total_loss = 0.0
        for step in range(num_unroll_steps):
            try:
                action = action_tensor[:, step]
            except IndexError:
                break  # Break if action_tensor is shorter than expected

            with autocast(device_type="cuda"):
                hidden_state, policy_logits, pred_value, pred_reward = (
                    model.recurrent_inference(hidden_state, action)
                )
                # Calculate losses
                reward_loss = F.mse_loss(pred_reward, reward_tensor[step].unsqueeze(0))
                value_loss = F.mse_loss(pred_value, reward_tensor[step].unsqueeze(0))
                pred_policy = F.softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(
                    policy_tensor * torch.log(pred_policy + 1e-8), dim=1
                ).mean()

                # Combine losses
                step_loss = (0.01 * policy_loss) + value_loss + reward_loss
                loss = (weights[i] * step_loss).mean()
                total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Append individual losses for logging
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        reward_losses.append(reward_loss.item())
        losses.append(total_loss)

    value = np.mean(value_losses)
    policy = np.mean(policy_losses)
    reward = np.mean(reward_losses)
    mean_loss = np.mean(losses)
    # Logging
    logger.losses(total_loss, mean_loss, value, policy, reward)
    #
    return
