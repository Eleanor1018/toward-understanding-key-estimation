import math

import torch
from torch import nn
from torch.distributions import Normal


class DiagonalGaussian(nn.Module):

    def __init__(
        self,
        action_dim: int,
        initial_std: float = 1.0,
        min_std: float = 1e-6,
        max_std: float = 1e6,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        initial_log_std = math.log(initial_std)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), initial_log_std)
        )

        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

    def forward(self, action_mean: torch.Tensor) -> Normal:

        bounded_log_std = self.log_std.clamp(
            min=self.min_log_std,
            max=self.max_log_std,
        )
        action_std = bounded_log_std.exp()

        return Normal(action_mean, action_std)

    def sample(
        self,
        action_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        distribution = self(action_mean)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)

        return actions, log_prob

    def log_prob(
        self,
        action_mean: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:

        distribution = self(action_mean)
        return distribution.log_prob(actions).sum(dim=-1)

    def entropy(self, action_mean: torch.Tensor) -> torch.Tensor:

        distribution = self(action_mean)
        return distribution.entropy().sum(dim=-1)

    @staticmethod
    def deterministic_action(action_mean: torch.Tensor) -> torch.Tensor:
        return action_mean


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.996,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:

    rewards = rewards.detach()
    values = values.detach()
    dones = dones.detach()
    last_value = last_value.detach()

    advantages = torch.zeros_like(values)

    next_advantage = torch.zeros_like(last_value)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_value = last_value
        else:
            next_value = values[step + 1]

        not_done = 1.0 - dones[step].to(dtype=values.dtype)

        delta = (
            rewards[step]
            + gamma * not_done * next_value
            - values[step]
        )

        next_advantage = (
            delta
            + gamma * gae_lambda * not_done * next_advantage
        )
        advantages[step] = next_advantage

    returns = advantages + values

    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:

    if advantages.numel() == 0:
        raise ValueError("advantages must not be empty")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    mean = advantages.mean()
    std = advantages.std(correction=0)
    return (advantages - mean) / (std + epsilon)


def clipped_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:

    old_log_prob = old_log_prob.detach()
    advantages = advantages.detach()

    ratio = torch.exp(new_log_prob - old_log_prob)

    unclipped_objective = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )
    clipped_objective = clipped_ratio * advantages

    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -objective.mean()

    return loss, ratio


def value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:

    old_values = old_values.detach()
    returns = returns.detach()

    value_change = new_values - old_values

    clipped_values = old_values + value_change.clamp(
        min=-clip_epsilon,
        max=clip_epsilon,
    )

    unclipped_loss = (new_values - returns).pow(2)
    clipped_loss = (clipped_values - returns).pow(2)

    loss = torch.maximum(
        unclipped_loss,
        clipped_loss,
    ).mean()

    return loss


def total_ppo_loss(
    policy_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    entropy: torch.Tensor,
    value_loss_coefficient: float = 1.0,
    entropy_coefficient: float = 0.01,
) -> torch.Tensor:

    loss = (
        policy_loss
        + value_loss_coefficient * critic_loss
        - entropy_coefficient * entropy.mean()
    )

    return loss
