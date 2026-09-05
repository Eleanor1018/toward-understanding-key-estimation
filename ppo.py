import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class DiagonalGaussian(nn.Module):
    distribution_type = "tanh_diagonal_gaussian_v1"

    def __init__(
        self,
        action_dim: int,
        initial_std: float = 0.3,
        min_std: float = 0.05,
        max_std: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0.0 < min_std <= initial_std <= max_std:
            raise ValueError(
                "Expected 0 < min_std <= initial_std <= max_std, got "
                f"{min_std}, {initial_std}, {max_std}."
            )
        self.action_dim = action_dim

        initial_log_std = math.log(initial_std)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

    def forward(self, action_mean: torch.Tensor) -> Normal:

        bounded_log_std = self.log_std.clamp(
            min=self.min_log_std,
            max=self.max_log_std,
        )
        action_std = bounded_log_std.exp()

        return Normal(action_mean, action_std)

    @staticmethod
    def _log_tanh_jacobian(pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Stable log(1 - tanh(x)^2) used by the change of variables."""

        return 2.0 * (
            math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action)
        )

    def _log_prob_from_pre_tanh(
        self,
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        return (
            distribution.log_prob(pre_tanh_action)
            - self._log_tanh_jacobian(pre_tanh_action)
        ).sum(dim=-1)

    def sample(
        self,
        action_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        distribution = self(action_mean)
        pre_tanh_actions = distribution.sample()
        actions = torch.tanh(pre_tanh_actions)
        log_prob = self._log_prob_from_pre_tanh(
            distribution,
            pre_tanh_actions,
        )

        return actions, log_prob

    def sample_for_ppo(
        self,
        action_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample bounded actions while retaining stable PPO coordinates.

        The tanh Jacobian depends only on the sampled action, so it cancels
        exactly in the new/old PPO likelihood ratio. Retaining the pre-tanh
        sample avoids numerically inverting actions that rounded to +/-1.
        """

        distribution = self(action_mean)
        pre_tanh_actions = distribution.sample()
        actions = torch.tanh(pre_tanh_actions)
        base_log_prob = distribution.log_prob(pre_tanh_actions).sum(dim=-1)
        return actions, pre_tanh_actions, base_log_prob

    def ppo_log_prob(
        self,
        action_mean: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Log probability in pre-tanh coordinates for PPO ratios."""

        return self(action_mean).log_prob(pre_tanh_actions).sum(dim=-1)

    def log_prob(
        self,
        action_mean: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:

        distribution = self(action_mean)
        epsilon = torch.finfo(actions.dtype).eps
        bounded_actions = actions.clamp(
            min=-1.0 + epsilon,
            max=1.0 - epsilon,
        )
        pre_tanh_actions = torch.atanh(bounded_actions)
        return self._log_prob_from_pre_tanh(
            distribution,
            pre_tanh_actions,
        )

    def entropy(self, action_mean: torch.Tensor) -> torch.Tensor:

        # A tanh-transformed Normal has no closed-form entropy. A single
        # reparameterized sample gives a differentiable Monte-Carlo estimate
        # and, unlike base-Normal entropy, also penalizes saturated means.
        distribution = self(action_mean)
        pre_tanh_actions = distribution.rsample()
        return -self._log_prob_from_pre_tanh(
            distribution,
            pre_tanh_actions,
        )

    @torch.no_grad()
    def clamp_std_parameters_(
        self,
        maximum_std: float | None = None,
    ) -> None:
        """Keep the stored parameter inside the same bounds used in forward."""

        maximum_log_std = self.max_log_std
        if maximum_std is not None:
            if maximum_std <= 0.0:
                raise ValueError("maximum_std must be positive")
            maximum_log_std = min(maximum_log_std, math.log(maximum_std))
        if maximum_log_std < self.min_log_std:
            maximum_log_std = self.min_log_std
        self.log_std.clamp_(
            min=self.min_log_std,
            max=maximum_log_std,
        )

    @torch.no_grad()
    def reset_std_parameters_(self, action_std: float) -> None:
        """Reset exploration scale at an explicit curriculum boundary."""

        if not math.isfinite(action_std) or action_std <= 0.0:
            raise ValueError("action_std must be finite and positive")
        action_log_std = math.log(action_std)
        if not self.min_log_std <= action_log_std <= self.max_log_std:
            raise ValueError(
                f"action_std must be within the configured bounds, got {action_std}"
            )
        self.log_std.fill_(action_log_std)

    @torch.no_grad()
    def std_statistics(self) -> tuple[float, float]:
        action_std = self.log_std.clamp(
            min=self.min_log_std,
            max=self.max_log_std,
        ).exp()
        return action_std.mean().item(), action_std.max().item()

    @staticmethod
    def deterministic_action(action_mean: torch.Tensor) -> torch.Tensor:
        return torch.tanh(action_mean)


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_value: torch.Tensor,
    time_out_bootstrap_values: torch.Tensor | None = None,
    gamma: float = 0.996,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:

    rewards = rewards.detach()
    values = values.detach()
    terminated = terminated.detach()
    truncated = truncated.detach()
    last_value = last_value.detach()
    successful_time_outs = torch.logical_and(
        truncated,
        torch.logical_not(terminated),
    )
    if time_out_bootstrap_values is None:
        if torch.any(successful_time_outs):
            raise ValueError("Timeout transitions require terminal-state values")
        time_out_bootstrap_values = torch.zeros_like(values)
    else:
        time_out_bootstrap_values = time_out_bootstrap_values.detach()

    advantages = torch.zeros_like(values)

    next_advantage = torch.zeros_like(last_value)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_value = last_value
        else:
            next_value = values[step + 1]

        next_value = torch.where(
            successful_time_outs[step],
            time_out_bootstrap_values[step],
            next_value,
        )
        bootstrap_mask = torch.logical_not(terminated[step]).to(dtype=values.dtype)
        trace_mask = torch.logical_not(
            torch.logical_or(terminated[step], truncated[step])
        ).to(dtype=values.dtype)

        delta = rewards[step] + gamma * bootstrap_mask * next_value - values[step]

        next_advantage = delta + gamma * gae_lambda * trace_mask * next_advantage
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
