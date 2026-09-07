"""Explicit-velocity EstNet: no latent representation or reconstruction head.

PPO gradients are deliberately allowed to pass through the estimator. The paper
does not specify this gradient convention; it is an explicit baseline choice.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class EstNet(nn.Module):
    """History -> three estimated velocities -> Gaussian actor, plus critic."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.obs_dim = int(cfg.obs_dim)
        self.command_dim = int(cfg.command_dim)
        self.critic_dim = int(cfg.critic_dim)
        self.action_dim = int(cfg.action_dim)
        self.history_steps = int(cfg.history_steps)
        if min(self.obs_dim, self.command_dim, self.critic_dim,
               self.action_dim, self.history_steps) <= 0:
            raise ValueError("Network dimensions must be positive")
        if not math.isfinite(cfg.init_std) or not math.exp(-3.0) <= cfg.init_std <= math.exp(1.0):
            raise ValueError("init_std must be within exp([-3, 1])")
        self.estimator = _mlp(
            self.history_steps * self.obs_dim, tuple(cfg.encoder_hidden), 3
        )
        self.actor = _mlp(
            self.obs_dim + self.command_dim + 3, tuple(cfg.actor_hidden), self.action_dim
        )
        self.critic = _mlp(self.critic_dim, tuple(cfg.critic_hidden), 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(cfg.init_std)))

    def estimate(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.obs_dim):
            raise ValueError(
                f"history must have shape [N, {self.history_steps}, {self.obs_dim}]"
            )
        return self.estimator(history.flatten(start_dim=1))

    def forward(
        self, history: torch.Tensor, obs: torch.Tensor, command: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        """Return distribution and the same differentiable velocity estimate."""
        velocity = self.estimate(history)
        batch = history.shape[0]
        if obs.shape != (batch, self.obs_dim) or command.shape != (batch, self.command_dim):
            raise ValueError("obs and command must match the history batch and configured dimensions")
        mean = self.actor(torch.cat((obs, command, velocity), dim=-1))
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), velocity

    def distribution(
        self, history: torch.Tensor, obs: torch.Tensor, command: torch.Tensor
    ) -> Normal:
        return self(history, obs, command)[0]

    def value(self, critic: torch.Tensor) -> torch.Tensor:
        if critic.ndim != 2 or critic.shape[-1] != self.critic_dim:
            raise ValueError(f"critic input must have shape [N, {self.critic_dim}]")
        return self.critic(critic).squeeze(-1)

    @torch.no_grad()
    def act(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample raw Gaussian actions; scaling/limits belong to the environment."""
        distribution = self.distribution(
            observation["history"], observation["obs"], observation["command"]
        )
        actions = distribution.sample()
        return (
            actions,
            distribution.log_prob(actions).sum(dim=-1),
            self.value(observation["critic"]),
            distribution.mean,
            distribution.stddev,
        )

    @torch.no_grad()
    def clamp_std_(self) -> None:
        """Keep the parameter inside its forward bounds so gradients can recover."""
        self.log_std.clamp_(-3.0, 1.0)
