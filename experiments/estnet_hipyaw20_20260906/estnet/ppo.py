"""PPO with joint supervised velocity estimation and explicit GAE boundaries.

Missing paper details are engineering choices: raw Gaussian, clipped value loss,
whole-rollout advantage normalization, and bidirectional KL learning-rate control.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .networks import EstNet


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = 0.996,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns) for [time, environment] transitions.

next_values[t] is V of the actual next state, BEFORE reset, including timeouts.
True termination removes the bootstrap. Both boundaries stop the GAE trace.
"""
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("GAE expects nonempty [time, environment] tensors")
    if any(t.shape != rewards.shape for t in (values, next_values, terminated, truncated)):
        raise ValueError("All GAE tensors must have the same shape")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("gamma and lam must lie in [0, 1]")
    terminal = terminated.bool()
    boundary = terminal | truncated.bool()
    bootstrap = torch.where(terminal, torch.zeros_like(next_values), next_values)
    deltas = rewards + gamma * bootstrap - values
    advantages = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for step in reversed(range(rewards.shape[0])):
        carry = deltas[step] + gamma * lam * torch.where(
            boundary[step], torch.zeros_like(carry), carry
        )
        advantages[step] = carry
    return advantages, advantages + values


@torch.no_grad()
def gaussian_kl(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(old || new), summed over action coordinates per sample."""
    return (
        torch.log(new_std / old_std)
        + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
        - 0.5
    ).sum(dim=-1)


class PPO:
    def __init__(self, model: EstNet, cfg: Any) -> None:
        self.model = model
        self.cfg = cfg
        if cfg.epochs < 1 or cfg.minibatches < 1:
            raise ValueError("epochs and minibatches must be positive")
        if not 0.0 < cfg.min_learning_rate <= cfg.learning_rate <= cfg.max_learning_rate:
            raise ValueError("Learning-rate bounds must contain the initial learning rate")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.updates = 0

    def state_dict(self) -> dict[str, Any]:
        """Optimizer/scheduler state; the runner saves model weights separately."""
        return {"optimizer": self.optimizer.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _adapt_learning_rate(self, kl: float) -> None:
        if self.cfg.desired_kl <= 0.0:
            return
        rate = self.learning_rate
        if kl > 2.0 * self.cfg.desired_kl:
            rate = max(self.cfg.min_learning_rate, rate / 1.5)
        elif 0.0 < kl < 0.5 * self.cfg.desired_kl:
            rate = min(self.cfg.max_learning_rate, rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Optimize one flat rollout, retaining PPO -> estimator gradients."""
        required = (
            "history", "obs", "command", "critic", "velocity", "action",
            "old_log_prob", "old_value", "old_mean", "old_std", "returns", "advantages",
        )
        missing = set(required) - batch.keys()
        if missing:
            raise ValueError(f"Missing rollout fields: {sorted(missing)}")
        data = {key: batch[key].detach() for key in required}
        count = data["action"].shape[0]
        if count == 0 or any(t.shape[0] != count for t in data.values()):
            raise ValueError("Rollout fields must have the same nonempty leading dimension")
        for name in ("old_log_prob", "old_value", "returns", "advantages"):
            if data[name].shape not in ((count,), (count, 1)):
                raise ValueError(f"{name} must contain one scalar per transition")
            data[name] = data[name].reshape(count)
        for name in ("action", "old_mean", "old_std"):
            if data[name].shape != (count, self.model.action_dim):
                raise ValueError(f"{name} has an incompatible action dimension")
        if data["velocity"].shape != (count, 3):
            raise ValueError("velocity targets must have shape [N, 3]")
        if not torch.isfinite(data["old_std"]).all() or not (data["old_std"] > 0).all():
            raise ValueError("old_std must be finite and positive")
        advantages = data["advantages"]
        data["advantages"] = (advantages - advantages.mean()) / (
            advantages.std(correction=0) + 1e-8
        )
        totals = {key: 0.0 for key in (
            "loss", "policy_loss", "value_loss", "velocity_loss", "entropy",
            "clip_fraction", "grad_norm",
        )}
        samples_seen = 0
        gradient_steps = 0
        last_kl = 0.0
        for _ in range(self.cfg.epochs):
            order = torch.randperm(count, device=data["action"].device)
            for indices in torch.tensor_split(order, self.cfg.minibatches):
                size = indices.numel()
                if size == 0:
                    continue
                distribution, velocity = self.model(
                    data["history"][indices], data["obs"][indices], data["command"][indices]
                )
                log_prob = distribution.log_prob(data["action"][indices]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["old_log_prob"][indices])
                advantage = data["advantages"][indices]
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * advantage,
                ).mean()
                value = self.model.value(data["critic"][indices])
                old_value = data["old_value"][indices]
                target = data["returns"][indices]
                clipped_value = old_value + (value - old_value).clamp(-self.cfg.clip, self.cfg.clip)
                value_loss = torch.maximum(
                    (value - target).square(), (clipped_value - target).square()
                ).mean()
                velocity_loss = F.mse_loss(velocity, data["velocity"][indices])
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = (policy_loss + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy
                        + self.cfg.velocity_coef * velocity_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite joint PPO/velocity loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                self.model.clamp_std_()
                values = {
                    "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                    "velocity_loss": velocity_loss, "entropy": entropy, "grad_norm": grad_norm,
                    "clip_fraction": ((ratio - 1.0).abs() > self.cfg.clip).float().mean(),
                }
                for key, value in values.items():
                    totals[key] += float(value.detach()) * size
                samples_seen += size
                gradient_steps += 1
            # Measure all samples and the entire estimator->actor policy. This
            # adjusts the next epoch/rollout LR; it neither rejects nor rolls back.
            with torch.no_grad():
                current = self.model.distribution(data["history"], data["obs"], data["command"])
                last_kl = gaussian_kl(
                    data["old_mean"], data["old_std"], current.mean, current.stddev
                ).mean().item()
            if not math.isfinite(last_kl):
                raise FloatingPointError("Non-finite post-epoch policy KL")
            last_kl = max(0.0, last_kl)
            self._adapt_learning_rate(last_kl)
        self.updates += 1
        metrics = {key: value / samples_seen for key, value in totals.items()}
        metrics.update({
            "kl": last_kl,
            "learning_rate": self.learning_rate,
            "velocity_rmse": math.sqrt(metrics["velocity_loss"]),
            "action_std": self.model.log_std.detach().exp().mean().item(),
            "gradient_steps": float(gradient_steps),
        })
        return metrics
