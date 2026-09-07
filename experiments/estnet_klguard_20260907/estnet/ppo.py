"""PPO with joint supervised velocity estimation and explicit GAE boundaries.

Missing paper details are engineering choices: raw Gaussian, clipped value loss,
whole-rollout advantage normalization, and bidirectional KL learning-rate control.
This independent diagnostic branch adds full-policy post-step acceptance/rollback.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .networks import EstNet
from .ppo_diagnostics import cpu_clone_tree, finite_tensor_tree, full_rollout_kl, loss_gradient_diagnostics


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
        self.kl_hard_limit = float(getattr(cfg, "kl_hard_limit", 0.02))
        self.kl_chunk_size = getattr(cfg, "kl_chunk_size", 2048)
        if not math.isfinite(self.kl_hard_limit) or self.kl_hard_limit <= 0:
            raise ValueError("kl_hard_limit must be finite and positive")
        if isinstance(self.kl_chunk_size, bool) or not isinstance(self.kl_chunk_size, int) or self.kl_chunk_size < 1:
            raise ValueError("kl_chunk_size must be a positive integer")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.updates = 0
        self.total_accepted_steps = 0
        self.diagnostic_callback = None
        self.replay_callback = None

    def state_dict(self) -> dict[str, Any]:
        """Optimizer/scheduler state; the runner saves model weights separately."""
        return {"optimizer": self.optimizer.state_dict(), "updates": self.updates,
                "total_accepted_steps": self.total_accepted_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("updates", "total_accepted_steps"):
            if name not in state or isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
                raise ValueError(f"Guard checkpoint requires a nonnegative integer {name}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.total_accepted_steps = state["total_accepted_steps"]

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

    def _emit(self, event: dict[str, Any]) -> None:
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(event)

    def _restore_rejected_step(self, model_state, optimizer_state) -> None:
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = max(self.cfg.min_learning_rate, float(group["lr"]) / 1.5)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Joint PPO/estimation with a post-step full-rollout KL acceptance test.

        A rejected step restores model + Adam and stops the ENTIRE joint update.
        Loss/norm summaries cover attempted minibatches; gradient_steps counts
        retained steps. Diagnostic callbacks contain JSON-safe per-attempt evidence.
        """
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
        attempted = accepted = rejected = 0
        if not finite_tensor_tree(self.optimizer.state_dict()):
            raise FloatingPointError("Pre-update optimizer state is non-finite")
        current_kl = full_rollout_kl(self.model, data, self.kl_chunk_size)
        if not current_kl["finite"] or current_kl["total"] > self.kl_hard_limit:
            raise FloatingPointError("Pre-update policy already violates full-rollout KL guard")
        initial_kl = copy.deepcopy(current_kl)
        last_candidate = copy.deepcopy(current_kl)
        iteration = self.updates + 1
        stopped = False
        for epoch_index in range(self.cfg.epochs):
            order = torch.randperm(count, device=data["action"].device)
            for minibatch_index, indices in enumerate(torch.tensor_split(order, self.cfg.minibatches)):
                size = indices.numel()
                if size == 0:
                    continue
                if attempted == 0 and self.replay_callback is not None:
                    # Construct large CPU data only when explicitly requested.
                    self.replay_callback({
                        "schema": "estnet-guard-first-minibatch-replay-v1",
                        "iteration": iteration, "epoch": epoch_index + 1,
                        "minibatch": minibatch_index + 1, "full_rollout_count": count,
                        "advantages_normalized_over_full_rollout": True,
                        "indices": indices.detach().cpu().clone(),
                        "data": cpu_clone_tree({key: value[indices] for key, value in data.items()}),
                        "model": cpu_clone_tree(self.model.state_dict()),
                        "optimizer": cpu_clone_tree(self.optimizer.state_dict()),
                        "cfg": copy.deepcopy(self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else vars(self.cfg)),
                        "total_accepted_steps": self.total_accepted_steps,
                        "note": "Current diagnostic minibatch, not a replay of historical iteration 415.",
                    })
                pre_kl = copy.deepcopy(current_kl)
                lr_before = self.learning_rate
                model_state = copy.deepcopy(self.model.state_dict())
                optimizer_state = copy.deepcopy(self.optimizer.state_dict())
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
                gradient_diagnostics = None
                if getattr(self.cfg, "gradient_diagnostics", False):
                    gradient_diagnostics = loss_gradient_diagnostics(
                        self.model,
                        {"policy": policy_loss, "value": value_loss, "velocity": velocity_loss, "entropy": entropy},
                        {"policy": 1.0, "value": self.cfg.value_coef,
                         "velocity": self.cfg.velocity_coef, "entropy": -self.cfg.entropy_coef},
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
                )
                attempted += 1
                try:
                    self.optimizer.step()
                    self.model.clamp_std_()
                    candidate_kl = full_rollout_kl(self.model, data, self.kl_chunk_size)
                    candidate_adam_finite = finite_tensor_tree(self.optimizer.state_dict())
                except Exception:
                    # Never leave a partly applied optimizer step on a contract/runtime error.
                    self.model.load_state_dict(model_state, strict=True)
                    self.optimizer.load_state_dict(optimizer_state)
                    self.optimizer.zero_grad(set_to_none=True)
                    raise
                last_candidate = copy.deepcopy(candidate_kl)
                reject_reason = ("nonfinite_optimizer_state" if not candidate_adam_finite else
                                 candidate_kl["reason"] if not candidate_kl["finite"] else
                                 "kl_hard_limit_exceeded" if candidate_kl["total"] > self.kl_hard_limit else None)
                if reject_reason is not None:
                    self._restore_rejected_step(model_state, optimizer_state)
                    current_kl = full_rollout_kl(self.model, data, self.kl_chunk_size)
                    rejected += 1
                    stopped = True
                else:
                    # Candidate is already the post-step model: this full-rollout
                    # measurement is the acceptance verification, not a sampled proxy.
                    current_kl = copy.deepcopy(candidate_kl)
                    accepted += 1
                    self.total_accepted_steps += 1
                del model_state, optimizer_state
                if not current_kl["finite"] or current_kl["total"] > self.kl_hard_limit:
                    raise FloatingPointError("Retained model violates KL guard after candidate/rollback")
                values = {
                    "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                    "velocity_loss": velocity_loss, "entropy": entropy, "grad_norm": grad_norm,
                    "clip_fraction": ((ratio - 1.0).abs() > self.cfg.clip).float().mean(),
                }
                scalar_values = {key: float(value.detach()) for key, value in values.items()}
                for key, value in scalar_values.items():
                    totals[key] += value * size
                samples_seen += size
                self._emit({
                    "event": "ppo_minibatch_guard", "iteration": iteration, "epoch": epoch_index + 1,
                    "minibatch": minibatch_index + 1, "minibatch_samples": size, "rollout_samples": count,
                    "attempted_steps": attempted, "accepted_steps": accepted, "rejected_steps": rejected,
                    "total_accepted_steps": self.total_accepted_steps, "status": "rejected" if stopped else "accepted",
                    "rejection_reason": reject_reason, "kl_hard_limit": self.kl_hard_limit,
                    "candidate_optimizer_state_finite": candidate_adam_finite,
                    "kl_chunk_size": self.kl_chunk_size, "pre_kl": pre_kl, "candidate_kl": candidate_kl,
                    "retained_kl": current_kl, "lr_before": lr_before, "lr_after": self.learning_rate,
                    "losses_and_preclip_norm": scalar_values, "gradient_diagnostics": gradient_diagnostics,
                    "loss_summary_scope": "attempted minibatch before its candidate step",
                })
                if stopped:
                    break
            if stopped:
                break
            lr_before_schedule = self.learning_rate
            self._adapt_learning_rate(current_kl["total"])
            self._emit({"event": "ppo_epoch_schedule", "iteration": iteration, "epoch": epoch_index + 1,
                        "retained_kl": current_kl, "lr_before": lr_before_schedule, "lr_after": self.learning_rate})
        # Re-read the final retained estimator->actor policy before returning success.
        current_kl = full_rollout_kl(self.model, data, self.kl_chunk_size)
        if not current_kl["finite"] or current_kl["total"] > self.kl_hard_limit:
            raise FloatingPointError("Final retained model violates full-rollout KL guard")
        if not finite_tensor_tree(self.optimizer.state_dict()):
            raise FloatingPointError("Final retained optimizer state is non-finite")
        self.updates += 1
        metrics = {key: value / samples_seen for key, value in totals.items()}
        metrics.update({
            "kl": current_kl["total"],
            "pre_kl": initial_kl["total"],
            "candidate_kl": last_candidate["total"],
            "retained_kl": current_kl["total"],
            "candidate_kl_finite": float(last_candidate["finite"]),
            "learning_rate": self.learning_rate,
            "velocity_rmse": math.sqrt(metrics["velocity_loss"]),
            "action_std": self.model.log_std.detach().exp().mean().item(),
            "gradient_steps": float(accepted),
            "attempted_steps": float(attempted), "accepted_steps": float(accepted),
            "rejected_steps": float(rejected), "total_accepted_steps": float(self.total_accepted_steps),
            "guard_stopped_update": float(stopped),
        })
        self._emit({"event": "ppo_update_guard_summary", "iteration": iteration,
                    "attempted_steps": attempted, "accepted_steps": accepted, "rejected_steps": rejected,
                    "total_accepted_steps": self.total_accepted_steps, "pre_kl": initial_kl,
                    "candidate_kl": last_candidate, "retained_kl": current_kl,
                    "learning_rate": self.learning_rate, "stopped_joint_update": stopped})
        return metrics
