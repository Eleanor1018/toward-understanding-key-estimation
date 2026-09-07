"""Joint PPO/velocity estimation with fixed LR and observation-only Gaussian KL.

The PPO ratio and value clips remain. Finite KL, however large, never changes
the learning rate, skips a minibatch, or rolls back model/Adam state.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .networks import EstNet
from .ppo_diagnostics import cpu_clone_tree, finite_tensor_tree, full_rollout_kl, loss_gradient_diagnostics


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
    terminated: torch.Tensor, truncated: torch.Tensor,
    gamma: float = 0.996, lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE [time, env]; bootstrap actual pre-reset next states, stop at boundaries."""
    if rewards.ndim != 2 or rewards.shape[0] == 0:
        raise ValueError("GAE expects nonempty [time, environment] tensors")
    if any(t.shape != rewards.shape for t in (values, next_values, terminated, truncated)):
        raise ValueError("All GAE tensors must have the same shape")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("gamma and lam must lie in [0, 1]")
    terminal = terminated.bool()
    boundary = terminal | truncated.bool()
    # 真终止不自举；超时仍使用 reset 前的下一状态价值，但两者都截断 GAE 递推。
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
def gaussian_kl(old_mean, old_std, new_mean, new_std):
    """Exact KL(old || new), summed over action coordinates per sample."""
    return (torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
            - 0.5).sum(dim=-1)


class PPO:
    def __init__(self, model: EstNet, cfg: Any) -> None:
        self.model, self.cfg = model, cfg
        for name in ("epochs", "minibatches"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.fixed_learning_rate = float(cfg.learning_rate)
        if not math.isfinite(self.fixed_learning_rate) or self.fixed_learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if getattr(cfg, "learning_rate_schedule", "fixed") != "fixed":
            raise ValueError("This PPO implementation requires a fixed learning rate")
        self.kl_chunk_size = getattr(cfg, "kl_chunk_size", 2048)
        if isinstance(self.kl_chunk_size, bool) or not isinstance(self.kl_chunk_size, int) or self.kl_chunk_size < 1:
            raise ValueError("kl_chunk_size must be a positive integer")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.fixed_learning_rate)
        self.updates = 0
        # 默认每轮 4 epoch × 4 minibatch = 16 个 Adam 步，和 rollout 轮数分别记录。
        self.total_optimizer_steps = 0
        self.diagnostic_callback = None
        self.replay_callback = None

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        """A resumable checkpoint is at a complete rollout boundary, never mid-step."""
        for name in ("updates", "total_optimizer_steps"):
            if name not in state or isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0:
                raise ValueError(f"Fixed-LR PPO checkpoint requires a nonnegative integer {name}")
        steps = state["total_optimizer_steps"]
        if steps != state["updates"] * self.cfg.epochs * self.cfg.minibatches:
            raise ValueError("Optimizer step count does not match completed rollout updates")
        raw = state["optimizer"]
        if not finite_tensor_tree(raw):
            raise FloatingPointError("Checkpoint optimizer state is non-finite")
        if len(raw["param_groups"]) != len(self.optimizer.param_groups):
            raise ValueError("Checkpoint optimizer parameter group count differs")
        parameters = {}
        for saved, current in zip(raw["param_groups"], self.optimizer.param_groups):
            if not math.isclose(float(saved["lr"]), self.fixed_learning_rate, rel_tol=1e-12, abs_tol=0.):
                raise ValueError("Checkpoint LR does not equal the configured fixed learning rate")
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("Checkpoint optimizer parameter count differs")
            parameters.update(zip(saved["params"], current["params"]))
        if steps == 0:
            if raw["state"]:
                raise ValueError("Untrained checkpoint must have empty Adam state")
            return
        if set(raw["state"]) != set(parameters):
            raise ValueError("Trained checkpoint must have complete Adam states")
        for identifier, parameter in parameters.items():
            moment = raw["state"][identifier]
            if not {"step", "exp_avg", "exp_avg_sq"} <= moment.keys():
                raise ValueError("Checkpoint Adam state lacks step or moments")
            if float(moment["step"]) != steps:
                raise ValueError("Checkpoint Adam step differs from total_optimizer_steps")
            if any(not isinstance(moment[key], torch.Tensor) or moment[key].shape != parameter.shape
                   for key in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Checkpoint Adam moment shape differs from model")

    def state_dict(self) -> dict[str, Any]:
        state = {"optimizer": self.optimizer.state_dict(), "updates": self.updates,
                 "total_optimizer_steps": self.total_optimizer_steps}
        self._validate_checkpoint_state(state)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._validate_checkpoint_state(state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.total_optimizer_steps = state["total_optimizer_steps"]

    def _emit(self, event: dict[str, Any]) -> None:
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(event)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Run every configured joint minibatch; KL is measured only after epochs.

        Numerical failures raise and do not emit a successful update summary.
        There is intentionally no transactional rollback, including after a failure.
        """
        self._validate_checkpoint_state({"optimizer": self.optimizer.state_dict(),
            "updates": self.updates, "total_optimizer_steps": self.total_optimizer_steps})
        required = ("history", "obs", "command", "critic", "velocity", "action",
                    "old_log_prob", "old_value", "old_mean", "old_std", "returns", "advantages")
        missing = set(required) - batch.keys()
        if missing:
            raise ValueError(f"Missing rollout fields: {sorted(missing)}")
        data = {key: batch[key].detach() for key in required}
        count = data["action"].shape[0]
        if count < self.cfg.minibatches or any(t.shape[0] != count for t in data.values()):
            raise ValueError("Rollout must provide matching leading dimensions and at least one sample per minibatch")
        if not finite_tensor_tree(data) or not finite_tensor_tree(self.model.state_dict()):
            raise FloatingPointError("Non-finite rollout or pre-update model state")
        for name in ("old_log_prob", "old_value", "returns", "advantages"):
            if data[name].shape not in ((count,), (count, 1)):
                raise ValueError(f"{name} must contain one scalar per transition")
            data[name] = data[name].reshape(count)
        for name in ("action", "old_mean", "old_std"):
            if data[name].shape != (count, self.model.action_dim):
                raise ValueError(f"{name} has an incompatible action dimension")
        if data["velocity"].shape != (count, 3):
            raise ValueError("velocity targets must have shape [N, 3]")
        if not (data["old_std"] > 0).all():
            raise ValueError("old_std must be positive")
        advantages = data["advantages"]
        # 延续原实现：先在完整 rollout 上归一化优势，再划分 minibatch。
        data["advantages"] = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
        totals = {key: 0.0 for key in ("loss", "policy_loss", "value_loss", "velocity_loss", "entropy",
                                      "clip_fraction", "grad_norm")}
        samples_seen = optimizer_steps = 0
        epoch_kls = []
        iteration = self.updates + 1
        for epoch_index in range(self.cfg.epochs):
            order = torch.randperm(count, device=data["action"].device)
            for minibatch_index, indices in enumerate(torch.tensor_split(order, self.cfg.minibatches)):
                size = indices.numel()
                if optimizer_steps == 0 and self.replay_callback is not None:
                    self.replay_callback({
                        "schema": "estnet-fixed-lr-first-minibatch-replay-v1", "iteration": iteration,
                        "epoch": epoch_index + 1, "minibatch": minibatch_index + 1,
                        "full_rollout_count": count, "advantages_normalized_over_full_rollout": True,
                        "indices": indices.detach().cpu().clone(),
                        "data": cpu_clone_tree({key: value[indices] for key, value in data.items()}),
                        "model": cpu_clone_tree(self.model.state_dict()),
                        "optimizer": cpu_clone_tree(self.optimizer.state_dict()),
                        "cfg": cpu_clone_tree(self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else vars(self.cfg)),
                        "total_optimizer_steps": self.total_optimizer_steps,
                        "note": "Snapshot before the first optimizer step of this rollout; not a historical replay.",
                    })
                distribution, velocity = self.model(data["history"][indices], data["obs"][indices], data["command"][indices])
                log_prob = distribution.log_prob(data["action"][indices]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["old_log_prob"][indices])
                if not torch.isfinite(log_prob).all() or not torch.isfinite(ratio).all():
                    raise FloatingPointError("Non-finite PPO log probability or probability ratio")
                advantage = data["advantages"][indices]
                # PPO clip=.2 裁剪的是新旧策略概率比，既不是关节角限制，也不是 KL 上限。
                policy_loss = -torch.minimum(ratio * advantage,
                    ratio.clamp(1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * advantage).mean()
                value = self.model.value(data["critic"][indices])
                old_value, target = data["old_value"][indices], data["returns"][indices]
                clipped_value = old_value + (value - old_value).clamp(-self.cfg.clip, self.cfg.clip)
                value_loss = torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
                velocity_loss = F.mse_loss(velocity, data["velocity"][indices])
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = (policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                        + self.cfg.velocity_coef * velocity_loss)
                # 估计器监督仍与 actor/critic 联合更新；没有停止或分离监督梯度。
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite joint PPO/velocity loss")
                gradient_diagnostics = None
                if getattr(self.cfg, "gradient_diagnostics", False):
                    gradient_diagnostics = loss_gradient_diagnostics(self.model,
                        {"policy": policy_loss, "value": value_loss, "velocity": velocity_loss, "entropy": entropy},
                        {"policy": 1.0, "value": self.cfg.value_coef,
                         "velocity": self.cfg.velocity_coef, "entropy": -self.cfg.entropy_coef})
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.model.clamp_std_()
                optimizer_steps += 1
                self.total_optimizer_steps += 1
                values = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                          "velocity_loss": velocity_loss, "entropy": entropy, "grad_norm": grad_norm,
                          "clip_fraction": ((ratio - 1.0).abs() > self.cfg.clip).float().mean()}
                scalar_values = {key: float(value.detach()) for key, value in values.items()}
                for key, value in scalar_values.items():
                    totals[key] += value * size
                samples_seen += size
                self._emit({"event": "ppo_minibatch_update", "iteration": iteration, "epoch": epoch_index + 1,
                    "minibatch": minibatch_index + 1, "minibatch_samples": size, "rollout_samples": count,
                    "optimizer_steps": optimizer_steps, "total_optimizer_steps": self.total_optimizer_steps,
                    "learning_rate": self.learning_rate, "losses_and_preclip_norm": scalar_values,
                    "gradient_diagnostics": gradient_diagnostics,
                    "loss_summary_scope": "executed minibatch before its optimizer step"})
            # 每个完整 epoch 仅做一次全 rollout KL 测量，默认一轮共四次。
            # 有限 KL 无论多大都只记录，不改 LR、不提前停止、不回滚已执行的 Adam 步。
            kl = full_rollout_kl(self.model, data, self.kl_chunk_size)
            if not kl["finite"]:
                raise FloatingPointError(f"Non-finite post-epoch policy/model: {kl['reason']}")
            if not finite_tensor_tree(self.optimizer.state_dict()):
                raise FloatingPointError("Non-finite post-epoch optimizer state")
            epoch_kls.append(kl)
            self._emit({"event": "ppo_epoch_kl", "iteration": iteration, "epoch": epoch_index + 1,
                        "kl": kl, "kl_role": "measurement_only", "kl_chunk_size": self.kl_chunk_size,
                        "optimizer_steps": optimizer_steps, "total_optimizer_steps": self.total_optimizer_steps,
                        "learning_rate": self.learning_rate})
        if optimizer_steps != self.cfg.epochs * self.cfg.minibatches:
            raise RuntimeError("Incomplete PPO update cannot be reported as successful")
        self.updates += 1
        metrics = {key: value / samples_seen for key, value in totals.items()}
        metrics.update({"kl": epoch_kls[-1]["total"], "max_epoch_kl": max(item["total"] for item in epoch_kls),
            "learning_rate": self.learning_rate, "velocity_rmse": math.sqrt(metrics["velocity_loss"]),
            "action_std": self.model.log_std.detach().exp().mean().item(),
            "gradient_steps": float(optimizer_steps), "optimizer_steps": float(optimizer_steps),
            "total_optimizer_steps": float(self.total_optimizer_steps)})
        self._emit({"event": "ppo_update_summary", "iteration": iteration,
                    "optimizer_steps": optimizer_steps, "total_optimizer_steps": self.total_optimizer_steps,
                    "kl": epoch_kls[-1], "max_epoch_kl": metrics["max_epoch_kl"],
                    "kl_role": "measurement_only", "learning_rate": self.learning_rate})
        return metrics
