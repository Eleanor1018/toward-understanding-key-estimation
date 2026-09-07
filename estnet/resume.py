"""恢复训练权重与Adam状态；环境和历史从新回合开始，不能宣称精确续训。"""
from __future__ import annotations

import math
from typing import Any

import torch

from .networks import EstNet
from .ppo import PPO
from .factory import build_model, build_ppo


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validate_rng(payload: dict[str, Any]) -> None:
    for name in ("torch_rng", "torch_cuda_rng"):
        if name not in payload:
            continue
        state = payload[name]
        if (not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or
                state.device.type != "cpu" or state.ndim != 1 or state.numel() == 0):
            raise ValueError(f"{name} must be a nonempty one-dimensional CPU ByteTensor")
        if name == "torch_rng":
            # 用独立CPU生成器检验状态，不污染当前进程的随机数序列。
            try:
                torch.Generator(device="cpu").set_state(state)
            except RuntimeError as exc:
                raise ValueError("Invalid CPU torch_rng state") from exc
        # CUDA生成器格式只能在实际CUDA恢复时由Torch检验；CPU预检不创建CUDA上下文。


def _load_and_validate_optimizer(model: EstNet, ppo: PPO, payload: dict[str, Any]) -> None:
    wrapped = payload.get("optimizer")
    if not isinstance(wrapped, dict) or not isinstance(wrapped.get("optimizer"), dict):
        raise ValueError("Training resume requires the complete PPO/Adam optimizer state")  # noqa: TRY004 - 损坏的检查点统一报告ValueError
    updates = wrapped.get("updates")
    if type(updates) is not int or updates < 0 or updates != payload.get("iteration"):
        raise ValueError("PPO updates must be a non-negative integer equal to checkpoint iteration")
    saved = wrapped["optimizer"]
    groups, states = saved.get("param_groups"), saved.get("state")
    if not isinstance(groups, list) or not groups or not isinstance(states, dict):
        raise ValueError("Invalid Adam parameter groups or state dictionary")
    parameter_ids = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("Invalid Adam parameter group")  # noqa: TRY004 - 检查点结构校验
        ids = group["params"]
        if any(type(index) is not int or index < 0 for index in ids):
            raise ValueError("Invalid Adam parameter identifiers")
        parameter_ids.extend(ids)
        rate = _finite_number(group.get("lr"), "Adam learning rate")
        if not 0 < ppo.cfg.min_learning_rate <= rate <= ppo.cfg.max_learning_rate:
            raise ValueError("Adam learning rate is outside the configured bounds")
        betas = group.get("betas")
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("Adam betas must contain two values")
        if any(not 0 <= _finite_number(beta, "Adam beta") < 1 for beta in betas):
            raise ValueError("Adam betas must lie in [0, 1)")
        if _finite_number(group.get("eps"), "Adam epsilon") <= 0:
            raise ValueError("Adam epsilon must be positive")
        if _finite_number(group.get("weight_decay"), "Adam weight decay") < 0:
            raise ValueError("Adam weight decay must be non-negative")
        for flag in ("amsgrad", "maximize", "capturable", "differentiable"):
            if flag in group and type(group[flag]) is not bool:
                raise ValueError(f"Adam {flag} must be boolean")
        for flag in ("foreach", "fused"):
            if flag in group and group[flag] is not None and type(group[flag]) is not bool:
                raise ValueError(f"Adam {flag} must be boolean or None")
    if len(parameter_ids) != len(set(parameter_ids)) or any(type(index) is not int for index in states):
        raise ValueError("Adam parameter identifiers must be unique integers")
    if not set(states).issubset(parameter_ids):
        raise ValueError("Adam state contains unknown parameters")
    if updates > 0 and set(states) != set(parameter_ids):
        raise ValueError("A trained checkpoint must retain Adam state for every parameter")
    try:
        ppo.load_state_dict(wrapped)
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise ValueError("PPO/Adam optimizer state cannot be loaded") from exc

    # Adam.load_state_dict只校验参数组数量；错误矩阵形状常要到step才报错，必须提前查。
    for group in ppo.optimizer.param_groups:
        for parameter in group["params"]:
            state = ppo.optimizer.state.get(parameter)
            if state is None:
                if updates == 0:
                    continue
                raise ValueError("Missing trained Adam parameter state")
            if not isinstance(state, dict):
                raise ValueError("Invalid Adam parameter state")  # noqa: TRY004 - 检查点结构校验
            required = ["exp_avg", "exp_avg_sq"]
            if group.get("amsgrad", False):
                required.append("max_exp_avg_sq")
            for name in required:
                value = state.get(name)
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or
                        value.dtype != parameter.dtype or not value.is_floating_point() or
                        not torch.isfinite(value).all()):
                    raise ValueError(f"Adam {name} must be finite and match its parameter shape/dtype")
                if name != "exp_avg" and (value < 0).any():
                    raise ValueError(f"Adam {name} cannot contain negative second moments")
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError("Adam step must be scalar")
                step = step.item()
            step = _finite_number(step, "Adam step")
            if step < 0 or not step.is_integer() or (updates > 0 and step == 0):
                raise ValueError("Adam step must be a non-negative integer, positive after training")
    if len(list(model.parameters())) != sum(len(group["params"]) for group in ppo.optimizer.param_groups):
        raise ValueError("Adam state does not cover the model parameters")


def load_training_checkpoint(path, asset):
    """严格CPU预检；缺优化器时拒绝续训，不静默变成仅加载权重的热启动。"""
    from .run import _validate_checkpoint

    payload = torch.load(path, map_location="cpu", weights_only=True)
    # 临时校验网络会初始化随机权重；保存并恢复调用方CPU RNG，且不触及CUDA生成器。
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        cfg = _validate_checkpoint(payload, asset)
        _validate_rng(payload)
        model = build_model(cfg)
        ppo = build_ppo(model, cfg)
        _load_and_validate_optimizer(model, ppo, payload)
    return payload, cfg


def restore_training_state(model: EstNet, ppo: PPO, payload: dict[str, Any]) -> dict[str, Any]:
    """恢复训练器；目标模型可在CUDA上，Adam矩状态由Torch迁移到参数设备。"""
    if ppo.model is not model:
        raise ValueError("PPO must own the model being restored")
    if type(payload.get("iteration")) is not int or payload["iteration"] < 0:
        raise ValueError("Checkpoint iteration must be a non-negative integer")
    _validate_rng(payload)
    model.load_state_dict(payload["model"], strict=True)
    _load_and_validate_optimizer(model, ppo, payload)
    cpu_restored = "torch_rng" in payload
    if cpu_restored:
        torch.set_rng_state(payload["torch_rng"])
    device = next(model.parameters()).device
    cuda_restored = False
    cuda_status = "not_saved"
    if "torch_cuda_rng" in payload:
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["torch_cuda_rng"], device=device)
            cuda_restored = True
            cuda_status = "restored"
        else:
            cuda_status = "not_applicable_cpu_model"
    # 即便新检查点带CUDA RNG，也未保存PhysX状态/历史/回合计时，不能声称精确续训。
    return {"iteration": payload["iteration"], "ppo_updates": ppo.updates,
            "learning_rate": ppo.learning_rate, "cpu_rng_restored": cpu_restored,
            "cuda_rng_restored": cuda_restored, "cuda_rng_status": cuda_status,
            "environment_state_restored": False, "history_restored": False,
            "resume_mode": "new_episodes", "exact_resume": False}
