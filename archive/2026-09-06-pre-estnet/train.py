"""Unitree G1 29-DOF 的 PPO 训练入口。

固定环境：Isaac Sim 4.5、PyTorch 2.5.1、CUDA 12.1。

环境需要返回五组基础观测：
    history、obs、command、privileged、explicit_target
V8 额外返回 future_reference。
"""

from __future__ import annotations

import argparse
import copy
import math
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from config import (
    FUTURE_REFERENCE_NORMALIZATION_TYPE,
    FUTURE_REFERENCE_REWARD_PROFILE,
    FULL_REFERENCE_REWARD_PROFILE,
    ModelConfig,
    model_config_for_reward_profile,
)
from normalization import (
    normalize_command,
    normalize_future_reference,
    normalize_obs,
    normalize_observation_batch,
    normalize_privileged,
    normalization_type_for_command_dim,
)
from policy import Actor, Critic, Decoder, Encoder
from ppo import (
    DiagonalGaussian,
    clipped_policy_loss,
    generalized_advantage_estimation,
    normalize_advantages,
    total_ppo_loss,
    value_loss,
)


# train.py 只依赖五组基础数据和 V8 可选的 future_reference。
# 这样更换机器人或奖励函数时，PPO 主循环不需要跟着重写。
OBSERVATION_KEYS = (
    "history",
    "obs",
    "command",
    "privileged",
    "explicit_target",
)
OPTIONAL_OBSERVATION_KEYS = ("future_reference",)
AUXILIARY_OBJECTIVE_TYPE = "current_observation_reconstruction_detached_v1"
V8_CURRICULUM_STEPS = 5000


def parse_args_and_launch_simulator() -> tuple[argparse.Namespace, Any]:
    """读取训练参数并启动 Isaac Sim。"""

    from isaaclab.app import AppLauncher

    config = ModelConfig()

    parser = argparse.ArgumentParser(description="训练 Unitree G1 29-DOF 策略。")

    # 环境和采样规模。
    parser.add_argument(
        "--task",
        type=str,
        default="Unitree-G1-29dof-KeyEstimation-v0",
        help="Isaac Lab 任务名称。",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--episode-length-s", type=float, default=None)
    parser.add_argument("--command-scale", type=float, default=1.0)
    parser.add_argument(
        "--command-profile",
        choices=(
            "stand",
            "stage1",
            "forward_walk",
            "walk",
            "stage2",
            "stage3",
            "full",
        ),
        default="full",
    )
    parser.add_argument(
        "--domain-randomization-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reward-profile",
        choices=(
            "paper",
            "p1_dense",
            "p1_stable",
            "p1_walk",
            "p1_walk_gait",
            "p1_walk_gait_v2",
            "p1_walk_stable_v3",
            "p1_walk_stable_v4",
            "p1_walk_stable_v5",
            "p1_walk_stable_v5_imitation",
            "p1_walk_stable_v6_phase_rsi",
            "p1_walk_stable_v6_phase_rsi_imitation",
            FULL_REFERENCE_REWARD_PROFILE,
            FUTURE_REFERENCE_REWARD_PROFILE,
        ),
        default="p1_stable",
    )

    # PPO 更新参数。
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=5.0e-5)
    parser.add_argument(
        "--minimum-policy-learning-rate",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=config.discount_gamma)
    parser.add_argument("--gae-lambda", type=float, default=config.gae_lambda)
    parser.add_argument(
        "--policy-clip",
        type=float,
        default=config.ppo_clip_epsilon,
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--max-post-update-kl", type=float, default=0.03)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
    parser.add_argument("--final-entropy-coef", type=float, default=0.0)
    parser.add_argument("--prediction-loss-coef", type=float, default=0.5)
    parser.add_argument("--estimation-loss-coef", type=float, default=1.0)
    parser.add_argument("--initial-imitation-weight", type=float, default=0.15)
    parser.add_argument("--final-imitation-weight", type=float, default=0.03)
    parser.add_argument(
        "--reference-state-initialization-probability",
        type=float,
        default=0.70,
    )
    parser.add_argument("--reference-motion-ramp-iterations", type=int, default=200)
    parser.add_argument("--reference-motion-origin-iteration", type=int, default=None)
    parser.add_argument("--v8-curriculum-origin-iteration", type=int, default=None)
    parser.add_argument("--v8-curriculum-end-iteration", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--initial-action-std", type=float, default=0.3)
    parser.add_argument("--final-action-std", type=float, default=0.1)
    parser.add_argument("--min-action-std", type=float, default=0.05)
    parser.add_argument("--max-action-std", type=float, default=0.6)

    # 复现实验和保存模型。
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/g1_key_estimation"),
    )
    # 添加 --headless、--device 等 Isaac Lab 参数。
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("--episode-length-s must be positive")
    if args.policy_learning_rate <= 0.0:
        parser.error("--policy-learning-rate must be positive")
    if not 0.0 < args.minimum_policy_learning_rate <= args.policy_learning_rate:
        parser.error(
            "--minimum-policy-learning-rate must be positive and no larger "
            "than --policy-learning-rate"
        )
    if args.max_post_update_kl <= 0.0:
        parser.error("--max-post-update-kl must be positive")
    if args.initial_imitation_weight < 0.0 or args.final_imitation_weight < 0.0:
        parser.error("imitation weights must be non-negative")
    if not 0.0 <= args.reference_state_initialization_probability <= 1.0:
        parser.error("--reference-state-initialization-probability must be in [0, 1]")
    if args.reference_motion_ramp_iterations < 0:
        parser.error("--reference-motion-ramp-iterations must be non-negative")
    if (
        args.reference_motion_origin_iteration is not None
        and args.reference_motion_origin_iteration < 1
    ):
        parser.error("--reference-motion-origin-iteration must be positive")
    for name in (
        "v8_curriculum_origin_iteration",
        "v8_curriculum_end_iteration",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.reward_profile
        in (FULL_REFERENCE_REWARD_PROFILE, FUTURE_REFERENCE_REWARD_PROFILE)
        and args.command_profile != "forward_walk"
    ):
        parser.error(f"{args.reward_profile} requires --command-profile forward_walk")

    app_launcher = AppLauncher(args, multi_gpu=False)
    return args, app_launcher.app


def create_environment(args: argparse.Namespace) -> Any:
    """创建并行仿真环境。"""

    # Isaac Lab 要求先启动 AppLauncher，再导入仿真相关模块。
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 -- 注册 Isaac Lab 任务
    from isaaclab_tasks.utils import parse_env_cfg
    import g1_env  # noqa: F401 -- 注册自定义任务

    # g1_env 负责注册任务；这里负责设置并行数量并创建实例。
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.command_scale = args.command_scale
    env_cfg.command_profile = args.command_profile
    env_cfg.domain_randomization_scale = args.domain_randomization_scale
    env_cfg.reward_profile = args.reward_profile
    env_cfg.imitation_reward_weight = args.initial_imitation_weight
    env_cfg.reference_state_initialization_probability = getattr(
        args,
        "initial_reference_state_initialization_probability",
        args.reference_state_initialization_probability,
    )
    env_cfg.reference_motion_progress = (
        1.0
        if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE
        else getattr(
            args,
            "initial_reference_motion_progress",
            0.0,
        )
    )
    env_cfg.task_mix_beta = getattr(args, "initial_task_mix_beta", 0.0)
    return gym.make(args.task, cfg=env_cfg)


def get_observation_batch(
    observation: Any,
) -> dict[str, torch.Tensor]:
    """取出并归一化网络需要的五组观测。"""

    raw_batch = {key: observation[key] for key in OBSERVATION_KEYS}
    for key in OPTIONAL_OBSERVATION_KEYS:
        if key in observation:
            raw_batch[key] = observation[key]
    return normalize_observation_batch(raw_batch)


def flattened_future_reference(
    observation: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Return the optional V8 future reference as one 63-D network input."""

    future_reference = observation.get("future_reference")
    if future_reference is None:
        return None
    if future_reference.shape[-2:] != (3, 21):
        raise ValueError(
            "future_reference must end in shape (3, 21), got "
            f"{tuple(future_reference.shape)}"
        )
    return future_reference.flatten(start_dim=-2)


def validate_initial_commands(
    observation: Any,
    num_envs: int,
    allow_all_zero: bool = False,
) -> tuple[float, float]:
    """Catch silent command-sampling failures before collecting rollouts."""

    commands = observation["command"][..., :3]
    if not allow_all_zero and num_envs >= 32 and not torch.any(commands != 0.0):
        raise RuntimeError("Command sampler returned all-zero commands")
    return commands.abs().mean().item(), commands.abs().max().item()


def linear_imitation_weight(
    iteration: int,
    start_iteration: int,
    end_iteration: int,
    initial_weight: float,
    final_weight: float,
) -> float:
    """Linearly anneal imitation over this run, independent of checkpoint age."""

    progress = (iteration - start_iteration) / max(
        end_iteration - start_iteration,
        1,
    )
    progress = min(max(progress, 0.0), 1.0)
    return initial_weight + progress * (final_weight - initial_weight)


def set_imitation_reward_weight(env: Any, weight: float) -> None:
    """Set the phase-imitation curriculum on the underlying Gym environment."""

    setter = getattr(env.unwrapped, "set_imitation_reward_weight", None)
    if setter is None:
        raise RuntimeError("Environment does not expose imitation-weight scheduling")
    setter(weight)


def reference_motion_progress(
    iteration: int,
    start_iteration: int,
    ramp_iterations: int,
) -> float:
    """Grow V7 motion amplitude over this run without changing its first step."""

    if ramp_iterations < 0:
        raise ValueError("ramp_iterations must be non-negative")
    if ramp_iterations == 0:
        return 1.0
    return min(max((iteration - start_iteration) / ramp_iterations, 0.0), 1.0)


def set_reference_motion_progress(env: Any, progress: float) -> None:
    """Set the synchronized V7 action/reference/RSI curriculum."""

    setter = getattr(env.unwrapped, "set_reference_motion_progress", None)
    if setter is None:
        raise RuntimeError("Environment does not expose reference-motion scheduling")
    setter(progress)


def _smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def v8_task_mix_beta(iteration: int, origin_iteration: int) -> float:
    """Return the staged V8 task-reward fraction at an absolute iteration."""

    new_step = iteration - origin_iteration + 1
    if new_step <= 1000:
        return 0.0
    if new_step <= 3000:
        return 0.9 * _smoothstep((new_step - 1000) / 2000.0)
    if new_step <= 3500:
        return 0.9 + 0.1 * _smoothstep((new_step - 3000) / 500.0)
    return 1.0


def v8_reference_state_initialization_probability(
    iteration: int,
    origin_iteration: int,
) -> float:
    """Return the staged V8 RSI probability at an absolute iteration."""

    new_step = iteration - origin_iteration + 1
    if new_step <= 500:
        return 1.0
    if new_step <= 1000:
        return 1.0 - 0.3 * (new_step - 500) / 500.0
    if new_step <= 2500:
        return 0.7 - 0.4 * (new_step - 1000) / 1500.0
    if new_step <= 3500:
        return 0.3 * (3500 - new_step) / 1000.0
    return 0.0


def prepare_v8_curriculum(
    args: argparse.Namespace,
    start_iteration: int,
    source_train_args: dict[str, Any] | None = None,
) -> None:
    """Resolve and validate V8's absolute staged curriculum."""

    if args.reward_profile != FUTURE_REFERENCE_REWARD_PROFILE:
        args.initial_task_mix_beta = 0.0
        args.initial_reference_state_initialization_probability = getattr(
            args, "reference_state_initialization_probability", 0.0
        )
        return

    source_train_args = source_train_args or {}
    source_is_v8 = (
        source_train_args.get("reward_profile") == FUTURE_REFERENCE_REWARD_PROFILE
    )
    if source_is_v8:
        saved_origin = source_train_args.get("v8_curriculum_origin_iteration")
        saved_end = source_train_args.get("v8_curriculum_end_iteration")
        if not isinstance(saved_origin, int) or not isinstance(saved_end, int):
            raise RuntimeError("V8 checkpoint omitted its curriculum bounds")
        if (
            args.v8_curriculum_origin_iteration is not None
            and args.v8_curriculum_origin_iteration != saved_origin
        ):
            raise RuntimeError(
                "V8 resume changed its curriculum origin: "
                f"checkpoint={saved_origin}, "
                f"requested={args.v8_curriculum_origin_iteration}"
            )
        if (
            args.v8_curriculum_end_iteration is not None
            and args.v8_curriculum_end_iteration != saved_end
        ):
            raise RuntimeError(
                "V8 resume changed its curriculum end: "
                f"checkpoint={saved_end}, requested={args.v8_curriculum_end_iteration}"
            )
        args.v8_curriculum_origin_iteration = saved_origin
        args.v8_curriculum_end_iteration = saved_end
    else:
        if args.v8_curriculum_origin_iteration is None:
            args.v8_curriculum_origin_iteration = start_iteration
        if args.v8_curriculum_end_iteration is None:
            args.v8_curriculum_end_iteration = (
                args.v8_curriculum_origin_iteration + V8_CURRICULUM_STEPS - 1
            )

    expected_end = args.v8_curriculum_origin_iteration + V8_CURRICULUM_STEPS - 1
    if args.v8_curriculum_end_iteration != expected_end:
        raise RuntimeError(
            "V8 curriculum must span exactly "
            f"{V8_CURRICULUM_STEPS} iterations: "
            f"origin={args.v8_curriculum_origin_iteration}, "
            f"end={args.v8_curriculum_end_iteration}"
        )
    requested_training_end = getattr(args, "iterations", None)
    if isinstance(requested_training_end, int):
        if requested_training_end < start_iteration:
            raise RuntimeError(
                "V8 training target precedes its next iteration: "
                f"target={requested_training_end}, next={start_iteration}"
            )
        if requested_training_end > args.v8_curriculum_end_iteration:
            raise RuntimeError(
                "V8 training target exceeds its 5000-iteration curriculum end: "
                f"target={requested_training_end}, "
                f"end={args.v8_curriculum_end_iteration}"
            )

    if source_is_v8:
        checkpoint_iteration = start_iteration - 1
        expected_beta = v8_task_mix_beta(
            checkpoint_iteration,
            args.v8_curriculum_origin_iteration,
        )
        expected_rsi = v8_reference_state_initialization_probability(
            checkpoint_iteration,
            args.v8_curriculum_origin_iteration,
        )
        for name, expected in (
            ("current_task_mix_beta", expected_beta),
            (
                "current_reference_state_initialization_probability",
                expected_rsi,
            ),
        ):
            saved = source_train_args.get(name)
            if not isinstance(saved, (float, int)) or not math.isclose(
                float(saved),
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                raise RuntimeError(
                    f"V8 checkpoint {name} is inconsistent with its schedule"
                )

    args.initial_task_mix_beta = v8_task_mix_beta(
        start_iteration,
        args.v8_curriculum_origin_iteration,
    )
    args.initial_reference_state_initialization_probability = (
        v8_reference_state_initialization_probability(
            start_iteration,
            args.v8_curriculum_origin_iteration,
        )
    )


def set_task_mix_beta(env: Any, beta: float) -> None:
    """Set V8's imitation/task reward mixture."""

    setter = getattr(env.unwrapped, "set_task_mix_beta", None)
    if setter is None:
        raise RuntimeError("Environment does not expose task-mix scheduling")
    setter(beta)


def set_reference_state_initialization_probability(
    env: Any,
    probability: float,
) -> None:
    """Set the reset-state curriculum on the underlying Gym environment."""

    setter = getattr(
        env.unwrapped,
        "set_reference_state_initialization_probability",
        None,
    )
    if setter is None:
        raise RuntimeError("Environment does not expose RSI scheduling")
    setter(probability)


def clone_for_rollout(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach normalized inputs before retaining them in rollout storage."""

    # normalize_observation_batch already allocates tensors that the simulator
    # does not own, so a second 50x93 history clone would only waste bandwidth.
    return {name: tensor.detach() for name, tensor in batch.items()}


# 采样阶段只记录数据，不建立反向传播图，可以显著减少显存占用。
@torch.no_grad()
def collect_rollout(
    env: Any,
    observation: Any,
    encoder: Encoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, torch.Tensor], Any]:
    """让所有并行环境运行一段 rollout。"""

    # 每个列表先按时间保存 tensor，rollout 结束后再 stack 成：
    # [rollout_steps, num_envs, ...]。
    observation_keys = list(OBSERVATION_KEYS)
    observation_keys.extend(
        key for key in OPTIONAL_OBSERVATION_KEYS if key in observation
    )
    storage: dict[str, list[torch.Tensor]] = {key: [] for key in observation_keys}
    for key in (
        "actions",
        "pre_tanh_actions",
        "old_log_prob",
        "old_values",
        "rewards",
        "terminated",
        "truncated",
        "time_out_bootstrap_values",
    ):
        storage[key] = []

    for _ in range(rollout_steps):
        batch = get_observation_batch(observation)
        if tuple(batch) != tuple(observation_keys):
            raise RuntimeError(
                "Environment observation keys changed during rollout: "
                f"expected {tuple(observation_keys)}, got {tuple(batch)}"
            )
        saved_batch = clone_for_rollout(batch)
        future_reference = flattened_future_reference(saved_batch)

        # history -> Encoder -> latent 和显式状态估计。
        # Actor 使用机器人可获得的信息；Critic 可以额外读取 privileged。
        latent, explicit_estimate = encoder(saved_batch["history"])
        action_mean = actor(
            saved_batch["obs"],
            saved_batch["command"],
            latent,
            explicit_estimate,
            future_reference,
        )
        actions, pre_tanh_actions, log_prob = action_distribution.sample_for_ppo(
            action_mean
        )
        values = critic(
            saved_batch["obs"],
            saved_batch["command"],
            saved_batch["privileged"],
            future_reference,
        )

        # 把 29 维动作交给环境，得到下一时刻的数据。
        next_observation, rewards, terminated, truncated, extras = env.step(actions)
        rewards = rewards.reshape(-1)
        terminated = terminated.reshape(-1)
        truncated = truncated.reshape(-1)
        successful_time_outs = torch.logical_and(
            truncated,
            torch.logical_not(terminated),
        )
        time_out_bootstrap_values = torch.zeros_like(values)
        time_out_observation = extras.get("time_out_critic_observation")
        if time_out_observation is not None:
            time_out_env_ids = time_out_observation["env_ids"]
            time_out_future_reference = None
            if "future_reference" in time_out_observation:
                time_out_future_reference = normalize_future_reference(
                    time_out_observation["future_reference"]
                ).flatten(start_dim=-2)
            time_out_bootstrap_values[time_out_env_ids] = critic(
                normalize_obs(time_out_observation["obs"]),
                normalize_command(time_out_observation["command"]),
                normalize_privileged(time_out_observation["privileged"]),
                time_out_future_reference,
            )
            captured_time_outs = torch.zeros_like(successful_time_outs)
            captured_time_outs[time_out_env_ids] = True
            if not torch.equal(captured_time_outs, successful_time_outs):
                raise RuntimeError("Timeout bootstrap observation IDs mismatch")
        elif torch.any(successful_time_outs):
            raise RuntimeError("Environment omitted timeout terminal observations")

        # old_log_prob 和 old_values 是采样时的快照，后面的 PPO 更新
        # 会把新网络输出与它们比较。
        for key in observation_keys:
            storage[key].append(saved_batch[key])
        storage["actions"].append(actions.detach().clone())
        storage["pre_tanh_actions"].append(pre_tanh_actions.detach().clone())
        storage["old_log_prob"].append(log_prob.detach().clone())
        storage["old_values"].append(values.detach().clone())
        storage["rewards"].append(rewards.detach().clone())
        # Both boundary types cut the GAE trace. A successful timeout retains
        # its pre-reset critic value above, while a true termination uses zero.
        storage["terminated"].append(terminated.detach().clone())
        storage["truncated"].append(truncated.detach().clone())
        storage["time_out_bootstrap_values"].append(
            time_out_bootstrap_values.detach().clone()
        )

        observation = next_observation

    # rollout 最后一帧没有存 value，需要额外计算一次给 GAE bootstrap。
    final_batch = get_observation_batch(observation)
    if tuple(final_batch) != tuple(observation_keys):
        raise RuntimeError("Environment observation keys changed after rollout")
    last_values = critic(
        final_batch["obs"],
        final_batch["command"],
        final_batch["privileged"],
        flattened_future_reference(final_batch),
    )

    rollout = {name: torch.stack(items) for name, items in storage.items()}

    # 用 reward、value 和 done 从后往前计算 advantage 与 return。
    advantages, returns = generalized_advantage_estimation(
        rewards=rollout["rewards"],
        values=rollout["old_values"],
        terminated=rollout["terminated"],
        truncated=rollout["truncated"],
        last_value=last_values,
        time_out_bootstrap_values=rollout["time_out_bootstrap_values"],
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    rollout["advantages"] = advantages
    rollout["returns"] = returns
    return rollout, observation


def flatten_time_and_env(tensor: torch.Tensor) -> torch.Tensor:
    """[时间, 环境, ...] → [时间 × 环境, ...]。"""

    return tensor.flatten(start_dim=0, end_dim=1)


@torch.no_grad()
def value_target_statistics(
    returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one affine value-loss scale over all distributed samples."""

    stable_returns = returns.to(torch.float64)
    statistics = torch.stack(
        (
            stable_returns.sum(),
            stable_returns.square().sum(),
            torch.tensor(
                returns.numel(),
                dtype=torch.float64,
                device=returns.device,
            ),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    mean = statistics[0] / statistics[2]
    variance = statistics[1] / statistics[2] - mean.square()
    scale = variance.clamp_min(0.0).sqrt().clamp_min(1.0)
    return mean.to(returns.dtype), scale.to(returns.dtype)


@torch.no_grad()
def normalized_input_clip_counts(
    rollout: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count values at each normalizer's explicit clipping boundary."""

    general_tensors = (
        rollout["history"],
        rollout["obs"],
        rollout["command"],
        rollout["explicit_target"],
        rollout["privileged"][..., :74],
    )
    if "future_reference" in rollout:
        general_tensors = (*general_tensors, rollout["future_reference"])
    clipped = sum(
        torch.count_nonzero(tensor.abs() >= 4.999) for tensor in general_tensors
    )
    privileged_torque = rollout["privileged"][..., 74:103]
    clipped += torch.count_nonzero(privileged_torque.abs() >= 1.499)
    total = torch.tensor(
        sum(tensor.numel() for tensor in general_tensors) + privileged_torque.numel(),
        device=rollout["obs"].device,
    )
    return clipped, total


def create_optimizer(
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    """Use a smaller trust-region-sensitive LR for the deployed policy."""

    policy_parameters = (
        list(encoder.parameters())
        + list(actor.parameters())
        + list(action_distribution.parameters())
    )
    return torch.optim.Adam(
        (
            {
                "params": policy_parameters,
                "lr": args.policy_learning_rate,
                "name": "policy",
            },
            {
                "params": list(critic.parameters()),
                "lr": args.learning_rate,
                "name": "critic",
            },
            {
                "params": list(decoder.parameters()),
                "lr": args.learning_rate,
                "name": "decoder",
            },
        )
    )


@torch.no_grad()
def compute_global_approximate_kl(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_ratio = new_log_prob - old_log_prob
    local_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
    statistics = torch.stack(
        (
            local_kl * new_log_prob.numel(),
            torch.tensor(
                new_log_prob.numel(),
                dtype=local_kl.dtype,
                device=local_kl.device,
            ),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    return local_kl, statistics[0] / statistics[1]


@torch.no_grad()
def snapshot_policy_update(
    policy_parameters: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> tuple[list[torch.Tensor], dict[nn.Parameter, dict[str, Any] | None]]:
    parameter_snapshot = [parameter.detach().clone() for parameter in policy_parameters]
    optimizer_snapshot: dict[nn.Parameter, dict[str, Any] | None] = {}
    for parameter in policy_parameters:
        state = optimizer.state.get(parameter)
        if not state:
            optimizer_snapshot[parameter] = None
            continue
        optimizer_snapshot[parameter] = {
            name: (
                value.detach().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )
            for name, value in state.items()
        }
    return parameter_snapshot, optimizer_snapshot


@torch.no_grad()
def restore_policy_update(
    policy_parameters: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    parameter_snapshot: list[torch.Tensor],
    optimizer_snapshot: dict[nn.Parameter, dict[str, Any] | None],
) -> None:
    for parameter, saved_parameter in zip(
        policy_parameters,
        parameter_snapshot,
    ):
        parameter.copy_(saved_parameter)
        saved_state = optimizer_snapshot[parameter]
        if saved_state is None:
            optimizer.state.pop(parameter, None)
        else:
            optimizer.state[parameter] = saved_state


def policy_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    for parameter_group in optimizer.param_groups:
        if parameter_group.get("name") == "policy":
            return float(parameter_group["lr"])
    raise RuntimeError("Optimizer is missing the policy parameter group")


def reduce_policy_learning_rate(
    optimizer: torch.optim.Optimizer,
    minimum_learning_rate: float,
) -> float:
    for parameter_group in optimizer.param_groups:
        if parameter_group.get("name") == "policy":
            parameter_group["lr"] = max(
                float(parameter_group["lr"]) * 0.5,
                minimum_learning_rate,
            )
            return float(parameter_group["lr"])
    raise RuntimeError("Optimizer is missing the policy parameter group")


def update_models(
    rollout: dict[str, torch.Tensor],
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[nn.Parameter],
    args: argparse.Namespace,
) -> dict[str, float]:
    """打乱 rollout，并执行多轮 PPO 更新。"""

    current_entropy_coefficient = getattr(
        args,
        "current_entropy_coef",
        args.entropy_coef,
    )
    # PPO 不再区分“时间”和“第几个环境”，统一摊平成训练样本。
    flat = {name: flatten_time_and_env(value) for name, value in rollout.items()}
    if "future_reference" in flat:
        flat["future_reference"] = flattened_future_reference(flat)
    value_target_mean, value_target_scale = value_target_statistics(flat["returns"])

    # 标准化只改变 advantage 的尺度，不改变样本的正负方向。
    flat["advantages"] = normalize_advantages(flat["advantages"])
    sample_count = flat["actions"].shape[0]
    policy_parameters = (
        list(encoder.parameters())
        + list(actor.parameters())
        + list(action_distribution.parameters())
    )
    guard_indices = torch.randperm(
        sample_count,
        device=flat["actions"].device,
    )[: min(sample_count, 4096)]

    # 记录所有 mini-batch 的平均指标，便于观察训练是否稳定。
    metric_sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "prediction_loss": 0.0,
        "estimation_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "latent_norm": 0.0,
        "deterministic_action_abs": 0.0,
        "deterministic_action_saturation": 0.0,
        "post_update_kl": 0.0,
    }
    update_count = 0
    maximum_approximate_kl = 0.0
    maximum_post_update_kl = 0.0
    stopped_for_kl = False
    accepted_policy_updates = 0
    rejected_policy_updates = 0

    for _ in range(args.update_epochs):
        # 同一批 rollout 重复使用，每轮重新打乱。
        shuffled_indices = torch.randperm(
            sample_count,
            device=flat["actions"].device,
        )

        for indices in torch.tensor_split(shuffled_indices, args.num_minibatches):
            if indices.numel() == 0:
                continue
            # 重新前向计算；旧概率和旧价值保持不变，作为 PPO 的参照。
            latent, explicit_estimate = encoder(flat["history"][indices])
            predicted_obs = decoder(
                latent.detach(),
                explicit_estimate.detach(),
            )
            action_mean = actor(
                flat["obs"][indices],
                flat["command"][indices],
                latent,
                explicit_estimate,
                (
                    flat["future_reference"][indices]
                    if "future_reference" in flat
                    else None
                ),
            )
            new_log_prob = action_distribution.ppo_log_prob(
                action_mean,
                flat["pre_tanh_actions"][indices],
            )
            entropy = action_distribution.entropy(action_mean)
            new_values = critic(
                flat["obs"][indices],
                flat["command"][indices],
                flat["privileged"][indices],
                (
                    flat["future_reference"][indices]
                    if "future_reference" in flat
                    else None
                ),
            )

            # Actor：限制新旧策略概率比，避免一次更新改变过大。
            actor_loss, probability_ratio = clipped_policy_loss(
                new_log_prob=new_log_prob,
                old_log_prob=flat["old_log_prob"][indices],
                advantages=flat["advantages"][indices],
                clip_epsilon=args.policy_clip,
            )
            with torch.no_grad():
                approximate_kl, global_approximate_kl = compute_global_approximate_kl(
                    new_log_prob,
                    flat["old_log_prob"][indices],
                )
                maximum_approximate_kl = max(
                    maximum_approximate_kl,
                    global_approximate_kl.item(),
                )
                stopped_for_kl = (
                    args.target_kl > 0.0
                    and update_count > 0
                    and global_approximate_kl.item() > 1.5 * args.target_kl
                )
            if stopped_for_kl:
                break
            # Critic：让预测价值靠近 GAE 得到的 return。
            critic_loss = value_loss(
                new_values=(new_values - value_target_mean) / value_target_scale,
                old_values=(flat["old_values"][indices] - value_target_mean)
                / value_target_scale,
                returns=(flat["returns"][indices] - value_target_mean)
                / value_target_scale,
                clip_epsilon=args.value_clip,
            )

            # Reconstruct the current normalized observation. Predicting the
            # next observation without conditioning on the sampled action is
            # under-specified; detaching also keeps this diagnostic head from
            # moving the policy encoder.
            prediction_loss = F.mse_loss(
                predicted_obs,
                flat["obs"][indices],
            )

            # 让 Encoder 估计的显式状态接近仿真器给出的真值。
            estimation_loss = F.mse_loss(
                explicit_estimate,
                flat["explicit_target"][indices],
            )

            # PPO 学习动作，两个辅助损失训练 Encoder 和 Decoder。
            ppo_loss = total_ppo_loss(
                policy_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                value_loss_coefficient=args.value_loss_coef,
                entropy_coefficient=current_entropy_coefficient,
            )
            loss = (
                ppo_loss
                + args.prediction_loss_coef * prediction_loss
                + args.estimation_loss_coef * estimation_loss
            )
            finite_loss_terms = torch.stack(
                (
                    actor_loss,
                    critic_loss,
                    entropy.mean(),
                    prediction_loss,
                    estimation_loss,
                    loss,
                )
            )
            if not torch.isfinite(finite_loss_terms).all().item():
                raise FloatingPointError(
                    "Non-finite PPO loss terms: "
                    f"{finite_loss_terms.detach().cpu().tolist()}"
                )

            # 清空旧梯度，反向传播总损失，再限制梯度范数后更新参数。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy_parameters,
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            nn.utils.clip_grad_norm_(
                decoder.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            nn.utils.clip_grad_norm_(
                critic.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            parameter_snapshot, optimizer_snapshot = snapshot_policy_update(
                policy_parameters,
                optimizer,
            )
            optimizer.step()
            action_distribution.clamp_std_parameters_(
                maximum_std=args.current_action_std_cap,
            )

            with torch.no_grad():
                guard_latent, guard_explicit = encoder(flat["history"][guard_indices])
                guard_action_mean = actor(
                    flat["obs"][guard_indices],
                    flat["command"][guard_indices],
                    guard_latent,
                    guard_explicit,
                    (
                        flat["future_reference"][guard_indices]
                        if "future_reference" in flat
                        else None
                    ),
                )
                guard_log_prob = action_distribution.ppo_log_prob(
                    guard_action_mean,
                    flat["pre_tanh_actions"][guard_indices],
                )
                _, post_update_kl = compute_global_approximate_kl(
                    guard_log_prob,
                    flat["old_log_prob"][guard_indices],
                )
                maximum_post_update_kl = max(
                    maximum_post_update_kl,
                    post_update_kl.item(),
                )
                reject_policy_update = (
                    not torch.isfinite(post_update_kl).item()
                    or post_update_kl.item() > args.max_post_update_kl
                )
            if reject_policy_update:
                restore_policy_update(
                    policy_parameters,
                    optimizer,
                    parameter_snapshot,
                    optimizer_snapshot,
                )
                reduce_policy_learning_rate(
                    optimizer,
                    args.minimum_policy_learning_rate,
                )
                rejected_policy_updates += 1
                stopped_for_kl = True
            else:
                accepted_policy_updates += 1

            # KL 和 clip fraction 只用于监控，不参与反向传播。
            with torch.no_grad():
                clip_fraction = (
                    ((probability_ratio - 1.0).abs() > args.policy_clip)
                    .to(torch.float32)
                    .mean()
                )

            batch_metrics = {
                "loss": loss,
                "policy_loss": actor_loss,
                "value_loss": critic_loss,
                "entropy": entropy.mean(),
                "prediction_loss": prediction_loss,
                "estimation_loss": estimation_loss,
                "approx_kl": approximate_kl,
                "clip_fraction": clip_fraction,
                "latent_norm": torch.linalg.vector_norm(
                    latent,
                    dim=-1,
                ).mean(),
                "deterministic_action_abs": torch.tanh(action_mean).abs().mean(),
                "deterministic_action_saturation": (
                    torch.tanh(action_mean).abs() >= 0.98
                )
                .to(torch.float32)
                .mean(),
                "post_update_kl": post_update_kl,
            }
            for name, value in batch_metrics.items():
                metric_sums[name] += value.detach().item()
            update_count += 1
            if reject_policy_update:
                break
        if stopped_for_kl:
            break

    metrics = {name: total / update_count for name, total in metric_sums.items()}
    std_mean, std_max = action_distribution.std_statistics()
    metrics["action_std_mean"] = std_mean
    metrics["action_std_max"] = std_max
    metrics["entropy_coefficient"] = current_entropy_coefficient
    metrics["maximum_approximate_kl"] = maximum_approximate_kl
    metrics["maximum_post_update_kl"] = maximum_post_update_kl
    metrics["stopped_for_kl"] = float(stopped_for_kl)
    metrics["accepted_policy_updates"] = float(accepted_policy_updates)
    metrics["rejected_policy_updates"] = float(rejected_policy_updates)
    metrics["policy_learning_rate"] = policy_learning_rate(optimizer)
    metrics["return_mean"] = value_target_mean.item()
    metrics["return_scale"] = value_target_scale.item()
    return metrics


def save_checkpoint(
    path: Path,
    iteration: int,
    config: ModelConfig,
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    """保存模型与优化器。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "action_distribution_type": action_distribution.distribution_type,
            "input_normalization_type": normalization_type_for_command_dim(
                config.command_dim
            ),
            "future_reference_normalization_type": (
                FUTURE_REFERENCE_NORMALIZATION_TYPE
                if config.future_reference_dim > 0
                else None
            ),
            "auxiliary_objective_type": AUXILIARY_OBJECTIVE_TYPE,
            "model_config": asdict(config),
            "train_args": dict(vars(args)),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "action_distribution": action_distribution.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def run_training(args: argparse.Namespace) -> None:
    """循环执行采样和网络更新。"""

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    start_iteration = 1
    prepare_v8_curriculum(args, start_iteration)
    if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
        args.initial_reference_motion_progress = 1.0
    elif args.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
        if args.reference_motion_origin_iteration is None:
            args.reference_motion_origin_iteration = start_iteration
        args.initial_reference_motion_progress = reference_motion_progress(
            start_iteration,
            args.reference_motion_origin_iteration,
            args.reference_motion_ramp_iterations,
        )
    else:
        args.initial_reference_motion_progress = 0.0
    config = model_config_for_reward_profile(args.reward_profile)
    env = create_environment(args)
    observation, _ = env.reset(seed=args.seed)
    command_abs_mean, command_abs_max = validate_initial_commands(
        observation,
        args.num_envs,
        allow_all_zero=args.command_profile == "stand",
    )
    device = torch.device(args.device)

    # 五个带参数的模块共同组成完整策略：
    # Encoder 提取状态，Decoder 做预测，Actor/Critic 完成 PPO，
    # action_distribution 保存可学习的动作标准差。
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    action_distribution = DiagonalGaussian(
        config.action_dim,
        initial_std=args.initial_action_std,
        min_std=args.min_action_std,
        max_std=args.max_action_std,
    ).to(device)

    # 放进同一个 optimizer，让 PPO loss 和两个辅助 loss 联合训练。
    modules: tuple[nn.Module, ...] = (
        encoder,
        decoder,
        actor,
        critic,
        action_distribution,
    )
    trainable_parameters = [
        parameter for module in modules for parameter in module.parameters()
    ]
    optimizer = create_optimizer(
        encoder,
        decoder,
        actor,
        critic,
        action_distribution,
        args,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {args.task} on {device}: {args.num_envs} envs, "
        f"{args.rollout_steps} steps per rollout, "
        f"iterations={args.iterations}."
    )
    print(
        "Initial command sampling: "
        f"abs_mean={command_abs_mean:.4f}, abs_max={command_abs_max:.4f}, "
        f"scale={args.command_scale:.2f}."
    )

    schedule_start_action_std = min(
        args.initial_action_std,
        args.max_action_std,
    )
    # 一次 iteration = 收集一批新数据 + 用这批数据更新若干次网络。
    for iteration in range(1, args.iterations + 1):
        if args.reward_profile in (
            "p1_walk_stable_v6_phase_rsi_imitation",
            FULL_REFERENCE_REWARD_PROFILE,
        ):
            args.current_imitation_reward_weight = linear_imitation_weight(
                iteration,
                start_iteration,
                args.iterations,
                args.initial_imitation_weight,
                args.final_imitation_weight,
            )
            set_imitation_reward_weight(
                env,
                args.current_imitation_reward_weight,
            )
        else:
            args.current_imitation_reward_weight = 0.0
        if args.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
            args.current_reference_motion_progress = reference_motion_progress(
                iteration,
                args.reference_motion_origin_iteration,
                args.reference_motion_ramp_iterations,
            )
            set_reference_motion_progress(
                env,
                args.current_reference_motion_progress,
            )
        elif args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
            args.current_reference_motion_progress = 1.0
        else:
            args.current_reference_motion_progress = 0.0
        if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
            args.current_task_mix_beta = v8_task_mix_beta(
                iteration,
                args.v8_curriculum_origin_iteration,
            )
            args.current_reference_state_initialization_probability = (
                v8_reference_state_initialization_probability(
                    iteration,
                    args.v8_curriculum_origin_iteration,
                )
            )
            set_task_mix_beta(env, args.current_task_mix_beta)
            set_reference_state_initialization_probability(
                env,
                args.current_reference_state_initialization_probability,
            )
        else:
            args.current_task_mix_beta = 0.0
            args.current_reference_state_initialization_probability = (
                args.reference_state_initialization_probability
            )
        schedule_progress = (iteration - 1) / max(args.iterations - 1, 1)
        args.current_entropy_coef = args.entropy_coef + schedule_progress * (
            args.final_entropy_coef - args.entropy_coef
        )
        args.current_action_std_cap = schedule_start_action_std + schedule_progress * (
            args.final_action_std - schedule_start_action_std
        )
        action_distribution.clamp_std_parameters_(
            maximum_std=args.current_action_std_cap,
        )
        if iteration == 1:
            print("First rollout collection started.")
        # 第一阶段：使用当前策略与环境交互。
        rollout, observation = collect_rollout(
            env=env,
            observation=observation,
            encoder=encoder,
            actor=actor,
            critic=critic,
            action_distribution=action_distribution,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        if iteration == 1:
            print("First rollout collection completed.")
        # 第二阶段：固定这批 rollout，执行 PPO mini-batch 更新。
        metrics = update_models(
            rollout=rollout,
            encoder=encoder,
            decoder=decoder,
            actor=actor,
            critic=critic,
            action_distribution=action_distribution,
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            args=args,
        )
        if iteration == 1:
            print("First PPO update completed.")

        if iteration == 1 or iteration % 10 == 0:
            mean_reward = rollout["rewards"].mean().item()
            termination_fraction = rollout["terminated"].float().mean().item()
            timeout_fraction = rollout["truncated"].float().mean().item()
            clipped_inputs, input_count = normalized_input_clip_counts(rollout)
            normalized_clip_fraction = clipped_inputs.item() / input_count.item()
            print(
                f"iteration={iteration:05d} "
                f"reward/step={mean_reward:+.4f} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"return={metrics['return_mean']:.3f}/"
                f"{metrics['return_scale']:.3f} "
                f"prediction={metrics['prediction_loss']:.4f} "
                f"estimation={metrics['estimation_loss']:.4f} "
                f"kl={metrics['approx_kl']:.6f} "
                f"max_kl={metrics['maximum_approximate_kl']:.6f} "
                f"post_kl={metrics['maximum_post_update_kl']:.6f} "
                f"kl_stop={int(metrics['stopped_for_kl'])} "
                f"updates={int(metrics['accepted_policy_updates'])}/"
                f"{int(metrics['rejected_policy_updates'])} "
                f"policy_lr={metrics['policy_learning_rate']:.2e} "
                f"std={metrics['action_std_mean']:.4f}/"
                f"{metrics['action_std_max']:.4f} "
                f"latent={metrics['latent_norm']:.3f} "
                f"action={metrics['deterministic_action_abs']:.3f}/"
                f"{metrics['deterministic_action_saturation']:.4f} "
                f"entropy_coef={metrics['entropy_coefficient']:.6f}"
                f" imitation_weight={args.current_imitation_reward_weight:.4f}"
                f" reference_progress={args.current_reference_motion_progress:.3f}"
                f" task_mix_beta={args.current_task_mix_beta:.4f}"
                " rsi_probability="
                f"{args.current_reference_state_initialization_probability:.4f}"
                f" term={termination_fraction:.6f}"
                f" timeout={timeout_fraction:.6f}"
                f" input_clip={normalized_clip_fraction:.6f}"
            )

        # 定期保存，也保证最后一轮一定保存。
        if iteration % args.save_interval == 0 or iteration == args.iterations:
            save_checkpoint(
                path=args.log_dir / f"checkpoint_{iteration:05d}.pt",
                iteration=iteration,
                config=config,
                encoder=encoder,
                decoder=decoder,
                actor=actor,
                critic=critic,
                action_distribution=action_distribution,
                optimizer=optimizer,
                args=args,
            )
    env.close()


if __name__ == "__main__":
    cli_args, simulation_app = parse_args_and_launch_simulator()

    # 无论训练正常结束还是中途报错，都关闭 Isaac Sim。
    try:
        run_training(cli_args)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
