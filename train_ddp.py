"""Multi-GPU data-parallel entry point for the existing PPO training code.

The protected single-GPU implementation in ``train.py`` is reused rather than
modified. Each rank owns an Isaac Sim process and a balanced shard of the total
environments. Gradients are weighted by local sample count during backward, so
all GPUs update one shared policy even when the environment count is not evenly
divisible by the number of available GPUs.
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from checkpoint_io import load_checkpoint
import train as single_gpu_train
from config import (
    FUTURE_REFERENCE_NORMALIZATION_TYPE,
    FUTURE_REFERENCE_REWARD_PROFILE,
    FULL_REFERENCE_REWARD_PROFILE,
    ModelConfig,
    model_config_for_reward_profile,
)
from policy import Actor, Critic, Decoder, Encoder
from ppo import DiagonalGaussian


def parse_args_and_launch_simulator() -> tuple[
    argparse.Namespace,
    Any,
    Any,
]:
    """Parse the original hyperparameters plus distributed-run arguments."""

    from isaaclab.app import AppLauncher

    config = ModelConfig()
    parser = argparse.ArgumentParser(
        description="Data-parallel PPO training for the Unitree G1 task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Unitree-G1-29dof-KeyEstimation-v0",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Total environments across all ranks.",
    )
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=2000)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/g1_key_estimation_ddp_8gpu"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a single/DDP checkpoint.",
    )
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=None,
        help="Reset action std after loading a checkpoint and clear its Adam state.",
    )
    parser.add_argument(
        "--reset-policy-optimizer-state",
        action="store_true",
        help=(
            "Clear Adam moments for encoder, actor, and action distribution after "
            "loading a checkpoint. Use this when the reward objective changes."
        ),
    )
    parser.add_argument(
        "--reset-optimizer-state",
        action="store_true",
        help="Clear all resumed Adam moments when the task objective changes.",
    )
    # Isaac Lab 2.0.2 consumes this flag but does not add it to argparse.
    parser.add_argument("--distributed", action="store_true")
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
    if args.resume_action_std is not None:
        if args.resume is None:
            parser.error("--resume-action-std requires --resume")
        if not args.min_action_std <= args.resume_action_std <= args.max_action_std:
            parser.error(
                "--resume-action-std must be between --min-action-std and "
                "--max-action-std"
            )
    if args.reset_policy_optimizer_state and args.resume is None:
        parser.error("--reset-policy-optimizer-state requires --resume")
    if args.reset_optimizer_state and args.resume is None:
        parser.error("--reset-optimizer-state requires --resume")
    if args.reset_optimizer_state and args.reset_policy_optimizer_state:
        parser.error(
            "--reset-optimizer-state and --reset-policy-optimizer-state are mutually exclusive"
        )
    if not args.distributed:
        parser.error("train_ddp.py must be launched with --distributed")

    app_launcher = AppLauncher(args)
    return args, app_launcher, app_launcher.app


def prepare_reference_motion_schedule(args: argparse.Namespace) -> int:
    """Resolve a V7 warm-start or crash-resume curriculum before env creation."""

    planned_start_iteration = 1
    source_train_args: dict[str, Any] = {}
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        planned_start_iteration = int(checkpoint["iteration"]) + 1
        source_train_args = checkpoint.get("train_args", {})
        if not isinstance(source_train_args, dict):
            raise RuntimeError("Resume checkpoint train_args must be a dictionary")

    single_gpu_train.prepare_v8_curriculum(
        args,
        planned_start_iteration,
        source_train_args,
    )
    if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
        args.initial_reference_motion_progress = 1.0
        return planned_start_iteration

    if args.reward_profile != FULL_REFERENCE_REWARD_PROFILE:
        args.initial_reference_motion_progress = 0.0
        return planned_start_iteration

    source_is_v7 = (
        source_train_args.get("reward_profile") == FULL_REFERENCE_REWARD_PROFILE
    )
    if source_is_v7:
        saved_origin = source_train_args.get("reference_motion_origin_iteration")
        saved_ramp = source_train_args.get("reference_motion_ramp_iterations")
        saved_end = source_train_args.get("iterations")
        saved_initial_weight = source_train_args.get("initial_imitation_weight")
        saved_final_weight = source_train_args.get("final_imitation_weight")
        if not isinstance(saved_origin, int) or saved_origin < 1:
            raise RuntimeError("V7 checkpoint omitted its curriculum origin")
        if saved_ramp != args.reference_motion_ramp_iterations:
            raise RuntimeError(
                "V7 resume changed reference-motion ramp iterations: "
                f"checkpoint={saved_ramp}, requested={args.reference_motion_ramp_iterations}"
            )
        if saved_end != args.iterations:
            raise RuntimeError(
                "V7 resume changed its curriculum end: "
                f"checkpoint={saved_end}, requested={args.iterations}"
            )
        for name, saved_value, requested_value in (
            (
                "initial_imitation_weight",
                saved_initial_weight,
                args.initial_imitation_weight,
            ),
            (
                "final_imitation_weight",
                saved_final_weight,
                args.final_imitation_weight,
            ),
        ):
            if not isinstance(saved_value, (float, int)) or not math.isclose(
                float(saved_value),
                float(requested_value),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"V7 resume changed {name}: checkpoint={saved_value}, "
                    f"requested={requested_value}"
                )
        if (
            args.reference_motion_origin_iteration is not None
            and args.reference_motion_origin_iteration != saved_origin
        ):
            raise RuntimeError(
                "V7 resume requested a different curriculum origin: "
                f"checkpoint={saved_origin}, "
                f"requested={args.reference_motion_origin_iteration}"
            )
        args.reference_motion_origin_iteration = saved_origin
        saved_progress = source_train_args.get("current_reference_motion_progress")
        expected_progress = single_gpu_train.reference_motion_progress(
            planned_start_iteration - 1,
            saved_origin,
            saved_ramp,
        )
        if not isinstance(saved_progress, (float, int)) or not math.isclose(
            float(saved_progress),
            expected_progress,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                "V7 checkpoint reference progress is inconsistent with its schedule"
            )
        saved_current_weight = source_train_args.get("current_imitation_reward_weight")
        expected_current_weight = single_gpu_train.linear_imitation_weight(
            planned_start_iteration - 1,
            saved_origin,
            saved_end,
            float(saved_initial_weight),
            float(saved_final_weight),
        )
        if not isinstance(saved_current_weight, (float, int)) or not math.isclose(
            float(saved_current_weight),
            expected_current_weight,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                "V7 checkpoint imitation weight is inconsistent with its schedule"
            )
    elif args.reference_motion_origin_iteration is None:
        args.reference_motion_origin_iteration = planned_start_iteration

    args.initial_reference_motion_progress = single_gpu_train.reference_motion_progress(
        planned_start_iteration,
        args.reference_motion_origin_iteration,
        args.reference_motion_ramp_iterations,
    )
    return planned_start_iteration


def initialize_distributed(
    args: argparse.Namespace,
    app_launcher: Any,
) -> tuple[int, int, int]:
    """Initialize NCCL and convert total environments to a per-rank shard."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = app_launcher.local_rank
    if world_size < 1:
        raise RuntimeError("Distributed world size must be positive")
    total_num_envs = args.num_envs
    environments_per_rank = total_num_envs // world_size
    remainder = total_num_envs % world_size
    if environments_per_rank == 0:
        raise ValueError(
            f"Total num_envs={total_num_envs} is smaller than world_size={world_size}."
        )
    args.total_num_envs = total_num_envs
    args.num_envs = environments_per_rank + int(rank < remainder)
    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank
    args.device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_training_state(
    checkpoint_path: Path,
    device: torch.device,
    config: ModelConfig,
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a compatible checkpoint and return its next iteration."""

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    contract_names = (
        "obs_dim",
        "command_dim",
        "future_reference_dim",
        "privileged_dim",
        "action_dim",
        "explicit_dim",
    )
    expected_contract = tuple(getattr(config, name) for name in contract_names)
    actual_contract = tuple(
        checkpoint["model_config"].get(name, 0)
        if name == "future_reference_dim"
        else checkpoint["model_config"][name]
        for name in contract_names
    )
    if actual_contract != expected_contract:
        raise RuntimeError("Resume checkpoint has an incompatible IO contract")
    if checkpoint.get("action_distribution_type") != (
        action_distribution.distribution_type
    ):
        raise RuntimeError("Resume checkpoint uses an incompatible action distribution")
    expected_normalization_type = single_gpu_train.normalization_type_for_command_dim(
        config.command_dim
    )
    if checkpoint.get("input_normalization_type") != expected_normalization_type:
        raise RuntimeError("Resume checkpoint uses incompatible input normalization")
    expected_future_reference_normalization = (
        FUTURE_REFERENCE_NORMALIZATION_TYPE if config.future_reference_dim > 0 else None
    )
    if (
        checkpoint.get("future_reference_normalization_type")
        != expected_future_reference_normalization
    ):
        raise RuntimeError(
            "Resume checkpoint uses incompatible future-reference normalization"
        )
    if checkpoint.get("auxiliary_objective_type") != (
        single_gpu_train.AUXILIARY_OBJECTIVE_TYPE
    ):
        raise RuntimeError("Resume checkpoint uses an incompatible auxiliary objective")
    checkpoint_args = checkpoint.get("train_args", {})
    action_std_bounds = {
        "min_action_std": math.exp(action_distribution.min_log_std),
        "max_action_std": math.exp(action_distribution.max_log_std),
    }
    for name, expected_value in action_std_bounds.items():
        checkpoint_value = checkpoint_args.get(name)
        if checkpoint_value is None or not math.isclose(
            checkpoint_value,
            expected_value,
            rel_tol=1e-6,
        ):
            raise RuntimeError(
                f"Resume checkpoint uses incompatible {name}: {checkpoint_value}"
            )

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    action_distribution.load_state_dict(
        checkpoint["action_distribution"],
        strict=True,
    )
    requested_learning_rates = {
        parameter_group["name"]: parameter_group["lr"]
        for parameter_group in optimizer.param_groups
    }
    optimizer.load_state_dict(checkpoint["optimizer"])
    for parameter_group in optimizer.param_groups:
        group_name = parameter_group.get("name")
        if group_name not in requested_learning_rates:
            raise RuntimeError(
                f"Resume checkpoint has an unknown optimizer group: {group_name}"
            )
        parameter_group["lr"] = requested_learning_rates[group_name]
    return int(checkpoint["iteration"]) + 1


def clear_optimizer_group_state(
    optimizer: torch.optim.Optimizer,
    group_name: str,
) -> int:
    """Clear Adam moments for one named parameter group and retain its LR."""

    matching_groups = [
        parameter_group
        for parameter_group in optimizer.param_groups
        if parameter_group.get("name") == group_name
    ]
    if len(matching_groups) != 1:
        raise RuntimeError(
            f"Expected one optimizer group named {group_name!r}; "
            f"found {len(matching_groups)}"
        )
    cleared_count = 0
    for parameter in matching_groups[0]["params"]:
        if parameter in optimizer.state:
            optimizer.state.pop(parameter)
            cleared_count += 1
    return cleared_count


def clear_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
    """Clear every resumed Adam moment while preserving parameter groups and LRs."""

    cleared_count = len(optimizer.state)
    optimizer.state.clear()
    return cleared_count


def broadcast_parameters(
    modules: tuple[nn.Module, ...],
) -> None:
    """Make rank zero authoritative before the first distributed update."""

    with torch.no_grad():
        for module in modules:
            for parameter in module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)


def register_gradient_average_hooks(
    trainable_parameters: list[nn.Parameter],
    local_sample_weight: float,
) -> list[Any]:
    """Compute a sample-weighted global gradient on every rank."""

    handles = []

    def average_gradient(gradient: torch.Tensor) -> torch.Tensor:
        weighted_gradient = gradient * local_sample_weight
        dist.all_reduce(weighted_gradient, op=dist.ReduceOp.SUM)
        return weighted_gradient

    for parameter in trainable_parameters:
        if parameter.requires_grad:
            handles.append(parameter.register_hook(average_gradient))
    return handles


def normalize_advantages_globally(
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Normalize advantages over the combined samples from every GPU."""

    statistics = torch.stack(
        (
            advantages.sum(),
            advantages.square().sum(),
            torch.tensor(
                advantages.numel(),
                dtype=advantages.dtype,
                device=advantages.device,
            ),
        )
    )
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    mean = statistics[0] / statistics[2]
    variance = statistics[1] / statistics[2] - mean.square()
    return (advantages - mean) / variance.clamp_min(0.0).sqrt().add(1e-8)


def update_models_distributed(
    rollout: dict[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, float]:
    """Call the existing PPO update with already-global-normalized advantages."""

    rollout["advantages"] = normalize_advantages_globally(rollout["advantages"])
    original_normalize = single_gpu_train.normalize_advantages
    single_gpu_train.normalize_advantages = lambda value: value
    try:
        return single_gpu_train.update_models(rollout=rollout, **kwargs)
    finally:
        single_gpu_train.normalize_advantages = original_normalize


def average_metrics(
    metrics: dict[str, float],
    device: torch.device,
    local_num_envs: int,
    total_num_envs: int,
) -> dict[str, float]:
    """Compute sample-weighted scalar logging metrics over all ranks."""

    names = tuple(metrics)
    values = torch.tensor(
        [metrics[name] for name in names],
        dtype=torch.float64,
        device=device,
    )
    values *= local_num_envs
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= total_num_envs
    return {name: values[index].item() for index, name in enumerate(names)}


def assert_models_synchronized(
    trainable_parameters: list[nn.Parameter],
    iteration: int,
) -> None:
    """Fail loudly if any rank's parameters diverge."""

    signature = torch.stack(
        (
            sum(parameter.detach().float().sum() for parameter in trainable_parameters),
            sum(
                parameter.detach().float().square().sum()
                for parameter in trainable_parameters
            ),
            sum(
                parameter.detach().float().abs().sum()
                for parameter in trainable_parameters
            ),
        )
    )
    minimum = signature.clone()
    maximum = signature.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    relative_difference = (
        (maximum - minimum).abs() / maximum.abs().clamp_min(1.0)
    ).max()
    if relative_difference.item() > 1e-6:
        raise RuntimeError(
            f"DDP parameters diverged at iteration {iteration}: "
            f"relative signature difference={relative_difference.item():.3e}"
        )


def run_training(
    args: argparse.Namespace,
    app_launcher: Any,
) -> None:
    """Run one synchronized PPO policy over all GPU environment shards."""

    rank, world_size, local_rank = initialize_distributed(args, app_launcher)
    device = torch.device(args.device)
    base_seed = args.seed
    rank_seed = base_seed + rank

    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    args.seed = rank_seed
    planned_start_iteration = prepare_reference_motion_schedule(args)
    env = single_gpu_train.create_environment(args)
    observation, _ = env.reset(seed=rank_seed)
    command_abs_mean, command_abs_max = single_gpu_train.validate_initial_commands(
        observation,
        args.num_envs,
        allow_all_zero=args.command_profile == "stand",
    )

    config = model_config_for_reward_profile(args.reward_profile)
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
    optimizer = single_gpu_train.create_optimizer(
        encoder,
        decoder,
        actor,
        critic,
        action_distribution,
        args,
    )

    start_iteration = 1
    if args.resume is not None:
        start_iteration = load_training_state(
            args.resume,
            device,
            config,
            encoder,
            decoder,
            actor,
            critic,
            action_distribution,
            optimizer,
        )
        if start_iteration != planned_start_iteration:
            raise RuntimeError("Resume iteration changed after environment creation")
        if args.resume_action_std is not None:
            action_distribution.reset_std_parameters_(args.resume_action_std)
            optimizer.state.pop(action_distribution.log_std, None)
        if args.reset_optimizer_state:
            cleared_optimizer_state_count = clear_optimizer_state(optimizer)
        elif args.reset_policy_optimizer_state:
            cleared_policy_state_count = clear_optimizer_group_state(
                optimizer,
                "policy",
            )

    broadcast_parameters(modules)
    gradient_hook_handles = register_gradient_average_hooks(
        trainable_parameters,
        args.num_envs / args.total_num_envs,
    )
    assert_models_synchronized(trainable_parameters, start_iteration - 1)

    args.seed = base_seed
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(
            "Distributed training "
            f"{args.task}: world_size={world_size}, "
            f"total_envs={args.total_num_envs}, "
            f"rank0_envs={args.num_envs}, "
            f"rollout_steps={args.rollout_steps}, "
            f"iterations={start_iteration}..{args.iterations}, "
            f"resume={args.resume}."
        )
        print(
            "Initial command sampling: "
            f"abs_mean={command_abs_mean:.4f}, "
            f"abs_max={command_abs_max:.4f}, "
            f"scale={args.command_scale:.2f}."
        )
        if args.resume_action_std is not None:
            print(
                "Reset resumed action std to "
                f"{args.resume_action_std:.4f} with fresh Adam state."
            )
        if args.reset_optimizer_state:
            print(
                "Cleared all resumed Adam state for "
                f"{cleared_optimizer_state_count} parameters."
            )
        elif args.reset_policy_optimizer_state:
            print(
                "Cleared resumed policy Adam state for "
                f"{cleared_policy_state_count} parameters."
            )

    schedule_start_action_std = action_distribution.std_statistics()[1]
    try:
        for iteration in range(start_iteration, args.iterations + 1):
            schedule_progress = (iteration - start_iteration) / max(
                args.iterations - start_iteration,
                1,
            )
            args.current_entropy_coef = args.entropy_coef + schedule_progress * (
                args.final_entropy_coef - args.entropy_coef
            )
            if args.reward_profile in (
                "p1_walk_stable_v6_phase_rsi_imitation",
                FULL_REFERENCE_REWARD_PROFILE,
            ):
                args.current_imitation_reward_weight = (
                    single_gpu_train.linear_imitation_weight(
                        iteration,
                        (
                            args.reference_motion_origin_iteration
                            if args.reward_profile == FULL_REFERENCE_REWARD_PROFILE
                            else start_iteration
                        ),
                        args.iterations,
                        args.initial_imitation_weight,
                        args.final_imitation_weight,
                    )
                )
                single_gpu_train.set_imitation_reward_weight(
                    env,
                    args.current_imitation_reward_weight,
                )
            else:
                args.current_imitation_reward_weight = 0.0
            if args.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
                args.current_reference_motion_progress = (
                    single_gpu_train.reference_motion_progress(
                        iteration,
                        args.reference_motion_origin_iteration,
                        args.reference_motion_ramp_iterations,
                    )
                )
                single_gpu_train.set_reference_motion_progress(
                    env,
                    args.current_reference_motion_progress,
                )
            elif args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
                args.current_reference_motion_progress = 1.0
            else:
                args.current_reference_motion_progress = 0.0
            if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
                args.current_task_mix_beta = single_gpu_train.v8_task_mix_beta(
                    iteration,
                    args.v8_curriculum_origin_iteration,
                )
                args.current_reference_state_initialization_probability = (
                    single_gpu_train.v8_reference_state_initialization_probability(
                        iteration,
                        args.v8_curriculum_origin_iteration,
                    )
                )
                single_gpu_train.set_task_mix_beta(
                    env,
                    args.current_task_mix_beta,
                )
                single_gpu_train.set_reference_state_initialization_probability(
                    env,
                    args.current_reference_state_initialization_probability,
                )
            else:
                args.current_task_mix_beta = 0.0
                args.current_reference_state_initialization_probability = (
                    args.reference_state_initialization_probability
                )
            args.current_action_std_cap = (
                schedule_start_action_std
                + schedule_progress
                * (args.final_action_std - schedule_start_action_std)
            )
            action_distribution.clamp_std_parameters_(
                maximum_std=args.current_action_std_cap,
            )
            if rank == 0 and iteration == start_iteration:
                print("First distributed rollout collection started.")
            rollout, observation = single_gpu_train.collect_rollout(
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
            if rank == 0 and iteration == start_iteration:
                print("First distributed rollout collection completed.")
            metrics = update_models_distributed(
                rollout,
                encoder=encoder,
                decoder=decoder,
                actor=actor,
                critic=critic,
                action_distribution=action_distribution,
                optimizer=optimizer,
                trainable_parameters=trainable_parameters,
                args=args,
            )
            if rank == 0 and iteration == start_iteration:
                print("First distributed PPO update completed.")
            metrics = average_metrics(
                metrics,
                device,
                args.num_envs,
                args.total_num_envs,
            )
            assert_models_synchronized(trainable_parameters, iteration)

            mean_reward = rollout["rewards"].mean().to(torch.float64) * args.num_envs
            dist.all_reduce(mean_reward, op=dist.ReduceOp.SUM)
            mean_reward /= args.total_num_envs
            clipped_inputs, input_count = single_gpu_train.normalized_input_clip_counts(
                rollout
            )
            boundary_counts = torch.stack(
                (
                    rollout["terminated"].sum(),
                    rollout["truncated"].sum(),
                    torch.tensor(
                        rollout["terminated"].numel(),
                        device=device,
                    ),
                    clipped_inputs,
                    input_count,
                )
            ).to(torch.float64)
            dist.all_reduce(boundary_counts, op=dist.ReduceOp.SUM)
            termination_fraction = boundary_counts[0] / boundary_counts[2]
            timeout_fraction = boundary_counts[1] / boundary_counts[2]
            normalized_clip_fraction = boundary_counts[3] / boundary_counts[4]
            if rank == 0 and (iteration == start_iteration or iteration % 10 == 0):
                print(
                    f"iteration={iteration:05d} "
                    f"reward/step={mean_reward.item():+.4f} "
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
                    f" term={termination_fraction.item():.6f}"
                    f" timeout={timeout_fraction.item():.6f}"
                    f" input_clip={normalized_clip_fraction.item():.6f}"
                )

            should_save = (
                iteration % args.save_interval == 0 or iteration == args.iterations
            )
            if should_save and rank == 0:
                single_gpu_train.save_checkpoint(
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
            if should_save:
                dist.barrier()
    finally:
        for handle in gradient_hook_handles:
            handle.remove()
        env.close()


if __name__ == "__main__":
    cli_args, launcher, simulation_app = parse_args_and_launch_simulator()
    try:
        run_training(cli_args, launcher)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        simulation_app.close()
