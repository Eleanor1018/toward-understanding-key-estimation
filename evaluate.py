"""Deterministic evaluation for G1 key-estimation checkpoints.

This script keeps the training observation semantics unchanged and reports
task-level quantities that can be interpreted without the PPO loss: command
tracking, falls, posture, base height, action magnitude, and explicit velocity
estimation error.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from checkpoint_io import load_checkpoint
from config import (
    FUTURE_REFERENCE_DIM,
    FUTURE_REFERENCE_NORMALIZATION_TYPE,
    FUTURE_REFERENCE_REWARD_PROFILE,
    FULL_REFERENCE_REWARD_PROFILE,
    ModelConfig,
)
from gait import classify_foot_landings
from motion_reference import (
    REFERENCE_POSE_SCALE,
    REFERENCE_POSE_WEIGHT,
    REFERENCE_VELOCITY_SCALE,
    REFERENCE_VELOCITY_WEIGHT,
    V7_REFERENCE_COMPONENT_FRACTIONS,
    split_imitation_reward_weight,
)
from normalization import (
    LEGACY_NORMALIZATION_TYPE,
    NORMALIZATION_TYPE,
    denormalize_explicit_velocity,
    normalize_observation_batch,
    normalization_type_for_command_dim,
)
from policy import Actor, Encoder
from ppo import DiagonalGaussian


V8_DIAGNOSTIC_SCORE_KEYS = (
    "imitation_score",
    "task_score",
    "mixed_score",
    "core_reward",
    "force_contact_score",
    "signed_contact_score",
    "clearance_score",
    "support_score",
    "landing_score",
    "task_mix_beta",
    "pose_score",
    "joint_velocity_score",
    "foot_position_score",
    "root_height_score",
    "root_velocity_score",
    "planar_tracking_score",
    "directional_progress_score",
    "yaw_tracking_score",
    "upright_score",
    "height_score",
)
V8_FORCE_CONTACT_STATE_NAMES = ("flight", "single_support", "double_support")


def force_contact_state_counts(actual_contact: torch.Tensor) -> torch.Tensor:
    """Count force-threshold flight, single-support, and double-support states."""

    if actual_contact.shape[-1:] != (2,):
        raise ValueError("actual_contact must have shape [..., 2]")
    if actual_contact.dtype != torch.bool:
        raise TypeError("actual_contact must be boolean")
    contact_count = actual_contact.to(torch.int64).sum(dim=-1)
    return torch.stack(
        tuple((contact_count == count).sum() for count in range(3)),
    )


def parse_args_and_launch_simulator() -> tuple[argparse.Namespace, Any]:
    """Parse evaluation arguments and start Isaac Sim."""

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Evaluate Unitree G1 key-estimation checkpoints.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Unitree-G1-29dof-KeyEstimation-v0",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--include-zero-baseline", action="store_true")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--continuous-autoreset",
        action="store_true",
        help="Continue scoring an environment after its first episode ends.",
    )
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
    parser.add_argument("--imitation-reward-weight", type=float, default=0.03)
    parser.add_argument(
        "--reference-state-initialization-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--eval-domain-randomization",
        action="store_true",
        help="Keep training-time physics randomization, latency, and noise enabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/g1_key_estimation_4096/evaluation_2000.json"),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.imitation_reward_weight < 0.0:
        parser.error("--imitation-reward-weight must be non-negative")
    if not 0.0 <= args.reference_state_initialization_probability <= 1.0:
        parser.error("--reference-state-initialization-probability must be in [0, 1]")
    if (
        args.reward_profile
        in (FULL_REFERENCE_REWARD_PROFILE, FUTURE_REFERENCE_REWARD_PROFILE)
        and args.command_profile != "forward_walk"
    ):
        parser.error(f"{args.reward_profile} requires --command-profile forward_walk")
    if not args.include_zero_baseline and not args.checkpoints:
        parser.error("provide --checkpoints or --include-zero-baseline")

    app_launcher = AppLauncher(args, multi_gpu=False)
    return args, app_launcher.app


def create_environment(args: argparse.Namespace) -> Any:
    """Create the same registered task used by training."""

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import g1_env  # noqa: F401

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    env_cfg.command_scale = args.command_scale
    env_cfg.command_profile = args.command_profile
    env_cfg.reward_profile = args.reward_profile
    env_cfg.imitation_reward_weight = args.imitation_reward_weight
    env_cfg.reference_state_initialization_probability = (
        args.reference_state_initialization_probability
    )
    env_cfg.reference_motion_progress = 1.0
    env_cfg.task_mix_beta = 1.0
    env_cfg.resample_commands_on_reset = False
    if not args.eval_domain_randomization:
        env_cfg.events = None
        env_cfg.enable_domain_randomization = False
        env_cfg.observation_noise = False
    return gym.make(args.task, cfg=env_cfg)


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    Encoder,
    Actor,
    ModelConfig,
    int,
    bool,
    str | None,
    str | None,
    dict[str, Any],
]:
    """Load the deterministic encoder/actor path from one checkpoint."""

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    expected_without_command = (93, 103, 29, 3, 50)
    actual = (
        config.obs_dim,
        config.privileged_dim,
        config.action_dim,
        config.explicit_dim,
        config.history_steps,
    )
    if (
        actual != expected_without_command
        or config.command_dim not in (3, 5)
        or config.future_reference_dim not in (0, FUTURE_REFERENCE_DIM)
    ):
        raise RuntimeError(
            "Checkpoint observation contract changed: "
            "expected fixed dimensions "
            f"{expected_without_command} and command 3 or 5, "
            f"got fixed dimensions {actual}, command {config.command_dim}, and "
            f"future reference {config.future_reference_dim}."
        )

    encoder = Encoder(config).to(device)
    actor = Actor(config).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    encoder.eval()
    actor.eval()
    action_distribution_type = checkpoint.get("action_distribution_type")
    if action_distribution_type not in (
        None,
        DiagonalGaussian.distribution_type,
    ):
        raise RuntimeError(
            f"Unsupported checkpoint action distribution: {action_distribution_type}"
        )
    uses_squashed_actions = (
        action_distribution_type == DiagonalGaussian.distribution_type
    )
    normalization_type = checkpoint.get("input_normalization_type")
    if normalization_type not in (
        None,
        LEGACY_NORMALIZATION_TYPE,
        NORMALIZATION_TYPE,
    ):
        raise RuntimeError(
            f"Unsupported checkpoint normalization: {normalization_type}"
        )
    expected_normalization_type = normalization_type_for_command_dim(config.command_dim)
    if config.command_dim == 5 and normalization_type != expected_normalization_type:
        raise RuntimeError(
            "Phase-visible checkpoint must use normalization "
            f"{expected_normalization_type!r}, got {normalization_type!r}"
        )
    future_reference_normalization_type = checkpoint.get(
        "future_reference_normalization_type"
    )
    expected_future_reference_normalization_type = (
        FUTURE_REFERENCE_NORMALIZATION_TYPE if config.future_reference_dim > 0 else None
    )
    if (
        future_reference_normalization_type
        != expected_future_reference_normalization_type
    ):
        raise RuntimeError(
            "Checkpoint future-reference normalization mismatch: "
            f"expected {expected_future_reference_normalization_type!r}, "
            f"got {future_reference_normalization_type!r}"
        )
    train_args = checkpoint.get("train_args", {})
    if not isinstance(train_args, dict):
        raise RuntimeError("Checkpoint train_args must be a dictionary")
    return (
        encoder,
        actor,
        config,
        int(checkpoint["iteration"]),
        uses_squashed_actions,
        normalization_type,
        future_reference_normalization_type,
        train_args,
    )


def resolve_v7_evaluation_curriculum(
    checkpoint_present: bool,
    checkpoint_train_args: dict[str, Any],
    fallback_imitation_weight: float,
) -> tuple[float, float]:
    """Restore the action/reference semantics used by one evaluated policy."""

    if not checkpoint_present:
        return 1.0, fallback_imitation_weight
    checkpoint_profile = checkpoint_train_args.get("reward_profile")
    if checkpoint_profile == FULL_REFERENCE_REWARD_PROFILE:
        progress = checkpoint_train_args.get("current_reference_motion_progress")
        imitation_weight = checkpoint_train_args.get("current_imitation_reward_weight")
        if not isinstance(progress, (float, int)):
            raise RuntimeError("V7 checkpoint omitted reference-motion progress")
        if not isinstance(imitation_weight, (float, int)):
            raise RuntimeError("V7 checkpoint omitted imitation reward weight")
        progress = float(progress)
        imitation_weight = float(imitation_weight)
        if not 0.0 <= progress <= 1.0 or imitation_weight < 0.0:
            raise RuntimeError("V7 checkpoint contains an invalid curriculum state")
        return progress, imitation_weight
    if checkpoint_profile in (
        "p1_walk_stable_v6_phase_rsi",
        "p1_walk_stable_v6_phase_rsi_imitation",
    ):
        return 0.0, fallback_imitation_weight
    raise RuntimeError(
        "V7 evaluation requires a V6 warm-start or V7 checkpoint, got "
        f"{checkpoint_profile!r}"
    )


def resolve_v8_evaluation_curriculum(
    checkpoint_present: bool,
    checkpoint_train_args: dict[str, Any],
) -> tuple[float, float]:
    """Return checkpoint task mix and training RSI for V8 evaluation."""

    if not checkpoint_present:
        return 1.0, 0.0
    checkpoint_profile = checkpoint_train_args.get("reward_profile")
    if checkpoint_profile == FUTURE_REFERENCE_REWARD_PROFILE:
        task_mix_beta = checkpoint_train_args.get("current_task_mix_beta")
        training_rsi = checkpoint_train_args.get(
            "current_reference_state_initialization_probability"
        )
        if not isinstance(task_mix_beta, (float, int)) or not isinstance(
            training_rsi,
            (float, int),
        ):
            raise RuntimeError("V8 checkpoint omitted its curriculum state")
        task_mix_beta = float(task_mix_beta)
        training_rsi = float(training_rsi)
        if (
            not math.isfinite(task_mix_beta)
            or not math.isfinite(training_rsi)
            or not 0.0 <= task_mix_beta <= 1.0
            or not 0.0 <= training_rsi <= 1.0
        ):
            raise RuntimeError("V8 checkpoint contains an invalid curriculum state")
        return task_mix_beta, training_rsi
    if checkpoint_profile == FULL_REFERENCE_REWARD_PROFILE:
        source_rsi = checkpoint_train_args.get(
            "current_reference_state_initialization_probability",
            checkpoint_train_args.get(
                "reference_state_initialization_probability",
                1.0,
            ),
        )
        if (
            not isinstance(source_rsi, (float, int))
            or not math.isfinite(float(source_rsi))
            or not 0.0 <= float(source_rsi) <= 1.0
        ):
            raise RuntimeError("V7 warm-start checkpoint contains invalid RSI metadata")
        return 0.0, float(source_rsi)
    raise RuntimeError(
        "V8 evaluation requires a V7 warm-start or V8 checkpoint, got "
        f"{checkpoint_profile!r}"
    )


@torch.inference_mode()
def evaluate_one(
    env: Any,
    args: argparse.Namespace,
    label: str,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Run one deterministic policy (or a zero-action baseline)."""

    from isaaclab.utils.math import quat_apply, quat_rotate_inverse, yaw_quat

    device = torch.device(args.device)
    encoder: Encoder | None = None
    actor: Actor | None = None
    config: ModelConfig | None = None
    checkpoint_iteration: int | None = None
    uses_squashed_actions = False
    uses_normalized_inputs = False
    input_normalization_type: str | None = None
    future_reference_normalization_type: str | None = None
    checkpoint_train_args: dict[str, Any] = {}
    training_task_mix_beta: float | None = None
    training_rsi_probability: float | None = None
    action_saturation_threshold = 1.0

    if checkpoint_path is not None:
        (
            encoder,
            actor,
            config,
            checkpoint_iteration,
            uses_squashed_actions,
            input_normalization_type,
            future_reference_normalization_type,
            checkpoint_train_args,
        ) = load_policy(checkpoint_path, device)
        uses_normalized_inputs = input_normalization_type is not None
        if uses_squashed_actions:
            action_saturation_threshold = 0.98

    if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
        training_task_mix_beta, training_rsi_probability = (
            resolve_v8_evaluation_curriculum(
                checkpoint_path is not None,
                checkpoint_train_args,
            )
        )
        env.unwrapped.set_task_mix_beta(training_task_mix_beta)
        env.unwrapped.set_reference_state_initialization_probability(
            args.reference_state_initialization_probability
        )
    elif args.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
        checkpoint_reference_progress, checkpoint_imitation_weight = (
            resolve_v7_evaluation_curriculum(
                checkpoint_path is not None,
                checkpoint_train_args,
                args.imitation_reward_weight,
            )
        )
        env.unwrapped.set_reference_motion_progress(checkpoint_reference_progress)
        env.unwrapped.set_imitation_reward_weight(checkpoint_imitation_weight)
    elif checkpoint_train_args.get("reward_profile") in (
        FULL_REFERENCE_REWARD_PROFILE,
        FUTURE_REFERENCE_REWARD_PROFILE,
    ):
        raise RuntimeError(
            "A V7/V8 checkpoint must use its matching evaluation profile"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    observation, _ = env.reset(seed=args.seed)
    if config is not None and observation["command"].shape[-1] != config.command_dim:
        raise RuntimeError(
            "Evaluation profile and checkpoint command dimensions differ: "
            f"environment={observation['command'].shape[-1]}, "
            f"checkpoint={config.command_dim}."
        )
    initial_commands = observation["command"][..., :3].detach().clone()
    initial_command_norm = torch.linalg.vector_norm(initial_commands, dim=-1)
    command_bins = {
        "stand": initial_command_norm < 0.05,
        "forward": initial_commands[:, 0] > 0.05,
        "backward": initial_commands[:, 0] < -0.05,
        "lateral": initial_commands[:, 1].abs() > 0.05,
        "turning": initial_commands[:, 2].abs() > 0.05,
    }

    reward_sum = 0.0
    planar_error_sq_sum = 0.0
    yaw_error_sq_sum = 0.0
    explicit_error_sq_sum = 0.0
    valid_state_count = 0
    explicit_state_count = 0
    upright_sum = 0.0
    height_sum = 0.0
    minimum_height = math.inf
    action_abs_sum = 0.0
    physical_action_delta_abs_sum = 0.0
    action_value_count = 0
    saturated_action_count = 0
    soft_limit_target_clipping_count = 0
    termination_count = 0
    timeout_count = 0
    completed_return_sum = 0.0
    completed_episode_count = 0
    episode_returns = torch.zeros(args.num_envs, device=device)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    successful_timeout = torch.zeros_like(active)
    evaluated_steps_per_env = torch.zeros(
        args.num_envs,
        dtype=torch.int64,
        device=device,
    )
    evaluated_transition_count = 0
    command_sum = torch.zeros(3, dtype=torch.float64, device=device)
    velocity_sum = torch.zeros(3, dtype=torch.float64, device=device)
    command_square_sum = torch.zeros_like(command_sum)
    velocity_square_sum = torch.zeros_like(velocity_sum)
    command_velocity_sum = torch.zeros_like(command_sum)
    directional_tracking = {
        name: {
            "error_sq_sum": 0.0,
            "directional_velocity_sum": 0.0,
            "velocity_ratio_sum": 0.0,
            "correct_direction_count": 0,
            "transition_count": 0,
        }
        for name in ("forward", "backward")
    }
    base_env = env.unwrapped
    last_landing_foot = torch.full(
        (args.num_envs,),
        -1,
        dtype=torch.long,
        device=device,
    )
    steps_since_landing = torch.zeros(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )
    gait_counters = torch.zeros(7, dtype=torch.int64, device=device)
    foot_landing_counts = torch.zeros(2, dtype=torch.int64, device=device)
    imitation_sums = torch.zeros(8, dtype=torch.float64, device=device)
    v8_score_sums = {name: 0.0 for name in V8_DIAGNOSTIC_SCORE_KEYS}
    v8_score_counts = {name: 0 for name in V8_DIAGNOSTIC_SCORE_KEYS}
    v8_physical_sums = torch.zeros(6, 2, dtype=torch.float64, device=device)
    v8_force_contact_state_counts = torch.zeros(
        3,
        dtype=torch.int64,
        device=device,
    )
    v8_diagnostic_count = 0

    start_time = time.perf_counter()
    for _ in range(args.steps):
        if not args.continuous_autoreset and not torch.any(active):
            break
        step_active = (
            torch.ones_like(active) if args.continuous_autoreset else active.clone()
        )
        if encoder is None or actor is None:
            actions = torch.zeros(
                args.num_envs,
                29,
                device=device,
            )
        else:
            network_observation = (
                normalize_observation_batch(observation)
                if uses_normalized_inputs
                else observation
            )
            latent, explicit_estimate = encoder(network_observation["history"])
            future_reference = None
            if config is not None and config.future_reference_dim > 0:
                raw_future_reference = network_observation.get("future_reference")
                if raw_future_reference is None:
                    raise RuntimeError("V8 environment omitted future_reference")
                future_reference = raw_future_reference.flatten(start_dim=-2)
            actions = actor(
                network_observation["obs"],
                network_observation["command"],
                latent,
                explicit_estimate,
                future_reference,
            )
            if uses_squashed_actions:
                actions = DiagonalGaussian.deterministic_action(actions)

        if not args.continuous_autoreset:
            actions = actions.clone()
            actions[torch.logical_not(step_active)] = 0.0

        clipped_actions = actions.clamp(-1.0, 1.0)
        action_abs_sum += clipped_actions[step_active].abs().sum().item()
        action_scales = base_env._action_scales
        physical_action_delta = clipped_actions * action_scales
        physical_action_delta_abs_sum += (
            physical_action_delta[step_active].abs().sum().item()
        )
        action_value_count += int(step_active.sum().item()) * 29
        saturated_action_count += (
            (actions[step_active].abs() >= action_saturation_threshold).sum().item()
        )
        default_joint_position = base_env._robot.data.default_joint_pos[
            :, base_env._joint_ids
        ]
        soft_joint_limits = base_env._robot.data.soft_joint_pos_limits[
            :, base_env._joint_ids
        ]
        unclipped_joint_target = default_joint_position + physical_action_delta
        soft_limit_target_clipping_count += (
            torch.logical_or(
                unclipped_joint_target[step_active]
                < soft_joint_limits[step_active, :, 0],
                unclipped_joint_target[step_active]
                > soft_joint_limits[step_active, :, 1],
            )
            .sum()
            .item()
        )

        next_observation, rewards, terminated, truncated, extras = env.step(actions)
        rewards = rewards.reshape(-1)
        terminated = terminated.reshape(-1)
        truncated = truncated.reshape(-1)
        done = torch.logical_or(terminated, truncated)
        scored_done = torch.logical_and(done, step_active)
        estimator_valid = torch.logical_and(
            step_active,
            torch.logical_not(done),
        )
        gait_active = step_active
        gait_diagnostics = extras.get("gait_diagnostics")
        if gait_diagnostics is None:
            raise RuntimeError("Environment omitted pre-reset gait diagnostics")
        imitation_diagnostics = extras.get("imitation_diagnostics")
        if imitation_diagnostics is None:
            raise RuntimeError("Environment omitted pre-reset imitation diagnostics")
        zero_imitation_metric = torch.zeros_like(
            imitation_diagnostics["pose_similarity"]
        )
        imitation_sums += torch.stack(
            (
                imitation_diagnostics["pose_similarity"][step_active].sum(),
                imitation_diagnostics["velocity_similarity"][step_active].sum(),
                imitation_diagnostics["pose_error"][step_active].sum(),
                imitation_diagnostics["velocity_error"][step_active].sum(),
                imitation_diagnostics.get(
                    "root_height_similarity",
                    zero_imitation_metric,
                )[step_active].sum(),
                imitation_diagnostics.get(
                    "root_velocity_similarity",
                    zero_imitation_metric,
                )[step_active].sum(),
                imitation_diagnostics.get(
                    "foot_position_similarity",
                    zero_imitation_metric,
                )[step_active].sum(),
                imitation_diagnostics.get(
                    "contact_similarity",
                    zero_imitation_metric,
                )[step_active].sum(),
            )
        ).to(torch.float64)
        v8_diagnostics = extras.get("v8_diagnostics")
        if v8_diagnostics is not None:
            active_count = int(step_active.sum().item())
            for name in V8_DIAGNOSTIC_SCORE_KEYS:
                value = v8_diagnostics.get(name)
                if value is not None:
                    if value.shape != (args.num_envs,):
                        raise RuntimeError(
                            f"V8 diagnostic {name!r} has shape {tuple(value.shape)}"
                        )
                    v8_score_sums[name] += value[step_active].sum().item()
                    v8_score_counts[name] += active_count

            physical_names = (
                "actual_contact",
                "target_contact",
                "actual_clearance_m",
                "target_clearance_m",
            )
            physical_values = []
            selected_physical_values = []
            for name in physical_names:
                value = v8_diagnostics.get(name)
                if value is None or value.shape != (args.num_envs, 2):
                    raise RuntimeError(
                        f"V8 diagnostic {name!r} must have shape ({args.num_envs}, 2)"
                    )
                selected = value[step_active]
                selected_physical_values.append(selected)
                physical_values.append(selected.to(torch.float64))
            actual_contact, target_contact, actual_clearance, target_clearance = (
                physical_values
            )
            v8_force_contact_state_counts += force_contact_state_counts(
                selected_physical_values[0]
            )
            clearance_error = actual_clearance - target_clearance
            v8_physical_sums += torch.stack(
                (
                    actual_contact.sum(dim=0),
                    target_contact.sum(dim=0),
                    actual_clearance.sum(dim=0),
                    target_clearance.sum(dim=0),
                    clearance_error.abs().sum(dim=0),
                    clearance_error.square().sum(dim=0),
                )
            )
            v8_diagnostic_count += active_count
        in_contact = gait_diagnostics["in_contact"]
        contact_count = in_contact.to(torch.int64).sum(dim=-1)
        valid_landing = gait_diagnostics["valid_landing"]
        steps_since_landing += gait_active.to(torch.long)
        (
            single_landing,
            alternating_landing,
            repeated_landing,
            next_last_landing_foot,
        ) = classify_foot_landings(valid_landing, last_landing_foot)
        inter_landing_time = steps_since_landing.to(torch.float32) * float(
            base_env.step_dt
        )
        valid_inter_landing_time = torch.logical_and(
            inter_landing_time >= 0.12,
            inter_landing_time <= 0.70,
        )
        alternating_landing = torch.logical_and(
            alternating_landing,
            valid_inter_landing_time,
        )
        repeated_landing = torch.logical_and(
            repeated_landing,
            valid_inter_landing_time,
        )
        scored_single_landing = torch.logical_and(single_landing, gait_active)
        scored_alternating_landing = torch.logical_and(
            alternating_landing,
            gait_active,
        )
        scored_repeated_landing = torch.logical_and(
            repeated_landing,
            gait_active,
        )
        landing_foot = valid_landing.to(torch.int64).argmax(dim=-1)
        gait_counters += torch.stack(
            (
                gait_active.sum(),
                torch.logical_and(contact_count == 1, gait_active).sum(),
                torch.logical_and(contact_count == 2, gait_active).sum(),
                torch.logical_and(contact_count == 0, gait_active).sum(),
                scored_single_landing.sum(),
                scored_alternating_landing.sum(),
                scored_repeated_landing.sum(),
            )
        )
        for foot_index in range(2):
            foot_landing_counts[foot_index] += torch.logical_and(
                scored_single_landing,
                landing_foot == foot_index,
            ).sum()
        last_landing_foot = torch.where(
            gait_active,
            next_last_landing_foot,
            last_landing_foot,
        )
        steps_since_landing = torch.where(
            scored_single_landing,
            torch.zeros_like(steps_since_landing),
            steps_since_landing,
        )
        last_landing_foot = torch.where(
            scored_done,
            torch.full_like(last_landing_foot, -1),
            last_landing_foot,
        )
        steps_since_landing = torch.where(
            scored_done,
            torch.zeros_like(steps_since_landing),
            steps_since_landing,
        )

        reward_sum += rewards[step_active].sum().item()
        evaluated_steps_per_env += step_active.to(torch.int64)
        evaluated_transition_count += int(step_active.sum().item())
        termination_count += int(
            torch.logical_and(terminated, step_active).sum().item()
        )
        timeout_count += int(torch.logical_and(truncated, step_active).sum().item())
        successful_timeout |= torch.logical_and(
            step_active,
            torch.logical_and(truncated, torch.logical_not(terminated)),
        )
        episode_returns[step_active] += rewards[step_active]
        if scored_done.any():
            completed_return_sum += episode_returns[scored_done].sum().item()
            completed_episode_count += int(scored_done.sum().item())
            episode_returns[scored_done] = 0.0
        if not args.continuous_autoreset:
            active[scored_done] = False

        metric_command = next_observation["command"][..., :3].clone()
        metric_body_velocity = next_observation["explicit_target"].clone()
        metric_privileged = next_observation["privileged"].clone()
        terminal_observation = extras.get("terminal_observation")
        if terminal_observation is not None:
            terminal_ids = terminal_observation["env_ids"]
            metric_command[terminal_ids] = terminal_observation["command"][..., :3]
            metric_privileged[terminal_ids] = terminal_observation["privileged"]
            metric_body_velocity[terminal_ids] = terminal_observation["privileged"][
                :, 7:10
            ]
            captured_terminal = torch.zeros_like(done)
            captured_terminal[terminal_ids] = True
            if torch.any(torch.logical_and(scored_done, ~captured_terminal)):
                raise RuntimeError("Environment omitted scored terminal observations")
        elif torch.any(scored_done):
            raise RuntimeError("Environment omitted terminal observations")

        if step_active.any():
            metric_orientation = metric_privileged[:, 3:7]
            metric_world_velocity = quat_apply(
                metric_orientation,
                metric_body_velocity,
            )
            metric_heading_velocity = quat_rotate_inverse(
                yaw_quat(metric_orientation),
                metric_world_velocity,
            )
            metric_world_angular_velocity = quat_apply(
                metric_orientation,
                metric_privileged[:, 10:13],
            )
            command = metric_command[step_active]
            velocity = metric_heading_velocity[step_active]
            privileged = metric_privileged[step_active]
            true_yaw_rate = metric_world_angular_velocity[step_active, 2]

            planar_error_sq_sum += (
                ((command[:, :2] - velocity[:, :2]).square().sum(dim=-1)).sum().item()
            )
            yaw_error_sq_sum += (command[:, 2] - true_yaw_rate).square().sum().item()
            tracked_velocity = torch.stack(
                (velocity[:, 0], velocity[:, 1], true_yaw_rate),
                dim=-1,
            ).to(torch.float64)
            command64 = command.to(torch.float64)
            command_sum += command64.sum(dim=0)
            velocity_sum += tracked_velocity.sum(dim=0)
            command_square_sum += command64.square().sum(dim=0)
            velocity_square_sum += tracked_velocity.square().sum(dim=0)
            command_velocity_sum += (command64 * tracked_velocity).sum(dim=0)
            for name, accumulator in directional_tracking.items():
                bin_active = torch.logical_and(command_bins[name], step_active)
                bin_count = int(bin_active.sum().item())
                if bin_count == 0:
                    continue
                bin_command = metric_command[bin_active, 0]
                bin_velocity = metric_heading_velocity[bin_active, 0]
                directional_velocity = torch.sign(bin_command) * bin_velocity
                accumulator["error_sq_sum"] += (
                    (bin_command - bin_velocity).square().sum().item()
                )
                accumulator["directional_velocity_sum"] += (
                    directional_velocity.sum().item()
                )
                accumulator["velocity_ratio_sum"] += (
                    (directional_velocity / bin_command.abs().clamp_min(1.0e-6))
                    .sum()
                    .item()
                )
                accumulator["correct_direction_count"] += int(
                    (directional_velocity > 0.05).sum().item()
                )
                accumulator["transition_count"] += bin_count
            upright_sum += (-privileged[:, 15]).sum().item()
            base_height = privileged[:, 2]
            height_sum += base_height.sum().item()
            minimum_height = min(
                minimum_height,
                base_height.min().item(),
            )
            valid_state_count += int(step_active.sum().item())

            if encoder is not None and estimator_valid.any():
                network_next_observation = (
                    normalize_observation_batch(next_observation)
                    if uses_normalized_inputs
                    else next_observation
                )
                _, explicit_estimate = encoder(
                    network_next_observation["history"][estimator_valid]
                )
                if uses_normalized_inputs:
                    explicit_estimate = denormalize_explicit_velocity(explicit_estimate)
                estimator_velocity = next_observation["explicit_target"][
                    estimator_valid
                ]
                explicit_error_sq_sum += (
                    ((explicit_estimate - estimator_velocity).square().sum(dim=-1))
                    .sum()
                    .item()
                )
                explicit_state_count += int(estimator_valid.sum().item())

        observation = next_observation

    elapsed_seconds = time.perf_counter() - start_time
    transition_count = evaluated_transition_count
    if valid_state_count == 0:
        raise RuntimeError(
            "Every evaluated transition terminated; no valid locomotion states remain."
        )

    def component_correlations(
        count: int,
        tracked_command_sum: torch.Tensor,
        tracked_velocity_sum: torch.Tensor,
        tracked_command_square_sum: torch.Tensor,
        tracked_velocity_square_sum: torch.Tensor,
        tracked_command_velocity_sum: torch.Tensor,
    ) -> list[float | None]:
        numerator = (
            count * tracked_command_velocity_sum
            - tracked_command_sum * tracked_velocity_sum
        )
        denominator = torch.sqrt(
            (
                count * tracked_command_square_sum - tracked_command_sum.square()
            ).clamp_min(0.0)
            * (
                count * tracked_velocity_square_sum - tracked_velocity_sum.square()
            ).clamp_min(0.0)
        )
        return [
            value / scale if scale > 1.0e-12 else None
            for value, scale in zip(numerator.tolist(), denominator.tolist())
        ]

    correlations = component_correlations(
        valid_state_count,
        command_sum,
        velocity_sum,
        command_square_sum,
        velocity_square_sum,
        command_velocity_sum,
    )

    command_bin_survival = None
    command_bin_tracking = {}
    for name, accumulator in directional_tracking.items():
        count = accumulator["transition_count"]
        command_bin_tracking[name] = {
            "transitions": count,
            "sagittal_velocity_rmse": (
                math.sqrt(accumulator["error_sq_sum"] / count) if count else None
            ),
            "mean_directional_velocity": (
                accumulator["directional_velocity_sum"] / count if count else None
            ),
            "mean_velocity_ratio": (
                accumulator["velocity_ratio_sum"] / count if count else None
            ),
            "correct_direction_fraction": (
                accumulator["correct_direction_count"] / count if count else None
            ),
        }

    survival_adjusted_planar_rmse = None
    survival_adjusted_yaw_rmse = None
    survival_adjusted_correlations = None
    survival_adjusted_command_bin_tracking = None
    if not args.continuous_autoreset:
        fixed_horizon_transition_count = args.num_envs * args.steps
        missed_steps = (
            (args.steps - evaluated_steps_per_env).clamp_min(0).to(torch.float64)
        )
        initial_commands64 = initial_commands.to(torch.float64)
        failure_planar_error_sq_sum = torch.sum(
            missed_steps * initial_commands64[:, :2].square().sum(dim=-1)
        ).item()
        failure_yaw_error_sq_sum = torch.sum(
            missed_steps * initial_commands64[:, 2].square()
        ).item()
        survival_adjusted_planar_rmse = math.sqrt(
            (planar_error_sq_sum + failure_planar_error_sq_sum)
            / fixed_horizon_transition_count
        )
        survival_adjusted_yaw_rmse = math.sqrt(
            (yaw_error_sq_sum + failure_yaw_error_sq_sum)
            / fixed_horizon_transition_count
        )
        fixed_command_sum = initial_commands64.sum(dim=0) * args.steps
        fixed_command_square_sum = initial_commands64.square().sum(dim=0) * args.steps
        survival_adjusted_correlations = component_correlations(
            fixed_horizon_transition_count,
            fixed_command_sum,
            velocity_sum,
            fixed_command_square_sum,
            velocity_square_sum,
            command_velocity_sum,
        )
        survival_adjusted_command_bin_tracking = {}
        for name, accumulator in directional_tracking.items():
            mask = command_bins[name]
            fixed_bin_count = int(mask.sum().item()) * args.steps
            missed_bin_steps = missed_steps[mask]
            bin_commands = initial_commands64[mask, 0]
            failure_error_sq_sum = torch.sum(
                missed_bin_steps * bin_commands.square()
            ).item()
            survival_adjusted_command_bin_tracking[name] = {
                "transitions": fixed_bin_count,
                "observed_transitions": accumulator["transition_count"],
                "sagittal_velocity_rmse": (
                    math.sqrt(
                        (accumulator["error_sq_sum"] + failure_error_sq_sum)
                        / fixed_bin_count
                    )
                    if fixed_bin_count
                    else None
                ),
                "mean_directional_velocity": (
                    accumulator["directional_velocity_sum"] / fixed_bin_count
                    if fixed_bin_count
                    else None
                ),
                "mean_velocity_ratio": (
                    accumulator["velocity_ratio_sum"] / fixed_bin_count
                    if fixed_bin_count
                    else None
                ),
                "correct_direction_fraction": (
                    accumulator["correct_direction_count"] / fixed_bin_count
                    if fixed_bin_count
                    else None
                ),
            }
    survival_fraction = None
    mean_first_episode_return = None
    if not args.continuous_autoreset:
        successful = torch.logical_or(active, successful_timeout)
        command_bin_survival = {}
        for name, mask in command_bins.items():
            count = int(mask.sum().item())
            command_bin_survival[name] = {
                "count": count,
                "survival_fraction": (
                    successful[mask].float().mean().item() if count else None
                ),
            }
        survival_fraction = successful.float().mean().item()
        mean_first_episode_return = (
            completed_return_sum + episode_returns[active].sum().item()
        ) / args.num_envs

    (
        gait_valid_transition_count,
        single_support_count,
        double_support_count,
        flight_count,
        single_landing_count,
        alternating_landing_count,
        repeated_landing_count,
    ) = (int(value) for value in gait_counters.cpu().tolist())
    classified_landing_count = alternating_landing_count + repeated_landing_count
    gait_metrics = {
        "valid_transitions": gait_valid_transition_count,
        "single_support_fraction": (
            single_support_count / gait_valid_transition_count
            if gait_valid_transition_count
            else None
        ),
        "double_support_fraction": (
            double_support_count / gait_valid_transition_count
            if gait_valid_transition_count
            else None
        ),
        "flight_fraction": (
            flight_count / gait_valid_transition_count
            if gait_valid_transition_count
            else None
        ),
        "single_foot_landings": single_landing_count,
        "landing_rate_hz": (
            single_landing_count
            / (gait_valid_transition_count * float(base_env.step_dt))
            if gait_valid_transition_count
            else None
        ),
        "fixed_horizon_alternating_landing_rate_hz": (
            alternating_landing_count
            / (args.num_envs * args.steps * float(base_env.step_dt))
            if not args.continuous_autoreset
            else None
        ),
        "alternating_landing_fraction": (
            alternating_landing_count / classified_landing_count
            if classified_landing_count
            else None
        ),
        "repeated_landing_fraction": (
            repeated_landing_count / classified_landing_count
            if classified_landing_count
            else None
        ),
        "left_landing_fraction": (
            foot_landing_counts[0].item() / single_landing_count
            if single_landing_count
            else None
        ),
        "left_landings": foot_landing_counts[0].item(),
        "right_landing_fraction": (
            foot_landing_counts[1].item() / single_landing_count
            if single_landing_count
            else None
        ),
        "right_landings": foot_landing_counts[1].item(),
        "landing_imbalance_fraction": (
            abs(foot_landing_counts[0].item() - foot_landing_counts[1].item())
            / single_landing_count
            if single_landing_count
            else None
        ),
    }
    (
        reference_pose_similarity_sum,
        reference_velocity_similarity_sum,
        reference_pose_error_sum,
        reference_velocity_error_sum,
        reference_root_height_similarity_sum,
        reference_root_velocity_similarity_sum,
        reference_foot_position_similarity_sum,
        reference_contact_similarity_sum,
    ) = imitation_sums.cpu().tolist()
    reference = base_env._walk_reference
    if args.reward_profile == "p1_walk_stable_v6_phase_rsi_imitation":
        effective_pose_weight, effective_velocity_weight = (
            split_imitation_reward_weight(base_env.cfg.imitation_reward_weight)
        )
    elif args.reward_profile == "p1_walk_stable_v5_imitation":
        effective_pose_weight = REFERENCE_POSE_WEIGHT
        effective_velocity_weight = REFERENCE_VELOCITY_WEIGHT
    elif args.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
        effective_pose_weight = (
            base_env.cfg.imitation_reward_weight
            * V7_REFERENCE_COMPONENT_FRACTIONS["pose"]
        )
        effective_velocity_weight = (
            base_env.cfg.imitation_reward_weight
            * V7_REFERENCE_COMPONENT_FRACTIONS["joint_velocity"]
        )
    else:
        effective_pose_weight = 0.0
        effective_velocity_weight = 0.0
    fixed_horizon_transition_count = args.num_envs * args.steps
    imitation_metrics = {
        "archive_sha256": reference.archive_sha256,
        "source_sha256": reference.source_sha256,
        "source_url": reference.source_url,
        "fps": reference.fps,
        "frame_count": reference.frame_count,
        "duration_s": reference.duration_s,
        "retarget_max_offset": reference.retarget_max_offset,
        "retarget_min_scale": reference.retarget_min_scale,
        "reference_motion_progress": base_env.cfg.reference_motion_progress,
        "natural_forward_speed": (
            base_env._reference_natural_forward_speed
            if args.reward_profile == FULL_REFERENCE_REWARD_PROFILE
            else None
        ),
        "action_scales": base_env._action_scales.detach().cpu().tolist(),
        "pose_weight": effective_pose_weight,
        "velocity_weight": effective_velocity_weight,
        "pose_scale": REFERENCE_POSE_SCALE,
        "velocity_scale": REFERENCE_VELOCITY_SCALE,
        "component_weights": {
            name: (
                base_env.cfg.imitation_reward_weight * fraction
                if args.reward_profile == FULL_REFERENCE_REWARD_PROFILE
                else 0.0
            )
            for name, fraction in V7_REFERENCE_COMPONENT_FRACTIONS.items()
        },
        "mean_pose_similarity": reference_pose_similarity_sum / transition_count,
        "mean_velocity_similarity": (
            reference_velocity_similarity_sum / transition_count
        ),
        "mean_pose_error": reference_pose_error_sum / transition_count,
        "mean_velocity_error": reference_velocity_error_sum / transition_count,
        "mean_root_height_similarity": (
            reference_root_height_similarity_sum / transition_count
        ),
        "mean_root_velocity_similarity": (
            reference_root_velocity_similarity_sum / transition_count
        ),
        "mean_foot_position_similarity": (
            reference_foot_position_similarity_sum / transition_count
        ),
        "mean_contact_similarity": reference_contact_similarity_sum / transition_count,
        "survival_adjusted_pose_similarity": (
            reference_pose_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
        "survival_adjusted_velocity_similarity": (
            reference_velocity_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
        "survival_adjusted_root_height_similarity": (
            reference_root_height_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
        "survival_adjusted_root_velocity_similarity": (
            reference_root_velocity_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
        "survival_adjusted_foot_position_similarity": (
            reference_foot_position_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
        "survival_adjusted_contact_similarity": (
            reference_contact_similarity_sum / fixed_horizon_transition_count
            if not args.continuous_autoreset
            else None
        ),
    }

    v8_metrics = None
    if args.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
        score_means = {
            name: (
                v8_score_sums[name] / v8_score_counts[name]
                if v8_score_counts[name] > 0
                else None
            )
            for name in V8_DIAGNOSTIC_SCORE_KEYS
        }
        physical_metrics: dict[str, Any] | None = None
        if v8_diagnostic_count > 0:
            (
                actual_contact_sum,
                target_contact_sum,
                actual_clearance_sum,
                target_clearance_sum,
                clearance_absolute_error_sum,
                clearance_squared_error_sum,
            ) = v8_physical_sums.cpu()

            def per_foot_mean(values: torch.Tensor) -> dict[str, float]:
                means = values / v8_diagnostic_count
                return {
                    "left": means[0].item(),
                    "right": means[1].item(),
                    "overall": means.mean().item(),
                }

            clearance_mse = clearance_squared_error_sum / v8_diagnostic_count
            force_state_counts = v8_force_contact_state_counts.cpu().tolist()
            force_contact_states = {
                "definition": "per-foot force > 0.05 * robot weight",
                "samples": v8_diagnostic_count,
            }
            for name, count in zip(
                V8_FORCE_CONTACT_STATE_NAMES,
                force_state_counts,
            ):
                force_contact_states[f"{name}_count"] = count
                force_contact_states[f"{name}_fraction"] = count / v8_diagnostic_count
            physical_metrics = {
                "actual_contact_fraction": per_foot_mean(actual_contact_sum),
                "target_contact_fraction": per_foot_mean(target_contact_sum),
                "force_threshold_contact_states": force_contact_states,
                "mean_actual_clearance_m": per_foot_mean(actual_clearance_sum),
                "mean_target_clearance_m": per_foot_mean(target_clearance_sum),
                "mean_clearance_error_m": per_foot_mean(
                    actual_clearance_sum - target_clearance_sum
                ),
                "clearance_mae_m": per_foot_mean(clearance_absolute_error_sum),
                "clearance_rmse_m": {
                    "left": math.sqrt(clearance_mse[0].item()),
                    "right": math.sqrt(clearance_mse[1].item()),
                    "overall": math.sqrt(clearance_mse.mean().item()),
                },
            }
        v8_metrics = {
            "training_task_mix_beta": training_task_mix_beta,
            "training_rsi_probability": training_rsi_probability,
            "evaluation_rsi_probability": (
                args.reference_state_initialization_probability
            ),
            "diagnostic_samples": v8_diagnostic_count,
            "mean_scores": score_means,
            "physical": physical_metrics,
        }

    result: dict[str, Any] = {
        "label": label,
        "checkpoint": (
            str(checkpoint_path.resolve()) if checkpoint_path is not None else None
        ),
        "checkpoint_iteration": checkpoint_iteration,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "transitions": transition_count,
        "mean_first_episode_steps": (
            transition_count / args.num_envs if not args.continuous_autoreset else None
        ),
        "tracking_velocity_frame": "yaw_heading",
        "mean_reward_per_step": reward_sum / transition_count,
        "planar_velocity_vector_rmse": math.sqrt(
            planar_error_sq_sum / valid_state_count
        ),
        "yaw_rate_rmse": math.sqrt(yaw_error_sq_sum / valid_state_count),
        "velocity_command_correlation": correlations,
        "survival_adjusted_planar_velocity_vector_rmse": (
            survival_adjusted_planar_rmse
        ),
        "survival_adjusted_yaw_rate_rmse": survival_adjusted_yaw_rmse,
        "survival_adjusted_velocity_command_correlation": (
            survival_adjusted_correlations
        ),
        "explicit_velocity_vector_rmse": (
            math.sqrt(explicit_error_sq_sum / explicit_state_count)
            if explicit_state_count > 0
            else None
        ),
        "mean_upright": upright_sum / valid_state_count,
        "mean_base_height": height_sum / valid_state_count,
        "minimum_base_height": minimum_height,
        "mean_absolute_clipped_action": (action_abs_sum / action_value_count),
        "mean_absolute_joint_target_delta_rad": (
            physical_action_delta_abs_sum / action_value_count
        ),
        "raw_action_saturation_fraction": (saturated_action_count / action_value_count),
        "soft_limit_target_clipping_fraction": (
            soft_limit_target_clipping_count / action_value_count
        ),
        "action_saturation_threshold": action_saturation_threshold,
        "terminations": termination_count,
        "timeouts": timeout_count,
        "completed_episodes": completed_episode_count,
        "survival_fraction": survival_fraction,
        "command_bin_survival": command_bin_survival,
        "command_bin_tracking": command_bin_tracking,
        "survival_adjusted_command_bin_tracking": (
            survival_adjusted_command_bin_tracking
        ),
        "gait": gait_metrics,
        "imitation": imitation_metrics,
        "v8": v8_metrics,
        "mean_completed_episode_return": (
            completed_return_sum / completed_episode_count
            if completed_episode_count > 0
            else None
        ),
        "mean_first_episode_return": mean_first_episode_return,
        "elapsed_seconds": elapsed_seconds,
        "transitions_per_second": transition_count / elapsed_seconds,
        "model_config": asdict(config) if config is not None else None,
        "input_normalization_type": input_normalization_type,
        "future_reference_normalization_type": future_reference_normalization_type,
        "training_reward_profile": checkpoint_train_args.get("reward_profile"),
    }
    return result


def run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate all requested policies in one simulator process."""

    missing = [path for path in args.checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing evaluation checkpoint(s): "
            + ", ".join(str(path) for path in missing)
        )

    env = create_environment(args)
    results: list[dict[str, Any]] = []
    try:
        if args.include_zero_baseline:
            baseline = evaluate_one(env, args, "zero_action", None)
            results.append(baseline)
            print(json.dumps(baseline, indent=2, sort_keys=True))

        for checkpoint_path in args.checkpoints:
            result = evaluate_one(
                env,
                args,
                checkpoint_path.stem,
                checkpoint_path,
            )
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        env.close()

    report = {
        "task": args.task,
        "seed": args.seed,
        "domain_randomization": args.eval_domain_randomization,
        "continuous_autoreset": args.continuous_autoreset,
        "command_scale": args.command_scale,
        "command_profile": args.command_profile,
        "reward_profile": args.reward_profile,
        "imitation_reward_weight": args.imitation_reward_weight,
        "reference_state_initialization_probability": (
            args.reference_state_initialization_probability
        ),
        "isaacsim_version": version("isaacsim"),
        "isaaclab_version": version("isaaclab"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    cli_args, simulation_app = parse_args_and_launch_simulator()
    try:
        run_evaluation(cli_args)
    finally:
        simulation_app.close()
