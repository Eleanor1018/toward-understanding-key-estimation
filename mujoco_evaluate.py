"""Deterministic MuJoCo rollout for repository G1 policy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from checkpoint_io import load_checkpoint
from config import ModelConfig
from motion_reference import CyclicJointReference
from normalization import DEFAULT_JOINT_POSITIONS, JOINT_EFFORT_LIMITS
from normalization import normalize_command, normalize_obs
from policy import Actor, Encoder
from ppo import DiagonalGaussian


POLICY_DT = 0.01
PHYSICS_DT = 0.001
HISTORY_STEPS = 50
OBS_DIM = 93
ACTION_DIM = 29
SOFT_JOINT_POSITION_LIMIT_FACTOR = 0.90
FOOT_FORCE_CONTACT_THRESHOLD_FRACTION = 0.05
ILLEGAL_CONTACT_FORCE_THRESHOLD_N = 10.0
WALK_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "assets" / "motions" / "g1_walk_mimickit.npz"
)
DEFAULT_SCENE_PATH = Path(
    os.environ.get(
        "MUJOCO_G1_SCENE",
        Path.home() / "mujoco_menagerie" / "unitree_g1" / "scene.xml",
    )
)

JOINT_STIFFNESS = (
    100,
    100,
    100,
    150,
    40,
    40,
    100,
    100,
    100,
    150,
    40,
    40,
    200,
    40,
    40,
    *([40] * 14),
)
JOINT_DAMPING = (
    2,
    2,
    2,
    4,
    2,
    2,
    2,
    2,
    2,
    4,
    2,
    2,
    5,
    5,
    5,
    *([1] * 14),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_inverse_rotate(
    quaternion_wxyz: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    inverse = np.empty(4, dtype=np.float64)
    rotated = np.empty(3, dtype=np.float64)
    mujoco.mju_negQuat(inverse, quaternion_wxyz)
    mujoco.mju_rotVecQuat(rotated, vector, inverse)
    return rotated


def validate_mujoco_contract(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos, qvel, and actuator indices after exact-name validation."""

    qpos_indices = []
    qvel_indices = []
    actuator_indices = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"MuJoCo model omitted joint/actuator {name!r}")
        qpos_indices.append(model.jnt_qposadr[joint_id])
        qvel_indices.append(model.jnt_dofadr[joint_id])
        actuator_indices.append(actuator_id)
    if len(set(actuator_indices)) != ACTION_DIM:
        raise RuntimeError("MuJoCo actuator mapping is not one-to-one")
    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


def configure_pd_actuators(model: mujoco.MjModel, actuator_indices: np.ndarray) -> None:
    stiffness = np.asarray(JOINT_STIFFNESS, dtype=np.float64)
    damping = np.asarray(JOINT_DAMPING, dtype=np.float64)
    effort = np.asarray(JOINT_EFFORT_LIMITS, dtype=np.float64)
    model.actuator_gainprm[actuator_indices, 0] = stiffness
    model.actuator_biasprm[actuator_indices, 1] = -stiffness
    model.actuator_biasprm[actuator_indices, 2] = -damping
    model.actuator_forcelimited[actuator_indices] = 1
    model.actuator_forcerange[actuator_indices, 0] = -effort
    model.actuator_forcerange[actuator_indices, 1] = effort


def soft_joint_position_limits(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    factor: float = SOFT_JOINT_POSITION_LIMIT_FACTOR,
) -> np.ndarray:
    """Build the same center-scaled joint limits used by Isaac Lab."""

    if not 0.0 < factor <= 1.0:
        raise ValueError("soft joint position limit factor must be in (0, 1]")
    limits = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not model.jnt_limited[joint_id]:
            raise RuntimeError(f"MuJoCo joint {name!r} has no finite position range")
        lower, upper = model.jnt_range[joint_id]
        center = 0.5 * (lower + upper)
        half_range = 0.5 * factor * (upper - lower)
        limits.append((center - half_range, center + half_range))
    return np.asarray(limits, dtype=np.float64)


def clamp_joint_position_targets(
    targets: np.ndarray,
    soft_limits: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Clamp one 29-D target and report how many joints were clipped."""

    if targets.shape != (ACTION_DIM,) or soft_limits.shape != (ACTION_DIM, 2):
        raise ValueError("joint targets and soft limits have incompatible shapes")
    clipped = np.clip(targets, soft_limits[:, 0], soft_limits[:, 1])
    return clipped, int(np.count_nonzero(clipped != targets))


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pelvis_body_id: int,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    previous_action: np.ndarray,
) -> torch.Tensor:
    spatial_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        pelvis_body_id,
        spatial_velocity,
        1,
    )
    projected_gravity = quaternion_inverse_rotate(
        data.xquat[pelvis_body_id],
        np.asarray((0.0, 0.0, -1.0)),
    )
    observation = np.concatenate(
        (
            projected_gravity,
            spatial_velocity[:3],
            data.qpos[qpos_indices],
            data.qvel[qvel_indices],
            previous_action,
        )
    ).astype(np.float32)
    if observation.shape != (OBS_DIM,) or not np.isfinite(observation).all():
        raise RuntimeError("MuJoCo produced an invalid 93-D policy observation")
    return torch.from_numpy(observation)


def contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_body_ids: tuple[int, int],
    robot_root_body_id: int,
) -> tuple[np.ndarray, bool]:
    """Return net foot-force magnitudes and Isaac-style illegal contact state."""

    foot_force_vectors = np.zeros((2, 3), dtype=np.float64)
    illegal_body_force_vectors: dict[int, np.ndarray] = {}
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body_a = model.geom_bodyid[contact.geom1]
        body_b = model.geom_bodyid[contact.geom2]
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        world_force_on_b = contact.frame.reshape(3, 3).T @ contact_force[:3]
        for foot_index, body_id in enumerate(foot_body_ids):
            if body_b == body_id:
                foot_force_vectors[foot_index] += world_force_on_b
            elif body_a == body_id:
                foot_force_vectors[foot_index] -= world_force_on_b

        for body_id, force in (
            (body_a, -world_force_on_b),
            (body_b, world_force_on_b),
        ):
            is_robot_body = model.body_rootid[body_id] == robot_root_body_id
            if is_robot_body and body_id not in foot_body_ids:
                illegal_body_force_vectors.setdefault(
                    int(body_id), np.zeros(3, dtype=np.float64)
                )
                illegal_body_force_vectors[int(body_id)] += force

    illegal_contact = any(
        np.linalg.norm(force) > ILLEGAL_CONTACT_FORCE_THRESHOLD_N
        for force in illegal_body_force_vectors.values()
    )
    return np.linalg.norm(foot_force_vectors, axis=-1), illegal_contact


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Encoder, Actor, ModelConfig, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    if config.obs_dim != OBS_DIM or config.command_dim != 5 or config.action_dim != 29:
        raise RuntimeError("MuJoCo rollout currently requires the V7 93/5/29 contract")
    if getattr(config, "future_reference_dim", 0) != 0:
        raise RuntimeError("Use a V7 checkpoint without future-reference inputs")
    encoder = Encoder(config).to(device).eval()
    actor = Actor(config).to(device).eval()
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    if checkpoint.get("action_distribution_type") != DiagonalGaussian.distribution_type:
        raise RuntimeError("MuJoCo rollout requires the tanh Gaussian action contract")
    return encoder, actor, config, checkpoint


@torch.inference_mode()
def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    encoder, actor, config, checkpoint = load_policy(args.checkpoint, device)
    train_args = checkpoint.get("train_args", {})
    progress = float(train_args.get("current_reference_motion_progress", 0.0))

    with np.load(WALK_REFERENCE_PATH, allow_pickle=False) as archive:
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
    reference = CyclicJointReference(WALK_REFERENCE_PATH, joint_names, device)
    default_position_torch = torch.tensor(
        DEFAULT_JOINT_POSITIONS,
        dtype=torch.float32,
        device=device,
    )
    reference.retarget_to_position_envelope(default_position_torch, max_offset=0.22)
    phase_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    reference_sample = reference.sample_full_reference(phase_tensor, progress)
    action_scales = reference_sample.action_scale[0].detach().cpu().numpy()
    natural_speed = float(reference.natural_forward_speed(progress).item())
    phase_rate = args.command_vx / max(natural_speed, 1.0e-6)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    qpos_indices, qvel_indices, actuator_indices = validate_mujoco_contract(
        model,
        joint_names,
    )
    soft_position_limits = soft_joint_position_limits(model, joint_names)
    configure_pd_actuators(model, actuator_indices)
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    foot_body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    robot_weight = float(model.body_mass.sum() * 9.81)
    contact_force_threshold = FOOT_FORCE_CONTACT_THRESHOLD_FRACTION * robot_weight

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (0.0, 0.0, 0.80)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_indices] = np.asarray(DEFAULT_JOINT_POSITIONS)
    data.ctrl[actuator_indices] = np.asarray(DEFAULT_JOINT_POSITIONS)
    mujoco.mj_forward(model, data)

    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    raw_observation = build_observation(
        model,
        data,
        pelvis_body_id,
        qpos_indices,
        qvel_indices,
        previous_action,
    )
    normalized_observation = normalize_obs(raw_observation.to(device))
    history = normalized_observation.expand(HISTORY_STEPS, -1).clone().unsqueeze(0)
    phase = 0.0
    maximum_steps = round(args.duration / POLICY_DT)
    render_stride = max(1, round(1.0 / (args.video_fps * PHYSICS_DT)))
    renderer = None
    writer = None
    if args.video is not None:
        if (
            args.video_width > model.vis.global_.offwidth
            or args.video_height > model.vis.global_.offheight
        ):
            raise ValueError(
                "Requested video size exceeds the model's offscreen framebuffer "
                f"({model.vis.global_.offwidth}x{model.vis.global_.offheight})"
            )
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(
            model, height=args.video_height, width=args.video_width
        )
        writer = imageio.get_writer(args.video, fps=args.video_fps, codec="libx264")

    start_x = float(data.qpos[0])
    support_counts = np.zeros(3, dtype=np.int64)
    initial_foot_forces, _ = contact_diagnostics(
        model,
        data,
        foot_body_ids,
        pelvis_body_id,
    )
    previous_contacts = initial_foot_forces > contact_force_threshold
    air_steps = np.zeros(2, dtype=np.int64)
    landing_counts = np.zeros(2, dtype=np.int64)
    first_fall_step: int | None = None
    first_fall_reason: str | None = None
    action_abs_sum = 0.0
    target_delta_abs_sum = 0.0
    soft_target_clipping_count = 0
    sampled_steps = 0
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 140.0
    camera.elevation = -18.0

    try:
        for policy_step in range(maximum_steps):
            command = torch.tensor(
                [
                    [
                        args.command_vx,
                        0.0,
                        0.0,
                        math.sin(2 * math.pi * phase),
                        math.cos(2 * math.pi * phase),
                    ]
                ],
                dtype=torch.float32,
                device=device,
            )
            latent, explicit = encoder(history)
            action_mean = actor(
                normalized_observation.unsqueeze(0),
                normalize_command(command),
                latent,
                explicit,
            )
            action = torch.tanh(action_mean)[0].cpu().numpy().clip(-1.0, 1.0)
            raw_joint_target = (
                np.asarray(DEFAULT_JOINT_POSITIONS) + action_scales * action
            )
            joint_target, clipped_target_count = clamp_joint_position_targets(
                raw_joint_target,
                soft_position_limits,
            )
            data.ctrl[actuator_indices] = joint_target
            previous_action = action.astype(np.float32)
            action_abs_sum += float(np.abs(action).mean())
            target_delta_abs_sum += float(
                np.abs(joint_target - np.asarray(DEFAULT_JOINT_POSITIONS)).mean()
            )
            soft_target_clipping_count += clipped_target_count

            illegal_contact = False
            foot_forces = np.zeros(2, dtype=np.float64)
            for _ in range(round(POLICY_DT / PHYSICS_DT)):
                mujoco.mj_step(model, data)
                foot_forces, substep_illegal_contact = contact_diagnostics(
                    model,
                    data,
                    foot_body_ids,
                    pelvis_body_id,
                )
                illegal_contact = illegal_contact or substep_illegal_contact
                if renderer is not None and writer is not None:
                    physics_index = round(data.time / PHYSICS_DT)
                    if physics_index % render_stride == 0:
                        camera.lookat[:] = data.xpos[pelvis_body_id]
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(renderer.render())

            contacts = foot_forces > contact_force_threshold
            support_count = int(contacts.sum())
            support_counts[support_count] += 1
            previous_air_steps = air_steps.copy()
            air_steps = np.where(contacts, 0, air_steps + 1)
            landed = contacts & ~previous_contacts
            valid_landed = landed & (previous_air_steps >= round(0.12 / POLICY_DT))
            landing_counts += valid_landed.astype(np.int64)
            previous_contacts = contacts

            raw_observation = build_observation(
                model,
                data,
                pelvis_body_id,
                qpos_indices,
                qvel_indices,
                previous_action,
            )
            normalized_observation = normalize_obs(raw_observation.to(device))
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1] = normalized_observation
            phase = (phase + POLICY_DT * phase_rate / reference.duration_s) % 1.0
            sampled_steps += 1

            projected_gravity_z = float(raw_observation[2].item())
            base_height = float(data.qpos[2])
            if projected_gravity_z > -0.70:
                first_fall_reason = "tipped"
            elif base_height < 0.55 or base_height > 1.20:
                first_fall_reason = "invalid_height"
            elif illegal_contact:
                first_fall_reason = "illegal_body_contact"
            if first_fall_step is None and first_fall_reason is not None:
                first_fall_step = policy_step + 1
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    elapsed = sampled_steps * POLICY_DT
    displacement = float(data.qpos[0]) - start_x
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "model": str(args.model.resolve()),
        "model_config": asdict(config),
        "mujoco_version": mujoco.__version__,
        "torch_version": torch.__version__,
        "duration_requested_s": args.duration,
        "duration_completed_s": elapsed,
        "first_fall_time_s": (
            first_fall_step * POLICY_DT if first_fall_step is not None else None
        ),
        "first_fall_reason": first_fall_reason,
        "survived_full_duration": first_fall_step is None,
        "command_vx_mps": args.command_vx,
        "forward_displacement_m": displacement,
        "mean_forward_velocity_mps": displacement / max(elapsed, POLICY_DT),
        "reference_motion_progress": progress,
        "reference_natural_speed_mps": natural_speed,
        "phase_rate": phase_rate,
        "mean_absolute_action": action_abs_sum / max(sampled_steps, 1),
        "mean_absolute_joint_target_delta_rad": (
            target_delta_abs_sum / max(sampled_steps, 1)
        ),
        "soft_joint_position_limit_factor": SOFT_JOINT_POSITION_LIMIT_FACTOR,
        "soft_limit_target_clipping_fraction": (
            soft_target_clipping_count / max(sampled_steps * ACTION_DIM, 1)
        ),
        "foot_force_contact_threshold_n": contact_force_threshold,
        "support": {
            "flight_fraction": support_counts[0] / max(sampled_steps, 1),
            "single_support_fraction": support_counts[1] / max(sampled_steps, 1),
            "double_support_fraction": support_counts[2] / max(sampled_steps, 1),
        },
        "landings": {
            "left": int(landing_counts[0]),
            "right": int(landing_counts[1]),
            "total": int(landing_counts.sum()),
            "rate_hz": int(landing_counts.sum()) / max(elapsed, POLICY_DT),
        },
        "transfer_dynamics": {
            "quantitative_equivalence_to_isaac": False,
            "model_source": "MuJoCo Menagerie Unitree G1 29-DOF",
            "foot_slide_friction": sorted(
                {
                    float(model.geom_friction[geom_id, 0])
                    for geom_id in range(model.ngeom)
                    if model.geom_bodyid[geom_id] in foot_body_ids
                    and model.geom_contype[geom_id] != 0
                }
            ),
            "actuated_joint_frictionloss": sorted(
                {float(value) for value in model.dof_frictionloss[qvel_indices]}
            ),
            "note": (
                "Policy IO, control rate, PD gains, effort limits, and soft joint "
                "limits match the Isaac evaluation contract; contact/friction and "
                "solver dynamics remain engine-specific."
            ),
        },
        "video": str(args.video.resolve()) if args.video is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.model.is_file():
        parser.error("checkpoint and model must exist")
    if (
        args.duration <= 0.0
        or args.video_fps <= 0
        or args.video_width <= 0
        or args.video_height <= 0
    ):
        parser.error("duration, video FPS, width, and height must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_rollout(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
