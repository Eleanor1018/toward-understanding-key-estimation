"""Safe cyclic G1 motion references and DeepMimic-style joint rewards."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REFERENCE_POSE_WEIGHT = 0.05
REFERENCE_VELOCITY_WEIGHT = 0.01
REFERENCE_POSE_SCALE = 2.0
REFERENCE_VELOCITY_SCALE = 0.1
REFERENCE_STATE_INITIALIZATION_PROBABILITY = 0.70
V7_BASE_ACTION_SCALE = 0.25
V7_LEG_ACTION_HEADROOM = 1.05
V7_CONTACT_TRANSITION_HEIGHT_M = 0.025
G1_FOOT_SOLE_OFFSET_M = 0.035
V8_FUTURE_HORIZONS_S = (1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0)
V8_FUTURE_REFERENCE_STEPS = 3
V8_FUTURE_FEATURES_PER_STEP = 21
V8_ANKLE_POSITION_CENTER = (
    (-0.000002326097, 0.118506455, -0.756863752),
    (-0.000002326097, -0.118506455, -0.756863752),
)
V8_ANKLE_POSITION_SCALES = (0.35, 0.15, 0.18)
V8_ROOT_HEIGHT_CENTER_M = 0.75
V8_ROOT_HEIGHT_SCALE_M = 0.08
V7_REFERENCE_COMPONENT_FRACTIONS = {
    "pose": 0.35,
    "joint_velocity": 0.07,
    "root_height": 0.08,
    "root_velocity": 0.10,
    "foot_position": 0.20,
    "contact": 0.20,
}
G1_WALK_ARCHIVE_SHA256 = (
    "06a5c950ec18bfbfb9506841ec7c9042d136d683924b6bb79729eb1da9f79b13"
)
G1_WALK_SOURCE_SHA256 = (
    "0030f5ba1db9497e7c9b511c674aefb820b946416e52ba26faf4813b0e998286"
)

# MimicKit's G1 joint weights with its fixed head body removed. The remaining
# order is exactly G1_JOINT_NAMES: legs, waist, left arm, then right arm.
G1_DEEPMIMIC_DOF_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    0.6,
    0.5,
    0.5,
    1.0,
    1.0,
    1.0,
    0.6,
    0.5,
    0.5,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.6,
    0.5,
    0.5,
    0.5,
    1.0,
    1.0,
    1.0,
    0.6,
    0.5,
    0.5,
    0.5,
)

G1_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Joint origins and fixed rotations are from Unitree's G1 29-DOF rev-1.0
# description. Signs differ only where the right leg mirrors the left leg.
_G1_LEG_JOINT_ORIGINS = (
    (
        (0.0, 0.064452, -0.1027),
        (0.0, 0.052, -0.030465),
        (0.025001, 0.0, -0.12412),
        (-0.078273, 0.0021489, -0.17734),
        (0.0, -0.000094445, -0.30001),
        (0.0, 0.0, -0.017558),
    ),
    (
        (0.0, -0.064452, -0.1027),
        (0.0, -0.052, -0.030465),
        (0.025001, 0.0, -0.12412),
        (-0.078273, -0.0021489, -0.17734),
        (0.0, 0.000094445, -0.30001),
        (0.0, 0.0, -0.017558),
    ),
)
_G1_LEG_FIXED_Y_ROTATIONS = (0.0, -0.1749, 0.0, 0.1749, 0.0, 0.0)
_G1_LEG_JOINT_AXES = (1, 0, 2, 1, 1, 0)


@dataclass(frozen=True)
class G1FullReferenceSample:
    """One batch of scheduled V7 reference targets."""

    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    phase: torch.Tensor
    progress: torch.Tensor
    action_scale: torch.Tensor
    ankle_position_pelvis: torch.Tensor
    ankle_velocity_pelvis: torch.Tensor
    contact_target: torch.Tensor
    contact_state: torch.Tensor
    source_root_position: torch.Tensor
    root_height: torch.Tensor
    root_rotation_exp_map: torch.Tensor
    root_linear_velocity: torch.Tensor


@dataclass(frozen=True)
class G1FullReferenceInitialState:
    """Reference-state initialization data for the scheduled V7 trajectory."""

    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    phase: torch.Tensor
    use_reference: torch.Tensor
    progress: torch.Tensor
    action_scale: torch.Tensor
    ankle_position_pelvis: torch.Tensor
    ankle_velocity_pelvis: torch.Tensor
    contact_target: torch.Tensor
    contact_state: torch.Tensor
    root_height: torch.Tensor
    root_rotation_exp_map: torch.Tensor
    root_linear_velocity: torch.Tensor


@dataclass(frozen=True)
class G1FutureReferenceSample:
    """Three compact future targets at MimicKit-equivalent time horizons."""

    features: torch.Tensor
    phase: torch.Tensor
    joint_position: torch.Tensor
    action_scale: torch.Tensor
    ankle_position_pelvis: torch.Tensor
    contact_target: torch.Tensor
    root_height: torch.Tensor


def _axis_rotation(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Return active rotation matrices about one local Cartesian axis."""

    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    if axis == 0:
        rows = (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, cosine, -sine), dim=-1),
            torch.stack((zero, sine, cosine), dim=-1),
        )
    elif axis == 1:
        rows = (
            torch.stack((cosine, zero, sine), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((-sine, zero, cosine), dim=-1),
        )
    elif axis == 2:
        rows = (
            torch.stack((cosine, -sine, zero), dim=-1),
            torch.stack((sine, cosine, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        )
    else:
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    return torch.stack(rows, dim=-2)


def _g1_leg_kinematics(
    leg_joint_position: torch.Tensor,
    leg_joint_velocity: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if leg_joint_position.shape[-1:] != (12,):
        raise ValueError("leg_joint_position must have shape [..., 12]")
    if not torch.is_floating_point(leg_joint_position):
        raise TypeError("leg_joint_position must be floating point")
    if leg_joint_velocity is not None:
        if leg_joint_velocity.shape != leg_joint_position.shape:
            raise ValueError("leg_joint_velocity must match leg_joint_position")
        if leg_joint_velocity.device != leg_joint_position.device:
            raise ValueError("leg joint position and velocity must share a device")

    batch_shape = leg_joint_position.shape[:-1]
    position_flat = leg_joint_position.reshape(-1, 2, 6)
    batch_count = position_flat.shape[0]
    joint_position = position_flat.reshape(-1, 6)
    origins = leg_joint_position.new_tensor(_G1_LEG_JOINT_ORIGINS)
    origins = origins.unsqueeze(0).expand(batch_count, -1, -1, -1).reshape(-1, 6, 3)

    rotation = (
        torch.eye(
            3,
            dtype=leg_joint_position.dtype,
            device=leg_joint_position.device,
        )
        .expand(joint_position.shape[0], -1, -1)
        .clone()
    )
    ankle_position = leg_joint_position.new_zeros(joint_position.shape[0], 3)

    ankle_velocity: torch.Tensor | None = None
    angular_velocity: torch.Tensor | None = None
    joint_velocity: torch.Tensor | None = None
    if leg_joint_velocity is not None:
        joint_velocity = leg_joint_velocity.reshape(-1, 2, 6).reshape(-1, 6)
        ankle_velocity = torch.zeros_like(ankle_position)
        angular_velocity = torch.zeros_like(ankle_position)

    for joint_index, axis in enumerate(_G1_LEG_JOINT_AXES):
        translated_origin = torch.matmul(
            rotation,
            origins[:, joint_index].unsqueeze(-1),
        ).squeeze(-1)
        ankle_position = ankle_position + translated_origin
        if ankle_velocity is not None and angular_velocity is not None:
            ankle_velocity = ankle_velocity + torch.linalg.cross(
                angular_velocity,
                translated_origin,
                dim=-1,
            )

        fixed_y_rotation = _G1_LEG_FIXED_Y_ROTATIONS[joint_index]
        if fixed_y_rotation != 0.0:
            fixed_angle = torch.full_like(
                joint_position[:, joint_index],
                fixed_y_rotation,
            )
            rotation = torch.matmul(rotation, _axis_rotation(fixed_angle, 1))

        local_axis = leg_joint_position.new_zeros(joint_position.shape[0], 3)
        local_axis[:, axis] = 1.0
        axis_pelvis = torch.matmul(rotation, local_axis.unsqueeze(-1)).squeeze(-1)
        if angular_velocity is not None and joint_velocity is not None:
            angular_velocity = (
                angular_velocity
                + axis_pelvis * joint_velocity[:, joint_index : joint_index + 1]
            )
        rotation = torch.matmul(
            rotation,
            _axis_rotation(joint_position[:, joint_index], axis),
        )

    output_shape = (*batch_shape, 2, 3)
    ankle_position = ankle_position.reshape(output_shape)
    if ankle_velocity is not None:
        ankle_velocity = ankle_velocity.reshape(output_shape)
    return ankle_position, ankle_velocity


def g1_leg_forward_kinematics(leg_joint_position: torch.Tensor) -> torch.Tensor:
    """Return left/right ankle-roll origins in the pelvis frame.

    The input follows the first twelve entries of the repository's G1 joint
    contract: six left-leg joints followed by six right-leg joints.
    """

    ankle_position, _ = _g1_leg_kinematics(leg_joint_position, None)
    return ankle_position


def g1_leg_forward_velocity(
    leg_joint_position: torch.Tensor,
    leg_joint_velocity: torch.Tensor,
) -> torch.Tensor:
    """Return exact FK ankle linear velocities in the pelvis frame."""

    _, ankle_velocity = _g1_leg_kinematics(
        leg_joint_position,
        leg_joint_velocity,
    )
    assert ankle_velocity is not None
    return ankle_velocity


def g1_ankle_contact_targets(
    ankle_position_pelvis: torch.Tensor,
    transition_height: float = V7_CONTACT_TRANSITION_HEIGHT_M,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smooth and binary contacts from height above the lower ankle."""

    if ankle_position_pelvis.shape[-2:] != (2, 3):
        raise ValueError("ankle_position_pelvis must have shape [..., 2, 3]")
    if not math.isfinite(transition_height) or transition_height <= 0.0:
        raise ValueError("transition_height must be finite and positive")
    ankle_height = ankle_position_pelvis[..., 2]
    relative_height = ankle_height - ankle_height.amin(dim=-1, keepdim=True)
    linear_contact = (1.0 - relative_height / transition_height).clamp(0.0, 1.0)
    smooth_contact = linear_contact.square() * (3.0 - 2.0 * linear_contact)
    binary_contact = smooth_contact >= 0.5
    return smooth_contact, binary_contact


def build_g1_future_reference_features(
    joint_position: torch.Tensor,
    action_scale: torch.Tensor,
    trajectory_center: torch.Tensor,
    ankle_position_pelvis: torch.Tensor,
    contact_target: torch.Tensor,
    root_height: torch.Tensor,
) -> torch.Tensor:
    """Build normalized ``[..., 3, 21]`` V8 future-reference features."""

    if joint_position.ndim < 3 or joint_position.shape[-2:] != (
        V8_FUTURE_REFERENCE_STEPS,
        29,
    ):
        raise ValueError("joint_position must have shape [..., 3, 29]")
    if action_scale.shape != joint_position.shape:
        raise ValueError("action_scale must match joint_position")
    if trajectory_center.shape != (29,):
        raise ValueError("trajectory_center must have shape [29]")
    batch_shape = joint_position.shape[:-2]
    expected_ankle_shape = (*batch_shape, V8_FUTURE_REFERENCE_STEPS, 2, 3)
    if ankle_position_pelvis.shape != expected_ankle_shape:
        raise ValueError(
            f"ankle_position_pelvis must have shape {expected_ankle_shape}"
        )
    expected_contact_shape = (*batch_shape, V8_FUTURE_REFERENCE_STEPS, 2)
    if contact_target.shape != expected_contact_shape:
        raise ValueError(f"contact_target must have shape {expected_contact_shape}")
    expected_root_shape = (*batch_shape, V8_FUTURE_REFERENCE_STEPS)
    if root_height.shape != expected_root_shape:
        raise ValueError(f"root_height must have shape {expected_root_shape}")
    tensors = (
        action_scale,
        trajectory_center,
        ankle_position_pelvis,
        contact_target,
        root_height,
    )
    if any(value.device != joint_position.device for value in tensors):
        raise ValueError("All future-reference tensors must share a device")
    if not all(torch.is_floating_point(value) for value in (joint_position, *tensors)):
        raise TypeError("Future-reference tensors must be floating point")
    if not all(
        torch.isfinite(value).all().item() for value in (joint_position, *tensors)
    ):
        raise ValueError("Future-reference tensors must contain only finite values")
    if torch.any(action_scale <= 0.0).item():
        raise ValueError("action_scale must be positive")
    if torch.any((contact_target < 0.0) | (contact_target > 1.0)).item():
        raise ValueError("contact_target must be in [0, 1]")

    leg_action_coordinate = (
        joint_position[..., :12] - trajectory_center[:12]
    ) / action_scale[..., :12]
    ankle_center = joint_position.new_tensor(V8_ANKLE_POSITION_CENTER)
    ankle_scale = joint_position.new_tensor(V8_ANKLE_POSITION_SCALES)
    normalized_ankle = (ankle_position_pelvis - ankle_center) / ankle_scale
    signed_contact = 2.0 * contact_target - 1.0
    normalized_height = (root_height - V8_ROOT_HEIGHT_CENTER_M) / V8_ROOT_HEIGHT_SCALE_M
    features = torch.cat(
        (
            leg_action_coordinate,
            normalized_ankle.flatten(start_dim=-2),
            signed_contact,
            normalized_height.unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape[-1] != V8_FUTURE_FEATURES_PER_STEP:
        raise RuntimeError("V8 future-reference feature count changed")
    return features


def cyclic_phase_features(phase: torch.Tensor) -> torch.Tensor:
    """Encode normalized cyclic phase as ``[sin(2 pi phase), cos(2 pi phase)]``."""

    if not torch.is_floating_point(phase):
        raise TypeError("phase must be a floating-point tensor")
    angle = 2.0 * torch.pi * torch.remainder(phase, 1.0)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def phase_visible_command(
    physical_command: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Append cyclic phase features without changing the physical command."""

    if physical_command.shape[-1:] != (3,):
        raise ValueError("physical_command must have shape [..., 3]")
    if phase.shape != physical_command.shape[:-1]:
        raise ValueError("phase shape must match physical_command batch dimensions")
    return torch.cat((physical_command, cyclic_phase_features(phase)), dim=-1)


def split_imitation_reward_weight(total_weight: float) -> tuple[float, float]:
    """Split an imitation-reward budget 5:1 between pose and velocity."""

    if not math.isfinite(total_weight) or total_weight < 0.0:
        raise ValueError(
            "total imitation reward weight must be finite and non-negative"
        )
    unit = total_weight / 6.0
    return 5.0 * unit, unit


def sanitize_root_pose(
    candidate_pose: torch.Tensor,
    fallback_pose: torch.Tensor,
    minimum_height: torch.Tensor | float,
) -> torch.Tensor:
    """Return a finite, upright-safe root pose with a unit quaternion.

    Reference-state initialization in this project imports joint state only.
    This guard keeps the simulator-derived root pose valid even if subsequent
    reset randomization produces a non-finite value or invalid quaternion.
    """

    if candidate_pose.ndim != 2 or candidate_pose.shape[-1] != 7:
        raise ValueError("candidate_pose must have shape [N, 7]")
    if fallback_pose.shape != candidate_pose.shape:
        raise ValueError("fallback_pose must have the same shape as candidate_pose")
    if candidate_pose.device != fallback_pose.device:
        raise ValueError("candidate_pose and fallback_pose must share a device")

    safe_pose = candidate_pose.clone()
    safe_pose[:, :3] = torch.where(
        torch.isfinite(candidate_pose[:, :3]),
        candidate_pose[:, :3],
        fallback_pose[:, :3],
    )
    minimum_height_tensor = torch.as_tensor(
        minimum_height,
        dtype=safe_pose.dtype,
        device=safe_pose.device,
    )
    if minimum_height_tensor.ndim > 1 or (
        minimum_height_tensor.ndim == 1
        and minimum_height_tensor.shape != safe_pose.shape[:1]
    ):
        raise ValueError("minimum_height must be scalar or have shape [N]")
    safe_pose[:, 2] = torch.where(
        safe_pose[:, 2] >= minimum_height_tensor,
        safe_pose[:, 2],
        torch.maximum(fallback_pose[:, 2], minimum_height_tensor),
    )

    identity = torch.zeros_like(candidate_pose[:, 3:7])
    identity[:, 0] = 1.0

    fallback_quaternion = fallback_pose[:, 3:7]
    fallback_norm = torch.linalg.vector_norm(fallback_quaternion, dim=-1)
    fallback_valid = torch.logical_and(
        torch.isfinite(fallback_quaternion).all(dim=-1),
        fallback_norm > 1.0e-6,
    )
    normalized_fallback = torch.where(
        fallback_valid.unsqueeze(-1),
        fallback_quaternion / fallback_norm.clamp_min(1.0e-6).unsqueeze(-1),
        identity,
    )

    candidate_quaternion = candidate_pose[:, 3:7]
    candidate_norm = torch.linalg.vector_norm(candidate_quaternion, dim=-1)
    candidate_valid = torch.logical_and(
        torch.isfinite(candidate_quaternion).all(dim=-1),
        candidate_norm > 1.0e-6,
    )
    safe_pose[:, 3:7] = torch.where(
        candidate_valid.unsqueeze(-1),
        candidate_quaternion / candidate_norm.clamp_min(1.0e-6).unsqueeze(-1),
        normalized_fallback,
    )
    return safe_pose


class CyclicJointReference:
    """Linearly interpolate a validated, cyclic joint trajectory on a device."""

    def __init__(
        self,
        path: Path,
        expected_joint_names: Sequence[str],
        device: torch.device | str,
        expected_archive_sha256: str | None = None,
        expected_source_sha256: str | None = None,
    ) -> None:
        archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            expected_archive_sha256 is not None
            and archive_sha256 != expected_archive_sha256
        ):
            raise ValueError(
                "Walking-reference archive checksum mismatch: "
                f"expected {expected_archive_sha256}, got {archive_sha256}"
            )
        with np.load(path, allow_pickle=False) as data:
            format_version = int(data["format_version"])
            fps = float(data["fps"])
            loop_mode = int(data["loop_mode"])
            joint_names = tuple(str(name) for name in data["joint_names"].tolist())
            joint_positions = np.asarray(data["joint_pos"], dtype=np.float32)
            root_positions = np.asarray(data["root_pos"], dtype=np.float32)
            root_rotation_exp_map = np.asarray(
                data["root_rot_exp_map"],
                dtype=np.float32,
            )
            source_sha256 = str(data["source_sha256"])
            source_url = str(data["source_url"])

        if format_version != 1:
            raise ValueError(f"Unsupported motion format version: {format_version}")
        if loop_mode != 1:
            raise ValueError("The walking reference must use cyclic wrap mode")
        if fps <= 0.0:
            raise ValueError("The walking reference FPS must be positive")
        if joint_names != tuple(expected_joint_names):
            raise ValueError(
                "Walking-reference joint order mismatch: "
                f"expected {tuple(expected_joint_names)}, got {joint_names}"
            )
        if joint_positions.ndim != 2 or joint_positions.shape[0] < 2:
            raise ValueError("Walking-reference joint_pos must have shape [F>=2, J]")
        if joint_positions.shape[1] != len(joint_names):
            raise ValueError("Walking-reference joint count does not match metadata")
        if not np.isfinite(joint_positions).all():
            raise ValueError("Walking reference contains non-finite joint positions")
        expected_root_shape = (joint_positions.shape[0], 3)
        if root_positions.shape != expected_root_shape:
            raise ValueError(
                "Walking-reference root_pos must have shape "
                f"{expected_root_shape}, got {root_positions.shape}"
            )
        if root_rotation_exp_map.shape != expected_root_shape:
            raise ValueError(
                "Walking-reference root_rot_exp_map must have shape "
                f"{expected_root_shape}, got {root_rotation_exp_map.shape}"
            )
        if not np.isfinite(root_positions).all():
            raise ValueError("Walking reference contains non-finite root positions")
        if not np.isfinite(root_rotation_exp_map).all():
            raise ValueError("Walking reference contains non-finite root rotations")
        if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
        ):
            raise ValueError(
                "Walking-reference source checksum mismatch: "
                f"expected {expected_source_sha256}, got {source_sha256}"
            )

        self.path = path
        self.archive_sha256 = archive_sha256
        self.fps = fps
        self.frame_count = joint_positions.shape[0]
        self.duration_s = (self.frame_count - 1) / fps
        self.joint_names = joint_names
        self.source_sha256 = source_sha256
        self.source_url = source_url
        self.retarget_max_offset: float | None = None
        self.retarget_min_scale = 1.0
        self.source_joint_positions = torch.as_tensor(
            joint_positions.copy(),
            dtype=torch.float32,
            device=device,
        )
        self.source_root_positions = torch.as_tensor(
            root_positions.copy(),
            dtype=torch.float32,
            device=device,
        )
        self.source_root_rotation_exp_map = torch.as_tensor(
            root_rotation_exp_map.copy(),
            dtype=torch.float32,
            device=device,
        )
        # These aliases preserve the V6 API. Retargeting replaces them with the
        # compressed trajectory while the immutable source tensors remain intact.
        self.joint_positions = self.source_joint_positions.clone()
        self.compressed_joint_positions: torch.Tensor | None = None
        self.compressed_joint_velocities: torch.Tensor | None = None
        self.full_joint_positions: torch.Tensor | None = None
        self.full_joint_velocities: torch.Tensor | None = None
        self.trajectory_center: torch.Tensor | None = None
        self.base_action_scales: torch.Tensor | None = None
        self.full_action_scales: torch.Tensor | None = None
        self._update_velocities()

    def _finite_difference(self, values: torch.Tensor) -> torch.Tensor:
        velocity = torch.empty_like(values)
        velocity[:-1] = (values[1:] - values[:-1]) * self.fps
        velocity[-1] = velocity[-2]
        return velocity

    def _update_velocities(self) -> None:
        self.joint_velocities = self._finite_difference(self.joint_positions)

    def retarget_to_position_envelope(
        self,
        center: torch.Tensor,
        max_offset: float,
        *,
        action_scale: float = V7_BASE_ACTION_SCALE,
        leg_action_headroom: float = V7_LEG_ACTION_HEADROOM,
    ) -> None:
        """Build the V6 compressed and V7 full-leg reference trajectories."""

        if center.shape != self.source_joint_positions.shape[-1:]:
            raise ValueError("Reference center must have shape [J]")
        if not math.isfinite(max_offset) or max_offset <= 0.0:
            raise ValueError("Reference max_offset must be finite and positive")
        if not math.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError("Reference action_scale must be finite and positive")
        if max_offset > action_scale:
            raise ValueError("Reference max_offset must not exceed action_scale")
        if not math.isfinite(leg_action_headroom) or leg_action_headroom < 1.0:
            raise ValueError("leg_action_headroom must be finite and at least one")
        if len(self.joint_names) < len(G1_LEG_JOINT_NAMES) or (
            self.joint_names[: len(G1_LEG_JOINT_NAMES)] != G1_LEG_JOINT_NAMES
        ):
            raise ValueError("Walking reference does not use the G1 leg joint order")
        center = center.to(
            device=self.source_joint_positions.device,
            dtype=self.source_joint_positions.dtype,
        )
        offset = self.source_joint_positions - center
        maximum_joint_offset = offset.abs().amax(dim=0).clamp_min(1.0e-8)
        scale = (max_offset / maximum_joint_offset).clamp(max=1.0)
        compressed_position = center + offset * scale
        full_position = compressed_position.clone()
        full_position[:, :12] = self.source_joint_positions[:, :12]

        base_action_scales = torch.full_like(center, action_scale)
        full_action_scales = base_action_scales.clone()
        full_action_scales[:12] = torch.maximum(
            full_action_scales[:12],
            leg_action_headroom * maximum_joint_offset[:12],
        )

        self.trajectory_center = center.clone()
        self.compressed_joint_positions = compressed_position
        self.compressed_joint_velocities = self._finite_difference(compressed_position)
        self.full_joint_positions = full_position
        self.full_joint_velocities = self._finite_difference(full_position)
        self.base_action_scales = base_action_scales
        self.full_action_scales = full_action_scales
        self.joint_positions = self.compressed_joint_positions
        self.joint_velocities = self.compressed_joint_velocities
        self.retarget_max_offset = max_offset
        self.retarget_min_scale = float(scale.min().item())

    def _frame_coordinates(
        self,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if phase.ndim != 1:
            raise ValueError("phase must have shape [N]")
        if phase.device != self.joint_positions.device:
            raise ValueError("phase and reference trajectory must share a device")
        normalized_phase = torch.remainder(phase, 1.0)
        frame_position = normalized_phase * (self.frame_count - 1)
        frame_index_0 = torch.floor(frame_position).to(torch.long)
        frame_index_1 = torch.clamp(frame_index_0 + 1, max=self.frame_count - 1)
        blend = frame_position - frame_index_0
        return normalized_phase, frame_index_0, frame_index_1, blend

    @staticmethod
    def _interpolate_frames(
        frames: torch.Tensor,
        frame_index_0: torch.Tensor,
        frame_index_1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        return torch.lerp(
            frames[frame_index_0],
            frames[frame_index_1],
            blend.to(frames.dtype).unsqueeze(-1),
        )

    def _scheduled_progress(
        self,
        progress: torch.Tensor | float,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        progress_tensor = torch.as_tensor(
            progress,
            dtype=self.joint_positions.dtype,
            device=self.joint_positions.device,
        )
        if progress_tensor.ndim == 0:
            progress_tensor = progress_tensor.expand(phase.shape)
        elif progress_tensor.shape != phase.shape:
            raise ValueError("progress must be scalar or have the same shape as phase")
        if not torch.isfinite(progress_tensor).all().item():
            raise ValueError("progress must contain only finite values")
        return progress_tensor.clamp(0.0, 1.0)

    def phase_at_time(
        self,
        time_s: torch.Tensor,
        phase_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized cyclic phase at elapsed times and optional offsets."""

        if time_s.ndim != 1:
            raise ValueError("time_s must have shape [N]")
        if phase_offset is None:
            phase_offset = torch.zeros_like(time_s)
        elif phase_offset.shape != time_s.shape:
            raise ValueError("phase_offset must have the same shape as time_s")
        reference_time_s = time_s + phase_offset * self.duration_s
        return torch.remainder(reference_time_s, self.duration_s) / self.duration_s

    def sample_phase(
        self,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return joint position and velocity at normalized cyclic phases."""

        (
            normalized_phase,
            frame_index_0,
            frame_index_1,
            blend,
        ) = self._frame_coordinates(phase)
        position = self._interpolate_frames(
            self.joint_positions,
            frame_index_0,
            frame_index_1,
            blend,
        )
        velocity = self.joint_velocities[frame_index_0]
        return position, velocity, normalized_phase

    def _assemble_full_reference(
        self,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        phase: torch.Tensor,
        progress: torch.Tensor,
        action_scale: torch.Tensor,
        source_root_position: torch.Tensor,
        root_rotation_exp_map: torch.Tensor,
        contact_target_override: torch.Tensor | None = None,
    ) -> G1FullReferenceSample:
        ankle_position = g1_leg_forward_kinematics(joint_position[:, :12])
        ankle_velocity = g1_leg_forward_velocity(
            joint_position[:, :12],
            joint_velocity[:, :12],
        )
        if contact_target_override is None:
            contact_target, contact_state = g1_ankle_contact_targets(ankle_position)
        else:
            if contact_target_override.shape != ankle_position.shape[:-1]:
                raise ValueError("contact target override must have shape [N, 2]")
            contact_target = contact_target_override
            contact_state = contact_target >= 0.5
        root_height = G1_FOOT_SOLE_OFFSET_M - ankle_position[..., 2].amin(dim=-1)
        contact_weight = contact_target / contact_target.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-8)
        root_linear_velocity = -torch.sum(
            contact_weight.unsqueeze(-1) * ankle_velocity,
            dim=-2,
        )
        return G1FullReferenceSample(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            phase=phase,
            progress=progress,
            action_scale=action_scale,
            ankle_position_pelvis=ankle_position,
            ankle_velocity_pelvis=ankle_velocity,
            contact_target=contact_target,
            contact_state=contact_state,
            source_root_position=source_root_position,
            root_height=root_height,
            root_rotation_exp_map=root_rotation_exp_map,
            root_linear_velocity=root_linear_velocity,
        )

    def sample_full_reference(
        self,
        phase: torch.Tensor,
        progress: torch.Tensor | float,
    ) -> G1FullReferenceSample:
        """Sample the V7 trajectory and its kinematic targets by phase/progress."""

        if any(
            value is None
            for value in (
                self.compressed_joint_positions,
                self.compressed_joint_velocities,
                self.full_joint_positions,
                self.full_joint_velocities,
                self.base_action_scales,
                self.full_action_scales,
            )
        ):
            raise RuntimeError("Reference must be retargeted before V7 sampling")
        (
            normalized_phase,
            frame_index_0,
            frame_index_1,
            frame_blend,
        ) = self._frame_coordinates(phase)
        scheduled_progress = self._scheduled_progress(progress, normalized_phase)
        blend_column = scheduled_progress.unsqueeze(-1)

        assert self.compressed_joint_positions is not None
        assert self.compressed_joint_velocities is not None
        assert self.full_joint_positions is not None
        assert self.full_joint_velocities is not None
        assert self.base_action_scales is not None
        assert self.full_action_scales is not None
        compressed_position = self._interpolate_frames(
            self.compressed_joint_positions,
            frame_index_0,
            frame_index_1,
            frame_blend,
        )
        full_position = self._interpolate_frames(
            self.full_joint_positions,
            frame_index_0,
            frame_index_1,
            frame_blend,
        )
        compressed_velocity = self.compressed_joint_velocities[frame_index_0]
        full_velocity = self.full_joint_velocities[frame_index_0]
        joint_position = torch.lerp(compressed_position, full_position, blend_column)
        joint_velocity = torch.lerp(compressed_velocity, full_velocity, blend_column)
        action_scale = torch.lerp(
            self.base_action_scales.expand_as(joint_position),
            self.full_action_scales.expand_as(joint_position),
            blend_column,
        )
        source_root_position = self._interpolate_frames(
            self.source_root_positions,
            frame_index_0,
            frame_index_1,
            frame_blend,
        )
        root_rotation_exp_map = self._interpolate_frames(
            self.source_root_rotation_exp_map,
            frame_index_0,
            frame_index_1,
            frame_blend,
        )
        full_ankle_position = g1_leg_forward_kinematics(full_position[:, :12])
        contact_target, _ = g1_ankle_contact_targets(full_ankle_position)
        return self._assemble_full_reference(
            joint_position,
            joint_velocity,
            normalized_phase,
            scheduled_progress,
            action_scale,
            source_root_position,
            root_rotation_exp_map,
            contact_target,
        )

    def sample_future_reference(
        self,
        current_phase: torch.Tensor,
        phase_rate: torch.Tensor | float,
        progress: torch.Tensor | float = 1.0,
    ) -> G1FutureReferenceSample:
        """Sample compact V8 targets 1/30, 2/30, and 3/30 seconds ahead."""

        if current_phase.ndim != 1:
            raise ValueError("current_phase must have shape [N]")
        if not torch.is_floating_point(current_phase):
            raise TypeError("current_phase must be floating point")
        if current_phase.device != self.joint_positions.device:
            raise ValueError("current_phase and reference must share a device")
        phase_rate_tensor = torch.as_tensor(
            phase_rate,
            dtype=self.joint_positions.dtype,
            device=self.joint_positions.device,
        )
        if phase_rate_tensor.ndim == 0:
            phase_rate_tensor = phase_rate_tensor.expand_as(current_phase)
        elif phase_rate_tensor.shape != current_phase.shape:
            raise ValueError("phase_rate must be scalar or match current_phase")
        if not torch.isfinite(phase_rate_tensor).all().item():
            raise ValueError("phase_rate must contain only finite values")
        if torch.any(phase_rate_tensor < 0.0).item():
            raise ValueError("phase_rate must be non-negative")

        horizon = current_phase.new_tensor(V8_FUTURE_HORIZONS_S)
        future_phase = torch.remainder(
            current_phase.unsqueeze(-1)
            + phase_rate_tensor.unsqueeze(-1) * horizon / self.duration_s,
            1.0,
        )
        flat_phase = future_phase.reshape(-1)
        progress_tensor = torch.as_tensor(
            progress,
            dtype=self.joint_positions.dtype,
            device=self.joint_positions.device,
        )
        if progress_tensor.ndim == 0:
            flat_progress: torch.Tensor | float = progress_tensor
        elif progress_tensor.shape == current_phase.shape:
            flat_progress = (
                progress_tensor.unsqueeze(-1)
                .expand(-1, V8_FUTURE_REFERENCE_STEPS)
                .reshape(-1)
            )
        else:
            raise ValueError("progress must be scalar or match current_phase")

        flat_reference = self.sample_full_reference(
            flat_phase,
            flat_progress,
        )
        batch_size = current_phase.shape[0]
        joint_position = flat_reference.joint_position.reshape(
            batch_size,
            V8_FUTURE_REFERENCE_STEPS,
            -1,
        )
        action_scale = flat_reference.action_scale.reshape_as(joint_position)
        ankle_position = flat_reference.ankle_position_pelvis.reshape(
            batch_size,
            V8_FUTURE_REFERENCE_STEPS,
            2,
            3,
        )
        contact_target = flat_reference.contact_target.reshape(
            batch_size,
            V8_FUTURE_REFERENCE_STEPS,
            2,
        )
        root_height = flat_reference.root_height.reshape(
            batch_size,
            V8_FUTURE_REFERENCE_STEPS,
        )
        if self.trajectory_center is None:
            raise RuntimeError("Reference must be retargeted before V8 sampling")
        features = build_g1_future_reference_features(
            joint_position,
            action_scale,
            self.trajectory_center,
            ankle_position,
            contact_target,
            root_height,
        )
        return G1FutureReferenceSample(
            features=features,
            phase=future_phase,
            joint_position=joint_position,
            action_scale=action_scale,
            ankle_position_pelvis=ankle_position,
            contact_target=contact_target,
            root_height=root_height,
        )

    def natural_forward_speed(
        self,
        progress: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return nominal mean forward speed for one trajectory progress value.

        Dividing a commanded forward speed by this value yields the dimensionless
        phase-rate multiplier to apply to the sampled joint, ankle, and root
        velocities.
        """

        progress_tensor = torch.as_tensor(
            progress,
            dtype=self.joint_positions.dtype,
            device=self.joint_positions.device,
        )
        if progress_tensor.ndim != 0:
            raise ValueError("progress must be scalar for natural_forward_speed")
        phase = torch.arange(
            self.frame_count - 1,
            dtype=self.joint_positions.dtype,
            device=self.joint_positions.device,
        ) / (self.frame_count - 1)
        sample = self.sample_full_reference(phase, progress_tensor)
        forward_speed = sample.root_linear_velocity[:, 0].mean()
        if not torch.isfinite(forward_speed).item() or forward_speed <= 0.0:
            raise RuntimeError(
                "Scheduled reference does not have a positive natural forward speed"
            )
        return forward_speed

    def sample_full_reference_initial_state(
        self,
        default_position: torch.Tensor,
        default_velocity: torch.Tensor,
        progress: torch.Tensor | float,
        reference_probability: float = REFERENCE_STATE_INITIALIZATION_PROBABILITY,
        generator: torch.Generator | None = None,
    ) -> G1FullReferenceInitialState:
        """Sample V7 RSI joint and root targets without changing the V6 API."""

        expected_shape = default_position.shape
        if default_position.ndim != 2 or expected_shape[-1] != len(self.joint_names):
            raise ValueError("default_position must have shape [N, J]")
        if default_velocity.shape != expected_shape:
            raise ValueError("default_velocity must match default_position")
        if (
            default_position.device != self.joint_positions.device
            or default_velocity.device != self.joint_positions.device
        ):
            raise ValueError("default states and reference must share a device")
        if not 0.0 <= reference_probability <= 1.0:
            raise ValueError("reference_probability must be in [0, 1]")

        count = expected_shape[0]
        random_options = {
            "device": self.joint_positions.device,
            "dtype": self.joint_positions.dtype,
            "generator": generator,
        }
        use_reference = torch.rand(count, **random_options) < reference_probability
        random_phase = torch.rand(count, **random_options)
        phase = torch.where(use_reference, random_phase, torch.zeros_like(random_phase))
        reference = self.sample_full_reference(phase, progress)
        joint_position = torch.where(
            use_reference.unsqueeze(-1),
            reference.joint_position.to(default_position.dtype),
            default_position,
        )
        joint_velocity = torch.where(
            use_reference.unsqueeze(-1),
            reference.joint_velocity.to(default_velocity.dtype),
            default_velocity,
        )
        root_rotation = torch.where(
            use_reference.unsqueeze(-1),
            reference.root_rotation_exp_map,
            torch.zeros_like(reference.root_rotation_exp_map),
        )
        contact_target = torch.where(
            use_reference.unsqueeze(-1),
            reference.contact_target,
            torch.ones_like(reference.contact_target),
        )
        assembled = self._assemble_full_reference(
            joint_position,
            joint_velocity,
            phase,
            reference.progress,
            reference.action_scale,
            reference.source_root_position,
            root_rotation,
            contact_target,
        )
        return G1FullReferenceInitialState(
            joint_position=assembled.joint_position,
            joint_velocity=assembled.joint_velocity,
            phase=assembled.phase,
            use_reference=use_reference,
            progress=assembled.progress,
            action_scale=assembled.action_scale,
            ankle_position_pelvis=assembled.ankle_position_pelvis,
            ankle_velocity_pelvis=assembled.ankle_velocity_pelvis,
            contact_target=assembled.contact_target,
            contact_state=assembled.contact_state,
            root_height=assembled.root_height,
            root_rotation_exp_map=assembled.root_rotation_exp_map,
            root_linear_velocity=assembled.root_linear_velocity,
        )

    def sample_initial_state(
        self,
        default_position: torch.Tensor,
        default_velocity: torch.Tensor,
        reference_probability: float = REFERENCE_STATE_INITIALIZATION_PROBABILITY,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix random reference states with phase-zero standing states."""

        expected_shape = default_position.shape
        if default_position.ndim != 2 or expected_shape[-1] != len(self.joint_names):
            raise ValueError("default_position must have shape [N, J]")
        if default_velocity.shape != expected_shape:
            raise ValueError("default_velocity must match default_position")
        if (
            default_position.device != self.joint_positions.device
            or default_velocity.device != self.joint_positions.device
        ):
            raise ValueError("default states and reference must share a device")
        if not 0.0 <= reference_probability <= 1.0:
            raise ValueError("reference_probability must be in [0, 1]")

        count = expected_shape[0]
        random_options = {
            "device": self.joint_positions.device,
            "dtype": self.joint_positions.dtype,
            "generator": generator,
        }
        use_reference = torch.rand(count, **random_options) < reference_probability
        random_phase = torch.rand(count, **random_options)
        phase = torch.where(use_reference, random_phase, torch.zeros_like(random_phase))
        reference_position, reference_velocity, _ = self.sample_phase(phase)

        position = torch.where(
            use_reference.unsqueeze(-1),
            reference_position.to(default_position.dtype),
            default_position,
        )
        velocity = torch.where(
            use_reference.unsqueeze(-1),
            reference_velocity.to(default_velocity.dtype),
            default_velocity,
        )
        return position, velocity, phase, use_reference

    def sample(
        self,
        time_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interpolated positions, finite-difference velocity, and phase."""

        phase = self.phase_at_time(time_s)
        return self.sample_phase(phase)


def deepmimic_joint_similarity(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    reference_position: torch.Tensor,
    reference_velocity: torch.Tensor,
    joint_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute original-paper exponential pose and joint-velocity kernels."""

    expected_shape = joint_position.shape
    if any(
        value.shape != expected_shape
        for value in (joint_velocity, reference_position, reference_velocity)
    ):
        raise ValueError("Joint and reference tensors must have identical shapes")
    if joint_weights.shape != expected_shape[-1:]:
        raise ValueError("joint_weights must have shape [J]")

    position_difference = torch.atan2(
        torch.sin(joint_position - reference_position),
        torch.cos(joint_position - reference_position),
    )
    velocity_difference = joint_velocity - reference_velocity
    pose_error = torch.sum(joint_weights * position_difference.square(), dim=-1)
    velocity_error = torch.sum(
        joint_weights * velocity_difference.square(),
        dim=-1,
    )
    pose_similarity = torch.exp(-REFERENCE_POSE_SCALE * pose_error)
    velocity_similarity = torch.exp(-REFERENCE_VELOCITY_SCALE * velocity_error)
    return pose_similarity, velocity_similarity, pose_error, velocity_error


def gated_imitation_rewards(
    pose_similarity: torch.Tensor,
    velocity_similarity: torch.Tensor,
    gate: torch.Tensor,
    *,
    pose_weight: float = REFERENCE_POSE_WEIGHT,
    velocity_weight: float = REFERENCE_VELOCITY_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the deliberately small treatment weights and a physical-state gate."""

    if (
        pose_similarity.shape != velocity_similarity.shape
        or gate.shape != pose_similarity.shape
    ):
        raise ValueError("Similarity and gate tensors must have identical shapes")
    if not math.isfinite(pose_weight) or pose_weight < 0.0:
        raise ValueError("pose_weight must be finite and non-negative")
    if not math.isfinite(velocity_weight) or velocity_weight < 0.0:
        raise ValueError("velocity_weight must be finite and non-negative")
    clipped_gate = gate.clamp(min=0.0, max=1.0)
    return (
        pose_weight * pose_similarity * clipped_gate,
        velocity_weight * velocity_similarity * clipped_gate,
    )
