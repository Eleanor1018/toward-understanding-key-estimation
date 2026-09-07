"""Fixed physical-unit normalization for the G1 policy and critic inputs."""

from __future__ import annotations

from functools import lru_cache

import torch


NORMALIZATION_TYPE = "g1_fixed_physical_scales_phase_v2"
LEGACY_NORMALIZATION_TYPE = "g1_fixed_physical_scales_v1"
NORMALIZED_VALUE_CLIP = 5.0

# Joint order is the public G1 29-DOF action/observation contract.
DEFAULT_JOINT_POSITIONS = (
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
    0.30,
    0.25,
    0.0,
    0.97,
    0.15,
    0.0,
    0.0,
    0.30,
    -0.25,
    0.0,
    0.97,
    -0.15,
    0.0,
    0.0,
)

# Effort limits mirror the actuator groups in g1_env.py. Dividing by these
# limits makes every privileged torque approximately comparable in magnitude.
JOINT_EFFORT_LIMITS = (
    88.0,
    139.0,
    88.0,
    139.0,
    25.0,
    25.0,
    88.0,
    139.0,
    88.0,
    139.0,
    25.0,
    25.0,
    88.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
)

# Commands and explicit velocity estimates share these scales, so equal
# physical values have equal normalized representations in the actor.
LINEAR_VELOCITY_SCALES = (1.0 / 1.20, 1.0 / 0.60, 1.0)
COMMAND_SCALES = (*LINEAR_VELOCITY_SCALES[:2], 1.0, 1.0, 1.0)
ANGULAR_VELOCITY_SCALES = (0.25, 0.25, 1.0)


def normalization_type_for_command_dim(command_dim: int) -> str:
    """Return truthful checkpoint metadata for either supported contract."""

    if command_dim == 3:
        return LEGACY_NORMALIZATION_TYPE
    if command_dim == 5:
        return NORMALIZATION_TYPE
    raise ValueError(f"unsupported command dimension: {command_dim}")


@lru_cache(maxsize=None)
def _constant_tensor(
    values: tuple[float, ...],
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(values, device=device, dtype=dtype)


def _constant_like(
    values: tuple[float, ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    return _constant_tensor(
        values,
        reference.device.type,
        reference.device.index,
        reference.dtype,
    )


def _check_last_dimension(
    tensor: torch.Tensor,
    expected: int,
    name: str,
) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} must end in dimension {expected}, got {tuple(tensor.shape)}"
        )


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    """Normalize one observation or a history ending in the 93-D contract."""

    _check_last_dimension(obs, 93, "obs")
    normalized = obs.clone()
    normalized[..., 3:6] *= _constant_like(
        ANGULAR_VELOCITY_SCALES,
        normalized,
    )
    normalized[..., 6:35] = 4.0 * (
        normalized[..., 6:35]
        - _constant_like(
            DEFAULT_JOINT_POSITIONS,
            normalized,
        )
    )
    normalized[..., 35:64] *= 0.10
    return normalized.clamp_(-NORMALIZED_VALUE_CLIP, NORMALIZED_VALUE_CLIP)


def normalize_command(command: torch.Tensor) -> torch.Tensor:
    command_dim = command.shape[-1]
    if command_dim not in (3, 5):
        raise ValueError(
            f"command must end in dimension 3 or 5, got {tuple(command.shape)}"
        )
    scales = COMMAND_SCALES[:command_dim]
    return (command * _constant_like(scales, command)).clamp(
        -NORMALIZED_VALUE_CLIP, NORMALIZED_VALUE_CLIP
    )


def normalize_explicit_velocity(velocity: torch.Tensor) -> torch.Tensor:
    _check_last_dimension(velocity, 3, "explicit velocity")
    return (velocity * _constant_like(LINEAR_VELOCITY_SCALES, velocity)).clamp(
        -NORMALIZED_VALUE_CLIP, NORMALIZED_VALUE_CLIP
    )


def denormalize_explicit_velocity(velocity: torch.Tensor) -> torch.Tensor:
    _check_last_dimension(velocity, 3, "explicit velocity")
    return velocity / _constant_like(LINEAR_VELOCITY_SCALES, velocity)


def normalize_privileged(privileged: torch.Tensor) -> torch.Tensor:
    """Normalize all 103 defined privileged values without padding."""

    _check_last_dimension(privileged, 103, "privileged")
    normalized = privileged.clone()
    # Relative x/y displacement, followed by base height centered at 0.8 m.
    normalized[..., 0:2] *= 0.10
    normalized[..., 2] = (normalized[..., 2] - 0.80) * 2.5
    # Quaternion 3:7 remains unit-scale.
    normalized[..., 7:10] *= _constant_like(
        LINEAR_VELOCITY_SCALES,
        normalized,
    )
    normalized[..., 10:13] *= _constant_like(
        ANGULAR_VELOCITY_SCALES,
        normalized,
    )
    # Projected gravity 13:16 remains unit-scale.
    normalized[..., 16:45] = 4.0 * (
        normalized[..., 16:45]
        - _constant_like(
            DEFAULT_JOINT_POSITIONS,
            normalized,
        )
    )
    normalized[..., 45:74] *= 0.10
    normalized[..., 74:103] /= _constant_like(
        JOINT_EFFORT_LIMITS,
        normalized,
    )
    normalized[..., 74:103].clamp_(-1.5, 1.5)
    return normalized.clamp_(-NORMALIZED_VALUE_CLIP, NORMALIZED_VALUE_CLIP)


def normalize_future_reference(future_reference: torch.Tensor) -> torch.Tensor:
    """Copy pre-normalized future-reference features and enforce their bound."""

    if future_reference.shape[-2:] != (3, 21):
        raise ValueError(
            "future_reference must end in shape (3, 21), got "
            f"{tuple(future_reference.shape)}"
        )
    if not torch.is_floating_point(future_reference):
        raise TypeError("future_reference must be floating point")
    return future_reference.clone().clamp_(
        -NORMALIZED_VALUE_CLIP,
        NORMALIZED_VALUE_CLIP,
    )


def normalize_observation_batch(
    observation: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return network inputs while leaving simulator-owned tensors untouched."""

    normalized = {
        "history": normalize_obs(observation["history"]),
        "obs": normalize_obs(observation["obs"]),
        "command": normalize_command(observation["command"]),
        "privileged": normalize_privileged(observation["privileged"]),
        "explicit_target": normalize_explicit_velocity(observation["explicit_target"]),
    }
    if "future_reference" in observation:
        normalized["future_reference"] = normalize_future_reference(
            observation["future_reference"]
        )
    return normalized
