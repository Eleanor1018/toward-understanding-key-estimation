"""Unitree G1 29-DOF mechanics with a 12-action leg policy.

The nominal pose and actuator parameters follow Unitree RL Lab's locomotion
UNITREE_G1_29DOF_CFG, not its differently tuned MIMIC configuration:
https://github.com/unitreerobotics/unitree_rl_lab/blob/
4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/
unitree_rl_lab/assets/robots/unitree.py#L397-L508

Engineering choices for this project: retain all 29 physical joints, hold the
17 upper-body joints at nominal PD targets, and expose only the 12 leg actions.
Only hip-yaw targets receive an extra angular cap; simulator mechanical limits
and actuator limits remain in force. Soft joint limits do not clip PD targets.
The action is an
unsquashed Gaussian coordinate, not a normalized +/-1 target. The raw-action
clip of 100 follows Unitree RL Gym's normalization convention (commit
276801e46c5d433564f24658bac64f254b7d2d4b, legged_robot_config.py:135);
it is not a physical joint range or a guarantee of successful locomotion.
"""

from __future__ import annotations

import math

import torch


JOINT_NAMES29 = (
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEG_JOINT_NAMES12 = JOINT_NAMES29[:12]
# 先由名称得到策略列，调用时再经leg_ids映射原生顺序，不能直接把2/8当原生ID。
HIP_YAW_JOINT_NAMES = ("left_hip_yaw_joint", "right_hip_yaw_joint")
HIP_YAW_POLICY_COLUMNS = tuple(LEG_JOINT_NAMES12.index(name) for name in HIP_YAW_JOINT_NAMES)

DEFAULT_JOINT_POS29 = (
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,
    0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,
)
STIFFNESS29 = (
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    200.0, 40.0, 40.0,
    *([40.0] * 14),
)
DAMPING29 = (
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    5.0, 5.0, 5.0,
    *([1.0] * 14),
)
EFFORT_LIMIT29 = (
    88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
    88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
    88.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
)
VELOCITY_LIMIT29 = (
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,
)
ARMATURE29 = (0.01,) * 29

ROOT_HEIGHT = 0.8
ACTION_SCALE = 0.25
RAW_ACTION_CLIP = 100.0
SOFT_JOINT_POSITION_LIMIT_FACTOR = 0.9


def make_joint_targets(
    raw_actions: torch.Tensor,
    default_pos: torch.Tensor,
    soft_limits: torch.Tensor,
    leg_ids: torch.Tensor,
    action_scale: float = ACTION_SCALE,
    *,
    hip_yaw_target_limit_rad: float,
    raw_action_clip: float = RAW_ACTION_CLIP,
) -> torch.Tensor:
    """Map policy-ordered leg actions into native simulator joint targets.

    ``leg_ids[k]`` is the native simulator index of LEG_JOINT_NAMES12[k].
    ``default_pos`` and ``soft_limits`` already use native simulator order.
    上半身保持默认 PD 目标。除 hip yaw 外不按 soft limits 截断目标，
    与普通 JointPositionAction 的 default + scale * action 语义一致。
    soft_limits 保留为接口元数据，不用它额外缩小膝/髋 pitch/踝的探索范围。
    左右 hip yaw 限制默认姿态±配置弧度；只约束 PD 目标，不保证实际角度不超调。
    """
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 12:
        raise ValueError("raw_actions must have shape [N, 12]")
    batch_size = raw_actions.shape[0]
    if default_pos.shape != (batch_size, 29):
        raise ValueError("default_pos must have shape [N, 29]")
    if soft_limits.shape != (batch_size, 29, 2):
        raise ValueError("soft_limits must have shape [N, 29, 2]")
    if leg_ids.shape != (12,) or leg_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("leg_ids must be a 12-element integer tensor")
    if any(t.device != raw_actions.device for t in (default_pos, soft_limits, leg_ids)):
        raise ValueError("all inputs must share one device")
    if not raw_actions.is_floating_point() or any(
        t.dtype != raw_actions.dtype for t in (default_pos, soft_limits)
    ):
        raise ValueError("actions, default positions and limits must share a floating dtype")
    if not math.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("action_scale must be finite and positive")
    if not math.isfinite(raw_action_clip) or raw_action_clip <= 0:
        raise ValueError("raw_action_clip must be finite and positive")
    if (isinstance(hip_yaw_target_limit_rad, bool)
            or not isinstance(hip_yaw_target_limit_rad, (int, float))
            or not math.isfinite(hip_yaw_target_limit_rad)
            or not 0.0 < hip_yaw_target_limit_rad <= math.pi):
        raise ValueError("hip_yaw_target_limit_rad must be finite and in (0, pi]")
    if torch.any((leg_ids < 0) | (leg_ids >= 29)).item() or leg_ids.unique().numel() != 12:
        raise ValueError("leg_ids must be distinct native indices in [0, 29)")
    if torch.any(soft_limits[..., 0] > soft_limits[..., 1]).item():
        raise ValueError("soft-limit lower bounds must not exceed upper bounds")

    targets = default_pos.clone()
    targets[:, leg_ids] += raw_actions.clamp(-raw_action_clip, raw_action_clip) * action_scale
    # 不能原地改 raw_actions：PPO 的 log_prob 对应采样的原始高斯动作。
    yaw_native_ids = leg_ids[list(HIP_YAW_POLICY_COLUMNS)]
    targets[:, yaw_native_ids] = torch.maximum(torch.minimum(
        targets[:, yaw_native_ids], default_pos[:, yaw_native_ids] + hip_yaw_target_limit_rad),
        default_pos[:, yaw_native_ids] - hip_yaw_target_limit_rad)
    return targets
