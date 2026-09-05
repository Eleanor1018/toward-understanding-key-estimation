"""Isaac Lab direct environment for paper-style G1 locomotion training.

The observation and action interface is the repository's fixed 29-DOF G1
contract.  The locomotion objective follows the Gaussian/Cauchy reward families
and system-identification randomization ranges in arXiv:2403.05868.  Existing
profiles keep ``command=[vx, vy, yaw_rate]``. The additive phase-visible RSI
profiles append ``[sin(phase), cos(phase)]`` while retaining the same three
physical command values internally. V8 additionally exposes three compact
21-value future-reference targets without changing proprioception or actions.

Observation contract
--------------------
``obs`` has 93 values in this exact order:

* projected gravity in the base frame: 3
* base angular velocity in the base frame: 3
* joint position in ``G1_JOINT_NAMES`` order: 29
* joint velocity in ``G1_JOINT_NAMES`` order: 29
* previous normalized action in ``G1_JOINT_NAMES`` order: 29

``privileged`` has 103 real simulator values in this exact order (no padding):

* base position relative to the environment origin: 3 (indices 0:3)
* base quaternion in the world frame, scalar-first: 4 (indices 3:7)
* base linear velocity in the base frame: 3 (indices 7:10)
* base angular velocity in the base frame: 3 (indices 10:13)
* projected gravity in the base frame: 3 (indices 13:16)
* joint position: 29 (indices 16:45)
* joint velocity: 29 (indices 45:74)
* applied joint torque: 29 (indices 74:103)

``explicit_target`` is the true base linear velocity in the base frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, yaw_quat

from config import (
    FULL_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_FEATURES_PER_STEP,
    FUTURE_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_STEPS,
)
from gait import classify_foot_landings
from motion_reference import (
    G1_FOOT_SOLE_OFFSET_M,
    G1_DEEPMIMIC_DOF_WEIGHTS,
    G1_WALK_ARCHIVE_SHA256,
    G1_WALK_SOURCE_SHA256,
    REFERENCE_STATE_INITIALIZATION_PROBABILITY,
    V7_REFERENCE_COMPONENT_FRACTIONS,
    CyclicJointReference,
    phase_visible_command,
    deepmimic_joint_similarity,
    gated_imitation_rewards,
    sanitize_root_pose,
    split_imitation_reward_weight,
)
from normalization import JOINT_EFFORT_LIMITS


TASK_ID = "Unitree-G1-29dof-KeyEstimation-v0"
OBS_DIM = 93
HISTORY_STEPS = 50
PRIVILEGED_DIM = 103
ACTION_DIM = 29
ACTION_SCALE = 0.25
GAIT_PERIOD_S = 0.80
DESIRED_BASE_HEIGHT = 0.80
PHYSICAL_COMMAND_DIM = 3
PHASE_VISIBLE_COMMAND_DIM = 5
PHASE_RSI_REWARD_PROFILES = (
    "p1_walk_stable_v6_phase_rsi",
    "p1_walk_stable_v6_phase_rsi_imitation",
    FULL_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_REWARD_PROFILE,
)
PHASE_RSI_IMITATION_REWARD_PROFILE = "p1_walk_stable_v6_phase_rsi_imitation"
FULL_REFERENCE_REWARD_PROFILES = (
    FULL_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_REWARD_PROFILE,
)
V7_FULL_ACTION_SCALES = (
    0.45,
    0.25,
    0.25,
    0.86,
    0.33,
    0.25,
    0.46,
    0.25,
    0.25,
    0.95,
    0.32,
    0.25,
    *([0.25] * 17),
)
V7_CADENCE_RAMP_PROGRESS = 0.25


def _signed_similarity(similarity: torch.Tensor) -> torch.Tensor:
    """Map a unit-interval similarity to a symmetric reward score."""

    return 2.0 * similarity.clamp(0.0, 1.0) - 1.0


def _v8_force_contact_score(
    foot_force: torch.Tensor,
    mass_times_gravity: torch.Tensor,
    target_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score force-derived foot contacts against signed smooth targets."""

    if foot_force.ndim != 2 or foot_force.shape[-1] != 2:
        raise ValueError("foot_force must have shape [N, 2]")
    if mass_times_gravity.shape != foot_force.shape[:-1]:
        raise ValueError("mass_times_gravity must have shape [N]")
    if target_contact.shape != foot_force.shape:
        raise ValueError("target_contact must have shape [N, 2]")
    actual_contact = foot_force > 0.05 * mass_times_gravity.unsqueeze(-1)
    actual_sign = 2.0 * actual_contact.to(foot_force.dtype) - 1.0
    target_sign = 2.0 * target_contact.clamp(0.0, 1.0) - 1.0
    return torch.mean(actual_sign * target_sign, dim=-1), actual_contact


def _v8_clearance_score(
    ankle_height_world: torch.Tensor,
    ground_height: torch.Tensor,
    reference_ankle_height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score signed swing clearance relative to the lower reference ankle."""

    if ankle_height_world.ndim != 2 or ankle_height_world.shape[-1] != 2:
        raise ValueError("ankle_height_world must have shape [N, 2]")
    if ground_height.shape != ankle_height_world.shape[:-1]:
        raise ValueError("ground_height must have shape [N]")
    if reference_ankle_height.shape != ankle_height_world.shape:
        raise ValueError("reference_ankle_height must have shape [N, 2]")
    actual_clearance = (
        ankle_height_world - ground_height.unsqueeze(-1) - G1_FOOT_SOLE_OFFSET_M
    )
    target_clearance = reference_ankle_height - reference_ankle_height.amin(
        dim=-1,
        keepdim=True,
    )
    similarity = torch.exp(-torch.square((actual_clearance - target_clearance) / 0.04))
    return (
        torch.mean(_signed_similarity(similarity), dim=-1),
        actual_clearance,
        target_clearance,
    )


def _v8_support_score(actual_contact: torch.Tensor) -> torch.Tensor:
    """Prefer single support and assign signed costs to double support/flight."""

    if actual_contact.ndim != 2 or actual_contact.shape[-1] != 2:
        raise ValueError("actual_contact must have shape [N, 2]")
    contact_count = torch.sum(actual_contact.to(torch.int32), dim=-1)
    return torch.where(
        contact_count == 1,
        torch.ones_like(contact_count, dtype=torch.float32),
        torch.where(
            contact_count == 2,
            torch.full_like(contact_count, -0.25, dtype=torch.float32),
            torch.full_like(contact_count, -1.0, dtype=torch.float32),
        ),
    )


def _v8_landing_score(
    alternating_landing: torch.Tensor,
    repeated_landing: torch.Tensor,
) -> torch.Tensor:
    """Return +1/-1 for valid alternating/repeated landings, else zero."""

    if alternating_landing.shape != repeated_landing.shape:
        raise ValueError("landing masks must have identical shapes")
    return alternating_landing.to(torch.float32) - repeated_landing.to(torch.float32)


def _v8_imitation_score(
    pose_similarity: torch.Tensor,
    joint_velocity_similarity: torch.Tensor,
    ankle_position_similarity: torch.Tensor,
    root_height_similarity: torch.Tensor,
    root_velocity_similarity: torch.Tensor,
    force_contact_score: torch.Tensor,
    clearance_score: torch.Tensor,
) -> torch.Tensor:
    """Combine normalized full-reference components into a signed score."""

    return (
        0.30 * _signed_similarity(pose_similarity)
        + 0.05 * _signed_similarity(joint_velocity_similarity)
        + 0.15 * _signed_similarity(ankle_position_similarity)
        + 0.05 * _signed_similarity(root_height_similarity)
        + 0.05 * _signed_similarity(root_velocity_similarity)
        + 0.20 * force_contact_score
        + 0.20 * clearance_score
    )


def _v8_task_score(
    planar_tracking_similarity: torch.Tensor,
    directional_progress: torch.Tensor,
    yaw_tracking_similarity: torch.Tensor,
    upright_similarity: torch.Tensor,
    height_similarity: torch.Tensor,
    support_score: torch.Tensor,
    landing_score: torch.Tensor,
) -> torch.Tensor:
    """Combine command following and gait events into a signed task score."""

    return (
        0.45 * _signed_similarity(planar_tracking_similarity)
        + 0.20 * directional_progress.clamp(-1.0, 1.0)
        + 0.10 * _signed_similarity(yaw_tracking_similarity)
        + 0.10 * _signed_similarity(upright_similarity)
        + 0.05 * _signed_similarity(height_similarity)
        + 0.05 * support_score
        + 0.05 * landing_score
    )


def _v8_core_reward(
    imitation_score: torch.Tensor,
    task_score: torch.Tensor,
    task_mix_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix signed imitation/task scores and apply the V8 core scale."""

    mixed_score = torch.lerp(imitation_score, task_score, task_mix_beta)
    return 3.0 * mixed_score, mixed_score


G1_JOINT_NAMES = (
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

_ASSET_PATH = (
    Path(__file__).resolve().parent / "assets" / "g1_29dof" / "g1_29dof_rev_1_0.usd"
)
_WALK_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "assets" / "motions" / "g1_walk_mimickit.npz"
)


G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            ".*_shoulder_pitch_joint": 0.30,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "hip_pitch_yaw_and_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_yaw_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "hip_roll_and_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_joint": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_roll_joint": 2.0, ".*_knee_joint": 4.0},
            armature=0.01,
        ),
        "ankle_waist_and_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_shoulder_.*",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping={
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
                ".*_shoulder_.*": 1.0,
                ".*_elbow_joint": 1.0,
                ".*_wrist_roll_joint": 1.0,
            },
            armature=0.01,
        ),
        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)


@configclass
class EventCfg:
    """Paper Appendix Table II randomization supported by Isaac Lab 2.0.2."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    base_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 12.5),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


@configclass
class G1KeyEstimationEnvCfg(DirectRLEnvCfg):
    """Paper-style flat-ground locomotion configuration."""

    episode_length_s = 10.0
    # The paper runs the policy at 100 Hz and its PD loop at 1 kHz.
    decimation = 10
    action_scale = ACTION_SCALE
    action_space = ACTION_DIM
    observation_space = {
        "history": [HISTORY_STEPS, OBS_DIM],
        "obs": OBS_DIM,
        "command": 3,
        "privileged": PRIVILEGED_DIM,
        "explicit_target": 3,
    }
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 1000.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_roll_link",
        history_length=3,
        update_period=1.0 / 1000.0,
        track_air_time=True,
    )
    termination_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        # Dones are evaluated once per 10-step control interval. Keep one
        # additional physics sample so a brief illegal contact cannot occur
        # and disappear between two policy steps.
        history_length=11,
        update_period=1.0 / 1000.0,
        track_air_time=False,
    )
    events: EventCfg | None = EventCfg()

    # Evaluation disables these flags and ``events`` without altering tensor
    # meanings.  Training defaults reproduce Appendix Table II.
    enable_domain_randomization = True
    observation_noise = True
    domain_randomization_scale = 1.0
    command_scale = 1.0
    reward_profile = "p1_dense"
    imitation_reward_weight = 0.03
    reference_state_initialization_probability = (
        REFERENCE_STATE_INITIALIZATION_PROBABILITY
    )
    reference_motion_progress = 0.0
    task_mix_beta = 0.0
    command_profile = "full"
    resample_commands_on_reset = True


class G1KeyEstimationEnv(DirectRLEnv):
    """G1 task exposing profile-specific tensors consumed by ``train.py``."""

    cfg: G1KeyEstimationEnvCfg

    def __init__(
        self,
        cfg: G1KeyEstimationEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        if not 0.0 <= cfg.domain_randomization_scale <= 1.0:
            raise ValueError("domain_randomization_scale must be in [0, 1]")
        if not 0.0 < cfg.command_scale <= 1.0:
            raise ValueError("command_scale must be in (0, 1]")
        if not math.isfinite(cfg.action_scale) or cfg.action_scale <= 0.0:
            raise ValueError("action_scale must be finite and positive")
        if cfg.reward_profile not in (
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
            *PHASE_RSI_REWARD_PROFILES,
        ):
            raise ValueError(f"Unsupported reward_profile={cfg.reward_profile!r}")
        if cfg.command_profile not in (
            "stand",
            "stage1",
            "forward_walk",
            "walk",
            "stage2",
            "stage3",
            "full",
        ):
            raise ValueError(f"Unsupported command_profile={cfg.command_profile!r}")

        self._future_reference_enabled = (
            cfg.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE
        )
        if self._future_reference_enabled:
            cfg.reference_motion_progress = 1.0
        self._phase_rsi_enabled = cfg.reward_profile in PHASE_RSI_REWARD_PROFILES
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space.pop("future_reference", None)
        cfg.observation_space["command"] = (
            PHASE_VISIBLE_COMMAND_DIM
            if self._phase_rsi_enabled
            else PHYSICAL_COMMAND_DIM
        )
        if self._future_reference_enabled:
            cfg.observation_space["future_reference"] = [
                FUTURE_REFERENCE_STEPS,
                FUTURE_REFERENCE_FEATURES_PER_STEP,
            ]
        if cfg.reward_profile in (
            PHASE_RSI_IMITATION_REWARD_PROFILE,
            FULL_REFERENCE_REWARD_PROFILE,
        ) and (
            not math.isfinite(cfg.imitation_reward_weight)
            or cfg.imitation_reward_weight < 0.0
        ):
            raise ValueError("imitation_reward_weight must be finite and non-negative")
        if self._phase_rsi_enabled and not (
            0.0 <= cfg.reference_state_initialization_probability <= 1.0
        ):
            raise ValueError(
                "reference_state_initialization_probability must be in [0, 1]"
            )
        if not 0.0 <= cfg.reference_motion_progress <= 1.0:
            raise ValueError("reference_motion_progress must be in [0, 1]")
        if self._future_reference_enabled and (
            not math.isfinite(cfg.task_mix_beta) or not 0.0 <= cfg.task_mix_beta <= 1.0
        ):
            raise ValueError("task_mix_beta must be finite and in [0, 1]")
        if (
            cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES
            and cfg.command_profile != "forward_walk"
        ):
            raise ValueError(
                f"{cfg.reward_profile} currently requires "
                "command_profile='forward_walk'"
            )

        randomization_scale = cfg.domain_randomization_scale
        if not cfg.enable_domain_randomization or randomization_scale == 0.0:
            cfg.events = None
            cfg.enable_domain_randomization = False
            cfg.observation_noise = False
        elif cfg.events is not None:
            cfg.events.physics_material.params["static_friction_range"] = (
                1.0 - 0.75 * randomization_scale,
                1.0 + 0.25 * randomization_scale,
            )
            cfg.events.physics_material.params["dynamic_friction_range"] = (
                1.0 - 0.75 * randomization_scale,
                1.0 + 0.25 * randomization_scale,
            )
            cfg.events.base_payload.params["mass_distribution_params"] = (
                -2.0 * randomization_scale,
                12.5 * randomization_scale,
            )
            cfg.events.actuator_gains.params["stiffness_distribution_params"] = (
                1.0 - 0.1 * randomization_scale,
                1.0 + 0.1 * randomization_scale,
            )
            cfg.events.actuator_gains.params["damping_distribution_params"] = (
                1.0 - 0.1 * randomization_scale,
                1.0 + 0.1 * randomization_scale,
            )
        super().__init__(cfg, render_mode, **kwargs)
        self._imitation_reward_weight = float(cfg.imitation_reward_weight)
        self._reference_motion_progress = float(cfg.reference_motion_progress)
        self._task_mix_beta = float(cfg.task_mix_beta)

        joint_ids, joint_names = self._robot.find_joints(
            list(G1_JOINT_NAMES),
            preserve_order=True,
        )
        if tuple(joint_names) != G1_JOINT_NAMES:
            raise RuntimeError(
                "G1 joint contract mismatch. "
                f"Expected {G1_JOINT_NAMES}, got {tuple(joint_names)}."
            )
        if self._robot.num_joints != ACTION_DIM:
            raise RuntimeError(
                f"The no-hand G1 asset must expose exactly 29 joints; "
                f"found {self._robot.num_joints}."
            )
        self._joint_ids = joint_ids
        self._base_action_scales = torch.full(
            (ACTION_DIM,),
            float(cfg.action_scale),
            device=self.device,
        )
        self._v7_full_action_scales = torch.tensor(
            V7_FULL_ACTION_SCALES,
            dtype=torch.float32,
            device=self.device,
        )
        self._action_scales = self._base_action_scales.clone()
        if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
            self._update_action_scales()

        foot_ids, foot_names = self._robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        if tuple(foot_names) != (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ):
            raise RuntimeError(f"G1 foot body mismatch: got {tuple(foot_names)}")
        self._foot_body_ids = foot_ids
        self._foot_sensor_ids, _ = self._contact_sensor.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        allowed_contact_names = set(foot_names)
        all_contact_sensor_ids, all_contact_names = (
            self._termination_contact_sensor.find_bodies(
                ".*",
                preserve_order=True,
            )
        )
        missing_allowed_contacts = allowed_contact_names.difference(all_contact_names)
        if missing_allowed_contacts:
            raise RuntimeError(
                "G1 termination sensor omitted allowed foot bodies: "
                f"{sorted(missing_allowed_contacts)}"
            )
        illegal_contacts = [
            (body_id, body_name)
            for body_id, body_name in zip(all_contact_sensor_ids, all_contact_names)
            if body_name not in allowed_contact_names
        ]
        if not illegal_contacts:
            raise RuntimeError("G1 termination sensor found no non-foot bodies")
        self._illegal_contact_sensor_ids = [body_id for body_id, _ in illegal_contacts]
        self._illegal_contact_body_names = tuple(
            body_name for _, body_name in illegal_contacts
        )
        self._pelvis_body_ids, _ = self._robot.find_bodies("pelvis")
        if len(self._pelvis_body_ids) != 1:
            raise RuntimeError("G1 asset must contain exactly one pelvis body")
        self._pelvis_body_id = self._pelvis_body_ids[0]

        # PhysX exposes these properties on CPU. Keep immutable defaults so
        # per-episode COM and motor-strength sampling never compounds.
        if self.cfg.enable_domain_randomization:
            self._default_coms = self._robot.root_physx_view.get_coms().clone()
            self._default_effort_limits = (
                self._robot.root_physx_view.get_dof_max_forces().clone()
            )

        self._previous_action = torch.zeros(
            self.num_envs, ACTION_DIM, device=self.device
        )
        self._action_before_step = torch.zeros_like(self._previous_action)
        self._joint_position_targets = torch.zeros_like(self._previous_action)
        # Physical commands remain strictly 3-D. Phase is appended only when
        # constructing policy-facing command tensors for phase-aware profiles.
        self._commands = torch.zeros(
            self.num_envs,
            PHYSICAL_COMMAND_DIM,
            device=self.device,
        )
        self._commands_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._reference_phase_offset = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._reference_phase = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._last_reset_from_reference = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._history = torch.zeros(
            self.num_envs,
            HISTORY_STEPS,
            OBS_DIM,
            device=self.device,
        )
        self._history_needs_fill = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._observation_delay = torch.zeros(
            self.num_envs,
            3,
            OBS_DIM,
            device=self.device,
        )
        self._observation_delay_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._delay_needs_fill = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._motor_strength = torch.ones(
            self.num_envs,
            1,
            device=self.device,
        )
        self._robot_mass = torch.sum(
            self._robot.data.default_mass,
            dim=1,
        ).to(self.device)
        self._joint_effort_limits = torch.tensor(
            JOINT_EFFORT_LIMITS,
            device=self.device,
            dtype=torch.float32,
        )
        self._walk_reference = CyclicJointReference(
            _WALK_REFERENCE_PATH,
            G1_JOINT_NAMES,
            self.device,
            expected_archive_sha256=G1_WALK_ARCHIVE_SHA256,
            expected_source_sha256=G1_WALK_SOURCE_SHA256,
        )
        self._walk_reference.retarget_to_position_envelope(
            self._robot.data.default_joint_pos[0, self._joint_ids],
            max_offset=0.22,
            action_scale=float(cfg.action_scale),
        )
        if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
            if self._walk_reference.full_action_scales is None:
                raise RuntimeError("Full reference did not initialize action scales")
            self._v7_full_action_scales.copy_(self._walk_reference.full_action_scales)
            self._update_action_scales()
            self._reference_natural_forward_speed = float(
                self._walk_reference.natural_forward_speed(
                    self._reference_motion_progress
                ).item()
            )
        else:
            self._reference_natural_forward_speed = 1.0
        self._deepmimic_joint_weights = torch.tensor(
            G1_DEEPMIMIC_DOF_WEIGHTS,
            device=self.device,
            dtype=torch.float32,
        )
        self._previous_foot_force = torch.zeros(
            self.num_envs,
            2,
            3,
            device=self.device,
        )
        self._previous_applied_torque = torch.zeros_like(self._previous_action)
        self._previous_joint_velocity = torch.zeros_like(self._previous_action)
        self._last_landing_foot = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._steps_since_landing = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "linear_velocity",
                "commanded_progress",
                "command_stall",
                "feet_air_time",
                "contact_pattern",
                "alternating_step",
                "reference_pose",
                "reference_joint_velocity",
                "reference_root_height",
                "reference_root_velocity",
                "reference_foot_position",
                "reference_contact",
                "v8_core",
                "torque",
                "impact",
                "joint_velocity",
                "angular_velocity",
                "orientation",
                "height",
                "gait_foot_velocity",
                "gait_foot_force",
                "impact_smoothness",
                "torque_smoothness",
                "joint_velocity_smoothness",
                "cost_of_transport",
                "action_magnitude",
                "action_rate",
                "action_saturation",
                "termination",
                "vertical_velocity",
                "roll_pitch_angular_velocity",
                "foot_slip",
                "joint_position",
            )
        }
        self._shape_contract_reported = False

    def _setup_scene(self) -> None:
        if not _ASSET_PATH.is_file():
            raise FileNotFoundError(f"Missing Unitree G1 USD asset: {_ASSET_PATH}")

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._termination_contact_sensor = ContactSensor(
            self.cfg.termination_contact_sensor
        )
        self.scene.sensors["termination_contact_sensor"] = (
            self._termination_contact_sensor
        )

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"Expected actions [{self.num_envs}, {ACTION_DIM}], "
                f"got {tuple(actions.shape)}."
            )

        self._action_before_step.copy_(self._previous_action)
        self._previous_action.copy_(actions.clamp(-1.0, 1.0))

        default_position = self._robot.data.default_joint_pos[:, self._joint_ids]
        targets = default_position + self._action_scales * self._previous_action
        soft_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]
        self._joint_position_targets.copy_(
            targets.clamp(min=soft_limits[..., 0], max=soft_limits[..., 1])
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(
            self._joint_position_targets,
            joint_ids=self._joint_ids,
        )

    def set_imitation_reward_weight(self, weight: float) -> None:
        """Set the total scheduled imitation weight for V6/V7 treatments."""

        if self.cfg.reward_profile not in (
            PHASE_RSI_IMITATION_REWARD_PROFILE,
            FULL_REFERENCE_REWARD_PROFILE,
        ):
            raise RuntimeError(
                "imitation reward scheduling is only available for "
                "the phase-RSI imitation or V7 full-reference profile"
            )
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("imitation reward weight must be finite and non-negative")
        self._imitation_reward_weight = float(weight)
        self.cfg.imitation_reward_weight = float(weight)

    def set_task_mix_beta(self, beta: float) -> None:
        """Set the V8 interpolation from imitation (zero) to task (one)."""

        if self.cfg.reward_profile != FUTURE_REFERENCE_REWARD_PROFILE:
            raise RuntimeError(
                f"task mixing is only available for {FUTURE_REFERENCE_REWARD_PROFILE}"
            )
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("task mix beta must be finite and in [0, 1]")
        self._task_mix_beta = float(beta)
        self.cfg.task_mix_beta = float(beta)

    def set_reference_state_initialization_probability(
        self,
        probability: float,
    ) -> None:
        """Set the reference-state reset probability for phase-aware profiles."""

        if not self._phase_rsi_enabled:
            raise RuntimeError(
                "reference-state initialization requires a phase-aware profile"
            )
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "reference-state initialization probability must be finite "
                "and in [0, 1]"
            )
        self.cfg.reference_state_initialization_probability = float(probability)

    def _update_action_scales(self) -> None:
        self._action_scales.copy_(
            torch.lerp(
                self._base_action_scales,
                self._v7_full_action_scales,
                self._reference_motion_progress,
            )
        )

    def set_reference_motion_progress(self, progress: float) -> None:
        """Synchronize V7 action reachability with its reference curriculum."""

        if self.cfg.reward_profile != FULL_REFERENCE_REWARD_PROFILE:
            raise RuntimeError(
                "reference-motion scheduling is only available for "
                f"{FULL_REFERENCE_REWARD_PROFILE}"
            )
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("reference motion progress must be finite and in [0, 1]")
        self._reference_motion_progress = float(progress)
        self.cfg.reference_motion_progress = float(progress)
        self._update_action_scales()
        self._reference_natural_forward_speed = float(
            self._walk_reference.natural_forward_speed(progress).item()
        )

    def _reference_phase_rate(self) -> torch.Tensor:
        if self.cfg.reward_profile not in FULL_REFERENCE_REWARD_PROFILES:
            return torch.ones(self.num_envs, device=self.device)
        command_matched_rate = self._commands[:, 0].abs() / max(
            self._reference_natural_forward_speed,
            1.0e-6,
        )
        cadence_progress = min(
            self._reference_motion_progress / V7_CADENCE_RAMP_PROGRESS,
            1.0,
        )
        return 1.0 + cadence_progress * (command_matched_rate - 1.0)

    def _advance_reference_phase(self) -> None:
        if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
            self._reference_phase.add_(
                self.step_dt
                * self._reference_phase_rate()
                / self._walk_reference.duration_s
            ).remainder_(1.0)

    def _current_reference_time_s(self) -> torch.Tensor:
        return self._current_reference_phase() * self._walk_reference.duration_s

    def _current_reference_phase(self) -> torch.Tensor:
        if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
            return self._reference_phase
        return self._walk_reference.phase_at_time(
            self.episode_length_buf.to(torch.float32) * self.step_dt,
            self._reference_phase_offset,
        )

    def _command_observation(self) -> torch.Tensor:
        if not self._phase_rsi_enabled:
            return self._commands
        return phase_visible_command(
            self._commands,
            self._current_reference_phase(),
        )

    def _future_reference_observation(self) -> torch.Tensor:
        if not self._future_reference_enabled:
            raise RuntimeError("future reference requested for a non-V8 profile")
        return self._walk_reference.sample_future_reference(
            self._current_reference_phase(),
            self._reference_phase_rate(),
            progress=1.0,
        ).features

    def _current_clean_obs_and_privileged(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read current physical state without advancing delay/history buffers."""

        joint_position = self._robot.data.joint_pos[:, self._joint_ids]
        joint_velocity = self._robot.data.joint_vel[:, self._joint_ids]
        clean_obs = torch.cat(
            (
                self._robot.data.projected_gravity_b,
                self._robot.data.root_ang_vel_b,
                joint_position,
                joint_velocity,
                self._previous_action,
            ),
            dim=-1,
        )
        base_position_relative = self._robot.data.root_pos_w - self._terrain.env_origins
        privileged = torch.cat(
            (
                base_position_relative,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                joint_position,
                joint_velocity,
                self._robot.data.applied_torque[:, self._joint_ids],
            ),
            dim=-1,
        )
        return clean_obs, privileged

    def _get_observations(self) -> dict[str, torch.Tensor]:
        clean_obs, privileged = self._current_clean_obs_and_privileged()

        policy_obs = clean_obs.clone()
        if self.cfg.observation_noise:
            # Stochastic proprioceptive noise is applied only to the actor
            # stream. The critic and explicit velocity target stay exact.
            scale = self.cfg.domain_randomization_scale
            policy_obs[:, 0:3] += 0.01 * scale * torch.randn_like(policy_obs[:, 0:3])
            policy_obs[:, 3:6] += 0.05 * scale * torch.randn_like(policy_obs[:, 3:6])
            policy_obs[:, 6:35] += 0.01 * scale * torch.randn_like(policy_obs[:, 6:35])
            policy_obs[:, 35:64] += (
                0.10 * scale * torch.randn_like(policy_obs[:, 35:64])
            )

        # A three-slot FIFO realizes the paper's uniformly sampled 0--2
        # network-step observation latency while preserving the 93-D meaning.
        self._observation_delay = torch.roll(
            self._observation_delay,
            shifts=-1,
            dims=1,
        )
        self._observation_delay[:, -1] = policy_obs
        delay_fill_ids = self._delay_needs_fill.nonzero(as_tuple=False).squeeze(-1)
        if delay_fill_ids.numel() > 0:
            self._observation_delay[delay_fill_ids] = (
                policy_obs[delay_fill_ids].unsqueeze(1).expand(-1, 3, -1)
            )
            self._delay_needs_fill[delay_fill_ids] = False
        delay_indices = 2 - self._observation_delay_steps
        obs = self._observation_delay[
            torch.arange(self.num_envs, device=self.device),
            delay_indices,
        ]

        # Keep the oldest observation at index 0 and newest at index 49.
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = obs
        fill_ids = self._history_needs_fill.nonzero(as_tuple=False).squeeze(-1)
        if fill_ids.numel() > 0:
            self._history[fill_ids] = (
                obs[fill_ids].unsqueeze(1).expand(-1, HISTORY_STEPS, -1)
            )
            self._history_needs_fill[fill_ids] = False

        observation = {
            "history": self._history,
            "obs": obs,
            "command": self._command_observation(),
            "privileged": privileged,
            "explicit_target": self._robot.data.root_lin_vel_b,
        }
        if self._future_reference_enabled:
            observation["future_reference"] = self._future_reference_observation()
        self._validate_observation_contract(observation)
        return observation

    def _validate_observation_contract(
        self,
        observation: dict[str, torch.Tensor],
    ) -> None:
        command_dim = (
            PHASE_VISIBLE_COMMAND_DIM
            if self._phase_rsi_enabled
            else PHYSICAL_COMMAND_DIM
        )
        expected_shapes = {
            "history": (self.num_envs, HISTORY_STEPS, OBS_DIM),
            "obs": (self.num_envs, OBS_DIM),
            "command": (self.num_envs, command_dim),
            "privileged": (self.num_envs, PRIVILEGED_DIM),
            "explicit_target": (self.num_envs, 3),
        }
        if self._future_reference_enabled:
            expected_shapes["future_reference"] = (
                self.num_envs,
                FUTURE_REFERENCE_STEPS,
                FUTURE_REFERENCE_FEATURES_PER_STEP,
            )
        actual_shapes = {
            name: tuple(value.shape) for name, value in observation.items()
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"Observation contract mismatch: expected {expected_shapes}, "
                f"got {actual_shapes}."
            )
        if not self._shape_contract_reported:
            print(
                "G1 observation contract: "
                + ", ".join(f"{name}={shape}" for name, shape in actual_shapes.items())
                + f", action=({self.num_envs}, {ACTION_DIM})"
            )
            self._shape_contract_reported = True

    def _get_rewards(self) -> torch.Tensor:
        def gaussian(x: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
            # Equation (1) is typeset without the leading minus sign, which
            # would reward increasing error. The intended bell-shaped kernel
            # described in the text is exp(-(x / sigma)^2).
            return alpha * torch.exp(-torch.square(x / sigma))

        def cauchy(
            x: torch.Tensor,
            alpha: float,
            beta: int,
            sigma: float,
        ) -> torch.Tensor:
            return alpha / (torch.pow(torch.abs(x) / sigma, 2 * beta) + 1.0)

        root_linear_velocity = self._robot.data.root_lin_vel_b
        root_linear_velocity_world = self._robot.data.root_lin_vel_w
        root_linear_velocity_heading = quat_rotate_inverse(
            yaw_quat(self._robot.data.root_quat_w),
            root_linear_velocity_world,
        )
        root_angular_velocity = self._robot.data.root_ang_vel_b
        root_angular_velocity_world = self._robot.data.root_ang_vel_w
        joint_position = self._robot.data.joint_pos[:, self._joint_ids]
        joint_velocity = self._robot.data.joint_vel[:, self._joint_ids]
        applied_torque = self._robot.data.applied_torque[:, self._joint_ids]
        full_reference = None
        if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
            reference_phase = self._current_reference_phase()
            full_reference = self._walk_reference.sample_full_reference(
                reference_phase,
                self._reference_motion_progress,
            )
            phase_rate = self._reference_phase_rate()
            reference_position = full_reference.joint_position
            reference_velocity = full_reference.joint_velocity * phase_rate.unsqueeze(
                -1
            )
        else:
            reference_position, reference_velocity, reference_phase = (
                self._walk_reference.sample(self._current_reference_time_s())
            )
        (
            reference_pose_similarity,
            reference_velocity_similarity,
            reference_pose_error,
            reference_velocity_error,
        ) = deepmimic_joint_similarity(
            joint_position,
            joint_velocity,
            reference_position,
            reference_velocity,
            self._deepmimic_joint_weights,
        )
        self.extras["imitation_diagnostics"] = {
            "phase": reference_phase.clone(),
            "pose_similarity": reference_pose_similarity.clone(),
            "velocity_similarity": reference_velocity_similarity.clone(),
            "pose_error": reference_pose_error.clone(),
            "velocity_error": reference_velocity_error.clone(),
        }
        desired_linear_velocity = torch.cat(
            (
                self._commands[:, :2],
                torch.zeros(self.num_envs, 1, device=self.device),
            ),
            dim=-1,
        )
        desired_angular_velocity = torch.cat(
            (
                torch.zeros(self.num_envs, 2, device=self.device),
                self._commands[:, 2:3],
            ),
            dim=-1,
        )
        base_height = (
            self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        )
        orientation_deviation = torch.abs(
            1.0 + self._robot.data.projected_gravity_b[:, 2]
        )
        foot_force_vectors = self._contact_sensor.data.net_forces_w[
            :, self._foot_sensor_ids
        ]
        foot_forces = torch.linalg.vector_norm(foot_force_vectors, dim=-1)
        mass_times_gravity = (9.81 * self._robot_mass).clamp_min(1.0)
        current_air_time = self._contact_sensor.data.current_air_time[
            :, self._foot_sensor_ids
        ]
        current_contact_time = self._contact_sensor.data.current_contact_time[
            :, self._foot_sensor_ids
        ]
        in_contact = current_contact_time > 0.0
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self._foot_sensor_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[
            :, self._foot_sensor_ids
        ]
        valid_landing = torch.logical_and(
            torch.logical_and(first_contact, last_air_time >= 0.12),
            foot_forces > 0.05 * mass_times_gravity.unsqueeze(-1),
        )
        self.extras["gait_diagnostics"] = {
            "in_contact": in_contact.clone(),
            "valid_landing": valid_landing.clone(),
        }
        reference_root_height_similarity = torch.zeros_like(base_height)
        reference_root_velocity_similarity = torch.zeros_like(base_height)
        reference_foot_position_similarity = torch.zeros_like(base_height)
        reference_contact_similarity = torch.zeros_like(base_height)
        if full_reference is not None:
            root_quaternion = (
                self._robot.data.root_quat_w.unsqueeze(1)
                .expand(-1, len(self._foot_body_ids), -1)
                .reshape(-1, 4)
            )
            ankle_position_world = self._robot.data.body_pos_w[:, self._foot_body_ids]
            ankle_position_relative = (
                ankle_position_world - self._robot.data.root_pos_w.unsqueeze(1)
            )
            ankle_position_pelvis = quat_rotate_inverse(
                root_quaternion,
                ankle_position_relative.reshape(-1, 3),
            ).reshape(self.num_envs, len(self._foot_body_ids), 3)
            foot_position_error = torch.sum(
                torch.square(
                    ankle_position_pelvis - full_reference.ankle_position_pelvis
                ),
                dim=(-1, -2),
            )
            reference_foot_position_similarity = torch.exp(-10.0 * foot_position_error)
            reference_root_height_similarity = torch.exp(
                -20.0 * torch.square(base_height - full_reference.root_height)
            )
            reference_root_velocity = (
                full_reference.root_linear_velocity * phase_rate.unsqueeze(-1)
            )
            root_velocity_error = torch.sum(
                torch.square(root_linear_velocity - reference_root_velocity),
                dim=-1,
            )
            reference_root_velocity_similarity = torch.exp(-root_velocity_error)
            reference_contact_similarity = 1.0 - torch.mean(
                torch.abs(in_contact.to(torch.float32) - full_reference.contact_target),
                dim=-1,
            )
            self.extras["imitation_diagnostics"].update(
                {
                    "root_height_similarity": reference_root_height_similarity.clone(),
                    "root_velocity_similarity": (
                        reference_root_velocity_similarity.clone()
                    ),
                    "foot_position_similarity": (
                        reference_foot_position_similarity.clone()
                    ),
                    "contact_similarity": reference_contact_similarity.clone(),
                    "target_contact": full_reference.contact_target.clone(),
                    "reference_progress": full_reference.progress.clone(),
                    "phase_rate": phase_rate.clone(),
                }
            )

        if self.cfg.reward_profile == FUTURE_REFERENCE_REWARD_PROFILE:
            if full_reference is None:
                raise RuntimeError("V8 requires a full reference sample")

            command_speed = torch.linalg.vector_norm(self._commands[:, :2], dim=-1)
            command_direction = self._commands[:, :2] / command_speed.clamp_min(
                0.05
            ).unsqueeze(-1)
            directional_velocity = torch.sum(
                command_direction * root_linear_velocity_heading[:, :2],
                dim=-1,
            )
            directional_progress = directional_velocity / command_speed.clamp_min(0.20)
            planar_tracking_error = torch.sum(
                torch.square(
                    self._commands[:, :2] - root_linear_velocity_heading[:, :2]
                ),
                dim=-1,
            )
            yaw_tracking_error = torch.square(
                self._commands[:, 2] - root_angular_velocity_world[:, 2]
            )
            tilt_error_squared = torch.sum(
                torch.square(self._robot.data.projected_gravity_b[:, :2]),
                dim=-1,
            )
            # Keep the task endpoint independent of the demonstration so
            # beta=1 is a genuine command-following objective.
            height_error = base_height - DESIRED_BASE_HEIGHT
            planar_tracking_similarity = torch.exp(-planar_tracking_error / 0.16)
            yaw_tracking_similarity = torch.exp(-yaw_tracking_error / 0.25)
            upright_similarity = torch.exp(-tilt_error_squared / 0.10)
            task_height_similarity = torch.exp(-torch.square(height_error / 0.10))

            force_contact_score, force_contact = _v8_force_contact_score(
                foot_forces,
                mass_times_gravity,
                full_reference.contact_target,
            )
            (
                clearance_score,
                actual_clearance,
                target_clearance,
            ) = _v8_clearance_score(
                ankle_position_world[..., 2],
                self._terrain.env_origins[:, 2],
                full_reference.ankle_position_pelvis[..., 2],
            )
            support_score = _v8_support_score(force_contact)

            self._steps_since_landing.add_(1).clamp_max_(10_000)
            (
                single_landing,
                alternating_landing,
                repeated_landing,
                next_last_landing_foot,
            ) = classify_foot_landings(
                valid_landing,
                self._last_landing_foot,
            )
            inter_landing_time = (
                self._steps_since_landing.to(torch.float32) * self.step_dt
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
            landing_score = _v8_landing_score(
                alternating_landing,
                repeated_landing,
            )
            self._last_landing_foot.copy_(next_last_landing_foot)
            self._steps_since_landing.copy_(
                torch.where(
                    single_landing,
                    torch.zeros_like(self._steps_since_landing),
                    self._steps_since_landing,
                )
            )

            signed_pose_score = _signed_similarity(reference_pose_similarity)
            signed_joint_velocity_score = _signed_similarity(
                reference_velocity_similarity
            )
            signed_foot_position_score = _signed_similarity(
                reference_foot_position_similarity
            )
            signed_root_height_score = _signed_similarity(
                reference_root_height_similarity
            )
            signed_root_velocity_score = _signed_similarity(
                reference_root_velocity_similarity
            )
            signed_planar_tracking_score = _signed_similarity(
                planar_tracking_similarity
            )
            signed_yaw_tracking_score = _signed_similarity(yaw_tracking_similarity)
            signed_upright_score = _signed_similarity(upright_similarity)
            signed_task_height_score = _signed_similarity(task_height_similarity)

            imitation_score = _v8_imitation_score(
                reference_pose_similarity,
                reference_velocity_similarity,
                reference_foot_position_similarity,
                reference_root_height_similarity,
                reference_root_velocity_similarity,
                force_contact_score,
                clearance_score,
            )
            task_score = _v8_task_score(
                planar_tracking_similarity,
                directional_progress,
                yaw_tracking_similarity,
                upright_similarity,
                task_height_similarity,
                support_score,
                landing_score,
            )
            core_reward, mixed_score = _v8_core_reward(
                imitation_score,
                task_score,
                self._task_mix_beta,
            )
            rewards = {"v8_core": core_reward}

            # Keep V7's safety costs and hard termination outside the V8 mix.
            vertical_velocity = root_linear_velocity_world[:, 2]
            rewards["vertical_velocity"] = -0.20 * torch.square(
                vertical_velocity
            ).clamp(max=4.0)
            rewards["roll_pitch_angular_velocity"] = -0.10 * torch.sum(
                torch.square(root_angular_velocity[:, :2]),
                dim=-1,
            ).clamp(max=10.0)
            foot_planar_speed_squared = torch.sum(
                torch.square(
                    self._robot.data.body_lin_vel_w[:, self._foot_body_ids, :2]
                ),
                dim=-1,
            )
            rewards["foot_slip"] = -0.10 * torch.sum(
                force_contact.to(torch.float32) * foot_planar_speed_squared,
                dim=-1,
            ).clamp(max=10.0)
            default_position = self._robot.data.default_joint_pos[:, self._joint_ids]
            joint_position_error = torch.mean(
                torch.square(joint_position - default_position),
                dim=-1,
            ).clamp(max=4.0)
            rewards["joint_position"] = -0.01 * joint_position_error
            rewards["action_magnitude"] = -0.20 * torch.mean(
                self._previous_action.square(),
                dim=-1,
            )
            rewards["action_rate"] = -0.10 * torch.mean(
                (self._previous_action - self._action_before_step).square(),
                dim=-1,
            )
            rewards["action_saturation"] = -2.00 * torch.mean(
                torch.square(torch.relu(self._previous_action.abs() - 0.80) / 0.20),
                dim=-1,
            )
            normalized_torque = applied_torque / self._joint_effort_limits
            rewards["torque"] = -0.05 * torch.mean(
                normalized_torque.square(),
                dim=-1,
            )
            rewards["joint_velocity"] = -0.02 * torch.mean(
                torch.square(joint_velocity / 10.0),
                dim=-1,
            )
            impact_delta = (foot_force_vectors - self._previous_foot_force).flatten(1)
            impact_ratio = (
                torch.linalg.vector_norm(impact_delta, dim=-1) / mass_times_gravity
            )
            impact_valid = (self.episode_length_buf > 1).to(torch.float32)
            rewards["impact"] = (
                -0.05 * torch.square(impact_ratio).clamp(max=4.0) * impact_valid
            )
            rewards["termination"] = -100.0 * self.reset_terminated.to(torch.float32)

            self.extras["v8_diagnostics"] = {
                "imitation_score": imitation_score.clone(),
                "task_score": task_score.clone(),
                "mixed_score": mixed_score.clone(),
                "core_reward": core_reward.clone(),
                "force_contact_score": force_contact_score.clone(),
                "clearance_score": clearance_score.clone(),
                "support_score": support_score.clone(),
                "landing_score": landing_score.clone(),
                "task_mix_beta": torch.full_like(
                    imitation_score,
                    self._task_mix_beta,
                ),
                "pose_score": signed_pose_score.clone(),
                "joint_velocity_score": signed_joint_velocity_score.clone(),
                "foot_position_score": signed_foot_position_score.clone(),
                "root_height_score": signed_root_height_score.clone(),
                "root_velocity_score": signed_root_velocity_score.clone(),
                "planar_tracking_score": signed_planar_tracking_score.clone(),
                "directional_progress_score": directional_progress.clamp(
                    -1.0,
                    1.0,
                ).clone(),
                "yaw_tracking_score": signed_yaw_tracking_score.clone(),
                "upright_score": signed_upright_score.clone(),
                "height_score": signed_task_height_score.clone(),
                "actual_contact": force_contact.clone(),
                "target_contact": full_reference.contact_target.clone(),
                "actual_clearance_m": actual_clearance.clone(),
                "target_clearance_m": target_clearance.clone(),
            }
        elif self.cfg.reward_profile in (
            "p1_stable",
            "p1_walk",
            "p1_walk_gait",
            "p1_walk_gait_v2",
            "p1_walk_stable_v3",
            "p1_walk_stable_v4",
            "p1_walk_stable_v5",
            "p1_walk_stable_v5_imitation",
            *PHASE_RSI_REWARD_PROFILES,
        ):
            walk_profile = self.cfg.reward_profile != "p1_stable"
            gait_profile = self.cfg.reward_profile in (
                "p1_walk_gait",
                "p1_walk_gait_v2",
                "p1_walk_stable_v3",
                "p1_walk_stable_v4",
                "p1_walk_stable_v5",
                "p1_walk_stable_v5_imitation",
                *PHASE_RSI_REWARD_PROFILES,
            )
            strong_gait_profile = self.cfg.reward_profile == "p1_walk_gait_v2"
            stable_v3_profile = self.cfg.reward_profile in (
                "p1_walk_stable_v3",
                "p1_walk_stable_v4",
                "p1_walk_stable_v5",
                "p1_walk_stable_v5_imitation",
                *PHASE_RSI_REWARD_PROFILES,
            )
            strong_motion_profile = self.cfg.reward_profile in (
                "p1_walk_stable_v4",
                "p1_walk_stable_v5",
                "p1_walk_stable_v5_imitation",
                *PHASE_RSI_REWARD_PROFILES,
            )
            alternating_v5_profile = self.cfg.reward_profile in (
                "p1_walk_stable_v5",
                "p1_walk_stable_v5_imitation",
                *PHASE_RSI_REWARD_PROFILES,
            )
            imitation_profile = self.cfg.reward_profile in (
                "p1_walk_stable_v5_imitation",
                PHASE_RSI_IMITATION_REWARD_PROFILE,
                FULL_REFERENCE_REWARD_PROFILE,
            )
            tracking_linear_velocity = (
                root_linear_velocity_heading
                if stable_v3_profile
                else root_linear_velocity
            )
            tracking_yaw_rate = (
                root_angular_velocity_world[:, 2]
                if stable_v3_profile
                else root_angular_velocity[:, 2]
            )
            tilt_error_squared = torch.sum(
                torch.square(self._robot.data.projected_gravity_b[:, :2]),
                dim=-1,
            )
            planar_tracking_error = torch.sum(
                torch.square(self._commands[:, :2] - tracking_linear_velocity[:, :2]),
                dim=-1,
            )
            yaw_tracking_error = torch.square(self._commands[:, 2] - tracking_yaw_rate)
            if strong_gait_profile:
                linear_velocity_weight = 3.50
                linear_velocity_width = 0.12
            elif stable_v3_profile:
                linear_velocity_weight = 2.50
                linear_velocity_width = 0.16
            else:
                linear_velocity_weight = 2.50 if walk_profile else 1.50
                linear_velocity_width = 0.16 if walk_profile else 0.25
            desired_base_height = (
                torch.lerp(
                    torch.full_like(base_height, DESIRED_BASE_HEIGHT),
                    full_reference.root_height,
                    self._reference_motion_progress,
                )
                if full_reference is not None
                else torch.full_like(base_height, DESIRED_BASE_HEIGHT)
            )
            rewards = {
                "linear_velocity": linear_velocity_weight
                * torch.exp(-planar_tracking_error / linear_velocity_width),
                "angular_velocity": 0.75 * torch.exp(-yaw_tracking_error / 0.25),
                "orientation": 0.50 * torch.exp(-tilt_error_squared / 0.10),
                "height": gaussian(
                    torch.abs(desired_base_height - base_height),
                    0.20,
                    0.10,
                ),
            }
            if stable_v3_profile:
                upright_gate = (
                    (-self._robot.data.projected_gravity_b[:, 2] - 0.70) / 0.20
                ).clamp(min=0.0, max=1.0)
                height_gate = ((base_height - 0.55) / 0.15).clamp(
                    min=0.0,
                    max=1.0,
                )
                locomotion_stability_gate = upright_gate * height_gate
                rewards["linear_velocity"] *= locomotion_stability_gate
            else:
                locomotion_stability_gate = torch.ones_like(base_height)
            if walk_profile:
                command_speed = torch.linalg.vector_norm(
                    self._commands[:, :2],
                    dim=-1,
                )
                moving_command = command_speed > 0.05
                command_direction = self._commands[:, :2] / command_speed.clamp_min(
                    0.05
                ).unsqueeze(-1)
                directional_velocity = torch.sum(
                    command_direction * tracking_linear_velocity[:, :2],
                    dim=-1,
                )
                relative_progress = directional_velocity / command_speed.clamp_min(0.20)
                progress_weight = (
                    1.00 if strong_gait_profile or strong_motion_profile else 0.50
                )
                rewards["commanded_progress"] = torch.where(
                    moving_command,
                    progress_weight
                    * relative_progress.clamp(min=-1.0, max=1.0)
                    * locomotion_stability_gate,
                    torch.zeros_like(relative_progress),
                )
                if strong_gait_profile or stable_v3_profile:
                    stall_error = ((0.15 - directional_velocity) / 0.15).clamp(
                        min=0.0, max=2.0
                    )
                    if strong_motion_profile:
                        stall_weight = 1.00
                    elif stable_v3_profile:
                        stall_weight = 0.15
                    else:
                        stall_weight = 0.75
                    rewards["command_stall"] = torch.where(
                        moving_command,
                        -stall_weight * stall_error,
                        torch.zeros_like(stall_error),
                    )
                if gait_profile:
                    mode_time = torch.where(
                        in_contact,
                        current_contact_time,
                        current_air_time,
                    )
                    single_stance = torch.sum(in_contact.to(torch.int32), dim=-1) == 1
                    single_stance_time = torch.min(
                        torch.where(
                            single_stance.unsqueeze(-1),
                            mode_time,
                            torch.zeros_like(mode_time),
                        ),
                        dim=-1,
                    ).values.clamp(max=0.30)
                    landing_air_time = torch.sum(
                        (last_air_time - 0.12).clamp(min=0.0, max=0.35) * first_contact,
                        dim=-1,
                    )
                    single_stance_weight = 0.50 if strong_gait_profile else 0.25
                    landing_weight = 1.00 if strong_gait_profile else 0.50
                    moving_float = moving_command.to(torch.float32)
                    if alternating_v5_profile:
                        swing_air_time = torch.max(
                            torch.where(
                                torch.logical_not(in_contact),
                                current_air_time,
                                torch.zeros_like(current_air_time),
                            ),
                            dim=-1,
                        ).values
                        stance_contact_time = torch.max(
                            torch.where(
                                in_contact,
                                current_contact_time,
                                torch.zeros_like(current_contact_time),
                            ),
                            dim=-1,
                        ).values
                        timed_single_stance = torch.logical_and(
                            single_stance,
                            torch.logical_and(
                                torch.logical_and(
                                    swing_air_time >= 0.04,
                                    swing_air_time <= 0.45,
                                ),
                                stance_contact_time <= 0.45,
                            ),
                        )
                        rewards["feet_air_time"] = (
                            0.03
                            * moving_float
                            * locomotion_stability_gate
                            * torch.logical_not(self.reset_terminated).to(torch.float32)
                            * timed_single_stance.to(torch.float32)
                        )
                    else:
                        rewards["feet_air_time"] = (
                            moving_float
                            * (
                                single_stance_weight * single_stance_time
                                + landing_weight * landing_air_time
                            )
                            * locomotion_stability_gate
                        )
                    if strong_gait_profile:
                        contact_count = torch.sum(in_contact.to(torch.int32), dim=-1)
                        contact_pattern = torch.where(
                            contact_count == 1,
                            torch.full_like(command_speed, 0.10),
                            torch.where(
                                contact_count == 0,
                                torch.full_like(command_speed, -0.10),
                                torch.full_like(command_speed, -0.05),
                            ),
                        )
                        rewards["contact_pattern"] = moving_float * contact_pattern
                    elif alternating_v5_profile:
                        contact_count = torch.sum(in_contact.to(torch.int32), dim=-1)
                        flight_penalty = torch.where(
                            contact_count == 0,
                            torch.full_like(command_speed, -0.10),
                            torch.zeros_like(command_speed),
                        )
                        rewards["contact_pattern"] = moving_float * flight_penalty
                        self._steps_since_landing.add_(1).clamp_max_(10_000)
                        (
                            single_landing,
                            alternating_landing,
                            repeated_landing,
                            next_last_landing_foot,
                        ) = classify_foot_landings(
                            valid_landing,
                            self._last_landing_foot,
                        )
                        inter_landing_time = (
                            self._steps_since_landing.to(torch.float32) * self.step_dt
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
                        rewards["alternating_step"] = (
                            moving_float
                            * locomotion_stability_gate
                            * (
                                torch.logical_not(self.reset_terminated).to(
                                    torch.float32
                                )
                                * alternating_landing.to(torch.float32)
                                - 0.5 * repeated_landing.to(torch.float32)
                            )
                        )
                        self._last_landing_foot.copy_(next_last_landing_foot)
                        self._steps_since_landing.copy_(
                            torch.where(
                                single_landing,
                                torch.zeros_like(self._steps_since_landing),
                                self._steps_since_landing,
                            )
                        )
                    elif stable_v3_profile:
                        contact_count = torch.sum(in_contact.to(torch.int32), dim=-1)
                        flight_penalty = torch.where(
                            contact_count == 0,
                            torch.full_like(command_speed, -0.10),
                            torch.zeros_like(command_speed),
                        )
                        rewards["contact_pattern"] = moving_float * flight_penalty
                if imitation_profile:
                    if self.cfg.reward_profile == FULL_REFERENCE_REWARD_PROFILE:
                        imitation_gate = moving_command.to(
                            torch.float32
                        ) * torch.logical_not(self.reset_terminated).to(torch.float32)
                        total_weight = self._imitation_reward_weight
                        rewards["reference_pose"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["pose"]
                            * reference_pose_similarity
                            * imitation_gate
                        )
                        rewards["reference_joint_velocity"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["joint_velocity"]
                            * reference_velocity_similarity
                            * imitation_gate
                        )
                        rewards["reference_root_height"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["root_height"]
                            * reference_root_height_similarity
                            * imitation_gate
                        )
                        rewards["reference_root_velocity"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["root_velocity"]
                            * reference_root_velocity_similarity
                            * imitation_gate
                        )
                        rewards["reference_foot_position"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["foot_position"]
                            * reference_foot_position_similarity
                            * imitation_gate
                        )
                        rewards["reference_contact"] = (
                            total_weight
                            * V7_REFERENCE_COMPONENT_FRACTIONS["contact"]
                            * reference_contact_similarity
                            * imitation_gate
                        )
                    else:
                        imitation_gate = (
                            moving_command.to(torch.float32)
                            * locomotion_stability_gate
                            * torch.logical_not(self.reset_terminated).to(torch.float32)
                        )
                        imitation_reward_weights = {}
                        if (
                            self.cfg.reward_profile
                            == PHASE_RSI_IMITATION_REWARD_PROFILE
                        ):
                            pose_weight, velocity_weight = (
                                split_imitation_reward_weight(
                                    self._imitation_reward_weight
                                )
                            )
                            imitation_reward_weights = {
                                "pose_weight": pose_weight,
                                "velocity_weight": velocity_weight,
                            }
                        reference_pose_reward, reference_velocity_reward = (
                            gated_imitation_rewards(
                                reference_pose_similarity,
                                reference_velocity_similarity,
                                imitation_gate,
                                **imitation_reward_weights,
                            )
                        )
                        rewards["reference_pose"] = reference_pose_reward
                        rewards["reference_joint_velocity"] = reference_velocity_reward
            vertical_velocity = (
                root_linear_velocity[:, 2]
                - self._reference_motion_progress * reference_root_velocity[:, 2]
                if full_reference is not None
                else (
                    root_linear_velocity_world[:, 2]
                    if stable_v3_profile
                    else root_linear_velocity[:, 2]
                )
            )
            vertical_velocity_weight = 0.20 if stable_v3_profile else 0.10
            rewards["vertical_velocity"] = -vertical_velocity_weight * torch.square(
                vertical_velocity
            ).clamp(max=4.0)
            roll_pitch_weight = 0.10 if stable_v3_profile else 0.05
            rewards["roll_pitch_angular_velocity"] = -roll_pitch_weight * torch.sum(
                torch.square(root_angular_velocity[:, :2]),
                dim=-1,
            ).clamp(max=10.0)

            contact_threshold = 0.05 * mass_times_gravity.unsqueeze(-1)
            foot_contacts = foot_forces > contact_threshold
            foot_planar_speed_squared = torch.sum(
                torch.square(
                    self._robot.data.body_lin_vel_w[:, self._foot_body_ids, :2]
                ),
                dim=-1,
            )
            foot_slip_weight = 0.10 if stable_v3_profile else 0.05
            rewards["foot_slip"] = -foot_slip_weight * torch.sum(
                foot_contacts.to(torch.float32) * foot_planar_speed_squared,
                dim=-1,
            ).clamp(max=10.0)

            default_position = self._robot.data.default_joint_pos[:, self._joint_ids]
            joint_position_error = torch.mean(
                torch.square(joint_position - default_position),
                dim=-1,
            ).clamp(max=4.0)
            standing = (
                torch.linalg.vector_norm(
                    self._commands,
                    dim=-1,
                )
                < 0.05
            )
            posture_weight = torch.where(
                standing,
                0.15,
                0.01 if stable_v3_profile else (0.005 if walk_profile else 0.01),
            )
            rewards["joint_position"] = -posture_weight * joint_position_error
            action_magnitude_weight = 0.20 if stable_v3_profile else 0.02
            action_rate_weight = 0.10 if stable_v3_profile else 0.02
            saturation_weight = 2.00 if stable_v3_profile else 0.25
            rewards["action_magnitude"] = -action_magnitude_weight * torch.mean(
                self._previous_action.square(),
                dim=-1,
            )
            rewards["action_rate"] = -action_rate_weight * torch.mean(
                (self._previous_action - self._action_before_step).square(),
                dim=-1,
            )
            rewards["action_saturation"] = -saturation_weight * torch.mean(
                torch.square(torch.relu(self._previous_action.abs() - 0.80) / 0.20),
                dim=-1,
            )
            if stable_v3_profile:
                normalized_torque = applied_torque / self._joint_effort_limits
                rewards["torque"] = -0.05 * torch.mean(
                    normalized_torque.square(),
                    dim=-1,
                )
                rewards["joint_velocity"] = -0.02 * torch.mean(
                    torch.square(joint_velocity / 10.0),
                    dim=-1,
                )
                impact_delta = (foot_force_vectors - self._previous_foot_force).flatten(
                    1
                )
                impact_ratio = (
                    torch.linalg.vector_norm(
                        impact_delta,
                        dim=-1,
                    )
                    / mass_times_gravity
                )
                impact_valid = (self.episode_length_buf > 1).to(torch.float32)
                rewards["impact"] = (
                    -0.05 * torch.square(impact_ratio).clamp(max=4.0) * impact_valid
                )
                termination_weight = 100.0
            else:
                termination_weight = 5.0
            rewards["termination"] = -termination_weight * self.reset_terminated.to(
                torch.float32
            )
        else:
            if self.cfg.reward_profile == "p1_dense":
                linear_parameters = (0.35, 0.50)
                angular_parameters = (0.25, 0.50)
                orientation_parameters = (0.15, 0.05)
                height_parameters = (0.15, 0.08)
                gait_weight = 0.05
                regularity_weight = 0.025
            else:
                linear_parameters = (0.1, 0.02)
                angular_parameters = (0.1, 0.02)
                orientation_parameters = (0.1, 0.0025)
                height_parameters = (0.2, 0.02)
                gait_weight = 0.1
                regularity_weight = 0.1

            rewards = {
                "linear_velocity": gaussian(
                    torch.linalg.vector_norm(
                        desired_linear_velocity - root_linear_velocity,
                        dim=-1,
                    ),
                    *linear_parameters,
                ),
                "angular_velocity": gaussian(
                    torch.linalg.vector_norm(
                        desired_angular_velocity - root_angular_velocity,
                        dim=-1,
                    ),
                    *angular_parameters,
                ),
                "orientation": gaussian(
                    orientation_deviation,
                    *orientation_parameters,
                ),
                "height": gaussian(
                    torch.abs(DESIRED_BASE_HEIGHT - base_height),
                    *height_parameters,
                ),
            }
            foot_velocities = torch.linalg.vector_norm(
                self._robot.data.body_lin_vel_w[:, self._foot_body_ids],
                dim=-1,
            )
            gait_phase = torch.remainder(
                self.episode_length_buf * self.step_dt / GAIT_PERIOD_S,
                1.0,
            )
            left_stance = (gait_phase < 0.5).to(torch.float32)
            stance_weights = torch.stack(
                (left_stance, 1.0 - left_stance),
                dim=-1,
            )
            moving = (
                torch.linalg.vector_norm(self._commands, dim=-1) > 0.10
            ).unsqueeze(-1)
            stance_weights = torch.where(
                moving,
                stance_weights,
                torch.ones_like(stance_weights),
            )
            swing_weights = torch.where(
                moving,
                1.0 - stance_weights,
                torch.zeros_like(stance_weights),
            )
            rewards["gait_foot_velocity"] = cauchy(
                torch.sum(stance_weights * foot_velocities, dim=-1),
                gait_weight,
                1,
                8.0,
            )
            rewards["gait_foot_force"] = cauchy(
                torch.sum(swing_weights * foot_forces, dim=-1),
                gait_weight,
                1,
                8.0,
            )

            planar_speed = torch.linalg.vector_norm(
                root_linear_velocity[:, :2],
                dim=-1,
            ).clamp_min(0.10)
            impact_delta = (foot_force_vectors - self._previous_foot_force).flatten(1)
            impact_ratio = (
                torch.linalg.vector_norm(
                    impact_delta,
                    dim=-1,
                )
                / mass_times_gravity
            )
            rewards["impact_smoothness"] = cauchy(
                impact_ratio,
                regularity_weight,
                3,
                0.2,
            )
            rewards["torque_smoothness"] = cauchy(
                torch.linalg.vector_norm(
                    applied_torque - self._previous_applied_torque,
                    dim=-1,
                ),
                regularity_weight,
                2,
                160.0,
            )
            rewards["joint_velocity_smoothness"] = cauchy(
                torch.linalg.vector_norm(
                    joint_velocity - self._previous_joint_velocity,
                    dim=-1,
                )
                / planar_speed,
                regularity_weight,
                1,
                8.0,
            )
            rewards["cost_of_transport"] = cauchy(
                torch.abs(torch.sum(applied_torque * joint_velocity, dim=-1))
                / (mass_times_gravity * planar_speed),
                regularity_weight,
                3,
                1.6,
            )
            if self.cfg.reward_profile == "p1_dense":
                rewards["action_magnitude"] = -0.02 * torch.mean(
                    self._previous_action.square(),
                    dim=-1,
                )
                rewards["action_rate"] = -0.05 * torch.mean(
                    (self._previous_action - self._action_before_step).square(),
                    dim=-1,
                )
                rewards["termination"] = -self.reset_terminated.to(torch.float32)

        self._previous_foot_force.copy_(foot_force_vectors)
        self._previous_applied_torque.copy_(applied_torque)
        self._previous_joint_velocity.copy_(joint_velocity)
        for name, value in rewards.items():
            self._episode_sums[name] += value

        # The paper reports undiscounted episode rewards around 1300. Reward
        # kernels are therefore summed per 100-Hz policy step, not multiplied
        # by dt (which would change that scale by 100x).
        return torch.sum(torch.stack(tuple(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._advance_reference_phase()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = (
            self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        )
        if self.cfg.reward_profile in (
            "p1_walk_stable_v3",
            "p1_walk_stable_v4",
            "p1_walk_stable_v5",
            "p1_walk_stable_v5_imitation",
            *PHASE_RSI_REWARD_PROFILES,
        ):
            tipped = self._robot.data.projected_gravity_b[:, 2] > -0.70
            invalid_height = torch.logical_or(
                base_height < 0.55,
                base_height > 1.20,
            )
            illegal_forces = self._termination_contact_sensor.data.net_forces_w_history[
                :, :, self._illegal_contact_sensor_ids, :
            ]
            illegal_contact = torch.any(
                torch.linalg.vector_norm(illegal_forces, dim=-1) > 10.0,
                dim=(1, 2),
            )
            terminated = torch.logical_or(
                torch.logical_or(tipped, invalid_height),
                illegal_contact,
            )
        else:
            tipped = self._robot.data.projected_gravity_b[:, 2] > -0.25
            invalid_height = torch.logical_or(
                base_height < 0.45,
                base_height > 1.20,
            )
            terminated = torch.logical_or(tipped, invalid_height)
        episode_end = torch.logical_or(terminated, time_out)
        episode_end_ids = episode_end.nonzero(as_tuple=False).squeeze(-1)
        successful_time_out = torch.logical_and(time_out, torch.logical_not(terminated))
        time_out_ids = successful_time_out.nonzero(as_tuple=False).squeeze(-1)
        if episode_end_ids.numel() > 0:
            clean_obs, privileged = self._current_clean_obs_and_privileged()
            command_observation = self._command_observation()
            future_reference_observation = (
                self._future_reference_observation()
                if self._future_reference_enabled
                else None
            )
            self.extras["terminal_observation"] = {
                "env_ids": episode_end_ids.clone(),
                "obs": clean_obs[episode_end_ids].clone(),
                "command": command_observation[episode_end_ids].clone(),
                "privileged": privileged[episode_end_ids].clone(),
            }
            if future_reference_observation is not None:
                self.extras["terminal_observation"]["future_reference"] = (
                    future_reference_observation[episode_end_ids].clone()
                )
            if time_out_ids.numel() > 0:
                self.extras["time_out_critic_observation"] = {
                    "env_ids": time_out_ids.clone(),
                    "obs": clean_obs[time_out_ids].clone(),
                    "command": command_observation[time_out_ids].clone(),
                    "privileged": privileged[time_out_ids].clone(),
                }
                if future_reference_observation is not None:
                    self.extras["time_out_critic_observation"]["future_reference"] = (
                        future_reference_observation[time_out_ids].clone()
                    )
            else:
                self.extras["time_out_critic_observation"] = None
        else:
            self.extras["terminal_observation"] = None
            self.extras["time_out_critic_observation"] = None
        return terminated, time_out

    def _signed_uniform(
        self,
        count: int,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        magnitude = torch.empty(count, device=self.device).uniform_(
            minimum,
            maximum,
        )
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -1.0,
            1.0,
        )
        return sign * magnitude

    def _sample_commands(self, count: int) -> torch.Tensor:
        """Sample an observable 3-D command curriculum without gait phase."""

        commands = torch.zeros(count, 3, device=self.device)
        selector = torch.rand(count, device=self.device)
        profile = self.cfg.command_profile

        def assign_signed(
            mask: torch.Tensor,
            dimension: int,
            minimum: float,
            maximum: float,
        ) -> None:
            values = self._signed_uniform(count, minimum, maximum)
            commands[mask, dimension] = values[mask]

        def assign_uniform(
            mask: torch.Tensor,
            dimension: int,
            minimum: float,
            maximum: float,
        ) -> None:
            values = torch.empty(count, device=self.device).uniform_(
                minimum,
                maximum,
            )
            commands[mask, dimension] = values[mask]

        if profile == "stand":
            pass
        elif profile == "stage1":
            sagittal = selector >= 0.20
            assign_signed(sagittal, 0, 0.20, 0.55)
        elif profile == "forward_walk":
            commands[:, 0].uniform_(0.25, 0.55)
        elif profile == "walk":
            sagittal = selector >= 0.10
            assign_signed(sagittal, 0, 0.25, 0.65)
        elif profile == "stage2":
            sagittal = torch.logical_and(selector >= 0.10, selector < 0.40)
            lateral = torch.logical_and(selector >= 0.40, selector < 0.55)
            turning = torch.logical_and(selector >= 0.55, selector < 0.70)
            mixed = selector >= 0.70
            assign_signed(sagittal, 0, 0.20, 0.80)
            assign_signed(lateral, 1, 0.10, 0.35)
            assign_signed(turning, 2, 0.15, 0.60)
            assign_uniform(mixed, 0, -0.80, 0.80)
            assign_uniform(mixed, 1, -0.35, 0.35)
            assign_uniform(mixed, 2, -0.60, 0.60)
        elif profile == "stage3":
            sagittal = torch.logical_and(selector >= 0.10, selector < 0.25)
            lateral = torch.logical_and(selector >= 0.25, selector < 0.35)
            turning = torch.logical_and(selector >= 0.35, selector < 0.45)
            mixed = selector >= 0.45
            assign_signed(sagittal, 0, 0.20, 1.20)
            assign_signed(lateral, 1, 0.10, 0.60)
            assign_signed(turning, 2, 0.15, 1.00)
            assign_uniform(mixed, 0, -1.20, 1.20)
            assign_uniform(mixed, 1, -0.60, 0.60)
            assign_uniform(mixed, 2, -1.00, 1.00)
        else:
            commands[:, 0].uniform_(-1.20, 1.20)
            commands[:, 1].uniform_(-0.60, 0.60)
            commands[:, 2].uniform_(-1.00, 1.00)
            commands[selector < 0.10] = 0.0

        commands *= self.cfg.command_scale
        if profile == "full":
            moving = torch.linalg.vector_norm(commands, dim=-1) > 0.0
            normalized_norm = torch.linalg.vector_norm(
                torch.stack(
                    (
                        commands[:, 0] / 1.20,
                        commands[:, 1] / 0.60,
                        commands[:, 2],
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            too_small = torch.logical_and(moving, normalized_norm < 0.20)
            assign_signed(
                too_small,
                0,
                0.24 * self.cfg.command_scale,
                0.40 * self.cfg.command_scale,
            )
        return commands

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        episode_log = {}
        for name, values in self._episode_sums.items():
            episode_log[f"Episode_Reward/{name}"] = torch.mean(values[env_ids])
            values[env_ids] = 0.0
        episode_log["Episode_Termination/fall"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        )
        episode_log["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        )
        self.extras["log"] = episode_log

        self._previous_action[env_ids] = 0.0
        self._action_before_step[env_ids] = 0.0
        self._joint_position_targets[env_ids] = self._robot.data.default_joint_pos[
            env_ids
        ][:, self._joint_ids]
        self._history[env_ids] = 0.0
        self._history_needs_fill[env_ids] = True
        self._observation_delay[env_ids] = 0.0
        self._delay_needs_fill[env_ids] = True
        self._previous_foot_force[env_ids] = 0.0
        self._previous_applied_torque[env_ids] = 0.0
        self._previous_joint_velocity[env_ids] = 0.0
        self._last_landing_foot[env_ids] = -1
        self._steps_since_landing[env_ids] = 0
        self._reference_phase_offset[env_ids] = 0.0
        self._reference_phase[env_ids] = 0.0
        self._last_reset_from_reference[env_ids] = False

        count = len(env_ids)
        if self.cfg.resample_commands_on_reset:
            command_env_ids = env_ids
        else:
            command_env_ids = env_ids[
                torch.logical_not(self._commands_initialized[env_ids])
            ]
        if command_env_ids.numel() > 0:
            random_state = None
            device = torch.device(self.device)
            if not self.cfg.resample_commands_on_reset:
                if device.type == "cuda":
                    random_state = torch.cuda.get_rng_state(device)
                else:
                    random_state = torch.random.get_rng_state()
            self._commands[command_env_ids] = self._sample_commands(
                len(command_env_ids)
            )
            self._commands_initialized[command_env_ids] = True
            if random_state is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(random_state, device)
                else:
                    torch.random.set_rng_state(random_state)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        fallback_root_pose = root_state[:, :7].clone()
        joint_position = self._robot.data.default_joint_pos[env_ids].clone()
        joint_velocity = self._robot.data.default_joint_vel[env_ids].clone()

        v7_initial_state = None
        if self._phase_rsi_enabled:
            if self.cfg.reward_profile in FULL_REFERENCE_REWARD_PROFILES:
                v7_initial_state = (
                    self._walk_reference.sample_full_reference_initial_state(
                        joint_position[:, self._joint_ids],
                        joint_velocity[:, self._joint_ids],
                        self._reference_motion_progress,
                        reference_probability=(
                            self.cfg.reference_state_initialization_probability
                        ),
                    )
                )
                reset_joint_position = v7_initial_state.joint_position
                reset_joint_velocity = v7_initial_state.joint_velocity
                reset_phase = v7_initial_state.phase
                reset_from_reference = v7_initial_state.use_reference
                phase_rate = self._reference_phase_rate()[env_ids]
                reset_joint_velocity = reset_joint_velocity * phase_rate.unsqueeze(-1)
                reference_root_velocity = (
                    v7_initial_state.root_linear_velocity * phase_rate.unsqueeze(-1)
                )
                root_state[reset_from_reference, 2] = (
                    self._terrain.env_origins[env_ids[reset_from_reference], 2]
                    + v7_initial_state.root_height[reset_from_reference]
                )
            else:
                (
                    reset_joint_position,
                    reset_joint_velocity,
                    reset_phase,
                    reset_from_reference,
                ) = self._walk_reference.sample_initial_state(
                    joint_position[:, self._joint_ids],
                    joint_velocity[:, self._joint_ids],
                    reference_probability=(
                        self.cfg.reference_state_initialization_probability
                    ),
                )
            joint_position[:, self._joint_ids] = reset_joint_position
            joint_velocity[:, self._joint_ids] = reset_joint_velocity
            self._reference_phase_offset[env_ids] = reset_phase
            self._reference_phase[env_ids] = reset_phase
            self._last_reset_from_reference[env_ids] = reset_from_reference
            self.extras["rsi_diagnostics"] = {
                "reference_fraction": reset_from_reference.to(torch.float32).mean(),
                "reference_progress": torch.tensor(
                    self._reference_motion_progress,
                    device=self.device,
                ),
            }

        if self.cfg.enable_domain_randomization:
            scale = self.cfg.domain_randomization_scale
            root_state[:, :2] += torch.empty(
                count,
                2,
                device=self.device,
            ).uniform_(-0.10 * scale, 0.10 * scale)
            root_state[:, 2] += torch.empty(
                count,
                device=self.device,
            ).uniform_(-0.02 * scale, 0.02 * scale)
            yaw = torch.empty(count, device=self.device).uniform_(-3.14159, 3.14159)
            root_state[:, 3] = torch.cos(0.5 * yaw)
            root_state[:, 4:6] = 0.0
            root_state[:, 6] = torch.sin(0.5 * yaw)
            root_state[:, 7:] += torch.empty(
                count,
                6,
                device=self.device,
            ).uniform_(-0.10 * scale, 0.10 * scale)
            joint_position += torch.empty_like(joint_position).uniform_(
                -0.05 * scale,
                0.05 * scale,
            )
            joint_velocity += torch.empty_like(joint_velocity).uniform_(
                -0.10 * scale,
                0.10 * scale,
            )

            maximum_delay = max(1, round(2 * scale))
            self._observation_delay_steps[env_ids] = torch.randint(
                low=0,
                high=maximum_delay + 1,
                size=(count,),
                device=self.device,
            )
            self._motor_strength[env_ids] = torch.empty(
                count,
                1,
                device=self.device,
            ).uniform_(1.0 - 0.20 * scale, 1.0 + 0.20 * scale)

            # Base COM and motor-strength ranges are the remaining Appendix
            # Table II quantities. Restore immutable defaults before sampling.
            env_ids_cpu = env_ids.detach().cpu()
            coms = self._robot.root_physx_view.get_coms()
            coms[env_ids_cpu, self._pelvis_body_id] = self._default_coms[
                env_ids_cpu,
                self._pelvis_body_id,
            ]
            coms[env_ids_cpu, self._pelvis_body_id, :3] += torch.empty(
                count,
                3,
                device="cpu",
            ).uniform_(-0.15 * scale, 0.15 * scale)
            self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

            effort_limits = (
                self._default_effort_limits[
                    env_ids_cpu[:, None],
                    self._joint_ids,
                ]
                * self._motor_strength[env_ids].detach().cpu()
            )
            self._robot.write_joint_effort_limit_to_sim(
                effort_limits,
                joint_ids=self._joint_ids,
                env_ids=env_ids,
            )
        else:
            self._observation_delay_steps[env_ids] = 0
            self._motor_strength[env_ids] = 1.0

        if self._phase_rsi_enabled:
            # V6 keeps the safe default root. Full-reference profiles additionally
            # align root height and velocity with the joint/contact reference.
            root_state[:, :7] = sanitize_root_pose(
                root_state[:, :7],
                fallback_root_pose,
                self._terrain.env_origins[env_ids, 2] + 0.55,
            )
            if v7_initial_state is not None and torch.any(
                v7_initial_state.use_reference
            ):
                reference_mask = v7_initial_state.use_reference
                reference_heading = yaw_quat(root_state[reference_mask, 3:7])
                root_state[reference_mask, 7:10] = quat_apply(
                    reference_heading,
                    reference_root_velocity[reference_mask],
                )
            default_joint_position = self._robot.data.default_joint_pos[env_ids][
                :, self._joint_ids
            ]
            reset_joint_position = joint_position[:, self._joint_ids]
            reset_action = (
                (reset_joint_position - default_joint_position) / self._action_scales
            ).clamp(-1.0, 1.0)
            self._joint_position_targets[env_ids] = reset_joint_position
            self._previous_action[env_ids] = reset_action
            self._action_before_step[env_ids] = reset_action
            self._previous_joint_velocity[env_ids] = joint_velocity[:, self._joint_ids]

        # The mass event runs in ``super()._reset_idx``. Read the resulting
        # physical mass for the force and transport normalizers.
        if self.cfg.enable_domain_randomization:
            env_ids_cpu = env_ids.detach().cpu()
            self._robot_mass[env_ids] = torch.sum(
                self._robot.root_physx_view.get_masses()[env_ids_cpu],
                dim=1,
            ).to(self.device)

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_position,
            joint_velocity,
            env_ids=env_ids,
        )


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="g1_env:G1KeyEstimationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "g1_env:G1KeyEstimationEnvCfg",
        },
    )
