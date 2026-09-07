"""Ten paper-inspired reward terms with explicit G1 engineering parameters.

The functional families and ten groups follow arXiv:2403.05868v1, Eqs. 3-12.
These defaults are NOT the paper's printed numerical configuration: tracking
widths are broadened, swing force is normalized by body weight, uprightness
uses 1-R22, and mechanical power
uses the sum of absolute per-joint powers. The paper's ambiguous parameters
are retained below only for documentation, never selected silently at runtime.

Each output is a per-policy-step reward. The caller sums these values without
an additional dt factor. No reference motion, foot-height trajectory, latent
encoding, progress bonus, or action-magnitude barrier is part of this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

GRAVITY_M_S2 = 9.81
TERM_WEIGHT = 0.1
HEIGHT_WEIGHT = 0.2

# Printed values are evidence, not a recommended/reproducible configuration.
# Eq. 1 omits the Gaussian minus sign; Eq. 5 and its prose disagree on R22^2;
# Eqs. 7-8 do not specify force/speed normalization; Eqs. 11-12 omit zero-speed
# handling. The gait's original Q construction is delegated to reference [38].
PAPER_PRINTED_PARAMETERS = {
    "linear_velocity": {"family": "Gaussian", "alpha": 0.1, "sigma": 0.02},
    "angular_velocity": {"family": "Gaussian", "alpha": 0.1, "sigma": 0.02},
    "upright": {"family": "Gaussian", "alpha": 0.1, "sigma": 0.0025},
    "height": {"family": "Gaussian", "alpha": 0.2, "sigma": 0.02},
    "stance_velocity": {"family": "Cauchy", "alpha": 0.1, "beta": 1, "sigma": 8.0},
    "swing_force": {"family": "Cauchy", "alpha": 0.1, "beta": 1, "sigma": 8.0},
    "impact": {"family": "Cauchy", "alpha": 0.1, "beta": 3, "sigma": 0.2},
    "torque_smoothness": {"family": "Cauchy", "alpha": 0.1, "beta": 2, "sigma": 160.0},
    "joint_velocity_smoothness": {
        "family": "Cauchy",
        "alpha": 0.1,
        "beta": 1,
        "sigma": 8.0,
    },
    "cost_of_transport": {"family": "Cauchy", "alpha": 0.1, "beta": 3, "sigma": 1.6},
}


@dataclass(frozen=True)
class RewardParameters:
    """Dense G1 defaults; dimensions belong to kernel INPUTS, not variances."""

    reward_linear_sigma: float = 0.5  # m/s, three-dimensional velocity error
    reward_yaw_sigma: float = (
        0.5  # rad/s, full angular-velocity error; desired roll/pitch rates are zero
    )
    reward_upright_sigma: float = 0.1  # dimensionless 1-R22
    reward_height_sigma: float = 0.05  # m
    reward_target_height: float = 0.78  # m; provisional official G1 locomotion target
    reward_stance_velocity_sigma: float = 0.25  # m/s
    reward_swing_force_sigma: float = 0.1  # force divided by total body weight
    reward_impact_sigma: float = 0.2  # force increment divided by body weight
    reward_torque_delta_sigma: float = 160.0  # N m
    reward_joint_velocity_delta_sigma: float = 8.0
    reward_cot_sigma: float = 1.6  # dimensionless
    reward_speed_floor: float = 0.1  # m/s, denominator regularization
    reward_termination_weight: float = 1.0  # engineering addition, not a paper term

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "reward_termination_weight":
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive")


def gaussian(x: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """Bell-shaped reward alpha*exp(-(x/sigma)^2), with an explicit width."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    return alpha * torch.exp(-torch.square(x / sigma))


def cauchy(x: torch.Tensor, alpha: float, beta: int, sigma: float) -> torch.Tensor:
    """Generalized Cauchy reward; its half-height occurs at abs(x)==sigma."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not isinstance(beta, int) or beta < 1:
        raise ValueError("beta must be a positive integer")
    return alpha / (1.0 + torch.pow(torch.abs(x) / sigma, 2 * beta))


def reward_terms(
    state: Mapping[str, torch.Tensor], cfg: RewardParameters
) -> dict[str, torch.Tensor]:
    """Compute ten paper-family terms plus a separately identified terminal cost.

    Required shapes: vel/ang_vel [N,3], command [N,7], up/height/mass/
    terminated [N], foot_vel/foot_force/prev_foot_force [N,2,3], stance [N,2],
    torque/prev_torque/joint_vel/prev_joint_vel [N,12]. Velocities, commands,
    and up must follow the caller's consistent base-frame convention. Foot
    velocities and forces must be world-frame vectors. ``up`` is R22.
    ``terminated`` excludes normal time-limit truncation.
    """
    vel = state["vel"]
    if vel.ndim != 2 or vel.shape[1] != 3:
        raise ValueError("vel must have shape [N, 3]")
    n = vel.shape[0]
    shapes = {
        "ang_vel": (n, 3),
        "command": (n, 7),
        "up": (n,),
        "height": (n,),
        "mass": (n,),
        "terminated": (n,),
        "stance": (n, 2),
        "foot_vel": (n, 2, 3),
        "foot_force": (n, 2, 3),
        "prev_foot_force": (n, 2, 3),
        "torque": (n, 12),
        "prev_torque": (n, 12),
        "joint_vel": (n, 12),
        "prev_joint_vel": (n, 12),
    }
    for name, shape in shapes.items():
        if state[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    commanded_vel = torch.cat(
        (state["command"][:, :2], torch.zeros_like(vel[:, :1])), dim=-1
    )
    commanded_ang_vel = torch.cat(
        (torch.zeros_like(vel[:, :2]), state["command"][:, 2:3]), dim=-1
    )
    speed = torch.linalg.vector_norm(vel, dim=-1).clamp_min(cfg.reward_speed_floor)
    body_weight = (state["mass"] * GRAVITY_M_S2).clamp_min(1.0e-6)
    stance = state["stance"]
    swing = 1.0 - stance
    foot_speed = torch.linalg.vector_norm(state["foot_vel"], dim=-1)
    foot_force = torch.linalg.vector_norm(state["foot_force"], dim=-1)
    force_change = (state["foot_force"] - state["prev_foot_force"]).flatten(1)
    torque_change = state["torque"] - state["prev_torque"]
    joint_velocity_change = state["joint_vel"] - state["prev_joint_vel"]
    # Sum absolute joint powers BEFORE summing: regenerative/braking work must
    # not cancel positive work of another motor and make a moving robot free.
    mechanical_power = torch.sum(
        torch.abs(state["torque"] * state["joint_vel"]), dim=-1
    )

    return {
        "linear_velocity": gaussian(
            torch.linalg.vector_norm(commanded_vel - vel, dim=-1),
            TERM_WEIGHT,
            cfg.reward_linear_sigma,
        ),
        "yaw_velocity": gaussian(
            torch.linalg.vector_norm(commanded_ang_vel - state["ang_vel"], dim=-1),
            TERM_WEIGHT,
            cfg.reward_yaw_sigma,
        ),
        "upright": gaussian(1.0 - state["up"], TERM_WEIGHT, cfg.reward_upright_sigma),
        "height": gaussian(
            cfg.reward_target_height - state["height"],
            HEIGHT_WEIGHT,
            cfg.reward_height_sigma,
        ),
        "stance_velocity": cauchy(
            torch.sum(stance * foot_speed, dim=-1),
            TERM_WEIGHT,
            1,
            cfg.reward_stance_velocity_sigma,
        ),
        "swing_force": cauchy(
            torch.sum(swing * foot_force, dim=-1) / body_weight,
            TERM_WEIGHT,
            1,
            cfg.reward_swing_force_sigma,
        ),
        "impact": cauchy(
            torch.linalg.vector_norm(force_change, dim=-1) / body_weight,
            TERM_WEIGHT,
            3,
            cfg.reward_impact_sigma,
        ),
        "torque_smoothness": cauchy(
            torch.linalg.vector_norm(torque_change, dim=-1),
            TERM_WEIGHT,
            2,
            cfg.reward_torque_delta_sigma,
        ),
        "joint_velocity_smoothness": cauchy(
            torch.linalg.vector_norm(joint_velocity_change, dim=-1) / speed,
            TERM_WEIGHT,
            1,
            cfg.reward_joint_velocity_delta_sigma,
        ),
        "cost_of_transport": cauchy(
            mechanical_power / (body_weight * speed),
            TERM_WEIGHT,
            3,
            cfg.reward_cot_sigma,
        ),
        "termination": -cfg.reward_termination_weight
        * state["terminated"].to(vel.dtype),
    }
