"""Observable periodic contact schedule, independent of a simulator.

Wei et al. (Humanoids 2023), Fig. 4, reports 1.5 Hz per leg and
equal stance/swing durations. Their text uses linear phase transitions but
does not publish a transition width. A 0.1-cycle width is our explicit
engineering choice. It is not a recovered parameter of arXiv:2403.05868.
"""

from __future__ import annotations

import math

import torch

GAIT_PERIOD_SECONDS = 2.0 / 3.0
GAIT_DUTY_FACTOR = 0.5
GAIT_TRANSITION_CYCLES = 0.1
FOOT_PHASE_OFFSETS = (0.0, 0.5)


def gait(
    phase: torch.Tensor,
    duty: float = GAIT_DUTY_FACTOR,
    transition: float = GAIT_TRANSITION_CYCLES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right phases and stance weights, each shaped ``[N, 2]``.

    ``phase`` is cycle time ``(elapsed_seconds / period + initial_phase)``;
    callers own its reset and advancement. This function wraps it into [0, 1).
    At the default duty, phase 0 and 0.5 are transition midpoints, phase 0.25
    is full stance, and phase 0.75 is full swing. ``transition`` is the FULL
    width of each linear transition, measured in cycles. Force penalties use
    ``1 - stance``; foot-speed penalties use ``stance``.
    """
    if phase.ndim != 1 or not phase.is_floating_point():
        raise ValueError("phase must be a floating-point tensor shaped [N]")
    if not math.isfinite(duty) or not 0.0 < duty < 1.0:
        raise ValueError("duty must be finite and strictly between 0 and 1")
    if not math.isfinite(transition) or not 0.0 < transition <= min(duty, 1.0 - duty):
        raise ValueError("transition must be positive and <= min(duty, 1-duty)")

    offsets = phase.new_tensor(FOOT_PHASE_OFFSETS)
    foot_phases = torch.remainder(phase[:, None] + offsets, 1.0)
    # Circular distance from the middle of the stance interval [0, duty].
    distance = torch.abs(torch.remainder(foot_phases - duty / 2.0 + 0.5, 1.0) - 0.5)
    stance = ((duty / 2.0 + transition / 2.0 - distance) / transition).clamp(0.0, 1.0)
    return foot_phases, stance


def command_with_phase(
    physical_command: torch.Tensor, foot_phases: torch.Tensor
) -> torch.Tensor:
    """Encode ``[vx, vy, wz, sinL, cosL, sinR, cosR]`` for actor and critic.

    This explicit seven-value interface is our implementation convention;
    the 2024 paper does not give the element-wise gait-command encoding.
    Sine AND cosine prevent the two halves of a cycle from aliasing.
    """
    if physical_command.ndim != 2 or physical_command.shape[1] != 3:
        raise ValueError("physical_command must have shape [N, 3]")
    if foot_phases.shape != (physical_command.shape[0], 2):
        raise ValueError("foot_phases must have shape [N, 2]")
    angles = 2.0 * math.pi * foot_phases
    clocks = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(1)
    return torch.cat((physical_command, clocks), dim=-1)
