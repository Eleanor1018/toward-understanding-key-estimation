"""Numerical contracts, not evidence that a simulated robot can walk."""

import math
import unittest

import torch

from estnet.gait import command_with_phase, gait
from estnet.rewards import (
    PAPER_PRINTED_PARAMETERS,
    RewardParameters,
    cauchy,
    gaussian,
    reward_terms,
)


def example_state(n: int = 1) -> dict[str, torch.Tensor]:
    """Idealized instantaneous physical values; not a dynamics trajectory."""
    foot_phases, stance = gait(torch.full((n,), 0.25))
    command = command_with_phase(
        torch.tensor([[0.4, 0.0, 0.0]]).repeat(n, 1), foot_phases
    )
    force = torch.zeros(n, 2, 3)
    force[:, 0, 2] = 40.0 * 9.81
    return {
        "vel": torch.tensor([[0.4, 0.0, 0.0]]).repeat(n, 1),
        "ang_vel": torch.zeros(n, 3),
        "command": command,
        "up": torch.ones(n),
        "height": torch.full((n,), RewardParameters().reward_target_height),
        "foot_vel": torch.zeros(n, 2, 3),
        "foot_force": force,
        "prev_foot_force": force.clone(),
        "torque": torch.zeros(n, 12),
        "prev_torque": torch.zeros(n, 12),
        "joint_vel": torch.zeros(n, 12),
        "prev_joint_vel": torch.zeros(n, 12),
        "mass": torch.full((n,), 40.0),
        "stance": stance,
        "terminated": torch.zeros(n, dtype=torch.bool),
    }


class GaitTest(unittest.TestCase):
    def test_boundaries_opposite_feet_and_wrap(self):
        phase = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, -0.25], dtype=torch.float64)
        foot_phase, stance = gait(phase)
        expected = torch.tensor(
            [[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(stance, expected)
        torch.testing.assert_close(foot_phase[:, 1], (foot_phase[:, 0] + 0.5) % 1)
        torch.testing.assert_close(stance.sum(-1), torch.ones(6, dtype=torch.float64))

    def test_linear_ramps_are_continuous_across_cycle_boundary(self):
        _, stance = gait(
            torch.tensor([0.95, 0.975, 0.0, 0.025, 0.05], dtype=torch.float64)
        )
        torch.testing.assert_close(
            stance[:, 0], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        )
        _, falling = gait(
            torch.tensor([0.45, 0.475, 0.5, 0.525, 0.55], dtype=torch.float64)
        )
        torch.testing.assert_close(
            falling[:, 0],
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64),
        )

    def test_clock_features_have_exact_order_and_disambiguate_phase(self):
        physical = torch.tensor(
            [[0.3, 0.1, -0.2], [0.3, 0.1, -0.2]], dtype=torch.float64
        )
        foot_phase, _ = gait(torch.tensor([0.125, 0.375], dtype=torch.float64))
        command = command_with_phase(physical, foot_phase)
        self.assertEqual(command.shape, (2, 7))
        torch.testing.assert_close(command[:, :3], physical)
        torch.testing.assert_close(
            command[:, 3], torch.sin(2 * math.pi * foot_phase[:, 0])
        )
        torch.testing.assert_close(
            command[:, 4], torch.cos(2 * math.pi * foot_phase[:, 0])
        )
        torch.testing.assert_close(
            command[:, 5], torch.sin(2 * math.pi * foot_phase[:, 1])
        )
        torch.testing.assert_close(
            command[:, 6], torch.cos(2 * math.pi * foot_phase[:, 1])
        )
        self.assertAlmostEqual(command[0, 3].item(), command[1, 3].item())
        self.assertNotAlmostEqual(command[0, 4].item(), command[1, 4].item())

    def test_reject_invalid_gait_parameters(self):
        for duty, width in [
            (0.0, 0.1),
            (1.0, 0.1),
            (0.5, 0.0),
            (0.1, 0.2),
            (0.5, math.nan),
        ]:
            with self.assertRaises(ValueError):
                gait(torch.zeros(1), duty, width)


class RewardTest(unittest.TestCase):
    def test_kernels_peak_decay_and_width_convention(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
        actual = gaussian(x, 0.1, 0.5)
        expected = torch.tensor(
            [0.1, 0.1 / math.e, 0.1 * math.exp(-4), 0.1 * math.exp(-16)],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected)
        for beta in (1, 2, 3):
            vals = cauchy(x, 0.1, beta, 0.5)
            self.assertAlmostEqual(vals[0].item(), 0.1)
            self.assertAlmostEqual(vals[1].item(), 0.05)
            self.assertTrue(torch.all(vals[:-1] > vals[1:]))
            torch.testing.assert_close(cauchy(-x, 0.1, beta, 0.5), vals)

    def test_static_reward_counterexamples_do_not_substitute_for_walking_test(self):
        state = example_state(3)
        # Rows: static double-support standing; double-support shuffling at
        # commanded speed; idealized mid-swing with stationary left support.
        state["vel"][0] = 0.0
        state["foot_force"][:2, :, 2] = 20.0 * 9.81
        state["prev_foot_force"] = state["foot_force"].clone()
        state["foot_vel"][1, :, 0] = 0.4
        state["foot_vel"][2, 1, 0] = 0.8
        terms = reward_terms(state, RewardParameters())
        total = sum(terms.values())
        self.assertEqual(len(terms), 11)
        self.assertAlmostEqual(terms["swing_force"][0].item(), 0.1 / 26.0, places=6)
        self.assertAlmostEqual(
            terms["stance_velocity"][1].item(), 0.1 / (1 + (0.4 / 0.25) ** 2), places=6
        )
        self.assertAlmostEqual(
            total[0].item(),
            0.9 + 0.1 * math.exp(-((0.4 / 0.5) ** 2)) + 0.1 / 26.0,
            places=6,
        )
        self.assertAlmostEqual(
            total[1].item(), 0.9 + 0.1 / 26.0 + 0.1 / (1 + (0.4 / 0.25) ** 2), places=6
        )
        self.assertAlmostEqual(total[2].item(), 1.1, places=6)
        # This deliberately records an unresolved tradeoff: modest shuffling
        # can be worse than safe standing; optimal kernels alone are no proof
        # that PPO will discover the better single-support behavior.
        self.assertGreater(total[2].item(), total[0].item())
        self.assertGreater(total[2].item(), total[1].item())
        self.assertGreater(total[0].item(), total[1].item())

    def test_wrong_foot_phase_is_not_equivalent_to_correct_support(self):
        state = example_state(2)
        state["stance"][1] = state["stance"][1].flip(0)
        terms = reward_terms(state, RewardParameters())
        self.assertAlmostEqual(terms["swing_force"][0].item(), 0.1, places=6)
        self.assertLess(terms["swing_force"][1].item(), 0.001)

    def test_velocity_tracking_includes_uncommanded_vertical_and_roll_rates(self):
        state = example_state(2)
        state["vel"][1, 2] = 0.5
        state["ang_vel"][1, 0] = 0.5
        terms = reward_terms(state, RewardParameters())
        for name in ("linear_velocity", "yaw_velocity"):
            self.assertAlmostEqual(terms[name][0].item(), 0.1, places=6)
            self.assertAlmostEqual(terms[name][1].item(), 0.1 / math.e, places=6)

    def test_zero_speed_is_finite_and_power_does_not_cancel_between_motors(self):
        state = example_state(2)
        state["vel"][:] = 0.0
        state["torque"][1, :2] = torch.tensor([100.0, -100.0])
        state["joint_vel"][1, :2] = 1.0
        terms = reward_terms(state, RewardParameters())
        self.assertTrue(all(torch.isfinite(value).all() for value in terms.values()))
        self.assertAlmostEqual(terms["cost_of_transport"][0].item(), 0.1, places=6)
        expected_cot = 200.0 / (40.0 * 9.81 * 0.1)
        expected = 0.1 / (1 + (expected_cot / 1.6) ** 6)
        self.assertAlmostEqual(terms["cost_of_transport"][1].item(), expected, places=6)

    def test_body_weight_normalization_and_termination_are_explicit(self):
        state = example_state(2)
        state["foot_force"][:, :, 2] = 20.0 * 9.81
        state["mass"][1] *= 2
        state["foot_force"][1] *= 2
        state["prev_foot_force"] = state["foot_force"].clone()
        state["terminated"][1] = True
        terms = reward_terms(state, RewardParameters())
        torch.testing.assert_close(terms["swing_force"][0], terms["swing_force"][1])
        torch.testing.assert_close(terms["termination"], torch.tensor([0.0, -1.0]))

    def test_printed_paper_values_cannot_be_mistaken_for_dense_defaults(self):
        cfg = RewardParameters()
        self.assertEqual(PAPER_PRINTED_PARAMETERS["linear_velocity"]["sigma"], 0.02)
        self.assertEqual(PAPER_PRINTED_PARAMETERS["swing_force"]["sigma"], 8.0)
        self.assertEqual(cfg.reward_linear_sigma, 0.5)
        self.assertEqual(cfg.reward_swing_force_sigma, 0.1)
        with self.assertRaises(ValueError):
            RewardParameters(reward_speed_floor=0.0)


if __name__ == "__main__":
    unittest.main()
