import unittest
from pathlib import Path

import numpy as np
import torch

from motion_reference import (
    G1_WALK_ARCHIVE_SHA256,
    G1_WALK_SOURCE_SHA256,
    CyclicJointReference,
    phase_visible_command,
    sanitize_root_pose,
    split_imitation_reward_weight,
)


class PhaseVisibleReferenceStateInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reference_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motions"
            / "g1_walk_mimickit.npz"
        )
        with np.load(reference_path, allow_pickle=False) as data:
            joint_names = tuple(str(name) for name in data["joint_names"].tolist())
        cls.reference = CyclicJointReference(
            reference_path,
            joint_names,
            "cpu",
            expected_archive_sha256=G1_WALK_ARCHIVE_SHA256,
            expected_source_sha256=G1_WALK_SOURCE_SHA256,
        )

    def test_phase_command_preserves_physical_values_and_wraps(self) -> None:
        physical = torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [-0.3, 0.5, -0.7],
                [0.0, 0.0, 0.0],
            ]
        )
        command = phase_visible_command(
            physical,
            torch.tensor([0.0, 0.25, 1.25]),
        )

        self.assertEqual(command.shape, (3, 5))
        torch.testing.assert_close(command[:, :3], physical)
        torch.testing.assert_close(
            command[:, 3:],
            torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_phase_offset_advances_and_wraps_with_reference_period(self) -> None:
        duration = self.reference.duration_s
        elapsed = torch.tensor([0.0, 0.2 * duration, 1.2 * duration])
        offset = torch.tensor([0.9, 0.9, 0.9])
        phase = self.reference.phase_at_time(elapsed, offset)
        torch.testing.assert_close(
            phase,
            torch.tensor([0.9, 0.1, 0.1]),
            atol=1.0e-6,
            rtol=0.0,
        )

        first = self.reference.sample_phase(torch.tensor([-0.1]))
        wrapped = self.reference.sample_phase(torch.tensor([0.9]))
        torch.testing.assert_close(first[0], wrapped[0])
        torch.testing.assert_close(first[1], wrapped[1])
        torch.testing.assert_close(first[2], wrapped[2])

    def test_rsi_mixes_about_seventy_percent_reference_states(self) -> None:
        count = 10_000
        default_position = torch.zeros(count, len(self.reference.joint_names))
        default_velocity = torch.zeros_like(default_position)
        generator = torch.Generator().manual_seed(1234)

        position, velocity, phase, use_reference = self.reference.sample_initial_state(
            default_position,
            default_velocity,
            reference_probability=0.70,
            generator=generator,
        )

        fraction = use_reference.to(torch.float32).mean().item()
        self.assertGreater(fraction, 0.68)
        self.assertLess(fraction, 0.72)
        torch.testing.assert_close(
            position[~use_reference],
            default_position[~use_reference],
        )
        torch.testing.assert_close(
            velocity[~use_reference],
            default_velocity[~use_reference],
        )
        torch.testing.assert_close(
            phase[~use_reference],
            torch.zeros_like(phase[~use_reference]),
        )
        sampled_position, sampled_velocity, _ = self.reference.sample_phase(
            phase[use_reference]
        )
        torch.testing.assert_close(position[use_reference], sampled_position)
        torch.testing.assert_close(velocity[use_reference], sampled_velocity)

    def test_rsi_probability_zero_is_standing_only(self) -> None:
        default_position = torch.full((8, len(self.reference.joint_names)), 0.25)
        default_velocity = torch.full_like(default_position, -0.5)
        position, velocity, phase, use_reference = self.reference.sample_initial_state(
            default_position,
            default_velocity,
            reference_probability=0.0,
        )

        self.assertFalse(torch.any(use_reference).item())
        torch.testing.assert_close(position, default_position)
        torch.testing.assert_close(velocity, default_velocity)
        torch.testing.assert_close(phase, torch.zeros(8))

    def test_root_pose_guard_rejects_bad_height_and_quaternion(self) -> None:
        fallback = torch.tensor(
            [
                [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 0.9, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        candidate = torch.tensor(
            [
                [0.1, 0.2, 0.7, 2.0, 0.0, 0.0, 0.0],
                [float("nan"), 4.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        safe = sanitize_root_pose(candidate, fallback, minimum_height=0.55)

        self.assertTrue(torch.isfinite(safe).all().item())
        torch.testing.assert_close(safe[:, 2], torch.tensor([0.7, 0.9]))
        torch.testing.assert_close(safe[1, :2], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(
            torch.linalg.vector_norm(safe[:, 3:7], dim=-1),
            torch.ones(2),
        )

    def test_imitation_budget_uses_five_to_one_split(self) -> None:
        pose_weight, velocity_weight = split_imitation_reward_weight(0.03)
        self.assertAlmostEqual(pose_weight, 0.025)
        self.assertAlmostEqual(velocity_weight, 0.005)
        self.assertAlmostEqual(pose_weight + velocity_weight, 0.03)


if __name__ == "__main__":
    unittest.main()
