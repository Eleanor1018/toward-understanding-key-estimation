import unittest
from pathlib import Path

import numpy as np
import torch

from motion_reference import (
    G1_FOOT_SOLE_OFFSET_M,
    G1_WALK_ARCHIVE_SHA256,
    G1_WALK_SOURCE_SHA256,
    CyclicJointReference,
    g1_leg_forward_kinematics,
    g1_leg_forward_velocity,
)
from normalization import DEFAULT_JOINT_POSITIONS


class G1LegForwardKinematicsTest(unittest.TestCase):
    def test_matches_fixed_unitree_model_values(self) -> None:
        # Fixtures were generated independently with MuJoCo 3.12 from Unitree's
        # g1_29dof_rev_1_0.xml, then expressed in the pelvis frame.
        configurations = torch.tensor(
            [
                [0.0] * 12,
                [
                    0.2,
                    -0.1,
                    0.15,
                    0.4,
                    -0.25,
                    0.05,
                    -0.3,
                    0.12,
                    -0.2,
                    0.55,
                    0.1,
                    -0.08,
                ],
                [
                    0.3,
                    0.2,
                    -0.4,
                    0.8,
                    -0.25,
                    0.1,
                    -0.2,
                    -0.3,
                    0.35,
                    1.0,
                    -0.4,
                    -0.1,
                ],
            ],
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [
                [
                    [-0.0000023283064, 0.1185064539, -0.7568637133],
                    [-0.0000023283064, -0.1185064539, -0.7568637133],
                ],
                [
                    [-0.239667827962, 0.024851915516, -0.690422793158],
                    [0.019190596485, 0.003166753997, -0.719620380644],
                ],
                [
                    [-0.348238468, 0.343343228, -0.539476871],
                    [-0.154669553, -0.365622044, -0.596334100],
                ],
            ],
            dtype=torch.float64,
        )

        actual = g1_leg_forward_kinematics(configurations)
        torch.testing.assert_close(actual, expected, atol=2.0e-7, rtol=0.0)

    def test_velocity_is_fk_time_derivative(self) -> None:
        position = torch.tensor(
            [
                0.2,
                -0.1,
                0.15,
                0.4,
                -0.25,
                0.05,
                -0.3,
                0.12,
                -0.2,
                0.55,
                0.1,
                -0.08,
            ],
            dtype=torch.float64,
        )
        velocity = torch.tensor(
            [
                0.3,
                -0.2,
                0.1,
                -0.4,
                0.25,
                0.8,
                -0.1,
                0.5,
                -0.3,
                0.2,
                0.4,
                -0.7,
            ],
            dtype=torch.float64,
        )
        epsilon = 1.0e-6
        finite_difference = (
            g1_leg_forward_kinematics(position + epsilon * velocity)
            - g1_leg_forward_kinematics(position - epsilon * velocity)
        ) / (2.0 * epsilon)

        torch.testing.assert_close(
            g1_leg_forward_velocity(position, velocity),
            finite_difference,
            atol=1.0e-9,
            rtol=1.0e-8,
        )

    def test_ankle_roll_does_not_move_its_own_origin(self) -> None:
        position = torch.zeros(4, 12)
        changed = position.clone()
        changed[:, 5] = torch.linspace(-0.25, 0.25, 4)
        changed[:, 11] = torch.linspace(0.25, -0.25, 4)
        torch.testing.assert_close(
            g1_leg_forward_kinematics(position),
            g1_leg_forward_kinematics(changed),
        )


class FullG1ReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motions"
            / "g1_walk_mimickit.npz"
        )
        with np.load(cls.reference_path, allow_pickle=False) as data:
            cls.joint_names = tuple(str(name) for name in data["joint_names"].tolist())
            cls.expected_root_position = torch.from_numpy(data["root_pos"].copy())
            cls.expected_root_rotation = torch.from_numpy(
                data["root_rot_exp_map"].copy()
            )
        cls.center = torch.tensor(DEFAULT_JOINT_POSITIONS)
        cls.reference = CyclicJointReference(
            cls.reference_path,
            cls.joint_names,
            "cpu",
            expected_archive_sha256=G1_WALK_ARCHIVE_SHA256,
            expected_source_sha256=G1_WALK_SOURCE_SHA256,
        )
        cls.reference.retarget_to_position_envelope(cls.center, max_offset=0.22)

    def test_loader_retains_exact_source_root_tracks(self) -> None:
        torch.testing.assert_close(
            self.reference.source_root_positions,
            self.expected_root_position,
        )
        torch.testing.assert_close(
            self.reference.source_root_rotation_exp_map,
            self.expected_root_rotation,
        )

    def test_schedule_preserves_v6_and_reaches_full_leg_motion(self) -> None:
        phase = torch.tensor([0.0, 0.17, 0.5, 0.83])
        compressed_position, compressed_velocity, _ = self.reference.sample_phase(phase)
        start = self.reference.sample_full_reference(phase, progress=0.0)
        middle = self.reference.sample_full_reference(phase, progress=0.5)
        full = self.reference.sample_full_reference(phase, progress=1.0)

        torch.testing.assert_close(start.joint_position, compressed_position)
        torch.testing.assert_close(start.joint_velocity, compressed_velocity)
        torch.testing.assert_close(
            start.action_scale,
            torch.full_like(start.action_scale, 0.25),
        )
        assert self.reference.full_joint_positions is not None
        assert self.reference.compressed_joint_positions is not None
        expected_full_position = self.reference._interpolate_frames(
            self.reference.full_joint_positions,
            *self.reference._frame_coordinates(phase)[1:],
        )
        torch.testing.assert_close(full.joint_position, expected_full_position)
        torch.testing.assert_close(
            full.joint_position[:, 12:],
            compressed_position[:, 12:],
        )
        torch.testing.assert_close(
            middle.joint_position,
            0.5 * (start.joint_position + full.joint_position),
        )
        torch.testing.assert_close(
            middle.joint_velocity,
            0.5 * (start.joint_velocity + full.joint_velocity),
        )

        assert self.reference.full_action_scales is not None
        expected_leg_scales = torch.maximum(
            torch.full((12,), 0.25),
            1.05
            * torch.max(
                torch.abs(
                    self.reference.source_joint_positions[:, :12] - self.center[:12]
                ),
                dim=0,
            ).values,
        )
        torch.testing.assert_close(
            self.reference.full_action_scales[:12],
            expected_leg_scales,
        )
        torch.testing.assert_close(
            self.reference.full_action_scales[12:],
            torch.full((17,), 0.25),
        )
        self.assertTrue(
            torch.all(
                torch.abs(full.joint_position - self.center) <= full.action_scale + 1e-6
            ).item()
        )

    def test_contact_and_root_targets_are_kinematically_consistent(self) -> None:
        phase = torch.arange(self.reference.frame_count - 1) / (
            self.reference.frame_count - 1
        )
        sample = self.reference.sample_full_reference(phase, progress=1.0)

        self.assertEqual(sample.contact_state.dtype, torch.bool)
        self.assertTrue(torch.all(sample.contact_state.any(dim=-1)).item())
        self.assertTrue(torch.all(sample.contact_target >= 0.0).item())
        self.assertTrue(torch.all(sample.contact_target <= 1.0).item())
        self.assertTrue(
            torch.any(
                torch.logical_and(
                    sample.contact_target > 0.0,
                    sample.contact_target < 1.0,
                )
            ).item()
        )
        double_support = sample.contact_state.all(dim=-1).to(torch.float32).mean()
        self.assertGreater(double_support.item(), 0.0)
        self.assertLess(double_support.item(), 0.15)
        self.assertTrue(
            torch.any(sample.contact_state[:, 0] & ~sample.contact_state[:, 1])
        )
        self.assertTrue(
            torch.any(sample.contact_state[:, 1] & ~sample.contact_state[:, 0])
        )

        ankle_world_height = sample.ankle_position_pelvis[
            ..., 2
        ] + sample.root_height.unsqueeze(-1)
        torch.testing.assert_close(
            ankle_world_height.amin(dim=-1),
            torch.full_like(sample.root_height, G1_FOOT_SOLE_OFFSET_M),
        )
        contact_weight = sample.contact_target / sample.contact_target.sum(
            dim=-1,
            keepdim=True,
        )
        expected_root_velocity = -torch.sum(
            contact_weight.unsqueeze(-1) * sample.ankle_velocity_pelvis,
            dim=-2,
        )
        torch.testing.assert_close(sample.root_linear_velocity, expected_root_velocity)

    def test_root_vertical_velocity_matches_height_derivative_in_single_support(
        self,
    ) -> None:
        phase = torch.tensor([0.2])
        epsilon = 1.0e-3
        center = self.reference.sample_full_reference(phase, progress=1.0)
        before = self.reference.sample_full_reference(phase - epsilon, progress=1.0)
        after = self.reference.sample_full_reference(phase + epsilon, progress=1.0)
        finite_difference = (after.root_height - before.root_height) / (
            2.0 * epsilon * self.reference.duration_s
        )
        self.assertEqual(int(center.contact_state.sum().item()), 1)
        torch.testing.assert_close(
            center.root_linear_velocity[:, 2],
            finite_difference,
            atol=5.0e-4,
            rtol=0.0,
        )

    def test_natural_speed_scales_with_full_leg_schedule(self) -> None:
        compressed_speed = self.reference.natural_forward_speed(0.0)
        full_speed = self.reference.natural_forward_speed(1.0)
        self.assertGreater(compressed_speed.item(), 0.0)
        self.assertAlmostEqual(full_speed.item(), 0.8343599, places=5)
        self.assertGreater(full_speed.item(), compressed_speed.item())

    def test_contact_schedule_is_full_motion_at_every_progress(self) -> None:
        phase = torch.arange(self.reference.frame_count - 1) / (
            self.reference.frame_count - 1
        )
        start = self.reference.sample_full_reference(phase, progress=0.0)
        middle = self.reference.sample_full_reference(phase, progress=0.25)
        full = self.reference.sample_full_reference(phase, progress=1.0)
        torch.testing.assert_close(start.contact_target, full.contact_target)
        torch.testing.assert_close(middle.contact_target, full.contact_target)
        self.assertLess(full.contact_state.all(dim=-1).float().mean().item(), 0.15)

    def test_full_reference_rsi_returns_complete_state(self) -> None:
        count = 32
        default_position = self.center.expand(count, -1).clone()
        default_velocity = torch.zeros_like(default_position)
        reference_state = self.reference.sample_full_reference_initial_state(
            default_position,
            default_velocity,
            progress=0.75,
            reference_probability=1.0,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.all(reference_state.use_reference).item())
        expected = self.reference.sample_full_reference(
            reference_state.phase,
            progress=0.75,
        )
        torch.testing.assert_close(
            reference_state.joint_position, expected.joint_position
        )
        torch.testing.assert_close(
            reference_state.joint_velocity, expected.joint_velocity
        )
        torch.testing.assert_close(reference_state.root_height, expected.root_height)
        torch.testing.assert_close(
            reference_state.root_linear_velocity,
            expected.root_linear_velocity,
        )

        standing = self.reference.sample_full_reference_initial_state(
            default_position,
            default_velocity,
            progress=0.75,
            reference_probability=0.0,
        )
        self.assertFalse(torch.any(standing.use_reference).item())
        torch.testing.assert_close(standing.joint_position, default_position)
        torch.testing.assert_close(standing.joint_velocity, default_velocity)
        torch.testing.assert_close(standing.phase, torch.zeros(count))
        torch.testing.assert_close(
            standing.root_rotation_exp_map,
            torch.zeros(count, 3),
        )
        torch.testing.assert_close(
            standing.root_linear_velocity,
            torch.zeros(count, 3),
            atol=1.0e-7,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
