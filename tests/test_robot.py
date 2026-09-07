import unittest

import torch

from estnet.robot import (
    ARMATURE29,
    DAMPING29,
    DEFAULT_JOINT_POS29,
    EFFORT_LIMIT29,
    JOINT_NAMES29,
    LEG_JOINT_NAMES12,
    STIFFNESS29,
    VELOCITY_LIMIT29,
    make_joint_targets,
)


class JointTargetContractTest(unittest.TestCase):
    def setUp(self):
        # Scramble native order: the first 12 columns are deliberately not legs.
        self.native_order = torch.arange(29).roll(9)
        self.leg_ids = torch.stack([
            (self.native_order == policy_id).nonzero().squeeze()
            for policy_id in range(12)
        ])
        self.defaults = torch.tensor(DEFAULT_JOINT_POS29)[self.native_order].expand(2, -1).clone()
        self.limits = torch.empty(2, 29, 2)
        self.limits[..., 0] = -3.0
        self.limits[..., 1] = 3.0
        self.actions = torch.zeros(2, 12)

    def test_constants_describe_same_29_joint_contract(self):
        self.assertEqual(len(set(JOINT_NAMES29)), 29)
        self.assertEqual(LEG_JOINT_NAMES12, JOINT_NAMES29[:12])
        for values in (DEFAULT_JOINT_POS29, STIFFNESS29, DAMPING29,
                       EFFORT_LIMIT29, VELOCITY_LIMIT29, ARMATURE29):
            self.assertIsInstance(values, tuple)
            self.assertEqual(len(values), 29)
        # Locomotion variant has 25-Nm ankles, unlike the 50-Nm mimic variant.
        self.assertEqual(EFFORT_LIMIT29[4:6], (25.0, 25.0))
        self.assertEqual(STIFFNESS29[3], 150.0)

    def test_raw_gaussian_knee_action_can_exceed_old_tanh_envelope(self):
        self.actions[:, 3] = 2.0
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        torch.testing.assert_close(targets[:, self.leg_ids[3]], torch.full((2,), 0.8))
        self.assertTrue(torch.all(targets[:, self.leg_ids[3]] > 0.55).item())

    def test_native_mapping_preserves_upper_body_and_inputs(self):
        before_defaults = self.defaults.clone()
        before_limits = self.limits.clone()
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        torch.testing.assert_close(targets, self.defaults)
        self.actions[0] = torch.linspace(-2.0, 2.0, 12)
        before_actions = self.actions.clone()
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        upper_ids = (self.native_order >= 12).nonzero().flatten()
        torch.testing.assert_close(targets[:, upper_ids], self.defaults[:, upper_ids])
        torch.testing.assert_close(targets[0, self.leg_ids], self.defaults[0, self.leg_ids] + 0.25 * self.actions[0])
        torch.testing.assert_close(self.defaults, before_defaults)
        torch.testing.assert_close(self.limits, before_limits)
        torch.testing.assert_close(self.actions, before_actions)

    def test_targets_cap_at_per_environment_soft_limits(self):
        self.actions[0, 3] = 2.0
        self.actions[1, 3] = -2.0
        knee_id = self.leg_ids[3]
        self.limits[0, knee_id, 1] = 0.7
        self.limits[1, knee_id, 0] = 0.1
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        torch.testing.assert_close(targets[:, knee_id], torch.tensor([0.7, 0.1]))

    def test_upper_body_nominal_target_also_obeys_soft_limits(self):
        upper_id = (self.native_order == 18).nonzero().squeeze()
        self.limits[:, upper_id, 1] = 0.9
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        torch.testing.assert_close(targets[:, upper_id], torch.full((2,), 0.9))

    def test_extreme_raw_action_uses_100_clip_not_unit_clip(self):
        self.limits[..., 0] = -100.0
        self.limits[..., 1] = 100.0
        self.actions[0, 3] = 200.0
        self.actions[1, 3] = -200.0
        targets = make_joint_targets(self.actions, self.defaults, self.limits, self.leg_ids)
        torch.testing.assert_close(targets[:, self.leg_ids[3]], torch.tensor([25.3, -24.7]))

    def test_invalid_shapes_and_mapping_are_rejected(self):
        cases = [
            (torch.zeros(2, 29), self.defaults, self.limits, self.leg_ids),
            (self.actions, self.defaults[:1], self.limits, self.leg_ids),
            (self.actions, self.defaults, self.limits[0], self.leg_ids),
            (self.actions, self.defaults, self.limits, self.leg_ids[:11]),
            (self.actions, self.defaults, self.limits, torch.zeros(12, dtype=torch.long)),
            (self.actions, self.defaults, self.limits, torch.arange(18, 30)),
            (self.actions, self.defaults, self.limits, self.leg_ids.float()),
        ]
        for values in cases:
            with self.subTest(shapes=[tuple(v.shape) for v in values]):
                with self.assertRaises(ValueError):
                    make_joint_targets(*values)


if __name__ == "__main__":
    unittest.main()
