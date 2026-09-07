import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from checkpoint_io import load_checkpoint
from config import model_config_for_reward_profile
from gait import classify_foot_landings
from motion_reference import (
    G1_DEEPMIMIC_DOF_WEIGHTS,
    G1_WALK_ARCHIVE_SHA256,
    G1_WALK_SOURCE_SHA256,
    REFERENCE_POSE_WEIGHT,
    REFERENCE_VELOCITY_WEIGHT,
    CyclicJointReference,
    deepmimic_joint_similarity,
    gated_imitation_rewards,
)
from normalization import (
    DEFAULT_JOINT_POSITIONS,
    JOINT_EFFORT_LIMITS,
    denormalize_explicit_velocity,
    normalize_command,
    normalize_explicit_velocity,
    normalize_obs,
    normalize_privileged,
)
from ppo import DiagonalGaussian, generalized_advantage_estimation
from train import (
    linear_imitation_weight,
    policy_learning_rate,
    reference_motion_progress,
    reduce_policy_learning_rate,
    restore_policy_update,
    snapshot_policy_update,
    value_target_statistics,
)
from train_ddp import (
    clear_optimizer_group_state,
    clear_optimizer_state,
    prepare_reference_motion_schedule,
)


class GeneralizedAdvantageEstimationTest(unittest.TestCase):
    def test_nonterminal_bootstrap(self) -> None:
        advantages, returns = generalized_advantage_estimation(
            rewards=torch.tensor([[1.0], [2.0]]),
            values=torch.tensor([[10.0], [20.0]]),
            terminated=torch.tensor([[False], [False]]),
            truncated=torch.tensor([[False], [False]]),
            last_value=torch.tensor([30.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(
            advantages[:, 0],
            torch.tensor([23.0, 12.0]),
        )
        torch.testing.assert_close(returns[:, 0], torch.tensor([33.0, 32.0]))

    def test_boundary_ignores_autoreset_value(self) -> None:
        arguments = {
            "rewards": torch.tensor([[1.0], [2.0]]),
            "values": torch.tensor([[10.0], [20.0]]),
            "terminated": torch.tensor([[False], [True]]),
            "truncated": torch.tensor([[False], [False]]),
            "gamma": 1.0,
            "gae_lambda": 1.0,
        }
        first = generalized_advantage_estimation(
            last_value=torch.tensor([30.0]),
            **arguments,
        )
        second = generalized_advantage_estimation(
            last_value=torch.tensor([1_000_000.0]),
            **arguments,
        )
        torch.testing.assert_close(first[0][:, 0], torch.tensor([-7.0, -18.0]))
        torch.testing.assert_close(first[1][:, 0], torch.tensor([3.0, 2.0]))
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_middle_boundary_blocks_later_episode(self) -> None:
        rewards = torch.tensor([[1.0], [2.0], [100.0], [200.0]])
        values = torch.tensor([[10.0], [20.0], [300.0], [400.0]])
        terminated = torch.tensor([[False], [True], [False], [False]])
        truncated = torch.zeros_like(terminated)
        first = generalized_advantage_estimation(
            rewards,
            values,
            terminated,
            truncated,
            torch.tensor([500.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        rewards[2:] = -10_000.0
        values[2:] = -20_000.0
        second = generalized_advantage_estimation(
            rewards,
            values,
            terminated,
            truncated,
            torch.tensor([-30_000.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(first[0][:2], second[0][:2])
        torch.testing.assert_close(first[1][:2], second[1][:2])

    def test_timeout_bootstraps_terminal_value_but_cuts_trace(self) -> None:
        rewards = torch.tensor([[1.0], [2.0], [1000.0]])
        values = torch.tensor([[10.0], [20.0], [3000.0]])
        terminated = torch.tensor([[False], [False], [False]])
        truncated = torch.tensor([[False], [True], [False]])
        timeout_values = torch.tensor([[0.0], [30.0], [0.0]])
        advantages, returns = generalized_advantage_estimation(
            rewards,
            values,
            terminated,
            truncated,
            torch.tensor([4000.0]),
            time_out_bootstrap_values=timeout_values,
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages[:2, 0], torch.tensor([23.0, 12.0]))
        torch.testing.assert_close(returns[:2, 0], torch.tensor([33.0, 32.0]))

    def test_timeout_requires_pre_reset_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminal-state values"):
            generalized_advantage_estimation(
                rewards=torch.tensor([[1.0]]),
                values=torch.tensor([[10.0]]),
                terminated=torch.tensor([[False]]),
                truncated=torch.tensor([[True]]),
                last_value=torch.tensor([1_000_000.0]),
            )


class PhaseFreeGaitTest(unittest.TestCase):
    def test_first_alternating_repeated_and_double_landings(self) -> None:
        previous = torch.tensor([-1, 0, 1, 0])
        first_contact = torch.tensor(
            [
                [True, False],
                [False, True],
                [False, True],
                [True, True],
            ]
        )
        single, alternating, repeated, next_previous = classify_foot_landings(
            first_contact,
            previous,
        )
        torch.testing.assert_close(
            single,
            torch.tensor([True, True, True, False]),
        )
        torch.testing.assert_close(
            alternating,
            torch.tensor([False, True, False, False]),
        )
        torch.testing.assert_close(
            repeated,
            torch.tensor([False, False, True, False]),
        )
        torch.testing.assert_close(next_previous, torch.tensor([0, 1, 1, 0]))


class MotionReferenceTest(unittest.TestCase):
    def test_cyclic_g1_reference_and_bounded_rewards(self) -> None:
        reference_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motions"
            / "g1_walk_mimickit.npz"
        )
        import numpy as np

        with np.load(reference_path, allow_pickle=False) as data:
            joint_names = tuple(str(name) for name in data["joint_names"].tolist())
            expected_middle_position = torch.from_numpy(data["joint_pos"][62].copy())

        reference = CyclicJointReference(
            reference_path,
            joint_names,
            "cpu",
            expected_archive_sha256=G1_WALK_ARCHIVE_SHA256,
            expected_source_sha256=G1_WALK_SOURCE_SHA256,
        )
        times = torch.tensor([0.0, 0.5 * reference.duration_s, reference.duration_s])
        position, velocity, phase = reference.sample(times)
        self.assertEqual(position.shape, (3, 29))
        self.assertEqual(velocity.shape, (3, 29))
        torch.testing.assert_close(phase, torch.tensor([0.0, 0.5, 0.0]))
        torch.testing.assert_close(position[0], position[2])
        torch.testing.assert_close(position[1], expected_middle_position)
        self.assertEqual(
            reference.source_sha256,
            G1_WALK_SOURCE_SHA256,
        )
        self.assertEqual(reference.archive_sha256, G1_WALK_ARCHIVE_SHA256)

        center = torch.tensor(DEFAULT_JOINT_POSITIONS)
        reference.retarget_to_position_envelope(center, max_offset=0.22)
        maximum_offset = torch.max(
            torch.abs(reference.joint_positions - center),
        ).item()
        self.assertLessEqual(maximum_offset, 0.220001)
        self.assertAlmostEqual(reference.retarget_max_offset, 0.22)
        self.assertLess(reference.retarget_min_scale, 1.0)

        position, velocity, _ = reference.sample(times)

        weights = torch.tensor(G1_DEEPMIMIC_DOF_WEIGHTS)
        pose_similarity, velocity_similarity, pose_error, velocity_error = (
            deepmimic_joint_similarity(
                position,
                velocity,
                position,
                velocity,
                weights,
            )
        )
        torch.testing.assert_close(pose_similarity, torch.ones(3))
        torch.testing.assert_close(velocity_similarity, torch.ones(3))
        torch.testing.assert_close(pose_error, torch.zeros(3))
        torch.testing.assert_close(velocity_error, torch.zeros(3))

        pose_reward, velocity_reward = gated_imitation_rewards(
            pose_similarity,
            velocity_similarity,
            torch.tensor([1.0, 0.5, 0.0]),
        )
        torch.testing.assert_close(
            pose_reward,
            torch.tensor([REFERENCE_POSE_WEIGHT, 0.5 * REFERENCE_POSE_WEIGHT, 0.0]),
        )
        torch.testing.assert_close(
            velocity_reward,
            torch.tensor(
                [REFERENCE_VELOCITY_WEIGHT, 0.5 * REFERENCE_VELOCITY_WEIGHT, 0.0]
            ),
        )


class NormalizationTest(unittest.TestCase):
    def test_observation_center_and_scales(self) -> None:
        obs = torch.zeros(2, 93)
        obs[:, 2] = -1.0
        obs[:, 3:6] = torch.tensor([4.0, -4.0, 1.0])
        obs[:, 6:35] = torch.tensor(DEFAULT_JOINT_POSITIONS)
        obs[:, 35:64] = 10.0
        obs[:, 64:93] = 1.0
        original = obs.clone()
        normalized = normalize_obs(obs)
        torch.testing.assert_close(obs, original)
        torch.testing.assert_close(
            normalized[0, 3:6],
            torch.tensor([1.0, -1.0, 1.0]),
        )
        torch.testing.assert_close(normalized[:, 6:35], torch.zeros(2, 29))
        torch.testing.assert_close(normalized[:, 35:64], torch.ones(2, 29))
        torch.testing.assert_close(normalized[:, 64:93], torch.ones(2, 29))

    def test_command_and_explicit_velocity_share_scales(self) -> None:
        physical = torch.tensor([[1.2, -0.6, 1.0]])
        command = torch.tensor([[1.2, -0.6, 1.0, 0.6, -0.8]])
        expected_command = torch.tensor([[1.0, -1.0, 1.0, 0.6, -0.8]])
        expected_velocity = torch.tensor([[1.0, -1.0, 1.0]])
        torch.testing.assert_close(normalize_command(command), expected_command)
        torch.testing.assert_close(
            normalize_command(physical),
            expected_velocity,
        )
        normalized = normalize_explicit_velocity(physical)
        torch.testing.assert_close(normalized, expected_velocity)
        torch.testing.assert_close(
            denormalize_explicit_velocity(normalized),
            physical,
        )

    def test_reward_profile_selects_command_contract(self) -> None:
        self.assertEqual(
            model_config_for_reward_profile("p1_walk_stable_v5").command_dim, 3
        )
        self.assertEqual(
            model_config_for_reward_profile("p1_walk_stable_v6_phase_rsi").command_dim,
            5,
        )

    def test_privileged_torque_uses_per_joint_limits(self) -> None:
        privileged = torch.zeros(1, 103)
        privileged[:, 2] = 0.8
        privileged[:, 74:103] = torch.tensor(JOINT_EFFORT_LIMITS)
        normalized = normalize_privileged(privileged)
        torch.testing.assert_close(normalized[:, 2], torch.zeros(1))
        torch.testing.assert_close(
            normalized[:, 74:103],
            torch.ones(1, 29),
        )


class ValueTargetStatisticsTest(unittest.TestCase):
    def test_large_offset_keeps_variance_precision(self) -> None:
        returns = torch.tensor([99_990.0, 100_000.0, 100_010.0])
        mean, scale = value_target_statistics(returns)
        torch.testing.assert_close(mean, torch.tensor(100_000.0))
        torch.testing.assert_close(
            scale,
            returns.to(torch.float64).std(correction=0).to(torch.float32),
        )

    def test_imitation_schedule_uses_only_current_run_span(self) -> None:
        self.assertAlmostEqual(
            linear_imitation_weight(3501, 3501, 4000, 0.15, 0.03), 0.15
        )
        self.assertAlmostEqual(
            linear_imitation_weight(4000, 3501, 4000, 0.15, 0.03), 0.03
        )
        middle = linear_imitation_weight(3750, 3501, 4000, 0.15, 0.03)
        self.assertGreater(middle, 0.03)
        self.assertLess(middle, 0.15)

    def test_reference_motion_ramp_starts_compatible_and_reaches_full(self) -> None:
        self.assertEqual(reference_motion_progress(4001, 4001, 200), 0.0)
        self.assertEqual(reference_motion_progress(4101, 4001, 200), 0.5)
        self.assertEqual(reference_motion_progress(4201, 4001, 200), 1.0)
        self.assertEqual(reference_motion_progress(5000, 4001, 200), 1.0)
        self.assertEqual(reference_motion_progress(4001, 4001, 0), 1.0)


class PolicyTransactionTest(unittest.TestCase):
    def test_restore_includes_adam_state_and_lr_backoff(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.Adam(
            [{"params": [parameter], "lr": 0.1, "name": "policy"}]
        )
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.tensor([0.3, -0.4])
        optimizer.step()

        parameter_snapshot, optimizer_snapshot = snapshot_policy_update(
            [parameter],
            optimizer,
        )
        saved_state = {
            name: value.clone() if torch.is_tensor(value) else value
            for name, value in optimizer.state[parameter].items()
        }
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.tensor([-10.0, 20.0])
        optimizer.step()
        restore_policy_update(
            [parameter],
            optimizer,
            parameter_snapshot,
            optimizer_snapshot,
        )

        torch.testing.assert_close(parameter, parameter_snapshot[0])
        for name, value in saved_state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(optimizer.state[parameter][name], value)
            else:
                self.assertEqual(optimizer.state[parameter][name], value)
        self.assertEqual(reduce_policy_learning_rate(optimizer, 0.01), 0.05)
        self.assertEqual(policy_learning_rate(optimizer), 0.05)

    def test_restricted_checkpoint_loader_supports_repository_state(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "iteration": 7,
                    "tensor": torch.tensor([1.0, 2.0]),
                    "train_args": {"log_dir": Path("logs/example")},
                },
                path,
            )
            checkpoint = load_checkpoint(path, map_location="cpu")
        self.assertEqual(checkpoint["iteration"], 7)
        self.assertEqual(checkpoint["train_args"]["log_dir"], Path("logs/example"))
        torch.testing.assert_close(checkpoint["tensor"], torch.tensor([1.0, 2.0]))

    def test_clear_named_optimizer_group_state(self) -> None:
        policy_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        critic_parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [policy_parameter],
                    "lr": 0.01,
                    "name": "policy",
                },
                {
                    "params": [critic_parameter],
                    "lr": 0.02,
                    "name": "critic",
                },
            ]
        )
        policy_parameter.grad = torch.ones_like(policy_parameter)
        critic_parameter.grad = torch.ones_like(critic_parameter)
        optimizer.step()

        self.assertEqual(clear_optimizer_group_state(optimizer, "policy"), 1)
        self.assertNotIn(policy_parameter, optimizer.state)
        self.assertIn(critic_parameter, optimizer.state)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_clear_all_optimizer_state_preserves_groups(self) -> None:
        policy_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        critic_parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.Adam(
            [
                {"params": [policy_parameter], "lr": 0.01, "name": "policy"},
                {"params": [critic_parameter], "lr": 0.02, "name": "critic"},
            ]
        )
        policy_parameter.grad = torch.ones_like(policy_parameter)
        critic_parameter.grad = torch.ones_like(critic_parameter)
        optimizer.step()

        self.assertEqual(clear_optimizer_state(optimizer), 2)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups], ["policy", "critic"]
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [0.01, 0.02]
        )

    def test_v7_schedule_distinguishes_warm_start_and_crash_resume(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            warm_path = root / "warm.pt"
            torch.save(
                {
                    "iteration": 4000,
                    "train_args": {
                        "reward_profile": "p1_walk_stable_v6_phase_rsi_imitation"
                    },
                },
                warm_path,
            )
            warm_args = Namespace(
                resume=warm_path,
                reward_profile="p1_walk_stable_v7_full_reference",
                reference_motion_origin_iteration=None,
                reference_motion_ramp_iterations=200,
                iterations=5000,
                initial_imitation_weight=0.75,
                final_imitation_weight=0.30,
            )
            self.assertEqual(prepare_reference_motion_schedule(warm_args), 4001)
            self.assertEqual(warm_args.reference_motion_origin_iteration, 4001)
            self.assertEqual(warm_args.initial_reference_motion_progress, 0.0)

            resumed_path = root / "resumed.pt"
            torch.save(
                {
                    "iteration": 4100,
                    "train_args": {
                        "reward_profile": "p1_walk_stable_v7_full_reference",
                        "reference_motion_origin_iteration": 4001,
                        "reference_motion_ramp_iterations": 200,
                        "current_reference_motion_progress": 0.495,
                        "iterations": 5000,
                        "initial_imitation_weight": 0.75,
                        "final_imitation_weight": 0.30,
                        "current_imitation_reward_weight": linear_imitation_weight(
                            4100,
                            4001,
                            5000,
                            0.75,
                            0.30,
                        ),
                    },
                },
                resumed_path,
            )
            resumed_args = Namespace(
                resume=resumed_path,
                reward_profile="p1_walk_stable_v7_full_reference",
                reference_motion_origin_iteration=None,
                reference_motion_ramp_iterations=200,
                iterations=5000,
                initial_imitation_weight=0.75,
                final_imitation_weight=0.30,
            )
            self.assertEqual(prepare_reference_motion_schedule(resumed_args), 4101)
            self.assertEqual(resumed_args.reference_motion_origin_iteration, 4001)
            self.assertEqual(resumed_args.initial_reference_motion_progress, 0.5)


class SquashedPolicyTest(unittest.TestCase):
    def test_ppo_log_probability_and_scheduled_cap(self) -> None:
        torch.manual_seed(5)
        distribution = DiagonalGaussian(
            29,
            initial_std=0.3,
            min_std=0.05,
            max_std=0.6,
        )
        mean = torch.randn(32, 29)
        actions, pre_tanh, old_log_prob = distribution.sample_for_ppo(mean)
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))
        torch.testing.assert_close(
            distribution.ppo_log_prob(mean, pre_tanh),
            old_log_prob,
        )
        with torch.no_grad():
            distribution.log_std.fill_(10.0)
        distribution.clamp_std_parameters_(maximum_std=0.2)
        _, maximum = distribution.std_statistics()
        self.assertLessEqual(maximum, 0.200001)

        distribution.reset_std_parameters_(0.12)
        mean_std, maximum_std = distribution.std_statistics()
        self.assertAlmostEqual(mean_std, 0.12, places=6)
        self.assertAlmostEqual(maximum_std, 0.12, places=6)
        with self.assertRaises(ValueError):
            distribution.reset_std_parameters_(0.01)


if __name__ == "__main__":
    unittest.main()
