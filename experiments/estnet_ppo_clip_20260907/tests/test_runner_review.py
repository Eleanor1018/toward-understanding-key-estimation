"""Static smoke bookkeeping regressions; these CPU fakes prove no physics."""

import unittest
from types import SimpleNamespace

import torch

from estnet.config import Config
from estnet.run import smoke_check


class SmokeVectorEnv:
    """Scripted measurements with reset episodes deliberately made misleading."""

    device = "cpu"

    def __init__(
        self,
        heights=(0.78, 0.78),
        speed=0.0,
        failures=None,
        after_reset=None,
        nan_observation=False,
    ):
        self.num_envs = len(heights)
        self.heights = torch.tensor(heights)
        self.speed = speed
        self.failures = failures or {}
        self.after_reset = after_reset
        self.nan_observation = nan_observation
        self.steps = 0
        self.robot = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(29)])
        self.leg_ids = torch.arange(12)
        self.mass = torch.full((self.num_envs,), 35.0)
        self.extras = {}

    def step(self, action):
        self.steps += 1
        torch.testing.assert_close(action, torch.zeros(self.num_envs, 12))
        terminal = torch.tensor(
            [
                self.steps in self.failures.get(env_id, ())
                for env_id in range(self.num_envs)
            ]
        )
        obs = torch.zeros(self.num_envs, 42)
        if self.nan_observation:
            obs[0, 0] = float("nan")
        metrics = {
            "base_height": self.heights.clone(),
            "double_support": torch.ones(self.num_envs),
            "forward_velocity": torch.full((self.num_envs,), self.speed),
            "horizontal_speed": torch.full((self.num_envs,), abs(self.speed)),
            "stance_slip": torch.full((self.num_envs,), abs(self.speed)),
            "upright_cosine": torch.ones(self.num_envs),
            "joint_tracking_error": torch.zeros(self.num_envs),
            "torque_saturation_fraction": torch.zeros(self.num_envs),
        }
        if self.after_reset is not None:
            for env_id, times in self.failures.items():
                if self.steps > min(times):
                    metrics["base_height"][env_id] = self.after_reset
                    metrics["forward_velocity"][env_id] = self.after_reset
                    metrics["horizontal_speed"][env_id] = self.after_reset
                    metrics["stance_slip"][env_id] = self.after_reset
        return (
            {"obs": obs},
            torch.zeros(self.num_envs),
            terminal,
            torch.zeros_like(terminal),
            {"metrics": metrics},
        )


class SmokeReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_smoke(self, env):
        return smoke_check(env, {}, Config())

    def test_consistently_stable_first_episodes_pass(self):
        result = self.run_smoke(SmokeVectorEnv())
        self.assertEqual(result["status"], "static_pd_smoke_passed")
        self.assertEqual(result["static_balance_pass_fraction"], 1.0)
        self.assertTrue(result["interface_finite"])
        self.assertTrue(result["height_matches_target"])
        self.assertEqual(result["terminal_events"], 0)
        self.assertFalse(result["walking_verified"])

    def test_opposite_height_errors_cannot_cancel_across_environments(self):
        result = self.run_smoke(SmokeVectorEnv(heights=(0.72,) * 8 + (0.84,) * 8))
        self.assertEqual(result["status"], "nominal_pose_unstable")
        self.assertEqual(result["static_balance_pass_fraction"], 0.0)
        self.assertFalse(result["height_matches_target"])
        self.assertTrue(result["interface_finite"])

    def test_double_support_with_large_drift_is_not_static_balance(self):
        result = self.run_smoke(SmokeVectorEnv(speed=0.4))
        self.assertEqual(result["status"], "nominal_pose_unstable")
        self.assertEqual(result["static_balance_pass_fraction"], 0.0)
        self.assertTrue(result["interface_finite"])

    def test_reset_episodes_neither_inflate_terminal_count_nor_settled_metrics(self):
        env = SmokeVectorEnv(failures={0: (50, 150, 250)}, after_reset=99.0)
        result = self.run_smoke(env)
        self.assertEqual(result["status"], "nominal_pose_unstable")
        self.assertEqual(result["terminal_events"], 1)
        self.assertEqual(result["static_balance_pass_fraction"], 0.5)
        self.assertFalse(result["height_matches_target"])
        self.assertTrue(result["interface_finite"])
        self.assertAlmostEqual(
            result["last_second_metrics"]["base_height"], 0.78, places=5
        )
        self.assertAlmostEqual(result["settled_height_std"], 0.0, places=5)

    def test_nan_in_inactive_reset_metrics_does_not_contaminate_first_episode(self):
        result = self.run_smoke(
            SmokeVectorEnv(failures={0: (50,)}, after_reset=float("nan"))
        )
        self.assertEqual(result["static_balance_pass_fraction"], 0.5)
        self.assertEqual(result["terminal_events"], 1)
        self.assertAlmostEqual(
            result["last_second_metrics"]["base_height"], 0.78, places=5
        )
        self.assertTrue(result["interface_finite"])

    def test_early_failures_have_no_claimed_settled_height(self):
        result = self.run_smoke(
            SmokeVectorEnv(failures={0: (50,), 1: (60,)}, after_reset=0.78)
        )
        self.assertEqual(result["status"], "nominal_pose_unstable")
        self.assertEqual(result["terminal_events"], 2)
        self.assertEqual(result["static_balance_pass_fraction"], 0.0)
        self.assertFalse(result["height_matches_target"])
        self.assertEqual(result["last_second_metrics"], {})
        self.assertIsNone(result["settled_height_std"])

    def test_invalid_interface_observation_is_an_error_not_balance_failure(self):
        with self.assertRaises(FloatingPointError):
            self.run_smoke(SmokeVectorEnv(nan_observation=True))


if __name__ == "__main__":
    unittest.main()
