"""真实CPU EstNet/PPO续训回归；不启动仿真或CUDA。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from estnet.config import Config
from estnet.networks import EstNet
from estnet.ppo import PPO
from estnet.preflight import ASSET_HASHES
from estnet.resume import load_training_checkpoint, restore_training_state
from estnet.run import save_checkpoint


def batch_for(model, count=12):
    observation = {"history": torch.randn(count, 50, 42), "obs": torch.randn(count, 42),
                   "command": torch.randn(count, 7), "critic": torch.randn(count, 61),
                   "velocity": torch.randn(count, 3)}
    action, logp, value, mean, std = model.act(observation)
    return {**observation, "action": action, "old_log_prob": logp, "old_value": value,
            "old_mean": mean, "old_std": std, "returns": value + torch.randn(count),
            "advantages": torch.randn(count)}


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(81)
        self.temp = tempfile.TemporaryDirectory(prefix="estnet-resume-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model_00001.pt"
        self.cfg = Config(num_envs=4, encoder_hidden=(12,), actor_hidden=(12,), critic_hidden=(12,),
                          epochs=2, minibatches=2)
        self.asset = {"ready": True, "files": [
            {"path": "/assets/g1/" + name, "sha256": digest} for name, digest in ASSET_HASHES.items()]}
        self.model = EstNet(self.cfg)
        self.ppo = PPO(self.model, self.cfg)
        self.ppo.update(batch_for(self.model))
        self.next_batch = batch_for(self.model)
        save_checkpoint(self.path, self.model, self.ppo, self.cfg, 1, self.asset)
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)

    def save_bad(self, payload):
        torch.save(payload, self.path)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_restored_next_update_matches_uninterrupted_model_adam_lr_and_rng(self):
        torch.set_rng_state(self.payload["torch_rng"])
        expected_metrics = self.ppo.update(self.next_batch)
        expected_rng = torch.get_rng_state().clone()
        payload, cfg = load_training_checkpoint(self.path, self.asset)
        restored_model = EstNet(cfg)
        restored_ppo = PPO(restored_model, cfg)
        record = restore_training_state(restored_model, restored_ppo, payload)
        self.assertEqual(record["iteration"], 1)
        self.assertEqual(record["ppo_updates"], 1)
        self.assertTrue(record["cpu_rng_restored"])
        self.assertFalse(record["cuda_rng_restored"])
        self.assertFalse(record["exact_resume"])
        self.assertEqual(record["resume_mode"], "new_episodes")
        torch.testing.assert_close(torch.get_rng_state(), self.payload["torch_rng"], rtol=0, atol=0)
        actual_metrics = restored_ppo.update(self.next_batch)
        self.assert_nested_equal(self.model.state_dict(), restored_model.state_dict())
        self.assert_nested_equal(self.ppo.state_dict(), restored_ppo.state_dict())
        self.assertEqual(expected_metrics, actual_metrics)
        self.assertEqual(restored_ppo.updates, 2)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_optimizer_is_required_and_missing_moments_do_not_become_warm_start(self):
        for missing in ("optimizer", "parameter_state"):
            with self.subTest(missing=missing):
                bad = copy.deepcopy(self.payload)
                if missing == "optimizer":
                    del bad["optimizer"]
                else:
                    state = bad["optimizer"]["optimizer"]["state"]
                    del state[next(iter(state))]
                self.save_bad(bad)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(self.path, self.asset)

    def test_corrupt_adam_moments_and_step_are_rejected_before_training(self):
        for corruption in ("shape", "nonfinite", "negative_second_moment", "fractional_step"):
            with self.subTest(corruption=corruption):
                bad = copy.deepcopy(self.payload)
                state = next(iter(bad["optimizer"]["optimizer"]["state"].values()))
                if corruption == "shape":
                    state["exp_avg"] = torch.zeros(1, 1, 1)
                elif corruption == "nonfinite":
                    state["exp_avg"].flatten()[0] = float("nan")
                elif corruption == "negative_second_moment":
                    state["exp_avg_sq"].flatten()[0] = -1.
                else:
                    state["step"] = torch.tensor(1.5)
                self.save_bad(bad)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(self.path, self.asset)

    def test_invalid_learning_rates_and_update_counts_are_rejected(self):
        for rate in (float("nan"), float("inf"), 0., -1., self.cfg.max_learning_rate * 2):
            with self.subTest(learning_rate=rate):
                bad = copy.deepcopy(self.payload)
                bad["optimizer"]["optimizer"]["param_groups"][0]["lr"] = rate
                self.save_bad(bad)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(self.path, self.asset)
        for updates in (None, True, -1, 0, 2, 1.0):
            with self.subTest(updates=updates):
                bad = copy.deepcopy(self.payload)
                bad["optimizer"]["updates"] = updates
                self.save_bad(bad)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(self.path, self.asset)

    def test_cpu_rng_is_validated_without_mutating_callers_rng(self):
        before = torch.get_rng_state().clone()
        load_training_checkpoint(self.path, self.asset)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        for state in (torch.ones(12), torch.ones(12, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)):
            with self.subTest(shape=state.shape, dtype=state.dtype):
                bad = copy.deepcopy(self.payload)
                bad["torch_rng"] = state
                self.save_bad(bad)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(self.path, self.asset)

    def test_optional_cuda_rng_does_not_initialize_cuda_for_cpu_model(self):
        payload = copy.deepcopy(self.payload)
        del payload["torch_rng"]
        payload["torch_cuda_rng"] = torch.zeros(16, dtype=torch.uint8)
        self.save_bad(payload)
        payload, cfg = load_training_checkpoint(self.path, self.asset)
        model = EstNet(cfg)
        ppo = PPO(model, cfg)
        with patch("torch.cuda.set_rng_state", side_effect=AssertionError("CPU restore must not call CUDA")):
            record = restore_training_state(model, ppo, payload)
        self.assertFalse(record["cpu_rng_restored"])
        self.assertFalse(record["cuda_rng_restored"])
        self.assertEqual(record["cuda_rng_status"], "not_applicable_cpu_model")
        self.assertFalse(record["exact_resume"])


if __name__ == "__main__":
    unittest.main()
