"""Key架构接入共用采样、历史和检查点时的边界验证；不启动Isaac。"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from estnet.factory import build_model, build_ppo, config_for_variant
from estnet.history import History
from estnet.preflight import ASSET_HASHES
from estnet.run import collect_rollout, load_evaluation_checkpoint, save_checkpoint
from estnet.resume import load_training_checkpoint


class KeyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self, variant):
        return replace(config_for_variant(variant), num_envs=4, horizon=2,
                       encoder_hidden=(16, 8), actor_hidden=(16,), critic_hidden=(16,),
                       decoder_hidden=(16,), epochs=1, minibatches=1)

    def test_history_excludes_current_without_reusing_previous_episode(self):
        history = History(2, 3, 1, "cpu")
        first = history.append(torch.tensor([[1.], [10.]]), exclude_current=True)
        second = history.append(torch.tensor([[2.], [20.]]), exclude_current=True)
        torch.testing.assert_close(first, torch.tensor([[[1.], [1.], [1.]], [[10.], [10.], [10.]]]))
        torch.testing.assert_close(second, first)
        third = history.append(torch.tensor([[3.], [30.]]), exclude_current=True)
        torch.testing.assert_close(third[:, :, 0], torch.tensor([[1., 1., 2.], [10., 10., 20.]]))
        history.reset(torch.tensor([0]))
        reset = history.append(torch.tensor([[99.], [40.]]), exclude_current=True)
        torch.testing.assert_close(reset[:, :, 0], torch.tensor([[99., 99., 99.], [10., 20., 30.]]))
        torch.testing.assert_close(third[0, :, 0], torch.tensor([1., 1., 2.]))

    def test_key_rollout_keeps_current_targets_before_inplace_environment_step(self):
        cfg = self.tiny("key2")
        model = build_model(cfg)

        class ReusingEnv:
            def __init__(self):
                self.current = {
                    "obs": torch.ones(4, 42), "history": torch.zeros(4, 50, 42),
                    "command": torch.zeros(4, 7), "critic": torch.ones(4, 152),
                    "velocity": torch.ones(4, 3), "heightmap": torch.ones(4, 18),
                }

            def step(self, action):
                for value in self.current.values():
                    value.add_(1.)
                return self.current, torch.ones(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {
                    "final_critic": self.current["critic"].clone(),
                    "metrics": {"marker": torch.ones(4)}, "reward_terms": {"reward": torch.ones(4)}}

        env = ReusingEnv()
        _, batch, _ = collect_rollout(env, model, env.current, cfg)
        for field in ("obs", "velocity", "heightmap"):
            torch.testing.assert_close(batch[field][:, 0], torch.tensor([1.] * 4 + [2.] * 4))
        torch.testing.assert_close(batch["history"][:, 0, 0], torch.tensor([0.] * 4 + [1.] * 4))
        metrics = build_ppo(model, cfg).update(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_schema_preserves_variant_and_roundtrips_trained_key_checkpoints(self):
        asset = {"ready": True, "files": [{"sha256": value} for value in ASSET_HASHES.values()]}
        with tempfile.TemporaryDirectory() as temporary:
            for variant in ("key1", "key2"):
                with self.subTest(variant=variant):
                    cfg = self.tiny(variant)
                    model, path = build_model(cfg), Path(temporary) / f"{variant}.pt"
                    learner = build_ppo(model, cfg)
                    save_checkpoint(path, model, learner, cfg, 0, asset)
                    evaluated, saved_cfg = load_evaluation_checkpoint(path, asset)
                    resumed, resume_cfg = load_training_checkpoint(path, asset)
                    self.assertEqual(saved_cfg.variant, variant)
                    self.assertEqual(resume_cfg.critic_dim, 152)
                    self.assertNotIn("optimizer", evaluated)
                    self.assertIn("optimizer", resumed)
                    broken = dict(resumed, schema="g1-estnet-flat-v1")
                    torch.save(broken, path)
                    with self.assertRaises(ValueError):
                        load_evaluation_checkpoint(path, asset)


if __name__ == "__main__":
    unittest.main()
