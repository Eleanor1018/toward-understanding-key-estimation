"""三组消融的采样标签时序、无显式监督路径与检查点恢复验证。"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from estnet.factory import build_model, build_ppo, canonical_variant, config_for_variant
from estnet.preflight import ASSET_HASHES
from estnet.resume import load_training_checkpoint
from estnet.run import collect_rollout, load_evaluation_checkpoint, save_checkpoint


class AblationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self, variant):
        return replace(config_for_variant(variant), num_envs=4, horizon=2,
                       encoder_hidden=(16, 8), actor_hidden=(16,), critic_hidden=(16,),
                       decoder_hidden=(16,), epochs=1, minibatches=1)

    def collect(self, cfg, model):
        class ReusingEnv:
            def __init__(self):
                self.current = {
                    "obs": torch.ones(4, 42), "history": torch.zeros(4, 50, 42),
                    "command": torch.zeros(4, 7), "critic": torch.ones(4, 152),
                    **{name: torch.ones(4, dim) for name, dim in cfg.supervision_dims.items()},
                }

            def step(self, action):
                for value in self.current.values():
                    value.add_(1.)
                return self.current, torch.ones(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {
                    "final_critic": self.current["critic"].clone(),
                    "metrics": {"marker": torch.ones(4)}, "reward_terms": {"reward": torch.ones(4)}}

        env = ReusingEnv()
        return collect_rollout(env, model, env.current, cfg)[1]

    def test_rollout_targets_are_current_and_survive_inplace_next_step(self):
        for variant in ("fullest", "irrest", "implicit"):
            with self.subTest(variant=variant):
                cfg = self.tiny(variant)
                model = build_model(cfg)
                batch = self.collect(cfg, model)
                for field in ("obs", *cfg.supervision_dims):
                    torch.testing.assert_close(batch[field][:, 0], torch.tensor([1.] * 4 + [2.] * 4))
                torch.testing.assert_close(batch["history"][:, 0, 0], torch.tensor([0.] * 4 + [1.] * 4))
                self.assertEqual("velocity" in batch, variant == "fullest")
                self.assertEqual("body_height" in batch, variant != "implicit")
                metrics = build_ppo(model, cfg).update(batch)
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
                self.assertEqual("velocity_rmse" in metrics, variant == "fullest")
                self.assertEqual("body_height_loss" in metrics, variant != "implicit")

    def test_trained_checkpoints_roundtrip_and_reject_cross_variant_weights(self):
        asset = {"ready": True, "files": [{"sha256": value} for value in ASSET_HASHES.values()]}
        with tempfile.TemporaryDirectory() as temporary:
            for variant in ("fullest", "irrest", "implicit"):
                with self.subTest(variant=variant):
                    cfg = self.tiny(variant)
                    model, path = build_model(cfg), Path(temporary) / f"{variant}.pt"
                    learner = build_ppo(model, cfg)
                    learner.update(self.collect(cfg, model))
                    save_checkpoint(path, model, learner, cfg, 1, asset)
                    evaluated, saved_cfg = load_evaluation_checkpoint(path, asset)
                    resumed, resume_cfg = load_training_checkpoint(path, asset)
                    self.assertEqual(saved_cfg.variant, variant)
                    self.assertEqual(resume_cfg.critic_dim, 152)
                    self.assertEqual(resumed["optimizer"]["updates"], 1)
                    self.assertNotIn("optimizer", evaluated)
                    torch.testing.assert_close(resumed["model"]["log_std"], model.log_std)
                    broken = dict(resumed, schema="g1-key1-flat-v1")
                    torch.save(broken, path)
                    with self.assertRaises(ValueError):
                        load_evaluation_checkpoint(path, asset)

    def test_irrest_spelling_and_legacy_checkpoint_default_are_unambiguous(self):
        self.assertEqual(canonical_variant("IllEst"), "irrest")
        self.assertEqual(config_for_variant("illest").schema, "g1-irrest-flat-v1")
        # 新增高度损失字段不影响旧EstNet/Key检查点；缺字段时采用dataclass默认值。
        from estnet.config import Config
        for variant in ("estnet", "key1", "key2"):
            saved = config_for_variant(variant).to_dict()
            del saved["body_height_coef"]
            restored = Config(**saved)
            restored.validate()
            self.assertEqual(restored.variant, variant)


if __name__ == "__main__":
    unittest.main()
