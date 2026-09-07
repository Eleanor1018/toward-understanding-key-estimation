"""Key系列的梯度、概率契约与辅助优化；使用真实CPU网络和Adam。"""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from estnet.key_networks import KeyPolicy
from estnet.key_ppo import KeyPPO


def key_config(variant="key1", **overrides):
    fields = {
        "variant": variant, "obs_dim": 42, "history_steps": 50, "command_dim": 7,
        "action_dim": 12, "critic_dim": 152, "heightmap_dim": 18, "latent_dim": 16,
        "encoder_hidden": (16, 8), "actor_hidden": (16, 8), "critic_hidden": (16, 8),
        "decoder_hidden": (8, 16), "init_std": .8, "learning_rate": 5e-4, "epochs": 2,
        "minibatches": 2, "gamma": .996, "gae_lambda": .95, "clip": .2, "value_coef": 1.,
        "entropy_coef": .008, "velocity_coef": 1., "max_grad_norm": 1., "desired_kl": .01,
        "min_learning_rate": 1e-5, "max_learning_rate": 1e-3, "heightmap_coef": .5,
        "prediction_coef": 2., "vae_beta": 50.,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def observation(count=8):
    return {
        "history": torch.randn(count, 50, 42), "obs": torch.randn(count, 42),
        "command": torch.randn(count, 7), "critic": torch.randn(count, 152),
        "velocity": torch.randn(count, 3), "heightmap": torch.randn(count, 18),
    }


def rollout(model, count=8):
    data = observation(count)
    action, log_prob, value, mean, std = model.act(data)
    return dict(data, action=action, old_log_prob=log_prob, old_value=value,
                old_mean=mean, old_std=std, returns=value + torch.randn(count),
                advantages=torch.randn(count))


def gradient_size(module):
    return sum(parameter.grad.abs().sum().item() for parameter in module.parameters()
               if parameter.grad is not None)


class KeyLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_key_architecture_and_raw_gaussian_contract(self):
        data = observation()
        for variant, explicit_dim in (("key1", 3), ("key2", 21)):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant))
                auxiliary = model.auxiliary(data["history"], data["obs"], data["command"])
                self.assertEqual(auxiliary["velocity"].shape, (8, 3))
                self.assertEqual(auxiliary["mu"].shape, (8, 16))
                self.assertEqual(auxiliary["logvar"].shape, (8, 16))
                self.assertEqual(auxiliary["prediction"].shape, (8, 42))
                self.assertEqual(hasattr(model, "heightmap_head"), variant == "key2")
                self.assertEqual("heightmap" in auxiliary, variant == "key2")
                if variant == "key2":
                    self.assertEqual(auxiliary["heightmap"].shape, (8, 18))
                self.assertEqual(model.actor[0].in_features, 42 + 7 + explicit_dim + 16)
                self.assertEqual(model.decoder[0].in_features, explicit_dim + 16)
                self.assertFalse(hasattr(model, "estimator"))
                action, log_prob, value, _, std = model.act(data)
                distribution = model.distribution(data["history"], data["obs"], data["command"])
                self.assertEqual(action.shape, (8, 12))
                self.assertEqual(value.shape, (8,))
                torch.testing.assert_close(log_prob, distribution.log_prob(action).sum(-1))
                torch.testing.assert_close(std, torch.full_like(std, .8))
                self.assertTrue((action.abs() > 1).any())

    def test_policy_gradient_reaches_explicit_heads_and_mu(self):
        data = observation()
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant))
                distribution = model.distribution(data["history"], data["obs"], data["command"])
                action = distribution.mean.detach() + .4
                (-distribution.log_prob(action).sum(-1).mean()).backward()
                for name in ("encoder", "velocity_head", "mu_head", "actor"):
                    self.assertGreater(gradient_size(getattr(model, name)), 0., name)
                if variant == "key2":
                    self.assertGreater(gradient_size(model.heightmap_head), 0.)
                # actor只使用mu：策略梯度不应通过采样方差或decoder绕行。
                for name in ("logvar_head", "decoder", "critic"):
                    self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))

    def test_reconstruction_gradient_reaches_mu_logvar_and_explicit_heads(self):
        data = observation()
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant)).train()
                auxiliary = model.auxiliary(data["history"], data["obs"], data["command"])
                F.mse_loss(auxiliary["prediction"], data["obs"]).backward()
                for name in ("encoder", "velocity_head", "mu_head", "logvar_head", "decoder"):
                    self.assertGreater(gradient_size(getattr(model, name)), 0., name)
                if variant == "key2":
                    self.assertGreater(gradient_size(model.heightmap_head), 0.)
                self.assertTrue(all(p.grad is None for p in model.actor.parameters()))

    def test_actor_log_probability_is_stable_across_decoder_sampling(self):
        data = observation()
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant)).train()
                fixed_action = torch.randn(8, 12)
                distribution = model.distribution(data["history"], data["obs"], data["command"])
                before = distribution.log_prob(fixed_action)
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"], data["command"])
                self.assertFalse(torch.equal(first["prediction"], second["prediction"]))
                after = model.distribution(data["history"], data["obs"], data["command"])
                torch.testing.assert_close(before, after.log_prob(fixed_action), rtol=0, atol=0)
                model.eval()
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"], data["command"])
                torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)

    def test_ground_truth_cannot_enter_actor_or_current_target_enter_decoder(self):
        data = observation()
        changed = dict(data, velocity=data["velocity"] + 1000.,
                       heightmap=data["heightmap"] - 1000., critic=data["critic"] + 1000.)
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant)).eval()
                torch.manual_seed(101)
                original_action = model.act(data)
                torch.manual_seed(101)
                changed_action = model.act(changed)
                # critic真值可以改变value；不允许改变动作、logprob、mean或std。
                for index in (0, 1, 3, 4):
                    torch.testing.assert_close(original_action[index], changed_action[index], rtol=0, atol=0)
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"] + 1000., data["command"] - 1000.)
                torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)

    def test_auxiliary_loss_coefficients_and_kl_reduction(self):
        data = {key: torch.zeros_like(value) for key, value in observation().items()}
        indices = torch.arange(8)
        # 给定mu=2、logvar=0，每维KL是2；不是16维求和后的32。
        fixed = {"velocity": torch.ones(8, 3), "prediction": torch.full((8, 42), 2.),
                 "mu": torch.full((8, 16), 2.), "logvar": torch.zeros(8, 16),
                 "heightmap": torch.full((8, 18), 3.)}
        for variant, expected in (("key1", 109.), ("key2", 113.5)):
            with self.subTest(variant=variant):
                model = KeyPolicy(key_config(variant))
                learner = KeyPPO(model, key_config(variant))
                with patch.object(model, "auxiliary", return_value=fixed):
                    loss, metrics = learner.auxiliary_loss(data, indices, None)
                self.assertEqual(loss.item(), expected)
                self.assertEqual(metrics["latent_kl"].item(), 2.)
                self.assertEqual(metrics["prediction_loss"].item(), 4.)
                self.assertEqual(metrics["velocity_loss"].item(), 1.)
                self.assertEqual("heightmap_loss" in metrics, variant == "key2")

    def test_finite_joint_update_and_exact_next_adam_step_after_restore(self):
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                cfg = key_config(variant)
                model = KeyPolicy(cfg)
                learner = KeyPPO(model, cfg)
                batch = rollout(model)
                original_batch = copy.deepcopy(batch)
                metrics = learner.update(batch)
                self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in metrics.values()))
                self.assertEqual(metrics["gradient_steps"], 4.)
                for name in ("velocity_loss", "prediction_loss", "latent_kl", "latent_mu_abs", "latent_std_mean"):
                    self.assertIn(name, metrics)
                self.assertEqual("heightmap_loss" in metrics, variant == "key2")
                self.assert_tree_equal(batch, original_batch)
                clone = KeyPolicy(cfg)
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                resumed = KeyPPO(clone, cfg)
                resumed.load_state_dict(copy.deepcopy(learner.state_dict()))
                self.assertEqual(resumed.updates, 1)
                next_batch = rollout(model)
                # randperm与decoder重参数采样都由同一CPU RNG状态恢复。
                torch.manual_seed(41)
                metrics_next = learner.update(next_batch)
                torch.manual_seed(41)
                metrics_resumed = resumed.update(next_batch)
                self.assert_tree_equal(model.state_dict(), clone.state_dict())
                self.assert_tree_equal(learner.state_dict(), resumed.state_dict())
                self.assertEqual(metrics_next, metrics_resumed)

    def test_actual_auxiliary_updates_reduce_joint_loss(self):
        for variant in ("key1", "key2"):
            with self.subTest(variant=variant):
                cfg = key_config(variant, epochs=1, minibatches=1, value_coef=0.,
                                 entropy_coef=0., desired_kl=0.)
                model = KeyPolicy(cfg)
                learner = KeyPPO(model, cfg)
                batch = rollout(model)
                batch["advantages"].zero_()
                indices = torch.arange(8)
                model.eval()
                with torch.no_grad():
                    before = learner.auxiliary_loss(batch, indices, None)[0].item()
                model.train()
                for _ in range(5):
                    learner.update(batch)
                model.eval()
                with torch.no_grad():
                    after = learner.auxiliary_loss(batch, indices, None)[0].item()
                self.assertLess(after, before)

    def test_key2_rejects_missing_or_broadcast_heightmap_targets(self):
        model = KeyPolicy(key_config("key2"))
        learner = KeyPPO(model, key_config("key2"))
        batch = rollout(model)
        with self.assertRaisesRegex(ValueError, "Missing rollout fields"):
            learner.update({key: value for key, value in batch.items() if key != "heightmap"})
        batch["heightmap"] = torch.zeros(8, 1)
        with self.assertRaisesRegex(ValueError, "Heightmap supervision"):
            learner.update(batch)
        self.assertEqual(learner.updates, 0)
        self.assertFalse(learner.optimizer.state)


if __name__ == "__main__":
    unittest.main()
