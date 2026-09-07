"""验证消融确实移除相应估计头、监督损失与指标，并保留可训练梯度。"""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from estnet.ablation_networks import AblationPolicy
from estnet.ablation_ppo import AblationPPO

VARIANTS = ("fullest", "irrest", "implicit")
HEADS = {"fullest": ("velocity", "heightmap", "body_height"),
         "irrest": ("body_height",), "implicit": ()}


def config(variant, **overrides):
    fields = {
        "variant": variant, "obs_dim": 42, "history_steps": 50, "command_dim": 7,
        "action_dim": 12, "critic_dim": 152, "heightmap_dim": 18, "latent_dim": 16,
        "encoder_hidden": (16, 8), "actor_hidden": (16, 8), "critic_hidden": (16, 8),
        "decoder_hidden": (8, 16), "init_std": .8, "learning_rate": 5e-4, "epochs": 2,
        "minibatches": 2, "clip": .2, "value_coef": 1., "entropy_coef": .008,
        "velocity_coef": 1., "max_grad_norm": 1., "desired_kl": .01,
        "min_learning_rate": 1e-5, "max_learning_rate": 1e-3, "heightmap_coef": .5,
        "body_height_coef": 2., "prediction_coef": 2., "vae_beta": 50.,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def observation(count=8):
    return {
        "history": torch.randn(count, 50, 42), "obs": torch.randn(count, 42),
        "command": torch.randn(count, 7), "critic": torch.randn(count, 152),
        "velocity": torch.randn(count, 3), "heightmap": torch.randn(count, 18),
        "body_height": torch.randn(count, 1),
    }


def rollout(model):
    data = observation()
    action, log_prob, value, mean, std = model.act(data)
    # 无显式头的消融连对应训练标签都不提供，确保没有隐藏的必需输入。
    data = {name: value for name, value in data.items()
            if name not in ("velocity", "heightmap", "body_height") or name in model.explicit_names}
    return dict(data, action=action, old_log_prob=log_prob, old_value=value,
                old_mean=mean, old_std=std, returns=value + torch.randn(8),
                advantages=torch.randn(8))


def gradient_size(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


class AblationLearningTests(unittest.TestCase):
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

    def test_architectures_have_exactly_the_paper_explicit_heads(self):
        data = observation()
        for variant, explicit_dim in (("fullest", 22), ("irrest", 1), ("implicit", 0)):
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant))
                auxiliary = model.auxiliary(data["history"], data["obs"], data["command"])
                self.assertEqual(set(auxiliary), {"mu", "logvar", "prediction", *HEADS[variant]})
                for name in ("velocity", "heightmap", "body_height"):
                    self.assertEqual(hasattr(model, f"{name}_head"), name in HEADS[variant])
                self.assertFalse(hasattr(model, "estimator"))
                self.assertEqual(model.has_velocity_estimate, variant == "fullest")
                self.assertEqual(model.actor[0].in_features, 42 + 7 + 16 + explicit_dim)
                self.assertEqual(model.decoder[0].in_features, 16 + explicit_dim)
                self.assertEqual(auxiliary["mu"].shape, (8, 16))
                self.assertEqual(auxiliary["prediction"].shape, (8, 42))
                if "body_height" in HEADS[variant]:
                    self.assertEqual(auxiliary["body_height"].shape, (8, 1))
                distribution, velocity = model(data["history"], data["obs"], data["command"])
                if variant == "fullest":
                    self.assertEqual(velocity.shape, (8, 3))
                    torch.testing.assert_close(model.estimate(data["history"]), velocity)
                else:
                    self.assertIsNone(velocity)
                    self.assertIsNone(model.estimate(data["history"]))
                action, log_prob, value, _, std = model.act(data)
                torch.testing.assert_close(log_prob, distribution.log_prob(action).sum(-1))
                torch.testing.assert_close(std, torch.full_like(std, .8))
                self.assertTrue((action.abs() > 1).any())
                self.assertEqual(value.shape, (8,))

    def test_policy_gradient_reaches_mu_and_all_actual_explicit_heads(self):
        data = observation()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant))
                distribution = model.distribution(data["history"], data["obs"], data["command"])
                (-distribution.log_prob(distribution.mean.detach() + .4).sum(-1).mean()).backward()
                for name in ("encoder", "actor", "mu_head", *(f"{head}_head" for head in HEADS[variant])):
                    self.assertGreater(gradient_size(getattr(model, name)), 0., name)
                for name in ("logvar_head", "decoder", "critic"):
                    self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))

    def test_reconstruction_gradient_reaches_mu_logvar_and_actual_explicit_heads(self):
        data = observation()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant)).train()
                auxiliary = model.auxiliary(data["history"], data["obs"], data["command"])
                F.mse_loss(auxiliary["prediction"], data["obs"]).backward()
                for name in ("encoder", "decoder", "mu_head", "logvar_head",
                             *(f"{head}_head" for head in HEADS[variant])):
                    self.assertGreater(gradient_size(getattr(model, name)), 0., name)
                self.assertTrue(all(p.grad is None for p in model.actor.parameters()))

    def test_decoder_sampling_does_not_randomize_actor_likelihood(self):
        data = observation()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant)).train()
                action = torch.randn(8, 12)
                before = model.distribution(data["history"], data["obs"], data["command"]).log_prob(action)
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"], data["command"])
                self.assertFalse(torch.equal(first["prediction"], second["prediction"]))
                after = model.distribution(data["history"], data["obs"], data["command"]).log_prob(action)
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                model.eval()
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"], data["command"])
                torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)

    def test_labels_do_not_leak_into_actor_and_current_obs_is_not_decoder_input(self):
        data = observation()
        changed = {name: value + 1000. if name in ("velocity", "heightmap", "body_height", "critic")
                   else value for name, value in data.items()}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant)).eval()
                torch.manual_seed(91)
                original = model.act(data)
                torch.manual_seed(91)
                modified = model.act(changed)
                for index in (0, 1, 3, 4):
                    torch.testing.assert_close(original[index], modified[index], rtol=0, atol=0)
                first = model.auxiliary(data["history"], data["obs"], data["command"])
                second = model.auxiliary(data["history"], data["obs"] + 1000., data["command"] - 1000.)
                torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)

    def test_auxiliary_coefficients_and_absence_of_removed_metrics(self):
        data = {name: torch.zeros_like(value) for name, value in observation().items()}
        fixed = {"velocity": torch.ones(8, 3), "heightmap": torch.full((8, 18), 3.),
                 "body_height": torch.full((8, 1), 4.), "prediction": torch.full((8, 42), 2.),
                 "mu": torch.full((8, 16), 2.), "logvar": torch.zeros(8, 16)}
        for variant, expected in (("fullest", 145.5), ("irrest", 140.), ("implicit", 108.)):
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant))
                learner = AblationPPO(model, config(variant))
                # 仅注入该变体实际输出，错误读取已移除的预测头会直接失败。
                actual = {name: value for name, value in fixed.items()
                          if name in ("prediction", "mu", "logvar", *HEADS[variant])}
                with patch.object(model, "auxiliary", return_value=actual):
                    loss, metrics = learner.auxiliary_loss(data, torch.arange(8), None)
                self.assertEqual(loss.item(), expected)
                self.assertEqual(metrics["latent_kl"].item(), 2.)
                self.assertEqual(metrics["prediction_loss"].item(), 4.)
                for name in ("velocity", "heightmap", "body_height"):
                    self.assertEqual(f"{name}_loss" in metrics, name in HEADS[variant])

    def test_real_ppo_without_absent_targets_and_exact_optimizer_restore(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = config(variant)
                model = AblationPolicy(cfg)
                learner = AblationPPO(model, cfg)
                batch = rollout(model)
                original_batch = copy.deepcopy(batch)
                metrics = learner.update(batch)
                self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in metrics.values()))
                self.assertEqual(metrics["gradient_steps"], 4.)
                for name in ("velocity", "heightmap", "body_height"):
                    self.assertEqual(f"{name}_loss" in metrics, name in HEADS[variant])
                self.assertEqual("velocity_rmse" in metrics, variant == "fullest")
                self.assert_tree_equal(batch, original_batch)
                clone = AblationPolicy(cfg)
                clone.load_state_dict(copy.deepcopy(model.state_dict()))
                resumed = AblationPPO(clone, cfg)
                resumed.load_state_dict(copy.deepcopy(learner.state_dict()))
                next_batch = rollout(model)
                torch.manual_seed(41)
                uninterrupted = learner.update(next_batch)
                torch.manual_seed(41)
                restored = resumed.update(next_batch)
                self.assert_tree_equal(model.state_dict(), clone.state_dict())
                self.assert_tree_equal(learner.state_dict(), resumed.state_dict())
                self.assertEqual(uninterrupted, restored)

    def test_real_auxiliary_updates_reduce_joint_loss(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = config(variant, epochs=1, minibatches=1, value_coef=0.,
                             entropy_coef=0., desired_kl=0.)
                model = AblationPolicy(cfg)
                learner = AblationPPO(model, cfg)
                batch = rollout(model)
                batch["advantages"].zero_()
                model.eval()
                with torch.no_grad():
                    before = learner.auxiliary_loss(batch, torch.arange(8), None)[0].item()
                model.train()
                for _ in range(5):
                    learner.update(batch)
                model.eval()
                with torch.no_grad():
                    after = learner.auxiliary_loss(batch, torch.arange(8), None)[0].item()
                self.assertLess(after, before)

    def test_height_targets_cannot_silently_broadcast(self):
        for variant in ("fullest", "irrest"):
            with self.subTest(variant=variant):
                model = AblationPolicy(config(variant))
                learner = AblationPPO(model, config(variant))
                batch = rollout(model)
                with self.assertRaisesRegex(ValueError, "Missing rollout fields"):
                    learner.update({name: value for name, value in batch.items() if name != "body_height"})
                batch["body_height"] = batch["body_height"].squeeze(-1)
                with self.assertRaisesRegex(ValueError, "body_height supervision"):
                    learner.update(batch)
                self.assertEqual(learner.updates, 0)
                self.assertFalse(learner.optimizer.state)


if __name__ == "__main__":
    unittest.main()
