"""六条真实网络的CPU接口/梯度测试；保留实际输入维度，只缩小隐藏层。"""
from dataclasses import replace
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from estnet.config import Config
from estnet.factory import VARIANTS, build_model, build_ppo, config_for_variant


EXPECTED = {
    # actor输入宽度 / decoder输入宽度 / 实际显式监督量。
    "estnet": (52, None, {"velocity": 3}),
    "key1": (68, 19, {"velocity": 3}),
    "key2": (86, 37, {"velocity": 3, "heightmap": 18}),
    "fullest": (87, 38, {"velocity": 3, "heightmap": 18, "body_height": 1}),
    "irrest": (66, 17, {"body_height": 1}),
    "implicit": (65, 16, {}),
}


def tiny_config(variant):
    return replace(config_for_variant(variant), encoder_hidden=(32, 16), actor_hidden=(24, 12),
                   critic_hidden=(24, 12), decoder_hidden=(16, 24), num_envs=8)


def inputs(cfg, count=8):
    return {"history": torch.randn(count, 50, 42), "obs": torch.randn(count, 42),
            "command": torch.randn(count, 7), "critic": torch.randn(count, cfg.critic_dim),
            "velocity": torch.randn(count, 3), "heightmap": torch.randn(count, 18),
            "body_height": torch.randn(count, 1), "base_heightmap": torch.randn(count, 81)}


def gradient_norm(module):
    return sum(float(parameter.grad.abs().sum()) for parameter in module.parameters() if parameter.grad is not None)


class VariantModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1701)

    def test_each_real_route_shapes_heads_and_raw_gaussian_contract(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                model = build_model(cfg)
                data = inputs(cfg)
                actor_width, decoder_width, labels = EXPECTED[variant]
                self.assertEqual(model.variant, variant)
                self.assertEqual(model.actor[0].in_features, actor_width)
                self.assertEqual(cfg.supervision_dims, labels)
                self.assertEqual(tuple(model.explicit_names), tuple(labels))
                self.assertEqual(model.has_velocity_estimate, "velocity" in labels)
                distribution, velocity = model(data["history"], data["obs"], data["command"])
                self.assertEqual(tuple(distribution.mean.shape), (8, 12))
                self.assertEqual(tuple(model.value(data["critic"]).shape), (8,))
                self.assertTrue(bool(torch.isfinite(distribution.mean).all()))
                self.assertTrue(bool(torch.isfinite(distribution.stddev).all()))
                if "velocity" in labels:
                    self.assertEqual(tuple(velocity.shape), (8, 3))
                else:
                    self.assertIsNone(velocity)
                    self.assertIsNone(model.estimate(data["history"]))
                    self.assertFalse(hasattr(model, "velocity_head"))
                if decoder_width is None:
                    self.assertFalse(hasattr(model, "encoder"))
                    self.assertFalse(hasattr(model, "decoder"))
                    self.assertFalse(hasattr(model, "mu_head"))
                else:
                    self.assertEqual(model.decoder[0].in_features, decoder_width)
                    aux = model.auxiliary(data["history"], data["obs"], data["command"])
                    self.assertEqual(set(aux), set(labels) | {"mu", "logvar", "prediction"})
                    self.assertEqual(tuple(aux["prediction"].shape), (8, 42))
                    self.assertEqual(tuple(aux["mu"].shape), (8, 16))
                    for name, dimension in labels.items():
                        self.assertEqual(tuple(aux[name].shape), (8, dimension))
                    for name in ("velocity", "heightmap", "body_height"):
                        self.assertEqual(hasattr(model, name + "_head"), name in labels)
                    self.assertFalse(hasattr(model, "base_heightmap_head"))
                # 将真实actor末层均值设为2，证明网络不会暗中变成tanh/±1策略。
                self.assertIsInstance(model.actor[-1], nn.Linear)
                with torch.no_grad():
                    model.actor[-1].weight.zero_()
                    model.actor[-1].bias.fill_(2.)
                action, log_prob, value, mean, std = model.act(data)
                torch.testing.assert_close(mean, torch.full((8, 12), 2.), rtol=0, atol=0)
                check = model.distribution(data["history"], data["obs"], data["command"])
                torch.testing.assert_close(log_prob, check.log_prob(action).sum(-1))
                self.assertEqual(tuple(value.shape), (8,))
                self.assertEqual(tuple(std.shape), (8, 12))

    def test_policy_gradient_reaches_history_estimates_and_mu_without_decoder_path(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                model = build_model(cfg)
                data = inputs(cfg)
                distribution = model.distribution(data["history"], data["obs"], data["command"])
                fixed_actions = distribution.mean.detach() + torch.linspace(.2, .6, 12)
                loss = -distribution.log_prob(fixed_actions).sum(-1).mean()
                loss.backward()
                self.assertGreater(gradient_norm(model.actor), 0.)
                self.assertGreater(gradient_norm(model.estimator if variant == "estnet" else model.encoder), 0.)
                if variant != "estnet":
                    self.assertGreater(gradient_norm(model.mu_head), 0.)
                    for name in model.explicit_names:
                        self.assertGreater(gradient_norm(getattr(model, name + "_head")), 0.)
                    self.assertEqual(gradient_norm(model.logvar_head), 0.)
                    self.assertEqual(gradient_norm(model.decoder), 0.)
                self.assertEqual(gradient_norm(model.critic), 0.)

    def test_reconstruction_backpropagates_through_sample_mu_logvar_and_explicit_heads(self):
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                model = build_model(cfg).train()
                data = inputs(cfg)
                aux = model.auxiliary(data["history"], data["obs"], data["command"])
                F.mse_loss(aux["prediction"], data["obs"]).backward()
                for module in (model.encoder, model.mu_head, model.logvar_head, model.decoder):
                    self.assertGreater(gradient_norm(module), 0.)
                for name in model.explicit_names:
                    self.assertGreater(gradient_norm(getattr(model, name + "_head")), 0.)
                self.assertEqual(gradient_norm(model.actor), 0.)
                self.assertEqual(gradient_norm(model.critic), 0.)

    def test_actor_cannot_read_ground_truth_or_resample_latent_for_likelihood(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                model = build_model(cfg).train()
                data = inputs(cfg)
                changed = {**data}
                for key in ("critic", "velocity", "heightmap", "body_height", "base_heightmap"):
                    changed[key] = torch.full_like(data[key], 12345.)
                torch.manual_seed(900)
                first = model.act(data)
                torch.manual_seed(900)
                second = model.act(changed)
                for index in (0, 1, 3, 4):
                    torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)
                before = model.distribution(data["history"], data["obs"], data["command"])
                if variant != "estnet":
                    # decoder采样可以消耗RNG，但同一状态actor仍使用确定mu。
                    model.auxiliary(data["history"], data["obs"], data["command"])
                after = model.distribution(data["history"], data["obs"], data["command"])
                torch.testing.assert_close(before.log_prob(first[0]), after.log_prob(first[0]), rtol=0, atol=0)

    def test_decoder_has_no_current_observation_or_command_shortcut(self):
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                model = build_model(cfg).train()
                data = inputs(cfg)
                torch.manual_seed(77)
                a = model.auxiliary(data["history"], data["obs"], data["command"])
                torch.manual_seed(77)
                b = model.auxiliary(data["history"], data["obs"] + 100, data["command"] - 100)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_config_schema_labels_and_implementation_choices_cannot_cross_routes(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = config_for_variant(variant)
                self.assertEqual(cfg.schema, f"g1-{variant}-ppo-clip-flat-isaac51-v1")
                self.assertEqual(cfg.learning_rate, 5e-4)
                self.assertEqual(cfg.learning_rate_schedule, "fixed")
                self.assertIs(cfg.self_collisions, True)
                self.assertIs(cfg.soft_joint_target_clipping, False)
                self.assertEqual(cfg.vae_beta, 50.)
                self.assertNotIn("kl_hard_limit", cfg.to_dict())
                self.assertNotIn("desired_kl", cfg.to_dict())
                # checkpoint JSON会把hidden tuple变为list，形状/约定仍可严格校验。
                reloaded = Config(**json.loads(json.dumps(cfg.to_dict())))
                reloaded.validate()
                self.assertEqual(reloaded.supervision_dims, EXPECTED[variant][2])
                for changed in ({"schema":"g1-key1-flat-v1"}, {"actor_latent_mode":"sample"},
                                {"reconstruction_target":"next_obs"}, {"vae_kl_reduction":"sum"},
                                {"learning_rate_schedule":"adaptive"}, {"self_collisions":False},
                                {"soft_joint_target_clipping":True}):
                    with self.subTest(changed=changed):
                        with self.assertRaises(ValueError):
                            replace(cfg, **changed).validate()
        self.assertEqual(config_for_variant("IllEst").variant, "irrest")
        for bad in ("unknown", "", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    config_for_variant(bad)
        with self.assertRaisesRegex(ValueError, "different variants"):
            build_ppo(build_model(tiny_config("key1")), tiny_config("key2"))


if __name__ == "__main__":
    unittest.main()
