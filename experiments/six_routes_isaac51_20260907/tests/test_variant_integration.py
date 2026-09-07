"""五组真实tiny模型的采样、当前标签、完整Adam续训与协议隔离测试。"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from estnet.factory import build_model, build_ppo, config_for_variant
from estnet.history import History
from estnet.preflight import ASSET_HASHES
from estnet.resume import load_training_checkpoint, restore_training_state
from estnet.run import collect_rollout, load_evaluation_checkpoint, save_checkpoint


VARIANTS = ("key1", "key2", "fullest", "irrest", "implicit")
EXPECTED_SUPERVISION = {
    "key1": {"velocity": 3},
    "key2": {"velocity": 3, "heightmap": 18},
    "fullest": {"velocity": 3, "heightmap": 18, "body_height": 1},
    "irrest": {"body_height": 1},
    "implicit": {},
}
PROTOCOL_FIELDS = (
    "variant", "hip_yaw_target_limit_rad", "learning_rate_schedule", "kl_chunk_size",
    "gradient_diagnostics", "soft_joint_target_clipping", "raw_action_clip", "self_collisions",
    "reconstruction_target", "actor_latent_mode", "vae_kl_reduction", "latent_dim",
    "heightmap_dim", "base_heightmap_dim", "decoder_hidden", "heightmap_coef",
    "body_height_coef", "prediction_coef", "vae_beta", "simulation_protocol",
)


def tiny_config(variant):
    """只缩小隐藏宽度和采样数；保留真实观测、历史、latent、critic及4×4更新。"""
    return replace(config_for_variant(variant), num_envs=4, horizon=2,
                   encoder_hidden=(16, 8), actor_hidden=(16,), critic_hidden=(16,),
                   decoder_hidden=(16,))


def asset_record():
    """只提供已有四份资产签名用于CPU协议验证；不读取或加载USD。"""
    return {"ready": True, "files": [{"sha256": value} for value in ASSET_HASHES.values()]}


class ReusingEnvironment:
    """每步原地覆盖同一批张量，以暴露collector忘记clone当前标签的错误。"""

    def __init__(self, cfg):
        self.num_envs, self.device = cfg.num_envs, "cpu"
        self.initial = {
            "obs": torch.full((cfg.num_envs, 42), .25),
            "history": torch.full((cfg.num_envs, 50, 42), -.25),
            "command": torch.full((cfg.num_envs, 7), .125),
            "critic": torch.full((cfg.num_envs, 152), .5),
            **{name: torch.full((cfg.num_envs, width), float(index + 1))
               for index, (name, width) in enumerate(cfg.supervision_dims.items())},
        }
        # 环境编号也各自不同，不能只通过所有环境都相同的样本。
        for value in self.initial.values():
            shape = (cfg.num_envs,) + (1,) * (value.ndim - 1)
            value.add_(torch.arange(cfg.num_envs).reshape(shape) * .01)
        self.current = {name: value.clone() for name, value in self.initial.items()}

    def step(self, _action):
        """下一帧刻意与当前帧相差1，真实奖励无需模拟动力学。"""
        for value in self.current.values():
            value.add_(1.)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.linspace(.1, .4, self.num_envs)
        return self.current, reward, zeros, zeros.clone(), {
            "final_critic": self.current["critic"].clone(),
            "metrics": {"marker": reward.clone()},
            "reward_terms": {"synthetic": reward.clone()},
        }


class VariantIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """限制纯CPU小矩阵线程，防止系统默认线程数拖慢测试。"""
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20260907)
        # CPU路径一旦意外初始化CUDA即失败，绝不占用真实训练设备。
        self.cuda_init = mock.patch("torch.cuda.init", side_effect=AssertionError("CPU integration test initialized CUDA"))
        self.cuda_init.start()
        self.addCleanup(self.cuda_init.stop)

    def assert_tree_equal(self, actual, expected):
        """递归核对全部参数组与Adam矩，不仅比较一个loss或第一层权重。"""
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assert_tree_equal(got, want)
        else:
            self.assertEqual(actual, expected)

    def assert_complete_adam(self, model, learner, updates):
        """每个实际参数都必须有完整Adam状态，step为更新轮数×4×4。"""
        self.assertEqual(learner.updates, updates)
        self.assertEqual(learner.total_optimizer_steps, updates * 16)
        self.assertEqual(learner.learning_rate, 5e-4)
        parameters = list(model.parameters())
        self.assertEqual(len(learner.optimizer.state), len(parameters))
        for parameter in parameters:
            moment = learner.optimizer.state[parameter]
            self.assertEqual(float(moment["step"]), updates * 16)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(moment[key].shape, parameter.shape)
                self.assertTrue(torch.isfinite(moment[key]).all())

    def test_current_labels_and_history_survive_inplace_steps_and_reconstruction_targets_current(self):
        """五组都重建obs_t；放入明显不同next_obs作为可被误用的反例。"""
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tiny_config(variant)
                self.assertEqual(cfg.supervision_dims, EXPECTED_SUPERVISION[variant])
                model = build_model(cfg)
                env = ReusingEnvironment(cfg)
                _, batch, summary = collect_rollout(env, model, env.current, cfg)
                for name, initial in env.initial.items():
                    expected = torch.cat((initial, initial + 1.), dim=0)
                    torch.testing.assert_close(batch[name], expected, rtol=0., atol=0.)
                    # 收集结束后环境再次覆盖，也不得污染已经返回的训练批。
                    env.current[name].fill_(999.)
                    torch.testing.assert_close(batch[name], expected, rtol=0., atol=0.)
                self.assertEqual(set(EXPECTED_SUPERVISION[variant]),
                    set(batch) & {"velocity", "heightmap", "body_height"})
                self.assertAlmostEqual(summary["reward/total"], .25, places=6)
                with torch.no_grad():
                    distribution = model.distribution(batch["history"], batch["obs"], batch["command"])
                    torch.testing.assert_close(distribution.log_prob(batch["action"]).sum(-1),
                                               batch["old_log_prob"], rtol=1e-5, atol=1e-6)
                learner = build_ppo(model, cfg)
                model.eval()  # decoder用mu，避免两次重建比较受到latent重采样干扰。
                batch["next_obs"] = batch["obs"] + 20.
                indices = torch.arange(batch["obs"].shape[0])
                auxiliary = model.auxiliary(batch["history"], batch["obs"], batch["command"])
                _, measured = learner.auxiliary_loss(batch, indices, None)
                current_loss = F.mse_loss(auxiliary["prediction"], batch["obs"])
                wrong_next_loss = F.mse_loss(auxiliary["prediction"], batch["next_obs"])
                torch.testing.assert_close(measured["prediction_loss"], current_loss, rtol=0., atol=0.)
                self.assertFalse(torch.isclose(current_loss, wrong_next_loss))
                model.train()
                metrics = learner.update(batch)
                self.assert_complete_adam(model, learner, 1)
                self.assertEqual("velocity_rmse" in metrics, "velocity" in cfg.supervision_dims)
                self.assertEqual("heightmap_loss" in metrics, "heightmap" in cfg.supervision_dims)
                self.assertEqual("body_height_loss" in metrics, "body_height" in cfg.supervision_dims)

    def test_trained_save_load_restore_and_next_update_match_complete_learning_state(self):
        """真实联合PPO更新后恢复，并在相同batch/RNG下逐张量复现下一次优化。"""
        with tempfile.TemporaryDirectory() as temporary:
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    cfg = tiny_config(variant)
                    model = build_model(cfg)
                    learner = build_ppo(model, cfg)
                    env = ReusingEnvironment(cfg)
                    _, batch, _ = collect_rollout(env, model, env.current, cfg)
                    learner.update(batch)
                    self.assert_complete_adam(model, learner, 1)
                    path = Path(temporary) / f"{variant}-00001.pt"
                    save_checkpoint(path, model, learner, cfg, 1, asset_record())
                    evaluated, eval_cfg = load_evaluation_checkpoint(path, asset_record())
                    loaded, resume_cfg = load_training_checkpoint(path, asset_record())
                    self.assertNotIn("optimizer", evaluated)
                    self.assertEqual(eval_cfg.to_dict(), cfg.to_dict())
                    self.assertEqual(resume_cfg.to_dict(), cfg.to_dict())
                    self.assertEqual(loaded["schema"], f"g1-{variant}-ppo-clip-flat-isaac51-v1")
                    restored_model = build_model(resume_cfg)
                    restored_learner = build_ppo(restored_model, resume_cfg)
                    restored = restore_training_state(restored_model, restored_learner, loaded)
                    self.assertFalse(restored["exact_resume"])
                    self.assertFalse(restored["history_restored"])
                    self.assertTrue(restored["cpu_rng_restored"])
                    self.assert_complete_adam(restored_model, restored_learner, 1)
                    self.assert_tree_equal(restored_model.state_dict(), model.state_dict())
                    self.assert_tree_equal(restored_learner.state_dict(), learner.state_dict())
                    # 环境未恢复；这里只验证同一已采集batch与同一随机流的优化等价。
                    _, next_batch, _ = collect_rollout(env, model, env.current, cfg)
                    before_update_rng = torch.get_rng_state().clone()
                    expected_metrics = learner.update(next_batch)
                    torch.set_rng_state(before_update_rng)
                    actual_metrics = restored_learner.update(next_batch)
                    self.assert_complete_adam(restored_model, restored_learner, 2)
                    self.assert_tree_equal(restored_model.state_dict(), model.state_dict())
                    self.assert_tree_equal(restored_learner.state_dict(), learner.state_dict())
                    self.assert_tree_equal(actual_metrics, expected_metrics)
                    final_path = path.with_name(f"{variant}-00002.pt")
                    save_checkpoint(final_path, restored_model, restored_learner, resume_cfg, 2, asset_record())
                    final, _ = load_training_checkpoint(final_path, asset_record())
                    self.assertEqual(final["optimizer"]["total_optimizer_steps"], 32)

    def test_checkpoint_requires_explicit_protocol_and_rejects_cross_variant(self):
        """旧schema/缺字段/仅改标签冒充其它模型，评估与训练加载都应拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            for variant_index, variant in enumerate(VARIANTS):
                with self.subTest(variant=variant):
                    cfg = tiny_config(variant)
                    model = build_model(cfg)
                    learner = build_ppo(model, cfg)
                    save_checkpoint(path, model, learner, cfg, 0, asset_record())
                    valid = torch.load(path, map_location="cpu", weights_only=True)
                    for field in PROTOCOL_FIELDS:
                        with self.subTest(missing_field=field):
                            bad = copy.deepcopy(valid)
                            del bad["config"][field]
                            torch.save(bad, path)
                            for loader in (load_evaluation_checkpoint, load_training_checkpoint):
                                with self.assertRaises(ValueError):
                                    loader(path, asset_record())
                    other = VARIANTS[(variant_index + 1) % len(VARIANTS)]
                    changed = tiny_config(other)
                    cases = [
                        dict(valid, schema=f"g1-{variant}-flat-v1"),
                        dict(valid, schema=changed.schema),
                        dict(valid, schema=changed.schema, config=changed.to_dict()),
                    ]
                    for bad in cases:
                        torch.save(bad, path)
                        for loader in (load_evaluation_checkpoint, load_training_checkpoint):
                            with self.assertRaises(ValueError):
                                loader(path, asset_record())
                    # 同变体也不能拿另一动作或辅助损失配置直接恢复训练。
                    mismatched_cfg = replace(cfg, hip_yaw_target_limit_rad=cfg.hip_yaw_target_limit_rad / 2.)
                    mismatched_model = build_model(mismatched_cfg)
                    mismatched_learner = build_ppo(mismatched_model, mismatched_cfg)
                    with self.assertRaises(ValueError):
                        restore_training_state(mismatched_model, mismatched_learner, valid)

    def test_history_reset_fills_only_new_episode_and_excludes_current(self):
        """真实History对象区分旧末帧、当前帧和reset出生帧。"""
        history = History(2, 3, 1, "cpu")
        history.append(torch.tensor([[1.], [10.]]), exclude_current=True)
        history.append(torch.tensor([[2.], [20.]]), exclude_current=True)
        past = history.append(torch.tensor([[3.], [30.]]), exclude_current=True)
        torch.testing.assert_close(past[:, :, 0], torch.tensor([[1., 1., 2.], [10., 10., 20.]]))
        history.reset(torch.tensor([0]))
        reset = history.append(torch.tensor([[99.], [40.]]), exclude_current=True)
        torch.testing.assert_close(reset[:, :, 0], torch.tensor([[99., 99., 99.], [10., 20., 30.]]))
        torch.testing.assert_close(past[0, :, 0], torch.tensor([1., 1., 2.]))


if __name__ == "__main__":
    unittest.main()
