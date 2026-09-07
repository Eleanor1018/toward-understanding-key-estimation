"""纯CPU回归：测试真实动作映射、runner收集和检查点，不初始化Isaac/CUDA。"""
import ast
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import torch

from estnet.config import Config
from estnet.networks import EstNet
from estnet.ppo import PPO
from estnet.resume import load_training_checkpoint
from estnet.robot import HIP_YAW_POLICY_COLUMNS, LEG_JOINT_NAMES12, make_joint_targets
from estnet.run import collect_rollout, load_evaluation_checkpoint, save_checkpoint


def actual_environment_method(name):
    """只抽取真实环境方法，避免为两个纯tensor方法导入Isaac。"""
    tree = ast.parse((SOURCE / "estnet/environment.py").read_text(encoding="utf-8"))
    method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "make_joint_targets": make_joint_targets}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "environment.py", "exec"), namespace)
    return namespace[name]


PRE_STEP = actual_environment_method("_pre_physics_step")
YAW_METRICS = actual_environment_method("_hip_yaw_metrics")


def inputs(batch=2):
    """故意打乱原生顺序；策略yaw列不是原生yaw ID。"""
    ids = torch.tensor([18, 0, 23, 8, 2, 26, 15, 3, 6, 10, 1, 20])
    default = torch.arange(29, dtype=torch.float64).repeat(batch, 1) * .001
    bounds = torch.empty(batch, 29, 2, dtype=torch.float64)
    bounds[..., 0], bounds[..., 1] = -2.5, 2.5
    actions = torch.zeros(batch, 12, dtype=torch.float64)
    return actions, default, bounds, ids


def tiny_config(**kwargs):
    """缩小隐藏层提高回归速度，保留42/7/61/12及50帧语义。"""
    return replace(Config(), num_envs=2, horizon=2, epochs=1, minibatches=2,
                   encoder_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,), **kwargs)


class FakeVectorEnv:
    """用真实pre-step检查动作边界；模拟返回值不代表物理步行。"""
    def __init__(self, cfg):
        self.params = cfg
        _, default, limits, self.leg_ids = inputs(cfg.num_envs)
        default, limits = default.float(), limits.float()
        self.robot = SimpleNamespace(data=SimpleNamespace(
            default_joint_pos=default, soft_joint_pos_limits=limits, joint_pos=default.clone()))
        self.hip_yaw_policy_columns = list(HIP_YAW_POLICY_COLUMNS)
        self.hip_yaw_native_ids = self.leg_ids[self.hip_yaw_policy_columns]
        self.previous_action = torch.zeros(cfg.num_envs, 12)
        self.target_clip_fraction = torch.zeros(cfg.num_envs)
        self.hip_yaw_target_clip_fraction = torch.zeros(cfg.num_envs, 2)
        self.received = []
        self.target_history = []
        self.observation = {
            "obs": torch.zeros(cfg.num_envs, 42),
            "history": torch.zeros(cfg.num_envs, 50, 42),
            "command": torch.zeros(cfg.num_envs, 7),
            "critic": torch.zeros(cfg.num_envs, 61),
            "velocity": torch.zeros(cfg.num_envs, 3),
        }

    def step(self, actions):
        before = actions.clone()
        PRE_STEP(self, actions)
        torch.testing.assert_close(actions, before, rtol=0, atol=0)
        self.received.append(before)
        self.target_history.append(self.targets.clone())
        return (self.observation, torch.arange(self.params.num_envs).float() + .1,
                torch.zeros(self.params.num_envs, dtype=torch.bool),
                torch.zeros(self.params.num_envs, dtype=torch.bool),
                {"final_critic": self.observation["critic"],
                 "metrics": YAW_METRICS(self),
                 "reward_terms": {"example": torch.zeros(self.params.num_envs)}})


class HipYawTargetTests(unittest.TestCase):
    def test_named_policy_columns_and_permuted_native_mapping(self):
        """错误地把策略列2/8当原生ID会令本测试失败。"""
        actions, default, limits, ids = inputs()
        yaw = ids[list(HIP_YAW_POLICY_COLUMNS)]
        self.assertEqual(tuple(LEG_JOINT_NAMES12[c] for c in HIP_YAW_POLICY_COLUMNS),
                         ("left_hip_yaw_joint", "right_hip_yaw_joint"))
        actions[:] = torch.linspace(-1, 1, 12)
        actions[:, list(HIP_YAW_POLICY_COLUMNS)] = torch.tensor([120., -120.], dtype=actions.dtype)
        before = [x.clone() for x in (actions, default, limits, ids)]
        limit = math.radians(20)
        targets = make_joint_targets(actions, default, limits, ids, hip_yaw_target_limit_rad=limit)
        torch.testing.assert_close(targets[:, yaw] - default[:, yaw],
                                   torch.tensor([[limit, -limit]]).to(actions).expand(2, 2),
                                   atol=1e-8, rtol=1e-7)
        # 所有非yaw关节保持旧公式，包括17个上身目标。
        original = default.clone()
        original[:, ids] += actions.clamp(-100, 100) * .25
        original = torch.maximum(torch.minimum(original, limits[..., 1]), limits[..., 0])
        other = [i for i in range(29) if i not in yaw.tolist()]
        torch.testing.assert_close(targets[:, other], original[:, other], rtol=0, atol=0)
        for actual, saved in zip((actions, default, limits, ids), before):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)

    def test_target_range_intersects_tighter_soft_limits(self):
        """两个限制必须同时满足，不能由yaw裁剪重新越过软限位。"""
        actions, default, limits, ids = inputs()
        yaw = ids[list(HIP_YAW_POLICY_COLUMNS)]
        limits[:, yaw[0], 1] = default[:, yaw[0]] + .12
        limits[:, yaw[1], 0] = default[:, yaw[1]] - .08
        actions[:, list(HIP_YAW_POLICY_COLUMNS)] = torch.tensor([10., -10.], dtype=actions.dtype)
        targets = make_joint_targets(actions, default, limits, ids, hip_yaw_target_limit_rad=.2)
        torch.testing.assert_close(targets[:, yaw] - default[:, yaw],
                                   torch.tensor([[.12, -.08]], dtype=actions.dtype).expand(2, 2))
        self.assertTrue((targets >= limits[..., 0]).all())
        self.assertTrue((targets <= limits[..., 1]).all())

    def test_incompatible_interval_and_bad_caps_rejected(self):
        """空交集和非法配置必须显式失败，不能得到貌似有效的目标。"""
        actions, default, limits, ids = inputs()
        for value in (0, -1, math.nan, math.inf, True, "0.2", 4):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(Config(), hip_yaw_target_limit_rad=value).validate()
                with self.assertRaises(ValueError):
                    make_joint_targets(actions, default, limits, ids, hip_yaw_target_limit_rad=value)
        limits[:, ids[HIP_YAW_POLICY_COLUMNS[0]], 0] = 1.
        with self.assertRaisesRegex(ValueError, "no intersection"):
            make_joint_targets(actions, default, limits, ids, hip_yaw_target_limit_rad=.2)
        with self.assertRaises(TypeError):
            make_joint_targets(actions, default, limits, ids)

    def test_actual_pre_step_preserves_raw_history_and_clip_metrics(self):
        """强制raw超过100，分别检查PPO输入、历史和物理目标三种量。"""
        env = FakeVectorEnv(tiny_config())
        raw = torch.zeros(2, 12)
        raw[:, HIP_YAW_POLICY_COLUMNS[0]] = 120
        raw[:, HIP_YAW_POLICY_COLUMNS[1]] = -8
        before = raw.clone()
        PRE_STEP(env, raw)
        torch.testing.assert_close(raw, before, rtol=0, atol=0)
        torch.testing.assert_close(env.previous_action, before.clamp(-100, 100), rtol=0, atol=0)
        torch.testing.assert_close(env.hip_yaw_target_clip_fraction, torch.ones(2, 2))
        torch.testing.assert_close(env.target_clip_fraction, torch.full((2,), 2 / 12))
        env.robot.data.joint_pos[:, env.hip_yaw_native_ids[0]] += .4
        metrics = YAW_METRICS(env)
        torch.testing.assert_close(metrics["hip_yaw_actual_outside_target_fraction"], torch.full((2,), .5))
        torch.testing.assert_close(metrics["hip_yaw_abs_offset_max_rad"], torch.full((2,), .4))

    def test_real_rollout_keeps_raw_log_probability_aligned(self):
        """在真实runner中env.step发生在保存action之前，确保限幅未污染旧logprob。"""
        torch.manual_seed(24)
        cfg = tiny_config()
        model = EstNet(cfg)
        with torch.no_grad():
            model.actor[-1].weight.zero_()
            model.actor[-1].bias.zero_()
            model.actor[-1].bias[HIP_YAW_POLICY_COLUMNS[0]] = 120.
            model.actor[-1].bias[HIP_YAW_POLICY_COLUMNS[1]] = -8.
        env = FakeVectorEnv(cfg)
        _, batch, summary = collect_rollout(env, model, env.observation, cfg)
        torch.testing.assert_close(batch["action"], torch.stack(env.received).flatten(0, 1), rtol=0, atol=0)
        self.assertTrue((batch["action"][:, HIP_YAW_POLICY_COLUMNS[0]] > 100).all())
        distribution = model.distribution(batch["history"], batch["obs"], batch["command"])
        reevaluated = distribution.log_prob(batch["action"]).sum(-1)
        torch.testing.assert_close(reevaluated, batch["old_log_prob"], rtol=0, atol=0)
        torch.testing.assert_close((reevaluated - batch["old_log_prob"]).exp(), torch.ones(4))
        for target in env.target_history:
            offset = target[:, env.hip_yaw_native_ids] - env.robot.data.default_joint_pos[:, env.hip_yaw_native_ids]
            self.assertTrue((offset.abs() <= cfg.hip_yaw_target_limit_rad + 1e-7).all())
        self.assertEqual(summary["env/hip_yaw_target_clip_fraction"], 1.)
        # 跑一次真实PPO/Adam更新，检查极端原始动作仍可通过合法rollout更新。
        ppo = PPO(model, cfg)
        metrics = ppo.update(batch)
        self.assertEqual(ppo.updates, 1)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.cuda.is_initialized())


class CheckpointContractTests(unittest.TestCase):
    def setUp(self):
        """使用真实网络/优化器检查点，只用模拟资产SHA避免依赖USD。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="hipyaw-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "model.pt"
        self.cfg = tiny_config(hip_yaw_target_limit_rad=.2)
        self.asset = {"ready": True, "files": [{"sha256": str(i) * 64} for i in range(4)]}
        model = EstNet(self.cfg)
        save_checkpoint(self.path, model, PPO(model, self.cfg), self.cfg, 0, self.asset)

    def test_train_and_evaluate_preserve_saved_cap(self):
        """非默认值必须按检查点恢复，不能被新进程默认20度覆盖。"""
        self.assertEqual(Config().hip_yaw_target_limit_rad, math.radians(20))
        for loader in (load_evaluation_checkpoint, load_training_checkpoint):
            with self.subTest(loader=loader.__name__):
                payload, cfg = loader(self.path, self.asset)
                self.assertEqual(cfg.hip_yaw_target_limit_rad, .2)
                self.assertEqual(payload["config"]["hip_yaw_target_limit_rad"], .2)
                self.assertEqual(cfg.critic_dim, 61)
                self.assertEqual(cfg.history_steps, 50)
        self.assertFalse(torch.cuda.is_initialized())

    def test_old_or_missing_field_checkpoint_refused_by_both_loaders(self):
        """即使伪造新外层schema，缺字段的旧配置也不能静默启用新映射。"""
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        variants = []
        old = copy.deepcopy(saved)
        old["schema"] = old["config"]["schema"] = "g1-estnet-flat-v1"
        del old["config"]["hip_yaw_target_limit_rad"]
        variants.append(old)
        missing = copy.deepcopy(saved)
        del missing["config"]["hip_yaw_target_limit_rad"]
        variants.append(missing)
        nested = copy.deepcopy(saved)
        nested["config"]["schema"] = "g1-estnet-flat-v1"
        variants.append(nested)
        invalid = copy.deepcopy(saved)
        invalid["config"]["hip_yaw_target_limit_rad"] = 0.
        variants.append(invalid)
        for index, payload in enumerate(variants):
            torch.save(payload, self.path)
            for loader in (load_evaluation_checkpoint, load_training_checkpoint):
                with self.subTest(case=index, loader=loader.__name__):
                    with self.assertRaises(ValueError):
                        loader(self.path, self.asset)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
