"""真实动作模块及环境方法的 CPU 集成测试；不导入 Isaac 或启动仿真。"""
import ast
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from estnet.config import Config
from estnet.factory import config_for_variant
from estnet.history import History
from estnet.gait import command_with_phase, gait
from estnet.robot import (
    DEFAULT_JOINT_POS29,
    HIP_YAW_POLICY_COLUMNS,
    JOINT_NAMES29,
    LEG_JOINT_NAMES12,
    make_joint_targets,
)


def load_environment_methods():
    """执行冻结源中的原方法，而不是复制公式或仅匹配源码字符串。"""
    path = Path(__file__).resolve().parents[1] / "estnet/environment.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    methods = {}
    scope = {"torch": torch, "make_joint_targets": make_joint_targets,
             "gait": gait, "command_with_phase": command_with_phase, "__package__": "estnet"}
    for name in ("_pre_physics_step", "_apply_action", "_read_state", "_get_observations"):
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one actual environment method: {name}")
        module = ast.Module(body=[matches[0]], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        methods[name] = scope[name]
    return methods


class RobotSink:
    def __init__(self, data):
        self.data = data
        self.applied = []

    def set_joint_position_target(self, targets):
        self.applied.append(targets.clone())


class ActionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods = load_environment_methods()

    def setUp(self):
        # 两个 yaw 的 native 序号为7/12，不能误用策略列2/8作为物理关节序号。
        permutation = [(7 * i + 11) % 29 for i in range(29)]
        self.native_names = tuple(JOINT_NAMES29[i] for i in permutation)
        self.leg_ids = torch.tensor([self.native_names.index(name) for name in LEG_JOINT_NAMES12])
        self.yaw_ids = self.leg_ids[list(HIP_YAW_POLICY_COLUMNS)]
        self.upper_ids = torch.tensor([self.native_names.index(name) for name in JOINT_NAMES29[12:]])
        self.default = torch.tensor(DEFAULT_JOINT_POS29)[permutation].repeat(4, 1)
        # 人为很窄的soft阈值，使旧全腿clamp会明确失败；物理仿真并不在本测试中执行。
        self.soft = torch.stack((self.default - .03, self.default + .03), dim=-1)
        for name in ("left_knee_joint", "right_knee_joint"):
            self.soft[:, self.native_names.index(name), :] = torch.tensor([math.radians(3.5), 2.9])
        self.raw = torch.tensor([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [-2., 2., 5., -2., 4., -4., 2., -2., -5., -4., 6., -6.],
            [-200., 200., -200., 200., -200., 200., 200., -200., 200., -200., 200., -200.],
            [0., 0., .1, 0., 0., 0., 0., 0., -.1, 0., 0., 0.],
        ])

    def targets(self, soft=None):
        return make_joint_targets(self.raw, self.default, self.soft if soft is None else soft,
                                  self.leg_ids, hip_yaw_target_limit_rad=math.radians(20))

    def by_name(self, tensor, name):
        return tensor[:, self.native_names.index(name)]

    def test_real_native_mapping_only_yaw_caps_and_inputs_remain_intact(self):
        before = [value.clone() for value in (self.raw, self.default, self.soft, self.leg_ids)]
        targets = self.targets()
        self.assertEqual(self.yaw_ids.tolist(), [7, 12])
        torch.testing.assert_close(targets[:, self.upper_ids], self.default[:, self.upper_ids], rtol=0, atol=0)
        torch.testing.assert_close(targets[0], self.default[0], rtol=0, atol=0)
        cap = math.radians(20)
        torch.testing.assert_close(self.by_name(targets, "left_hip_yaw_joint"), torch.tensor([0., cap, -cap, .025]))
        torch.testing.assert_close(self.by_name(targets, "right_hip_yaw_joint"), torch.tensor([0., -cap, cap, -.025]))
        # 手算具体反例：±2必须形成±0.5rad偏移，不能被悄悄压成±1动作。
        self.assertAlmostEqual(targets[1, self.native_names.index("left_hip_pitch_joint")].item(), -.6, places=6)
        self.assertAlmostEqual(targets[1, self.native_names.index("left_hip_roll_joint")].item(), .5, places=6)
        for current, saved in zip((self.raw, self.default, self.soft, self.leg_ids), before):
            torch.testing.assert_close(current, saved, rtol=0, atol=0)

    def test_knee_targets_cross_soft_limits_and_soft_metadata_does_not_change_output(self):
        targets = self.targets()
        left = self.native_names.index("left_knee_joint")
        right = self.native_names.index("right_knee_joint")
        self.assertAlmostEqual(targets[1, left].item(), -.2, places=6)
        self.assertAlmostEqual(targets[1, right].item(), -.7, places=6)
        self.assertLess(targets[1, left].item(), self.soft[1, left, 0].item())
        self.assertLess(targets[1, right].item(), self.soft[1, right, 0].item())
        radically_different_soft = torch.stack((self.default - .001, self.default + .001), dim=-1)
        torch.testing.assert_close(self.targets(radically_different_soft), targets, rtol=0, atol=0)
        # ±200仅由raw±100保护截断，因此非yaw目标偏移仍为±25rad，不伪装成机械可达角度。
        self.assertAlmostEqual(targets[2, left].item(), 25.3, places=5)
        self.assertAlmostEqual(targets[2, right].item(), -24.7, places=5)

    def stub_env(self):
        count = len(self.raw)
        cfg = Config(num_envs=count)
        data = SimpleNamespace(default_joint_pos=self.default, soft_joint_pos_limits=self.soft,
                               joint_pos=self.default.clone(), joint_vel=torch.zeros(count, 29),
                               projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(count, 1),
                               root_ang_vel_b=torch.zeros(count, 3), root_lin_vel_b=torch.zeros(count, 3),
                               root_pos_w=torch.tensor([0., 0., .8]).repeat(count, 1),
                               root_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(count, 1),
                               body_pos_w=torch.zeros(count, 2, 3), body_lin_vel_w=torch.zeros(count, 2, 3),
                               applied_torque=torch.zeros(count, 29))
        env = SimpleNamespace(params=cfg, robot=RobotSink(data), leg_ids=self.leg_ids,
                              hip_yaw_policy_columns=list(HIP_YAW_POLICY_COLUMNS), hip_yaw_native_ids=self.yaw_ids,
                              previous_action=torch.full_like(self.raw, 999.), targets=self.default.clone(),
                              hip_yaw_target_clip_fraction=torch.zeros(count, 2),
                              contacts=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(count, 2, 3))),
                              phase_offset=torch.zeros(count), episode_length_buf=torch.zeros(count),
                              step_dt=cfg.step_dt, physical_command=torch.tensor([.4, 0., 0.]).repeat(count, 1),
                              sensor_feet=torch.tensor([0, 1]), foot_ids=torch.tensor([0, 1]),
                              scene=SimpleNamespace(env_origins=torch.zeros(count, 3)), mass=torch.full((count,), 35.),
                              previous_force=torch.zeros(count, 2, 3), previous_torque=torch.zeros(count, 12),
                              previous_joint_velocity=torch.zeros(count, 12))
        return env

    def test_actual_environment_step_apply_and_observation_share_action_contract(self):
        env = self.stub_env()
        sampled = self.raw.clone()
        original = sampled.clone()
        self.methods["_pre_physics_step"](env, sampled)
        self.methods["_apply_action"](env)
        obs, critic, _ = self.methods["_read_state"](env)
        torch.testing.assert_close(sampled, original, rtol=0, atol=0)
        self.assertEqual(len(env.robot.applied), 1)
        torch.testing.assert_close(env.robot.applied[0], self.targets(), rtol=0, atol=0)
        # 该帧下一次可见的previous_action记录宽松raw裁剪后的动作，而不是限幅后的关节角。
        expected_previous = original.clone()
        expected_previous[2] = torch.tensor([-100.,100.,-100.,100.,-100.,100.,100.,-100.,100.,-100.,100.,-100.])
        torch.testing.assert_close(env.previous_action, expected_previous, rtol=0, atol=0)
        torch.testing.assert_close(obs[:, -12:], expected_previous, rtol=0, atol=0)
        self.assertEqual(tuple(obs.shape), (4, 42))
        self.assertEqual(tuple(critic.shape), (4, 61))
        torch.testing.assert_close(critic[:, :42], obs, rtol=0, atol=0)
        torch.testing.assert_close(env.hip_yaw_target_clip_fraction, torch.tensor([[0.,0.],[1.,1.],[1.,1.],[0.,0.]]))
        torch.testing.assert_close(env.target_clip_fraction, torch.tensor([0.,2./12,2./12,0.]))
        # 原始PPO buffer在环境之后再改变，也不能倒过来修改已保存的previous_action/targets。
        saved_previous = env.previous_action.clone()
        saved_target = env.targets.clone()
        sampled.fill_(12345.)
        torch.testing.assert_close(env.previous_action, saved_previous, rtol=0, atol=0)
        torch.testing.assert_close(env.targets, saved_target, rtol=0, atol=0)

    def test_all_variant_observations_and_history_use_true_production_methods(self):
        for variant in ("estnet", "key1", "key2", "fullest", "irrest", "implicit"):
            with self.subTest(variant=variant):
                env = self.stub_env()
                env.params = replace(config_for_variant(variant), num_envs=4)
                env.history = History(4, 50, 42, "cpu")
                env._read_state = lambda: self.methods["_read_state"](env)
                env.robot.data.root_lin_vel_b[:] = torch.tensor([.3, .1, -.2])
                self.methods["_pre_physics_step"](env, self.raw)
                self.methods["_apply_action"](env)
                torch.testing.assert_close(env.robot.applied[0], self.targets())
                first = self.methods["_get_observations"](env)
                self.assertEqual(set(first), {"obs", "command", "critic", "history", *env.params.supervision_dims})
                self.assertEqual(first["critic"].shape, (4, env.params.critic_dim))
                # 出生帧填充缺失历史；随后VAE历史不能包含当前帧。
                env.robot.data.joint_pos += .2
                second = self.methods["_get_observations"](env)
                last_history = first["obs"] if variant != "estnet" else second["obs"]
                torch.testing.assert_close(second["history"][:, -1], last_history)
                torch.testing.assert_close(second["critic"][:, 49:52], env.robot.data.root_lin_vel_b)
                if "velocity" in second:
                    torch.testing.assert_close(second["velocity"], env.robot.data.root_lin_vel_b)
                if "body_height" in second:
                    torch.testing.assert_close(second["body_height"], torch.full((4, 1), .8))
                if "heightmap" in second:
                    torch.testing.assert_close(second["heightmap"], second["critic"][:, 53:71])
                # 重置单个环境后，历史不能继续含有上一回合动作。
                env.history.reset(torch.tensor([0]))
                third = self.methods["_get_observations"](env)
                torch.testing.assert_close(third["history"][0], third["obs"][0].expand(50, -1))

    def test_invalid_environment_action_does_not_advance_buffers(self):
        for bad in (torch.zeros(4, 11), torch.full((4, 12), float("nan"))):
            with self.subTest(shape=tuple(bad.shape), finite=bool(torch.isfinite(bad).all())):
                env = self.stub_env()
                old_previous, old_target = env.previous_action.clone(), env.targets.clone()
                with self.assertRaises(ValueError):
                    self.methods["_pre_physics_step"](env, bad)
                torch.testing.assert_close(env.previous_action, old_previous, rtol=0, atol=0)
                torch.testing.assert_close(env.targets, old_target, rtol=0, atol=0)

    def test_config_rejects_disabling_collision_or_reintroducing_global_soft_target_clip(self):
        cfg = Config()
        cfg.validate()
        self.assertIs(cfg.self_collisions, True)
        self.assertIs(cfg.soft_joint_target_clipping, False)
        for changed in ({"self_collisions":False}, {"self_collisions":1},
                        {"soft_joint_target_clipping":True}, {"soft_joint_target_clipping":0}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    replace(cfg, **changed).validate()


if __name__ == "__main__":
    unittest.main()
