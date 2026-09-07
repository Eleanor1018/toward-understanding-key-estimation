"""纯CPU验证真实奖励、前后时序拷贝、记录上限及离线CLI；不导入Isaac。"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from estnet.config import Config
from estnet.reward_diagnostics import RewardTraceRecorder, analyze_trace, verify_trace, kernel_inputs
from estnet.rewards import reward_terms, gaussian, cauchy
from estnet.gait import gait
from estnet.robot import HIP_YAW_POLICY_COLUMNS, make_joint_targets


def fixture(n=4):
    """有接触变化、非零功率和零速度的确定性输入；仅用于软件测试。"""
    mass = torch.full((n,), 33.)
    state = {"vel": torch.zeros(n,3), "ang_vel": torch.zeros(n,3), "command": torch.zeros(n,7),
        "up": torch.ones(n), "height": torch.full((n,), .78), "mass": mass,
        "terminated": torch.zeros(n,dtype=torch.bool), "stance": torch.tensor([[1.,0.],[0.,1.],[.5,.5],[1.,0.]])[:n],
        "foot_vel": torch.zeros(n,2,3), "foot_force": torch.zeros(n,2,3),
        "prev_foot_force": torch.zeros(n,2,3), "torque": torch.full((n,12), 10.),
        "prev_torque": torch.full((n,12), 8.), "joint_vel": torch.full((n,12), .3),
        "prev_joint_vel": torch.full((n,12), .1)}
    state["vel"][:,0] = torch.tensor([0.,.4,.2,-.1])[:n]
    state["command"][:,0] = .4
    state["foot_force"][:,:,2] = state["stance"]*mass[:,None]*9.81
    state["foot_vel"][:,1,0] = .5
    obs = {"obs": torch.arange(n*42).float().reshape(n,42), "history": torch.ones(n,50,42),
           "command": state["command"].clone(), "critic": torch.zeros(n,61), "velocity": state["vel"].clone()}
    action = torch.arange(n*12).float().reshape(n,12)
    return obs, action, state


def complete(recorder, *, offset=0, include_geometry=True):
    """按与runner相同的三个生命周期点记录一帧。"""
    obs, action, state = fixture()
    state["height"] += offset
    recorder.capture_pre(obs, action, action+.1, torch.ones_like(action),
                         phase_offset=torch.zeros(4), episode_length=torch.ones(4,dtype=torch.long))
    fields = {"total_reward": torch.stack(list(reward_terms(state,Config()).values())).sum(0)}
    if include_geometry:
        fields.update(foot_height=torch.tensor([[.02,.08],[.08,.02],[.04,.04],[.02,.02]]),
                      foot_contact=state["stance"]>.5)
    recorder.capture_reward(state, reward_terms(state,Config()), **fields)
    recorder.capture_post(terminated=state["terminated"], truncated=torch.zeros(4,dtype=torch.bool),
                          reset_mask=state["terminated"])


class RewardRecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="reward-diagnostic-cpu-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "trace"

    def test_real_reward_verification_all_terms_and_source_inputs(self):
        """实际reward_terms逐项一致，保存的kernel输入也确实重现每个核。"""
        recorder = RewardTraceRecorder(Config(), num_envs=2, max_steps=2, output_dir=self.directory)
        complete(recorder)
        complete(recorder,offset=.02)
        summary = recorder.flush()
        self.assertEqual(summary["steps"],2)
        self.assertEqual(summary["stop_reason"],"step_cap")
        self.assertEqual(verify_trace(self.directory)["status"],"passed")
        with np.load(self.directory/"reward_trace.npz") as data:
            state = {key.removeprefix("reward/state/"):torch.from_numpy(data[key]).flatten(0,1)
                     for key in data.files if key.startswith("reward/state/")}
            values = kernel_inputs(state,Config())
            terms = reward_terms(state,Config())
            parameters = {"linear_velocity":(.1,.5),"yaw_velocity":(.1,.5),"upright":(.1,.1),"height":(.2,.05)}
            for name,(alpha,width) in parameters.items():
                torch.testing.assert_close(gaussian(values[name],alpha,width),terms[name])
            for name,beta,width in (("stance_velocity",1,.25),("swing_force",1,.1),("impact",3,.2),
                ("torque_smoothness",2,160.),("joint_velocity_smoothness",1,8.),("cost_of_transport",3,1.6)):
                torch.testing.assert_close(cauchy(values[name],.1,beta,width),terms[name])
            self.assertEqual(data["pre/observation/history"].shape,(2,2,50,42))
        self.assertFalse(torch.cuda.is_initialized())

    def test_copy_survives_history_previous_buffers_and_reset_mutation(self):
        """env写回previous或reset之后，旧奖励状态与原始动作仍保持不变。"""
        recorder = RewardTraceRecorder(Config(),num_envs=2,max_steps=1,output_dir=self.directory)
        obs, action, state = fixture()
        state["terminated"][0] = True
        saved_action = action.clone()
        saved_previous_force = state["prev_foot_force"].clone()
        recorder.capture_pre(obs,action,action+.1,torch.ones_like(action),episode_length=torch.ones(4,dtype=torch.long))
        action.fill_(999);obs["history"].fill_(999)
        terms = reward_terms(state,Config())
        recorder.capture_reward(state,terms,foot_height=torch.full((4,2),.03))
        state["prev_foot_force"].copy_(state["foot_force"])
        state["height"].zero_()
        for term in terms.values():term.zero_()
        recorder.capture_post(terminated=state["terminated"],truncated=torch.zeros(4,dtype=torch.bool),
                              next_obs=torch.full((4,42),-999.))
        recorder.flush()
        with np.load(self.directory/"reward_trace.npz") as data:
            np.testing.assert_array_equal(data["pre/action"][0],saved_action[:2].numpy())
            np.testing.assert_array_equal(data["pre/observation/history"],1.)
            np.testing.assert_array_equal(data["reward/state/prev_foot_force"][0],saved_previous_force[:2].numpy())
            self.assertAlmostEqual(float(data["reward/state/height"][0,0]),.78,places=6)
            self.assertEqual(float(data["reward/terms/termination"][0,0]),-1.)
            self.assertEqual(float(data["post/next_obs"][0,0,0]),-999.)
        self.assertEqual(verify_trace(self.directory)["status"],"passed")

    def test_caps_stop_new_capture_and_byte_limit_discards_pending_only(self):
        """超过步数/字节上限后不再继续保存；绝不把残缺帧当完整帧。"""
        recorder = RewardTraceRecorder(Config(),num_envs=1,max_steps=1)
        complete(recorder)
        self.assertFalse(recorder.capture_pre({},torch.zeros(4,12),torch.zeros(4,12),torch.ones(4,12)))
        self.assertFalse(recorder.capture_reward({},{}))
        self.assertFalse(recorder.capture_post(terminated=torch.zeros(4),truncated=torch.zeros(4)))
        self.assertEqual(recorder.steps,1)
        limited = RewardTraceRecorder(Config(),max_steps=2,num_envs=2,max_bytes=10)
        obs,action,_ = fixture()
        self.assertFalse(limited.capture_pre(obs,action,action,torch.ones_like(action)))
        self.assertEqual(limited.stop_reason,"byte_cap")
        self.assertEqual(limited.steps,0)
        self.assertIsNone(limited.pending)
        self.assertEqual(limited.retained_bytes,0)
        for kwargs in ({"max_steps":1001},{"num_envs":17},{"max_steps":0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                RewardTraceRecorder(Config(),**kwargs)

    def test_mismatch_is_detected_and_incomplete_trace_cannot_pass(self):
        """不匹配的项明确失败，已完成前缀可以落盘但不能冒充完整通过。"""
        recorder = RewardTraceRecorder(Config(),num_envs=2,max_steps=3,output_dir=self.directory)
        complete(recorder)
        obs,action,state = fixture()
        recorder.capture_pre(obs,action,action,torch.ones_like(action))
        terms = reward_terms(state,Config())
        terms["swing_force"] += .01
        with self.assertRaisesRegex(FloatingPointError,"swing_force"):
            recorder.capture_reward(state,terms)
        summary = recorder.flush()
        self.assertEqual(summary["steps"],1)
        self.assertTrue(summary["incomplete_frame_saved"])
        self.assertTrue((self.directory/"incomplete_frame.npz").is_file())
        self.assertEqual(verify_trace(self.directory)["status"],"failed")
        with self.assertRaises(ValueError):analyze_trace(self.directory)

    def test_phase_height_contact_groups_and_missing_values_are_explicit(self):
        """按实际提供字段分组，缺失足高/接触时不猜值。"""
        recorder = RewardTraceRecorder(Config(),num_envs=2,max_steps=1,output_dir=self.directory,
            static_metadata={"foot_height_semantics":"ankle origin above plane; not sole clearance"})
        complete(recorder)
        recorder.flush()
        result = analyze_trace(self.directory)
        self.assertEqual(result["phase_groups"]["left/stance"]["transitions"],1)
        self.assertEqual(result["phase_height_groups"]["left/stance/0_to_0p03m"]["transitions"],1)
        self.assertEqual(result["phase_contact_groups"]["left/stance/contact"]["transitions"],1)
        self.assertEqual(result["missing_optional_fields"],[])
        self.assertIn("not sole clearance",result["height_definition"])
        empty = result["phase_groups"]["left/transition"]
        self.assertEqual(empty["transitions"],0)
        self.assertIsNone(empty["rewards"]["linear_velocity"])
        other = RewardTraceRecorder(Config(),num_envs=2,max_steps=1,output_dir=Path(self.tmp.name)/"missing")
        complete(other,include_geometry=False);other.flush()
        missing = analyze_trace(Path(self.tmp.name)/"missing")
        self.assertEqual(set(missing["missing_optional_fields"]),{"reward/foot_height","reward/foot_contact"})

    def test_cli_cpu_verify_and_trace_checksum(self):
        """真正启动CPU CLI验证保存产物；篡改NPZ后必须拒绝。"""
        recorder = RewardTraceRecorder(Config(),num_envs=2,max_steps=1,output_dir=self.directory)
        complete(recorder);recorder.flush()
        result = subprocess.run([sys.executable,"-B","-m","estnet.reward_diagnostics","verify",str(self.directory)],
                                cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((self.directory/"offline_verify.json").read_text())["status"],"passed")
        with (self.directory/"reward_trace.npz").open("ab") as stream:stream.write(b"changed")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):verify_trace(self.directory)

    def test_actual_environment_reward_hook_before_previous_copy_and_reset(self):
        """执行当前env真实奖励方法：不用Isaac，也不拿手写仿制hook代替实际接入。"""
        tree = ast.parse((ROOT/"estnet/environment.py").read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node,ast.FunctionDef) and node.name == "_get_rewards")
        namespace = {"torch":torch,"reward_terms":reward_terms,"gait":gait,
                     "HIP_YAW_POLICY_COLUMNS":HIP_YAW_POLICY_COLUMNS}
        exec(compile(ast.Module(body=[method],type_ignores=[]),"actual_environment_reward_hook","exec"),namespace)
        cfg = Config()
        obs,action,state = fixture()
        leg_ids = torch.tensor([18,0,23,8,2,26,15,3,6,10,1,20])
        default = torch.zeros(4,29)
        soft_limits = torch.tensor([-1.,1.]).expand(4,29,2).clone()
        targets = make_joint_targets(action,default,soft_limits,leg_ids,cfg.action_scale,
                                     hip_yaw_target_limit_rad=cfg.hip_yaw_target_limit_rad)
        origins = torch.zeros(4,3);origins[:,2] = torch.tensor([.0,.2,.4,.6])
        body_pos = origins[:,None,:].expand(4,2,3).clone();body_pos[:,:,2] += .08
        quaternion = torch.tensor([1.,0.,0.,0.]).expand(4,4).clone()
        data = SimpleNamespace(default_joint_pos=default,joint_pos=targets.clone(),
            root_pos_w=origins+torch.tensor([0.,0.,.78]),root_quat_w=quaternion,
            body_pos_w=body_pos,body_quat_w=quaternion[:,None,:].expand(4,2,4).clone())
        recorder = RewardTraceRecorder(cfg,num_envs=2,max_steps=1,output_dir=self.directory)
        env = SimpleNamespace(params=cfg,step_dt=cfg.step_dt,extras={},reward_recorder=recorder,
            phase_offset=torch.tensor([.025,.2,.55,.9]),episode_length_buf=torch.tensor([10,20,30,40]),
            robot=SimpleNamespace(data=data),leg_ids=leg_ids,foot_ids=[0,1],targets=targets,
            scene=SimpleNamespace(env_origins=origins),previous_action=action.clamp(-100.,100.),
            previous_force=state["prev_foot_force"],previous_torque=state["prev_torque"],
            previous_joint_velocity=state["prev_joint_vel"],last_contact=state["stance"]>.5,
            _transition_state=state)
        pre_phase,pre_stance = gait(env.phase_offset+env.episode_length_buf*cfg.step_dt/cfg.gait_period,
                                   cfg.gait_duty,cfg.gait_transition)
        recorder.capture_pre(obs,action,action+.1,torch.ones_like(action),phase_offset=env.phase_offset,
            episode_length=env.episode_length_buf,pre_phase=pre_phase,pre_stance=pre_stance)
        # DirectRLEnv在奖励前已推进episode计数；模拟这一控制步边界。
        env.episode_length_buf += 1
        reward_phase,reward_stance = gait(env.phase_offset+env.episode_length_buf*cfg.step_dt/cfg.gait_period,
                                         cfg.gait_duty,cfg.gait_transition)
        state["stance"] = reward_stance
        state["terminated"][0] = True
        expected = torch.stack(list(reward_terms(state,cfg).values()),dim=-1).sum(-1)
        saved_previous = env.previous_force.clone()
        actual = namespace["_get_rewards"](env)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(env.previous_force,state["foot_force"])
        # 随后首环境reset：确认奖励时刻的前一帧缓冲、phase和几何不会随之改变。
        env.previous_force[0] = 0.;env.episode_length_buf[0] = 0;env.phase_offset[0] = .75
        env.robot.data.body_pos_w[0] = -999.
        post_phase,post_stance = gait(env.phase_offset+env.episode_length_buf*cfg.step_dt/cfg.gait_period,
                                     cfg.gait_duty,cfg.gait_transition)
        recorder.capture_post(terminated=state["terminated"],truncated=torch.zeros(4,dtype=torch.bool),
            post_episode_length=env.episode_length_buf,post_phase=post_phase,post_stance=post_stance,
            reset_mask=state["terminated"],next_obs=torch.full((4,42),-999.))
        recorder.flush()
        self.assertEqual(verify_trace(self.directory)["status"],"passed")
        with np.load(self.directory/"reward_trace.npz") as trace:
            np.testing.assert_allclose(trace["reward/state/prev_foot_force"][0],saved_previous[:2])
            np.testing.assert_allclose(trace["reward/reward_phase"][0],reward_phase[:2])
            np.testing.assert_allclose(trace["reward/reward_stance"][0],trace["reward/state/stance"][0])
            np.testing.assert_allclose(trace["reward/final_target"][0],targets[:2,leg_ids])
            np.testing.assert_allclose(trace["reward/foot_height"][0],.08,atol=2e-8)
            np.testing.assert_allclose(trace["reward/requested_target"][0],action[:2]*cfg.action_scale)
            self.assertLessEqual(np.abs(trace["reward/hip_capped_target"][0,:,list(HIP_YAW_POLICY_COLUMNS)]).max(),
                                 cfg.hip_yaw_target_limit_rad+1e-7)
            self.assertEqual(trace["pre/episode_length"][0,0],10)
            self.assertEqual(trace["reward/reward_episode_length"][0,0],11)
            self.assertEqual(trace["post/post_episode_length"][0,0],0)
            self.assertEqual(len([key for key in trace.files if key.startswith("reward/terms/")]),11)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
