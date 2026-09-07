"""固定Isaac Lab 2.3.2 / Isaac Sim 5.1适配；必须在AppLauncher之后导入。

This is a new flat-ground task, not a verified simulator result. The no-hand
29-joint asset is retained; the policy controls 12 legs and PD holds the torso.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .config import Config
from .gait import gait, command_with_phase
from .history import History
from .rewards import reward_terms
from .robot import (JOINT_NAMES29 as JOINT_NAMES, LEG_JOINT_NAMES12 as LEG_JOINT_NAMES,
                    DEFAULT_JOINT_POS29 as DEFAULT_JOINT_POS, STIFFNESS29 as STIFFNESS,
                    DAMPING29 as DAMPING, EFFORT_LIMIT29 as EFFORT_LIMIT,
                    VELOCITY_LIMIT29 as VELOCITY_LIMIT, ARMATURE29 as ARMATURE,
                    make_joint_targets, HIP_YAW_POLICY_COLUMNS)


@configclass
class EnvironmentCfg(DirectRLEnvCfg):
    # 逐个拷贝失败姿态到 CPU 只用于短时诊断，常规训练保留批量统计即可。
    detailed_diagnostics = False
    decimation = 10
    episode_length_s = 10.0
    action_space = 12
    observation_space = {"obs": 42, "history": [50, 42], "command": 7,
                         "critic": 61, "velocity": 3}
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.001, render_interval=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5,
                                                     replicate_physics=True)


def build_env_cfg(config, asset_path, device):
    config.validate()
    cfg = EnvironmentCfg()
    cfg.seed = config.seed
    cfg.sim.device = device
    cfg.sim.dt = config.sim_dt
    # Lab2.3.2把稳定化默认改为False；显式保持原Lab2.0.2训练物理设置。
    cfg.sim.physx.enable_stabilization = True
    cfg.sim.physx.solve_articulation_contact_last = False
    cfg.sim.physx.enable_external_forces_every_iteration = False
    cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    cfg.decimation = config.decimation
    cfg.sim.render_interval = config.decimation
    cfg.scene.num_envs = config.num_envs
    cfg.episode_length_s = config.episode_seconds
    cfg.baseline = config
    # 五条VAE路线使用152维特权critic；只输出各自确实监督的标签。
    # observation字典里的critic/监督真值不是actor输入，由网络接口明确隔离。
    cfg.observation_space = {"obs": config.obs_dim,
                             "history": [config.history_steps, config.obs_dim],
                             "command": config.command_dim, "critic": config.critic_dim,
                             **config.supervision_dims}
    # All actuator dictionaries are keyed by exact names, never implicit ordering.
    cfg.robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=config.self_collisions, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., .8), joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=.9,
        actuators={"joints": ImplicitActuatorCfg(
            joint_names_expr=list(JOINT_NAMES),
            stiffness=dict(zip(JOINT_NAMES, STIFFNESS)),
            damping=dict(zip(JOINT_NAMES, DAMPING)),
            effort_limit_sim=dict(zip(JOINT_NAMES, EFFORT_LIMIT)),
            velocity_limit_sim=dict(zip(JOINT_NAMES, VELOCITY_LIMIT)),
            armature=dict(zip(JOINT_NAMES, ARMATURE)))})
    return cfg


class EstNetEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        self.params: Config = cfg.baseline
        super().__init__(cfg, **kwargs)
        ids, names = self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != tuple(JOINT_NAMES) or self.robot.num_joints != 29:
            raise RuntimeError("Asset must be the expected no-hand 29-DOF G1")
        self.joint_ids = ids
        self.leg_ids = torch.tensor(self.robot.find_joints(list(LEG_JOINT_NAMES),
                                   preserve_order=True)[0], device=self.device, dtype=torch.long)
        # yaw策略列按名称产生，再映射到当前资产的原生关节顺序。
        self.hip_yaw_policy_columns = list(HIP_YAW_POLICY_COLUMNS)
        self.hip_yaw_native_ids = self.leg_ids[self.hip_yaw_policy_columns]
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_ids, foot_names = self.robot.find_bodies(feet, preserve_order=True)
        if foot_names != feet:
            raise RuntimeError(f"Unexpected foot mapping: {foot_names}")
        self.sensor_feet = self.contacts.find_bodies(feet, preserve_order=True)[0]
        self.illegal_bodies = [i for i, name in enumerate(self.contacts.body_names) if name not in feet]
        self.mass = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)
        self.leg_effort_limits = torch.tensor(EFFORT_LIMIT[:12], device=self.device)
        n = self.num_envs
        self.phase_offset = torch.zeros(n, device=self.device)
        self.physical_command = torch.zeros(n, 3, device=self.device)
        self.previous_action = torch.zeros(n, 12, device=self.device)
        self.previous_force = torch.zeros(n, 2, 3, device=self.device)
        self.previous_torque = torch.zeros(n, 12, device=self.device)
        self.previous_joint_velocity = torch.zeros(n, 12, device=self.device)
        self.targets = self.robot.data.default_joint_pos.clone()
        self.history = History(n, self.params.history_steps, self.params.obs_dim, self.device)
        self.last_contact = torch.zeros(n, 2, dtype=torch.bool, device=self.device)
        self.air_time = torch.zeros(n, 2, device=self.device)
        self.last_landing = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.target_clip_fraction = torch.zeros(n, device=self.device)
        self.hip_yaw_target_clip_fraction = torch.zeros(n, 2, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            num_envs=self.cfg.scene.num_envs, env_spacing=self.cfg.scene.env_spacing,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        self.terrain = self.terrain.class_type(self.terrain)
        self.contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", update_period=self.params.sim_dt,
            history_length=self.params.decimation + 1, track_air_time=True))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contacts"] = self.contacts
        light = sim_utils.DomeLightCfg(intensity=1800., color=(.8, .8, .8))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions):
        if actions.shape != self.previous_action.shape or not torch.isfinite(actions).all():
            raise ValueError("Non-finite or incorrectly shaped policy actions")
        raw = actions.clamp(-self.params.raw_action_clip, self.params.raw_action_clip)
        self.targets = make_joint_targets(raw, self.robot.data.default_joint_pos,
                                         self.robot.data.soft_joint_pos_limits,
                                         self.leg_ids, self.params.action_scale,
                                         hip_yaw_target_limit_rad=self.params.hip_yaw_target_limit_rad,
                                         raw_action_clip=self.params.raw_action_clip)
        requested = self.robot.data.default_joint_pos[:, self.leg_ids] + raw * self.params.action_scale
        self.target_clip_fraction = (requested != self.targets[:, self.leg_ids]).float().mean(-1)
        # 现在只有 hip yaw 会触发目标裁剪。soft-limit 越界另作诊断，不截断动作或加奖励惩罚。
        self.hip_yaw_target_clip_fraction.copy_(
            (requested[:, self.hip_yaw_policy_columns]
             != self.targets[:, self.hip_yaw_native_ids]).float())
        self.previous_action.copy_(raw)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _hip_yaw_metrics(self):
        """记录目标裁剪与实际角度；实际角度可能因动力学越过目标边界，不额外终止回合。"""
        actual = self.robot.data.joint_pos[:, self.hip_yaw_native_ids]
        default = self.robot.data.default_joint_pos[:, self.hip_yaw_native_ids]
        offset = actual - default
        return {
            "hip_yaw_target_clip_fraction": self.hip_yaw_target_clip_fraction.mean(-1),
            "hip_yaw_target_clip_left": self.hip_yaw_target_clip_fraction[:, 0].clone(),
            "hip_yaw_target_clip_right": self.hip_yaw_target_clip_fraction[:, 1].clone(),
            "hip_yaw_left_rad": actual[:, 0].clone(),
            "hip_yaw_right_rad": actual[:, 1].clone(),
            "hip_yaw_abs_offset_max_rad": offset.abs().amax(-1),
            "hip_yaw_actual_outside_target_fraction": (
                offset.abs() > self.params.hip_yaw_target_limit_rad).float().mean(-1),
        }

    def _read_state(self):
        data = self.robot.data
        phase = self.phase_offset + self.episode_length_buf * self.step_dt / self.params.gait_period
        foot_phase, stance = gait(phase, self.params.gait_duty, self.params.gait_transition)
        command = command_with_phase(self.physical_command, foot_phase)
        q = data.joint_pos[:, self.leg_ids]
        dq = data.joint_vel[:, self.leg_ids]
        obs = torch.cat((data.projected_gravity_b,
                         data.root_ang_vel_b * self.params.angular_velocity_scale,
                         q - data.default_joint_pos[:, self.leg_ids],
                         dq * self.params.joint_velocity_scale, self.previous_action), dim=-1)
        force = self.contacts.data.net_forces_w[:, self.sensor_feet, :]
        height = data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        foot_height = data.body_pos_w[:, self.foot_ids, 2] - self.scene.env_origins[:, 2:3]
        force_scaled = force / (9.81 * self.mass[:, None, None])
        if self.params.variant == "estnet":
            critic = torch.cat((obs, command, data.root_lin_vel_b, height[:, None],
                                force_scaled.flatten(1), foot_height), dim=-1)
        else:
            from .heightmaps import flat_heightmaps
            # 当前任务明确是水平plane，用解析地表交点计算高度；不是粗糙地形的替代传感器。
            # 小图各点以对应脚踝link为垂直参考，大图以base为参考；均是米。
            foot_map, base_map = flat_heightmaps(
                data.root_pos_w, data.root_quat_w, data.body_pos_w[:, self.foot_ids],
                self.scene.env_origins[:, 2])
            # 103D privilege = velocity3 + baseheight1 + smallmap18 + largemap81。
            # 18/81具体网格为工程选择；论文只给总103D，不能据此声称恢复作者网格。
            critic = torch.cat((obs, command, data.root_lin_vel_b, height[:, None], foot_map, base_map), -1)
        state = {"vel": data.root_lin_vel_b, "ang_vel": data.root_ang_vel_b,
                 "command": command, "up": -data.projected_gravity_b[:, 2],
                 "height": height, "foot_vel": data.body_lin_vel_w[:, self.foot_ids],
                 "foot_force": force, "prev_foot_force": self.previous_force,
                 "torque": data.applied_torque[:, self.leg_ids],
                 "prev_torque": self.previous_torque, "joint_vel": dq,
                 "prev_joint_vel": self.previous_joint_velocity,
                 "mass": self.mass, "stance": stance}
        if self.params.variant != "estnet":
            state["heightmap"] = foot_map
        return obs, critic, state

    def _get_dones(self):
        _, critic, state = self._read_state()
        history_force = self.contacts.data.net_forces_w_history[:, :self.params.decimation]
        illegal = history_force[:, :, self.illegal_bodies].norm(dim=-1).amax(dim=(1, 2)) > 1.
        terminated = illegal | (state["height"] < .45) | (state["up"] < .5)
        truncated = self.episode_length_buf >= self.max_episode_length
        state["terminated"] = terminated
        self._transition_state = state
        # DirectRLEnv resets before returning obs; save the true final state first.
        self.extras["final_critic"] = critic.clone()
        contact = history_force[:, :, self.sensor_feet].norm(dim=-1).amax(dim=1) > (.05 * self.mass[:, None] * 9.81)
        landing = contact & ~self.last_contact
        valid_landing = landing & (self.air_time >= .08) & (self.air_time <= .45)
        single = valid_landing.sum(-1) == 1
        foot = valid_landing.long().argmax(-1)
        alternating = single & (self.last_landing >= 0) & (foot != self.last_landing)
        self.last_landing[single] = foot[single]
        slip = (state["foot_vel"][..., :2].norm(dim=-1) * contact).sum(-1) / contact.sum(-1).clamp_min(1)
        self.extras["metrics"] = {
            "termination_illegal_contact": illegal.float(),
            "termination_low_height": (state["height"] < .45).float(),
            "termination_tilt": (state["up"] < .5).float(),
            "upright_cosine": state["up"].clone(),
            "joint_tracking_error": (self.robot.data.joint_pos[:, self.leg_ids] - self.targets[:, self.leg_ids]).abs().mean(-1),
            "torque_saturation_fraction": ((state["torque"].abs() / self.leg_effort_limits) >= .98).float().mean(-1),
            "velocity_error": (state["vel"] - torch.cat((self.physical_command[:, :2], torch.zeros_like(self.physical_command[:, :1])), -1)).norm(dim=-1).clone(),
            "forward_velocity": state["vel"][:, 0].clone(),
            "horizontal_speed": state["vel"][:, :2].norm(dim=-1),
            "command_vx": self.physical_command[:, 0].clone(),
            "double_support": contact.all(-1).float(),
            "single_support": (contact.sum(-1) == 1).float(),
            "flight": (~contact.any(-1)).float(),
            "contact_phase_error": (contact.float() - state["stance"]).abs().mean(-1),
            "stance_slip": slip, "valid_landing": single.float(),
            "alternating_landing": alternating.float(),
            "target_clip_fraction": self.target_clip_fraction.clone(),
            "base_height": state["height"].clone(),
            "ankle_height_left": self.robot.data.body_pos_w[:, self.foot_ids[0], 2].clone() - self.scene.env_origins[:, 2],
            "ankle_height_right": self.robot.data.body_pos_w[:, self.foot_ids[1], 2].clone() - self.scene.env_origins[:, 2],
            "terminated": terminated.float(), "truncated": truncated.float()}
        self.extras["metrics"].update(self._hip_yaw_metrics())
        if self.cfg.detailed_diagnostics and terminated.any():
            failed = terminated.nonzero().flatten()
            peak = history_force[failed].norm(dim=-1).amax(dim=(0, 1))
            self.extras["last_failure"] = {
                "episode_steps": self.episode_length_buf[failed].detach().cpu().tolist(),
                "root_position": self.robot.data.root_pos_w[failed].detach().cpu().tolist(),
                "projected_gravity": self.robot.data.projected_gravity_b[failed].detach().cpu().tolist(),
                "contact_force_peaks": {name: round(peak[i].item(), 3) for i, name in enumerate(self.contacts.body_names) if peak[i] > 1.}}
        self.air_time = torch.where(contact, 0., self.air_time + self.step_dt)
        self.last_contact.copy_(contact)
        return terminated, truncated

    def _get_rewards(self):
        terms = reward_terms(self._transition_state, self.params)
        self.extras["reward_terms"] = {k: v.clone() for k, v in terms.items()}
        reward = torch.stack(list(terms.values()), dim=-1).sum(-1)
        # 奖励输入中的previous_*仍指向上一控制步缓冲；必须先只读采样，
        # 再copy当前值。自动reset也尚未发生，不把新回合混进本步奖励证据。
        recorder = getattr(self, "reward_recorder", None)
        if recorder is not None and recorder.active:
            phase = self.phase_offset + self.episode_length_buf * self.step_dt / self.params.gait_period
            foot_phase, stance = gait(phase, self.params.gait_duty, self.params.gait_transition)
            default = self.robot.data.default_joint_pos[:, self.leg_ids]
            requested = default + self.previous_action * self.params.action_scale
            hip_capped = requested.clone()
            yaw = list(HIP_YAW_POLICY_COLUMNS)
            hip_capped[:, yaw] = torch.maximum(torch.minimum(hip_capped[:, yaw],
                default[:, yaw] + self.params.hip_yaw_target_limit_rad),
                default[:, yaw] - self.params.hip_yaw_target_limit_rad)
            recorder.capture_reward(self._transition_state, terms,
                reward_phase=foot_phase, reward_stance=stance,
                reward_episode_length=self.episode_length_buf,
                foot_height=self.robot.data.body_pos_w[:, self.foot_ids, 2] - self.scene.env_origins[:, 2:3],
                foot_contact=self.last_contact,
                final_target=self.targets[:, self.leg_ids], requested_target=requested,
                hip_capped_target=hip_capped, action_after_raw_clip=self.previous_action,
                actual_q=self.robot.data.joint_pos[:, self.leg_ids],
                base_pos_w=self.robot.data.root_pos_w, base_quat_wxyz=self.robot.data.root_quat_w,
                foot_pos_w=self.robot.data.body_pos_w[:, self.foot_ids],
                foot_quat_wxyz=self.robot.data.body_quat_w[:, self.foot_ids],
                total_reward=reward)
        self.previous_force.copy_(self._transition_state["foot_force"])
        self.previous_torque.copy_(self._transition_state["torque"])
        self.previous_joint_velocity.copy_(self._transition_state["joint_vel"])
        return reward

    def _get_observations(self):
        obs, critic, state = self._read_state()
        if self.params.observation_noise:
            scales = obs.new_tensor([.02]*3 + [.05 * self.params.angular_velocity_scale]*3
                                    + [.01]*12 + [1.5*self.params.joint_velocity_scale]*12 + [0.]*12)
            obs = obs + (torch.rand_like(obs) * 2 - 1) * scales
        result = {"obs": obs.clone(), "history": self.history.append(obs, exclude_current=self.params.variant != "estnet"),
                  "command": state["command"].clone(), "critic": critic.clone()}
        # 监督量与当前obs来自同一物理时刻，在step/reset前由runner复制。
        # IrrEst和Implicit的critic仍可读真值速度，但不因此产生速度估计监督。
        sources = {"velocity": state["vel"], "body_height": state["height"][:, None]}
        if "heightmap" in state:
            sources["heightmap"] = state["heightmap"]
        for name in self.params.supervision_dims:
            result[name] = sources[name].clone()
        return result

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        q = self.robot.data.default_joint_pos[env_ids].clone()
        dq = torch.zeros_like(q)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(q, dq, None, env_ids)
        self.targets[env_ids] = q
        self.previous_action[env_ids] = 0.
        self.previous_force[env_ids] = 0.
        self.previous_torque[env_ids] = 0.
        self.previous_joint_velocity[env_ids] = 0.
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device)
        self.physical_command[env_ids] = 0.
        self.physical_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.params.command_vx_min, self.params.command_vx_max)
        self.history.reset(env_ids)
        self.last_contact[env_ids] = False
        self.air_time[env_ids] = 0.
        self.last_landing[env_ids] = -1
