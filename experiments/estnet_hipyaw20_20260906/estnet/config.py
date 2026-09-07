"""All first-run choices. Provenance and unverified choices are in docs/PLAN.md."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    # 新动作映射使用独立schema，旧未限幅模型不能静默进入本实验。
    schema: str = "g1-estnet-hipyaw-limited-flat-v1"
    obs_dim: int = 42
    command_dim: int = 7
    critic_dim: int = 61
    action_dim: int = 12
    history_steps: int = 50
    encoder_hidden: tuple[int, ...] = (1024, 256, 64)
    actor_hidden: tuple[int, ...] = (2048, 512, 128)
    critic_hidden: tuple[int, ...] = (2048, 512, 128)
    init_std: float = 0.8
    learning_rate: float = 5e-4
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.996
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.008
    velocity_coef: float = 1.0
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-3
    horizon: int = 24
    num_envs: int = 4096
    seed: int = 42
    sim_dt: float = 0.001
    decimation: int = 10
    episode_seconds: float = 10.0
    gait_period: float = 2.0 / 3.0
    gait_duty: float = 0.5
    gait_transition: float = 0.1
    command_vx_min: float = 0.25
    command_vx_max: float = 0.55
    action_scale: float = 0.25
    # 工程初值；限制左右hip yaw相对默认姿态的目标偏移，不修改USD机械限位。
    hip_yaw_target_limit_rad: float = math.radians(20.0)
    observation_noise: bool = False
    angular_velocity_scale: float = 0.25
    joint_velocity_scale: float = 0.05
    reward_linear_sigma: float = 0.5
    reward_yaw_sigma: float = 0.5
    reward_upright_sigma: float = 0.1
    reward_height_sigma: float = 0.05
    reward_target_height: float = 0.78
    reward_stance_velocity_sigma: float = 0.25
    reward_swing_force_sigma: float = 0.1
    reward_impact_sigma: float = 0.2
    reward_torque_delta_sigma: float = 160.0
    reward_joint_velocity_delta_sigma: float = 8.0
    reward_cot_sigma: float = 1.6
    reward_speed_floor: float = 0.1
    reward_termination_weight: float = 1.0

    @property
    def step_dt(self):
        return self.sim_dt * self.decimation

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.schema != "g1-estnet-hipyaw-limited-flat-v1":
            raise ValueError("This experiment requires the hip-yaw-limited EstNet schema")
        if (isinstance(self.hip_yaw_target_limit_rad, bool)
                or not isinstance(self.hip_yaw_target_limit_rad, (int, float))
                or not math.isfinite(self.hip_yaw_target_limit_rad)
                or not 0.0 < self.hip_yaw_target_limit_rad <= math.pi):
            raise ValueError("hip_yaw_target_limit_rad must be finite and in (0, pi]")
        if not math.isfinite(self.reward_target_height) or self.reward_target_height <= 0:
            raise ValueError("reward_target_height must be finite and positive")
        if (self.obs_dim, self.command_dim, self.critic_dim, self.action_dim) != (42, 7, 61, 12):
            raise ValueError("The v1 environment has a fixed 42/7/61/12 tensor contract")
        if abs(self.history_steps * self.step_dt - 0.5) > 1e-8:
            raise ValueError("History must span 0.5 seconds")
        if self.history_steps != 50 or abs(self.sim_dt - .001) > 1e-10 or self.decimation != 10:
            raise ValueError("v1 is fixed to 50 history frames, 1 kHz physics and 100 Hz policy")
        if self.num_envs * self.horizon < self.minibatches or self.num_envs < 1:
            raise ValueError("Insufficient rollout samples")
        if not 0 < self.gait_transition < min(self.gait_duty, 1 - self.gait_duty):
            raise ValueError("Invalid gait transition")
        if not 0 < self.command_vx_min <= self.command_vx_max:
            raise ValueError("This baseline requires strictly positive forward commands")
