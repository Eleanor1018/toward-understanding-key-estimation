"""六路线共用的物理与PPO配置；显式标签和模型schema按variant严格分开。"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    # 独立协议：旧 KL 回退 / 全关节 soft-target 截断检查点不能静默续训。
    schema: str = "g1-estnet-ppo-clip-flat-v1"
    variant: str = "estnet"
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
    learning_rate_schedule: str = "fixed"
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.996
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.008
    velocity_coef: float = 1.0
    # 表III的辅助学习系数。VAE latent KL进入损失；PPO策略KL只是观测指标。
    heightmap_coef: float = 0.5
    body_height_coef: float = 2.0
    prediction_coef: float = 2.0
    vae_beta: float = 50.0
    latent_dim: int = 16
    heightmap_dim: int = 18
    base_heightmap_dim: int = 81
    decoder_hidden: tuple[int, ...] = (64, 256, 1024)
    # 论文未完整规定以下细节；保持既有VAE路线的明确工程约定。
    reconstruction_target: str = "current_obs_from_past_history"
    actor_latent_mode: str = "mean"
    vae_kl_reduction: str = "mean_batch_and_latent"
    max_grad_norm: float = 1.0
    # KL 只在每个完整优化 epoch 后分块测量，不调 LR、不拒绝更新、不提前结束。
    kl_chunk_size: int = 2048
    gradient_diagnostics: bool = False
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
    # 100 是宽松的原始动作数值保护（上游 Unitree RL Gym 的约定），不是 ±1 动作限制。
    raw_action_clip: float = 100.0
    # 仅 hip yaw 有额外角度目标限制；其余关节不再按 90% soft limits 截断 PD 目标。
    soft_joint_target_clipping: bool = False
    self_collisions: bool = True
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

    @property
    def supervision_dims(self):
        """只向rollout保存本路线确实估计的真值；这些标签绝不直接送actor。"""
        return {
            "estnet": {"velocity": 3},
            "key1": {"velocity": 3},
            "key2": {"velocity": 3, "heightmap": self.heightmap_dim},
            "fullest": {"velocity": 3, "heightmap": self.heightmap_dim, "body_height": 1},
            "irrest": {"body_height": 1},
            "implicit": {},
        }[self.variant]

    def validate(self):
        variants = ("estnet", "key1", "key2", "fullest", "irrest", "implicit")
        if self.variant not in variants or self.schema != f"g1-{self.variant}-ppo-clip-flat-v1":
            raise ValueError("Policy variant and PPO-clip checkpoint schema must agree")
        if (self.reconstruction_target != "current_obs_from_past_history"
                or self.actor_latent_mode != "mean"
                or self.vae_kl_reduction != "mean_batch_and_latent"):
            raise ValueError("Reconstruction target, actor latent mode and VAE KL reduction must match the recorded contract")
        if (self.latent_dim, self.heightmap_dim, self.base_heightmap_dim) != (16, 18, 81):
            raise ValueError("The common VAE contract is latent16, foot-map18 and base-map81")
        for name in ("value_coef", "entropy_coef", "velocity_coef", "heightmap_coef", "body_height_coef", "prediction_coef", "vae_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("encoder_hidden", "actor_hidden", "critic_hidden", "decoder_hidden"):
            widths = getattr(self, name)
            if not isinstance(widths, (tuple, list)) or not widths or any(type(width) is not int or width < 1 for width in widths):
                raise ValueError(f"{name} must contain positive integer widths")
        if self.learning_rate_schedule != "fixed":
            raise ValueError("KL is observation-only; learning_rate_schedule must be fixed")
        for name in ("learning_rate", "action_scale", "raw_action_clip", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("epochs", "minibatches", "horizon", "num_envs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.clip < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid PPO clip or GAE discount")
        if self.soft_joint_target_clipping is not False or self.self_collisions is not True:
            raise ValueError("This baseline requires self collision and no global soft-target clipping")
        if type(self.kl_chunk_size) is not int or self.kl_chunk_size < 1:
            raise ValueError("kl_chunk_size must be a positive integer")
        if type(self.gradient_diagnostics) is not bool:
            raise ValueError("gradient_diagnostics must be boolean")
        if (isinstance(self.hip_yaw_target_limit_rad, bool)
                or not isinstance(self.hip_yaw_target_limit_rad, (int, float))
                or not math.isfinite(self.hip_yaw_target_limit_rad)
                or not 0.0 < self.hip_yaw_target_limit_rad <= math.pi):
            raise ValueError("hip_yaw_target_limit_rad must be finite and in (0, pi]")
        if not math.isfinite(self.reward_target_height) or self.reward_target_height <= 0:
            raise ValueError("reward_target_height must be finite and positive")
        expected_critic = 61 if self.variant == "estnet" else 152
        if (self.obs_dim, self.command_dim, self.critic_dim, self.action_dim) != (42, 7, expected_critic, 12):
            raise ValueError("Observation/action/critic dimensions do not match the selected variant")
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
