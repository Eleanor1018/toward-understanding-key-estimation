from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # Unitree G1 29-DOF body (12 leg + 3 waist + 14 arm joints).
    # Observation: gravity(3) + base angular velocity(3)
    #              + joint position(29) + joint velocity(29)
    #              + previous action(29) = 93.
    obs_dim: int = 93
    command_dim: int = 3

    # Provisional until the G1 privileged observation group is finalized.
    # It is not determined by the number of controlled joints.
    privileged_dim: int = 103
    latent_dim: int = 16
    action_dim: int = 29
    explicit_dim: int = 3

    history_seconds: float = 0.5
    policy_hz: int = 100

    encoder_hidden_dims: tuple[int, ...] = (1024, 256, 64)
    decoder_hidden_dims: tuple[int, ...] = (64, 128)
    actor_hidden_dims: tuple[int, ...] = (2048, 512, 128)
    critic_hidden_dims: tuple[int, ...] = (2048, 512, 128)

    # PPO / GAE hyperparameters
    discount_gamma: float = 0.996
    gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.2

    @property
    def history_steps(self) -> int:
        return round(self.history_seconds * self.policy_hz)

    @property
    def encoder_input_dim(self) -> int:
        return self.history_steps * self.obs_dim

    @property
    def actor_input_dim(self) -> int:
        return (
            self.obs_dim
            + self.command_dim
            + self.latent_dim
            + self.explicit_dim
        )

    @property
    def critic_input_dim(self) -> int:
        return (
            self.obs_dim
            + self.command_dim
            + self.privileged_dim
        )
