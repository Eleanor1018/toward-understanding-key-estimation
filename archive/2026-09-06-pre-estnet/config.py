from dataclasses import dataclass


FUTURE_REFERENCE_REWARD_PROFILE = "p1_walk_stable_v8_future_reference"
FUTURE_REFERENCE_NORMALIZATION_TYPE = "g1_future_reference_3x21_v1"
FUTURE_REFERENCE_STEPS = 3
FUTURE_REFERENCE_FEATURES_PER_STEP = 21
FUTURE_REFERENCE_DIM = FUTURE_REFERENCE_STEPS * FUTURE_REFERENCE_FEATURES_PER_STEP
PHASE_VISIBLE_REWARD_PROFILES = frozenset(
    {
        "p1_walk_stable_v6_phase_rsi",
        "p1_walk_stable_v6_phase_rsi_imitation",
        "p1_walk_stable_v7_full_reference",
        FUTURE_REFERENCE_REWARD_PROFILE,
    }
)
FULL_REFERENCE_REWARD_PROFILE = "p1_walk_stable_v7_full_reference"


@dataclass(frozen=True)
class ModelConfig:
    # Unitree G1 29-DOF body (12 leg + 3 waist + 14 arm joints).
    # Observation: gravity(3) + base angular velocity(3)
    #              + joint position(29) + joint velocity(29)
    #              + previous action(29) = 93.
    obs_dim: int = 93
    # Physical command (vx, vy, yaw rate) plus a visible cyclic gait phase
    # represented as (sin(2*pi*phase), cos(2*pi*phase)).
    command_dim: int = 5
    # Three future reference frames with 21 compact features per frame.
    future_reference_dim: int = 0

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
            + self.future_reference_dim
            + self.latent_dim
            + self.explicit_dim
        )

    @property
    def critic_input_dim(self) -> int:
        return (
            self.obs_dim
            + self.command_dim
            + self.future_reference_dim
            + self.privileged_dim
        )


def model_config_for_reward_profile(reward_profile: str) -> ModelConfig:
    """Select the explicit 3-D or phase-visible 5-D command contract."""

    command_dim = 5 if reward_profile in PHASE_VISIBLE_REWARD_PROFILES else 3
    future_reference_dim = (
        FUTURE_REFERENCE_DIM if reward_profile == FUTURE_REFERENCE_REWARD_PROFILE else 0
    )
    return ModelConfig(
        command_dim=command_dim,
        future_reference_dim=future_reference_dim,
    )
