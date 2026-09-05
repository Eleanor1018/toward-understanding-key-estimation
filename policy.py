import torch
from torch import nn

from config import (
    FUTURE_REFERENCE_FEATURES_PER_STEP,
    FUTURE_REFERENCE_STEPS,
    ModelConfig,
)


def _flatten_future_reference(
    future_reference: torch.Tensor | None,
    reference_tensor: torch.Tensor,
    expected_dim: int,
) -> torch.Tensor:
    """Validate and flatten an optional compact future-reference tensor."""

    batch_shape = reference_tensor.shape[:-1]
    if expected_dim == 0:
        if future_reference is not None:
            raise ValueError("This model does not accept a future reference")
        return reference_tensor.new_empty((*batch_shape, 0))
    if future_reference is None:
        raise ValueError("This model requires a future reference")
    if future_reference.device != reference_tensor.device:
        raise ValueError("future reference and policy inputs must share a device")
    if future_reference.dtype != reference_tensor.dtype:
        raise ValueError("future reference and policy inputs must share a dtype")
    if future_reference.shape == (*batch_shape, expected_dim):
        return future_reference
    if (
        expected_dim == FUTURE_REFERENCE_STEPS * FUTURE_REFERENCE_FEATURES_PER_STEP
        and future_reference.shape
        == (
            *batch_shape,
            FUTURE_REFERENCE_STEPS,
            FUTURE_REFERENCE_FEATURES_PER_STEP,
        )
    ):
        return future_reference.reshape(*batch_shape, expected_dim)
    raise ValueError(
        "future reference must have shape "
        f"{(*batch_shape, expected_dim)} or batch-compatible unflattened shape; "
        f"got {tuple(future_reference.shape)}"
    )


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.history_steps = config.history_steps
        self.obs_dim = config.obs_dim

        # MLP 4650 -> 1024 -> 256 -> 64
        layers = []
        input_dim = config.encoder_input_dim

        for hidden_dim in config.encoder_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        self.latent_head = nn.Linear(
            input_dim,
            config.latent_dim,
        )

        self.explicit_head = nn.Linear(
            input_dim,
            config.explicit_dim,
        )

    def forward(
        self,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # [B, 50, 93] -> [B, 4650]
        flattened_history = history.flatten(start_dim=1)

        # [B, 4650] -> [B, 64]
        features = self.trunk(flattened_history)

        latent = self.latent_head(features)
        explicit_estimate = self.explicit_head(features)
        return latent, explicit_estimate


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # MLP 19 -> 64 -> 128 -> 93
        layers = []
        input_dim = config.latent_dim + config.explicit_dim

        for hidden_dim in config.decoder_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, config.obs_dim))
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        explicit: torch.Tensor,
    ) -> torch.Tensor:

        concatenated = torch.cat([latent, explicit], dim=-1)
        features = self.trunk(concatenated)

        return features


class Actor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.future_reference_dim = config.future_reference_dim

        # MLP (93 + command + 16 + 3) -> 2048 -> 512 -> 128 -> 29
        layers = []
        input_dim = config.actor_input_dim

        for hidden_dim in config.actor_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, config.action_dim))
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        command: torch.Tensor,
        latent: torch.Tensor,
        explicit: torch.Tensor,
        future_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:

        future = _flatten_future_reference(
            future_reference,
            obs,
            self.future_reference_dim,
        )
        concatenated = torch.cat(
            [obs, command, future, latent, explicit],
            dim=-1,
        )
        action_mean = self.trunk(concatenated)

        return action_mean


class Critic(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.future_reference_dim = config.future_reference_dim

        # MLP (93 + command + 103) -> 2048 -> 512 -> 128 -> 1
        layers = []
        input_dim = config.critic_input_dim

        for hidden_dim in config.critic_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        command: torch.Tensor,
        privileged: torch.Tensor,
        future_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:

        future = _flatten_future_reference(
            future_reference,
            obs,
            self.future_reference_dim,
        )
        concatenated = torch.cat(
            [obs, command, future, privileged],
            dim=-1,
        )
        value = self.trunk(concatenated)

        return value.squeeze(-1)  # [B, 1] -> [B]
