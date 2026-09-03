import torch
from torch import nn

from config import ModelConfig


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
        latent : torch.Tensor,
        explicit : torch.Tensor,
    ) -> torch.Tensor:

        concatenated = torch.cat([latent, explicit], dim=-1)
        features = self.trunk(concatenated)

        return features


class Actor(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()

        # MLP 115 -> 2048 -> 512 -> 128 -> 29
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
    ) -> torch.Tensor:

        concatenated = torch.cat([obs, command, latent, explicit], dim=-1)
        action_mean = self.trunk(concatenated)

        return action_mean


class Critic(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # MLP 199 -> 2048 -> 512 -> 128 -> 1
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
    ) -> torch.Tensor:

        concatenated = torch.cat([obs, command, privileged], dim=-1)
        value = self.trunk(concatenated)

        return value.squeeze(-1)  # [B, 1] -> [B]
