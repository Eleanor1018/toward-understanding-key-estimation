"""Key1/Key2显式估计与16维VAE；潜变量梯度不切断。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .networks import EstNet, _mlp


class KeyPolicy(EstNet):
    """保持EstNet的原始Gaussian动作接口，扩展共享编码器和重建支路。

    工程约定：actor始终使用mu；仅训练时的decoder使用重参数采样。
    这样同一观测和动作的PPO likelihood不会因重新采样latent而改变。
    decoder重建当前本体观测；历史排除当前帧由环境负责。
    Key1显式速度3；Key2另估计足周heightmap18。Key2的latent16沿用共同框架约定。
    机体周围base-map81仅是critic特权信息，不是Key1/Key2的估计头。
    """

    def __init__(self, cfg):
        # 不先创建再丢弃EstNet网络，避免无用参数和额外随机数消耗。
        nn.Module.__init__(self)
        self.variant = cfg.variant
        if self.variant not in ("key1", "key2"):
            raise ValueError("KeyPolicy variant must be key1 or key2")
        self.obs_dim = int(cfg.obs_dim)
        self.command_dim = int(cfg.command_dim)
        self.critic_dim = int(cfg.critic_dim)
        self.action_dim = int(cfg.action_dim)
        self.history_steps = int(cfg.history_steps)
        self.latent_dim = int(cfg.latent_dim)
        self.heightmap_dim = int(cfg.heightmap_dim)
        if min(self.obs_dim, self.command_dim, self.critic_dim, self.action_dim,
               self.history_steps, self.latent_dim, self.heightmap_dim) <= 0:
            raise ValueError("KeyPolicy dimensions must be positive")
        if not math.isfinite(cfg.init_std) or not math.exp(-3) <= cfg.init_std <= math.exp(1):
            raise ValueError("init_std must lie within exp([-3, 1])")
        hidden = tuple(cfg.encoder_hidden)
        if not hidden or any(type(width) is not int or width <= 0 for width in hidden):
            raise ValueError("Shared encoder must have positive hidden dimensions")

        layers = []
        width = self.history_steps * self.obs_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.ELU()))
            width = next_width
        self.encoder = nn.Sequential(*layers)
        self.explicit_names = ("velocity", "heightmap") if self.variant == "key2" else ("velocity",)
        self.has_velocity_estimate = True
        self.velocity_head = nn.Linear(width, 3)
        if self.variant == "key2":
            self.heightmap_head = nn.Linear(width, self.heightmap_dim)
        self.mu_head = nn.Linear(width, self.latent_dim)
        self.logvar_head = nn.Linear(width, self.latent_dim)
        self.explicit_dim = 3 + (self.heightmap_dim if self.variant == "key2" else 0)
        self.actor = _mlp(self.obs_dim + self.command_dim + self.explicit_dim + self.latent_dim,
                          tuple(cfg.actor_hidden), self.action_dim)
        self.critic = _mlp(self.critic_dim, tuple(cfg.critic_hidden), 1)
        self.decoder = _mlp(self.explicit_dim + self.latent_dim, tuple(cfg.decoder_hidden), self.obs_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(cfg.init_std)))

    def _encode(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.obs_dim):
            raise ValueError("KeyPolicy history has an incompatible shape")
        encoded = self.encoder(history.flatten(1))
        result = {"velocity": self.velocity_head(encoded), "mu": self.mu_head(encoded),
                  # 这是防止指数溢出的工程界限，不是论文给出的额外损失。
                  "logvar": self.logvar_head(encoded).clamp(-10., 10.)}
        if self.variant == "key2":
            result["heightmap"] = self.heightmap_head(encoded)
        return result

    def _check_current_inputs(self, history, obs, command):
        count = history.shape[0]
        if obs.shape != (count, self.obs_dim) or command.shape != (count, self.command_dim):
            raise ValueError("Current observation and command must match the history batch")

    def _explicit(self, encoded):
        if self.variant == "key2":
            return torch.cat((encoded["velocity"], encoded["heightmap"]), dim=-1)
        return encoded["velocity"]

    def estimate(self, history):
        return self._encode(history)["velocity"]

    def forward(self, history, obs, command):
        encoded = self._encode(history)
        self._check_current_inputs(history, obs, command)
        # actor没有真值速度或真值高度图入口；所有显式量都来自同一个历史编码器。
        actor_input = torch.cat((obs, command, self._explicit(encoded), encoded["mu"]), dim=-1)
        mean = self.actor(actor_input)
        std = self.log_std.clamp(-3., 1.).exp().expand_as(mean)
        return Normal(mean, std), encoded["velocity"]

    def auxiliary(self, history, obs, command):
        """返回监督/重建所需量；当前obs只校验形状，不泄露给decoder。"""
        encoded = self._encode(history)
        self._check_current_inputs(history, obs, command)
        latent = encoded["mu"]
        if self.training:
            latent = latent + (0.5 * encoded["logvar"]).exp() * torch.randn_like(latent)
        # sampled z与显式估计都保持梯度，重建误差能回传到各head和共享trunk。
        encoded["prediction"] = self.decoder(torch.cat((latent, self._explicit(encoded)), dim=-1))
        return encoded
