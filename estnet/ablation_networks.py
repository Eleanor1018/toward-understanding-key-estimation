"""FullEst、IrrEst与Implicit消融；只创建各自存在的显式估计头。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .networks import EstNet, _mlp


class AblationPolicy(EstNet):
    """沿用Key的actor用mu、decoder重建当前观测的工程约定。

    原文IrrEst是高度估计加16维隐变量；Implicit没有显式估计。
    FullEst的16维隐变量宽度是沿用Key框架的选择，原文未单独给出。
    """

    def __init__(self, cfg):
        # 不创建EstNet的速度网络再弃用，消融参数中没有隐藏的无效估计头。
        nn.Module.__init__(self)
        self.variant = cfg.variant
        if self.variant not in ("fullest", "irrest", "implicit"):
            raise ValueError("AblationPolicy variant must be fullest, irrest or implicit")
        self.obs_dim = int(cfg.obs_dim)
        self.command_dim = int(cfg.command_dim)
        self.critic_dim = int(cfg.critic_dim)
        self.action_dim = int(cfg.action_dim)
        self.history_steps = int(cfg.history_steps)
        self.latent_dim = int(cfg.latent_dim)
        self.heightmap_dim = int(cfg.heightmap_dim)
        if min(self.obs_dim, self.command_dim, self.critic_dim, self.action_dim,
               self.history_steps, self.latent_dim, self.heightmap_dim) <= 0:
            raise ValueError("AblationPolicy dimensions must be positive")
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
        dimensions = {"velocity": 3, "heightmap": self.heightmap_dim, "body_height": 1}
        self.explicit_names = {
            "fullest": ("velocity", "heightmap", "body_height"),
            "irrest": ("body_height",),
            "implicit": (),
        }[self.variant]
        self.has_velocity_estimate = "velocity" in self.explicit_names
        self.explicit_dim = sum(dimensions[name] for name in self.explicit_names)
        for name in self.explicit_names:
            setattr(self, f"{name}_head", nn.Linear(width, dimensions[name]))
        self.mu_head = nn.Linear(width, self.latent_dim)
        self.logvar_head = nn.Linear(width, self.latent_dim)
        self.actor = _mlp(self.obs_dim + self.command_dim + self.explicit_dim + self.latent_dim,
                          tuple(cfg.actor_hidden), self.action_dim)
        self.critic = _mlp(self.critic_dim, tuple(cfg.critic_hidden), 1)
        self.decoder = _mlp(self.explicit_dim + self.latent_dim, tuple(cfg.decoder_hidden), self.obs_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(cfg.init_std)))

    def _encode(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, self.obs_dim):
            raise ValueError("AblationPolicy history has an incompatible shape")
        encoded = self.encoder(history.flatten(1))
        result = {name: getattr(self, f"{name}_head")(encoded) for name in self.explicit_names}
        result.update({"mu": self.mu_head(encoded),
                       # 与Key一致的指数稳定界限；论文没有规定这一裁剪。
                       "logvar": self.logvar_head(encoded).clamp(-10., 10.)})
        return result

    def _check_current_inputs(self, history, obs, command):
        count = history.shape[0]
        if obs.shape != (count, self.obs_dim) or command.shape != (count, self.command_dim):
            raise ValueError("Current observation and command must match the history batch")

    def estimate(self, history):
        # None明确表示该模型没有速度估计；不伪造零速度或误差指标。
        return self._encode(history)["velocity"] if self.has_velocity_estimate else None

    def forward(self, history, obs, command):
        encoded = self._encode(history)
        self._check_current_inputs(history, obs, command)
        estimates = tuple(encoded[name] for name in self.explicit_names)
        # 真值标签不在接口中；隐变量固定用mu，不给PPO likelihood增加采样噪声。
        mean = self.actor(torch.cat((obs, command, *estimates, encoded["mu"]), dim=-1))
        std = self.log_std.clamp(-3., 1.).exp().expand_as(mean)
        return Normal(mean, std), encoded.get("velocity")

    def auxiliary(self, history, obs, command):
        """当前obs只用于形状校验，重建输入仅有历史估计和隐变量。"""
        encoded = self._encode(history)
        self._check_current_inputs(history, obs, command)
        latent = encoded["mu"]
        if self.training:
            latent = latent + (0.5 * encoded["logvar"]).exp() * torch.randn_like(latent)
        estimates = tuple(encoded[name] for name in self.explicit_names)
        # 所有支路共同训练；重建可回传到mu、logvar及实际存在的显式估计头。
        encoded["prediction"] = self.decoder(torch.cat((latent, *estimates), dim=-1))
        return encoded
