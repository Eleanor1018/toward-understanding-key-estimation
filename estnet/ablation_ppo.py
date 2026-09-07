"""消融只扩展辅助损失；PPO裁剪、GAE和自适应LR保持基类实现。"""
import math

from torch.nn import functional as F

from .ppo import PPO


class AblationPPO(PPO):
    def __init__(self, model, cfg):
        if cfg.variant not in ("fullest", "irrest", "implicit") or model.variant != cfg.variant:
            raise ValueError("AblationPPO requires matching policy and configuration variants")
        for name in ("velocity_coef", "heightmap_coef", "body_height_coef", "prediction_coef", "vae_beta"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        super().__init__(model, cfg)

    def additional_batch_fields(self):
        # velocity由基类has_velocity_estimate协议声明；其余只声明实际监督量。
        return tuple(name for name in self.model.explicit_names if name != "velocity")

    def auxiliary_loss(self, data, indices, velocity):
        del velocity
        # 每个minibatch重算辅助图，不复用actor图缓存或切断编码器梯度。
        auxiliary = self.model.auxiliary(data["history"][indices], data["obs"][indices],
                                         data["command"][indices])
        prediction_loss = F.mse_loss(auxiliary["prediction"], data["obs"][indices])
        mu, logvar = auxiliary["mu"], auxiliary["logvar"]
        latent_kl = 0.5 * (mu.square() + logvar.exp() - 1. - logvar).mean()
        weighted = self.cfg.prediction_coef * prediction_loss + self.cfg.vae_beta * latent_kl
        metrics = {"prediction_loss": prediction_loss, "latent_kl": latent_kl,
                   "latent_mu_abs": mu.abs().mean(),
                   "latent_std_mean": (0.5 * logvar).exp().mean()}
        for name in self.model.explicit_names:
            target = data[name][indices]
            if target.shape != auxiliary[name].shape:
                raise ValueError(f"{name} supervision must exactly match its prediction shape")
            loss = F.mse_loss(auxiliary[name], target)
            weighted = weighted + getattr(self.cfg, f"{name}_coef") * loss
            # 不存在的头没有loss或rmse键，避免用零值造成“完美估计”的假象。
            metrics[f"{name}_loss"] = loss
        return weighted, metrics
