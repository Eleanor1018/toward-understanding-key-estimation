"""Key1/Key2仅扩展PPO辅助损失；策略裁剪、GAE和自适应LR沿用基类。"""
import math

from torch.nn import functional as F

from .ppo import PPO


class KeyPPO(PPO):
    def __init__(self, model, cfg):
        if cfg.variant not in ("key1", "key2") or model.variant != cfg.variant:
            raise ValueError("KeyPPO requires matching KeyPolicy and configuration variants")
        for name in ("velocity_coef", "heightmap_coef", "prediction_coef", "vae_beta"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        super().__init__(model, cfg)

    def additional_batch_fields(self):
        return ("heightmap",) if self.model.variant == "key2" else ()

    def auxiliary_loss(self, data, indices, velocity):
        # 重新计算辅助分支，避免跨minibatch缓存计算图；actor的mu没有重采样。
        del velocity
        auxiliary = self.model.auxiliary(data["history"][indices], data["obs"][indices],
                                         data["command"][indices])
        velocity_loss = F.mse_loss(auxiliary["velocity"], data["velocity"][indices])
        prediction_loss = F.mse_loss(auxiliary["prediction"], data["obs"][indices])
        mu, logvar = auxiliary["mu"], auxiliary["logvar"]
        # KL对batch和latent维同时取均值；beta=50不再隐含乘以latent维数。
        latent_kl = 0.5 * (mu.square() + logvar.exp() - 1. - logvar).mean()
        weighted = (self.cfg.velocity_coef * velocity_loss
                    + self.cfg.prediction_coef * prediction_loss + self.cfg.vae_beta * latent_kl)
        metrics = {"velocity_loss": velocity_loss, "prediction_loss": prediction_loss,
                   "latent_kl": latent_kl, "latent_mu_abs": mu.abs().mean(),
                   "latent_std_mean": (0.5 * logvar).exp().mean()}
        if self.model.variant == "key2":
            target = data["heightmap"][indices]
            if target.shape != auxiliary["heightmap"].shape:
                raise ValueError("Heightmap supervision must match the predicted [N, heightmap_dim] shape")
            heightmap_loss = F.mse_loss(auxiliary["heightmap"], target)
            weighted = weighted + self.cfg.heightmap_coef * heightmap_loss
            metrics["heightmap_loss"] = heightmap_loss
        return weighted, metrics
