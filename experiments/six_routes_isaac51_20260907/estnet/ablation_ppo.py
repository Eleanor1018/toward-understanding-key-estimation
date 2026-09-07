"""FullEst / IrrEst / Implicit 的显式监督和 VAE 目标，复用固定 LR PPO。"""
import math

from .key_ppo import checked_mse, reconstruction_and_latent_kl
from .ppo import PPO


class AblationPPO(PPO):
    def __init__(self, model, cfg):
        if cfg.variant not in ("fullest", "irrest", "implicit") or model.variant != cfg.variant:
            raise ValueError("AblationPPO requires matching policy and configuration variants")
        for name in ("velocity_coef", "heightmap_coef", "body_height_coef", "prediction_coef", "vae_beta"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        super().__init__(model, cfg)

    def additional_batch_fields(self):
        # FullEst: velocity3+脚周map18+body_height1；IrrEst仅body_height1；Implicit无显式头。
        # 81维机体高度图属于 critic 特权输入，不是 IrrEst 的辅助目标。
        return tuple(name for name in self.model.explicit_names if name != "velocity")

    def auxiliary_loss(self, data, indices, velocity):
        del velocity
        auxiliary = self.model.auxiliary(data["history"][indices], data["obs"][indices], data["command"][indices])
        prediction_loss, latent_kl, metrics = reconstruction_and_latent_kl(auxiliary, data["obs"][indices])
        weighted_latent = self.cfg.vae_beta * latent_kl
        weighted = self.cfg.prediction_coef * prediction_loss + weighted_latent
        metrics["latent_kl_weighted"] = weighted_latent
        for name in self.model.explicit_names:
            loss = checked_mse(auxiliary[name], data[name][indices], name)
            weighted = weighted + getattr(self.cfg, f"{name}_coef") * loss
            metrics[f"{name}_loss"] = loss
        return weighted, metrics
