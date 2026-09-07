"""Key1/Key2 辅助目标；策略 KL 只测量，VAE latent KL(beta50)仍参与训练。"""
import math

from torch.nn import functional as F

from .ppo import PPO


def checked_mse(prediction, target, name):
    """禁止 PyTorch 广播悄悄掩盖监督维度错误。"""
    if prediction.shape != target.shape:
        raise ValueError(f"{name} supervision must exactly match its prediction shape")
    return F.mse_loss(prediction, target)


def reconstruction_and_latent_kl(auxiliary, current_obs):
    # 本轮沿用旧时序约定：历史不含当前帧，decoder 重构当前 obs_t。
    # 论文图中的 t+1 标记与文字 current 有歧义；这里不静默换成 next_obs。
    prediction_loss = checked_mse(auxiliary["prediction"], current_obs, "current observation reconstruction")
    mu, logvar = auxiliary["mu"], auxiliary["logvar"]
    if mu.ndim != 2 or logvar.shape != mu.shape or mu.shape[0] != current_obs.shape[0]:
        raise ValueError("Latent mu/logvar must have matching [N, latent_dim] shapes")
    # batch 与 latent 两维同时平均；beta=50，不额外乘 latent_dim。
    latent_kl = 0.5 * (mu.square() + logvar.exp() - 1. - logvar).mean()
    return prediction_loss, latent_kl, {"prediction_loss": prediction_loss, "latent_kl": latent_kl,
        "latent_mu_abs": mu.abs().mean(), "latent_std_mean": (0.5 * logvar).exp().mean()}


class KeyPPO(PPO):
    def __init__(self, model, cfg):
        if cfg.variant not in ("key1", "key2") or model.variant != cfg.variant:
            raise ValueError("KeyPPO requires matching KeyPolicy and configuration variants")
        for name in ("velocity_coef", "heightmap_coef", "prediction_coef", "vae_beta"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        super().__init__(model, cfg)

    def additional_batch_fields(self):
        return ("heightmap",) if self.model.variant == "key2" else ()

    def auxiliary_loss(self, data, indices, velocity):
        del velocity
        # 每 minibatch 重算辅助图；actor 使用 mu，decoder 按模型约定重参数采样。
        auxiliary = self.model.auxiliary(data["history"][indices], data["obs"][indices], data["command"][indices])
        velocity_loss = checked_mse(auxiliary["velocity"], data["velocity"][indices], "velocity")
        prediction_loss, latent_kl, metrics = reconstruction_and_latent_kl(auxiliary, data["obs"][indices])
        weighted_latent = self.cfg.vae_beta * latent_kl
        weighted = self.cfg.velocity_coef * velocity_loss + self.cfg.prediction_coef * prediction_loss + weighted_latent
        metrics.update(velocity_loss=velocity_loss, latent_kl_weighted=weighted_latent)
        if self.model.variant == "key2":
            heightmap_loss = checked_mse(auxiliary["heightmap"], data["heightmap"][indices], "heightmap")
            weighted = weighted + self.cfg.heightmap_coef * heightmap_loss
            metrics["heightmap_loss"] = heightmap_loss
        return weighted, metrics
