"""按检查点/CLI中的显式variant构造模型；不同架构不混用权重。"""
from dataclasses import replace

from .config import Config


def canonical_variant(variant):
    """论文名称为IrrEst；接受用户沿用的IllEst拼写，但保存统一架构名。"""
    name = variant.lower()
    return "irrest" if name == "illest" else name


def config_for_variant(variant):
    variant = canonical_variant(variant)
    if variant == "estnet":
        return Config()
    if variant not in ("key1", "key2", "fullest", "irrest", "implicit"):
        raise ValueError(f"Unknown policy variant: {variant}")
    return replace(Config(), variant=variant, schema=f"g1-{variant}-flat-v1", critic_dim=152)


def build_model(cfg):
    if getattr(cfg, "variant", "estnet") == "estnet":
        from .networks import EstNet
        return EstNet(cfg)
    if cfg.variant in ("fullest", "irrest", "implicit"):
        from .ablation_networks import AblationPolicy
        return AblationPolicy(cfg)
    from .key_networks import KeyPolicy
    return KeyPolicy(cfg)


def build_ppo(model, cfg):
    if getattr(cfg, "variant", "estnet") == "estnet":
        from .ppo import PPO
        return PPO(model, cfg)
    if cfg.variant in ("fullest", "irrest", "implicit"):
        from .ablation_ppo import AblationPPO
        return AblationPPO(model, cfg)
    from .key_ppo import KeyPPO
    return KeyPPO(model, cfg)
