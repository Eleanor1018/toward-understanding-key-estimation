"""配置、网络和优化器统一按variant路由；跨路线权重不得被静默解释。"""
from dataclasses import replace

from .config import Config

VARIANTS = ("estnet", "key1", "key2", "fullest", "irrest", "implicit")


def canonical_variant(variant):
    """论文拼写IrrEst；只将用户沿用的IllEst别名规范化，不猜其它未知名称。"""
    if not isinstance(variant, str):
        raise ValueError("variant must be a string")
    name = variant.strip().lower()
    name = "irrest" if name == "illest" else name
    if name not in VARIANTS:
        raise ValueError(f"Unknown policy variant: {variant}")
    return name


def config_for_variant(variant):
    variant = canonical_variant(variant)
    # 五个VAE路线critic统一152维；EstNet沿用61维。actor始终不接受critic真值。
    cfg = replace(Config(), variant=variant, schema=f"g1-{variant}-ppo-clip-flat-v1",
                  critic_dim=61 if variant == "estnet" else 152)
    cfg.validate()
    return cfg


def _checked_variant(cfg):
    cfg.validate()
    variant = canonical_variant(cfg.variant)
    if variant != cfg.variant:
        raise ValueError("Saved configuration must use the canonical variant name")
    return variant


def build_model(cfg):
    variant = _checked_variant(cfg)
    if variant == "estnet":
        from .networks import EstNet
        return EstNet(cfg)
    if variant in ("key1", "key2"):
        from .key_networks import KeyPolicy
        return KeyPolicy(cfg)
    from .ablation_networks import AblationPolicy
    return AblationPolicy(cfg)


def build_ppo(model, cfg):
    variant = _checked_variant(cfg)
    if getattr(model, "variant", None) != variant:
        raise ValueError("Model and PPO configuration belong to different variants")
    if variant == "estnet":
        from .ppo import PPO
        return PPO(model, cfg)
    if variant in ("key1", "key2"):
        from .key_ppo import KeyPPO
        return KeyPPO(model, cfg)
    from .ablation_ppo import AblationPPO
    return AblationPPO(model, cfg)
