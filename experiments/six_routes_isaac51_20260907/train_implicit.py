"""implicit从零训练/续训入口；自碰撞开启、仅hip yaw额外限幅。

本文件只选择路线；物理、PPO与记录逻辑集中在estnet/run.py，避免各路线代码漂移。
默认目标10000轮；--iterations是训练结束时的总轮数，--resume从新回合续训。
"""
from estnet.run import main

if __name__ == "__main__":
    raise SystemExit(main(mode_override="train", variant_override="implicit"))
