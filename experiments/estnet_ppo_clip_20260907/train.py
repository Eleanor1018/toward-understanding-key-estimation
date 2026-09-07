"""新 EstNet 训练入口：固定 LR 的 PPO-clip，自碰撞 + 仅髋 yaw 额外限幅。

用法：python train.py --asset /path/to/g1_29dof_rev_1_0.usd --headless --iterations 500
默认从随机权重开始；--resume 仅接受本版本的完整检查点，iterations 表示目标总轮数。
具体采样/GAE/保存逻辑在 estnet/run.py，联合优化在 estnet/ppo.py。
"""
from estnet.run import main


if __name__ == "__main__":
    # 薄入口保证只有一套训练循环，避免 train.py 和 run.py 的更新规则分叉。
    # 此处不启动 Isaac；--help 和参数错误可在普通 CPU Python 环境返回。
    raise SystemExit(main(mode_override="train"))
