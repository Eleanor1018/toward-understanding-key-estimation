**EstNet：完整策略 KL 保护与逐步奖励诊断**

本目录从已冻结的自碰撞开启、hip yaw 目标 ±20° 版本独立复制。保留 42 维本体观测、50 帧历史、显式三维速度估计器、61 维非对称 critic；没有 latent、decoder 或 mimic。奖励函数、全部奖励参数、步态周期/相位、PD 参数与动作映射保持该基线。

本次工程改动：

- 每个 minibatch 更新后，分块遍历完整 rollout，计算旧策略到候选策略的解析 KL（12维求和后的样本均值）。包含 estimator→actor 全路径；平均值限制不是每个状态的独立上界。
- 候选 KL 大于 0.02 或候选模型/分布非有限，恢复该步之前的全部模型和 Adam 状态，降低学习率并结束该轮剩余联合更新。0.02 是工程候选上限，论文没有给出此回滚规则。
- `updates` 仍计 rollout 数；`total_accepted_steps` 单独计真正保留的 Adam 步数。记录的 candidate KL 可以超限，retained KL 才是最终保留值。
- 每次尝试记录 KL 的均值/方差及逐关节贡献，以及 policy/value/velocity/entropy 的梯度范数和估计器监督与 PPO 梯度夹角。梯度范数并非因果贡献百分比。
- 仅选前 16 个环境，最多记录 480 个控制步。保存动作前完整 obs/history/command、相位，以及 reset 前完整奖励输入和各项输出，每 5 轮落盘一次；足高度明确为 ankle link 原点相对平地的高度，不称为足底离地间隙。
- 第 10 轮第一个真实 minibatch 在更新前保存 CPU 快照，包含模型、Adam、原始动作、旧概率、整 rollout 归一化后的优势等。一步损失分支重放只说明这个新采集样本上的条件结果，不能复盘历史第 415 轮。

第一阶段是全新 seed42、4096 环境、20 轮烟测：验证记录、回滚、优化器计数与实际数据复算，不用它宣布机器人学会走路。旧 500 轮模型及训练目录不被覆盖。新 schema 为 `g1-estnet-klguard-flat-v1`，不允许把旧检查点静默当成新算法续训状态。

主要入口为 `estnet/run.py`，更新实现为 `estnet/ppo.py`，奖励记录与离线核验为 `estnet/reward_diagnostics.py`，条件重放为 `estnet/replay_diagnostics.py`。训练仍使用 Isaac Lab 2.0.2 / Isaac Sim 4.5.0；CPU 测试不要求 Isaac。

在本目录设好 PYTHONPATH 后可运行 `python -m pytest tests -q`。实际资产文件必须沿用已经核对的四份 G1 USD 内容签名，GPU 必须在启动前重新检查 UUID、利用率、显存和现有进程。

完整实验参数、源代码 SHA、实际服务器目录与结果以配套 `SOURCE_MANIFEST.json` 和本次中文结果报告为准；不得把烟测通过替代 32 环境首回合与视频的行走验收。
