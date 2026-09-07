# EstNet：PPO-clip + 自碰撞 + 仅髋 yaw 额外限幅

2026-09-07 新实验，协议 `g1-estnet-ppo-clip-flat-v1`。入口为 `train.py`，实际训练循环在 `estnet/run.py`，参数集中在 `estnet/config.py`。已有 KL-guard 实验及其运行源码保持独立。本目录默认从随机权重训练；旧实验 checkpoint 不兼容，不能直接续接已经学成站立的策略来冒充本版从零试验。

这版同时取消 KL 对优化的干预和全关节 soft-target 截断，按用户要求仅保留髋 yaw 的额外角度限制。它是新的训练基线，不是单变量因果实验；尚不能把此前不走全部归因于 KL，也没有证据保证这一版一定走起来。

## 实际改动

| 项目 | 上一版 KL-guard | 本版 |
|---|---|---|
| KL 超阈值 | 拒绝候选、还原模型和 Adam、提前停止本轮 | 删除；有限 KL 无论多大均只记录 |
| 学习率 | KL 自适应，有下降到 1e-5 的记录 | 固定 5e-4，无 KL 调度 |
| 每轮优化 | 4×4 为上限，实际可能跳过多步 | 正常完成必须有 16 次 Adam；非有限数明确失败 |
| 所有腿关节的 soft-target clamp | PD 目标截在 90% soft limits 内 | 删除；膝、髋 pitch/roll、踝目标不再被此层截断 |
| 髋 yaw 目标偏移 | 默认姿态 ±20° | 保留，仅作用左右 hip yaw |
| 自碰撞 | 开启 | 保留，并在 reset 后读回源机器人 USD 属性 |
| KL 诊断开销 | 每个候选后遍历 rollout，再可能回滚 | 每完整优化 epoch 测量一次；默认每轮4次 |
| 逐步轨迹/单步快照 | 默认开启 | 默认关闭，可用 CLI 显式开启；不生成视频 |
| 分项梯度探针 | 默认开启 | `gradient_diagnostics=False`，关闭昂贵的只读探针；正常反向传播不受影响 |

PPO 的 `clip=0.2`、value loss clip、梯度范数裁剪1、高斯 log-std 范围[-3,1]继续使用。它们分别作用于概率比损失、价值损失、优化梯度和探索标准差，**不是“每一步关节最多转0.2弧度”**。`clip` 也不保证最终所有概率比都落在[0.8,1.2]。本版不存在按关节目标变化量设定的额外逐步硬限幅。

## 动作链与保留边界

```text
策略：无 latent/decoder/mimic 的 EstNet → 12维 Normal 原始动作 a_raw
记录：PPO 始终保存 a_raw 及其原始高斯 log_prob
a = clamp(a_raw, -100, 100)
q_requested = q_default + 0.25 * a
仅两侧 hip_yaw：q_target = clamp(q_requested, q_default - 20°, q_default + 20°)
其它10个腿关节：q_target = q_requested
上半身17关节：q_target = q_default
PD → 仿真器的原始机械关节限位及电机力矩/速度限制
```

策略列先按关节名称解析，再映射到 USD 原生顺序。±100 是 Unitree RL Gym 使用的宽松原始动作保护，保留它；此前800轮轨迹没有触发它。0.25是输出到角度的比例，不会把无界高斯变成有界动作。±20°是此前方案的工程初值，**作者只说限制 hip yaw，没有给出这个数值；所查官方 G1 速度配置也没有给出±20°**。限制的是 PD 目标，实际角度仍可能有动力学超调。

soft limits 仍作为资产元数据读取，但不截断任何 PD 目标，也不新增 soft-limit 惩罚。这里取消的是额外软件截断，USD 机械限位、刚度阻尼、力矩与速度上限未移除。环境为纯平地、4096个机器人、12维腿动作、29自由度物理资产，保持当前 Isaac Lab 2.0.2 / Isaac Sim 4.5 API；没有同时迁移到新版模拟器。

## 参数和来源

| 参数 | 本版值 | 来源/选择 |
|---|---|---|
| 环境数 | 4096 | 论文 Table III |
| learning epochs / minibatches | 4 / 4 | 论文 Table III |
| 初始 LR / schedule | 5e-4 / fixed | 数值来自论文；固定 schedule 是本次工程选择 |
| gamma / lambda | 0.996 / 0.95 | 论文 Table III |
| actor / critic | 2048→512→128，ELU | 论文 Table III |
| EstNet 估计器 | 1024→256→64→3维速度 | 论文网络规模与 EstNet 任务 |
| 本体 / 命令 / critic | 42 / 7 / 61 | G1环境接口；维数一致不证明观测定义与作者完全相同 |
| 历史 / policy / physics | 50帧0.5s / 100Hz / 1000Hz | 保留现有论文方向实现 |
| horizon | 24 | 常用 locomotion 配置；论文未公布此项 |
| PPO clip / value clip | 0.2 / 开启 | RSL-RL、IsaacLab G1 常用配置；主论文未公布 clip |
| value / entropy / velocity系数 | 1 / 0.008 / 1 | IsaacLab G1参考 / 论文 velocity系数 |
| 初始动作标准差 / max grad norm | 0.8 / 1 | 保留现有工程选择 |
| gait | 周期2/3s，占空比0.5，左右相位差0.5，过渡0.1cycle | 保留现有相位实现；过渡形状与宽度为工程选择 |
| 命令 | vx∈[0.25,0.55]m/s，vy=wz=0 | 平地起步实验选择 |

一轮训练采集4096×24=98,304个转移；每个 minibatch 24,576样本，共4×4=16次 Adam。这里 CLI 的 `iterations` 是采样＋优化的外层轮次，每轮内部有4个 learning epochs，不能将两个计数混为一谈。checkpoint 同时保存 `iteration` 和 `total_optimizer_steps`，完整检查点要求后者等于前者×16。

本次联网核对到的关注度：IsaacLab 8,055★，Unitree RL Gym 3,528★，legged_gym 3,114★，RSL-RL 2,953★，Humanoid-Gym 2,079★，Unitree RL Lab 1,322★（2026-09-07 约01:55–01:57香港时间 GitHub API）。逐仓提交与参数见 `references/` 下两份研究报告及 JSON；关注度不等于在我们的资产上验证成功。

主流参考通常是 LR=1e-3、adaptive、desired_KL=.01、5epochs/4batches；其已核对的 RSL 实现按 KL 调 LR 后继续更新，没有我们此前的模型/Adam回滚。本版为排除 KL 对更新的干预，明确选择固定5e-4、保留论文4epochs，而不是宣称照搬了开源的完整默认值。[RSL-RL v2.1.1 PPO](https://github.com/leggedrobotics/rsl_rl/blob/8682834e98d7a512bf2728ed144cc293dbacc212/rsl_rl/algorithms/ppo.py#L128-L179)，[IsaacLab v2.0.2 G1配置](https://github.com/isaac-sim/IsaacLab/blob/b5fa0eb031a2413c182eeb54fa3a9295e8fd867c/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py#L12-L48)。

Unitree 的普通 PD 动作没有全腿 soft-target 截断；它们会用实际关节偏差/越界奖励项约束姿态。本次遵循用户范围，不把这些髋部L1、越界或 mimic 奖励一并加入。[Unitree G1动作/PD](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L309-L330)，[soft-limit仅作奖励阈值](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L251-L266)。

## 奖励的实际边界

奖励模块与上一版逐字一致。本次没有加入动作幅度惩罚、髋部姿态惩罚、参考动作或脚高轨迹，也没有把已有平滑项全部删掉。保留论文10项函数族及峰值权重（height .2，其余各.1），以及已有的跌倒扣1工程项。

| 核输入 | sigma | 与原文/实现的关系 |
|---|---:|---|
| 3D线速度误差 | .5m/s | 既有加宽，原文印.02，未充分说明归一化 |
| 3D角速度误差 | .5rad/s | 既有加宽，原文印.02 |
| 1−R22 | .1 | 原文公式/文字矛盾后的既有选择 |
| base高度−.78m | .05m | G1适配 |
| 门控支撑脚速之和 | .25m/s | 已明确量纲的工程选择 |
| 门控摆动脚力之和 / mg | .1 | 已有重量归一化与工程尺度 |
| 脚力差范数 / mg | .2 | impact，beta=3 |
| 腿力矩差范数 | 160 | 平滑奖励，beta=2 |
| 腿关节速度差范数 / max(速度,.1) | 8 | 平滑奖励，beta=1；低速分母保护是工程选择 |
| sum(abs(tau*dq)) / (mg*max(速度,.1)) | 1.6 | CoT，beta=3；绝对功率/分母约定 |

这些是软奖励，不会像硬截断那样直接替换动作，但仍可能影响站立与步行的回报取舍。因此本版不是“无任何平滑激励”，也不能称主论文所有未公布细节已精确复现。[论文v1](https://arxiv.org/pdf/2403.05868v1#page=8)。

## 使用方法

先将**本实验整个目录**放入服务器工作目录，使用已有 Isaac Sim 4.5 / Isaac Lab 2.0.2 / Torch2.5.1+cu121 / Python3.10环境。不要在最新IsaacLab3.x环境中直接混用这份适配器。`--help` 不依赖 Isaac 或 GPU。

在实验目录中运行，`GPU_LOGICAL_ID`和`KIT_RENDER_ID`须分别核对到分配GPU的UUID；下面命令要求提前赋值，不默认使用0号卡：

```bash
python train.py --asset /data/nora/toward-understanding-key-estimation/assets/g1_29dof/g1_29dof_rev_1_0.usd \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 4096 --iterations 500 --run-dir logs/estnet-ppo-clip-500
```

这是准备好的500轮使用示例，**不表示已启动本版GPU训练**。运行目录必须新建。外部调度仍按现有授权核实利用率与显存，避开Pro6000D及4090-48-3；不要停止其他人的进程。

评估同版本 checkpoint：

```bash
python run.py evaluate --asset /data/nora/toward-understanding-key-estimation/assets/g1_29dof/g1_29dof_rev_1_0.usd \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 32 --seed 43 --checkpoint logs/estnet-ppo-clip-500/model_00500.pt \
  --run-dir logs/estnet-ppo-clip-eval500
```

`--resume model_00500.pt --iterations 10000`只接受本schema，表示追加9,500轮；恢复模型、Adam、LR和Torch随机状态，从新仿真回合开始，不声称PhysX/历史逐位续接。何时扩大训练应依据真实评估；本次写代码没有启动这条长训命令。

主要输出：配置/源码/资产哈希 `manifest.json`，源机器人属性 `self_collision_readback.json`，每轮指标 `metrics.jsonl`，16次更新与4次KL事件 `ppo_events.jsonl`，每100轮及最终 checkpoint，完成/失败结果 `result.json`。训练任务正常结束不等于学会走路；评估统计首个10秒回合的速度、交替落脚、支撑和滑脚，存活不能替代行走。

可选 `--trace-steps 480 --trace-envs 16`记录有限奖励输入，`--replay-iteration 10`保存一次更新前的独立CPU快照；二者默认关闭。只读分析入口为 `python -m estnet.reward_diagnostics --help` 和 `python -m estnet.replay_diagnostics --help`。

## 验证

CPU环境安装Torch2.5.1和pytest后，在本目录运行 `python -B -m pytest -q tests`。关键检查包含：真实高KL仍满16步、固定LR、Adam/计数/RNG恢复一致、旧schema拒载、打乱原生关节顺序后的仅髋yaw截断、膝目标越softlimit不再被截断，以及CPU替代环境中的真实训练循环与检查点续训。

测试只证明这些代码路径，不能证明物理接触或真实机器人已经行走。最终测试命令、结果和文件SHA见 `verification.json` 与 `source_manifest.json`。
