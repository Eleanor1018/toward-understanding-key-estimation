# FullEst / IrrEst / Implicit：论文核对与 G1 实现契约

本说明依据用户提供的 **arXiv:2403.05868v1**，核对全文提取和 PDF 第 2、4、5、8 页图表。
页码按 PDF 页序计数；不把旧版失败代码、其他论文或本项目的工程参数当成作者原始实现。
原文名称是 **Irrelevant Estimation（IrrEst）**，用户所写的 IllEst 在本说明中对应 IrrEst。

本文规定新增三组的实现契约，区分论文事实、缺失细节和沿用的工程选择；代码与实际运行是否满足契约须另行验证。
它不以源码存在、训练轮次或平均回报代替行走证据。

## 1. 论文明确的三组定义

| 组别 | 显式估计 | 隐式编码 | 原文位置与确定性 |
|---|---|---|---|
| FullEst | 机身线速度、脚附近高度图、机身高度 | 有，维数未单独写出 | 第 5 页 III.C 说结构与显著性分析组相同；第 4 页 III.B 明确三类显式量之外还有固定宽度隐式信息 |
| IrrEst | 仅机身高度 | 明确 16 维 | 第 5 页 III.C 明确估计 base height，并搭配 16 维 latent |
| Implicit | 无 | 明确 16 维 | 第 5 页 III.C 明确没有显式估计，encoder 只输出 16 维 latent |

速度是三维向量，机身高度是标量，见第 3 页速度和高度奖励的变量定义。用 H 表示作者未公布的足周高度图维数，
三组显式输出分别为 **3 + H + 1、1、0**。FullEst 的“全估计”不包括大范围基座高度图：第 2 页 II.B 将它列为
特权信息，并把大图排除在可显式估计的量之外；第 5 页 FullEst 条目列的是脚周高度图。

FullEst 使用 16 维 latent 是为与 Key1、IrrEst、Implicit 保持一致的实现假设。
原文支持 FullEst **有隐式编码**，但不能据此声称“FullEst 的 16 维由该条目直接给出”。

IrrEst 和 Implicit 都不应创建速度估计头，也不应计算速度监督损失。Implicit 同样没有高度或高度图监督头。
这不影响三组共用任务中的速度跟踪奖励，也不妨碍 critic 读取真实速度；奖励/critic 与 actor 的显式估计不是同一接口。

## 2. 公共架构及未充分说明的部分

第 2 页图 2 将历史送入 encoder，隐式编码与显式估计一起送入 actor 和 decoder；actor 还读取当前本体观测。
图注说明 decoder 重建本体观测，显式估计拟合对应物理真值。critic 单独接收当前观测和特权信息。
第 2 页 II.A 说明各部分是全连接网络，并联合使用 PPO、观测预测和估计损失。

三组沿用这个 decoder 框架：FullEst / IrrEst 输入 latent 与各自的显式估计，Implicit 的 decoder 只输入 latent。
这来自公共图 2 的一致实现，三组实验条目没有各自重画 decoder。与之相对，第 4 页 EstNet 条目明确删除隐式编码和 decoder。

| 项目 | 论文提供的信息 | 本轮明示实现选择 |
|---|---|---|
| 历史 | 0.5 秒；图 2 写 `o_{t-1:t-h}` | 正常时刻使用过去 50 帧，排除当前帧；重置后用新 episode 首帧填充缺失历史 |
| 重建时刻 | 图输出标 `o_hat_{t+1}`，图注却说 current，重建虚线连接 `o_t`，三者有冲突 | 沿用 Key1/Key2：过去历史预测当前 `o_t`，不把这个选择说成原文毫无歧义 |
| encoder / backbone | 第 8 页表 III：`[1024,256,64]` / `[2048,512,128]`，ELU | 沿用这些宽度和激活 |
| decoder / critic 隐藏宽度 | 未单独给出 | decoder `[64,256,1024]`；critic `[2048,512,128]` |
| VAE 参数化 | 表 III 有 VAE beta，但未给概率头、先验和重参数化细节 | 共享 encoder 后接 mu/logvar；标准正态先验；logvar 截断 `[-10,10]` |
| latent 的 actor 用法 | 未说明训练/部署采样约定 | actor 使用 mu；训练 decoder 使用重参数采样，评估 decoder 使用 mu |
| 梯度 | 未明确 stop-gradient、detach 或分离更新 | PPO、监督及重建共同更新 encoder，不 detach |

actor 用 mu 是为了让 PPO 重算动作概率时不因重新采样 latent 而改变条件分布。这是本项目的数值实现选择，
不是论文公布的必需机制。当前命令单独拼接到 actor；decoder 不能偷读当前观测或物理真值作为重建输入。

## 3. 本轮维度和高度真值

沿用 [Key1/Key2 文档](KEY1_KEY2.md) 的 G1 观测、命令和高度图工程约定。

| 张量 | FullEst | IrrEst | Implicit |
|---|---:|---:|---:|
| 历史输入 | 50 × 42 | 50 × 42 | 50 × 42 |
| 显式输出 | 速度 3 + 足图 18 + 高度 1 = 22 | 高度 1 | 0 |
| latent | 16，FullEst 维数为明示假设 | 16 | 16 |
| actor 输入 | obs42 + command7 + explicit22 + mu16 = 87 | 42 + 7 + 1 + 16 = 66 | 42 + 7 + 16 = 65 |
| decoder 输入 | z16 + explicit22 = 38 | z16 + height1 = 17 | z16 |
| decoder 输出 | 当前本体观测 42 | 当前本体观测 42 | 当前本体观测 42 |
| critic 输入 | 152 | 152 | 152 |

critic152 为 `obs42 + command7 + velocity3 + bodyheight1 + footmap18 + basemap81`。
所有真值只进入 critic、适用组的监督标签或环境指标；actor 的显式量必须来自历史 encoder 的预测。

机身高度真值为水平地面上的 `base_world_z - ground_world_z`，单位米；它是机身参考点高度，
不是机器人总身高、脚踝高度，也不是固定目标高度 0.78 的常数标签。高度预测和监督 target 均使用 `[batch,1]`，
避免与 `[batch]` 发生静默广播。FullEst 同时有足图和机身高度，这两种监督不可互相替换。

足图每脚 3 × 3，x 偏移 `[-0.1,0,0.1]` m、y 偏移 `[-0.05,0,0.05]` m，先左脚九点后右脚九点；
大图为基座附近 9 × 9、x/y 范围 `[-0.4,0.4]` m、间隔 0.1 m。网格仅随基座 yaw 旋转，
输出各采样原点到地面的世界竖直距离，不裁剪或归一化。足图原点参考各脚踝 link，大图参考 base。
以上范围、点数、排序和参考系都不是论文已给参数，也不能从总特权维数 103 唯一反推。

初试只使用水平平面，`flat_heightmaps` 可以解析查询真实水平地面；同一脚的九个距离相同是平地的退化性质，
不是硬编码标签为零。抬脚会改变足图，基座运动会改变大图。坡面或粗糙地形必须改成逐点真实地表查询。

## 4. 各组损失：原文系数与本轮组合

第 8 页表 III 明确列出速度系数 1、足周高度图系数 0.5、机身高度系数 2、prediction 系数 2、VAE beta 50。
论文没有给完整展开公式，没有给 MSE/KL 的求和或平均规则，也没有说明 beta 与 prediction 系数的嵌套关系。

本轮沿用 Key1/Key2 的实现约定：各 MSE 对 batch 与对应全部特征取 mean；
`KL = 0.5 * mean(mu^2 + exp(logvar) - 1 - logvar)`，对 batch 和 latent 维同时取 mean。
定义 `L_control` 为现有 PPO 策略裁剪损失、value 损失和熵项，三组辅助项分别为：

```text
FullEst:  L_control + 1*MSE_velocity + 0.5*MSE_footmap + 2*MSE_bodyheight
                   + 2*MSE_current_observation + 50*KL
IrrEst:   L_control + 2*MSE_bodyheight + 2*MSE_current_observation + 50*KL
Implicit: L_control + 2*MSE_current_observation + 50*KL
```

这三行是把论文系数落实到本项目的明确约定，不是原文逐字给出的公式。没有相应显式输出的组就不计算对应监督项，
不能保留一个无用 head 然后将 loss 系数设为零，或用零预测/真值填充固定的速度估计接口。
IrrEst / Implicit 的“速度估计 RMSE”应缺省或记为不适用；它们的环境速度跟踪误差仍应正常报告。

## 5. 与现有 EstNet 工程基线共用的部分和比较边界

物理与优化继续沿用 [PLAN.md](PLAN.md) 中的 G1 基线，不为这三组另改奖励或任务难度。
共用 29 活动关节 G1 资产、12 维腿动作和其余关节 PD、动作尺度 0.25、相同软限位和 PD 参数；
共用 100 Hz 策略 / 1 kHz 物理、10 秒 episode、平地正向命令 0.25–0.55 m/s、相同时钟和奖励。
初试仍关闭噪声、延迟、域随机化与复杂地形；不增加 mimic、参考轨迹或专门的脚高奖励。

共用 PPO / GAE / timeout 处理、4096 环境、24 步 rollout、每次更新内部 4 epochs / 4 minibatches、
初始学习率 5e-4、gamma 0.996、lambda 0.95，以及原基线的 clipping、熵、梯度范数和自适应学习率约定。
共用每回合 reset、终帧 critic 在 reset 前保存、训练标签在 env.step 前复制的接口。
不同架构应使用独立检查点 schema，不能拿 EstNet 权重或优化器状态冒作这些组从零训练的结果。

现有 EstNet 的 critic 是 61 维、历史包含当前帧；本轮三组沿用 Key 系列的 critic152 与排除当前帧的历史。
这些差别必须保留在比较记录中：即使 EstNet 已有成功行走结果，也不能把现有日志直接视为
只改变“显式量/latent”的严格消融。要做严格比较，需要另开统一 critic 和历史时序的 EstNet 对照。
本文不改动已经运行或冻结的 EstNet 源码，也不重新定义其成功标准。

500 次外层 iteration 的初试与论文第 5 页图 5 的 10000 iterations 不是同等训练预算；
表 III 的 4 learning epochs 是每轮采样后的内部优化次数。论文未公布 rollout 长度，不能仅凭迭代数声称总样本数相同。

三组应按相同的独立均值动作、首 episode 评估记录存活、速度跟踪、支撑滑移、离地和交替落地，并结合视频判断迈步。
Implicit / IrrEst 不含速度监督，不意味着它们一定不会行走；FullEst 的监督更多，也不保证胜过较简单的组。
实际趋势必须以本轮检查点评估为依据，论文中的排序不能代替 G1 实验。
