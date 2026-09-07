# 常用机器人 PPO：固定版本核对与本次新基线

按用户最终要求，本次保留自碰撞、hip yaw目标±20°、raw动作±100与scale .25、物理关节/力矩/速度限位及原EstNet网络、奖励和PD；同时取消全腿按soft joint limits裁剪PD目标，采用固定学习率`5e-4`与`clip=0.2`，移除KL驱动LR调整、更新拒绝、提前结束和回滚，KL仅作记录。**这是动作映射与PPO更新策略同时变化的新基线，不是只改变KL的一项因果对照。** 主流配置支持clip=.2，但通常使用adaptive LR，不能说本次固定LR就是它们的完整默认方案。

## 资料与版本

GitHub API取样为香港时间 **2026-09-07 01:55:39–47**（UTC前一日17:55）。星标是当时计数，不代表效果排名；各项目存在继承关系，也不是四份独立算法验证。原始API结果、完整提交号和文件SHA均保存在配套JSON。

| 官方仓库 | 星标 | 取样时默认分支 | 固定提交 / 提交日期UTC |
|---|---:|---|---|
| [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 3,114 | `master` | [8fa29acc6fd1](https://github.com/leggedrobotics/legged_gym/commit/8fa29acc6fd1910c3d9659eef6310bdd301cde0a) · 2025-05-29 |
| [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | 2,953 | `main` | [00e13d1aa49b](https://github.com/leggedrobotics/rsl_rl/commit/00e13d1aa49b398ae512f1765297f7ab8c50ca07) · 2026-08-31 |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 8,055 | `release/3.0.0-beta2` | [99f1423e5d4a](https://github.com/isaac-sim/IsaacLab/commit/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1) · 2026-09-04 |
| [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | 1,322 | `main` | [4960b84732b0](https://github.com/unitreerobotics/unitree_rl_lab/commit/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) · 2025-11-19 |

计数来源为各仓库的GitHub API，例如[IsaacLab官方API](https://api.github.com/repos/isaac-sim/IsaacLab)。后续页面数字可能变化。

## 实际运行配置，不混同底层构造函数默认

| 参数 | legged_gym任务基类 | IsaacLab G1粗糙/平地：v2.0.2及当前默认分支 | Unitree locomotion基类 | 本次新基线 |
|---|---:|---:|---:|---:|
| PPO clip ε | .2 | .2 | .2 | .2 |
| value clip | True | True | True | 保留当前实现 |
| horizon | 24 | 24 | 24 | 24 |
| epochs / minibatches | 5 / 4 | 5 / 4 | 5 / 4 | 4 / 4 |
| 初始LR / schedule | 1e-3 / adaptive | 1e-3 / adaptive | 1e-3 / adaptive | 5e-4 / fixed |
| γ / λ | .99 / .95 | .99 / .95 | .99 / .95 | .996 / .95 |
| desired KL | .01 | .01 | .01 | 仅诊断，不控制更新 |
| max grad norm | 1 | 1 | 1 | 保留1 |
| entropy coefficient | .01 | .008 | .01 | 保留原值 |
| value coefficient | 1 | 1 | 1 | 保留原值 |

表中数值逐项由固定源码AST提取核对：[legged_gym](<https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot_config.py#L202-L238>), [IsaacLab v2.0.2 G1](<https://github.com/isaac-sim/IsaacLab/blob/b5fa0eb031a2413c182eeb54fa3a9295e8fd867c/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py#L12-L48>), [IsaacLab当前G1](<https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py#L12-L60>), [Unitree locomotion](<https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py#L11-L36>)。Unitree的G1-29dof-Velocity任务明确[注册该BasePPORunnerCfg](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py#L3-L11)。IsaacLab平地类继承同一PPO算法参数，只另改训练轮数和网络宽度。legged_gym的`learning_rate=1e-3`后虽有`#5.e-4`注释，生效值仍是`1e-3`。

RSL-RL本身是算法库，不自带机器人任务horizon。当前5.5和IsaacLab当前依赖的5.0.1构造函数默认5epochs/4minibatches、gamma.99、adaptive；老v1.0.2与v2.1.1构造函数则默认1epoch/1minibatch、gamma.998、fixed，具体机器人任务会覆盖它们。因此不能把旧库的构造默认冒充legged_gym实际训练参数。[当前构造函数](<https://github.com/leggedrobotics/rsl_rl/blob/00e13d1aa49b398ae512f1765297f7ab8c50ca07/rsl_rl/algorithms/ppo.py#L39-L61>), [v1.0.2](<https://github.com/leggedrobotics/rsl_rl/blob/2ad79cf0caa85b91721abfe358105f869a784121/rsl_rl/algorithms/ppo.py#L36-L56>), [v2.1.1](<https://github.com/leggedrobotics/rsl_rl/blob/8682834e98d7a512bf2728ed144cc293dbacc212/rsl_rl/algorithms/ppo.py#L19-L34>), [v5.0.1](<https://github.com/leggedrobotics/rsl_rl/blob/3ac56acd3376f2952eb636a133f4b5aa30142552/rsl_rl/algorithms/ppo.py#L35-L55>)。

## clip、KL和回滚分别做什么

PPO clip作用于损失中的新旧策略概率比，`.2`对应裁剪分支的`.8–1.2`。它不是动作限幅，也不保证每个样本最终概率比都在这个范围，更不保证KL小于某个值。value clip是另一个损失操作：裁剪相对旧value的差值，并取裁剪/未裁剪平方误差的较大者；不是将value输出硬限在±.2。[当前PPO实现](<https://github.com/leggedrobotics/rsl_rl/blob/00e13d1aa49b398ae512f1765297f7ab8c50ca07/rsl_rl/algorithms/ppo.py#L268-L286>)。

这些版本的adaptive KL只控制学习率：当mini-batch平均KL大于`2×desired_kl`，LR除以1.5、最低1e-5；当`0<KL<desired_kl/2`，LR乘以1.5、最高1e-2。之后仍继续当前mini-batch的反向传播和Adam更新。**已核对的v1.0.2、v2.1.1、v5.0.1、当前5.5没有基于KL的硬早停、更新拒绝或模型+Adam回滚。** 这不等于其他PPO项目都没有这些机制。[旧实现完整update](<https://github.com/leggedrobotics/rsl_rl/blob/2ad79cf0caa85b91721abfe358105f869a784121/rsl_rl/algorithms/ppo.py#L120-L185>), [v2.1.1](<https://github.com/leggedrobotics/rsl_rl/blob/8682834e98d7a512bf2728ed144cc293dbacc212/rsl_rl/algorithms/ppo.py#L128-L179>), [v5.0.1](<https://github.com/leggedrobotics/rsl_rl/blob/3ac56acd3376f2952eb636a133f4b5aa30142552/rsl_rl/algorithms/ppo.py#L268-L379>), [当前实现](<https://github.com/leggedrobotics/rsl_rl/blob/00e13d1aa49b398ae512f1765297f7ab8c50ca07/rsl_rl/algorithms/ppo.py#L240-L311>)。

数值相同也不代表实现完全一致：旧RSL对actor+critic合并参数做grad-norm裁剪，新5.x分别对actor、critic裁剪。此次不要借移除额外KL机制，同时切换这类基础实现细节。[旧梯度步骤](<https://github.com/leggedrobotics/rsl_rl/blob/8682834e98d7a512bf2728ed144cc293dbacc212/rsl_rl/algorithms/ppo.py#L174-L179>), [新梯度步骤](<https://github.com/leggedrobotics/rsl_rl/blob/00e13d1aa49b398ae512f1765297f7ab8c50ca07/rsl_rl/algorithms/ppo.py#L308-L311>)。

## 与现有Isaac Sim 4.5环境的关系

- **legged_gym**当前README明确Isaac Gym Preview3和RSL v1.0.2，是参数参考，不能当作Sim4.5项目直接移植。[安装说明](<https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/README.md#L22-L35>)。
- **IsaacLab v2.0.2**明确torch2.5.1、Sim4.5.0，RSL依赖为`>=2.1.1`，并非精确锁版本；本报告审v2.1.1仅表示审其下界，不推断用户安装了该版本。[官方依赖](<https://github.com/isaac-sim/IsaacLab/blob/b5fa0eb031a2413c182eeb54fa3a9295e8fd867c/source/isaaclab_rl/setup.py#L20-L47>), [运行版本标记](<https://github.com/isaac-sim/IsaacLab/blob/b5fa0eb031a2413c182eeb54fa3a9295e8fd867c/source/isaaclab_rl/setup.py#L71-L76>)。
- **IsaacLab当前默认分支**已是3.0.0-beta2，目标Sim6.0/6.0.1，固定RSL5.0.1及torch≥2.10；不是当前RSL main5.5。官方README中的`main`行与默认beta分支是两回事。[兼容表](<https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/README.md#L67-L80>), [依赖](<https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab_rl/setup.py#L20-L57>)。
- **Unitree当前README**标Sim5.1/Lab2.3，URDF路径要求Sim≥5；setup仍残留Sim4.5 classifier。不能据此声称整个当前版本已在4.5验证，故本次只借鉴参数，保留现有资产与runtime。[README](<https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/README.md#L1-L58>), [setup](<https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/setup.py#L29-L37>)。

## 本次如何保持结论可解释

论文第8页Table III明确4096环境、4learning epochs、初始LR5e-4、γ.996、λ.95、4batches；表中没有clip、horizon、schedule或KL回滚。因此保留这些已知参数与原网络、奖励是合理的；**固定schedule和clip=.2是明示工程选择，不能补写成作者公布过的细节。** [用户论文v1，Table III](https://arxiv.org/pdf/2403.05868v1#page=8)。

4096×24得到每轮98,304个transition；4个minibatch，每个24,576样本；4epochs一共16次Adam步骤。学习率始终5e-4；取消KL驱动LR变化、KL阈值拒绝/早停/回滚，保留KL、ratio、clip fraction、梯度范数和非有限数失败记录。自碰撞与hip yaw目标±20°照旧，其他腿关节不再由soft limits额外裁剪PD目标，但物理机械限位与执行器限制仍有效；原始采样动作和log-prob路径不改。是否真正行走，仍以同seed、同命令的确定性mean-policy评估、足部姿态与连续视频判断，不能由训练奖励上涨或星标数推出。

这次能回答的是“这个新基线在相同评估条件下是否改善行走”。因为同时调整了学习率schedule、KL限制和soft-target裁剪，结果不能单独证明KL控制或关节目标裁剪是旧方案失败的原因；也不保证放开这些限制就能学会走路。
