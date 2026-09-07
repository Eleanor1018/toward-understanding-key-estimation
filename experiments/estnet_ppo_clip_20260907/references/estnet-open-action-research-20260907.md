# 同类开源项目的动作与关节约束核对

查证时间：2026-09-07 01:56–01:57 香港时间。星标来自本次 GitHub API，代码链接固定到实际提交。星标用于说明项目关注度，不代表这些配置已在我们的机器人上复现成功。

| 仓库 / 当前API记录 | 星标 | 本次提交 | 默认分支 |
|---|---:|---|---|
| [unitreerobotics/unitree_rl_lab](https://api.github.com/repos/unitreerobotics/unitree_rl_lab) | 1,322 | `4960b84732b0` | main |
| [unitreerobotics/unitree_rl_gym](https://api.github.com/repos/unitreerobotics/unitree_rl_gym) | 3,528 | `276801e46c5d` | main |
| [roboterax/humanoid-gym](https://api.github.com/repos/roboterax/humanoid-gym) | 2,079 | `ae46e201c85a` | main |
| [isaac-sim/IsaacLab](https://api.github.com/repos/isaac-sim/IsaacLab) | 8,055 | `99f1423e5d4a` | release/3.0.0-beta2 |

最直接的参考是 Unitree 自家的 G1 29DOF 速度任务：普通位置动作采用默认姿态加 0.25 倍输出，自碰撞开启；髋 yaw/roll 的约束还依赖姿态偏差惩罚。所查 G1 速度任务都没有明确指定髋 yaw ±20°。因此 ±20°应写成这次实验的工程初值，不能写成官方推荐值。[动作](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py#L180-L185)、[自碰撞](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py#L24-L47)、[髋部惩罚](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py#L288-L292)。

**实际代码的区别**

| 项目 | 动作到PD目标 | 关节约束 / 自碰撞 |
|---|---|---|
| Unitree RL Lab，G1 29DOF | 所有关节 scale=0.25，叠加默认姿态；此速度任务未设置专门 hip yaw cap | 自碰撞 True；soft limit factor=0.9；实际髋 roll/yaw 相对默认姿态 L1 惩罚 −1；实际关节越 soft limit 惩罚 −5 |
| Unitree RL Gym，G1 12DOF | raw action 先裁 ±100，再乘0.25加默认姿态；显式PD后裁力矩 | self_collisions=0 表示启用；soft limit factor=0.9 用于实际关节位置惩罚；髋 roll/yaw 平方惩罚 −1 |
| Humanoid-Gym，XBot-L | 自带 Normal 策略；raw action 裁 ±18，再乘0.25；显式PD后裁力矩 | 自碰撞启用；有默认姿态、脚/膝间距和解析参考关节姿态奖励；不是我们的 G1 论文基线 |
| IsaacLab 当前 G1_MINIMAL | 继承统一 scale=0.5＋默认姿态的普通位置动作 | 此资产自碰撞 False；髋 roll/yaw L1 惩罚 −0.1；踝实际越界惩罚 −1；并非相同29DOF rev1.0资产 |

表内 Unitree Gym 的动作链可以直接追到 [raw裁剪](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L49-L63)、[±100配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot_config.py#L128-L135)、[PD及力矩裁剪](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L309-L330)、[G1配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_config.py#L3-L87)；索引式髋惩罚见[函数](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_env.py#L122-L123)。Humanoid-Gym 的 [Normal与确定性均值评估](https://github.com/roboterax/humanoid-gym/blob/ae46e201c85a2b17e7f2cea59a441dae7ea88a8f/humanoid/algo/ppo/actor_critic.py#L110-L124)、[±18配置](https://github.com/roboterax/humanoid-gym/blob/ae46e201c85a2b17e7f2cea59a441dae7ea88a8f/humanoid/envs/custom/humanoid_config.py#L218-L227)、[0.25缩放](https://github.com/roboterax/humanoid-gym/blob/ae46e201c85a2b17e7f2cea59a441dae7ea88a8f/humanoid/envs/custom/humanoid_config.py#L118-L128)、[参考动作奖励](https://github.com/roboterax/humanoid-gym/blob/ae46e201c85a2b17e7f2cea59a441dae7ea88a8f/humanoid/envs/custom/humanoid_config.py#L188-L216) 均在作者仓库。IsaacLab 的 [scale=0.5](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L154-L159)、[髋与踝惩罚](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py#L54-L65)、[不同资产/自碰撞设置](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L274-L313) 不能与 Unitree 29DOF 配置混用。

**soft limit 惩罚不是目标硬裁剪。** Unitree Gym 把原始机械范围缩成0.9倍，只存作奖励阈值，回传的物理 DOF props 未被该段改写。奖励检查实际 q 是否越过阈值；PD目标未先按该阈值截断。IsaacLab 普通 JointPositionAction 同样先做仿射变换，只有显式设置 clip 才截断处理后的目标，不会自动套用 soft limits。[阈值生成](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L251-L266)、[实际q惩罚](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L678-L682)、[普通位置动作](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py#L169-L201)。

**缩放与有界策略也不同。** 缩小 action_scale 不能给无界高斯设定输出范围。IsaacLab 支持逐关节 scale，也另外提供 JointPositionToLimitsAction：把输入裁到[-1,1]后映射到 soft limits，但所查 G1 速度任务没有采用这个算子。Unitree 文件中的 `0.25 × effort / stiffness` 逐关节缩放属于另一个 MIMIC 配置，不能声称是同仓库速度任务默认方法。[逐关节API](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py#L29-L61)、[有界算子](https://github.com/isaac-sim/IsaacLab/blob/99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py#L152-L173)、[MIMIC专用缩放](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py#L706-L717)。两份 Unitree 环境代码把策略实现交给外部 RSL-RL；本报告不把没有锁定版本的依赖推断成已核实的完整策略分布。[依赖声明](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/setup.py#L5-L11)。

**本轮采用的最小动作方案**

新基线同时移除自定义 KL 硬限制干预，以及额外的全腿 soft-target clamp。保留自碰撞、普通高斯动作、raw ±100 数值保护、0.25 缩放、原默认姿态/PD/机械关节限位/力矩速度限制；仅左右 hip yaw 的目标偏移保留 ±20°。其余目标不再人为卡到 soft limit；不新增 L1髋部奖励、动作惩罚或 mimic。上游确实存在这些奖励，但这不构成本轮自动加入它们的理由。

具体顺序：`a = clip(raw, −100, 100)`；`q_requested = q_default + 0.25 × a`；只把两侧 hip yaw 的 `q_requested` 限制在 `q_default ± 0.34906585 rad`，其它目标原样交给PD。这个限制针对目标，不保证实际关节每一物理步都严格在±20°内。

把 yaw 的目标裁剪挪到 raw 动作 ±1.3962634，在同一0.25缩放和默认偏移下是等价的，不能仅凭移动裁剪位置解决网络原始均值漂移。既有轨迹中的髋输出远超目标范围、膝目标长期卡下界，说明出现了饱和；它们不足以单独证明不能走的根因。这里同时改了 KL 干预与全腿目标裁剪，所以属于新训练基线，不能当作单变量因果实验，也不能预先保证能正常行走。

本次只进行了网页/源码阅读；没有改动训练项目、启动GPU、修改模型或奖励。所有原始文件的提交与SHA记录在同名JSON中。
