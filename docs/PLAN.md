# EstNet 重建方案与参数登记

2026-09-06。先做平地向前步行，之后再检验噪声、延迟、地形和鲁棒性。

## 1. 这次复现的边界

目标论文为用户提供的 [Toward Understanding Key Estimation in Learning Robust Humanoid Locomotion，2403.05868v1](https://arxiv.org/abs/2403.05868v1)。其 EstNet 消融明确不使用 latent 和 decoder，只估计机体线速度；论文中的各组方法都学会了基本运动。因此应先验证速度估计、策略、控制和步态奖励这条最短路径。

原论文机器人是 Wukong IV，当前是 G1 29DOF、Isaac Sim 4.5 / Isaac Lab 2.0.2。本文称新代码为 **EstNet 架构与论文奖励形式在 G1 上的移植基线**，不是作者未公开配置的逐位复刻。第一阶段简化任务的结果，也不能当作论文的地形与实机鲁棒性复现结果。

## 2. 论文疑点与缺失信息

| 项目 | 原文证据／问题 | 本轮处理 |
|---|---|---|
| 高斯核符号 | 式1排版没有负号，与“误差越小奖励越高”不符 | 明确采用 `alpha*exp(-(error/sigma)^2)` |
| 速度／角速度 sigma | 式3/4为 `.02`，未说明是否先做额外归一化 | 不把它直接当原始单位可用宽度；基线两者用`.5` |
| 数值尺度 | 按上述核，速度误差`.1m/s`时`.1*exp(-(.1/.02)^2)=1.39e-12` | 这是回报几乎为零，不是说PPO必须直接对奖励函数求导；更宽误差区间能提供学习信号 |
| 直立误差 | 式5写`1-R22²`，文字为`1-R22`；sigma`.0025` | 采用文字约定`1-R22`，sigma`.1`；另由跌倒条件排除翻倒 |
| 脚力/脚速 | 式7/8都写sigma8，量纲、缩放和具体Q函数不全 | 速度用m/s；力除总重mg；明确登记新尺度，不能声称“找回作者参数” |
| gait command | 主论文command7维，但未逐项列出 | 新接口明列速度3维＋双足sin/cos4维；7维相同不证明编码相同 |
| gait周期/占空比/相位/过渡 | 主论文继承[38]，自身没有完整数值 | 引用论文的1.5Hz、支撑摆动各半；相位差半周期，线性过渡宽度`.1 cycle`为工程选择 |
| 零速度 | 式11/12除以速度，没有epsilon | 分母`max(norm(v),.1m/s)`，防止NaN；同时承认会改变低速成本 |
| 功率 | 式12点积可能允许正负功率相互抵消 | `sum(abs(tau_i*dq_i))`，明确是绝对机械功率约定 |
| 网络/优化 | 公布encoder/backbone宽度；未单列EstNet、critic宽度、梯度截断、优化器更新细节 | 明确3维head、critic尺寸、联合梯度；不推断latent存在 |
| PPO细节 | 未公布rollout长度、clip、熵系数、value系数、KL控制和优势归一化 | 借鉴固定版本RSL-RL/IsaacLab，见下表 |
| 控制/物理 | 未给可移植到G1的PD/action scale/限幅/名义姿态/精确资产 | 用同型号Unitree配置和资产哈希，先做静态标定 |
| 历史 | 明确0.5秒但未详述采样步长/重置/噪声与延迟的完整实现 | 100Hz取连续50帧，重置后用首帧填满，不跨episode |
| episode/奖励单位 | 没给足够信息保证曲线回报可直接对齐 | 新任务10秒，各项每策略步相加，不另乘dt；不与论文约1300回报直接比较 |

主论文表III的环境数4096、初始LR5e-4、4轮更新、4个minibatch、gamma.996、lambda.95，以及速度监督系数1，予以保留。`4 minibatches`不是“每个batch只有4个样本”。

## 3. gait 的出处和完整实现

[38] 为 Wei 等人的 [Learning Gait-conditioned Bipedal Locomotion with Motor Adaptation](https://www.researchgate.net/profile/Wang-Zhicheng-6/publication/376349059_Learning_Gait-conditioned_Bipedal_Locomotion_with_Motor_Adaptation/links/65731b766610947889ab3223/Learning-Gait-conditioned-Bipedal-Locomotion-with-Motor-Adaptation.pdf)，此版本为作者上传。其第3页描述线性插值切换，第4页图4给出每腿1.5Hz、支撑/摆动各半。它本身的command是8维，也没有完整列出编码，不能把它冒认为主论文7维命令的逐元素说明。

本轮固定：`T=2/3s`、duty`.5`、left offset0/right`.5`；每次reset随机共同起始相位。`phase=t/T+offset`，足相位0/.5是切换中点，.25完全支撑、.75完全摆动；过渡全宽`.1周期`。每腿的 `Q_velocity=stance`，`Q_force=1-stance`，全部在[0,1]。策略与critic都看到同一个时钟；命令是 `[vx,vy,wz,sinL,cosL,sinR,cosR]`。周期固定，因而不需要额外频率输入；以后随机频率必须修改观察契约。

其祖先 [Siekmann等，Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/pdf/2011.01387) 描述周期状态与相位偏移进入策略。作者项目 [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py) 同样让支撑概率门控脚速、其补集门控脚力。但该项目`exp(-x²/sigma)`的sigma是分母，不是本方案核中的标准差；其训练脚本还覆盖默认值，不能只拷一个配置类的数字。

旧`.8s`也不是“错误到不能走”：Unitree另一个已开源G1任务确实采用`.8s`。问题在于它不是从本论文完整查证的取值，而且早期策略看不到对应时钟。新方案把来源和缺项写明。

## 4. G1、观察和动作

机器人动力学沿用29个活动关节，只让策略控制两腿12关节；其余17个关节PD追踪默认姿态，**没有焊死或删除上身**。这减少初期动作搜索维度，是本轮G1工程适配，不能仅依据论文obs42就认定作者也这样固定上身。

| 张量 | 维数 | 内容 |
|---|---:|---|
| obs | 42 | projected gravity3；角速度3×.25；腿q减默认12；腿dq12×.05；上次raw action12 |
| history | 50×42 | 连续0.5秒，仅现实可获取本体量；无真实线速度 |
| command | 7 | 速度命令3＋相位编码4 |
| estimator | 3 | 机体坐标系线速度m/s；MSE对同一时刻真实速度，mean跨batch及3坐标 |
| actor input | 52 | obs42＋command7＋estimated velocity3 |
| critic input | 61 | clean obs42＋command7＋true velocity3＋base height1＋foot force6/mg＋foot height2 |
| action | 12 | 普通Gaussian样本，无tanh；`q_target=q_default+.25*action`，再按USD软关节限位裁剪 |

观察固定物理尺度，速度命令与估计都保留m/s；不做随训练改变的running normalization。首轮关闭观察噪声、延迟、域随机化和terrain curriculum，只用平地、每episode固定向前`.25–.55m/s`，vy/wz=0；不采样站立。此简化专门隔离基础步行，后续单独恢复论文随机化。

同型号官方参数来自 [Unitree RL Lab 固定提交4960b847](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)。旧项目这组PD基本一致，不能把它认定为旧训练失败的原因。

| 关节 | Kp/Kd | 力矩上限Nm | 速度上限rad/s |
|---|---|---:|---:|
| hip pitch/yaw |100/2|88|32|
| hip roll |100/2|139|20|
| knee |150/4|139|20|
| ankle pitch/roll |40/2|25|37|
| waist yaw |200/5|88|32|
| waist roll/pitch |40/5|25|37|
| shoulder/elbow/wrist roll |40/1|25|37|
| wrist pitch/yaw |40/1|5|22|

armature全部`.01`。两腿默认hip pitch−.1、knee.3、ankle pitch−.2，其余0；上身默认按同官方配置。spawn root`.8m`，但同款 [官方locomotion reward](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) 的高度目标为`.78m`。本轮以`.78`为待物理预检的起点，不把生成高度误作站稳高度。

raw action clip100借鉴 [Unitree RL Gym默认限幅](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot_config.py)，最终仍有真实关节软限位和力矩限位。`scale=.25`不等于动作限制±1：例如膝action2对应`.8rad`目标，而旧tanh限幅最多`.55rad`。这些主流项目的机器人、输入与频率并不完全相同，不能将其配置整体等同本方案。

## 5. EstNet 与 PPO

历史MLP `[1024,256,64]→3`；actor `[2048,512,128]→12`；critic `[2048,512,128]→1`；ELU。主论文表III给出了encoder/backbone，critic同宽与EstNet继承该encoder宽度是明示假设。

`L = clipped_PPO + 1*clipped_value_loss - .008*entropy + 1*mean_squared_velocity_error`。
PPO梯度经过actor输入回传estimator；每个minibatch重算整个分布。主论文总loss表述支持这种实现，但没有明确公布EstNet的detach，因此不能声称它就是作者原始代码。

[Ji等，Concurrent Training…，2202.05481v2](https://arxiv.org/html/2202.05481v2) 的完整估计器还预测足高/接触，不能把它的完整模型等同2403的“仅速度EstNet”消融；其论文没有补齐本问题的梯度截断细节。作者仓库是部署代码，不能当作已核验的训练发布。

| 参数 | 本轮固定值 | 来源 |
|---|---|---|
| env/horizon |4096/24|4096论文；24借鉴IsaacLab，100Hz下0.24秒 |
| epochs/minibatches |4/4|论文 |
| initialLR |5e-4|论文 |
| gamma/lambda |.996/.95|论文 |
| clip/value clip |.2/.2|RSL-RL |
| value/entropy coef |1/.008|IsaacLab G1 |
| initial std |.8|参考Unitree RL Gym G1；结合本轮`.25`动作尺度 |
| logstd bounds |[-3,1]|工程稳定界，非退火到小探索 |
| KL / LR bounds |.01 / [1e-5,1e-3]|KL参考RSL-RL；边界工程选择 |
| LR规则 |KL高于2倍目标除1.5；低于.5倍且>0乘1.5|借鉴双向自适应；无旧rollback |
| maxgrad/weightdecay |1/0|主流设置/简化选择 |
| optimizer |Adam，共同loss，默认betas/eps|实现约定 |
| advantages |整批均值0、标准差1|明确实现约定 |
| returns |不标准化|明确实现约定，记录value loss观测实际尺度 |

固定主来源：[IsaacLab v2.0.2 G1 PPO](https://github.com/isaac-sim/IsaacLab/blob/v2.0.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py)、[RSL-RL v2.2.4 PPO](https://github.com/leggedrobotics/rsl_rl/blob/v2.2.4/rsl_rl/algorithms/ppo.py)。本方案保留主论文100Hz/1kHz，而Unitree常用50Hz/200Hz；没有把50Hz的gamma直接搬来。

GAE对真实跌倒不bootstrap，对超时使用**reset前终帧**价值；两者都切断trace。没有照搬RSL-RL旧版本近似用V(s_t)处理timeout的路径。

## 6. 奖励的保留与改动

保留论文10项核结构及相对峰值权重：height`.2`，其余各`.1`，总峰值1.1/步。下列sigma都指核中`error/sigma`的尺度，不是方差。

| 项目 | 输入约定 | sigma | 与原文关系 |
|---|---|---:|---|
| linear |3D线速度误差norm|.5m/s|宽度改为稠密，参考主流tracking尺度 |
| angular |3D角速度误差norm，目标[0,0,wz]|.5rad/s|同上 |
| upright |1-R22|.1|处理公式矛盾并加宽 |
| height |base高度减.78m|.05m|G1适配、待标定 |
| stance velocity |sum(Qv*foot_speed)|.25m/s|量纲明确后的工程选择 |
| swing force |sum(Qf*norm(foot_force))/mg|.1|重量归一后的工程选择 |
| impact |norm(delta foot forces)/mg|.2，beta3|保留原文 |
| torque smoothness |norm(delta legtorques)|160，beta2|保留原文尺度，维数适配 |
| dq smoothness |norm(delta legdq)/max(norm(v),.1)|8，beta1|保留尺度，增加分母保护 |
| CoT |sum(abs(tau*dq))/(mg*max(norm(v),.1))|1.6，beta3|保留尺度，明确功率/分母约定 |

此外真实跌倒扣1，超时不扣，这是单独登记的工程项。第一轮没有mimic、参考关节轨迹、脚高目标奖励、额外progress bonus或大动作惩罚；若后续必须增加任何一项，应另开实验编号，不能悄悄改称同一基线。

静态反例检查（不是动态rollout）：cmd`.4m/s`、各姿态项理想时，新核下站立约`.9566`、双脚同速蹭行约`.9319`、理想单支撑约`1.1`。这表明该构造下蹭行不再胜过站立；也诚实暴露站立仍有较高回报，**不能凭排序宣布已经解决不走**。仍须依赖实际探索、运动可行性和闭环测量。

## 7. 按证据推进，不再盲跑10000次

1. 静态检查固定资产4个哈希、29关节/脚部名字、运行时版本。记录GPU/驱动、代码快照、参数、种子。
2. 16环境默认PD姿态3秒，每个环境仅测首回合，不用重置后的帧补证据。最后1秒逐环境检查高度波动<.01m、距目标≤.03m、双支撑>.95、水平速度和支撑滑速均<.05m/s；没有有效稳态窗口时返回空统计。零动作不稳记为`nominal_pose_unstable`，不单独阻断PPO训练。脚底几何、碰撞和力矩饱和须结合实际状态查看；CPU测试不能代替。
3. 从随机初始化做短步行试训，默认计划500次，每100次保存。先检查数值、动作/力矩限幅、reward分项、速度估计RMSE、KL和相位对齐，再决定是否扩大训练。
4. 固定checkpoint用均值动作评估首个完整10秒episode；记录存活、vx比率、速度误差、单双支撑、相位偏差、支撑脚滑动、离地时间及左右交替落地。跌倒重置后的帧不补算原episode。
5. 工程行走gate：完整存活；速度误差<.2m/s；vx/command在`.7–1.3`；双支撑占比<.8；10秒至少6次合格交替落地；支撑脚滑速<.15m/s；双方脚踝高度在排除首秒后各有>.03m变化。落地要求此前腾空`.08–.45s`。这些是本轮检查标准，不是作者论文指标，最终还要看视频中的实际迈步。
6. 单个种子出现步行后换至少两个种子，并扩展速度。先恢复噪声/延迟，再恢复域随机化与地形。能走之后才谈论文鲁棒性和EstNet对照Key1等结构。

## 8. 归档和当前验证状态

原工作树91个文件全部移至`archive/2026-09-06-pre-estnet/`，移动前后SHA-256一致，Git历史保留。旧训练脚本、模型、mimic、tests与说明保留原相对目录；新代码不import归档。原本地目录没有实际checkpoint、训练日志、USD或NPZ，因此此次没有归档远端的3500/10000训练结果。

本地已完成纯CPU核函数/梯度/GAE/动作映射验证，运行时验证将单独记录。用户已授权从其原服务器清单选择空闲GPU并配置缺失环境；仿真和训练状态以新运行的manifest、JSONL与result为准，不以代码存在为完成证据。

2026-09-06 04:19（香港时间）在4090-6的GPU7启动pilot-001：4096环境、500次更新、seed42。
04:24已观测到62次实际PPO更新，每次约5秒。使用冻结的`pilot-source-001`代码；并行完成的入口审阅修正
用于后续运行和评估，未热替换正在训练的源码。选择计算/渲染GPU7，不因原生插件初始化临时CUDA上下文出现在其他卡而停止。
此记录只证明试训已运行，不证明已经形成步态。后续状态见实际日志与独立评估。

04:38更新：试训已完成225次更新，第100/200轮检查点已保存。独立修正版入口对第100轮检查点的32环境评估
已正常完成，存活满10秒比例0，工程行走gate比例0，平均回合1.7797秒；尚未学会稳定行走。
真实评估同时发现并修复Isaac Sim 4.5快速关闭导致结果文件缺失的问题：关闭前保存测量并标记cleanup_pending，
禁用fast_shutdown，清理返回后再提交正常最终结果。该修正用于独立评估，不修改运行中的训练快照。

用户随后授权将总训练轮次追加至10000。2026-09-06 05:19:43在同服务器GPU7从第500轮检查点续训，
独立目录为`logs/resume-001-to10000`，冻结源码为`resume-source-001`；仅追加501–10000共9500轮。
模型、完整Adam、自适应LR=1e-5及更新计数500已通过真实检查点CPU预检；任务/PPO配置沿用保存值。
旧检查点缺CUDA RNG与PhysX/历史现场，续训从新回合开始，不承诺与不中断仿真逐位一致。
论文第5页图5是10000 iterations，第8页表III Learning epochs=4，故保留内部epochs4；
论文rollout长度未公布，迭代数对齐不等于总采样量完全相同。
