# EstNet / Key1 / Key2 / FullEst / IrrEst / Implicit：Blackwell 六路线 10000 轮版

2026-09-07。本版将六条路线放在同一套 Isaac Sim 5.1 / Isaac Lab 2.3.2 运行协议中，默认各自从零训练到总计 **10000 轮**。EstNet仅有显式速度估计；另外五条路线保留各自估计头、16维latent和decoder。本发布验证是CPU代码验证，不含新机器的GPU部署状态或真实行走结论。

本轮是六组 **fresh训练（0→10000）**，不加载已有500轮或其它运行时检查点。500只是普通保存点，不因到达500或当时步态未通过而自动停训；代码仍按既定目标继续到10000。只有明确运行故障才会写失败状态，不把故障伪报为完成。

入口与训练循环都补了中文注释。五个专用入口固定相应VAE路线，`train.py --variant estnet`运行EstNet，实际代码共享 `estnet/run.py`、`environment.py` 和 `ppo.py`，因此动作、奖励或续训修复不会漏掉其中某一路。六条路线均是本次正式fresh训练目标。

## 六条路线

| 路线 | 训练文件 | 显式估计头 | Actor输入维度 | Decoder输入→输出 |
|---|---|---|---:|---|
| EstNet | `train.py --variant estnet` | 机身速度3，无latent | 52 | 无decoder |
| Key1 | `train_key1.py` | 机身速度3 | 68 | 19→42 |
| Key2 | `train_key2.py` | 机身速度3＋足周高度图18 | 86 | 37→42 |
| FullEst | `train_fullest.py` | 速度3＋足图18＋机身高度1 | 87 | 38→42 |
| IrrEst | `train_irrest.py` | 仅机身高度1 | 66 | 17→42 |
| Implicit | `train_implicit.py` | 无显式估计头 | 65 | 16→42 |

另外五组VAE actor都读取本体观测42、命令7、自己的显式预测和latent均值16；监督真值只用于辅助损失与critic，不能直接喂给actor。decoder输入为latent与本组显式预测，当前观测只作为重建目标。IrrEst不是“估计错误速度”，也没有81维高度图估计头；Implicit不会创建无用速度头或报告假的速度估计RMSE=0。环境实际速度跟踪指标仍可记录。

论文名称为 **IrrEst**；统一入口兼容此前的 `IllEst` 拼写并规范化为 `irrest`。Key1旧接口曾附带未消费的heightmap标签，本版已删除冗余标签，并未给Key1添加高度图估计。

## 与新版 EstNet 一致的训练基础

| 项目 | 本版 |
|---|---|
| 自碰撞 | 开启；reset后读取源机器人合成USD属性，未开启则报错 |
| Hip yaw | 左右hip yaw的PD目标相对默认姿态额外限制为±20° |
| 其他腿关节 | 默认角＋0.25×动作，不再按90% soft limits截断目标 |
| 原始动作 | Normal采样；PPO保存采样值及其log probability；送PD前宽松保护±100 |
| 上半身 | 17关节按原默认姿态PD保持；保留机械限位、电机刚度/阻尼/力矩/速度限制 |
| PPO clip / value clip | 0.2 / 开启 |
| LR | 固定5e-4，不做KL自适应、回滚、拒绝候选或提前结束epoch |
| 每轮更新 | 24步rollout，4 epochs × 4 minibatches = 16次Adam |
| 策略KL | 每完整epoch测一次；有限值再大也仅记录，不改变更新流程 |
| 数值保护 | 梯度范数上限1，策略log-std范围[-3,1]；非有限数明确失败 |
| 奖励 | 与新版EstNet相同的10项奖励函数族及已有跌倒扣分，无新增mimic或姿态奖励 |
| 环境 | 固定Isaac Lab 2.3.2 / Isaac Sim 5.1.0、29关节G1、12腿动作、纯平地 |
| PhysX迁移 | 显式stabilization=True；contact-last与external-forces-every-iteration=False |

±20°是当前工程初值：作者只建议“限制hip yaw”，未给这个数值。它限制目标角，实际角仍可能因动力学超调。PPO的0.2是概率比损失裁剪参数，不是关节角上限。保留论文方向的平滑奖励也不等于重新引入动作硬限幅。

## 两种 KL 与辅助损失

**移除的是PPO策略KL对更新的干预；保留的是VAE latent KL损失。** 所有模型参数共用Adam：策略与辅助目标都可回传到共享编码器及真实存在的估计头，没有stop-gradient切断。actor在采样和重算likelihood时都用latent均值mu，避免重复采样latent改变PPO条件分布；训练decoder使用重参数采样。

```text
latent_KL = 0.5 * mean(mu² + exp(logvar) - 1 - logvar)
共同VAE损失 = 2 * MSE(predicted_current_obs, current_obs) + 50 * latent_KL
EstNet   = 1 * MSE(速度)，无VAE项
Key1     = 共同VAE损失 + 1 * MSE(速度)
Key2     = 共同VAE损失 + 1 * MSE(速度) + 0.5 * MSE(足图)
FullEst  = 共同VAE损失 + 1 * MSE(速度) + 0.5 * MSE(足图) + 2 * MSE(高度)
IrrEst   = 共同VAE损失 + 2 * MSE(高度)
Implicit = 共同VAE损失
```

KL与MSE均对batch及相应特征维一起取mean。β=50来自论文Table III；这套归约和损失组合沿用既有实现。将latent改为求sum会放大16倍，不能只保持β数字就说没有改参数。`logvar`的[-10,10]界限用于指数稳定，也是工程选择。

日志明确分开 `ppo/policy_kl`（兼容别名`ppo/kl`）、`ppo/latent_kl` 和 `ppo/latent_kl_weighted`。`prediction_loss`、各头MSE及联合`auxiliary_loss`单独记录，只有有速度头的路线才有估计速度RMSE。

## 观测时序、特权信息与论文未明确的地方

五条VAE路线沿用“**过去50帧，不含当前帧 → 重建当前obs_t**”；收集器在`env.step`前复制obs、历史和监督标签，防止复用缓冲区或reset错配。初始/重置回合用首帧填充缺失历史。GAE使用自动reset前保存的终帧critic，并区分跌倒与时间截断；这不是下一回合的重建标签。

论文图2的输出标注`t+1`，图注又说current、虚线指向当前`t`，时序存在冲突。本轮保留现有约定，没有同时换成next-observation预测；检查点显式记录 `reconstruction_target=current_obs_from_past_history`、`actor_latent_mode=mean` 和 `vae_kl_reduction=mean_batch_and_latent`。详细逐页核对见 `references/paper-variants-definition-audit-20260907.md`。

五组critic均为152维：obs42＋command7＋速度3＋身体高度1＋足图18＋base图81。足图为每脚3×3，base图9×9；这些网格与米制量纲是工程选择，论文给103维privileged information不足以唯一推出它们。脚图以脚踝link原点为参考，不等于足底净空。

目前地图用平地解析高度，**有高度图输入不代表已经加入粗糙地形训练**。日后加地形须用每个采样点的实际地面查询替换平面实现。

现有新版EstNet使用critic61、包含当前帧的50帧历史；这五组使用critic152、排除当前帧。因此这里统一的是物理、动作、PPO与奖励基础，并非严格“只替换估计头”的消融实验。没有修改正在运行的EstNet实现或其检查点。

旧独立EstNet以及旧五路线的检查点继续由各自冻结代码读取。本包六路线均使用新的 `g1-{variant}-ppo-clip-flat-isaac51-v1` schema，并要求配置显式记录 `simulation_protocol=isaacsim-5.1.0-lab-2.3.2-stabilized-v1`。旧schema、缺协议字段或其它运行时检查点均拒绝加载，不自动补字段迁移。

## 参数与来源

| 参数 | 默认值 | 来源 |
|---|---|---|
| 环境数 / learning epochs / minibatches | 4096 / 4 / 4 | 论文Table III |
| LR / gamma / GAE lambda | 5e-4 / .996 / .95 | 论文Table III；固定LR schedule为本次选择 |
| Actor、critic隐藏层 / 激活 | 2048→512→128 / ELU | 论文网络规模；独立critic沿用backbone宽度 |
| Encoder隐藏层 | 1024→256→64 | 论文Table III |
| Decoder隐藏层 | 64→256→1024 | 既有工程选择，论文未单独给出 |
| Latent维数 | 16 | 原文给Key1、IrrEst、Implicit；Key2/FullEst沿用共同框架 |
| 速度/足图/高度/prediction/β系数 | 1 / .5 / 2 / 2 / 50 | 论文Table III；归约与组合如上 |
| Horizon / PPO clip / max grad norm | 24 / .2 / 1 | 主流locomotion参考与既有选择，主论文未公布clip |
| Value / entropy / 初始动作std | 1 / .008 / .8 | 新版EstNet既有工程配置 |
| Physics / policy / history | 1000Hz / 100Hz / 0.5s | 既有实现，50帧历史 |
| Gait | 周期2/3s，占空比.5、左右差半周期、过渡.1cycle | 既有论文方向实现，具体过渡形状为工程选择 |
| 前进命令 | vx∈[.25,.55]m/s，vy=wz=0 | 平地起步实验 |
| 高度目标 | .78m | G1适配 |

奖励宽度等适配项与新版EstNet完全一致，参数来源和开源固定提交核查保存在 `references/estnet-foundation-README.md` 及研究报告中；不将原论文缺省项称为已精确还原。各路线完整默认配置在 `protocols/*.json`，这些JSON是可审阅快照，CLI参数入口仍由代码配置控制。

## 运行

将整个包复制到已配置的独立环境；不要修改系统包或覆盖正在使用的旧实验源。在本包目录启动，避免被旧仓库同名`estnet`包覆盖。真实runner在CUDA与AppLauncher初始化前严格检查：

- Python 3.11；安装元数据和实际导入的Torch均为 `2.7.0+cu128`，Torch CUDA为12.8。
- Isaac Sim安装版本 `5.1.0.0`；Isaac Lab实际import源必须来自提交 `37ddf626871758333d6ed89cf64ad702aef127d0`（v2.3.2），其tracked `source/`、`apps/` 不得修改。只设置 `ISAACLAB_PATH` 指向另一份checkout不会绕过核验。
- `run --device cpu`仍是仿真入口，同样执行运行时核验；独立CPU模型/检查点单元测试不调用该真实入口边界。

Lab2.3.2将稳定化默认从True改成False；本包显式保留True，同时固定两个新增物理开关为False。29关节PD、资产内容、动作和奖励保持既有设置，但**更换Sim/PhysX/Torch版本不等于物理轨迹或接触行为完全等价**。必须分别保存运行时版本和实际配置，不能把迁移叫作逐位复现。Sim5.1是本项目固定兼容组合，不宣称为当前最新维护版本。

以下示例假设外部调度已按UUID选定GPU，分别提供CUDA逻辑编号、Kit渲染编号和本地G1资产路径。六组各运行一个独立单GPU训练进程；此示例不自动抢占GPU或选择机器。

```bash
python train_key1.py \
  --asset "${G1_USD:?}" \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 4096 --iterations 10000 --run-dir logs/key1-isaac51-10000
```

其余路线替换入口和独立日志目录；EstNet用 `python train.py --variant estnet ...`。六组默认配置在 `protocols/estnet.json` 与其余五个同名路线文件中。也可用 `python train.py --variant key2 ...`。专用入口拒绝与自身不符的`--variant`或检查点。帮助 `python train_fullest.py --help` 无需Isaac/CUDA。

```bash
# 仅在本次同运行协议任务中断后显式恢复；新六组首次启动不传resume。
python train_key1.py \
  --asset "${G1_USD:?}" \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --resume "${CHECKPOINT_FROM_THIS_PROTOCOL:?}" --iterations 10000 \
  --run-dir logs/key1-isaac51-resumed-to10000

# 从检查点读取路线；确定性均值动作评估32个首回合，不录制视频。
python run.py evaluate \
  --asset "${G1_USD:?}" \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 32 --seed 43 --checkpoint logs/key1-isaac51-10000/model_10000.pt \
  --run-dir logs/key1-isaac51-eval10000
```

这是单GPU训练入口，不通过这份脚本自动启用DDP。`iterations`是外层采样＋优化轮次；每轮含4个learning epochs，默认总16次Adam。所有路线checkpoint协议独立为`g1-{variant}-ppo-clip-flat-isaac51-v1`，旧KL-guard、旧平地协议、缺新参数语义或跨路线检查点不能静默续训。恢复模型、完整Adam、计数和可适用的Torch RNG；PhysX、历史和回合状态从新回合开始，不承诺物理过程逐位续接。

主要输出是 `manifest.json`（`runtime_protocol`含实际包版本/Lab commit，`resolved_physics`含实际传入的Sim/PhysX/render配置，另有复制源码与资产哈希）、`metrics.jsonl`、`ppo_events.jsonl`、每100轮及最终checkpoint、`result.json`。自碰撞读回验证env_0合成USD属性，不等于验证每一对几何的真实接触响应。训练完成状态明确为`training_finished_requires_evaluation`，不能以运行稳定或存活代替会走。

可选 `--trace-steps 480 --trace-envs 16` 记录有限奖励输入；`--replay-iteration 10` 保存首minibatch更新前的路线、模型、优化器和监督数据，默认都关闭。新快照协议为 `variant-fixed-lr-first-minibatch-replay-v1`。旧仅支持EstNet速度损失的反事实分析器没有随本包发布，不能拿它分析VAE联合目标。

## 六路监督器（独立于训练算法）

`operators/launch_all.py`派发六个独立监督器；`operators/train_supervisor.py`只管理自己创建且PID/出生时间匹配的进程。两者只做启动、证据核验和资源清理，不修改PPO、奖励或策略。这套operator已另行通过35项CPU测试；发布副本仅修正测试定位包根的一行路径，相关六路线检查点测试单独复验，未重跑其余operator测试。可从包根执行 `PYTHONPATH=. pytest operators/test_supervisor.py`。

```bash
# deployment.json必须由外部调度依据新鲜GPU/路径检查生成，存放在源码包外。
python operators/launch_all.py --deployment "${DEPLOYMENT_JSON:?}" --attempt attempt-001

# 单独派发一条时，deployment仍须给出完整六路线映射。
python operators/launch_all.py --deployment "${DEPLOYMENT_JSON:?}" --variant estnet --attempt attempt-001
```

部署JSON字段如下；包内不提供带机器IP、实际UUID或部署状态的样例。

| 字段 | 含义 |
|---|---|
| `root` / `source` | 新运行结果根目录／本冻结源码包的绝对路径 |
| `runtime_python` / `runtime_env` | 已准备好的运行解释器／可source的运行环境脚本绝对路径 |
| `asset` | 已核对四份内容SHA的G1主USD绝对路径 |
| `source_manifest_sha256` | 本包 `source_manifest.json` 的完整SHA256 |
| `operator_sha256` | `train_supervisor.py`、`launch_all.py`两文件各自完整SHA256 |
| `gpus` | 六个规范路线名各对应 `{index, uuid, render_index}`；完整GPU UUID与index不得跨路线重复，render_index可省略并沿用index |
| `target` | 本轮目标10000；代码允许显式1..10000作独立短验证，但短任务不能冒称完成10000 |
| `final_evaluation` | 默认True：训练完成并核对checkpoint后，重新准入运行32环境/seed43的最终首回合评估；显式False则记录评估延后 |

operator是 **fresh-only**：它不给训练命令添加resume，不重用旧attempt或覆盖旧结果。500轮不会触发停止或重新初始化。默认训练采用seed42、4096环境，评估使用seed43、32环境。最终检查点须证明完整目标和每轮16步Adam；有限KL不触发跳步。GPU利用率为0仍可能有已有上下文，准入记录可用显存及背景上下文，不能因此声称获得排他预留。

进程正常退出、测量有效、已知native关闭警告、自有PID/context释放和最终评估分别记录；普通Python失败不能降级为关闭警告。最终评估受资源准入阻挡时明确记录deferred，不去终止别人的任务；本包不提供视频录制脚本，也不以数值门槛自动宣称真实正常行走。

## 验证

在本目录使用带Torch与pytest的CPU Python运行 `python -B -m pytest -q tests`。测试执行真实网络、联合损失、优化器、生产采样/保存/恢复代码；仅物理场景和Isaac启动器用替身。覆盖五组完整train入口、自动读检查点续训、全部参数Adam状态、标签不泄漏、历史边界、原生关节顺序打乱后的仅yaw限幅、跨路线拒载及KL超过旧阈值仍完整更新。

完整CPU测试 **109项＋243子测试通过**。新增真实运行协议检查、六路线跨运行时拒载和显式PhysX设置测试；真实模拟器入口的版本检查在CPU替身测试中被显式patch，没有给生产CLI添加绕过开关。

相对之前PPO-clip六路线包，下列12个模块字节完全相同：`ppo.py`、`key_ppo.py`、`ablation_ppo.py`、`networks.py`、`key_networks.py`、`ablation_networks.py`、`robot.py`、`rewards.py`、`gait.py`、`history.py`、`heightmaps.py`、`resume.py`。运行时适配没有改动其中训练算法、网络、动作、奖励或恢复数学；完整SHA证明保存在verification。

CPU结果、真实JUnit及发布文件SHA见 `verification.json`、`verification-junit.xml`、`source_manifest.json`。清单采用相对路径，每项记录path/bytes/sha256，不包含清单自身以避免循环哈希；部署方另行记录清单SHA。它们证明代码路径通过，不表示六条路线已经经过新机器GPU仿真或学会行走。
