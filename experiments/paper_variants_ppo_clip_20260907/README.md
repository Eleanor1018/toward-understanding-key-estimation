# Key1 / Key2 / FullEst / IrrEst / Implicit：自碰撞与 hip yaw 限幅版

2026-09-07。五条路线采用刚完成的 EstNet PPO-clip 版本作为共同训练基础，分别保留论文各自的估计头、16维 latent 和 decoder。默认从零训练500轮；本次交付训练代码与CPU验证，没有启动这五条路线的GPU任务。

入口与训练循环都补了中文注释。五个入口只固定路线，实际代码共享 `estnet/run.py`、`environment.py` 和 `ppo.py`，因此动作、奖励或续训修复不会漏掉其中某一路。统一入口也支持 EstNet，作为接口回归参考。

## 五条路线

| 路线 | 训练文件 | 显式估计头 | Actor输入维度 | Decoder输入→输出 |
|---|---|---|---:|---|
| Key1 | `train_key1.py` | 机身速度3 | 68 | 19→42 |
| Key2 | `train_key2.py` | 机身速度3＋足周高度图18 | 86 | 37→42 |
| FullEst | `train_fullest.py` | 速度3＋足图18＋机身高度1 | 87 | 38→42 |
| IrrEst | `train_irrest.py` | 仅机身高度1 | 66 | 17→42 |
| Implicit | `train_implicit.py` | 无显式估计头 | 65 | 16→42 |

五组actor都读取本体观测42、命令7、自己的显式预测和latent均值16；监督真值只用于辅助损失与critic，不能直接喂给actor。decoder输入为latent与本组显式预测，当前观测只作为重建目标。IrrEst不是“估计错误速度”，也没有81维高度图估计头；Implicit不会创建无用速度头或报告假的速度估计RMSE=0。环境实际速度跟踪指标仍可记录。

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
| 环境 | Isaac Lab 2.0.2 / Isaac Sim 4.5、29关节G1、12腿动作、纯平地 |

±20°是当前工程初值：作者只建议“限制hip yaw”，未给这个数值。它限制目标角，实际角仍可能因动力学超调。PPO的0.2是概率比损失裁剪参数，不是关节角上限。保留论文方向的平滑奖励也不等于重新引入动作硬限幅。

## 两种 KL 与辅助损失

**移除的是PPO策略KL对更新的干预；保留的是VAE latent KL损失。** 所有模型参数共用Adam：策略与辅助目标都可回传到共享编码器及真实存在的估计头，没有stop-gradient切断。actor在采样和重算likelihood时都用latent均值mu，避免重复采样latent改变PPO条件分布；训练decoder使用重参数采样。

```text
latent_KL = 0.5 * mean(mu² + exp(logvar) - 1 - logvar)
共同VAE损失 = 2 * MSE(predicted_current_obs, current_obs) + 50 * latent_KL
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

正在运行的独立EstNet500仍应使用原 `experiments/estnet_ppo_clip_20260907` 冻结入口评估和续训。本包的EstNet选项用于共同代码回归；虽然保留相同EstNet外层schema，加载器新增显式元数据要求，不能直接读取原独立版本缺这些字段的检查点，也不自动补默认值迁移。

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

将本实验整个目录复制到服务器，使用已配置的Python3.10 / Torch2.5.1+cu121 / Isaac Sim4.5 / Isaac Lab2.0.2环境。在本实验目录启动，避免被旧仓库同名`estnet`包覆盖。

以下示例假设已按UUID选定空闲GPU，并分别设定CUDA逻辑编号与Kit渲染编号；不会默认占用0号卡。调度沿用避开Pro6000D和4090-48-3的安排。

```bash
python train_key1.py \
  --asset /data/nora/toward-understanding-key-estimation/assets/g1_29dof/g1_29dof_rev_1_0.usd \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 4096 --iterations 500 --run-dir logs/key1-ppo-clip-500
```

其余路线替换入口和独立日志目录即可。也可用 `python train.py --variant key2 ...`。专用入口拒绝与自身不符的`--variant`或检查点。帮助 `python train_fullest.py --help` 无需Isaac/CUDA。

```bash
# 同版本Key1从500续到10000，只追加9500次外层更新。
python train_key1.py \
  --asset /data/nora/toward-understanding-key-estimation/assets/g1_29dof/g1_29dof_rev_1_0.usd \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --resume logs/key1-ppo-clip-500/model_00500.pt --iterations 10000 \
  --run-dir logs/key1-ppo-clip-to10000

# 从检查点读取路线；确定性均值动作评估32个首回合，不录制视频。
python run.py evaluate \
  --asset /data/nora/toward-understanding-key-estimation/assets/g1_29dof/g1_29dof_rev_1_0.usd \
  --device "cuda:${GPU_LOGICAL_ID:?}" --render-gpu "${KIT_RENDER_ID:?}" --headless \
  --num-envs 32 --seed 43 --checkpoint logs/key1-ppo-clip-500/model_00500.pt \
  --run-dir logs/key1-ppo-clip-eval500
```

这是单GPU训练入口，不通过这份脚本自动启用DDP。`iterations`是外层采样＋优化轮次；每轮含4个learning epochs，默认总16次Adam。所有路线checkpoint协议独立为`g1-{variant}-ppo-clip-flat-v1`，旧KL-guard、旧平地协议、缺新参数语义或跨路线检查点不能静默续训。恢复模型、完整Adam、计数和可适用的Torch RNG；PhysX、历史和回合状态从新回合开始，不承诺物理过程逐位续接。

主要输出是 `manifest.json`（版本、配置、实际复制源码和资产哈希）、`metrics.jsonl`、`ppo_events.jsonl`、每100轮及最终checkpoint、`result.json`。自碰撞读回验证env_0合成USD属性，不等于验证每一对几何的真实接触响应。训练完成状态明确为`training_finished_requires_evaluation`，不能以运行稳定或存活代替会走。

可选 `--trace-steps 480 --trace-envs 16` 记录有限奖励输入；`--replay-iteration 10` 保存首minibatch更新前的路线、模型、优化器和监督数据，默认都关闭。新快照协议为 `variant-fixed-lr-first-minibatch-replay-v1`。旧仅支持EstNet速度损失的反事实分析器没有随本包发布，不能拿它分析VAE联合目标。

## 验证

在本目录使用带Torch与pytest的CPU Python运行 `python -B -m pytest -q tests`。测试执行真实网络、联合损失、优化器、生产采样/保存/恢复代码；仅物理场景和Isaac启动器用替身。覆盖五组完整train入口、自动读检查点续训、全部参数Adam状态、标签不泄漏、历史边界、原生关节顺序打乱后的仅yaw限幅、跨路线拒载及KL超过旧阈值仍完整更新。

CPU结果与发布文件SHA见 `verification.json`、`source_manifest.json`。它们证明代码路径通过，不表示五条路线已经经过真实GPU仿真或学会行走。
