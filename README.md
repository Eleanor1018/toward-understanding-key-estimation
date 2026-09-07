# G1论文路线复现：EstNet与显式/隐式估计变体

当前训练版本使用**自碰撞开启、仅hip yaw额外目标限幅±20°、固定学习率5e-4的PPO-clip**，取消策略KL回退与全腿soft-limit目标截断。PPO clip仍为0.2；五条VAE路线保留latent KL损失β=50。具体参数中哪些来自论文、哪些是G1适配或工程选择，均在各实验说明中标注。

| 当前版本 | 入口与说明 |
|---|---|
| Blackwell 六路线：同一运行时，各10000轮 | [Isaac 5.1 六路线](experiments/six_routes_isaac51_20260907/README.md)；`train.py --variant ...`或各自专用入口 |
| EstNet：只估计速度，无latent/decoder | [EstNet PPO-clip](experiments/estnet_ppo_clip_20260907/README.md)；该目录下`train.py` |
| Key1、Key2、FullEst、IrrEst、Implicit | [五路线PPO-clip](experiments/paper_variants_ppo_clip_20260907/README.md)；各自`train_*.py`或`train.py --variant ...` |
| EstNet新版500轮结果 | [评估摘要](docs/ESTNET_PPO_CLIP_500_RESULT.md)；完成训练，但尚未通过行走验证 |

新的Blackwell六路线目录固定使用Isaac Lab 2.3.2 / Isaac Sim 5.1.0 / Torch 2.7.0+cu128，六组均从零训练到10000轮，保留已有动作、奖励与PPO设置，并显式固定迁移涉及的PhysX选项。此前两个PPO-clip目录继续使用Isaac Lab 2.0.2 / Isaac Sim 4.5、默认500轮。每个目录含独立源码、测试、配置与SHA256清单；使用对应运行环境和G1资产，不要混用根目录旧包或跨协议检查点。公开源码验证记录不等同于GPU训练或行走成功。

```bash
cd experiments/six_routes_isaac51_20260907
python train_key1.py --help
python -B -m pytest -q tests
```

目前仍为平地训练。五条VAE路线使用152维critic与不含当前帧的历史，独立EstNet沿用61维critic与含当前帧的历史，因此不是严格仅改变估计头的消融。CPU验证和运行完成均不代表机器人已学会行走。

`archive/2026-09-06-pre-estnet`保存V8及更早的代码、测试和文档；`estnet_hipyaw20_20260906`与`estnet_klguard_20260907`保存后续对照版本。根目录`estnet/`保留早期EstNet/变体及DDP实现，下面是它的历史用法，当前训练应优先使用上表入口。服务器运行记录、GPU清单与模型checkpoint留在本地，不随源码上传。

## 早期EstNet/变体基线与历史用法

新增 `--variant key1` / `--variant key2`：含16维VAE与decoder的论文结构，Key2另外估计足周高度图。
明确参数、原文歧义、网格与比较限制见 [Key1/Key2说明](docs/KEY1_KEY2.md)。旧EstNet配置和检查点兼容保留。

新增 `--variant fullest` / `--variant irrest` / `--variant implicit`，分别估计全部三类显式量、仅身体高度、以及不做显式估计。
三者均使用latent和decoder；原文名称为IrrEst，CLI也接受 `--variant illest` 并规范为irrest。
网络、损失系数和论文缺失参数见 [FullEst/IrrEst/Implicit说明](docs/FULL_IRR_IMPLICIT.md)。

新基线采用 **0.5秒历史 → 三维机体线速度估计 → Gaussian actor**，以及可见真实状态的 critic。
无 latent、decoder、mimic 或归档旧模型迁移。同一EstNet基线支持保留优化器的续训。完整参数来源、论文疑点、移植差异和验收顺序见
[方案](docs/PLAN.md)。这是一份明确标注假设的 G1 移植实现，不宣称恢复了作者未公开的全部参数。

- `estnet/config.py`：唯一主配置；100Hz策略、1kHz仿真、12腿动作、上身17关节PD保持。
- `estnet/gait.py`：可见双足相位；1.5Hz、0.5占空比、半周期左右偏移。
- `estnet/networks.py`, `estnet/ppo.py`：显式速度 EstNet、联合 PPO/MSE、正确的超时 bootstrap。
- `estnet/environment.py`, `estnet/run.py`：Isaac Lab 2.0.2 / Isaac Sim 4.5 接口、训练与评估。
- `archive/2026-09-06-pre-estnet/`：完整旧实验和逐文件 SHA-256 清单。

先在 Isaac Lab 的 Python 环境运行以下命令。`G1_USD` 是已存在的同版本 USD 的绝对路径，
其 `configuration/` 引用文件必须齐全。每次运行创建新目录，不覆盖旧结果。

```bash
python -m estnet.preflight --asset "$G1_USD"
python -m estnet.run smoke --asset "$G1_USD" --num-envs 16 --headless
python -m estnet.run train --asset "$G1_USD" --num-envs 4096 --iterations 500 --headless
python -m estnet.run evaluate --asset "$G1_USD" --checkpoint /absolute/run/model_00500.pt --num-envs 32 --headless
python -m estnet.run train --asset "$G1_USD" --resume /absolute/run/model_00500.pt --iterations 10000 --headless
python -m estnet.run train --variant fullest --asset "$G1_USD" --num-envs 4096 --iterations 500 --headless
python -m estnet.run train --variant irrest --asset "$G1_USD" --num-envs 4096 --iterations 500 --headless
python -m estnet.run train --variant implicit --asset "$G1_USD" --num-envs 4096 --iterations 500 --headless
```

`--iterations` 是目标总轮数：从500轮检查点到10000轮只追加9500轮，日志编号从501开始。
论文图5横轴为iterations，表III内部Learning epochs为4，本代码保留每批数据的4次内部优化。
续训沿用保存的全部任务/PPO配置，恢复模型、Adam动量、当前自适应学习率和更新计数；不会重置回初始学习率。
缺失或损坏优化器会在CPU预检失败，不退化成只加载权重的训练。旧500轮检查点没有CUDA随机状态；
PhysX状态和历史观测也未保存，因此从新回合开始采样，不承诺与连续不中断的仿真轨迹一致。
每次续训创建独立日志目录并记录父检查点哈希，不覆盖原500轮记录。

`smoke` 测量零动作下的默认PD姿态、接触、漂移和高度，**不等于行走成功**。目标高度默认 `.78m`，
与生成姿态的 `.8m` 不同。`nominal_pose_unstable` 表示默认姿态需要主动平衡，不单独作为训练阻断条件；
非有限数、错误资产和接口异常才是需要排查的运行故障。第一轮关闭噪声、延迟和域随机化，
平地只采样向前 `.25–.55m/s`。先检查连续迈步，再扩大任务范围。

每100次保存 checkpoint；每次记录 reward分项、速度、接触、交替落地、滑脚、估计误差、KL和学习率。
训练结束仅表示需要评估。评估使用每个环境的首个完整 episode，跌倒后的重置不会伪造存活率。

`run.py` 已补中文注释。评估先在CPU核对检查点的配置、资产签名和模型权重，再初始化GPU；
模型沿用保存的高度目标，因此评估不接受 `--target-height`。运行目录保存最终配置、代码副本/哈希和版本；
Python捕获到的启动、训练、清理异常会写入 `failure.txt` 和失败结果。Isaac Sim 4.5的native关闭可能直接退出进程，
因此关闭前先原子保存 `measurement.json`，`result.json` 暂记 `cleanup_pending`；正常关闭返回后才提交最终状态。
如果进程退出而仍是 `cleanup_pending`，代表尚未证实正常清理，不能只凭退出码0算完成。
是否会走仍看 `walking_gate_fraction` 与视频。GPU选择不代表排他占用，也不因初始化临时上下文触及其他卡而停止。

CPU 验证（不需要 Isaac）：

```bash
python -m pytest -q
```

本地运行记录见 `docs/PLAN.md` 及本次交付报告。仿真验证必须以实际运行结果为准。
