# EstNet：自碰撞与 hip yaw 目标 ±20° 独立实验

本目录是 2026-09-06 已部署源码的独立审查副本，12 个训练模块与 `SOURCE_MANIFEST.json` 的 `experiment_sha256` 逐文件一致。仓库顶层 `estnet` 不受此实验影响。

本轮从头训练到总第 500 次 PPO 更新，4096 个环境，seed 42。自碰撞保持开启；相对上一轮自碰撞实验，控制变化仅为左右 hip yaw 的 PD 位置目标限制在默认角度 ±20°，同时遵守原有软限位。默认 hip yaw 为零。20° 是用户批准的工程初值，并非论文或作者给出的具体角度。

保持原 EstNet 的 obs42、command7、critic61、50 帧含当前帧历史、网络、PPO、奖励、PD、上身默认姿态和平地。限制目标不等于修改机械关节范围；实际关节仍可能因动力学短暂越过目标范围。

动作链：原始高斯采样 → 非原地裁剪至 ±100 → 默认姿态 + 0.25 × 动作 → 仅 yaw 目标区间与仿真软限位求交。PPO 保存原始采样及其 log probability；历史仍使用旧的 ±100 动作语义。新增指标记录每侧目标裁剪与实际角度。

配置/检查点使用 `g1-estnet-hipyaw-limited-flat-v1` schema，必须显式保存 `hip_yaw_target_limit_rad`。旧 schema 或缺失字段的模型均不能直接评估或续训，防止静默改变动作映射；新模型评估和续训沿用已保存角度配置。

## 运行方式

以下为审查与复现实例，需在与已部署任务一致的 Isaac Lab/Isaac Sim 环境中运行，并从本目录启动 Python。GPU 编号须按现场 UUID 和 Kit GPU 表核对；已运行任务不应重复启动。

```bash
python -m estnet.run train --asset /path/to/verified/G1.usd --headless --device cuda:5 --render-gpu 5 --num-envs 4096 --seed 42 --iterations 500 --run-dir /new/path/train-001
python -m estnet.run evaluate --asset /path/to/verified/G1.usd --headless --device cuda:5 --render-gpu 5 --num-envs 32 --seed 43 --checkpoint /path/train-001/model_00500.pt --run-dir /new/path/evaluate-001
python -B tests/test_hip_yaw_limit.py
```

普通 `evaluate` 给出原有首回合步行指标；此次还要求独立姿态/转向检查和视频审查，完整验收门槛见 `SOURCE_MANIFEST.json` 的 `acceptance_gate`。至少 29/32 个首回合同时通过原有步行与新增姿态条件，并通过视频检查，才准备明确记录参数的论文参考地形课程，续训到总第 10000 次更新。未通过则不增加地形、不延长到 10000。当前目录仍仅实现平地，不得仅修改 `--iterations` 就宣称已加入论文地形。

`--iterations` 始终表示总 PPO 更新次数；未来有合格地形版本后从 500 续至 10000 才是追加 9500 次。每次更新内部的 `epochs=4` 是另一个概念。

## 来源与验证

部署来源记录：`C:/Users/Nora/Documents/Codex/2026-09-06/xiao/outputs/estnet-hipyaw20-source-manifest.json`。本地 `SOURCE_MANIFEST.json` 是同一记录的副本，`HIP_YAW20.patch` 是相对上一轮自碰撞源码的差异。

CPU 回归 7 项通过，含 17 个参数/加载器子案例：乱序原生映射、其他关节/上身保持、软限位交集、原始动作不变、真实 rollout 的 logprob 对齐、真实 PPO/Adam 更新、目标与实际角度指标、检查点兼容性。CPU 验证不代表仿真步态已成功；结果需等待本轮实际评估和视频。


## 本轮实际结果

已完成500轮和独立评估：32/32存活10秒，0/32通过正常步行门槛；因此未加入地形，也未续至10000轮。模型与视频均完整保存，相关进程已清理。完整记录：[500轮验证报告](<C:/Users/Nora/Documents/Codex/2026-09-06/xiao/outputs/EstNet髋Yaw20度500轮验证.md>)。
