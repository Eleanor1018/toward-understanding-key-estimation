"""EstNet 运行入口：python -m estnet.run {smoke,train,evaluate} --asset /path/to/G1.usd。

smoke 测量零动作下的默认 PD 姿态；train 从零开始或恢复 PPO 训练；evaluate 测量检查点。
策略输入为本体观测、0.5 秒历史估计出的三维速度和带步态相位的命令。
critic 额外读取仿真真值；本版本没有 latent、decoder 或 mimic。
具体网络和更新公式分别在 networks.py、ppo.py；论文差异及参数来源见 docs/PLAN.md。
"""
import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from .config import Config
from .preflight import inspect_asset


def write_json(path, data):
    """保存可跨平台读取的记录；不把 NaN/Infinity 写成看似正常的结果。"""
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def runtime_manifest(args):
    """版本和设备环境只作运行记录；此函数不初始化 CUDA 或仿真器。"""
    versions = {}
    for name in ("torch", "isaacsim", "isaaclab", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "requested_device": args.device,
            "requested_render_gpu": args.render_gpu,
            "environment": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "PXR_WORK_THREAD_LIMIT")}}


def _asset_signature(asset):
    """按四份 USD 的内容比较资产；服务器目录不同不应导致拒绝评估。"""
    if not isinstance(asset, dict) or asset.get("ready") is not True:
        raise ValueError("Checkpoint asset signature is missing or incomplete")
    files = asset.get("files", [])
    if not isinstance(files, list) or len(files) != 4:
        raise ValueError("Expected four asset SHA256 signatures")
    hashes = [entry.get("sha256") if isinstance(entry, dict) else None for entry in files]
    if any(not isinstance(value, str) or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("Invalid asset SHA256 signature")
    return tuple(sorted(hashes))


def _validate_checkpoint(payload, asset):
    """训练恢复和评估共用的 CPU 配置、资产与模型验证。"""
    import torch
    from .networks import EstNet

    schema = Config().schema
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError("Checkpoint schema is incompatible with this EstNet baseline")
    saved_config = payload.get("config")
    # 即使外层schema被手改，也拒绝用Config默认值给旧模型悄悄加上新动作映射。
    if not isinstance(saved_config, dict) or not {
            "hip_yaw_target_limit_rad", "kl_hard_limit", "kl_chunk_size", "gradient_diagnostics"
        }.issubset(saved_config):
        raise ValueError("Checkpoint must explicitly record the hip-yaw limit and KL guard configuration")
    cfg = Config(**saved_config)
    if cfg.schema != schema:
        raise ValueError("Checkpoint config schema is incompatible")
    # 配置中的任意 NaN/Infinity 都会破坏动力学或审计记录，不能只检查高度。
    json.dumps(cfg.to_dict(), allow_nan=False)
    cfg.validate()
    if type(payload.get("iteration")) is not int or payload["iteration"] < 0:
        raise ValueError("Checkpoint iteration must be a non-negative integer")
    if _asset_signature(payload.get("asset")) != _asset_signature(asset):
        raise ValueError("Checkpoint asset signature differs from the current asset")
    # strict 同时检查键名和张量形状；单凭外层 schema 相同不能证明模型可加载。
    validator = EstNet(cfg)
    try:
        validator.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Checkpoint model weights are incompatible") from exc
    if any(not torch.isfinite(value).all() for value in validator.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model weights")
    return cfg


def load_evaluation_checkpoint(path, asset):
    """在 CPU 上验证完整检查点，只返回评估需要的权重和配置。"""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = _validate_checkpoint(payload, asset)
    # Adam 状态仍可能临时占用 CPU 内存，但不会随着返回值被搬到 GPU。
    model_only = {key: payload[key] for key in ("schema", "model", "config", "iteration")}
    return model_only, cfg


def _check_active_metrics(metrics, active):
    """仅检查仍在被评估的首回合，重置后的回合不参与成功率或统计。"""
    import torch
    for name, value in metrics.items():
        if not torch.isfinite(value[active]).all():
            raise FloatingPointError(f"Non-finite first-episode metric: {name}")


def close_resources(env, app):
    """环境清理抛错时仍尝试关闭 Kit，避免跳过应用资源释放。"""
    try:
        if env is not None:
            env.close()
    finally:
        if app is not None:
            app.close()


def collect_rollout(env, model, obs, cfg):
    """采集 horizon × num_envs 个样本，计算 GAE，再摊平成 PPO 的训练批次。"""
    import torch
    from .ppo import compute_gae
    rows, metrics, rewards_by_term = [], [], []
    with torch.no_grad():
        for _ in range(cfg.horizon):
            action, logp, value, mean, std = model.act(obs)
            recorder = getattr(env, "reward_recorder", None)
            if recorder is not None and recorder.active:
                from .gait import gait
                phase = env.phase_offset + env.episode_length_buf * cfg.step_dt / cfg.gait_period
                foot_phase, stance = gait(phase, cfg.gait_duty, cfg.gait_transition)
                recorder.capture_pre(obs, action, mean, std,
                    phase_offset=env.phase_offset, episode_length=env.episode_length_buf,
                    pre_phase=foot_phase, pre_stance=stance)
            # 先复制动作对应的旧观测，避免环境内部复用缓冲区造成时序错位。
            row = {k: v.clone() for k, v in obs.items()}
            next_obs, reward, terminated, truncated, info = env.step(action)
            if recorder is not None and recorder.active:
                phase = env.phase_offset + env.episode_length_buf * cfg.step_dt / cfg.gait_period
                foot_phase, stance = gait(phase, cfg.gait_duty, cfg.gait_transition)
                recorder.capture_post(terminated=terminated, truncated=truncated,
                    post_episode_length=env.episode_length_buf,
                    post_phase=foot_phase, post_stance=stance,
                    reset_mask=terminated | truncated, next_obs=next_obs["obs"])
            # DirectRLEnv 会自动重置：next_obs 可能已属于下一回合。
            # bootstrap 必须读取环境保存的终帧 critic；GAE 会对真实终止清零，
            # 对时间截断保留 bootstrap，并在两种边界上都切断优势向后递推。
            row.update(action=action.clone(), old_log_prob=logp.clone(), old_value=value.clone(),
                       old_mean=mean.clone(), old_std=std.clone(), reward=reward.clone(),
                       next_value=model.value(info["final_critic"]).clone(),
                       terminated=terminated.clone(), truncated=truncated.clone())
            rows.append(row)
            metrics.append({k: v.mean().detach() for k, v in info["metrics"].items()})
            rewards_by_term.append({k: v.mean().detach() for k, v in info["reward_terms"].items()})
            obs = next_obs
        stacked = {k: torch.stack([row[k] for row in rows]) for k in rows[0]}
        advantages, returns = compute_gae(stacked["reward"], stacked["old_value"],
                                         stacked["next_value"], stacked["terminated"],
                                         stacked["truncated"], cfg.gamma, cfg.gae_lambda)
        stacked.update(advantages=advantages, returns=returns)
        # 前两维是时间 T 和环境 N；其余观测/动作维度保持原样。
        batch = {k: v.flatten(0, 1) for k, v in stacked.items()}
        summary = {"env/" + k: torch.stack([m[k] for m in metrics]).mean().item() for k in metrics[0]}
        summary.update({"reward/" + k: torch.stack([m[k] for m in rewards_by_term]).mean().item()
                        for k in rewards_by_term[0]})
        summary["reward/total"] = stacked["reward"].mean().item()
    return obs, batch, summary


def save_checkpoint(path, model, ppo, config, iteration, asset):
    """先写临时文件再替换，降低中断留下半个检查点的概率。"""
    import torch
    # 模型、Adam和随机状态可恢复，但不保存PhysX状态与历史观测；
    # 续训从新的回合开始，不能声称与不中断仿真逐位一致。
    state = {"schema": config.schema, "config": config.to_dict(), "iteration": iteration,
             "model": model.state_dict(), "optimizer": ppo.state_dict(), "asset": asset,
             "torch_rng": torch.get_rng_state()}
    device = next(model.parameters()).device
    if device.type == "cuda":
        state["torch_cuda_rng"] = torch.cuda.get_rng_state(device)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def evaluate_first_episodes(env, model, obs, cfg):
    """每个环境只评估首回合，用确定性均值动作观察策略的实际行为。"""
    import torch
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    survived = torch.zeros_like(active)
    sums = {}
    ankle_min = torch.full((env.num_envs, 2), float("inf"), device=env.device)
    ankle_max = torch.full_like(ankle_min, -float("inf"))
    with torch.no_grad():
        for step in range(env.max_episode_length):
            # 已结束环境继续接收零动作以满足向量接口，但不再调用其策略或计分。
            action = torch.zeros(env.num_envs, cfg.action_dim, device=env.device)
            action[active] = model.distribution(obs["history"][active], obs["obs"][active],
                                                obs["command"][active]).mean
            obs, _, terminal, timeout, info = env.step(action)
            _check_active_metrics(info["metrics"], active)
            for name, value in info["metrics"].items():
                sums.setdefault(name, torch.zeros_like(lengths))
                # NaN * 0 仍为 NaN；where 才能隔离已结束回合后的异常值。
                sums[name] += torch.where(active, value, torch.zeros_like(value))
            lengths += active
            # 排除出生后的前一秒，避免把自然下落误当成主动抬脚。
            if step * cfg.step_dt >= 1.:
                heights = torch.stack([info["metrics"]["ankle_height_left"],
                                       info["metrics"]["ankle_height_right"]], -1)
                ankle_min = torch.where(active[:, None], torch.minimum(ankle_min, heights), ankle_min)
                ankle_max = torch.where(active[:, None], torch.maximum(ankle_max, heights), ankle_max)
            survived |= active & timeout & ~terminal
            active &= ~(terminal | timeout)
            if not active.any():
                break
    means = {k: v / lengths.clamp_min(1) for k, v in sums.items()}
    ratio = means["forward_velocity"] / means["command_vx"].clamp_min(.01)
    excursion = (ankle_max - ankle_min).clamp_min(0.)
    # 同时看存活、速度、交替落脚、支撑比例和滑脚；这些是工程验收门槛，
    # 不是论文公布的成功率定义，仍需配合视频判断是否真正形成步态。
    gates = {"survived": survived, "velocity_error": means["velocity_error"] < .2,
             "speed_ratio": (ratio > .7) & (ratio < 1.3),
             "double_support": means["double_support"] < .8,
             "alternation": sums["alternating_landing"] >= 6,
             "stance_slip": means["stance_slip"] < .15,
             "both_feet_excursion": (excursion > .03).all(-1)}
    passed = torch.stack(list(gates.values())).all(0)
    return {"status": "measured_first_episodes", "num_envs": env.num_envs,
            "survival_fraction": survived.float().mean().item(),
            "walking_gate_fraction": passed.float().mean().item(),
            "gate_fractions": {k: v.float().mean().item() for k, v in gates.items()},
            "mean_episode_seconds": (lengths.mean() * cfg.step_dt).item(),
            "mean_alternating_landings": sums["alternating_landing"].mean().item(),
            "mean_velocity_ratio": ratio.mean().item(),
            "metrics": {k: v.mean().item() for k, v in means.items()},
            "interpretation": "工程验收门槛，并非论文的成功率定义；需结合视频检查步态。"}


def smoke_check(env, obs, cfg):
    """用零动作检查默认姿态三秒；接口有效与无需主动平衡是两个结论。

    每个环境只统计首回合。默认姿态无法自行站稳时，返回
    nominal_pose_unstable，不能据此断言仿真器或机器人模型损坏。
    """
    import torch

    # 动作零表示追踪默认关节角；不会调用随机策略或执行 PPO 更新。
    action = torch.zeros(env.num_envs, 12, device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    terminal_events = 0
    sample_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    sums = {}
    # 用双精度累计高度矩，减少几乎恒定高度的方差相消误差。
    height_sum = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    height_square_sum = torch.zeros_like(height_sum)

    with torch.no_grad():
        for step in range(round(3.0 / cfg.step_dt)):
            obs, reward, terminal, timeout, info = env.step(action)
            if any(not torch.isfinite(value).all() for value in obs.values()):
                raise FloatingPointError("Non-finite simulator observation")
            if not torch.isfinite(reward[active]).all():
                raise FloatingPointError("Non-finite simulator reward in first episode")
            metrics = info["metrics"]
            _check_active_metrics(metrics, active)

            # 当前返回的是这次动作后的终帧；先计入首回合，再关闭结束的环境。
            terminal_events += int((active & terminal).sum().item())
            if (step + 1) * cfg.step_dt > 2.0:
                sample_count += active.long()
                for name, value in metrics.items():
                    sums.setdefault(name, torch.zeros_like(value))
                    # NaN * False 仍是 NaN，必须用 where 排除重置后的帧。
                    sums[name] += torch.where(active, value, torch.zeros_like(value))
                height = metrics["base_height"].double()
                height = torch.where(active, height, torch.zeros_like(height))
                height_sum += height
                height_square_sum += height.square()

            # 超时也结束首回合；之后的新回合不能补充存活或稳态证据。
            active &= ~(terminal | timeout)
            if not active.any():
                break

    has_samples = sample_count > 0
    denominator = sample_count.clamp_min(1)
    means = {name: value / denominator for name, value in sums.items()}
    height_mean = height_sum / denominator
    height_variance = (height_square_sum / denominator - height_mean.square()).clamp_min(0.0)
    height_std_per_env = height_variance.sqrt()
    height_ok_per_env = has_samples & ((height_mean - cfg.reward_target_height).abs() <= 0.03)

    if has_samples.any():
        # 先按各环境自己的有效样本求均值，显示时只汇总确实有样本的环境。
        # 是否通过仍逐环境判断，不能让一高一低相互抵消。
        last_second_metrics = {
            name: value[has_samples].mean().item() for name, value in means.items()
        }
        settled_height_std = height_std_per_env[has_samples].mean().item()
        static_pass = (
            active
            & has_samples
            & height_ok_per_env
            & (height_std_per_env < 0.01)
            & (means["double_support"] > 0.95)
            & (means["horizontal_speed"] < 0.05)
            & (means["stance_slip"] < 0.05)
        )
    else:
        # 全部首回合在测量窗口前结束时，没有“稳态高度”可报告。
        last_second_metrics = {}
        settled_height_std = None
        static_pass = torch.zeros_like(active)

    all_static = bool(static_pass.all().item())
    return {
        "status": "static_pd_smoke_passed" if all_static else "nominal_pose_unstable",
        "static_balance_pass_fraction": static_pass.float().mean().item(),
        "interface_finite": True,
        "terminal_events": terminal_events,
        "last_second_metrics": last_second_metrics,
        "settled_height_std": settled_height_std,
        "height_matches_target": bool(height_ok_per_env.all().item()),
        "joint_names": list(env.robot.joint_names),
        "leg_indices": env.leg_ids.tolist(),
        "mass_kg": env.mass.mean().item(),
        "last_failure": env.extras.get("last_failure"),
        "walking_verified": False,
    }


def main():
    """先完成 CPU 预检，再启动模拟器；清理结束后才提交最终运行结果。"""
    import math
    import traceback

    parser = argparse.ArgumentParser(description="G1 EstNet：默认姿态诊断、训练/续训和检查点评估。")
    parser.add_argument("mode", choices=["smoke", "train", "evaluate"],
                        help="smoke诊断默认姿态；train训练或续训；evaluate测量检查点的首个回合")
    parser.add_argument("--asset", type=Path, required=True, help="已核对内容签名的29关节G1 USD文件")
    parser.add_argument("--num-envs", type=int, help="并行环境数；训练默认4096，诊断和评估默认16")
    parser.add_argument("--iterations", type=int, default=500,
                        help="训练结束时的总PPO更新数，默认500；从500续至10000时仅追加9500轮")
    parser.add_argument("--seed", type=int, default=None,
                        help="新运行默认42；续训省略时沿用检查点种子，评估可独立指定")
    parser.add_argument("--device", default="cuda:0",
                        help="张量与物理设备，例如cpu或cuda:3；CPU物理模式不保证Kit完全不使用GPU")
    parser.add_argument("--render-gpu", type=int,
                        help="Kit GPU表中的渲染编号；须按UUID核对，不能假定等于CUDA编号")
    parser.add_argument("--cpu-threads", type=int, default=8, help="本进程的CPU线程上限，默认8")
    parser.add_argument("--headless", action="store_true", help="不打开模拟器交互窗口")
    parser.add_argument("--run-dir", type=Path, help="新建运行目录；已存在的目录不会被覆盖")
    parser.add_argument("--checkpoint", type=Path, help="仅用于evaluate的模型检查点")
    parser.add_argument("--resume", type=Path, help="仅用于train；恢复模型、Adam、学习率及更新计数")
    parser.add_argument("--target-height", type=float, default=None,
                        help="诊断或从零训练的高度目标；省略时使用默认配置；评估必须沿用检查点配置")
    parser.add_argument("--trace-steps", type=int, default=480,
                        help="训练只读逐步记录上限，0关闭；最多1000步")
    parser.add_argument("--trace-envs", type=int, default=16,
                        help="只记录前若干环境；不改变总仿真环境数")
    parser.add_argument("--replay-iteration", type=int, default=10,
                        help="只保存此轮第一个真实minibatch及更新前模型/Adam，0关闭")
    args = parser.parse_args()

    # 能在纯参数阶段拒绝的输入，不应等到初始化GPU或加载USD时才报错。
    if args.iterations < 1:
        parser.error("iterations必须为正整数")
    if args.cpu_threads < 1:
        parser.error("cpu-threads必须为正整数")
    if not 0 <= args.trace_steps <= 1000 or args.trace_envs < 1 or args.replay_iteration < 0:
        parser.error("trace-steps应为0..1000，trace-envs为正，replay-iteration非负")
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("num-envs必须为正整数")
    if args.render_gpu is not None and args.render_gpu < 0:
        parser.error("render-gpu不能为负数")
    if args.target_height is not None and (not math.isfinite(args.target_height) or args.target_height <= 0):
        parser.error("target-height必须是有限正数")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("evaluate必须提供checkpoint")
    if args.mode != "evaluate" and args.checkpoint is not None:
        parser.error("checkpoint仅用于evaluate；续训请用train --resume")
    if args.resume is not None and args.mode != "train":
        parser.error("resume仅用于train")
    if (args.mode == "evaluate" or args.resume is not None) and args.target_height is not None:
        parser.error("评估和续训沿用检查点中的高度目标，不能传入target-height")

    run_dir = (args.run_dir or Path("logs") / f"{args.mode}-{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"无法创建新的运行目录 {run_dir}：{exc}", file=sys.stderr)
        return 2

    # 所有资源都从空状态开始，初始化中途失败也会进入统一清理路径。
    env = None
    app = None
    recorder = None
    result = None
    failure = None
    failure_text = ""
    exit_code = 0
    phase = "cpu_preflight"
    manifest = {"mode": args.mode, "argv": list(sys.argv),
                "created_at": datetime.now().astimezone().isoformat(),
                "status": "preparing", "config": None, "runtime": {"status": "not_initialized"}}
    try:
        source_dir = run_dir / "source"
        source_dir.mkdir()
        source_hashes = {}
        for source in sorted(Path(__file__).parent.glob("*.py")):
            copied_source = source_dir / source.name
            shutil.copy2(source, copied_source)
            # 哈希对应实际归档副本，避免复制后又读取正在编辑的源文件。
            source_hashes[source.name] = hashlib.sha256(copied_source.read_bytes()).hexdigest()
        manifest["source_sha256"] = source_hashes
        manifest["source_directory"] = str(source_dir)

        asset = inspect_asset(args.asset)
        manifest["asset"] = asset
        if not asset["ready"]:
            raise ValueError("机器人资产缺失或内容签名与本基线不一致")

        # 线程限制必须早于Torch和检查点验证模型的初始化；这里尚未调用CUDA API。
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PXR_WORK_THREAD_LIMIT"):
            os.environ[key] = str(args.cpu_threads)
        import torch
        torch.set_num_threads(args.cpu_threads)

        checkpoint = None
        checkpoint_iteration = None
        start_iteration = 0
        num_envs = args.num_envs if args.num_envs is not None else (4096 if args.mode == "train" else 16)
        if args.mode == "evaluate":
            # 此函数只在CPU加载，核对两层schema、配置、四份资产签名及模型权重形状。
            # 返回值不含Adam状态；评估只需要模型，不将训练优化器搬到GPU。
            checkpoint, saved_cfg = load_evaluation_checkpoint(args.checkpoint, asset)
            cfg = replace(saved_cfg, num_envs=num_envs, seed=42 if args.seed is None else args.seed)
            checkpoint_iteration = checkpoint["iteration"]
            manifest["checkpoint"] = str(args.checkpoint.resolve())
            manifest["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            manifest["checkpoint_iteration"] = checkpoint_iteration
        elif args.resume is not None:
            from .resume import load_training_checkpoint
            checkpoint, cfg = load_training_checkpoint(args.resume, asset)
            # 追加轮数保留已保存的任务与PPO参数，不将CLI默认值覆盖到旧配置。
            if args.num_envs is not None and args.num_envs != cfg.num_envs:
                raise ValueError("续训必须沿用检查点的num-envs")
            if args.seed is not None and args.seed != cfg.seed:
                raise ValueError("续训必须沿用检查点的seed")
            start_iteration = checkpoint["iteration"]
            if args.iterations <= start_iteration:
                raise ValueError("iterations是目标总轮数，必须大于检查点中已完成的轮数")
            manifest["resume"] = {
                "checkpoint": str(args.resume.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                "start_iteration": start_iteration, "target_iteration": args.iterations,
                "additional_iterations": args.iterations - start_iteration,
                "simulation_state_restored": False, "history_restored": False,
                "note": "恢复学习状态，从新回合开始；CUDA随机状态是否可恢复见restored_state。"}
        else:
            cfg = Config(num_envs=num_envs, seed=42 if args.seed is None else args.seed)
            if args.target_height is not None:
                cfg = replace(cfg, reward_target_height=args.target_height)
        cfg.validate()
        manifest["config"] = cfg.to_dict()
        manifest["runtime"] = runtime_manifest(args)
        write_json(run_dir / "manifest.json", manifest)

        # 在全部CPU预检完成后才选择CUDA设备；编号选择不表示拥有GPU的排他使用权。
        phase = "device_initialization"
        try:
            device = torch.device(args.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"无法解析device={args.device!r}") from exc
        if device.type not in ("cpu", "cuda"):
            raise ValueError("此入口仅支持cpu或cuda设备")
        if device.type == "cpu" and device.index is not None:
            raise ValueError("CPU设备请使用cpu，不使用带编号的CPU设备")
        selected_device = {"type": device.type, "exclusive_reservation": False}
        if device.type == "cuda":
            device = torch.device("cuda", 0 if device.index is None else device.index)
            torch.cuda.set_device(device)
            logical_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(logical_index)
            selected_device.update(logical_index=logical_index, name=properties.name,
                                   uuid=str(getattr(properties, "uuid", "unavailable")))
        args.device = str(device)
        torch.manual_seed(cfg.seed)
        try:
            runtime = runtime_manifest(args)
        except Exception as exc:
            # 版本信息采集失败单独记录；不能伪造版本，也不将此辅助诊断当成动力学故障。
            runtime = {"collection_error": f"{type(exc).__name__}: {exc}"}
        runtime["selected_torch_device"] = selected_device
        runtime["requested_kit_render_gpu"] = args.render_gpu
        manifest["runtime"] = runtime
        write_json(run_dir / "manifest.json", manifest)

        phase = "application_initialization"
        try:
            from isaaclab.app import AppLauncher
        except ImportError as exc:
            raise RuntimeError("请使用Isaac Lab 2.0.2 / Isaac Sim 4.5 Python环境；当前无法导入Isaac Lab") from exc

        class SelectedGpuLauncher(AppLauncher):
            def _config_resolution(self, launcher_args):
                super()._config_resolution(launcher_args)
                # Isaac Sim 4.5 默认快速关闭可能在 native shutdown 中直接退出进程。
                # 关闭快速退出；关键测量仍须在 app.close() 前落盘，不能依赖它返回。
                self._sim_app_config["fast_shutdown"] = False
                if args.render_gpu is not None:
                    # 这是固定Isaac Lab 2.0.2版本的钩子，必须在SimulationApp创建前设置。
                    # Kit渲染编号和CUDA逻辑编号分别记录，不能根据整数相等推断物理GPU相同。
                    self._sim_app_config["active_gpu"] = args.render_gpu
                    self._sim_app_config["multi_gpu"] = False

        original_argv = list(sys.argv)
        sys.argv.extend(["--portable-root", str(run_dir / "kit-user")])
        try:
            launcher = SelectedGpuLauncher(
                headless=args.headless, device=args.device, multi_gpu=False,
                kit_args=f"--/plugins/carb.tasking.plugin/threadCount={args.cpu_threads}")
            app = launcher.app
        finally:
            # AppLauncher构造失败时也必须恢复进程参数，避免污染后续错误处理。
            sys.argv[:] = original_argv

        phase = "environment_initialization"
        from .environment import EstNetEnv, build_env_cfg
        env_cfg = build_env_cfg(cfg, args.asset.resolve(), args.device)
        env_cfg.detailed_diagnostics = args.mode == "smoke"
        env = EstNetEnv(env_cfg)
        obs, _ = env.reset()
        if args.mode == "train" and args.trace_steps:
            from .reward_diagnostics import RewardTraceRecorder
            from .robot import LEG_JOINT_NAMES12
            cpu = lambda x: x.detach().cpu().tolist()
            d = env.robot.data
            ids = slice(0, min(args.trace_envs, env.num_envs))
            recorder = RewardTraceRecorder(cfg, max_steps=args.trace_steps,
                num_envs=min(args.trace_envs, env.num_envs), output_dir=run_dir / "reward_trace",
                static_metadata={"joint_names":list(LEG_JOINT_NAMES12),
                    "native_joint_names":list(env.robot.joint_names), "leg_ids":cpu(env.leg_ids),
                    "foot_body_names":[env.robot.body_names[i] for i in env.foot_ids],
                    "default_q":cpu(d.default_joint_pos[ids][:, env.leg_ids]),
                    "soft_limits":cpu(d.soft_joint_pos_limits[ids][:, env.leg_ids]),
                    "env_origins":cpu(env.scene.env_origins[ids]),
                    "foot_height_semantics":"ankle_roll link origin z relative to flat ground; not sole clearance",
                    "ground_height_semantics":"flat plane at each environment origin z",
                    "policy":"raw Gaussian sampling during training; no deterministic evaluation in this trace"})
            env.reward_recorder = recorder
        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)

        if args.mode == "smoke":
            phase = "nominal_pose_measurement"
            # 只执行零动作接口与默认姿态诊断，不为此构造大型actor、critic或估计器。
            # 零策略姿态不稳是一项测量结果，不等于模拟器损坏或学习策略无法站稳。
            result = smoke_check(env, obs, cfg)
        else:
            from .networks import EstNet
            model = EstNet(cfg).to(env.device)
            if args.mode == "evaluate":
                phase = "checkpoint_evaluation"
                model.load_state_dict(checkpoint["model"], strict=True)
                checkpoint = None
                result = evaluate_first_episodes(env, model.eval(), obs, cfg)
                result["checkpoint"] = str(args.checkpoint.resolve())
                result["iteration"] = checkpoint_iteration
            else:
                phase = "training"
                from .ppo import PPO
                ppo = PPO(model, cfg)
                if args.resume is not None:
                    from .resume import restore_training_state
                    restored = restore_training_state(model, ppo, checkpoint)
                    manifest["resume"]["restored_state"] = restored
                    write_json(run_dir / "manifest.json", manifest)
                    print(json.dumps({"event": "training_resumed", **manifest["resume"]},
                                     ensure_ascii=False), flush=True)
                    checkpoint = None
                with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log, \
                     (run_dir / "ppo_events.jsonl").open("w", encoding="utf-8") as event_log:
                    def record_event(event):
                        event_log.write(json.dumps(event, allow_nan=False) + "\n")
                        event_log.flush()
                    def save_replay(snapshot):
                        path = run_dir / "replay_minibatch.pt"
                        if path.exists():
                            raise RuntimeError("Only one bounded replay snapshot may be saved")
                        temporary = path.with_suffix(".tmp")
                        torch.save(snapshot, temporary)
                        temporary.replace(path)
                    ppo.diagnostic_callback = record_event
                    # 护栏只改变更新是否被接受；奖励、动作映射和网络结构保持本次基线。
                    for iteration in range(start_iteration + 1, args.iterations + 1):
                        started = time.perf_counter()
                        obs, batch, summary = collect_rollout(env, model, obs, cfg)
                        ppo.replay_callback = save_replay if iteration == args.replay_iteration else None
                        summary.update({"ppo/" + key: value for key, value in ppo.update(batch).items()})
                        if ppo.updates != iteration:
                            raise RuntimeError("PPO更新计数与训练全局轮次不一致")
                        summary.update(iteration=iteration, ppo_updates=ppo.updates,
                                       seconds=time.perf_counter() - started)
                        log.write(json.dumps(summary) + "\n")
                        log.flush()
                        print(json.dumps(summary), flush=True)
                        if recorder is not None and iteration % 5 == 0:
                            # 有界烟测每5轮持久化一次，避免只依赖native shutdown之前的finally。
                            manifest["reward_trace"] = recorder.flush()
                        if iteration % 100 == 0 or iteration == args.iterations:
                            save_checkpoint(run_dir / f"model_{iteration:05d}.pt", model, ppo, cfg, iteration, asset)
                result = {"status": "pilot_finished_requires_evaluation", "iterations": args.iterations,
                          "start_iteration": start_iteration,
                          "updates_this_run": args.iterations - start_iteration,
                          "walking_verified": False,
                          "total_accepted_steps":ppo.total_accepted_steps,
                          "diagnostic_only":True}
    except BaseException as exc:
        # 包括人工中断；操作系统强制杀进程时无法保证Python有机会写最终文件。
        failure = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        failure_text = traceback.format_exc()
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        try:
            if recorder is not None:
                try:
                    trace_summary = recorder.flush()
                    manifest["reward_trace"] = trace_summary
                except BaseException as exc:
                    # 诊断落盘失败也必须继续保存主运行事实，不能被native close吞掉。
                    failure = {"phase":"diagnostic_flush", "type":type(exc).__name__,
                               "message":str(exc), "earlier_failure":failure}
                    failure_text += "\n诊断落盘失败：\n" + traceback.format_exc()
                    exit_code = exit_code or 2
            # Kit 的 native shutdown 可能直接结束 Python，finally 之后并不保证可达。
            # 测量先单独持久化，最终记录先标记 cleanup_pending，不提前宣称清理成功。
            # 若初始化或测量已失败，同样必须在调用 Kit 关闭前写出失败原因。
            if failure is not None:
                if result is not None:
                    write_json(run_dir / "measurement.json", result)
                (run_dir / "failure.txt").write_text(failure_text, encoding="utf-8")
                write_json(run_dir / "result.json", {"status": "failed", "mode": args.mode,
                           "failure": failure, "walking_verified": False, "cleanup_status": "pending"})
                manifest["status"] = "failed"
            elif result is not None:
                write_json(run_dir / "measurement.json", result)
                write_json(run_dir / "result.json", {"status": "cleanup_pending", "mode": args.mode,
                           "measurement_file": "measurement.json", "walking_verified": False})
                manifest["status"] = "cleanup_pending"
            write_json(run_dir / "manifest.json", manifest)
        except BaseException as exc:
            failure = {"phase": "pre_shutdown_recording", "type": type(exc).__name__,
                       "message": str(exc), "earlier_failure": failure}
            failure_text += "\n关闭前保存记录失败：\n" + traceback.format_exc()
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else (exit_code or 2)
        try:
            # helper内部用try/finally：即使env.close抛错，也继续调用app.close。
            close_resources(env, app)
        except BaseException as exc:
            cleanup_failure = {"phase": "cleanup", "type": type(exc).__name__, "message": str(exc)}
            if failure is None:
                failure = cleanup_failure
            else:
                failure["cleanup_failure"] = cleanup_failure
            failure_text += "\n清理资源时发生异常：\n" + traceback.format_exc()
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else (exit_code or 2)

    # 清理完成后再提交结果；清理失败不能在磁盘上留下“已成功”的最终状态。
    finished_at = datetime.now().astimezone().isoformat()
    if failure is not None:
        result = {"status": "failed", "mode": args.mode, "failure": failure,
                  "walking_verified": False, "finished_at": finished_at}
        manifest["status"] = "failed"
    else:
        result["finished_at"] = finished_at
        result["cleanup_status"] = "completed"
        manifest["status"] = "completed"
    manifest["finished_at"] = finished_at
    try:
        if failure is not None:
            (run_dir / "failure.txt").write_text(failure_text, encoding="utf-8")
            # 即使诊断元数据本身损坏，也优先留下可读取的失败结果。
            write_json(run_dir / "result.json", result)
        write_json(run_dir / "manifest.json", manifest)
        if failure is None:
            write_json(run_dir / "result.json", result)
    except Exception as exc:
        print(f"无法保存最终运行记录到 {run_dir}：{type(exc).__name__}: {exc}", file=sys.stderr)
        return exit_code or 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # 0表示这次诊断/测量/训练任务完成；是否会走必须查看评估门槛与具体结果。
    # static_pd_smoke_passed和nominal_pose_unstable都是已完成的默认姿态测量。
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
