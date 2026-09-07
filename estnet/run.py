"""论文六组消融的统一入口：python -m estnet.run train --variant fullest --asset /path/to/G1.usd。

smoke 测量零动作下的默认 PD 姿态；train 从零开始或恢复 PPO 训练；evaluate 测量检查点。
EstNet无latent/decoder；Key1有速度估计、16维latent和decoder，Key2额外估计足周高度图。
FullEst再估计身体高度；IrrEst只估计身体高度；Implicit只有latent，没有任何显式估计头。
critic读取仿真真值，actor只接收当前本体观测、命令和从历史估计出的信息；各模式均无mimic。
具体网络和更新公式分别在 networks.py、ppo.py；论文差异及参数来源见 docs/PLAN.md。
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from .config import Config
from .factory import build_model, build_ppo, canonical_variant, config_for_variant
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


def _parse_indices(value):
    """解析单机各 rank 的明确 GPU 编号，拒绝重复或隐式共享同一张卡。"""
    try:
        indices = [int(item.strip()) for item in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("GPU编号必须是逗号分隔的整数") from exc
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise ValueError("GPU编号必须非负且互不重复")
    return indices


def _distributed_layout(args):
    """只读 torchrun 环境变量，尚不创建进程组、目录或 CUDA 上下文。"""
    if not args.distributed:
        if args.gpu_indices is not None or args.render_gpu_indices is not None:
            raise ValueError("gpu-indices/render-gpu-indices仅用于distributed训练")
        return None
    if args.mode != "train" or args.run_dir is None:
        raise ValueError("distributed只用于train，且必须提供共享run-dir")
    if args.render_gpu is not None:
        raise ValueError("distributed请使用逐rank的render-gpu-indices")
    try:
        rank_id, local_rank, size = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", size))
    except (KeyError, ValueError) as exc:
        raise ValueError("distributed必须通过torchrun提供RANK/LOCAL_RANK/WORLD_SIZE") from exc
    if size < 2 or local_size != size or rank_id != local_rank or not 0 <= rank_id < size:
        raise ValueError("当前仅支持同一主机上至少两个rank，global/local rank必须一致")
    physical = _parse_indices(args.gpu_indices) if args.gpu_indices is not None else None
    render = _parse_indices(args.render_gpu_indices) if args.render_gpu_indices is not None else physical
    if args.device == "cpu":
        if physical is not None or render is not None:
            raise ValueError("CPU Gloo运行不接受GPU编号")
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            raise ValueError("distributed禁止CUDA_VISIBLE_DEVICES mask；请取消该变量并使用gpu-indices")
        if physical is None or len(physical) != size or len(render) != size:
            raise ValueError("每个rank必须有一个gpu-indices及对应渲染编号")
    return {"rank": rank_id, "local_rank": local_rank, "world_size": size,
            "gpu_indices": physical, "render_gpu_indices": render,
            "backend": "gloo" if args.device == "cpu" else "nccl",
            "root_directory": str(args.run_dir.resolve()), "result_scope": "per_rank"}


def _uuid_key(value):
    """兼容Torch与nvidia-smi的GPU UUID格式，不把ordinal相等当成设备相同。"""
    return str(value).lower().removeprefix("gpu-").replace("-", "")


def _distributed_cuda_device(args, layout):
    """按nvidia-smi物理编号找UUID，再定位对应Torch ordinal，返回已验证设备。"""
    import torch
    queried = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20)
    inventory = {int(row[0].strip()): row[1].strip()
                 for row in csv.reader(queried.stdout.splitlines()) if len(row) == 2}
    physical = layout["gpu_indices"][layout["local_rank"]]
    if physical not in inventory:
        raise ValueError(f"nvidia-smi没有物理GPU{physical}")
    expected = inventory[physical]
    matches = [(index, torch.cuda.get_device_properties(index)) for index in range(torch.cuda.device_count())]
    matches = [(index, props) for index, props in matches
               if _uuid_key(getattr(props, "uuid", "unavailable")) == _uuid_key(expected)]
    if len(matches) != 1:
        raise ValueError(f"不能将物理GPU{physical} UUID={expected}唯一映射到Torch设备")
    ordinal, props = matches[0]
    args.render_gpu = layout["render_gpu_indices"][layout["local_rank"]]
    return torch.device("cuda", ordinal), {
        "type": "cuda", "physical_index": physical, "logical_index": ordinal,
        "uuid": str(props.uuid), "nvidia_smi_uuid": expected, "uuid_verified": True,
        "name": props.name, "exclusive_reservation": False,
        "kit_index_mapping": "explicit_render_indices" if args.render_gpu_indices else "defaults_to_physical_indices_requires_Kit_verification"}


def _initialize_process_group(device):
    """CUDA用NCCL、CPU接口测试用Gloo；torchrun提供env://会合信息。"""
    import torch.distributed as dist
    if dist.is_initialized():
        raise ValueError("runner要求自己管理进程组，不能复用未声明的已有进程组")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=180))


def _local_training_config(cfg, layout, start_iteration):
    """checkpoint/PPO保留全局配置，只给仿真和采样构造每rank的环境数量。"""
    if layout is None:
        return cfg
    size = layout["world_size"]
    if cfg.num_envs % size:
        raise ValueError("全局num-envs必须能被world_size整除")
    local_count = cfg.num_envs // size
    if local_count * cfg.horizon < cfg.minibatches:
        raise ValueError("每rank采样量不足以覆盖配置中的minibatches")
    if (local_count * cfg.horizon) % cfg.minibatches:
        raise ValueError("每rank的num_envs*horizon必须能被minibatches整除")
    seed = (cfg.seed + 1000003 * layout["rank"] + (97 * start_iteration if layout["rank"] else 0)) % (2**63 - 1)
    layout.update(global_num_envs=cfg.num_envs, local_num_envs=local_count,
                  global_rollout_samples=cfg.num_envs * cfg.horizon,
                  local_rollout_samples=local_count * cfg.horizon, environment_seed=seed)
    return replace(cfg, num_envs=local_count, seed=seed)


def _validate_distributed_rng(payload):
    """在CPU预检每rank保存的RNG结构，不用CUDA生成器检查CUDA字节格式。"""
    import torch
    states = payload.get("distributed_rng")
    if states is None:
        return
    metadata = payload.get("distributed")
    if not isinstance(metadata, dict) or type(metadata.get("world_size")) is not int:
        raise ValueError("distributed_rng需要明确的world_size元数据")
    size = metadata["world_size"]
    if not isinstance(states, list) or len(states) != size or size < 2:
        raise ValueError("distributed_rng必须完整覆盖保存时的所有rank")
    for index, item in enumerate(states):
        if not isinstance(item, dict) or item.get("rank") != index:
            raise ValueError("distributed_rng必须按rank排序且不能重复")
        for name in ("torch_rng", "torch_cuda_rng"):
            if name not in item and name == "torch_cuda_rng":
                continue
            value = item.get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or
                    value.device.type != "cpu" or value.ndim != 1 or value.numel() == 0):
                raise ValueError(f"rank{index}的{name}必须是非空CPU字节向量")
            if name == "torch_rng":
                torch.Generator(device="cpu").set_state(value)


def _restore_rank_rng(payload, layout, device, cfg, iteration):
    """完整Adam恢复后分配各rank随机流；新回合/变更world均不是精确现场续训。"""
    import torch
    index, size = layout["rank"], layout["world_size"]
    states = payload.get("distributed_rng")
    saved_size = payload.get("distributed", {}).get("world_size", 1)
    if states is not None and saved_size == size:
        selected = states[index]
        torch.set_rng_state(selected["torch_rng"])
        cuda_restored = device.type == "cuda" and "torch_cuda_rng" in selected
        if cuda_restored:
            torch.cuda.set_rng_state(selected["torch_cuda_rng"], device)
        elif device.type == "cuda":
            torch.cuda.manual_seed((cfg.seed + 1000003 * index + 97 * iteration) % (2**63 - 1))
        mode = "saved_per_rank_rng"
    elif index == 0:
        # 上层restore_training_state已恢复旧checkpoint的rank0/单卡随机状态。
        cuda_restored = device.type == "cuda" and "torch_cuda_rng" in payload
        mode = "legacy_or_changed_world_rank0_rng"
    else:
        seed = (cfg.seed + 1000003 * index + 97 * iteration) % (2**63 - 1)
        torch.manual_seed(seed)
        cuda_restored = False
        mode = "derived_rank_seed"
    return {"mode": mode, "saved_world_size": saved_size, "world_size": size,
            "rank": index, "cuda_rng_restored": cuda_restored,
            "environment_state_restored": False, "exact_resume": False}


def _save_distributed_checkpoint(path, model, ppo, cfg, iteration, asset, layout):
    """所有rank同时收集RNG，仅rank0写公共权重/Adam，避免并发覆盖检查点。"""
    import torch
    import torch.distributed as dist
    local = {"rank": layout["rank"], "torch_rng": torch.get_rng_state()}
    device = next(model.parameters()).device
    if device.type == "cuda":
        local["torch_cuda_rng"] = torch.cuda.get_rng_state(device)
    states = [None] * layout["world_size"]
    dist.all_gather_object(states, local)
    if layout["rank"] == 0:
        metadata = {name: layout[name] for name in (
            "world_size", "global_num_envs", "local_num_envs", "global_rollout_samples", "local_rollout_samples", "devices")}
        save_checkpoint(path, model, ppo, cfg, iteration, asset,
                        distributed_rng=states, distributed_metadata=metadata)


def _resolved_physics_manifest(env_cfg):
    """记录真正交给环境的物理参数，不序列化带callable的整个Isaac配置。"""
    sim = getattr(env_cfg, "sim", None)
    scene = getattr(env_cfg, "scene", None)
    result = {"sim": {name: getattr(sim, name, None) for name in ("dt", "device", "render_interval")},
              "decimation": getattr(env_cfg, "decimation", None),
              "episode_length_s": getattr(env_cfg, "episode_length_s", None),
              "scene": {name: getattr(scene, name, None) for name in
                        ("num_envs", "env_spacing", "replicate_physics", "clone_in_fabric")}}
    physx = getattr(sim, "physx", None)
    if physx is not None:
        try:
            values = physx.to_dict()
            json.dumps(values, allow_nan=False)
            result["physx"] = values
        except Exception as exc:  # noqa: BLE001 - 记录配置序列化问题，不阻断资源清理
            result["physx_serialization_error"] = f"{type(exc).__name__}: {exc}"
            result["physx_enable_stabilization"] = getattr(physx, "enable_stabilization", None)
    return result


def _training_state_digest(model, ppo):
    """仅在首个协同更新后哈希模型及完整Adam，用内容校验而不是日志相同作证据。"""
    import torch
    digest = hashlib.sha256()

    def visit(value):
        """跨设备采用相同dtype/shape/原始字节，字典键排序消除插入顺序差异。"""
        if isinstance(value, torch.Tensor):
            descriptor = ["tensor", str(value.dtype), list(value.shape)]
            digest.update(json.dumps(descriptor).encode())
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps([type(value).__name__, value], sort_keys=True, allow_nan=False).encode())

    visit({"model": model.state_dict(), "ppo": ppo.state_dict()})
    return digest.hexdigest()


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
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dictionary")  # noqa: TRY004 - 损坏检查点统一ValueError
    cfg = Config(**payload["config"])
    if cfg.schema != payload.get("schema"):
        raise ValueError("Checkpoint config schema is incompatible")
    # 配置中的任意 NaN/Infinity 都会破坏动力学或审计记录，不能只检查高度。
    json.dumps(cfg.to_dict(), allow_nan=False)
    cfg.validate()
    _validate_distributed_rng(payload)
    if type(payload.get("iteration")) is not int or payload["iteration"] < 0:
        raise ValueError("Checkpoint iteration must be a non-negative integer")
    if _asset_signature(payload.get("asset")) != _asset_signature(asset):
        raise ValueError("Checkpoint asset signature differs from the current asset")
    # strict 同时检查键名和张量形状；单凭外层 schema 相同不能证明模型可加载。
    validator = build_model(cfg)
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
            # 先复制动作对应的旧观测，避免环境内部复用缓冲区造成时序错位。
            row = {k: v.clone() for k, v in obs.items()}
            next_obs, reward, terminated, truncated, info = env.step(action)
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


def save_checkpoint(path, model, ppo, config, iteration, asset, *, distributed_rng=None, distributed_metadata=None):
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
    if distributed_rng is not None:
        state["distributed_rng"] = distributed_rng
        state["distributed"] = distributed_metadata
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

    parser = argparse.ArgumentParser(description="G1论文六组消融：训练、续训、评估与姿态诊断。")
    parser.add_argument("mode", choices=["smoke", "train", "evaluate"],
                        help="smoke诊断默认姿态；train训练或续训；evaluate测量检查点的首个回合")
    parser.add_argument("--asset", type=Path, required=True, help="已核对内容签名的29关节G1 USD文件")
    parser.add_argument("--variant", type=canonical_variant,
                        choices=["estnet", "key1", "key2", "fullest", "irrest", "implicit"], default=None,
                        help="新训练默认estnet；IllEst拼写等同irrest；评估和续训默认读取检查点架构")
    parser.add_argument("--num-envs", type=int, help="并行环境数；训练默认4096，诊断和评估默认16")
    parser.add_argument("--iterations", type=int, default=500,
                        help="训练结束时的总PPO更新数，默认500；从500续至10000时仅追加9500轮")
    parser.add_argument("--seed", type=int, default=None,
                        help="新运行默认42；续训省略时沿用检查点种子，评估可独立指定")
    parser.add_argument("--device", default="cuda:0",
                        help="张量与物理设备，例如cpu或cuda:3；CPU物理模式不保证Kit完全不使用GPU")
    parser.add_argument("--render-gpu", type=int,
                        help="Kit GPU表中的渲染编号；须按UUID核对，不能假定等于CUDA编号")
    parser.add_argument("--distributed", action="store_true",
                        help="由torchrun启动同一策略的多rank同步训练，num-envs仍表示全局环境数")
    parser.add_argument("--gpu-indices", help="各LOCAL_RANK对应的nvidia-smi物理GPU编号，例如0,1；禁止CVD mask")
    parser.add_argument("--render-gpu-indices", help="各rank的Kit渲染编号；省略时取gpu-indices，仍需核对Kit映射")
    parser.add_argument("--cpu-threads", type=int, default=8, help="本进程的CPU线程上限，默认8")
    parser.add_argument("--headless", action="store_true", help="不打开模拟器交互窗口")
    parser.add_argument("--run-dir", type=Path, help="新建运行目录；已存在的目录不会被覆盖")
    parser.add_argument("--checkpoint", type=Path, help="仅用于evaluate的模型检查点")
    parser.add_argument("--resume", type=Path, help="仅用于train；恢复模型、Adam、学习率及更新计数")
    parser.add_argument("--target-height", type=float, default=None,
                        help="诊断或从零训练的高度目标；省略时使用默认配置；评估必须沿用检查点配置")
    args = parser.parse_args()

    # 能在纯参数阶段拒绝的输入，不应等到初始化GPU或加载USD时才报错。
    if args.iterations < 1:
        parser.error("iterations必须为正整数")
    if args.cpu_threads < 1:
        parser.error("cpu-threads必须为正整数")
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
    try:
        layout = _distributed_layout(args)
    except ValueError as exc:
        parser.error(str(exc))

    run_root = (args.run_dir or Path("logs") / f"{args.mode}-{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}").resolve()
    run_dir = run_root / f"rank-{layout['rank']}" if layout is not None else run_root
    try:
        if layout is not None:
            # torchrun父进程可预建共享根；不同rank只竞争创建父目录，不共享rank目录。
            run_root.mkdir(parents=True, exist_ok=True)
            if any(run_root.glob("model_*.pt")):
                raise FileExistsError("共享run-dir已含公共检查点，请创建新的续训目录")
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"无法创建新的运行目录 {run_dir}：{exc}", file=sys.stderr)
        return 2

    # 所有资源都从空状态开始，初始化中途失败也会进入统一清理路径。
    env = None
    app = None
    process_group_started = False
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
            cfg = replace(config_for_variant(args.variant or "estnet"),
                          num_envs=num_envs, seed=42 if args.seed is None else args.seed)
            if args.target_height is not None:
                cfg = replace(cfg, reward_target_height=args.target_height)
        if args.variant is not None and cfg.variant != args.variant:
            raise ValueError("variant与检查点架构不一致，不能跨论文消融架构加载权重")
        cfg.validate()
        local_cfg = _local_training_config(cfg, layout, start_iteration)
        local_cfg.validate()
        manifest["config"] = cfg.to_dict()
        manifest["variant"] = cfg.variant
        if layout is not None:
            manifest["distributed"] = layout
            manifest["local_config"] = local_cfg.to_dict()
        if checkpoint is not None and (layout is not None or "distributed" in checkpoint) and args.resume is not None:
            saved_size = checkpoint.get("distributed", {}).get("world_size", 1)
            manifest["resume"]["distributed_transition"] = {
                "saved_world_size": saved_size, "world_size": layout["world_size"] if layout else 1,
                "global_num_envs_preserved": cfg.num_envs, "new_episodes": True, "exact_resume": False,
                "note": "保留全局环境数、PPO参数和完整Adam；进程拓扑、仿真现场与随机流分配可能改变。"}
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
            if layout is not None:
                device, selected_device = _distributed_cuda_device(args, layout)
            else:
                device = torch.device("cuda", 0 if device.index is None else device.index)
            torch.cuda.set_device(device)
            logical_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(logical_index)
            selected_device.update(logical_index=logical_index, name=properties.name,
                                   uuid=str(getattr(properties, "uuid", "unavailable")))
        args.device = str(device)
        torch.manual_seed(local_cfg.seed)
        try:
            runtime = runtime_manifest(args)
        except Exception as exc:  # noqa: BLE001 - 版本采集失败须记录，不伪造环境信息
            # 版本信息采集失败单独记录；不能伪造版本，也不将此辅助诊断当成动力学故障。
            runtime = {"collection_error": f"{type(exc).__name__}: {exc}"}
        runtime["selected_torch_device"] = selected_device
        runtime["requested_kit_render_gpu"] = args.render_gpu
        manifest["runtime"] = runtime
        write_json(run_dir / "manifest.json", manifest)
        if layout is not None:
            phase = "distributed_initialization"
            import torch.distributed as dist

            from . import distributed as sync
            _initialize_process_group(device)
            process_group_started = True
            devices = [None] * layout["world_size"]
            dist.all_gather_object(devices, {"rank": layout["rank"], **selected_device,
                                            "render_gpu": args.render_gpu})
            if device.type == "cuda" and len({_uuid_key(item["uuid"]) for item in devices}) != layout["world_size"]:
                raise ValueError("多个rank解析到了同一GPU UUID")
            layout["devices"] = devices
            manifest["distributed"] = layout
            write_json(run_dir / "manifest.json", manifest)
            if layout["rank"] == 0:
                write_json(run_root / "distributed.json", {
                    **layout, "config": cfg.to_dict(),
                    "checkpoint_directory": str(run_root),
                    "rank_result_files": [str(run_root / f"rank-{index}" / "result.json") for index in range(layout["world_size"])],
                    "note": "此文件只描述协同布局；必须检查每个rank结果和进程退出，不能用rank0独自结束认定全部完成。"})

        phase = "application_initialization"
        try:
            from isaaclab.app import AppLauncher
        except ImportError as exc:
            raise RuntimeError("请使用本实验已核验的Isaac Lab/Isaac Sim Python环境；当前无法导入Isaac Lab") from exc

        class SelectedGpuLauncher(AppLauncher):
            def _config_resolution(self, launcher_args):
                if layout is not None:
                    # Lab2.3.2在distributed=True时会按LOCAL_RANK覆盖设备；同步由本runner管理。
                    launcher_args["distributed"] = False
                super()._config_resolution(launcher_args)
                # Isaac Sim 4.5 默认快速关闭可能在 native shutdown 中直接退出进程。
                # 关闭快速退出；关键测量仍须在 app.close() 前落盘，不能依赖它返回。
                self._sim_app_config["fast_shutdown"] = False
                if args.render_gpu is not None:
                    # 这是固定Isaac Lab 2.0.2版本的钩子，必须在SimulationApp创建前设置。
                    # Kit渲染编号和CUDA逻辑编号分别记录，不能根据整数相等推断物理GPU相同。
                    self._sim_app_config["active_gpu"] = args.render_gpu
                    self._sim_app_config["multi_gpu"] = False
                if layout is not None and device.type == "cuda" and (
                    self.device_id != device.index or self._sim_app_config.get("physics_gpu") != device.index
                ):
                    raise RuntimeError("AppLauncher改变了已按UUID验证的CUDA物理设备")

        original_argv = list(sys.argv)
        sys.argv.extend(["--portable-root", str(run_dir / "kit-user")])
        try:
            launcher = SelectedGpuLauncher(
                headless=args.headless, device=args.device, multi_gpu=False,
                **({"distributed": False} if layout is not None else {}),
                kit_args=f"--/plugins/carb.tasking.plugin/threadCount={args.cpu_threads}")
            app = launcher.app
        finally:
            # AppLauncher构造失败时也必须恢复进程参数，避免污染后续错误处理。
            sys.argv[:] = original_argv

        phase = "environment_initialization"
        if layout is not None and device.type == "cuda":
            # Kit原生插件可能切换当前CUDA设备，张量/通信仍必须使用该rank的已验证设备。
            torch.cuda.set_device(device)
        from .environment import EstNetEnv, build_env_cfg
        env_cfg = build_env_cfg(local_cfg, args.asset.resolve(), args.device)
        env_cfg.detailed_diagnostics = args.mode == "smoke"
        manifest["resolved_physics"] = _resolved_physics_manifest(env_cfg)
        write_json(run_dir / "manifest.json", manifest)
        env = EstNetEnv(env_cfg)
        obs, _ = env.reset()
        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)

        if args.mode == "smoke":
            phase = "nominal_pose_measurement"
            # 只执行零动作接口与默认姿态诊断，不为此构造大型actor、critic或估计器。
            # 零策略姿态不稳是一项测量结果，不等于模拟器损坏或学习策略无法站稳。
            result = smoke_check(env, obs, cfg)
        else:
            model = build_model(cfg).to(env.device)
            if args.mode == "evaluate":
                phase = "checkpoint_evaluation"
                model.load_state_dict(checkpoint["model"], strict=True)
                checkpoint = None
                result = evaluate_first_episodes(env, model.eval(), obs, cfg)
                result["checkpoint"] = str(args.checkpoint.resolve())
                result["iteration"] = checkpoint_iteration
            else:
                phase = "training"
                ppo = build_ppo(model, cfg)
                if args.resume is not None:
                    from .resume import restore_training_state
                    restored = restore_training_state(model, ppo, checkpoint)
                    if layout is not None:
                        restored["rank_rng"] = _restore_rank_rng(checkpoint, layout, device, cfg, start_iteration)
                    manifest["resume"]["restored_state"] = restored
                    write_json(run_dir / "manifest.json", manifest)
                    print(json.dumps({"event": "training_resumed", **manifest["resume"]},
                                     ensure_ascii=False), flush=True)
                    checkpoint = None
                if layout is not None:
                    # 每rank均已恢复完整Adam；此处只统一权重/buffer，绝不以广播替代Adam恢复。
                    sync.broadcast_model(model)
                with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
                    # 保持既有PPO更新、参数和保存间隔；这里只管理运行与记录。
                    for iteration in range(start_iteration + 1, args.iterations + 1):
                        started = time.perf_counter()
                        obs, batch, summary = collect_rollout(env, model, obs, local_cfg)
                        if layout is not None:
                            # 各rank具有相同环境数和步数，因此局部均值的平均就是全局均值。
                            summary = {key: sync.mean_scalar(summary[key]) for key in sorted(summary)}
                        summary.update({"ppo/" + key: value for key, value in ppo.update(batch).items()})
                        if ppo.updates != iteration:
                            raise RuntimeError("PPO更新计数与训练全局轮次不一致")
                        if layout is not None and iteration == start_iteration + 1:
                            # 只做一次完整状态校验，避免每轮将大型模型和Adam拷回CPU。
                            digests = [None] * layout["world_size"]
                            dist.all_gather_object(digests, _training_state_digest(model, ppo))
                            manifest["first_distributed_update"] = {
                                "iteration": iteration, "model_and_adam_sha256_by_rank": digests,
                                "all_equal": len(set(digests)) == 1}
                            write_json(run_dir / "manifest.json", manifest)
                            if len(set(digests)) != 1:
                                raise RuntimeError("首个协同更新后各rank的模型/Adam状态不一致")
                        duration = time.perf_counter() - started
                        summary.update(iteration=iteration, ppo_updates=ppo.updates,
                                       seconds=sync.mean_scalar(duration) if layout is not None else duration)
                        log.write(json.dumps(summary) + "\n")
                        log.flush()
                        print(json.dumps(summary), flush=True)
                        if iteration % 100 == 0 or iteration == args.iterations:
                            if layout is None:
                                save_checkpoint(run_dir / f"model_{iteration:05d}.pt", model, ppo, cfg, iteration, asset)
                            else:
                                _save_distributed_checkpoint(run_root / f"model_{iteration:05d}.pt", model, ppo, cfg, iteration, asset, layout)
                result = {"status": "pilot_finished_requires_evaluation", "iterations": args.iterations,
                          "start_iteration": start_iteration,
                          "updates_this_run": args.iterations - start_iteration,
                          "walking_verified": False}
                if layout is not None:
                    result.update(result_scope="local_rank", rank=layout["rank"], world_size=layout["world_size"],
                                  global_num_envs=cfg.num_envs, local_num_envs=local_cfg.num_envs,
                                  checkpoint_directory=str(run_root), all_ranks_cleanup_verified=False)
    except BaseException as exc:  # noqa: BLE001 - 包括中断也必须落盘并释放仿真资源
        # 包括人工中断；操作系统强制杀进程时无法保证Python有机会写最终文件。
        failure = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        failure_text = traceback.format_exc()
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        try:
            # Kit 的 native shutdown 可能直接结束 Python，finally 之后并不保证可达。
            # 测量先单独持久化，最终记录先标记 cleanup_pending，不提前宣称清理成功。
            # 若初始化或测量已失败，同样必须在调用 Kit 关闭前写出失败原因。
            if failure is not None:
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
        except BaseException as exc:  # noqa: BLE001 - 记录失败不能跳过后续清理
            failure = {"phase": "pre_shutdown_recording", "type": type(exc).__name__,
                       "message": str(exc), "earlier_failure": failure}
            failure_text += "\n关闭前保存记录失败：\n" + traceback.format_exc()
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else (exit_code or 2)
        if process_group_started:
            try:
                # 不调用barrier：任一rank已失败时，清理阶段等待同伴会造成死锁。
                # 先结束NCCL/Gloo，再关闭Kit，防止native shutdown抢先退出Python。
                dist.destroy_process_group()
                manifest["process_group_cleanup"] = "completed"
            except BaseException as exc:  # noqa: BLE001 - 进程组清理失败仍须关闭Kit
                group_failure = {"phase": "distributed_cleanup", "type": type(exc).__name__, "message": str(exc)}
                if failure is None:
                    failure = group_failure
                else:
                    failure["distributed_cleanup_failure"] = group_failure
                failure_text += "\n进程组清理失败：\n" + traceback.format_exc()
                exit_code = exit_code or 2
        try:
            # helper内部用try/finally：即使env.close抛错，也继续调用app.close。
            close_resources(env, app)
        except BaseException as exc:  # noqa: BLE001 - 清理异常也必须留下最终失败结果
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
    except Exception as exc:  # noqa: BLE001 - 最终写盘失败必须返回非零退出码
        print(f"无法保存最终运行记录到 {run_dir}：{type(exc).__name__}: {exc}", file=sys.stderr)
        return exit_code or 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # 0表示这次诊断/测量/训练任务完成；是否会走必须查看评估门槛与具体结果。
    # static_pd_smoke_passed和nominal_pose_unstable都是已完成的默认姿态测量。
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
