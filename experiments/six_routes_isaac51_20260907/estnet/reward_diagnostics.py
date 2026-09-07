"""有上限的奖励输入记录器及CPU复算；不改变动作、奖励或仿真状态。

调用顺序为capture_pre -> capture_reward -> capture_post。capture_reward必须在
奖励previous缓冲更新和环境reset前；capture_post明确属于env.step返回之后，可能含重置状态。
CLI：python -m estnet.reward_diagnostics {verify,analyze} /path/to/reward-trace
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np
import torch

from .rewards import reward_terms


def _sha(path: Path) -> str:
    """文件内容签名用于确认离线重算仍使用同一个奖励实现。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """只转换显式提供的元数据，不补造缺失的物理量。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_json(path: Path, value: dict) -> None:
    """先写临时文件并同步，再原子替换本记录器自己的输出。"""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RewardTraceRecorder:
    """记录前若干环境的有限步数，默认16环境×480步，上限16环境×1000步。

    fields可以是批量Tensor、批量numpy数组、标量或它们的字典。
    非标量首维必须等于完整环境数；全局关节名称/限位等放static_metadata。
    每次读取都立即复制所选环境到独立CPU内存，禁止持有可变仿真缓冲的view。
    """

    def __init__(self, cfg: Any, *, max_steps: int = 480, num_envs: int = 16,
                 output_dir: str | Path | None = None, static_metadata: dict | None = None,
                 atol: float = 1e-6, rtol: float = 1e-5,
                 max_bytes: int = 256 * 1024 * 1024, enabled: bool = True):
        if type(max_steps) is not int or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer in [1,1000]")
        if type(num_envs) is not int or not 1 <= num_envs <= 16:
            raise ValueError("num_envs must be an integer in [1,16]")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
            raise ValueError("comparison tolerances must be finite and non-negative")
        config = cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg) if is_dataclass(cfg) else vars(cfg)
        self.config = _json_value(config)
        # 固定奖励配置副本，防止调用方后来修改可变配置污染复算。
        self.cfg = SimpleNamespace(**json.loads(json.dumps(self.config, allow_nan=False)))
        self.max_steps, self.num_envs = max_steps, num_envs
        self.max_bytes, self.atol, self.rtol = max_bytes, atol, rtol
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.static_metadata = _json_value(static_metadata or {})
        self.recorder_id = uuid.uuid4().hex
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.frames: list[dict[str, torch.Tensor]] = []
        self.pending: dict[str, torch.Tensor] | None = None
        self.full_envs: int | None = None
        self.selected_envs: int | None = None
        self.retained_bytes = 0
        self.stop_reason: str | None = None if enabled else "disabled"
        self.errors_by_term: dict[str, float] = {}
        self.verification_failure: dict | None = None
        self.reward_source_sha256 = _sha(Path(__file__).with_name("rewards.py"))

    @property
    def steps(self) -> int:
        """只计完整pre/reward/post三阶段记录，不把残缺帧算成完成。"""
        return len(self.frames)

    @property
    def active(self) -> bool:
        """容量用尽后各捕获函数均返回False，不继续占用训练资源。"""
        return self.stop_reason is None and self.steps < self.max_steps

    def _flatten(self, prefix: str, value: Any) -> dict[str, torch.Tensor]:
        """嵌套字典展开成NPZ字段，先选择环境再进行CPU复制。"""
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if not isinstance(key, str) or "/" in key:
                    raise ValueError("trace dictionary keys must be strings without '/'")
                result.update(self._flatten(prefix + "/" + key, item))
            return result
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim:
            if tensor.shape[0] != self.full_envs:
                raise ValueError(f"{prefix}: expected batch dimension {self.full_envs}, got {tuple(tensor.shape)}")
            tensor = tensor[:self.selected_envs]
        if tensor.is_complex():
            raise ValueError("Complex diagnostic values are unsupported")
        tensor = tensor.detach().cpu().clone()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()  # numpy不支持bfloat16；float32能精确保存其值。
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite recorded value: {prefix}")
        return {prefix: tensor}

    def _append_pending(self, values: dict[str, torch.Tensor]) -> bool:
        """字节容量不足时丢弃未完成帧，保留已完成帧并明确停止原因。"""
        if self.pending is None:
            raise RuntimeError("capture_pre must begin a frame")
        if set(values) & self.pending.keys():
            raise RuntimeError("A trace stage/field was captured twice")
        size = sum(value.numel() * value.element_size() for value in self.pending.values())
        size += sum(value.numel() * value.element_size() for value in values.values())
        if self.retained_bytes + size > self.max_bytes:
            self.pending = None
            self.stop_reason = "byte_cap"
            return False
        self.pending.update(values)
        return True

    def capture_pre(self, observation: dict, action: torch.Tensor,
                    mean: torch.Tensor, std: torch.Tensor, **fields) -> bool:
        """在env.step前保存原始采样/均值/标准差及历史，动作不做限幅。"""
        if not self.active:
            return False
        if self.pending is not None:
            raise RuntimeError("Previous frame has not completed capture_post")
        if action.ndim != 2 or action.shape != mean.shape or action.shape != std.shape:
            raise ValueError("action/mean/std must have the same [N,A] shape")
        if self.full_envs is None:
            self.full_envs = action.shape[0]
            if self.full_envs < 1:
                raise ValueError("An empty environment batch cannot be recorded")
            self.selected_envs = min(self.num_envs, self.full_envs)
        elif action.shape[0] != self.full_envs:
            raise ValueError("Environment count changed within one recorder")
        if not (std[:self.selected_envs] > 0).all():
            raise ValueError("Recorded policy std must be positive")
        self.pending = {}
        payload = {"observation": observation, "action": action, "mean": mean, "std": std, **fields}
        return self._append_pending(self._flatten("pre", payload))

    def capture_reward(self, state: dict, terms: dict, **fields) -> bool:
        """在previous缓存更新和reset前捕获完整state，并用真实奖励逐项核对。"""
        if not self.active:
            return False
        if self.pending is None:
            raise RuntimeError("capture_reward requires capture_pre")
        values = self._flatten("reward", {"state": state, "terms": terms, **fields})
        if not self._append_pending(values):
            return False
        cpu_state = {key.removeprefix("reward/state/"): value for key, value in values.items()
                     if key.startswith("reward/state/")}
        actual = {key.removeprefix("reward/terms/"): value for key, value in values.items()
                  if key.startswith("reward/terms/")}
        expected = reward_terms(cpu_state, self.cfg)
        if set(actual) != set(expected):
            raise ValueError("Recorded reward names differ from real reward_terms")
        failed = []
        for name, value in expected.items():
            if value.shape != actual[name].shape or not torch.isfinite(value).all():
                failed.append(name)
                continue
            error = float((value-actual[name]).abs().max())
            self.errors_by_term[name] = max(self.errors_by_term.get(name, 0.), error)
            if not torch.allclose(value, actual[name], atol=self.atol, rtol=self.rtol):
                failed.append(name)
        if "reward/total_reward" in values:
            summed = torch.stack(list(expected.values())).sum(0)
            if not torch.allclose(summed, values["reward/total_reward"], atol=self.atol, rtol=self.rtol):
                failed.append("total_reward")
        if failed:
            self.verification_failure = {"completed_steps_before_failure": self.steps,
                                         "terms": failed, "max_abs_errors": dict(self.errors_by_term)}
            raise FloatingPointError("Recorded rewards differ from CPU reward_terms: " + ", ".join(failed))
        return True

    def capture_post(self, *, terminated: torch.Tensor, truncated: torch.Tensor, **fields) -> bool:
        """记录step返回后的done/reset信息；该阶段不覆盖已保存的pre-reset奖励状态。"""
        if not self.active:
            return False
        if self.pending is None or not any(key.startswith("reward/state/") for key in self.pending):
            raise RuntimeError("capture_post requires a captured reward state")
        values = self._flatten("post", {"terminated": terminated, "truncated": truncated, **fields})
        if not self._append_pending(values):
            return False
        assert self.pending is not None
        if not torch.equal(self.pending["reward/state/terminated"].bool(), values["post/terminated"].bool()):
            raise ValueError("Reward terminal mask differs from env.step terminal mask")
        if self.frames and set(self.frames[0]) != set(self.pending):
            raise ValueError("Recorded field schema changed between steps")
        if self.frames and any(self.frames[0][key].shape != value.shape for key, value in self.pending.items()):
            raise ValueError("Recorded field shapes changed between steps")
        self.retained_bytes += sum(value.numel()*value.element_size() for value in self.pending.values())
        self.frames.append(self.pending)
        self.pending = None
        if self.steps >= self.max_steps:
            self.stop_reason = "step_cap"
        return True

    def arrays(self) -> dict[str, np.ndarray]:
        """只导出完整帧；stack产生独立数组，调用方不会改写记录器内部数据。"""
        if not self.frames:
            return {}
        return {key: torch.stack([row[key] for row in self.frames]).numpy() for key in self.frames[0]}

    def flush(self, output_dir: str | Path | None = None) -> dict:
        """可在未满容量时主动落盘；残缺失败帧单独保存，不伪装完整轨迹。"""
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        if directory is None:
            raise ValueError("flush needs output_dir")
        directory.mkdir(parents=True, exist_ok=True)
        static_path = directory / "static.json"
        if static_path.exists():
            existing = json.loads(static_path.read_text(encoding="utf-8"))
            if existing.get("recorder_id") != self.recorder_id:
                raise FileExistsError("Refuse to overwrite another recorder's directory")
        arrays = self.arrays()
        temporary = directory / "reward_trace.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / "reward_trace.npz")
        if self.pending is not None:
            with (directory / "incomplete_frame.npz").open("wb") as stream:
                np.savez_compressed(stream, **{key: value.numpy() for key, value in self.pending.items()})
        metadata = {"format_version": 1, "recorder_id": self.recorder_id, "created_at": self.created_at,
            "config": self.config, "reward_source_sha256": self.reward_source_sha256,
            "selected_env_ids": list(range(self.selected_envs or 0)), "full_environment_count": self.full_envs,
            "max_steps": self.max_steps, "max_bytes": self.max_bytes, "atol": self.atol, "rtol": self.rtol,
            "field_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
            "timing": {"pre": "before env.step", "reward": "before previous buffers update and before reset",
                       "post": "after env.step; may contain reset episode observations"},
            "static_metadata": self.static_metadata}
        _save_json(static_path, metadata)
        summary = {"recorder_id": self.recorder_id, "steps": self.steps, "selected_envs": self.selected_envs,
            "retained_tensor_bytes": self.retained_bytes, "stop_reason": self.stop_reason,
            "incomplete_frame_saved": self.pending is not None, "verification_failure": self.verification_failure,
            "max_abs_errors_by_term": self.errors_by_term, "reward_trace_sha256": _sha(directory / "reward_trace.npz"),
            "static_sha256": _sha(static_path), "sampling": "bounded first selected environments; not an unbiased population estimate"}
        _save_json(directory / "summary.json", summary)
        return summary


def _load(directory: str | Path):
    """校验完整文件与奖励源码；损坏文件或不同奖励版本不能直接被标成复算通过。"""
    directory = Path(directory)
    metadata = json.loads((directory / "static.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if _sha(directory / "reward_trace.npz") != summary["reward_trace_sha256"] or _sha(directory / "static.json") != summary["static_sha256"]:
        raise ValueError("Trace/static file SHA mismatch")
    if _sha(Path(__file__).with_name("rewards.py")) != metadata["reward_source_sha256"]:
        raise ValueError("Offline reward source differs from recorded source")
    with np.load(directory / "reward_trace.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if summary["steps"] < 1:
        raise ValueError("No complete reward frames available")
    shape = (summary["steps"], summary["selected_envs"])
    state = {name.removeprefix("reward/state/"): torch.from_numpy(value).flatten(0, 1)
             for name, value in arrays.items() if name.startswith("reward/state/")}
    return arrays, metadata, summary, state, SimpleNamespace(**metadata["config"]), shape


def verify_trace(directory: str | Path) -> dict:
    """离线逐项复算全部完整帧；失败/残缺记录必须保留失败状态。"""
    arrays, metadata, summary, state, cfg, shape = _load(directory)
    computed = reward_terms(state, cfg)
    names = {key.removeprefix("reward/terms/") for key in arrays if key.startswith("reward/terms/")}
    if names != set(computed):
        raise ValueError("Saved reward term names do not match source")
    errors, failed = {}, []
    for name, expected in computed.items():
        actual = torch.from_numpy(arrays["reward/terms/" + name]).flatten()
        if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            failed.append(name)
            continue
        errors[name] = float((actual-expected).abs().max())
        if not torch.allclose(actual, expected, atol=metadata["atol"], rtol=metadata["rtol"]):
            failed.append(name)
    if "reward/total_reward" in arrays:
        total = torch.from_numpy(arrays["reward/total_reward"]).flatten()
        if not torch.allclose(total, torch.stack(list(computed.values())).sum(0), atol=metadata["atol"], rtol=metadata["rtol"]):
            failed.append("total_reward")
    return {"status": "passed" if not failed and summary["verification_failure"] is None and not summary["incomplete_frame_saved"] else "failed",
            "frames": shape[0], "environment_transitions": shape[0]*shape[1], "failed_terms": failed,
            "max_abs_errors_by_term": errors, "incomplete_frame_saved": summary["incomplete_frame_saved"],
            "prior_verification_failure": summary["verification_failure"]}


def kernel_inputs(state: dict, cfg: Any) -> dict[str, torch.Tensor]:
    """明确报告原始核输入；与冻结奖励相同的单位、归约和full3D速度分母。"""
    vel = state["vel"]
    desired_v = torch.cat((state["command"][:, :2], torch.zeros_like(vel[:, :1])), -1)
    desired_w = torch.cat((torch.zeros_like(vel[:, :2]), state["command"][:, 2:3]), -1)
    speed = vel.norm(dim=-1).clamp_min(cfg.reward_speed_floor)
    weight = (state["mass"]*9.81).clamp_min(1e-6)
    return {"linear_velocity": (vel-desired_v).norm(dim=-1), "yaw_velocity": (state["ang_vel"]-desired_w).norm(dim=-1),
        "upright": 1-state["up"], "height": cfg.reward_target_height-state["height"],
        "stance_velocity": (state["stance"]*state["foot_vel"].norm(dim=-1)).sum(-1),
        "swing_force": ((1-state["stance"])*state["foot_force"].norm(dim=-1)).sum(-1)/weight,
        "impact": (state["foot_force"]-state["prev_foot_force"]).flatten(1).norm(dim=-1)/weight,
        "torque_smoothness": (state["torque"]-state["prev_torque"]).norm(dim=-1),
        "joint_velocity_smoothness": (state["joint_vel"]-state["prev_joint_vel"]).norm(dim=-1)/speed,
        "cost_of_transport": (state["torque"]*state["joint_vel"]).abs().sum(-1)/(weight*speed)}


def _statistics(value: np.ndarray) -> dict | None:
    """空分组返回None；不以零伪装缺失数据。"""
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if not value.size:
        return None
    if not np.isfinite(value).all():
        raise FloatingPointError("Non-finite analysis input")
    return {"count": int(value.size), "mean": float(value.mean()), "min": float(value.min()),
            "p10": float(np.quantile(value, .1)), "median": float(np.median(value)),
            "p90": float(np.quantile(value, .9)), "p99": float(np.quantile(value, .99)), "max": float(value.max())}


def analyze_trace(directory: str | Path) -> dict:
    """按显式相位、实际提供的足高、速度方向分组；不从接触力猜离地高度。"""
    verified = verify_trace(directory)
    if verified["status"] != "passed":
        raise ValueError("Cannot analyze a trace whose rewards failed verification")
    arrays, metadata, summary, state, cfg, shape = _load(directory)
    inputs = {name: value.numpy() for name, value in kernel_inputs(state, cfg).items()}
    terms = {name.removeprefix("reward/terms/"): value.reshape(-1) for name, value in arrays.items()
             if name.startswith("reward/terms/")}

    def group(mask):
        mask = np.asarray(mask).reshape(-1)
        return {"transitions": int(mask.sum()), "kernel_inputs": {name: _statistics(value[mask]) for name, value in inputs.items()},
                "rewards": {name: _statistics(value[mask]) for name, value in terms.items()}}

    phase = state["stance"].numpy()
    height = arrays.get("reward/foot_height")
    if height is not None and height.shape != (*shape, 2):
        raise ValueError("reward/foot_height must be [T,selected_envs,2]")
    height = height.reshape(-1, 2) if height is not None else None
    contact = arrays.get("reward/foot_contact", arrays.get("reward/contact"))
    if contact is not None and contact.shape != (*shape, 2):
        raise ValueError("Recorded foot contact must be [T,selected_envs,2]")
    contact = contact.reshape(-1,2).astype(bool) if contact is not None else None
    phase_groups, phase_height_groups, phase_contact_groups = {}, {}, {}
    height_bins = [(-np.inf,0., "below_zero"), (0.,.03,"0_to_0p03m"), (.03,.06,"0p03_to_0p06m"),
                   (.06,.1,"0p06_to_0p10m"), (.1,np.inf,"at_least_0p10m")]
    for foot, name in enumerate(("left", "right")):
        for label, mask in (("stance",phase[:,foot]>=.95), ("swing",phase[:,foot]<=.05),
                            ("transition",(phase[:,foot]>.05)&(phase[:,foot]<.95))):
            phase_groups[name+"/"+label] = group(mask)
            if contact is not None:
                phase_contact_groups[name+"/"+label+"/contact"] = group(mask&contact[:,foot])
                phase_contact_groups[name+"/"+label+"/no_contact"] = group(mask&~contact[:,foot])
            if height is not None:
                for low, high, bin_name in height_bins:
                    phase_height_groups[name+"/"+label+"/"+bin_name] = group(mask&(height[:,foot]>=low)&(height[:,foot]<high))
    velocity = state["vel"].numpy()
    norm = np.linalg.norm(velocity, axis=-1)
    orthogonal = np.linalg.norm(velocity[:,1:], axis=-1)
    forward_dominates = abs(velocity[:,0]) >= orthogonal
    direction_groups = {"near_rest_3d_below_floor": group(norm<cfg.reward_speed_floor),
        "forward_dominated": group((norm>=cfg.reward_speed_floor)&forward_dominates&(velocity[:,0]>=0)),
        "backward_dominated": group((norm>=cfg.reward_speed_floor)&forward_dominates&(velocity[:,0]<0)),
        "lateral_or_vertical_dominated": group((norm>=cfg.reward_speed_floor)&~forward_dominates)}
    return {"verification": verified, "global": group(np.ones(shape[0]*shape[1],dtype=bool)),
        "phase_groups": phase_groups, "phase_height_groups": phase_height_groups, "phase_contact_groups": phase_contact_groups,
        "speed_direction_groups": direction_groups,
        "missing_optional_fields": (["reward/foot_height"] if height is None else []) + (["reward/foot_contact"] if contact is None else []),
        "height_definition": metadata["static_metadata"].get("foot_height_definition", metadata["static_metadata"].get("foot_height_semantics", "Unspecified; height bins are raw supplied coordinates and must not be called sole clearance")),
        "grouping_semantics": "Phase thresholds >=.95 stance, <=.05 swing; intermediate transition. Each foot conditions the shared two-foot/global reward; groups are not extra reward terms.",
        "speed_definition": "full3D norm with configured floor; direction dominance compares abs(vx) to norm(vy,vz)",
        "sampling": summary["sampling"], "not_a_walking_success_test": True}


def main() -> int:
    """CLI只进行CPU文件核验/分析，不初始化Isaac或CUDA。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "analyze"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_trace(args.directory) if args.mode == "verify" else analyze_trace(args.directory)
    output = args.output or args.directory / ("offline_"+args.mode+".json")
    _save_json(output, result)
    print(json.dumps({"output": str(output), "status": result.get("status", result.get("verification",{}).get("status"))}))
    return 0 if result.get("status", "passed") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
