"""Isaac 5.1 的明确运行协议；纯模型/优化器测试不导入模拟器或检查本机GPU。

只有真实runner在CPU预检阶段调用verify_runtime。没有跳过版本校验的CLI或环境变量；
CPU集成测试必须显式替换这个边界，不能让真实运行沿用测试的宽松规则。
"""
from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata, util
from pathlib import Path

SIMULATION_PROTOCOL = "isaacsim-5.1.0-lab-2.3.2-stabilized-v1"
ISAACLAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
PHYSX_PROTOCOL = {
    "enable_stabilization": True,
    "solve_articulation_contact_last": False,
    "enable_external_forces_every_iteration": False,
}


def _check_versions(record):
    """核对实际包与实际导入的Torch，不能只按显卡名字猜wheel兼容性。"""
    if tuple(record["python_version"][:2]) != (3, 11):
        raise RuntimeError("Isaac51 protocol requires Python 3.11")
    for name, expected in (("torch", "2.7.0+cu128"), ("isaacsim", "5.1.0.0")):
        if record["packages"].get(name) != expected:
            raise RuntimeError(f"Isaac51 protocol requires {name}=={expected}")
    if record["torch_import_version"] != "2.7.0+cu128" or record["torch_cuda"] != "12.8":
        raise RuntimeError("Imported Torch must be 2.7.0+cu128 with CUDA 12.8")


def _lab_checkout():
    """定位实际import源，再校验固定git提交；不相信可能指向另一份仓库的环境变量。"""
    spec = util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed Isaac Lab source")
    origin = Path(spec.origin).resolve()

    def git(directory, *arguments):
        try:
            return subprocess.run(["git", "-C", str(directory), *arguments], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot verify the actual Isaac Lab git checkout") from exc

    checkout = Path(git(origin.parent, "rev-parse", "--show-toplevel")).resolve()
    expected_origin = checkout / "source/isaaclab/isaaclab/__init__.py"
    if origin != expected_origin.resolve():
        raise RuntimeError("Installed isaaclab does not resolve to the expected checkout source")
    commit = git(checkout, "rev-parse", "HEAD")
    if commit != ISAACLAB_COMMIT:
        raise RuntimeError(f"Isaac Lab must be pinned to {ISAACLAB_COMMIT}")
    # editable安装不应修改tracked实现；忽略构建缓存等untracked文件。
    dirty = git(checkout, "status", "--porcelain", "--untracked-files=no", "--",
                "source", "apps")
    if dirty:
        raise RuntimeError("Isaac Lab tracked source/apps differ from the pinned commit")
    return {"commit": commit, "checkout": str(checkout), "import_origin": str(origin),
            "tracked_source_apps_clean": True, "tag": "v2.3.2"}


def verify_runtime():
    """真实入口的CPU预检；无需import Isaac/Omni，也不初始化CUDA上下文。"""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Runtime verification must precede CUDA initialization")
    packages = {}
    for name in ("torch", "isaacsim", "isaaclab", "isaaclab_assets"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = {"simulation_protocol": SIMULATION_PROTOCOL,
              "python_version": list(platform.python_version_tuple()),
              "packages": packages, "torch_import_version": str(torch.__version__),
              "torch_cuda": torch.version.cuda}
    record["python_version"] = [int(value) for value in record["python_version"]]
    _check_versions(record)
    record["isaaclab"] = _lab_checkout()
    record["cuda_initialized"] = torch.cuda.is_initialized()
    if record["cuda_initialized"]:
        raise RuntimeError("Runtime verification unexpectedly initialized CUDA")
    record["status"] = "verified_cpu_preflight"
    return record


def resolved_physics_manifest(env_cfg):
    """记录实际传给环境的配置，区分配置证据与PhysX运行时getter回读。"""
    sim = getattr(env_cfg, "sim", None)
    scene = getattr(env_cfg, "scene", None)
    physx = getattr(sim, "physx", None)
    physics = physx.to_dict() if physx is not None and hasattr(physx, "to_dict") else {
        name: getattr(physx, name, None) for name in PHYSX_PROTOCOL}
    render = getattr(sim, "render", None)
    rendered = render.to_dict() if render is not None and hasattr(render, "to_dict") else None
    result = {"evidence": "resolved configuration before environment construction; not live solver readback",
              "simulation_protocol": SIMULATION_PROTOCOL,
              "sim": {name: getattr(sim, name, None) for name in
                      ("dt", "device", "render_interval", "use_fabric", "gravity")},
              "physx": physics, "render": rendered,
              "decimation": getattr(env_cfg, "decimation", None),
              "episode_length_s": getattr(env_cfg, "episode_length_s", None),
              "scene": {name: getattr(scene, name, None) for name in
                        ("num_envs", "env_spacing", "replicate_physics", "clone_in_fabric")}}
    json.dumps(result, allow_nan=False)
    return result
