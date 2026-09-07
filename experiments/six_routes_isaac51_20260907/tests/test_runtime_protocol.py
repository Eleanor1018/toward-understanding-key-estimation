"""真实运行时边界、协议拒载和物理默认显式固定；不导入Isaac或初始化CUDA。"""
import ast
import copy
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from estnet import runtime_protocol as protocol
from estnet.config import Config
from estnet.factory import VARIANTS, build_model, build_ppo, config_for_variant
from estnet.preflight import ASSET_HASHES
from estnet.resume import load_training_checkpoint
from estnet.run import load_evaluation_checkpoint, save_checkpoint


def good_versions():
    return {"python_version": [3, 11, 16],
            "packages": {"torch": "2.7.0+cu128", "isaacsim": "5.1.0.0"},
            "torch_import_version": "2.7.0+cu128", "torch_cuda": "12.8"}


@pytest.mark.parametrize("field,value", [
    ("python_version", [3, 10, 16]), ("python_version", [3, 12, 0]),
    ("torch_import_version", "2.7.0+cpu"), ("torch_cuda", "12.1"),
    ("torch", "2.8.0+cu128"), ("isaacsim", "4.5.0.0"),
])
def test_wrong_real_runtime_versions_are_rejected(field, value):
    record = good_versions()
    protocol._check_versions(record)
    target = record["packages"] if field in ("torch", "isaacsim") else record
    target[field] = value
    with pytest.raises(RuntimeError):
        protocol._check_versions(record)


def test_runtime_preflight_reports_actual_import_checkout_without_cuda(monkeypatch, tmp_path):
    """模拟包元数据/文件系统边界，但执行真实检查器，确保导入源决定checkout。"""
    record = good_versions()
    monkeypatch.setattr(protocol.platform, "python_version_tuple", lambda: ("3", "11", "16"))
    monkeypatch.setattr(protocol.metadata, "version", lambda name: record["packages"].get(name, "record-only"))
    monkeypatch.setattr(torch, "__version__", "2.7.0+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CPU preflight initialized CUDA"))
    checkout = (tmp_path / "actual-installed-lab").resolve()
    origin = checkout / "source/isaaclab/isaaclab/__init__.py"
    monkeypatch.setattr(protocol.util, "find_spec", lambda name: SimpleNamespace(origin=str(origin)))
    monkeypatch.setenv("ISAACLAB_PATH", str(tmp_path / "unrelated-checkout"))
    replies = {"commit": protocol.ISAACLAB_COMMIT, "dirty": ""}
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 10 and kwargs["check"] is True
        if "--show-toplevel" in command:
            assert Path(command[2]) == origin.parent
            value = str(checkout)
        elif "HEAD" in command:
            value = replies["commit"]
        else:
            assert command[-2:] == ["source", "apps"]
            value = replies["dirty"]
        return subprocess.CompletedProcess(command, 0, value + "\n", "")

    monkeypatch.setattr(protocol.subprocess, "run", git)
    result = protocol.verify_runtime()
    assert result["status"] == "verified_cpu_preflight"
    assert result["isaaclab"]["commit"] == protocol.ISAACLAB_COMMIT
    assert result["isaaclab"]["checkout"] == str(checkout)
    assert result["cuda_initialized"] is False
    assert len(calls) == 3
    replies["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="pinned"):
        protocol.verify_runtime()
    replies.update(commit=protocol.ISAACLAB_COMMIT, dirty=" M source/isaaclab/isaaclab/sim/simulation_cfg.py")
    with pytest.raises(RuntimeError, match="differ"):
        protocol.verify_runtime()


@pytest.mark.parametrize("variant", VARIANTS)
def test_old_runtime_or_missing_protocol_cannot_load_same_weights(tmp_path, variant):
    cfg = replace(config_for_variant(variant), num_envs=2, encoder_hidden=(4,),
                  actor_hidden=(4,), critic_hidden=(4,), decoder_hidden=(4,))
    assert cfg.schema == f"g1-{variant}-ppo-clip-flat-isaac51-v1"
    assert cfg.simulation_protocol == protocol.SIMULATION_PROTOCOL
    asset = {"ready": True, "files": [{"sha256": value} for value in ASSET_HASHES.values()]}
    model = build_model(cfg)
    path = tmp_path / "initial.pt"
    save_checkpoint(path, model, build_ppo(model, cfg), cfg, 0, asset)
    valid = torch.load(path, map_location="cpu", weights_only=True)
    for loader in (load_evaluation_checkpoint, load_training_checkpoint):
        loader(path, asset)  # 纯CPU检查点预检允许本地CPU Torch，不检查真实仿真环境。
    for change in ("missing", "different", "old_schema"):
        payload = copy.deepcopy(valid)
        if change == "missing":
            del payload["config"]["simulation_protocol"]
        elif change == "different":
            payload["config"]["simulation_protocol"] = "isaacsim-4.5.0-lab-2.0.2-v1"
        else:
            payload["schema"] = payload["config"]["schema"] = f"g1-{variant}-ppo-clip-flat-v1"
        torch.save(payload, path)
        for loader in (load_evaluation_checkpoint, load_training_checkpoint):
            with pytest.raises(ValueError):
                loader(path, asset)


def test_actual_build_cfg_overrides_new_physx_defaults_and_records_them():
    """执行生产build_env_cfg AST；以相反初值证明三个开关都被显式写入。"""
    from estnet import robot
    source = Path(__file__).resolve().parents[1] / "estnet/environment.py"
    function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_env_cfg")

    class Record(SimpleNamespace):
        def to_dict(self):
            return vars(self).copy()

    class Articulation(Record):
        InitialStateCfg = Record

    cfg_stub = Record(sim=Record(physx=Record(enable_stabilization=False,
                      solve_articulation_contact_last=True, enable_external_forces_every_iteration=True)),
                      scene=Record(env_spacing=2.5, replicate_physics=True))
    namespace = {"EnvironmentCfg": lambda: cfg_stub, "ArticulationCfg": Articulation,
                 "ImplicitActuatorCfg": Record,
                 "sim_utils": SimpleNamespace(**{name: Record for name in
                     ("RigidBodyMaterialCfg", "UsdFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg")})}
    for short, name in (("JOINT_NAMES", "JOINT_NAMES29"), ("DEFAULT_JOINT_POS", "DEFAULT_JOINT_POS29"),
                        ("STIFFNESS", "STIFFNESS29"), ("DAMPING", "DAMPING29"),
                        ("EFFORT_LIMIT", "EFFORT_LIMIT29"), ("VELOCITY_LIMIT", "VELOCITY_LIMIT29"),
                        ("ARMATURE", "ARMATURE29")):
        namespace[short] = getattr(robot, name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - 仅执行本项目已审阅的生产函数AST
    cfg = Config()
    built = namespace["build_env_cfg"](cfg, "g1.usd", "cuda:5")
    assert vars(built.sim.physx) == protocol.PHYSX_PROTOCOL
    assert built.robot.spawn.articulation_props.enabled_self_collisions is True
    assert built.sim.dt == .001 and built.decimation == 10
    receipt = protocol.resolved_physics_manifest(built)
    assert receipt["physx"] == protocol.PHYSX_PROTOCOL
    assert receipt["sim"]["device"] == "cuda:5"
    assert receipt["simulation_protocol"] == cfg.simulation_protocol
