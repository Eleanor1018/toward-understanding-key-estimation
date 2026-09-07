"""CPU operator tests; never import Isaac or initialize CUDA, never start remote jobs."""
import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_supervisor as ops
import launch_all as launch


def jsonl(path, rows, final_newline=True):
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if final_newline else ""), encoding="utf-8")


def event_rows(target):
    for i in range(1, target + 1):
        for epoch in range(1, 5):
            for mini in range(1, 5):
                step = (epoch - 1) * 4 + mini
                yield dict(event="ppo_minibatch_update", iteration=i, epoch=epoch, minibatch=mini,
                    learning_rate=.0005, optimizer_steps=step, total_optimizer_steps=(i - 1) * 16 + step)
            yield dict(event="ppo_epoch_kl", iteration=i, epoch=epoch, learning_rate=.0005,
                optimizer_steps=epoch * 4, total_optimizer_steps=(i - 1) * 16 + epoch * 4,
                kl={"finite": True, "total": .3}, kl_role="measurement_only")
        yield dict(event="ppo_update_summary", iteration=i, learning_rate=.0005,
            optimizer_steps=16, total_optimizer_steps=i * 16,
            kl={"finite": True, "total": .3}, kl_role="measurement_only")


def metric(i):
    return {"iteration": i, "ppo_updates": i, "ppo/optimizer_steps": 16,
            "ppo/total_optimizer_steps": i * 16, "ppo/learning_rate": .0005, "seconds": 2.}


def test_incremental_reader_holds_partial_and_keeps_only50(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tail = ops.JsonlTail(path)
    assert tail.poll() == 0
    path.write_bytes(b'{"iteration":1}\n{"iteration":')
    assert tail.poll() == 1
    assert tail.poll() == 0
    assert tail.count == 1
    with path.open("ab") as stream:
        stream.write(b'2}\n')
        for i in range(3, 102):
            stream.write(json.dumps({"iteration": i}).encode() + b"\n")
    assert tail.poll(final=True) == 100
    assert len(tail.last) == 50
    assert tail.last[0]["iteration"] == 52 and tail.last[-1]["iteration"] == 101


@pytest.mark.parametrize("bad", [b'{"x":2}', b'{"x":NaN}\n', b'{malformed}\n'])
def test_final_reader_rejects_partial_nan_and_invalid(tmp_path, bad):
    path = tmp_path / "x.jsonl"
    path.write_bytes(bad)
    with pytest.raises(ValueError):
        ops.JsonlTail(path).poll(final=True)


def test_reader_rejects_truncated_file(tmp_path):
    path = tmp_path / "x.jsonl"
    jsonl(path, [metric(1), metric(2)])
    tail = ops.JsonlTail(path)
    tail.poll()
    jsonl(path, [metric(1)])
    with pytest.raises(ValueError, match="truncated"):
        tail.poll()


def test_full10000_event_contract_linear_stream():
    audit = ops.EventAudit(10000)
    for event in event_rows(10000):
        audit.consume(event)
    report = audit.finish()
    assert report["minibatch_events"] == 160000
    assert report["epoch_kl_events"] == 40000
    assert report["summaries"] == 10000
    assert report["max_policy_kl"] == .3  # KL超过.02不应成为训练失败门槛。


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "wrong_total", "lr", "kl_epoch", "kl_nonfinite"])
def test_event_audit_rejects_counts_or_order_even_if_total_similar(corruption):
    rows = list(event_rows(2))
    if corruption == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif corruption == "missing":
        rows.pop(0)
    elif corruption == "wrong_total":
        rows[0]["total_optimizer_steps"] = 2
    elif corruption == "lr":
        rows[0]["learning_rate"] = .0001
    elif corruption == "kl_epoch":
        rows[4]["epoch"] = 2
    else:
        rows[4]["kl"]["total"] = float("nan")
    with pytest.raises(ValueError):
        audit = ops.EventAudit(2)
        for row in rows:
            audit.consume(row)
        audit.finish()


def test_final_audit_requires_complete_metrics_and_events(tmp_path):
    jsonl(tmp_path / "metrics.jsonl", [metric(1), metric(2)])
    jsonl(tmp_path / "ppo_events.jsonl", list(event_rows(2)))
    assert ops.audit_logs(tmp_path, 2)["metric_count"] == 2
    rows = [metric(1), metric(2)]
    rows[-1]["ppo/total_optimizer_steps"] = 31
    jsonl(tmp_path / "metrics.jsonl", rows)
    with pytest.raises(ValueError):
        ops.audit_logs(tmp_path, 2)


@pytest.mark.parametrize("rc,pending,expected", [(0, False, "completed"), (-11, False, "completed_with_shutdown_warning"),
    (-11, True, "completed_with_shutdown_warning"), (0, True, None), (-9, False, None), (-15, False, None), (2, False, None)])
def test_native_shutdown_contract(rc, pending, expected):
    result = {"status": "cleanup_pending"} if pending else {"status": "training_finished_requires_evaluation", "cleanup_status": "completed"}
    if expected:
        assert ops.completion_status(rc, result, {"iterations": 10000}, released=True) == expected
        assert ("cleanup_status" not in result) == pending
    else:
        with pytest.raises(ValueError):
            ops.completion_status(rc, result, {"iterations": 10000}, released=True)


@pytest.mark.parametrize("bad", [{"failure_file": True}, {"survivors": True}, {"released": False}])
def test_native_minus11_never_hides_python_failure_or_unreleased_gpu(bad):
    with pytest.raises(ValueError):
        ops.completion_status(-11, {"status": "cleanup_pending"}, {"iterations": 10000}, **{"released": True, **bad})


def test_identity_reuse_never_signals_unknown_pid(monkeypatch):
    identities = {91: {"pid": 91, "ppid": 1, "pgid": 91, "state": "R", "starttime": 100}}
    monkeypatch.setattr(ops, "process_identity", lambda pid: identities.get(pid))
    child = SimpleNamespace(pid=91, poll=lambda: None)
    tree = ops.OwnedTree(child)
    sent = []
    monkeypatch.setattr(tree, "refresh", lambda: None)
    monkeypatch.setattr(ops.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    identities[91] = {**identities[91], "starttime": 101}
    tree.signal_all(signal.SIGTERM)
    assert sent == [] and tree.live() == []
    identities[91]["starttime"] = 100
    tree.signal_all(signal.SIGTERM)
    assert sent == [(91, signal.SIGTERM)]


def gpu(index=0):
    return {"index": index, "uuid": f"GPU-{index:08x}-1234-1234-1234-123456789abc", "render_index": index}


def snapshot(g, **overrides):
    return {"index": g["index"], "uuid": g["uuid"], "utilization": 0, "free_mib": 25000,
            "compute_contexts": [{"pid": 999, "used_mib": 500}], **overrides}


def test_util0_idle_context_allowed_but_capacity_busy_uuid_mismatch_declined():
    g = gpu()
    assert ops.admit(snapshot(g), g)["compute_contexts"]
    for change in ({"free_mib": 20479}, {"utilization": 1}, {"uuid": gpu(1)["uuid"]}):
        with pytest.raises(ops.AdmissionDeclined):
            ops.admit(snapshot(g, **change), g)


def test_kit_busid_accepts_real_format_and_detects_other_physical_gpu():
    row = "|     |                                  |        |     |            | a1        |            |"
    assert ops.kit_pci_matches(row, "00000000:A1:00.0")
    assert not ops.kit_pci_matches(row, "00000000:41:00.0")
    assert ops.kit_pci_matches(row.replace("a1        ", "0000:a1:00.0"), "00000000:A1:00.0")
    assert not ops.kit_pci_matches("unrecognized table", "00000000:A1:00.0")


def test_runtime_env_isolates_job_paths_and_does_not_mask_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LD_PRELOAD", "old_override.so")
    env = ops.runtime_env({}, tmp_path / "estnet/attempt-001", "train-001")
    assert "CUDA_VISIBLE_DEVICES" not in env and "LD_PRELOAD" not in env
    for key in ("XDG_CACHE_HOME", "TMPDIR", "CUDA_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH"):
        assert Path(env[key]).is_dir()
        assert Path(env[key]).is_relative_to(tmp_path / "estnet/attempt-001/runtime/train-001")


def deployment(tmp_path):
    return dict(root=str(tmp_path / "root"), source=str(tmp_path / "source"),
        runtime_python=str(tmp_path / "runtime/python"), runtime_env=str(tmp_path / "runtime/runtime_env.sh"),
        asset=str(tmp_path / "assets/robot.usd"), source_manifest_sha256="a" * 64,
        gpus={v: gpu(i) for i, v in enumerate(ops.VARIANTS)})


def test_deployment_refuses_invented_or_duplicate_mapping(tmp_path):
    d = deployment(tmp_path)
    assert ops.validate_deployment(d) is d
    d["gpus"]["key1"] = d["gpus"]["estnet"]
    with pytest.raises(ValueError):
        ops.validate_deployment(d)


def test_launch_single_variant_no_other_job_and_source_safe_argv(tmp_path, monkeypatch):
    d = deployment(tmp_path)
    d["runtime_env"] = str(tmp_path / 'spaces $literal/runtime env.sh')
    path = tmp_path / "deployment.json"
    ops.write(path, d)
    monkeypatch.setattr(launch, "verify_files", lambda _: {})
    monkeypatch.setattr(launch, "gpu_snapshot", lambda g: snapshot(g))
    commands = []
    monkeypatch.setattr(launch.subprocess, "Popen", lambda argv, **kwargs: commands.append((argv, kwargs)) or SimpleNamespace(pid=345))
    assert launch.main(["--deployment", str(path), "--variant", "key2"]) == 0
    assert len(commands) == 1
    argv, kwargs = commands[0]
    assert d["runtime_env"] in argv and d["runtime_env"] not in argv[2]
    assert argv[-3:] == ["key2", "--attempt", "attempt-001"]
    assert kwargs["start_new_session"] is True
    assert not (Path(d["root"]) / "estnet").exists()
    # 同一label日志已经存在，第二次绝不再派生。
    assert launch.main(["--deployment", str(path), "--variant", "key2"]) == 1
    assert len(commands) == 1


@pytest.mark.parametrize("variant", ops.VARIANTS)
def test_real_cpu_checkpoint_strict_model_adam_schema_and_counts(tmp_path, variant):
    import torch
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1]  # 发布包根；不依赖外层work目录名。
    sys.path.insert(0, str(source))
    factory = importlib.import_module("estnet.factory")
    run = importlib.import_module("estnet.run")
    cfg = replace(factory.config_for_variant(variant), num_envs=4,
                  encoder_hidden=(12, 8), actor_hidden=(12, 8), critic_hidden=(12, 8), decoder_hidden=(8, 12), kl_chunk_size=8)
    assert cfg.schema == f"g1-{variant}-ppo-clip-flat-isaac51-v1"
    model, ppo = factory.build_model(cfg), None
    ppo = factory.build_ppo(model, cfg)
    n = 16
    history, obs = torch.randn(n, 50, 42) * .1, torch.randn(n, 42) * .1
    command, critic = torch.randn(n, 7) * .1, torch.randn(n, cfg.critic_dim) * .1
    with torch.no_grad():
        dist = model.distribution(history, obs, command)
        actions, value = dist.sample(), model.value(critic)
    batch = dict(history=history, obs=obs, command=command, critic=critic, action=actions,
                 old_mean=dist.mean.clone(), old_std=dist.stddev.clone(), old_log_prob=dist.log_prob(actions).sum(-1),
                 old_value=value, returns=value + .1, advantages=torch.linspace(-1., 1., n))
    for name, width in cfg.supervision_dims.items():
        batch[name] = torch.randn(n, width) * .1
    events = []
    ppo.diagnostic_callback = events.append
    m = ppo.update(batch)
    assert ppo.total_optimizer_steps == 16
    asset = {"ready": True, "files": [{"sha256": str(i) * 64} for i in range(1, 5)]}
    path = tmp_path / "model_00001.pt"
    run.save_checkpoint(path, model, ppo, cfg, 1, asset)
    receipt = ops.validate_checkpoint(path, 1, cfg.to_dict(), asset)
    assert receipt["total_optimizer_steps"] == 16
    jsonl(tmp_path / "metrics.jsonl", [{**{"ppo/" + k: v for k, v in m.items()}, "iteration": 1, "ppo_updates": 1}])
    jsonl(tmp_path / "ppo_events.jsonl", events)
    measurement = dict(status="training_finished_requires_evaluation", iterations=1, start_iteration=0,
        updates_this_run=1, total_optimizer_steps=16, variant=variant, learning_rate_schedule="fixed", kl_controls_updates=False)
    assert ops.validate_training(tmp_path, measurement, {"target": 1}, cfg.to_dict(), asset)["logs"]["minibatch_events"] == 16
    original = torch.load(path, weights_only=True, map_location="cpu")
    for corruption in ("count", "missing_adam", "schema", "nan_model", "asset"):
        bad = copy.deepcopy(original)
        if corruption == "count":
            bad["optimizer"]["total_optimizer_steps"] = 15
        elif corruption == "missing_adam":
            bad["optimizer"]["optimizer"]["state"].pop(next(iter(bad["optimizer"]["optimizer"]["state"])))
        elif corruption == "schema":
            bad["schema"] = "g1-estnet-flat-v1"
        elif corruption == "nan_model":
            next(iter(bad["model"].values())).fill_(float("nan"))
        else:
            bad["asset"]["files"][0]["sha256"] = "f" * 64
        torch.save(bad, path)
        with pytest.raises((ValueError, RuntimeError)):
            ops.validate_checkpoint(path, 1, cfg.to_dict(), asset)
    assert not torch.cuda.is_initialized()
