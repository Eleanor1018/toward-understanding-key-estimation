"""六路线独立监督器：不持有CUDA；只控制自己启动且出生时间匹配的PID。

部署JSON显式给出root/source/runtime_python/runtime_env/asset、source_manifest_sha256、
operator_sha256和gpus[variant]={index,uuid,render_index?}。默认每路从零10000轮；
短验证可显式覆盖target，所有验收按实际target记录，不能冒充完成10000轮。
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

VARIANTS = ("estnet", "key1", "key2", "fullest", "irrest", "implicit")
DEFAULT_ROOT = "/data/nora/g1-six-routes-isaac51-10000-20260907"
MAX_RUN = 36 * 3600
START_TIMEOUT = STALE_TIMEOUT = 1800


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


class JsonlTail:
    """增量读取完整行；写到一半的行保留到下次，最终验收拒绝残行。"""
    def __init__(self, path, keep=50):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.count = 0
        self.last = deque(maxlen=keep)
        self.inode = None

    def poll(self, callback=lambda row: None, *, final=False):
        if not self.path.exists():
            if final:
                raise ValueError(f"Missing JSONL: {self.path}")
            return 0
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.inode is not None and (identity != self.inode or stat.st_size < self.offset):
            raise ValueError("JSONL replaced or truncated")
        self.inode = identity
        added = 0
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            # 有界块读取，不在每次轮询重新加载整个数十万行事件文件。
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                self.offset += len(block)
                lines = (self.pending + block).split(b"\n")
                self.pending = lines.pop()
                if len(self.pending) > 16 * 1024 * 1024:
                    raise ValueError("Unbounded JSONL line")
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or not finite(row):
                        raise ValueError("Nonfinite or nonobject JSONL row")
                    callback(row)
                    self.count += 1
                    added += 1
                    self.last.append(row)
        if final and self.pending.strip():
            raise ValueError("Final JSONL has an unterminated row")
        return added


def metric_contract(row, iteration):
    if (row.get("iteration") != iteration or row.get("ppo_updates") != iteration
            or row.get("ppo/optimizer_steps") != 16
            or row.get("ppo/total_optimizer_steps") != iteration * 16
            or row.get("ppo/learning_rate") != 5e-4 or not finite(row)):
        raise ValueError(f"Incomplete/nonfinite/flexible-LR metric at iteration {iteration}")


class EventAudit:
    """一次O(n)最终扫描；按实际时序核每轮4×4 Adam及4个观测KL。"""
    def __init__(self, target):
        self.target = target
        self.iteration = 1
        self.epoch = 1
        self.minibatch = 1
        self.expect = "ppo_minibatch_update"
        self.minis = self.epochs = self.summaries = 0
        self.max_policy_kl = 0.0

    def consume(self, row):
        if (self.iteration > self.target or row.get("event") != self.expect
                or row.get("iteration") != self.iteration or row.get("learning_rate") != 5e-4
                or not finite(row)):
            raise ValueError(f"Optimization event order/protocol mismatch at {self.iteration}: {row.get('event')}")
        step = (self.epoch - 1) * 4 + self.minibatch
        if self.expect == "ppo_minibatch_update":
            if (row.get("epoch"), row.get("minibatch")) != (self.epoch, self.minibatch):
                raise ValueError("Missing/duplicated minibatch")
            self.minis += 1
            if self.minibatch == 4:
                self.expect = "ppo_epoch_kl"
            else:
                self.minibatch += 1
        else:
            step = self.epoch * 4
            kl = row.get("policy_kl", row.get("kl", {}))
            if row.get("kl_role") != "measurement_only" or kl.get("finite") is not True:
                raise ValueError("Policy KL must be finite observation-only data")
            if not isinstance(kl.get("total"), (int, float)):
                raise ValueError("Missing measured policy KL")
            self.max_policy_kl = max(self.max_policy_kl, kl["total"])
            if self.expect == "ppo_epoch_kl":
                if row.get("epoch") != self.epoch:
                    raise ValueError("Missing/duplicated epoch KL")
                self.epochs += 1
                if self.epoch == 4:
                    self.expect = "ppo_update_summary"
                else:
                    self.epoch += 1
                    self.minibatch = 1
                    self.expect = "ppo_minibatch_update"
            else:
                self.summaries += 1
                self.iteration += 1
                self.epoch = self.minibatch = 1
                self.expect = "ppo_minibatch_update"
        # summary 的iteration已推进，使用原始行作为全局计数基准。
        if (row.get("optimizer_steps") != step
                or row.get("total_optimizer_steps") != (row["iteration"] - 1) * 16 + step):
            raise ValueError("Event Adam count mismatch")

    def finish(self):
        if (self.iteration, self.minis, self.epochs, self.summaries) != (
                self.target + 1, self.target * 16, self.target * 4, self.target):
            raise ValueError("Incomplete final optimization event stream")
        return {"minibatch_events": self.minis, "epoch_kl_events": self.epochs,
                "summaries": self.summaries, "max_policy_kl": self.max_policy_kl,
                "policy_kl_is_a_success_gate": False}


def audit_logs(run, target):
    metrics = JsonlTail(Path(run) / "metrics.jsonl")
    def check(row):
        metric_contract(row, metrics.count + 1)
    metrics.poll(check, final=True)
    if metrics.count != target:
        raise ValueError("Final metric sequence incomplete")
    audit = EventAudit(target)
    JsonlTail(Path(run) / "ppo_events.jsonl", keep=0).poll(audit.consume, final=True)
    return {**audit.finish(), "metric_count": metrics.count, "last_50": list(metrics.last)}


def process_identity(pid):
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        fields = data[data.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "pgid": int(fields[2]), "starttime": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


class OwnedTree:
    """按已验证父子关系发现后代；从不按名称/显存/设备号终止进程。"""
    def __init__(self, child):
        self.child, self.records, self.actions = child, {}, []
        identity = process_identity(child.pid)
        if identity:
            self.records[child.pid] = identity
        elif child.poll() is None:
            raise RuntimeError("Cannot prove child birth identity")

    def current(self, pid):
        old, current = self.records.get(pid), process_identity(pid)
        return current if old and current and old["starttime"] == current["starttime"] else None

    def refresh(self):
        pending, seen = list(self.records), set()
        while pending:
            pid = pending.pop()
            if pid in seen or self.current(pid) is None:
                continue
            seen.add(pid)
            try:
                tasks = list(Path(f"/proc/{pid}/task").iterdir())
            except OSError:
                continue
            for task in tasks:
                try:
                    children = [int(x) for x in (task / "children").read_text().split()]
                except (OSError, ValueError):
                    continue
                for child in children:
                    identity = process_identity(child)
                    if identity and self.current(pid) and identity["ppid"] == pid:
                        saved = self.records.get(child)
                        if saved and saved["starttime"] != identity["starttime"]:
                            continue
                        self.records[child] = identity
                        pending.append(child)

    def live(self):
        return [current for pid in self.records if (current := self.current(pid))
                and current["state"] not in ("Z", "X")]

    def signal_all(self, sig):
        self.refresh()
        for entry in sorted(self.live(), key=lambda x: x["pid"] == self.child.pid):
            # 信号发送前再次核出生时间。只发单个已验证PID，不扩大到整个进程组。
            current = self.current(entry["pid"])
            if current is None or current["state"] in ("Z", "X"):
                continue
            try:
                os.kill(entry["pid"], sig)
                self.actions.append({"pid": entry["pid"], "starttime": entry["starttime"], "signal": int(sig)})
            except ProcessLookupError:
                pass

    def terminate(self, grace=45):
        for sig, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
            self.signal_all(sig)
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                self.refresh()
                self.child.poll()
                if not self.live() and self.child.returncode is not None:
                    return {"completed": True, "actions": self.actions, "survivors": []}
                time.sleep(.25)
        return {"completed": False, "actions": self.actions, "survivors": self.live()}


def gpu_snapshot(gpu):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu["index"]),
        "--query-gpu=uuid,pci.bus_id,utilization.gpu,memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()
    uuid, pci, util, free, total = [part.strip() for part in raw.split(",")]
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu["index"]),
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=15)
    contexts = []
    for line in output.splitlines():
        if line.strip():
            pid, memory = [part.strip() for part in line.split(",")]
            contexts.append({"pid": int(pid), "used_mib": int(memory) if memory.isdigit() else None})
    return {"at": now(), "index": gpu["index"], "uuid": uuid, "pci": pci,
            "utilization": int(util), "free_mib": int(free), "total_mib": int(total),
            "compute_contexts": contexts, "graphics_processes_enumerated": False,
            "exclusive_reservation": False}


class AdmissionDeclined(RuntimeError):
    """此异常仅发生在启动前，最终评估因此可安全记deferred。"""


def admit(snapshot, gpu, minimum_free=20 * 1024):
    if (snapshot["uuid"].lower() != gpu["uuid"].lower() or snapshot["index"] != gpu["index"]
            or snapshot["utilization"] != 0 or snapshot["free_mib"] < minimum_free):
        raise AdmissionDeclined("GPU admission declined: " + json.dumps(snapshot))
    # 用户允许util0的空闲上下文共用；保留其证据，不冒充独占也绝不清理它们。
    return snapshot


def fresh_admission(gpu, minimum_free):
    try:
        return admit(gpu_snapshot(gpu), gpu, minimum_free)
    except Exception as exc:
        raise AdmissionDeclined(str(exc)) from exc


def owned_gpu_release(owned, tries=10):
    last = []
    for attempt in range(tries):
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        last = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            pid_text, uuid, memory = [part.strip() for part in line.split(",")]
            pid = int(pid_text)
            if pid not in owned.records:
                continue
            current = process_identity(pid)
            if current and current["starttime"] != owned.records[pid]["starttime"]:
                continue
            last.append({"pid": pid, "uuid": uuid, "used_mib": memory, "current": current})
        if not last:
            return {"released": True, "owned_compute_contexts": [], "at": now(), "observations": attempt + 1}
        if attempt + 1 < tries:
            time.sleep(3)
    raise RuntimeError("Own or stale-driver contexts remain; never kill unverified PIDs: " + json.dumps(last))


def validate_deployment(d):
    for key in ("root", "source", "runtime_python", "runtime_env", "asset"):
        if not isinstance(d.get(key), str) or not Path(d[key]).is_absolute():
            raise ValueError(f"Deployment requires absolute {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", d.get("source_manifest_sha256", "")):
        raise ValueError("Missing pinned source manifest SHA256")
    if set(d.get("gpus", {})) != set(VARIANTS):
        raise ValueError("Deployment must explicitly map all six variants")
    indices, uuids = [], []
    for variant in VARIANTS:
        gpu = d["gpus"][variant]
        if (type(gpu.get("index")) is not int or gpu["index"] < 0
                or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu.get("uuid", ""))
                or type(gpu.get("render_index", gpu["index"])) is not int
                or gpu.get("render_index", gpu["index"]) < 0):
            raise ValueError("Invalid explicit GPU mapping")
        indices.append(gpu["index"])
        uuids.append(gpu["uuid"].lower())
    if len(set(indices)) != 6 or len(set(uuids)) != 6:
        raise ValueError("Six routes must not share the same training GPU")
    target = d.get("target", 10000)
    if type(target) is not int or not 1 <= target <= 10000:
        raise ValueError("target must be an explicit 1..10000 total")
    return d


def verify_files(d):
    source = Path(d["source"]).resolve()
    manifest = source / "source_manifest.json"
    if sha(manifest) != d["source_manifest_sha256"]:
        raise ValueError("Deployment source manifest SHA mismatch")
    entries = read(manifest)["files"]
    seen = set()
    for entry in entries:
        path = (source / entry["path"]).resolve()
        if not path.is_relative_to(source) or entry["path"] in seen:
            raise ValueError("Invalid/duplicate manifest path")
        seen.add(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError("Frozen source mismatch: " + entry["path"])
    required = {str(p.relative_to(source)).replace("\\", "/") for p in (source / "estnet").glob("*.py")}
    if not required.issubset(seen) or "run.py" not in seen:
        raise ValueError("Manifest omits executable source")
    operators = d.get("operator_sha256", {})
    if set(operators) != {"train_supervisor.py", "launch_all.py"}:
        raise ValueError("Deployment must pin both operator files")
    for name, digest in operators.items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError("Operator source hash mismatch: " + name)
    return {p.name: sha(p) for p in (source / "estnet").glob("*.py")}


def kit_active_rows(log):
    # 实际Kit输出的UUID前缀是证据；不能从render整数等于CUDA整数推断同卡。
    lines = Path(log).read_text(errors="replace").splitlines()
    active = []
    for index, line in enumerate(lines[:-2]):
        columns = line.split("|")
        if len(columns) >= 9 and columns[1].strip().isdigit() and columns[3].strip().startswith("Yes"):
            active.append({"index": int(columns[1]), "uuid_row": lines[index + 1], "pci_row": lines[index + 2]})
    return active


def kit_pci_matches(row, nvml_pci):
    """Kit现有表只给十六进制Bus-ID；新版若给完整PCI则严格比较bus/device/function。"""
    nvml = re.fullmatch(r"(?:[0-9a-f]+:)?([0-9a-f]+):([0-9a-f]+)\.([0-7])", nvml_pci.lower())
    columns = row.split("|")
    if nvml is None or len(columns) < 8:
        return False
    actual = columns[6].strip().lower()
    full = re.fullmatch(r"(?:[0-9a-f]+:)?([0-9a-f]+):([0-9a-f]+)\.([0-7])", actual)
    if full:
        return tuple(int(x, 16) for x in full.groups()) == tuple(int(x, 16) for x in nvml.groups())
    return bool(re.fullmatch(r"[0-9a-f]+", actual)) and int(actual, 16) == int(nvml.group(1), 16)


def validate_runtime(run, log, expected, source_hashes, gpu, admission):
    manifest = read(Path(run) / "manifest.json")
    if manifest.get("config") != normalized(expected) or manifest.get("source_sha256") != source_hashes:
        raise ValueError("Actual runner config/source differs from deployment")
    protocol = manifest.get("runtime_protocol", {})
    physics = manifest.get("resolved_physics", {}).get("physx", {})
    if (protocol.get("status") != "verified_cpu_preflight" or protocol.get("cuda_initialized") is not False
            or protocol.get("isaaclab", {}).get("commit") != "37ddf626871758333d6ed89cf64ad702aef127d0"
            or protocol.get("isaaclab", {}).get("tracked_source_apps_clean") is not True
            or physics.get("enable_stabilization") is not True
            or physics.get("solve_articulation_contact_last") is not False
            or physics.get("enable_external_forces_every_iteration") is not False):
        raise ValueError("Isaac51 runtime/explicit physics protocol readback missing")
    selected = manifest["runtime"]["selected_torch_device"]
    if (selected.get("uuid", "").removeprefix("GPU-").lower() != gpu["uuid"].removeprefix("GPU-").lower()
            or selected.get("logical_index") != gpu["index"]):
        raise ValueError("Actual Torch GPU UUID differs from deployment")
    rows = kit_active_rows(log)
    prefix = gpu["uuid"].removeprefix("GPU-").split("-")[0].lower()
    if not rows or any(row["index"] != gpu.get("render_index", gpu["index"])
            or prefix not in row["uuid_row"].lower() or not kit_pci_matches(row["pci_row"], admission["pci"]) for row in rows):
        raise ValueError("Kit Active GPU index/UUID-prefix/PCI readback mismatch")
    collision = read(Path(run) / "self_collision_readback.json")
    if collision.get("configured") is not True or not collision.get("attributes") or not all(
            row.get("enabled") is True for row in collision["attributes"]):
        raise ValueError("Live self-collision readback not true")
    return {"source_and_config_match": True, "torch": selected, "kit_active_rows": rows,
            "kit_uuid_evidence": "8-character UUID prefix plus printed PCI Bus-ID; full UUID verified by Torch",
            "self_collision": collision, "runtime": manifest["runtime"],
            "runtime_protocol": protocol, "resolved_physics": manifest["resolved_physics"]}


def runtime_env(d, job, label):
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE",
                "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE"):
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", PXR_WORK_THREAD_LIMIT="8", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMNI_KIT_ACCEPT_EULA="YES")
    for key, part in (("XDG_CACHE_HOME", "cache"), ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                      ("TMPDIR", "tmp"), ("CUDA_CACHE_PATH", "cuda"), ("__GL_SHADER_DISK_CACHE_PATH", "gl")):
        path = job / "runtime" / label / part
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    if d.get("vulkan_icd"):
        icd = Path(d["vulkan_icd"])
        if not icd.is_absolute() or not icd.is_file():
            raise ValueError("Explicit Vulkan ICD does not exist")
        env.update(VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd))
    return env


def completion_status(returncode, result, measurement, *, failure_file=False, survivors=False, released=False):
    if (failure_file or result.get("failure") or result.get("status") == "failed"
            or not measurement or not finite(measurement)
            or survivors or not released or returncode not in (0, -11)):
        raise ValueError("Incomplete runner result, Python failure, native failure or resource cleanup")
    if returncode == 0 and (result.get("cleanup_status") != "completed" or result.get("status") == "cleanup_pending"):
        raise ValueError("Exit zero does not prove pending runner cleanup completed")
    if returncode == -11 and result.get("cleanup_status") != "completed" and result.get("status") != "cleanup_pending":
        raise ValueError("Unrecognized native failure result")
    # -11仅容许已知native关闭异常；最终全局成功还要过checkpoint/events严格验收。
    # runner是否返回与监督器确认自有资源释放分开记录，绝不把pending改写completed。
    return "completed" if returncode == 0 else "completed_with_shutdown_warning"


def execute(d, variant, job, status, expected, source_hashes, *, train, checkpoint=None):
    gpu = d["gpus"][variant]
    label = "train-001" if train else "evaluate-final-001"
    run, log_path = job / label, job / (label + ".log")
    # 同卡评估也要再次准入；训练结束不代表GPU一直归本任务所有。
    admission = fresh_admission(gpu, 20 * 1024 if train else 6 * 1024)
    args = ["train", "--iterations", str(d.get("target", 10000))] if train else [
        "evaluate", "--checkpoint", str(checkpoint)]
    command = [d["runtime_python"], "-u", "-B", str(Path(d["source"]) / "run.py"), *args,
        "--variant", variant, "--num-envs", "4096" if train else "32", "--seed", "42" if train else "43",
        "--asset", d["asset"], "--device", f"cuda:{gpu['index']}", "--render-gpu", str(gpu.get("render_index", gpu["index"])),
        "--cpu-threads", "8", "--headless", "--run-dir", str(run)]
    phase = {"status": "starting", "started_at": now(), "command": command, "gpu_admission": admission,
             "run_dir": str(run)}
    status[label] = phase
    status["status"] = "training" if train else "evaluating"
    write(job / "state.json", status)
    env = runtime_env(d, job, label)
    owned = None
    try:
        with log_path.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=d["source"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            owned = OwnedTree(child)
            phase.update(status="running", child_pid=child.pid, owned_processes=list(owned.records.values()))
            write(job / "state.json", status)
            tail = JsonlTail(run / "metrics.jsonl")
            started = last_progress = time.monotonic()
            measured_at = None
            def check(row):
                metric_contract(row, tail.count + 1)
            while child.poll() is None:
                owned.refresh()
                current = time.monotonic()
                if train and tail.poll(check):
                    last_progress = current
                    if "runtime_validation" not in phase:
                        phase["runtime_validation"] = validate_runtime(run, log_path, expected, source_hashes, gpu, admission)
                    phase.update(iteration=tail.count, last_50=list(tail.last), last_progress_at=now(),
                                 owned_processes=list(owned.records.values()))
                    # 500只是文件产生事实，不在这里评估、暂停或抢另外的卡。
                    if tail.count >= 500 and (run / "model_00500.pt").is_file():
                        phase.setdefault("checkpoint_500_observed", {"path": str(run / "model_00500.pt"), "at": now()})
                    write(job / "state.json", status)
                if (run / "failure.txt").exists():
                    raise RuntimeError("Runner recorded a Python failure")
                if (run / "measurement.json").exists() and measured_at is None:
                    measured_at = current
                elapsed = current - started
                if train and ((tail.count == 0 and elapsed > START_TIMEOUT)
                        or (tail.count > 0 and current - last_progress > STALE_TIMEOUT) or elapsed > MAX_RUN):
                    raise TimeoutError("Training startup/stale/36-hour bound exceeded")
                if not train and elapsed > 1200:
                    raise TimeoutError("Final evaluation exceeded 1200 seconds")
                if measured_at is not None and current - measured_at > 120:
                    raise TimeoutError("Runner native cleanup stalled after saving measurement")
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        owned.refresh()
        if owned.live():
            phase["cleanup"] = owned.terminate()
        phase.update(native_exit_code=child.poll(), owned_processes=list(owned.records.values()),
                     survivors=owned.live())
        phase["gpu_release"] = owned_gpu_release(owned)
        result, measurement = read(run / "result.json"), read(run / "measurement.json")
        phase["runtime_validation"] = validate_runtime(run, log_path, expected, source_hashes, gpu, admission)
        phase["status"] = completion_status(child.returncode, result, measurement,
            failure_file=(run / "failure.txt").exists(), survivors=bool(owned.live()), released=phase["gpu_release"]["released"])
        phase.update(result=result, measurement=measurement, runner_cleanup_status=result.get("cleanup_status", "pending"),
                     finished_at=now())
        write(job / "state.json", status)
        return run, measurement
    except BaseException:
        if owned is not None:
            phase["cleanup"] = owned.terminate()
            phase["native_exit_code"] = owned.child.poll()
            phase["owned_processes"] = list(owned.records.values())
            try:
                phase["gpu_release"] = owned_gpu_release(owned)
            except Exception as exc:
                phase["gpu_release"] = {"released": False, "error": str(exc)}
        phase.update(status="failed", finished_at=now())
        raise


def validate_checkpoint(path, target, expected, asset):
    from estnet.resume import load_training_checkpoint
    payload, cfg = load_training_checkpoint(path, asset)  # CPU + weights_only + model/Adam/asset strict validation
    if (payload["iteration"] != target or normalized(cfg.to_dict()) != normalized(expected)
            or payload["optimizer"]["updates"] != target
            or payload["optimizer"]["total_optimizer_steps"] != target * 16):
        raise ValueError("Final checkpoint config/iteration/Adam count mismatch")
    return {"path": str(path), "sha256": sha(path), "iteration": target, "schema": cfg.schema,
            "total_optimizer_steps": target * 16, "strict_cpu_model_adam_asset_validation": True}


def validate_training(run, measurement, d, expected, asset):
    target = d.get("target", 10000)
    if (measurement.get("status") != "training_finished_requires_evaluation"
            or measurement.get("iterations") != target or measurement.get("start_iteration") != 0
            or measurement.get("updates_this_run") != target
            or measurement.get("total_optimizer_steps") != target * 16
            or measurement.get("variant") != expected["variant"]
            or measurement.get("learning_rate_schedule") != "fixed"
            or measurement.get("kl_controls_updates") is not False):
        raise ValueError("Training measurement does not prove fresh complete target")
    checkpoint = validate_checkpoint(Path(run) / f"model_{target:05d}.pt", target, expected, asset)
    return {"checkpoint": checkpoint, "logs": audit_logs(run, target)}


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Supervisor received signal {signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--attempt", default="attempt-001")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"attempt-\d{3}", args.attempt):
        raise ValueError("Attempt must have form attempt-001")
    d = validate_deployment(read(args.deployment))
    job = Path(d["root"]) / args.variant / args.attempt
    job.parent.mkdir(parents=True, exist_ok=True)
    job.mkdir()  # 原子拒绝重复；永不覆盖旧日志、结果或缓存。
    status = {"status": "preflight", "at": now(), "supervisor_pid": os.getpid(),
        "supervisor_identity": process_identity(os.getpid()), "variant": args.variant,
        "target": d.get("target", 10000), "initialization": "fresh_seed42_4096env",
        "gpu": d["gpus"][args.variant], "deployment": str(args.deployment.resolve()),
        "deployment_sha256": sha(args.deployment), "exclusive_reservation": False,
        "walking_verified": False, "max_runtime_seconds": MAX_RUN}
    write(job / "state.json", status)
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        hashes = verify_files(d)
        if Path(sys.executable).resolve() != Path(d["runtime_python"]).resolve():
            raise ValueError("Supervisor must run in the specified activated runtime")
        sys.path.insert(0, d["source"])
        from estnet.factory import config_for_variant
        from estnet.preflight import inspect_asset
        cfg = replace(config_for_variant(args.variant), num_envs=4096, seed=42)
        cfg.validate()
        if cfg.schema != f"g1-{args.variant}-ppo-clip-flat-isaac51-v1":
            raise ValueError("Not the explicitly separated Isaac 5.1 protocol")
        expected, asset = cfg.to_dict(), inspect_asset(d["asset"])
        if not asset["ready"]:
            raise ValueError("Four USD assets missing or hash mismatch")
        status.update(config=normalized(expected), source_sha256=hashes, asset=asset)
        write(job / "state.json", status)
        run, measured = execute(d, args.variant, job, status, expected, hashes, train=True)
        status["training_validation"] = validate_training(run, measured, d, expected, asset)
        status["training_completed"] = True
        write(job / "state.json", status)
        if not d.get("final_evaluation", True):
            status["status"] = "training_completed_evaluation_deferred"
            status["evaluation_deferred_reason"] = "Explicit deployment final_evaluation=false"
        else:
            # 这里只捕获准入失败；已启动的评估执行失败必须如实记failed。
            try:
                path = status["training_validation"]["checkpoint"]["path"]
                erun, em = execute(d, args.variant, job, status, replace(cfg, num_envs=32, seed=43).to_dict(),
                                   hashes, train=False, checkpoint=path)
                if (em.get("status") != "measured_first_episodes" or em.get("iteration") != d.get("target", 10000)
                        or em.get("num_envs") != 32 or not finite(em)
                        or read(erun / "manifest.json").get("checkpoint_sha256") != sha(path)):
                    raise ValueError("Final 32env mean first-episode evaluation/provenance mismatch")
                write(job / "evaluation_summary.json", em)
                warning = any(status[x]["status"] != "completed" for x in ("train-001", "evaluate-final-001"))
                status["status"] = "completed_with_shutdown_warning" if warning else "completed"
            except AdmissionDeclined as exc:
                status.update(status="training_completed_evaluation_deferred", evaluation_deferred_reason=str(exc))
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (job / "supervisor-failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        status["finished_at"] = now()
        write(job / "state.json", status)
    return int(status["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
