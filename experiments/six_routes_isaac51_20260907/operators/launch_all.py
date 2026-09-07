"""只派生六个独立监督器；不会直接创建CUDA上下文或停止其他任务。

用法: runtime_python launch_all.py --deployment deployment.json [--variant estnet]
部署target默认10000；短验证使用独立root及显式小target。准入后仍无独占预留。
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time

from train_supervisor import VARIANTS, admit, gpu_snapshot, now, read, sha, validate_deployment, verify_files, write


def command_for(d, deployment, variant, attempt):
    # shell文本固定；路径通过独立位置参数传入，不进行字符串拼接或eval。
    return ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"', "runtime-activation",
            d["runtime_env"], d["runtime_python"], "-u", "-B", str(Path(__file__).with_name("train_supervisor.py")),
            "--deployment", str(Path(deployment).resolve()), "--variant", variant, "--attempt", attempt]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, help="只派发这一条；不启动其余五条")
    parser.add_argument("--attempt", default="attempt-001")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"attempt-\d{3}", args.attempt):
        raise ValueError("Invalid attempt label")
    d = validate_deployment(read(args.deployment))
    verify_files(d)
    root = Path(d["root"])
    root.mkdir(parents=True, exist_ok=True)
    selected = [args.variant] if args.variant else list(VARIANTS)
    receipt_path = root / f"launch-{time.time_ns()}.json"
    receipt = {"at": now(), "deployment_sha256": sha(args.deployment), "requested": selected,
               "target": d.get("target", 10000), "routes": {}, "exclusive_reservation": False}
    write(receipt_path, receipt)
    for variant in selected:
        row = receipt["routes"][variant] = {"status": "checking", "at": now()}
        try:
            if (root / variant / args.attempt).exists():
                raise ValueError("Attempt already exists; never duplicate a recorded job")
            row["admission"] = admit(gpu_snapshot(d["gpus"][variant]), d["gpus"][variant])
            command = command_for(d, args.deployment, variant, args.attempt)
            # x模式同样防并发重复派发。后续单路失败不停止已派发的其它独立路线。
            with (root / f"operator-{variant}-{args.attempt}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
            row.update(status="dispatched_not_yet_validated", supervisor_pid=child.pid, command=command)
        except Exception as exc:
            row.update(status="not_launched", error=f"{type(exc).__name__}: {exc}")
        write(receipt_path, receipt)
    print(str(receipt_path), flush=True)
    return int(any(row["status"] == "not_launched" for row in receipt["routes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
