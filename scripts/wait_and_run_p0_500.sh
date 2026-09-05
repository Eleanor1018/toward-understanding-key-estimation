#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
final_log_dir="${FINAL_LOG_DIR:-logs/g1_p0_squashed_4096_500_seed42}"
wait_log="${WAIT_LOG:-logs/g1_p0_gpu_wait.log}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"

mkdir -p "$(dirname "${wait_log}")"
printf '%s waiting for GPUs %s without preempting any process\n' \
    "$(date -Is)" "${gpu_list}" >> "${wait_log}"

idle_checks=0
while true; do
    busy=()
    for gpu_index in "${gpu_indices[@]}"; do
        if nvidia-smi -i "${gpu_index}" \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null \
            | grep -Eq '^[[:space:]]*[0-9]+'; then
            busy+=("${gpu_index}")
        fi
    done
    if (( ${#busy[@]} == 0 )); then
        idle_checks=$((idle_checks + 1))
        printf '%s idle confirmation %d/3\n' \
            "$(date -Is)" "${idle_checks}" >> "${wait_log}"
        if (( idle_checks >= 3 )); then
            break
        fi
    else
        idle_checks=0
        printf '%s still busy: %s\n' \
            "$(date -Is)" "${busy[*]}" >> "${wait_log}"
    fi
    sleep 30
done

printf '%s GPUs free; starting P0 preflight\n' "$(date -Is)" >> "${wait_log}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
export PYTHONUNBUFFERED=1

preflight_root="logs/g1_p0_preflight_$(date +%Y%m%dT%H%M%S)"
smoke_dir="${preflight_root}/single_1"
ddp_smoke_dir="${preflight_root}/ddp_512"
preflight_dir="${preflight_root}/ddp_4096_20"
mkdir -p "${preflight_root}"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python train.py \
    --headless \
    --device cuda:0 \
    --num-envs 1 \
    --iterations 2 \
    --rollout-steps 4 \
    --log-dir "${smoke_dir}" \
    2>&1 | tee "${smoke_dir}.log"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
world_size="${#gpu_indices[@]}"
torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 512 \
    --iterations 2 \
    --rollout-steps 4 \
    --log-dir "${ddp_smoke_dir}" \
    2>&1 | tee "${ddp_smoke_dir}.log"

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations 20 \
    --rollout-steps 24 \
    --log-dir "${preflight_dir}" \
    2>&1 | tee "${preflight_dir}.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${preflight_dir}" <<'PY'
import math
import re
import sys
from pathlib import Path

import torch

log_dir = Path(sys.argv[1])
checkpoint = torch.load(
    log_dir / "checkpoint_00020.pt",
    map_location="cpu",
    weights_only=False,
)
if checkpoint.get("action_distribution_type") != "tanh_diagonal_gaussian_v1":
    raise SystemExit("P0 preflight checkpoint has the wrong distribution type")
log_std = checkpoint["action_distribution"]["log_std"].float()
if not torch.isfinite(log_std).all():
    raise SystemExit("P0 preflight produced non-finite action std")
std = log_std.exp()
if std.min().item() < 0.05 - 1e-6 or std.max().item() > 0.8 + 1e-6:
    raise SystemExit(f"P0 preflight std escaped bounds: {std.min()}..{std.max()}")
text = Path(f"{log_dir}.log").read_text()
matches = re.findall(r"iteration=00020 .*?kl=([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit("P0 preflight did not report iteration 20 KL")
kl = float(matches[-1])
if not math.isfinite(kl) or kl >= 0.2:
    raise SystemExit(f"P0 preflight KL is unstable: {kl}")
print(
    "P0 preflight passed: "
    f"std={std.mean().item():.4f}/{std.max().item():.4f}, kl={kl:.6f}"
)
PY

printf '%s preflight passed; starting 500 iterations\n' \
    "$(date -Is)" >> "${wait_log}"
GPU_LIST="${gpu_list}" "${script_dir}/run_p0_500.sh" "${final_log_dir}"
printf '%s training and evaluation finished\n' "$(date -Is)" >> "${wait_log}"
