#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
resume_checkpoint="${1:-logs/g1_stage1_4096_iter200_to_2000_seed46/checkpoint_01000.pt}"
final_log_dir="${2:-logs/g1_walk_estimation_from1000_plus2000_seed47}"
wait_log="${WAIT_LOG:-logs/g1_walk_estimation_gpu_wait.log}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"

if (( ${#gpu_indices[@]} < 2 )); then
    echo "Walking preflight requires at least two GPUs" >&2
    exit 2
fi
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Missing resume checkpoint: ${resume_checkpoint}" >&2
    exit 2
fi

mkdir -p "$(dirname "${wait_log}")"

wait_for_free_gpus() {
    local idle_checks=0
    while true; do
        local busy=()
        local gpu_index
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
                return
            fi
        else
            idle_checks=0
            printf '%s waiting without preemption; GPUs busy: %s\n' \
                "$(date -Is)" "${busy[*]}" >> "${wait_log}"
        fi
        sleep 20
    done
}

printf '%s waiting for GPUs %s without preempting any process\n' \
    "$(date -Is)" "${gpu_list}" >> "${wait_log}"
wait_for_free_gpus

source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
export PYTHONUNBUFFERED=1
preflight_root="logs/g1_walk_preflight_$(date +%Y%m%dT%H%M%S)"
single_dir="${preflight_root}/single_1"
ddp_dir="${preflight_root}/ddp_resume_128"
mkdir -p "${preflight_root}"

printf '%s GPUs free; starting one-environment walking smoke\n' \
    "$(date -Is)" >> "${wait_log}"
export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python train.py \
    --headless \
    --device cuda:0 \
    --num-envs 1 \
    --iterations 2 \
    --rollout-steps 4 \
    --learning-rate 0.0001 \
    --policy-learning-rate 0.00000625 \
    --target-kl 0.01 \
    --max-post-update-kl 0.02 \
    --entropy-coef 0.0002 \
    --final-entropy-coef 0.00002 \
    --prediction-loss-coef 0.1 \
    --estimation-loss-coef 0.15 \
    --initial-action-std 0.10 \
    --final-action-std 0.06 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile walk \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk \
    --seed 47 \
    --save-interval 2 \
    --log-dir "${single_dir}" \
    2>&1 | tee "${preflight_root}/single_1.log"

printf '%s single smoke passed; starting two-GPU resume smoke\n' \
    "$(date -Is)" >> "${wait_log}"
smoke_gpu_list="${gpu_indices[0]},${gpu_indices[1]}"
export CUDA_VISIBLE_DEVICES="${smoke_gpu_list}"
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --distributed \
    --headless \
    --num-envs 128 \
    --iterations 1001 \
    --rollout-steps 4 \
    --learning-rate 0.0001 \
    --policy-learning-rate 0.00000625 \
    --minimum-policy-learning-rate 0.000001 \
    --target-kl 0.01 \
    --max-post-update-kl 0.02 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0002 \
    --final-entropy-coef 0.00002 \
    --prediction-loss-coef 0.1 \
    --estimation-loss-coef 0.15 \
    --initial-action-std 0.10 \
    --resume-action-std 0.10 \
    --final-action-std 0.06 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile walk \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk \
    --seed 47 \
    --save-interval 1 \
    --resume "${resume_checkpoint}" \
    --log-dir "${ddp_dir}" \
    2>&1 | tee "${preflight_root}/ddp_resume_128.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 64 \
    --steps 100 \
    --command-profile walk \
    --reward-profile p1_walk \
    --checkpoints "${ddp_dir}/checkpoint_01001.pt" \
    --output "${preflight_root}/evaluation.json" \
    2>&1 | tee "${preflight_root}/evaluation.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${preflight_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "evaluation.json").read_text())
result = report["results"][0]
required = (
    result["mean_reward_per_step"],
    result["planar_velocity_vector_rmse"],
    result["explicit_velocity_vector_rmse"],
    result["mean_absolute_clipped_action"],
)
if not all(value is not None and math.isfinite(value) for value in required):
    raise SystemExit("Walking preflight produced non-finite metrics")
for name in ("forward", "backward"):
    tracking = result["command_bin_tracking"][name]
    if tracking["transitions"] <= 0:
        raise SystemExit(f"Walking preflight omitted {name} tracking samples")
print("WALK_PREFLIGHT_PASS", result)
PY

printf '%s all walking preflights passed; confirming all GPUs again\n' \
    "$(date -Is)" >> "${wait_log}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
wait_for_free_gpus

printf '%s GPUs free; launching 2000-iteration walking phase\n' \
    "$(date -Is)" >> "${wait_log}"
GPU_LIST="${gpu_list}" "${script_dir}/run_walk_estimation_2000.sh" \
    "${resume_checkpoint}" "${final_log_dir}"
printf '%s walking training and evaluation finished\n' \
    "$(date -Is)" >> "${wait_log}"
