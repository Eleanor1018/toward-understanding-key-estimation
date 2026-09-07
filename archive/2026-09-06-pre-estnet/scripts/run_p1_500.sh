#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
    echo "Legacy P1 launcher disabled; use scripts/run_p1_fixed_stage1.sh" >&2
    exit 2
fi

source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
root_dir="${1:-logs/g1_p1_curriculum_4096_500_seed43}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "P1 DDP training requires at least two GPUs" >&2
    exit 2
fi
if compgen -G "${root_dir}/stage*/checkpoint_*.pt" >/dev/null; then
    echo "Refusing to overwrite checkpoints in ${root_dir}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${root_dir}"

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
            if (( idle_checks >= 2 )); then
                return
            fi
        else
            idle_checks=0
            printf '%s waiting between stages; GPUs busy: %s\n' \
                "$(date -Is)" "${busy[*]}" \
                >> "${root_dir}/stage_transition_wait.log"
        fi
        sleep 30
    done
}

run_stage() {
    local stage_dir="$1"
    local end_iteration="$2"
    local command_scale="$3"
    local randomization_scale="$4"
    local entropy_start="$5"
    local entropy_end="$6"
    local resume_checkpoint="${7:-}"
    local resume_args=()
    if [[ -n "${resume_checkpoint}" ]]; then
        resume_args=(--resume "${resume_checkpoint}")
    fi

    mkdir -p "${stage_dir}"
    torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
        --distributed \
        --headless \
        --num-envs 4096 \
        --iterations "${end_iteration}" \
        --rollout-steps 24 \
        --learning-rate 0.0002 \
        --target-kl 0.02 \
        --value-loss-coef 0.5 \
        --entropy-coef "${entropy_start}" \
        --final-entropy-coef "${entropy_end}" \
        --initial-action-std 0.8 \
        --min-action-std 0.05 \
        --max-action-std 0.8 \
        --command-scale "${command_scale}" \
        --domain-randomization-scale "${randomization_scale}" \
        --reward-profile p1_dense \
        --seed 43 \
        --save-interval 50 \
        --log-dir "${stage_dir}" \
        "${resume_args[@]}" \
        2>&1 | tee "${stage_dir}/train_ddp_${world_size}gpu.log"
}

stage1="${root_dir}/stage1_clean"
stage2="${root_dir}/stage2_mild_randomization"
stage3="${root_dir}/stage3_full_randomization"

# Learn command-conditioned balance and motion before adding disturbances.
run_stage "${stage1}" 150 0.40 0.0 0.0005 0.0002

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 512 \
    --steps 500 \
    --command-scale 0.40 \
    --reward-profile p1_dense \
    --include-zero-baseline \
    --checkpoints "${stage1}/checkpoint_00150.pt" \
    --output "${root_dir}/stage1_evaluation.json" \
    2>&1 | tee "${root_dir}/stage1_evaluation.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${root_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = json.loads((root / "stage1_evaluation.json").read_text())["results"]
zero, learned = results
metrics = (
    learned["planar_velocity_vector_rmse"],
    learned["yaw_rate_rmse"],
    learned["raw_action_saturation_fraction"],
)
if not all(math.isfinite(value) for value in metrics):
    raise SystemExit("P1 stage 1 produced non-finite evaluation metrics")
if learned["raw_action_saturation_fraction"] >= 0.10:
    raise SystemExit("P1 stage 1 action saturation exceeded 10%")
if learned["terminations"] > 1.5 * zero["terminations"]:
    raise SystemExit("P1 stage 1 falls substantially more often than zero action")
if (
    learned["planar_velocity_vector_rmse"]
    + learned["yaw_rate_rmse"]
    >= 1.10
    * (
        zero["planar_velocity_vector_rmse"]
        + zero["yaw_rate_rmse"]
    )
):
    raise SystemExit("P1 stage 1 did not improve aggregate command tracking")
print("P1 stage 1 gate passed", learned)
PY

export CUDA_VISIBLE_DEVICES="${gpu_list}"
wait_for_free_gpus
run_stage \
    "${stage2}" 300 0.70 0.35 0.0002 0.00005 \
    "${stage1}/checkpoint_00150.pt"
wait_for_free_gpus
run_stage \
    "${stage3}" 500 1.00 1.00 0.00005 0.0 \
    "${stage2}/checkpoint_00300.pt"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --command-scale 1.0 \
    --reward-profile p1_dense \
    --include-zero-baseline \
    --checkpoints \
        "${stage1}/checkpoint_00050.pt" \
        "${stage2}/checkpoint_00250.pt" \
        "${stage3}/checkpoint_00500.pt" \
    --output "${root_dir}/evaluation_500.json" \
    2>&1 | tee "${root_dir}/evaluation.log"
