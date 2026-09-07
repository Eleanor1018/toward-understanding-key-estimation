#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
    echo "Legacy P0 launcher disabled; use scripts/run_p1_fixed_stage1.sh" >&2
    exit 2
fi

source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
log_dir="${1:-logs/g1_p0_squashed_4096_500_seed42}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "P0 DDP training requires at least two GPUs" >&2
    exit 2
fi
if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
    echo "Refusing to overwrite checkpoints in ${log_dir}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations 500 \
    --rollout-steps 24 \
    --learning-rate 0.0002 \
    --target-kl 0.02 \
    --entropy-coef 0.001 \
    --final-entropy-coef 0.0 \
    --initial-action-std 0.8 \
    --min-action-std 0.05 \
    --max-action-std 0.8 \
    --seed 42 \
    --save-interval 50 \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --include-zero-baseline \
    --checkpoints \
        "${log_dir}/checkpoint_00050.pt" \
        "${log_dir}/checkpoint_00250.pt" \
        "${log_dir}/checkpoint_00500.pt" \
    --output "${log_dir}/evaluation_500.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
