#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
    echo "Legacy paper DDP launcher disabled after the input-contract fix" >&2
    exit 2
fi

source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
checkpoint="${1:?usage: scripts/run_paper_ddp.sh CHECKPOINT [LOG_DIR]}"
log_dir="${2:-logs/g1_paper_4096_gpu1_2000_1khz_reset}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "DDP requires at least two GPUs" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations 2000 \
    --rollout-steps 24 \
    --resume "${checkpoint}" \
    --log-dir "${log_dir}" \
    2>&1 | tee -a "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --include-zero-baseline \
    --checkpoints \
        "${log_dir}/checkpoint_00050.pt" \
        "${log_dir}/checkpoint_00500.pt" \
        "${log_dir}/checkpoint_02000.pt" \
    --output "${log_dir}/evaluation_2000.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
