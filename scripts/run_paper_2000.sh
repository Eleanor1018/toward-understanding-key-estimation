#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_EXPERIMENT:-0}" != "1" ]]; then
    echo "Legacy paper launcher disabled after the input-contract fix" >&2
    exit 2
fi

source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_index="${GPU_INDEX:-1}"
log_dir="${1:-logs/g1_paper_4096_gpu1_2000}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export PYTHONUNBUFFERED=1

mkdir -p "${log_dir}"
python train.py \
    --headless \
    --device cuda:0 \
    --num-envs 4096 \
    --iterations 2000 \
    --rollout-steps 24 \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train.log"

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
