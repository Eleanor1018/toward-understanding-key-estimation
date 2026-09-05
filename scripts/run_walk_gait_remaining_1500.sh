#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
resume_checkpoint="${1:-logs/g1_walk_estimation_from1000_plus2000_seed47/checkpoint_01500.pt}"
log_dir="${2:-logs/g1_walk_gait_from1500_to3000_seed48}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "Walking-gait DDP training requires at least two GPUs" >&2
    exit 2
fi
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Missing resume checkpoint: ${resume_checkpoint}" >&2
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
    --iterations 3000 \
    --rollout-steps 24 \
    --learning-rate 0.0001 \
    --policy-learning-rate 0.000002 \
    --minimum-policy-learning-rate 0.000001 \
    --target-kl 0.015 \
    --max-post-update-kl 0.03 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.00015 \
    --final-entropy-coef 0.00002 \
    --prediction-loss-coef 0.1 \
    --estimation-loss-coef 0.15 \
    --initial-action-std 0.09 \
    --resume-action-std 0.09 \
    --final-action-std 0.06 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile walk \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk_gait \
    --seed 48 \
    --save-interval 250 \
    --resume "${resume_checkpoint}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
evaluation_checkpoints=(
    "${resume_checkpoint}"
    "${log_dir}/checkpoint_01750.pt"
    "${log_dir}/checkpoint_02000.pt"
    "${log_dir}/checkpoint_02250.pt"
    "${log_dir}/checkpoint_02500.pt"
    "${log_dir}/checkpoint_02750.pt"
    "${log_dir}/checkpoint_03000.pt"
)
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --command-profile walk \
    --reward-profile p1_walk_gait \
    --checkpoints "${evaluation_checkpoints[@]}" \
    --output "${log_dir}/evaluation_3000.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
