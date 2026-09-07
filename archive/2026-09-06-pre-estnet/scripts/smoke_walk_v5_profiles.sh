#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

output_root="${1:?usage: scripts/smoke_walk_v5_profiles.sh OUTPUT_ROOT}"
gpu_index="${GPU_INDEX:-0}"
physical_gpu_index="${PHYSICAL_GPU_INDEX:-}"
if [[ -n "${physical_gpu_index}" ]]; then
    unset CUDA_VISIBLE_DEVICES
    device="cuda:${physical_gpu_index}"
else
    export CUDA_VISIBLE_DEVICES="${gpu_index}"
    device="cuda:0"
fi
export PYTHONUNBUFFERED=1

for reward_profile in p1_walk_stable_v5 p1_walk_stable_v5_imitation; do
    log_dir="${output_root}/${reward_profile}"
    if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
        echo "Refusing to overwrite smoke checkpoint in ${log_dir}" >&2
        exit 2
    fi
    mkdir -p "${log_dir}"
    python train.py \
        --headless \
        --device "${device}" \
        --num-envs 1 \
        --iterations 2 \
        --rollout-steps 4 \
        --learning-rate 0.00005 \
        --policy-learning-rate 0.000001 \
        --minimum-policy-learning-rate 0.0000005 \
        --target-kl 0.01 \
        --max-post-update-kl 0.02 \
        --value-loss-coef 0.5 \
        --entropy-coef 0.0 \
        --final-entropy-coef 0.0 \
        --prediction-loss-coef 0.05 \
        --estimation-loss-coef 0.10 \
        --initial-action-std 0.05 \
        --final-action-std 0.05 \
        --min-action-std 0.05 \
        --max-action-std 0.60 \
        --command-profile forward_walk \
        --command-scale 1.0 \
        --domain-randomization-scale 0.0 \
        --reward-profile "${reward_profile}" \
        --seed 67 \
        --save-interval 2 \
        --log-dir "${log_dir}" \
        2>&1 | tee "${log_dir}/console.log"
    test -s "${log_dir}/checkpoint_00002.pt"
done
