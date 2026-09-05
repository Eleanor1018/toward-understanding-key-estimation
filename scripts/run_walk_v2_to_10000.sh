#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

resume_checkpoint="${1:?usage: scripts/run_walk_v2_to_10000.sh CHECKPOINT LOG_DIR}"
log_dir="${2:?usage: scripts/run_walk_v2_to_10000.sh CHECKPOINT LOG_DIR}"
gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
target_iteration=10000
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "Long walking comparison requires at least two GPUs" >&2
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

start_iteration="$(
    /data/nora/isaacsim-4.5.0/env/bin/python - "${resume_checkpoint}" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state["iteration"]))
PY
)"
if (( start_iteration >= target_iteration )); then
    echo "Checkpoint iteration ${start_iteration} already reached target" >&2
    exit 2
fi

evaluation_checkpoints=("${resume_checkpoint}")
checkpoint_iteration=$((((start_iteration / 500) + 1) * 500))
while (( checkpoint_iteration <= target_iteration )); do
    evaluation_checkpoints+=(
        "${log_dir}/checkpoint_$(printf '%05d' "${checkpoint_iteration}").pt"
    )
    checkpoint_iteration=$((checkpoint_iteration + 500))
done

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

echo "Continuing iteration $((start_iteration + 1)) through ${target_iteration}"
torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations "${target_iteration}" \
    --rollout-steps 24 \
    --learning-rate 0.0001 \
    --policy-learning-rate 0.0000015 \
    --minimum-policy-learning-rate 0.000001 \
    --target-kl 0.02 \
    --max-post-update-kl 0.04 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.00015 \
    --final-entropy-coef 0.00002 \
    --prediction-loss-coef 0.05 \
    --estimation-loss-coef 0.10 \
    --initial-action-std 0.10 \
    --resume-action-std 0.10 \
    --final-action-std 0.06 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile forward_walk \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk_gait_v2 \
    --seed 50 \
    --save-interval 500 \
    --resume "${resume_checkpoint}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --command-profile forward_walk \
    --reward-profile p1_walk_gait_v2 \
    --checkpoints "${evaluation_checkpoints[@]}" \
    --output "${log_dir}/evaluation_10000.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
