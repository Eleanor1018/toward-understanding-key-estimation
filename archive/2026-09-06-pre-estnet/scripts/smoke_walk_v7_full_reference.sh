#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

checkpoint="${1:?usage: GPU_INDEX=N scripts/smoke_walk_v7_full_reference.sh CHECKPOINT OUTPUT_ROOT}"
output_root="${2:?usage: GPU_INDEX=N scripts/smoke_walk_v7_full_reference.sh CHECKPOINT OUTPUT_ROOT}"
gpu_index="${GPU_INDEX:?GPU_INDEX must be a genuinely idle physical GPU}"

if [[ ! "${gpu_index}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GPU_INDEX must be one physical GPU index" >&2
    exit 2
fi

if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
fi
if [[ -e "${output_root}" ]]; then
    echo "Refusing existing smoke output: ${output_root}" >&2
    exit 2
fi
mkdir -p "${output_root}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export PYTHONUNBUFFERED=1

checkpoint_iteration="$(
    /data/nora/isaacsim-4.5.0/env/bin/python - "${checkpoint}" <<'PY'
import sys
from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
if checkpoint["model_config"]["command_dim"] != 5:
    raise SystemExit("V7 smoke requires a phase-visible checkpoint")
if checkpoint.get("train_args", {}).get("reward_profile") not in (
    "p1_walk_stable_v6_phase_rsi",
    "p1_walk_stable_v6_phase_rsi_imitation",
):
    raise SystemExit("V7 smoke requires a V6 warm-start checkpoint")
print(int(checkpoint["iteration"]))
PY
)"
start_iteration=$((checkpoint_iteration + 1))
end_iteration=$((checkpoint_iteration + 2))
end_checkpoint="checkpoint_$(printf '%05d' "${end_iteration}").pt"

python train.py \
    --headless --device cuda:0 \
    --num-envs 1 --iterations 2 --rollout-steps 4 \
    --command-profile forward_walk \
    --reward-profile p1_walk_stable_v7_full_reference \
    --domain-randomization-scale 0.0 \
    --reference-state-initialization-probability 0.70 \
    --reference-motion-ramp-iterations 1 \
    --initial-imitation-weight 0.75 --final-imitation-weight 0.30 \
    --initial-action-std 0.10 --final-action-std 0.10 \
    --min-action-std 0.05 --save-interval 1 \
    --log-dir "${output_root}/single_1" \
    2>&1 | tee "${output_root}/single_1.log"

run_resume_smoke() {
    local num_envs="$1"
    local name="$2"
    local seed="$3"
    local log_dir="${output_root}/${name}"
    mkdir -p "${log_dir}"
    torchrun --standalone --nproc_per_node=1 train_ddp.py \
        --distributed --headless \
        --num-envs "${num_envs}" --iterations "${end_iteration}" --rollout-steps 24 \
        --learning-rate 0.00005 \
        --policy-learning-rate 0.000001 \
        --minimum-policy-learning-rate 0.00000025 \
        --target-kl 0.01 --max-post-update-kl 0.02 \
        --value-loss-coef 0.5 --entropy-coef 0.0 --final-entropy-coef 0.0 \
        --prediction-loss-coef 0.05 --estimation-loss-coef 0.10 \
        --initial-action-std 0.10 --resume-action-std 0.10 \
        --final-action-std 0.10 --min-action-std 0.05 --max-action-std 0.60 \
        --initial-imitation-weight 0.75 --final-imitation-weight 0.30 \
        --reference-state-initialization-probability 0.70 \
        --reference-motion-ramp-iterations 200 \
        --reference-motion-origin-iteration "${start_iteration}" \
        --command-profile forward_walk --domain-randomization-scale 0.0 \
        --reward-profile p1_walk_stable_v7_full_reference \
        --seed "${seed}" --save-interval 1 \
        --resume "${checkpoint}" --reset-optimizer-state \
        --log-dir "${log_dir}" \
        2>&1 | tee "${log_dir}/train.log"
}

run_resume_smoke 64 resume_64 77
run_resume_smoke 512 resume_512 78

python evaluate.py \
    --headless --device cuda:0 \
    --num-envs 64 --steps 100 --seed 123 \
    --command-profile forward_walk \
    --reward-profile p1_walk_stable_v7_full_reference \
    --reference-state-initialization-probability 0.0 \
    --checkpoints "${output_root}/resume_512/${end_checkpoint}" \
    --output "${output_root}/evaluation_64x100.json" \
    2>&1 | tee "${output_root}/evaluation_64x100.log"

test -s "${output_root}/single_1/checkpoint_00002.pt"
test -s "${output_root}/resume_64/${end_checkpoint}"
test -s "${output_root}/resume_512/${end_checkpoint}"
test -s "${output_root}/evaluation_64x100.json"
printf 'V7 full-reference smoke suite passed: %s\n' "${output_root}"
