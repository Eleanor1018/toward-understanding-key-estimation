#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/v8_ops_common.sh
source "${script_dir}/v8_ops_common.sh"

if (( $# != 2 )); then
    v8_die "usage: GPU_INDEX=N $0 MIGRATED_V8_CHECKPOINT OUTPUT_ROOT"
fi
checkpoint="$(v8_resolve_existing_file "$1")"
output_root="$(v8_resolve_new_path "$2")"
gpu_index="${GPU_INDEX:?GPU_INDEX must name one genuinely idle physical GPU}"
v8_parse_gpu_list "${gpu_index}"
(( V8_WORLD_SIZE == 1 )) || v8_die "GPU_INDEX must name exactly one GPU"
v8_require_runtime
cd "${repo_dir}"

checkpoint_metadata="$(
    "${V8_PYTHON_BIN}" - "${checkpoint}" <<'PY'
import sys

from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
migration = checkpoint.get("v8_checkpoint_migration", {})
print(
    checkpoint.get("iteration"),
    checkpoint.get("model_config", {}).get("command_dim"),
    checkpoint.get("model_config", {}).get("future_reference_dim"),
    checkpoint.get("input_normalization_type"),
    checkpoint.get("future_reference_normalization_type"),
    migration.get("type"),
    migration.get("source_iteration"),
    sep="\t",
)
PY
)"
IFS=$'\t' read -r \
    checkpoint_iteration command_dim future_dim normalization_type \
    future_normalization_type migration_type migration_source_iteration \
    <<<"${checkpoint_metadata}"
v8_require_nonnegative_integer CHECKPOINT_ITERATION "${checkpoint_iteration}"
[[ "${command_dim}" == "5" ]] || v8_die "migrated command_dim must be 5"
[[ "${future_dim}" == "63" ]] || v8_die "migrated future_reference_dim must be 63"
[[ "${normalization_type}" == "g1_fixed_physical_scales_phase_v2" ]] \
    || v8_die "migrated checkpoint has wrong base normalization"
[[ "${future_normalization_type}" == "g1_future_reference_3x21_v1" ]] \
    || v8_die "migrated checkpoint has wrong future normalization"
[[ "${migration_type}" == "future_reference_0_to_63_v1" ]] \
    || v8_die "checkpoint was not produced by v8_checkpoint.py"
[[ "${migration_source_iteration}" == "${checkpoint_iteration}" ]] \
    || v8_die "migration source iteration mismatch"

curriculum_origin=$((10#${checkpoint_iteration} + 1))
curriculum_end=$((10#${checkpoint_iteration} + 5000))
smoke_end=$((10#${checkpoint_iteration} + 2))
smoke_checkpoint="checkpoint_$(printf '%05d' "${smoke_end}").pt"

mkdir -p -- "${output_root}"
cat >"${output_root}/run_identity.txt" <<EOF
schema=v8_future_reference_smoke_v1
migrated_checkpoint=${checkpoint}
migrated_checkpoint_sha256=$(v8_sha256_file "${checkpoint}")
checkpoint_iteration=${checkpoint_iteration}
curriculum_origin_iteration=${curriculum_origin}
curriculum_end_iteration=${curriculum_end}
gpu_index=${gpu_index}
fresh_smoke=1_env_x_2_iterations_x_4_steps
resume_smoke=64_then_512_envs_x_2_iterations_x_24_steps
started_at=$(date --iso-8601=seconds)
EOF

export CUDA_VISIBLE_DEVICES="${gpu_index}"
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
v8_activate_runtime

v8_require_idle_gpus "${gpu_index}"
python train.py \
    --headless --device cuda:0 \
    --num-envs 1 --iterations 2 --rollout-steps 4 \
    --learning-rate 0.00005 \
    --policy-learning-rate 0.000001 \
    --minimum-policy-learning-rate 0.00000025 \
    --target-kl 0.01 --max-post-update-kl 0.02 \
    --value-loss-coef 0.5 --entropy-coef 0.0 --final-entropy-coef 0.0 \
    --prediction-loss-coef 0.05 --estimation-loss-coef 0.10 \
    --initial-action-std 0.10 --final-action-std 0.10 \
    --min-action-std 0.05 --max-action-std 0.60 \
    --v8-curriculum-origin-iteration 1 \
    --v8-curriculum-end-iteration 5000 \
    --command-profile forward_walk --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk_stable_v8_future_reference \
    --seed 81 --save-interval 1 \
    --log-dir "${output_root}/fresh_1" \
    2>&1 | tee "${output_root}/fresh_1.log"

run_resume_smoke() {
    local num_envs="$1"
    local name="$2"
    local seed="$3"
    local log_dir="${output_root}/${name}"

    v8_require_idle_gpus "${gpu_index}"
    mkdir -p -- "${log_dir}"
    torchrun --standalone --nproc_per_node=1 train_ddp.py \
        --distributed --headless \
        --num-envs "${num_envs}" \
        --iterations "${smoke_end}" --rollout-steps 24 \
        --learning-rate 0.00005 \
        --policy-learning-rate 0.000001 \
        --minimum-policy-learning-rate 0.00000025 \
        --target-kl 0.01 --max-post-update-kl 0.02 \
        --value-loss-coef 0.5 --entropy-coef 0.0 --final-entropy-coef 0.0 \
        --prediction-loss-coef 0.05 --estimation-loss-coef 0.10 \
        --initial-action-std 0.10 --resume-action-std 0.10 \
        --final-action-std 0.10 --min-action-std 0.05 --max-action-std 0.60 \
        --v8-curriculum-origin-iteration "${curriculum_origin}" \
        --v8-curriculum-end-iteration "${curriculum_end}" \
        --command-profile forward_walk --command-scale 1.0 \
        --domain-randomization-scale 0.0 \
        --reward-profile p1_walk_stable_v8_future_reference \
        --seed "${seed}" --save-interval 1 \
        --resume "${checkpoint}" --reset-optimizer-state \
        --log-dir "${log_dir}" \
        2>&1 | tee "${log_dir}/train.log"
}

run_resume_smoke 64 resume_64 82
run_resume_smoke 512 resume_512 83

test -s "${output_root}/fresh_1/checkpoint_00002.pt"
test -s "${output_root}/resume_64/${smoke_checkpoint}"
test -s "${output_root}/resume_512/${smoke_checkpoint}"
printf 'V8 future-reference smoke suite passed: %s\n' "${output_root}"
