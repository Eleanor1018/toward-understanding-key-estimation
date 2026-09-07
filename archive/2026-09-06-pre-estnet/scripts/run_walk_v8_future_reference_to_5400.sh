#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/v8_ops_common.sh
source "${script_dir}/v8_ops_common.sh"

if (( $# != 2 )); then
    v8_die "usage: GPU_LIST=... $0 MIGRATED_V8_CHECKPOINT LOG_DIR"
fi
checkpoint="$(v8_resolve_existing_file "$1")"
log_dir="$(v8_resolve_new_path "$2")"
gpu_list="${GPU_LIST:?GPU_LIST must name genuinely idle physical GPUs}"
train_seed="${TRAIN_SEED:-84}"
eval_num_envs="${EVAL_NUM_ENVS:-512}"
eval_steps="${EVAL_STEPS:-1000}"
save_interval=50
target_iteration=5400
curriculum_origin=4901
curriculum_end=9900

v8_parse_gpu_list "${gpu_list}"
v8_require_runtime
v8_require_nonnegative_integer TRAIN_SEED "${train_seed}"
v8_require_positive_integer EVAL_NUM_ENVS "${eval_num_envs}"
v8_require_positive_integer EVAL_STEPS "${eval_steps}"
train_seed=$((10#${train_seed}))
eval_num_envs=$((10#${eval_num_envs}))
eval_steps=$((10#${eval_steps}))
num_envs="${NUM_ENVS:-$((V8_WORLD_SIZE * 512))}"
v8_require_positive_integer NUM_ENVS "${num_envs}"
num_envs=$((10#${num_envs}))
if (( num_envs < V8_WORLD_SIZE || num_envs % V8_WORLD_SIZE != 0 )); then
    v8_die "NUM_ENVS must be at least GPU count and divisible by it"
fi
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
    checkpoint.get("train_args", {}).get("reward_profile"),
    migration.get("type"),
    migration.get("source_checkpoint_path"),
    migration.get("source_checkpoint_sha256"),
    sep="\t",
)
PY
)"
IFS=$'\t' read -r \
    checkpoint_iteration command_dim future_dim normalization_type \
    future_normalization_type source_profile migration_type v7_checkpoint \
    v7_checkpoint_sha256 \
    <<<"${checkpoint_metadata}"
[[ "${checkpoint_iteration}" == "4900" ]] \
    || v8_die "formal V8 segment requires source iteration 4900"
[[ "${command_dim}" == "5" && "${future_dim}" == "63" ]] \
    || v8_die "formal V8 segment requires command=5 and future_reference=63"
[[ "${normalization_type}" == "g1_fixed_physical_scales_phase_v2" ]] \
    || v8_die "migrated checkpoint has wrong base normalization"
[[ "${future_normalization_type}" == "g1_future_reference_3x21_v1" ]] \
    || v8_die "migrated checkpoint has wrong future normalization"
[[ "${source_profile}" == "p1_walk_stable_v7_full_reference" ]] \
    || v8_die "migration must preserve a V7 full-reference source profile"
[[ "${migration_type}" == "future_reference_0_to_63_v1" ]] \
    || v8_die "checkpoint was not produced by v8_checkpoint.py"

mkdir -p -- "${log_dir}"
cat >"${log_dir}/run_identity.txt" <<EOF
schema=v8_future_reference_segment_4901_5400_v1
migrated_checkpoint=${checkpoint}
migrated_checkpoint_sha256=$(v8_sha256_file "${checkpoint}")
v7_checkpoint=${v7_checkpoint}
v7_checkpoint_sha256=${v7_checkpoint_sha256}
start_iteration=4901
target_iteration=${target_iteration}
new_iterations=500
v8_curriculum_origin_iteration=${curriculum_origin}
v8_curriculum_end_iteration=${curriculum_end}
gpu_list=${gpu_list}
world_size=${V8_WORLD_SIZE}
num_envs=${num_envs}
train_seed=${train_seed}
rollout_steps=24
model_learning_rate=0.00005
policy_learning_rate=0.000001
minimum_policy_learning_rate=0.00000025
target_kl=0.01
max_post_update_kl=0.02
prediction_loss_coefficient=0.05
estimation_loss_coefficient=0.10
action_std=0.10
domain_randomization_scale=0.0
command_profile=forward_walk
reward_profile=p1_walk_stable_v8_future_reference
save_interval=${save_interval}
automatic_continuation_after_5400=false
started_at=$(date --iso-8601=seconds)
EOF

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
v8_activate_runtime
v8_require_idle_gpus "${V8_GPU_INDICES[@]}"

torchrun --standalone --nproc_per_node="${V8_WORLD_SIZE}" train_ddp.py \
    --distributed --headless \
    --num-envs "${num_envs}" \
    --iterations "${target_iteration}" --rollout-steps 24 \
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
    --seed "${train_seed}" --save-interval "${save_interval}" \
    --resume "${checkpoint}" --reset-optimizer-state \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train.log"

final_checkpoint="${log_dir}/checkpoint_05400.pt"
test -s "${final_checkpoint}"

eval_gpu="${V8_GPU_INDICES[0]}"
v8_require_idle_gpus "${eval_gpu}"
export CUDA_VISIBLE_DEVICES="${eval_gpu}"
python evaluate.py \
    --headless --device cuda:0 \
    --num-envs "${eval_num_envs}" --steps "${eval_steps}" --seed 123 \
    --command-profile forward_walk \
    --reward-profile p1_walk_stable_v8_future_reference \
    --reference-state-initialization-probability 0.0 \
    --checkpoints "${final_checkpoint}" \
    --output "${log_dir}/evaluation_05400.json" \
    2>&1 | tee "${log_dir}/evaluation_05400.log"

test -s "${log_dir}/evaluation_05400.json"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" \
    >>"${log_dir}/run_identity.txt"
printf 'V8 stopped at iteration 5400 after evaluation; no continuation launched.\n'
