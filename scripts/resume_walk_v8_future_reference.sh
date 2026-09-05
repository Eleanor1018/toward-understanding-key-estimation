#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/v8_ops_common.sh
source "${script_dir}/v8_ops_common.sh"

if (( $# != 3 )); then
    v8_die "usage: GPU_LIST=... $0 V8_CHECKPOINT LOG_DIR TARGET_ITERATION"
fi
checkpoint="$(v8_resolve_existing_file "$1")"
log_dir="$(v8_resolve_new_path "$2")"
target_iteration="$3"
gpu_list="${GPU_LIST:?GPU_LIST must name genuinely idle physical GPUs}"
eval_num_envs="${EVAL_NUM_ENVS:-512}"
eval_steps="${EVAL_STEPS:-1000}"
save_interval=50

v8_parse_gpu_list "${gpu_list}"
v8_require_runtime
v8_require_positive_integer TARGET_ITERATION "${target_iteration}"
v8_require_positive_integer EVAL_NUM_ENVS "${eval_num_envs}"
v8_require_positive_integer EVAL_STEPS "${eval_steps}"
target_iteration=$((10#${target_iteration}))
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
import math
import sys

import torch

from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
config = checkpoint.get("model_config", {})
args = checkpoint.get("train_args", {})
groups = {
    group.get("name"): group for group in checkpoint.get("optimizer", {}).get("param_groups", [])
}
required_groups = {"policy", "critic", "decoder"}
if set(groups) != required_groups:
    raise SystemExit(f"optimizer groups must be {sorted(required_groups)}, got {sorted(groups)}")
if groups["critic"].get("lr") != groups["decoder"].get("lr"):
    raise SystemExit("critic and decoder learning rates must match for train_ddp.py")
log_std = checkpoint.get("action_distribution", {}).get("log_std")
if not isinstance(log_std, torch.Tensor) or tuple(log_std.shape) != (29,):
    raise SystemExit("checkpoint action log_std must have shape (29,)")
action_std = log_std.exp()
values = (
    checkpoint.get("iteration"),
    config.get("command_dim"),
    config.get("future_reference_dim"),
    checkpoint.get("input_normalization_type"),
    checkpoint.get("future_reference_normalization_type"),
    args.get("reward_profile"),
    args.get("v8_curriculum_origin_iteration"),
    args.get("v8_curriculum_end_iteration"),
    args.get("current_task_mix_beta"),
    args.get("current_reference_state_initialization_probability"),
    groups["policy"].get("lr"),
    groups["critic"].get("lr"),
    args.get("minimum_policy_learning_rate"),
    args.get("final_action_std"),
    args.get("min_action_std"),
    args.get("max_action_std"),
    args.get("target_kl"),
    args.get("max_post_update_kl"),
    args.get("value_loss_coef"),
    args.get("entropy_coef"),
    args.get("final_entropy_coef"),
    args.get("prediction_loss_coef"),
    args.get("estimation_loss_coef"),
    args.get("rollout_steps"),
    args.get("seed"),
    args.get("command_scale"),
    args.get("domain_randomization_scale"),
    action_std.mean().item(),
    action_std.max().item(),
)
if any(value is None for value in values):
    missing = [index for index, value in enumerate(values) if value is None]
    raise SystemExit(f"checkpoint omitted required resume metadata fields {missing}")
if not all(
    math.isfinite(float(value))
    for value in values[8:23] + values[25:]
):
    raise SystemExit("checkpoint contains non-finite resume metadata")
print(*values, sep="\t")
PY
)"
IFS=$'\t' read -r \
    checkpoint_iteration command_dim future_dim normalization_type \
    future_normalization_type source_profile curriculum_origin curriculum_end \
    current_beta current_rsi policy_learning_rate model_learning_rate \
    minimum_policy_learning_rate final_action_std minimum_action_std \
    maximum_action_std target_kl max_post_update_kl value_loss_coef entropy_coef \
    final_entropy_coef prediction_loss_coef estimation_loss_coef rollout_steps \
    train_seed command_scale domain_randomization_scale action_std_mean action_std_max \
    <<<"${checkpoint_metadata}"

v8_require_nonnegative_integer CHECKPOINT_ITERATION "${checkpoint_iteration}"
v8_require_positive_integer CURRICULUM_ORIGIN "${curriculum_origin}"
v8_require_positive_integer CURRICULUM_END "${curriculum_end}"
v8_require_positive_integer ROLLOUT_STEPS "${rollout_steps}"
v8_require_nonnegative_integer TRAIN_SEED "${train_seed}"
[[ "${command_dim}" == "5" && "${future_dim}" == "63" ]] \
    || v8_die "resume checkpoint does not use the V8 model contract"
[[ "${normalization_type}" == "g1_fixed_physical_scales_phase_v2" ]] \
    || v8_die "resume checkpoint has wrong base normalization"
[[ "${future_normalization_type}" == "g1_future_reference_3x21_v1" ]] \
    || v8_die "resume checkpoint has wrong future normalization"
[[ "${source_profile}" == "p1_walk_stable_v8_future_reference" ]] \
    || v8_die "resume checkpoint is not a trained V8 checkpoint"
[[ "${curriculum_origin}" == "4901" && "${curriculum_end}" == "9900" ]] \
    || v8_die "resume checkpoint must retain V8 bounds 4901..9900"
if (( target_iteration <= 10#${checkpoint_iteration} )); then
    v8_die "TARGET_ITERATION must exceed checkpoint iteration ${checkpoint_iteration}"
fi
if (( target_iteration > 10#${curriculum_end} )); then
    v8_die "TARGET_ITERATION must not exceed curriculum end ${curriculum_end}"
fi

mkdir -p -- "${log_dir}"
cat >"${log_dir}/run_identity.txt" <<EOF
schema=v8_future_reference_crash_resume_v1
source_checkpoint=${checkpoint}
source_checkpoint_sha256=$(v8_sha256_file "${checkpoint}")
source_iteration=${checkpoint_iteration}
target_iteration=${target_iteration}
v8_curriculum_origin_iteration=${curriculum_origin}
v8_curriculum_end_iteration=${curriculum_end}
source_task_mix_beta=${current_beta}
source_training_rsi_probability=${current_rsi}
source_policy_optimizer_lr=${policy_learning_rate}
source_model_optimizer_lr=${model_learning_rate}
source_action_std_mean=${action_std_mean}
source_action_std_max=${action_std_max}
resume_action_std_reset=false
optimizer_reset=false
gpu_list=${gpu_list}
world_size=${V8_WORLD_SIZE}
num_envs=${num_envs}
train_seed=${train_seed}
save_interval=${save_interval}
started_at=$(date --iso-8601=seconds)
EOF

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
v8_activate_runtime
v8_require_idle_gpus "${V8_GPU_INDICES[@]}"

# Deliberately omit both --resume-action-std and optimizer-reset flags.
torchrun --standalone --nproc_per_node="${V8_WORLD_SIZE}" train_ddp.py \
    --distributed --headless \
    --num-envs "${num_envs}" \
    --iterations "${target_iteration}" --rollout-steps "${rollout_steps}" \
    --learning-rate "${model_learning_rate}" \
    --policy-learning-rate "${policy_learning_rate}" \
    --minimum-policy-learning-rate "${minimum_policy_learning_rate}" \
    --target-kl "${target_kl}" --max-post-update-kl "${max_post_update_kl}" \
    --value-loss-coef "${value_loss_coef}" \
    --entropy-coef "${entropy_coef}" --final-entropy-coef "${final_entropy_coef}" \
    --prediction-loss-coef "${prediction_loss_coef}" \
    --estimation-loss-coef "${estimation_loss_coef}" \
    --initial-action-std "${action_std_max}" \
    --final-action-std "${final_action_std}" \
    --min-action-std "${minimum_action_std}" \
    --max-action-std "${maximum_action_std}" \
    --v8-curriculum-origin-iteration "${curriculum_origin}" \
    --v8-curriculum-end-iteration "${curriculum_end}" \
    --command-profile forward_walk --command-scale "${command_scale}" \
    --domain-randomization-scale "${domain_randomization_scale}" \
    --reward-profile p1_walk_stable_v8_future_reference \
    --seed "${train_seed}" --save-interval "${save_interval}" \
    --resume "${checkpoint}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train.log"

target_checkpoint="${log_dir}/checkpoint_$(printf '%05d' "${target_iteration}").pt"
test -s "${target_checkpoint}"

eval_gpu="${V8_GPU_INDICES[0]}"
v8_require_idle_gpus "${eval_gpu}"
export CUDA_VISIBLE_DEVICES="${eval_gpu}"
python evaluate.py \
    --headless --device cuda:0 \
    --num-envs "${eval_num_envs}" --steps "${eval_steps}" --seed 123 \
    --command-profile forward_walk \
    --reward-profile p1_walk_stable_v8_future_reference \
    --reference-state-initialization-probability 0.0 \
    --checkpoints "${target_checkpoint}" \
    --output "${log_dir}/evaluation_$(printf '%05d' "${target_iteration}").json" \
    2>&1 | tee "${log_dir}/evaluation_$(printf '%05d' "${target_iteration}").log"

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" \
    >>"${log_dir}/run_identity.txt"
printf 'V8 crash resume completed at explicit target %d.\n' "${target_iteration}"
