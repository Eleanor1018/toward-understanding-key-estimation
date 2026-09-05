#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

checkpoint="${1:?usage: GPU_LIST=... scripts/run_walk_v7_full_reference_1000.sh CHECKPOINT LOG_DIR}"
log_dir="${2:?usage: GPU_LIST=... scripts/run_walk_v7_full_reference_1000.sh CHECKPOINT LOG_DIR}"
gpu_list="${GPU_LIST:?GPU_LIST must name genuinely idle physical GPU indices}"
train_seed="${TRAIN_SEED:-76}"
run_iterations="${RUN_ITERATIONS:-1000}"
save_interval="${SAVE_INTERVAL:-100}"
eval_num_envs="${EVAL_NUM_ENVS:-512}"
eval_steps="${EVAL_STEPS:-1000}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        echo "${name} must be a positive integer, got '${value}'" >&2
        exit 2
    fi
}

if [[ ! "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "GPU_LIST must be comma-separated physical GPU indices" >&2
    exit 2
fi
IFS=',' read -r -a gpu_indices <<<"${gpu_list}"
declare -A seen_gpu_indices=()
for gpu_index in "${gpu_indices[@]}"; do
    if [[ -n "${seen_gpu_indices[${gpu_index}]+present}" ]]; then
        echo "GPU_LIST contains duplicate index ${gpu_index}" >&2
        exit 2
    fi
    seen_gpu_indices[${gpu_index}]=1
done
world_size="${#gpu_indices[@]}"
num_envs="${NUM_ENVS:-$((world_size * 512))}"

require_positive_integer RUN_ITERATIONS "${run_iterations}"
require_positive_integer SAVE_INTERVAL "${save_interval}"
require_positive_integer NUM_ENVS "${num_envs}"
require_positive_integer EVAL_NUM_ENVS "${eval_num_envs}"
require_positive_integer EVAL_STEPS "${eval_steps}"
if [[ ! "${train_seed}" =~ ^[0-9]+$ ]]; then
    echo "TRAIN_SEED must be a non-negative integer" >&2
    exit 2
fi
run_iterations=$((10#${run_iterations}))
save_interval=$((10#${save_interval}))
num_envs=$((10#${num_envs}))
eval_num_envs=$((10#${eval_num_envs}))
eval_steps=$((10#${eval_steps}))
train_seed=$((10#${train_seed}))
if (( num_envs < world_size || num_envs % world_size != 0 )); then
    echo "NUM_ENVS must be at least GPU count and divisible by it" >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
fi
if [[ -e "${log_dir}" ]]; then
    echo "Refusing existing V7 output directory: ${log_dir}" >&2
    exit 2
fi

checkpoint_metadata="$(
    /data/nora/isaacsim-4.5.0/env/bin/python - "${checkpoint}" <<'PY'
import sys
from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
train_args = checkpoint.get("train_args", {})
print(
    checkpoint["iteration"],
    checkpoint["model_config"]["command_dim"],
    checkpoint.get("input_normalization_type"),
    train_args.get("reward_profile"),
    train_args.get("reference_motion_origin_iteration"),
    train_args.get("reference_motion_ramp_iterations"),
    train_args.get("iterations"),
    train_args.get("initial_imitation_weight"),
    train_args.get("final_imitation_weight"),
    train_args.get("minimum_policy_learning_rate"),
    train_args.get("final_action_std"),
    {group.get("name"): group.get("lr") for group in checkpoint["optimizer"]["param_groups"]}.get("policy"),
    {group.get("name"): group.get("lr") for group in checkpoint["optimizer"]["param_groups"]}.get("critic"),
    {group.get("name"): group.get("lr") for group in checkpoint["optimizer"]["param_groups"]}.get("decoder"),
    sep="\t",
)
PY
)"
IFS=$'\t' read -r \
    checkpoint_iteration command_dim normalization_type source_profile source_origin \
    source_ramp source_end source_initial_weight source_final_weight \
    source_minimum_policy_lr source_final_action_std source_policy_lr \
    source_critic_lr source_decoder_lr \
    <<<"${checkpoint_metadata}"

if [[ "${command_dim}" != "5" \
    || "${normalization_type}" != "g1_fixed_physical_scales_phase_v2" ]]; then
    echo "V7 requires a compatible phase-visible checkpoint" >&2
    exit 2
fi
if [[ "${source_profile}" != "p1_walk_stable_v6_phase_rsi_imitation" \
    && "${source_profile}" != "p1_walk_stable_v7_full_reference" ]]; then
    echo "Unsupported V7 source profile: ${source_profile}" >&2
    exit 2
fi

start_iteration=$((checkpoint_iteration + 1))
if [[ "${source_profile}" == "p1_walk_stable_v7_full_reference" ]]; then
    if [[ ! "${source_origin}" =~ ^[0-9]+$ ]]; then
        echo "V7 resume checkpoint omitted its curriculum origin" >&2
        exit 2
    fi
    reference_origin="${source_origin}"
    if [[ -z "${TARGET_ITERATION:-}" ]]; then
        echo "V7 crash resume requires TARGET_ITERATION to preserve the schedule" >&2
        exit 2
    fi
    require_positive_integer TARGET_ITERATION "${TARGET_ITERATION}"
    end_iteration=$((10#${TARGET_ITERATION}))
    if [[ "${end_iteration}" != "${source_end}" ]]; then
        echo "TARGET_ITERATION must preserve checkpoint schedule end ${source_end}" >&2
        exit 2
    fi
    reference_ramp="${source_ramp}"
    initial_imitation_weight="${source_initial_weight}"
    final_imitation_weight="${source_final_weight}"
    policy_learning_rate="${source_policy_lr}"
    minimum_policy_learning_rate="${source_minimum_policy_lr}"
    if [[ "${source_critic_lr}" != "${source_decoder_lr}" ]]; then
        echo "V7 checkpoint critic/decoder learning rates differ" >&2
        exit 2
    fi
    model_learning_rate="${source_critic_lr}"
    final_action_std="${source_final_action_std}"
    reset_optimizer_args=()
    resume_action_std_args=()
else
    reference_origin="${start_iteration}"
    end_iteration=$((checkpoint_iteration + run_iterations))
    reference_ramp=200
    initial_imitation_weight=0.75
    final_imitation_weight=0.30
    policy_learning_rate=0.000001
    minimum_policy_learning_rate=0.00000025
    model_learning_rate=0.00005
    final_action_std=0.10
    reset_optimizer_args=(--reset-optimizer-state)
    resume_action_std_args=(--resume-action-std 0.10)
fi
if (( end_iteration < start_iteration )); then
    echo "Target iteration must be at least ${start_iteration}" >&2
    exit 2
fi

mkdir -p "${log_dir}"
checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
cat >"${log_dir}/run_identity.txt" <<EOF
schema=v7_full_reference_1000_v1
source_checkpoint=$(realpath "${checkpoint}")
source_checkpoint_sha256=${checkpoint_sha256}
source_profile=${source_profile}
start_iteration=${start_iteration}
end_iteration=${end_iteration}
reference_motion_origin_iteration=${reference_origin}
reference_motion_ramp_iterations=${reference_ramp}
initial_imitation_weight=${initial_imitation_weight}
final_imitation_weight=${final_imitation_weight}
gpu_list=${gpu_list}
world_size=${world_size}
num_envs=${num_envs}
train_seed=${train_seed}
EOF

evaluation_checkpoints=("$(realpath "${checkpoint}")")
first_save_iteration=$((((checkpoint_iteration / save_interval) + 1) * save_interval))
for ((iteration = first_save_iteration; iteration <= end_iteration; iteration += save_interval)); do
    evaluation_checkpoints+=(
        "$(realpath -m "${log_dir}")/checkpoint_$(printf '%05d' "${iteration}").pt"
    )
done
if (( end_iteration % save_interval != 0 )); then
    evaluation_checkpoints+=(
        "$(realpath -m "${log_dir}")/checkpoint_$(printf '%05d' "${end_iteration}").pt"
    )
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs "${num_envs}" \
    --iterations "${end_iteration}" \
    --rollout-steps 24 \
    --learning-rate "${model_learning_rate}" \
    --policy-learning-rate "${policy_learning_rate}" \
    --minimum-policy-learning-rate "${minimum_policy_learning_rate}" \
    --target-kl 0.01 \
    --max-post-update-kl 0.02 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0 \
    --final-entropy-coef 0.0 \
    --prediction-loss-coef 0.05 \
    --estimation-loss-coef 0.10 \
    --initial-action-std 0.10 \
    "${resume_action_std_args[@]}" \
    --final-action-std "${final_action_std}" \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --initial-imitation-weight "${initial_imitation_weight}" \
    --final-imitation-weight "${final_imitation_weight}" \
    --reference-state-initialization-probability 0.70 \
    --reference-motion-ramp-iterations "${reference_ramp}" \
    --reference-motion-origin-iteration "${reference_origin}" \
    --command-profile forward_walk \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk_stable_v7_full_reference \
    --seed "${train_seed}" \
    --save-interval "${save_interval}" \
    --resume "${checkpoint}" \
    "${reset_optimizer_args[@]}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs "${eval_num_envs}" \
    --steps "${eval_steps}" \
    --seed 123 \
    --command-profile forward_walk \
    --reward-profile p1_walk_stable_v7_full_reference \
    --imitation-reward-weight 0.75 \
    --reference-state-initialization-probability 0.0 \
    --checkpoints "${evaluation_checkpoints[@]}" \
    --output "${log_dir}/evaluation.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
