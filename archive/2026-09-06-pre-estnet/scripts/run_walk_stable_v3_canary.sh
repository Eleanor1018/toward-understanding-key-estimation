#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

checkpoint="${1:?usage: scripts/run_walk_stable_v3_canary.sh CHECKPOINT LOG_DIR}"
log_dir="${2:?usage: scripts/run_walk_stable_v3_canary.sh CHECKPOINT LOG_DIR}"
gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
reset_policy_optimizer_state="${RESET_POLICY_OPTIMIZER_STATE:-0}"
reward_profile="${REWARD_PROFILE:-p1_walk_stable_v3}"
eval_reward_profile="${EVAL_REWARD_PROFILE:-${reward_profile}}"
train_seed="${TRAIN_SEED:-51}"
canary_iterations="${CANARY_ITERATIONS:-250}"
save_interval="${SAVE_INTERVAL:-50}"
learning_rate="${LEARNING_RATE:-0.00005}"
policy_learning_rate="${POLICY_LEARNING_RATE:-0.000001}"
minimum_policy_learning_rate="${MINIMUM_POLICY_LEARNING_RATE:-0.0000005}"
resume_action_std="${RESUME_ACTION_STD:-0.05}"
final_action_std="${FINAL_ACTION_STD:-0.05}"
minimum_action_std="${MINIMUM_ACTION_STD:-0.05}"
initial_imitation_weight="${INITIAL_IMITATION_WEIGHT:-0.15}"
final_imitation_weight="${FINAL_IMITATION_WEIGHT:-0.03}"
rsi_probability="${RSI_PROBABILITY:-0.70}"
eval_num_envs="${EVAL_NUM_ENVS:-512}"
eval_steps="${EVAL_STEPS:-1000}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"
num_envs="${NUM_ENVS:-$((world_size * 512))}"
reset_optimizer_args=()

if [[ "${reset_policy_optimizer_state}" == "1" ]]; then
    reset_optimizer_args+=(--reset-policy-optimizer-state)
elif [[ "${reset_policy_optimizer_state}" != "0" ]]; then
    echo "RESET_POLICY_OPTIMIZER_STATE must be 0 or 1" >&2
    exit 2
fi
if [[ "${reward_profile}" != "p1_walk_stable_v3" \
    && "${reward_profile}" != "p1_walk_stable_v4" \
    && "${reward_profile}" != "p1_walk_stable_v5" \
    && "${reward_profile}" != "p1_walk_stable_v5_imitation" \
    && "${reward_profile}" != "p1_walk_stable_v6_phase_rsi" \
    && "${reward_profile}" != "p1_walk_stable_v6_phase_rsi_imitation" ]]; then
    echo "Unsupported stable-walk REWARD_PROFILE: ${reward_profile}" >&2
    exit 2
fi
if [[ "${eval_reward_profile}" != "p1_walk_stable_v3" \
    && "${eval_reward_profile}" != "p1_walk_stable_v4" \
    && "${eval_reward_profile}" != "p1_walk_stable_v5" \
    && "${eval_reward_profile}" != "p1_walk_stable_v5_imitation" \
    && "${eval_reward_profile}" != "p1_walk_stable_v6_phase_rsi" \
    && "${eval_reward_profile}" != "p1_walk_stable_v6_phase_rsi_imitation" ]]; then
    echo "Unsupported stable-walk EVAL_REWARD_PROFILE: ${eval_reward_profile}" >&2
    exit 2
fi

if (( world_size < 1 )); then
    echo "Stable-walk canary requires at least one GPU" >&2
    exit 2
fi
if (( num_envs < world_size || num_envs % world_size != 0 )); then
    echo "NUM_ENVS must be at least WORLD_SIZE and divisible by it" >&2
    exit 2
fi
if (( canary_iterations < 1 || save_interval < 1 )); then
    echo "CANARY_ITERATIONS and SAVE_INTERVAL must be positive integers" >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
fi
if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
    echo "Refusing to overwrite checkpoints in ${log_dir}" >&2
    exit 2
fi

start_iteration="$(
    /data/nora/isaacsim-4.5.0/env/bin/python - "${checkpoint}" <<'PY'
import sys

from checkpoint_io import load_checkpoint

state = load_checkpoint(sys.argv[1], map_location="cpu")
print(int(state["iteration"]))
PY
)"
end_iteration=$((start_iteration + canary_iterations))
evaluation_checkpoints=("${checkpoint}")
first_save_iteration=$((((start_iteration / save_interval) + 1) * save_interval))
for ((save_iteration = first_save_iteration; save_iteration <= end_iteration; save_iteration += save_interval)); do
    evaluation_checkpoints+=(
        "${log_dir}/checkpoint_$(printf '%05d' "${save_iteration}").pt"
    )
done
if (( end_iteration % save_interval != 0 )); then
    evaluation_checkpoints+=(
        "${log_dir}/checkpoint_$(printf '%05d' "${end_iteration}").pt"
    )
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs "${num_envs}" \
    --iterations "${end_iteration}" \
    --rollout-steps 24 \
    --learning-rate "${learning_rate}" \
    --policy-learning-rate "${policy_learning_rate}" \
    --minimum-policy-learning-rate "${minimum_policy_learning_rate}" \
    --target-kl 0.01 \
    --max-post-update-kl 0.02 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0 \
    --final-entropy-coef 0.0 \
    --prediction-loss-coef 0.05 \
    --estimation-loss-coef 0.10 \
    --initial-action-std "${resume_action_std}" \
    --resume-action-std "${resume_action_std}" \
    --final-action-std "${final_action_std}" \
    --min-action-std "${minimum_action_std}" \
    --max-action-std 0.60 \
    --initial-imitation-weight "${initial_imitation_weight}" \
    --final-imitation-weight "${final_imitation_weight}" \
    --reference-state-initialization-probability "${rsi_probability}" \
    --command-profile forward_walk \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile "${reward_profile}" \
    --seed "${train_seed}" \
    --save-interval "${save_interval}" \
    --resume "${checkpoint}" \
    "${reset_optimizer_args[@]}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs "${eval_num_envs}" \
    --steps "${eval_steps}" \
    --reference-state-initialization-probability 0.0 \
    --command-profile forward_walk \
    --reward-profile "${eval_reward_profile}" \
    --checkpoints "${evaluation_checkpoints[@]}" \
    --output "${log_dir}/canary_evaluation.json" \
    2>&1 | tee "${log_dir}/evaluation.log"
