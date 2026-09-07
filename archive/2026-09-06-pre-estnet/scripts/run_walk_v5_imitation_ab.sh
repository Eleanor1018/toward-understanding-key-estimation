#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

checkpoint="${1:?usage: scripts/run_walk_v5_imitation_ab.sh CHECKPOINT OUTPUT_ROOT}"
output_root="${2:?usage: scripts/run_walk_v5_imitation_ab.sh CHECKPOINT OUTPUT_ROOT}"
train_seed="${TRAIN_SEED:-61}"
gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"

run_arm() {
    local arm_name="$1"
    local reward_profile="$2"
    local log_dir="${output_root}/${arm_name}_seed${train_seed}"
    local identity_file="${log_dir}/ab_identity.txt"
    local checkpoint_sha256
    local expected_identity

    checkpoint_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    expected_identity="checkpoint_sha256=${checkpoint_sha256}
reward_profile=${reward_profile}
evaluation_profile=p1_walk_stable_v5
train_seed=${train_seed}
gpu_list=${gpu_list}
num_envs=${NUM_ENVS:-auto}"

    if [[ -e "${log_dir}" && ! -f "${identity_file}" ]]; then
        echo "Refusing arm without an A/B identity manifest: ${log_dir}" >&2
        exit 2
    fi
    if [[ -f "${identity_file}" && "$(<"${identity_file}")" != "${expected_identity}" ]]; then
        echo "A/B identity mismatch in ${identity_file}" >&2
        exit 2
    fi

    if [[ -s "${log_dir}/canary_evaluation.json" ]]; then
        echo "Skipping completed arm: ${log_dir}"
        return
    fi
    if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
        echo "Refusing partial arm with existing checkpoints: ${log_dir}" >&2
        exit 2
    fi
    mkdir -p "${log_dir}"
    printf '%s\n' "${expected_identity}" >"${identity_file}"

    env \
        GPU_LIST="${gpu_list}" \
        NUM_ENVS="${NUM_ENVS:-}" \
        TRAIN_SEED="${train_seed}" \
        RESET_POLICY_OPTIMIZER_STATE=1 \
        REWARD_PROFILE="${reward_profile}" \
        EVAL_REWARD_PROFILE=p1_walk_stable_v5 \
        "${script_dir}/run_walk_stable_v3_canary.sh" \
        "${checkpoint}" \
        "${log_dir}"
}

run_arm control p1_walk_stable_v5
run_arm deepmimic_lite p1_walk_stable_v5_imitation
