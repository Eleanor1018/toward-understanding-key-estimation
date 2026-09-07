#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: GPU_LIST=INDEX[,INDEX...] scripts/run_walk_v6_phase_rsi_ab.sh LEGACY_CHECKPOINT OUTPUT_ROOT

Optional environment:
  TRAIN_SEED          Shared control/treatment seed (default: 61)
  CANARY_ITERATIONS   Iterations per arm (default: 500)
  NUM_ENVS            Total training environments (default: 512 per GPU)
  SAVE_INTERVAL       Checkpoint interval (default: 50)
  EVAL_NUM_ENVS       Evaluation environments (default: 512)
  EVAL_STEPS          Evaluation horizon (default: 1000)
EOF
}

die() {
    printf 'Phase-RSI A/B error: %s\n' "$*" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        die "${name} must be a positive integer, got '${value}'"
    fi
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

write_new_manifest() {
    local destination="$1"
    local contents="$2"
    local temporary

    [[ ! -e "${destination}" ]] || die "refusing to replace ${destination}"
    temporary="$(mktemp "${destination}.tmp.XXXXXX")"
    printf '%s\n' "${contents}" >"${temporary}"
    chmod 0644 "${temporary}"
    mv -T -- "${temporary}" "${destination}"
}

if (( $# != 2 )); then
    usage
    exit 2
fi

legacy_checkpoint_input="$1"
output_root_input="$2"
gpu_list="${GPU_LIST:-}"
[[ -n "${gpu_list}" ]] || die "GPU_LIST must be supplied explicitly"
if [[ ! "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    die "GPU_LIST must be a comma-separated list of physical GPU indices"
fi

IFS=',' read -r -a gpu_indices <<<"${gpu_list}"
declare -A seen_gpu_indices=()
for gpu_index in "${gpu_indices[@]}"; do
    if [[ -n "${seen_gpu_indices[${gpu_index}]+present}" ]]; then
        die "GPU_LIST contains duplicate physical index ${gpu_index}"
    fi
    seen_gpu_indices[${gpu_index}]=1
done
world_size="${#gpu_indices[@]}"

train_seed="${TRAIN_SEED:-61}"
canary_iterations="${CANARY_ITERATIONS:-500}"
save_interval="${SAVE_INTERVAL:-50}"
eval_num_envs="${EVAL_NUM_ENVS:-512}"
eval_steps="${EVAL_STEPS:-1000}"
num_envs="${NUM_ENVS:-$((world_size * 512))}"

[[ "${train_seed}" =~ ^[0-9]+$ ]] \
    || die "TRAIN_SEED must be a non-negative integer, got '${train_seed}'"
require_positive_integer CANARY_ITERATIONS "${canary_iterations}"
require_positive_integer SAVE_INTERVAL "${save_interval}"
require_positive_integer EVAL_NUM_ENVS "${eval_num_envs}"
require_positive_integer EVAL_STEPS "${eval_steps}"
require_positive_integer NUM_ENVS "${num_envs}"

train_seed=$((10#${train_seed}))
canary_iterations=$((10#${canary_iterations}))
save_interval=$((10#${save_interval}))
eval_num_envs=$((10#${eval_num_envs}))
eval_steps=$((10#${eval_steps}))
num_envs=$((10#${num_envs}))
if (( num_envs < world_size || num_envs % world_size != 0 )); then
    die "NUM_ENVS must be at least GPU count and divisible by it"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
python_bin="/data/nora/isaacsim-4.5.0/env/bin/python"
migration_script="${repo_dir}/phase_checkpoint.py"
canary_script="${script_dir}/run_walk_stable_v3_canary.sh"
evaluator_script="${script_dir}/evaluate_phase_rsi_ab.py"
orchestrator_script="${script_dir}/run_walk_v6_phase_rsi_ab.sh"

[[ -x "${python_bin}" ]] || die "missing executable Python: ${python_bin}"
for required_file in \
    "${migration_script}" \
    "${canary_script}" \
    "${evaluator_script}" \
    "${orchestrator_script}"; do
    [[ -f "${required_file}" ]] || die "missing required file: ${required_file}"
done
[[ -f "${legacy_checkpoint_input}" ]] \
    || die "legacy checkpoint does not exist: ${legacy_checkpoint_input}"

legacy_checkpoint="$(realpath -- "${legacy_checkpoint_input}")"
if [[ -e "${output_root_input}" && ! -d "${output_root_input}" ]]; then
    die "OUTPUT_ROOT exists and is not a directory: ${output_root_input}"
fi
mkdir -p -- "${output_root_input}"
output_root="$(realpath -- "${output_root_input}")"
cd "${repo_dir}"
migrated_checkpoint="${output_root}/migrated_checkpoint_03500_phase.pt"
migration_identity_file="${output_root}/phase_checkpoint_identity.txt"
lock_dir="${output_root}/.phase_rsi_ab.lock"

if ! mkdir -- "${lock_dir}" 2>/dev/null; then
    die "another run is active or a stale lock exists: ${lock_dir}"
fi
cleanup_lock() {
    rmdir -- "${lock_dir}" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

legacy_sha256="$(sha256_file "${legacy_checkpoint}")"
migration_script_sha256="$(sha256_file "${migration_script}")"
canary_script_sha256="$(sha256_file "${canary_script}")"
evaluator_script_sha256="$(sha256_file "${evaluator_script}")"
orchestrator_script_sha256="$(sha256_file "${orchestrator_script}")"

build_migration_identity() {
    local migrated_sha256_value="$1"
    cat <<EOF
schema=phase_checkpoint_identity_v1
source_checkpoint=${legacy_checkpoint}
source_checkpoint_sha256=${legacy_sha256}
migrated_checkpoint=${migrated_checkpoint}
migrated_checkpoint_sha256=${migrated_sha256_value}
migration_script=${migration_script}
migration_script_sha256=${migration_script_sha256}
migration_type=command_phase_3_to_5_v1
EOF
}

if [[ -e "${migrated_checkpoint}" || -e "${migration_identity_file}" ]]; then
    [[ -f "${migrated_checkpoint}" && -f "${migration_identity_file}" ]] \
        || die "refusing partial migration output in ${output_root}"
    migrated_sha256="$(sha256_file "${migrated_checkpoint}")"
    expected_migration_identity="$(build_migration_identity "${migrated_sha256}")"
    actual_migration_identity="$(<"${migration_identity_file}")"
    [[ "${actual_migration_identity}" == "${expected_migration_identity}" ]] \
        || die "migration identity mismatch: ${migration_identity_file}"
    printf 'Reusing verified migrated checkpoint: %s\n' "${migrated_checkpoint}"
else
    unmanaged_output="$(
        find "${output_root}" -mindepth 1 -maxdepth 1 \
            ! -path "${lock_dir}" -print -quit 2>/dev/null
    )"
    [[ -z "${unmanaged_output}" ]] \
        || die "refusing non-empty OUTPUT_ROOT without migration identity"

    "${python_bin}" "${migration_script}" \
        "${legacy_checkpoint}" "${migrated_checkpoint}"
    [[ "$(sha256_file "${legacy_checkpoint}")" == "${legacy_sha256}" ]] \
        || die "legacy checkpoint changed during migration"
    migrated_sha256="$(sha256_file "${migrated_checkpoint}")"
    expected_migration_identity="$(build_migration_identity "${migrated_sha256}")"
    write_new_manifest \
        "${migration_identity_file}" "${expected_migration_identity}"
fi

migration_metadata="$("${python_bin}" - "${migrated_checkpoint}" <<'PY'
import sys

from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
migration = checkpoint.get("phase_checkpoint_migration", {})
print(
    checkpoint.get("iteration"),
    migration.get("type"),
    migration.get("source_checkpoint_sha256"),
    migration.get("source_checkpoint_path"),
    sep="\t",
)
PY
)"
IFS=$'\t' read -r \
    start_iteration migration_type recorded_source_sha256 recorded_source_path \
    <<<"${migration_metadata}"
[[ "${start_iteration}" =~ ^[0-9]+$ ]] \
    || die "migrated checkpoint has invalid iteration '${start_iteration}'"
[[ "${migration_type}" == "command_phase_3_to_5_v1" ]] \
    || die "migrated checkpoint has unexpected migration type"
[[ "${recorded_source_sha256}" == "${legacy_sha256}" ]] \
    || die "migrated checkpoint records a different source digest"
[[ "${recorded_source_path}" == "${legacy_checkpoint}" ]] \
    || die "migrated checkpoint records a different source path"
[[ "$(sha256_file "${legacy_checkpoint}")" == "${legacy_sha256}" ]] \
    || die "legacy checkpoint changed after identity verification"

start_iteration=$((10#${start_iteration}))
end_iteration=$((start_iteration + canary_iterations))
final_checkpoint_name="checkpoint_$(printf '%05d' "${end_iteration}").pt"

# The resume loader requires the legacy min/max bounds before it resets std.
# Reset and final values of 0.10 make the scheduled action-std cap constant.
critic_learning_rate="0.00005"
policy_learning_rate="0.000001"
minimum_policy_learning_rate="0.00000025"
resume_action_std="0.10"
final_action_std="0.10"
minimum_action_std="0.05"
maximum_action_std="0.60"
initial_imitation_weight="0.15"
final_imitation_weight="0.03"
training_rsi_probability="0.70"
evaluation_rsi_probability="0.0"
reset_policy_optimizer_state="0"

build_arm_identity() {
    local arm_name="$1"
    local reward_profile="$2"
    cat <<EOF
schema=phase_rsi_ab_arm_v1
arm=${arm_name}
source_checkpoint=${legacy_checkpoint}
source_checkpoint_sha256=${legacy_sha256}
migrated_checkpoint=${migrated_checkpoint}
migrated_checkpoint_sha256=${migrated_sha256}
migration_script_sha256=${migration_script_sha256}
canary_script_sha256=${canary_script_sha256}
evaluator_script_sha256=${evaluator_script_sha256}
orchestrator_script_sha256=${orchestrator_script_sha256}
start_iteration=${start_iteration}
end_iteration=${end_iteration}
canary_iterations=${canary_iterations}
save_interval=${save_interval}
reward_profile=${reward_profile}
evaluation_reward_profile=${reward_profile}
train_seed=${train_seed}
gpu_list=${gpu_list}
world_size=${world_size}
num_envs=${num_envs}
environments_per_gpu=$((num_envs / world_size))
eval_num_envs=${eval_num_envs}
eval_steps=${eval_steps}
critic_and_decoder_learning_rate=${critic_learning_rate}
policy_learning_rate=${policy_learning_rate}
minimum_policy_learning_rate=${minimum_policy_learning_rate}
initial_action_std=${resume_action_std}
resume_action_std=${resume_action_std}
final_action_std=${final_action_std}
scheduled_action_std_cap=${final_action_std}
minimum_action_std=${minimum_action_std}
maximum_action_std=${maximum_action_std}
initial_imitation_weight=${initial_imitation_weight}
final_imitation_weight=${final_imitation_weight}
training_rsi_probability=${training_rsi_probability}
evaluation_rsi_probability=${evaluation_rsi_probability}
reset_policy_optimizer_state=${reset_policy_optimizer_state}
command_profile=forward_walk
command_scale=1.0
domain_randomization_scale=0.0
rollout_steps=24
target_kl=0.01
max_post_update_kl=0.02
value_loss_coefficient=0.5
entropy_coefficient=0.0
final_entropy_coefficient=0.0
prediction_loss_coefficient=0.05
estimation_loss_coefficient=0.10
EOF
}

validate_evaluation() {
    local evaluation_file="$1"
    local expected_final_checkpoint="$2"
    "${python_bin}" - \
        "${evaluation_file}" "${end_iteration}" "${expected_final_checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.evaluate_phase_rsi_ab import select_final_checkpoint_result

evaluation_path = Path(sys.argv[1])
expected_iteration = int(sys.argv[2])
expected_checkpoint = Path(sys.argv[3]).resolve()
report = json.loads(evaluation_path.read_text(encoding="utf-8"))
result = select_final_checkpoint_result(report)
if result["checkpoint_iteration"] != expected_iteration:
    raise SystemExit(
        f"{evaluation_path}: final iteration {result['checkpoint_iteration']} "
        f"does not match {expected_iteration}"
    )
if Path(result["checkpoint"]).resolve() != expected_checkpoint:
    raise SystemExit(
        f"{evaluation_path}: final checkpoint {result['checkpoint']} "
        f"does not match {expected_checkpoint}"
    )
PY
}

run_arm() {
    local arm_name="$1"
    local reward_profile="$2"
    local log_dir="${output_root}/${arm_name}_seed${train_seed}"
    local identity_file="${log_dir}/ab_identity.txt"
    local evaluation_file="${log_dir}/canary_evaluation.json"
    local final_checkpoint="${log_dir}/${final_checkpoint_name}"
    local expected_identity
    local actual_identity

    expected_identity="$(build_arm_identity "${arm_name}" "${reward_profile}")"
    if [[ -e "${log_dir}" ]]; then
        [[ -d "${log_dir}" ]] \
            || die "arm output exists and is not a directory: ${log_dir}"
        [[ -f "${identity_file}" ]] \
            || die "refusing arm without identity manifest: ${log_dir}"
        actual_identity="$(<"${identity_file}")"
        [[ "${actual_identity}" == "${expected_identity}" ]] \
            || die "A/B identity mismatch: ${identity_file}"
        [[ -s "${evaluation_file}" && -s "${final_checkpoint}" ]] \
            || die "refusing partial arm output: ${log_dir}"
        validate_evaluation "${evaluation_file}" "${final_checkpoint}"
        printf 'Skipping verified completed arm: %s\n' "${log_dir}"
        return
    fi

    mkdir -- "${log_dir}"
    write_new_manifest "${identity_file}" "${expected_identity}"
    env \
        GPU_LIST="${gpu_list}" \
        NUM_ENVS="${num_envs}" \
        TRAIN_SEED="${train_seed}" \
        CANARY_ITERATIONS="${canary_iterations}" \
        SAVE_INTERVAL="${save_interval}" \
        EVAL_NUM_ENVS="${eval_num_envs}" \
        EVAL_STEPS="${eval_steps}" \
        RESET_POLICY_OPTIMIZER_STATE="${reset_policy_optimizer_state}" \
        REWARD_PROFILE="${reward_profile}" \
        EVAL_REWARD_PROFILE="${reward_profile}" \
        LEARNING_RATE="${critic_learning_rate}" \
        POLICY_LEARNING_RATE="${policy_learning_rate}" \
        MINIMUM_POLICY_LEARNING_RATE="${minimum_policy_learning_rate}" \
        RESUME_ACTION_STD="${resume_action_std}" \
        FINAL_ACTION_STD="${final_action_std}" \
        MINIMUM_ACTION_STD="${minimum_action_std}" \
        INITIAL_IMITATION_WEIGHT="${initial_imitation_weight}" \
        FINAL_IMITATION_WEIGHT="${final_imitation_weight}" \
        RSI_PROBABILITY="${training_rsi_probability}" \
        "${canary_script}" "${migrated_checkpoint}" "${log_dir}"

    [[ -s "${evaluation_file}" && -s "${final_checkpoint}" ]] \
        || die "arm completed without required final outputs: ${log_dir}"
    validate_evaluation "${evaluation_file}" "${final_checkpoint}"
}

control_dir="${output_root}/control_seed${train_seed}"
treatment_dir="${output_root}/treatment_seed${train_seed}"
gate_report="${output_root}/phase_rsi_ab_gate_report_seed${train_seed}.json"
if [[ -e "${gate_report}" \
    && ! -s "${control_dir}/canary_evaluation.json" ]]; then
    die "refusing stale gate report without a completed control arm: ${gate_report}"
fi
if [[ -e "${gate_report}" \
    && ! -s "${treatment_dir}/canary_evaluation.json" ]]; then
    die "refusing stale gate report without a completed treatment arm: ${gate_report}"
fi

run_arm control p1_walk_stable_v6_phase_rsi
run_arm treatment p1_walk_stable_v6_phase_rsi_imitation

set +e
"${python_bin}" "${evaluator_script}" \
    "${control_dir}/canary_evaluation.json" \
    "${treatment_dir}/canary_evaluation.json" \
    --output "${gate_report}"
gate_status=$?
set -e

printf 'Phase-RSI gate evaluator exit code: %s\n' "${gate_status}"
if (( gate_status == 0 )); then
    printf 'Phase-RSI A/B gates passed; report: %s\n' "${gate_report}"
else
    printf 'Phase-RSI A/B gates failed or could not be evaluated.\n' >&2
    printf 'Completed run outputs are preserved at: %s\n' "${output_root}" >&2
    printf 'Gate report path: %s\n' "${gate_report}" >&2
fi
exit "${gate_status}"
