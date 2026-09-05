#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/v8_ops_common.sh
source "${script_dir}/v8_ops_common.sh"

if (( $# != 2 )); then
    v8_die "usage: GPU_LIST=... $0 V7_CHECKPOINT OUTPUT_ROOT"
fi
v7_checkpoint="$(v8_resolve_existing_file "$1")"
output_root="$(v8_resolve_new_path "$2")"
gpu_list="${GPU_LIST:?GPU_LIST must name genuinely idle physical GPUs}"
v8_parse_gpu_list "${gpu_list}"
v8_require_runtime
cd "${repo_dir}"

[[ -f "${repo_dir}/v8_checkpoint.py" ]] \
    || v8_die "missing migration utility: ${repo_dir}/v8_checkpoint.py"
v7_metadata="$(
    "${V8_PYTHON_BIN}" - "${v7_checkpoint}" <<'PY'
import sys

from checkpoint_io import load_checkpoint

checkpoint = load_checkpoint(sys.argv[1], map_location="cpu")
print(
    checkpoint.get("iteration"),
    checkpoint.get("model_config", {}).get("command_dim"),
    checkpoint.get("model_config", {}).get("future_reference_dim", 0),
    checkpoint.get("input_normalization_type"),
    checkpoint.get("future_reference_normalization_type"),
    checkpoint.get("train_args", {}).get("reward_profile"),
    sep="\t",
)
PY
)"
IFS=$'\t' read -r \
    source_iteration command_dim future_dim normalization_type \
    future_normalization_type source_profile \
    <<<"${v7_metadata}"
[[ "${source_iteration}" == "4900" ]] \
    || v8_die "validated V8 launch requires the V7 iteration-4900 checkpoint"
[[ "${command_dim}" == "5" && "${future_dim}" == "0" ]] \
    || v8_die "source checkpoint is not the V7 model contract"
[[ "${normalization_type}" == "g1_fixed_physical_scales_phase_v2" ]] \
    || v8_die "source checkpoint has wrong base normalization"
[[ "${future_normalization_type}" == "None" ]] \
    || v8_die "source checkpoint already declares a future reference"
[[ "${source_profile}" == "p1_walk_stable_v7_full_reference" ]] \
    || v8_die "source checkpoint is not V7 full-reference"

v8_require_idle_gpus "${V8_GPU_INDICES[@]}"
mkdir -p -- "${output_root}"
source_sha256="$(v8_sha256_file "${v7_checkpoint}")"
migrated_checkpoint="${output_root}/migrated_checkpoint_04900_v8.pt"
cat >"${output_root}/run_identity.txt" <<EOF
schema=v8_validated_launch_v1
v7_checkpoint=${v7_checkpoint}
v7_checkpoint_sha256=${source_sha256}
v7_checkpoint_iteration=${source_iteration}
migrated_checkpoint=${migrated_checkpoint}
gpu_list=${gpu_list}
world_size=${V8_WORLD_SIZE}
smoke_sequence=fresh_1_then_resume_64_then_resume_512
formal_segment=4901..5400
v8_curriculum=4901..9900
automatic_continuation_after_5400=false
started_at=$(date --iso-8601=seconds)
EOF

"${V8_PYTHON_BIN}" v8_checkpoint.py \
    "${v7_checkpoint}" "${migrated_checkpoint}" \
    2>&1 | tee "${output_root}/migration.log"
[[ "$(v8_sha256_file "${v7_checkpoint}")" == "${source_sha256}" ]] \
    || v8_die "V7 source changed during migration"
test -s "${migrated_checkpoint}"
printf 'migrated_checkpoint_sha256=%s\n' \
    "$(v8_sha256_file "${migrated_checkpoint}")" \
    >>"${output_root}/run_identity.txt"

GPU_INDEX="${V8_GPU_INDICES[0]}" \
    "${script_dir}/smoke_walk_v8_future_reference.sh" \
    "${migrated_checkpoint}" "${output_root}/smoke"

GPU_LIST="${gpu_list}" \
    "${script_dir}/run_walk_v8_future_reference_to_5400.sh" \
    "${migrated_checkpoint}" "${output_root}/train_to_5400"

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" \
    >>"${output_root}/run_identity.txt"
printf 'Validated V8 launch completed and stopped at global iteration 5400.\n'
