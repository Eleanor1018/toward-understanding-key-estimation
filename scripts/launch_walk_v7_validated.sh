#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

checkpoint="${1:?usage: scripts/launch_walk_v7_validated.sh CHECKPOINT OUTPUT_ROOT GPU_LIST}"
output_root="${2:?usage: scripts/launch_walk_v7_validated.sh CHECKPOINT OUTPUT_ROOT GPU_LIST}"
gpu_list="${3:?usage: scripts/launch_walk_v7_validated.sh CHECKPOINT OUTPUT_ROOT GPU_LIST}"

if [[ -e "${output_root}" ]]; then
    echo "Refusing existing validated-run output: ${output_root}" >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
fi

IFS=',' read -r -a gpu_indices <<<"${gpu_list}"
for gpu_index in "${gpu_indices[@]}"; do
    gpu_memory="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${gpu_index}"
    )"
    gpu_uuid="$(
        nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu_index}"
    )"
    if (( gpu_memory >= 500 )); then
        echo "GPU ${gpu_index} is no longer idle (${gpu_memory} MiB)" >&2
        exit 3
    fi
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
        | grep -Fxq "${gpu_uuid}"; then
        echo "GPU ${gpu_index} acquired a compute process before launch" >&2
        exit 3
    fi
done

mkdir -p "${output_root}"
printf 'checkpoint=%s\ngpu_list=%s\nstarted_at=%s\n' \
    "$(realpath "${checkpoint}")" "${gpu_list}" "$(date --iso-8601=seconds)" \
    >"${output_root}/validated_launch_identity.txt"

GPU_INDEX="${gpu_indices[0]}" \
    "${script_dir}/smoke_walk_v7_full_reference.sh" \
    "${checkpoint}" "${output_root}/smoke"

GPU_LIST="${gpu_list}" \
    "${script_dir}/run_walk_v7_full_reference_1000.sh" \
    "${checkpoint}" "${output_root}/train"
