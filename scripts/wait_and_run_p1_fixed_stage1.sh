#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
log_dir="${P1_FIXED_LOG_DIR:-logs/g1_p1_fixed_stage1_4096_75_seed44}"
wait_log="${P1_FIXED_WAIT_LOG:-logs/g1_p1_fixed_gpu_wait.log}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"

mkdir -p "$(dirname "${wait_log}")"
printf '%s waiting for GPUs %s without terminating any process\n' \
    "$(date -Is)" "${gpu_list}" >> "${wait_log}"

idle_checks=0
while true; do
    busy=()
    for gpu_index in "${gpu_indices[@]}"; do
        if nvidia-smi -i "${gpu_index}" \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null \
            | grep -Eq '^[[:space:]]*[0-9]+'; then
            busy+=("${gpu_index}")
        fi
    done
    if (( ${#busy[@]} == 0 )); then
        idle_checks=$((idle_checks + 1))
        printf '%s idle confirmation %d/3\n' \
            "$(date -Is)" "${idle_checks}" >> "${wait_log}"
        if (( idle_checks >= 3 )); then
            break
        fi
    else
        idle_checks=0
        printf '%s still busy: %s\n' \
            "$(date -Is)" "${busy[*]}" >> "${wait_log}"
    fi
    sleep 30
done

printf '%s GPUs free; starting corrected Stage 1\n' \
    "$(date -Is)" >> "${wait_log}"
GPU_LIST="${gpu_list}" "${script_dir}/run_p1_fixed_stage1.sh" "${log_dir}"
printf '%s corrected Stage 1 and evaluation finished\n' \
    "$(date -Is)" >> "${wait_log}"
