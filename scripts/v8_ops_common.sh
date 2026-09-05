#!/usr/bin/env bash

# Shared validation helpers for V8 launch scripts. This file is sourced.

V8_ISAAC_ROOT="${V8_ISAAC_ROOT:-${ISAACSIM_ROOT:-}}"
V8_PYTHON_BIN="${V8_PYTHON_BIN:-${V8_ISAAC_ROOT:+${V8_ISAAC_ROOT}/env/bin/python}}"
V8_IDLE_MEMORY_LIMIT_MIB="${V8_IDLE_MEMORY_LIMIT_MIB:-500}"

v8_die() {
    printf 'V8 operation error: %s\n' "$*" >&2
    exit 2
}

v8_require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( 10#${value} < 1 )); then
        v8_die "${name} must be a positive integer, got '${value}'"
    fi
}

v8_require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        v8_die "${name} must be a non-negative integer, got '${value}'"
    fi
}

v8_require_runtime() {
    [[ -n "${V8_ISAAC_ROOT}" ]] \
        || v8_die "set V8_ISAAC_ROOT to the Isaac Sim 4.5 installation"
    [[ -x "${V8_PYTHON_BIN}" ]] \
        || v8_die "missing Isaac Sim Python: ${V8_PYTHON_BIN}"
    [[ -f "${V8_ISAAC_ROOT}/activate_isaacsim.sh" ]] \
        || v8_die "missing Isaac Sim activation script"
    command -v nvidia-smi >/dev/null \
        || v8_die "nvidia-smi is required for idle-GPU validation"
    v8_require_positive_integer \
        V8_IDLE_MEMORY_LIMIT_MIB "${V8_IDLE_MEMORY_LIMIT_MIB}"
}

v8_activate_runtime() {
    # shellcheck source=/dev/null
    source "${V8_ISAAC_ROOT}/activate_isaacsim.sh" >/dev/null
}

v8_parse_gpu_list() {
    local gpu_list="$1"
    local gpu_index
    local -A seen=()

    if [[ ! "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
        v8_die "GPU_LIST must be comma-separated physical GPU indices"
    fi
    IFS=',' read -r -a V8_GPU_INDICES <<<"${gpu_list}"
    for gpu_index in "${V8_GPU_INDICES[@]}"; do
        if [[ -n "${seen[${gpu_index}]+present}" ]]; then
            v8_die "GPU_LIST contains duplicate physical index ${gpu_index}"
        fi
        seen[${gpu_index}]=1
    done
    V8_WORLD_SIZE="${#V8_GPU_INDICES[@]}"
    (( V8_WORLD_SIZE > 0 )) || v8_die "GPU_LIST must not be empty"
}

v8_require_idle_gpus() {
    local compute_uuids
    local gpu_index
    local gpu_memory
    local gpu_uuid

    compute_uuids="$(
        nvidia-smi --query-compute-apps=gpu_uuid \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    for gpu_index in "$@"; do
        if ! gpu_memory="$(
            nvidia-smi --query-gpu=memory.used \
                --format=csv,noheader,nounits -i "${gpu_index}" 2>/dev/null
        )"; then
            v8_die "GPU ${gpu_index} does not exist or cannot be queried"
        fi
        gpu_memory="${gpu_memory//[[:space:]]/}"
        [[ "${gpu_memory}" =~ ^[0-9]+$ ]] \
            || v8_die "GPU ${gpu_index} returned invalid memory usage"
        gpu_uuid="$(
            nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
                -i "${gpu_index}" 2>/dev/null
        )"
        gpu_uuid="${gpu_uuid//$'\r'/}"
        gpu_uuid="${gpu_uuid//$'\n'/}"
        [[ -n "${gpu_uuid}" ]] || v8_die "GPU ${gpu_index} has no UUID"

        if (( 10#${gpu_memory} >= 10#${V8_IDLE_MEMORY_LIMIT_MIB} )); then
            v8_die "GPU ${gpu_index} is not idle (${gpu_memory} MiB used)"
        fi
        if grep -Fqx -- "${gpu_uuid}" <<<"${compute_uuids}"; then
            v8_die "GPU ${gpu_index} has an active compute process"
        fi
    done
}

v8_sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

v8_resolve_existing_file() {
    local path="$1"
    [[ -f "${path}" ]] || v8_die "missing file: ${path}"
    realpath -- "${path}"
}

v8_resolve_new_path() {
    local path="$1"
    [[ ! -e "${path}" ]] || v8_die "refusing existing output: ${path}"
    realpath -m -- "${path}"
}
