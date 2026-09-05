#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
resume_checkpoint="${1:-logs/g1_stage1_4096_iter200_to_2000_seed46/checkpoint_01000.pt}"
log_dir="${2:-logs/g1_walk_estimation_from1000_plus2000_seed47}"
additional_iterations=2000
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "Walking DDP training requires at least two GPUs" >&2
    exit 2
fi
if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Missing resume checkpoint: ${resume_checkpoint}" >&2
    exit 2
fi
if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
    echo "Refusing to overwrite checkpoints in ${log_dir}" >&2
    exit 2
fi

checkpoint_iteration="$(
    /data/nora/isaacsim-4.5.0/env/bin/python - "${resume_checkpoint}" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["iteration"]))
PY
)"
end_iteration=$((checkpoint_iteration + additional_iterations))
evaluation_checkpoints=("${resume_checkpoint}")
for offset in 250 500 750 1000 1250 1500 1750 2000; do
    evaluation_checkpoints+=(
        "${log_dir}/checkpoint_$(printf '%05d' "$((checkpoint_iteration + offset))").pt"
    )
done

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

echo "Warm-starting walking phase at $((checkpoint_iteration + 1)); target=${end_iteration}"
torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations "${end_iteration}" \
    --rollout-steps 24 \
    --learning-rate 0.0001 \
    --policy-learning-rate 0.00000625 \
    --minimum-policy-learning-rate 0.000001 \
    --target-kl 0.01 \
    --max-post-update-kl 0.02 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0002 \
    --final-entropy-coef 0.00002 \
    --prediction-loss-coef 0.1 \
    --estimation-loss-coef 0.15 \
    --initial-action-std 0.10 \
    --resume-action-std 0.10 \
    --final-action-std 0.06 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile walk \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_walk \
    --seed 47 \
    --save-interval 250 \
    --resume "${resume_checkpoint}" \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 1024 \
    --steps 1000 \
    --command-profile walk \
    --reward-profile p1_walk \
    --include-zero-baseline \
    --checkpoints "${evaluation_checkpoints[@]}" \
    --output "${log_dir}/evaluation_${end_iteration}.json" \
    2>&1 | tee "${log_dir}/evaluation.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${log_dir}" "${end_iteration}" <<'PY'
import json
import math
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
end_iteration = int(sys.argv[2])
report = json.loads(
    (log_dir / f"evaluation_{end_iteration}.json").read_text()
)
candidates = report["results"][2:]

def metric(result, name, fallback):
    value = result[name]
    return fallback if value is None else value

def score(result):
    correlation = metric(result, "velocity_command_correlation", [None])[0]
    correlation_for_score = -1.0 if correlation is None else correlation
    return (
        5.0 * (1.0 - result["survival_fraction"])
        + 2.0 * result["planar_velocity_vector_rmse"]
        + result["yaw_rate_rmse"]
        + 0.5 * result["explicit_velocity_vector_rmse"]
        + 0.5 * (1.0 - correlation_for_score)
        + result["mean_absolute_clipped_action"]
        + 5.0 * result["raw_action_saturation_fraction"]
    )

def passes_gate(result):
    correlation = result["velocity_command_correlation"][0]
    forward = result["command_bin_survival"]["forward"]["survival_fraction"]
    backward = result["command_bin_survival"]["backward"]["survival_fraction"]
    stand = result["command_bin_survival"]["stand"]["survival_fraction"]
    forward_tracking = result["command_bin_tracking"]["forward"]
    backward_tracking = result["command_bin_tracking"]["backward"]
    required = (
        result["planar_velocity_vector_rmse"],
        result["yaw_rate_rmse"],
        result["explicit_velocity_vector_rmse"],
        result["mean_absolute_clipped_action"],
        result["raw_action_saturation_fraction"],
        correlation,
        forward_tracking["sagittal_velocity_rmse"],
        forward_tracking["mean_velocity_ratio"],
        forward_tracking["correct_direction_fraction"],
        backward_tracking["sagittal_velocity_rmse"],
        backward_tracking["mean_velocity_ratio"],
        backward_tracking["correct_direction_fraction"],
    )
    return (
        all(value is not None and math.isfinite(value) for value in required)
        and result["survival_fraction"] >= 0.95
        and forward is not None and forward >= 0.95
        and backward is not None and backward >= 0.95
        and stand is not None and stand >= 0.95
        and result["planar_velocity_vector_rmse"] <= 0.20
        and result["yaw_rate_rmse"] <= 0.25
        and result["explicit_velocity_vector_rmse"] <= 0.25
        and result["raw_action_saturation_fraction"] < 0.01
        and result["mean_absolute_clipped_action"] < 0.35
        and result["mean_upright"] >= 0.95
        and correlation >= 0.70
        and forward_tracking["sagittal_velocity_rmse"] <= 0.20
        and 0.50 <= forward_tracking["mean_velocity_ratio"] <= 1.50
        and forward_tracking["correct_direction_fraction"] >= 0.70
        and backward_tracking["sagittal_velocity_rmse"] <= 0.20
        and 0.50 <= backward_tracking["mean_velocity_ratio"] <= 1.50
        and backward_tracking["correct_direction_fraction"] >= 0.70
    )

passing_candidates = [result for result in candidates if passes_gate(result)]
passed = bool(passing_candidates)
best = min(passing_candidates or candidates, key=score)
selection = {
    "passed": passed,
    "best_checkpoint": best["checkpoint"],
    "best_score": score(best),
    "criteria": {
        "survival_fraction_min": 0.95,
        "planar_velocity_vector_rmse_max": 0.20,
        "yaw_rate_rmse_max": 0.25,
        "explicit_velocity_vector_rmse_max": 0.25,
        "raw_action_saturation_fraction_max": 0.01,
        "mean_absolute_clipped_action_max": 0.35,
        "velocity_command_correlation_min": 0.70,
        "directional_sagittal_velocity_rmse_max": 0.20,
        "directional_mean_velocity_ratio_range": [0.50, 1.50],
        "directional_correct_fraction_min": 0.70,
        "mean_upright_min": 0.95,
    },
    "best": best,
}
(log_dir / "walk_selection.json").write_text(
    json.dumps(selection, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(selection, indent=2, sort_keys=True))
print("WALK_STAGE_PASS" if passed else "WALK_STAGE_HOLD")
raise SystemExit(0 if passed else 3)
PY
