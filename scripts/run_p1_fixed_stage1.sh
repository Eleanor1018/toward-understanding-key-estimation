#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
log_dir="${1:-logs/g1_p1_fixed_stage1_4096_75_seed44}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "Stage 1 DDP training requires at least two GPUs" >&2
    exit 2
fi
if compgen -G "${log_dir}/checkpoint_*.pt" >/dev/null; then
    echo "Refusing to overwrite checkpoints in ${log_dir}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
mkdir -p "${log_dir}"

torchrun --standalone --nproc_per_node="${world_size}" train_ddp.py \
    --distributed \
    --headless \
    --num-envs 4096 \
    --iterations 75 \
    --rollout-steps 24 \
    --learning-rate 0.0002 \
    --target-kl 0.02 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0002 \
    --final-entropy-coef 0.0 \
    --prediction-loss-coef 0.5 \
    --estimation-loss-coef 1.0 \
    --initial-action-std 0.3 \
    --final-action-std 0.15 \
    --min-action-std 0.05 \
    --max-action-std 0.6 \
    --command-profile stage1 \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_stable \
    --seed 44 \
    --save-interval 25 \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 512 \
    --steps 1000 \
    --command-profile stage1 \
    --command-scale 1.0 \
    --reward-profile p1_stable \
    --include-zero-baseline \
    --checkpoints \
        "${log_dir}/checkpoint_00025.pt" \
        "${log_dir}/checkpoint_00050.pt" \
        "${log_dir}/checkpoint_00075.pt" \
    --output "${log_dir}/evaluation_75.json" \
    2>&1 | tee "${log_dir}/evaluation.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${log_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
results = json.loads((log_dir / "evaluation_75.json").read_text())["results"]
baseline = results[0]

def score(result):
    return (
        result["planar_velocity_vector_rmse"]
        + result["yaw_rate_rmse"]
        + 2.0 * (1.0 - result["survival_fraction"])
        + 5.0 * result["raw_action_saturation_fraction"]
        + 0.2 * result["mean_absolute_clipped_action"]
    )

best = min(results[1:], key=score)
correlation = best["velocity_command_correlation"][0]
forward_survival = best["command_bin_survival"]["forward"]["survival_fraction"]
backward_survival = best["command_bin_survival"]["backward"]["survival_fraction"]
finite = all(
    math.isfinite(value)
    for value in (
        best["planar_velocity_vector_rmse"],
        best["yaw_rate_rmse"],
        best["raw_action_saturation_fraction"],
        best["mean_absolute_clipped_action"],
    )
)
passed = (
    finite
    and best["survival_fraction"] >= 0.70
    and forward_survival is not None
    and forward_survival >= 0.65
    and backward_survival is not None
    and backward_survival >= 0.65
    and best["planar_velocity_vector_rmse"]
    <= 0.80 * baseline["planar_velocity_vector_rmse"]
    and best["yaw_rate_rmse"] <= 0.25
    and best["raw_action_saturation_fraction"] < 0.01
    and best["mean_absolute_clipped_action"] < 0.35
    and correlation is not None
    and correlation >= 0.50
)
selection = {
    "passed": passed,
    "best_checkpoint": best["checkpoint"],
    "best_score": score(best),
    "baseline": baseline,
    "best": best,
}
(log_dir / "stage1_selection.json").write_text(
    json.dumps(selection, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(selection, indent=2, sort_keys=True))
print("P1_FIXED_STAGE1_PASS" if passed else "P1_FIXED_STAGE1_HOLD")
raise SystemExit(0 if passed else 3)
PY
