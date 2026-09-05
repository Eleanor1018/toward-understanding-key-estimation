#!/usr/bin/env bash
set -euo pipefail

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
source /data/nora/isaacsim-4.5.0/activate_isaacsim.sh >/dev/null
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
log_dir="${1:-logs/g1_balance_stage0_4096_200_seed46}"
IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
world_size="${#gpu_indices[@]}"

if (( world_size < 2 )); then
    echo "Balance Stage 0 requires at least two GPUs" >&2
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
    --iterations 200 \
    --rollout-steps 24 \
    --learning-rate 0.0002 \
    --policy-learning-rate 0.00005 \
    --minimum-policy-learning-rate 0.000001 \
    --target-kl 0.02 \
    --max-post-update-kl 0.03 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.0001 \
    --final-entropy-coef 0.0 \
    --prediction-loss-coef 0.5 \
    --estimation-loss-coef 0.0 \
    --initial-action-std 0.20 \
    --final-action-std 0.10 \
    --min-action-std 0.05 \
    --max-action-std 0.60 \
    --command-profile stand \
    --command-scale 1.0 \
    --domain-randomization-scale 0.0 \
    --reward-profile p1_stable \
    --seed 46 \
    --save-interval 25 \
    --log-dir "${log_dir}" \
    2>&1 | tee "${log_dir}/train_ddp_${world_size}gpu.log"

export CUDA_VISIBLE_DEVICES="${gpu_indices[0]}"
python evaluate.py \
    --headless \
    --device cuda:0 \
    --num-envs 512 \
    --steps 1000 \
    --command-profile stand \
    --reward-profile p1_stable \
    --include-zero-baseline \
    --checkpoints \
        "${log_dir}/checkpoint_00050.pt" \
        "${log_dir}/checkpoint_00100.pt" \
        "${log_dir}/checkpoint_00150.pt" \
        "${log_dir}/checkpoint_00200.pt" \
    --output "${log_dir}/evaluation_200.json" \
    2>&1 | tee "${log_dir}/evaluation.log"

/data/nora/isaacsim-4.5.0/env/bin/python - "${log_dir}" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
results = json.loads((log_dir / "evaluation_200.json").read_text())["results"]

def score(result):
    return (
        4.0 * (1.0 - result["survival_fraction"])
        + result["planar_velocity_vector_rmse"]
        + result["yaw_rate_rmse"]
        + result["mean_absolute_clipped_action"]
        + 5.0 * result["raw_action_saturation_fraction"]
    )

best = min(results[1:], key=score)
passed = (
    best["survival_fraction"] >= 0.70
    and best["raw_action_saturation_fraction"] < 0.05
    and best["mean_absolute_clipped_action"] < 0.60
)
selection = {
    "passed": passed,
    "best_checkpoint": best["checkpoint"],
    "best_score": score(best),
    "baseline": results[0],
    "best": best,
}
(log_dir / "stage0_selection.json").write_text(
    json.dumps(selection, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(selection, indent=2, sort_keys=True))
print("BALANCE_STAGE0_PASS" if passed else "BALANCE_STAGE0_HOLD")
raise SystemExit(0 if passed else 3)
PY
