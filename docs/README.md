# Project documentation

The repository keeps generated outputs, robot assets, operational scripts, and
long-form notes in dedicated directories:

| Directory | Contents |
|---|---|
| `assets/` | Local third-party asset manifests and provenance; binaries are Git-ignored |
| `docs/` | Environment contract, experiment history, and evaluation reports |
| `logs/` | Generated checkpoints, metrics, and console logs (Git-ignored) |
| `scripts/` | Training, evaluation, and GPU-wait launchers |
| `tests/` | Python unit tests |

## Documents

- [Environment contract](ENVIRONMENT.md)
- [2000-iteration evaluation](EVALUATION_2000.md)
- [Changes from the initial version](CHANGES_FROM_INITIAL.md)
- [V8 future-reference experiment](FUTURE_REFERENCE_V8_EXPERIMENT.md)

Machine-specific historical logs are kept locally rather than published; they
contain internal host aliases and absolute filesystem paths.

## Launchers

Current launchers:

- `scripts/run_balance_stage0.sh`
- `scripts/run_p1_fixed_stage1.sh`
- `scripts/run_stage1_to_2000.sh`
- `scripts/run_walk_estimation_2000.sh`
- `scripts/run_walk_gait_remaining_1500.sh`
- `scripts/run_walk_v2_canary.sh`
- `scripts/run_walk_v2_to_10000.sh`
- `scripts/run_walk_stable_v3_canary.sh`
- `scripts/run_walk_v5_imitation_ab.sh`
- `scripts/run_walk_v6_phase_rsi_ab.sh`
- `scripts/evaluate_phase_rsi_ab.py`
- `scripts/smoke_walk_v7_full_reference.sh`
- `scripts/run_walk_v7_full_reference_1000.sh`
- `scripts/launch_walk_v7_validated.sh`
- `scripts/smoke_walk_v8_future_reference.sh`
- `scripts/run_walk_v8_future_reference_to_5400.sh`
- `scripts/resume_walk_v8_future_reference.sh`
- `scripts/launch_walk_v8_validated.sh`
- `scripts/smoke_walk_v5_profiles.sh`
- `scripts/wait_and_run_walk_estimation_2000.sh`
- `scripts/wait_and_run_p1_fixed_stage1.sh`

Historical launchers are retained for reproducibility. The `run_p0_500.sh`,
`run_p1_500.sh`, `run_paper_2000.sh`, and `run_paper_ddp.sh` launchers require
`ALLOW_LEGACY_EXPERIMENT=1`; their corresponding waiters are historical too.

Run every launcher from any working directory using its repository-relative
path, for example `scripts/run_balance_stage0.sh`. Each launcher resolves the
repository root from its own location before starting Python.
`run_walk_stable_v3_canary.sh` supports the versioned stable-walk profiles through
`REWARD_PROFILE`, selects devices with `GPU_LIST`, defaults to 512 environments
per selected GPU, and accepts `TRAIN_SEED` for independent repeats.
