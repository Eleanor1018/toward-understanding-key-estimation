# Changes from the initial version

## Baseline

The comparison baseline is commit
`86419152d9247b3154562e520cbfe9e289d685a9` (`Initial G1 key estimation
training pipeline`). It is the repository's only commit and currently matches
both `main` and `origin/main`.

The initial commit contained only `.gitignore`, `config.py`, `deploy.py`,
`policy.py`, `ppo.py`, and `train.py`. The training entry point referenced a
`g1_env` module that did not yet exist, so the environment could not be created.

## File-level changes

The original `config.py`, `policy.py`, `ppo.py`, and `train.py` now support the
phase-visible and future-reference experiment contracts while retaining legacy
profiles. The empty `deploy.py` remains unchanged. Generated outputs and
third-party binary assets are excluded from Git.

New implementation files:

- `g1_env.py`: Isaac Lab environment and Gym registration.
- `normalization.py`: shared train/evaluation input normalization.
- `evaluate.py`: deterministic checkpoint and zero-action evaluation.
- `train_ddp.py`: multi-GPU distributed PPO training.
- `tests/test_training_math.py`: training-math regression tests.

New supporting material:

- `assets/`: provenance and hash manifests for local-only robot and motion data.
- `scripts/`: current and historical experiment launchers.
- `docs/`: environment contract, experiment log, evaluation report, and this
  comparison.

Generated checkpoints and metrics remain under the Git-ignored `logs/`
directory.

## Functional changes

### Runnable Unitree G1 environment

- Registered `Unitree-G1-29dof-KeyEstimation-v0` with Isaac Lab's
  `DirectRLEnv` API.
- Added the no-hand Unitree G1 with 29 joint-position targets and local USD
  asset loading, avoiding a runtime Nucleus dependency.
- Implemented the fixed observation contract: `history [N,50,93]`,
  `obs [N,93]`, `command [N,3]`, `privileged [N,103]`, and
  `explicit_target [N,3]`.
- Added a 1 kHz physics/PD loop, 100 Hz policy loop, reset and termination
  logic, command profiles, locomotion rewards, contact sensing, observation
  latency/noise, and controllable domain randomization.

### Stable network inputs and evaluation

- Added fixed physical normalization shared by training and evaluation.
- Centered joint positions on the default pose and scaled velocities,
  commands, explicit velocity, and privileged torques without changing field
  meanings or dimensions.
- Added deterministic first-episode evaluation with fixed commands,
  zero-action comparison, survival, tracking, action saturation, correlation,
  and command-bin metrics written to JSON.

### PPO and rollout correctness

- Replaced the unbounded Normal action output with a tanh-squashed Gaussian,
  including stable pre-tanh PPO likelihoods and bounded standard-deviation
  scheduling.
- Split true termination from timeout handling in GAE. Successful timeouts
  bootstrap from the captured pre-reset critic state while every episode
  boundary cuts the advantage trace.
- Added command-sampling validation, return standardization, separate gradient
  clipping, policy-specific learning rate, KL early stopping, and transactional
  rollback when a policy update exceeds the post-update KL limit.
- Replaced under-conditioned next-observation prediction with detached current
  normalized-observation reconstruction.
- Versioned action-distribution, normalization, and auxiliary-objective
  contracts inside checkpoints and expanded numerical training diagnostics.

### Multi-GPU operation

- Added `torchrun`/NCCL training across multiple GPUs.
- Added balanced environment sharding, including totals that are not evenly
  divisible by rank count, with sample-weighted gradient and metric reduction.
- Added strict resume compatibility checks, parameter synchronization checks,
  and rank-zero-only checkpoint output.

### Experiment workflow

- Added smoke, balance, staged locomotion, continuation, GPU-wait, evaluation,
  and quality-gate launchers under `scripts/`.
- Kept legacy P0/P1 and paper launchers for reproducibility but disabled them
  by default after later input-contract and stability fixes.
- Recorded the environment contract, system bring-up, failed first 2000-run
  diagnosis, and subsequent corrections under `docs/`.

## Later locomotion experiments

- Added phase-visible V6 commands, reference-state initialization, V7 full-body
  reference rewards, and strict checkpoint migrations.
- Added V8's optional three-frame, 63-D future-reference input to actor and
  critic while leaving the 93-D proprioceptive observation unchanged.
- Added deterministic MuJoCo transfer diagnostics with matching policy IO,
  control rate, PD gains, effort limits, and soft joint limits.

## Compatibility and verification

Legacy profiles retain their original three-value physical command semantics.
Phase-visible profiles append sine/cosine phase values, and only V8 adds the
separate future-reference tensor. These changes are versioned and validated in
checkpoints rather than silently reinterpreting an existing input.

Current verification:

- All 28 Shell/Python launchers pass syntax checks.
- All 81 local unit tests pass with the documented local assets installed.
- `git diff --check` reports no whitespace errors.
