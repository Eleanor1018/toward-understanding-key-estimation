# Toward Understanding Key Estimation

Research reproduction code for Unitree G1 29-DOF locomotion with Isaac Sim
4.5, PyTorch 2.5.1, and CUDA 12.1. The repository contains the environment,
PPO training and evaluation paths, distributed launchers, checkpoint contract
migrations, and regression tests used by the experiments.

The current experimental branch is `experiment/future-reference-v8`. It adds a
phase-visible command and a compact three-frame future motion reference while
preserving the original 93-D proprioceptive observation and 29-D action.

## Local assets

Third-party robot and motion-data binaries are intentionally not redistributed
by this public repository. Before starting Isaac Sim, provide the files listed
in:

- [`assets/g1_29dof/README.md`](assets/g1_29dof/README.md)
- [`assets/motions/README.md`](assets/motions/README.md)

Runtime loading validates the motion archive hashes. Generated checkpoints,
evaluation reports, videos, and logs are Git-ignored.

## Verification

With the local assets installed:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The minimal environment/PPO contract is exercised with:

```bash
python train.py --headless --num-envs 1 --iterations 2 --rollout-steps 4
```

Operational launchers and experiment notes are indexed in
[`docs/README.md`](docs/README.md).

The V8 launchers require `V8_ISAAC_ROOT` to point at an Isaac Sim 4.5
installation before they are run. Historical launchers retain the absolute
paths used by their original experiments.
