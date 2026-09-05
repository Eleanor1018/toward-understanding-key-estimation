# V8 future-reference experiment

## Contract

V8 keeps the 93-D proprioceptive observation, 50-step history, 103-D
privileged state, and 29-D joint-position action. It adds three future reference
frames at 1/30, 2/30, and 3/30 seconds. Each frame has 21 normalized values:
12 leg joint targets, two ankle positions (six values), two contact targets,
and root height. The resulting 63-D tensor is provided explicitly to the actor
and critic without changing the meaning of any legacy input.

`v8_checkpoint.py` migrates a compatible V7 checkpoint by inserting zero
columns for the new input into the first actor and critic layers. It preserves
the learned policy exactly at migration time, clears Adam state, records source
provenance, and validates all checkpoint contracts.

## V8a curriculum

The first tested recipe schedules 5,000 new updates. Task mixing stays at zero
for the first 1,000 updates, reaches 0.9 at update 3,000 and 1.0 at update 3,500.
Reference-state initialization (RSI) stays at 1.0 for 500 updates and reaches
zero at update 3,500. The policy LR is `1e-6`; critic and decoder LRs are
`5e-5`. Domain randomization is disabled during this locomotion bring-up.

Smoke tests passed at 1, 64, and 512 environments. The 512-environment smoke
reported finite PPO/GAE/backward metrics, zero normalized-input clipping, and a
raw action saturation fraction below 0.4%.

## Intermediate gate

The run was warm-started from V7 iteration 4,900 and deliberately stopped after
500 new updates at global iteration 5,400. Evaluation uses 512 deterministic
standing-start environments for 1,000 policy steps with randomization disabled.

| Metric | Update 250 | Update 500 |
|---|---:|---:|
| Survival fraction | 0.0977 | 0.4727 |
| Survival-adjusted forward velocity | 0.0513 m/s | 0.0178 m/s |
| Survival-adjusted velocity ratio | 0.1268 | 0.0398 |
| Force-threshold double support | 0.9762 | 0.9894 |
| Force-threshold single support | 0.0133 | 0.0054 |
| Valid landings | 0 | 0 |
| Raw action saturation | 0.1201% | 0.0009% |
| Soft-limit target clipping | 0.0734% | 0.1563% |

At update 500, the actor's future-input first-layer weights had reached only
0.66% of the mean absolute magnitude of its legacy input weights, while the
critic's ratio was 19.4%. The critic learned to use the target far faster than
the policy. Stability recovered, but gait formation and forward progress did
not. The run therefore failed its locomotion gate and was not extended with the
same recipe. Checkpoints and evaluation JSON remain local generated artifacts,
not repository content.

## MuJoCo transfer check

The V7 iteration-5,000 policy was also replayed deterministically in the MuJoCo
Menagerie G1 model with matching policy IO, 100 Hz control, 1 kHz physics, PD
gains, effort limits, and 90% soft joint limits. It remained upright for 10
seconds but moved only 0.069 m, spent 98.6% of samples in double support, and
had no valid landings. MuJoCo friction/contact and solver dynamics differ from
Isaac, so this is a qualitative transfer diagnostic rather than a numerically
equivalent evaluation.
