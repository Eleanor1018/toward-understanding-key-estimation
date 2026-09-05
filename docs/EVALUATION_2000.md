# G1 robust-locomotion evaluation after 2000 iterations

Date: 2026-09-03 (Asia/Singapore)

## Result

The 2000-iteration run completed successfully at the systems level, but the
final policy does **not** pass a normal robust-walking criterion.  Checkpoint
500 is the most stable of the evaluated learned policies: all 1024 environments
remain upright for the full 1000-step horizon and planar velocity tracking is
substantially better than the zero-action baseline.  It nevertheless has severe
yaw-rate error and frequent action saturation.  By checkpoint 2000 the policy
has regressed: virtually every action is saturated, yaw tracking is worse,
velocity-estimation error has doubled, and 167 early terminations occur.

Evaluation used 1024 environments for 1000 policy steps, deterministic actions,
seed 123, and domain randomization disabled.

| Policy | Planar velocity RMSE | Yaw-rate RMSE | Explicit velocity RMSE | Early terminations | Timeouts | Action saturation | Mean height | Minimum height |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero action | 0.7145 | 0.0066 | n/a | 7168 | 0 | 0.00% | 0.7343 | 0.4587 |
| Checkpoint 50 | 0.6889 | 0.4220 | 0.2518 | 6144 | 0 | 0.00% | 0.7310 | 0.4500 |
| Checkpoint 500 | **0.3654** | 4.6583 | **0.2702** | **0** | 1024 | 58.85% | 0.7850 | **0.7415** |
| Checkpoint 2000 | 0.5011 | 6.4373 | 0.5169 | 167 | 869 | 99.98% | **0.7897** | 0.4501 |

`Timeouts=1024` for checkpoint 500 means every environment reached the full
evaluation horizon rather than falling early.  The zero-action and checkpoint
50 policies repeatedly terminate and reset, so their apparently competitive
mean reward is not evidence of locomotion quality.

## Interpretation

- Checkpoint 500 learns stable upright motion and useful planar command
  tracking, but the very large yaw error means it is not normal commanded
  locomotion.
- The final checkpoint exhibits policy-output divergence: 99.98% saturation is
  incompatible with a healthy joint-target controller and explains why longer
  training does not improve the task.
- Mean reward is not a reliable selector in the current reward design: the
  zero-action baseline receives more reward per step than either learned policy.
  Survival/posture terms therefore dominate the objective too strongly relative
  to command tracking and action regularization.
- The next experiment should resume from checkpoint 500, strengthen yaw and
  command-tracking terms, add an explicit action/action-rate penalty, and reduce
  update aggressiveness or stop early when saturation and yaw RMSE rise.

The raw machine-readable report is
`logs/g1_paper_4096_gpu1_2000_1khz_reset/evaluation_2000.json`.
