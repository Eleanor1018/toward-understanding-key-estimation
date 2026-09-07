# Unitree G1 29-DOF environment contract

The registered task is `Unitree-G1-29dof-KeyEstimation-v0`. It uses Unitree's
no-hand G1 29-DOF USD and controls the joints in the `G1_JOINT_NAMES` order in
`g1_env.py`.

The policy action is a normalized 29-value vector. The environment clips it to
`[-1, 1]`, multiplies it by `0.25` radians, adds the robot's default pose, and
then clamps the result to the USD soft position limits. The resulting 29 values
are joint position targets.

The policy observation is exactly 93 values:

| Slice | Size | Quantity | Frame / convention |
|---|---:|---|---|
| `0:3` | 3 | projected gravity | base frame |
| `3:6` | 3 | base angular velocity | base frame |
| `6:35` | 29 | joint position | `G1_JOINT_NAMES` order, radians |
| `35:64` | 29 | joint velocity | `G1_JOINT_NAMES` order, rad/s |
| `64:93` | 29 | previous action | normalized and clipped |

`history` is a chronological rolling buffer of 50 policy observations with
shape `[N, 50, 93]`. At reset, all 50 entries are filled with the reset
observation rather than zero padding. During training the actor stream receives
small Gaussian proprioceptive noise and a uniformly sampled 0--2 policy-step
delay. These change measurement fidelity, not the ordering or meaning of the
93 fields. The privileged stream and `explicit_target` remain exact.

Legacy profiles expose desired base-frame `(vx, vy, yaw_rate)` as `command`
with shape `[N, 3]`. The phase-visible V6 profiles append
`(sin(2*pi*phase), cos(2*pi*phase))`, producing `[N, 5]`; the first three fields
retain exactly the same physical meaning.
`explicit_target` is the simulator's true base-frame linear velocity with shape
`[N, 3]`.

The privileged observation is exactly 103 simulator values; no values are
padding:

| Slice | Size | Quantity | Frame / convention |
|---|---:|---|---|
| `0:3` | 3 | base position relative to its environment origin | world axes |
| `3:7` | 4 | base orientation quaternion | world frame, scalar-first `(w,x,y,z)` |
| `7:10` | 3 | base linear velocity | base frame |
| `10:13` | 3 | base angular velocity | base frame |
| `13:16` | 3 | projected gravity | base frame |
| `16:45` | 29 | joint position | `G1_JOINT_NAMES` order |
| `45:74` | 29 | joint velocity | `G1_JOINT_NAMES` order |
| `74:103` | 29 | applied joint torque | `G1_JOINT_NAMES` order |

## Paper-style locomotion objective

The policy runs at 100 Hz (`step_dt=0.01`) and Isaac Sim's implicit PD targets
are solved at 1 kHz (`sim.dt=0.001`, `decimation=10`), matching the timing in
arXiv:2403.05868. Episodes last 10 seconds. Desired commands are constant for
one episode and sampled as `vx in [-1.2, 1.2] m/s`, `vy in [-0.6, 0.6] m/s`,
and `yaw_rate in [-1, 1] rad/s`; ten percent are stand commands.

The reward directly implements equations (3)--(12):

| Group | Terms |
|---|---|
| base tracking | 3-D linear velocity, 3-D angular velocity, upright base, 0.80 m base height |
| gait | stance-foot velocity and swing-foot contact force |
| smoothness / energy | foot-force change, torque change, joint-velocity change, cost of transport |

The paper's printed Gaussian equation omits the leading minus sign even though
its text defines a bell-shaped error reward. The implementation therefore uses
`alpha * exp(-(x/sigma)^2)`; the generalized Cauchy terms use
`alpha / ((abs(x)/sigma)^(2*beta) + 1)`. Rewards are summed per policy step,
not multiplied by `dt`, so episode-return scale remains comparable to the
paper's approximately 1300-point curves.

The original Wukong-IV method includes a gait signal in a larger command
vector. Legacy profiles keep the three-value command and generate a 0.8-second
left/right gait clock internally. V6 instead exposes its cyclic reference phase
to the Actor and Critic while keeping physical command handling three-dimensional.

## Domain randomization and resets

At the beginning of each training episode the environment samples the paper's
Appendix Table II ranges:

| Quantity | Range |
|---|---:|
| pelvis COM offset | `[-0.15, 0.15] m` per axis |
| pelvis payload mass | `[-2.0, 12.5] kg` |
| rigid-body friction | `[0.25, 1.25]` |
| motor effort strength | `[0.8, 1.2]` |
| stiffness (`Kp`) factor | `[0.9, 1.1]` |
| damping (`Kd`) factor | `[0.9, 1.1]` |
| observation latency | `0, 1, or 2` policy steps |

Initial root pose/velocity and joint state also receive small perturbations.
Evaluation explicitly disables all physics randomization, latency, and noise.
Termination is a 10-second time limit, invalid base height, or a tipped base.

This first 2000-iteration experiment intentionally uses flat ground to answer
the requested question, "can the G1 acquire normal walking?" The paper's
easy-to-hard terrain curriculum and height-map target are a later robustness
stage; adding them now would change the current network inputs and would not be
a faithful silent modification.

## Corrected locomotion-training profile

The historical `paper` and `p1_dense` reward profiles remain available for
reproducing earlier checkpoints. New locomotion runs use `p1_stable` and a
versioned network-boundary normalizer. The environment still returns physical
values with exactly the same shapes and slice meanings documented above.

Before entering the networks, observations are centered/scaled as follows:

| Quantity | Network transform |
|---|---|
| projected gravity | unchanged |
| base angular velocity | divide by `(4, 4, 1) rad/s` |
| joint position | `(q - default_q) / 0.25 rad` |
| joint velocity | divide by `10 rad/s` |
| previous action | unchanged |
| command / explicit velocity | divide by `(1.2, 0.6, 1.0)` |
| privileged torque | divide by the corresponding actuator effort limit |

Dynamic normalized quantities are clipped to `[-5, 5]`; normalized torques
are clipped to `[-1.5, 1.5]`. Checkpoints record
`input_normalization_type=g1_fixed_physical_scales_v1`, and evaluation applies
the same transform before inference and reverses it for explicit-velocity RMSE.

`p1_stable` separates planar velocity and yaw tracking, removes the hidden
left/right gait clock, and replaces it with a phase-free stance-foot slip cost.
It also includes vertical velocity, roll/pitch angular velocity, joint posture,
action, action-rate, a strong barrier above `|action|=0.8`, and terminal costs.
The three command profiles are:

1. `stage1`: 20% stand and 80% forward/backward commands.
2. `forward_walk`: 100% forward commands in `vx in [0.25, 0.55] m/s`.
3. `walk`: 10% stand and 90% forward/backward commands in
   `|vx| in [0.25, 0.65] m/s`.
4. `stage2`: stand, sagittal, lateral, yaw-only, and mixed commands.
5. `stage3`: the same mixture over the complete command range.

The `p1_walk` reward profile retains the stability and action protections from
`p1_stable`, increases the planar tracking reward, and adds a bounded signed
progress term for nonzero planar commands. It is a clean sagittal-locomotion
stage, not the later domain-randomized robustness stage.

`p1_walk_gait` adds a phase-free biped gait term on top of `p1_walk`. It gives a
small dense reward while exactly one foot is in stance and a bounded landing
bonus after a meaningful swing duration. The term is disabled for stand
commands and does not require adding a gait clock to the observation.

`p1_walk_gait_v2` is an escape-stage objective for policies trapped in stable
standing. It strengthens planar tracking and signed progress, penalizes failure
to reach 0.15 m/s in the commanded direction, and increases the phase-free
single-stance and landing signals. Existing orientation, height, slip, action,
saturation, and termination protections remain active.

`p1_walk_stable_v3` is the stabilization objective used after motion has been
discovered. Command tracking uses yaw-heading horizontal velocity rather than
the fully tilted body frame, and positive locomotion rewards are gated by
uprightness and base height. It terminates at roughly 45 degrees of tilt,
base height below 0.55 m, or ground contact by any rigid body other than the two
feet. Terminal, action, action-rate, saturation, slip, torque, impact, and
joint-velocity costs are strengthened to prevent short forward dives, crawling,
or arm-supported motion from outscoring sustained walking. The illegal-contact
sensor retains 11 physics samples, covering the full 10 ms interval between
policy steps.

Evaluation holds one command fixed per environment after reset and scores only
the first episode. This prevents a policy that falls early from receiving a
different sequence of easier or harder commands than another checkpoint.
Terminal physical state is captured before Isaac Lab's automatic reset, so
tracking and minimum-height metrics include the failure transition. Successful
time limits use the same pre-reset critic state for value bootstrap while still
cutting the GAE trace between episodes. Evaluation also reports
`survival_adjusted_*` tracking metrics over the requested fixed horizon; after a
termination, the remaining steps are scored as zero velocity instead of being
dropped. These metrics avoid favoring policies whose difficult episodes simply
end early.

`p1_walk_stable_v4` keeps all V3 frame, stability-gate, early-termination, and
regularization rules. It raises signed progress from 0.5 to 1.0 and the
below-0.15 m/s stall penalty from 0.15 to 1.0. This profile is a controlled
follow-up for the observed V3 failure mode in which survival reaches 100% by
converging to near-zero forward velocity.

`p1_walk_stable_v5` keeps V4 and adds a phase-free contact objective compatible
with the fixed three-value command. A short, physically plausible single-support
window receives `+0.03` per step. A force-validated landing after at least 0.12 s
of swing anchors the internal self-paced gait; subsequent opposite-foot landings
0.12-0.70 s apart receive `+1`, while same-foot repeats receive `-0.5`. Flight
retains V3's penalty, all positive gait rewards are stability-gated, and fall
transitions receive no gait bonus. Evaluation reports support fractions, landing
rate, fixed-horizon alternating rate, left/right balance, and
alternating/repeated landing fractions so this objective cannot be judged from
velocity alone.

`p1_walk_stable_v5_imitation` is the treatment profile on the
`experiment/deepmimic-lite` branch. It inherits V5 unchanged and adds a cyclic
G1 walking-reference pose kernel bounded by `0.05` and joint-velocity kernel
bounded by `0.01`. Both use the original DeepMimic exponential scales and
MimicKit's G1 joint weights, and both are gated off for standing commands,
unstable states, and true termination transitions. The policy does not receive
reference phase or future targets, so this is a low-weight DeepMimic-inspired
experiment rather than a complete DeepMimic implementation. Per-joint motion
is scaled into a `+/-0.22 rad` envelope around the simulator default pose so it
is reachable under the unchanged action contract.

`p1_walk_stable_v6_phase_rsi` and
`p1_walk_stable_v6_phase_rsi_imitation` are the matched control and treatment
profiles on `experiment/deepmimic-phase-visible`. Both expose a 5-D command and
initialize 70% of training resets from a randomly phased, retargeted reference
joint pose and velocity; the remaining 30% start from standing. They retain the
simulator's safe default root state. Evaluation sets the RSI probability to zero.
Only the treatment adds imitation reward, with a total coefficient scheduled
from 0.15 to 0.03 and split 5:1 between joint pose and velocity.

The stable-walk canary launcher defaults to 512 environments per selected GPU;
`GPU_LIST` controls the devices and `NUM_ENVS` can override the total when it is
divisible by the number of selected GPUs. `TRAIN_SEED` selects the environment
and policy sampling seed for independent repeats.
`EVAL_REWARD_PROFILE` can score a treatment with baseline V5; the paired
DeepMimic-lite launcher uses this to keep returns comparable between arms.
