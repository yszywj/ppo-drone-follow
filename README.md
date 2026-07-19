# Pegasus Iris Fast Trajectory-Follow PPO

This package provides a no-PX4 Pegasus Iris vector environment for dynamic
target following. It runs inside the Isaac Sim Docker image and uses the
Pegasus Simulator Iris asset plus a local CTBR body-rate backend. No file in the
Pegasus Simulator extension is modified.

## Control rates

- PPO policy and task reference: 5 Hz (`step_dt_sim_sec=0.2`)
- Isaac physics and local body-rate controller: 250 Hz (`physics_dt=0.004`)
- Policy action: normalized roll rate, pitch rate, yaw rate, collective thrust
- Helper/PPO ratio mix is retained for curriculum learning

The body-rate loop remains a permanent low-level stabilizer. The task-level
helper is temporary and is reduced through the training curriculum.

## Actor and critic

The deployed Actor has 32 causal inputs. They are measurable vehicle state and
current task references only:

- local NED position, velocity and acceleration
- roll, pitch, sine/cosine yaw and FRD body rates
- current following-point relative position
- target relative position
- desired following-point velocity
- three reserved following-point acceleration slots
- previous applied CTBR command

No future trajectory point, primitive ID, phase ID or trajectory progress is an
Actor input. In particular, a path sampled by the simulator cannot leak into
the policy that will be deployed against an unknown real target.

`actor_mask_target_acceleration=true` keeps the 32-value checkpoint contract
but fills the three following-point acceleration slots with zero. This is the
default in the masked bridge and vertical curriculum configs. Vehicle
acceleration remains available to the Actor; exact target/reference
acceleration remains available only to the training-time privileged Critic and
the temporary helper controller.

The Actor is:

```text
32 causal values -> 256-256-128 ELU frame encoder
frame embedding history -> GRU 128
small gated recurrent residual -> tanh-squashed Gaussian CTBR action
```

The frame encoder, GRU and action head keep their parameter names. A previous
47-input checkpoint from this project can therefore transfer all shared Actor
tensors; its removed future-reference encoder and gate are explicitly skipped.
For the bridge run, the recurrent residual starts at a small nonzero gate, is
held fixed for 15 updates, then becomes trainable. The inherited frame policy
uses a lower learning rate while the GRU branch uses a higher learning rate.

The asymmetric Critic remains 77-dimensional: 32 Actor values, 15 future
following-point values at 0.2, 0.4, 0.8, 1.2 and 1.6 seconds, and 30 privileged
values. The last group contains simulator-only phase, primitive ID, segment
progress, remaining episode time, true target/reference velocity and
acceleration, and control mix ratio. The generated future trajectory is thus a
training-only value baseline aid; neither privileged group reaches the Actor.

Recurrent PPO minibatches preserve complete time sequences and split by
environment rather than randomly mixing individual transitions. Actor and
Critic have separate optimizers and gradient clipping. PPO also supports value
clipping, target-KL early stopping, learning-rate decay and entropy decay.

## Motion task pool

`motion_task.py` builds a complete reference trajectory when an environment is
reset. Capture still occurs before trajectory time starts. A sampled episode is:

```text
capture -> random motion segments -> bounded final braking -> stopped dwell
```

Selectable primitive IDs are:

```text
hold, accelerate, cruise, decelerate, turn, climb, descend
```

`final_stop`, `stopped` and `capture` are internal IDs. A config selects the
training pool through `primitive_ids`. `prefix_ids` fixes an initial sequence,
and `required_ids` guarantees that selected primitives occur at least once.
`required_one_of_ids` reserves one sampled slot for one member of a group; the
vertical curriculum uses `["climb", "descend"]` so every episode contains at
least one altitude-change segment without forcing both directions. Keep those
IDs out of `primitive_ids` when the curriculum must contain exactly one vertical
segment per episode.

The helper's vertical CTBR correction is:

```text
delta_thrust = z_feedback_scale * (
    z_pos_gain * z_error
  + z_vel_gain * own_vertical_velocity
  - z_target_velocity_gain * target_vertical_velocity
  - z_target_accel_gain * target_vertical_acceleration
)
```

`z_target_velocity_gain` defaults to zero for backward compatibility. Setting it
near `z_vel_gain` adds target vertical-velocity feedforward while retaining
position feedback and velocity damping. `z_target_accel_gain` also defaults to
zero and is helper-only; enabling it compensates the smooth acceleration at the
start and end of a vertical trajectory segment.

Every sampled segment has a random duration, target horizontal speed,
acceleration limit, curvature and vertical speed. The generator enforces global
limits on:

- horizontal and vertical speed
- longitudinal, lateral and vertical acceleration
- horizontal and vertical jerk
- curvature and curvature rate
- yaw rate
- horizontal workspace radius and vertical displacement

Transitions are smooth. Velocity and heading are never changed instantaneously.
Trajectory samples that leave the configured workspace are rejected and
resampled.

The horizontal following point is defined from the trajectory tangent:

```text
goal_xy = target_xy - follow_distance * horizontal_tangent
goal_z  = target_z + follow_vertical_offset
```

The tangent remains defined while braking and stopped, so the following point
does not jump when target speed approaches zero. `climb` uses negative NED
vertical speed; `descend` uses positive NED vertical speed.

## Reward and success

The dense tracking reward is unified across moving, final braking and successful
stopped tracking:

```text
q_position = exp(-(xy_error / xy_sigma)^2 - (z_error / z_sigma)^2)
q_velocity = exp(-(3D_velocity_error / velocity_sigma)^2)

r_tracking = position_weight * q_position
           + velocity_weight * q_velocity
           + joint_weight * q_position * q_velocity
```

Small tilt, action magnitude and action-difference penalties are added. Dense
terms are multiplied by the policy timestep, making reward scale stable when
the action frequency changes. Capture entry and terminal rewards are one-time
events and are not timestep-scaled.

The optional moving progress term is signed and bounded:

```text
r_moving_progress = progress_weight
                  * clip(previous_xy_error - current_xy_error, -limit, limit)
```

It rewards closing the follow-point error and gives an equal penalty when the
vehicle drifts away. Unlike absolute position quality, it cannot be collected
indefinitely by remaining at a mediocre tracking error. In ratio control mode,
configs that use this term can disable the raw PPO action-magnitude penalty
while retaining the action-difference penalty; otherwise a partly mixed PPO
action is penalized at full scale even though only its configured ratio reaches
the vehicle.

`moving_good` remains a small bonus and a success diagnostic. It is not the
main learning signal. Height and vertical velocity are part of the continuous
3D tracking objective. If final stopping begins after the moving fraction has
already failed, the episode ends with `moving_success_failed`; it cannot collect
positive stopped rewards until timeout.

For episodes containing `climb` or `descend`, success also requires a dedicated
vertical fraction. A vertical step is good only when both follow-point height
error and vertical-velocity error meet their tighter tolerances. Failure is
reported as `vertical_success_failed`, and stopped reward is disabled when this
gate has failed.

An optional primitive-independent local credit signal summarizes the last fixed
time window (2 seconds in Stage 7). It blends continuous joint quality with the
minimum XY/velocity/Z good fraction and emits once per configured interval. A
separate penalty is applied only when XY error has increased beyond a deadband
across the window. The history is not reset at primitive boundaries, so it also
applies to continuous random trajectories.

## JSON training configs

The normal training command now only needs `--config`:

```bash
./python.sh /home/1234/workspace/runpy/pegasus_iris_fast_line_follow/train_pegasus_iris_fast_line_follow_ppo.py \
  --config /home/1234/workspace/runpy/pegasus_iris_fast_line_follow/configs/stage1_speed_pool_5hz_ratio_5to5.json
```

Config sections are organizational; scalar keys map to the existing command-line
argument names. A command-line scalar still overrides the config value. Relative
`results_root` and `load_checkpoint` paths are resolved from the config file.
An optional `output.task_description` string is copied to `args.json` and printed
at startup. If the config omits it, the field is omitted from `args.json`.
Configs may use `"extends": "base.json"`; nested sections are inherited and only
the listed values are overridden.

Available curriculum configs:

- `bridge_causal_actor_5hz_ratio_5to5.json`: 100k-step compatibility run from
  `seed43_20260715_004143`; removes future points from the Actor while keeping
  the current speed-only task and resets the GRU gate/optimizers.
- `stage1_speed_pool_5hz_ratio_5to5.json`: random acceleration, cruise and
  deceleration segments; no turns or altitude changes.
- `stage2_turn_pool_5hz_ratio_5to5.json`: adds turn segments and requires at
  least one real turn per episode. Replace `REPLACE_WITH_STAGE1_RUN` with the
  selected Stage 1 result directory before running it.
- `stage2b_turn_pool_masked_accel_bridge_5hz_ratio_5to5.json`: keeps the Stage 2
  task for a 100k-step distribution bridge while masking Actor target
  acceleration. It points to the completed `seed43_20260715_155524` checkpoint.
- `stage3_vertical_pool_5hz_ratio_5to5.json`: requires at least one straight
  moving `climb` or `descend` segment per episode, retains the straight speed
  primitives, and adds vertical-specific success diagnostics.
  Replace `REPLACE_WITH_MASKED_ACCEL_BRIDGE_RUN` after completing Stage 2b.
- `stage4_combined_long_pool_5hz_ratio_5to5.json`: transfers from the completed
  Stage 3 policy and combines acceleration, cruise, deceleration, turns, straight
  climbs and straight descents. Each episode starts with acceleration, samples
  six to eight additional segments, and contains at least one turn and one
  vertical segment. The policy horizon remains 5 Hz while the episode limit is
  extended to 60 seconds.
- `stage4_combined_long_pool_reward_rebalance_50k.json`: a short bridge from
  Stage 4 update 30 that keeps the same task and success gates, shifts dense
  reward weight from standalone velocity tracking toward continuous position
  and joint tracking, resets optimizer momentum, and lowers the GRU learning
  rate multiplier.
- `stage5_combined_extra_long_pool_400k_seed2.json`: transfers the rebalanced
  Stage 4 update-10 policy to seed 2, samples ten to twelve pool segments after
  the initial acceleration, and extends the episode limit to 80 seconds without
  increasing primitive speed, acceleration, curvature or vertical-speed limits.
- `stage5a_corrective_reward_intermediate_pool_100k_seed3.json`: rolls back to
  the pre-regression Stage 4 update-10 policy, uses eight to ten sampled segments
  and a 128-step rollout, and aligns velocity reward with position recovery by
  adding a bounded corrective velocity reference. A broad recovery term keeps
  position reward informative when tracking error is outside the narrow Gaussian
  region. Success still uses the original position and goal-velocity gates.
- `stage5b_progress_reward_bridge_50k_seed5.json`: rolls back to the best early
  seed-4 interval checkpoint and keeps the same eight-to-ten segment pool and
  5:5 control mix. It adds signed moving XY progress, reduces the broad recovery
  reward, removes the raw action-magnitude penalty, keeps a small action-change
  penalty, resets optimizer momentum, and uses separate constant Actor/Critic
  learning rates.
- `stage7_local_credit_popart_5hz_ratio_5to5_100k_seed7.json`: transfers the
  seed-5 best Actor, uses six to eight segments, a 256-step rollout and
  `lambda=0.98`, adds the 2-second local signal, trains the Critic in PopArt
  normalized units, freezes the transferred GRU branch, and limits cumulative
  Actor drift with a fixed-reference KL term.
- `eval_fixed_seed5_best_50k.json`: deterministic no-update evaluation with a
  fixed per-environment task bank. `ablation/run_fixed_ablation.sh` compares the
  retained GRU, disabled GRU, wider CTBR roll/pitch limits and helper-only control.

Stage 7 sets `allow_partial_checkpoint=true` because six causal local-window
statistics were added to the privileged Critic input. Actor tensors remain exact;
only the added columns of the Critic input layer start from fresh initialization.

New checkpoints include Actor and Critic optimizer states, PopArt state when
enabled, the fixed reference-policy anchor when used, plus Python, NumPy, Torch
and environment RNG states. A legacy
checkpoint has no optimizer state,
so the first migrated run intentionally starts new optimizers. Set
`reset_rng_on_load=true` when a transferred checkpoint should retain optimizer
state but use the new config's seed and task sampling. Interrupting training
saves both `actor_critic.pt` and `actor_critic_interrupted.pt` when shutdown is
handled by the training process.

When `best_checkpoint_window` is enabled, the trainer also saves
`models/actor_critic_best.pt` and a JSON sidecar. Its rolling score combines
moving joint quality (50%), moving XY quality (40%) and the moving success gate
(10%). Optional early stopping counts consecutive updates whose rolling score
falls by more than `early_stop_drop_threshold` from the best score.

## Results

New runs are written directly below `result/ppo_train` using the naming style
`seed1_YYYYMMDD_HHMMSS`; the seed remains an actual reproducibility parameter
and the timestamp distinguishes repeated runs. Each run stores:

- resolved arguments/config and run summary
- current, interval and interrupted checkpoints
- terminal log and optional TensorBoard events
- update and episode CSV files
- tracking, reward, PPO, outcome, direction/radius and primitive plots

Update metrics include the raw/effective GRU gate, separate backbone/recurrent
learning rates, raw return/value statistics, value clipping, reference KL,
per-axis policy/helper/final CTBR commands and saturation, local-window quality,
and moving/stopped XY, Z, velocity and joint good fractions.
Vertical-task rows additionally contain height/vertical-velocity good
fractions, vertical gate state and per-primitive vertical velocity error.
Those conditions are also split by primitive in
`plots/primitive_conditions.png`; `condition_diagnostics.png` and
`episode_condition_fractions.png` expose the main success bottleneck directly.
Episode rows include the sampled primitive sequence, trajectory duration,
curvature and actual 3D target/follow endpoint.

## Host-side tests

The trajectory generator and recurrent network can be tested without Isaac Sim:

```bash
/home/ry/miniconda3/envs/px4control/bin/python -m unittest discover \
  -s pegasus_iris_fast_line_follow/tests -v
```

The controller probe remains compatible with the legacy straight-line mode.
PX4/MAVLink-only status such as armed state, flight mode, failsafe and estimator
timestamps remains unavailable in this fast environment.
