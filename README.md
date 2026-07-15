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
- desired following-point velocity and acceleration
- previous applied CTBR command

No future trajectory point, primitive ID, phase ID or trajectory progress is an
Actor input. In particular, a path sampled by the simulator cannot leak into
the policy that will be deployed against an unknown real target.

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

`moving_good` remains a small bonus and a success diagnostic. It is not the
main learning signal. Height and vertical velocity are part of the continuous
3D tracking objective. If final stopping begins after the moving fraction has
already failed, the episode ends with `moving_success_failed`; it cannot collect
positive stopped rewards until timeout.

## JSON training configs

The normal training command now only needs `--config`:

```bash
./python.sh /home/1234/workspace/runpy/pegasus_iris_fast_line_follow/train_pegasus_iris_fast_line_follow_ppo.py \
  --config /home/1234/workspace/runpy/pegasus_iris_fast_line_follow/configs/stage1_speed_pool_5hz_ratio_5to5.json
```

Config sections are organizational; scalar keys map to the existing command-line
argument names. A command-line scalar still overrides the config value. Relative
`results_root` and `load_checkpoint` paths are resolved from the config file.

Available curriculum configs:

- `bridge_causal_actor_5hz_ratio_5to5.json`: 100k-step compatibility run from
  `seed43_20260715_004143`; removes future points from the Actor while keeping
  the current speed-only task and resets the GRU gate/optimizers.
- `stage1_speed_pool_5hz_ratio_5to5.json`: random acceleration, cruise and
  deceleration segments; no turns or altitude changes.
- `stage2_turn_pool_5hz_ratio_5to5.json`: adds turn segments and requires at
  least one real turn per episode. Replace `REPLACE_WITH_STAGE1_RUN` with the
  selected Stage 1 result directory before running it.

The bridge config sets `allow_partial_checkpoint=true` because the old
checkpoint contains the removed Actor future-reference branch. The 77-value
Critic layout and all shared tensor shapes are preserved, so migration should
report exact shared tensors and only skip the obsolete reference tensors.

New checkpoints include Actor and Critic optimizer states plus Python, NumPy,
Torch and environment RNG states. A legacy checkpoint has no optimizer state,
so the first migrated run intentionally starts new optimizers. Interrupting
training saves both `actor_critic.pt` and `actor_critic_interrupted.pt`.

## Results

Each run keeps the timestamp naming style `seed43_YYYYMMDD_HHMMSS` and stores:

- resolved arguments/config and run summary
- current, interval and interrupted checkpoints
- terminal log and optional TensorBoard events
- update and episode CSV files
- tracking, reward, PPO, outcome, direction/radius and primitive plots

Update metrics include the raw/effective GRU gate, separate backbone/recurrent
learning rates, and moving/stopped XY, Z, velocity and joint good fractions.
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
