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

The Actor uses measurable vehicle state and task references only. Its base 32
values contain:

- local NED position, velocity and acceleration
- roll, pitch, sine/cosine yaw and FRD body rates
- current following-point relative position
- target relative position
- desired following-point velocity and acceleration
- previous applied CTBR command

The default task config appends five future following points at 0.2, 0.4, 0.8,
1.2 and 1.6 seconds. This produces 47 Actor inputs: 32 base values plus 15
future-reference values. Future points are relative to the drone in the same
policy frame. During capture they are held at the current capture point.

The Actor is:

```text
base observation -> 256-256-128 ELU frame encoder
frame embedding history -> GRU 128
future points -> reference encoder 128
gated fusion -> tanh-squashed Gaussian CTBR action
```

The old frame encoder and action head keep their parameter names. A previous
32-observation MLP checkpoint can therefore initialize the new Actor exactly.
The GRU and future-reference branches start behind zero-valued residual gates,
so loading the old checkpoint does not immediately change its action output.

The Critic is asymmetric. In addition to the Actor observation it receives
simulator-only phase, primitive ID, segment progress, remaining episode time,
true target/reference velocity and acceleration, and control mix ratio. These
values are never inputs to the deployed Actor.

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

- `stage1_speed_pool_5hz_ratio_5to5.json`: random acceleration, cruise and
  deceleration segments; no turns or altitude changes.
- `stage2_turn_pool_5hz_ratio_5to5.json`: adds turn segments and requires at
  least one real turn per episode. Replace `REPLACE_WITH_STAGE1_RUN` with the
  selected Stage 1 result directory before running it.

The Stage 1 config migrates the latest straight-line 5:5 Actor checkpoint. It
sets `allow_partial_checkpoint=true` because the asymmetric Critic input layer
is wider. The migration is expected to report exact Actor tensors and one
partially expanded Critic input tensor.

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

Update metrics include per-primitive sample counts, XY/Z/velocity error and
good-step fraction. `plots/primitive_performance.png` compares primitive types.
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
