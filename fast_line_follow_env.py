from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import carb
import numpy as np
import omni.timeline
from scipy.spatial.transform import Rotation

try:
    from isaacsim.core.api import World
except Exception:
    from omni.isaac.core.world import World

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

from .ctbr_backend import (
    CTBRActionLimits,
    GoalPoint,
    HomePoint,
    ObservationData,
    RotorCTBRBackend,
    RotorCTBRBackendConfig,
    SafetyLimits,
    clamp,
    enu_to_ned,
    future_reference_vector,
    goal_distance,
    map_policy_action_to_ctbr,
    ned_to_enu,
    observation_vector,
    rotate_world_xy_to_policy_frame,
    wrap_angle_pi,
)
from .motion_task import (
    PRIMITIVE_CODES,
    GeneratedTrajectory,
    MotionPoolConfig,
    MotionTaskGenerator,
)


REWARD_TERM_KEYS = [
    "reward_alive",
    "reward_time",
    "reward_progress",
    "reward_distance",
    "reward_z",
    "reward_speed",
    "reward_braking",
    "reward_stop_overspeed",
    "reward_capture",
    "reward_moving_position",
    "reward_moving_velocity",
    "reward_moving_good",
    "reward_stopped_position",
    "reward_tilt",
    "reward_control",
    "reward_goal_zone",
    "reward_dwell",
    "reward_success",
    "reward_crash",
    "reward_timeout",
]


@dataclass
class LineFollowTaskConfig:
    follow_distance_m: float = 1.0
    line_length_m: float = 4.0
    randomize_line_length: bool = False
    line_length_min_m: float = 3.5
    line_length_max_m: float = 4.5
    line_length_sampling: str = "uniform_area"
    target_speed_mps: float = 0.35
    target_accel_sec: float = 2.0
    target_decel_sec: float = 2.0
    target_stopped_speed_threshold_mps: float = 0.05
    target_z_delta_m: float = 0.0
    follow_vertical_offset_m: float = 0.0
    line_yaw_deg: float = 0.0
    randomize_line_yaw: bool = False
    line_yaw_min_deg: float = -20.0
    line_yaw_max_deg: float = 20.0
    tracking_xy_tolerance_m: float = 0.45
    tracking_z_tolerance_m: float = 0.40
    tracking_velocity_tolerance_mps: float = 0.45
    stopped_speed_xy_tolerance_mps: float = 0.25
    stopped_speed_z_tolerance_mps: float = 0.25
    moving_success_dwell_sec: float = 1.0
    moving_reward_min_progress_fraction: float = 0.20
    moving_success_min_fraction: float = 0.50
    moving_success_xy_tolerance_m: float = 0.60
    moving_success_velocity_tolerance_mps: float = 0.35
    stopped_success_dwell_sec: float = 2.0
    capture_radius_m: float = 0.80
    capture_hold_sec: float = 1.0
    reward_capture_once: float = 3.0
    reward_capture_hold: float = 0.15
    reward_capture_tracking_scale: float = 1.0
    reward_moving_good: float = 0.10
    reward_moving_joint_scale: float = 1.0
    max_tracking_error_m: float = 3.0
    min_target_distance_m: float = 0.35
    reward_position_scale: float = 2.0
    reward_velocity_scale: float = 0.8
    reward_stop_speed_scale: float = 2.0
    reward_braking_scale: float = 1.0
    reward_stop_overspeed_scale: float = 1.0
    reward_stopped_progress_multiplier: float = 2.0
    reward_stopped_time_penalty: float = -0.05
    stopped_approach_speed_gain: float = 0.8
    stopped_max_approach_speed_mps: float = 0.6
    reward_too_close_scale: float = 1.5
    position_sigma_m: float = 0.75
    z_sigma_m: float = 0.50
    velocity_sigma_mps: float = 0.50
    stop_speed_sigma_mps: float = 0.30
    reward_action_delta_scale: float = 0.02
    reward_tilt_scale: float = 0.08
    reward_scale_by_dt: bool = True
    motion_pool: MotionPoolConfig = field(default_factory=MotionPoolConfig.disabled)


@dataclass
class FastLineFollowEnvConfig:
    num_envs: int = 64
    env_spacing_m: float = 8.0
    physics_dt: float = 0.004
    rendering_dt: float = 1.0 / 60.0
    step_dt_sim_sec: float = 0.2
    episode_length: int = 160
    takeoff_altitude: float = 5.0
    render: bool = False
    seed: int = 43
    randomize_yaw_on_reset: bool = False
    random_yaw_max_offset_deg: float = 180.0
    yaw_hold_kp: float = 1.0
    yaw_hold_max_rate: float = math.radians(15.0)
    yaw_success_tolerance_deg: float = 5.0
    reward_alive: float = 0.0
    reward_progress_scale: float = 1.0
    reward_distance_scale: float = 0.10
    reward_z_scale: float = 0.40
    reward_control_scale: float = 0.05
    reward_goal_zone: float = 0.20
    reward_dwell_scale: float = 0.50
    reward_success: float = 50.0
    reward_crash: float = -40.0
    reward_timeout: float = -30.0
    target_accel_observation_filter_alpha: float = 0.5
    action_limits: CTBRActionLimits = field(default_factory=CTBRActionLimits)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    backend_config: RotorCTBRBackendConfig = field(default_factory=RotorCTBRBackendConfig)


class FastIrisLineFollowVecEnv:
    """Pegasus Iris vector environment for line following without PX4/MAVLink."""

    base_obs_dim = 32
    action_dim = 4

    def __init__(self, config: FastLineFollowEnvConfig, task_config: LineFollowTaskConfig):
        self.config = config
        self.task_config = task_config
        self.num_envs = int(config.num_envs)
        self.reference_horizon_sec = tuple(task_config.motion_pool.reference_horizon_sec)
        self.future_reference_dim = 3 * len(self.reference_horizon_sec)
        self.obs_dim = self.base_obs_dim
        self.privileged_obs_dim = 4 + len(PRIMITIVE_CODES) + 16
        self.critic_obs_dim = (
            self.obs_dim + self.future_reference_dim + self.privileged_obs_dim
        )
        self._rng = np.random.default_rng(config.seed)
        self._sim_steps_per_policy_step = max(1, int(round(config.step_dt_sim_sec / config.physics_dt)))
        self._episode_id = np.zeros(self.num_envs, dtype=np.int64)
        self._step_id = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._last_tracking_xy_err = np.full(self.num_envs, np.nan, dtype=np.float64)
        self._last_goal_xy_progress = np.zeros(self.num_envs, dtype=np.float64)
        self._last_velocity_error = np.zeros(self.num_envs, dtype=np.float64)
        self._last_speed_xy = np.full(self.num_envs, np.nan, dtype=np.float64)
        self._last_braking_progress = np.zeros(self.num_envs, dtype=np.float64)
        self._desired_approach_speed = np.zeros(self.num_envs, dtype=np.float64)
        self._allowed_stopped_speed = np.zeros(self.num_envs, dtype=np.float64)
        self._stopped_velocity_error = np.zeros(self.num_envs, dtype=np.float64)
        self._target_distance = np.zeros(self.num_envs, dtype=np.float64)
        self._inside_goal_zone = np.zeros(self.num_envs, dtype=bool)
        self._goal_dwell_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._goal_dwell_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._capture_dwell_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._capture_dwell_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._capture_entered = np.zeros(self.num_envs, dtype=bool)
        self._capture_acquired = np.zeros(self.num_envs, dtype=bool)
        self._motion_start_step = np.full(self.num_envs, -1, dtype=np.int64)
        self._moving_track_dwell_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_eligible_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._moving_xy_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_z_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_velocity_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._moving_xy_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._moving_z_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._moving_velocity_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._moving_xy_good = np.zeros(self.num_envs, dtype=bool)
        self._moving_z_good = np.zeros(self.num_envs, dtype=bool)
        self._moving_velocity_good = np.zeros(self.num_envs, dtype=bool)
        self._moving_good = np.zeros(self.num_envs, dtype=bool)
        self._stopped_track_dwell_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stopped_eligible_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stopped_xy_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stopped_z_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stopped_speed_good_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stopped_xy_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._stopped_z_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._stopped_speed_good_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._stopped_xy_good = np.zeros(self.num_envs, dtype=bool)
        self._stopped_z_good = np.zeros(self.num_envs, dtype=bool)
        self._stopped_speed_good = np.zeros(self.num_envs, dtype=bool)
        self._moving_success_met = np.zeros(self.num_envs, dtype=bool)
        self._line_dir = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (self.num_envs, 1))
        self._line_length_m = np.full(
            self.num_envs,
            max(0.0, float(task_config.line_length_m)),
            dtype=np.float64,
        )
        self._target_start_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._target_pos_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._target_vel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._target_accel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._goal_vel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._goal_accel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._observed_target_accel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._previous_target_vel_ned = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._target_phase = np.array(["stopped"] * self.num_envs, dtype=object)
        self._primitive_id = np.array(["stopped"] * self.num_envs, dtype=object)
        self._primitive_code = np.full(
            self.num_envs,
            PRIMITIVE_CODES["stopped"],
            dtype=np.int64,
        )
        self._primitive_segment_index = np.zeros(self.num_envs, dtype=np.int64)
        self._primitive_segment_progress = np.zeros(self.num_envs, dtype=np.float64)
        self._target_heading_rad = np.zeros(self.num_envs, dtype=np.float64)
        self._target_curvature_per_m = np.zeros(self.num_envs, dtype=np.float64)
        self._motion_sequence_ids: List[Tuple[str, ...]] = [tuple() for _ in range(self.num_envs)]
        self._trajectories: List[Optional[GeneratedTrajectory]] = [None for _ in range(self.num_envs)]
        self._target_progress_fraction = np.zeros(self.num_envs, dtype=np.float64)
        self._line_accel_sec = np.zeros(self.num_envs, dtype=np.float64)
        self._line_cruise_sec = np.zeros(self.num_envs, dtype=np.float64)
        self._line_decel_sec = np.zeros(self.num_envs, dtype=np.float64)
        self._line_total_motion_sec = np.zeros(self.num_envs, dtype=np.float64)
        self._line_peak_speed_mps = np.zeros(self.num_envs, dtype=np.float64)
        self._home = [HomePoint(0.0, 0.0, -config.takeoff_altitude) for _ in range(self.num_envs)]
        self._goal = [GoalPoint(0.0, 0.0, -config.takeoff_altitude) for _ in range(self.num_envs)]
        self._policy_yaw_reference: List[Optional[float]] = [None for _ in range(self.num_envs)]
        self._yaw_target: List[Optional[float]] = [None for _ in range(self.num_envs)]
        self._episode_start_yaw = np.zeros(self.num_envs, dtype=np.float64)
        self._max_abs_yaw_error = np.zeros(self.num_envs, dtype=np.float64)
        self._prev_action = np.tile(
            np.array(
                [0.0, 0.0, 0.0, config.action_limits.hover_thrust],
                dtype=np.float32,
            ),
            (self.num_envs, 1),
        )
        self._previous_policy_action = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self._motion_generator = (
            MotionTaskGenerator(
                task_config.motion_pool,
                config.step_dt_sim_sec,
                task_config.follow_distance_m,
                task_config.follow_vertical_offset_m,
            )
            if task_config.motion_pool.enabled
            else None
        )
        self._last_reward_terms: List[Dict[str, float]] = [
            {key: 0.0 for key in REWARD_TERM_KEYS} for _ in range(self.num_envs)
        ]
        self._last_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.world.set_simulation_dt(physics_dt=config.physics_dt, rendering_dt=config.rendering_dt)
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        self.env_origins_enu = self._make_env_origins()
        self.vehicles: List[Multirotor] = []
        self.backends: List[RotorCTBRBackend] = []
        for env_id in range(self.num_envs):
            self._spawn_vehicle(env_id)

        self.world.reset()
        self.timeline.play()
        for _ in range(10):
            self.world.step(render=False)
        self.reset()

    def _make_env_origins(self) -> np.ndarray:
        cols = int(math.ceil(math.sqrt(self.num_envs)))
        origins = np.zeros((self.num_envs, 3), dtype=np.float64)
        for i in range(self.num_envs):
            row = i // cols
            col = i % cols
            origins[i, 0] = col * self.config.env_spacing_m
            origins[i, 1] = row * self.config.env_spacing_m
        return origins

    def _spawn_vehicle(self, env_id: int) -> None:
        backend_cfg = self.config.backend_config
        backend_cfg.hover_thrust_command = float(self.config.action_limits.hover_thrust)
        backend = RotorCTBRBackend(backend_cfg)
        cfg = MultirotorConfig()
        cfg.backends = [backend]
        cfg.sensors = []
        cfg.graphical_sensors = []
        cfg.graphs = []
        origin = self.env_origins_enu[env_id]
        init_pos = origin + np.array([0.0, 0.0, self.config.takeoff_altitude], dtype=np.float64)
        vehicle = Multirotor(
            f"/World/quadrotor_fast_{env_id}",
            ROBOTS["Iris"],
            env_id,
            init_pos.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=False).as_quat(),
            config=cfg,
        )
        self.vehicles.append(vehicle)
        self.backends.append(backend)

    def close(self) -> None:
        try:
            for backend in self.backends:
                backend.stop()
            self.timeline.stop()
        except Exception:
            pass

    def reset(self, env_ids: Optional[Iterable[int]] = None) -> np.ndarray:
        ids = np.arange(self.num_envs, dtype=np.int64) if env_ids is None else np.asarray(list(env_ids), dtype=np.int64)
        if ids.size == 0:
            return self._build_obs_batch()
        for env_id in ids:
            self._reset_one(int(env_id))
        for _ in range(3):
            self.world.step(render=False)
        return self._build_obs_batch()

    def _reset_one(self, env_id: int) -> None:
        self._episode_id[env_id] += 1
        self._step_id[env_id] = 0
        self._episode_return[env_id] = 0.0
        self._last_tracking_xy_err[env_id] = np.nan
        self._last_goal_xy_progress[env_id] = 0.0
        self._last_velocity_error[env_id] = 0.0
        self._last_speed_xy[env_id] = np.nan
        self._last_braking_progress[env_id] = 0.0
        self._desired_approach_speed[env_id] = 0.0
        self._allowed_stopped_speed[env_id] = self.task_config.stopped_speed_xy_tolerance_mps
        self._stopped_velocity_error[env_id] = 0.0
        self._target_distance[env_id] = self.task_config.follow_distance_m
        self._inside_goal_zone[env_id] = False
        self._goal_dwell_steps[env_id] = 0
        self._goal_dwell_fraction[env_id] = 0.0
        self._capture_dwell_steps[env_id] = 0
        self._capture_dwell_fraction[env_id] = 0.0
        self._capture_entered[env_id] = False
        self._capture_acquired[env_id] = False
        self._motion_start_step[env_id] = -1
        self._moving_track_dwell_steps[env_id] = 0
        self._moving_eligible_steps[env_id] = 0
        self._moving_good_steps[env_id] = 0
        self._moving_good_fraction[env_id] = 0.0
        self._moving_xy_good_steps[env_id] = 0
        self._moving_z_good_steps[env_id] = 0
        self._moving_velocity_good_steps[env_id] = 0
        self._moving_xy_good_fraction[env_id] = 0.0
        self._moving_z_good_fraction[env_id] = 0.0
        self._moving_velocity_good_fraction[env_id] = 0.0
        self._moving_xy_good[env_id] = False
        self._moving_z_good[env_id] = False
        self._moving_velocity_good[env_id] = False
        self._moving_good[env_id] = False
        self._stopped_track_dwell_steps[env_id] = 0
        self._stopped_eligible_steps[env_id] = 0
        self._stopped_xy_good_steps[env_id] = 0
        self._stopped_z_good_steps[env_id] = 0
        self._stopped_speed_good_steps[env_id] = 0
        self._stopped_xy_good_fraction[env_id] = 0.0
        self._stopped_z_good_fraction[env_id] = 0.0
        self._stopped_speed_good_fraction[env_id] = 0.0
        self._stopped_xy_good[env_id] = False
        self._stopped_z_good[env_id] = False
        self._stopped_speed_good[env_id] = False
        self._moving_success_met[env_id] = False
        self._target_accel_ned[env_id, :] = 0.0
        self._goal_vel_ned[env_id, :] = 0.0
        self._goal_accel_ned[env_id, :] = 0.0
        self._observed_target_accel_ned[env_id, :] = 0.0
        self._previous_target_vel_ned[env_id, :] = 0.0
        self._prev_action[env_id, :] = [
            0.0,
            0.0,
            0.0,
            self.config.action_limits.hover_thrust,
        ]
        self._previous_policy_action[env_id, :] = 0.0
        self._primitive_id[env_id] = "capture"
        self._primitive_code[env_id] = PRIMITIVE_CODES["capture"]
        self._primitive_segment_index[env_id] = 0
        self._primitive_segment_progress[env_id] = 0.0
        self._target_heading_rad[env_id] = 0.0
        self._target_curvature_per_m[env_id] = 0.0
        self._motion_sequence_ids[env_id] = tuple()
        self._trajectories[env_id] = None
        self._last_reward_terms[env_id] = {key: 0.0 for key in REWARD_TERM_KEYS}

        yaw_ned = 0.0
        self._policy_yaw_reference[env_id] = None
        self._yaw_target[env_id] = None
        if self.config.randomize_yaw_on_reset:
            max_offset = math.radians(max(0.0, min(180.0, self.config.random_yaw_max_offset_deg)))
            yaw_ned = float(self._rng.uniform(-max_offset, max_offset))
            self._policy_yaw_reference[env_id] = 0.0
            self._yaw_target[env_id] = yaw_ned

        pos_enu = self.env_origins_enu[env_id] + ned_to_enu([0.0, 0.0, -self.config.takeoff_altitude])
        self._set_vehicle_pose_velocity(env_id, pos_enu, yaw_ned)
        self.backends[env_id].reset()
        self.backends[env_id].set_safe_command()
        if self._motion_generator is None:
            self._sample_line_goal(env_id)
        else:
            self._sample_motion_task(env_id)
        self._previous_target_vel_ned[env_id] = self._target_vel_ned[env_id]
        obs = self._obs_data(env_id)
        self._episode_start_yaw[env_id] = obs.yaw
        self._max_abs_yaw_error[env_id] = 0.0
        self._last_infos[env_id] = self._build_info(env_id, "reset")

    def _set_vehicle_pose_velocity(self, env_id: int, pos_enu: Sequence[float], yaw_ned: float) -> None:
        vehicle = self.vehicles[env_id]
        yaw_enu = math.pi / 2.0 - float(yaw_ned)
        quat_xyzw = Rotation.from_euler("XYZ", [0.0, 0.0, yaw_enu], degrees=False).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        try:
            vehicle.set_world_pose(position=np.asarray(pos_enu, dtype=np.float64), orientation=quat_wxyz)
            vehicle.set_linear_velocity(np.zeros(3, dtype=np.float64))
            vehicle.set_angular_velocity(np.zeros(3, dtype=np.float64))
            return
        except Exception:
            pass

        dc = vehicle.get_dc_interface()
        body = dc.get_rigid_body(vehicle._stage_prefix + "/body")
        try:
            transform = vehicle.get_world_pose()
            _ = transform
        except Exception:
            pass
        dc.set_rigid_body_linear_velocity(body, carb._carb.Float3([0.0, 0.0, 0.0]))
        dc.set_rigid_body_angular_velocity(body, carb._carb.Float3([0.0, 0.0, 0.0]))

    def _sample_line_goal(self, env_id: int) -> None:
        task = self.task_config
        yaw_deg = (
            self._rng.uniform(task.line_yaw_min_deg, task.line_yaw_max_deg)
            if task.randomize_line_yaw
            else task.line_yaw_deg
        )
        yaw = math.radians(float(yaw_deg))
        self._line_dir[env_id] = [math.cos(yaw), math.sin(yaw)]
        if task.randomize_line_length:
            min_length = max(0.0, float(task.line_length_min_m))
            max_length = max(min_length, float(task.line_length_max_m))
            if task.line_length_sampling == "uniform_area":
                squared_length = self._rng.uniform(
                    min_length * min_length,
                    max_length * max_length,
                )
                line_length = math.sqrt(float(squared_length))
            elif task.line_length_sampling == "uniform_radius":
                line_length = float(self._rng.uniform(min_length, max_length))
            else:
                raise ValueError(
                    "line_length_sampling must be 'uniform_area' or "
                    f"'uniform_radius', got {task.line_length_sampling!r}"
                )
        else:
            line_length = max(0.0, float(task.line_length_m))
        self._line_length_m[env_id] = line_length
        home = self._home[env_id]
        self._target_start_ned[env_id] = [
            home.x + task.follow_distance_m * self._line_dir[env_id, 0],
            home.y + task.follow_distance_m * self._line_dir[env_id, 1],
            home.z + task.target_z_delta_m,
        ]
        self._compute_motion_profile(env_id)
        self._sync_goal_to_target(env_id)

    def _sample_motion_task(self, env_id: int) -> None:
        task = self.task_config
        yaw_deg = (
            self._rng.uniform(task.line_yaw_min_deg, task.line_yaw_max_deg)
            if task.randomize_line_yaw
            else task.line_yaw_deg
        )
        yaw = math.radians(float(yaw_deg))
        self._line_dir[env_id] = [math.cos(yaw), math.sin(yaw)]
        home = self._home[env_id]
        trajectory = self._motion_generator.sample(
            self._rng,
            [home.x, home.y, home.z + task.target_z_delta_m],
            yaw,
        )
        self._trajectories[env_id] = trajectory
        self._motion_sequence_ids[env_id] = trajectory.sequence_ids
        endpoint_delta = trajectory.goal_positions[-1, :2] - np.array([home.x, home.y])
        self._line_length_m[env_id] = float(np.linalg.norm(endpoint_delta))
        self._line_accel_sec[env_id] = 0.0
        self._line_cruise_sec[env_id] = max(
            0.0,
            float(trajectory.final_stop_index * self.config.step_dt_sim_sec),
        )
        self._line_decel_sec[env_id] = max(
            0.0,
            trajectory.duration_sec - self._line_cruise_sec[env_id] - task.motion_pool.stop_hold_sec,
        )
        self._line_total_motion_sec[env_id] = trajectory.duration_sec - task.motion_pool.stop_hold_sec
        self._line_peak_speed_mps[env_id] = float(
            np.max(np.linalg.norm(trajectory.velocities[:, :2], axis=1))
        )
        self._target_start_ned[env_id] = trajectory.positions[0]
        self._sync_goal_to_target(env_id)

    def _compute_motion_profile(self, env_id: int) -> None:
        task = self.task_config
        line_length = max(0.0, float(self._line_length_m[env_id]))
        target_speed = max(0.0, float(task.target_speed_mps))
        accel_sec = max(0.0, float(task.target_accel_sec))
        decel_sec = max(0.0, float(task.target_decel_sec))
        if line_length <= 1e-6 or target_speed <= 1e-6:
            self._line_accel_sec[env_id] = 0.0
            self._line_cruise_sec[env_id] = 0.0
            self._line_decel_sec[env_id] = 0.0
            self._line_total_motion_sec[env_id] = 0.0
            self._line_peak_speed_mps[env_id] = 0.0
            return

        accel_dist = 0.5 * target_speed * accel_sec
        decel_dist = 0.5 * target_speed * decel_sec
        if accel_dist + decel_dist <= line_length:
            self._line_accel_sec[env_id] = accel_sec
            self._line_decel_sec[env_id] = decel_sec
            self._line_peak_speed_mps[env_id] = target_speed
            self._line_cruise_sec[env_id] = (
                line_length - accel_dist - decel_dist
            ) / target_speed
            self._line_total_motion_sec[env_id] = (
                self._line_accel_sec[env_id]
                + self._line_cruise_sec[env_id]
                + self._line_decel_sec[env_id]
            )
            return

        accel_rate = target_speed / accel_sec if accel_sec > 1e-6 else math.inf
        decel_rate = target_speed / decel_sec if decel_sec > 1e-6 else math.inf
        if math.isinf(accel_rate) and math.isinf(decel_rate):
            peak_speed = target_speed
            actual_accel_sec = 0.0
            actual_decel_sec = 0.0
            cruise_sec = line_length / target_speed
        else:
            inv_accel = 0.0 if math.isinf(accel_rate) else 1.0 / accel_rate
            inv_decel = 0.0 if math.isinf(decel_rate) else 1.0 / decel_rate
            peak_speed = math.sqrt(max(0.0, 2.0 * line_length / max(1e-9, inv_accel + inv_decel)))
            peak_speed = min(target_speed, peak_speed)
            actual_accel_sec = 0.0 if math.isinf(accel_rate) else peak_speed / accel_rate
            actual_decel_sec = 0.0 if math.isinf(decel_rate) else peak_speed / decel_rate
            cruise_sec = 0.0

        self._line_accel_sec[env_id] = actual_accel_sec
        self._line_cruise_sec[env_id] = cruise_sec
        self._line_decel_sec[env_id] = actual_decel_sec
        self._line_total_motion_sec[env_id] = actual_accel_sec + cruise_sec + actual_decel_sec
        self._line_peak_speed_mps[env_id] = peak_speed

    def _trajectory_at(self, env_id: int, t_sec: float) -> Tuple[float, float, float, str, float]:
        line_length = max(0.0, float(self._line_length_m[env_id]))
        peak_speed = max(0.0, float(self._line_peak_speed_mps[env_id]))
        t = max(0.0, float(t_sec))
        if line_length <= 1e-6 or peak_speed <= 1e-6:
            return 0.0, 0.0, 0.0, "stopped", 1.0

        accel_sec = float(self._line_accel_sec[env_id])
        cruise_sec = float(self._line_cruise_sec[env_id])
        decel_sec = float(self._line_decel_sec[env_id])
        accel_dist = 0.5 * peak_speed * accel_sec
        cruise_dist = peak_speed * cruise_sec

        if accel_sec > 1e-6 and t < accel_sec:
            speed = peak_speed * t / accel_sec
            accel = peak_speed / accel_sec
            s = 0.5 * peak_speed * t * t / accel_sec
            return min(line_length, s), speed, accel, "moving", min(1.0, s / line_length)

        cruise_start = accel_sec
        cruise_end = accel_sec + cruise_sec
        if t < cruise_end:
            tau = t - cruise_start
            s = accel_dist + peak_speed * tau
            return min(line_length, s), peak_speed, 0.0, "moving", min(1.0, s / line_length)

        if decel_sec <= 1e-6:
            return line_length, 0.0, 0.0, "stopped", 1.0

        if t < self._line_total_motion_sec[env_id]:
            tau = t - cruise_end
            speed = peak_speed * max(0.0, 1.0 - tau / decel_sec)
            accel = -peak_speed / decel_sec
            decel_dist = peak_speed * (tau - 0.5 * tau * tau / decel_sec)
            s = min(line_length, accel_dist + cruise_dist + decel_dist)
            return s, speed, accel, "decelerating", s / line_length
        return line_length, 0.0, 0.0, "stopped", 1.0

    def _sync_goal_to_target(self, env_id: int) -> None:
        trajectory = self._trajectories[env_id]
        if trajectory is not None:
            if not self._capture_acquired[env_id]:
                state = trajectory.state_at(0.0)
                phase = "capture"
                primitive_id = "capture"
                primitive_code = PRIMITIVE_CODES[primitive_id]
                progress = 0.0
                segment_progress = 0.0
            else:
                start_step = int(self._motion_start_step[env_id])
                if start_step < 0:
                    start_step = int(self._step_id[env_id])
                    self._motion_start_step[env_id] = start_step
                motion_time_sec = max(
                    0.0,
                    float(self._step_id[env_id] - start_step) * self.config.step_dt_sim_sec,
                )
                state = trajectory.state_at(motion_time_sec)
                phase = state.phase
                primitive_id = state.primitive_id
                primitive_code = state.primitive_code
                progress = state.motion_progress
                segment_progress = state.segment_progress
            self._target_phase[env_id] = phase
            self._primitive_id[env_id] = primitive_id
            self._primitive_code[env_id] = primitive_code
            self._primitive_segment_index[env_id] = state.segment_index
            self._primitive_segment_progress[env_id] = segment_progress
            self._target_progress_fraction[env_id] = float(np.clip(progress, 0.0, 1.0))
            self._target_pos_ned[env_id] = state.position
            if phase == "capture":
                self._target_vel_ned[env_id] = 0.0
                self._target_accel_ned[env_id] = 0.0
                self._goal_vel_ned[env_id] = 0.0
                self._goal_accel_ned[env_id] = 0.0
            else:
                self._target_vel_ned[env_id] = state.velocity
                self._target_accel_ned[env_id] = state.acceleration
                self._goal_vel_ned[env_id] = state.goal_velocity
                self._goal_accel_ned[env_id] = state.goal_acceleration
            self._target_heading_rad[env_id] = state.heading_rad
            self._target_curvature_per_m[env_id] = state.curvature_per_m
            desired = state.goal_position
            self._goal[env_id] = GoalPoint(
                float(desired[0]),
                float(desired[1]),
                float(desired[2]),
            )
            return

        if not self._capture_acquired[env_id]:
            s, speed, accel, phase, progress = 0.0, 0.0, 0.0, "capture", 0.0
        else:
            start_step = int(self._motion_start_step[env_id])
            if start_step < 0:
                start_step = int(self._step_id[env_id])
                self._motion_start_step[env_id] = start_step
            motion_time_sec = max(
                0.0,
                float(self._step_id[env_id] - start_step) * self.config.step_dt_sim_sec,
            )
            s, speed, accel, phase, progress = self._trajectory_at(env_id, motion_time_sec)
        line_dir = self._line_dir[env_id]
        self._target_phase[env_id] = phase
        self._target_progress_fraction[env_id] = float(np.clip(progress, 0.0, 1.0))
        target = self._target_start_ned[env_id].copy()
        target[0] += s * line_dir[0]
        target[1] += s * line_dir[1]
        self._target_pos_ned[env_id] = target
        self._target_vel_ned[env_id] = [speed * line_dir[0], speed * line_dir[1], 0.0]
        self._target_accel_ned[env_id] = [accel * line_dir[0], accel * line_dir[1], 0.0]
        self._goal_vel_ned[env_id] = self._target_vel_ned[env_id]
        self._goal_accel_ned[env_id] = self._target_accel_ned[env_id]
        self._primitive_id[env_id] = (
            "capture"
            if phase == "capture"
            else "final_stop"
            if phase == "decelerating"
            else "stopped"
            if phase == "stopped"
            else "cruise"
        )
        self._primitive_code[env_id] = PRIMITIVE_CODES[self._primitive_id[env_id]]
        self._primitive_segment_progress[env_id] = float(np.clip(progress, 0.0, 1.0))
        self._target_heading_rad[env_id] = math.atan2(line_dir[1], line_dir[0])
        self._target_curvature_per_m[env_id] = 0.0
        desired = target.copy()
        desired[0] -= self.task_config.follow_distance_m * line_dir[0]
        desired[1] -= self.task_config.follow_distance_m * line_dir[1]
        self._goal[env_id] = GoalPoint(float(desired[0]), float(desired[1]), float(desired[2]))

    def _future_reference_positions(self, env_id: int) -> Optional[np.ndarray]:
        if not self.reference_horizon_sec:
            return None
        trajectory = self._trajectories[env_id]
        if trajectory is None or not self._capture_acquired[env_id]:
            goal = self._goal[env_id]
            current = np.array([goal.x, goal.y, goal.z], dtype=np.float64)
            return np.repeat(current[None, :], len(self.reference_horizon_sec), axis=0)
        start_step = int(self._motion_start_step[env_id])
        current_time_sec = max(
            0.0,
            float(self._step_id[env_id] - start_step) * self.config.step_dt_sim_sec,
        )
        return np.stack(
            [
                trajectory.state_at(current_time_sec + offset).goal_position
                for offset in self.reference_horizon_sec
            ],
            axis=0,
        )

    def _update_target_acceleration_observation(self, env_id: int) -> None:
        dt = max(1e-6, float(self.config.step_dt_sim_sec))
        raw_accel = (
            self._target_vel_ned[env_id] - self._previous_target_vel_ned[env_id]
        ) / dt
        alpha = float(np.clip(self.config.target_accel_observation_filter_alpha, 0.0, 1.0))
        self._observed_target_accel_ned[env_id] = (
            alpha * raw_accel
            + (1.0 - alpha) * self._observed_target_accel_ned[env_id]
        )
        self._previous_target_vel_ned[env_id] = self._target_vel_ned[env_id]

    @property
    def required_capture_steps(self) -> int:
        sec = max(0.0, float(self.task_config.capture_hold_sec))
        return max(1, int(math.ceil(sec / self.config.step_dt_sim_sec))) if sec > 0.0 else 1

    @property
    def required_moving_track_steps(self) -> int:
        sec = max(0.0, float(self.task_config.moving_success_dwell_sec))
        return max(1, int(math.ceil(sec / self.config.step_dt_sim_sec))) if sec > 0.0 else 1

    @property
    def required_stopped_track_steps(self) -> int:
        sec = max(0.0, float(self.task_config.stopped_success_dwell_sec))
        return max(1, int(math.ceil(sec / self.config.step_dt_sim_sec))) if sec > 0.0 else 1

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.action_dim}), got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)

        for env_id in range(self.num_envs):
            self._sync_goal_to_target(env_id)
            cmd = self._policy_action_to_command(env_id, actions[env_id])
            self.backends[env_id].set_ctbr_command(*cmd)
            self._prev_action[env_id] = np.asarray(cmd, dtype=np.float32)

        for _ in range(self._sim_steps_per_policy_step):
            self.world.step(render=self.config.render)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []
        done_ids: List[int] = []
        for env_id in range(self.num_envs):
            self._step_id[env_id] += 1
            self._sync_goal_to_target(env_id)
            self._update_target_acceleration_observation(env_id)
            reward, done, reason = self._compute_reward_done(env_id, actions[env_id])
            self._previous_policy_action[env_id] = actions[env_id]
            self._episode_return[env_id] += reward
            rewards[env_id] = float(reward)
            dones[env_id] = bool(done)
            info = self._build_info(env_id, reason)
            info["episode_return"] = float(self._episode_return[env_id])
            infos.append(info)
            self._last_infos[env_id] = info
            if done:
                done_ids.append(env_id)

        obs = self._build_obs_batch()
        if done_ids:
            reset_obs = self.reset(done_ids)
            obs[done_ids] = reset_obs[done_ids]
        return obs, rewards, dones, infos

    def _policy_action_to_command(self, env_id: int, action: Sequence[float]) -> Tuple[float, float, float, float]:
        limits = self.config.action_limits
        pol_roll, pol_pitch, pol_yaw, pol_thrust = map_policy_action_to_ctbr(action, limits)
        roll, pitch, yaw, thrust = pol_roll, pol_pitch, pol_yaw, pol_thrust

        obs = self._obs_data(env_id)
        goal = self._goal[env_id]
        x_err = float(obs.x) - goal.x
        y_err = float(obs.y) - goal.y
        z_err = float(obs.z) - goal.z
        x_err, y_err = rotate_world_xy_to_policy_frame(x_err, y_err, obs.yaw, self._policy_yaw_reference[env_id])
        vx, vy = rotate_world_xy_to_policy_frame(obs.vx, obs.vy, obs.yaw, self._policy_yaw_reference[env_id])
        target_vx, target_vy = rotate_world_xy_to_policy_frame(
            self._goal_vel_ned[env_id, 0],
            self._goal_vel_ned[env_id, 1],
            obs.yaw,
            self._policy_yaw_reference[env_id],
        )
        target_ax, target_ay = rotate_world_xy_to_policy_frame(
            self._goal_accel_ned[env_id, 0],
            self._goal_accel_ned[env_id, 1],
            obs.yaw,
            self._policy_yaw_reference[env_id],
        )
        vz = float(obs.vz)

        goal_feedback_scale = limits.goal_feedback_scale
        if goal_feedback_scale is None:
            goal_feedback_scale = limits.pd_feedback_scale
        x_tilt_term = (
            goal_feedback_scale * limits.goal_xy_pos_gain * x_err
            + limits.xy_velocity_damping_gain * vx
            - limits.xy_target_velocity_gain * target_vx
            - limits.xy_target_accel_gain * target_ax
        )
        y_tilt_term = (
            goal_feedback_scale * limits.goal_xy_pos_gain * y_err
            + limits.xy_velocity_damping_gain * vy
            - limits.xy_target_velocity_gain * target_vy
            - limits.xy_target_accel_gain * target_ay
        )
        if limits.xy_control_mode == "legacy":
            roll_des = clamp(x_tilt_term, -limits.xy_max_tilt_cmd, limits.xy_max_tilt_cmd)
            pitch_des = clamp(y_tilt_term, -limits.xy_max_tilt_cmd, limits.xy_max_tilt_cmd)
        else:
            pitch_des = clamp(x_tilt_term, -limits.xy_max_tilt_cmd, limits.xy_max_tilt_cmd)
            roll_des = clamp(-y_tilt_term, -limits.xy_max_tilt_cmd, limits.xy_max_tilt_cmd)
        kp_att = 1.0
        max_feedback_rate = min(0.080, limits.max_roll_rate, limits.max_pitch_rate)
        pitch_fb = clamp(kp_att * (pitch_des - float(obs.pitch)), -max_feedback_rate, max_feedback_rate)
        roll_fb = clamp(kp_att * (roll_des - float(obs.roll)), -max_feedback_rate, max_feedback_rate)
        controller_pitch_raw = limits.attitude_feedback_scale * pitch_fb
        controller_roll_raw = limits.attitude_feedback_scale * roll_fb

        yaw_target = self._yaw_target[env_id]
        if yaw_target is None or limits.yaw_hold_kp <= 0.0:
            controller_yaw = 0.0
        else:
            yaw_error = wrap_angle_pi(yaw_target - float(obs.yaw))
            yaw_feedback_limit = min(limits.yaw_hold_max_rate, limits.max_yaw_rate)
            controller_yaw = clamp(
                limits.yaw_hold_kp * yaw_error,
                -yaw_feedback_limit,
                yaw_feedback_limit,
            )

        controller_thrust_delta = limits.z_feedback_scale * (
            limits.z_pos_gain * z_err + limits.z_vel_gain * vz
        )
        policy_thrust_delta = pol_thrust - limits.hover_thrust

        if limits.control_mix_mode == "ratio":
            policy_ratio = clamp(limits.policy_ratio, 0.0, 1.0)
            controller_ratio = 1.0 - policy_ratio
            controller_roll = clamp(
                controller_roll_raw,
                -limits.max_roll_rate,
                limits.max_roll_rate,
            )
            controller_pitch = clamp(
                controller_pitch_raw,
                -limits.max_pitch_rate,
                limits.max_pitch_rate,
            )
            controller_thrust = clamp(
                limits.hover_thrust + controller_thrust_delta,
                limits.thrust_min,
                limits.thrust_max,
            )
            roll = controller_ratio * controller_roll + policy_ratio * pol_roll
            pitch = controller_ratio * controller_pitch + policy_ratio * pol_pitch
            yaw = controller_ratio * controller_yaw + policy_ratio * pol_yaw
            thrust = controller_ratio * controller_thrust + policy_ratio * pol_thrust
        else:
            # Legacy mode retained so previous probe/training results remain reproducible.
            roll = controller_roll_raw + limits.residual_gain * pol_roll
            pitch = controller_pitch_raw + limits.residual_gain * pol_pitch
            yaw = controller_yaw
            thrust = (
                limits.hover_thrust
                + controller_thrust_delta
                + limits.residual_gain * policy_thrust_delta
            )

        roll = clamp(roll, -limits.max_roll_rate, limits.max_roll_rate)
        pitch = clamp(pitch, -limits.max_pitch_rate, limits.max_pitch_rate)
        yaw = clamp(yaw, -limits.max_yaw_rate, limits.max_yaw_rate)
        thrust = clamp(thrust, limits.thrust_min, limits.thrust_max)
        return float(roll), float(pitch), float(yaw), float(thrust)

    def _obs_data(self, env_id: int) -> ObservationData:
        state = self.vehicles[env_id].state
        local_pos_enu = np.asarray(state.position, dtype=np.float64) - self.env_origins_enu[env_id]
        pos_ned = enu_to_ned(local_pos_enu)
        vel_ned = enu_to_ned(state.linear_velocity)
        try:
            quat_ned_frd = state.get_attitude_ned_frd()
            roll, pitch, yaw = Rotation.from_quat(quat_ned_frd).as_euler("XYZ", degrees=False)
        except Exception:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        try:
            rates_frd = state.get_angular_velocity_frd()
        except Exception:
            rates_frd = np.zeros(3, dtype=np.float64)
        try:
            accel_ned = state.get_linear_acceleration_ned()
        except Exception:
            try:
                accel_ned = enu_to_ned(state.linear_acceleration)
            except Exception:
                accel_ned = np.zeros(3, dtype=np.float64)
        return ObservationData(
            x=float(pos_ned[0]),
            y=float(pos_ned[1]),
            z=float(pos_ned[2]),
            vx=float(vel_ned[0]),
            vy=float(vel_ned[1]),
            vz=float(vel_ned[2]),
            ax=float(accel_ned[0]),
            ay=float(accel_ned[1]),
            az=float(accel_ned[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(wrap_angle_pi(yaw)),
            rollspeed=float(rates_frd[0]),
            pitchspeed=float(rates_frd[1]),
            yawspeed=float(rates_frd[2]),
        )

    def _build_obs_batch(self) -> np.ndarray:
        return np.stack([self._build_obs_one(env_id) for env_id in range(self.num_envs)], axis=0).astype(np.float32)

    def _build_obs_one(self, env_id: int) -> np.ndarray:
        self._sync_goal_to_target(env_id)
        obs = self._obs_data(env_id)
        policy_obs = observation_vector(
            own=obs,
            goal=self._goal[env_id],
            target_position=self._target_pos_ned[env_id],
            target_velocity=self._goal_vel_ned[env_id],
            target_acceleration=self._goal_accel_ned[env_id],
            prev_command=self._prev_action[env_id],
            action_limits=self.config.action_limits,
            yaw_reference=self._policy_yaw_reference[env_id],
        )
        if policy_obs.shape != (self.obs_dim,):
            raise RuntimeError(
                f"observation contract produced {policy_obs.shape}, expected ({self.obs_dim},)"
            )
        return policy_obs

    def build_critic_obs(self, actor_obs: Optional[np.ndarray] = None) -> np.ndarray:
        if actor_obs is None:
            actor_obs = self._build_obs_batch()
        actor_obs = np.asarray(actor_obs, dtype=np.float32)
        if actor_obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"actor_obs must have shape ({self.num_envs}, {self.obs_dim}), "
                f"got {actor_obs.shape}"
            )
        privileged = np.stack(
            [self._build_privileged_obs_one(env_id) for env_id in range(self.num_envs)],
            axis=0,
        ).astype(np.float32)
        if self.future_reference_dim > 0:
            future_reference = np.stack(
                [self._build_critic_future_reference_one(env_id) for env_id in range(self.num_envs)],
                axis=0,
            ).astype(np.float32)
        else:
            future_reference = np.zeros((self.num_envs, 0), dtype=np.float32)
        critic_obs = np.concatenate(
            (actor_obs, future_reference, privileged),
            axis=1,
        ).astype(np.float32)
        if critic_obs.shape != (self.num_envs, self.critic_obs_dim):
            raise RuntimeError(
                f"critic observation has shape {critic_obs.shape}, expected "
                f"({self.num_envs}, {self.critic_obs_dim})"
            )
        return critic_obs

    def _build_critic_future_reference_one(self, env_id: int) -> np.ndarray:
        future_positions = self._future_reference_positions(env_id)
        if future_positions is None:
            return np.zeros(0, dtype=np.float32)
        return future_reference_vector(
            self._obs_data(env_id),
            future_positions,
            yaw_reference=self._policy_yaw_reference[env_id],
        )

    def _build_privileged_obs_one(self, env_id: int) -> np.ndarray:
        phase_names = ("capture", "moving", "decelerating", "stopped")
        phase_one_hot = np.zeros(len(phase_names), dtype=np.float32)
        phase = str(self._target_phase[env_id])
        if phase in phase_names:
            phase_one_hot[phase_names.index(phase)] = 1.0
        primitive_one_hot = np.zeros(len(PRIMITIVE_CODES), dtype=np.float32)
        primitive_code = int(np.clip(self._primitive_code[env_id], 0, len(PRIMITIVE_CODES) - 1))
        primitive_one_hot[primitive_code] = 1.0
        remaining_fraction = max(
            0.0,
            1.0 - float(self._step_id[env_id]) / max(1.0, float(self.config.episode_length)),
        )
        scalars = np.array(
            [
                self._target_progress_fraction[env_id],
                self._primitive_segment_progress[env_id],
                remaining_fraction,
            ],
            dtype=np.float32,
        )
        dynamics = np.concatenate(
            (
                self._target_vel_ned[env_id] / 3.0,
                self._target_accel_ned[env_id] / 3.0,
                self._goal_vel_ned[env_id] / 3.0,
                self._goal_accel_ned[env_id] / 3.0,
            )
        ).astype(np.float32)
        control_ratio = np.array(
            [float(self.config.action_limits.policy_ratio)],
            dtype=np.float32,
        )
        privileged = np.concatenate(
            (phase_one_hot, primitive_one_hot, scalars, dynamics, control_ratio)
        ).astype(np.float32)
        if privileged.shape != (self.privileged_obs_dim,):
            raise RuntimeError(
                f"privileged observation has shape {privileged.shape}, "
                f"expected ({self.privileged_obs_dim},)"
            )
        return np.clip(privileged, -5.0, 5.0)

    def _target_is_stopped(self, env_id: int) -> bool:
        speed = float(np.linalg.norm(self._target_vel_ned[env_id]))
        return self._target_phase[env_id] == "stopped" and speed <= self.task_config.target_stopped_speed_threshold_mps

    def _tracking_status(self, env_id: int) -> Dict[str, float | bool]:
        obs = self._obs_data(env_id)
        goal = self._goal[env_id]
        xy_err = math.sqrt((obs.x - goal.x) ** 2 + (obs.y - goal.y) ** 2)
        z_err = abs(obs.z - goal.z)
        vel_err_xy = math.sqrt(
            (obs.vx - self._goal_vel_ned[env_id, 0]) ** 2
            + (obs.vy - self._goal_vel_ned[env_id, 1]) ** 2
        )
        vel_err_z = abs(obs.vz - self._goal_vel_ned[env_id, 2])
        vel_err = math.sqrt(vel_err_xy * vel_err_xy + vel_err_z * vel_err_z)
        speed_xy = math.sqrt(obs.vx ** 2 + obs.vy ** 2)
        speed_z = abs(obs.vz)
        target_distance = math.sqrt(
            (obs.x - self._target_pos_ned[env_id, 0]) ** 2
            + (obs.y - self._target_pos_ned[env_id, 1]) ** 2
            + (obs.z - self._target_pos_ned[env_id, 2]) ** 2
        )
        target_stopped = self._target_is_stopped(env_id)
        if target_stopped:
            inside = (
                xy_err <= self.task_config.tracking_xy_tolerance_m
                and z_err <= self.task_config.tracking_z_tolerance_m
                and speed_xy <= self.task_config.stopped_speed_xy_tolerance_mps
                and speed_z <= self.task_config.stopped_speed_z_tolerance_mps
            )
        else:
            inside = (
                xy_err <= self.task_config.tracking_xy_tolerance_m
                and z_err <= self.task_config.tracking_z_tolerance_m
                and vel_err <= self.task_config.tracking_velocity_tolerance_mps
            )
        return {
            "inside": bool(inside),
            "target_stopped": bool(target_stopped),
            "xy_err": float(xy_err),
            "z_err": float(z_err),
            "vel_err_xy": float(vel_err_xy),
            "vel_err_z": float(vel_err_z),
            "vel_err": float(vel_err),
            "speed_xy": float(speed_xy),
            "speed_z": float(speed_z),
            "accel_x": float(obs.ax),
            "accel_y": float(obs.ay),
            "accel_z": float(obs.az),
            "target_distance": float(target_distance),
        }

    def _safety_reason(self, env_id: int) -> Optional[str]:
        obs = self._obs_data(env_id)
        limits = self.config.safety_limits
        alt = -float(obs.z)
        tilt_deg = math.degrees(math.sqrt(obs.roll ** 2 + obs.pitch ** 2))
        max_rate = max(abs(obs.rollspeed), abs(obs.pitchspeed), abs(obs.yawspeed))
        if alt < limits.min_altitude:
            return f"near_ground_alt={alt:.2f}m"
        if alt > limits.max_altitude:
            return f"too_high_alt={alt:.2f}m"
        if tilt_deg > limits.max_tilt_deg:
            return f"tilt_too_large={tilt_deg:.1f}deg"
        if max_rate > limits.max_body_rate:
            return f"body_rate_too_large={max_rate:.2f}rad/s"
        if obs.vz > limits.max_down_speed:
            return f"falling_fast_vz={obs.vz:.2f}m/s"
        home = self._home[env_id]
        xy_dist = math.sqrt((obs.x - home.x) ** 2 + (obs.y - home.y) ** 2)
        z_err = abs(obs.z - home.z)
        if xy_dist > limits.max_xy_from_home:
            return f"xy_out_of_bounds={xy_dist:.2f}m"
        if z_err > limits.max_z_error_from_home:
            return f"z_out_of_bounds={z_err:.2f}m"
        return None

    def _compute_reward_done(self, env_id: int, action: np.ndarray) -> Tuple[float, bool, str]:
        obs = self._obs_data(env_id)
        goal = self._goal[env_id]
        done = False
        done_reason = "running"
        reward_crash = 0.0

        safety_reason = self._safety_reason(env_id)
        if safety_reason is not None:
            done = True
            done_reason = safety_reason
            reward_crash += self.config.reward_crash

        status = self._tracking_status(env_id)
        xy_err = float(status["xy_err"])
        z_err = float(status["z_err"])
        vel_err_xy = float(status["vel_err_xy"])
        vel_err = float(status["vel_err"])
        speed_xy = float(status["speed_xy"])
        speed_z = float(status["speed_z"])
        target_distance = float(status["target_distance"])
        target_stopped = bool(status["target_stopped"])
        target_progress = float(self._target_progress_fraction[env_id])
        moving_track_eligible = target_progress >= float(self.task_config.moving_reward_min_progress_fraction)
        capture_entered_now = False
        capture_acquired_now = False
        capture_position_inside = False
        capture_inside = False
        moving_good = False
        self._moving_xy_good[env_id] = False
        self._moving_z_good[env_id] = False
        self._moving_velocity_good[env_id] = False
        self._moving_good[env_id] = False
        self._stopped_xy_good[env_id] = False
        self._stopped_z_good[env_id] = False
        self._stopped_speed_good[env_id] = False
        self._last_velocity_error[env_id] = vel_err
        self._target_distance[env_id] = target_distance
        previous_speed_xy = (
            speed_xy
            if np.isnan(self._last_speed_xy[env_id])
            else float(self._last_speed_xy[env_id])
        )
        braking_progress = float(np.clip(previous_speed_xy - speed_xy, -0.5, 0.5))
        self._last_speed_xy[env_id] = speed_xy
        self._last_braking_progress[env_id] = braking_progress

        prev_xy_err = xy_err if np.isnan(self._last_tracking_xy_err[env_id]) else self._last_tracking_xy_err[env_id]
        goal_xy_progress = float(np.clip(prev_xy_err - xy_err, -0.5, 0.5))
        self._last_tracking_xy_err[env_id] = xy_err
        self._last_goal_xy_progress[env_id] = goal_xy_progress

        if not self._capture_acquired[env_id]:
            capture_position_inside = (
                xy_err <= self.task_config.capture_radius_m
                and z_err <= self.task_config.tracking_z_tolerance_m
            )
            capture_inside = (
                capture_position_inside
                and speed_xy <= self.task_config.tracking_velocity_tolerance_mps
                and speed_z <= self.task_config.stopped_speed_z_tolerance_mps
            )
            if capture_position_inside and not self._capture_entered[env_id]:
                capture_entered_now = True
                self._capture_entered[env_id] = True
            self._capture_dwell_steps[env_id] = (
                self._capture_dwell_steps[env_id] + 1 if capture_inside else 0
            )
            self._capture_dwell_fraction[env_id] = min(
                1.0,
                float(self._capture_dwell_steps[env_id]) / float(self.required_capture_steps),
            )
            self._goal_dwell_steps[env_id] = self._capture_dwell_steps[env_id]
            self._goal_dwell_fraction[env_id] = self._capture_dwell_fraction[env_id]
            if self._capture_dwell_steps[env_id] >= self.required_capture_steps:
                self._capture_acquired[env_id] = True
                self._motion_start_step[env_id] = int(self._step_id[env_id])
                capture_acquired_now = True
        elif bool(status["target_stopped"]):
            self._stopped_xy_good[env_id] = (
                xy_err <= self.task_config.tracking_xy_tolerance_m
            )
            self._stopped_z_good[env_id] = (
                z_err <= self.task_config.tracking_z_tolerance_m
            )
            self._stopped_speed_good[env_id] = (
                speed_xy <= self.task_config.stopped_speed_xy_tolerance_mps
                and speed_z <= self.task_config.stopped_speed_z_tolerance_mps
            )
            self._stopped_eligible_steps[env_id] += 1
            self._stopped_xy_good_steps[env_id] += int(self._stopped_xy_good[env_id])
            self._stopped_z_good_steps[env_id] += int(self._stopped_z_good[env_id])
            self._stopped_speed_good_steps[env_id] += int(self._stopped_speed_good[env_id])
            stopped_denominator = float(max(1, self._stopped_eligible_steps[env_id]))
            self._stopped_xy_good_fraction[env_id] = (
                float(self._stopped_xy_good_steps[env_id]) / stopped_denominator
            )
            self._stopped_z_good_fraction[env_id] = (
                float(self._stopped_z_good_steps[env_id]) / stopped_denominator
            )
            self._stopped_speed_good_fraction[env_id] = (
                float(self._stopped_speed_good_steps[env_id]) / stopped_denominator
            )
            self._stopped_track_dwell_steps[env_id] = self._stopped_track_dwell_steps[env_id] + 1 if status["inside"] else 0
            self._goal_dwell_steps[env_id] = self._stopped_track_dwell_steps[env_id]
            self._goal_dwell_fraction[env_id] = min(
                1.0,
                float(self._stopped_track_dwell_steps[env_id]) / float(self.required_stopped_track_steps),
            )
        else:
            inside_moving_track = bool(status["inside"]) and moving_track_eligible
            self._moving_track_dwell_steps[env_id] = self._moving_track_dwell_steps[env_id] + 1 if inside_moving_track else 0
            self._moving_xy_good[env_id] = (
                moving_track_eligible
                and xy_err <= self.task_config.moving_success_xy_tolerance_m
            )
            self._moving_z_good[env_id] = (
                moving_track_eligible
                and z_err <= self.task_config.tracking_z_tolerance_m
            )
            self._moving_velocity_good[env_id] = (
                moving_track_eligible
                and vel_err <= self.task_config.moving_success_velocity_tolerance_mps
            )
            moving_good = bool(
                self._moving_xy_good[env_id]
                and self._moving_z_good[env_id]
                and self._moving_velocity_good[env_id]
            )
            self._moving_good[env_id] = moving_good
            if moving_track_eligible:
                self._moving_eligible_steps[env_id] += 1
                self._moving_xy_good_steps[env_id] += int(self._moving_xy_good[env_id])
                self._moving_z_good_steps[env_id] += int(self._moving_z_good[env_id])
                self._moving_velocity_good_steps[env_id] += int(
                    self._moving_velocity_good[env_id]
                )
                if moving_good:
                    self._moving_good_steps[env_id] += 1
                moving_denominator = float(max(1, self._moving_eligible_steps[env_id]))
                self._moving_good_fraction[env_id] = float(
                    self._moving_good_steps[env_id]
                ) / moving_denominator
                self._moving_xy_good_fraction[env_id] = float(
                    self._moving_xy_good_steps[env_id]
                ) / moving_denominator
                self._moving_z_good_fraction[env_id] = float(
                    self._moving_z_good_steps[env_id]
                ) / moving_denominator
                self._moving_velocity_good_fraction[env_id] = float(
                    self._moving_velocity_good_steps[env_id]
                ) / moving_denominator
                self._moving_success_met[env_id] = (
                    self._moving_good_fraction[env_id]
                    >= self.task_config.moving_success_min_fraction
                )
            self._goal_dwell_steps[env_id] = self._moving_track_dwell_steps[env_id]
            self._goal_dwell_fraction[env_id] = min(
                1.0,
                float(self._moving_track_dwell_steps[env_id]) / float(self.required_moving_track_steps),
            )
        if not self._capture_acquired[env_id] or capture_acquired_now:
            self._inside_goal_zone[env_id] = bool(capture_inside)
        else:
            self._inside_goal_zone[env_id] = bool(status["inside"]) and (
                target_stopped or moving_track_eligible
            )

        if not done and xy_err > self.task_config.max_tracking_error_m:
            done = True
            done_reason = "tracking_lost"
            reward_crash += self.config.reward_crash
        if not done and target_distance < self.task_config.min_target_distance_m:
            done = True
            done_reason = "too_close_to_target"
            reward_crash += self.config.reward_crash

        pos_sigma = max(1e-3, float(self.task_config.position_sigma_m))
        z_sigma = max(1e-3, float(self.task_config.z_sigma_m))
        vel_sigma = max(1e-3, float(self.task_config.velocity_sigma_mps))
        stop_sigma = max(1e-3, float(self.task_config.stop_speed_sigma_mps))
        position_quality = math.exp(
            -((xy_err / pos_sigma) ** 2)
            -((z_err / z_sigma) ** 2)
        )
        velocity_quality = math.exp(-((vel_err / vel_sigma) ** 2))
        dense_scale = (
            float(self.config.step_dt_sim_sec)
            if self.task_config.reward_scale_by_dt
            else 1.0
        )
        too_close_margin = max(0.0, self.task_config.min_target_distance_m - target_distance)
        too_close_penalty = -self.task_config.reward_too_close_scale * (
            too_close_margin / max(1e-3, self.task_config.min_target_distance_m)
        )

        capture_reward_active = not self._capture_acquired[env_id] or capture_acquired_now
        moving_reward_active = (
            self._capture_acquired[env_id]
            and not capture_acquired_now
            and not target_stopped
        )
        stopped_reward_active = (
            self._capture_acquired[env_id]
            and target_stopped
            and self._moving_success_met[env_id]
        )
        reward_alive = dense_scale * self.config.reward_alive
        reward_time = (
            dense_scale * self.task_config.reward_stopped_time_penalty
            if stopped_reward_active
            else 0.0
        )
        reward_progress = (
            self.config.reward_progress_scale * goal_xy_progress
            if capture_reward_active
            else 0.0
        )
        reward_distance = too_close_penalty
        if capture_reward_active:
            reward_distance += -dense_scale * self.config.reward_distance_scale * min(
                xy_err,
                self.task_config.max_tracking_error_m,
            )
        # Height is part of the normalized 3D position quality instead of a separate objective.
        reward_z = 0.0
        reward_moving_position = (
            dense_scale * self.task_config.reward_position_scale * position_quality
            if moving_reward_active
            else 0.0
        )
        reward_moving_velocity = (
            dense_scale
            * (
                self.task_config.reward_velocity_scale * velocity_quality
                + self.task_config.reward_moving_joint_scale
                * position_quality
                * velocity_quality
            )
            if moving_reward_active
            else 0.0
        )
        reward_stopped_position = (
            dense_scale * self.task_config.reward_position_scale * position_quality
            if stopped_reward_active
            else 0.0
        )
        reward_speed = (
            dense_scale
            * (
                self.task_config.reward_velocity_scale * velocity_quality
                + self.task_config.reward_moving_joint_scale
                * position_quality
                * velocity_quality
            )
            if stopped_reward_active
            else 0.0
        )
        reward_braking = 0.0
        reward_stop_overspeed = 0.0
        if self._capture_acquired[env_id] and target_stopped:
            distance_outside_zone = max(
                0.0,
                xy_err - self.task_config.tracking_xy_tolerance_m,
            )
            desired_approach_speed = min(
                self.task_config.stopped_max_approach_speed_mps,
                self.task_config.stopped_approach_speed_gain * distance_outside_zone,
            )
            if xy_err > 1e-6:
                goal_dir_x = (goal.x - obs.x) / xy_err
                goal_dir_y = (goal.y - obs.y) / xy_err
            else:
                goal_dir_x = 0.0
                goal_dir_y = 0.0
            desired_vx = desired_approach_speed * goal_dir_x
            desired_vy = desired_approach_speed * goal_dir_y
            stopped_velocity_error = math.sqrt(
                (obs.vx - desired_vx) ** 2
                + (obs.vy - desired_vy) ** 2
                + speed_z * speed_z
            )
            allowed_stopped_speed = (
                desired_approach_speed
                + self.task_config.stopped_speed_xy_tolerance_mps
            )
            self._desired_approach_speed[env_id] = desired_approach_speed
            self._allowed_stopped_speed[env_id] = allowed_stopped_speed
            self._stopped_velocity_error[env_id] = stopped_velocity_error
            overspeed_xy = max(
                0.0,
                speed_xy - allowed_stopped_speed,
            )
            normalized_overspeed = overspeed_xy / stop_sigma
            if stopped_reward_active:
                reward_stop_overspeed = (
                    -dense_scale
                    * self.task_config.reward_stop_overspeed_scale
                    * min(normalized_overspeed * normalized_overspeed, 4.0)
                )
        else:
            self._desired_approach_speed[env_id] = 0.0
            self._allowed_stopped_speed[env_id] = self.task_config.stopped_speed_xy_tolerance_mps
            self._stopped_velocity_error[env_id] = 0.0
        reward_capture = (
            dense_scale
            * self.task_config.reward_capture_tracking_scale
            * (position_quality * (0.7 + 0.3 * velocity_quality) - 1.0)
            if capture_reward_active
            else 0.0
        )
        if capture_entered_now:
            reward_capture += self.task_config.reward_capture_once
        if (not target_stopped) and (not self._capture_acquired[env_id] or capture_acquired_now) and capture_inside:
            reward_capture += (
                dense_scale
                * self.task_config.reward_capture_hold
                * max(0.0, self._capture_dwell_fraction[env_id])
            )
        reward_moving_good = (
            dense_scale * self.task_config.reward_moving_good
            if moving_good
            else 0.0
        )
        tilt = math.sqrt(obs.roll ** 2 + obs.pitch ** 2)
        control_penalty = float(np.mean(np.square(np.clip(action, -1.0, 1.0))))
        action_delta_penalty = float(
            np.mean(np.square(action - self._previous_policy_action[env_id]))
        )
        reward_tilt = -dense_scale * self.task_config.reward_tilt_scale * tilt
        reward_control = -dense_scale * (
            self.config.reward_control_scale * control_penalty
            + self.task_config.reward_action_delta_scale * action_delta_penalty
        )
        reward_goal_zone = (
            dense_scale * self.config.reward_goal_zone
            if stopped_reward_active and self._inside_goal_zone[env_id]
            else 0.0
        )
        reward_dwell = (
            dense_scale * self.config.reward_dwell_scale * self._goal_dwell_fraction[env_id]
            if stopped_reward_active
            else 0.0
        )
        reward_success = 0.0
        reward_timeout = 0.0

        if (
            not done
            and target_stopped
            and self._capture_acquired[env_id]
            and self._moving_eligible_steps[env_id] > 0
            and not self._moving_success_met[env_id]
        ):
            done = True
            done_reason = "moving_success_failed"
            reward_timeout = self.config.reward_timeout

        if (
            not done
            and bool(status["target_stopped"])
            and self._capture_acquired[env_id]
            and self._moving_success_met[env_id]
            and self._stopped_track_dwell_steps[env_id] >= self.required_stopped_track_steps
        ):
            done = True
            done_reason = "success"
            reward_success = self.config.reward_success
        if self._step_id[env_id] >= self.config.episode_length and not done:
            done = True
            done_reason = "timeout"
            reward_timeout = self.config.reward_timeout

        reward = (
            reward_alive
            + reward_time
            + reward_progress
            + reward_distance
            + reward_z
            + reward_speed
            + reward_braking
            + reward_stop_overspeed
            + reward_capture
            + reward_moving_position
            + reward_moving_velocity
            + reward_moving_good
            + reward_stopped_position
            + reward_tilt
            + reward_control
            + reward_goal_zone
            + reward_dwell
            + reward_success
            + reward_crash
            + reward_timeout
        )
        self._last_reward_terms[env_id] = {
            "reward_alive": float(reward_alive),
            "reward_time": float(reward_time),
            "reward_progress": float(reward_progress),
            "reward_distance": float(reward_distance),
            "reward_z": float(reward_z),
            "reward_speed": float(reward_speed),
            "reward_braking": float(reward_braking),
            "reward_stop_overspeed": float(reward_stop_overspeed),
            "reward_capture": float(reward_capture),
            "reward_moving_position": float(reward_moving_position),
            "reward_moving_velocity": float(reward_moving_velocity),
            "reward_moving_good": float(reward_moving_good),
            "reward_stopped_position": float(reward_stopped_position),
            "reward_tilt": float(reward_tilt),
            "reward_control": float(reward_control),
            "reward_goal_zone": float(reward_goal_zone),
            "reward_dwell": float(reward_dwell),
            "reward_success": float(reward_success),
            "reward_crash": float(reward_crash),
            "reward_timeout": float(reward_timeout),
            "reward_total": float(reward),
        }
        return float(reward), bool(done), done_reason

    def _build_info(self, env_id: int, done_reason: str) -> Dict[str, Any]:
        obs = self._obs_data(env_id)
        goal = self._goal[env_id]
        home = self._home[env_id]
        xy_err = math.sqrt((obs.x - goal.x) ** 2 + (obs.y - goal.y) ** 2)
        z_err = abs(obs.z - goal.z)
        goal_rel_x = goal.x - obs.x
        goal_rel_y = goal.y - obs.y
        goal_rel_z = goal.z - obs.z
        speed_xy = math.sqrt(obs.vx ** 2 + obs.vy ** 2)
        speed_z = abs(obs.vz)
        yaw_target = self._yaw_target[env_id]
        yaw_error = None if yaw_target is None else wrap_angle_pi(yaw_target - obs.yaw)
        if yaw_error is not None:
            self._max_abs_yaw_error[env_id] = max(self._max_abs_yaw_error[env_id], abs(yaw_error))
        phase = str(self._target_phase[env_id])
        if phase == "capture":
            required_goal_dwell_steps = self.required_capture_steps
        elif phase == "stopped":
            required_goal_dwell_steps = self.required_stopped_track_steps
        else:
            required_goal_dwell_steps = self.required_moving_track_steps
        motion_start_step = int(self._motion_start_step[env_id])
        motion_time_sec = (
            0.0
            if motion_start_step < 0
            else max(0.0, float(self._step_id[env_id] - motion_start_step) * self.config.step_dt_sim_sec)
        )
        trajectory = self._trajectories[env_id]
        if trajectory is None:
            desired_endpoint = np.array(
                [
                    home.x + self._line_length_m[env_id] * self._line_dir[env_id, 0],
                    home.y + self._line_length_m[env_id] * self._line_dir[env_id, 1],
                    home.z,
                ],
                dtype=np.float64,
            )
            target_endpoint = np.array(
                [
                    home.x
                    + (self._line_length_m[env_id] + self.task_config.follow_distance_m)
                    * self._line_dir[env_id, 0],
                    home.y
                    + (self._line_length_m[env_id] + self.task_config.follow_distance_m)
                    * self._line_dir[env_id, 1],
                    home.z,
                ],
                dtype=np.float64,
            )
            trajectory_duration_sec = float(self._line_total_motion_sec[env_id])
        else:
            desired_endpoint = trajectory.goal_positions[-1]
            target_endpoint = trajectory.positions[-1]
            trajectory_duration_sec = trajectory.duration_sec
        return {
            "env_id": int(env_id),
            "episode_id": int(self._episode_id[env_id]),
            "step_id": int(self._step_id[env_id]),
            "done_reason": done_reason,
            "xy_err": float(xy_err),
            "z_err": float(z_err),
            "goal_distance": float(goal_distance(obs, goal)),
            "goal_xy_progress": float(self._last_goal_xy_progress[env_id]),
            "inside_goal_zone": bool(self._inside_goal_zone[env_id]),
            "goal_dwell_steps": int(self._goal_dwell_steps[env_id]),
            "required_goal_dwell_steps": int(required_goal_dwell_steps),
            "goal_dwell_fraction": float(self._goal_dwell_fraction[env_id]),
            "goal_dwell_time_sec": float(self._goal_dwell_steps[env_id] * self.config.step_dt_sim_sec),
            "required_goal_dwell_time_sec": float(required_goal_dwell_steps * self.config.step_dt_sim_sec),
            "goal_rel_x": float(goal_rel_x),
            "goal_rel_y": float(goal_rel_y),
            "goal_rel_z": float(goal_rel_z),
            "signed_z_err": float(obs.z - goal.z),
            "home_xy_err": float(math.sqrt((obs.x - home.x) ** 2 + (obs.y - home.y) ** 2)),
            "goal_home_xy_err": float(math.sqrt((goal.x - home.x) ** 2 + (goal.y - home.y) ** 2)),
            "speed_xy": float(speed_xy),
            "speed_z": float(speed_z),
            "yaw": float(obs.yaw),
            "yaw_rate": float(obs.yawspeed),
            "yaw_start": float(self._episode_start_yaw[env_id]),
            "yaw_target": yaw_target,
            "yaw_error": yaw_error,
            "max_abs_yaw_error": float(self._max_abs_yaw_error[env_id]),
            "policy_yaw_reference": self._policy_yaw_reference[env_id],
            "cmd_roll_rate": float(self._prev_action[env_id, 0]),
            "cmd_pitch_rate": float(self._prev_action[env_id, 1]),
            "cmd_yaw_rate": float(self._prev_action[env_id, 2]),
            "cmd_thrust": float(self._prev_action[env_id, 3]),
            "target_phase": phase,
            "primitive_id": str(self._primitive_id[env_id]),
            "primitive_code": int(self._primitive_code[env_id]),
            "primitive_segment_index": int(self._primitive_segment_index[env_id]),
            "primitive_segment_progress": float(self._primitive_segment_progress[env_id]),
            "motion_sequence_ids": "|".join(self._motion_sequence_ids[env_id]),
            "episode_time_sec": float(self._step_id[env_id] * self.config.step_dt_sim_sec),
            "target_time_sec": float(motion_time_sec),
            "trajectory_duration_sec": float(trajectory_duration_sec),
            "target_accel_sec": float(self._line_accel_sec[env_id]),
            "target_cruise_sec": float(self._line_cruise_sec[env_id]),
            "target_decel_sec": float(self._line_decel_sec[env_id]),
            "target_motion_total_sec": float(self._line_total_motion_sec[env_id]),
            "target_progress_fraction": float(self._target_progress_fraction[env_id]),
            "target_speed": float(np.linalg.norm(self._target_vel_ned[env_id, :2])),
            "target_peak_speed": float(self._line_peak_speed_mps[env_id]),
            "target_vel_x": float(self._target_vel_ned[env_id, 0]),
            "target_vel_y": float(self._target_vel_ned[env_id, 1]),
            "target_vel_z": float(self._target_vel_ned[env_id, 2]),
            "target_accel_x": float(self._target_accel_ned[env_id, 0]),
            "target_accel_y": float(self._target_accel_ned[env_id, 1]),
            "target_accel_z": float(self._target_accel_ned[env_id, 2]),
            "goal_ref_vel_x": float(self._goal_vel_ned[env_id, 0]),
            "goal_ref_vel_y": float(self._goal_vel_ned[env_id, 1]),
            "goal_ref_vel_z": float(self._goal_vel_ned[env_id, 2]),
            "goal_ref_accel_x": float(self._goal_accel_ned[env_id, 0]),
            "goal_ref_accel_y": float(self._goal_accel_ned[env_id, 1]),
            "goal_ref_accel_z": float(self._goal_accel_ned[env_id, 2]),
            "target_heading_rad": float(self._target_heading_rad[env_id]),
            "target_curvature_per_m": float(self._target_curvature_per_m[env_id]),
            "observed_target_accel_x": float(self._observed_target_accel_ned[env_id, 0]),
            "observed_target_accel_y": float(self._observed_target_accel_ned[env_id, 1]),
            "observed_target_accel_z": float(self._observed_target_accel_ned[env_id, 2]),
            "target_x": float(self._target_pos_ned[env_id, 0]),
            "target_y": float(self._target_pos_ned[env_id, 1]),
            "target_z": float(self._target_pos_ned[env_id, 2]),
            "target_distance": float(self._target_distance[env_id]),
            "tracking_velocity_error": float(self._last_velocity_error[env_id]),
            "braking_progress": float(self._last_braking_progress[env_id]),
            "target_stopped": bool(self._target_is_stopped(env_id)),
            "desired_approach_speed": float(self._desired_approach_speed[env_id]),
            "allowed_stopped_speed": float(self._allowed_stopped_speed[env_id]),
            "stopped_velocity_error": float(self._stopped_velocity_error[env_id]),
            "line_yaw_deg": float(
                math.degrees(
                    math.atan2(
                        self._line_dir[env_id, 1],
                        self._line_dir[env_id, 0],
                    )
                )
            ),
            "line_dir_x": float(self._line_dir[env_id, 0]),
            "line_dir_y": float(self._line_dir[env_id, 1]),
            "sampled_line_length_m": float(self._line_length_m[env_id]),
            "desired_endpoint_x": float(desired_endpoint[0]),
            "desired_endpoint_y": float(desired_endpoint[1]),
            "desired_endpoint_z": float(desired_endpoint[2]),
            "target_endpoint_x": float(target_endpoint[0]),
            "target_endpoint_y": float(target_endpoint[1]),
            "target_endpoint_z": float(target_endpoint[2]),
            "follow_distance_m": float(self.task_config.follow_distance_m),
            "capture_acquired": bool(self._capture_acquired[env_id]),
            "capture_entered": bool(self._capture_entered[env_id]),
            "capture_dwell_steps": int(self._capture_dwell_steps[env_id]),
            "required_capture_steps": int(self.required_capture_steps),
            "capture_dwell_fraction": float(self._capture_dwell_fraction[env_id]),
            "capture_radius_m": float(self.task_config.capture_radius_m),
            "capture_hold_sec": float(self.task_config.capture_hold_sec),
            "moving_reward_min_progress_fraction": float(self.task_config.moving_reward_min_progress_fraction),
            "moving_track_eligible": bool(
                self._capture_acquired[env_id]
                and not self._target_is_stopped(env_id)
                and self._target_progress_fraction[env_id]
                >= self.task_config.moving_reward_min_progress_fraction
            ),
            "moving_success_met": bool(self._moving_success_met[env_id]),
            "moving_track_dwell_steps": int(self._moving_track_dwell_steps[env_id]),
            "required_moving_track_steps": int(self.required_moving_track_steps),
            "moving_eligible_steps": int(self._moving_eligible_steps[env_id]),
            "moving_good_steps": int(self._moving_good_steps[env_id]),
            "moving_good_fraction": float(self._moving_good_fraction[env_id]),
            "moving_xy_good": bool(self._moving_xy_good[env_id]),
            "moving_z_good": bool(self._moving_z_good[env_id]),
            "moving_velocity_good": bool(self._moving_velocity_good[env_id]),
            "moving_good": bool(self._moving_good[env_id]),
            "moving_xy_good_steps": int(self._moving_xy_good_steps[env_id]),
            "moving_z_good_steps": int(self._moving_z_good_steps[env_id]),
            "moving_velocity_good_steps": int(
                self._moving_velocity_good_steps[env_id]
            ),
            "moving_xy_good_fraction": float(
                self._moving_xy_good_fraction[env_id]
            ),
            "moving_z_good_fraction": float(self._moving_z_good_fraction[env_id]),
            "moving_velocity_good_fraction": float(
                self._moving_velocity_good_fraction[env_id]
            ),
            "moving_success_min_fraction": float(self.task_config.moving_success_min_fraction),
            "moving_success_xy_tolerance_m": float(self.task_config.moving_success_xy_tolerance_m),
            "moving_success_velocity_tolerance_mps": float(
                self.task_config.moving_success_velocity_tolerance_mps
            ),
            "stopped_track_dwell_steps": int(self._stopped_track_dwell_steps[env_id]),
            "required_stopped_track_steps": int(self.required_stopped_track_steps),
            "stopped_eligible_steps": int(self._stopped_eligible_steps[env_id]),
            "stopped_xy_good": bool(self._stopped_xy_good[env_id]),
            "stopped_z_good": bool(self._stopped_z_good[env_id]),
            "stopped_speed_good": bool(self._stopped_speed_good[env_id]),
            "stopped_xy_good_steps": int(self._stopped_xy_good_steps[env_id]),
            "stopped_z_good_steps": int(self._stopped_z_good_steps[env_id]),
            "stopped_speed_good_steps": int(self._stopped_speed_good_steps[env_id]),
            "stopped_xy_good_fraction": float(
                self._stopped_xy_good_fraction[env_id]
            ),
            "stopped_z_good_fraction": float(self._stopped_z_good_fraction[env_id]),
            "stopped_speed_good_fraction": float(
                self._stopped_speed_good_fraction[env_id]
            ),
            **self._last_reward_terms[env_id],
        }
