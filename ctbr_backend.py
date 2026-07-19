from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


@dataclass
class HomePoint:
    x: float
    y: float
    z: float


@dataclass
class GoalPoint:
    x: float
    y: float
    z: float


@dataclass
class ObservationData:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0


@dataclass
class CTBRActionLimits:
    max_roll_rate: float = 0.080
    max_pitch_rate: float = 0.080
    max_yaw_rate: float = 0.010
    yaw_hold_kp: float = 1.0
    yaw_hold_max_rate: float = math.radians(15.0)
    hover_thrust: float = 0.60
    thrust_delta: float = 0.015
    thrust_min: float = 0.50
    thrust_max: float = 0.72
    control_mix_mode: str = "ratio"
    policy_ratio: float = 0.20
    residual_gain: float = 0.8
    pd_feedback_scale: float = 1.0
    goal_feedback_scale: Optional[float] = 1.0
    attitude_feedback_scale: float = 1.0
    xy_control_mode: str = "body"
    goal_xy_pos_gain: float = 0.025
    xy_velocity_damping_gain: float = 0.100
    xy_target_velocity_gain: float = 0.100
    xy_target_accel_gain: float = 0.080
    xy_max_tilt_cmd: float = 0.16
    z_feedback_scale: float = 1.0
    z_pos_gain: float = 0.040
    z_vel_gain: float = 0.060
    z_target_velocity_gain: float = 0.0
    z_target_accel_gain: float = 0.0

    def __post_init__(self) -> None:
        self.control_mix_mode = str(self.control_mix_mode).lower()
        if self.control_mix_mode not in {"ratio", "additive"}:
            raise ValueError(
                f"control_mix_mode must be 'ratio' or 'additive', got {self.control_mix_mode!r}"
            )
        if not 0.0 <= float(self.policy_ratio) <= 1.0:
            raise ValueError(f"policy_ratio must be in [0, 1], got {self.policy_ratio}")
        self.xy_control_mode = str(self.xy_control_mode).lower()
        if self.xy_control_mode not in {"body", "legacy"}:
            raise ValueError(
                f"xy_control_mode must be 'body' or 'legacy', got {self.xy_control_mode!r}"
            )


@dataclass
class SafetyLimits:
    min_altitude: float = 0.35
    max_altitude: float = 11.0
    max_tilt_deg: float = 55.0
    max_body_rate: float = 4.0
    max_down_speed: float = 3.0
    max_xy_from_home: float = 5.5
    max_z_error_from_home: float = 4.0


@dataclass
class RotorCTBRBackendConfig(BackendConfig):
    mass_kg: float = 1.52
    gravity_mps2: float = 9.81
    inertia_diag: Tuple[float, float, float] = (0.029125, 0.029125, 0.055225)
    hover_thrust_command: float = 0.60
    max_force_ratio: float = 2.0
    rate_kp: Tuple[float, float, float] = (0.52, 0.52, 0.18)
    max_torque: Tuple[float, float, float] = (2.0, 2.0, 0.5)
    motor_time_constant: float = 0.025
    idle_rotor_velocity: float = 80.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def wrap_angle_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def enu_to_ned(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return np.array([v[1], v[0], -v[2]], dtype=np.float64)


def ned_to_enu(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return np.array([v[1], v[0], -v[2]], dtype=np.float64)


def frd_to_flu(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float64)


def rotate_world_xy_to_policy_frame(
    x: float,
    y: float,
    current_yaw: float,
    reference_yaw: Optional[float],
) -> Tuple[float, float]:
    if reference_yaw is None:
        return float(x), float(y)
    delta_yaw = wrap_angle_pi(float(current_yaw) - float(reference_yaw))
    c = math.cos(delta_yaw)
    s = math.sin(delta_yaw)
    return c * float(x) + s * float(y), -s * float(x) + c * float(y)


def map_policy_action_to_ctbr(action: Sequence[float], limits: CTBRActionLimits) -> Tuple[float, float, float, float]:
    if len(action) != 4:
        raise ValueError(f"CTBR policy action must have shape (4,), got {len(action)}")
    a = [clamp(float(v), -1.0, 1.0) for v in action]
    roll_rate = a[0] * limits.max_roll_rate
    pitch_rate = a[1] * limits.max_pitch_rate
    yaw_rate = a[2] * limits.max_yaw_rate
    thrust = limits.hover_thrust + a[3] * limits.thrust_delta
    return roll_rate, pitch_rate, yaw_rate, clamp(thrust, limits.thrust_min, limits.thrust_max)


def observation_vector(
    own: ObservationData,
    goal: GoalPoint,
    target_position: Sequence[float],
    target_velocity: Sequence[float],
    target_acceleration: Sequence[float],
    prev_command: np.ndarray,
    action_limits: CTBRActionLimits,
    yaw_reference: Optional[float] = None,
    future_reference_positions: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Build policy inputs from vehicle/target state and the task reference.

    Success flags, phase labels, trajectory progress and dwell counters are
    intentionally excluded. The helper controller may use those internals, but
    the policy only receives measurable state and the requested following point.
    """
    own_pos = np.array([own.x, own.y, own.z], dtype=np.float32) / 10.0
    own_vx, own_vy = rotate_world_xy_to_policy_frame(own.vx, own.vy, own.yaw, yaw_reference)
    own_vel = np.array([own_vx, own_vy, own.vz], dtype=np.float32) / 3.0
    own_ax, own_ay = rotate_world_xy_to_policy_frame(own.ax, own.ay, own.yaw, yaw_reference)
    own_accel = np.array([own_ax, own_ay, own.az], dtype=np.float32) / 5.0
    policy_yaw = own.yaw if yaw_reference is None else wrap_angle_pi(own.yaw - yaw_reference)
    own_att = np.array(
        [own.roll / math.pi, own.pitch / math.pi, math.sin(policy_yaw), math.cos(policy_yaw)],
        dtype=np.float32,
    )
    own_rates = np.array([own.rollspeed, own.pitchspeed, own.yawspeed], dtype=np.float32) / 4.0

    goal_dx, goal_dy = rotate_world_xy_to_policy_frame(goal.x - own.x, goal.y - own.y, own.yaw, yaw_reference)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    target_acceleration = np.asarray(target_acceleration, dtype=np.float64)
    if target_position.shape != (3,) or target_velocity.shape != (3,) or target_acceleration.shape != (3,):
        raise ValueError("target position, velocity and acceleration must each have shape (3,)")
    target_dx, target_dy = rotate_world_xy_to_policy_frame(
        target_position[0] - own.x,
        target_position[1] - own.y,
        own.yaw,
        yaw_reference,
    )
    target_vx, target_vy = rotate_world_xy_to_policy_frame(
        target_velocity[0], target_velocity[1], own.yaw, yaw_reference
    )
    target_ax, target_ay = rotate_world_xy_to_policy_frame(
        target_acceleration[0], target_acceleration[1], own.yaw, yaw_reference
    )
    goal_rel = np.array([goal_dx, goal_dy, goal.z - own.z], dtype=np.float32) / 5.0
    target_rel_pos = np.array(
        [target_dx, target_dy, target_position[2] - own.z], dtype=np.float32
    ) / 5.0
    target_vel = np.array([target_vx, target_vy, target_velocity[2]], dtype=np.float32) / 3.0
    target_accel = np.array(
        [target_ax, target_ay, target_acceleration[2]], dtype=np.float32
    ) / 3.0
    prev = np.asarray(prev_command, dtype=np.float32).copy()
    if prev.shape == (4,):
        prev = np.array(
            [
                prev[0] / max(1e-6, action_limits.max_roll_rate),
                prev[1] / max(1e-6, action_limits.max_pitch_rate),
                prev[2] / max(1e-6, action_limits.max_yaw_rate),
                (prev[3] - action_limits.hover_thrust) / max(1e-6, action_limits.thrust_delta),
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"prev_command must have shape (4,), got {prev.shape}")
    components = [
        own_pos,
        own_vel,
        own_accel,
        own_att,
        own_rates,
        goal_rel,
        target_rel_pos,
        target_vel,
        target_accel,
        prev,
    ]
    if future_reference_positions is not None:
        components.append(
            future_reference_vector(
                own,
                future_reference_positions,
                yaw_reference=yaw_reference,
            )
        )
    vec = np.concatenate(components).astype(np.float32)
    return np.clip(vec, -5.0, 5.0).astype(np.float32)


def future_reference_vector(
    own: ObservationData,
    future_reference_positions: Sequence[Sequence[float]],
    yaw_reference: Optional[float] = None,
) -> np.ndarray:
    """Encode simulator trajectory preview for privileged critic input."""
    future = np.asarray(future_reference_positions, dtype=np.float64)
    if future.ndim != 2 or future.shape[1] != 3:
        raise ValueError(
            "future_reference_positions must have shape (horizon, 3), "
            f"got {future.shape}"
        )
    future_relative = np.zeros_like(future, dtype=np.float32)
    for index, reference in enumerate(future):
        dx, dy = rotate_world_xy_to_policy_frame(
            reference[0] - own.x,
            reference[1] - own.y,
            own.yaw,
            yaw_reference,
        )
        future_relative[index] = [dx, dy, reference[2] - own.z]
    return np.clip(future_relative.reshape(-1) / 5.0, -5.0, 5.0).astype(np.float32)


def goal_distance(obs: ObservationData, goal: GoalPoint) -> float:
    return math.sqrt((obs.x - goal.x) ** 2 + (obs.y - goal.y) ** 2 + (obs.z - goal.z) ** 2)


class RotorCTBRBackend(Backend):
    """No-MAVLink backend that converts CTBR body-rate/thrust commands to rotor speeds."""

    def __init__(self, config: Optional[RotorCTBRBackendConfig] = None):
        super().__init__(config or RotorCTBRBackendConfig())
        self._state = None
        self._command_rates_frd = np.zeros(3, dtype=np.float64)
        self._command_thrust = float(self.config.hover_thrust_command)
        self._rotor_velocity = np.full(4, float(self.config.idle_rotor_velocity), dtype=np.float64)
        self._last_rotor_reference = self._rotor_velocity.copy()

    def set_ctbr_command(self, roll_rate: float, pitch_rate: float, yaw_rate: float, thrust: float) -> None:
        self._command_rates_frd[:] = [float(roll_rate), float(pitch_rate), float(yaw_rate)]
        self._command_thrust = float(thrust)

    def set_safe_command(self) -> None:
        self.set_ctbr_command(0.0, 0.0, 0.0, self.config.hover_thrust_command)

    def input_reference(self):
        return self._last_rotor_reference.tolist()

    def update_state(self, state):
        self._state = state

    def update(self, dt: float):
        if self._vehicle is None or self._state is None:
            return

        cfg = self.config
        hover = max(1e-6, float(cfg.hover_thrust_command))
        total_force = cfg.mass_kg * cfg.gravity_mps2 * clamp(self._command_thrust, 0.0, 1.0) / hover
        max_force = cfg.mass_kg * cfg.gravity_mps2 * cfg.max_force_ratio
        total_force = clamp(total_force, 0.0, max_force)

        current_rates_frd = frd_to_flu(self._state.angular_velocity)
        rate_error = self._command_rates_frd - current_rates_frd
        rate_kp = np.asarray(cfg.rate_kp, dtype=np.float64)
        torque_frd = rate_kp * rate_error
        max_torque = np.asarray(cfg.max_torque, dtype=np.float64)
        torque_frd = np.clip(torque_frd, -max_torque, max_torque)
        torque_flu = frd_to_flu(torque_frd)

        try:
            target = np.asarray(self._vehicle.force_and_torques_to_velocities(total_force, torque_flu), dtype=np.float64)
        except Exception:
            target = np.full(4, math.sqrt(max(total_force / 4.0, 0.0) / 8.54858e-6), dtype=np.float64)

        tau = max(0.0, float(cfg.motor_time_constant))
        if tau > 1e-6 and dt > 0.0:
            alpha = 1.0 - math.exp(-float(dt) / tau)
            self._rotor_velocity += alpha * (target - self._rotor_velocity)
        else:
            self._rotor_velocity[:] = target
        self._last_rotor_reference = np.maximum(self._rotor_velocity, float(cfg.idle_rotor_velocity))

    def update_sensor(self, sensor_type: str, data):
        return None

    def update_graphical_sensor(self, sensor_type: str, data):
        return None

    def start(self):
        self.set_safe_command()

    def stop(self):
        self._last_rotor_reference[:] = 0.0

    def reset(self):
        self.set_safe_command()
        self._rotor_velocity[:] = float(self.config.idle_rotor_velocity)
        self._last_rotor_reference[:] = self._rotor_velocity
