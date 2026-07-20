from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def mix_ctbr_commands(
    controller_command: Sequence[float],
    policy_command: Sequence[float],
    policy_ratio: float,
) -> np.ndarray:
    """Mix helper and policy CTBR commands with one fixed convex ratio."""
    controller = np.asarray(controller_command, dtype=np.float64)
    policy = np.asarray(policy_command, dtype=np.float64)
    ratio = float(policy_ratio)
    if controller.shape != (4,) or policy.shape != (4,):
        raise ValueError("controller_command and policy_command must have shape (4,)")
    if not np.all(np.isfinite(controller)) or not np.all(np.isfinite(policy)):
        raise ValueError("CTBR commands must contain finite values")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("policy_ratio must be finite and in [0, 1]")
    return (1.0 - ratio) * controller + ratio * policy


def wrap_angle_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def vehicle_yaw_for_target_bearing(
    target_world_yaw: float,
    target_bearing: float,
) -> float:
    """Choose vehicle yaw so a world direction appears at a body-frame bearing."""
    return wrap_angle_pi(float(target_world_yaw) - float(target_bearing))


def quaternion_ned_frd_to_euler(
    quaternion_xyzw: Sequence[float],
) -> Tuple[float, float, float]:
    """Return aerospace roll, pitch and yaw from a body-FRD to world-NED quaternion."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    yaw, pitch, roll = Rotation.from_quat(quaternion).as_euler(
        "ZYX",
        degrees=False,
    )
    return float(roll), float(pitch), float(wrap_angle_pi(yaw))


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


def xy_tracking_attitude_targets(
    position_error_ned_xy: Sequence[float],
    own_velocity_ned_xy: Sequence[float],
    target_velocity_ned_xy: Sequence[float],
    target_acceleration_ned_xy: Sequence[float],
    current_yaw: float,
    goal_feedback_scale: float,
    position_gain: float,
    velocity_damping_gain: float,
    target_velocity_gain: float,
    target_acceleration_gain: float,
    max_tilt_cmd: float,
    control_mode: str = "body",
) -> Tuple[float, float]:
    """Map world-frame XY tracking errors to yaw-invariant body attitude targets."""
    vectors = tuple(
        np.asarray(vector, dtype=np.float64)
        for vector in (
            position_error_ned_xy,
            own_velocity_ned_xy,
            target_velocity_ned_xy,
            target_acceleration_ned_xy,
        )
    )
    if any(vector.shape != (2,) for vector in vectors):
        raise ValueError("all helper XY vectors must have shape (2,)")
    if any(not np.all(np.isfinite(vector)) for vector in vectors):
        raise ValueError("all helper XY vectors must contain finite values")
    if control_mode not in {"body", "legacy"}:
        raise ValueError(f"unsupported XY control mode: {control_mode!r}")

    body_vectors = [
        rotate_world_xy_to_policy_frame(
            float(vector[0]),
            float(vector[1]),
            current_yaw,
            0.0,
        )
        for vector in vectors
    ]
    (x_err, y_err), (vx, vy), (target_vx, target_vy), (target_ax, target_ay) = (
        body_vectors
    )
    x_tilt_term = (
        goal_feedback_scale * position_gain * x_err
        + velocity_damping_gain * vx
        - target_velocity_gain * target_vx
        - target_acceleration_gain * target_ax
    )
    y_tilt_term = (
        goal_feedback_scale * position_gain * y_err
        + velocity_damping_gain * vy
        - target_velocity_gain * target_vy
        - target_acceleration_gain * target_ay
    )
    if control_mode == "legacy":
        roll_des = np.clip(x_tilt_term, -max_tilt_cmd, max_tilt_cmd)
        pitch_des = np.clip(y_tilt_term, -max_tilt_cmd, max_tilt_cmd)
    else:
        pitch_des = np.clip(x_tilt_term, -max_tilt_cmd, max_tilt_cmd)
        roll_des = np.clip(-y_tilt_term, -max_tilt_cmd, max_tilt_cmd)
    return float(roll_des), float(pitch_des)
