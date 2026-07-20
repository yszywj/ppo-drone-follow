from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


CAMERA_OBSERVATION_NAMES = (
    "image_u_normalized",
    "image_v_normalized",
    "forward_depth_normalized",
    "range_3d_normalized",
    "visible",
    "center_quality",
)
CAMERA_OBSERVATION_DIM = len(CAMERA_OBSERVATION_NAMES)


@dataclass(frozen=True)
class CameraModelConfig:
    """Pinhole camera fixed to a vehicle with NED/FRD state conventions.

    Camera coordinates are x-forward, y-right and z-down. Mount yaw is
    positive to the right, mount pitch is positive down, and mount roll is a
    right-handed rotation about the optical axis.
    """

    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    mount_roll_deg: float = 0.0
    mount_pitch_down_deg: float = 0.0
    mount_yaw_right_deg: float = 0.0
    near_clip_m: float = 0.10
    far_clip_m: float = 20.0
    success_margin: float = 0.90
    center_sigma_u: float = 0.50
    center_sigma_v: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < float(self.horizontal_fov_deg) < 180.0:
            raise ValueError("camera horizontal FOV must be in (0, 180) degrees")
        if not 0.0 < float(self.vertical_fov_deg) < 180.0:
            raise ValueError("camera vertical FOV must be in (0, 180) degrees")
        if float(self.near_clip_m) < 0.0:
            raise ValueError("camera near clip must be non-negative")
        if float(self.far_clip_m) <= float(self.near_clip_m):
            raise ValueError("camera far clip must exceed near clip")
        if not 0.0 < float(self.success_margin) <= 1.0:
            raise ValueError("camera success margin must be in (0, 1]")
        if float(self.center_sigma_u) <= 0.0 or float(self.center_sigma_v) <= 0.0:
            raise ValueError("camera center sigmas must be positive")


@dataclass(frozen=True)
class CameraProjection:
    camera_x_m: float
    camera_y_m: float
    camera_z_m: float
    range_m: float
    normalized_u: float
    normalized_v: float
    bearing_rad: float
    elevation_rad: float
    in_front: bool
    visible: bool
    success_region: bool
    center_quality: float


def camera_observation_vector(
    projection: CameraProjection,
    config: CameraModelConfig,
) -> np.ndarray:
    """Build detector-style camera features without exposing raw pixels.

    Horizontal and vertical positions are normalized by the image half-width
    and half-height. Optical-axis depth and 3-D range are normalized to the
    configured camera range. An invisible target has zero-valued measurements;
    the visibility bit disambiguates that case from a centered detection.
    """
    if not projection.visible:
        return np.zeros(CAMERA_OBSERVATION_DIM, dtype=np.float32)
    depth_span = max(1e-6, float(config.far_clip_m) - float(config.near_clip_m))
    normalized_depth = (
        float(projection.camera_x_m) - float(config.near_clip_m)
    ) / depth_span
    normalized_range = float(projection.range_m) / max(
        1e-6,
        float(config.far_clip_m),
    )
    return np.asarray(
        [
            np.clip(projection.normalized_u, -1.0, 1.0),
            np.clip(projection.normalized_v, -1.0, 1.0),
            np.clip(normalized_depth, 0.0, 1.0),
            np.clip(normalized_range, 0.0, 1.0),
            1.0,
            np.clip(projection.center_quality, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def project_target_to_camera(
    own_position_ned: Sequence[float],
    attitude_body_to_ned_xyzw: Sequence[float],
    target_position_ned: Sequence[float],
    config: CameraModelConfig,
) -> CameraProjection:
    """Project a NED target through a body-FRD-mounted pinhole camera."""
    own = np.asarray(own_position_ned, dtype=np.float64)
    target = np.asarray(target_position_ned, dtype=np.float64)
    quaternion = np.asarray(attitude_body_to_ned_xyzw, dtype=np.float64)
    if own.shape != (3,) or target.shape != (3,):
        raise ValueError("camera positions must each have shape (3,)")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("camera attitude quaternion must have shape (4,) and be finite")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        raise ValueError("camera attitude quaternion must be non-zero")

    qx, qy, qz, qw = quaternion / quaternion_norm
    body_to_ned = np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    relative_body_frd = body_to_ned.T @ (target - own)

    # Active camera-to-body rotation. Positive mount pitch is defined as down,
    # hence the sign inversion for the right-handed y-axis rotation.
    yaw = math.radians(float(config.mount_yaw_right_deg))
    pitch = -math.radians(float(config.mount_pitch_down_deg))
    roll = math.radians(float(config.mount_roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rotate_z = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotate_y = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotate_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    camera_to_body = rotate_z @ rotate_y @ rotate_x
    camera = camera_to_body.T @ relative_body_frd
    camera_x, camera_y, camera_z = (float(value) for value in camera)
    target_range = float(np.linalg.norm(camera))
    in_front = bool(camera_x > max(1e-9, float(config.near_clip_m)))

    if in_front:
        tan_half_horizontal = math.tan(
            0.5 * math.radians(float(config.horizontal_fov_deg))
        )
        tan_half_vertical = math.tan(
            0.5 * math.radians(float(config.vertical_fov_deg))
        )
        normalized_u = camera_y / (camera_x * tan_half_horizontal)
        normalized_v = camera_z / (camera_x * tan_half_vertical)
        bearing = math.atan2(camera_y, camera_x)
        elevation = math.atan2(camera_z, camera_x)
    else:
        normalized_u = math.copysign(math.inf, camera_y) if camera_y else 0.0
        normalized_v = math.copysign(math.inf, camera_z) if camera_z else 0.0
        bearing = math.atan2(camera_y, camera_x)
        elevation = math.atan2(camera_z, camera_x)

    depth_valid = (
        float(config.near_clip_m)
        <= camera_x
        <= float(config.far_clip_m)
    )
    visible = bool(
        in_front
        and depth_valid
        and abs(normalized_u) <= 1.0 + 1e-9
        and abs(normalized_v) <= 1.0 + 1e-9
    )
    success_region = bool(
        in_front
        and depth_valid
        and abs(normalized_u) <= float(config.success_margin) + 1e-9
        and abs(normalized_v) <= float(config.success_margin) + 1e-9
    )
    if in_front and depth_valid:
        bounded_u = float(np.clip(normalized_u, -4.0, 4.0))
        bounded_v = float(np.clip(normalized_v, -4.0, 4.0))
        center_quality = math.exp(
            -0.5
            * (
                (bounded_u / float(config.center_sigma_u)) ** 2
                + (bounded_v / float(config.center_sigma_v)) ** 2
            )
        )
    else:
        center_quality = 0.0

    return CameraProjection(
        camera_x_m=camera_x,
        camera_y_m=camera_y,
        camera_z_m=camera_z,
        range_m=target_range,
        normalized_u=float(normalized_u),
        normalized_v=float(normalized_v),
        bearing_rad=float(bearing),
        elevation_rad=float(elevation),
        in_front=in_front,
        visible=visible,
        success_region=success_region,
        center_quality=float(center_quality),
    )


def camera_centering_yaw_rate(
    bearing_rad: float,
    current_yaw_rate_rad_s: float,
    proportional_gain: float,
    damping_gain: float,
    max_rate_rad_s: float,
    deadband_rad: float = 0.0,
) -> float:
    """Return a body-FRD yaw-rate command that centers a target horizontally."""
    values = (
        bearing_rad,
        current_yaw_rate_rad_s,
        proportional_gain,
        damping_gain,
        max_rate_rad_s,
        deadband_rad,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("camera yaw-helper inputs must be finite")
    if proportional_gain < 0.0 or damping_gain < 0.0:
        raise ValueError("camera yaw-helper gains must be non-negative")
    if max_rate_rad_s < 0.0 or deadband_rad < 0.0:
        raise ValueError("camera yaw-helper limits must be non-negative")
    wrapped_bearing = math.atan2(math.sin(bearing_rad), math.cos(bearing_rad))
    effective_magnitude = max(0.0, abs(wrapped_bearing) - deadband_rad)
    effective_bearing = math.copysign(effective_magnitude, wrapped_bearing)
    command = (
        proportional_gain * effective_bearing
        - damping_gain * current_yaw_rate_rad_s
    )
    return float(np.clip(command, -max_rate_rad_s, max_rate_rad_s))


@dataclass(frozen=True)
class CameraWindowSnapshot:
    ready: bool
    reward_event: bool
    visible_fraction: float
    success_fraction: float
    mean_center_quality: float
    joint_quality: float


class CameraQualityWindow:
    """Causal image-space quality window, independent of motion primitives."""

    def __init__(self, window_steps: int, interval_steps: int) -> None:
        self.window_steps = max(1, int(window_steps))
        self.interval_steps = max(1, int(interval_steps))
        self._visible: deque[bool] = deque(maxlen=self.window_steps)
        self._success: deque[bool] = deque(maxlen=self.window_steps)
        self._center_quality: deque[float] = deque(maxlen=self.window_steps)
        self._active_steps = 0

    def reset(self) -> None:
        self._visible.clear()
        self._success.clear()
        self._center_quality.clear()
        self._active_steps = 0

    def append(
        self,
        visible: bool,
        success_region: bool,
        center_quality: float,
    ) -> CameraWindowSnapshot:
        self._visible.append(bool(visible))
        self._success.append(bool(success_region))
        self._center_quality.append(float(np.clip(center_quality, 0.0, 1.0)))
        self._active_steps += 1
        ready = len(self._visible) == self.window_steps
        reward_event = ready and self._active_steps % self.interval_steps == 0
        if not ready:
            return CameraWindowSnapshot(False, False, 0.0, 0.0, 0.0, 0.0)
        denominator = float(self.window_steps)
        visible_fraction = sum(self._visible) / denominator
        success_fraction = sum(self._success) / denominator
        mean_center_quality = sum(self._center_quality) / denominator
        joint_quality = mean_center_quality * success_fraction
        return CameraWindowSnapshot(
            ready=True,
            reward_event=reward_event,
            visible_fraction=float(visible_fraction),
            success_fraction=float(success_fraction),
            mean_center_quality=float(mean_center_quality),
            joint_quality=float(joint_quality),
        )
