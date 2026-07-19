from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def corrective_velocity_reference(
    goal_velocity: Sequence[float],
    position_error: Sequence[float],
    gain: float,
    max_correction_speed: float,
) -> Tuple[np.ndarray, float]:
    """Return goal velocity plus a bounded velocity that closes position error."""
    goal_velocity_array = np.asarray(goal_velocity, dtype=np.float64)
    position_error_array = np.asarray(position_error, dtype=np.float64)
    correction = max(0.0, float(gain)) * position_error_array
    correction_speed = float(np.linalg.norm(correction))
    correction_limit = max(0.0, float(max_correction_speed))
    if correction_speed > correction_limit and correction_speed > 1e-12:
        correction *= correction_limit / correction_speed
        correction_speed = correction_limit
    return goal_velocity_array + correction, correction_speed


def position_reward_qualities(
    xy_error: float,
    z_error: float,
    position_sigma: float,
    z_sigma: float,
    recovery_position_sigma: float,
    recovery_z_sigma: float,
) -> Tuple[float, float]:
    """Return a precise Gaussian quality and a broad recovery quality."""
    position_sigma = max(1e-3, float(position_sigma))
    z_sigma = max(1e-3, float(z_sigma))
    recovery_position_sigma = max(1e-3, float(recovery_position_sigma))
    recovery_z_sigma = max(1e-3, float(recovery_z_sigma))
    precise_quality = math.exp(
        -((float(xy_error) / position_sigma) ** 2)
        -((float(z_error) / z_sigma) ** 2)
    )
    recovery_quality = 1.0 / (
        1.0
        + (float(xy_error) / recovery_position_sigma) ** 2
        + (float(z_error) / recovery_z_sigma) ** 2
    )
    return precise_quality, recovery_quality


def clipped_error_progress(
    previous_error: float,
    current_error: float,
    max_abs_progress: float,
) -> float:
    """Return bounded positive progress when an error decreases."""
    limit = max(0.0, float(max_abs_progress))
    return float(
        np.clip(float(previous_error) - float(current_error), -limit, limit)
    )
