from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


def seconds_to_steps(seconds: float, step_dt_sec: float) -> int:
    """Convert a duration to policy steps without shortening the duration."""
    step_dt_sec = float(step_dt_sec)
    if step_dt_sec <= 0.0:
        raise ValueError("step_dt_sec must be greater than zero")
    return max(1, int(math.ceil(max(0.0, float(seconds)) / step_dt_sec)))


@dataclass(frozen=True)
class TrackingWindowSnapshot:
    ready: bool
    reward_event: bool
    soft_joint_quality: float
    xy_good_fraction: float
    velocity_good_fraction: float
    z_good_fraction: float
    drift_delta_m: float


class TrackingQualityWindow:
    """Causal fixed-time tracking history independent of trajectory primitives."""

    def __init__(
        self,
        window_steps: int,
        interval_steps: int,
        xy_tolerance_m: float,
        velocity_tolerance_mps: float,
        z_tolerance_m: float,
        xy_sigma_m: float,
        velocity_sigma_mps: float,
        z_sigma_m: float,
    ) -> None:
        self.window_steps = max(1, int(window_steps))
        self.interval_steps = max(1, int(interval_steps))
        self.xy_tolerance_m = max(1e-6, float(xy_tolerance_m))
        self.velocity_tolerance_mps = max(
            1e-6, float(velocity_tolerance_mps)
        )
        self.z_tolerance_m = max(1e-6, float(z_tolerance_m))
        self.xy_sigma_m = max(1e-6, float(xy_sigma_m))
        self.velocity_sigma_mps = max(1e-6, float(velocity_sigma_mps))
        self.z_sigma_m = max(1e-6, float(z_sigma_m))
        self._xy_errors: deque[float] = deque(maxlen=self.window_steps)
        self._velocity_errors: deque[float] = deque(maxlen=self.window_steps)
        self._z_errors: deque[float] = deque(maxlen=self.window_steps)
        self._active_steps = 0

    def reset(self) -> None:
        self._xy_errors.clear()
        self._velocity_errors.clear()
        self._z_errors.clear()
        self._active_steps = 0

    def append(
        self,
        xy_error_m: float,
        corrected_velocity_error_mps: float,
        z_error_m: float,
    ) -> TrackingWindowSnapshot:
        self._xy_errors.append(max(0.0, float(xy_error_m)))
        self._velocity_errors.append(
            max(0.0, float(corrected_velocity_error_mps))
        )
        self._z_errors.append(max(0.0, float(z_error_m)))
        self._active_steps += 1
        ready = len(self._xy_errors) == self.window_steps
        reward_event = ready and self._active_steps % self.interval_steps == 0
        if not ready:
            return TrackingWindowSnapshot(
                ready=False,
                reward_event=False,
                soft_joint_quality=0.0,
                xy_good_fraction=0.0,
                velocity_good_fraction=0.0,
                z_good_fraction=0.0,
                drift_delta_m=0.0,
            )

        xy_values = tuple(self._xy_errors)
        velocity_values = tuple(self._velocity_errors)
        z_values = tuple(self._z_errors)
        soft_joint_quality = sum(
            math.exp(
                -((xy / self.xy_sigma_m) ** 2)
                -((velocity / self.velocity_sigma_mps) ** 2)
                -((z / self.z_sigma_m) ** 2)
            )
            for xy, velocity, z in zip(
                xy_values, velocity_values, z_values
            )
        ) / float(self.window_steps)
        xy_good_fraction = sum(
            value <= self.xy_tolerance_m for value in xy_values
        ) / float(self.window_steps)
        velocity_good_fraction = sum(
            value <= self.velocity_tolerance_mps for value in velocity_values
        ) / float(self.window_steps)
        z_good_fraction = sum(
            value <= self.z_tolerance_m for value in z_values
        ) / float(self.window_steps)
        return TrackingWindowSnapshot(
            ready=True,
            reward_event=reward_event,
            soft_joint_quality=float(soft_joint_quality),
            xy_good_fraction=float(xy_good_fraction),
            velocity_good_fraction=float(velocity_good_fraction),
            z_good_fraction=float(z_good_fraction),
            drift_delta_m=float(xy_values[-1] - xy_values[0]),
        )


def tracking_event_rewards(
    snapshot: TrackingWindowSnapshot,
    hard_fraction_weight: float,
    drift_deadband_m: float,
    xy_tolerance_m: float,
    tracking_scale: float,
    drift_scale: float,
) -> tuple[float, float]:
    if not snapshot.reward_event:
        return 0.0, 0.0
    hard_weight = min(1.0, max(0.0, float(hard_fraction_weight)))
    hard_joint_fraction = min(
        snapshot.xy_good_fraction,
        snapshot.velocity_good_fraction,
        snapshot.z_good_fraction,
    )
    quality = (
        (1.0 - hard_weight) * snapshot.soft_joint_quality
        + hard_weight * hard_joint_fraction
    )
    tracking_reward = max(0.0, float(tracking_scale)) * quality
    drift_excess = max(
        0.0,
        snapshot.drift_delta_m - max(0.0, float(drift_deadband_m)),
    )
    normalized_drift = drift_excess / max(1e-6, float(xy_tolerance_m))
    drift_reward = -max(0.0, float(drift_scale)) * min(
        2.0, normalized_drift
    )
    return float(tracking_reward), float(drift_reward)
