from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PRIMITIVE_CODES = {
    "capture": 0,
    "hold": 1,
    "accelerate": 2,
    "cruise": 3,
    "decelerate": 4,
    "turn": 5,
    "climb": 6,
    "descend": 7,
    "final_stop": 8,
    "stopped": 9,
}


def _range(value: Any, name: str, *, non_negative: bool = False) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        lo = hi = float(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    else:
        raise ValueError(f"{name} must be a number or [min, max]")
    if lo > hi:
        raise ValueError(f"{name} minimum must not exceed maximum")
    if non_negative and lo < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return lo, hi


def _sample_range(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return lo if abs(hi - lo) <= 1e-12 else float(rng.uniform(lo, hi))


def _move_towards(current: float, target: float, max_delta: float) -> float:
    delta = float(np.clip(target - current, -max_delta, max_delta))
    return current + delta


@dataclass(frozen=True)
class PrimitiveTemplate:
    primitive_id: str
    weight: float
    duration_sec: Tuple[float, float]
    target_speed_mps: Tuple[float, float]
    acceleration_mps2: Tuple[float, float]
    curvature_per_m: Tuple[float, float]
    min_abs_curvature_per_m: float
    vertical_speed_mps: Tuple[float, float]
    min_heading_change_deg: float
    min_vertical_displacement_m: float
    min_speed_change_mps: float

    @classmethod
    def from_dict(cls, primitive_id: str, data: Mapping[str, Any]) -> "PrimitiveTemplate":
        if primitive_id not in PRIMITIVE_CODES or primitive_id in ("capture", "final_stop", "stopped"):
            raise ValueError(f"unsupported selectable primitive id: {primitive_id!r}")
        weight = float(data.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"primitive {primitive_id!r} weight must be non-negative")
        return cls(
            primitive_id=primitive_id,
            weight=weight,
            duration_sec=_range(data.get("duration_sec", [2.0, 4.0]), f"{primitive_id}.duration_sec", non_negative=True),
            target_speed_mps=_range(data.get("target_speed_mps", [0.2, 0.4]), f"{primitive_id}.target_speed_mps", non_negative=True),
            acceleration_mps2=_range(data.get("acceleration_mps2", [0.15, 0.30]), f"{primitive_id}.acceleration_mps2", non_negative=True),
            curvature_per_m=_range(data.get("curvature_per_m", 0.0), f"{primitive_id}.curvature_per_m"),
            min_abs_curvature_per_m=max(0.0, float(data.get("min_abs_curvature_per_m", 0.0))),
            vertical_speed_mps=_range(data.get("vertical_speed_mps", 0.0), f"{primitive_id}.vertical_speed_mps"),
            min_heading_change_deg=max(
                0.0, float(data.get("min_heading_change_deg", 0.0))
            ),
            min_vertical_displacement_m=max(
                0.0, float(data.get("min_vertical_displacement_m", 0.0))
            ),
            min_speed_change_mps=max(
                0.0, float(data.get("min_speed_change_mps", 0.0))
            ),
        )


@dataclass(frozen=True)
class MotionLimits:
    max_speed_mps: float = 0.60
    max_acceleration_mps2: float = 0.40
    max_jerk_mps3: float = 1.0
    max_curvature_per_m: float = 0.80
    max_curvature_rate_per_m_sec: float = 0.80
    max_yaw_rate_rad_s: float = 0.60
    max_lateral_acceleration_mps2: float = 0.50
    max_vertical_speed_mps: float = 0.30
    max_vertical_acceleration_mps2: float = 0.30
    max_vertical_jerk_mps3: float = 0.80
    max_horizontal_radius_m: float = 5.0
    max_vertical_displacement_m: float = 2.0
    speed_response_time_sec: float = 0.60
    vertical_response_time_sec: float = 0.60
    final_brake_max_sec: float = 6.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotionLimits":
        values = {field: float(data.get(field, getattr(cls(), field))) for field in cls.__dataclass_fields__}
        for key, value in values.items():
            if value <= 0.0:
                raise ValueError(f"motion limit {key!r} must be positive")
        return cls(**values)


@dataclass(frozen=True)
class MotionPoolConfig:
    enabled: bool
    min_segments: int
    max_segments: int
    primitive_ids: Tuple[str, ...]
    prefix_ids: Tuple[str, ...]
    required_ids: Tuple[str, ...]
    required_one_of_ids: Tuple[str, ...]
    prevent_consecutive_same: bool
    stop_hold_sec: float
    reference_horizon_sec: Tuple[float, ...]
    max_resample_attempts: int
    limits: MotionLimits
    templates: Mapping[str, PrimitiveTemplate]

    @classmethod
    def disabled(cls) -> "MotionPoolConfig":
        return cls(
            enabled=False,
            min_segments=0,
            max_segments=0,
            primitive_ids=(),
            prefix_ids=(),
            required_ids=(),
            required_one_of_ids=(),
            prevent_consecutive_same=True,
            stop_hold_sec=2.0,
            reference_horizon_sec=(),
            max_resample_attempts=1,
            limits=MotionLimits(),
            templates={},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionPoolConfig":
        if not data or not bool(data.get("enabled", False)):
            return cls.disabled()
        library = data.get("primitive_library", {})
        if not isinstance(library, Mapping):
            raise ValueError("motion_pool.primitive_library must be an object")
        templates = {
            str(primitive_id): PrimitiveTemplate.from_dict(str(primitive_id), spec)
            for primitive_id, spec in library.items()
        }
        primitive_ids = tuple(str(value) for value in data.get("primitive_ids", ()))
        prefix_ids = tuple(str(value) for value in data.get("prefix_ids", ()))
        required_ids = tuple(str(value) for value in data.get("required_ids", ()))
        required_one_of_ids = tuple(
            str(value) for value in data.get("required_one_of_ids", ())
        )
        if not primitive_ids and not prefix_ids:
            raise ValueError("motion_pool must contain primitive_ids or prefix_ids")
        missing = [
            value
            for value in (
                *primitive_ids,
                *prefix_ids,
                *required_ids,
                *required_one_of_ids,
            )
            if value not in templates
        ]
        if missing:
            raise ValueError(f"motion_pool references undefined primitive ids: {sorted(set(missing))}")
        if primitive_ids and sum(templates[value].weight for value in primitive_ids) <= 0.0:
            raise ValueError("at least one selectable primitive must have positive weight")
        if required_one_of_ids and sum(
            templates[value].weight for value in required_one_of_ids
        ) <= 0.0:
            raise ValueError(
                "motion_pool.required_one_of_ids must contain a positive-weight primitive"
            )
        if len(set(required_ids)) != len(required_ids):
            raise ValueError("motion_pool.required_ids must not contain duplicates")
        if len(set(required_one_of_ids)) != len(required_one_of_ids):
            raise ValueError(
                "motion_pool.required_one_of_ids must not contain duplicates"
            )
        overlap = sorted(set(required_ids).intersection(required_one_of_ids))
        if overlap:
            raise ValueError(
                "motion_pool.required_ids and required_one_of_ids must be disjoint; "
                f"overlap={overlap}"
            )
        min_segments = int(data.get("min_segments", 2))
        max_segments = int(data.get("max_segments", min_segments))
        if min_segments < 0 or max_segments < min_segments:
            raise ValueError("motion_pool segment limits are invalid")
        required_slots = len(required_ids) + int(bool(required_one_of_ids))
        if required_slots > min_segments:
            raise ValueError(
                "motion_pool required primitives cannot exceed min_segments"
            )
        reference_horizon = tuple(float(value) for value in data.get("reference_horizon_sec", ()))
        if any(value <= 0.0 for value in reference_horizon):
            raise ValueError("reference_horizon_sec entries must be positive")
        if tuple(sorted(reference_horizon)) != reference_horizon:
            raise ValueError("reference_horizon_sec must be sorted")
        stop_hold_sec = float(data.get("stop_hold_sec", 2.0))
        if stop_hold_sec <= 0.0:
            raise ValueError("motion_pool.stop_hold_sec must be positive")
        prevent_consecutive_same = bool(data.get("prevent_consecutive_same", True))
        if prevent_consecutive_same and any(
            left == right for left, right in zip(prefix_ids, prefix_ids[1:])
        ):
            raise ValueError(
                "motion_pool.prefix_ids contains consecutive duplicate ids while "
                "prevent_consecutive_same is enabled"
            )
        return cls(
            enabled=True,
            min_segments=min_segments,
            max_segments=max_segments,
            primitive_ids=primitive_ids,
            prefix_ids=prefix_ids,
            required_ids=required_ids,
            required_one_of_ids=required_one_of_ids,
            prevent_consecutive_same=prevent_consecutive_same,
            stop_hold_sec=stop_hold_sec,
            reference_horizon_sec=reference_horizon,
            max_resample_attempts=max(1, int(data.get("max_resample_attempts", 64))),
            limits=MotionLimits.from_dict(data.get("limits", {})),
            templates=templates,
        )


@dataclass(frozen=True)
class TrajectoryState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    goal_position: np.ndarray
    goal_velocity: np.ndarray
    goal_acceleration: np.ndarray
    heading_rad: float
    curvature_per_m: float
    primitive_id: str
    primitive_code: int
    segment_index: int
    segment_progress: float
    phase: str
    motion_progress: float


@dataclass
class GeneratedTrajectory:
    dt: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    goal_positions: np.ndarray
    goal_velocities: np.ndarray
    goal_accelerations: np.ndarray
    headings: np.ndarray
    curvatures: np.ndarray
    primitive_ids: List[str]
    primitive_codes: np.ndarray
    segment_indices: np.ndarray
    segment_progress: np.ndarray
    phases: List[str]
    motion_progress: np.ndarray
    sequence_ids: Tuple[str, ...]
    final_stop_index: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, float((len(self.positions) - 1) * self.dt))

    @property
    def reference_dim(self) -> int:
        return 3

    def index_at(self, time_sec: float) -> int:
        return int(np.clip(round(max(0.0, float(time_sec)) / self.dt), 0, len(self.positions) - 1))

    def state_at(self, time_sec: float) -> TrajectoryState:
        index = self.index_at(time_sec)
        return TrajectoryState(
            position=self.positions[index],
            velocity=self.velocities[index],
            acceleration=self.accelerations[index],
            goal_position=self.goal_positions[index],
            goal_velocity=self.goal_velocities[index],
            goal_acceleration=self.goal_accelerations[index],
            heading_rad=float(self.headings[index]),
            curvature_per_m=float(self.curvatures[index]),
            primitive_id=self.primitive_ids[index],
            primitive_code=int(self.primitive_codes[index]),
            segment_index=int(self.segment_indices[index]),
            segment_progress=float(self.segment_progress[index]),
            phase=self.phases[index],
            motion_progress=float(self.motion_progress[index]),
        )


class MotionTaskGenerator:
    def __init__(
        self,
        config: MotionPoolConfig,
        dt: float,
        follow_distance_m: float,
        follow_vertical_offset_m: float = 0.0,
    ):
        if dt <= 0.0:
            raise ValueError("trajectory dt must be positive")
        self.config = config
        self.dt = float(dt)
        self.follow_distance_m = max(0.0, float(follow_distance_m))
        self.follow_vertical_offset_m = float(follow_vertical_offset_m)

    def sample(
        self,
        rng: np.random.Generator,
        home_position: Sequence[float],
        initial_heading_rad: float,
    ) -> GeneratedTrajectory:
        if not self.config.enabled:
            raise RuntimeError("motion task generator is disabled")
        home = np.asarray(home_position, dtype=np.float64)
        tangent = np.array([math.cos(initial_heading_rad), math.sin(initial_heading_rad)], dtype=np.float64)
        target_start = home.copy()
        target_start[:2] += self.follow_distance_m * tangent
        last_error = "trajectory violated configured workspace limits"
        for _ in range(self.config.max_resample_attempts):
            sequence = self._sample_sequence(rng)
            trajectory = self._generate(rng, home, target_start, initial_heading_rad, sequence)
            if not self._within_limits(trajectory, home):
                last_error = "trajectory violated configured workspace limits"
                continue
            amplitudes_valid, amplitude_error = self._meets_template_amplitudes(
                trajectory
            )
            if amplitudes_valid:
                return trajectory
            last_error = amplitude_error
        raise RuntimeError(last_error)

    def _sample_sequence(self, rng: np.random.Generator) -> Tuple[str, ...]:
        sequence = list(self.config.prefix_ids)
        if self.config.primitive_ids:
            count = int(rng.integers(self.config.min_segments, self.config.max_segments + 1))
            weights = np.asarray(
                [self.config.templates[value].weight for value in self.config.primitive_ids],
                dtype=np.float64,
            )
            weights /= weights.sum()
            attempts = max(16, self.config.max_resample_attempts)
            for _ in range(attempts):
                sampled = list(self.config.required_ids)
                if self.config.required_one_of_ids:
                    required_weights = np.asarray(
                        [
                            self.config.templates[value].weight
                            for value in self.config.required_one_of_ids
                        ],
                        dtype=np.float64,
                    )
                    required_weights /= required_weights.sum()
                    sampled.append(
                        str(
                            rng.choice(
                                self.config.required_one_of_ids,
                                p=required_weights,
                            )
                        )
                    )
                sampled.extend(
                    str(rng.choice(self.config.primitive_ids, p=weights))
                    for _ in range(count - len(sampled))
                )
                rng.shuffle(sampled)
                candidate = sequence + sampled
                if not self.config.prevent_consecutive_same or all(
                    left != right for left, right in zip(candidate, candidate[1:])
                ):
                    return tuple(candidate)
            raise RuntimeError(
                "unable to sample a motion sequence without consecutive duplicate ids; "
                "adjust primitive weights, segment count, or disable "
                "prevent_consecutive_same"
            )
        return tuple(sequence)

    def _sample_targets(
        self,
        rng: np.random.Generator,
        primitive_id: str,
        current_speed: float,
        current_vertical_speed: float,
    ) -> Tuple[float, float, float, float, float]:
        template = self.config.templates[primitive_id]
        duration = max(self.dt, _sample_range(rng, template.duration_sec))
        target_speed = _sample_range(rng, template.target_speed_mps)
        acceleration = _sample_range(rng, template.acceleration_mps2)
        curvature = _sample_range(rng, template.curvature_per_m)
        if 0.0 < abs(curvature) < template.min_abs_curvature_per_m:
            curvature = math.copysign(template.min_abs_curvature_per_m, curvature)
        elif abs(curvature) <= 1e-12 and template.min_abs_curvature_per_m > 0.0:
            curvature = template.min_abs_curvature_per_m * (-1.0 if rng.random() < 0.5 else 1.0)
        vertical_speed = _sample_range(rng, template.vertical_speed_mps)
        if primitive_id == "hold":
            target_speed = current_speed
            curvature = 0.0
            vertical_speed = current_vertical_speed
        elif primitive_id == "accelerate":
            target_speed = max(current_speed, target_speed)
        elif primitive_id == "decelerate":
            target_speed = min(current_speed, target_speed)
        target_speed = float(np.clip(target_speed, 0.0, self.config.limits.max_speed_mps))
        acceleration = float(np.clip(acceleration, 1e-4, self.config.limits.max_acceleration_mps2))
        curvature = float(np.clip(curvature, -self.config.limits.max_curvature_per_m, self.config.limits.max_curvature_per_m))
        vertical_speed = float(np.clip(
            vertical_speed,
            -self.config.limits.max_vertical_speed_mps,
            self.config.limits.max_vertical_speed_mps,
        ))
        return duration, target_speed, acceleration, curvature, vertical_speed

    def _generate(
        self,
        rng: np.random.Generator,
        home: np.ndarray,
        start: np.ndarray,
        initial_heading: float,
        sequence: Tuple[str, ...],
    ) -> GeneratedTrajectory:
        dt = self.dt
        limits = self.config.limits
        positions = [start.copy()]
        velocities = [np.zeros(3, dtype=np.float64)]
        accelerations = [np.zeros(3, dtype=np.float64)]
        headings = [float(initial_heading)]
        curvatures = [0.0]
        primitive_ids = [sequence[0] if sequence else "hold"]
        segment_indices = [0]
        segment_progress = [0.0]
        phases = ["moving"]

        speed = 0.0
        vertical_speed = 0.0
        longitudinal_accel = 0.0
        vertical_accel = 0.0
        curvature = 0.0
        heading = float(initial_heading)

        def append_step(
            primitive_id: str,
            segment_index: int,
            progress: float,
            target_speed: float,
            acceleration_limit: float,
            target_curvature: float,
            target_vertical_speed: float,
            phase: str,
        ) -> None:
            nonlocal speed, vertical_speed, longitudinal_accel, vertical_accel, curvature, heading
            desired_accel = float(np.clip(
                (target_speed - speed) / limits.speed_response_time_sec,
                -acceleration_limit,
                acceleration_limit,
            ))
            longitudinal_accel = _move_towards(
                longitudinal_accel,
                desired_accel,
                limits.max_jerk_mps3 * dt,
            )
            next_speed = float(np.clip(speed + longitudinal_accel * dt, 0.0, limits.max_speed_mps))

            desired_vertical_accel = float(np.clip(
                (target_vertical_speed - vertical_speed) / limits.vertical_response_time_sec,
                -limits.max_vertical_acceleration_mps2,
                limits.max_vertical_acceleration_mps2,
            ))
            vertical_accel = _move_towards(
                vertical_accel,
                desired_vertical_accel,
                limits.max_vertical_jerk_mps3 * dt,
            )
            next_vertical_speed = float(np.clip(
                vertical_speed + vertical_accel * dt,
                -limits.max_vertical_speed_mps,
                limits.max_vertical_speed_mps,
            ))

            curvature = _move_towards(
                curvature,
                target_curvature,
                limits.max_curvature_rate_per_m_sec * dt,
            )
            curvature_limit = limits.max_curvature_per_m
            if next_speed > 1e-3:
                available_lateral_accel = math.sqrt(
                    max(
                        0.0,
                        limits.max_acceleration_mps2 ** 2
                        - longitudinal_accel ** 2,
                    )
                )
                curvature_limit = min(
                    curvature_limit,
                    min(limits.max_lateral_acceleration_mps2, available_lateral_accel)
                    / max(next_speed * next_speed, 1e-6),
                    limits.max_yaw_rate_rad_s / next_speed,
                )
            curvature = float(np.clip(curvature, -curvature_limit, curvature_limit))
            yaw_rate = float(np.clip(
                next_speed * curvature,
                -limits.max_yaw_rate_rad_s,
                limits.max_yaw_rate_rad_s,
            ))
            heading = math.atan2(math.sin(heading + yaw_rate * dt), math.cos(heading + yaw_rate * dt))
            next_velocity = np.array(
                [
                    next_speed * math.cos(heading),
                    next_speed * math.sin(heading),
                    next_vertical_speed,
                ],
                dtype=np.float64,
            )
            previous_velocity = velocities[-1]
            next_position = positions[-1] + 0.5 * (previous_velocity + next_velocity) * dt
            next_acceleration = (next_velocity - previous_velocity) / dt
            positions.append(next_position)
            velocities.append(next_velocity)
            accelerations.append(next_acceleration)
            headings.append(heading)
            curvatures.append(curvature)
            primitive_ids.append(primitive_id)
            segment_indices.append(segment_index)
            segment_progress.append(float(np.clip(progress, 0.0, 1.0)))
            phases.append(phase)
            speed = next_speed
            vertical_speed = next_vertical_speed

        for segment_index, primitive_id in enumerate(sequence):
            duration, target_speed, accel_limit, target_curvature, target_vertical_speed = self._sample_targets(
                rng,
                primitive_id,
                speed,
                vertical_speed,
            )
            steps = max(1, int(math.ceil(duration / dt)))
            for step in range(steps):
                append_step(
                    primitive_id,
                    segment_index,
                    (step + 1) / steps,
                    target_speed,
                    accel_limit,
                    target_curvature,
                    target_vertical_speed,
                    "moving",
                )

        final_stop_index = len(positions) - 1
        brake_steps = max(1, int(math.ceil(limits.final_brake_max_sec / dt)))
        for step in range(brake_steps):
            append_step(
                "final_stop",
                len(sequence),
                (step + 1) / brake_steps,
                0.0,
                limits.max_acceleration_mps2,
                0.0,
                0.0,
                "decelerating",
            )
            if (
                step >= 1
                and speed <= 0.01
                and abs(vertical_speed) <= 0.01
                and abs(longitudinal_accel) <= 0.02
                and abs(vertical_accel) <= 0.02
                and abs(curvature) <= 0.01
            ):
                break

        speed = 0.0
        vertical_speed = 0.0
        longitudinal_accel = 0.0
        vertical_accel = 0.0
        curvature = 0.0
        hold_steps = max(1, int(math.ceil(self.config.stop_hold_sec / dt)))
        for step in range(hold_steps):
            positions.append(positions[-1].copy())
            velocities.append(np.zeros(3, dtype=np.float64))
            accelerations.append(np.zeros(3, dtype=np.float64))
            headings.append(heading)
            curvatures.append(0.0)
            primitive_ids.append("stopped")
            segment_indices.append(len(sequence) + 1)
            segment_progress.append((step + 1) / hold_steps)
            phases.append("stopped")

        positions_array = np.asarray(positions, dtype=np.float64)
        velocities_array = np.asarray(velocities, dtype=np.float64)
        accelerations_array = np.asarray(accelerations, dtype=np.float64)
        headings_array = np.asarray(headings, dtype=np.float64)
        tangent = np.stack((np.cos(headings_array), np.sin(headings_array)), axis=1)
        goal_positions = positions_array.copy()
        goal_positions[:, :2] -= self.follow_distance_m * tangent
        goal_positions[:, 2] += self.follow_vertical_offset_m
        goal_velocities = np.gradient(goal_positions, dt, axis=0, edge_order=1)
        goal_accelerations = np.gradient(goal_velocities, dt, axis=0, edge_order=1)
        stopped_mask = np.asarray([phase == "stopped" for phase in phases], dtype=bool)
        goal_velocities[stopped_mask] = 0.0
        goal_accelerations[stopped_mask] = 0.0
        motion_denominator = max(1, final_stop_index)
        motion_progress = np.minimum(
            1.0,
            np.arange(len(positions_array), dtype=np.float64) / motion_denominator,
        )
        return GeneratedTrajectory(
            dt=dt,
            positions=positions_array,
            velocities=velocities_array,
            accelerations=accelerations_array,
            goal_positions=goal_positions,
            goal_velocities=goal_velocities,
            goal_accelerations=goal_accelerations,
            headings=headings_array,
            curvatures=np.asarray(curvatures, dtype=np.float64),
            primitive_ids=primitive_ids,
            primitive_codes=np.asarray([PRIMITIVE_CODES[value] for value in primitive_ids], dtype=np.int64),
            segment_indices=np.asarray(segment_indices, dtype=np.int64),
            segment_progress=np.asarray(segment_progress, dtype=np.float64),
            phases=phases,
            motion_progress=motion_progress,
            sequence_ids=sequence,
            final_stop_index=final_stop_index,
        )

    def _within_limits(self, trajectory: GeneratedTrajectory, home: np.ndarray) -> bool:
        limits = self.config.limits
        target_xy = np.linalg.norm(trajectory.positions[:, :2] - home[:2], axis=1)
        goal_xy = np.linalg.norm(trajectory.goal_positions[:, :2] - home[:2], axis=1)
        target_z = np.abs(trajectory.positions[:, 2] - home[2])
        goal_z = np.abs(trajectory.goal_positions[:, 2] - home[2])
        return bool(
            np.max(target_xy) <= limits.max_horizontal_radius_m
            and np.max(goal_xy) <= limits.max_horizontal_radius_m
            and np.max(target_z) <= limits.max_vertical_displacement_m
            and np.max(goal_z) <= limits.max_vertical_displacement_m
        )

    def _meets_template_amplitudes(
        self,
        trajectory: GeneratedTrajectory,
    ) -> tuple[bool, str]:
        horizontal_speeds = np.linalg.norm(trajectory.velocities[:, :2], axis=1)
        for segment_index, primitive_id in enumerate(trajectory.sequence_ids):
            template = self.config.templates[primitive_id]
            if (
                template.min_heading_change_deg <= 0.0
                and template.min_vertical_displacement_m <= 0.0
                and template.min_speed_change_mps <= 0.0
            ):
                continue
            indices = np.flatnonzero(
                trajectory.segment_indices == segment_index
            )
            if indices.size == 0:
                return False, f"trajectory segment {segment_index} has no samples"
            start_index = max(0, int(indices[0]) - 1)
            end_index = int(indices[-1])

            unwrapped_heading = np.unwrap(
                trajectory.headings[start_index : end_index + 1]
            )
            heading_change_deg = abs(
                math.degrees(float(unwrapped_heading[-1] - unwrapped_heading[0]))
            )
            vertical_displacement = abs(
                float(
                    trajectory.positions[end_index, 2]
                    - trajectory.positions[start_index, 2]
                )
            )
            speed_change = abs(
                float(horizontal_speeds[end_index] - horizontal_speeds[start_index])
            )
            if heading_change_deg + 1e-6 < template.min_heading_change_deg:
                return (
                    False,
                    f"{primitive_id} segment heading change {heading_change_deg:.2f}deg "
                    f"is below {template.min_heading_change_deg:.2f}deg",
                )
            if (
                vertical_displacement + 1e-6
                < template.min_vertical_displacement_m
            ):
                return (
                    False,
                    f"{primitive_id} segment vertical displacement "
                    f"{vertical_displacement:.3f}m is below "
                    f"{template.min_vertical_displacement_m:.3f}m",
                )
            if speed_change + 1e-6 < template.min_speed_change_mps:
                return (
                    False,
                    f"{primitive_id} segment speed change {speed_change:.3f}m/s "
                    f"is below {template.min_speed_change_mps:.3f}m/s",
                )
        return True, ""
