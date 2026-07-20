from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CheckpointScoreConfig:
    final_xy_tolerance_m: float
    camera_enabled: bool = False
    guardrail_drop: float = 0.08


POSITIVE_WEIGHTS = {
    "moving_joint": 0.18,
    "moving_xy": 0.10,
    "overall_success": 0.18,
    "stopped_xy": 0.10,
    "stopped_position": 0.05,
    "stopped_stationary": 0.07,
    "final_stop": 0.07,
    "final_xy_quality": 0.08,
    "camera_visibility": 0.07,
}
PENALTY_WEIGHTS = {
    "timeout_rate": 0.07,
    "other_failure_rate": 0.05,
}
GUARDRAIL_COMPONENTS = (
    "overall_success",
    "stopped_xy",
    "final_stop",
    "final_xy_quality",
    "camera_visibility",
)


def _sample_weight(row: Mapping[str, object], count_key: str) -> float:
    if count_key not in row:
        return 1.0
    try:
        value = float(row[count_key])
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _weighted_mean(
    rows: Sequence[Mapping[str, object]],
    key: str,
    count_key: str,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        weight = _sample_weight(row, count_key)
        weighted_sum += float(row.get(key, 0.0)) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0.0 else 0.0


def _primitive_fraction(row: Mapping[str, object], primitive_id: str) -> float:
    value = row.get("primitive_good_fraction", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
    if not isinstance(value, Mapping):
        return 0.0
    return float(value.get(primitive_id, 0.0))


def _primitive_sample_count(
    row: Mapping[str, object],
    primitive_id: str,
) -> float:
    explicit_key = f"{primitive_id}_sample_count"
    if explicit_key in row:
        return _sample_weight(row, explicit_key)
    value = row.get("primitive_sample_counts")
    if value is None:
        return 1.0
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0
    if not isinstance(value, Mapping):
        return 0.0
    try:
        count = float(value.get(primitive_id, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return count if math.isfinite(count) and count > 0.0 else 0.0


def _weighted_primitive_fraction(
    rows: Sequence[Mapping[str, object]],
    primitive_id: str,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        weight = _primitive_sample_count(row, primitive_id)
        weighted_sum += _primitive_fraction(row, primitive_id) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0.0 else 0.0


def aggregate_checkpoint_score(
    rows: Sequence[Mapping[str, object]],
    config: CheckpointScoreConfig,
) -> tuple[float, dict[str, float]]:
    if not rows:
        return 0.0, {}
    completed = sum(int(row.get("completed_episode_count", 0)) for row in rows)
    successes = sum(int(row.get("success_count", 0)) for row in rows)
    timeouts = sum(int(row.get("timeout_count", 0)) for row in rows)
    other_failures = sum(int(row.get("other_done_count", 0)) for row in rows)
    outcome_denominator = float(max(1, completed))

    final_xy_weight = sum(
        int(row.get("completed_episode_count", 0)) for row in rows
    )
    if final_xy_weight > 0:
        mean_final_xy = sum(
            float(row.get("mean_completed_final_xy_err", 0.0))
            * int(row.get("completed_episode_count", 0))
            for row in rows
        ) / float(final_xy_weight)
    else:
        mean_final_xy = 0.0
    final_xy_tolerance = max(1e-6, float(config.final_xy_tolerance_m))
    final_xy_quality = (
        math.exp(-((mean_final_xy / final_xy_tolerance) ** 2))
        if completed > 0
        else 0.0
    )
    camera_visibility = (
        _weighted_mean(
            rows,
            "camera_good_sample_fraction",
            "camera_sample_count",
        )
        if config.camera_enabled
        else 1.0
    )
    components = {
        "completed_episodes": float(completed),
        "moving_joint": _weighted_mean(
            rows,
            "moving_good_sample_fraction",
            "moving_eligible_sample_count",
        ),
        "moving_xy": _weighted_mean(
            rows,
            "moving_xy_good_sample_fraction",
            "moving_eligible_sample_count",
        ),
        "overall_success": successes / outcome_denominator,
        "stopped_xy": _weighted_mean(
            rows,
            "stopped_xy_zone_fraction",
            "stopped_sample_count",
        ),
        "stopped_position": _weighted_mean(
            rows,
            "stopped_position_zone_fraction",
            "stopped_sample_count",
        ),
        "stopped_stationary": _weighted_mean(
            rows,
            "stopped_stationary_fraction",
            "stopped_sample_count",
        ),
        "final_stop": _weighted_primitive_fraction(rows, "final_stop"),
        "mean_final_xy_error_m": float(mean_final_xy),
        "final_xy_quality": float(final_xy_quality),
        "camera_visibility": float(camera_visibility),
        "timeout_rate": timeouts / outcome_denominator,
        "other_failure_rate": other_failures / outcome_denominator,
    }
    score = sum(
        POSITIVE_WEIGHTS[name] * components[name] for name in POSITIVE_WEIGHTS
    ) - sum(
        PENALTY_WEIGHTS[name] * components[name] for name in PENALTY_WEIGHTS
    )
    return float(score), components


def violates_checkpoint_guardrails(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float],
    config: CheckpointScoreConfig,
) -> bool:
    if not incumbent:
        return False
    for name in GUARDRAIL_COMPONENTS:
        if name == "camera_visibility" and not config.camera_enabled:
            continue
        if float(candidate.get(name, 0.0)) < (
            float(incumbent.get(name, 0.0)) - float(config.guardrail_drop)
        ):
            return True
    return False


def checkpoint_score_formula() -> str:
    positive = " + ".join(
        f"{weight:g}*{name}" for name, weight in POSITIVE_WEIGHTS.items()
    )
    penalties = " - ".join(
        f"{weight:g}*{name}" for name, weight in PENALTY_WEIGHTS.items()
    )
    return f"{positive} - {penalties}"
