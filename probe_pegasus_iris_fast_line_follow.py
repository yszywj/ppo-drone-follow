#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
RUNPY_ROOT = SCRIPT_DIR.parent
if str(RUNPY_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNPY_ROOT))

DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "result" / "probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Probe no-PX4 Pegasus Iris CTBR backend and vector env throughput")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--physics_dt", type=float, default=0.004)
    parser.add_argument("--rendering_dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--step_dt_sim_sec", type=float, default=0.5)
    parser.add_argument("--env_spacing_m", type=float, default=8.0)
    parser.add_argument("--takeoff_altitude", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--print_defaults_only", action="store_true", default=False)

    parser.add_argument("--hover_sec", type=float, default=4.0)
    parser.add_argument("--pulse_sec", type=float, default=1.0)
    parser.add_argument("--settle_sec", type=float, default=1.0)
    parser.add_argument("--throughput_policy_steps", type=int, default=20)
    parser.add_argument("--sample_dt", type=float, default=0.02)
    parser.add_argument("--rate_step", type=float, default=0.025)
    parser.add_argument("--yaw_rate_step", type=float, default=0.006)
    parser.add_argument("--thrust_step", type=float, default=0.015)

    parser.add_argument(
        "--control_mix_mode",
        choices=("ratio", "additive"),
        default="ratio",
        help="ratio uses an explicit controller/PPO convex mix; additive reproduces the legacy gain sum.",
    )
    parser.add_argument(
        "--policy_ratio",
        type=float,
        default=0.20,
        help="PPO share in ratio mode; controller share is 1-policy_ratio.",
    )
    parser.add_argument("--residual_gain", type=float, default=0.8, help="Legacy additive mode only.")
    parser.add_argument("--goal_feedback_scale", type=float, default=1.0)
    parser.add_argument("--attitude_feedback_scale", type=float, default=1.0)
    parser.add_argument("--randomize_yaw", action="store_true", default=False)
    parser.add_argument("--random_yaw_max_offset_deg", type=float, default=180.0)
    parser.add_argument("--max_roll_rate", type=float, default=0.080)
    parser.add_argument("--max_pitch_rate", type=float, default=0.080)
    parser.add_argument("--max_yaw_rate", type=float, default=0.010)
    parser.add_argument("--xy_control_mode", choices=("body", "legacy"), default="body")
    parser.add_argument("--goal_xy_pos_gain", type=float, default=0.025)
    parser.add_argument("--xy_velocity_damping_gain", type=float, default=0.100)
    parser.add_argument("--xy_target_velocity_gain", type=float, default=0.100)
    parser.add_argument("--xy_target_accel_gain", type=float, default=0.080)
    parser.add_argument("--xy_max_tilt_cmd", type=float, default=0.16)
    parser.add_argument("--yaw_hold_kp", type=float, default=1.0)
    parser.add_argument("--yaw_hold_max_rate_deg_s", type=float, default=15.0)

    parser.add_argument("--hover_thrust", type=float, default=0.60)
    parser.add_argument("--thrust_delta", type=float, default=0.015)
    parser.add_argument("--thrust_min", type=float, default=0.50)
    parser.add_argument("--thrust_max", type=float, default=0.72)
    parser.add_argument("--z_feedback_scale", type=float, default=1.0)
    parser.add_argument("--z_pos_gain", type=float, default=0.04)
    parser.add_argument("--z_vel_gain", type=float, default=0.06)
    parser.add_argument("--z_target_velocity_gain", type=float, default=0.0)
    parser.add_argument("--z_target_accel_gain", type=float, default=0.0)
    parser.add_argument("--safety_max_xy_from_home", type=float, default=5.5)

    parser.add_argument("--mass_kg", type=float, default=1.52)
    parser.add_argument("--inertia_xx", type=float, default=0.029125)
    parser.add_argument("--inertia_yy", type=float, default=0.029125)
    parser.add_argument("--inertia_zz", type=float, default=0.055225)
    parser.add_argument("--rate_kp_roll", type=float, default=0.52)
    parser.add_argument("--rate_kp_pitch", type=float, default=0.52)
    parser.add_argument("--rate_kp_yaw", type=float, default=0.18)
    parser.add_argument("--motor_time_constant", type=float, default=0.025)

    parser.add_argument("--follow_distance_m", type=float, default=1.0)
    parser.add_argument("--line_length_m", type=float, default=4.0)
    parser.add_argument("--randomize_line_length", action="store_true", default=False)
    parser.add_argument("--line_length_min_m", type=float, default=3.5)
    parser.add_argument("--line_length_max_m", type=float, default=4.5)
    parser.add_argument(
        "--line_length_sampling",
        choices=("uniform_area", "uniform_radius"),
        default="uniform_area",
    )
    parser.add_argument("--target_speed_mps", type=float, default=0.35)
    parser.add_argument("--target_accel_sec", type=float, default=2.0)
    parser.add_argument("--target_decel_sec", type=float, default=2.0)
    parser.add_argument("--target_stopped_speed_threshold_mps", type=float, default=0.05)
    parser.add_argument("--target_accel_observation_filter_alpha", type=float, default=0.5)
    parser.add_argument("--target_z_delta_m", type=float, default=0.0)
    parser.add_argument("--line_yaw_deg", type=float, default=0.0)
    parser.add_argument("--randomize_line_yaw", action="store_true", default=False)
    parser.add_argument("--line_yaw_min_deg", type=float, default=-20.0)
    parser.add_argument("--line_yaw_max_deg", type=float, default=20.0)

    parser.add_argument("--tracking_xy_tolerance_m", type=float, default=0.45)
    parser.add_argument("--tracking_z_tolerance_m", type=float, default=0.40)
    parser.add_argument("--tracking_velocity_tolerance_mps", type=float, default=0.45)
    parser.add_argument("--stopped_speed_xy_tolerance_mps", type=float, default=0.25)
    parser.add_argument("--stopped_speed_z_tolerance_mps", type=float, default=0.25)
    parser.add_argument("--moving_success_dwell_sec", type=float, default=1.0)
    parser.add_argument("--moving_reward_min_progress_fraction", type=float, default=0.20)
    parser.add_argument("--moving_success_min_fraction", type=float, default=0.50)
    parser.add_argument("--moving_success_xy_tolerance_m", type=float, default=0.60)
    parser.add_argument("--moving_success_velocity_tolerance_mps", type=float, default=0.35)
    parser.add_argument("--stopped_success_dwell_sec", type=float, default=2.0)
    parser.add_argument("--capture_radius_m", type=float, default=0.80)
    parser.add_argument("--capture_hold_sec", type=float, default=1.0)
    parser.add_argument("--reward_capture_once", type=float, default=3.0)
    parser.add_argument("--reward_capture_hold", type=float, default=0.15)
    parser.add_argument("--reward_capture_tracking_scale", type=float, default=1.0)
    parser.add_argument("--reward_moving_good", type=float, default=0.10)
    parser.add_argument("--reward_moving_joint_scale", type=float, default=1.0)
    parser.add_argument("--max_tracking_error_m", type=float, default=3.0)
    parser.add_argument("--min_target_distance_m", type=float, default=0.35)

    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--baseline_summary",
        type=str,
        default=None,
        help="Optional PX4 or previous fast-probe summary.json to compute normalized gap scores.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.policy_ratio <= 1.0:
        parser.error("--policy_ratio must be in [0, 1]")
    for name in (
        "target_accel_sec",
        "target_decel_sec",
        "line_length_m",
        "line_length_min_m",
        "line_length_max_m",
        "moving_reward_min_progress_fraction",
        "moving_success_min_fraction",
        "moving_success_xy_tolerance_m",
        "moving_success_velocity_tolerance_mps",
        "capture_radius_m",
        "capture_hold_sec",
        "reward_capture_once",
        "reward_capture_hold",
        "reward_capture_tracking_scale",
        "reward_moving_good",
        "reward_moving_joint_scale",
        "goal_xy_pos_gain",
        "xy_velocity_damping_gain",
        "xy_target_velocity_gain",
        "xy_target_accel_gain",
        "xy_max_tilt_cmd",
        "max_roll_rate",
        "max_pitch_rate",
        "max_yaw_rate",
        "z_feedback_scale",
        "z_pos_gain",
        "z_vel_gain",
        "z_target_velocity_gain",
        "z_target_accel_gain",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name} must be non-negative")
    for name in ("moving_reward_min_progress_fraction", "moving_success_min_fraction"):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    if not 0.0 <= args.target_accel_observation_filter_alpha <= 1.0:
        parser.error("--target_accel_observation_filter_alpha must be in [0, 1]")
    if args.line_length_min_m > args.line_length_max_m:
        parser.error("--line_length_min_m must be <= --line_length_max_m")
    return args


def default_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "sim": {
            "num_envs": args.num_envs,
            "physics_dt": args.physics_dt,
            "rendering_dt": args.rendering_dt,
            "step_dt_sim_sec": args.step_dt_sim_sec,
            "env_spacing_m": args.env_spacing_m,
        },
        "start": {
            "drone_start_local_ned_m": [0.0, 0.0, -args.takeoff_altitude],
            "takeoff_altitude_m": args.takeoff_altitude,
            "randomize_yaw": args.randomize_yaw,
            "random_yaw_range_deg": [-args.random_yaw_max_offset_deg, args.random_yaw_max_offset_deg],
        },
        "line_follow_task": {
            "follow_distance_m": args.follow_distance_m,
            "target_start_local_ned_m": [args.follow_distance_m, 0.0, -args.takeoff_altitude + args.target_z_delta_m],
            "line_length_m": args.line_length_m,
            "randomize_line_length": args.randomize_line_length,
            "line_length_range_m": [args.line_length_min_m, args.line_length_max_m],
            "line_length_sampling": args.line_length_sampling,
            "target_speed_mps": args.target_speed_mps,
            "target_accel_sec": args.target_accel_sec,
            "target_decel_sec": args.target_decel_sec,
            "target_accel_observation_filter_alpha": args.target_accel_observation_filter_alpha,
            "target_z_delta_m": args.target_z_delta_m,
            "line_yaw_deg": args.line_yaw_deg,
            "randomize_line_yaw": args.randomize_line_yaw,
            "line_yaw_range_deg": [args.line_yaw_min_deg, args.line_yaw_max_deg],
            "capture_radius_m": args.capture_radius_m,
            "capture_hold_sec": args.capture_hold_sec,
        },
        "success": {
            "tracking_xy_tolerance_m": args.tracking_xy_tolerance_m,
            "tracking_z_tolerance_m": args.tracking_z_tolerance_m,
            "tracking_velocity_tolerance_mps": args.tracking_velocity_tolerance_mps,
            "stopped_speed_xy_tolerance_mps": args.stopped_speed_xy_tolerance_mps,
            "stopped_speed_z_tolerance_mps": args.stopped_speed_z_tolerance_mps,
            "moving_success_dwell_sec": args.moving_success_dwell_sec,
            "moving_reward_min_progress_fraction": args.moving_reward_min_progress_fraction,
            "moving_success_min_fraction": args.moving_success_min_fraction,
            "moving_success_xy_tolerance_m": args.moving_success_xy_tolerance_m,
            "moving_success_velocity_tolerance_mps": args.moving_success_velocity_tolerance_mps,
            "stopped_success_dwell_sec": args.stopped_success_dwell_sec,
        },
        "reward": {
            "reward_capture_once": args.reward_capture_once,
            "reward_capture_hold": args.reward_capture_hold,
            "reward_capture_tracking_scale": args.reward_capture_tracking_scale,
            "reward_moving_good": args.reward_moving_good,
            "reward_moving_joint_scale": args.reward_moving_joint_scale,
        },
        "ctbr": {
            "xy_control_mode": args.xy_control_mode,
            "hover_thrust": args.hover_thrust,
            "thrust_delta": args.thrust_delta,
            "thrust_min": args.thrust_min,
            "thrust_max": args.thrust_max,
            "max_roll_rate": args.max_roll_rate,
            "max_pitch_rate": args.max_pitch_rate,
            "max_yaw_rate": args.max_yaw_rate,
            "goal_xy_pos_gain": args.goal_xy_pos_gain,
            "xy_velocity_damping_gain": args.xy_velocity_damping_gain,
            "xy_target_velocity_gain": args.xy_target_velocity_gain,
            "xy_target_accel_gain": args.xy_target_accel_gain,
            "xy_max_tilt_cmd": args.xy_max_tilt_cmd,
            "control_mix_mode": args.control_mix_mode,
            "policy_ratio": args.policy_ratio,
            "controller_ratio": 1.0 - args.policy_ratio,
            "residual_gain": args.residual_gain,
            "goal_feedback_scale": args.goal_feedback_scale,
            "attitude_feedback_scale": args.attitude_feedback_scale,
            "rate_kp": [args.rate_kp_roll, args.rate_kp_pitch, args.rate_kp_yaw],
            "motor_time_constant": args.motor_time_constant,
            "z_feedback_scale": args.z_feedback_scale,
            "z_pos_gain": args.z_pos_gain,
            "z_vel_gain": args.z_vel_gain,
            "z_target_velocity_gain": args.z_target_velocity_gain,
            "z_target_accel_gain": args.z_target_accel_gain,
        },
        "probe_metrics": {
            "hover": ["xy_rms_m", "z_rms_m", "speed_rms_mps", "max_tilt_deg"],
            "axis_step": [
                "steady_rate_mean_rad_s",
                "steady_rate_error_rad_s",
                "peak_abs_rate_rad_s",
                "rise_time_63_sec",
                "delta_xy_m",
                "delta_z_m",
                "max_tilt_deg",
            ],
            "throughput": ["env_steps_per_wall_sec", "sim_rtf"],
            "gap_score": "mean normalized difference of matched metrics against --baseline_summary; lower is better.",
        },
    }


def print_defaults(args: argparse.Namespace) -> None:
    print(json.dumps(default_payload(args), indent=2, sort_keys=True))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_csv_row(path: Path, fieldnames: Sequence[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def rms(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return 0.0
    return math.sqrt(sum(v * v for v in arr) / len(arr))


def percentile_tail(values: Sequence[float], fraction: float = 0.5) -> List[float]:
    arr = list(values)
    if not arr:
        return []
    start = max(0, int(len(arr) * (1.0 - fraction)))
    return arr[start:]


def mean(values: Sequence[float]) -> float:
    arr = list(values)
    return float(sum(arr) / len(arr)) if arr else 0.0


def max_abs(values: Sequence[float]) -> float:
    arr = list(values)
    return max((abs(float(v)) for v in arr), default=0.0)


def make_configs(args):
    import numpy as np

    from pegasus_iris_fast_line_follow.ctbr_backend import CTBRActionLimits, RotorCTBRBackendConfig, SafetyLimits
    from pegasus_iris_fast_line_follow.fast_line_follow_env import FastLineFollowEnvConfig, LineFollowTaskConfig

    action_limits = CTBRActionLimits(
        max_roll_rate=args.max_roll_rate,
        max_pitch_rate=args.max_pitch_rate,
        max_yaw_rate=args.max_yaw_rate,
        hover_thrust=args.hover_thrust,
        thrust_delta=args.thrust_delta,
        thrust_min=args.thrust_min,
        thrust_max=args.thrust_max,
        control_mix_mode=args.control_mix_mode,
        policy_ratio=args.policy_ratio,
        residual_gain=args.residual_gain,
        goal_feedback_scale=args.goal_feedback_scale,
        attitude_feedback_scale=args.attitude_feedback_scale,
        xy_control_mode=args.xy_control_mode,
        goal_xy_pos_gain=args.goal_xy_pos_gain,
        xy_velocity_damping_gain=args.xy_velocity_damping_gain,
        xy_target_velocity_gain=args.xy_target_velocity_gain,
        xy_target_accel_gain=args.xy_target_accel_gain,
        xy_max_tilt_cmd=args.xy_max_tilt_cmd,
        yaw_hold_kp=args.yaw_hold_kp,
        yaw_hold_max_rate=float(np.deg2rad(args.yaw_hold_max_rate_deg_s)),
        z_feedback_scale=args.z_feedback_scale,
        z_pos_gain=args.z_pos_gain,
        z_vel_gain=args.z_vel_gain,
        z_target_velocity_gain=args.z_target_velocity_gain,
        z_target_accel_gain=args.z_target_accel_gain,
    )
    max_requested_line_length = (
        args.line_length_max_m
        if args.randomize_line_length
        else args.line_length_m
    )
    safety_limits = SafetyLimits(
        min_altitude=0.35,
        max_altitude=11.0,
        max_tilt_deg=55.0,
        max_body_rate=4.0,
        max_down_speed=3.0,
        max_xy_from_home=max(
            float(args.safety_max_xy_from_home),
            float(max_requested_line_length) + float(args.follow_distance_m) + 1.0,
        ),
        max_z_error_from_home=4.0,
    )
    backend_cfg = RotorCTBRBackendConfig(
        mass_kg=args.mass_kg,
        inertia_diag=(args.inertia_xx, args.inertia_yy, args.inertia_zz),
        hover_thrust_command=args.hover_thrust,
        rate_kp=(args.rate_kp_roll, args.rate_kp_pitch, args.rate_kp_yaw),
        motor_time_constant=args.motor_time_constant,
    )
    env_cfg = FastLineFollowEnvConfig(
        num_envs=args.num_envs,
        env_spacing_m=args.env_spacing_m,
        physics_dt=args.physics_dt,
        rendering_dt=args.rendering_dt,
        step_dt_sim_sec=args.step_dt_sim_sec,
        episode_length=200,
        takeoff_altitude=args.takeoff_altitude,
        render=args.render,
        seed=args.seed,
        randomize_yaw_on_reset=args.randomize_yaw,
        random_yaw_max_offset_deg=args.random_yaw_max_offset_deg,
        yaw_hold_kp=args.yaw_hold_kp,
        yaw_hold_max_rate=float(np.deg2rad(args.yaw_hold_max_rate_deg_s)),
        reward_alive=0.0,
        reward_progress_scale=0.5,
        reward_distance_scale=0.10,
        reward_z_scale=0.40,
        target_accel_observation_filter_alpha=args.target_accel_observation_filter_alpha,
        action_limits=action_limits,
        safety_limits=safety_limits,
        backend_config=backend_cfg,
    )
    task_cfg = LineFollowTaskConfig(
        follow_distance_m=args.follow_distance_m,
        line_length_m=args.line_length_m,
        randomize_line_length=args.randomize_line_length,
        line_length_min_m=args.line_length_min_m,
        line_length_max_m=args.line_length_max_m,
        line_length_sampling=args.line_length_sampling,
        target_speed_mps=args.target_speed_mps,
        target_accel_sec=args.target_accel_sec,
        target_decel_sec=args.target_decel_sec,
        target_stopped_speed_threshold_mps=args.target_stopped_speed_threshold_mps,
        target_z_delta_m=args.target_z_delta_m,
        line_yaw_deg=args.line_yaw_deg,
        randomize_line_yaw=args.randomize_line_yaw,
        line_yaw_min_deg=args.line_yaw_min_deg,
        line_yaw_max_deg=args.line_yaw_max_deg,
        tracking_xy_tolerance_m=args.tracking_xy_tolerance_m,
        tracking_z_tolerance_m=args.tracking_z_tolerance_m,
        tracking_velocity_tolerance_mps=args.tracking_velocity_tolerance_mps,
        stopped_speed_xy_tolerance_mps=args.stopped_speed_xy_tolerance_mps,
        stopped_speed_z_tolerance_mps=args.stopped_speed_z_tolerance_mps,
        moving_success_dwell_sec=args.moving_success_dwell_sec,
        moving_reward_min_progress_fraction=args.moving_reward_min_progress_fraction,
        moving_success_min_fraction=args.moving_success_min_fraction,
        moving_success_xy_tolerance_m=args.moving_success_xy_tolerance_m,
        moving_success_velocity_tolerance_mps=args.moving_success_velocity_tolerance_mps,
        stopped_success_dwell_sec=args.stopped_success_dwell_sec,
        capture_radius_m=args.capture_radius_m,
        capture_hold_sec=args.capture_hold_sec,
        reward_capture_once=args.reward_capture_once,
        reward_capture_hold=args.reward_capture_hold,
        reward_capture_tracking_scale=args.reward_capture_tracking_scale,
        reward_moving_good=args.reward_moving_good,
        reward_moving_joint_scale=args.reward_moving_joint_scale,
        max_tracking_error_m=args.max_tracking_error_m,
        min_target_distance_m=args.min_target_distance_m,
    )
    return env_cfg, task_cfg


def sample_env(env, label: str, t_sec: float, command: Tuple[float, float, float, float]) -> List[dict]:
    rows = []
    for env_id in range(env.num_envs):
        obs = env._obs_data(env_id)
        home = env._home[env_id]
        rows.append(
            {
                "test": label,
                "time_sec": float(t_sec),
                "env_id": int(env_id),
                "x": obs.x,
                "y": obs.y,
                "z": obs.z,
                "vx": obs.vx,
                "vy": obs.vy,
                "vz": obs.vz,
                "roll": obs.roll,
                "pitch": obs.pitch,
                "yaw": obs.yaw,
                "rollspeed": obs.rollspeed,
                "pitchspeed": obs.pitchspeed,
                "yawspeed": obs.yawspeed,
                "xy_from_home": math.sqrt((obs.x - home.x) ** 2 + (obs.y - home.y) ** 2),
                "z_err": obs.z - home.z,
                "cmd_roll_rate": command[0],
                "cmd_pitch_rate": command[1],
                "cmd_yaw_rate": command[2],
                "cmd_thrust": command[3],
            }
        )
    return rows


def apply_direct_command(env, command: Tuple[float, float, float, float]) -> None:
    for backend in env.backends:
        backend.set_ctbr_command(*command)


def run_direct_segment(
    env,
    label: str,
    duration_sec: float,
    command: Tuple[float, float, float, float],
    sample_dt: float,
    samples_path: Optional[Path],
) -> Tuple[List[dict], float, int]:
    apply_direct_command(env, command)
    physics_dt = float(env.config.physics_dt)
    total_steps = max(1, int(round(duration_sec / physics_dt)))
    sample_every = max(1, int(round(sample_dt / physics_dt)))
    rows: List[dict] = []
    wall0 = time.perf_counter()
    for step_idx in range(total_steps):
        env.world.step(render=env.config.render)
        if step_idx % sample_every == 0 or step_idx == total_steps - 1:
            rows.extend(sample_env(env, label, (step_idx + 1) * physics_dt, command))
    wall = time.perf_counter() - wall0
    if samples_path is not None:
        fieldnames = list(rows[0].keys()) if rows else []
        for row in rows:
            append_csv_row(samples_path, fieldnames, row)
    return rows, wall, total_steps


def summarize_hover(rows: List[dict], wall_sec: float, sim_sec: float, world_steps: int, num_envs: int) -> dict:
    return {
        "test": "direct_hover",
        "command_axis": "none",
        "command_value": 0.0,
        "duration_sec": sim_sec,
        "wall_sec": wall_sec,
        "world_steps": world_steps,
        "num_envs": num_envs,
        "sim_rtf": sim_sec / max(wall_sec, 1e-9),
        "world_steps_per_wall_sec": world_steps / max(wall_sec, 1e-9),
        "env_steps_per_wall_sec": world_steps * num_envs / max(wall_sec, 1e-9),
        "xy_rms_m": rms([r["xy_from_home"] for r in rows]),
        "xy_max_m": max([r["xy_from_home"] for r in rows], default=0.0),
        "z_rms_m": rms([r["z_err"] for r in rows]),
        "z_max_abs_m": max_abs([r["z_err"] for r in rows]),
        "speed_rms_mps": rms(
            [math.sqrt(r["vx"] ** 2 + r["vy"] ** 2 + r["vz"] ** 2) for r in rows]
        ),
        "max_tilt_deg": math.degrees(
            max([math.sqrt(r["roll"] ** 2 + r["pitch"] ** 2) for r in rows], default=0.0)
        ),
    }


def first_rise_time(rows: List[dict], axis_rate_key: str, command_value: float) -> Optional[float]:
    threshold = 0.63 * abs(command_value)
    sign = 1.0 if command_value >= 0.0 else -1.0
    if threshold <= 1e-9:
        return None
    first_env_rows = [r for r in rows if int(r["env_id"]) == 0]
    for row in first_env_rows:
        if sign * float(row[axis_rate_key]) >= threshold:
            return float(row["time_sec"])
    return None


def summarize_axis(
    test_name: str,
    rows: List[dict],
    command_axis: str,
    command_value: float,
    wall_sec: float,
    sim_sec: float,
    world_steps: int,
    num_envs: int,
) -> dict:
    rate_key = {"roll": "rollspeed", "pitch": "pitchspeed", "yaw": "yawspeed"}[command_axis]
    tail = percentile_tail(rows, 0.5)
    steady_rates = [float(r[rate_key]) for r in tail]
    all_rates = [float(r[rate_key]) for r in rows]
    first_by_env: Dict[int, dict] = {}
    last_by_env: Dict[int, dict] = {}
    for row in rows:
        env_id = int(row["env_id"])
        first_by_env.setdefault(env_id, row)
        last_by_env[env_id] = row
    deltas_xy = []
    deltas_z = []
    for env_id, first in first_by_env.items():
        last = last_by_env[env_id]
        deltas_xy.append(math.sqrt((last["x"] - first["x"]) ** 2 + (last["y"] - first["y"]) ** 2))
        deltas_z.append(last["z"] - first["z"])
    steady = mean(steady_rates)
    return {
        "test": test_name,
        "command_axis": command_axis,
        "command_value": command_value,
        "duration_sec": sim_sec,
        "wall_sec": wall_sec,
        "world_steps": world_steps,
        "num_envs": num_envs,
        "sim_rtf": sim_sec / max(wall_sec, 1e-9),
        "world_steps_per_wall_sec": world_steps / max(wall_sec, 1e-9),
        "env_steps_per_wall_sec": world_steps * num_envs / max(wall_sec, 1e-9),
        "steady_rate_mean_rad_s": steady,
        "steady_rate_error_rad_s": steady - command_value,
        "steady_rate_abs_error_rad_s": abs(steady - command_value),
        "peak_abs_rate_rad_s": max_abs(all_rates),
        "rise_time_63_sec": first_rise_time(rows, rate_key, command_value),
        "delta_xy_m": mean(deltas_xy),
        "delta_z_m": mean(deltas_z),
        "max_tilt_deg": math.degrees(
            max([math.sqrt(r["roll"] ** 2 + r["pitch"] ** 2) for r in rows], default=0.0)
        ),
    }


def summarize_thrust(
    rows: List[dict],
    command_value: float,
    wall_sec: float,
    sim_sec: float,
    world_steps: int,
    num_envs: int,
) -> dict:
    first_by_env: Dict[int, dict] = {}
    last_by_env: Dict[int, dict] = {}
    for row in rows:
        env_id = int(row["env_id"])
        first_by_env.setdefault(env_id, row)
        last_by_env[env_id] = row
    dz = []
    dvz = []
    for env_id, first in first_by_env.items():
        last = last_by_env[env_id]
        dz.append(last["z"] - first["z"])
        dvz.append(last["vz"] - first["vz"])
    return {
        "test": "direct_thrust_step",
        "command_axis": "thrust",
        "command_value": command_value,
        "duration_sec": sim_sec,
        "wall_sec": wall_sec,
        "world_steps": world_steps,
        "num_envs": num_envs,
        "sim_rtf": sim_sec / max(wall_sec, 1e-9),
        "world_steps_per_wall_sec": world_steps / max(wall_sec, 1e-9),
        "env_steps_per_wall_sec": world_steps * num_envs / max(wall_sec, 1e-9),
        "delta_z_m": mean(dz),
        "delta_vz_mps": mean(dvz),
        "vz_peak_abs_mps": max_abs([r["vz"] for r in rows]),
        "max_tilt_deg": math.degrees(
            max([math.sqrt(r["roll"] ** 2 + r["pitch"] ** 2) for r in rows], default=0.0)
        ),
    }


def summarize_policy_throughput(env, policy_steps: int) -> dict:
    import numpy as np

    actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
    wall0 = time.perf_counter()
    reward_sum = 0.0
    done_count = 0
    xy = []
    z = []
    for _ in range(max(1, policy_steps)):
        _, rewards, dones, infos = env.step(actions)
        reward_sum += float(np.sum(rewards))
        done_count += int(np.sum(dones))
        xy.extend(float(info.get("xy_err", 0.0)) for info in infos)
        z.extend(float(info.get("z_err", 0.0)) for info in infos)
    wall = time.perf_counter() - wall0
    sim_sec = policy_steps * float(env.config.step_dt_sim_sec)
    env_steps = policy_steps * env.num_envs
    return {
        "test": "policy_zero_throughput",
        "command_axis": "policy_action_zero",
        "command_value": 0.0,
        "duration_sec": sim_sec,
        "wall_sec": wall,
        "world_steps": policy_steps * env._sim_steps_per_policy_step,
        "num_envs": env.num_envs,
        "policy_steps": policy_steps,
        "env_policy_steps": env_steps,
        "sim_rtf": sim_sec / max(wall, 1e-9),
        "policy_steps_per_wall_sec": policy_steps / max(wall, 1e-9),
        "env_steps_per_wall_sec": env_steps / max(wall, 1e-9),
        "mean_reward": reward_sum / max(env_steps, 1),
        "done_count": done_count,
        "mean_xy_err": mean(xy),
        "mean_z_err": mean(z),
    }


GAP_FIELDS = {
    "direct_hover": {
        "xy_rms_m": 0.05,
        "z_rms_m": 0.05,
        "speed_rms_mps": 0.05,
        "max_tilt_deg": 1.0,
    },
    "direct_roll_step": {
        "steady_rate_mean_rad_s": 0.005,
        "peak_abs_rate_rad_s": 0.005,
        "rise_time_63_sec": 0.05,
        "delta_xy_m": 0.05,
        "max_tilt_deg": 1.0,
    },
    "direct_pitch_step": {
        "steady_rate_mean_rad_s": 0.005,
        "peak_abs_rate_rad_s": 0.005,
        "rise_time_63_sec": 0.05,
        "delta_xy_m": 0.05,
        "max_tilt_deg": 1.0,
    },
    "direct_yaw_step": {
        "steady_rate_mean_rad_s": 0.002,
        "peak_abs_rate_rad_s": 0.002,
        "rise_time_63_sec": 0.05,
        "max_tilt_deg": 1.0,
    },
    "direct_thrust_step": {
        "delta_z_m": 0.05,
        "delta_vz_mps": 0.05,
        "vz_peak_abs_mps": 0.05,
    },
}


def compute_gap(metrics: List[dict], baseline_summary: Optional[str]) -> Optional[dict]:
    if not baseline_summary:
        return None
    baseline_path = Path(baseline_summary).expanduser()
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline_summary does not exist: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", [])
    by_name = {row.get("test"): row for row in metrics}
    base_by_name = {row.get("test"): row for row in baseline_metrics}
    per_test = {}
    all_values = []
    for test_name, fields in GAP_FIELDS.items():
        if test_name not in by_name or test_name not in base_by_name:
            continue
        row = by_name[test_name]
        base = base_by_name[test_name]
        values = []
        for field, floor in fields.items():
            if row.get(field) is None or base.get(field) is None:
                continue
            denom = max(abs(float(base[field])), float(floor))
            values.append(abs(float(row[field]) - float(base[field])) / denom)
        if values:
            per_test[test_name] = float(sum(values) / len(values))
            all_values.extend(values)
    return {
        "baseline_summary": str(baseline_path),
        "gap_score": float(sum(all_values) / len(all_values)) if all_values else None,
        "per_test_gap_score": per_test,
        "definition": "Mean normalized absolute metric difference against baseline; 0 is identical, lower is better.",
    }


def metric_fields(metrics: List[dict]) -> List[str]:
    keys = []
    for row in metrics:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    return keys


def main() -> None:
    args = parse_args()
    if args.print_defaults_only:
        print_defaults(args)
        return

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})

    import numpy as np

    from pegasus_iris_fast_line_follow.fast_line_follow_env import FastIrisLineFollowVecEnv

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.results_root).expanduser() / f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    metrics_path = run_dir / "metrics.csv"
    samples_path = run_dir / "samples.csv"
    env = None
    metrics: List[dict] = []
    try:
        env_cfg, task_cfg = make_configs(args)
        env = FastIrisLineFollowVecEnv(env_cfg, task_cfg)
        write_json(run_dir / "probe_config.json", default_payload(args))

        hover_cmd = (0.0, 0.0, 0.0, args.hover_thrust)
        rows, wall, steps = run_direct_segment(
            env,
            "direct_hover",
            args.hover_sec,
            hover_cmd,
            args.sample_dt,
            samples_path,
        )
        metrics.append(summarize_hover(rows, wall, args.hover_sec, steps, env.num_envs))

        tests = [
            ("direct_roll_step", "roll", (args.rate_step, 0.0, 0.0, args.hover_thrust)),
            ("direct_pitch_step", "pitch", (0.0, args.rate_step, 0.0, args.hover_thrust)),
            ("direct_yaw_step", "yaw", (0.0, 0.0, args.yaw_rate_step, args.hover_thrust)),
        ]
        for test_name, axis, command in tests:
            env.reset()
            rows, wall, steps = run_direct_segment(
                env,
                test_name,
                args.pulse_sec,
                command,
                args.sample_dt,
                samples_path,
            )
            metrics.append(summarize_axis(test_name, rows, axis, command[{"roll": 0, "pitch": 1, "yaw": 2}[axis]], wall, args.pulse_sec, steps, env.num_envs))
            run_direct_segment(env, f"{test_name}_settle", args.settle_sec, hover_cmd, args.sample_dt, samples_path)

        env.reset()
        thrust_cmd = (0.0, 0.0, 0.0, args.hover_thrust + args.thrust_step)
        rows, wall, steps = run_direct_segment(
            env,
            "direct_thrust_step",
            args.pulse_sec,
            thrust_cmd,
            args.sample_dt,
            samples_path,
        )
        metrics.append(summarize_thrust(rows, thrust_cmd[3], wall, args.pulse_sec, steps, env.num_envs))
        run_direct_segment(env, "direct_thrust_step_settle", args.settle_sec, hover_cmd, args.sample_dt, samples_path)

        env.reset()
        metrics.append(summarize_policy_throughput(env, args.throughput_policy_steps))

        gap = compute_gap(metrics, args.baseline_summary)
        fields = metric_fields(metrics)
        for row in metrics:
            append_csv_row(metrics_path, fields, row)
        summary = {
            "run_dir": str(run_dir),
            "config": default_payload(args),
            "metrics": metrics,
            "gap": gap,
            "notes": [
                "Direct tests bypass the PPO residual adapter and set CTBR commands directly on the no-PX4 backend.",
                "policy_zero_throughput uses env.step() with zero normalized policy actions and measures full vector-env stepping cost.",
                "PX4-only signals are unavailable here: armed state, flight mode, failsafe text, MAVLink freshness and estimator delay/noise.",
            ],
        }
        write_json(run_dir / "summary.json", summary)

        print("=" * 88)
        print("Fast Pegasus Iris no-PX4 probe complete")
        print(f"run_dir: {run_dir}")
        for row in metrics:
            print(
                f"[{row['test']}] "
                f"rtf={row.get('sim_rtf', 0.0):.2f}, "
                f"env_steps/s={row.get('env_steps_per_wall_sec', 0.0):.1f}, "
                f"xy_rms={row.get('xy_rms_m', 0.0):.3f}, "
                f"z_rms={row.get('z_rms_m', 0.0):.3f}, "
                f"steady_err={row.get('steady_rate_abs_error_rad_s', 0.0):.4f}, "
                f"max_tilt={row.get('max_tilt_deg', 0.0):.2f}"
            )
        if gap is not None:
            print(f"gap_score: {gap.get('gap_score')}")
        print(f"saved metrics: {metrics_path}")
        print(f"saved samples: {samples_path}")
        print(f"saved summary: {run_dir / 'summary.json'}")
        print("=" * 88)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
