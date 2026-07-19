#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNPY_ROOT = SCRIPT_DIR.parent
if str(RUNPY_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNPY_ROOT))

from pegasus_iris_fast_line_follow.training_config import (
    load_training_config,
    resolve_path_from_config,
)

DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "result" / "ppo_train"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, _ = config_parser.parse_known_args()
    config_defaults = {}
    config_special = {}
    raw_config = {}
    if config_args.config:
        config_defaults, config_special, raw_config = load_training_config(config_args.config)

    parser = argparse.ArgumentParser("Fast no-PX4 Pegasus Iris trajectory-follow PPO")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--num_env_steps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=160)
    parser.add_argument("--step_dt_sim_sec", type=float, default=0.2)
    parser.add_argument("--physics_dt", type=float, default=0.004)
    parser.add_argument("--rendering_dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--env_spacing_m", type=float, default=8.0)
    parser.add_argument("--takeoff_altitude", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--actor_hidden_sizes", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--critic_hidden_sizes", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--recurrent_hidden_size", type=int, default=128)
    parser.add_argument(
        "--actor_recurrent_mode",
        choices=("train", "frozen", "disabled"),
        default="train",
        help="Train, freeze, or zero-disable the Actor GRU residual branch.",
    )
    parser.add_argument("--temporal_gate_init", type=float, default=0.05)
    parser.add_argument("--temporal_gate_warmup_updates", type=int, default=15)
    parser.add_argument("--reset_temporal_gate_on_load", action="store_true", default=False)
    parser.add_argument("--actor_backbone_lr_multiplier", type=float, default=0.5)
    parser.add_argument("--actor_recurrent_lr_multiplier", type=float, default=3.0)
    parser.add_argument("--activation", choices=("elu", "relu", "tanh"), default="elu")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Legacy shorthand: use two equally sized layers for both actor and critic.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", choices=("constant", "linear"), default="linear")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_clip_param", type=float, default=0.2)
    parser.add_argument(
        "--value_loss_type",
        choices=("clipped_mse", "unclipped_huber"),
        default="clipped_mse",
    )
    parser.add_argument("--use_popart", action="store_true", default=False)
    parser.add_argument(
        "--no_popart",
        action="store_false",
        dest="use_popart",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--popart_beta", type=float, default=0.999)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--reference_kl_coef", type=float, default=0.0)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--entropy_coef_final", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--init_action_std", type=float, default=0.20)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--allow_partial_checkpoint", action="store_true", default=False)
    parser.add_argument("--reset_optimizer", action="store_true", default=False)
    parser.add_argument(
        "--reset_rng_on_load",
        action="store_true",
        default=False,
        help="Keep the configured seed instead of restoring RNG states from a checkpoint.",
    )
    parser.add_argument("--from_scratch", action="store_true", default=False)
    parser.add_argument("--evaluation_only", action="store_true", default=False)
    parser.add_argument("--deterministic_actions", action="store_true", default=False)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--best_checkpoint_window", type=int, default=0)
    parser.add_argument("--early_stop_patience_updates", type=int, default=0)
    parser.add_argument("--early_stop_drop_threshold", type=float, default=0.0)

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
    parser.add_argument("--independent_env_rng", action="store_true", default=False)
    parser.add_argument(
        "--shared_env_rng",
        action="store_false",
        dest="independent_env_rng",
        default=argparse.SUPPRESS,
    )
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
    parser.add_argument("--yaw_success_tolerance_deg", type=float, default=5.0)

    parser.add_argument("--hover_thrust", type=float, default=0.60)
    parser.add_argument("--thrust_delta", type=float, default=0.015)
    parser.add_argument("--thrust_min", type=float, default=0.50)
    parser.add_argument("--thrust_max", type=float, default=0.72)
    parser.add_argument("--z_feedback_scale", type=float, default=1.0)
    parser.add_argument("--z_pos_gain", type=float, default=0.04)
    parser.add_argument("--z_vel_gain", type=float, default=0.06)
    parser.add_argument(
        "--z_target_velocity_gain",
        type=float,
        default=0.0,
        help="Target NED vertical-velocity feedforward gain; zero reproduces the old helper.",
    )
    parser.add_argument(
        "--z_target_accel_gain",
        type=float,
        default=0.0,
        help="Target NED vertical-acceleration feedforward gain; helper-only privileged input.",
    )
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
    parser.add_argument(
        "--actor_mask_target_acceleration",
        action="store_true",
        default=False,
        help="Keep the 3 Actor acceleration slots but fill them with zero.",
    )
    parser.add_argument(
        "--no_actor_mask_target_acceleration",
        action="store_false",
        dest="actor_mask_target_acceleration",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--target_z_delta_m", type=float, default=0.0)
    parser.add_argument("--follow_vertical_offset_m", type=float, default=0.0)
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
    parser.add_argument("--vertical_motion_z_tolerance_m", type=float, default=0.30)
    parser.add_argument(
        "--vertical_motion_velocity_tolerance_mps",
        type=float,
        default=0.10,
    )
    parser.add_argument("--vertical_success_min_fraction", type=float, default=0.75)
    parser.add_argument("--stopped_success_dwell_sec", type=float, default=2.0)
    parser.add_argument("--capture_radius_m", type=float, default=0.80)
    parser.add_argument("--capture_hold_sec", type=float, default=1.0)
    parser.add_argument("--reward_capture_once", type=float, default=3.0)
    parser.add_argument("--reward_capture_hold", type=float, default=0.15)
    parser.add_argument("--reward_capture_tracking_scale", type=float, default=1.0)
    parser.add_argument("--reward_moving_good", type=float, default=0.10)
    parser.add_argument("--reward_moving_joint_scale", type=float, default=1.0)
    parser.add_argument("--reward_moving_progress_scale", type=float, default=0.0)
    parser.add_argument("--moving_progress_clip_m", type=float, default=0.10)
    parser.add_argument("--reward_position_recovery_scale", type=float, default=0.0)
    parser.add_argument("--reward_velocity_correction_gain", type=float, default=0.0)
    parser.add_argument("--reward_velocity_correction_max_mps", type=float, default=0.0)
    parser.add_argument("--local_tracking_window_sec", type=float, default=2.0)
    parser.add_argument(
        "--local_tracking_reward_interval_sec", type=float, default=2.0
    )
    parser.add_argument(
        "--local_tracking_velocity_tolerance_mps", type=float, default=0.25
    )
    parser.add_argument(
        "--local_tracking_hard_fraction_weight", type=float, default=0.30
    )
    parser.add_argument(
        "--local_tracking_drift_deadband_m", type=float, default=0.05
    )
    parser.add_argument("--reward_local_tracking_scale", type=float, default=0.0)
    parser.add_argument("--reward_local_drift_scale", type=float, default=0.0)
    parser.add_argument("--max_tracking_error_m", type=float, default=3.0)
    parser.add_argument("--min_target_distance_m", type=float, default=0.35)

    parser.add_argument("--reward_alive", type=float, default=0.0)
    parser.add_argument("--reward_progress_scale", type=float, default=1.0)
    parser.add_argument("--reward_distance_scale", type=float, default=0.10)
    parser.add_argument("--reward_z_scale", type=float, default=0.40)
    parser.add_argument("--reward_control_scale", type=float, default=0.05)
    parser.add_argument("--reward_goal_zone", type=float, default=0.20)
    parser.add_argument("--reward_dwell_scale", type=float, default=0.50)
    parser.add_argument("--reward_success", type=float, default=50.0)
    parser.add_argument("--reward_crash", type=float, default=-40.0)
    parser.add_argument("--reward_timeout", type=float, default=-30.0)
    parser.add_argument("--reward_position_scale", type=float, default=2.0)
    parser.add_argument("--reward_velocity_scale", type=float, default=0.8)
    parser.add_argument("--reward_stop_speed_scale", type=float, default=2.0)
    parser.add_argument("--reward_braking_scale", type=float, default=1.0)
    parser.add_argument("--reward_stop_overspeed_scale", type=float, default=1.0)
    parser.add_argument("--reward_stopped_progress_multiplier", type=float, default=2.0)
    parser.add_argument("--reward_stopped_time_penalty", type=float, default=-0.05)
    parser.add_argument("--stopped_approach_speed_gain", type=float, default=0.8)
    parser.add_argument("--stopped_max_approach_speed_mps", type=float, default=0.6)
    parser.add_argument("--reward_too_close_scale", type=float, default=1.5)
    parser.add_argument("--position_sigma_m", type=float, default=0.75)
    parser.add_argument("--z_sigma_m", type=float, default=0.50)
    parser.add_argument("--position_recovery_sigma_m", type=float, default=1.50)
    parser.add_argument("--z_recovery_sigma_m", type=float, default=0.80)
    parser.add_argument("--velocity_sigma_mps", type=float, default=0.50)
    parser.add_argument("--stop_speed_sigma_mps", type=float, default=0.30)
    parser.add_argument("--reward_action_delta_scale", type=float, default=0.02)
    parser.add_argument("--reward_tilt_scale", type=float, default=0.08)
    parser.add_argument(
        "--no_reward_scale_by_dt",
        action="store_false",
        dest="reward_scale_by_dt",
        default=True,
    )

    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--task_description", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--no_terminal_log", action="store_true", default=False)
    parser.add_argument("--no_tensorboard", action="store_true", default=False)
    parser.add_argument("--live_plot_interval", type=int, default=5)
    valid_dests = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(config_defaults) - valid_dests)
    if unknown_config_keys:
        parser.error(
            "unknown keys in --config: " + ", ".join(unknown_config_keys)
        )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    args.motion_pool_config = config_special.get("motion_pool", {})
    args.raw_config = raw_config
    if args.config:
        args.config = str(Path(args.config).expanduser().resolve())
        args.results_root = resolve_path_from_config(args.config, args.results_root)
        args.load_checkpoint = resolve_path_from_config(args.config, args.load_checkpoint)
    for name in ("num_env_steps", "num_envs", "rollout_steps", "episode_length", "num_mini_batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("step_dt_sim_sec", "physics_dt"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name} must be positive")
    if args.hidden_size is not None:
        args.actor_hidden_sizes = [args.hidden_size, args.hidden_size]
        args.critic_hidden_sizes = [args.hidden_size, args.hidden_size]
    if any(size <= 0 for size in (*args.actor_hidden_sizes, *args.critic_hidden_sizes)):
        parser.error("all actor/critic hidden layer sizes must be positive")
    if args.recurrent_hidden_size <= 0:
        parser.error("--recurrent_hidden_size must be positive")
    if args.temporal_gate_warmup_updates < 0:
        parser.error("--temporal_gate_warmup_updates must be non-negative")
    for name in ("best_checkpoint_window", "early_stop_patience_updates"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if args.early_stop_patience_updates > 0 and args.best_checkpoint_window <= 0:
        parser.error("--early_stop_patience_updates requires --best_checkpoint_window")
    if abs(args.temporal_gate_init) > 3.0:
        parser.error("--temporal_gate_init must be in [-3, 3]")
    for name in ("actor_backbone_lr_multiplier", "actor_recurrent_lr_multiplier"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name} must be positive")
    if args.num_mini_batch > args.num_envs:
        parser.error("--num_mini_batch cannot exceed --num_envs for recurrent PPO")
    for name in (
        "lr",
        "critic_lr",
        "min_lr",
        "value_clip_param",
        "target_kl",
        "reference_kl_coef",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name} must be non-negative")
    if args.init_action_std <= 0.0:
        parser.error("--init_action_std must be positive")
    if not 0.0 <= args.popart_beta < 1.0:
        parser.error("--popart_beta must be in [0, 1)")
    if args.local_tracking_window_sec <= 0.0:
        parser.error("--local_tracking_window_sec must be positive")
    if args.local_tracking_reward_interval_sec <= 0.0:
        parser.error("--local_tracking_reward_interval_sec must be positive")
    if not 0.0 <= args.local_tracking_hard_fraction_weight <= 1.0:
        parser.error("--local_tracking_hard_fraction_weight must be in [0, 1]")
    if not 0.0 <= args.policy_ratio <= 1.0:
        parser.error("--policy_ratio must be in [0, 1]")
    for name in (
        "reward_position_scale",
        "reward_velocity_scale",
        "reward_capture_tracking_scale",
        "reward_moving_joint_scale",
        "reward_moving_progress_scale",
        "moving_progress_clip_m",
        "reward_position_recovery_scale",
        "reward_velocity_correction_gain",
        "reward_velocity_correction_max_mps",
        "local_tracking_velocity_tolerance_mps",
        "local_tracking_drift_deadband_m",
        "reward_local_tracking_scale",
        "reward_local_drift_scale",
        "reward_stop_speed_scale",
        "reward_braking_scale",
        "reward_stop_overspeed_scale",
        "reward_stopped_progress_multiplier",
        "stopped_approach_speed_gain",
        "stopped_max_approach_speed_mps",
        "position_sigma_m",
        "z_sigma_m",
        "position_recovery_sigma_m",
        "z_recovery_sigma_m",
        "velocity_sigma_mps",
        "stop_speed_sigma_mps",
        "target_accel_sec",
        "target_decel_sec",
        "line_length_m",
        "line_length_min_m",
        "line_length_max_m",
        "moving_reward_min_progress_fraction",
        "moving_success_min_fraction",
        "moving_success_xy_tolerance_m",
        "moving_success_velocity_tolerance_mps",
        "vertical_motion_z_tolerance_m",
        "vertical_motion_velocity_tolerance_mps",
        "vertical_success_min_fraction",
        "capture_radius_m",
        "capture_hold_sec",
        "reward_capture_once",
        "reward_capture_hold",
        "reward_moving_good",
        "reward_action_delta_scale",
        "reward_tilt_scale",
        "reward_control_scale",
        "early_stop_drop_threshold",
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
    for name in (
        "moving_reward_min_progress_fraction",
        "moving_success_min_fraction",
        "vertical_success_min_fraction",
    ):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    if not 0.0 <= args.target_accel_observation_filter_alpha <= 1.0:
        parser.error("--target_accel_observation_filter_alpha must be in [0, 1]")
    if args.line_length_min_m > args.line_length_max_m:
        parser.error("--line_length_min_m must be <= --line_length_max_m")
    if args.reward_stopped_time_penalty > 0.0:
        parser.error("--reward_stopped_time_penalty must be non-positive")
    if args.reward_timeout > 0.0:
        parser.error("--reward_timeout must be non-positive")
    if args.min_lr > min(args.lr, args.critic_lr):
        parser.error("--min_lr must not exceed either actor or critic learning rate")
    if args.entropy_coef < 0.0 or args.entropy_coef_final < 0.0:
        parser.error("entropy coefficients must be non-negative")
    return args


def main():
    args = parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})

    import numpy as np
    import torch

    from pegasus_iris_fast_line_follow.ctbr_backend import CTBRActionLimits, RotorCTBRBackendConfig, SafetyLimits
    from pegasus_iris_fast_line_follow.fast_line_follow_env import (
        FastIrisLineFollowVecEnv,
        FastLineFollowEnvConfig,
        LineFollowTaskConfig,
        REWARD_TERM_KEYS,
    )
    from pegasus_iris_fast_line_follow.motion_task import PRIMITIVE_CODES, MotionPoolConfig
    from pegasus_iris_fast_line_follow.ppo_core import (
        ActorCritic,
        PopArtValueNormalizer,
        append_csv_row,
        compute_gae_vec,
        dump_json,
        explained_variance,
        load_transplanted_checkpoint,
        restore_training_state,
        save_policy_checkpoint,
        save_training_plots,
        start_tensorboard_writer,
        start_terminal_log,
        write_tensorboard_scalars,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    run_dir = Path(args.results_root).expanduser() / f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    terminal_log = None if args.no_terminal_log else start_terminal_log(run_dir)
    writer = None if args.no_tensorboard else start_tensorboard_writer(run_dir)
    dump_json(run_dir / "args.json", vars(args))

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
    motion_pool_cfg = MotionPoolConfig.from_dict(args.motion_pool_config)
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
            (
                motion_pool_cfg.limits.max_horizontal_radius_m + 0.5
                if motion_pool_cfg.enabled
                else 0.0
            ),
        ),
        max_z_error_from_home=max(
            4.0,
            (
                motion_pool_cfg.limits.max_vertical_displacement_m + 0.5
                if motion_pool_cfg.enabled
                else 0.0
            ),
        ),
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
        episode_length=args.episode_length,
        takeoff_altitude=args.takeoff_altitude,
        render=args.render,
        seed=args.seed,
        independent_env_rng=args.independent_env_rng,
        randomize_yaw_on_reset=args.randomize_yaw,
        random_yaw_max_offset_deg=args.random_yaw_max_offset_deg,
        yaw_hold_kp=args.yaw_hold_kp,
        yaw_hold_max_rate=float(np.deg2rad(args.yaw_hold_max_rate_deg_s)),
        yaw_success_tolerance_deg=args.yaw_success_tolerance_deg,
        reward_alive=args.reward_alive,
        reward_progress_scale=args.reward_progress_scale,
        reward_distance_scale=args.reward_distance_scale,
        reward_z_scale=args.reward_z_scale,
        reward_control_scale=args.reward_control_scale,
        reward_goal_zone=args.reward_goal_zone,
        reward_dwell_scale=args.reward_dwell_scale,
        reward_success=args.reward_success,
        reward_crash=args.reward_crash,
        reward_timeout=args.reward_timeout,
        target_accel_observation_filter_alpha=args.target_accel_observation_filter_alpha,
        actor_mask_target_acceleration=args.actor_mask_target_acceleration,
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
        follow_vertical_offset_m=args.follow_vertical_offset_m,
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
        vertical_motion_z_tolerance_m=args.vertical_motion_z_tolerance_m,
        vertical_motion_velocity_tolerance_mps=(
            args.vertical_motion_velocity_tolerance_mps
        ),
        vertical_success_min_fraction=args.vertical_success_min_fraction,
        stopped_success_dwell_sec=args.stopped_success_dwell_sec,
        capture_radius_m=args.capture_radius_m,
        capture_hold_sec=args.capture_hold_sec,
        reward_capture_once=args.reward_capture_once,
        reward_capture_hold=args.reward_capture_hold,
        reward_capture_tracking_scale=args.reward_capture_tracking_scale,
        reward_moving_good=args.reward_moving_good,
        reward_moving_joint_scale=args.reward_moving_joint_scale,
        reward_moving_progress_scale=args.reward_moving_progress_scale,
        moving_progress_clip_m=args.moving_progress_clip_m,
        reward_position_recovery_scale=args.reward_position_recovery_scale,
        reward_velocity_correction_gain=args.reward_velocity_correction_gain,
        reward_velocity_correction_max_mps=(
            args.reward_velocity_correction_max_mps
        ),
        local_tracking_window_sec=args.local_tracking_window_sec,
        local_tracking_reward_interval_sec=(
            args.local_tracking_reward_interval_sec
        ),
        local_tracking_velocity_tolerance_mps=(
            args.local_tracking_velocity_tolerance_mps
        ),
        local_tracking_hard_fraction_weight=(
            args.local_tracking_hard_fraction_weight
        ),
        local_tracking_drift_deadband_m=(
            args.local_tracking_drift_deadband_m
        ),
        reward_local_tracking_scale=args.reward_local_tracking_scale,
        reward_local_drift_scale=args.reward_local_drift_scale,
        max_tracking_error_m=args.max_tracking_error_m,
        min_target_distance_m=args.min_target_distance_m,
        reward_position_scale=args.reward_position_scale,
        reward_velocity_scale=args.reward_velocity_scale,
        reward_stop_speed_scale=args.reward_stop_speed_scale,
        reward_braking_scale=args.reward_braking_scale,
        reward_stop_overspeed_scale=args.reward_stop_overspeed_scale,
        reward_stopped_progress_multiplier=args.reward_stopped_progress_multiplier,
        reward_stopped_time_penalty=args.reward_stopped_time_penalty,
        stopped_approach_speed_gain=args.stopped_approach_speed_gain,
        stopped_max_approach_speed_mps=args.stopped_max_approach_speed_mps,
        reward_too_close_scale=args.reward_too_close_scale,
        position_sigma_m=args.position_sigma_m,
        z_sigma_m=args.z_sigma_m,
        position_recovery_sigma_m=args.position_recovery_sigma_m,
        z_recovery_sigma_m=args.z_recovery_sigma_m,
        velocity_sigma_mps=args.velocity_sigma_mps,
        stop_speed_sigma_mps=args.stop_speed_sigma_mps,
        reward_action_delta_scale=args.reward_action_delta_scale,
        reward_tilt_scale=args.reward_tilt_scale,
        reward_scale_by_dt=args.reward_scale_by_dt,
        motion_pool=motion_pool_cfg,
    )

    env = None
    policy = None
    actor_optimizer = None
    critic_optimizer = None
    value_normalizer = None
    reference_policy = None
    reference_hidden = None
    update_rows = []
    episode_rows = []
    total_steps = 0
    update = 0
    early_stopped = False
    start_wall = time.perf_counter()
    run_started_at = time.time()
    update_fields = [
        "update",
        "total_steps",
        "wall_time",
        "elapsed_wall_sec",
        "update_wall_sec",
        "sps",
        "eta_sec",
        "estimated_finish_time",
        "mean_rollout_reward",
        "mean_xy_err",
        "max_xy_err",
        "mean_z_err",
        "max_z_err",
        "mean_signed_z_err",
        "mean_vertical_velocity_error",
        "mean_vertical_motion_velocity_error",
        "max_vertical_motion_velocity_error",
        "mean_goal_xy_progress",
        "goal_zone_fraction",
        "mean_goal_dwell_fraction",
        "max_goal_dwell_fraction",
        "mean_speed_xy",
        "max_speed_xy",
        "mean_speed_z",
        "max_speed_z",
        "success_count",
        "timeout_count",
        "other_done_count",
        "checkpoint_score",
        "best_checkpoint_score",
        "best_checkpoint_update",
        "degradation_updates",
        "early_stop_triggered",
        "done_reasons",
        "primitive_sample_counts",
        "primitive_mean_xy_err",
        "primitive_mean_z_err",
        "primitive_mean_velocity_error",
        "primitive_mean_vertical_velocity_error",
        "primitive_xy_good_fraction",
        "primitive_z_good_fraction",
        "primitive_velocity_good_fraction",
        "primitive_vertical_velocity_good_fraction",
        "primitive_good_fraction",
        "approx_kl",
        "clip_fraction",
        "reference_kl",
        "value_clip_fraction",
        "kl_early_stop",
        "actor_lr",
        "actor_backbone_lr",
        "actor_recurrent_lr",
        "critic_lr",
        "temporal_gate_raw",
        "temporal_gate_effective",
        "temporal_gate_frozen",
        "actor_recurrent_mode",
        "entropy_coef_current",
        "explained_variance",
        "return_mean",
        "return_std",
        "raw_value_mean",
        "raw_value_std",
        "popart_mean",
        "popart_std",
        "action_mean",
        "action_std",
        "action_abs_mean",
        "cmd_saturation_fraction",
        "cmd_roll_saturation_fraction",
        "cmd_pitch_saturation_fraction",
        "cmd_yaw_saturation_fraction",
        "cmd_thrust_saturation_fraction",
        "mean_abs_policy_cmd_roll_rate",
        "mean_abs_policy_cmd_pitch_rate",
        "mean_abs_controller_cmd_roll_rate",
        "mean_abs_controller_cmd_pitch_rate",
        "mean_abs_final_cmd_roll_rate",
        "mean_abs_final_cmd_pitch_rate",
        "mean_target_speed",
        "mean_tracking_velocity_error",
        "mean_reward_velocity_error",
        "mean_reward_velocity_correction_speed",
        "mean_target_distance",
        "mean_target_progress_fraction",
        "capture_acquired_fraction",
        "mean_capture_dwell_fraction",
        "moving_success_met_fraction",
        "mean_moving_good_fraction",
        "moving_xy_good_sample_fraction",
        "moving_z_good_sample_fraction",
        "moving_velocity_good_sample_fraction",
        "moving_good_sample_fraction",
        "local_tracking_ready_fraction",
        "local_tracking_event_count",
        "mean_local_tracking_soft_joint_quality",
        "mean_local_tracking_xy_good_fraction",
        "mean_local_tracking_velocity_good_fraction",
        "mean_local_tracking_z_good_fraction",
        "mean_local_tracking_xy_drift_delta_m",
        "vertical_success_met_fraction",
        "mean_vertical_good_fraction",
        "vertical_position_good_sample_fraction",
        "vertical_velocity_good_sample_fraction",
        "vertical_good_sample_fraction",
        "capture_phase_fraction",
        "moving_phase_fraction",
        "decelerating_phase_fraction",
        "stopped_phase_fraction",
        "mean_stopped_xy_err",
        "mean_stopped_speed_xy",
        "mean_stopped_speed_z",
        "mean_desired_approach_speed",
        "mean_allowed_stopped_speed",
        "mean_stopped_velocity_error",
        "stopped_xy_zone_fraction",
        "stopped_z_zone_fraction",
        "stopped_position_zone_fraction",
        "stopped_stationary_fraction",
        "stopped_xy_good_sample_fraction",
        "stopped_z_good_sample_fraction",
        "stopped_speed_good_sample_fraction",
        "mean_braking_progress",
        "policy_loss",
        "value_loss",
        "entropy",
        *[f"mean_{key}" for key in REWARD_TERM_KEYS],
    ]
    episode_fields = [
        "episode",
        "env_id",
        "total_steps",
        "episode_steps",
        "return",
        "done_reason",
        "final_goal_xy_err",
        "final_z_err",
        "final_goal_distance",
        "final_goal_rel_x",
        "final_goal_rel_y",
        "final_goal_rel_z",
        "final_signed_z_err",
        "final_speed_xy",
        "final_speed_z",
        "final_yaw",
        "final_yaw_rate",
        "yaw_start",
        "final_cmd_roll_rate",
        "final_cmd_pitch_rate",
        "final_cmd_yaw_rate",
        "final_cmd_thrust",
        "final_policy_cmd_roll_rate",
        "final_policy_cmd_pitch_rate",
        "final_policy_cmd_yaw_rate",
        "final_policy_cmd_thrust",
        "final_controller_cmd_roll_rate",
        "final_controller_cmd_pitch_rate",
        "final_controller_cmd_yaw_rate",
        "final_controller_cmd_thrust",
        "final_cmd_any_saturated",
        "final_target_phase",
        "final_primitive_id",
        "final_primitive_code",
        "motion_sequence_ids",
        "trajectory_duration_sec",
        "final_target_curvature_per_m",
        "final_target_progress_fraction",
        "final_target_speed",
        "final_target_distance",
        "final_tracking_velocity_error",
        "final_vertical_velocity_error",
        "final_desired_approach_speed",
        "final_allowed_stopped_speed",
        "final_stopped_velocity_error",
        "line_yaw_deg",
        "line_dir_x",
        "line_dir_y",
        "sampled_line_length_m",
        "desired_endpoint_x",
        "desired_endpoint_y",
        "desired_endpoint_z",
        "target_endpoint_x",
        "target_endpoint_y",
        "target_endpoint_z",
        "capture_acquired",
        "capture_dwell_steps",
        "required_capture_steps",
        "capture_dwell_fraction",
        "moving_success_met",
        "moving_track_dwell_steps",
        "required_moving_track_steps",
        "moving_eligible_steps",
        "moving_good_steps",
        "moving_good_fraction",
        "moving_xy_good_steps",
        "moving_z_good_steps",
        "moving_velocity_good_steps",
        "moving_xy_good_fraction",
        "moving_z_good_fraction",
        "moving_velocity_good_fraction",
        "local_tracking_soft_joint_quality",
        "local_tracking_xy_good_fraction",
        "local_tracking_velocity_good_fraction",
        "local_tracking_z_good_fraction",
        "local_tracking_xy_drift_delta_m",
        "vertical_success_met",
        "vertical_eligible_steps",
        "vertical_position_good_steps",
        "vertical_velocity_good_steps",
        "vertical_good_steps",
        "vertical_position_good_fraction",
        "vertical_velocity_good_fraction",
        "vertical_good_fraction",
        "stopped_track_dwell_steps",
        "required_stopped_track_steps",
        "stopped_eligible_steps",
        "stopped_xy_good_steps",
        "stopped_z_good_steps",
        "stopped_speed_good_steps",
        "stopped_xy_good_fraction",
        "stopped_z_good_fraction",
        "stopped_speed_good_fraction",
        "mean_goal_xy_err",
        "max_goal_xy_err",
        "mean_z_err",
        "max_z_err",
        "mean_speed_xy",
        "max_speed_xy",
        "mean_tracking_velocity_error",
        "max_tracking_velocity_error",
        "mean_vertical_motion_velocity_error",
        "max_vertical_motion_velocity_error",
        "max_goal_dwell_fraction",
        "capture_phase_steps",
        "moving_phase_steps",
        "decelerating_phase_steps",
        "stopped_phase_steps",
        *[f"return_{key.removeprefix('reward_')}" for key in REWARD_TERM_KEYS],
        "success",
    ]
    episode_step_counts = np.zeros(args.num_envs, dtype=np.int64)
    episode_xy_sums = np.zeros(args.num_envs, dtype=np.float64)
    episode_xy_max = np.zeros(args.num_envs, dtype=np.float64)
    episode_z_sums = np.zeros(args.num_envs, dtype=np.float64)
    episode_z_max = np.zeros(args.num_envs, dtype=np.float64)
    episode_speed_xy_sums = np.zeros(args.num_envs, dtype=np.float64)
    episode_speed_xy_max = np.zeros(args.num_envs, dtype=np.float64)
    episode_tracking_vel_sums = np.zeros(args.num_envs, dtype=np.float64)
    episode_tracking_vel_max = np.zeros(args.num_envs, dtype=np.float64)
    episode_vertical_vel_sums = np.zeros(args.num_envs, dtype=np.float64)
    episode_vertical_vel_max = np.zeros(args.num_envs, dtype=np.float64)
    episode_vertical_vel_samples = np.zeros(args.num_envs, dtype=np.int64)
    episode_max_dwell = np.zeros(args.num_envs, dtype=np.float64)
    episode_phase_steps = {
        phase: np.zeros(args.num_envs, dtype=np.int64)
        for phase in ("capture", "moving", "decelerating", "stopped")
    }
    episode_reward_returns = {
        key: np.zeros(args.num_envs, dtype=np.float64)
        for key in REWARD_TERM_KEYS
    }

    try:
        print("[FAST PPO] initializing vector environment...", flush=True)
        env = FastIrisLineFollowVecEnv(env_cfg, task_cfg)
        print(
            f"[FAST PPO] vector environment initialized: num_envs={env.num_envs}, "
            f"actor_obs={env.obs_dim}, critic_obs={env.critic_obs_dim}",
            flush=True,
        )
        print(f"[FAST PPO] initializing ActorCritic on {device}...", flush=True)
        policy = ActorCritic(
            env.obs_dim,
            env.action_dim,
            critic_obs_dim=env.critic_obs_dim,
            actor_hidden_sizes=args.actor_hidden_sizes,
            critic_hidden_sizes=args.critic_hidden_sizes,
            recurrent_hidden_size=args.recurrent_hidden_size,
            temporal_gate_init=args.temporal_gate_init,
            activation=args.activation,
            init_std=args.init_action_std,
        ).to(device)
        print("[FAST PPO] ActorCritic initialized", flush=True)
        value_normalizer = (
            PopArtValueNormalizer(device=device, beta=args.popart_beta)
            if args.use_popart
            else None
        )
        checkpoint_payload = {}
        if args.from_scratch:
            print("[FAST PPO] starting from scratch")
        elif args.load_checkpoint:
            checkpoint = Path(args.load_checkpoint).expanduser()
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
            checkpoint_payload = load_transplanted_checkpoint(
                policy,
                checkpoint,
                device,
                allow_partial=args.allow_partial_checkpoint,
            )
            if args.reset_temporal_gate_on_load:
                with torch.no_grad():
                    policy.temporal_gate.fill_(args.temporal_gate_init)
                print(
                    "[FAST PPO] reset temporal gate after checkpoint load: "
                    f"raw={args.temporal_gate_init:.6f}, "
                    f"effective={float(torch.tanh(policy.temporal_gate)):.6f}"
                )
        else:
            print("[FAST PPO] starting from scratch (no checkpoint supplied)")
        if args.actor_recurrent_mode == "disabled":
            with torch.no_grad():
                policy.temporal_gate.zero_()
            for parameter in policy.actor_recurrent_parameters():
                parameter.requires_grad_(False)
            print("[FAST PPO] Actor GRU residual branch disabled (gate=0)")
        elif args.actor_recurrent_mode == "frozen":
            for parameter in policy.actor_recurrent_parameters():
                parameter.requires_grad_(False)
            print(
                "[FAST PPO] Actor GRU residual branch frozen at "
                f"effective_gate={float(torch.tanh(policy.temporal_gate)):.6f}"
            )

        actor_backbone_parameters = list(policy.actor_backbone_parameters())
        actor_recurrent_parameters = (
            list(policy.actor_recurrent_parameters())
            if args.actor_recurrent_mode == "train"
            else []
        )
        actor_parameters = [
            parameter
            for parameter in (*actor_backbone_parameters, *actor_recurrent_parameters)
            if parameter.requires_grad
        ]
        critic_parameters = list(policy.critic_parameters())
        actor_parameter_groups = [
            {
                "params": actor_backbone_parameters,
                "lr_multiplier": args.actor_backbone_lr_multiplier,
                "group_name": "backbone",
            }
        ]
        if actor_recurrent_parameters:
            actor_parameter_groups.append(
                {
                    "params": actor_recurrent_parameters,
                    "lr_multiplier": args.actor_recurrent_lr_multiplier,
                    "group_name": "recurrent",
                }
            )
        actor_optimizer = torch.optim.Adam(
            actor_parameter_groups,
            lr=args.lr,
        )
        critic_optimizer = torch.optim.Adam(critic_parameters, lr=args.critic_lr)
        if checkpoint_payload and not args.reset_optimizer:
            restore_training_state(
                checkpoint_payload,
                actor_optimizer,
                critic_optimizer,
                env_rng=env._rng,
                restore_rng=not args.reset_rng_on_load,
                value_normalizer=value_normalizer,
            )
        elif checkpoint_payload and value_normalizer is not None:
            normalizer_state = checkpoint_payload.get("value_normalizer_state")
            if normalizer_state is not None:
                value_normalizer.load_state_dict(normalizer_state)
                print("[FAST PPO] restored PopArt state with fresh optimizers")
            else:
                print(
                    "[FAST PPO] checkpoint has no PopArt state; "
                    "using identity initialization"
                )
        temporal_gate_frozen = bool(
            args.actor_recurrent_mode != "train"
            or args.temporal_gate_warmup_updates > 0
        )
        if args.actor_recurrent_mode == "train":
            policy.temporal_gate.requires_grad_(not temporal_gate_frozen)
        if temporal_gate_frozen:
            if args.actor_recurrent_mode == "train":
                print(
                    "[FAST PPO] temporal gate fixed for the first "
                    f"{args.temporal_gate_warmup_updates} updates"
                )
        if args.reference_kl_coef > 0.0 and not args.evaluation_only:
            reference_policy = copy.deepcopy(policy).to(device)
            reference_state = checkpoint_payload.get(
                "reference_policy_state_dict"
            )
            if reference_state is not None:
                reference_policy.load_state_dict(reference_state)
                print("[FAST PPO] restored fixed reference-policy anchor")
            reference_policy.eval()
            for parameter in reference_policy.parameters():
                parameter.requires_grad_(False)
            print(
                "[FAST PPO] fixed reference-policy KL enabled: "
                f"coef={args.reference_kl_coef}"
            )

        print("=" * 88)
        print("Fast no-PX4 Pegasus Iris line-follow PPO")
        print(f"run_dir: {run_dir}")
        if hasattr(args, "task_description"):
            print(f"task_description: {args.task_description}")
        print(f"device: {device}")
        print(f"num_envs: {args.num_envs}, rollout_steps: {args.rollout_steps}, batch/update: {args.num_envs * args.rollout_steps}")
        print(
            f"physics_dt: {args.physics_dt}, step_dt_sim_sec: {args.step_dt_sim_sec}, "
            f"policy_hz: {1.0 / args.step_dt_sim_sec:.2f}, "
            f"sim_steps/policy_step: {env._sim_steps_per_policy_step}"
        )
        print(
            f"actor_obs_dim: {env.obs_dim} (causal current state only), "
            f"critic_obs_dim: {env.critic_obs_dim} "
            f"(future_ref={env.future_reference_dim} privileged), "
            f"action_dim: {env.action_dim}, "
            f"actor={args.actor_hidden_sizes}, critic={args.critic_hidden_sizes}, "
            f"gru={args.recurrent_hidden_size}, recurrent_mode={args.actor_recurrent_mode}, "
            f"temporal_gate_init={args.temporal_gate_init}, "
            f"activation={args.activation}, distribution=tanh_squashed_normal, "
            f"parameters={sum(parameter.numel() for parameter in policy.parameters())}"
        )
        print(
            "policy_observation: vehicle position/velocity/acceleration/attitude/body rates, "
            "desired follow point, target position, follow-reference velocity/acceleration, "
            "and previous CTBR command; future follow points and phase/task ids are critic-only"
        )
        print(
            "actor_learning_rates: "
            f"backbone={args.actor_backbone_lr_multiplier}x, "
            f"recurrent={args.actor_recurrent_lr_multiplier}x base actor LR"
        )
        print(
            "critic_learning: "
            f"value_loss={args.value_loss_type}, popart={args.use_popart}, "
            f"popart_beta={args.popart_beta}, value_clip={args.value_clip_param}; "
            f"gamma={args.gamma}, gae_lambda={args.gae_lambda}, "
            f"reference_kl_coef={args.reference_kl_coef}"
        )
        line_length_description = (
            f"[{args.line_length_min_m}, {args.line_length_max_m}]m "
            f"({args.line_length_sampling})"
            if args.randomize_line_length
            else f"{args.line_length_m}m"
        )
        if motion_pool_cfg.enabled:
            print(
                "task_pool: "
                f"prefix={list(motion_pool_cfg.prefix_ids)}, "
                f"sample_from={list(motion_pool_cfg.primitive_ids)}, "
                f"required={list(motion_pool_cfg.required_ids)}, "
                f"required_one_of={list(motion_pool_cfg.required_one_of_ids)}, "
                f"segments=[{motion_pool_cfg.min_segments}, {motion_pool_cfg.max_segments}], "
                f"future_reference={list(motion_pool_cfg.reference_horizon_sec)}s"
            )
            print(
                "motion_limits: "
                f"speed<={motion_pool_cfg.limits.max_speed_mps}m/s, "
                f"accel<={motion_pool_cfg.limits.max_acceleration_mps2}m/s^2, "
                f"curvature<={motion_pool_cfg.limits.max_curvature_per_m}1/m, "
                f"vertical_speed<={motion_pool_cfg.limits.max_vertical_speed_mps}m/s"
            )
        else:
            print(
                "legacy_line_task: "
                f"line_length={line_length_description}, target_speed={args.target_speed_mps}m/s, "
                f"accel={args.target_accel_sec}s, decel={args.target_decel_sec}s"
            )
        print(
            f"capture_radius={args.capture_radius_m}m, capture_hold={args.capture_hold_sec}s, "
            f"episode_length={args.episode_length} steps "
            f"({args.episode_length * args.step_dt_sim_sec:.1f}s)"
        )
        if args.control_mix_mode == "ratio":
            print(
                "control_mix: ratio "
                f"(controller={1.0 - args.policy_ratio:.3f}, PPO={args.policy_ratio:.3f})"
            )
        else:
            print(
                "control_mix: additive "
                f"(attitude_feedback_scale={args.attitude_feedback_scale}, "
                f"residual_gain={args.residual_gain})"
            )
        print(
            "aux_controller: "
            f"xy_mode={args.xy_control_mode}, goal_feedback_scale={args.goal_feedback_scale}, "
            f"xy_pos_gain={args.goal_xy_pos_gain}, xy_damping={args.xy_velocity_damping_gain}, "
            f"target_velocity_gain={args.xy_target_velocity_gain}, "
            f"target_accel_gain={args.xy_target_accel_gain}, "
            f"xy_tilt_limit={args.xy_max_tilt_cmd}, "
            f"rate_limits=({args.max_roll_rate}, {args.max_pitch_rate}, {args.max_yaw_rate}), "
            f"z_feedback_scale={args.z_feedback_scale}, "
            f"z_gains=(pos={args.z_pos_gain}, own_vel={args.z_vel_gain}, "
            f"target_vel={args.z_target_velocity_gain}, "
            f"target_accel={args.z_target_accel_gain})"
        )
        print(
            "actor_observation: "
            f"target_acceleration_masked={args.actor_mask_target_acceleration}; "
            "Critic privileged target/reference acceleration remains enabled"
        )
        print(
            "reward_3d_continuous: "
            f"position={args.reward_position_scale}, velocity={args.reward_velocity_scale}, "
            f"joint={args.reward_moving_joint_scale}, "
            f"moving_progress={args.reward_moving_progress_scale}, "
            f"recovery={args.reward_position_recovery_scale}, "
            f"position_sigma_xy={args.position_sigma_m}, sigma_z={args.z_sigma_m}, "
            f"recovery_sigma_xy={args.position_recovery_sigma_m}, "
            f"recovery_sigma_z={args.z_recovery_sigma_m}, "
            f"velocity_sigma={args.velocity_sigma_mps}, "
            f"velocity_correction=(gain={args.reward_velocity_correction_gain}, "
            f"max={args.reward_velocity_correction_max_mps}), "
            f"action_delta={args.reward_action_delta_scale}, tilt={args.reward_tilt_scale}, "
            f"dt_scaled={args.reward_scale_by_dt}, "
            f"capture_once={args.reward_capture_once}, capture_hold={args.reward_capture_hold}, "
            f"capture_tracking={args.reward_capture_tracking_scale}, "
            f"moving_good={args.reward_moving_good}, "
            f"local_window={args.local_tracking_window_sec}s, "
            f"local_interval={args.local_tracking_reward_interval_sec}s, "
            f"local_tracking={args.reward_local_tracking_scale}, "
            f"local_drift={args.reward_local_drift_scale}, "
            f"stopped_time={args.reward_stopped_time_penalty}, "
            f"goal_zone={args.reward_goal_zone}, dwell={args.reward_dwell_scale}, "
            f"success={args.reward_success}, timeout={args.reward_timeout}, crash={args.reward_crash}"
        )
        print(
            "stopped_approach: "
            f"speed_gain={args.stopped_approach_speed_gain}, "
            f"max_speed={args.stopped_max_approach_speed_mps} m/s"
        )
        print(
            "success gates: "
            f"capture_radius={args.capture_radius_m}, capture_hold={args.capture_hold_sec}s, "
            f"moving_min_progress={args.moving_reward_min_progress_fraction}, "
            f"moving_good_fraction>={args.moving_success_min_fraction}, "
            f"moving_xy_tol={args.moving_success_xy_tolerance_m}, "
            f"moving_vel_tol={args.moving_success_velocity_tolerance_mps}, "
            f"vertical_good_fraction>={args.vertical_success_min_fraction}, "
            f"vertical_z_tol={args.vertical_motion_z_tolerance_m}, "
            f"vertical_vel_tol={args.vertical_motion_velocity_tolerance_mps}, "
            f"stopped_dwell={args.stopped_success_dwell_sec}s"
        )
        print("PX4/MAVLink: disabled; CTBR commands are converted by local rate controller backend")
        print("Unavailable PX4-only signals: armed, flight_mode, failsafe, MAVLink freshness/timestamps")
        print("=" * 88)

        obs = env.reset()
        critic_obs = env.build_critic_obs(obs)
        recurrent_hidden = policy.initial_state(args.num_envs, device=device)
        if reference_policy is not None:
            reference_hidden = reference_policy.initial_state(
                args.num_envs, device=device
            )
        episode_start = np.ones(args.num_envs, dtype=bool)
        best_checkpoint_score = -math.inf
        best_checkpoint_update = 0
        degradation_updates = 0
        while total_steps < args.num_env_steps:
            if (
                args.actor_recurrent_mode == "train"
                and temporal_gate_frozen
                and update >= args.temporal_gate_warmup_updates
            ):
                policy.temporal_gate.requires_grad_(True)
                temporal_gate_frozen = False
                print(
                    "[FAST PPO] temporal gate is now trainable at "
                    f"update={update}, effective={float(torch.tanh(policy.temporal_gate)):.6f}"
                )
            update_start_wall = time.perf_counter()
            obs_buf = []
            critic_obs_buf = []
            action_buf = []
            logprob_buf = []
            reward_buf = []
            done_buf = []
            value_buf = []
            episode_start_buf = []
            rollout_initial_hidden = recurrent_hidden.detach().clone()
            rollout_initial_reference_hidden = (
                reference_hidden.detach().clone()
                if reference_hidden is not None
                else None
            )
            stats_rewards = []
            stats_reasons = Counter()
            stats_xy = []
            stats_z = []
            stats_signed_z = []
            stats_goal_progress = []
            stats_inside = []
            stats_dwell = []
            stats_speed_xy = []
            stats_speed_z = []
            stats_target_speed = []
            stats_tracking_vel = []
            stats_reward_velocity_error = []
            stats_reward_velocity_correction_speed = []
            stats_vertical_vel = []
            stats_vertical_motion_vel = []
            stats_target_distance = []
            stats_target_progress = []
            stats_capture_acquired = []
            stats_capture_dwell = []
            stats_moving_met = []
            stats_moving_good_fraction = []
            stats_moving_xy_good_sample = []
            stats_moving_z_good_sample = []
            stats_moving_velocity_good_sample = []
            stats_moving_good_sample = []
            stats_local_tracking_ready = []
            stats_local_tracking_events = []
            stats_local_tracking_soft_quality = []
            stats_local_tracking_xy_fraction = []
            stats_local_tracking_velocity_fraction = []
            stats_local_tracking_z_fraction = []
            stats_local_tracking_drift = []
            stats_cmd_saturation = {
                key: [] for key in ("any", "roll", "pitch", "yaw", "thrust")
            }
            stats_abs_commands = {
                key: []
                for key in (
                    "policy_roll",
                    "policy_pitch",
                    "controller_roll",
                    "controller_pitch",
                    "final_roll",
                    "final_pitch",
                )
            }
            stats_vertical_met = []
            stats_vertical_good_fraction = []
            stats_vertical_position_good_sample = []
            stats_vertical_velocity_good_sample = []
            stats_vertical_good_sample = []
            stats_phases = Counter()
            stats_primitives = Counter()
            stats_primitive_xy = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_z = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_velocity = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_vertical_velocity = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_xy_good = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_z_good = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_velocity_good = {key: [] for key in PRIMITIVE_CODES}
            stats_primitive_vertical_velocity_good = {
                key: [] for key in PRIMITIVE_CODES
            }
            stats_primitive_good = {key: [] for key in PRIMITIVE_CODES}
            stats_stopped_xy = []
            stats_stopped_speed_xy = []
            stats_stopped_speed_z = []
            stats_desired_approach_speed = []
            stats_allowed_stopped_speed = []
            stats_stopped_velocity_error = []
            stats_stopped_xy_met = []
            stats_stopped_z_met = []
            stats_stopped_position_met = []
            stats_stopped_stationary = []
            stats_braking_progress = []
            stats_reward_terms = {key: [] for key in REWARD_TERM_KEYS}

            for _ in range(args.rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                critic_obs_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=device)
                episode_start_t = torch.as_tensor(
                    episode_start,
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    act_fn = (
                        policy.act_deterministic
                        if args.deterministic_actions
                        else policy.act
                    )
                    action_t, logprob_t, value_t, recurrent_hidden = act_fn(
                        obs_t, critic_obs_t, recurrent_hidden, episode_start_t
                    )
                    raw_value_t = (
                        value_normalizer.denormalize(value_t)
                        if value_normalizer is not None
                        else value_t
                    )
                    if reference_policy is not None:
                        _, reference_hidden = reference_policy.deterministic_action(
                            obs_t,
                            reference_hidden,
                            episode_start_t,
                        )
                actions = action_t.detach().cpu().numpy().astype(np.float32)
                next_obs, rewards, dones, infos = env.step(actions)
                next_critic_obs = env.build_critic_obs(next_obs)

                obs_buf.append(obs.copy())
                critic_obs_buf.append(critic_obs.copy())
                action_buf.append(actions.copy())
                logprob_buf.append(logprob_t.detach().cpu().numpy().astype(np.float32))
                value_buf.append(
                    raw_value_t.detach().cpu().numpy().astype(np.float32)
                )
                reward_buf.append(rewards.copy())
                done_buf.append(dones.copy())
                episode_start_buf.append(episode_start.copy())

                stats_rewards.extend(rewards.astype(float).tolist())
                total_steps += args.num_envs
                for env_id, info in enumerate(infos):
                    reason = info.get("done_reason", "running")
                    xy_err = float(info.get("xy_err", 0.0))
                    z_err = float(info.get("z_err", 0.0))
                    speed_xy = float(info.get("speed_xy", 0.0))
                    speed_z = float(info.get("speed_z", 0.0))
                    tracking_vel_err = float(info.get("tracking_velocity_error", 0.0))
                    reward_vel_err = float(info.get("reward_velocity_error", 0.0))
                    reward_vel_correction_speed = float(
                        info.get("reward_velocity_correction_speed", 0.0)
                    )
                    vertical_vel_err = float(info.get("vertical_velocity_error", 0.0))
                    dwell_fraction = float(info.get("goal_dwell_fraction", 0.0))
                    phase = str(info.get("target_phase", "unknown"))
                    stats_reasons[reason] += 1
                    stats_xy.append(xy_err)
                    stats_z.append(z_err)
                    stats_signed_z.append(float(info.get("signed_z_err", 0.0)))
                    stats_goal_progress.append(float(info.get("goal_xy_progress", 0.0)))
                    stats_inside.append(1.0 if info.get("inside_goal_zone") else 0.0)
                    stats_dwell.append(dwell_fraction)
                    stats_speed_xy.append(speed_xy)
                    stats_speed_z.append(speed_z)
                    stats_target_speed.append(float(info.get("target_speed", 0.0)))
                    stats_tracking_vel.append(tracking_vel_err)
                    stats_reward_velocity_error.append(reward_vel_err)
                    stats_reward_velocity_correction_speed.append(
                        reward_vel_correction_speed
                    )
                    stats_vertical_vel.append(vertical_vel_err)
                    stats_target_distance.append(float(info.get("target_distance", 0.0)))
                    stats_target_progress.append(float(info.get("target_progress_fraction", 0.0)))
                    stats_capture_acquired.append(1.0 if info.get("capture_acquired") else 0.0)
                    stats_capture_dwell.append(float(info.get("capture_dwell_fraction", 0.0)))
                    stats_moving_met.append(1.0 if info.get("moving_success_met") else 0.0)
                    stats_moving_good_fraction.append(float(info.get("moving_good_fraction", 0.0)))
                    local_ready = bool(
                        info.get("local_tracking_window_ready", False)
                    )
                    stats_local_tracking_ready.append(1.0 if local_ready else 0.0)
                    if local_ready:
                        stats_local_tracking_soft_quality.append(
                            float(
                                info.get(
                                    "local_tracking_soft_joint_quality", 0.0
                                )
                            )
                        )
                        stats_local_tracking_xy_fraction.append(
                            float(
                                info.get(
                                    "local_tracking_xy_good_fraction", 0.0
                                )
                            )
                        )
                        stats_local_tracking_velocity_fraction.append(
                            float(
                                info.get(
                                    "local_tracking_velocity_good_fraction", 0.0
                                )
                            )
                        )
                        stats_local_tracking_z_fraction.append(
                            float(
                                info.get(
                                    "local_tracking_z_good_fraction", 0.0
                                )
                            )
                        )
                        stats_local_tracking_drift.append(
                            float(
                                info.get(
                                    "local_tracking_xy_drift_delta_m", 0.0
                                )
                            )
                        )
                    if info.get("local_tracking_reward_event", False):
                        stats_local_tracking_events.append(1.0)
                    stats_cmd_saturation["any"].append(
                        1.0 if info.get("cmd_any_saturated", False) else 0.0
                    )
                    for axis in ("roll", "pitch", "yaw", "thrust"):
                        stats_cmd_saturation[axis].append(
                            1.0
                            if info.get(f"cmd_{axis}_saturated", False)
                            else 0.0
                        )
                    stats_abs_commands["policy_roll"].append(
                        abs(float(info.get("policy_cmd_roll_rate", 0.0)))
                    )
                    stats_abs_commands["policy_pitch"].append(
                        abs(float(info.get("policy_cmd_pitch_rate", 0.0)))
                    )
                    stats_abs_commands["controller_roll"].append(
                        abs(float(info.get("controller_cmd_roll_rate", 0.0)))
                    )
                    stats_abs_commands["controller_pitch"].append(
                        abs(float(info.get("controller_cmd_pitch_rate", 0.0)))
                    )
                    stats_abs_commands["final_roll"].append(
                        abs(float(info.get("cmd_roll_rate", 0.0)))
                    )
                    stats_abs_commands["final_pitch"].append(
                        abs(float(info.get("cmd_pitch_rate", 0.0)))
                    )
                    moving_track_eligible = bool(info.get("moving_track_eligible", False))
                    if moving_track_eligible:
                        stats_moving_xy_good_sample.append(
                            1.0 if info.get("moving_xy_good") else 0.0
                        )
                        stats_moving_z_good_sample.append(
                            1.0 if info.get("moving_z_good") else 0.0
                        )
                        stats_moving_velocity_good_sample.append(
                            1.0 if info.get("moving_velocity_good") else 0.0
                        )
                        stats_moving_good_sample.append(
                            1.0 if info.get("moving_good") else 0.0
                        )
                    vertical_motion_eligible = bool(
                        info.get("vertical_motion_eligible", False)
                    )
                    if vertical_motion_eligible:
                        stats_vertical_motion_vel.append(vertical_vel_err)
                        stats_vertical_met.append(
                            1.0 if info.get("vertical_success_met") else 0.0
                        )
                        stats_vertical_good_fraction.append(
                            float(info.get("vertical_good_fraction", 0.0))
                        )
                        stats_vertical_position_good_sample.append(
                            1.0 if info.get("vertical_position_good") else 0.0
                        )
                        stats_vertical_velocity_good_sample.append(
                            1.0 if info.get("vertical_velocity_good") else 0.0
                        )
                        stats_vertical_good_sample.append(
                            1.0 if info.get("vertical_good") else 0.0
                        )
                    stats_phases[phase] += 1
                    primitive_id = str(info.get("primitive_id", "unknown"))
                    stats_primitives[primitive_id] += 1
                    if primitive_id in stats_primitive_xy and moving_track_eligible:
                        stats_primitive_xy[primitive_id].append(xy_err)
                        stats_primitive_z[primitive_id].append(z_err)
                        stats_primitive_velocity[primitive_id].append(tracking_vel_err)
                        stats_primitive_xy_good[primitive_id].append(
                            1.0 if info.get("moving_xy_good") else 0.0
                        )
                        stats_primitive_z_good[primitive_id].append(
                            1.0 if info.get("moving_z_good") else 0.0
                        )
                        stats_primitive_velocity_good[primitive_id].append(
                            1.0 if info.get("moving_velocity_good") else 0.0
                        )
                        stats_primitive_good[primitive_id].append(
                            1.0 if info.get("moving_good") else 0.0
                        )
                    if primitive_id in stats_primitive_xy and vertical_motion_eligible:
                        stats_primitive_vertical_velocity[primitive_id].append(
                            vertical_vel_err
                        )
                        stats_primitive_vertical_velocity_good[primitive_id].append(
                            1.0 if info.get("vertical_velocity_good") else 0.0
                        )
                    stats_braking_progress.append(float(info.get("braking_progress", 0.0)))
                    if bool(info.get("target_stopped", False)):
                        stats_stopped_xy.append(xy_err)
                        stats_stopped_speed_xy.append(speed_xy)
                        stats_stopped_speed_z.append(speed_z)
                        stats_desired_approach_speed.append(
                            float(info.get("desired_approach_speed", 0.0))
                        )
                        stats_allowed_stopped_speed.append(
                            float(info.get("allowed_stopped_speed", 0.0))
                        )
                        stats_stopped_velocity_error.append(
                            float(info.get("stopped_velocity_error", 0.0))
                        )
                        stats_stopped_xy_met.append(
                            1.0 if info.get("stopped_xy_good") else 0.0
                        )
                        stats_stopped_z_met.append(
                            1.0 if info.get("stopped_z_good") else 0.0
                        )
                        stats_stopped_position_met.append(
                            1.0
                            if info.get("stopped_xy_good")
                            and info.get("stopped_z_good")
                            else 0.0
                        )
                        stats_stopped_stationary.append(
                            1.0 if info.get("stopped_speed_good") else 0.0
                        )

                    episode_step_counts[env_id] += 1
                    episode_xy_sums[env_id] += xy_err
                    episode_xy_max[env_id] = max(episode_xy_max[env_id], xy_err)
                    episode_z_sums[env_id] += z_err
                    episode_z_max[env_id] = max(episode_z_max[env_id], z_err)
                    episode_speed_xy_sums[env_id] += speed_xy
                    episode_speed_xy_max[env_id] = max(episode_speed_xy_max[env_id], speed_xy)
                    episode_tracking_vel_sums[env_id] += tracking_vel_err
                    episode_tracking_vel_max[env_id] = max(
                        episode_tracking_vel_max[env_id],
                        tracking_vel_err,
                    )
                    if vertical_motion_eligible:
                        episode_vertical_vel_sums[env_id] += vertical_vel_err
                        episode_vertical_vel_max[env_id] = max(
                            episode_vertical_vel_max[env_id],
                            vertical_vel_err,
                        )
                        episode_vertical_vel_samples[env_id] += 1
                    episode_max_dwell[env_id] = max(
                        episode_max_dwell[env_id],
                        dwell_fraction,
                    )
                    if phase in episode_phase_steps:
                        episode_phase_steps[phase][env_id] += 1
                    for key in REWARD_TERM_KEYS:
                        reward_value = float(info.get(key, 0.0))
                        stats_reward_terms[key].append(reward_value)
                        episode_reward_returns[key][env_id] += reward_value
                    if dones[env_id]:
                        episode_steps = max(1, int(episode_step_counts[env_id]))
                        row = {
                            "episode": int(info.get("episode_id", 0)),
                            "env_id": int(env_id),
                            "total_steps": int(total_steps),
                            "episode_steps": int(info.get("step_id", 0)),
                            "return": float(info.get("episode_return", 0.0)),
                            "done_reason": str(reason),
                            "final_goal_xy_err": float(info.get("xy_err", 0.0)),
                            "final_z_err": float(info.get("z_err", 0.0)),
                            "final_goal_distance": float(info.get("goal_distance", 0.0)),
                            "final_goal_rel_x": float(info.get("goal_rel_x", 0.0)),
                            "final_goal_rel_y": float(info.get("goal_rel_y", 0.0)),
                            "final_goal_rel_z": float(info.get("goal_rel_z", 0.0)),
                            "final_signed_z_err": float(info.get("signed_z_err", 0.0)),
                            "final_speed_xy": float(info.get("speed_xy", 0.0)),
                            "final_speed_z": float(info.get("speed_z", 0.0)),
                            "final_yaw": float(info.get("yaw", 0.0)),
                            "final_yaw_rate": float(info.get("yaw_rate", 0.0)),
                            "yaw_start": float(info.get("yaw_start", 0.0)),
                            "final_cmd_roll_rate": float(info.get("cmd_roll_rate", 0.0)),
                            "final_cmd_pitch_rate": float(info.get("cmd_pitch_rate", 0.0)),
                            "final_cmd_yaw_rate": float(info.get("cmd_yaw_rate", 0.0)),
                            "final_cmd_thrust": float(info.get("cmd_thrust", 0.0)),
                            "final_policy_cmd_roll_rate": float(
                                info.get("policy_cmd_roll_rate", 0.0)
                            ),
                            "final_policy_cmd_pitch_rate": float(
                                info.get("policy_cmd_pitch_rate", 0.0)
                            ),
                            "final_policy_cmd_yaw_rate": float(
                                info.get("policy_cmd_yaw_rate", 0.0)
                            ),
                            "final_policy_cmd_thrust": float(
                                info.get("policy_cmd_thrust", 0.0)
                            ),
                            "final_controller_cmd_roll_rate": float(
                                info.get("controller_cmd_roll_rate", 0.0)
                            ),
                            "final_controller_cmd_pitch_rate": float(
                                info.get("controller_cmd_pitch_rate", 0.0)
                            ),
                            "final_controller_cmd_yaw_rate": float(
                                info.get("controller_cmd_yaw_rate", 0.0)
                            ),
                            "final_controller_cmd_thrust": float(
                                info.get("controller_cmd_thrust", 0.0)
                            ),
                            "final_cmd_any_saturated": bool(
                                info.get("cmd_any_saturated", False)
                            ),
                            "final_target_phase": str(info.get("target_phase", "unknown")),
                            "final_primitive_id": str(info.get("primitive_id", "unknown")),
                            "final_primitive_code": int(info.get("primitive_code", -1)),
                            "motion_sequence_ids": str(info.get("motion_sequence_ids", "")),
                            "trajectory_duration_sec": float(
                                info.get("trajectory_duration_sec", 0.0)
                            ),
                            "final_target_curvature_per_m": float(
                                info.get("target_curvature_per_m", 0.0)
                            ),
                            "final_target_progress_fraction": float(info.get("target_progress_fraction", 0.0)),
                            "final_target_speed": float(info.get("target_speed", 0.0)),
                            "final_target_distance": float(info.get("target_distance", 0.0)),
                            "final_tracking_velocity_error": float(info.get("tracking_velocity_error", 0.0)),
                            "final_vertical_velocity_error": float(
                                info.get("vertical_velocity_error", 0.0)
                            ),
                            "final_desired_approach_speed": float(
                                info.get("desired_approach_speed", 0.0)
                            ),
                            "final_allowed_stopped_speed": float(
                                info.get("allowed_stopped_speed", 0.0)
                            ),
                            "final_stopped_velocity_error": float(
                                info.get("stopped_velocity_error", 0.0)
                            ),
                            "line_yaw_deg": float(info.get("line_yaw_deg", 0.0)),
                            "line_dir_x": float(info.get("line_dir_x", 1.0)),
                            "line_dir_y": float(info.get("line_dir_y", 0.0)),
                            "sampled_line_length_m": float(
                                info.get("sampled_line_length_m", args.line_length_m)
                            ),
                            "desired_endpoint_x": float(info.get("desired_endpoint_x", 0.0)),
                            "desired_endpoint_y": float(info.get("desired_endpoint_y", 0.0)),
                            "desired_endpoint_z": float(info.get("desired_endpoint_z", 0.0)),
                            "target_endpoint_x": float(info.get("target_endpoint_x", 0.0)),
                            "target_endpoint_y": float(info.get("target_endpoint_y", 0.0)),
                            "target_endpoint_z": float(info.get("target_endpoint_z", 0.0)),
                            "capture_acquired": bool(info.get("capture_acquired", False)),
                            "capture_dwell_steps": int(info.get("capture_dwell_steps", 0)),
                            "required_capture_steps": int(info.get("required_capture_steps", 0)),
                            "capture_dwell_fraction": float(info.get("capture_dwell_fraction", 0.0)),
                            "moving_success_met": bool(info.get("moving_success_met", False)),
                            "moving_track_dwell_steps": int(info.get("moving_track_dwell_steps", 0)),
                            "required_moving_track_steps": int(info.get("required_moving_track_steps", 0)),
                            "moving_eligible_steps": int(info.get("moving_eligible_steps", 0)),
                            "moving_good_steps": int(info.get("moving_good_steps", 0)),
                            "moving_good_fraction": float(info.get("moving_good_fraction", 0.0)),
                            "moving_xy_good_steps": int(info.get("moving_xy_good_steps", 0)),
                            "moving_z_good_steps": int(info.get("moving_z_good_steps", 0)),
                            "moving_velocity_good_steps": int(
                                info.get("moving_velocity_good_steps", 0)
                            ),
                            "moving_xy_good_fraction": float(
                                info.get("moving_xy_good_fraction", 0.0)
                            ),
                            "moving_z_good_fraction": float(
                                info.get("moving_z_good_fraction", 0.0)
                            ),
                            "moving_velocity_good_fraction": float(
                                info.get("moving_velocity_good_fraction", 0.0)
                            ),
                            "local_tracking_soft_joint_quality": float(
                                info.get(
                                    "local_tracking_soft_joint_quality", 0.0
                                )
                            ),
                            "local_tracking_xy_good_fraction": float(
                                info.get("local_tracking_xy_good_fraction", 0.0)
                            ),
                            "local_tracking_velocity_good_fraction": float(
                                info.get(
                                    "local_tracking_velocity_good_fraction", 0.0
                                )
                            ),
                            "local_tracking_z_good_fraction": float(
                                info.get("local_tracking_z_good_fraction", 0.0)
                            ),
                            "local_tracking_xy_drift_delta_m": float(
                                info.get("local_tracking_xy_drift_delta_m", 0.0)
                            ),
                            "vertical_success_met": bool(
                                info.get("vertical_success_met", False)
                            ),
                            "vertical_eligible_steps": int(
                                info.get("vertical_eligible_steps", 0)
                            ),
                            "vertical_position_good_steps": int(
                                info.get("vertical_position_good_steps", 0)
                            ),
                            "vertical_velocity_good_steps": int(
                                info.get("vertical_velocity_good_steps", 0)
                            ),
                            "vertical_good_steps": int(
                                info.get("vertical_good_steps", 0)
                            ),
                            "vertical_position_good_fraction": float(
                                info.get("vertical_position_good_fraction", 0.0)
                            ),
                            "vertical_velocity_good_fraction": float(
                                info.get("vertical_velocity_good_fraction", 0.0)
                            ),
                            "vertical_good_fraction": float(
                                info.get("vertical_good_fraction", 0.0)
                            ),
                            "stopped_track_dwell_steps": int(info.get("stopped_track_dwell_steps", 0)),
                            "required_stopped_track_steps": int(info.get("required_stopped_track_steps", 0)),
                            "stopped_eligible_steps": int(info.get("stopped_eligible_steps", 0)),
                            "stopped_xy_good_steps": int(info.get("stopped_xy_good_steps", 0)),
                            "stopped_z_good_steps": int(info.get("stopped_z_good_steps", 0)),
                            "stopped_speed_good_steps": int(
                                info.get("stopped_speed_good_steps", 0)
                            ),
                            "stopped_xy_good_fraction": float(
                                info.get("stopped_xy_good_fraction", 0.0)
                            ),
                            "stopped_z_good_fraction": float(
                                info.get("stopped_z_good_fraction", 0.0)
                            ),
                            "stopped_speed_good_fraction": float(
                                info.get("stopped_speed_good_fraction", 0.0)
                            ),
                            "mean_goal_xy_err": float(episode_xy_sums[env_id] / episode_steps),
                            "max_goal_xy_err": float(episode_xy_max[env_id]),
                            "mean_z_err": float(episode_z_sums[env_id] / episode_steps),
                            "max_z_err": float(episode_z_max[env_id]),
                            "mean_speed_xy": float(episode_speed_xy_sums[env_id] / episode_steps),
                            "max_speed_xy": float(episode_speed_xy_max[env_id]),
                            "mean_tracking_velocity_error": float(
                                episode_tracking_vel_sums[env_id] / episode_steps
                            ),
                            "max_tracking_velocity_error": float(
                                episode_tracking_vel_max[env_id]
                            ),
                            "mean_vertical_motion_velocity_error": float(
                                episode_vertical_vel_sums[env_id]
                                / episode_vertical_vel_samples[env_id]
                            )
                            if episode_vertical_vel_samples[env_id] > 0
                            else 0.0,
                            "max_vertical_motion_velocity_error": float(
                                episode_vertical_vel_max[env_id]
                            ),
                            "max_goal_dwell_fraction": float(episode_max_dwell[env_id]),
                            "capture_phase_steps": int(episode_phase_steps["capture"][env_id]),
                            "moving_phase_steps": int(episode_phase_steps["moving"][env_id]),
                            "decelerating_phase_steps": int(
                                episode_phase_steps["decelerating"][env_id]
                            ),
                            "stopped_phase_steps": int(episode_phase_steps["stopped"][env_id]),
                            **{
                                f"return_{key.removeprefix('reward_')}": float(
                                    episode_reward_returns[key][env_id]
                                )
                                for key in REWARD_TERM_KEYS
                            },
                            "success": str(reason) == "success",
                        }
                        episode_rows.append(row)
                        append_csv_row(metrics_dir / "episode_metrics.csv", episode_fields, row)
                        episode_step_counts[env_id] = 0
                        episode_xy_sums[env_id] = 0.0
                        episode_xy_max[env_id] = 0.0
                        episode_z_sums[env_id] = 0.0
                        episode_z_max[env_id] = 0.0
                        episode_speed_xy_sums[env_id] = 0.0
                        episode_speed_xy_max[env_id] = 0.0
                        episode_tracking_vel_sums[env_id] = 0.0
                        episode_tracking_vel_max[env_id] = 0.0
                        episode_vertical_vel_sums[env_id] = 0.0
                        episode_vertical_vel_max[env_id] = 0.0
                        episode_vertical_vel_samples[env_id] = 0
                        episode_max_dwell[env_id] = 0.0
                        for phase_name in episode_phase_steps:
                            episode_phase_steps[phase_name][env_id] = 0
                        for key in REWARD_TERM_KEYS:
                            episode_reward_returns[key][env_id] = 0.0
                obs = next_obs
                critic_obs = next_critic_obs
                episode_start = dones.copy()
                if total_steps >= args.num_env_steps:
                    break

            with torch.no_grad():
                last_values_t = policy.value(
                    torch.as_tensor(critic_obs, dtype=torch.float32, device=device)
                )
                if value_normalizer is not None:
                    last_values_t = value_normalizer.denormalize(last_values_t)
                last_values = last_values_t.detach().cpu().numpy()
            rewards_np = np.asarray(reward_buf, dtype=np.float32)
            dones_np = np.asarray(done_buf, dtype=bool)
            values_np = np.asarray(value_buf, dtype=np.float32)
            returns, advantages = compute_gae_vec(rewards_np, dones_np, values_np, last_values, args.gamma, args.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_seq = torch.as_tensor(np.asarray(obs_buf, dtype=np.float32), dtype=torch.float32, device=device)
            critic_obs_seq = torch.as_tensor(
                np.asarray(critic_obs_buf, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            actions_seq = torch.as_tensor(np.asarray(action_buf, dtype=np.float32), dtype=torch.float32, device=device)
            old_logprob_seq = torch.as_tensor(np.asarray(logprob_buf, dtype=np.float32), dtype=torch.float32, device=device)
            raw_old_values_seq = torch.as_tensor(
                values_np, dtype=torch.float32, device=device
            )
            raw_returns_seq = torch.as_tensor(
                returns, dtype=torch.float32, device=device
            )
            if value_normalizer is not None:
                if not args.evaluation_only:
                    value_normalizer.update(
                        raw_returns_seq,
                        policy.critic,
                        optimizer=critic_optimizer,
                    )
                old_values_seq = value_normalizer.normalize(raw_old_values_seq)
                returns_seq = value_normalizer.normalize(raw_returns_seq)
            else:
                old_values_seq = raw_old_values_seq
                returns_seq = raw_returns_seq
            advantages_seq = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            episode_starts_seq = torch.as_tensor(
                np.asarray(episode_start_buf, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )

            schedule_progress = min(1.0, float(total_steps) / max(1.0, float(args.num_env_steps)))
            if args.lr_schedule == "linear":
                actor_lr = max(args.min_lr, args.lr * (1.0 - schedule_progress))
                critic_lr = max(args.min_lr, args.critic_lr * (1.0 - schedule_progress))
            else:
                actor_lr = args.lr
                critic_lr = args.critic_lr
            for group in actor_optimizer.param_groups:
                group["lr"] = actor_lr * float(group.get("lr_multiplier", 1.0))
            for group in critic_optimizer.param_groups:
                group["lr"] = critic_lr
            actor_backbone_lr = actor_lr * args.actor_backbone_lr_multiplier
            actor_recurrent_lr = (
                actor_lr * args.actor_recurrent_lr_multiplier
                if actor_recurrent_parameters
                else 0.0
            )
            entropy_coef = (
                args.entropy_coef
                + schedule_progress * (args.entropy_coef_final - args.entropy_coef)
            )

            pg_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            clip_fractions = []
            reference_kls = []
            value_clip_fractions = []
            kl_early_stop = False

            for _ in range(0 if args.evaluation_only else args.ppo_epoch):
                env_indices = np.arange(args.num_envs)
                np.random.shuffle(env_indices)
                for mb_env_ids in np.array_split(env_indices, args.num_mini_batch):
                    if mb_env_ids.size == 0:
                        continue
                    mb = torch.as_tensor(mb_env_ids, dtype=torch.long, device=device)
                    new_logprob, entropy, value, _ = policy.evaluate_actions_sequence(
                        obs_seq[:, mb],
                        critic_obs_seq[:, mb],
                        actions_seq[:, mb],
                        rollout_initial_hidden[mb],
                        episode_starts_seq[:, mb],
                    )
                    log_ratio = new_logprob - old_logprob_seq[:, mb]
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantages_seq[:, mb]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                        * advantages_seq[:, mb]
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_delta = value - old_values_seq[:, mb]
                    with torch.no_grad():
                        value_clip_fraction = (
                            (value_delta.abs() > args.value_clip_param)
                            .float()
                            .mean()
                            if args.value_clip_param > 0.0
                            else torch.zeros((), device=device)
                        )
                    value_clip_fractions.append(
                        float(value_clip_fraction.item())
                    )
                    if args.value_loss_type == "unclipped_huber":
                        value_loss = torch.nn.functional.smooth_l1_loss(
                            value,
                            returns_seq[:, mb],
                        )
                    else:
                        value_clipped = old_values_seq[:, mb] + torch.clamp(
                            value_delta,
                            -args.value_clip_param,
                            args.value_clip_param,
                        )
                        value_loss_unclipped = torch.square(
                            value - returns_seq[:, mb]
                        )
                        value_loss_clipped = torch.square(
                            value_clipped - returns_seq[:, mb]
                        )
                        value_loss = 0.5 * torch.maximum(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    entropy_loss = entropy.mean()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = ((ratio - 1.0).abs() > args.clip_param).float().mean()
                    approx_kls.append(float(approx_kl.item()))
                    clip_fractions.append(float(clip_fraction.item()))
                    if args.target_kl > 0.0 and float(approx_kl.item()) > args.target_kl:
                        kl_early_stop = True
                        break

                    reference_kl = torch.zeros((), device=device)
                    if reference_policy is not None:
                        current_mean, current_std, _ = (
                            policy.distribution_parameters_sequence(
                                obs_seq[:, mb],
                                rollout_initial_hidden[mb],
                                episode_starts_seq[:, mb],
                            )
                        )
                        with torch.no_grad():
                            reference_mean, reference_std, _ = (
                                reference_policy.distribution_parameters_sequence(
                                    obs_seq[:, mb],
                                    rollout_initial_reference_hidden[mb],
                                    episode_starts_seq[:, mb],
                                )
                            )
                        reference_kl = (
                            torch.log(reference_std / current_std)
                            + (
                                torch.square(current_std)
                                + torch.square(current_mean - reference_mean)
                            )
                            / (2.0 * torch.square(reference_std))
                            - 0.5
                        ).sum(dim=-1).mean()
                        reference_kls.append(float(reference_kl.item()))
                    actor_loss = (
                        policy_loss
                        - entropy_coef * entropy_loss
                        + args.reference_kl_coef * reference_kl
                    )
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm)
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    (args.value_loss_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm)
                    critic_optimizer.step()
                    pg_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_loss.item()))
                if kl_early_stop:
                    break

            update += 1
            if not args.evaluation_only:
                save_policy_checkpoint(
                    model_dir / "actor_critic.pt",
                    policy,
                    update,
                    total_steps,
                    actor_optimizer,
                    critic_optimizer,
                    env._rng,
                    value_normalizer,
                    reference_policy,
                )
            if (
                not args.evaluation_only
                and args.save_interval > 0
                and update % args.save_interval == 0
            ):
                save_policy_checkpoint(
                    model_dir / f"actor_critic_update_{update}.pt",
                    policy,
                    update,
                    total_steps,
                    actor_optimizer,
                    critic_optimizer,
                    env._rng,
                    value_normalizer,
                    reference_policy,
                )
            with torch.no_grad():
                value_pred_t = policy.value(
                    critic_obs_seq.reshape(-1, env.critic_obs_dim)
                )
                if value_normalizer is not None:
                    value_pred_t = value_normalizer.denormalize(value_pred_t)
                value_pred = value_pred_t.detach().cpu().numpy()
            recurrent_hidden = recurrent_hidden.detach()
            if reference_hidden is not None:
                reference_hidden = reference_hidden.detach()
            actions_np = np.asarray(action_buf, dtype=np.float32)
            mean_reward_terms = {
                f"mean_{key}": float(np.mean(values)) if values else 0.0
                for key, values in stats_reward_terms.items()
            }
            success_count = int(stats_reasons.get("success", 0))
            timeout_count = int(stats_reasons.get("timeout", 0))
            other_done_count = int(sum(v for k, v in stats_reasons.items() if k not in ("running", "success", "timeout")))
            elapsed = max(1e-9, time.perf_counter() - start_wall)
            update_wall_sec = max(0.0, time.perf_counter() - update_start_wall)
            sps = total_steps / elapsed
            eta_sec = max(0.0, (args.num_env_steps - total_steps) / max(sps, 1e-9))
            wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
            estimated_finish_time = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time() + eta_sec),
            )
            phase_sample_count = max(1, sum(stats_phases.values()))
            update_row = {
                "update": int(update),
                "total_steps": int(total_steps),
                "wall_time": wall_time,
                "elapsed_wall_sec": float(elapsed),
                "update_wall_sec": float(update_wall_sec),
                "sps": float(sps),
                "eta_sec": float(eta_sec),
                "estimated_finish_time": estimated_finish_time,
                "mean_rollout_reward": float(np.mean(stats_rewards)) if stats_rewards else 0.0,
                "mean_xy_err": float(np.mean(stats_xy)) if stats_xy else 0.0,
                "max_xy_err": float(np.max(stats_xy)) if stats_xy else 0.0,
                "mean_z_err": float(np.mean(stats_z)) if stats_z else 0.0,
                "max_z_err": float(np.max(stats_z)) if stats_z else 0.0,
                "mean_signed_z_err": float(np.mean(stats_signed_z)) if stats_signed_z else 0.0,
                "mean_vertical_velocity_error": (
                    float(np.mean(stats_vertical_vel)) if stats_vertical_vel else 0.0
                ),
                "mean_vertical_motion_velocity_error": (
                    float(np.mean(stats_vertical_motion_vel))
                    if stats_vertical_motion_vel
                    else 0.0
                ),
                "max_vertical_motion_velocity_error": (
                    float(np.max(stats_vertical_motion_vel))
                    if stats_vertical_motion_vel
                    else 0.0
                ),
                "mean_goal_xy_progress": float(np.mean(stats_goal_progress)) if stats_goal_progress else 0.0,
                "goal_zone_fraction": float(np.mean(stats_inside)) if stats_inside else 0.0,
                "mean_goal_dwell_fraction": float(np.mean(stats_dwell)) if stats_dwell else 0.0,
                "max_goal_dwell_fraction": float(np.max(stats_dwell)) if stats_dwell else 0.0,
                "mean_speed_xy": float(np.mean(stats_speed_xy)) if stats_speed_xy else 0.0,
                "max_speed_xy": float(np.max(stats_speed_xy)) if stats_speed_xy else 0.0,
                "mean_speed_z": float(np.mean(stats_speed_z)) if stats_speed_z else 0.0,
                "max_speed_z": float(np.max(stats_speed_z)) if stats_speed_z else 0.0,
                "success_count": success_count,
                "timeout_count": timeout_count,
                "other_done_count": other_done_count,
                "done_reasons": json.dumps(dict(stats_reasons), sort_keys=True),
                "primitive_sample_counts": json.dumps(
                    dict(stats_primitives),
                    sort_keys=True,
                ),
                "primitive_mean_xy_err": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_xy.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_mean_z_err": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_z.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_mean_velocity_error": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_velocity.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_mean_vertical_velocity_error": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_vertical_velocity.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_xy_good_fraction": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_xy_good.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_z_good_fraction": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_z_good.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_velocity_good_fraction": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_velocity_good.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_vertical_velocity_good_fraction": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_vertical_velocity_good.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "primitive_good_fraction": json.dumps(
                    {
                        key: float(np.mean(values))
                        for key, values in stats_primitive_good.items()
                        if values
                    },
                    sort_keys=True,
                ),
                "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "reference_kl": (
                    float(np.mean(reference_kls)) if reference_kls else 0.0
                ),
                "value_clip_fraction": (
                    float(np.mean(value_clip_fractions))
                    if value_clip_fractions
                    else 0.0
                ),
                "kl_early_stop": bool(kl_early_stop),
                "actor_lr": float(actor_lr),
                "actor_backbone_lr": float(actor_backbone_lr),
                "actor_recurrent_lr": float(actor_recurrent_lr),
                "critic_lr": float(critic_lr),
                "temporal_gate_raw": float(policy.temporal_gate.detach().cpu()),
                "temporal_gate_effective": float(
                    torch.tanh(policy.temporal_gate).detach().cpu()
                ),
                "temporal_gate_frozen": bool(temporal_gate_frozen),
                "actor_recurrent_mode": args.actor_recurrent_mode,
                "entropy_coef_current": float(entropy_coef),
                "explained_variance": explained_variance(value_pred, returns.reshape(-1)),
                "return_mean": float(np.mean(returns)),
                "return_std": float(np.std(returns)),
                "raw_value_mean": float(np.mean(value_pred)),
                "raw_value_std": float(np.std(value_pred)),
                "popart_mean": (
                    float(value_normalizer.mean.detach().cpu())
                    if value_normalizer is not None
                    else 0.0
                ),
                "popart_std": (
                    float(value_normalizer.std.detach().cpu())
                    if value_normalizer is not None
                    else 1.0
                ),
                "action_mean": float(np.mean(actions_np)) if actions_np.size else 0.0,
                "action_std": float(np.std(actions_np)) if actions_np.size else 0.0,
                "action_abs_mean": float(np.mean(np.abs(actions_np))) if actions_np.size else 0.0,
                "cmd_saturation_fraction": float(
                    np.mean(stats_cmd_saturation["any"])
                ),
                "cmd_roll_saturation_fraction": float(
                    np.mean(stats_cmd_saturation["roll"])
                ),
                "cmd_pitch_saturation_fraction": float(
                    np.mean(stats_cmd_saturation["pitch"])
                ),
                "cmd_yaw_saturation_fraction": float(
                    np.mean(stats_cmd_saturation["yaw"])
                ),
                "cmd_thrust_saturation_fraction": float(
                    np.mean(stats_cmd_saturation["thrust"])
                ),
                "mean_abs_policy_cmd_roll_rate": float(
                    np.mean(stats_abs_commands["policy_roll"])
                ),
                "mean_abs_policy_cmd_pitch_rate": float(
                    np.mean(stats_abs_commands["policy_pitch"])
                ),
                "mean_abs_controller_cmd_roll_rate": float(
                    np.mean(stats_abs_commands["controller_roll"])
                ),
                "mean_abs_controller_cmd_pitch_rate": float(
                    np.mean(stats_abs_commands["controller_pitch"])
                ),
                "mean_abs_final_cmd_roll_rate": float(
                    np.mean(stats_abs_commands["final_roll"])
                ),
                "mean_abs_final_cmd_pitch_rate": float(
                    np.mean(stats_abs_commands["final_pitch"])
                ),
                "mean_target_speed": float(np.mean(stats_target_speed)) if stats_target_speed else 0.0,
                "mean_tracking_velocity_error": float(np.mean(stats_tracking_vel)) if stats_tracking_vel else 0.0,
                "mean_reward_velocity_error": (
                    float(np.mean(stats_reward_velocity_error))
                    if stats_reward_velocity_error
                    else 0.0
                ),
                "mean_reward_velocity_correction_speed": (
                    float(np.mean(stats_reward_velocity_correction_speed))
                    if stats_reward_velocity_correction_speed
                    else 0.0
                ),
                "mean_target_distance": float(np.mean(stats_target_distance)) if stats_target_distance else 0.0,
                "mean_target_progress_fraction": float(np.mean(stats_target_progress)) if stats_target_progress else 0.0,
                "capture_acquired_fraction": (
                    float(np.mean(stats_capture_acquired)) if stats_capture_acquired else 0.0
                ),
                "mean_capture_dwell_fraction": (
                    float(np.mean(stats_capture_dwell)) if stats_capture_dwell else 0.0
                ),
                "moving_success_met_fraction": float(np.mean(stats_moving_met)) if stats_moving_met else 0.0,
                "mean_moving_good_fraction": (
                    float(np.mean(stats_moving_good_fraction))
                    if stats_moving_good_fraction
                    else 0.0
                ),
                "moving_xy_good_sample_fraction": (
                    float(np.mean(stats_moving_xy_good_sample))
                    if stats_moving_xy_good_sample
                    else 0.0
                ),
                "moving_z_good_sample_fraction": (
                    float(np.mean(stats_moving_z_good_sample))
                    if stats_moving_z_good_sample
                    else 0.0
                ),
                "moving_velocity_good_sample_fraction": (
                    float(np.mean(stats_moving_velocity_good_sample))
                    if stats_moving_velocity_good_sample
                    else 0.0
                ),
                "moving_good_sample_fraction": (
                    float(np.mean(stats_moving_good_sample))
                    if stats_moving_good_sample
                    else 0.0
                ),
                "local_tracking_ready_fraction": (
                    float(np.mean(stats_local_tracking_ready))
                    if stats_local_tracking_ready
                    else 0.0
                ),
                "local_tracking_event_count": int(
                    len(stats_local_tracking_events)
                ),
                "mean_local_tracking_soft_joint_quality": (
                    float(np.mean(stats_local_tracking_soft_quality))
                    if stats_local_tracking_soft_quality
                    else 0.0
                ),
                "mean_local_tracking_xy_good_fraction": (
                    float(np.mean(stats_local_tracking_xy_fraction))
                    if stats_local_tracking_xy_fraction
                    else 0.0
                ),
                "mean_local_tracking_velocity_good_fraction": (
                    float(np.mean(stats_local_tracking_velocity_fraction))
                    if stats_local_tracking_velocity_fraction
                    else 0.0
                ),
                "mean_local_tracking_z_good_fraction": (
                    float(np.mean(stats_local_tracking_z_fraction))
                    if stats_local_tracking_z_fraction
                    else 0.0
                ),
                "mean_local_tracking_xy_drift_delta_m": (
                    float(np.mean(stats_local_tracking_drift))
                    if stats_local_tracking_drift
                    else 0.0
                ),
                "vertical_success_met_fraction": (
                    float(np.mean(stats_vertical_met)) if stats_vertical_met else 0.0
                ),
                "mean_vertical_good_fraction": (
                    float(np.mean(stats_vertical_good_fraction))
                    if stats_vertical_good_fraction
                    else 0.0
                ),
                "vertical_position_good_sample_fraction": (
                    float(np.mean(stats_vertical_position_good_sample))
                    if stats_vertical_position_good_sample
                    else 0.0
                ),
                "vertical_velocity_good_sample_fraction": (
                    float(np.mean(stats_vertical_velocity_good_sample))
                    if stats_vertical_velocity_good_sample
                    else 0.0
                ),
                "vertical_good_sample_fraction": (
                    float(np.mean(stats_vertical_good_sample))
                    if stats_vertical_good_sample
                    else 0.0
                ),
                "capture_phase_fraction": float(stats_phases.get("capture", 0) / phase_sample_count),
                "moving_phase_fraction": float(stats_phases.get("moving", 0) / phase_sample_count),
                "decelerating_phase_fraction": float(
                    stats_phases.get("decelerating", 0) / phase_sample_count
                ),
                "stopped_phase_fraction": float(stats_phases.get("stopped", 0) / phase_sample_count),
                "mean_stopped_xy_err": float(np.mean(stats_stopped_xy)) if stats_stopped_xy else 0.0,
                "mean_stopped_speed_xy": (
                    float(np.mean(stats_stopped_speed_xy)) if stats_stopped_speed_xy else 0.0
                ),
                "mean_stopped_speed_z": (
                    float(np.mean(stats_stopped_speed_z)) if stats_stopped_speed_z else 0.0
                ),
                "mean_desired_approach_speed": (
                    float(np.mean(stats_desired_approach_speed))
                    if stats_desired_approach_speed
                    else 0.0
                ),
                "mean_allowed_stopped_speed": (
                    float(np.mean(stats_allowed_stopped_speed))
                    if stats_allowed_stopped_speed
                    else 0.0
                ),
                "mean_stopped_velocity_error": (
                    float(np.mean(stats_stopped_velocity_error))
                    if stats_stopped_velocity_error
                    else 0.0
                ),
                "stopped_xy_zone_fraction": (
                    float(np.mean(stats_stopped_xy_met)) if stats_stopped_xy_met else 0.0
                ),
                "stopped_z_zone_fraction": (
                    float(np.mean(stats_stopped_z_met)) if stats_stopped_z_met else 0.0
                ),
                "stopped_position_zone_fraction": (
                    float(np.mean(stats_stopped_position_met)) if stats_stopped_position_met else 0.0
                ),
                "stopped_stationary_fraction": (
                    float(np.mean(stats_stopped_stationary)) if stats_stopped_stationary else 0.0
                ),
                "stopped_xy_good_sample_fraction": (
                    float(np.mean(stats_stopped_xy_met)) if stats_stopped_xy_met else 0.0
                ),
                "stopped_z_good_sample_fraction": (
                    float(np.mean(stats_stopped_z_met)) if stats_stopped_z_met else 0.0
                ),
                "stopped_speed_good_sample_fraction": (
                    float(np.mean(stats_stopped_stationary))
                    if stats_stopped_stationary
                    else 0.0
                ),
                "mean_braking_progress": (
                    float(np.mean(stats_braking_progress)) if stats_braking_progress else 0.0
                ),
                "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                **mean_reward_terms,
            }
            checkpoint_score = 0.0
            best_checkpoint_saved = False
            early_stop_triggered = False
            if args.best_checkpoint_window > 0:
                history_count = args.best_checkpoint_window - 1
                score_rows = [
                    *(update_rows[-history_count:] if history_count > 0 else []),
                    update_row,
                ]
                if len(score_rows) >= args.best_checkpoint_window:
                    checkpoint_score = float(
                        np.mean(
                            [
                                0.5 * row["moving_good_sample_fraction"]
                                + 0.4 * row["moving_xy_good_sample_fraction"]
                                + 0.1 * row["moving_success_met_fraction"]
                                for row in score_rows
                            ]
                        )
                    )
                    if checkpoint_score > best_checkpoint_score:
                        best_checkpoint_score = checkpoint_score
                        best_checkpoint_update = update
                        degradation_updates = 0
                        best_checkpoint_saved = True
                    elif (
                        checkpoint_score
                        < best_checkpoint_score - args.early_stop_drop_threshold
                    ):
                        degradation_updates += 1
                    else:
                        degradation_updates = 0
                    early_stop_triggered = bool(
                        args.early_stop_patience_updates > 0
                        and degradation_updates
                        >= args.early_stop_patience_updates
                    )
            update_row.update(
                {
                    "checkpoint_score": checkpoint_score,
                    "best_checkpoint_score": (
                        best_checkpoint_score
                        if math.isfinite(best_checkpoint_score)
                        else 0.0
                    ),
                    "best_checkpoint_update": int(best_checkpoint_update),
                    "degradation_updates": int(degradation_updates),
                    "early_stop_triggered": early_stop_triggered,
                }
            )
            update_rows.append(update_row)
            append_csv_row(metrics_dir / "update_metrics.csv", update_fields, update_row)
            write_tensorboard_scalars(writer, update_row, total_steps)
            if best_checkpoint_saved and not args.evaluation_only:
                save_policy_checkpoint(
                    model_dir / "actor_critic_best.pt",
                    policy,
                    update,
                    total_steps,
                    actor_optimizer,
                    critic_optimizer,
                    env._rng,
                    value_normalizer,
                    reference_policy,
                )
                dump_json(
                    model_dir / "actor_critic_best.json",
                    {
                        "update": int(update),
                        "total_steps": int(total_steps),
                        "window": int(args.best_checkpoint_window),
                        "score": float(best_checkpoint_score),
                        "metric": (
                            "0.5*moving_good_sample_fraction + "
                            "0.4*moving_xy_good_sample_fraction + "
                            "0.1*moving_success_met_fraction"
                        ),
                    },
                )
            print("=" * 88)
            print(
                f"[FastIrisLineFollowPPO] update={update}, "
                f"steps={total_steps}/{args.num_env_steps}, SPS={sps:.1f}"
            )
            print(
                f"  wall_time: {wall_time}, elapsed: {format_duration(elapsed)}, "
                f"update_time: {format_duration(update_wall_sec)}"
            )
            print(
                f"  ETA: {format_duration(eta_sec)}, "
                f"estimated_finish: {estimated_finish_time}"
            )
            print(f"  mean_rollout_reward: {update_row['mean_rollout_reward']:.4f}")
            print(f"  mean_follow_xy_err: {update_row['mean_xy_err']:.3f}, mean_z_err: {update_row['mean_z_err']:.3f}")
            print(
                f"  mean_speed_xy: {update_row['mean_speed_xy']:.3f}, "
                f"mean_tracking_velocity_error: {update_row['mean_tracking_velocity_error']:.3f}, "
                f"reward_velocity_error: {update_row['mean_reward_velocity_error']:.3f}, "
                f"correction_speed: {update_row['mean_reward_velocity_correction_speed']:.3f}"
            )
            print(
                f"  follow_zone_fraction: {update_row['goal_zone_fraction']:.3f}, "
                f"mean_dwell: {update_row['mean_goal_dwell_fraction']:.3f}, "
                f"capture_acquired_fraction: {update_row['capture_acquired_fraction']:.3f}, "
                f"moving_success_met_fraction: {update_row['moving_success_met_fraction']:.3f}, "
                f"mean_moving_good_fraction: {update_row['mean_moving_good_fraction']:.3f}"
            )
            print(
                "  moving_conditions: "
                f"xy={update_row['moving_xy_good_sample_fraction']:.3f}, "
                f"z={update_row['moving_z_good_sample_fraction']:.3f}, "
                f"velocity={update_row['moving_velocity_good_sample_fraction']:.3f}, "
                f"joint={update_row['moving_good_sample_fraction']:.3f}"
            )
            print(
                "  vertical_conditions: "
                f"mean_vz_error={update_row['mean_vertical_motion_velocity_error']:.3f}, "
                f"position={update_row['vertical_position_good_sample_fraction']:.3f}, "
                f"velocity={update_row['vertical_velocity_good_sample_fraction']:.3f}, "
                f"joint={update_row['vertical_good_sample_fraction']:.3f}, "
                f"gate={update_row['vertical_success_met_fraction']:.3f}"
            )
            print(
                f"  phases: capture={update_row['capture_phase_fraction']:.3f}, "
                f"moving={update_row['moving_phase_fraction']:.3f}, "
                f"decelerating={update_row['decelerating_phase_fraction']:.3f}, "
                f"stopped={update_row['stopped_phase_fraction']:.3f}"
            )
            print(
                f"  stopped: xy_err={update_row['mean_stopped_xy_err']:.3f}, "
                f"speed_xy={update_row['mean_stopped_speed_xy']:.3f}, "
                f"desired_speed={update_row['mean_desired_approach_speed']:.3f}, "
                f"velocity_error={update_row['mean_stopped_velocity_error']:.3f}"
            )
            print(
                f"  stopped_zones: xy={update_row['stopped_xy_zone_fraction']:.3f}, "
                f"z={update_row['stopped_z_zone_fraction']:.3f}, "
                f"position={update_row['stopped_position_zone_fraction']:.3f}, "
                f"stationary={update_row['stopped_stationary_fraction']:.3f}"
            )
            print(f"  done_reasons: {dict(stats_reasons)}")
            primitive_good_summary = {
                key: round(float(np.mean(values)), 3)
                for key, values in stats_primitive_good.items()
                if values
            }
            if primitive_good_summary:
                print(f"  primitive_good_fraction: {primitive_good_summary}")
            primitive_vertical_summary = {
                key: round(float(np.mean(values)), 3)
                for key, values in stats_primitive_vertical_velocity_good.items()
                if values
            }
            if primitive_vertical_summary:
                print(
                    "  primitive_vertical_velocity_good_fraction: "
                    f"{primitive_vertical_summary}"
                )
            print(
                f"  approx_kl: {update_row['approx_kl']:.6f}, "
                f"clip_fraction: {update_row['clip_fraction']:.3f}, "
                f"reference_kl: {update_row['reference_kl']:.6f}, "
                f"kl_early_stop: {update_row['kl_early_stop']}"
            )
            print(
                f"  actor_lr: base={update_row['actor_lr']:.7f}, "
                f"backbone={update_row['actor_backbone_lr']:.7f}, "
                f"recurrent={update_row['actor_recurrent_lr']:.7f}, "
                f"critic_lr: {update_row['critic_lr']:.7f}, "
                f"entropy_coef: {update_row['entropy_coef_current']:.7f}"
            )
            print(
                f"  temporal_gate: raw={update_row['temporal_gate_raw']:.6f}, "
                f"effective={update_row['temporal_gate_effective']:.6f}, "
                f"frozen={update_row['temporal_gate_frozen']}, "
                f"mode={update_row['actor_recurrent_mode']}"
            )
            print(
                f"  critic: explained_variance={update_row['explained_variance']:.3f}, "
                f"return={update_row['return_mean']:.3f}+/-{update_row['return_std']:.3f}, "
                f"value_clip_fraction={update_row['value_clip_fraction']:.3f}, "
                f"popart={update_row['popart_mean']:.3f}+/-{update_row['popart_std']:.3f}"
            )
            print(
                f"  control: action_abs_mean={update_row['action_abs_mean']:.3f}, "
                f"saturation={update_row['cmd_saturation_fraction']:.3f}, "
                f"roll(policy/helper/final)="
                f"{update_row['mean_abs_policy_cmd_roll_rate']:.4f}/"
                f"{update_row['mean_abs_controller_cmd_roll_rate']:.4f}/"
                f"{update_row['mean_abs_final_cmd_roll_rate']:.4f}, "
                f"pitch(policy/helper/final)="
                f"{update_row['mean_abs_policy_cmd_pitch_rate']:.4f}/"
                f"{update_row['mean_abs_controller_cmd_pitch_rate']:.4f}/"
                f"{update_row['mean_abs_final_cmd_pitch_rate']:.4f}"
            )
            print(
                "  local_tracking: "
                f"ready={update_row['local_tracking_ready_fraction']:.3f}, "
                f"events={update_row['local_tracking_event_count']}, "
                f"quality={update_row['mean_local_tracking_soft_joint_quality']:.3f}, "
                f"xy={update_row['mean_local_tracking_xy_good_fraction']:.3f}, "
                f"velocity={update_row['mean_local_tracking_velocity_good_fraction']:.3f}, "
                f"z={update_row['mean_local_tracking_z_good_fraction']:.3f}, "
                f"drift={update_row['mean_local_tracking_xy_drift_delta_m']:.3f}m"
            )
            print(
                "  reward_terms: "
                f"time={update_row['mean_reward_time']:.4f}, "
                f"progress={update_row['mean_reward_progress']:.4f}, "
                f"distance={update_row['mean_reward_distance']:.4f}, "
                f"speed={update_row['mean_reward_speed']:.4f}, "
                f"braking={update_row['mean_reward_braking']:.4f}, "
                f"stop_overspeed={update_row['mean_reward_stop_overspeed']:.4f}, "
                f"capture={update_row['mean_reward_capture']:.4f}, "
                f"moving_position={update_row['mean_reward_moving_position']:.4f}, "
                f"moving_velocity={update_row['mean_reward_moving_velocity']:.4f}, "
                f"moving_good={update_row['mean_reward_moving_good']:.4f}, "
                f"moving_progress={update_row['mean_reward_moving_progress']:.4f}, "
                f"local_tracking={update_row['mean_reward_local_tracking']:.4f}, "
                f"local_drift={update_row['mean_reward_local_drift']:.4f}, "
                f"stopped_position={update_row['mean_reward_stopped_position']:.4f}, "
                f"goal_zone={update_row['mean_reward_goal_zone']:.4f}, "
                f"dwell={update_row['mean_reward_dwell']:.4f}, "
                f"crash={update_row['mean_reward_crash']:.4f}, "
                f"timeout={update_row['mean_reward_timeout']:.4f}"
            )
            print(f"  policy_loss: {update_row['policy_loss']:.6f}, value_loss: {update_row['value_loss']:.6f}, entropy: {update_row['entropy']:.6f}")
            if args.live_plot_interval > 0 and update % args.live_plot_interval == 0:
                save_training_plots(run_dir, update_rows, episode_rows)
            if early_stop_triggered:
                early_stopped = True
                print(
                    "[FAST PPO] early stop: rolling checkpoint score "
                    f"{checkpoint_score:.4f} stayed more than "
                    f"{args.early_stop_drop_threshold:.4f} below best "
                    f"{best_checkpoint_score:.4f} for "
                    f"{degradation_updates} updates; best update="
                    f"{best_checkpoint_update}"
                )
                break
    except KeyboardInterrupt:
        print("\n[FAST PPO] interrupted")
        if (
            not args.evaluation_only
            and policy is not None
            and actor_optimizer is not None
            and critic_optimizer is not None
        ):
            save_policy_checkpoint(
                model_dir / "actor_critic_interrupted.pt",
                policy,
                update,
                total_steps,
                actor_optimizer,
                critic_optimizer,
                env._rng if env is not None else None,
                value_normalizer,
                reference_policy,
            )
            save_policy_checkpoint(
                model_dir / "actor_critic.pt",
                policy,
                update,
                total_steps,
                actor_optimizer,
                critic_optimizer,
                env._rng if env is not None else None,
                value_normalizer,
                reference_policy,
            )
            print("[FAST PPO] saved interrupted checkpoint")
    except BaseException as exc:
        print(
            f"[FAST PPO] fatal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if env is not None:
            env.close()
        save_training_plots(run_dir, update_rows, episode_rows)
        final_elapsed = max(0.0, time.perf_counter() - start_wall)
        success_episodes = sum(1 for row in episode_rows if row["success"])
        dump_json(
            run_dir / "run_summary.json",
            {
                "status": (
                    "early_stopped"
                    if early_stopped
                    else "completed"
                    if total_steps >= args.num_env_steps
                    else "interrupted"
                ),
                "started_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(run_started_at),
                ),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_wall_sec": final_elapsed,
                "elapsed_wall": format_duration(final_elapsed),
                "total_steps": int(total_steps),
                "updates": int(update),
                "average_sps": float(total_steps / max(final_elapsed, 1e-9)),
                "episodes": len(episode_rows),
                "success_episodes": int(success_episodes),
                "success_rate": (
                    float(success_episodes / len(episode_rows))
                    if episode_rows
                    else 0.0
                ),
                "last_update": update_rows[-1] if update_rows else None,
            },
        )
        if writer is not None:
            writer.flush()
            writer.close()
        if terminal_log is not None:
            log_file, original_stdout, original_stderr = terminal_log
            print(f"[FAST PPO] saved terminal output to {log_file.name}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
