from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pegasus_iris_fast_line_follow.train_pegasus_iris_fast_line_follow_ppo import parse_args


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class TrainingConfigTest(unittest.TestCase):
    def _parse(self, config_name: str):
        config_path = PACKAGE_DIR / "configs" / config_name
        with patch.object(sys, "argv", ["train", "--config", str(config_path)]):
            return parse_args()

    def test_optional_task_description_is_only_emitted_when_configured(self) -> None:
        stage3 = self._parse("stage3_vertical_pool_5hz_ratio_5to5.json")
        stage4 = self._parse("stage4_combined_long_pool_5hz_ratio_5to5.json")
        self.assertFalse(hasattr(stage3, "task_description"))
        self.assertIn("combined long-horizon pool", stage4.task_description)

    def test_stage4_uses_flat_result_layout_and_fresh_seed(self) -> None:
        args = self._parse("stage4_combined_long_pool_5hz_ratio_5to5.json")
        self.assertEqual(args.seed, 1)
        self.assertTrue(args.reset_rng_on_load)
        self.assertEqual(Path(args.results_root).name, "ppo_train")

    def test_stage5_advances_seed_and_episode_length(self) -> None:
        args = self._parse("stage5_combined_extra_long_pool_400k_seed2.json")
        self.assertEqual(args.seed, 2)
        self.assertEqual(args.episode_length, 400)
        self.assertEqual(args.num_env_steps, 400000)

    def test_stage5a_uses_corrective_reward_intermediate_curriculum(self) -> None:
        args = self._parse(
            "stage5a_corrective_reward_intermediate_pool_100k_seed3.json"
        )
        self.assertEqual(args.seed, 3)
        self.assertEqual(args.rollout_steps, 128)
        self.assertEqual(args.episode_length, 350)
        self.assertEqual(args.num_env_steps, 100000)
        self.assertTrue(args.reset_optimizer)
        self.assertAlmostEqual(args.gamma, 0.997)
        self.assertAlmostEqual(args.reward_velocity_scale, 0.05)
        self.assertAlmostEqual(args.reward_position_recovery_scale, 0.5)
        self.assertAlmostEqual(args.reward_velocity_correction_gain, 0.5)
        self.assertAlmostEqual(args.reward_velocity_correction_max_mps, 0.25)

    def test_stage5b_rolls_back_and_removes_raw_action_magnitude_penalty(self) -> None:
        args = self._parse("stage5b_progress_reward_bridge_50k_seed5.json")
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.num_env_steps, 50000)
        self.assertEqual(args.policy_ratio, 0.5)
        self.assertTrue(args.reset_optimizer)
        self.assertEqual(args.lr_schedule, "constant")
        self.assertAlmostEqual(args.lr, 1e-6)
        self.assertAlmostEqual(args.critic_lr, 5e-6)
        self.assertAlmostEqual(args.reward_control_scale, 0.0)
        self.assertAlmostEqual(args.reward_action_delta_scale, 0.005)
        self.assertAlmostEqual(args.reward_moving_progress_scale, 3.0)
        self.assertAlmostEqual(args.reward_position_recovery_scale, 0.1)
        self.assertIn("actor_critic_update_10.pt", args.load_checkpoint)
        self.assertEqual(args.best_checkpoint_window, 5)

    def test_stage7_inherits_baseline_and_overrides_credit_assignment(self) -> None:
        args = self._parse(
            "stage7_local_credit_popart_5hz_ratio_5to5_100k_seed7.json"
        )
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.rollout_steps, 256)
        self.assertEqual(args.policy_ratio, 0.5)
        self.assertEqual(args.actor_recurrent_mode, "frozen")
        self.assertTrue(args.use_popart)
        self.assertAlmostEqual(args.gae_lambda, 0.98)
        self.assertAlmostEqual(args.reward_local_tracking_scale, 2.0)
        self.assertAlmostEqual(args.reference_kl_coef, 0.05)
        self.assertEqual(args.motion_pool_config["min_segments"], 6)
        self.assertEqual(args.motion_pool_config["max_segments"], 8)
        self.assertIn("cruise", args.motion_pool_config["primitive_library"])

    def test_stage8_enables_camera_gru_and_short_high_amplitude_pool(self) -> None:
        args = self._parse(
            "stage8_camera_short_high_amplitude_5hz_ratio_5to5_100k_seed8.json"
        )
        self.assertEqual(args.seed, 8)
        self.assertEqual(args.rollout_steps, 256)
        self.assertEqual(args.episode_length, 320)
        self.assertEqual(args.actor_recurrent_mode, "train")
        self.assertTrue(args.camera_tracking_enabled)
        self.assertTrue(args.allow_partial_checkpoint)
        self.assertAlmostEqual(args.camera_yaw_helper_kp, 1.0)
        self.assertAlmostEqual(args.camera_yaw_helper_kd, 0.15)
        self.assertAlmostEqual(args.camera_yaw_helper_max_rate_rad_s, 0.6)
        self.assertAlmostEqual(args.camera_yaw_helper_deadband_deg, 2.0)
        self.assertAlmostEqual(args.attitude_feedback_scale, 2.0)
        self.assertAlmostEqual(args.xy_velocity_damping_gain, 0.18)
        self.assertAlmostEqual(args.xy_target_velocity_gain, 0.18)
        self.assertAlmostEqual(args.xy_target_accel_gain, 0.102)
        self.assertAlmostEqual(args.tracking_xy_tolerance_m, 0.3)
        self.assertAlmostEqual(args.moving_success_xy_tolerance_m, 0.3)
        self.assertAlmostEqual(args.tracking_z_tolerance_m, 0.3)
        self.assertAlmostEqual(args.vertical_motion_z_tolerance_m, 0.3)
        self.assertEqual(args.motion_pool_config["min_segments"], 2)
        self.assertEqual(args.motion_pool_config["max_segments"], 4)
        self.assertIn("turn", args.motion_pool_config["required_ids"])
        turn = args.motion_pool_config["primitive_library"]["turn"]
        climb = args.motion_pool_config["primitive_library"]["climb"]
        self.assertAlmostEqual(turn["min_heading_change_deg"], 35.0)
        self.assertAlmostEqual(climb["min_vertical_displacement_m"], 0.6)
        self.assertEqual(args.best_checkpoint_min_episodes, 24)

    def test_stage8_500k_preserves_camera_curriculum_and_seed(self) -> None:
        args = self._parse(
            "stage8_camera_short_high_amplitude_5hz_ratio_5to5_500k_seed8.json"
        )
        self.assertEqual(args.seed, 8)
        self.assertEqual(args.num_env_steps, 500000)
        self.assertEqual(args.policy_ratio, 0.5)
        self.assertTrue(args.camera_tracking_enabled)
        self.assertTrue(args.allow_partial_checkpoint)
        self.assertIn("seed7_20260719_154139", args.load_checkpoint)

    def test_stage9_resets_actor_output_and_correlates_global_yaw(self) -> None:
        args = self._parse(
            "stage9_global_yaw_visible_bridge_5hz_ratio_8to2_100k_seed9.json"
        )
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.num_env_steps, 100000)
        self.assertAlmostEqual(args.policy_ratio, 0.2)
        self.assertTrue(args.reset_actor_output_on_load)
        self.assertFalse(args.allow_partial_checkpoint)
        self.assertEqual(args.early_stop_patience_updates, 0)
        self.assertAlmostEqual(args.lr, 1e-5)
        self.assertAlmostEqual(args.reference_kl_coef, 0.0)
        self.assertTrue(args.align_initial_yaw_to_line)
        self.assertAlmostEqual(args.line_yaw_min_deg, -180.0)
        self.assertAlmostEqual(args.line_yaw_max_deg, 180.0)
        self.assertAlmostEqual(args.initial_camera_bearing_min_deg, -40.0)
        self.assertAlmostEqual(args.initial_camera_bearing_max_deg, 40.0)
        self.assertEqual(args.motion_pool_config["prefix_ids"], ["accelerate"])
        self.assertEqual(args.motion_pool_config["required_ids"], ["turn"])
        self.assertEqual(args.motion_pool_config["required_one_of_ids"], [])
        self.assertEqual(args.motion_pool_config["min_segments"], 1)
        self.assertEqual(args.motion_pool_config["max_segments"], 2)
        self.assertIn("seed8_20260720_005211", args.load_checkpoint)

    def test_stage10_preserves_policy_for_7_to_3_handoff(self) -> None:
        args = self._parse(
            "stage10_global_yaw_visible_bridge_5hz_ratio_7to3_64k_seed10.json"
        )
        self.assertEqual(args.seed, 10)
        self.assertEqual(args.num_env_steps, 65536)
        self.assertAlmostEqual(args.policy_ratio, 0.3)
        self.assertFalse(args.reset_optimizer)
        self.assertFalse(args.reset_actor_output_on_load)
        self.assertIn("seed9_20260720_103119", args.load_checkpoint)

    def test_stage11_preserves_policy_for_6_to_4_handoff(self) -> None:
        args = self._parse(
            "stage11_global_yaw_visible_bridge_5hz_ratio_6to4_64k_seed11.json"
        )
        self.assertEqual(args.seed, 11)
        self.assertEqual(args.num_env_steps, 65536)
        self.assertAlmostEqual(args.policy_ratio, 0.4)
        self.assertFalse(args.reset_optimizer)
        self.assertFalse(args.reset_actor_output_on_load)
        self.assertIn("seed10_20260720_120604", args.load_checkpoint)
        self.assertIn("actor_critic_update_4.pt", args.load_checkpoint)

    def test_full_helper_short_evaluation_has_zero_policy_mix(self) -> None:
        args = self._parse("eval_stage8_full_helper_short.json")
        self.assertTrue(args.evaluation_only)
        self.assertTrue(args.deterministic_actions)
        self.assertTrue(args.from_scratch)
        self.assertEqual(args.num_env_steps, 20480)
        self.assertEqual(args.rollout_steps, 320)
        self.assertEqual(args.policy_ratio, 0.0)
        self.assertEqual(args.best_checkpoint_window, 0)
        self.assertEqual(Path(args.results_root).name, "controller_helper_test")

    def test_full_helper_v2_is_a_paired_seed80_evaluation(self) -> None:
        args = self._parse("eval_stage8_full_helper_v2_short.json")
        self.assertTrue(args.evaluation_only)
        self.assertEqual(args.seed, 80)
        self.assertEqual(args.policy_ratio, 0.0)
        self.assertAlmostEqual(args.attitude_feedback_scale, 2.0)
        self.assertAlmostEqual(args.xy_velocity_damping_gain, 0.18)

    def test_fixed_evaluation_is_deterministic_and_does_not_train(self) -> None:
        args = self._parse("eval_fixed_seed5_best_50k.json")
        self.assertTrue(args.evaluation_only)
        self.assertTrue(args.deterministic_actions)
        self.assertTrue(args.independent_env_rng)
        self.assertEqual(args.actor_recurrent_mode, "frozen")
        self.assertEqual(args.best_checkpoint_window, 0)


if __name__ == "__main__":
    unittest.main()
